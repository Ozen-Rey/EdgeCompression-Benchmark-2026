"""Offline interpretability audit of the content-aware kNN predictor.

This module is read-only against artefacts already produced by the
content-aware pipeline (oracle by-image CSV, metadata/features CSV, and
optionally the classifier decisions CSV). It does not re-run the router,
does not re-run benchmarks, does not change the ranking score, does not
change the operational report schema, and does not modify any
calibration or feedback file.

The audit answers three questions that the headline regret-reduction
number alone does not address:

1. **Class balance.** What is the oracle class distribution globally
   and per source? Are minority classes present in every LODO training
   fold? When at least one fold drops a class, the audit flags
   ``class_missing_in_lodo_training_fold`` so downstream claims account
   for it. When the global count of a class is small (≤ 5), the audit
   flags ``minority_class_too_small_for_structural_claim`` so the
   reader is reminded that no decision-boundary claim can be made on
   that class from this benchmark.
2. **Surrogate decision tree.** A depth-sweep of decision trees fit to
   imitate the kNN predictions on the binary JPEG-vs-JXL subproblem
   (HEVC is kept as a qualitative case study, not as a structural
   claim). For each depth we report ``fidelity_to_knn``,
   ``fidelity_to_oracle``, the number of leaves, and an
   ``export_text`` rule listing. The tree is meant to be a surrogate
   *interpretation* of the kNN, not a replacement of it.
3. **Logistic-regression pairwise interactions and feature
   attribution.** A logistic regression on the binary subproblem with
   pairwise interactions (``megapixels × aspect_ratio``,
   ``megapixels × orientation``, ``aspect_ratio × orientation``)
   reports the coefficients. Two model-agnostic attribution methods
   (leave-one-feature-out and permutation) are also run against the
   surrogate's fidelity to the kNN.

The wording in the auto-generated interpretation strings is
deliberately prudent: ``suggests``, ``is consistent with``, ``within
this benchmark``. No claim of statistical significance is made unless a
proper test is run — and none is, here, by design.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


__all__ = [
    "load_unified_rows",
    "audit_class_balance",
    "detect_lodo_missing_classes",
    "filter_binary_jpeg_jxl",
    "build_hevc_case_study",
    "run_surrogate_tree_sweep",
    "run_logistic_pairwise_interactions",
    "run_feature_attribution",
    "build_interpretation",
    "main",
]


try:
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix
    from sklearn.tree import DecisionTreeClassifier, export_text

    SKLEARN_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - exercised only when sklearn is absent.
    ConvergenceWarning = Warning  # type: ignore[assignment]
    LogisticRegression = None  # type: ignore[assignment]
    accuracy_score = None  # type: ignore[assignment]
    confusion_matrix = None  # type: ignore[assignment]
    DecisionTreeClassifier = None  # type: ignore[assignment]
    export_text = None  # type: ignore[assignment]
    SKLEARN_IMPORT_ERROR = exc


_HEVC_LABEL_DEFAULT = "HEVC|crf=15"
_MINORITY_CLASS_THRESHOLD = 5

_NUMERIC_FEATURES = ["megapixels", "aspect_ratio"]
_CATEGORICAL_FEATURES = ["resolution_class", "orientation_class"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_sklearn() -> None:
    if SKLEARN_IMPORT_ERROR is not None:
        raise RuntimeError(
            "scikit-learn is required for content predictor interpretability. "
            "Install it with: python -m pip install scikit-learn"
        ) from SKLEARN_IMPORT_ERROR


def _read_csv(path: str) -> List[Dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8-sig") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() == "none":
        return None
    try:
        return float(text.replace(",", "."))
    except ValueError:
        return None


def _label(codec: Any, config: Any) -> str:
    return f"{codec}|{config}"


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _fmt(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}g}"


# ---------------------------------------------------------------------------
# Unified row loader
# ---------------------------------------------------------------------------


def _load_oracle_rows(path: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in _read_csv(path):
        image_id = str(row.get("image_id", "")).strip()
        if not image_id:
            continue
        out[image_id] = row
    if not out:
        raise ValueError(f"No usable oracle rows in {path}")
    return out


def _load_metadata_rows(path: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in _read_csv(path):
        image_id = str(row.get("image_id", "")).strip()
        if not image_id:
            continue
        out[image_id] = row
    if not out:
        raise ValueError(f"No usable metadata rows in {path}")
    return out


def _filter_classifier_decisions(
    rows: List[Dict[str, str]],
    *,
    feature_set: Optional[str],
    k: Optional[int],
    evaluation_mode: Optional[str],
) -> List[Dict[str, str]]:
    has_feature_set = bool(rows) and "feature_set" in rows[0]
    has_k = bool(rows) and "k" in rows[0]
    has_evaluation_mode = bool(rows) and "evaluation_mode" in rows[0]

    out: List[Dict[str, str]] = []
    for row in rows:
        if (
            feature_set is not None
            and has_feature_set
            and str(row.get("feature_set", "")).strip() != feature_set
        ):
            continue
        if k is not None and has_k:
            row_k = _to_float(row.get("k"))
            if row_k is None or int(row_k) != int(k):
                continue
        if (
            evaluation_mode is not None
            and has_evaluation_mode
            and str(row.get("evaluation_mode", "")).strip() != evaluation_mode
        ):
            continue
        out.append(row)
    return out


def _index_decisions_by_image(
    rows: List[Dict[str, str]],
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        image_id = str(row.get("image_id", "")).strip()
        if image_id and image_id not in out:
            out[image_id] = row
    return out


def load_unified_rows(
    *,
    oracle_by_image_path: str,
    metadata_features_path: str,
    classifier_decisions_path: Optional[str] = None,
    loio_decisions_path: Optional[str] = None,
    lodo_decisions_path: Optional[str] = None,
    feature_set: Optional[str] = "metadata_no_source",
    k: Optional[int] = 7,
) -> Dict[str, Any]:
    """Load and join the per-image artefacts into a single list of rows."""
    oracle_rows = _load_oracle_rows(oracle_by_image_path)
    metadata_rows = _load_metadata_rows(metadata_features_path)

    loio_index: Dict[str, Dict[str, Any]] = {}
    lodo_index: Dict[str, Dict[str, Any]] = {}

    if loio_decisions_path:
        raw = _read_csv(loio_decisions_path)
        filtered = _filter_classifier_decisions(
            raw,
            feature_set=feature_set,
            k=k,
            evaluation_mode="leave_one_image_out",
        )
        loio_index = _index_decisions_by_image(filtered)

    if lodo_decisions_path:
        raw = _read_csv(lodo_decisions_path)
        filtered = _filter_classifier_decisions(
            raw,
            feature_set=feature_set,
            k=k,
            evaluation_mode="leave_one_dataset_out",
        )
        lodo_index = _index_decisions_by_image(filtered)

    if classifier_decisions_path:
        raw = _read_csv(classifier_decisions_path)
        if (
            raw
            and "evaluation_mode" in raw[0]
            and not loio_index
        ):
            loio_filtered = _filter_classifier_decisions(
                raw,
                feature_set=feature_set,
                k=k,
                evaluation_mode="leave_one_image_out",
            )
            loio_index = _index_decisions_by_image(loio_filtered)
        if (
            raw
            and "evaluation_mode" in raw[0]
            and not lodo_index
        ):
            lodo_filtered = _filter_classifier_decisions(
                raw,
                feature_set=feature_set,
                k=k,
                evaluation_mode="leave_one_dataset_out",
            )
            lodo_index = _index_decisions_by_image(lodo_filtered)
        if not (loio_index or lodo_index):
            filtered = _filter_classifier_decisions(
                raw,
                feature_set=feature_set,
                k=k,
                evaluation_mode=None,
            )
            loio_index = _index_decisions_by_image(filtered)

    unified: List[Dict[str, Any]] = []
    for image_id, oracle in oracle_rows.items():
        meta = metadata_rows.get(image_id)
        if meta is None:
            continue

        oracle_codec = oracle.get("oracle_codec") or meta.get("oracle_codec")
        oracle_config = oracle.get("oracle_config") or meta.get("oracle_config")
        oracle_label = _label(oracle_codec, oracle_config)

        loio_decision = loio_index.get(image_id)
        lodo_decision = lodo_index.get(image_id)

        loio_predicted_label = (
            _label(
                loio_decision.get("predicted_codec"),
                loio_decision.get("predicted_config"),
            )
            if loio_decision
            else None
        )
        lodo_predicted_label = (
            _label(
                lodo_decision.get("predicted_codec"),
                lodo_decision.get("predicted_config"),
            )
            if lodo_decision
            else None
        )

        unified.append(
            {
                "image_id": image_id,
                "dataset": oracle.get("dataset") or meta.get("dataset"),
                "image": oracle.get("image") or meta.get("image"),
                "oracle_codec": oracle_codec,
                "oracle_config": oracle_config,
                "oracle_label": oracle_label,
                "megapixels": _to_float(meta.get("megapixels")),
                "aspect_ratio": _to_float(meta.get("aspect_ratio")),
                "resolution_class": str(meta.get("resolution_class", "unknown")),
                "orientation_class": str(meta.get("orientation_class", "unknown")),
                "loio_predicted_label": loio_predicted_label,
                "lodo_predicted_label": lodo_predicted_label,
            }
        )

    if not unified:
        raise ValueError(
            "load_unified_rows produced no rows; check that the oracle and "
            "metadata CSVs share image_id values."
        )

    return {
        "rows": unified,
        "has_loio": bool(loio_index),
        "has_lodo": bool(lodo_index),
    }


# ---------------------------------------------------------------------------
# Class balance audit
# ---------------------------------------------------------------------------


def audit_class_balance(
    rows: List[Dict[str, Any]],
    *,
    hevc_label: str = _HEVC_LABEL_DEFAULT,
    minority_threshold: int = _MINORITY_CLASS_THRESHOLD,
) -> Dict[str, Any]:
    oracle_counter: Counter = Counter(r["oracle_label"] for r in rows)
    per_dataset: Dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        per_dataset[str(r.get("dataset", "unknown"))][r["oracle_label"]] += 1

    loio_counter: Counter = Counter(
        r["loio_predicted_label"]
        for r in rows
        if r.get("loio_predicted_label")
    )
    lodo_counter: Counter = Counter(
        r["lodo_predicted_label"]
        for r in rows
        if r.get("lodo_predicted_label")
    )

    warnings: List[str] = []
    for label, count in oracle_counter.items():
        if count <= minority_threshold:
            warnings.append(
                f"minority_class_too_small_for_structural_claim:{label}={count}"
            )

    return {
        "global_oracle_distribution": dict(oracle_counter),
        "per_dataset_oracle_distribution": {
            ds: dict(counter) for ds, counter in per_dataset.items()
        },
        "loio_prediction_distribution": dict(loio_counter),
        "lodo_prediction_distribution": dict(lodo_counter),
        "hevc_count": int(oracle_counter.get(hevc_label, 0)),
        "minority_threshold": minority_threshold,
        "warnings": warnings,
    }


def detect_lodo_missing_classes(
    rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """For each LODO fold (one per dataset), list classes absent from training."""
    datasets = sorted({str(r.get("dataset", "unknown")) for r in rows})
    all_classes = {r["oracle_label"] for r in rows}

    out: List[Dict[str, Any]] = []
    for left_out in datasets:
        training_labels = {
            r["oracle_label"]
            for r in rows
            if str(r.get("dataset")) != left_out
        }
        missing = sorted(all_classes - training_labels)
        if missing:
            out.append(
                {
                    "left_out_dataset": left_out,
                    "missing_classes": missing,
                    "training_classes": sorted(training_labels),
                }
            )
    return out


# ---------------------------------------------------------------------------
# Binary JPEG-vs-JXL split + HEVC case study
# ---------------------------------------------------------------------------


def filter_binary_jpeg_jxl(
    rows: List[Dict[str, Any]],
    *,
    hevc_label: str = _HEVC_LABEL_DEFAULT,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    binary = [r for r in rows if r["oracle_label"] != hevc_label]
    hevc = [r for r in rows if r["oracle_label"] == hevc_label]
    return binary, hevc


def build_hevc_case_study(
    hevc_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    listing = [
        {
            "image_id": r["image_id"],
            "dataset": r["dataset"],
            "image": r["image"],
            "megapixels": r.get("megapixels"),
            "aspect_ratio": r.get("aspect_ratio"),
            "resolution_class": r.get("resolution_class"),
            "orientation_class": r.get("orientation_class"),
            "oracle_label": r["oracle_label"],
            "loio_predicted_label": r.get("loio_predicted_label"),
            "lodo_predicted_label": r.get("lodo_predicted_label"),
        }
        for r in hevc_rows
    ]
    return {
        "num_samples": len(listing),
        "rows": listing,
        "note": (
            "HEVC oracle-optimal cases are too few in the current benchmark "
            "to support a structural decision-boundary claim. They are "
            "listed here as a qualitative case study."
        ),
    }


# ---------------------------------------------------------------------------
# Encoding for sklearn surrogates
# ---------------------------------------------------------------------------


def _build_feature_columns(rows: List[Dict[str, Any]]) -> Tuple[
    List[str],
    Dict[str, List[str]],
]:
    """Return the ordered feature column names plus the categorical levels."""
    levels: Dict[str, List[str]] = {}
    for feat in _CATEGORICAL_FEATURES:
        levels[feat] = sorted({str(r.get(feat, "unknown")) for r in rows})

    columns: List[str] = list(_NUMERIC_FEATURES)
    for feat in _CATEGORICAL_FEATURES:
        for value in levels[feat]:
            columns.append(f"{feat}={value}")
    return columns, levels


def _encode(
    rows: List[Dict[str, Any]],
    *,
    columns: List[str],
    levels: Dict[str, List[str]],
) -> List[List[float]]:
    out: List[List[float]] = []
    for r in rows:
        vec: List[float] = []
        for feat in _NUMERIC_FEATURES:
            value = _to_float(r.get(feat))
            vec.append(0.0 if value is None else value)
        for feat in _CATEGORICAL_FEATURES:
            current = str(r.get(feat, "unknown"))
            for level in levels[feat]:
                vec.append(1.0 if current == level else 0.0)
        out.append(vec)
    return out


def _surrogate_targets(
    rows: List[Dict[str, Any]],
    *,
    target: str,
) -> List[str]:
    if target == "knn_loio":
        return [str(r.get("loio_predicted_label") or "") for r in rows]
    if target == "knn_lodo":
        return [str(r.get("lodo_predicted_label") or "") for r in rows]
    if target == "oracle":
        return [str(r["oracle_label"]) for r in rows]
    raise ValueError(f"Unknown surrogate target: {target}")


def _resolve_target(
    rows: List[Dict[str, Any]],
) -> Tuple[str, str]:
    """Pick the best available surrogate target.

    Order of preference: kNN LOIO predictions > kNN LODO predictions >
    oracle labels (fallback when no classifier decisions were provided).
    """
    if any(r.get("loio_predicted_label") for r in rows):
        return "knn_loio", "fidelity_to_knn_loio"
    if any(r.get("lodo_predicted_label") for r in rows):
        return "knn_lodo", "fidelity_to_knn_lodo"
    return "oracle", "fidelity_to_oracle_self"


# ---------------------------------------------------------------------------
# Surrogate decision tree sweep
# ---------------------------------------------------------------------------


def run_surrogate_tree_sweep(
    binary_rows: List[Dict[str, Any]],
    *,
    max_depths: List[int],
    seed: int = 42,
) -> Dict[str, Any]:
    """Fit a decision tree for each depth and report fidelity + rules."""
    _require_sklearn()

    if not binary_rows:
        return {
            "depth_rows": [],
            "rules_by_depth": {},
            "target": "unavailable",
            "fidelity_target_column": None,
            "warnings": ["binary_jpeg_jxl_set_is_empty"],
        }

    target_kind, fidelity_column = _resolve_target(binary_rows)
    columns, levels = _build_feature_columns(binary_rows)
    X = _encode(binary_rows, columns=columns, levels=levels)

    y_target = _surrogate_targets(binary_rows, target=target_kind)
    y_oracle = _surrogate_targets(binary_rows, target="oracle")

    # When the kNN target is missing for some rows, fall back to oracle for
    # those rows so the surrogate has a label to learn from.
    y_target_filled = [
        label if label else y_oracle[i]
        for i, label in enumerate(y_target)
    ]

    unique_targets = sorted(set(y_target_filled))
    warnings: List[str] = []
    if len(unique_targets) < 2:
        warnings.append("surrogate_target_collapsed_to_single_class")

    depth_rows: List[Dict[str, Any]] = []
    rules_by_depth: Dict[int, str] = {}

    for depth in max_depths:
        if len(unique_targets) < 2:
            depth_rows.append(
                {
                    "max_depth": depth,
                    "num_leaves": 0,
                    "fidelity_to_knn": None,
                    "fidelity_to_oracle": None,
                    "accuracy_binary_jpeg_jxl": None,
                    "confusion_matrix_vs_knn": None,
                    "confusion_matrix_vs_oracle": None,
                    "note": "single_class_target_no_meaningful_tree",
                }
            )
            rules_by_depth[depth] = "single class target; no tree fit"
            continue

        tree = DecisionTreeClassifier(
            max_depth=int(depth),
            min_samples_leaf=2,
            random_state=int(seed),
        )
        tree.fit(X, y_target_filled)

        predicted = list(tree.predict(X))
        fidelity_to_knn = float(accuracy_score(y_target_filled, predicted))
        fidelity_to_oracle = float(accuracy_score(y_oracle, predicted))

        confusion_vs_knn = confusion_matrix(
            y_target_filled,
            predicted,
            labels=unique_targets,
        ).tolist()
        confusion_vs_oracle = confusion_matrix(
            y_oracle,
            predicted,
            labels=unique_targets,
        ).tolist()

        depth_rows.append(
            {
                "max_depth": int(depth),
                "num_leaves": int(tree.get_n_leaves()),
                "fidelity_to_knn": fidelity_to_knn,
                "fidelity_to_oracle": fidelity_to_oracle,
                "accuracy_binary_jpeg_jxl": fidelity_to_oracle,
                "confusion_matrix_vs_knn": confusion_vs_knn,
                "confusion_matrix_vs_oracle": confusion_vs_oracle,
                "labels": unique_targets,
                "note": "",
            }
        )

        rules_by_depth[int(depth)] = export_text(
            tree,
            feature_names=columns,
            decimals=4,
        )

    return {
        "depth_rows": depth_rows,
        "rules_by_depth": rules_by_depth,
        "target": target_kind,
        "fidelity_target_column": fidelity_column,
        "feature_columns": columns,
        "labels": unique_targets,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Logistic regression with pairwise interactions
# ---------------------------------------------------------------------------


def _build_interaction_columns(
    rows: List[Dict[str, Any]],
    columns: List[str],
    levels: Dict[str, List[str]],
) -> Tuple[List[str], List[List[float]]]:
    base = _encode(rows, columns=columns, levels=levels)
    column_index = {name: idx for idx, name in enumerate(columns)}

    interaction_columns: List[str] = []
    interaction_vectors: List[List[float]] = [[] for _ in rows]

    def _interact(a: str, b: str) -> None:
        i = column_index.get(a)
        j = column_index.get(b)
        if i is None or j is None:
            return
        name = f"{a} * {b}"
        interaction_columns.append(name)
        for k, vec in enumerate(base):
            interaction_vectors[k].append(vec[i] * vec[j])

    _interact("megapixels", "aspect_ratio")

    for level in levels.get("orientation_class", []):
        _interact("megapixels", f"orientation_class={level}")
        _interact("aspect_ratio", f"orientation_class={level}")

    for level in levels.get("resolution_class", []):
        _interact("aspect_ratio", f"resolution_class={level}")

    combined_columns = list(columns) + interaction_columns
    combined_vectors = [
        base_row + interaction_row
        for base_row, interaction_row in zip(base, interaction_vectors)
    ]
    return combined_columns, combined_vectors


def run_logistic_pairwise_interactions(
    binary_rows: List[Dict[str, Any]],
    *,
    seed: int = 42,
) -> Dict[str, Any]:
    _require_sklearn()

    if not binary_rows:
        return {
            "coefficients": [],
            "model_score": None,
            "warnings": ["binary_jpeg_jxl_set_is_empty"],
        }

    targets = [r["oracle_label"] for r in binary_rows]
    if len(set(targets)) < 2:
        return {
            "coefficients": [],
            "model_score": None,
            "warnings": ["logistic_regression_skipped_single_class_target"],
        }

    columns, levels = _build_feature_columns(binary_rows)
    expanded_columns, X = _build_interaction_columns(
        binary_rows,
        columns,
        levels,
    )

    import warnings as _warnings

    warnings_emitted: List[str] = []

    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", ConvergenceWarning)
        try:
            model = LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=int(seed),
            )
            model.fit(X, targets)
            score = float(model.score(X, targets))
            coefs = model.coef_[0].tolist()
        except Exception as exc:
            return {
                "coefficients": [],
                "model_score": None,
                "warnings": [
                    f"logistic_regression_fit_failed:{type(exc).__name__}"
                ],
            }

    coefficient_rows: List[Dict[str, Any]] = []
    for name, coef in zip(expanded_columns, coefs):
        coefficient_rows.append(
            {
                "feature": name,
                "coefficient": float(coef),
                "absolute_coefficient": float(abs(coef)),
                "sign": (
                    "positive" if coef > 0
                    else "negative" if coef < 0
                    else "zero"
                ),
            }
        )
    coefficient_rows.sort(
        key=lambda row: row["absolute_coefficient"],
        reverse=True,
    )

    return {
        "coefficients": coefficient_rows,
        "model_score": score,
        "labels": sorted(set(targets)),
        "warnings": warnings_emitted,
    }


# ---------------------------------------------------------------------------
# Model-agnostic feature attribution
# ---------------------------------------------------------------------------


def _fit_surrogate_for_attribution(
    *,
    X: List[List[float]],
    y: List[str],
    seed: int,
) -> Optional[Any]:
    if len(set(y)) < 2:
        return None
    tree = DecisionTreeClassifier(
        max_depth=4,
        min_samples_leaf=2,
        random_state=int(seed),
    )
    tree.fit(X, y)
    return tree


def _zero_feature(
    X: List[List[float]],
    feature_index: int,
) -> List[List[float]]:
    return [
        [value if i != feature_index else 0.0 for i, value in enumerate(row)]
        for row in X
    ]


def _permute_feature(
    X: List[List[float]],
    feature_index: int,
    *,
    rng: random.Random,
) -> List[List[float]]:
    column = [row[feature_index] for row in X]
    rng.shuffle(column)
    return [
        [
            (column[k] if i == feature_index else value)
            for i, value in enumerate(row)
        ]
        for k, row in enumerate(X)
    ]


def run_feature_attribution(
    binary_rows: List[Dict[str, Any]],
    *,
    seed: int = 42,
) -> Dict[str, Any]:
    _require_sklearn()

    if not binary_rows:
        return {
            "rows": [],
            "baseline_fidelity": None,
            "target": "unavailable",
            "warnings": ["binary_jpeg_jxl_set_is_empty"],
        }

    target_kind, _ = _resolve_target(binary_rows)
    columns, levels = _build_feature_columns(binary_rows)
    X = _encode(binary_rows, columns=columns, levels=levels)

    y_target = _surrogate_targets(binary_rows, target=target_kind)
    y_oracle = _surrogate_targets(binary_rows, target="oracle")
    y_filled = [label or y_oracle[i] for i, label in enumerate(y_target)]

    surrogate = _fit_surrogate_for_attribution(X=X, y=y_filled, seed=seed)
    if surrogate is None:
        return {
            "rows": [],
            "baseline_fidelity": None,
            "target": target_kind,
            "warnings": ["surrogate_target_collapsed_to_single_class"],
        }

    baseline_predictions = list(surrogate.predict(X))
    baseline_fidelity = float(accuracy_score(y_filled, baseline_predictions))

    attribution_rows: List[Dict[str, Any]] = []
    rng = random.Random(int(seed))

    for feature_index, feature_name in enumerate(columns):
        X_loo = _zero_feature(X, feature_index)
        loo_predictions = list(surrogate.predict(X_loo))
        loo_fidelity = float(accuracy_score(y_filled, loo_predictions))
        attribution_rows.append(
            {
                "feature": feature_name,
                "method": "leave_one_feature_out",
                "baseline_score": baseline_fidelity,
                "perturbed_score": loo_fidelity,
                "delta": baseline_fidelity - loo_fidelity,
                "target": target_kind,
            }
        )

        X_perm = _permute_feature(X, feature_index, rng=rng)
        perm_predictions = list(surrogate.predict(X_perm))
        perm_fidelity = float(accuracy_score(y_filled, perm_predictions))
        attribution_rows.append(
            {
                "feature": feature_name,
                "method": "permutation",
                "baseline_score": baseline_fidelity,
                "perturbed_score": perm_fidelity,
                "delta": baseline_fidelity - perm_fidelity,
                "target": target_kind,
            }
        )

    attribution_rows.sort(key=lambda r: r["delta"], reverse=True)
    for rank, row in enumerate(attribution_rows, start=1):
        row["rank"] = rank

    return {
        "rows": attribution_rows,
        "baseline_fidelity": baseline_fidelity,
        "target": target_kind,
        "warnings": [],
    }


# ---------------------------------------------------------------------------
# Auto-generated interpretation strings
# ---------------------------------------------------------------------------


_SHALLOW_FIDELITY_THRESHOLD = 0.80


def _depth_row(rows: List[Dict[str, Any]], depth: int) -> Optional[Dict[str, Any]]:
    for row in rows:
        if row.get("max_depth") == depth:
            return row
    return None


def build_interpretation(
    *,
    class_balance: Dict[str, Any],
    surrogate_sweep: Dict[str, Any],
    logistic_report: Dict[str, Any],
    attribution_report: Dict[str, Any],
    hevc_case_study: Dict[str, Any],
    lodo_missing: List[Dict[str, Any]],
) -> List[str]:
    notes: List[str] = []

    shallow = _depth_row(surrogate_sweep.get("depth_rows", []), 2)
    deep = _depth_row(surrogate_sweep.get("depth_rows", []), 4)

    if shallow and shallow.get("fidelity_to_knn") is not None:
        fid = shallow["fidelity_to_knn"]
        if fid >= _SHALLOW_FIDELITY_THRESHOLD:
            notes.append(
                f"A depth-2 decision tree surrogate matches the kNN on "
                f"{fid:.2%} of the binary JPEG-vs-JXL cases, which "
                "suggests that the metadata-only predictor is largely "
                "explained by a small number of interpretable thresholds "
                "within this benchmark."
            )
        elif deep and deep.get("fidelity_to_knn") is not None:
            notes.append(
                f"A depth-2 surrogate reaches only "
                f"{fid:.2%} fidelity to the kNN, while depth=4 reaches "
                f"{deep['fidelity_to_knn']:.2%}; this is consistent with a "
                "decision boundary that requires depth-3 splits or "
                "feature interactions to be captured."
            )

    attribution_rows = attribution_report.get("rows", [])
    permutation_only = [
        r for r in attribution_rows if r["method"] == "permutation"
    ]
    if permutation_only:
        permutation_only.sort(key=lambda r: r["delta"], reverse=True)
        top = permutation_only[0]
        if top["delta"] > 0:
            base = top["feature"].split("=", 1)[0]
            notes.append(
                f"Permutation attribution ranks '{top['feature']}' as the "
                f"most influential feature for the surrogate (Δ fidelity = "
                f"{top['delta']:.3f}); within this benchmark, "
                f"'{base}' appears to carry the dominant metadata signal."
            )

    coefficients = logistic_report.get("coefficients", [])
    if coefficients:
        main_effects = [
            c for c in coefficients if " * " not in c["feature"]
        ]
        interactions = [
            c for c in coefficients if " * " in c["feature"]
        ]
        if interactions and main_effects:
            max_main = max(c["absolute_coefficient"] for c in main_effects)
            max_interaction = max(
                c["absolute_coefficient"] for c in interactions
            )
            if max_interaction >= max_main and max_interaction > 0:
                notes.append(
                    "At least one pairwise interaction term in the logistic "
                    "regression has a magnitude comparable to or larger "
                    "than the main effects; pairwise interactions may "
                    "contribute additional structure to the JPEG-vs-JXL "
                    "boundary within this benchmark."
                )

    hevc_count = hevc_case_study.get("num_samples", 0)
    if hevc_count <= _MINORITY_CLASS_THRESHOLD:
        notes.append(
            f"HEVC oracle-optimal cases are too few in the current "
            f"benchmark (n={hevc_count}) for a structural decision-boundary "
            "claim; they are reported as a qualitative case study only."
        )

    if lodo_missing:
        descriptions = ", ".join(
            f"{entry['left_out_dataset']}({', '.join(entry['missing_classes'])})"
            for entry in lodo_missing
        )
        notes.append(
            f"In at least one LODO fold the training set does not contain "
            f"every oracle class: {descriptions}. Generalization claims for "
            "the missing class in that fold cannot be supported by this "
            "benchmark."
        )

    notes.append(
        "These observations describe patterns within the current 96-image "
        "multi-source benchmark and do not establish universal "
        "generalization to arbitrary natural-image distributions."
    )

    return notes


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def _write_class_balance_csv(
    path: Path,
    class_balance: Dict[str, Any],
    lodo_missing: List[Dict[str, Any]],
) -> None:
    rows: List[Dict[str, Any]] = []

    for label, count in sorted(
        class_balance["global_oracle_distribution"].items()
    ):
        rows.append(
            {
                "scope": "global_oracle",
                "key": label,
                "count": count,
            }
        )

    for dataset, dist in sorted(
        class_balance["per_dataset_oracle_distribution"].items()
    ):
        for label, count in sorted(dist.items()):
            rows.append(
                {
                    "scope": f"dataset:{dataset}",
                    "key": label,
                    "count": count,
                }
            )

    for label, count in sorted(
        class_balance["loio_prediction_distribution"].items()
    ):
        rows.append(
            {
                "scope": "loio_prediction",
                "key": label,
                "count": count,
            }
        )

    for label, count in sorted(
        class_balance["lodo_prediction_distribution"].items()
    ):
        rows.append(
            {
                "scope": "lodo_prediction",
                "key": label,
                "count": count,
            }
        )

    for entry in lodo_missing:
        rows.append(
            {
                "scope": f"lodo_missing:{entry['left_out_dataset']}",
                "key": ",".join(entry["missing_classes"]),
                "count": len(entry["missing_classes"]),
            }
        )

    _write_csv(path, rows)


def _write_tree_surrogate_csv(
    path: Path,
    sweep: Dict[str, Any],
) -> None:
    rows: List[Dict[str, Any]] = []
    for entry in sweep.get("depth_rows", []):
        rows.append(
            {
                "max_depth": entry["max_depth"],
                "num_leaves": entry["num_leaves"],
                "fidelity_to_knn": _fmt(entry.get("fidelity_to_knn"), 5),
                "fidelity_to_oracle": _fmt(entry.get("fidelity_to_oracle"), 5),
                "accuracy_binary_jpeg_jxl": _fmt(
                    entry.get("accuracy_binary_jpeg_jxl"), 5
                ),
                "target": sweep.get("target"),
                "note": entry.get("note") or "",
            }
        )
    _write_csv(path, rows)


def _write_tree_rules_files(
    out_dir: Path,
    sweep: Dict[str, Any],
) -> Dict[int, str]:
    out_paths: Dict[int, str] = {}
    for depth, rules in sweep.get("rules_by_depth", {}).items():
        path = out_dir / f"content_predictor_tree_rules_depth_{depth}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rules, encoding="utf-8")
        out_paths[int(depth)] = str(path)
    return out_paths


def _write_logistic_csv(
    path: Path,
    logistic: Dict[str, Any],
) -> None:
    rows = [
        {
            "feature": row["feature"],
            "coefficient": _fmt(row["coefficient"], 6),
            "absolute_coefficient": _fmt(row["absolute_coefficient"], 6),
            "sign": row["sign"],
            "model_score": _fmt(logistic.get("model_score"), 5),
        }
        for row in logistic.get("coefficients", [])
    ]
    _write_csv(path, rows)


def _write_attribution_csv(
    path: Path,
    attribution: Dict[str, Any],
) -> None:
    rows = [
        {
            "feature": row["feature"],
            "method": row["method"],
            "baseline_score": _fmt(row["baseline_score"], 5),
            "perturbed_score": _fmt(row["perturbed_score"], 5),
            "delta": _fmt(row["delta"], 5),
            "rank": row["rank"],
            "target": row["target"],
        }
        for row in attribution.get("rows", [])
    ]
    _write_csv(path, rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="src.router.analysis.content_predictor_interpretability",
        description=(
            "Offline interpretability audit of the content-aware kNN "
            "predictor: class-balance audit, decision-tree surrogate "
            "depth sweep, logistic regression with pairwise interactions, "
            "and model-agnostic feature attribution. Read-only against "
            "existing pipeline artefacts; does not change the router."
        ),
    )

    parser.add_argument(
        "--oracle-by-image",
        required=True,
        help="Path to v09_content_oracle_by_image.csv.",
    )
    parser.add_argument(
        "--metadata-features",
        required=True,
        help=(
            "Path to v09_content_metadata_oracle.csv (or any CSV with "
            "image_id, megapixels, aspect_ratio, resolution_class, "
            "orientation_class)."
        ),
    )
    parser.add_argument(
        "--classifier-decisions",
        default=None,
        help=(
            "Optional per-image classifier decisions CSV (single-run or "
            "sweep). Used to align the surrogate target with the kNN "
            "predictions instead of the oracle labels."
        ),
    )
    parser.add_argument(
        "--loio-decisions",
        default=None,
        help="Optional LOIO decisions CSV (used in place of --classifier-decisions).",
    )
    parser.add_argument(
        "--lodo-decisions",
        default=None,
        help="Optional LODO decisions CSV (used in place of --classifier-decisions).",
    )
    parser.add_argument("--feature-set", default="metadata_no_source")
    parser.add_argument("--k", type=int, default=7)

    parser.add_argument(
        "--out-dir",
        default="results/routing_context/content_predictor_interpretability",
    )
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--out-summary-csv", default=None)
    parser.add_argument(
        "--tree-depths",
        default="1,2,3,4",
        help="Comma-separated list of max_depth values for the surrogate sweep.",
    )
    parser.add_argument("--seed", type=int, default=42)

    return parser


def _parse_depth_list(value: str) -> List[int]:
    out = [int(item.strip()) for item in str(value).split(",") if item.strip()]
    if not out:
        raise ValueError("--tree-depths must contain at least one integer.")
    return out


def _run(args: argparse.Namespace) -> Dict[str, Any]:
    _require_sklearn()

    loaded = load_unified_rows(
        oracle_by_image_path=args.oracle_by_image,
        metadata_features_path=args.metadata_features,
        classifier_decisions_path=args.classifier_decisions,
        loio_decisions_path=args.loio_decisions,
        lodo_decisions_path=args.lodo_decisions,
        feature_set=args.feature_set,
        k=args.k,
    )
    rows = loaded["rows"]

    class_balance = audit_class_balance(rows)
    lodo_missing = detect_lodo_missing_classes(rows)

    binary_rows, hevc_rows = filter_binary_jpeg_jxl(rows)
    hevc_case_study = build_hevc_case_study(hevc_rows)

    binary_distribution = Counter(r["oracle_label"] for r in binary_rows)
    binary_block = {
        "num_samples": len(binary_rows),
        "class_distribution": dict(binary_distribution),
    }

    depths = _parse_depth_list(args.tree_depths)
    surrogate_sweep = run_surrogate_tree_sweep(
        binary_rows,
        max_depths=depths,
        seed=args.seed,
    )
    logistic_report = run_logistic_pairwise_interactions(
        binary_rows,
        seed=args.seed,
    )
    attribution_report = run_feature_attribution(
        binary_rows,
        seed=args.seed,
    )

    interpretation = build_interpretation(
        class_balance=class_balance,
        surrogate_sweep=surrogate_sweep,
        logistic_report=logistic_report,
        attribution_report=attribution_report,
        hevc_case_study=hevc_case_study,
        lodo_missing=lodo_missing,
    )

    inputs_meta = {
        k: getattr(args, k)
        for k in (
            "oracle_by_image",
            "metadata_features",
            "classifier_decisions",
            "loio_decisions",
            "lodo_decisions",
            "feature_set",
            "k",
            "tree_depths",
            "seed",
        )
    }
    inputs_meta = {k: v for k, v in inputs_meta.items() if v is not None}

    report = {
        "generated_at_unix": int(time.time()),
        "router_target": "content_predictor_interpretability",
        "inputs": inputs_meta,
        "num_rows": len(rows),
        "has_loio_decisions": loaded["has_loio"],
        "has_lodo_decisions": loaded["has_lodo"],
        "class_balance": class_balance,
        "lodo_missing_classes": lodo_missing,
        "binary_jpeg_jxl": binary_block,
        "hevc_case_study": hevc_case_study,
        "surrogate_tree_sweep": surrogate_sweep,
        "logistic_pairwise_interactions": logistic_report,
        "feature_attribution": attribution_report,
        "interpretation": interpretation,
        "provenance": {
            "surrogate_target": surrogate_sweep.get("target"),
            "feature_attribution_target": attribution_report.get("target"),
            "logistic_skipped": bool(logistic_report.get("warnings")),
            "shap_used": False,
        },
    }
    return report


def _write_outputs(
    report: Dict[str, Any],
    *,
    out_dir: Path,
    out_json: Optional[Path],
    out_summary_csv: Optional[Path],
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    class_balance_path = out_dir / "content_predictor_class_balance.csv"
    _write_class_balance_csv(
        class_balance_path,
        report["class_balance"],
        report["lodo_missing_classes"],
    )

    tree_csv = out_dir / "content_predictor_tree_surrogates.csv"
    _write_tree_surrogate_csv(tree_csv, report["surrogate_tree_sweep"])
    rule_paths = _write_tree_rules_files(
        out_dir,
        report["surrogate_tree_sweep"],
    )

    logistic_csv = out_dir / "content_predictor_logistic_interactions.csv"
    _write_logistic_csv(logistic_csv, report["logistic_pairwise_interactions"])

    attribution_csv = out_dir / "content_predictor_feature_attribution.csv"
    _write_attribution_csv(attribution_csv, report["feature_attribution"])

    json_path = out_json or (out_dir / "content_predictor_interpretability.json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )

    summary_path = (
        out_summary_csv
        if out_summary_csv is not None
        else (out_dir / "content_predictor_interpretability_summary.csv")
    )
    summary_rows = [
        {"section": "class_balance", "key": "hevc_count",
         "value": report["class_balance"]["hevc_count"]},
        {"section": "binary_jpeg_jxl", "key": "num_samples",
         "value": report["binary_jpeg_jxl"]["num_samples"]},
    ]
    for entry in report["surrogate_tree_sweep"].get("depth_rows", []):
        summary_rows.append(
            {
                "section": f"surrogate_depth_{entry['max_depth']}",
                "key": "fidelity_to_knn",
                "value": _fmt(entry.get("fidelity_to_knn"), 5),
            }
        )
        summary_rows.append(
            {
                "section": f"surrogate_depth_{entry['max_depth']}",
                "key": "fidelity_to_oracle",
                "value": _fmt(entry.get("fidelity_to_oracle"), 5),
            }
        )
    _write_csv(summary_path, summary_rows)

    return {
        "json": str(json_path),
        "summary_csv": str(summary_path),
        "class_balance_csv": str(class_balance_path),
        "tree_surrogates_csv": str(tree_csv),
        "logistic_csv": str(logistic_csv),
        "attribution_csv": str(attribution_csv),
        "tree_rules": rule_paths,
    }


def main(argv: Optional[List[str]] = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    report = _run(args)
    out_dir = Path(args.out_dir)
    out_json = Path(args.out_json) if args.out_json else None
    out_summary_csv = (
        Path(args.out_summary_csv) if args.out_summary_csv else None
    )
    written = _write_outputs(
        report,
        out_dir=out_dir,
        out_json=out_json,
        out_summary_csv=out_summary_csv,
    )

    print("\n=== R-D-E Content Predictor Interpretability ===")
    print(f"Rows analysed:             {report['num_rows']}")
    print(
        "Binary (JPEG vs JXL):      "
        f"{report['binary_jpeg_jxl']['num_samples']} rows"
    )
    print(
        "HEVC case study:           "
        f"{report['hevc_case_study']['num_samples']} rows"
    )
    print(f"Surrogate target:          {report['provenance']['surrogate_target']}")
    print(f"Tree depths swept:         {args.tree_depths}")
    print(f"Output directory:          {written['json']}")
    for note in report["interpretation"]:
        print(f"- {note}")


if __name__ == "__main__":
    main()
