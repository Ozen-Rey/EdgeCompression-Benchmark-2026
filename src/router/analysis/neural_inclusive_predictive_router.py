"""Neural-inclusive *predictive router* evaluation (offline / paper-facing).

v0.43.3 computed the per-image full-pool oracle and answered the
*theoretical* question: "where, and under which profile, are neural
codecs oracle-optimal in this benchmark?". This module answers the
*operational* question: "can a lightweight routing policy anticipate
those cases *before* the compression, using only training-time
information?".

The methodological rule that defines the module is strict:

    A policy is allowed to use the test image's pixel metadata
    (megapixels, aspect_ratio, resolution_class, orientation_class)
    and the test image's dataset/source label. It is **not** allowed
    to look at the test image's measured R-D-E candidates while
    choosing which codec/configuration to run. The R-D-E candidates
    of the test image may be used only afterwards: to look up the
    *realised* J_RDE of the predicted candidate, to compute the
    test image's oracle, and to compute regret. Quality floor
    violations on the test image are reported but **never** trigger
    a retroactive re-selection.

Five policies are evaluated:

A. ``robust_global_full_pool_baseline`` — pick a single
   (codec, config) from training (lowest mean J_RDE under full
   training coverage), apply uniformly on test images.
B. ``source_aware_full_pool_majority`` — per source/dataset majority
   vote of the per-training-image full-pool oracle; fallback to the
   global baseline for unknown sources.
C. ``knn_metadata_full_pool`` — kNN on the per-training-image
   full-pool oracle labels, encoded over metadata features.
D. ``classic_only_predictive_baseline`` — same kNN as C, but the
   training labels are the classical-pool oracle (not the full pool).
   Quantifies how much regret a classic-only predictor incurs
   against the full-pool oracle.
E. ``full_pool_oracle`` — upper-bound reference; not a router.

Two protocols:

- ``loio`` (leave-one-image-out): training = all images except the
  test image.
- ``lodo`` (leave-one-dataset-out): training = all images whose
  dataset differs from the test image's dataset. Stricter
  generalization test because the entire target source is held out.

For every (policy, protocol, profile, quality_floor) combination, the
module reports per-image decisions, summary statistics with paired
bootstrap CIs for ``mean_regret`` and
``relative_reduction_vs_global``, plus neural-family precision and
recall against the full-pool oracle's neural-vs-classical label.

The module does not change the router runtime, the ranking score, the
operational report schema, the rde_router CLI, or the benchmark raw
data. It is read-only against the existing R-D-E CSV and optionally a
metadata-features CSV; when the metadata file is omitted, features
are derived from the rde-csv's ``width`` and ``height`` columns.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Optional, Tuple

from src.router.adaptation.content_metadata_features import (
    _orientation_class,
    _resolution_class,
)
from src.router.analysis.neural_inclusive_oracle import (
    _GLOBAL_GROUP_KEY,
    classify_codec_family,
    load_pool_rows,
    normalize_full_pool,
)
from src.router.core.profiles import PROFILES as ROUTER_PROFILES


__all__ = [
    "evaluate_predictive_router",
    "build_per_image_oracle_labels",
    "policy_robust_global",
    "policy_source_aware_majority",
    "policy_knn_metadata",
    "bootstrap_paired_ci",
    "build_interpretation",
    "main",
]


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


Pair = Tuple[str, str]


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


def _quantile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    s = sorted(values)
    q = max(0.0, min(1.0, q))
    pos = (len(s) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    if lo == hi:
        return s[lo]
    frac = pos - lo
    return s[lo] * (1.0 - frac) + s[hi] * frac


def _fmt(value: Optional[float], digits: int = 6) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}g}"


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _resolve_profile_weights(name: str) -> Dict[str, float]:
    key = name.strip().lower().replace("_", "-")
    if key not in ROUTER_PROFILES:
        raise ValueError(
            f"Unknown profile '{name}'. Known profiles: "
            f"{', '.join(sorted(ROUTER_PROFILES))}"
        )
    profile = ROUTER_PROFILES[key]
    total = profile.w_e + profile.w_r + profile.w_d
    if total <= 0:
        raise ValueError(f"Profile '{name}' has non-positive weight sum.")
    return {
        "w_R": profile.w_r / total,
        "w_E": profile.w_e / total,
        "w_D": profile.w_d / total,
    }


def _j_rde(row: Dict[str, Any], weights: Dict[str, float]) -> float:
    return (
        weights["w_R"] * row["norm_rate"]
        + weights["w_E"] * row["norm_energy"]
        + weights["w_D"] * row["norm_distortion"]
    )


def _pair(codec: Any, config: Any) -> Pair:
    return str(codec), str(config)


def _label(pair: Pair) -> str:
    return f"{pair[0]}|{pair[1]}"


def _split_label(label: str) -> Pair:
    if "|" not in label:
        raise ValueError(f"Invalid codec|config label: {label!r}")
    codec, config = label.split("|", 1)
    return codec, config


# ---------------------------------------------------------------------------
# Metadata features
# ---------------------------------------------------------------------------


_METADATA_NUMERIC = ["megapixels", "aspect_ratio"]
_METADATA_CATEGORICAL = ["resolution_class", "orientation_class"]


def _derive_metadata_from_rde_rows(
    rows: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Compute per-image metadata features from the R-D-E CSV's raw fields."""
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        image_id = row["image_id"]
        if image_id in out or image_id == _GLOBAL_GROUP_KEY:
            continue
        raw = row.get("raw") or {}
        width = _to_float(raw.get("width"))
        height = _to_float(raw.get("height"))
        if width is None or height is None or height <= 0:
            continue
        megapixels = width * height / 1_000_000.0
        aspect_ratio = width / height
        out[image_id] = {
            "image_id": image_id,
            "dataset": row.get("dataset"),
            "megapixels": megapixels,
            "aspect_ratio": aspect_ratio,
            "resolution_class": _resolution_class(megapixels),
            "orientation_class": _orientation_class(width, height),
        }
    return out


def _load_metadata_features_csv(path: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    with Path(path).open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            image_id = str(raw.get("image_id", "")).strip()
            if not image_id:
                continue
            megapixels = _to_float(raw.get("megapixels"))
            aspect_ratio = _to_float(raw.get("aspect_ratio"))
            resolution = str(raw.get("resolution_class", "unknown"))
            orientation = str(raw.get("orientation_class", "unknown"))
            out[image_id] = {
                "image_id": image_id,
                "dataset": raw.get("dataset"),
                "megapixels": megapixels,
                "aspect_ratio": aspect_ratio,
                "resolution_class": resolution,
                "orientation_class": orientation,
            }
    if not out:
        raise ValueError(f"No metadata rows loaded from {path}")
    return out


def _build_metadata_columns(
    meta_rows: List[Dict[str, Any]],
) -> Tuple[List[str], Dict[str, List[str]]]:
    levels: Dict[str, List[str]] = {}
    for feat in _METADATA_CATEGORICAL:
        levels[feat] = sorted({str(m.get(feat, "unknown")) for m in meta_rows})
    columns: List[str] = list(_METADATA_NUMERIC)
    for feat in _METADATA_CATEGORICAL:
        for value in levels[feat]:
            columns.append(f"{feat}={value}")
    return columns, levels


def _fit_numeric_stats(
    meta_rows: List[Dict[str, Any]],
) -> Dict[str, Tuple[float, float]]:
    stats: Dict[str, Tuple[float, float]] = {}
    for feat in _METADATA_NUMERIC:
        values = [
            _to_float(m.get(feat))
            for m in meta_rows
            if _to_float(m.get(feat)) is not None
        ]
        if not values:
            stats[feat] = (0.0, 1.0)
            continue
        mu = mean(values)
        var = mean((x - mu) ** 2 for x in values)
        sigma = math.sqrt(var)
        if sigma <= 1e-12:
            sigma = 1.0
        stats[feat] = (mu, sigma)
    return stats


def _encode_metadata(
    meta: Dict[str, Any],
    *,
    numeric_stats: Dict[str, Tuple[float, float]],
    categorical_levels: Dict[str, List[str]],
) -> List[float]:
    vector: List[float] = []
    for feat in _METADATA_NUMERIC:
        value = _to_float(meta.get(feat))
        mu, sigma = numeric_stats[feat]
        vector.append(0.0 if value is None else (value - mu) / sigma)
    for feat in _METADATA_CATEGORICAL:
        current = str(meta.get(feat, "unknown"))
        for level in categorical_levels[feat]:
            vector.append(1.0 if current == level else 0.0)
    return vector


# ---------------------------------------------------------------------------
# Pool definitions
# ---------------------------------------------------------------------------


def _filter_pool(
    rows: List[Dict[str, Any]],
    *,
    pool: str,
) -> List[Dict[str, Any]]:
    if pool == "full_pool":
        return list(rows)
    if pool == "classic_pool":
        return [r for r in rows if r["codec_family"] == "classical"]
    if pool == "neural_pool":
        return [r for r in rows if r["codec_family"] == "neural"]
    raise ValueError(f"Unknown pool '{pool}'.")


# ---------------------------------------------------------------------------
# Per-image oracle labels (training-only when applied properly)
# ---------------------------------------------------------------------------


def _oracle_for_image(
    candidates: List[Dict[str, Any]],
    *,
    weights: Dict[str, float],
    quality_floor: Optional[float],
) -> Optional[Dict[str, Any]]:
    feasible = [
        r for r in candidates
        if quality_floor is None or r["quality"] >= quality_floor
    ]
    if not feasible:
        return None
    best: Optional[Dict[str, Any]] = None
    best_cost = math.inf
    for row in feasible:
        cost = _j_rde(row, weights)
        if cost < best_cost:
            best_cost = cost
            best = row
    if best is None:
        return None
    return {
        "image_id": best["image_id"],
        "dataset": best["dataset"],
        "codec": best["codec"],
        "config": best["config"],
        "family": best["codec_family"],
        "rate": best["rate"],
        "quality": best["quality"],
        "energy": best["energy"],
        "J_RDE": best_cost,
        "pair": _pair(best["codec"], best["config"]),
    }


def build_per_image_oracle_labels(
    rows: List[Dict[str, Any]],
    *,
    pool: str,
    weights: Dict[str, float],
    quality_floor: Optional[float],
) -> Dict[str, Optional[Dict[str, Any]]]:
    """For each image in ``rows``, compute its oracle within the given pool."""
    pool_rows = _filter_pool(rows, pool=pool)
    by_image: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in pool_rows:
        by_image[r["image_id"]].append(r)
    out: Dict[str, Optional[Dict[str, Any]]] = {}
    for image_id, candidates in by_image.items():
        out[image_id] = _oracle_for_image(
            candidates,
            weights=weights,
            quality_floor=quality_floor,
        )
    return out


# ---------------------------------------------------------------------------
# Realization on the test image (the only legal use of test R-D-E data
# during evaluation: look up a predicted pair's measured J_RDE)
# ---------------------------------------------------------------------------


def _build_candidate_index(
    rows: List[Dict[str, Any]],
) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    index: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for row in rows:
        key = (row["image_id"], row["codec"], row["config"])
        index[key] = row
    return index


def _realize_on_test_image(
    *,
    image_id: str,
    predicted_pair: Pair,
    candidate_index: Dict[Tuple[str, str, str], Dict[str, Any]],
    weights: Dict[str, float],
    quality_floor: Optional[float],
) -> Dict[str, Any]:
    """Look up the predicted pair's measured J_RDE on the test image.

    Quality floor violations are reported as a flag but never trigger
    re-selection; that would leak target information into the policy.
    """
    row = candidate_index.get((image_id, predicted_pair[0], predicted_pair[1]))
    if row is None:
        return {
            "found": False,
            "selected_J": None,
            "rate": None,
            "quality": None,
            "energy": None,
            "quality_violation": False,
        }
    return {
        "found": True,
        "selected_J": _j_rde(row, weights),
        "rate": row["rate"],
        "quality": row["quality"],
        "energy": row["energy"],
        "quality_violation": (
            quality_floor is not None and row["quality"] < quality_floor
        ),
    }


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------


def policy_robust_global(
    *,
    training_rows: List[Dict[str, Any]],
    pool: str,
    weights: Dict[str, float],
    quality_floor: Optional[float],
) -> Optional[Pair]:
    """Pick the (codec, config) with lowest mean training J_RDE under full coverage."""
    pool_rows = _filter_pool(training_rows, pool=pool)
    if not pool_rows:
        return None

    training_images = {r["image_id"] for r in pool_rows}
    if not training_images:
        return None

    grouped: Dict[Pair, List[Dict[str, Any]]] = defaultdict(list)
    for r in pool_rows:
        grouped[(r["codec"], r["config"])].append(r)

    best: Optional[Pair] = None
    best_cost = math.inf
    for pair, candidates in grouped.items():
        feasible = [
            c for c in candidates
            if quality_floor is None or c["quality"] >= quality_floor
        ]
        covered_images = {c["image_id"] for c in feasible}
        if covered_images != training_images:
            continue
        cost = mean(_j_rde(c, weights) for c in feasible)
        if cost < best_cost:
            best_cost = cost
            best = pair

    if best is not None:
        return best

    # Soft fallback: pick the pair with the most coverage and lowest mean J,
    # because under harsh floors no pair may meet full coverage. The
    # selected pair is still derived strictly from training data.
    best_score: Optional[Tuple[int, float]] = None
    for pair, candidates in grouped.items():
        feasible = [
            c for c in candidates
            if quality_floor is None or c["quality"] >= quality_floor
        ]
        if not feasible:
            continue
        coverage = len({c["image_id"] for c in feasible})
        cost = mean(_j_rde(c, weights) for c in feasible)
        score = (coverage, -cost)
        if best_score is None or score > best_score:
            best_score = score
            best = pair
    return best


def policy_source_aware_majority(
    *,
    training_rows: List[Dict[str, Any]],
    pool: str,
    weights: Dict[str, float],
    quality_floor: Optional[float],
) -> Tuple[Dict[str, Pair], Optional[Pair]]:
    """Build ``{source -> (codec, config)}`` from the per-image training oracle."""
    oracle_labels = build_per_image_oracle_labels(
        training_rows,
        pool=pool,
        weights=weights,
        quality_floor=quality_floor,
    )
    per_dataset: Dict[str, Counter] = defaultdict(Counter)
    image_to_dataset: Dict[str, str] = {}
    for r in training_rows:
        image_to_dataset[r["image_id"]] = str(r.get("dataset"))
    for image_id, oracle in oracle_labels.items():
        if oracle is None:
            continue
        dataset = image_to_dataset.get(image_id, "unknown")
        per_dataset[dataset][oracle["pair"]] += 1

    rules: Dict[str, Pair] = {}
    for dataset, counter in per_dataset.items():
        if counter:
            rules[dataset] = sorted(
                counter.items(),
                key=lambda item: (-item[1], item[0][0], item[0][1]),
            )[0][0]

    global_baseline = policy_robust_global(
        training_rows=training_rows,
        pool=pool,
        weights=weights,
        quality_floor=quality_floor,
    )
    return rules, global_baseline


def _knn_predict_label(
    *,
    test_meta: Dict[str, Any],
    training_metas: List[Dict[str, Any]],
    training_labels: List[str],
    columns: List[str],
    levels: Dict[str, List[str]],
    numeric_stats: Dict[str, Tuple[float, float]],
    k: int,
) -> Tuple[str, float]:
    """Predict an oracle label by kNN over metadata, with confidence as vote share."""
    if not training_metas or not training_labels:
        raise ValueError("Cannot predict kNN label on empty training set.")
    test_vec = _encode_metadata(
        test_meta,
        numeric_stats=numeric_stats,
        categorical_levels=levels,
    )
    distances: List[Tuple[float, str]] = []
    for meta, label in zip(training_metas, training_labels):
        train_vec = _encode_metadata(
            meta,
            numeric_stats=numeric_stats,
            categorical_levels=levels,
        )
        dsq = sum((a - b) ** 2 for a, b in zip(test_vec, train_vec))
        distances.append((dsq, label))
    distances.sort(key=lambda item: item[0])
    nearest = distances[: min(k, len(distances))]
    votes = Counter(label for _, label in nearest)
    top = sorted(
        votes.items(),
        key=lambda item: (-item[1], item[0]),
    )[0]
    confidence = top[1] / len(nearest)
    return top[0], confidence


def policy_knn_metadata(
    *,
    training_rows: List[Dict[str, Any]],
    metadata_by_image: Dict[str, Dict[str, Any]],
    pool: str,
    weights: Dict[str, float],
    quality_floor: Optional[float],
    test_meta: Dict[str, Any],
    k: int,
) -> Tuple[Optional[Pair], Optional[float], Optional[Pair]]:
    """Return ``(predicted_pair, confidence, training_global_baseline)``."""
    oracle_labels = build_per_image_oracle_labels(
        training_rows,
        pool=pool,
        weights=weights,
        quality_floor=quality_floor,
    )
    training_metas: List[Dict[str, Any]] = []
    training_labels: List[str] = []
    for image_id, oracle in oracle_labels.items():
        if oracle is None:
            continue
        meta = metadata_by_image.get(image_id)
        if meta is None:
            continue
        training_metas.append(meta)
        training_labels.append(_label(oracle["pair"]))

    global_baseline = policy_robust_global(
        training_rows=training_rows,
        pool=pool,
        weights=weights,
        quality_floor=quality_floor,
    )

    if not training_metas:
        return None, None, global_baseline

    columns, levels = _build_metadata_columns(training_metas + [test_meta])
    numeric_stats = _fit_numeric_stats(training_metas)

    label, confidence = _knn_predict_label(
        test_meta=test_meta,
        training_metas=training_metas,
        training_labels=training_labels,
        columns=columns,
        levels=levels,
        numeric_stats=numeric_stats,
        k=k,
    )
    predicted_pair = _split_label(label)
    return predicted_pair, confidence, global_baseline


# ---------------------------------------------------------------------------
# LOIO / LODO folds
# ---------------------------------------------------------------------------


def _loio_training_rows(
    rows: List[Dict[str, Any]],
    *,
    test_image_id: str,
) -> List[Dict[str, Any]]:
    return [r for r in rows if r["image_id"] != test_image_id]


def _lodo_training_rows(
    rows: List[Dict[str, Any]],
    *,
    test_dataset: str,
) -> List[Dict[str, Any]]:
    return [r for r in rows if str(r.get("dataset")) != str(test_dataset)]


# ---------------------------------------------------------------------------
# Bootstrap (paired)
# ---------------------------------------------------------------------------


def bootstrap_paired_ci(
    *,
    policy_regrets: List[float],
    baseline_regrets: List[float],
    iterations: int,
    seed: int,
    ci_low_q: float = 0.025,
    ci_high_q: float = 0.975,
) -> Dict[str, Optional[float]]:
    n = len(policy_regrets)
    if n == 0 or n != len(baseline_regrets):
        return {
            "mean_regret_ci_low": None,
            "mean_regret_ci_high": None,
            "relative_reduction_ci_low": None,
            "relative_reduction_ci_high": None,
        }
    rng = random.Random(seed)
    mean_samples: List[float] = []
    rel_samples: List[float] = []
    for _ in range(iterations):
        indices = [rng.randrange(n) for _ in range(n)]
        sampled_policy = [policy_regrets[i] for i in indices]
        sampled_baseline = [baseline_regrets[i] for i in indices]
        m_policy = mean(sampled_policy)
        m_baseline = mean(sampled_baseline)
        mean_samples.append(m_policy)
        if m_baseline > 0.0:
            rel_samples.append((m_baseline - m_policy) / m_baseline)
    return {
        "mean_regret_ci_low": _quantile(mean_samples, ci_low_q),
        "mean_regret_ci_high": _quantile(mean_samples, ci_high_q),
        "relative_reduction_ci_low": _quantile(rel_samples, ci_low_q),
        "relative_reduction_ci_high": _quantile(rel_samples, ci_high_q),
    }


# ---------------------------------------------------------------------------
# Decision row + summary
# ---------------------------------------------------------------------------


def _make_decision_row(
    *,
    image_id: str,
    dataset: Any,
    protocol: str,
    profile: str,
    quality_floor: Optional[float],
    policy_name: str,
    predicted_pair: Optional[Pair],
    confidence: Optional[float],
    fallback_used: bool,
    fallback_reason: str,
    candidate_index: Dict[Tuple[str, str, str], Dict[str, Any]],
    weights: Dict[str, float],
    quality_floor_value: Optional[float],
    oracle: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if predicted_pair is None:
        return {
            "image_id": image_id,
            "dataset": dataset,
            "protocol": protocol,
            "profile": profile,
            "quality_floor": quality_floor,
            "policy_name": policy_name,
            "predicted_codec": None,
            "predicted_config": None,
            "predicted_family": None,
            "oracle_codec": oracle["codec"] if oracle else None,
            "oracle_config": oracle["config"] if oracle else None,
            "oracle_family": oracle["family"] if oracle else None,
            "selected_J_on_test": None,
            "oracle_J_on_test": oracle["J_RDE"] if oracle else None,
            "regret": None,
            "family_match": None,
            "exact_match": None,
            "confidence": confidence,
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason,
            "quality_violation": False,
            "provenance": "no_predicted_pair_available",
        }

    realization = _realize_on_test_image(
        image_id=image_id,
        predicted_pair=predicted_pair,
        candidate_index=candidate_index,
        weights=weights,
        quality_floor=quality_floor_value,
    )
    predicted_family = classify_codec_family(predicted_pair[0])

    regret = None
    if (
        realization["selected_J"] is not None
        and oracle is not None
        and oracle["J_RDE"] is not None
    ):
        regret = realization["selected_J"] - oracle["J_RDE"]

    return {
        "image_id": image_id,
        "dataset": dataset,
        "protocol": protocol,
        "profile": profile,
        "quality_floor": quality_floor,
        "policy_name": policy_name,
        "predicted_codec": predicted_pair[0],
        "predicted_config": predicted_pair[1],
        "predicted_family": predicted_family,
        "oracle_codec": oracle["codec"] if oracle else None,
        "oracle_config": oracle["config"] if oracle else None,
        "oracle_family": oracle["family"] if oracle else None,
        "selected_J_on_test": realization["selected_J"],
        "oracle_J_on_test": oracle["J_RDE"] if oracle else None,
        "regret": regret,
        "family_match": (
            oracle is not None and predicted_family == oracle["family"]
        ),
        "exact_match": (
            oracle is not None
            and predicted_pair[0] == oracle["codec"]
            and predicted_pair[1] == oracle["config"]
        ),
        "confidence": confidence,
        "fallback_used": fallback_used,
        "fallback_reason": fallback_reason,
        "quality_violation": realization["quality_violation"],
        "provenance": (
            "predicted_pair_realised_on_test_image"
            if realization["found"]
            else "predicted_pair_not_present_on_test_image"
        ),
    }


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------


_POLICIES = [
    "robust_global_full_pool_baseline",
    "source_aware_full_pool_majority",
    "knn_metadata_full_pool",
    "classic_only_predictive_baseline",
    "full_pool_oracle",
]


def _policy_pool(name: str) -> str:
    if name == "classic_only_predictive_baseline":
        return "classic_pool"
    return "full_pool"


def evaluate_predictive_router(
    *,
    rows: List[Dict[str, Any]],
    metadata_by_image: Dict[str, Dict[str, Any]],
    profile_names: List[str],
    quality_floors: List[Optional[float]],
    protocols: List[str],
    k: int,
    bootstrap_iterations: int,
    seed: int,
) -> Dict[str, Any]:
    candidate_index = _build_candidate_index(rows)
    image_ids = sorted({r["image_id"] for r in rows})
    image_to_dataset: Dict[str, str] = {}
    for r in rows:
        image_to_dataset.setdefault(r["image_id"], str(r.get("dataset")))

    profile_weights = {
        name: _resolve_profile_weights(name) for name in profile_names
    }

    decisions: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []

    for protocol in protocols:
        if protocol not in {"loio", "lodo"}:
            raise ValueError(f"Unknown protocol '{protocol}'.")

        for profile in profile_names:
            weights = profile_weights[profile]
            for floor in quality_floors:
                full_oracle_by_image = build_per_image_oracle_labels(
                    rows,
                    pool="full_pool",
                    weights=weights,
                    quality_floor=floor,
                )

                for policy in _POLICIES:
                    pool = _policy_pool(policy)
                    policy_decisions: List[Dict[str, Any]] = []
                    for image_id in image_ids:
                        dataset = image_to_dataset[image_id]
                        oracle = full_oracle_by_image.get(image_id)

                        if protocol == "loio":
                            training_rows = _loio_training_rows(
                                rows, test_image_id=image_id
                            )
                        else:
                            training_rows = _lodo_training_rows(
                                rows, test_dataset=dataset
                            )

                        decision = _evaluate_single(
                            policy=policy,
                            pool=pool,
                            image_id=image_id,
                            dataset=dataset,
                            protocol=protocol,
                            profile=profile,
                            quality_floor=floor,
                            training_rows=training_rows,
                            metadata_by_image=metadata_by_image,
                            candidate_index=candidate_index,
                            weights=weights,
                            k=k,
                            oracle=oracle,
                        )
                        policy_decisions.append(decision)
                        decisions.append(decision)

                    summaries.append(
                        _summarise_policy(
                            policy_decisions,
                            policy_name=policy,
                            protocol=protocol,
                            profile=profile,
                            quality_floor=floor,
                        )
                    )

    _attach_bootstrap_cis(
        summaries=summaries,
        decisions=decisions,
        iterations=bootstrap_iterations,
        seed=seed,
    )

    return {
        "decisions": decisions,
        "summaries": summaries,
        "profile_weights": profile_weights,
    }


def _evaluate_single(
    *,
    policy: str,
    pool: str,
    image_id: str,
    dataset: str,
    protocol: str,
    profile: str,
    quality_floor: Optional[float],
    training_rows: List[Dict[str, Any]],
    metadata_by_image: Dict[str, Dict[str, Any]],
    candidate_index: Dict[Tuple[str, str, str], Dict[str, Any]],
    weights: Dict[str, float],
    k: int,
    oracle: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if policy == "full_pool_oracle":
        predicted_pair = oracle["pair"] if oracle else None
        return _make_decision_row(
            image_id=image_id,
            dataset=dataset,
            protocol=protocol,
            profile=profile,
            quality_floor=quality_floor,
            policy_name=policy,
            predicted_pair=predicted_pair,
            confidence=1.0 if predicted_pair else None,
            fallback_used=False,
            fallback_reason="oracle_reference_no_fallback",
            candidate_index=candidate_index,
            weights=weights,
            quality_floor_value=quality_floor,
            oracle=oracle,
        )

    if policy == "robust_global_full_pool_baseline":
        baseline = policy_robust_global(
            training_rows=training_rows,
            pool=pool,
            weights=weights,
            quality_floor=quality_floor,
        )
        return _make_decision_row(
            image_id=image_id,
            dataset=dataset,
            protocol=protocol,
            profile=profile,
            quality_floor=quality_floor,
            policy_name=policy,
            predicted_pair=baseline,
            confidence=None,
            fallback_used=False,
            fallback_reason="" if baseline else "no_training_baseline",
            candidate_index=candidate_index,
            weights=weights,
            quality_floor_value=quality_floor,
            oracle=oracle,
        )

    if policy == "source_aware_full_pool_majority":
        rules, fallback = policy_source_aware_majority(
            training_rows=training_rows,
            pool=pool,
            weights=weights,
            quality_floor=quality_floor,
        )
        predicted = rules.get(str(dataset))
        if predicted is None:
            return _make_decision_row(
                image_id=image_id,
                dataset=dataset,
                protocol=protocol,
                profile=profile,
                quality_floor=quality_floor,
                policy_name=policy,
                predicted_pair=fallback,
                confidence=None,
                fallback_used=True,
                fallback_reason="source_not_in_training_rules",
                candidate_index=candidate_index,
                weights=weights,
                quality_floor_value=quality_floor,
                oracle=oracle,
            )
        return _make_decision_row(
            image_id=image_id,
            dataset=dataset,
            protocol=protocol,
            profile=profile,
            quality_floor=quality_floor,
            policy_name=policy,
            predicted_pair=predicted,
            confidence=None,
            fallback_used=False,
            fallback_reason="",
            candidate_index=candidate_index,
            weights=weights,
            quality_floor_value=quality_floor,
            oracle=oracle,
        )

    if policy in {"knn_metadata_full_pool", "classic_only_predictive_baseline"}:
        test_meta = metadata_by_image.get(image_id)
        if test_meta is None:
            return _make_decision_row(
                image_id=image_id,
                dataset=dataset,
                protocol=protocol,
                profile=profile,
                quality_floor=quality_floor,
                policy_name=policy,
                predicted_pair=None,
                confidence=None,
                fallback_used=True,
                fallback_reason="no_metadata_for_test_image",
                candidate_index=candidate_index,
                weights=weights,
                quality_floor_value=quality_floor,
                oracle=oracle,
            )
        predicted, confidence, fallback = policy_knn_metadata(
            training_rows=training_rows,
            metadata_by_image=metadata_by_image,
            pool=pool,
            weights=weights,
            quality_floor=quality_floor,
            test_meta=test_meta,
            k=k,
        )
        used_fallback = predicted is None
        if used_fallback:
            predicted = fallback
        return _make_decision_row(
            image_id=image_id,
            dataset=dataset,
            protocol=protocol,
            profile=profile,
            quality_floor=quality_floor,
            policy_name=policy,
            predicted_pair=predicted,
            confidence=confidence,
            fallback_used=used_fallback,
            fallback_reason="knn_no_training_labels" if used_fallback else "",
            candidate_index=candidate_index,
            weights=weights,
            quality_floor_value=quality_floor,
            oracle=oracle,
        )

    raise ValueError(f"Unknown policy '{policy}'.")


# ---------------------------------------------------------------------------
# Summarisation
# ---------------------------------------------------------------------------


def _summarise_policy(
    decisions: List[Dict[str, Any]],
    *,
    policy_name: str,
    protocol: str,
    profile: str,
    quality_floor: Optional[float],
) -> Dict[str, Any]:
    num_images = len(decisions)
    regrets = [d["regret"] for d in decisions if d["regret"] is not None]
    mean_regret = mean(regrets) if regrets else None
    median_regret = median(regrets) if regrets else None
    p90_regret = _quantile(regrets, 0.90) if regrets else None
    max_regret = max(regrets) if regrets else None

    predicted_family_counts = Counter(
        d["predicted_family"] for d in decisions if d["predicted_family"]
    )
    oracle_family_counts = Counter(
        d["oracle_family"] for d in decisions if d["oracle_family"]
    )

    neural_predicted = predicted_family_counts.get("neural", 0)
    neural_oracle = oracle_family_counts.get("neural", 0)

    neural_selection_rate = (
        neural_predicted / num_images if num_images else None
    )
    oracle_neural_rate = (
        neural_oracle / num_images if num_images else None
    )

    # Precision/recall against neural oracle label.
    tp = sum(
        1 for d in decisions
        if d["predicted_family"] == "neural"
        and d["oracle_family"] == "neural"
    )
    fp = sum(
        1 for d in decisions
        if d["predicted_family"] == "neural"
        and d["oracle_family"] == "classical"
    )
    fn = sum(
        1 for d in decisions
        if d["predicted_family"] == "classical"
        and d["oracle_family"] == "neural"
    )

    neural_family_precision = tp / (tp + fp) if (tp + fp) > 0 else None
    neural_family_recall = tp / (tp + fn) if (tp + fn) > 0 else None

    exact_matches = sum(1 for d in decisions if d.get("exact_match") is True)
    family_matches = sum(1 for d in decisions if d.get("family_match") is True)
    fallbacks = sum(1 for d in decisions if d.get("fallback_used") is True)
    violations = sum(
        1 for d in decisions if d.get("quality_violation") is True
    )

    return {
        "policy_name": policy_name,
        "protocol": protocol,
        "profile": profile,
        "quality_floor": quality_floor,
        "num_images": num_images,
        "mean_regret": mean_regret,
        "median_regret": median_regret,
        "p90_regret": p90_regret,
        "max_regret": max_regret,
        "relative_reduction_vs_global": None,
        "neural_selection_rate": neural_selection_rate,
        "oracle_neural_rate": oracle_neural_rate,
        "neural_family_precision": neural_family_precision,
        "neural_family_recall": neural_family_recall,
        "exact_match_rate": exact_matches / num_images if num_images else None,
        "family_match_rate": family_matches / num_images if num_images else None,
        "fallback_rate": fallbacks / num_images if num_images else None,
        "quality_violation_rate": violations / num_images if num_images else None,
        "mean_regret_ci_low": None,
        "mean_regret_ci_high": None,
        "relative_reduction_ci_low": None,
        "relative_reduction_ci_high": None,
    }


def _attach_bootstrap_cis(
    *,
    summaries: List[Dict[str, Any]],
    decisions: List[Dict[str, Any]],
    iterations: int,
    seed: int,
) -> None:
    by_key: Dict[Tuple[str, str, str, Optional[float], str], List[Dict[str, Any]]] = (
        defaultdict(list)
    )
    for d in decisions:
        by_key[
            (
                d["protocol"],
                d["profile"],
                str(d["quality_floor"]),
                d["quality_floor"],
                d["policy_name"],
            )
        ].append(d)

    baseline_name = "robust_global_full_pool_baseline"

    for summary in summaries:
        key = (
            summary["protocol"],
            summary["profile"],
            str(summary["quality_floor"]),
            summary["quality_floor"],
            summary["policy_name"],
        )
        baseline_key = (
            summary["protocol"],
            summary["profile"],
            str(summary["quality_floor"]),
            summary["quality_floor"],
            baseline_name,
        )

        policy_decisions = by_key.get(key, [])
        baseline_decisions = by_key.get(baseline_key, [])
        baseline_by_image = {
            d["image_id"]: d for d in baseline_decisions
        }

        paired_policy: List[float] = []
        paired_baseline: List[float] = []
        for d in policy_decisions:
            b = baseline_by_image.get(d["image_id"])
            if (
                d["regret"] is None
                or b is None
                or b.get("regret") is None
            ):
                continue
            paired_policy.append(float(d["regret"]))
            paired_baseline.append(float(b["regret"]))

        if not paired_baseline:
            continue

        baseline_mean = mean(paired_baseline)
        policy_mean = mean(paired_policy)
        if baseline_mean > 0:
            summary["relative_reduction_vs_global"] = (
                (baseline_mean - policy_mean) / baseline_mean
            )

        ci = bootstrap_paired_ci(
            policy_regrets=paired_policy,
            baseline_regrets=paired_baseline,
            iterations=iterations,
            seed=seed,
        )
        summary.update(ci)


# ---------------------------------------------------------------------------
# Interpretation
# ---------------------------------------------------------------------------


def build_interpretation(
    *,
    summaries: List[Dict[str, Any]],
    profiles: List[str],
    quality_floors: List[Optional[float]],
    protocols: List[str],
) -> List[str]:
    notes: List[str] = []
    notes.append(
        "The full-pool oracle audit (v0.43.3) describes where neural "
        "codecs are theoretically optimal in this benchmark; this "
        "predictive evaluation measures whether lightweight routing "
        "policies can anticipate those cases before the compression."
    )

    knn_full = [
        s for s in summaries
        if s["policy_name"] == "knn_metadata_full_pool"
    ]
    for summary in knn_full:
        oracle_rate = summary.get("oracle_neural_rate")
        predicted_rate = summary.get("neural_selection_rate")
        if oracle_rate is None or predicted_rate is None:
            continue
        if oracle_rate > 0 and predicted_rate < oracle_rate * 0.5:
            notes.append(
                f"Under protocol='{summary['protocol']}', "
                f"profile='{summary['profile']}', "
                f"quality_floor={summary['quality_floor']}, the kNN "
                f"predictor proposes a neural codec for "
                f"{predicted_rate:.2%} of images while the oracle "
                f"selects neural for {oracle_rate:.2%}; the gap "
                "suggests the predictor is conservative or lacks "
                "sufficient signal in metadata features alone."
            )

    has_lodo = "lodo" in protocols
    if has_lodo:
        notes.append(
            "LODO is the stricter protocol because the target "
            "dataset is excluded from training; reductions that "
            "survive LODO are more indicative of cross-source "
            "generalization within this benchmark."
        )

    notes.append(
        "These observations describe patterns within the current "
        "image R-D-E benchmark and do not establish universal "
        "generalization or universal neural/classical dominance."
    )
    return notes


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


_DECISION_COLUMNS = [
    "image_id",
    "dataset",
    "protocol",
    "profile",
    "quality_floor",
    "policy_name",
    "predicted_codec",
    "predicted_config",
    "predicted_family",
    "oracle_codec",
    "oracle_config",
    "oracle_family",
    "selected_J_on_test",
    "oracle_J_on_test",
    "regret",
    "family_match",
    "exact_match",
    "confidence",
    "fallback_used",
    "fallback_reason",
    "quality_violation",
    "provenance",
]


_SUMMARY_COLUMNS = [
    "policy_name",
    "protocol",
    "profile",
    "quality_floor",
    "num_images",
    "mean_regret",
    "median_regret",
    "p90_regret",
    "max_regret",
    "relative_reduction_vs_global",
    "neural_selection_rate",
    "oracle_neural_rate",
    "neural_family_precision",
    "neural_family_recall",
    "exact_match_rate",
    "family_match_rate",
    "fallback_rate",
    "quality_violation_rate",
    "mean_regret_ci_low",
    "mean_regret_ci_high",
    "relative_reduction_ci_low",
    "relative_reduction_ci_high",
]


def _decision_to_row(decision: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for col in _DECISION_COLUMNS:
        value = decision.get(col)
        if value is None:
            out[col] = ""
        elif isinstance(value, bool):
            out[col] = "true" if value else "false"
        elif isinstance(value, float):
            out[col] = _fmt(value, 6)
        else:
            out[col] = str(value)
    return out


def _summary_to_row(summary: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for col in _SUMMARY_COLUMNS:
        value = summary.get(col)
        if value is None:
            out[col] = ""
        elif isinstance(value, bool):
            out[col] = "true" if value else "false"
        elif isinstance(value, float):
            out[col] = _fmt(value, 6)
        else:
            out[col] = str(value)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="src.router.analysis.neural_inclusive_predictive_router",
        description=(
            "Offline neural-inclusive predictive router evaluation. For "
            "each (policy, protocol, profile, quality_floor) the test "
            "image's measured R-D-E candidates are never used to "
            "choose the codec; they are used only to look up the "
            "realised J_RDE of the predicted pair and to compute "
            "regret against the per-image full-pool oracle. Read-only "
            "against existing CSV artefacts; does not change the "
            "router runtime."
        ),
    )

    parser.add_argument("--rde-csv", required=True)
    parser.add_argument("--metadata-features", default=None)

    parser.add_argument(
        "--out-dir",
        default="results/routing_context/neural_inclusive_predictive_router",
    )
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--out-summary-csv", default=None)
    parser.add_argument("--out-decisions-csv", default=None)

    parser.add_argument("--image-id-col", default="image_id")
    parser.add_argument("--dataset-col", default="dataset")
    parser.add_argument("--codec-col", default="codec")
    parser.add_argument("--config-col", default="param")
    parser.add_argument("--rate-col", default="bpp")
    parser.add_argument("--quality-col", default="ssimulacra2")
    parser.add_argument("--energy-col", default="energy_per_image_j")

    parser.add_argument(
        "--profiles",
        default="balanced,energy-limited,bandwidth-limited,quality-first",
    )
    parser.add_argument("--quality-floors", default="60,70,80")
    parser.add_argument("--protocols", default="loio,lodo")
    parser.add_argument("--k", type=int, default=7)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)

    return parser


def _parse_csv_list(value: str) -> List[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _parse_floor_list(value: str) -> List[Optional[float]]:
    out: List[Optional[float]] = []
    for token in _parse_csv_list(value):
        if token.lower() in {"none", "null", ""}:
            out.append(None)
            continue
        out.append(float(token))
    if not out:
        raise ValueError("--quality-floors must contain at least one entry.")
    return out


def _resolve_metadata(
    args: argparse.Namespace,
    rows: List[Dict[str, Any]],
) -> Tuple[Dict[str, Dict[str, Any]], str]:
    if args.metadata_features:
        return (
            _load_metadata_features_csv(args.metadata_features),
            "metadata_features_csv",
        )
    return _derive_metadata_from_rde_rows(rows), "derived_from_rde_csv"


def _run(args: argparse.Namespace) -> Dict[str, Any]:
    rows = load_pool_rows(
        args.rde_csv,
        codec_col=args.codec_col,
        config_col=args.config_col,
        rate_col=args.rate_col,
        quality_col=args.quality_col,
        energy_col=args.energy_col,
        image_id_col=args.image_id_col,
        dataset_col=args.dataset_col,
    )
    normalization_stats = normalize_full_pool(rows)

    metadata_by_image, metadata_source = _resolve_metadata(args, rows)

    profile_names = _parse_csv_list(args.profiles)
    quality_floors = _parse_floor_list(args.quality_floors)
    protocols = _parse_csv_list(args.protocols)

    evaluation = evaluate_predictive_router(
        rows=rows,
        metadata_by_image=metadata_by_image,
        profile_names=profile_names,
        quality_floors=quality_floors,
        protocols=protocols,
        k=args.k,
        bootstrap_iterations=args.bootstrap_iterations,
        seed=args.seed,
    )

    interpretation = build_interpretation(
        summaries=evaluation["summaries"],
        profiles=profile_names,
        quality_floors=quality_floors,
        protocols=protocols,
    )

    codec_inventory = Counter(
        f"{r['codec']}|{r['codec_family']}" for r in rows
    )

    report = {
        "generated_at_unix": int(time.time()),
        "router_target": "neural_inclusive_predictive_router",
        "inputs": {
            "rde_csv": args.rde_csv,
            "metadata_features": args.metadata_features,
            "image_id_col": args.image_id_col,
            "dataset_col": args.dataset_col,
            "codec_col": args.codec_col,
            "config_col": args.config_col,
            "rate_col": args.rate_col,
            "quality_col": args.quality_col,
            "energy_col": args.energy_col,
        },
        "num_rows_loaded": len(rows),
        "num_images": len({r["image_id"] for r in rows}),
        "codec_inventory": dict(codec_inventory),
        "normalization": {"scope": "full_pool_global", **normalization_stats},
        "profiles": profile_names,
        "profile_weights": evaluation["profile_weights"],
        "quality_floors": quality_floors,
        "protocols": protocols,
        "k": args.k,
        "bootstrap_iterations": args.bootstrap_iterations,
        "seed": args.seed,
        "summaries": evaluation["summaries"],
        "interpretation": interpretation,
        "provenance": {
            "metadata_source": metadata_source,
            "normalization_scope": "full_pool_global",
            "policy_does_not_see_test_image_rde": True,
            "test_image_rde_used_only_for_realisation_and_oracle": True,
            "quality_floor_violations_reported_not_corrected": True,
            "uses_router_profiles": True,
            "profile_source": "src.router.core.profiles.PROFILES",
        },
    }
    report["_decisions"] = evaluation["decisions"]
    return report


def _write_outputs(
    report: Dict[str, Any],
    *,
    out_dir: Path,
    out_json: Optional[Path],
    out_summary_csv: Optional[Path],
    out_decisions_csv: Optional[Path],
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    decisions_path = (
        out_decisions_csv
        if out_decisions_csv is not None
        else (out_dir / "neural_inclusive_predictive_router_decisions.csv")
    )
    _write_csv(decisions_path, [_decision_to_row(d) for d in report["_decisions"]])

    summary_path = (
        out_summary_csv
        if out_summary_csv is not None
        else (out_dir / "neural_inclusive_predictive_router_summary.csv")
    )
    _write_csv(summary_path, [_summary_to_row(s) for s in report["summaries"]])

    json_path = (
        out_json
        if out_json is not None
        else (out_dir / "neural_inclusive_predictive_router_report.json")
    )
    public_report = {k: v for k, v in report.items() if not k.startswith("_")}
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(public_report, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )

    return {
        "decisions_csv": str(decisions_path),
        "summary_csv": str(summary_path),
        "json": str(json_path),
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
    out_decisions_csv = (
        Path(args.out_decisions_csv) if args.out_decisions_csv else None
    )
    written = _write_outputs(
        report,
        out_dir=out_dir,
        out_json=out_json,
        out_summary_csv=out_summary_csv,
        out_decisions_csv=out_decisions_csv,
    )

    print("\n=== R-D-E Neural-Inclusive Predictive Router Evaluation ===")
    print(f"Rows loaded:           {report['num_rows_loaded']}")
    print(f"Images:                {report['num_images']}")
    print(f"Protocols:             {', '.join(report['protocols'])}")
    print(f"Profiles:              {', '.join(report['profiles'])}")
    print(f"k:                     {report['k']}")
    print(f"Bootstrap iterations:  {report['bootstrap_iterations']}")
    print(f"Decisions CSV:         {written['decisions_csv']}")
    print(f"Summary CSV:           {written['summary_csv']}")
    print(f"JSON report:           {written['json']}")
    for note in report["interpretation"]:
        print(f"- {note}")


if __name__ == "__main__":
    main()
