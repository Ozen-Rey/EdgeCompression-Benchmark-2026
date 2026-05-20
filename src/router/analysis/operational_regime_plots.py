"""Best-effort plot rendering for operational-regime simulation CSVs.

The simulation module writes scientific CSV/JSON artifacts only. This module
is a derived renderer: it can be rerun independently, skipped entirely, or run
in ``basic`` mode for quick reproducible figures.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Dict, List, Optional


CSV_FILES = {
    "plot_data": "operational_regime_plot_data.csv",
    "winners": "operational_regime_winner_distribution.csv",
    "oracle_prediction": "operational_regime_oracle_vs_prediction.csv",
    "confusion": "operational_regime_family_confusion.csv",
    "rate_pressure": "operational_regime_rate_pressure_sweep.csv",
    "switch_summary": "operational_regime_switch_summary.csv",
    "switch_reason": "switch_reason_by_rate_weight.csv",
    "tradeoff": "neural_vs_classic_tradeoff_scatter.csv",
    "quality_floor": "quality_floor_switch_summary.csv",
}

BASIC_PLOTS = [
    "energy_saving_vs_regret_reduction.png",
    "neural_selection_rate_by_regime.png",
    "winner_family_by_regime.png",
    "oracle_vs_predicted_neural_rate.png",
    "rate_weight_switch_boundary.png",
    "switch_reason_by_rate_weight.png",
    "quality_floor_vs_neural_necessity.png",
]

ALL_EXTRA_PLOTS = [
    "winner_codec_by_regime.png",
    "family_confusion_heatmap.png",
    "neural_vs_classic_tradeoff_scatter.png",
    "rate_pressure_family_shift.png",
    "rate_pressure_codec_shift.png",
    "rate_reduction_vs_energy_penalty_sweep.png",
    "quality_violation_by_regime.png",
]


def _read_csv(path: Path) -> Optional[List[Dict[str, str]]]:
    if not path.exists():
        return None
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _float(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    if not math.isfinite(out):
        return None
    return out


def _sample_rows(
    rows: List[Dict[str, str]],
    *,
    max_plot_points: int,
    seed: int,
) -> tuple[List[Dict[str, str]], bool]:
    if max_plot_points <= 0 or len(rows) <= max_plot_points:
        return rows, False
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(rows)), max_plot_points))
    return [rows[i] for i in indices], True


def _mean_metric(rows: List[Dict[str, str]], field: str) -> float:
    values = [_float(row.get(field)) for row in rows]
    values = [v for v in values if v is not None]
    return mean(values) if values else 0.0


def _require(data: Dict[str, Optional[List[Dict[str, str]]]], *keys: str) -> Optional[str]:
    for key in keys:
        if data.get(key) is None:
            return CSV_FILES[key]
    return None


def _save(plt: Any, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_energy_regret(plt: Any, data: Dict[str, Any], out_path: Path, max_plot_points: int, seed: int) -> Dict[str, Any]:
    rows, sampled = _sample_rows(data["plot_data"], max_plot_points=max_plot_points, seed=seed)
    plt.figure(figsize=(8, 5))
    for row in rows:
        x = _float(row.get("x_energy_saving"))
        y = _float(row.get("y_regret_reduction"))
        if x is None or y is None:
            continue
        size = 30 + 120 * max(0.0, min(1.0, _float(row.get("neural_selection_rate")) or 0.0))
        plt.scatter(x, y, s=size, alpha=0.65)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("Energy saving vs global baseline")
    plt.ylabel("Regret reduction vs global baseline")
    _save(plt, out_path)
    return {"sampled": sampled, "input_rows": len(data["plot_data"]), "plotted_rows": len(rows)}


def _plot_bar_by_regime(plt: Any, rows: List[Dict[str, str]], metric: str, out_path: Path) -> Dict[str, Any]:
    regimes = sorted({row.get("regime", "") for row in rows})
    policies = sorted({row.get("policy", "") for row in rows})
    width = 0.8 / max(1, len(policies))
    plt.figure(figsize=(10, 5))
    for p_idx, policy in enumerate(policies):
        xs: List[float] = []
        values: List[float] = []
        for r_idx, regime in enumerate(regimes):
            matches = [r for r in rows if r.get("policy") == policy and r.get("regime") == regime]
            xs.append(r_idx + p_idx * width)
            values.append(_mean_metric(matches, metric))
        plt.bar(xs, values, width=width, label=policy)
    plt.xticks([i + width for i in range(len(regimes))], regimes, rotation=35, ha="right")
    plt.ylabel(metric)
    plt.legend(fontsize=7)
    _save(plt, out_path)
    return {"sampled": False, "input_rows": len(rows), "plotted_rows": len(rows)}


def _plot_neural_selection(plt: Any, data: Dict[str, Any], out_path: Path, max_plot_points: int, seed: int) -> Dict[str, Any]:
    return _plot_bar_by_regime(plt, data["plot_data"], "neural_selection_rate", out_path)


def _plot_quality_violation(plt: Any, data: Dict[str, Any], out_path: Path, max_plot_points: int, seed: int) -> Dict[str, Any]:
    return _plot_bar_by_regime(plt, data["plot_data"], "quality_violation_rate", out_path)


def _plot_winner_stack(plt: Any, rows: List[Dict[str, str]], field: str, out_path: Path) -> Dict[str, Any]:
    filtered = [r for r in rows if r.get("policy") == "metadata_plus_system_full_pool"] or rows
    regimes = sorted({r.get("regime", "") for r in filtered})
    labels = sorted({r.get(field, "none") for r in filtered})
    bottoms = [0.0 for _ in regimes]
    plt.figure(figsize=(10, 5))
    for label in labels:
        values: List[float] = []
        for regime in regimes:
            values.append(
                sum(
                    _float(r.get("selection_rate")) or 0.0
                    for r in filtered
                    if r.get("regime") == regime and r.get(field) == label
                )
            )
        plt.bar(regimes, values, bottom=bottoms, label=label)
        bottoms = [b + v for b, v in zip(bottoms, values)]
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Selection rate")
    plt.legend(fontsize=7)
    _save(plt, out_path)
    return {"sampled": False, "input_rows": len(rows), "plotted_rows": len(filtered)}


def _plot_winner_family(plt: Any, data: Dict[str, Any], out_path: Path, max_plot_points: int, seed: int) -> Dict[str, Any]:
    return _plot_winner_stack(plt, data["winners"], "selected_family", out_path)


def _plot_winner_codec(plt: Any, data: Dict[str, Any], out_path: Path, max_plot_points: int, seed: int) -> Dict[str, Any]:
    return _plot_winner_stack(plt, data["winners"], "selected_codec", out_path)


def _plot_oracle_predicted(plt: Any, data: Dict[str, Any], out_path: Path, max_plot_points: int, seed: int) -> Dict[str, Any]:
    rows = [r for r in data["oracle_prediction"] if r.get("family") == "neural"]
    regimes = sorted({r.get("regime", "") for r in rows})
    xs = list(range(len(regimes)))
    oracle = [_mean_metric([r for r in rows if r.get("regime") == regime], "oracle_rate") for regime in regimes]
    pred = [_mean_metric([r for r in rows if r.get("regime") == regime], "predicted_rate") for regime in regimes]
    plt.figure(figsize=(9, 5))
    plt.bar([x - 0.2 for x in xs], oracle, width=0.4, label="oracle")
    plt.bar([x + 0.2 for x in xs], pred, width=0.4, label="predicted")
    plt.xticks(xs, regimes, rotation=35, ha="right")
    plt.ylabel("Neural rate")
    plt.legend()
    _save(plt, out_path)
    return {"sampled": False, "input_rows": len(data["oracle_prediction"]), "plotted_rows": len(rows)}


def _plot_rate_switch_boundary(plt: Any, data: Dict[str, Any], out_path: Path, max_plot_points: int, seed: int) -> Dict[str, Any]:
    # ``operational_regime_switch_summary.csv`` aggregates by
    # (regime, protocol, quality_floor); its rate_weight column is empty.
    # The actual rate_weight sweep lives in
    # ``operational_regime_rate_pressure_sweep.csv``, so we take the weight
    # axis from there. switch_summary still provides the neural_win_rate,
    # which is constant in rate_weight by construction (the switch analysis
    # is computed once per quality floor), so we plot it as a horizontal
    # reference line per quality floor.
    rp = data["rate_pressure"]
    rs = data["switch_summary"]
    weights = sorted({w for w in (_float(r.get("rate_weight")) for r in rp) if w is not None})
    if not weights:
        return {"sampled": False, "input_rows": len(rp), "plotted_rows": 0, "skipped": "no rate_weight values in rate_pressure CSV"}

    predicted = []
    oracle = []
    for w in weights:
        matches = [r for r in rp if _float(r.get("rate_weight")) == w and r.get("policy") == "metadata_plus_system_full_pool"]
        predicted.append(max([_float(r.get("neural_selection_rate")) or 0.0 for r in matches] or [0.0]))
        oracle.append(max([_float(r.get("oracle_neural_rate")) or 0.0 for r in matches] or [0.0]))

    floors = sorted({f for f in (_float(r.get("quality_floor")) for r in rs) if f is not None})

    plt.figure(figsize=(8, 5))
    plt.plot(weights, predicted, marker="s", label="predicted_neural_rate")
    plt.plot(weights, oracle, marker="^", label="oracle_neural_rate")
    for floor in floors:
        floor_rows = [r for r in rs if _float(r.get("quality_floor")) == floor]
        win_rate = _mean_metric(floor_rows, "neural_win_rate")
        plt.axhline(win_rate, linestyle="--", alpha=0.5, label=f"switch_neural_win_rate (Q≥{floor:g})")
    plt.xlabel("Rate weight")
    plt.ylabel("Neural selection / win rate")
    plt.legend(fontsize=8)
    _save(plt, out_path)
    return {"sampled": False, "input_rows": len(rp), "plotted_rows": len(weights)}


def _plot_switch_reason(plt: Any, data: Dict[str, Any], out_path: Path, max_plot_points: int, seed: int) -> Dict[str, Any]:
    rows = data["switch_reason"]
    weights = sorted({_float(r.get("rate_weight")) for r in rows if _float(r.get("rate_weight")) is not None})
    reasons = sorted({r.get("switch_reason", "") for r in rows})
    bottoms = [0.0 for _ in weights]
    plt.figure(figsize=(9, 5))
    for reason in reasons:
        values = [
            sum(
                _float(r.get("share")) or 0.0
                for r in rows
                if _float(r.get("rate_weight")) == weight and r.get("switch_reason") == reason
            )
            for weight in weights
        ]
        plt.bar(weights, values, width=0.06, bottom=bottoms, label=reason)
        bottoms = [b + v for b, v in zip(bottoms, values)]
    plt.xlabel("Rate weight")
    plt.ylabel("Share")
    plt.legend(fontsize=6)
    _save(plt, out_path)
    return {"sampled": False, "input_rows": len(rows), "plotted_rows": len(rows)}


def _plot_quality_floor(plt: Any, data: Dict[str, Any], out_path: Path, max_plot_points: int, seed: int) -> Dict[str, Any]:
    rows = data["quality_floor"]
    floors = sorted({_float(r.get("quality_floor")) for r in rows if _float(r.get("quality_floor")) is not None})
    necessity = [_mean_metric([r for r in rows if _float(r.get("quality_floor")) == f], "neural_necessary_rate") for f in floors]
    wins = [_mean_metric([r for r in rows if _float(r.get("quality_floor")) == f], "neural_win_rate") for f in floors]

    # Guard: skip the plot when both series are degenerate (constant) — a
    # flat line at 0 across 3 points is misleading and adds no information.
    # The summary table is the authoritative artifact in that case.
    necessity_varies = len(set(round(v, 6) for v in necessity)) > 1
    wins_varies = len(set(round(v, 6) for v in wins)) > 1
    if not (necessity_varies or wins_varies):
        return {
            "sampled": False,
            "input_rows": len(rows),
            "plotted_rows": 0,
            "skipped": (
                "both neural_necessary_rate and neural_win_rate are constant "
                "across quality floors; rendering would mislead"
            ),
        }

    plt.figure(figsize=(7, 5))
    if necessity_varies:
        plt.plot(floors, necessity, marker="o", label="neural_necessary_rate")
    if wins_varies:
        plt.plot(floors, wins, marker="s", label="neural_win_rate")
    plt.xlabel("Quality floor")
    plt.ylabel("Rate")
    plt.legend()
    _save(plt, out_path)
    return {"sampled": False, "input_rows": len(rows), "plotted_rows": len(floors)}


def _plot_confusion(plt: Any, data: Dict[str, Any], out_path: Path, max_plot_points: int, seed: int) -> Dict[str, Any]:
    rows = data["confusion"]
    families = sorted({r.get("oracle_family", "") for r in rows} | {r.get("predicted_family", "") for r in rows})
    matrix = []
    for oracle_family in families:
        matrix.append([
            sum(
                _float(r.get("rate")) or 0.0
                for r in rows
                if r.get("oracle_family") == oracle_family and r.get("predicted_family") == predicted_family
            )
            for predicted_family in families
        ])
    plt.figure(figsize=(5, 4))
    plt.imshow(matrix, cmap="Blues")
    plt.xticks(range(len(families)), families)
    plt.yticks(range(len(families)), families)
    plt.xlabel("Predicted family")
    plt.ylabel("Oracle family")
    _save(plt, out_path)
    return {"sampled": False, "input_rows": len(rows), "plotted_rows": len(rows)}


def _plot_tradeoff(plt: Any, data: Dict[str, Any], out_path: Path, max_plot_points: int, seed: int) -> Dict[str, Any]:
    valid = []
    for row in data["tradeoff"]:
        x = _float(row.get("bitrate_reduction_neural_vs_classic"))
        y = _float(row.get("delta_energy_neural_minus_classic"))
        if x is None or y is None:
            continue
        valid.append(row)
    rows, sampled = _sample_rows(valid, max_plot_points=max_plot_points, seed=seed)

    xs = [_float(r.get("bitrate_reduction_neural_vs_classic")) for r in rows]
    ys = [_float(r.get("delta_energy_neural_minus_classic")) for r in rows]
    qs = [_float(r.get("delta_quality_neural_minus_classic")) or 0.0 for r in rows]
    sizes = [max(15.0, min(120.0, 20.0 + 20.0 * abs(q))) for q in qs]

    # Single vectorized call so every point gets a deterministic color from
    # the colormap (driven by the quality delta) instead of a different
    # cycler color per iteration, which produced the rainbow scatter.
    fig, ax = plt.subplots(figsize=(7.5, 5))
    scatter = ax.scatter(xs, ys, s=sizes, c=qs, cmap="coolwarm", alpha=0.65,
                         edgecolors="none")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Delta quality (neural - classic)")

    # Clip y-axis to the 95th percentile to keep the bulk of the points
    # readable; outliers (e.g. ~1200 J on one image) are kept off-axis and
    # noted in the caption rather than allowed to crush the plot.
    finite_ys = [y for y in ys if y is not None and math.isfinite(y)]
    if finite_ys:
        sorted_ys = sorted(finite_ys)
        p95 = sorted_ys[int(0.95 * (len(sorted_ys) - 1))]
        if p95 > 0:
            ax.set_ylim(top=p95 * 1.1)

    ax.axhline(0, color="black", linewidth=0.6, alpha=0.4)
    ax.axvline(0, color="black", linewidth=0.6, alpha=0.4)
    ax.set_xlabel("Bitrate reduction neural vs classic")
    ax.set_ylabel("Delta energy neural minus classic")
    _save(plt, out_path)
    return {"sampled": sampled, "input_rows": len(valid), "plotted_rows": len(rows)}


def _plot_rate_pressure_family(plt: Any, data: Dict[str, Any], out_path: Path, max_plot_points: int, seed: int) -> Dict[str, Any]:
    rows = data["rate_pressure"]
    weights = sorted({_float(r.get("rate_weight")) for r in rows if _float(r.get("rate_weight")) is not None})
    families = sorted({r.get("selected_family", "") for r in rows})
    plt.figure(figsize=(8, 5))
    for family in families:
        values = [_mean_metric([r for r in rows if _float(r.get("rate_weight")) == w and r.get("selected_family") == family], "selection_rate") for w in weights]
        plt.plot(weights, values, marker="o", label=family)
    plt.xlabel("Rate weight")
    plt.ylabel("Selection share")
    plt.legend()
    _save(plt, out_path)
    return {"sampled": False, "input_rows": len(rows), "plotted_rows": len(rows)}


def _plot_rate_pressure_codec(plt: Any, data: Dict[str, Any], out_path: Path, max_plot_points: int, seed: int) -> Dict[str, Any]:
    rows = data["rate_pressure"]
    weights = sorted({_float(r.get("rate_weight")) for r in rows if _float(r.get("rate_weight")) is not None})
    codecs = sorted({r.get("top_selected_codec", "") for r in rows})
    plt.figure(figsize=(9, 5))
    for codec in codecs:
        values = [_mean_metric([r for r in rows if _float(r.get("rate_weight")) == w and r.get("top_selected_codec") == codec], "top_selected_codec_rate") for w in weights]
        plt.plot(weights, values, marker="o", label=codec)
    plt.xlabel("Rate weight")
    plt.ylabel("Top codec selection share")
    plt.legend(fontsize=7)
    _save(plt, out_path)
    return {"sampled": False, "input_rows": len(rows), "plotted_rows": len(rows)}


def _plot_rate_energy(plt: Any, data: Dict[str, Any], out_path: Path, max_plot_points: int, seed: int) -> Dict[str, Any]:
    rows = []
    for row in data["rate_pressure"]:
        if _float(row.get("rate_reduction_vs_baseline")) is not None and _float(row.get("energy_delta_vs_baseline")) is not None:
            rows.append(row)
    rows, sampled = _sample_rows(rows, max_plot_points=max_plot_points, seed=seed)
    plt.figure(figsize=(7, 5))
    for row in rows:
        plt.scatter(
            _float(row.get("rate_reduction_vs_baseline")),
            _float(row.get("energy_delta_vs_baseline")),
            s=24,
            alpha=0.65,
        )
    plt.xlabel("Rate reduction vs baseline")
    plt.ylabel("Energy delta vs baseline")
    _save(plt, out_path)
    return {"sampled": sampled, "input_rows": len(data["rate_pressure"]), "plotted_rows": len(rows)}


PLOT_SPECS: Dict[str, Dict[str, Any]] = {
    "energy_saving_vs_regret_reduction.png": {"inputs": ["plot_data"], "fn": _plot_energy_regret},
    "neural_selection_rate_by_regime.png": {"inputs": ["plot_data"], "fn": _plot_neural_selection},
    "winner_family_by_regime.png": {"inputs": ["winners"], "fn": _plot_winner_family},
    "oracle_vs_predicted_neural_rate.png": {"inputs": ["oracle_prediction"], "fn": _plot_oracle_predicted},
    "rate_weight_switch_boundary.png": {"inputs": ["switch_summary", "rate_pressure"], "fn": _plot_rate_switch_boundary},
    "switch_reason_by_rate_weight.png": {"inputs": ["switch_reason"], "fn": _plot_switch_reason},
    "quality_floor_vs_neural_necessity.png": {"inputs": ["quality_floor"], "fn": _plot_quality_floor},
    "winner_codec_by_regime.png": {"inputs": ["winners"], "fn": _plot_winner_codec},
    "family_confusion_heatmap.png": {"inputs": ["confusion"], "fn": _plot_confusion},
    "neural_vs_classic_tradeoff_scatter.png": {"inputs": ["tradeoff"], "fn": _plot_tradeoff},
    "rate_pressure_family_shift.png": {"inputs": ["rate_pressure"], "fn": _plot_rate_pressure_family},
    "rate_pressure_codec_shift.png": {"inputs": ["rate_pressure"], "fn": _plot_rate_pressure_codec},
    "rate_reduction_vs_energy_penalty_sweep.png": {"inputs": ["rate_pressure"], "fn": _plot_rate_energy},
    "quality_violation_by_regime.png": {"inputs": ["plot_data"], "fn": _plot_quality_violation},
}


def generate_plots(
    *,
    input_dir: Path,
    out_dir: Path,
    plot_mode: str,
    max_plot_points: int,
    seed: int,
) -> Dict[str, Any]:
    started = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    report: Dict[str, Any] = {
        "plot_mode": plot_mode,
        "input_dir": str(input_dir),
        "out_dir": str(out_dir),
        "generated_plots": [],
        "failed_plots": [],
        "skipped_plots": [],
        "max_plot_points": max_plot_points,
        "sampled_plots": [],
    }

    plot_names = BASIC_PLOTS if plot_mode == "basic" else BASIC_PLOTS + ALL_EXTRA_PLOTS
    data = {key: _read_csv(input_dir / filename) for key, filename in CSV_FILES.items()}

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:
        for name in plot_names:
            report["skipped_plots"].append(
                {"plot": name, "reason": "matplotlib_unavailable", "detail": type(exc).__name__}
            )
        report["elapsed_seconds"] = time.time() - started
        _write_json(out_dir / "operational_regime_plot_report.json", report)
        return report

    for name in plot_names:
        spec = PLOT_SPECS[name]
        missing = _require(data, *spec["inputs"])
        if missing is not None:
            report["skipped_plots"].append(
                {"plot": name, "reason": "missing_input_csv", "detail": missing}
            )
            continue
        try:
            meta = spec["fn"](plt, data, out_dir / name, max_plot_points, seed)
            report["generated_plots"].append(str(out_dir / name))
            if meta.get("sampled"):
                report["sampled_plots"].append({"plot": name, **meta})
        except Exception as exc:
            try:
                plt.close()
            except Exception:
                pass
            report["failed_plots"].append(
                {"plot": name, "error": type(exc).__name__, "detail": str(exc)}
            )

    report["elapsed_seconds"] = time.time() - started
    _write_json(out_dir / "operational_regime_plot_report.json", report)
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render operational-regime plots from existing CSV artifacts."
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--out-dir")
    parser.add_argument("--plot-mode", choices=["basic", "all"], default="basic")
    parser.add_argument("--max-plot-points", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir) if args.out_dir else input_dir
    generate_plots(
        input_dir=input_dir,
        out_dir=out_dir,
        plot_mode=args.plot_mode,
        max_plot_points=args.max_plot_points,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
