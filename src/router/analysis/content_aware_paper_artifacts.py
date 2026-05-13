import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional


def _read_csv(path: str) -> List[Dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        rows = [dict(row) for row in csv.DictReader(f)]

    if not rows:
        raise ValueError(f"No rows loaded from {path}")

    return rows


def _write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _to_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None or str(value).strip() == "":
            return default
        return float(str(value).replace(",", "."))
    except Exception:
        return default


def _fmt_float(value: Any, digits: int = 4) -> str:
    x = _to_float(value)

    if x is None:
        return "--"

    return f"{x:.{digits}f}"


def _fmt_percent(value: Any, digits: int = 1) -> str:
    x = _to_float(value)

    if x is None:
        return "--"

    return f"{100.0 * x:.{digits}f}"


def _latex_escape(value: Any) -> str:
    text = str(value)

    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


def _write_latex_table(
    *,
    path: str,
    rows: List[Dict[str, Any]],
    columns: List[str],
    headers: List[str],
    caption: str,
    label: str,
    resize_to_textwidth: bool = False,
    alignment: str | None = None,
) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    alignment = alignment or ("l" + "r" * (len(columns) - 1))

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
    ]

    if resize_to_textwidth:
        lines.append(r"\resizebox{\textwidth}{!}{%")

    lines.extend(
        [
            rf"\begin{{tabular}}{{{alignment}}}",
            r"\toprule",
            " & ".join(headers) + r" \\",
            r"\midrule",
        ]
    )

    for row in rows:
        lines.append(
            " & ".join(_latex_escape(row.get(col, "")) for col in columns) + r" \\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )

    if resize_to_textwidth:
        lines.append(r"}")

    lines.extend([r"\end{table}", ""])

    out.write_text("\n".join(lines), encoding="utf-8")


def build_main_paper_table(rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    friendly_names = {
        "robust_global_baseline": "Baseline globale robusta",
        "source_aware_dataset_majority_policy": "Policy source-aware",
        "best_source_agnostic_knn_leave_one_image_out": "kNN source-agnostic",
        "best_source_agnostic_knn_leave_one_dataset_out": "kNN source-agnostic",
        "per_image_oracle": "Oracle per immagine",
    }

    protocol_names = {
        "global_coverage_oracle_analysis": "Global",
        "leave-one-out": "Source LOO",
        "leave_one_image_out": "LOIO",
        "leave_one_dataset_out": "LODO",
        "oracle": "Oracle",
    }

    deployment_names = {
        "source_agnostic": "Source-agnostic",
        "batch_known_source": "Batch con sorgente",
        "not_deployable": "Non deployable",
    }

    out = []

    for row in rows:
        method_id = row.get("method_id", "")

        if method_id not in friendly_names:
            continue

        feature_set = row.get("feature_set", "")
        k = row.get("k", "")

        detail = ""
        if feature_set and feature_set not in {"none", "dataset", "oracle"}:
            detail = feature_set
            if k:
                detail += f", k={k}"
        elif feature_set == "dataset":
            detail = "dataset majority"
        elif feature_set == "oracle":
            detail = "oracle"
        else:
            detail = row.get("selected_policy", "")

        out.append(
            {
                "method": friendly_names[method_id],
                "protocol": protocol_names.get(
                    row.get("evaluation_protocol", ""),
                    row.get("evaluation_protocol", ""),
                ),
                "deployment": deployment_names.get(
                    row.get("deployment_setting", ""),
                    row.get("deployment_setting", ""),
                ),
                "detail": detail,
                "mean_regret": _fmt_float(row.get("mean_regret"), 5),
                "regret_reduction_percent": _fmt_percent(
                    row.get("relative_regret_reduction"), 1
                ),
                "oracle_match_percent": _fmt_percent(row.get("accuracy"), 1),
                "fallback_percent": _fmt_percent(row.get("fallback_rate"), 1),
            }
        )

    return out


def build_overhead_paper_table(rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    by_case = {row.get("case_id"): row for row in rows}

    jpeg_global = _to_float(by_case.get("jpeg_q85_global", {}).get("mean_ms"))
    hevc_global = _to_float(by_case.get("hevc_crf15_global", {}).get("mean_ms"))

    wanted = [
        ("pixel_features_long_side_256", "Feature pixel", "Lato lungo 256"),
        ("metadata_no_source", "Feature metadata-only", "Header / metadati noti"),
        ("jpeg_q85_tecnick", "Codifica JPEG q=85", "Tecnick"),
        ("jpeg_q85_global", "Codifica JPEG q=85", "Globale"),
        ("jxl_d1_global", "Codifica JXL d=1.0", "Globale"),
        ("hevc_crf15_global", "Codifica HEVC crf=15", "Globale"),
    ]

    out = []

    for case_id, component, scope in wanted:
        row = by_case.get(case_id, {})
        mean_ms = _to_float(row.get("mean_ms"))

        ratio_jpeg = None
        ratio_hevc = None

        if mean_ms is not None and jpeg_global and jpeg_global > 0:
            ratio_jpeg = mean_ms / jpeg_global

        if mean_ms is not None and hevc_global and hevc_global > 0:
            ratio_hevc = mean_ms / hevc_global

        out.append(
            {
                "component": component,
                "scope": scope,
                "samples": row.get("num_samples", ""),
                "mean_ms": _fmt_float(mean_ms, 2),
                "p90_ms": _fmt_float(row.get("p90_ms"), 2),
                "ratio_vs_jpeg_global": _fmt_float(ratio_jpeg, 2),
                "ratio_vs_hevc_global": _fmt_float(ratio_hevc, 2),
            }
        )

    return out


def build_best_k_table(rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    best_rows = [
        row for row in rows
        if str(row.get("best_for_evaluation_mode", "")).lower() == "true"
    ]

    protocol_names = {
        "leave_one_image_out": "LOIO",
        "leave_one_dataset_out": "LODO",
    }

    out = []

    for row in best_rows:
        out.append(
            {
                "protocol": protocol_names.get(
                    row.get("evaluation_mode", ""),
                    row.get("evaluation_mode", ""),
                ),
                "feature_set": row.get("feature_set", ""),
                "k": row.get("k", ""),
                "mean_regret": _fmt_float(row.get("mean_regret"), 5),
                "regret_reduction_percent": _fmt_percent(
                    row.get("relative_regret_reduction"), 1
                ),
                "oracle_match_percent": _fmt_percent(row.get("accuracy"), 1),
                "fallback_percent": _fmt_percent(row.get("fallback_rate"), 1),
            }
        )

    out.sort(key=lambda r: r["protocol"])

    return out


def _plot_bar(
    *,
    rows: List[Dict[str, Any]],
    x_key: str,
    y_key: str,
    title: str,
    ylabel: str,
    out_path: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [str(row[x_key]) for row in rows]
    values = [_to_float(row[y_key], 0.0) or 0.0 for row in rows]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(labels, values)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def _plot_k_sensitivity(rows: List[Dict[str, str]], out_dir: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    feature_sets = ["metadata_no_source", "pixel_no_source"]

    for protocol in ["leave_one_image_out", "leave_one_dataset_out"]:
        fig, ax = plt.subplots(figsize=(7.2, 4.5))

        for feature_set in feature_sets:
            selected = [
                row for row in rows
                if row.get("evaluation_mode") == protocol
                and row.get("feature_set") == feature_set
            ]

            selected.sort(key=lambda r: int(float(r["k"])))

            xs = [int(float(row["k"])) for row in selected]
            ys = [_to_float(row["mean_regret"], 0.0) or 0.0 for row in selected]

            ax.plot(xs, ys, marker="o", label=feature_set)

        protocol_label = {
            "leave_one_image_out": "LOIO",
            "leave_one_dataset_out": "LODO",
        }.get(protocol, protocol)

        ax.set_title(f"Sensibilita rispetto a k ({protocol_label})")
        ax.set_xlabel("k")
        ax.set_ylabel("Regret medio")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / f"v09_k_sensitivity_{protocol}.png", dpi=200)
        plt.close(fig)


def _plot_oracle_distribution(oracle_summary_csv: str, out_path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = _read_csv(oracle_summary_csv)

    selected = [row for row in rows if row.get("section") == "oracle_count"]

    labels = [
        row.get("key") or row.get("label") or row.get("codec") or "unknown"
        for row in selected
    ]
    values = [
        int(float(row.get("value") or row.get("count") or 0))
        for row in selected
    ]

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.bar(labels, values)
    ax.set_title("Distribuzione dell'oracle per immagine")
    ax.set_ylabel("Immagini")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def build_artifacts(
    *,
    benchmark_table_csv: str,
    overhead_table_csv: str,
    sensitivity_table_csv: str,
    oracle_summary_csv: str,
    out_dir: str,
    make_plots: bool,
) -> Dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    benchmark_rows = _read_csv(benchmark_table_csv)
    overhead_rows = _read_csv(overhead_table_csv)
    sensitivity_rows = _read_csv(sensitivity_table_csv)

    main_table = build_main_paper_table(benchmark_rows)
    overhead_table = build_overhead_paper_table(overhead_rows)
    best_k_table = build_best_k_table(sensitivity_rows)

    paths = {
        "main_csv": str(out / "v09_content_aware_final_table.csv"),
        "main_tex": str(out / "v09_content_aware_final_table.tex"),
        "overhead_csv": str(out / "v09_content_aware_overhead_table_paper.csv"),
        "overhead_tex": str(out / "v09_content_aware_overhead_table_paper.tex"),
        "best_k_csv": str(out / "v09_content_aware_best_k_table.csv"),
        "best_k_tex": str(out / "v09_content_aware_best_k_table.tex"),
    }

    _write_csv(paths["main_csv"], main_table)
    _write_csv(paths["overhead_csv"], overhead_table)
    _write_csv(paths["best_k_csv"], best_k_table)

    _write_latex_table(
        path=paths["main_tex"],
        rows=main_table,
        columns=[
            "method",
            "protocol",
            "deployment",
            "mean_regret",
            "regret_reduction_percent",
            "oracle_match_percent",
            "fallback_percent",
        ],
        headers=[
            "Metodo",
            "Protocollo",
            "Deployment",
            "Regret medio",
            "Riduz. (\\%)",
            "Match oracle (\\%)",
            "Fallback (\\%)",
        ],
        caption="Benchmark del routing R-D-E content-aware.",
        label="tab:content-aware-rde",
        resize_to_textwidth=True,
        alignment="lllrrrr",
    )

    _write_latex_table(
        path=paths["overhead_tex"],
        rows=overhead_table,
        columns=[
            "component",
            "scope",
            "mean_ms",
            "p90_ms",
            "ratio_vs_jpeg_global",
            "ratio_vs_hevc_global",
        ],
        headers=[
            "Componente",
            "Ambito",
            "Media ms",
            "P90 ms",
            "vs JPEG",
            "vs HEVC",
        ],
        caption="Overhead e tempi di codifica di riferimento per il routing content-aware.",
        label="tab:content-aware-overhead",
        resize_to_textwidth=True,
        alignment="llrrrr",
    )

    _write_latex_table(
        path=paths["best_k_tex"],
        rows=best_k_table,
        columns=[
            "protocol",
            "feature_set",
            "k",
            "mean_regret",
            "regret_reduction_percent",
            "oracle_match_percent",
            "fallback_percent",
        ],
        headers=[
            "Protocollo",
            "Feature set",
            r"\(k\)",
            "Regret medio",
            "Riduz. (\\%)",
            "Match oracle (\\%)",
            "Fallback (\\%)",
        ],
        caption="Migliori configurazioni kNN per ciascun protocollo di valutazione.",
        label="tab:content-aware-best-k",
        resize_to_textwidth=True,
        alignment="llrrrrr",
    )

    if make_plots:
        fig_dir = out / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

        plot_rows = [
            {
                "method": row["method"],
                "mean_regret_raw": _to_float(row["mean_regret"], 0.0),
                "reduction_raw": (
                    (_to_float(row["regret_reduction_percent"], 0.0) or 0.0)
                    / 100.0
                ),
            }
            for row in main_table
        ]

        _plot_bar(
            rows=plot_rows,
            x_key="method",
            y_key="mean_regret_raw",
            title="Regret medio per metodo",
            ylabel="Regret medio",
            out_path=str(fig_dir / "v09_mean_regret_methods.png"),
        )

        _plot_bar(
            rows=plot_rows,
            x_key="method",
            y_key="reduction_raw",
            title="Riduzione relativa del regret per metodo",
            ylabel="Riduzione relativa",
            out_path=str(fig_dir / "v09_relative_regret_reduction_methods.png"),
        )

        overhead_plot_rows = [
            {
                "component": row["component"],
                "mean_ms_raw": _to_float(row["mean_ms"], 0.0),
            }
            for row in overhead_table
            if row["mean_ms"] != "--"
        ]

        _plot_bar(
            rows=overhead_plot_rows,
            x_key="component",
            y_key="mean_ms_raw",
            title="Overhead delle feature e tempi di codifica",
            ylabel="Tempo medio (ms)",
            out_path=str(fig_dir / "v09_overhead_vs_encoding.png"),
        )

        _plot_k_sensitivity(sensitivity_rows, str(fig_dir))

        _plot_oracle_distribution(
            oracle_summary_csv,
            str(fig_dir / "v09_oracle_distribution.png"),
        )

        paths["figures_dir"] = str(fig_dir)

    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate paper-ready tables and figures for v0.9 content-aware "
            "R-D-E routing."
        )
    )

    parser.add_argument(
        "--benchmark-table",
        default="results/routing_context/v09_content_aware_benchmark_table.csv",
    )
    parser.add_argument(
        "--overhead-table",
        default="results/routing_context/v09_content_aware_overhead_table.csv",
    )
    parser.add_argument(
        "--sensitivity-table",
        default="results/routing_context/v09_knn_sensitivity_table.csv",
    )
    parser.add_argument(
        "--oracle-summary",
        default="results/routing_context/v09_content_oracle_summary.csv",
    )
    parser.add_argument(
        "--out-dir",
        default="results/routing_context/paper_artifacts_v09",
    )
    parser.add_argument("--no-plots", action="store_true")

    args = parser.parse_args()

    paths = build_artifacts(
        benchmark_table_csv=args.benchmark_table,
        overhead_table_csv=args.overhead_table,
        sensitivity_table_csv=args.sensitivity_table,
        oracle_summary_csv=args.oracle_summary,
        out_dir=args.out_dir,
        make_plots=not args.no_plots,
    )

    print("\n=== R-D-E v0.9 Paper Artifacts ===")
    for key, value in paths.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
