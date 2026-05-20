import argparse

from src.router.core.profiles import available_profiles


def build_router_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prototype R-D-E router for adaptive codec selection."
    )

    parser.add_argument(
        "--config",
        default=None,
        help="Router configuration JSON file. Expanded before normal argument parsing.",
    )

    parser.add_argument("--csv", required=True, help="Path to the CSV file containing R-D-E points.")

    parser.add_argument(
        "--calibration-bundle-manifest",
        default=None,
        help=(
            "Explicit calibration bundle manifest. When provided, the router "
            "validates the manifest and uses its calibrated CSV as the R-D-E input."
        ),
    )

    parser.add_argument(
        "--calibration-bundle-validation",
        default=None,
        help=(
            "Optional explicit shadow decision validation report. When provided, "
            "it must be accepted before the calibration bundle can be used."
        ),
    )

    parser.add_argument(
        "--calibration-file",
        default=None,
        help="Local calibration JSON file to apply to the R-D-E points.",
    )

    parser.add_argument(
        "--normalization-file",
        default=None,
        help="JSON file with precomputed normalization scales.",
    )

    parser.add_argument(
        "--normalization-mode",
        default="auto",
        choices=["auto", "runtime", "global", "dataset", "local"],
        help=(
            "Normalization policy: auto, runtime, global, dataset, local. "
            "runtime computes normalization on the fly; global/dataset/local "
            "require --normalization-file."
        ),
    )

    parser.add_argument(
        "--previous-decision-receipt",
        default=None,
        help=(
            "Optional explicit previous decision receipt/router report used only "
            "to audit normalization comparability in the output report."
        ),
    )

    parser.add_argument(
        "--input",
        default=None,
        help="Input file used to generate an execution plan.",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Desired output file for the execution plan.",
    )

    parser.add_argument(
        "--generate-command",
        action="store_true",
        help="Generate an execution plan for the selected codec.",
    )

    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute the generated command directly when the plan is executable.",
    )

    parser.add_argument(
        "--domain",
        default="image",
        choices=["image", "audio", "video"],
        help="Media domain.",
    )

    parser.add_argument(
        "--domain-spec",
        default=None,
        help=(
            "Built-in DomainSpec name or JSON path used to resolve R-D-E "
            "column defaults without changing the ranking formula."
        ),
    )

    parser.add_argument(
        "--profile",
        default="balanced",
        choices=available_profiles(),
        help="Operating profile to use when --all-profiles is not set.",
    )

    parser.add_argument(
        "--all-profiles",
        action="store_true",
        help="Run the router over all available profiles and produce a summary CSV.",
    )

    parser.add_argument(
        "--auto-weights",
        action="store_true",
        help="Automatically derive the R-D-E weights from the operating context.",
    )

    parser.add_argument(
        "--power-mode",
        choices=["ac", "battery", "unknown"],
        default="ac",
        help="Power mode used by the contextual policy.",
    )

    parser.add_argument(
        "--battery-percent",
        type=float,
        default=None,
        help="Battery percentage used by the contextual policy.",
    )

    parser.add_argument(
        "--thermal-state",
        choices=["nominal", "warm", "hot", "critical"],
        default="nominal",
        help="Thermal state used by the contextual policy.",
    )

    parser.add_argument(
        "--network-profile",
        choices=["normal", "limited", "very-limited"],
        default="normal",
        help="Network profile used by the contextual policy.",
    )

    parser.add_argument(
        "--quality-target",
        choices=["preview", "normal", "high", "very-high"],
        default="normal",
        help="Quality target used by the contextual policy.",
    )

    parser.add_argument(
        "--quality-thresholds-file",
        default="configs/quality_thresholds.json",
        help="JSON file with domain-specific quality thresholds.",
    )

    parser.add_argument(
        "--codec-registry-file",
        default=None,
        help="External codec/backend registry JSON file.",
    )

    parser.add_argument(
        "--external-codec-manifest",
        action="append",
        default=[],
        help=(
            "Explicit external codec router manifest. Repeat to load multiple "
            "validated external R-D-E exports. No automatic discovery is performed."
        ),
    )

    parser.add_argument(
        "--system-load",
        choices=["normal", "high", "very-high"],
        default="normal",
        help="System load used by the contextual policy.",
    )

    parser.add_argument(
        "--aggregate-by-config",
        action="store_true",
        help="Aggregate rows by codec+config using the mean of rate, quality, and energy.",
    )

    parser.add_argument(
        "--normalization-scope",
        choices=["global", "filtered"],
        default="global",
        help=(
            "global = normalize over the points before codec filtering; "
            "filtered = normalize only over the points that survive the filters."
        ),
    )

    parser.add_argument(
        "--available-codecs",
        default=None,
        help="Comma-separated list of available codecs. Example: JPEG,JXL,HEVC",
    )

    parser.add_argument(
        "--exclude-codecs",
        default=None,
        help="Comma-separated list of codecs to exclude. Example: DCAE,JPEG_AI",
    )

    parser.add_argument(
        "--exclude-neural",
        action="store_true",
        help="Exclude neural or learning-based codecs.",
    )

    parser.add_argument(
        "--system-aware",
        action="store_true",
        help="Use the real system profile to automatically filter the admissible candidate pool.",
    )

    parser.add_argument(
        "--system-features",
        action="store_true",
        help="Extract cheap system-aware features and include them in the report.",
    )

    parser.add_argument(
        "--system-probe-level",
        default="basic",
        choices=["basic", "gpu", "full"],
        help="System feature probe level: basic, gpu, or full.",
    )

    parser.add_argument(
        "--system-feature-cache-ttl-s",
        type=float,
        default=5.0,
        help="Cache TTL in seconds for system feature probes.",
    )

    parser.add_argument(
        "--system-feature-cpu-interval-s",
        type=float,
        default=0.0,
        help="psutil CPU sampling interval for system feature extraction.",
    )

    parser.add_argument(
        "--system-policy",
        action="store_true",
        help="Build a system-aware policy from extracted system features.",
    )

    parser.add_argument(
        "--system-policy-mode",
        default="report-only",
        choices=["report-only", "apply"],
        help="report-only records suggested changes; apply uses adjusted weights.",
    )

    parser.add_argument(
        "--system-policy-simulate",
        default=None,
        help=(
            "Comma-separated simulated system classes, e.g. "
            "'battery=critical,cpu=busy,memory=constrained'. "
            "Overrides measured classes for system-policy evaluation."
        ),
    )

    parser.add_argument(
        "--system-penalty",
        action="store_true",
        help="Compute a codec/backend system penalty from resource profiles.",
    )

    parser.add_argument(
        "--system-penalty-mode",
        default="report-only",
        choices=["report-only", "apply"],
        help="report-only records J_total; apply ranks by J_total and applies hard exclusions.",
    )

    parser.add_argument(
        "--system-penalty-lambda",
        type=float,
        default=0.25,
        help="Weight of the system penalty term in J_total.",
    )

    parser.add_argument(
        "--system-penalty-weights-file",
        default=None,
        help="Optional JSON file with configurable system penalty coefficients.",
    )

    parser.add_argument(
        "--content-policy",
        action="store_true",
        help="Enable source-aware content policy reporting.",
    )

    parser.add_argument(
        "--content-policy-mode",
        default="report-only",
        choices=["report-only", "apply"],
        help="Content policy mode: report-only or safe apply integration.",
    )

    parser.add_argument(
        "--content-policy-rules-file",
        default=None,
        help="CSV rules file produced by content metadata policy evaluation.",
    )

    parser.add_argument(
        "--content-policy-key",
        default="dataset",
        help="Metadata key used by the content policy rules, e.g. dataset/source.",
    )

    parser.add_argument(
        "--content-source",
        default=None,
        help="Optional homogeneous content source/dataset label, e.g. tecnick, kodak, clic2020.",
    )

    parser.add_argument(
        "--content-source-filter",
        action="store_true",
        help="Filter the benchmark candidate pool using the provided content source/context.",
    )

    parser.add_argument(
        "--content-filter-column",
        default=None,
        help="CSV/raw column used for source-conditioned filtering. Defaults to --content-policy-key.",
    )

    parser.add_argument(
        "--content-filter-value",
        default=None,
        help="Value used for source-conditioned filtering. Defaults to --content-source.",
    )

    parser.add_argument(
        "--content-classifier",
        action="store_true",
        help="Enable source-agnostic content classifier reporting.",
    )

    parser.add_argument(
        "--content-classifier-mode",
        default="report-only",
        choices=["report-only", "apply"],
        help="Content classifier mode: report-only or safe apply integration.",
    )

    parser.add_argument(
        "--content-classifier-config",
        default=None,
        help="JSON config for the source-agnostic content classifier.",
    )

    parser.add_argument(
        "--content-classifier-image",
        default=None,
        help="Optional image path used to extract source-agnostic content classifier features.",
    )

    parser.add_argument(
        "--content-classifier-width",
        type=int,
        default=None,
        help="Optional image width used when no content-classifier image path is provided.",
    )

    parser.add_argument(
        "--content-classifier-height",
        type=int,
        default=None,
        help="Optional image height used when no content-classifier image path is provided.",
    )

    parser.add_argument(
        "--capability-aware",
        action="store_true",
        help="Filter codecs using the hardware/software requirement registry.",
    )

    parser.add_argument(
        "--strict-executables",
        action="store_true",
        help="When set, exclude codecs whose required executables are not on PATH.",
    )

    parser.add_argument(
        "--simulate-no-cuda",
        action="store_true",
        help="Debug: simulate the absence of CUDA to test the system-aware filter.",
    )

    parser.add_argument(
        "--safe-mode",
        action="store_true",
        help="Enable the robust quality guard: use p10 when unspecified and a minimum floor of 60.",
    )

    parser.add_argument(
        "--quality-constraint-stat",
        choices=["mean", "p25", "p10", "min"],
        default=None,
        help="Statistic used as the hard quality constraint.",
    )

    parser.add_argument(
        "--quality-floor",
        type=float,
        default=None,
        help="Absolute minimum acceptable quality threshold.",
    )

    parser.add_argument(
        "--near-quality-floor",
        type=float,
        default=None,
        help="Near-usable quality threshold for degraded fallback.",
    )

    parser.add_argument(
        "--allow-degraded-fallback",
        action="store_true",
        help="Allow degraded fallback when no point clears the safe threshold.",
    )

    parser.add_argument(
        "--min-quality",
        type=float,
        default=None,
        help="Minimum admissible quality. If omitted, the profile/policy value is used.",
    )

    parser.add_argument("--max-rate", type=float, default=None, help="Maximum admissible rate.")
    parser.add_argument("--max-energy", type=float, default=None, help="Maximum admissible energy.")
    parser.add_argument("--max-time-ms", type=float, default=None, help="Maximum admissible time in milliseconds.")
    parser.add_argument(
        "--strict-time",
        action="store_true",
        help=(
            "When used with --max-time-ms, require every candidate point "
            "to have a time_ms value."
        ),
    )

    parser.add_argument("--wE", "--w-e", dest="wE", type=float, default=None, help="Custom energy weight.")
    parser.add_argument("--wR", "--w-r", dest="wR", type=float, default=None, help="Custom rate weight.")
    parser.add_argument("--wD", "--w-d", dest="wD", type=float, default=None, help="Custom distortion weight.")

    parser.add_argument("--codec-col", default=None)
    parser.add_argument("--config-col", default=None)
    parser.add_argument("--rate-col", default=None)
    parser.add_argument("--quality-col", default=None)
    parser.add_argument("--quality-metric", default=None)
    parser.add_argument("--energy-col", default=None)
    parser.add_argument("--time-col", default=None)

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of best configurations to save in the JSON report.",
    )

    parser.add_argument(
        "--export-topk",
        action="store_true",
        help="Also export the top-k candidates as a CSV file.",
    )

    parser.add_argument(
        "--feedback-out",
        default="results/routing_context/online_feedback.csv",
        help=(
            "Append-only CSV for observed execution feedback. "
            "Written only when --execute is used."
        ),
    )

    parser.add_argument(
        "--out",
        default="results/routing/router_decision_report.json",
        help="Path to the JSON report when a single profile is used.",
    )

    parser.add_argument(
        "--out-dir",
        default="results/routing",
        help="Output directory when --all-profiles is used.",
    )

    parser.add_argument(
        "--summary-out",
        default=None,
        help="Path to the summary CSV. Defaults to results/routing/router_summary.csv.",
    )

    return parser
