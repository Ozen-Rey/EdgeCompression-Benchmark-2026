import argparse

try:
    from .calibration_apply import apply_local_calibration
    from .rde_database import aggregate_points_by_config, load_rde_points
    from .normalization_profile import build_normalization_profile, save_normalization_profile
except ImportError:
    from calibration_apply import apply_local_calibration
    from rde_database import aggregate_points_by_config, load_rde_points
    from normalization_profile import build_normalization_profile, save_normalization_profile


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a precomputed normalization profile for the R-D-E router."
    )

    parser.add_argument("--csv", required=True)
    parser.add_argument("--domain", default="image")
    parser.add_argument("--codec-col", default="codec")
    parser.add_argument("--config-col", default="config")
    parser.add_argument("--rate-col", default="rate")
    parser.add_argument("--quality-col", default="quality")
    parser.add_argument("--energy-col", default="energy")
    parser.add_argument("--time-col", default=None)
    parser.add_argument("--aggregate-by-config", action="store_true")
    parser.add_argument(
        "--mode",
        default="global",
        choices=["global", "dataset", "local"],
        help="Tipo semantico del profilo di normalizzazione.",
    )
    parser.add_argument(
        "--calibration-file",
        default=None,
        help="File JSON di calibrazione locale da applicare prima di costruire il profilo.",
    )
    parser.add_argument(
        "--normalization-build-scope",
        default="all",
        choices=["all", "calibrated-only"],
        help=(
            "Scope usato per costruire le scale: 'all' usa tutti i punti; "
            "'calibrated-only' usa solo i punti presenti nella calibrazione locale."
        ),
    )
    parser.add_argument("--out", required=True)

    args = parser.parse_args()

    points = load_rde_points(
        csv_path=args.csv,
        codec_col=args.codec_col,
        config_col=args.config_col,
        rate_col=args.rate_col,
        quality_col=args.quality_col,
        energy_col=args.energy_col,
        time_col=args.time_col,
    )

    if args.aggregate_by_config:
        points = aggregate_points_by_config(points)

    if args.calibration_file:
        points, calibration_report = apply_local_calibration(
            points=points,
            calibration_file=args.calibration_file,
        )
    else:
        calibration_report = {"enabled": False}

    num_points_before_build_scope = len(points)

    if args.normalization_build_scope == "calibrated-only":
        if not args.calibration_file:
            raise ValueError(
                "--normalization-build-scope calibrated-only richiede --calibration-file."
            )

        applied_keys = {
            (str(item["codec"]), str(item["config"]))
            for item in calibration_report.get("applied", [])
        }

        points = [
            p for p in points
            if (str(p.codec), str(p.config)) in applied_keys
        ]

        if not points:
            raise ValueError(
                "Nessun punto disponibile per normalization-build-scope=calibrated-only."
            )

    num_points_after_build_scope = len(points)

    warning = None
    comparability = "global"

    if args.mode == "local":
        comparability = "local_only"
        warning = (
            "This normalization profile is local to the calibration dataset and "
            "target machine; costs are not globally comparable."
        )
    elif args.mode == "dataset":
        comparability = "dataset_only"
        warning = (
            "This normalization profile is dataset-specific; costs are comparable "
            "only within the same dataset/profile."
        )

    profile = build_normalization_profile(
        points=points,
        domain=args.domain,
        mode=args.mode,
        source=args.calibration_file,
        comparability=comparability,
        warning=warning,
        build_scope=args.normalization_build_scope,
    )

    profile["calibration"] = calibration_report
    profile["num_points_before_build_scope"] = num_points_before_build_scope
    profile["num_points_after_build_scope"] = num_points_after_build_scope

    save_normalization_profile(profile, args.out)

    print("\n=== R-D-E normalization profile ===")
    print(f"Domain: {profile['domain']}")
    print(f"Points: {profile['num_points']}")
    print(f"Build scope: {profile.get('build_scope')}")
    print(
        "Build-scope points: "
        f"{profile.get('num_points_after_build_scope')} / "
        f"{profile.get('num_points_before_build_scope')}"
    )
    print(f"Output: {args.out}")
    print("Scales:")
    for key, value in profile["scales"].items():
        print(f"  {key}: min={value['min']}, max={value['max']}")


if __name__ == "__main__":
    main()
