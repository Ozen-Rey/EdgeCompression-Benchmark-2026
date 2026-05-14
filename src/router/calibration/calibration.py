import argparse
import csv
import json
import statistics
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.router.version import ROUTER_VERSION
from src.utils.energy_backends import (
    CompositeEnergyMeter,
    collect_energy_backend_diagnostics,
)
from src.router.codecs.codec_capabilities import build_execution_plan
from src.router.adaptation.system_probe import probe_system


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

ENERGY_MODES = ("auto", "require-measured-total", "benchmark-only")


DEFAULT_CONFIGS = {
    "quick": {
        "JPEG": ["q=60"],
        "JXL": ["d=1.0"],
        "HEVC": ["crf=25"],
    },
    "standard": {
        "JPEG": ["q=60", "q=85"],
        "JXL": ["d=1.0", "d=3.0"],
        "HEVC": ["crf=15", "crf=25"],
    },
    "full": {
        "JPEG": ["q=10", "q=30", "q=60", "q=85"],
        "JXL": ["d=1.0", "d=3.0", "d=7.0", "d=12.0"],
        "HEVC": ["crf=15", "crf=25", "crf=35", "crf=45"],
    },
}


DEFAULT_MAX_IMAGES = {
    "quick": 1,
    "standard": 8,
    "full": None,
}


DEFAULT_REPEATS = {
    "quick": 1,
    "standard": 3,
    "full": 3,
}


def _safe_name(text: str) -> str:
    return (
        text.replace("\\", "_")
        .replace("/", "_")
        .replace("=", "")
        .replace(".", "p")
        .replace(" ", "_")
        .replace(":", "_")
    )


def _find_images(input_dir: str, max_images: Optional[int]) -> List[Path]:
    root = Path(input_dir)

    if not root.exists():
        raise FileNotFoundError(f"Input directory not found: {root}")

    images = [
        p
        for p in sorted(root.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if not images:
        raise ValueError(f"No images found in: {root}")

    if max_images is not None:
        images = images[:max_images]

    return images


def _get_image_pixels(image_path: Path) -> Optional[int]:
    try:
        from PIL import Image

        with Image.open(image_path) as img:
            width, height = img.size
            return int(width * height)
    except Exception:
        return None


def _run_command(command: List[str]) -> Dict[str, Any]:
    meter = CompositeEnergyMeter()

    def run_once():
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )

    result, energy = meter.measure_callable(run_once)
    return {
        "success": result.returncode == 0,
        "returncode": result.returncode,
        "time_ms": energy.time_s * 1000.0,
        "local_cpu_energy_j": energy.cpu_j,
        "local_gpu_energy_j": energy.gpu_j,
        "local_energy_j": energy.total_j,
        "energy_backend": energy.energy_backend,
        "energy_method": energy.energy_method,
        "energy_is_measured": energy.energy_is_measured,
        "energy_quality": energy.energy_quality,
        "energy_scope": energy.energy_scope,
        "energy_usable_for_total": energy.energy_usable_for_total,
        "energy_warnings": ";".join(energy.warnings),
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def _apply_energy_mode(run: Dict[str, Any], energy_mode: str) -> Dict[str, Any]:
    out = dict(run)

    if energy_mode == "benchmark-only":
        warnings = []
        if out.get("energy_warnings"):
            warnings.append(str(out["energy_warnings"]))
        warnings.append("benchmark_only_energy_mode")

        out.update(
            {
                "local_cpu_energy_j": None,
                "local_gpu_energy_j": None,
                "local_energy_j": None,
                "energy_backend": "benchmark_only",
                "energy_method": "benchmark_only_energy_mode",
                "energy_is_measured": False,
                "energy_quality": "disabled",
                "energy_scope": "disabled",
                "energy_usable_for_total": False,
                "energy_warnings": ";".join(warnings),
            }
        )
        return out

    if energy_mode == "require-measured-total" and not out.get(
        "energy_usable_for_total", False
    ):
        raise RuntimeError(
            "Strict energy mode requires usable total hardware energy, "
            f"but got scope={out.get('energy_scope')}, "
            f"backend={out.get('energy_backend')}, "
            f"method={out.get('energy_method')}"
        )

    return out


def _summarize(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "std": None,
        }

    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def _unique_nonempty(values: List[Any]) -> List[str]:
    seen = set()
    out: List[str] = []

    for value in values:
        if value is None:
            continue

        text = str(value).strip()
        if not text or text in seen:
            continue

        seen.add(text)
        out.append(text)

    return out


def _select_configs(level: str, codecs: List[str]) -> Dict[str, List[str]]:
    level_configs = DEFAULT_CONFIGS[level]
    selected = {}

    for codec in codecs:
        codec_clean = codec.strip()
        if not codec_clean:
            continue

        codec_upper = codec_clean.upper()

        # Mantiene il nome canonico usato dal router.
        if codec_upper == "JPEG":
            selected["JPEG"] = level_configs.get("JPEG", [])
        elif codec_upper in {"JXL", "JPEGXL", "JPEG_XL"}:
            selected["JXL"] = level_configs.get("JXL", [])
        elif codec_upper == "HEVC":
            selected["HEVC"] = level_configs.get("HEVC", [])
        else:
            raise ValueError(
                f"Codec non supportato nella calibrazione v0.3: {codec_clean}. "
                "Per ora sono supportati JPEG, JXL, HEVC."
            )

    return selected


def _make_output_path(
    output_root: Path,
    image_path: Path,
    codec: str,
    config: str,
    repeat_index: int,
) -> Path:
    name = (
        f"{image_path.stem}__"
        f"{_safe_name(codec)}__"
        f"{_safe_name(config)}__"
        f"r{repeat_index}"
    )

    # L'estensione finale viene eventualmente corretta da build_execution_plan.
    return output_root / f"{name}.bin"


def run_calibration(
    level: str,
    input_dir: str,
    codecs: List[str],
    out_path: str,
    max_images: Optional[int],
    repeats: Optional[int],
    dry_run: bool,
    energy_mode: str = "auto",
    summary_csv: Optional[str] = None,
) -> Dict[str, Any]:
    if energy_mode not in ENERGY_MODES:
        raise ValueError(
            "energy_mode must be one of: " + ", ".join(ENERGY_MODES)
        )

    system_state = probe_system()

    if max_images is None:
        max_images = DEFAULT_MAX_IMAGES[level]

    if repeats is None:
        repeats = DEFAULT_REPEATS[level]

    images = _find_images(input_dir, max_images=max_images)
    configs_by_codec = _select_configs(level, codecs)

    out_path_obj = Path(out_path)
    out_path_obj.parent.mkdir(parents=True, exist_ok=True)

    output_root = out_path_obj.parent / "calibration_outputs" / level
    output_root.mkdir(parents=True, exist_ok=True)

    measurements: List[Dict[str, Any]] = []

    for image_path in images:
        pixels = _get_image_pixels(image_path)

        for codec, configs in configs_by_codec.items():
            for config in configs:
                for repeat_idx in range(1, repeats + 1):
                    provisional_output = _make_output_path(
                        output_root=output_root,
                        image_path=image_path,
                        codec=codec,
                        config=config,
                        repeat_index=repeat_idx,
                    )

                    plan = build_execution_plan(
                        codec_name=codec,
                        config=config,
                        input_path=str(image_path),
                        output_path=str(provisional_output),
                        system_state=system_state,
                        requested=True,
                    )

                    record: Dict[str, Any] = {
                        "image": str(image_path),
                        "pixels": pixels,
                        "codec": codec,
                        "config": config,
                        "repeat": repeat_idx,
                        "backend": plan.get("execution_backend"),
                        "can_execute": plan.get("can_execute"),
                        "command": plan.get("command"),
                        "output": plan.get("output"),
                        "plan_reasons": plan.get("reasons", []),
                        "plan_warnings": plan.get("warnings", []),
                        "dry_run": dry_run,
                        "energy_mode": energy_mode,
                        "success": False,
                        "returncode": None,
                        "time_ms": None,
                        "output_bytes": None,
                        "local_bpp": None,
                        "local_cpu_energy_j": None,
                        "local_gpu_energy_j": None,
                        "local_energy_j": None,
                        "energy_backend": None,
                        "energy_method": None,
                        "energy_is_measured": False,
                        "energy_quality": None,
                        "energy_scope": "none",
                        "energy_usable_for_total": False,
                        "energy_warnings": None,
                    }

                    if dry_run:
                        measurements.append(record)
                        continue

                    if not plan.get("can_execute"):
                        record["success"] = False
                        record["failure_reason"] = "execution_plan_not_executable"
                        measurements.append(record)
                        continue

                    command = plan.get("command")
                    if not command:
                        record["success"] = False
                        record["failure_reason"] = "missing_command"
                        measurements.append(record)
                        continue

                    run = _apply_energy_mode(_run_command(command), energy_mode)
                    record["success"] = run["success"]
                    record["returncode"] = run["returncode"]
                    record["time_ms"] = run["time_ms"]
                    record["local_cpu_energy_j"] = run["local_cpu_energy_j"]
                    record["local_gpu_energy_j"] = run["local_gpu_energy_j"]
                    record["local_energy_j"] = run["local_energy_j"]
                    record["energy_backend"] = run["energy_backend"]
                    record["energy_method"] = run["energy_method"]
                    record["energy_is_measured"] = run["energy_is_measured"]
                    record["energy_quality"] = run["energy_quality"]
                    record["energy_scope"] = run["energy_scope"]
                    record["energy_usable_for_total"] = run["energy_usable_for_total"]
                    record["energy_warnings"] = run["energy_warnings"]

                    output = plan.get("output")
                    if output and Path(output).exists():
                        output_bytes = Path(output).stat().st_size
                        record["output_bytes"] = output_bytes

                        if pixels and pixels > 0:
                            record["local_bpp"] = (output_bytes * 8.0) / pixels

                    if not run["success"]:
                        record["stderr"] = run["stderr"]
                        record["stdout"] = run["stdout"]

                    measurements.append(record)

    summary = _build_summary(measurements)

    report = {
        "version": ROUTER_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "level": level,
        "energy_mode": energy_mode,
        "dry_run": dry_run,
        "input_dir": str(input_dir),
        "num_images": len(images),
        "images": [str(p) for p in images],
        "codecs": list(configs_by_codec.keys()),
        "configs_by_codec": configs_by_codec,
        "repeats": repeats,
        "measured": [
            "time_ms",
            "output_bytes",
            "local_bpp",
            "local_energy_j_if_backend_available",
        ],
        "estimated": [],
        "not_calibrated": ["quality"],
        "energy_measurement": {
            "enabled": energy_mode != "benchmark-only",
            "mode": energy_mode,
            "backend_dependent": True,
            "fallback_behavior": "energy_is_measured_false_when_no_backend_available",
            "strict_behavior": (
                "raise_when_no_usable_total_energy_in_require_measured_total_mode"
            ),
        },
        "energy_backend_diagnostics": collect_energy_backend_diagnostics(),
        "system_state": system_state,
        "summary": summary,
        "summary_csv": summary_csv,
        "measurements": measurements,
    }

    with out_path_obj.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    if summary_csv:
        _write_summary_csv(summary, summary_csv)

    return report


def _build_summary(measurements: List[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

    for m in measurements:
        codec = m["codec"]
        config = m["config"]

        grouped.setdefault(codec, {})
        grouped[codec].setdefault(config, [])
        grouped[codec][config].append(m)

    summary: Dict[str, Any] = {}

    for codec, config_map in grouped.items():
        summary[codec] = {}

        for config, rows in config_map.items():
            successful = [r for r in rows if r.get("success")]
            time_values = [
                float(r["time_ms"])
                for r in successful
                if r.get("time_ms") is not None
            ]
            bpp_values = [
                float(r["local_bpp"])
                for r in successful
                if r.get("local_bpp") is not None
            ]
            bytes_values = [
                float(r["output_bytes"])
                for r in successful
                if r.get("output_bytes") is not None
            ]
            cpu_energy_values = [
                float(r["local_cpu_energy_j"])
                for r in successful
                if r.get("local_cpu_energy_j") is not None
            ]
            gpu_energy_values = [
                float(r["local_gpu_energy_j"])
                for r in successful
                if r.get("local_gpu_energy_j") is not None
            ]
            local_energy_values = [
                float(r["local_energy_j"])
                for r in successful
                if r.get("local_energy_j") is not None
            ]
            measured_energy_values = [
                float(r["local_energy_j"])
                for r in successful
                if r.get("local_energy_j") is not None
                and bool(r.get("energy_is_measured", False))
            ]
            usable_total_energy_values = [
                float(r["local_energy_j"])
                for r in successful
                if r.get("local_energy_j") is not None
                and bool(r.get("energy_usable_for_total", False))
            ]
            energy_backends = _unique_nonempty(
                [r.get("energy_backend") for r in successful]
            )
            energy_methods = _unique_nonempty(
                [r.get("energy_method") for r in successful]
            )
            energy_qualities = _unique_nonempty(
                [r.get("energy_quality") for r in successful]
            )
            energy_scopes = _unique_nonempty(
                [r.get("energy_scope") for r in successful]
            )
            energy_warnings = _unique_nonempty(
                [r.get("energy_warnings") for r in successful]
            )
            energy_usable_for_total = bool(usable_total_energy_values) and len(
                usable_total_energy_values
            ) == len(local_energy_values)

            summary[codec][config] = {
                "num_runs": len(rows),
                "num_success": len(successful),
                "success_rate": len(successful) / len(rows) if rows else 0.0,
                "time_ms": _summarize(time_values),
                "local_bpp": _summarize(bpp_values),
                "output_bytes": _summarize(bytes_values),
                "local_cpu_energy_j": _summarize(cpu_energy_values),
                "local_gpu_energy_j": _summarize(gpu_energy_values),
                "local_energy_j": _summarize(local_energy_values),
                "local_measured_energy_j": _summarize(measured_energy_values),
                "local_usable_total_energy_j": _summarize(usable_total_energy_values),
                "energy_is_measured": bool(measured_energy_values),
                "energy_scope": "+".join(energy_scopes) if energy_scopes else "none",
                "energy_usable_for_total": energy_usable_for_total,
                "energy_backend": ";".join(energy_backends) or None,
                "energy_method": ";".join(energy_methods) or None,
                "energy_quality": ";".join(energy_qualities) or None,
                "energy_warnings": ";".join(energy_warnings) or None,
            }

    return summary


def _write_summary_csv(summary: Dict[str, Any], path: str) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "codec",
        "config",
        "num_runs",
        "num_success",
        "success_rate",
        "time_ms_mean",
        "time_ms_median",
        "time_ms_min",
        "time_ms_max",
        "time_ms_std",
        "local_bpp_mean",
        "local_bpp_median",
        "local_bpp_min",
        "local_bpp_max",
        "local_bpp_std",
        "output_bytes_mean",
        "output_bytes_median",
        "output_bytes_min",
        "output_bytes_max",
        "output_bytes_std",
        "local_cpu_energy_j_mean",
        "local_gpu_energy_j_mean",
        "local_energy_j_mean",
        "energy_is_measured",
        "energy_scope",
        "energy_usable_for_total",
        "energy_backend",
        "energy_method",
        "energy_quality",
        "energy_warnings",
    ]

    rows = []

    for codec, config_map in summary.items():
        for config, stats in config_map.items():
            time_stats = stats.get("time_ms", {})
            bpp_stats = stats.get("local_bpp", {})
            bytes_stats = stats.get("output_bytes", {})
            cpu_energy_stats = stats.get("local_cpu_energy_j", {})
            gpu_energy_stats = stats.get("local_gpu_energy_j", {})
            local_energy_stats = stats.get("local_energy_j", {})

            rows.append(
                {
                    "codec": codec,
                    "config": config,
                    "num_runs": stats.get("num_runs"),
                    "num_success": stats.get("num_success"),
                    "success_rate": stats.get("success_rate"),

                    "time_ms_mean": time_stats.get("mean"),
                    "time_ms_median": time_stats.get("median"),
                    "time_ms_min": time_stats.get("min"),
                    "time_ms_max": time_stats.get("max"),
                    "time_ms_std": time_stats.get("std"),

                    "local_bpp_mean": bpp_stats.get("mean"),
                    "local_bpp_median": bpp_stats.get("median"),
                    "local_bpp_min": bpp_stats.get("min"),
                    "local_bpp_max": bpp_stats.get("max"),
                    "local_bpp_std": bpp_stats.get("std"),

                    "output_bytes_mean": bytes_stats.get("mean"),
                    "output_bytes_median": bytes_stats.get("median"),
                    "output_bytes_min": bytes_stats.get("min"),
                    "output_bytes_max": bytes_stats.get("max"),
                    "output_bytes_std": bytes_stats.get("std"),

                    "local_cpu_energy_j_mean": cpu_energy_stats.get("mean"),
                    "local_gpu_energy_j_mean": gpu_energy_stats.get("mean"),
                    "local_energy_j_mean": local_energy_stats.get("mean"),
                    "energy_is_measured": stats.get("energy_is_measured"),
                    "energy_scope": stats.get("energy_scope"),
                    "energy_usable_for_total": stats.get("energy_usable_for_total"),
                    "energy_backend": stats.get("energy_backend"),
                    "energy_method": stats.get("energy_method"),
                    "energy_quality": stats.get("energy_quality"),
                    "energy_warnings": stats.get("energy_warnings"),
                }
            )

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Local calibration utility for the R-D-E router."
    )

    parser.add_argument(
        "--level",
        required=True,
        choices=["quick", "standard", "full"],
        help="Local calibration level.",
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing calibration images.",
    )

    parser.add_argument(
        "--codecs",
        default="JPEG,JXL,HEVC",
        help="Comma-separated codec list. Default: JPEG,JXL,HEVC.",
    )

    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum number of images. If omitted, the level default is used.",
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=None,
        help="Number of repetitions per point. If omitted, the level default is used.",
    )

    parser.add_argument(
        "--out",
        required=True,
        help="Path to the calibration JSON output file.",
    )

    parser.add_argument(
        "--summary-csv",
        default=None,
        help="Optional path to export a CSV calibration summary.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Produce the calibration plan without running the encoders.",
    )

    parser.add_argument(
        "--energy-mode",
        default="auto",
        choices=ENERGY_MODES,
        help="Policy for local energy measurements.",
    )

    args = parser.parse_args()

    codecs = [c.strip() for c in args.codecs.split(",") if c.strip()]

    report = run_calibration(
        level=args.level,
        input_dir=args.input_dir,
        codecs=codecs,
        out_path=args.out,
        max_images=args.max_images,
        repeats=args.repeats,
        dry_run=args.dry_run,
        energy_mode=args.energy_mode,
        summary_csv=args.summary_csv,
    )

    print("\n=== R-D-E Router Local Calibration ===")
    print(f"Level:       {report['level']}")
    print(f"Energy mode: {report['energy_mode']}")
    print(f"Dry run:     {report['dry_run']}")
    print(f"Images:      {report['num_images']}")
    print(f"Codecs:      {', '.join(report['codecs'])}")
    print(f"Repeats:     {report['repeats']}")
    print(f"Output JSON: {args.out}")
    if args.summary_csv:
        print(f"Summary CSV: {args.summary_csv}")
    print()

    for codec, config_map in report["summary"].items():
        for config, stats in config_map.items():
            t = stats["time_ms"]["mean"]
            bpp = stats["local_bpp"]["mean"]

            t_str = f"{t:.3f} ms" if t is not None else "n/a"
            bpp_str = f"{bpp:.6f}" if bpp is not None else "n/a"

            print(
                f"{codec:5s} {config:8s} | "
                f"success {stats['num_success']}/{stats['num_runs']} | "
                f"time {t_str} | "
                f"bpp {bpp_str}"
            )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\n=== R-D-E Calibration: failed ===")
        print(str(exc))
        raise SystemExit(2)
