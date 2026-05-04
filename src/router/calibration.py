import argparse
import csv
import json
import statistics
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from .codec_capabilities import build_execution_plan
    from .system_probe import probe_system
except ImportError:
    from codec_capabilities import build_execution_plan
    from system_probe import probe_system


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


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
        raise FileNotFoundError(f"Input directory non trovata: {root}")

    images = [
        p
        for p in sorted(root.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if not images:
        raise ValueError(f"Nessuna immagine trovata in: {root}")

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
    t0 = time.perf_counter()

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )

    t1 = time.perf_counter()

    return {
        "success": result.returncode == 0,
        "returncode": result.returncode,
        "time_ms": (t1 - t0) * 1000.0,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


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
    summary_csv: Optional[str] = None,
) -> Dict[str, Any]:
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
                        "success": False,
                        "returncode": None,
                        "time_ms": None,
                        "output_bytes": None,
                        "local_bpp": None,
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

                    run = _run_command(command)
                    record["success"] = run["success"]
                    record["returncode"] = run["returncode"]
                    record["time_ms"] = run["time_ms"]

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
        "version": "0.3",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "level": level,
        "dry_run": dry_run,
        "input_dir": str(input_dir),
        "num_images": len(images),
        "images": [str(p) for p in images],
        "codecs": list(configs_by_codec.keys()),
        "configs_by_codec": configs_by_codec,
        "repeats": repeats,
        "measured": ["time_ms", "output_bytes", "local_bpp"],
        "estimated": [],
        "not_calibrated": ["quality", "energy"],
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

            summary[codec][config] = {
                "num_runs": len(rows),
                "num_success": len(successful),
                "success_rate": len(successful) / len(rows) if rows else 0.0,
                "time_ms": _summarize(time_values),
                "local_bpp": _summarize(bpp_values),
                "output_bytes": _summarize(bytes_values),
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
    ]

    rows = []

    for codec, config_map in summary.items():
        for config, stats in config_map.items():
            time_stats = stats.get("time_ms", {})
            bpp_stats = stats.get("local_bpp", {})
            bytes_stats = stats.get("output_bytes", {})

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
        help="Livello di calibrazione locale.",
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        help="Cartella contenente immagini di calibrazione.",
    )

    parser.add_argument(
        "--codecs",
        default="JPEG,JXL,HEVC",
        help="Lista codec separata da virgole. Default: JPEG,JXL,HEVC.",
    )

    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Numero massimo di immagini. Se assente, usa il default del livello.",
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=None,
        help="Numero di ripetizioni per punto. Se assente, usa il default del livello.",
    )

    parser.add_argument(
        "--out",
        required=True,
        help="Path del file JSON di calibrazione.",
    )

    parser.add_argument(
        "--summary-csv",
        default=None,
        help="Path opzionale per esportare una sintesi CSV della calibrazione.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Genera il piano di calibrazione senza eseguire gli encoder.",
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
        summary_csv=args.summary_csv,
    )

    print("\n=== R-D-E Router Local Calibration ===")
    print(f"Level:       {report['level']}")
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
