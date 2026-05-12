import csv
import hashlib
import json
from pathlib import Path
from uuid import uuid4

from src.router.calibration_apply import main


def _tmp_dir(name: str) -> Path:
    root = Path(__file__).with_name("_tmp") / "calibration_bundle" / f"{name}_{uuid4().hex}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_benchmark(path: Path) -> None:
    path.write_text(
        "codec,config,rate,quality,energy,time_ms\n"
        "JXL,d=1.0,1.2,85,2.0,100\n",
        encoding="utf-8",
    )


def _write_calibration(path: Path) -> None:
    _write_json(
        path,
        {
            "version": "0.18.0",
            "level": "quick",
            "energy_mode": "auto",
            "created_at": "test",
            "measured": ["time_ms", "output_bytes", "local_bpp"],
            "summary": {
                "JXL": {
                    "d=1.0": {
                        "success_rate": 1.0,
                        "time_ms": {"mean": 300.0},
                        "local_bpp": {"mean": 0.8},
                    }
                }
            },
        },
    )


def _write_promotion(path: Path, *, include_gpu_only_energy: bool = False) -> None:
    calibration_profile = [
        {
            "codec": "JXL",
            "config": "d=1.0",
            "axis": "rate",
            "scale": 1.25,
            "status": "accepted",
            "num_eval_rows": 4,
            "mean_abs_log_error_before": 0.3,
            "mean_abs_log_error_after": 0.1,
            "improvement_ratio": 0.66,
        }
    ]

    if include_gpu_only_energy:
        calibration_profile.append(
            {
                "codec": "JXL",
                "config": "d=1.0",
                "axis": "energy",
                "scale": 2.0,
                "status": "accepted",
                "num_eval_rows": 4,
                "energy_scope": "gpu",
            }
        )

    _write_json(
        path,
        {
            "version": "0.16.0",
            "mode": "candidate_profile_only",
            "applied_by_router": False,
            "calibration_profile": calibration_profile,
            "rejected": [
                {
                    "codec": "JXL",
                    "config": "d=1.0",
                    "axis": "time",
                    "scale": 0.5,
                    "status": "rejected",
                    "num_eval_rows": 4,
                }
            ],
        },
    )


def _run_bundle(root: Path, *, manifest: bool = True, gpu_only_energy: bool = False):
    benchmark = root / "benchmark.csv"
    calibration = root / "calibration.json"
    promotion = root / "promotion.json"
    out = root / "calibrated.csv"
    manifest_out = root / "manifest.json"

    _write_benchmark(benchmark)
    _write_calibration(calibration)
    _write_promotion(promotion, include_gpu_only_energy=gpu_only_energy)

    args = [
        "--benchmark",
        str(benchmark),
        "--calibration",
        str(calibration),
        "--promotion-profile",
        str(promotion),
        "--out",
        str(out),
    ]

    if manifest:
        args.extend(["--manifest-out", str(manifest_out)])

    main(args)
    return benchmark, calibration, promotion, out, manifest_out


def test_manifest_is_created_when_manifest_out_is_passed():
    root = _tmp_dir("created")
    *_, manifest = _run_bundle(root)

    assert manifest.exists()


def test_manifest_contains_version_mode_source_paths_and_output_path():
    root = _tmp_dir("fields")
    benchmark, calibration, promotion, out, manifest = _run_bundle(root)
    data = json.loads(manifest.read_text(encoding="utf-8"))

    assert data["artifact_type"] == "promoted_calibration_bundle"
    assert data["router_version"] == "0.18.0"
    assert data["mode"] == "explicit_opt_in_calibration_apply"
    assert data["source_benchmark"] == str(benchmark)
    assert data["source_calibration"] == str(calibration)
    assert data["promotion_profile"] == str(promotion)
    assert data["output_csv"] == str(out)
    assert data["energy_policy"]["requires_energy_usable_for_total"] is True
    assert data["energy_policy"]["gpu_only_energy_excluded"] is True


def test_manifest_output_csv_sha256_is_correct():
    root = _tmp_dir("hash")
    *_, out, manifest = _run_bundle(root)
    data = json.loads(manifest.read_text(encoding="utf-8"))

    assert data["hashes"]["output_csv_sha256"] == _sha256(out)


def test_manifest_accepted_scales_contains_only_applied_accepted_scales():
    root = _tmp_dir("accepted")
    *_, manifest = _run_bundle(root)
    data = json.loads(manifest.read_text(encoding="utf-8"))

    assert len(data["accepted_scales"]) == 1
    assert data["accepted_scales"][0]["axis"] == "rate"
    assert data["accepted_scales"][0]["status"] == "accepted"
    assert data["accepted_scales"][0]["scale"] == 1.25


def test_rejected_scales_do_not_appear_among_applied_scales():
    root = _tmp_dir("rejected")
    *_, manifest = _run_bundle(root)
    data = json.loads(manifest.read_text(encoding="utf-8"))

    axes = {item["axis"] for item in data["accepted_scales"]}
    assert "time" not in axes
    assert data["rejected_scales_count"] >= 1


def test_gpu_only_energy_does_not_produce_accepted_energy_scale():
    root = _tmp_dir("gpu_energy")
    *_, out, manifest = _run_bundle(root, gpu_only_energy=True)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    with out.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    axes = {item["axis"] for item in data["accepted_scales"]}
    assert "energy" not in axes
    assert float(rows[0]["energy"]) == 6.0
    assert data["rejected_scales_count"] >= 1


def test_without_manifest_out_no_manifest_is_written():
    root = _tmp_dir("no_manifest")
    *_, manifest = _run_bundle(root, manifest=False)

    assert not manifest.exists()
