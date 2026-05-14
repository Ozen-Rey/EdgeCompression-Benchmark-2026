import contextlib
import io
import json
from pathlib import Path
from uuid import uuid4

from src.router.observability.decision_replay import (
    main as decision_replay_main,
    replay_decision_receipt,
)
from src.router.rde_router import main as router_main
from src.router.version import ROUTER_VERSION


def _tmp_dir(name: str) -> Path:
    root = (
        Path(__file__).with_name("_tmp")
        / "decision_replay"
        / f"{name}_{uuid4().hex}"
    )
    root.mkdir(parents=True, exist_ok=True)
    return root


def _write_points(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "codec,config,rate,quality,energy,time_ms",
                "JPEG,q=85,0.4,95,1.0,10",
                "JXL,d=1.0,0.8,95,2.0,20",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _run_router_report(root: Path) -> tuple[Path, Path]:
    csv_path = root / "points.csv"
    report_path = root / "router_report.json"
    _write_points(csv_path)

    with contextlib.redirect_stdout(io.StringIO()):
        router_main(
            [
                "--csv",
                str(csv_path),
                "--codec-col",
                "codec",
                "--config-col",
                "config",
                "--rate-col",
                "rate",
                "--quality-col",
                "quality",
                "--energy-col",
                "energy",
                "--time-col",
                "time_ms",
                "--quality-target",
                "preview",
                "--quality-floor",
                "50",
                "--normalization-mode",
                "runtime",
                "--out",
                str(report_path),
            ]
        )

    return csv_path, report_path


def test_decision_replay_reproduces_router_report_decision():
    root = _tmp_dir("success")
    _, report_path = _run_router_report(root)
    out = root / "replay.json"

    replay = replay_decision_receipt(
        receipt_path=str(report_path),
        out_path=str(out),
    )

    assert out.exists()
    assert replay["mode"] == "decision_replay_validation"
    assert replay["router_version"] == ROUTER_VERSION
    assert replay["input_hashes_match"] is True
    assert replay["replay_success"] is True
    assert replay["decision_reproduced"] is True
    assert replay["mismatches"] == []


def test_decision_replay_detects_tampered_decision_receipt():
    root = _tmp_dir("tampered_decision")
    _, report_path = _run_router_report(root)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    receipt = report["decision_receipt"]
    receipt["decision"]["selected_codec"] = "NOT_THE_CODEC"
    tampered = root / "tampered_receipt.json"
    tampered.write_text(json.dumps(receipt), encoding="utf-8")

    replay = replay_decision_receipt(
        receipt_path=str(tampered),
        out_path=str(root / "tampered_replay.json"),
    )

    assert replay["input_hashes_match"] is True
    assert replay["replay_success"] is True
    assert replay["decision_reproduced"] is False
    assert replay["mismatches"][0]["field"] == "selected_codec"


def test_decision_replay_rejects_input_hash_mismatch_without_replay():
    root = _tmp_dir("hash_mismatch")
    csv_path, report_path = _run_router_report(root)
    csv_path.write_text(
        csv_path.read_text(encoding="utf-8")
        + "HEVC,crf=15,0.1,99,0.1,1\n",
        encoding="utf-8",
    )

    replay = replay_decision_receipt(
        receipt_path=str(report_path),
        out_path=str(root / "hash_mismatch_replay.json"),
    )

    assert replay["input_hashes_match"] is False
    assert replay["replay_success"] is False
    assert replay["decision_reproduced"] is False
    assert any(not item["match"] for item in replay["artifact_checks"])


def test_decision_replay_cli_writes_report():
    root = _tmp_dir("cli")
    _, report_path = _run_router_report(root)
    out = root / "cli_replay.json"

    decision_replay_main(
        [
            "--receipt",
            str(report_path),
            "--out",
            str(out),
        ]
    )

    replay = json.loads(out.read_text(encoding="utf-8"))
    assert replay["decision_reproduced"] is True
