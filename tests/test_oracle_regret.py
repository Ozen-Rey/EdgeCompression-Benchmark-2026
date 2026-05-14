import csv
import json
import math
from pathlib import Path

from src.router.rde_router import main
from tests.conftest import scratch_root


def _tmp_path(name: str) -> Path:
    tmp_dir = scratch_root() / "oracle_regret"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    return tmp_dir / name


def _norm_log(value, values):
    logs = [math.log10(v) for v in values]
    lo = min(logs)
    hi = max(logs)

    if hi <= lo:
        return 0.0

    return (math.log10(value) - lo) / (hi - lo)


def _norm_quality_to_distortion(value, values):
    lo = min(values)
    hi = max(values)

    if hi <= lo:
        return 0.0

    quality_norm = (value - lo) / (hi - lo)
    return 1.0 - quality_norm


def _oracle_argmin(csv_path, weights, min_quality):
    rows = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            quality = float(row["ssimulacra2"])
            if quality < min_quality:
                continue

            rows.append(
                {
                    "codec": row["codec"],
                    "config": row["param"],
                    "rate": float(row["bpp"]),
                    "quality": quality,
                    "energy": float(row["energy_per_image_j"]),
                }
            )

    rates = [r["rate"] for r in rows]
    qualities = [r["quality"] for r in rows]
    energies = [r["energy"] for r in rows]

    scored = []

    for row in rows:
        r_n = _norm_log(row["rate"], rates)
        e_n = _norm_log(row["energy"], energies)
        d_n = _norm_quality_to_distortion(row["quality"], qualities)

        cost = (
            weights["w_R"] * r_n
            + weights["w_E"] * e_n
            + weights["w_D"] * d_n
        )

        scored.append((cost, row))

    scored.sort(key=lambda item: item[0])
    return scored[0][1]


def test_router_has_zero_regret_against_independent_rde_oracle():
    csv_path = _tmp_path("oracle.csv")
    out_path = _tmp_path("report.json")

    csv_path.write_text(
        "\n".join(
            [
                "codec,param,bpp,ssimulacra2,energy_per_image_j,time_ms",
                "CodecA,mode=a,1.0,90.0,100.0,100.0",
                "CodecB,mode=b,2.0,95.0,1.0,100.0",
                "CodecC,mode=c,0.5,80.0,10.0,100.0",
            ]
        ),
        encoding="utf-8",
    )

    weights = {
        "w_R": 0.2,
        "w_E": 0.2,
        "w_D": 0.6,
    }

    expected = _oracle_argmin(
        csv_path=csv_path,
        weights=weights,
        min_quality=50.0,
    )

    main(
        [
            "--csv",
            str(csv_path),
            "--codec-col",
            "codec",
            "--config-col",
            "param",
            "--rate-col",
            "bpp",
            "--quality-col",
            "ssimulacra2",
            "--energy-col",
            "energy_per_image_j",
            "--time-col",
            "time_ms",
            "--quality-target",
            "normal",
            "--wR",
            str(weights["w_R"]),
            "--wE",
            str(weights["w_E"]),
            "--wD",
            str(weights["w_D"]),
            "--out",
            str(out_path),
        ]
    )

    report = json.loads(out_path.read_text(encoding="utf-8"))
    selected = report["decision"]["selected"]

    assert selected["codec"] == expected["codec"]
    assert selected["config"] == expected["config"]
