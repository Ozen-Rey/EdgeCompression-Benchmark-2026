from pathlib import Path
import json

from src.router.quality_thresholds import resolve_quality_floor


def _write_thresholds_fixture(name: str, thresholds: dict) -> Path:
    tmp_dir = Path(__file__).with_name("_tmp")
    tmp_dir.mkdir(exist_ok=True)
    path = tmp_dir / name
    path.write_text(json.dumps(thresholds), encoding="utf-8")
    return path


def test_quality_threshold_uses_domain_target_floor():
    thresholds = {
        "image": {
            "default_metric": "ssimulacra2",
            "higher_is_better": True,
            "targets": {
                "preview": 50.0,
                "normal": 50.0,
                "high": 75.0,
                "very-high": 90.0,
            },
        }
    }

    path = _write_thresholds_fixture("quality_thresholds_domain.json", thresholds)

    report = resolve_quality_floor(
        domain="image",
        quality_metric="ssimulacra2",
        quality_target="high",
        user_quality_floor=70.0,
        thresholds_file=str(path),
    )

    assert report["target_floor"] == 75.0
    assert report["user_quality_floor"] == 70.0
    assert report["effective_quality_floor"] == 75.0


def test_quality_threshold_user_floor_can_be_stricter():
    thresholds = {
        "image": {
            "default_metric": "ssimulacra2",
            "higher_is_better": True,
            "targets": {
                "high": 80.0,
            },
        }
    }

    path = _write_thresholds_fixture("quality_thresholds_user.json", thresholds)

    report = resolve_quality_floor(
        domain="image",
        quality_metric="ssimulacra2",
        quality_target="high",
        user_quality_floor=85.0,
        thresholds_file=str(path),
    )

    assert report["target_floor"] == 80.0
    assert report["user_quality_floor"] == 85.0
    assert report["effective_quality_floor"] == 85.0
    assert report["policy"] == "max(target_floor,user_floor)"
