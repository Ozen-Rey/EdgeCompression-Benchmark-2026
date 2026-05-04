from pathlib import Path

from src.router.normalization_profile import (
    build_normalization_profile,
    load_normalization_profile,
    normalize_with_profile,
    save_normalization_profile,
)
from src.router.rde_database import RDEPoint


def _tmp_path(name: str) -> Path:
    tmp_dir = Path(__file__).with_name("_tmp") / "normalization_idempotence"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    return tmp_dir / name


def test_normalization_profile_save_load_is_idempotent():
    points = [
        RDEPoint(
            codec="JPEG",
            config="q=85",
            rate=1.6,
            quality=81.1,
            energy=0.1,
            time_ms=6.3,
            raw={},
        ),
        RDEPoint(
            codec="JXL",
            config="d=1.0",
            rate=1.37,
            quality=85.18,
            energy=2.55,
            time_ms=134.0,
            raw={},
        ),
        RDEPoint(
            codec="HEVC",
            config="crf=15",
            rate=2.95,
            quality=91.36,
            energy=15.06,
            time_ms=342.0,
            raw={},
        ),
    ]

    profile = build_normalization_profile(
        points=points,
        domain="image",
        mode="global",
        comparability="global",
    )

    path = _tmp_path("normalization.json")
    save_normalization_profile(profile, str(path))
    loaded = load_normalization_profile(str(path))

    point = points[1]

    direct = normalize_with_profile(point, profile)
    reloaded = normalize_with_profile(point, loaded)

    assert direct == reloaded
