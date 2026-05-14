import pytest
from pathlib import Path
from PIL import Image

from src.router.adaptation.content_image_manifest import (
    _parse_roots,
    build_image_manifest,
)


def _case_dir(name: str) -> Path:
    path = Path(".test_tmp") / "content_image_manifest" / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_parse_roots_parses_semicolon_mappings():
    roots = _parse_roots("kodak=datasets/kodak;tecnick=datasets/tecnick")

    assert str(roots["kodak"]) == "datasets\\kodak" or str(roots["kodak"]) == "datasets/kodak"
    assert "tecnick" in roots


def test_build_image_manifest_matches_expected_files():
    case_dir = _case_dir("match")
    csv_path = case_dir / "bench.csv"
    root = case_dir / "kodak"
    root.mkdir(exist_ok=True)

    img_path = root / "kodim01.png"
    Image.new("RGB", (10, 20)).save(img_path)

    csv_path.write_text(
        "\n".join(
            [
                "dataset,image,codec,param",
                "kodak,kodim01.png,JPEG,q=85",
                "kodak,kodim01.png,JXL,d=1.0",
            ]
        ),
        encoding="utf-8",
    )

    manifest = build_image_manifest(
        csv_path=str(csv_path),
        roots={"kodak": root},
    )

    assert len(manifest) == 1
    assert manifest[0]["dataset"] == "kodak"
    assert manifest[0]["image"] == "kodim01.png"
    assert manifest[0]["width"] == 10
    assert manifest[0]["height"] == 20
    assert manifest[0]["pixels"] == 200
    assert manifest[0]["mode"] == "RGB"


def test_build_image_manifest_raises_on_missing_file():
    case_dir = _case_dir("missing")
    csv_path = case_dir / "bench.csv"
    root = case_dir / "kodak"
    root.mkdir(exist_ok=True)

    csv_path.write_text(
        "\n".join(
            [
                "dataset,image,codec,param",
                "kodak,kodim01.png,JPEG,q=85",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="Missing 1 expected images"):
        build_image_manifest(
            csv_path=str(csv_path),
            roots={"kodak": root},
        )
