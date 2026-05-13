import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def _parse_roots(value: str) -> Dict[str, Path]:
    roots = {}

    for item in str(value).split(";"):
        item = item.strip()
        if not item:
            continue

        if "=" not in item:
            raise ValueError(f"Invalid root mapping: {item}")

        dataset, path = item.split("=", 1)
        dataset = dataset.strip()
        path = path.strip()

        if not dataset or not path:
            raise ValueError(f"Invalid root mapping: {item}")

        roots[dataset] = Path(path)

    if not roots:
        raise ValueError("No dataset roots provided.")

    return roots


def _read_expected_images(
    csv_path: str, *, dataset_col: str, image_col: str
) -> List[Dict[str, str]]:
    seen = set()
    rows = []

    with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            dataset = str(row.get(dataset_col, "")).strip()
            image = str(row.get(image_col, "")).strip()

            if not dataset or not image:
                continue

            key = (dataset, image)

            if key in seen:
                continue

            seen.add(key)
            rows.append({"dataset": dataset, "image": image})

    if not rows:
        raise ValueError("No expected dataset/image rows found.")

    return rows


def _list_image_files(root: Path) -> Dict[str, Path]:
    files = {}

    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            files.setdefault(p.name, p)

    return files


def build_image_manifest(
    *,
    csv_path: str,
    roots: Dict[str, Path],
    dataset_col: str = "dataset",
    image_col: str = "image",
    require_all: bool = True,
) -> List[Dict[str, Any]]:
    expected = _read_expected_images(
        csv_path,
        dataset_col=dataset_col,
        image_col=image_col,
    )

    files_by_dataset = {
        dataset: _list_image_files(root) for dataset, root in roots.items()
    }

    manifest = []
    missing = []

    for row in expected:
        dataset = row["dataset"]
        image = row["image"]

        if dataset not in roots:
            missing.append((dataset, image, "missing_dataset_root"))
            continue

        matched = files_by_dataset[dataset].get(Path(image).name)

        if matched is None:
            missing.append((dataset, image, "missing_file"))
            continue

        with Image.open(matched) as im:
            width = int(im.width)
            height = int(im.height)
            mode = str(im.mode)

        manifest.append(
            {
                "dataset": dataset,
                "image": image,
                "path": str(matched),
                "width": width,
                "height": height,
                "pixels": width * height,
                "mode": mode,
                "status": "matched",
            }
        )

    if missing and require_all:
        preview = "; ".join(f"{d}/{i}:{reason}" for d, i, reason in missing[:10])
        raise FileNotFoundError(
            f"Missing {len(missing)} expected images. Preview: {preview}"
        )

    return manifest


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build an image manifest by matching benchmark CSV dataset/image names "
            "to local files."
        )
    )

    parser.add_argument("--csv", required=True)
    parser.add_argument("--roots", required=True)
    parser.add_argument("--dataset-col", default="dataset")
    parser.add_argument("--image-col", default="image")
    parser.add_argument("--out", default="results/routing_context/v09_image_manifest.csv")
    parser.add_argument("--allow-missing", action="store_true")

    args = parser.parse_args()

    roots = _parse_roots(args.roots)

    manifest = build_image_manifest(
        csv_path=args.csv,
        roots=roots,
        dataset_col=args.dataset_col,
        image_col=args.image_col,
        require_all=not args.allow_missing,
    )

    write_csv(args.out, manifest)

    print("\n=== R-D-E Image Manifest ===")
    print(f"Images matched: {len(manifest)}")
    print(f"Output:         {args.out}")

    by_dataset = {}
    for row in manifest:
        by_dataset[row["dataset"]] = by_dataset.get(row["dataset"], 0) + 1

    for dataset, count in sorted(by_dataset.items()):
        print(f"  {dataset}: {count}")


if __name__ == "__main__":
    main()
