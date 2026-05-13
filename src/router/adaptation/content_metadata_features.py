import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


def _parse_float(value: Any) -> Optional[float]:
    try:
        if value is None or str(value).strip() == "":
            return None
        return float(str(value).replace(",", "."))
    except Exception:
        return None


def _make_image_id(dataset: str, image: str) -> str:
    return f"{dataset}::{image}"


def _resolution_class(megapixels: Optional[float]) -> str:
    if megapixels is None:
        return "unknown"
    if megapixels < 0.5:
        return "small"
    if megapixels < 2.0:
        return "medium"
    if megapixels < 8.0:
        return "large"
    return "huge"


def _orientation_class(width: Optional[float], height: Optional[float]) -> str:
    if width is None or height is None or width <= 0 or height <= 0:
        return "unknown"

    ratio = width / height

    if ratio > 1.15:
        return "landscape"
    if ratio < 1.0 / 1.15:
        return "portrait"
    return "squareish"


def extract_metadata_features(
    csv_path: str,
    *,
    dataset_col: str = "dataset",
    image_col: str = "image",
    width_col: str = "width",
    height_col: str = "height",
    pixels_col: str = "pixels",
) -> List[Dict[str, Any]]:
    images: Dict[str, Dict[str, Any]] = {}

    with Path(csv_path).open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        for row in reader:
            dataset = str(row.get(dataset_col, "")).strip()
            image = str(row.get(image_col, "")).strip()

            if not dataset or not image:
                continue

            image_id = _make_image_id(dataset, image)

            if image_id in images:
                continue

            width = _parse_float(row.get(width_col))
            height = _parse_float(row.get(height_col))
            pixels = _parse_float(row.get(pixels_col))

            if pixels is None and width is not None and height is not None:
                pixels = width * height

            megapixels = pixels / 1_000_000.0 if pixels is not None else None
            aspect_ratio = (
                width / height
                if width is not None and height is not None and height > 0
                else None
            )

            images[image_id] = {
                "dataset": dataset,
                "image": image,
                "image_id": image_id,
                "width": width,
                "height": height,
                "pixels": pixels,
                "megapixels": megapixels,
                "aspect_ratio": aspect_ratio,
                "orientation_class": _orientation_class(width, height),
                "resolution_class": _resolution_class(megapixels),
            }

    if not images:
        raise ValueError("No image metadata features were extracted.")

    return list(images.values())


def load_oracle_by_image(path: str) -> Dict[str, Dict[str, Any]]:
    oracle: Dict[str, Dict[str, Any]] = {}

    with Path(path).open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        for row in reader:
            image_id = str(row.get("image_id", "")).strip()
            if image_id:
                oracle[image_id] = row

    if not oracle:
        raise ValueError("No oracle rows were loaded.")

    return oracle


def join_metadata_with_oracle(
    metadata_rows: List[Dict[str, Any]],
    oracle_rows_by_image: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    joined = []

    for meta in metadata_rows:
        image_id = meta["image_id"]
        oracle = oracle_rows_by_image.get(image_id)

        if oracle is None:
            continue

        row = dict(meta)

        for key in [
            "oracle_codec",
            "oracle_config",
            "oracle_rate",
            "oracle_quality",
            "oracle_energy",
            "oracle_time_ms",
            "oracle_cost",
            "global_codec",
            "global_config",
            "global_cost",
            "global_feasible",
            "regret",
        ]:
            row[key] = oracle.get(key)

        joined.append(row)

    if not joined:
        raise ValueError("Metadata/oracle join produced no rows.")

    return joined


def summarize_metadata_oracle(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summary = []

    def add(section: str, key: str, value: Any) -> None:
        summary.append(
            {
                "section": section,
                "key": key,
                "value": value,
            }
        )

    add("summary", "num_images", len(rows))

    oracle_counter = Counter(
        f"{r.get('oracle_codec')}|{r.get('oracle_config')}"
        for r in rows
    )

    add("oracle", "num_distinct_oracle_configs", len(oracle_counter))

    for key, count in oracle_counter.most_common():
        add("oracle_count", key, count)

    for group_key in ["dataset", "resolution_class", "orientation_class"]:
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for row in rows:
            grouped[str(row.get(group_key, "unknown"))].append(row)

        for group_value, group_rows in sorted(grouped.items()):
            add(f"{group_key}_count", group_value, len(group_rows))

            group_oracles = Counter(
                f"{r.get('oracle_codec')}|{r.get('oracle_config')}"
                for r in group_rows
            )

            for oracle_key, count in group_oracles.most_common():
                add(
                    f"{group_key}_oracle_count",
                    f"{group_value}::{oracle_key}",
                    count,
                )

    return summary


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())

    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract metadata-only content features and join them with "
            "per-image R-D-E oracle labels."
        )
    )

    parser.add_argument("--csv", required=True)
    parser.add_argument("--oracle-by-image", required=True)

    parser.add_argument("--dataset-col", default="dataset")
    parser.add_argument("--image-col", default="image")
    parser.add_argument("--width-col", default="width")
    parser.add_argument("--height-col", default="height")
    parser.add_argument("--pixels-col", default="pixels")

    parser.add_argument(
        "--features-out",
        default="results/routing_context/v09_content_metadata_features.csv",
    )

    parser.add_argument(
        "--joined-out",
        default="results/routing_context/v09_content_metadata_oracle.csv",
    )

    parser.add_argument(
        "--summary-out",
        default="results/routing_context/v09_content_metadata_summary.csv",
    )

    args = parser.parse_args()

    metadata = extract_metadata_features(
        args.csv,
        dataset_col=args.dataset_col,
        image_col=args.image_col,
        width_col=args.width_col,
        height_col=args.height_col,
        pixels_col=args.pixels_col,
    )

    oracle = load_oracle_by_image(args.oracle_by_image)
    joined = join_metadata_with_oracle(metadata, oracle)
    summary = summarize_metadata_oracle(joined)

    write_csv(args.features_out, metadata)
    write_csv(args.joined_out, joined)
    write_csv(args.summary_out, summary)

    print("\n=== R-D-E Content Metadata Features ===")
    print(f"Images:          {len(metadata)}")
    print(f"Joined images:   {len(joined)}")
    print(f"Features CSV:    {args.features_out}")
    print(f"Joined CSV:      {args.joined_out}")
    print(f"Summary CSV:     {args.summary_out}")


if __name__ == "__main__":
    main()
