import argparse
import csv
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

import numpy as np
from PIL import Image, ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or str(value).strip() == "":
            return default
        return float(value)
    except Exception:
        return default


def _resize_for_features(image: Image.Image, resize_long_side: int) -> Image.Image:
    if resize_long_side <= 0:
        return image.copy()

    width, height = image.size
    long_side = max(width, height)

    if long_side <= resize_long_side:
        return image.copy()

    scale = resize_long_side / float(long_side)
    new_size = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )

    try:
        resample = Image.Resampling.LANCZOS
    except AttributeError:
        resample = Image.LANCZOS

    return image.resize(new_size, resample=resample)


def _entropy_bits_from_uint8(values: np.ndarray) -> float:
    flat = values.astype(np.uint8).ravel()
    hist = np.bincount(flat, minlength=256).astype(np.float64)

    total = hist.sum()
    if total <= 0:
        return 0.0

    probs = hist / total
    probs = probs[probs > 0.0]

    return float(-(probs * np.log2(probs)).sum())


def _class_entropy(entropy_norm: float) -> str:
    if entropy_norm < 0.45:
        return "low_entropy"
    if entropy_norm < 0.70:
        return "medium_entropy"
    return "high_entropy"


def _class_edges(edge_density: float) -> str:
    if edge_density < 0.05:
        return "low_edge"
    if edge_density < 0.15:
        return "medium_edge"
    return "high_edge"


def _class_texture(texture_score: float) -> str:
    if texture_score < 0.20:
        return "flat"
    if texture_score < 0.45:
        return "moderate"
    if texture_score < 0.70:
        return "textured"
    return "highly_textured"


def _class_color(colorfulness: float) -> str:
    if colorfulness < 5.0:
        return "grayscale_like"
    if colorfulness < 25.0:
        return "normal_color"
    return "colorful"


def _class_resolution(megapixels: float) -> str:
    if megapixels < 0.5:
        return "small"
    if megapixels < 2.0:
        return "medium"
    if megapixels < 8.0:
        return "large"
    return "huge"


def _class_orientation(width: int, height: int) -> str:
    if width <= 0 or height <= 0:
        return "unknown"

    ratio = width / height

    if ratio > 1.15:
        return "landscape"
    if ratio < 1.0 / 1.15:
        return "portrait"
    return "squareish"


def extract_image_features(
    path: str,
    *,
    resize_long_side: int = 256,
    edge_threshold: float = 20.0,
    flat_threshold: float = 2.0,
) -> Dict[str, Any]:
    start = time.perf_counter()

    image_path = Path(path)

    with Image.open(image_path) as im:
        original_mode = im.mode
        original_width, original_height = im.size
        has_alpha = original_mode in {"RGBA", "LA"} or (
            original_mode == "P" and "transparency" in im.info
        )

        resized = _resize_for_features(im, resize_long_side)
        resized_width, resized_height = resized.size

        rgb = resized.convert("RGB")
        arr = np.asarray(rgb, dtype=np.float32)

    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]

    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    luminance_uint8 = np.clip(np.rint(luminance), 0, 255).astype(np.uint8)

    luminance_mean = float(luminance.mean())
    luminance_std = float(luminance.std())

    entropy_bits = _entropy_bits_from_uint8(luminance_uint8)
    entropy_norm = entropy_bits / 8.0

    gx = np.zeros_like(luminance)
    gy = np.zeros_like(luminance)

    gx[:, 1:] = luminance[:, 1:] - luminance[:, :-1]
    gy[1:, :] = luminance[1:, :] - luminance[:-1, :]

    gradient = np.sqrt(gx * gx + gy * gy)

    gradient_mean = float(gradient.mean())
    gradient_std = float(gradient.std())
    edge_density = float((gradient > edge_threshold).mean())
    flat_area_ratio = float((gradient < flat_threshold).mean())

    rg = r - g
    yb = 0.5 * (r + g) - b

    std_rg = float(rg.std())
    std_yb = float(yb.std())
    mean_rg = float(rg.mean())
    mean_yb = float(yb.mean())

    colorfulness = math.sqrt(std_rg**2 + std_yb**2) + 0.3 * math.sqrt(
        mean_rg**2 + mean_yb**2
    )

    pixels = int(original_width * original_height)
    megapixels = pixels / 1_000_000.0
    aspect_ratio = original_width / original_height if original_height > 0 else 0.0

    texture_score = 0.5 * min(luminance_std / 64.0, 1.0) + 0.5 * min(
        gradient_mean / 32.0,
        1.0,
    )

    overhead_ms = (time.perf_counter() - start) * 1000.0

    return {
        "path": str(image_path),
        "width": int(original_width),
        "height": int(original_height),
        "pixels": pixels,
        "megapixels": megapixels,
        "aspect_ratio": aspect_ratio,
        "original_mode": original_mode,
        "has_alpha": bool(has_alpha),
        "resize_long_side": int(resize_long_side),
        "resized_width": int(resized_width),
        "resized_height": int(resized_height),
        "luminance_mean": luminance_mean,
        "luminance_std": luminance_std,
        "luminance_entropy_bits": entropy_bits,
        "luminance_entropy_norm": entropy_norm,
        "gradient_mean": gradient_mean,
        "gradient_std": gradient_std,
        "edge_density": edge_density,
        "flat_area_ratio": flat_area_ratio,
        "colorfulness": float(colorfulness),
        "resolution_class": _class_resolution(megapixels),
        "orientation_class": _class_orientation(original_width, original_height),
        "entropy_class": _class_entropy(entropy_norm),
        "edge_class": _class_edges(edge_density),
        "texture_class": _class_texture(texture_score),
        "color_class": _class_color(float(colorfulness)),
        "feature_overhead_ms": overhead_ms,
    }


def load_manifest(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("path"):
                rows.append(dict(row))

    if not rows:
        raise ValueError("No manifest rows loaded.")

    return rows


def extract_features_from_manifest(
    manifest_path: str,
    *,
    resize_long_side: int = 256,
) -> List[Dict[str, Any]]:
    manifest = load_manifest(manifest_path)
    rows: List[Dict[str, Any]] = []

    for item in manifest:
        features = extract_image_features(
            item["path"],
            resize_long_side=resize_long_side,
        )

        row = {
            "dataset": item.get("dataset"),
            "image": item.get("image"),
            "image_id": f"{item.get('dataset')}::{item.get('image')}",
            **features,
        }

        rows.append(row)

    return rows


def summarize_feature_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summary: List[Dict[str, Any]] = []

    def add(section: str, key: str, value: Any) -> None:
        summary.append(
            {
                "section": section,
                "key": key,
                "value": value,
            }
        )

    add("summary", "num_images", len(rows))

    overheads = [_safe_float(r.get("feature_overhead_ms")) for r in rows]
    add("overhead", "mean_ms", mean(overheads) if overheads else None)
    add("overhead", "max_ms", max(overheads) if overheads else None)

    for key in [
        "dataset",
        "resolution_class",
        "orientation_class",
        "entropy_class",
        "edge_class",
        "texture_class",
        "color_class",
    ]:
        counter = Counter(str(r.get(key, "unknown")) for r in rows)

        for value, count in counter.most_common():
            add(f"{key}_count", value, count)

    for dataset, group in _group_by(rows, "dataset").items():
        luminance_std = [_safe_float(r.get("luminance_std")) for r in group]
        entropy = [_safe_float(r.get("luminance_entropy_norm")) for r in group]
        edge_density = [_safe_float(r.get("edge_density")) for r in group]
        colorfulness = [_safe_float(r.get("colorfulness")) for r in group]

        add("dataset_mean_luminance_std", dataset, mean(luminance_std))
        add("dataset_mean_entropy_norm", dataset, mean(entropy))
        add("dataset_mean_edge_density", dataset, mean(edge_density))
        add("dataset_mean_colorfulness", dataset, mean(colorfulness))

    return summary


def _group_by(rows: List[Dict[str, Any]], key: str) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for row in rows:
        grouped[str(row.get(key, "unknown"))].append(row)

    return grouped


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
        description="Extract pixel-level content features from an image manifest."
    )

    parser.add_argument("--manifest", required=True)
    parser.add_argument("--resize-long-side", type=int, default=256)
    parser.add_argument(
        "--features-out",
        default="results/routing_context/v09_image_pixel_features.csv",
    )
    parser.add_argument(
        "--summary-out",
        default="results/routing_context/v09_image_pixel_features_summary.csv",
    )

    args = parser.parse_args()

    rows = extract_features_from_manifest(
        args.manifest,
        resize_long_side=args.resize_long_side,
    )

    summary = summarize_feature_rows(rows)

    write_csv(args.features_out, rows)
    write_csv(args.summary_out, summary)

    overheads = [_safe_float(r.get("feature_overhead_ms")) for r in rows]

    print("\n=== R-D-E Image Pixel Features ===")
    print(f"Images:             {len(rows)}")
    print(f"Resize long side:   {args.resize_long_side}")
    print(f"Mean overhead ms:   {mean(overheads) if overheads else None}")
    print(f"Max overhead ms:    {max(overheads) if overheads else None}")
    print(f"Features CSV:       {args.features_out}")
    print(f"Summary CSV:        {args.summary_out}")


if __name__ == "__main__":
    main()
