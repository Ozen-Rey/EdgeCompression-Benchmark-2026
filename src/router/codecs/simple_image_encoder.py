import argparse
from pathlib import Path


def encode_jpeg(input_path: str, output_path: str, quality: int) -> None:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Pillow non è installato. Installa con: pip install pillow"
        ) from exc

    input_path = Path(input_path)
    output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(input_path) as img:
        if img.mode not in {"RGB", "L"}:
            img = img.convert("RGB")

        img.save(
            output_path,
            format="JPEG",
            quality=quality,
            optimize=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal image encoder backend.")
    parser.add_argument("--codec", required=True, choices=["jpeg"])
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--quality", type=int, default=85)

    args = parser.parse_args()

    if args.codec == "jpeg":
        encode_jpeg(args.input, args.output, args.quality)


if __name__ == "__main__":
    main()