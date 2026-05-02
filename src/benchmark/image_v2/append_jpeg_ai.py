import argparse
from tqdm import tqdm

from .config import DATASETS
from .io_utils import get_images, dataset_metrics_csv, append_to_csv
from .codecs_jpeg_ai import compress_jpeg_ai_metrics


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default="kodak", choices=list(DATASETS.keys()))
    parser.add_argument("--jpeg_ai_root", required=True)
    parser.add_argument("--conda_env", default=None)
    parser.add_argument("--limit", type=int, default=None)

    parser.add_argument(
        "--target_bpps",
        nargs="+",
        type=float,
        default=[0.15, 0.25, 0.35, 0.50, 0.75, 1.00],
    )

    parser.add_argument(
        "--extra_encoder_args",
        nargs=argparse.REMAINDER,
        default=[],
    )

    args = parser.parse_args()

    dataset_name = args.dataset
    dataset_dir = DATASETS[dataset_name]
    images = get_images(dataset_dir)

    if args.limit is not None:
        images = images[: args.limit]

    print("=" * 80)
    print("JPEG AI APPEND")
    print(f"dataset: {dataset_name}")
    print(f"dataset_dir: {dataset_dir}")
    print(f"images: {len(images)}")
    print(f"jpeg_ai_root: {args.jpeg_ai_root}")
    print(f"conda_env: {args.conda_env}")
    print(f"target_bpps: {args.target_bpps}")
    print(f"extra_encoder_args: {args.extra_encoder_args}")
    print("=" * 80)

    rows = []

    for target_bpp in args.target_bpps:
        print(f"\nJPEG AI target_bpp={target_bpp}")

        for img_path in tqdm(images):
            try:
                row = compress_jpeg_ai_metrics(
                    dataset_name=dataset_name,
                    img_path=img_path,
                    target_bpp=target_bpp,
                    jpeg_ai_root=args.jpeg_ai_root,
                    conda_env=args.conda_env,
                    extra_encoder_args=args.extra_encoder_args,
                )
                rows.append(row)

            except Exception as e:
                print()
                print(f"[ERROR] {img_path.name} target_bpp={target_bpp}")
                print(str(e)[-4000:])
                print()

    out_csv = dataset_metrics_csv(dataset_name)
    df = append_to_csv(rows, out_csv)

    if len(rows) > 0:
        print("\nJPEG AI SUMMARY")
        sub = df[df["codec"] == "JPEG_AI"]
        print(
            sub.groupby(["codec", "param"])[
                [
                    "bpp",
                    "psnr",
                    "ssim",
                    "ms_ssim",
                    "lpips",
                    "dists",
                    "ssimulacra2",
                    "time_ms",
                ]
            ]
            .mean(numeric_only=True)
            .sort_values("bpp")
            .to_string()
        )


if __name__ == "__main__":
    main()
