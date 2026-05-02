import argparse
from tqdm import tqdm

from .config import DATASETS
from .io_utils import get_images, dataset_metrics_csv, append_to_csv
from .codecs_classical import (
    compress_jpeg_metrics,
    compress_jxl_metrics,
    compress_hevc_metrics,
)


def run_dataset_metrics(dataset_name, codecs=("jpeg", "jxl", "hevc"), limit=None):
    dataset_dir = DATASETS[dataset_name]
    images = get_images(dataset_dir)

    if limit is not None:
        images = images[:limit]

    print("=" * 80)
    print(f"DATASET: {dataset_name}")
    print(f"DIR: {dataset_dir}")
    print(f"IMAGES: {len(images)}")
    print("=" * 80)

    rows = []

    if "jpeg" in codecs:
        for q in [10, 30, 60, 85]:
            print(f"\nJPEG q={q}")
            for img_path in tqdm(images):
                rows.append(compress_jpeg_metrics(dataset_name, img_path, q))

    if "jxl" in codecs:
        for d in [1.0, 3.0, 7.0, 12.0]:
            print(f"\nJXL d={d}")
            for img_path in tqdm(images):
                rows.append(compress_jxl_metrics(dataset_name, img_path, d))

    if "hevc" in codecs:
        for crf in [15, 25, 35, 45]:
            print(f"\nHEVC crf={crf}")
            for img_path in tqdm(images):
                rows.append(compress_hevc_metrics(dataset_name, img_path, crf))

    out_csv = dataset_metrics_csv(dataset_name)
    df = append_to_csv(rows, out_csv)

    print("\nRIEPILOGO")
    print(
        df.groupby(["codec", "param"])[["bpp", "psnr", "ms_ssim", "lpips", "time_ms"]]
        .mean()
        .sort_values(["codec", "bpp"])
        .to_string()
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(DATASETS.keys()))
    parser.add_argument("--codecs", nargs="+", default=["jpeg", "jxl", "hevc"])
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    run_dataset_metrics(
        dataset_name=args.dataset,
        codecs=args.codecs,
        limit=args.limit,
    )
