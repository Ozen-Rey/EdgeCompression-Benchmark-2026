import argparse
from pathlib import Path
import pandas as pd

from .config import DATASETS


def keep_row(row):
    codec = row["codec"]
    pipeline = row["pipeline"]

    if codec in ["JPEG", "JXL", "HEVC", "JPEG_AI"]:
        return pipeline == "end_to_end"

    if codec in ["Ballé", "Cheng", "ELIC", "TCM", "DCAE"]:
        return pipeline == "actual_bitstream"

    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(DATASETS.keys()))
    args = parser.parse_args()

    dataset = args.dataset
    in_csv = Path(
        f"/home/user/tesi/results/images/{dataset}/{dataset}_12metric_benchmark.csv"
    )
    out_csv = Path(
        f"/home/user/tesi/results/images/{dataset}/{dataset}_12metric_paper_ready.csv"
    )

    df = pd.read_csv(in_csv)
    paper = df[df.apply(keep_row, axis=1)].copy()
    paper = paper.sort_values(["codec", "param", "image"]).reset_index(drop=True)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    paper.to_csv(out_csv, index=False)

    print(f"[OK] Salvato: {out_csv}")
    print("Righe:", len(paper))
    print()
    print(paper.groupby(["codec", "param", "pipeline"]).size())
    print()
    print(
        paper.groupby(["codec", "param"])[
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
        .sort_values(["codec", "bpp"])
        .to_string()
    )


if __name__ == "__main__":
    main()
