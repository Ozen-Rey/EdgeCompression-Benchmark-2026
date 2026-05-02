from pathlib import Path
import pandas as pd

IN_CSV = Path("/home/user/tesi/results/images/kodak/kodak_12metric_benchmark.csv")
OUT_CSV = Path("/home/user/tesi/results/images/kodak/kodak_12metric_paper_ready.csv")


def keep_row(row):
    codec = row["codec"]
    pipeline = row["pipeline"]

    if codec in ["JPEG", "JXL", "HEVC", "JPEG_AI"]:
        return pipeline == "end_to_end"

    if codec == "ELIC":
        return pipeline == "full_pipeline"

    if codec in ["Ballé", "Cheng", "TCM", "DCAE"]:
        return pipeline == "forward_pass_only"

    return False


def main():
    df = pd.read_csv(IN_CSV)

    paper = df[df.apply(keep_row, axis=1)].copy()

    paper = paper.sort_values(["codec", "param", "image"]).reset_index(drop=True)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    paper.to_csv(OUT_CSV, index=False)

    print(f"[OK] Salvato: {OUT_CSV}")
    print()
    print("Righe:", len(paper))
    print()
    print(paper.groupby(["codec", "param"]).size())
    print()
    print("Pipeline:")
    print(paper.groupby(["codec", "pipeline"]).size())

    print()
    print("Medie principali:")
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
