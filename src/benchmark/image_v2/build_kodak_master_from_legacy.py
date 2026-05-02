from pathlib import Path
import pandas as pd

from .config import RESULTS_ROOT, ALL_COLS

ROOT = Path("/home/user/tesi")
IMAGES_RESULTS = ROOT / "results" / "images"

LEGACY_12METRIC_FILES = [
    IMAGES_RESULTS / "missing_12metric_benchmark.csv",
    IMAGES_RESULTS / "sota_12metric_benchmark.csv",
    ROOT / "results" / "jxl_vs_elic_benchmark.csv",
]

OUT_CSV = RESULTS_ROOT / "kodak" / "kodak_12metric_benchmark.csv"


def normalize_param_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "param" not in df.columns:
        if "quality" in df.columns:
            df["param"] = "q=" + df["quality"].astype(str)
        elif "crf" in df.columns:
            df["param"] = "crf=" + df["crf"].astype(str)
        elif "distance" in df.columns:
            df["param"] = "d=" + df["distance"].astype(str)
        elif "lambda" in df.columns:
            df["param"] = "lam=" + df["lambda"].astype(str)
        elif "lam" in df.columns:
            df["param"] = "lam=" + df["lam"].astype(str)
        else:
            df["param"] = "unknown"

    return df


def add_missing_spatial_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in ["width", "height", "pixels"]:
        if col not in df.columns:
            df[col] = None

    return df


def normalize_codec_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "codec" in df.columns:
        df["codec"] = df["codec"].replace(
            {
                "JPEGXL": "JXL",
                "JPEG XL": "JXL",
                "jpegxl": "JXL",
                "jxl": "JXL",
                "elic": "ELIC",
                "balle": "Ballé",
                "Balle": "Ballé",
            }
        )

    return df


def load_legacy_file(path: Path) -> pd.DataFrame:
    print(f"\n[LOAD] {path}")

    df = pd.read_csv(path)
    print(f"  rows: {len(df)}")
    print(f"  columns: {df.columns.tolist()}")

    df = normalize_param_columns(df)
    df = normalize_codec_names(df)
    df = add_missing_spatial_columns(df)

    if "dataset" not in df.columns:
        df.insert(0, "dataset", "kodak")
    else:
        df["dataset"] = "kodak"

    if "pipeline" not in df.columns:
        df["pipeline"] = "unknown"

    for col in ALL_COLS:
        if col not in df.columns:
            df[col] = None

    standard = [c for c in ALL_COLS if c in df.columns]
    extras = [c for c in df.columns if c not in standard]
    df = df[standard + extras]

    if "codec" in df.columns:
        print("  codecs:")
        print(df["codec"].value_counts().to_string())

    return df


def main():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    dfs = []

    for path in LEGACY_12METRIC_FILES:
        if not path.exists():
            print(f"[SKIP] Non trovato: {path}")
            continue

        df = load_legacy_file(path)
        dfs.append(df)

    if not dfs:
        raise RuntimeError("Nessun CSV legacy trovato.")

    master = pd.concat(dfs, ignore_index=True)

    key_cols = ["dataset", "codec", "param", "image", "pipeline"]
    available_keys = [c for c in key_cols if c in master.columns]

    before = len(master)
    master = master.drop_duplicates(subset=available_keys, keep="last")
    after = len(master)

    print(f"\n[DEDUP] {before} -> {after} rows")

    sort_cols = [c for c in ["codec", "param", "image"] if c in master.columns]
    master = master.sort_values(sort_cols).reset_index(drop=True)

    master.to_csv(OUT_CSV, index=False)

    print(f"\n[OK] Kodak master salvato in:")
    print(f"  {OUT_CSV}")

    print("\n[COUNTS]")
    print(master.groupby(["codec", "param"]).size().to_string())

    print("\n[PIPELINES]")
    print(master.groupby(["codec", "pipeline"]).size().to_string())

    print("\n[MEANS]")
    cols = [
        c
        for c in ["bpp", "psnr", "ms_ssim", "lpips", "dists", "ssimulacra2", "time_ms"]
        if c in master.columns
    ]
    print(
        master.groupby(["codec", "param"])[cols]
        .mean(numeric_only=True)
        .sort_values(["codec", "bpp"])
        .to_string()
    )


if __name__ == "__main__":
    main()
