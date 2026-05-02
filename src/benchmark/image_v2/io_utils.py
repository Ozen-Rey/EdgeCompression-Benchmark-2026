from pathlib import Path
import pandas as pd

from .config import IMAGE_EXTS, RESULTS_ROOT


def get_images(dataset_dir):
    dataset_dir = Path(dataset_dir).expanduser()

    images = []
    for p in dataset_dir.rglob("*"):
        if p.suffix.lower() in IMAGE_EXTS:
            images.append(p)

    return sorted(images)


def dataset_output_dir(dataset_name):
    out_dir = RESULTS_ROOT / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def dataset_metrics_csv(dataset_name):
    return dataset_output_dir(dataset_name) / f"{dataset_name}_12metric_benchmark.csv"


def dataset_energy_csv(dataset_name):
    return dataset_output_dir(dataset_name) / f"{dataset_name}_energy_benchmark.csv"


def append_to_csv(
    rows, out_csv, key_cols=("dataset", "codec", "param", "image", "pipeline")
):
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    new_df = pd.DataFrame(rows)

    if len(new_df) == 0:
        print(f"[WARN] Nessuna riga da salvare in {out_csv}")
        return new_df

    if out_csv.exists():
        old_df = pd.read_csv(out_csv)

        all_cols = list(dict.fromkeys(list(old_df.columns) + list(new_df.columns)))
        old_df = old_df.reindex(columns=all_cols)
        new_df = new_df.reindex(columns=all_cols)

        df = pd.concat([old_df, new_df], ignore_index=True)

        existing_keys = [c for c in key_cols if c in df.columns]
        if existing_keys:
            df = df.drop_duplicates(subset=existing_keys, keep="last")
    else:
        df = new_df

    df.to_csv(out_csv, index=False)
    print(f"[OK] Salvato: {out_csv}")
    return df
