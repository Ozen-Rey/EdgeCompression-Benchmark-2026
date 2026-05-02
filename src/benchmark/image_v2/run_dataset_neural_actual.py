import argparse
import gc
import torch
from tqdm import tqdm

from .config import DATASETS
from .io_utils import get_images, dataset_metrics_csv, append_to_csv
from .codecs_neural_actual import (
    benchmark_neural_actual_one,
    load_balle,
    load_cheng,
    load_elic,
    load_tcm,
    load_dcae,
)

torch.backends.cudnn.enabled = False


def run_model_on_dataset(
    dataset_name,
    images,
    codec_name,
    param,
    loader_fn,
    pad_multiple=64,
    pad_mode="right",
):
    print("\n" + "=" * 80)
    print(f"{codec_name} {param} | actual_bitstream")
    print("=" * 80)

    net, params_m = loader_fn()

    rows = []

    for img_path in tqdm(images):
        try:
            row = benchmark_neural_actual_one(
                net=net,
                dataset_name=dataset_name,
                codec_name=codec_name,
                param=param,
                img_path=img_path,
                pad_multiple=pad_multiple,
                pad_mode=pad_mode,
                params_m=params_m,
            )
            rows.append(row)

        except Exception as e:
            print()
            print(f"[ERROR] {codec_name} {param} su {img_path.name}")
            print(str(e)[-3000:])
            print()

    del net
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return rows


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", required=True, choices=list(DATASETS.keys()))
    parser.add_argument(
        "--codecs",
        nargs="+",
        default=["balle", "cheng", "elic", "tcm", "dcae"],
        choices=["balle", "cheng", "elic", "tcm", "dcae"],
    )
    parser.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()

    dataset_name = args.dataset
    dataset_dir = DATASETS[dataset_name]
    images = get_images(dataset_dir)

    if args.limit is not None:
        images = images[: args.limit]

    print("=" * 80)
    print("NEURAL ACTUAL BITSTREAM BENCHMARK")
    print(f"dataset: {dataset_name}")
    print(f"dataset_dir: {dataset_dir}")
    print(f"images: {len(images)}")
    print(f"codecs: {args.codecs}")
    print("=" * 80)

    all_rows = []

    if "balle" in args.codecs:
        for q in [1, 3, 5, 7]:
            all_rows.extend(
                run_model_on_dataset(
                    dataset_name=dataset_name,
                    images=images,
                    codec_name="Ballé",
                    param=f"q={q}",
                    loader_fn=lambda q=q: load_balle(q),
                    pad_multiple=64,
                    pad_mode="right",
                )
            )

    if "cheng" in args.codecs:
        for q in [1, 3, 5]:
            all_rows.extend(
                run_model_on_dataset(
                    dataset_name=dataset_name,
                    images=images,
                    codec_name="Cheng",
                    param=f"q={q}",
                    loader_fn=lambda q=q: load_cheng(q),
                    pad_multiple=64,
                    pad_mode="right",
                )
            )

    if "elic" in args.codecs:
        for lam in ["0.008", "0.032", "0.150", "0.450"]:
            all_rows.extend(
                run_model_on_dataset(
                    dataset_name=dataset_name,
                    images=images,
                    codec_name="ELIC",
                    param=f"lam={lam}",
                    loader_fn=lambda lam=lam: load_elic(lam),
                    pad_multiple=64,
                    pad_mode="right",
                )
            )

    if "tcm" in args.codecs:
        for lam in ["0.0035", "0.013"]:
            all_rows.extend(
                run_model_on_dataset(
                    dataset_name=dataset_name,
                    images=images,
                    codec_name="TCM",
                    param=f"lam={lam}",
                    loader_fn=lambda lam=lam: load_tcm(lam),
                    pad_multiple=128,
                    pad_mode="right",
                )
            )

    if "dcae" in args.codecs:
        for lam in ["0.0035", "0.013"]:
            all_rows.extend(
                run_model_on_dataset(
                    dataset_name=dataset_name,
                    images=images,
                    codec_name="DCAE",
                    param=f"lam={lam}",
                    loader_fn=lambda lam=lam: load_dcae(lam),
                    pad_multiple=128,
                    pad_mode="center",
                )
            )

        if len(all_rows) == 0:
            print("\n[WARN] Nessuna riga prodotta. Controlla gli errori sopra.")
            return

    out_csv = dataset_metrics_csv(dataset_name)
    df = append_to_csv(all_rows, out_csv)

    print("\nSUMMARY")
    if "pipeline" in df.columns:
        sub = df[df["pipeline"] == "actual_bitstream"]
    else:
        sub = df

    if len(sub) == 0:
        print("[WARN] Nessuna riga actual_bitstream nel CSV.")
        return

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
        .sort_values(["codec", "bpp"])
        .to_string()
    )


if __name__ == "__main__":
    main()
