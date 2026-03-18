import os
import sys
import time
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from pytorch_msssim import ms_ssim
import lpips

sys.path.insert(0, os.path.expanduser("~/tesi"))

KODAK_DIR = os.path.expanduser("~/tesi/datasets/kodak")
RESULTS_DIR = os.path.expanduser("~/tesi/results/images")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

loss_fn_lpips = lpips.LPIPS(net="alex").to(device)


def psnr(img1, img2):
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return 100.0
    return 20 * np.log10(255.0 / np.sqrt(mse))


def compute_msssim(img1_np, img2_np):
    t1 = torch.from_numpy(img1_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    t2 = torch.from_numpy(img2_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return ms_ssim(t1, t2, data_range=1.0).item()


def compute_lpips(img1_np, img2_np):
    def to_tensor(img):
        t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        t = t * 2.0 - 1.0
        return t.to(device)

    with torch.no_grad():
        d = loss_fn_lpips(to_tensor(img1_np), to_tensor(img2_np))
    return d.item()


def img_to_tensor(img_path):
    img = Image.open(img_path).convert("RGB")
    x = torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    
    # Padding a multiplo di 64 richiesto da CompressAI
    h, w = x.shape[2], x.shape[3]
    pad_h = (64 - h % 64) % 64
    pad_w = (64 - w % 64) % 64
    if pad_h > 0 or pad_w > 0:
        x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
    
    return x.to(device), np.array(img), h, w


def compress_neural(model, img_path):
    x, orig_np, orig_h, orig_w = img_to_tensor(img_path)
    total_pixels = orig_h * orig_w  # usa dimensioni originali per bpp

    with torch.no_grad():
        torch.cuda.synchronize()
        start = time.time()
        out = model.compress(x)
        torch.cuda.synchronize()
        elapsed_enc = time.time() - start

        torch.cuda.synchronize()
        start = time.time()
        x_hat = model.decompress(out["strings"], out["shape"])["x_hat"]
        torch.cuda.synchronize()
        elapsed_dec = time.time() - start

    # crop alle dimensioni originali
    x_hat = x_hat[:, :, :orig_h, :orig_w]
    
    num_bits = sum(len(s[0]) for s in out["strings"]) * 8
    bpp = num_bits / total_pixels

    rec_np = (np.nan_to_num(x_hat.squeeze(0).permute(1,2,0).cpu().numpy()) * 255).clip(0,255).astype(np.uint8)

    return {
        "bpp": bpp,
        "psnr": psnr(orig_np, rec_np),
        "ms_ssim": compute_msssim(orig_np, rec_np),
        "lpips": compute_lpips(orig_np, rec_np),
        "enc_ms": round(elapsed_enc * 1000, 2),
        "dec_ms": round(elapsed_dec * 1000, 2)
    }


def run_neural_benchmark():
    from compressai.zoo import bmshj2018_hyperprior, cheng2020_anchor

    os.makedirs(RESULTS_DIR, exist_ok=True)
    images = sorted(Path(KODAK_DIR).glob("*.png"))
    rows = []

    # ── Ballé 2018 ──────────────────────────────────────────────
    qualities = [1, 2, 3, 4, 5, 6, 7, 8]
    for q in qualities:
        print(f"\nBallé2018 qualità {q}...")
        model = bmshj2018_hyperprior(quality=q, pretrained=True).to(device)
        model.eval()

        results = []
        start_batch = time.time()
        for img_path in tqdm(images, desc=f"Ballé2018 q={q}"):
            metrics = compress_neural(model, img_path)
            results.append(metrics)
            rows.append(
                {"image": img_path.name, "codec": "Balle2018", "quality": q, **metrics}
            )

            pd.DataFrame(rows).to_csv(
                os.path.join(RESULTS_DIR, "neural_images_kodak.csv"), index=False
            )

        elapsed_batch = time.time() - start_batch
        fps = len(images) / elapsed_batch

        del model
        torch.cuda.empty_cache()

        print(
            f"q={q} | bpp={np.mean([r['bpp'] for r in results]):.3f} | "
            f"psnr={np.mean([r['psnr'] for r in results]):.2f} dB | "
            f"lpips={np.mean([r['lpips'] for r in results]):.4f} | "
            f"enc_ms={np.mean([r['enc_ms'] for r in results]):.1f} | "
            f"fps={fps:.2f}"
        )

    # ── Cheng 2020 ──────────────────────────────────────────────
    qualities_cheng = [1, 2, 3, 4, 5, 6]
    for q in qualities_cheng:
        print(f"\nCheng2020 qualità {q}...")
        model = cheng2020_anchor(quality=q, pretrained=True).to(device)
        model.eval()

        results = []
        start_batch = time.time()
        for img_path in tqdm(images, desc=f"Cheng2020 q={q}"):
            metrics = compress_neural(model, img_path)
            results.append(metrics)
            rows.append(
                {"image": img_path.name, "codec": "Cheng2020", "quality": q, **metrics}
            )

            pd.DataFrame(rows).to_csv(
                os.path.join(RESULTS_DIR, "neural_images_kodak.csv"), index=False
            )
        elapsed_batch = time.time() - start_batch
        fps = len(images) / elapsed_batch

        del model
        torch.cuda.empty_cache()

        print(
            f"q={q} | bpp={np.mean([r['bpp'] for r in results]):.3f} | "
            f"psnr={np.mean([r['psnr'] for r in results]):.2f} dB | "
            f"lpips={np.mean([r['lpips'] for r in results]):.4f} | "
            f"enc_ms={np.mean([r['enc_ms'] for r in results]):.1f} | "
            f"fps={fps:.2f}"
        )

    df = pd.DataFrame(rows)
    rd_curve = df.groupby(["codec", "quality"])[
        ["bpp", "psnr", "ms_ssim", "lpips", "enc_ms", "dec_ms"]
    ].mean()
    print("\nCurva R-D su Kodak:")
    print(rd_curve.to_string())

    # salva separati per codec
    for codec_name in ["Balle2018", "Cheng2020"]:
        subset = df[df["codec"] == codec_name]
        out = os.path.join(RESULTS_DIR, f"{codec_name.lower()}_kodak.csv")
        subset.to_csv(out, index=False)
        print(f"Salvato {out}")


if __name__ == "__main__":
    run_neural_benchmark()
