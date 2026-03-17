import pandas as pd
import matplotlib.pyplot as plt
import os

RESULTS_DIR = os.path.expanduser("~/tesi/results/images")

jpeg = pd.read_csv(f"{RESULTS_DIR}/jpeg_kodak.csv")
hevc = pd.read_csv(f"{RESULTS_DIR}/hevc_kodak.csv")
balle = pd.read_csv(f"{RESULTS_DIR}/balle2018_kodak.csv")
cheng = pd.read_csv(f"{RESULTS_DIR}/cheng2020_kodak.csv")

jpeg_rd = jpeg.groupby("quality")[["bpp", "psnr", "lpips"]].mean().sort_values("bpp")
hevc_rd = hevc.groupby("crf")[["bpp", "psnr", "lpips"]].mean().sort_values("bpp")
balle_rd = balle.groupby("quality")[["bpp", "psnr", "lpips"]].mean().sort_values("bpp")
cheng_rd = cheng.groupby("quality")[["bpp", "psnr", "lpips"]].mean().sort_values("bpp")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

for ax, metric, ylabel in [
    (ax1, "psnr", "PSNR (dB)"),
    (ax2, "lpips", "LPIPS (↓ meglio)"),
]:
    ax.plot(jpeg_rd["bpp"], jpeg_rd[metric], "b-o", label="JPEG", markersize=4)
    ax.plot(hevc_rd["bpp"], hevc_rd[metric], "r-s", label="H.265/HEVC", markersize=4)
    ax.plot(balle_rd["bpp"], balle_rd[metric], "g-^", label="Ballé2018", markersize=4)
    ax.plot(cheng_rd["bpp"], cheng_rd[metric], "m-D", label="Cheng2020", markersize=4)
    ax.set_xlabel("Bitrate (bpp)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Rate-Distortion Immagini: {metric.upper()}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 2.5)

plt.tight_layout()
out = f"{RESULTS_DIR}/rd_curves_images_v2.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Salvato {out}")
plt.show()
