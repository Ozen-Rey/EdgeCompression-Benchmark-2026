import os
import sys
import time
import subprocess
import tempfile
import torch
import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from pytorch_msssim import ms_ssim
import lpips

sys.path.insert(0, os.path.expanduser("~/tesi"))

UVG_DIR = os.path.expanduser("~/tesi/datasets/uvg")
RESULTS_DIR = os.path.expanduser("~/tesi/results/video")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loss_fn_lpips = lpips.LPIPS(net="alex").to(device)

WIDTH, HEIGHT, FPS, N_FRAMES = 1920, 1080, 120, 120

SEQUENCES = [
    "Beauty_1920x1080_120fps_420_8bit_YUV.yuv",
    "Bosphorus_1920x1080_120fps_420_8bit_YUV.yuv",
    "HoneyBee_1920x1080_120fps_420_8bit_YUV.yuv",
]


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
    def to_t(img):
        t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        return (t * 2.0 - 1.0).to(device)

    with torch.no_grad():
        return loss_fn_lpips(to_t(img1_np), to_t(img2_np)).item()


def extract_frames(yuv_path, n_frames=N_FRAMES):
    """Estrae i primi n_frames dal file YUV come array numpy RGB."""
    frames = []
    with tempfile.TemporaryDirectory() as tmp:
        out_pattern = os.path.join(tmp, "frame_%04d.png")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "rawvideo",
                "-pixel_format",
                "yuv420p",
                "-video_size",
                f"{WIDTH}x{HEIGHT}",
                "-framerate",
                str(FPS),
                "-i",
                yuv_path,
                "-frames:v",
                str(n_frames),
                out_pattern,
            ],
            capture_output=True,
            check=True,
        )
        for i in range(1, n_frames + 1):
            p = os.path.join(tmp, f"frame_{i:04d}.png")
            if os.path.exists(p):
                from PIL import Image

                frames.append(np.array(Image.open(p).convert("RGB")))
    return frames


def compress_video(yuv_path, codec, param_value, n_frames=N_FRAMES):
    """Comprime n_frames con il codec specificato e calcola le metriche."""
    total_pixels = WIDTH * HEIGHT

    with tempfile.TemporaryDirectory() as tmp:
        # input: primi n_frames del YUV
        input_clip = os.path.join(tmp, "input.yuv")
        bytes_per_frame = WIDTH * HEIGHT * 3 // 2  # YUV420
        with open(yuv_path, "rb") as fin, open(input_clip, "wb") as fout:
            fout.write(fin.read(bytes_per_frame * n_frames))

        output_video = os.path.join(
            tmp, f"compressed.{'mp4' if codec != 'av1' else 'mkv'}"
        )
        decoded_dir = os.path.join(tmp, "decoded")
        os.makedirs(decoded_dir)

        # formato raw senza container overhead
        # scegli encoder
        if codec == "h264":
            vcodec = "libx264"
            output_video = os.path.join(tmp, "compressed.264")
            quality_param = ["-crf", str(param_value)]
        elif codec == "h265":
            vcodec = "libx265"
            output_video = os.path.join(tmp, "compressed.265")
            quality_param = [
                "-crf",
                str(param_value),
                "-x265-params",
                "log-level=error",
            ]
        elif codec == "av1":
            vcodec = "libaom-av1"
            output_video = os.path.join(tmp, "compressed.ivf")
            quality_param = ["-crf", str(param_value), "-b:v", "0"]
        else:
            raise ValueError(f"Codec non supportato: {codec}")

        # codifica
        start = time.time()
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "rawvideo",
                "-pixel_format",
                "yuv420p",
                "-video_size",
                f"{WIDTH}x{HEIGHT}",
                "-framerate",
                str(FPS),
                "-i",
                input_clip,
                "-c:v",
                vcodec,
                "-preset",
                "medium" if codec != "av1" else "4",
                *quality_param,
                output_video,
            ],
            capture_output=True,
            check=True,
        )
        enc_time = time.time() - start

        compressed_bits = os.path.getsize(output_video) * 8
        bpp = compressed_bits / (total_pixels * n_frames)
        fps_enc = n_frames / enc_time

        # decodifica — per raw streams serve specificare il codec
        if codec == "h264":
            dec_cmd = [
                "ffmpeg",
                "-y",
                "-c:v",
                "h264",
                "-i",
                output_video,
                os.path.join(decoded_dir, "frame_%04d.png"),
            ]
        elif codec == "h265":
            dec_cmd = [
                "ffmpeg",
                "-y",
                "-c:v",
                "hevc",
                "-i",
                output_video,
                os.path.join(decoded_dir, "frame_%04d.png"),
            ]
        elif codec == "av1":
            dec_cmd = [
                "ffmpeg",
                "-y",
                "-i",
                output_video,
                os.path.join(decoded_dir, "frame_%04d.png"),
            ]
        else:
            raise ValueError(f"Codec non supportato per decodifica: {codec}")
        subprocess.run(dec_cmd, capture_output=True, check=True)

        # calcola metriche frame per frame
        from PIL import Image

        orig_frames = extract_frames(yuv_path, n_frames)
        psnr_vals, msssim_vals, lpips_vals = [], [], []

        for i, orig in enumerate(orig_frames):
            dec_path = os.path.join(decoded_dir, f"frame_{i+1:04d}.png")
            if not os.path.exists(dec_path):
                continue
            rec = np.array(Image.open(dec_path).convert("RGB"))
            psnr_vals.append(psnr(orig, rec))
            msssim_vals.append(compute_msssim(orig, rec))
            lpips_vals.append(compute_lpips(orig, rec))

        return {
            "bpp": round(bpp, 6),
            "psnr": round(np.mean(psnr_vals), 4),
            "ms_ssim": round(np.mean(msssim_vals), 6),
            "lpips": round(np.mean(lpips_vals), 6),
            "fps_enc": round(fps_enc, 3),
            "enc_time_s": round(enc_time, 2),
        }


def run_video_benchmark():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = []
    out = os.path.join(RESULTS_DIR, "video_benchmark.csv")

    codecs = {
        "h264": [51, 45, 40, 35, 30, 25, 20],
        "h265": [51, 45, 40, 35, 30, 25, 20],
        # "av1":  [63, 55, 48, 40, 32],  # rimosso crf=24
    }

    for codec, crf_values in codecs.items():
        print(f"\n=== {codec.upper()} ===")
        n_frames = 24 if codec == "av1" else 120

        for crf in crf_values:
            psnr_seq, msssim_seq, lpips_seq, bpp_seq, fps_seq = [], [], [], [], []

            for seq_name in tqdm(SEQUENCES, desc=f"{codec} crf={crf}"):
                yuv_path = os.path.join(UVG_DIR, seq_name)
                try:
                    m = compress_video(yuv_path, codec, crf, n_frames=n_frames)
                    psnr_seq.append(m["psnr"])
                    msssim_seq.append(m["ms_ssim"])
                    lpips_seq.append(m["lpips"])
                    bpp_seq.append(m["bpp"])
                    fps_seq.append(m["fps_enc"])
                    rows.append(
                        {
                            "sequence": seq_name.split("_")[0],
                            "codec": codec,
                            "crf": crf,
                            **m,
                        }
                    )
                    # salva dopo ogni sequenza
                    pd.DataFrame(rows).to_csv(out, index=False)
                except Exception as e:
                    print(f"  ERRORE {seq_name}: {e}")

            if psnr_seq:
                print(
                    f"  crf={crf} | "
                    f"bpp={np.mean(bpp_seq):.4f} | "
                    f"psnr={np.mean(psnr_seq):.2f}dB | "
                    f"ms_ssim={np.mean(msssim_seq):.4f} | "
                    f"lpips={np.mean(lpips_seq):.4f} | "
                    f"fps={np.mean(fps_seq):.1f}"
                )

    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(f"\nRisultati salvati in {out}")
    print("\n--- Curva R-D media per codec ---")
    print(
        df.groupby(["codec", "crf"])[["bpp", "psnr", "ms_ssim", "lpips", "fps_enc"]]
        .mean()
        .to_string()
    )


if __name__ == "__main__":
    run_video_benchmark()
