import os
import sys
import time
import torch
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import subprocess
import tempfile
from pesq import pesq as pesq_score
from pystoi import stoi

sys.path.insert(0, os.path.expanduser("~/tesi"))

AUDIO_TEST_DIR = os.path.expanduser("~/tesi/datasets/audio_test")
RESULTS_DIR = os.path.expanduser("~/tesi/results/audio")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_audio(path, target_sr=24000):
    y, sr = librosa.load(path, sr=target_sr, mono=True)
    return y, target_sr


def snr(original, reconstructed):
    noise = original - reconstructed
    signal_power = np.mean(original**2)
    noise_power = np.mean(noise**2)
    if noise_power == 0:
        return 100.0
    return 10 * np.log10(signal_power / noise_power)


def segmental_snr(original, reconstructed, sr, frame_ms=20):
    frame_len = int(sr * frame_ms / 1000)
    n_frames = len(original) // frame_len
    snr_values = []
    for i in range(n_frames):
        s = original[i * frame_len : (i + 1) * frame_len]
        r = reconstructed[i * frame_len : (i + 1) * frame_len]
        if np.mean(s**2) > 1e-6:
            snr_values.append(snr(s, r))
    return np.mean(snr_values) if snr_values else 0.0


def compute_pesq(original, reconstructed, sr):
    try:
        target_sr = 16000
        orig_16k = librosa.resample(original, orig_sr=sr, target_sr=target_sr)
        rec_16k = librosa.resample(reconstructed, orig_sr=sr, target_sr=target_sr)
        min_len = min(len(orig_16k), len(rec_16k))
        return round(
            pesq_score(target_sr, orig_16k[:min_len], rec_16k[:min_len], "wb"), 4
        )
    except Exception:
        return None


def compute_stoi(original, reconstructed, sr):
    try:
        target_sr = 16000
        orig_16k = librosa.resample(original, orig_sr=sr, target_sr=target_sr)
        rec_16k = librosa.resample(reconstructed, orig_sr=sr, target_sr=target_sr)
        min_len = min(len(orig_16k), len(rec_16k))
        return round(
            stoi(orig_16k[:min_len], rec_16k[:min_len], target_sr, extended=False), 4
        )
    except Exception:
        return None


def compute_si_sdr(original, reconstructed):
    try:
        min_len = min(len(original), len(reconstructed))
        s = original[:min_len]
        r = reconstructed[:min_len]
        # rimuovi media
        s = s - np.mean(s)
        r = r - np.mean(r)
        # proiezione
        alpha = np.dot(r, s) / (np.dot(s, s) + 1e-8)
        s_target = alpha * s
        noise = r - s_target
        return round(
            10 * np.log10(np.dot(s_target, s_target) / (np.dot(noise, noise) + 1e-8)), 4
        )
    except Exception:
        return None


def compute_mel_distance(original, reconstructed, sr):
    try:
        kwargs = {"sr": sr, "n_mels": 128, "fmax": int(8000)}
        mel1 = librosa.feature.melspectrogram(y=original, **kwargs)
        mel2 = librosa.feature.melspectrogram(y=reconstructed, **kwargs)
        min_len = min(mel1.shape[1], mel2.shape[1])
        diff = librosa.power_to_db(mel1[:, :min_len]) - librosa.power_to_db(
            mel2[:, :min_len]
        )
        return round(float(np.sqrt(np.mean(diff**2))), 4)
    except Exception:
        return None


# ── CODEC CLASSICI ──────────────────────────────────────────────


def compress_opus(audio_path, bitrate_kbps):
    y, sr = load_audio(audio_path)
    duration = len(y) / sr

    with tempfile.TemporaryDirectory() as tmp:
        wav_in = os.path.join(tmp, "in.wav")
        opus_f = os.path.join(tmp, "out.opus")
        wav_out = os.path.join(tmp, "out.wav")
        sf.write(wav_in, y, sr)

        start = time.time()
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                wav_in,
                "-c:a",
                "libopus",
                "-b:a",
                f"{bitrate_kbps}k",
                opus_f,
            ],
            capture_output=True,
            check=True,
        )
        enc_ms = (time.time() - start) * 1000

        compressed_bits = os.path.getsize(opus_f) * 8
        actual_kbps = compressed_bits / duration / 1000

        subprocess.run(
            ["ffmpeg", "-y", "-i", opus_f, wav_out], capture_output=True, check=True
        )

        y_rec, _ = librosa.load(wav_out, sr=sr, mono=True)
        min_len = min(len(y), len(y_rec))
        y, y_rec = y[:min_len], y_rec[:min_len]

        return {
            "bitrate_kbps": actual_kbps,
            "snr_db": snr(y, y_rec),
            "segsnr_db": segmental_snr(y, y_rec, sr),
            "pesq": compute_pesq(y, y_rec, sr),
            "stoi": compute_stoi(y, y_rec, sr),
            "si_sdr": compute_si_sdr(y, y_rec),
            "mel_dist": compute_mel_distance(y, y_rec, sr),
            "enc_ms": round(enc_ms, 2),
            "duration_s": duration,
        }


def compress_aac(audio_path, bitrate_kbps):
    y, sr = load_audio(audio_path)
    duration = len(y) / sr

    with tempfile.TemporaryDirectory() as tmp:
        wav_in = os.path.join(tmp, "in.wav")
        aac_f = os.path.join(tmp, "out.m4a")
        wav_out = os.path.join(tmp, "out.wav")
        sf.write(wav_in, y, sr)

        start = time.time()
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                wav_in,
                "-c:a",
                "aac",
                "-b:a",
                f"{bitrate_kbps}k",
                aac_f,
            ],
            capture_output=True,
            check=True,
        )
        enc_ms = (time.time() - start) * 1000

        compressed_bits = os.path.getsize(aac_f) * 8
        actual_kbps = compressed_bits / duration / 1000

        subprocess.run(
            ["ffmpeg", "-y", "-i", aac_f, wav_out], capture_output=True, check=True
        )

        y_rec, _ = librosa.load(wav_out, sr=sr, mono=True)
        min_len = min(len(y), len(y_rec))
        y, y_rec = y[:min_len], y_rec[:min_len]

        return {
            "bitrate_kbps": actual_kbps,
            "snr_db": snr(y, y_rec),
            "segsnr_db": segmental_snr(y, y_rec, sr),
            "pesq": compute_pesq(y, y_rec, sr),
            "stoi": compute_stoi(y, y_rec, sr),
            "si_sdr": compute_si_sdr(y, y_rec),
            "mel_dist": compute_mel_distance(y, y_rec, sr),
            "enc_ms": round(enc_ms, 2),
            "duration_s": duration,
        }


# ── ENCODEC ─────────────────────────────────────────────────────


def compress_encodec(audio_path, bandwidth):
    from encodec.model import EncodecModel
    from encodec.utils import convert_audio

    y, sr = load_audio(audio_path, target_sr=24000)
    duration = len(y) / sr

    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(bandwidth)
    model.to(device)
    model.eval()

    wav = torch.from_numpy(y).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        torch.cuda.synchronize()
        start = time.time()
        encoded = model.encode(wav)
        decoded = model.decode(encoded)
        torch.cuda.synchronize()
        enc_ms = (time.time() - start) * 1000

    y_rec = decoded.squeeze().cpu().numpy()
    min_len = min(len(y), len(y_rec))
    y, y_rec = y[:min_len], y_rec[:min_len]

    actual_kbps = bandwidth * 1000 / 1000

    del model
    torch.cuda.empty_cache()

    return {
        "bitrate_kbps": actual_kbps,
        "snr_db": snr(y, y_rec),
        "segsnr_db": segmental_snr(y, y_rec, sr),
        "pesq": compute_pesq(y, y_rec, sr),
        "stoi": compute_stoi(y, y_rec, sr),
        "si_sdr": compute_si_sdr(y, y_rec),
        "mel_dist": compute_mel_distance(y, y_rec, sr),
        "enc_ms": round(enc_ms, 2),
        "duration_s": duration,
    }


# ── BENCHMARK ───────────────────────────────────────────────────


def run_audio_benchmark():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # separa file per tipo
    speech_files = sorted(Path(AUDIO_TEST_DIR).glob("6930*.wav"))
    music_files = [
        Path(AUDIO_TEST_DIR) / "nutcracker.wav",
        Path(AUDIO_TEST_DIR) / "trumpet.wav",
    ]
    all_files = speech_files + music_files

    print(f"File parlato: {len(speech_files)}")
    print(f"File musicali: {len(music_files)}")

    rows = []

    print("\n=== Opus ===")
    for bitrate in [6, 12, 24, 48, 96, 128]:
        for f in all_files:
            file_type = "speech" if f in speech_files else "music"
            m = compress_opus(str(f), bitrate)
            rows.append(
                {
                    "file": f.name,
                    "type": file_type,
                    "codec": "Opus",
                    "target_kbps": bitrate,
                    **m,
                }
            )
            pd.DataFrame(rows).to_csv(
                os.path.join(RESULTS_DIR, "audio_benchmark.csv"), index=False
            )
            print(
                f"  [{file_type}] {f.name} | {bitrate}kbps | "
                f"SNR={m['snr_db']:.1f}dB | PESQ={m['pesq']}"
            )

    print("\n=== AAC ===")
    for bitrate in [32, 48, 64, 96, 128]:
        for f in all_files:
            file_type = "speech" if f in speech_files else "music"
            m = compress_aac(str(f), bitrate)
            rows.append(
                {
                    "file": f.name,
                    "type": file_type,
                    "codec": "AAC",
                    "target_kbps": bitrate,
                    **m,
                }
            )
            pd.DataFrame(rows).to_csv(
                os.path.join(RESULTS_DIR, "audio_benchmark.csv"), index=False
            )
            print(
                f"  [{file_type}] {f.name} | {bitrate}kbps | "
                f"SNR={m['snr_db']:.1f}dB | PESQ={m['pesq']}"
            )

    print("\n=== EnCodec ===")
    for bw in [1.5, 3.0, 6.0, 12.0, 24.0]:
        for f in all_files:
            file_type = "speech" if f in speech_files else "music"
            m = compress_encodec(str(f), bw)
            rows.append(
                {
                    "file": f.name,
                    "type": file_type,
                    "codec": "EnCodec",
                    "target_kbps": bw,
                    **m,
                }
            )
            pd.DataFrame(rows).to_csv(
                os.path.join(RESULTS_DIR, "audio_benchmark.csv"), index=False
            )
            print(
                f"  [{file_type}] {f.name} | {bw}kbps | "
                f"SNR={m['snr_db']:.1f}dB | PESQ={m['pesq']}"
            )

    df = pd.DataFrame(rows)
    out = os.path.join(RESULTS_DIR, "audio_benchmark.csv")
    df.to_csv(out, index=False)

    print("\n--- Riepilogo parlato ---")
    speech_df = df[df["type"] == "speech"]
    print(
        speech_df.groupby(["codec", "target_kbps"])[
            ["bitrate_kbps", "snr_db", "pesq", "stoi", "si_sdr", "mel_dist"]
        ]
        .mean()
        .to_string()
    )

    print("\n--- Riepilogo musica ---")
    music_df = df[df["type"] == "music"]
    print(
        music_df.groupby(["codec", "target_kbps"])[
            ["bitrate_kbps", "snr_db", "pesq", "stoi", "si_sdr", "mel_dist"]
        ]
        .mean()
        .to_string()
    )


if __name__ == "__main__":
    run_audio_benchmark()
