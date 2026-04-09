import sys
import os
import time
import csv
import gc
import warnings
import subprocess
import glob
import numpy as np
import torch
import librosa
import soundfile as sf
from pathlib import Path

warnings.filterwarnings("ignore")

# ================== CONFIGURAZIONE ==================
DATASETS = {
    "librispeech": os.path.expanduser("~/tesi/datasets_audio/librispeech_sample"),
    "esc50": os.path.expanduser("~/tesi/datasets_audio/esc50_sample"),
    "musdb": os.path.expanduser("~/tesi/datasets_audio/musdb_10s"),
}
MAX_PER_DS = 50
MAX_DURATION = 10.0
TARGET_SR = 24000
IDLE_DURATION = 3.0  # Tempo per stabilizzare la misura di idle

OUTPUT_CSV = os.path.expanduser("~/tesi/results/audio/audio_energy_rigorous_batch.csv")
DEVICE = "cuda"
device = torch.device(DEVICE)


# ================== SENSORI HARDWARE ==================
def read_rapl_uj():
    with open("/sys/class/powercap/intel-rapl:0/energy_uj") as f:
        return int(f.read().strip())


RAPL_MAX = int(open("/sys/class/powercap/intel-rapl:0/max_energy_range_uj").read())

from zeus.device.gpu import get_gpus

gpu = get_gpus().gpus[0]


# ================== LOGICA DI MISURA ==================
def measure_idle():
    """Misura la potenza di idle (W) corrente per CPU e GPU."""
    torch.cuda.synchronize()
    time.sleep(1.0)
    r0, g0, t0 = read_rapl_uj(), gpu.getTotalEnergyConsumption(), time.perf_counter()
    time.sleep(IDLE_DURATION)
    r1, g1, t1 = read_rapl_uj(), gpu.getTotalEnergyConsumption(), time.perf_counter()
    dt = t1 - t0
    dr = r1 - r0
    if dr < 0:
        dr += RAPL_MAX
    return (dr / 1e6) / dt, ((g1 - g0) / 1000) / dt


def measure_start():
    torch.cuda.synchronize()
    return read_rapl_uj(), gpu.getTotalEnergyConsumption(), time.perf_counter()


def measure_end(r0, g0, t0, cpu_idle_w, gpu_idle_w, is_neural):
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    r1, g1 = read_rapl_uj(), gpu.getTotalEnergyConsumption()
    dt = t1 - t0
    dr = r1 - r0
    if dr < 0:
        dr += RAPL_MAX

    cpu_gross = dr / 1e6
    gpu_gross = (g1 - g0) / 1000

    # Sottrazione dell'idle calcolato sul tempo di esecuzione effettivo
    cpu_net = max(cpu_gross - cpu_idle_w * dt, 0)
    gpu_net = max(gpu_gross - gpu_idle_w * dt, 0) if is_neural else 0.0

    return dt, cpu_net, gpu_net


# ================== FUNZIONI CODEC (LOOP BATCH) ==================
def run_opus_batch(audio_list, bitrate_kbps):
    tmp_in = "/dev/shm/opus_in.wav"
    tmp_opus = "/dev/shm/opus_out.opus"
    tmp_out = "/dev/shm/opus_dec.wav"
    for a in audio_list:
        sf.write(tmp_in, a, TARGET_SR)
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                tmp_in,
                "-c:a",
                "libopus",
                "-b:a",
                f"{bitrate_kbps}k",
                "-ar",
                "24000",
                "-ac",
                "1",
                tmp_opus,
            ],
            capture_output=True,
        )
        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_opus, "-ar", "24000", "-ac", "1", tmp_out],
            capture_output=True,
        )


def run_neural_batch(model, audio_list, codec_type):
    with torch.no_grad():
        for a in audio_list:
            x = torch.from_numpy(a).float().unsqueeze(0).unsqueeze(0).to(device)
            if codec_type == "encodec":
                frames = model.encode(x)
                model.decode(frames)
            elif codec_type == "dac":
                z, _, _, _, _ = model.encode(x)
                model.decode(z)
            elif codec_type == "snac":
                codes = model.encode(x)
                model.decode(codes)
            elif codec_type == "wavtokenizer":
                # WavTokenizer vuole [1, T] e non [1, 1, T]
                x_wt = torch.from_numpy(a).float().unsqueeze(0).to(device)
                bid = torch.tensor([0]).to(device)
                feat, _ = model.encode_infer(x_wt, bandwidth_id=bid)
                model.decode(feat, bandwidth_id=bid)


# ================== MAIN ==================
def main():
    print("=" * 70)
    print("BENCHMARK ENERGETICO AUDIO RIGOROSO (BATCH MODE)")
    print("=" * 70)

    # 1. Caricamento in RAM
    print("\nCaricamento file audio in RAM...")
    all_audio = []
    total_sec = 0
    for ds_name, ds_path in DATASETS.items():
        files = []
        for ext in ["*.flac", "*.wav"]:
            files.extend(glob.glob(os.path.join(ds_path, "**", ext), recursive=True))
        files = sorted(files)[:MAX_PER_DS]
        for f in files:
            a, _ = librosa.load(f, sr=TARGET_SR, mono=True, duration=MAX_DURATION)
            if len(a) >= TARGET_SR:
                all_audio.append(a)
                total_sec += len(a) / TARGET_SR

    print(f"Pronti {len(all_audio)} file ({total_sec:.1f} secondi totali)")

    summary = []

    def execute_benchmark(name, param, batch_fn, is_neural):
        # Warmup (1 file)
        batch_fn([all_audio[0]])

        # Misura Idle specifica per questo stato termico
        c_idle, g_idle = measure_idle()
        print(f"  {name} {param} (Idle: CPU={c_idle:.1f}W, GPU={g_idle:.1f}W)")

        # Esecuzione Batch
        gc.collect()
        gc.disable()
        r0, g0, t0 = measure_start()
        batch_fn(all_audio)
        dt, cpu_net, gpu_net = measure_end(r0, g0, t0, c_idle, g_idle, is_neural)
        gc.enable()

        tot_net = cpu_net + gpu_net
        jps = tot_net / total_sec
        print(f"    Completato in {dt:.1f}s. E_net={tot_net:.2f}J -> {jps:.4f} J/s")
        summary.append(
            {
                "codec": name,
                "param": param,
                "j_per_s": round(jps, 4),
                "cpu_j": round(cpu_net, 3),
                "gpu_j": round(gpu_net, 3),
                "time_s": round(dt, 1),
            }
        )

    # === TEST LOOP ===
    # Opus
    for k in [12, 24, 48]:
        execute_benchmark(
            "Opus", str(k), lambda al, kb=k: run_opus_batch(al, kb), False
        )

    # EnCodec
    from encodec.model import EncodecModel

    for bw in [1.5, 3.0, 6.0]:
        m = EncodecModel.encodec_model_24khz().to(device).eval()
        m.set_target_bandwidth(bw)
        execute_benchmark(
            "EnCodec",
            str(bw),
            lambda al, mod=m: run_neural_batch(mod, al, "encodec"),
            True,
        )
        del m
        torch.cuda.empty_cache()

    # DAC
    import dac

    m = dac.DAC.load(str(dac.utils.download(model_type="24khz"))).to(device).eval()
    execute_benchmark(
        "DAC", "8.0", lambda al, mod=m: run_neural_batch(mod, al, "dac"), True
    )
    del m
    torch.cuda.empty_cache()

    # SNAC
    from snac import SNAC

    m = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(device).eval()
    execute_benchmark(
        "SNAC", "0.8", lambda al, mod=m: run_neural_batch(mod, al, "snac"), True
    )
    del m
    torch.cuda.empty_cache()

    # WavTokenizer
    sys.path.insert(0, os.path.expanduser("~/tesi/external_codecs/WavTokenizer"))
    from decoder.pretrained import WavTokenizer as WT  # type: ignore

    cfg = "/home/user/.cache/huggingface/hub/models--novateur--WavTokenizer-medium-music-audio-75token/snapshots/1bcbe86db88fc1f1f86c7ea192e791b3ede17da2/wavtokenizer_mediumdata_music_audio_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
    ckp = "/home/user/.cache/huggingface/hub/models--novateur--WavTokenizer-medium-music-audio-75token/snapshots/1bcbe86db88fc1f1f86c7ea192e791b3ede17da2/wavtokenizer_medium_music_audio_320_24k_v2.ckpt"
    m = WT.from_pretrained0802(cfg, ckp).to(device).eval()
    execute_benchmark(
        "WavTokenizer",
        "0.9",
        lambda al, mod=m: run_neural_batch(mod, al, "wavtokenizer"),
        True,
    )

    # Salvataggio
    with open(OUTPUT_CSV, "w") as f:
        w = csv.DictWriter(f, fieldnames=summary[0].keys())
        w.writeheader()
        w.writerows(summary)
    print(f"\nRisultati salvati in: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
