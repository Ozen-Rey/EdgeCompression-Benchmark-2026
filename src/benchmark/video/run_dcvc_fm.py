"""
Benchmark energetico DCVC-FM.

Protocollo coerente con i codec classici (run_x265.py, run_svtav1.py, ...):
  Per ogni (seq, q_index):
    1. Idle cal #1
    2. Encode 600 frame (misurato) → scrive bitstream persistente
    3. Idle cal #2
    4. Decode dal bitstream (misurato) → scrive YUV decoded in /dev/shm
    5. YUV decoded cancellato DOPO misura (non tocca il timer)

Differenze chiave rispetto ai classici:
  - is_neural=True → la funzione measure_phase contabilizza anche la GPU
  - Profilo solo LDP (neurali non supportano RA per design)
  - "param" è "q=<q_index>" (es "q=21"), non "crf=<n>"
  - Modelli caricati una sola volta all'inizio (riutilizzati tra tutte le config)

SMOKE_TEST=True limita a Beauty+q=21 (un solo punto): serve per verificare
che la strumentazione non rompa prima del full run (~3h).

Uso:
  conda activate dcvc-eval
  cd ~/tesi/src/benchmark/video
  # smoke test (5 min)
  python run_dcvc_fm.py
  # full run (~3h), in background con log live
  SMOKE_TEST=0 nohup python -u run_dcvc_fm.py > dcvc_fm_full_run.log 2>&1 &
"""

import os
import sys
import time
from pathlib import Path

# Repo DCVC-FM a sys.path (prima di qualsiasi import src.*)
DCVC_FM_REPO = (
    Path.home() / "tesi" / "external_codecs" / "dcvc" / "DCVC-family" / "DCVC-FM"
)
sys.path.insert(0, str(DCVC_FM_REPO))

import torch
import torch.nn.functional as F
import numpy as np

# Common benchmark framework
from common import (
    UVG_DIR,
    RESULTS_DIR,
    BITSTREAMS_DIR,
    DECODED_DIR,
    UVG_SEQUENCES,
    measure_idle,
    measure_phase,
    init_csv,
    append_row,
    setup_dirs,
    bitstream_path,
    decoded_path,
    compute_actual_mbps,
)

# DCVC-FM internals
from src.models.video_model import DMC
from src.models.image_model import DMCI
from src.utils.stream_helper import (
    get_padding_size,
    get_state_dict,
    SPSHelper,
    NalType,
    write_sps,
    read_header,
    read_sps_remaining,
    read_ip_remaining,
)
from src.utils.video_reader import YUVReader
from src.utils.video_writer import YUVWriter
from src.transforms.functional import ycbcr420_to_444, ycbcr444_to_420


# ================== CONFIG ==================

CODEC = "dcvc_fm"
CSV_PATH = RESULTS_DIR / "dcvc_fm_energy.csv"

CKPT_I = DCVC_FM_REPO / "checkpoints" / "cvpr2024_image.pth.tar"
CKPT_P = DCVC_FM_REPO / "checkpoints" / "cvpr2024_video.pth.tar"

# Operating points: rate_num=4, q_index linspace(0, qp_num-1) come test_video.py
Q_INDEXES = [0, 21, 42, 63]  # verificato con DMC.get_qp_num() = 64
RATE_GOP_SIZE = 8
RESET_INTERVAL = 64  # come da README Microsoft

# Model instantiation params (coerenti con init_func di Microsoft)
EC_THREAD = False
STREAM_PART_I = 1
STREAM_PART_P = 1

# Smoke test: solo Beauty a q_index=21, limitato a 600 frame reali
SMOKE_TEST = os.environ.get("SMOKE_TEST", "1") == "1"


# ================== UTILITY ==================


def np_image_to_tensor(img):
    return torch.from_numpy(img).to(dtype=torch.float32).unsqueeze(0)


def load_models():
    """Carica DCVC-FM I-model e P-model, ready for encode/decode."""
    print(f"[load_models] loading DCVC-FM checkpoints...")
    t0 = time.time()

    torch.backends.cudnn.benchmark = False
    torch.manual_seed(0)

    i_state_dict = get_state_dict(str(CKPT_I))
    i_model = DMCI(ec_thread=EC_THREAD, stream_part=STREAM_PART_I, inplace=True)
    i_model.load_state_dict(i_state_dict)
    i_model = i_model.cuda().eval()

    p_state_dict = get_state_dict(str(CKPT_P))
    p_model = DMC(ec_thread=EC_THREAD, stream_part=STREAM_PART_P, inplace=True)
    p_model.load_state_dict(p_state_dict)
    p_model = p_model.cuda().eval()

    i_model.update(force=True)
    p_model.update(force=True)

    print(
        f"[load_models] ready in {time.time()-t0:.1f}s, "
        f"VRAM={torch.cuda.memory_allocated()/1024**2:.1f} MB"
    )
    return i_model, p_model


def load_yuv_frames_to_memory(seq_path, width, height, n_frames):
    """
    Carica le n_frames della sequenza in RAM come lista di tensori cuda.
    Questo evita I/O dal disco nel loop misurato.
    Ritorna: list di n_frames tensori su cuda, già padded e formato YUV444.
    Peso: 1920x1080x3x4byte x 600 = ~14 GB. Troppo per tutto.
    → Torniamo a leggere frame-by-frame dal disco nel loop.
      L'I/O è cachato dal filesystem dopo il primo pass, quindi costa poco
      e il ruolo è identico per tutti i codec classici + neurali.
    Decisione: NON pre-carichiamo. Leggiamo frame-per-frame nel loop come
    fanno anche i classici via ffmpeg pipe. Coerenza metodologica.
    """
    raise NotImplementedError("non usato: leggiamo on-the-fly nel loop come i classici")


# ================== ENCODE ==================


def encode_sequence(
    i_model, p_model, seq_path, width, height, n_frames, q_index_i, q_index_p, bs_path
):
    """
    Encode completo di una sequenza. Coerente con run_one_point_with_stream
    del test_helper.py Microsoft, adattato a LDP puro + intra solo al primo frame.

    Ritorna:
      dict con total_bits, actual_mbps (al frame rate reale della sequenza),
      n_i_frames, n_p_frames (sanity).
    """
    device = next(i_model.parameters()).device
    src_reader = YUVReader(str(seq_path), width, height)
    padding_l, padding_r, padding_t, padding_b = get_padding_size(height, width, 16)

    output_file = bs_path.open("wb")
    sps_helper = SPSHelper()
    outstanding_sps_bytes = 0
    index_map = [0, 1, 0, 2, 0, 2, 0, 2]

    dpb: dict = {}
    total_bits = 0
    n_i = 0
    n_p = 0

    with torch.no_grad():
        for frame_idx in range(n_frames):
            y, uv = src_reader.read_one_frame(dst_format="420")
            yuv = ycbcr420_to_444(y, uv)
            x = np_image_to_tensor(yuv).to(device)
            x_padded = F.pad(
                x, (padding_l, padding_r, padding_t, padding_b), mode="replicate"
            )

            if frame_idx == 0:
                # I-frame (unico, force_intra_period=-1)
                sps = {
                    "sps_id": -1,
                    "height": height,
                    "width": width,
                    "qp": q_index_i,
                    "fa_idx": 0,
                }
                sps_id, sps_new = sps_helper.get_sps_id(sps)
                sps["sps_id"] = sps_id
                if sps_new:
                    outstanding_sps_bytes += write_sps(output_file, sps)
                result = i_model.encode(x_padded, q_index_i, sps_id, output_file)
                dpb = {
                    "ref_frame": result["x_hat"],
                    "ref_feature": None,
                    "ref_mv_feature": None,
                    "ref_y": None,
                    "ref_mv_y": None,
                }
                total_bits += result["bit"] + outstanding_sps_bytes * 8
                outstanding_sps_bytes = 0
                n_i += 1
            else:
                # P-frame
                fa_idx = index_map[frame_idx % RATE_GOP_SIZE]
                if RESET_INTERVAL > 0 and frame_idx % RESET_INTERVAL == 1:
                    dpb["ref_feature"] = None
                    fa_idx = 3
                sps = {
                    "sps_id": -1,
                    "height": height,
                    "width": width,
                    "qp": q_index_p,
                    "fa_idx": fa_idx,
                }
                sps_id, sps_new = sps_helper.get_sps_id(sps)
                sps["sps_id"] = sps_id
                if sps_new:
                    outstanding_sps_bytes += write_sps(output_file, sps)
                result = p_model.encode(
                    x_padded, dpb, q_index_p, fa_idx, sps_id, output_file
                )
                dpb = result["dpb"]
                total_bits += result["bit"] + outstanding_sps_bytes * 8
                outstanding_sps_bytes = 0
                n_p += 1

    output_file.close()
    src_reader.close()
    torch.cuda.synchronize()

    return {"total_bits": total_bits, "n_i": n_i, "n_p": n_p}


# ================== DECODE ==================


def decode_sequence(i_model, p_model, bs_path, width, height, n_frames, dec_yuv_path):
    """
    Decode completo di una sequenza dal bitstream, scrivendo YUV su dec_yuv_path.
    Il YUV decodificato è persistente al termine per permettere il quality calc.
    """
    device = next(i_model.parameters()).device
    padding_l, padding_r, padding_t, padding_b = get_padding_size(height, width, 16)

    input_file = bs_path.open("rb")
    sps_helper = SPSHelper()
    pending_frame_spss: list = []
    dpb: dict = {}

    recon_writer = YUVWriter(str(dec_yuv_path), width, height)

    with torch.no_grad():
        decoded_frame_number = 0
        while decoded_frame_number < n_frames:
            new_stream = False
            header = None
            sps_id = None

            if len(pending_frame_spss) == 0:
                header = read_header(input_file)
                if header["nal_type"] == NalType.NAL_SPS:
                    sps = read_sps_remaining(input_file, header["sps_id"])
                    sps_helper.add_sps_by_id(sps)
                    continue
                if header["nal_type"] == NalType.NAL_Ps:
                    pending_frame_spss = header["sps_ids"][1:]
                    sps_id = header["sps_ids"][0]
                else:
                    sps_id = header["sps_id"]
                new_stream = True
            else:
                sps_id = pending_frame_spss[0]
                pending_frame_spss.pop(0)

            sps = sps_helper.get_sps_by_id(sps_id)
            bit_stream = read_ip_remaining(input_file) if new_stream else None

            if header is not None and header["nal_type"] == NalType.NAL_I:
                decoded = i_model.decompress(bit_stream, sps)
                dpb = {
                    "ref_frame": decoded["x_hat"],
                    "ref_feature": None,
                    "ref_mv_feature": None,
                    "ref_y": None,
                    "ref_mv_y": None,
                }
                recon_frame = decoded["x_hat"]
            else:
                if sps["fa_idx"] == 3:
                    dpb["ref_feature"] = None
                decoded = p_model.decompress(bit_stream, dpb, sps)
                dpb = decoded["dpb"]
                recon_frame = dpb["ref_frame"]

            recon_frame = recon_frame.clamp_(0, 1)
            x_hat = F.pad(recon_frame, (-padding_l, -padding_r, -padding_t, -padding_b))
            yuv_rec = x_hat.squeeze(0).cpu().numpy()
            y_rec, uv_rec = ycbcr444_to_420(yuv_rec)
            recon_writer.write_one_frame(y=y_rec, uv=uv_rec, src_format="420")

            decoded_frame_number += 1

    input_file.close()
    recon_writer.close()
    torch.cuda.synchronize()


# ================== MAIN ==================


def process_config(i_model, p_model, seq_entry, q_index):
    """
    Processa un singolo punto operativo (seq, q_index) LDP.
    Scrive due righe CSV: encode + decode.
    """
    seq_name, fname, width, height, fps, n_frames = seq_entry
    seq_path = UVG_DIR / fname

    if not seq_path.exists():
        print(f"[SKIP] {seq_path} not found")
        return

    profile = "LDP"
    param = f"q={q_index}"
    bs_path = bitstream_path(CODEC, seq_name, profile, f"q{q_index}", "bin")
    dec_yuv_path = decoded_path(CODEC, seq_name, profile, f"q{q_index}")

    print(f"\n[{seq_name} {profile} {param}] {n_frames} frames @ {width}x{height}")

    # --- IDLE CAL #1 ---
    cpu_idle_1, gpu_idle_1 = measure_idle()
    print(f"  idle: CPU={cpu_idle_1:.1f}W, GPU={gpu_idle_1:.1f}W")

    # --- ENCODE (measured) ---
    enc_info: dict = {}

    def do_encode():
        nonlocal enc_info
        enc_info = encode_sequence(
            i_model,
            p_model,
            seq_path,
            width,
            height,
            n_frames,
            q_index,
            q_index,
            bs_path,
        )

    enc_result = measure_phase(do_encode, cpu_idle_1, gpu_idle_1, is_neural=True)
    actual_mbps = compute_actual_mbps(bs_path, n_frames, fps)
    print(
        f"  ENCODE: {enc_result['time_s']:.2f}s, "
        f"CPU={enc_result['cpu_net_j']:.1f}J GPU={enc_result['gpu_net_j']:.1f}J, "
        f"{actual_mbps:.2f} Mbps, I={enc_info['n_i']} P={enc_info['n_p']}"
    )

    append_row(
        CSV_PATH,
        {
            "codec": CODEC,
            "seq": seq_name,
            "profile": profile,
            "param": param,
            "phase": "encode",
            "n_frames": n_frames,
            "time_s": round(enc_result["time_s"], 3),
            "energy_cpu_j": round(enc_result["cpu_net_j"], 3),
            "energy_gpu_j": round(enc_result["gpu_net_j"], 3),
            "energy_total_j": round(enc_result["total_net_j"], 3),
            "idle_cpu_w": round(cpu_idle_1, 2),
            "idle_gpu_w": round(gpu_idle_1, 2),
            "actual_mbps": round(actual_mbps, 3),
        },
    )

    # --- IDLE CAL #2 ---
    cpu_idle_2, gpu_idle_2 = measure_idle()
    print(f"  idle #2: CPU={cpu_idle_2:.1f}W, GPU={gpu_idle_2:.1f}W")

    # --- DECODE (measured) ---
    def do_decode():
        decode_sequence(
            i_model, p_model, bs_path, width, height, n_frames, dec_yuv_path
        )

    dec_result = measure_phase(do_decode, cpu_idle_2, gpu_idle_2, is_neural=True)
    print(
        f"  DECODE: {dec_result['time_s']:.2f}s, "
        f"CPU={dec_result['cpu_net_j']:.1f}J GPU={dec_result['gpu_net_j']:.1f}J"
    )

    append_row(
        CSV_PATH,
        {
            "codec": CODEC,
            "seq": seq_name,
            "profile": profile,
            "param": param,
            "phase": "decode",
            "n_frames": n_frames,
            "time_s": round(dec_result["time_s"], 3),
            "energy_cpu_j": round(dec_result["cpu_net_j"], 3),
            "energy_gpu_j": round(dec_result["gpu_net_j"], 3),
            "energy_total_j": round(dec_result["total_net_j"], 3),
            "idle_cpu_w": round(cpu_idle_2, 2),
            "idle_gpu_w": round(gpu_idle_2, 2),
            "actual_mbps": round(actual_mbps, 3),
        },
    )

    # Cleanup decoded YUV (non misurato)
    try:
        os.remove(dec_yuv_path)
    except OSError:
        pass


def main():
    setup_dirs()
    init_csv(CSV_PATH)

    print("=" * 70)
    print(f"DCVC-FM ENERGY BENCHMARK")
    print(f"Output CSV:     {CSV_PATH}")
    print(f"Bitstream dir:  {BITSTREAMS_DIR}")
    print(f"Q-indexes:      {Q_INDEXES}")
    print(f"Reset interval: {RESET_INTERVAL}")
    print(f"SMOKE_TEST:     {SMOKE_TEST}")
    print("=" * 70)

    # Load models ONCE
    i_model, p_model = load_models()

    # Select config matrix
    if SMOKE_TEST:
        # Solo Beauty + q=21 (punto operativo medio)
        seqs_to_run = [s for s in UVG_SEQUENCES if s[0] == "Beauty"]
        q_indexes_to_run = [21]
    else:
        seqs_to_run = UVG_SEQUENCES
        q_indexes_to_run = Q_INDEXES

    n_configs = len(seqs_to_run) * len(q_indexes_to_run)
    t_run_start = time.time()
    done = 0

    for seq_entry in seqs_to_run:
        for q_index in q_indexes_to_run:
            done += 1
            print(f"\n========= [{done}/{n_configs}] =========")
            process_config(i_model, p_model, seq_entry, q_index)

    total_elapsed = time.time() - t_run_start
    print(
        f"\n\nDONE. {done}/{n_configs} configs in {total_elapsed:.1f}s "
        f"({total_elapsed/60:.1f} min)"
    )
    print(f"CSV: {CSV_PATH}")


if __name__ == "__main__":
    main()
