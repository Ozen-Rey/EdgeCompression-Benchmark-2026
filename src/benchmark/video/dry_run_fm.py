"""
Dry-run DCVC-FM: test di encoding + decoding completo su una singola sequenza UVG.

Non misura energia. Serve solo a verificare:
  - I modelli si caricano correttamente
  - L'inferenza gira senza crash (grid_sample, tensori, DPB)
  - Il bitstream viene scritto e riletto correttamente
  - Il YUV ricostruito ha senso (dimensioni, range, no NaN)
  - I tempi frame-per-frame sono ragionevoli

Questo script sta in ~/tesi/src/benchmark/video/ per coerenza con gli altri script
di benchmark. Il repo DCVC-FM viene aggiunto a sys.path dinamicamente.

Uso:
  conda activate dcvc-eval
  cd ~/tesi/src/benchmark/video
  python dry_run_fm.py
"""

import os
import sys
import time
from pathlib import Path

# Path del repo DCVC-FM (aggiunto a sys.path per gli import from src.models.*)
DCVC_FM_REPO = (
    Path.home() / "tesi" / "external_codecs" / "dcvc" / "DCVC-family" / "DCVC-FM"
)
sys.path.insert(0, str(DCVC_FM_REPO))

import torch
import torch.nn.functional as F
import numpy as np

# Import dal repo DCVC-FM
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
from src.transforms.functional import ycbcr420_to_444, ycbcr444_to_420


# ================== CONFIG ==================

UVG_DIR = Path.home() / "tesi" / "datasets" / "uvg"
BITSTREAMS_DIR = Path.home() / "tesi" / "bitstreams_video"
BITSTREAMS_DIR.mkdir(parents=True, exist_ok=True)

CKPT_I = DCVC_FM_REPO / "checkpoints" / "cvpr2024_image.pth.tar"
CKPT_P = DCVC_FM_REPO / "checkpoints" / "cvpr2024_video.pth.tar"

# Dry-run: solo Beauty, solo primi 30 frame, 1 operating point (q_index medio)
SEQ_NAME = "Beauty"
SEQ_PATH = UVG_DIR / "Beauty_1920x1080_120fps_420_8bit_YUV.yuv"
WIDTH, HEIGHT = 1920, 1080
FPS = 120
N_FRAMES = 30  # solo prime 30 frame per dry-run rapido
Q_INDEX_I = 32  # DCVC-FM supporta q_index 0-63, scelgo medio
Q_INDEX_P = 32
RATE_GOP_SIZE = 8
RESET_INTERVAL = 64  # come raccomandato nel README

# Argomenti modello (coerenti con il loro init_func)
EC_THREAD = False  # single-thread entropy coder (più stabile per misure)
STREAM_PART_I = 1  # 1 partition per I-frame stream
STREAM_PART_P = 1  # 1 partition per P-frame stream


# ================== UTILITÀ (ridotte dal test_helper) ==================


def np_image_to_tensor(img):
    # evita il torch.FloatTensor deprecato: uso dtype esplicito
    return torch.from_numpy(img).to(dtype=torch.float32).unsqueeze(0)


def load_models():
    """
    Carica i modelli DCVC-FM replicando fedelmente init_func dal repo Microsoft:
      - Istanzia con ec_thread/stream_part/inplace
      - Carica state_dict via get_state_dict() (gestisce format diversi)
      - Sposta su CUDA, eval mode
      - Chiama update(force=True) per inizializzare entropy_coder
    """
    print("=== Loading models ===")
    t0 = time.time()

    # Setup deterministico (come init_func loro)
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(0)

    # I-model
    i_state_dict = get_state_dict(str(CKPT_I))
    i_model = DMCI(ec_thread=EC_THREAD, stream_part=STREAM_PART_I, inplace=True)
    i_model.load_state_dict(i_state_dict)
    i_model = i_model.cuda().eval()

    # P-model
    p_state_dict = get_state_dict(str(CKPT_P))
    p_model = DMC(ec_thread=EC_THREAD, stream_part=STREAM_PART_P, inplace=True)
    p_model.load_state_dict(p_state_dict)
    p_model = p_model.cuda().eval()

    # CRITICO: init entropy coder (serve per encode/decode reali)
    i_model.update(force=True)
    p_model.update(force=True)

    print(
        f"  loaded in {time.time()-t0:.1f}s, VRAM={torch.cuda.memory_allocated()/1024**2:.1f} MB"
    )
    return i_model, p_model


# ================== ENCODE ==================


def encode_sequence(
    i_model,
    p_model,
    seq_path,
    width,
    height,
    n_frames,
    q_index_i,
    q_index_p,
    bitstream_path,
):
    """
    Encoding di una sequenza completa.
    Ritorna dict con tempi, bits per frame, numero di I e P frame.
    """
    device = next(i_model.parameters()).device
    src_reader = YUVReader(str(seq_path), width, height)
    padding_l, padding_r, padding_t, padding_b = get_padding_size(height, width, 16)

    output_file = bitstream_path.open("wb")
    sps_helper = SPSHelper()
    outstanding_sps_bytes = 0

    # Index map per P-frame rate_gop_size
    index_map = [0, 1, 0, 2, 0, 2, 0, 2]

    # Inizializzo dpb a dict vuoto (sarà popolato al primo I-frame)
    dpb: dict = {}
    bits_per_frame = []
    times_per_frame = []
    n_i_frames = 0
    n_p_frames = 0

    print(f"\n=== Encode: {seq_path.name} ({n_frames} frames) ===")
    t_total_start = time.time()

    with torch.no_grad():
        for frame_idx in range(n_frames):
            t_frame_start = time.time()

            # Read YUV frame
            y, uv = src_reader.read_one_frame(dst_format="420")
            yuv = ycbcr420_to_444(y, uv)
            x = np_image_to_tensor(yuv).to(device)

            # Pad to multiple of 16
            x_padded = F.pad(
                x, (padding_l, padding_r, padding_t, padding_b), mode="replicate"
            )

            if frame_idx == 0:
                # I-frame (primo frame sempre intra in LDP puro)
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
                bits_per_frame.append(result["bit"] + outstanding_sps_bytes * 8)
                outstanding_sps_bytes = 0
                n_i_frames += 1
                frame_type = "I"
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
                bits_per_frame.append(result["bit"] + outstanding_sps_bytes * 8)
                outstanding_sps_bytes = 0
                n_p_frames += 1
                frame_type = "P"

            torch.cuda.synchronize()
            t_frame_end = time.time()
            times_per_frame.append(t_frame_end - t_frame_start)

            if frame_idx < 5 or frame_idx % 10 == 0:
                print(
                    f"  frame {frame_idx:3d} [{frame_type}] "
                    f"{times_per_frame[-1]*1000:.0f} ms, "
                    f"{bits_per_frame[-1]} bits"
                )

    output_file.close()
    src_reader.close()
    t_total = time.time() - t_total_start

    total_bits = sum(bits_per_frame)
    fps_enc = n_frames / t_total
    bitrate_mbps = total_bits / (n_frames / FPS) / 1e6

    print(f"  ENCODE done: {t_total:.2f}s total, {fps_enc:.2f} fps")
    print(
        f"  Total bits: {total_bits}, bitstream: {bitstream_path.stat().st_size} bytes"
    )
    print(f"  Effective bitrate: {bitrate_mbps:.2f} Mbps")

    return {
        "time_total": t_total,
        "fps": fps_enc,
        "n_i": n_i_frames,
        "n_p": n_p_frames,
        "total_bits": total_bits,
        "bitstream_size": bitstream_path.stat().st_size,
        "bitrate_mbps": bitrate_mbps,
    }


# ================== DECODE ==================


def decode_sequence(i_model, p_model, bitstream_path, width, height, n_frames):
    """
    Decode di una sequenza dal bitstream prodotto.
    Ritorna dict con tempi e lista dei YUV ricostruiti.
    """
    device = next(i_model.parameters()).device
    padding_l, padding_r, padding_t, padding_b = get_padding_size(height, width, 16)

    input_file = bitstream_path.open("rb")
    sps_helper = SPSHelper()
    pending_frame_spss: list = []
    dpb: dict = {}

    times_per_frame = []
    decoded_yuvs = []  # Lista di (y, uv) numpy arrays

    print(f"\n=== Decode: {bitstream_path.name} ===")
    t_total_start = time.time()

    with torch.no_grad():
        decoded_frame_number = 0
        while decoded_frame_number < n_frames:
            t_frame_start = time.time()
            new_stream = False
            # Inizializza variabili che potrebbero essere non assegnate nei rami
            header = None
            sps_id = None
            recon_frame = None
            frame_type = "?"

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

            # Se header è None (ramo di pending_frame_spss), usiamo l'ultimo header letto
            # via pending. Ma in quel caso NAL_Ps header era già stato letto.
            # Se sei qui con header=None, siamo in un caso di NAL_Ps continuation.
            # Il nal_type in quel caso è implicitamente NAL_Ps → trattiamo come P.
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
                frame_type = "I"
            else:
                # P-frame (sia NAL_P sia NAL_Ps sia continuation)
                if sps["fa_idx"] == 3:
                    dpb["ref_feature"] = None
                decoded = p_model.decompress(bit_stream, dpb, sps)
                dpb = decoded["dpb"]
                recon_frame = dpb["ref_frame"]
                frame_type = "P"

            # Rimuovi padding
            recon_frame = recon_frame.clamp_(0, 1)
            x_hat = F.pad(recon_frame, (-padding_l, -padding_r, -padding_t, -padding_b))

            # Converte YUV444 → YUV420 (per il quality calc sarà identico al source)
            yuv_rec = x_hat.squeeze(0).cpu().numpy()
            y_rec, uv_rec = ycbcr444_to_420(yuv_rec)
            decoded_yuvs.append((y_rec, uv_rec))

            torch.cuda.synchronize()
            t_frame_end = time.time()
            times_per_frame.append(t_frame_end - t_frame_start)

            if decoded_frame_number < 5 or decoded_frame_number % 10 == 0:
                print(
                    f"  frame {decoded_frame_number:3d} [{frame_type}] "
                    f"{times_per_frame[-1]*1000:.0f} ms"
                )

            decoded_frame_number += 1

    input_file.close()
    t_total = time.time() - t_total_start
    fps_dec = n_frames / t_total

    print(f"  DECODE done: {t_total:.2f}s total, {fps_dec:.2f} fps")

    return {
        "time_total": t_total,
        "fps": fps_dec,
        "decoded_yuvs": decoded_yuvs,
    }


# ================== MAIN ==================


def main():
    print("=" * 70)
    print(f"DCVC-FM DRY RUN")
    print(f"Sequence: {SEQ_NAME} @ {WIDTH}x{HEIGHT}, first {N_FRAMES} frames")
    print(f"Q-index: I={Q_INDEX_I}, P={Q_INDEX_P}")
    print(f"Reset interval: {RESET_INTERVAL}")
    print("=" * 70)

    if not SEQ_PATH.exists():
        print(f"[FATAL] Sequence not found: {SEQ_PATH}")
        sys.exit(1)
    if not CKPT_I.exists():
        print(f"[FATAL] Checkpoint not found: {CKPT_I}")
        sys.exit(1)
    if not CKPT_P.exists():
        print(f"[FATAL] Checkpoint not found: {CKPT_P}")
        sys.exit(1)

    bitstream_path = BITSTREAMS_DIR / f"__dryrun_dcvc_fm_{SEQ_NAME}_q{Q_INDEX_P}.bin"

    # 1. Load models
    i_model, p_model = load_models()

    # 2. Encode
    enc_result = encode_sequence(
        i_model,
        p_model,
        SEQ_PATH,
        WIDTH,
        HEIGHT,
        N_FRAMES,
        Q_INDEX_I,
        Q_INDEX_P,
        bitstream_path,
    )

    # 3. Decode
    dec_result = decode_sequence(
        i_model, p_model, bitstream_path, WIDTH, HEIGHT, N_FRAMES
    )

    # 4. Sanity check sul YUV ricostruito
    print(f"\n=== Sanity check decoded YUV ===")
    y0, uv0 = dec_result["decoded_yuvs"][0]
    print(f"  First frame Y shape: {y0.shape}, dtype: {y0.dtype}")
    print(f"  Y range: [{y0.min():.3f}, {y0.max():.3f}]")
    print(f"  UV shape: {uv0.shape}, range [{uv0.min():.3f}, {uv0.max():.3f}]")
    print(f"  Any NaN? Y: {np.isnan(y0).any()}, UV: {np.isnan(uv0).any()}")

    print(f"\n=== Summary ===")
    print(f"  Encode: {enc_result['time_total']:.2f}s ({enc_result['fps']:.2f} fps)")
    print(f"  Decode: {dec_result['time_total']:.2f}s ({dec_result['fps']:.2f} fps)")
    print(f"  Bitrate: {enc_result['bitrate_mbps']:.2f} Mbps")
    print(f"  Bitstream: {bitstream_path}")


if __name__ == "__main__":
    main()
