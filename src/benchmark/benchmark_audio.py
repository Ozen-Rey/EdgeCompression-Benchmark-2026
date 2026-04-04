import os
import sys
import time
import torch
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
import subprocess
import tempfile
from pesq import pesq as pesq_score
from pystoi import stoi
import torchaudio

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Variabile globale per evitare di ricaricare il modello EnCodec a ogni file
_ENCODEC_MODEL = None

def get_encodec_model():
    global _ENCODEC_MODEL
    if _ENCODEC_MODEL is None:
        from encodec.model import EncodecModel
        print("Caricamento modello EnCodec in VRAM...")
        _ENCODEC_MODEL = EncodecModel.encodec_model_24khz()
        _ENCODEC_MODEL.to(device)
        _ENCODEC_MODEL.eval()
    return _ENCODEC_MODEL

_DAC_MODEL = None

def get_dac_model():
    global _DAC_MODEL
    if _DAC_MODEL is None:
        import dac
        print("Caricamento modello DAC in VRAM...")
        # Usiamo il modello a 24kHz per fare un confronto equo con EnCodec
        model_path = dac.utils.download(model_type="24khz") 
        _DAC_MODEL = dac.DAC.load(model_path) #type: ignore
        _DAC_MODEL.to(device)
        _DAC_MODEL.eval()
    return _DAC_MODEL

# --- METRICHE ---
def load_audio(path: str, target_sr: int = 24000):
    """Carica l'audio forzando il path a stringa per il linter."""
    safe_path = str(path)
    y, sr = librosa.load(safe_path, sr=target_sr, mono=True)
    return y, target_sr

def compute_pesq(original, reconstructed, sr):
    try:
        target_sr = 16000
        orig_16k = librosa.resample(original, orig_sr=sr, target_sr=target_sr)
        rec_16k = librosa.resample(reconstructed, orig_sr=sr, target_sr=target_sr)
        min_len = min(len(orig_16k), len(rec_16k))
        # Il PESQ va da 1.0 (pessimo) a 4.5 (perfetto)
        return round(pesq_score(target_sr, orig_16k[:min_len], rec_16k[:min_len], "wb"), 4)
    except Exception:
        return 1.0 # Peggior score possibile se fallisce

def compute_mel_distance(original, reconstructed, sr):
    try:
        kwargs = {"sr": sr, "n_mels": 128, "fmax": int(8000)}
        mel1 = librosa.feature.melspectrogram(y=original, **kwargs)
        mel2 = librosa.feature.melspectrogram(y=reconstructed, **kwargs)
        min_len = min(mel1.shape[1], mel2.shape[1])
        diff = librosa.power_to_db(mel1[:, :min_len]) - librosa.power_to_db(mel2[:, :min_len])
        return round(float(np.sqrt(np.mean(diff**2))), 4)
    except Exception:
        return 999.0

# ── CODEC CLASSICI ──────────────────────────────────────────────

def compress_opus(audio_path, bitrate_kbps):
    y, sr = load_audio(audio_path)
    duration = len(y) / sr

    with tempfile.TemporaryDirectory() as tmp:
        wav_in = os.path.join(tmp, "in.wav")
        opus_f = os.path.join(tmp, "out.opus")
        wav_out = os.path.join(tmp, "out.wav")
        sf.write(wav_in, y, sr)

        start = time.perf_counter()
        subprocess.run(["ffmpeg", "-y", "-i", wav_in, "-c:a", "libopus", "-b:a", f"{bitrate_kbps}k", opus_f], 
                       capture_output=True, check=True)
        enc_ms = (time.perf_counter() - start) * 1000

        compressed_bits = os.path.getsize(opus_f) * 8
        actual_kbps = compressed_bits / duration / 1000

        subprocess.run(["ffmpeg", "-y", "-i", opus_f, wav_out], capture_output=True, check=True)
        y_rec, _ = librosa.load(wav_out, sr=sr, mono=True)
        
        min_len = min(len(y), len(y_rec))
        y, y_rec = y[:min_len], y_rec[:min_len]
        
        pesq_val = compute_pesq(y, y_rec, sr)
        # TRUCCO ORACLE: Creiamo una metrica dove PIU' BASSO = MEGLIO (come LPIPS)
        audio_dist = 4.5 - pesq_val 

        return {
            "kbps": round(actual_kbps, 2),
            "pesq": pesq_val,
            "mel_dist": compute_mel_distance(y, y_rec, sr),
            "audio_dist": round(audio_dist, 4), # Questa è la colonna che userà l'Oracolo!
            "enc_ms": round(enc_ms, 2)
        }

# ── ENCODEC (Neurale) ───────────────────────────────────────────

def compress_encodec(audio_path, bandwidth):
    model = get_encodec_model()
    y, sr = load_audio(audio_path, target_sr=24000)
    
    # Encodec lavora solo se la shape è giusta
    wav = torch.from_numpy(y).unsqueeze(0).unsqueeze(0).to(device)

    model.set_target_bandwidth(bandwidth)

    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        encoded = model.encode(wav)
        decoded = model.decode(encoded)
    torch.cuda.synchronize()
    enc_ms = (time.perf_counter() - start) * 1000

    y_rec = decoded.squeeze().cpu().numpy()
    min_len = min(len(y), len(y_rec))
    y, y_rec = y[:min_len], y_rec[:min_len]

    pesq_val = compute_pesq(y, y_rec, sr)
    audio_dist = 4.5 - pesq_val

    return {
        "kbps": bandwidth,
        "pesq": pesq_val,
        "mel_dist": compute_mel_distance(y, y_rec, sr),
        "audio_dist": round(audio_dist, 4),
        "enc_ms": round(enc_ms, 2)
    }

def compress_dac(audio_path, target_kbps=8.0):
    """Codec Neurale SOTA: DAC (Descript Audio Codec)"""
    model = get_dac_model()
    y, sr = load_audio(audio_path, target_sr=24000)
    
    # DAC richiede shape [Batch, Channels, Time]
    wav = torch.from_numpy(y).unsqueeze(0).unsqueeze(0).to(device)
    
    # FIX: Calcoliamo la durata direttamente dall'array in memoria!
    duration = len(y) / sr

    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        # DAC preprocessa, codifica e decodifica nativamente
        x = model.preprocess(wav, 24000)
        z, codes, latents, _, _ = model.encode(x)
        y_rec_tensor = model.decode(z)
    torch.cuda.synchronize()
    enc_ms = (time.perf_counter() - start) * 1000

    y_rec = y_rec_tensor.squeeze().cpu().numpy()
    min_len = min(len(y), len(y_rec))
    y, y_rec = y[:min_len], y_rec[:min_len]

    pesq_val = compute_pesq(y, y_rec, sr)
    audio_dist = 4.5 - pesq_val
    
    kbps = target_kbps 

    return {
        "kbps": round(kbps, 2),
        "pesq": pesq_val,
        "mel_dist": compute_mel_distance(y, y_rec, sr),
        "audio_dist": round(audio_dist, 4),
        "enc_ms": round(enc_ms, 2)
    }