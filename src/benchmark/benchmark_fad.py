import os
import shutil
import subprocess
import glob
import torch
from pathlib import Path

# ================== HACK PER FADTK / TORCHCODEC ==================
# Forziamo brutalmente l'uso della CPU per aggirare il bug di libnvrtc.so.13
# in torchcodec. Nascondendo la GPU, PyTorch disabilita le ottimizzazioni
# hardware difettose e usa i binari CPU standard.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# ================== HEALTH CHECK ==================
print("=" * 60)
print("SYSTEM HEALTH CHECK (MODALITÀ CPU FORZATA)")
print("=" * 60)
print("✅ GPU intenzionalmente nascosta (CUDA_VISIBLE_DEVICES='')")
print("✅ Evitato il crash dei driver CUDA in libtorchcodec")
print(
    "FADTK estrarrà gli embedding VGGish su CPU. Nessun problema, ci metterà pochi minuti."
)
print("=" * 60)


# ================== CONFIGURAZIONE ==================
# Le cartelle originali usate nel benchmark
DATASETS = [
    os.path.expanduser("~/tesi/datasets_audio/librispeech_sample"),
    os.path.expanduser("~/tesi/datasets_audio/esc50_sample"),
    os.path.expanduser("~/tesi/datasets_audio/musdb_10s"),
]
RECONSTRUCTIONS_DIR = os.path.expanduser("~/tesi/results/audio/reconstructions")
FAD_WORKSPACE = os.path.expanduser("~/tesi/results/audio/fad_workspace")

# Codec da testare (uguali a quelli dello script principale)
CODEC_PARAMS = [
    "Opus_12",
    "Opus_24",
    "Opus_48",
    "EnCodec_1.5",
    "EnCodec_3.0",
    "EnCodec_6.0",
    "DAC_8.0",
    "SNAC_0.8",
]


def main():
    print("\n" + "=" * 60)
    print("INIZIO VALUTAZIONE FAD (Fréchet Audio Distance)")
    print("=" * 60)

    # 1. Creiamo una cartella "Reference" unita che contiene tutti i file originali
    ref_dir = os.path.join(FAD_WORKSPACE, "reference")
    os.makedirs(ref_dir, exist_ok=True)

    print("1. Preparazione cartella Reference...")
    copied_ref = 0
    for ds in DATASETS:
        files = glob.glob(f"{ds}/**/*.wav", recursive=True) + glob.glob(
            f"{ds}/**/*.flac", recursive=True
        )
        for f in files:
            dst = os.path.join(ref_dir, Path(f).name)
            # Copiamo solo se non esiste, evitando PermissionError su file Read-Only
            if not os.path.exists(dst):
                shutil.copy(f, dst)
                copied_ref += 1

    total_ref = len(os.listdir(ref_dir))
    print(
        f"   -> {total_ref} file totali in Reference (di cui {copied_ref} copiati in questa sessione)."
    )

    # 2. Separiamo i file ricostruiti per Codec
    print("\n2. Smistamento file compressi per Codec...")
    for cp in CODEC_PARAMS:
        cp_dir = os.path.join(FAD_WORKSPACE, cp)
        os.makedirs(cp_dir, exist_ok=True)

        # Cerca tutti i file che finiscono con _Codec_Param.wav
        rec_files = glob.glob(os.path.join(RECONSTRUCTIONS_DIR, f"*_{cp}.wav"))
        copied_rec = 0
        for f in rec_files:
            dst = os.path.join(cp_dir, Path(f).name)
            # Copiamo solo se non esiste
            if not os.path.exists(dst):
                shutil.copy(f, dst)
                copied_rec += 1

        total_cp = len(os.listdir(cp_dir))
        print(
            f"   -> {cp}: {total_cp} file preparati (di cui {copied_rec} copiati in questa sessione)."
        )

    # 3. Eseguiamo FADTK
    print("\n3. Calcolo FAD (Modello: VGGish)...")
    print("   (L'elaborazione del Reference richiederà qualche minuto la prima volta)")
    print("-" * 60)

    results = {}
    for cp in CODEC_PARAMS:
        cp_dir = os.path.join(FAD_WORKSPACE, cp)
        if len(os.listdir(cp_dir)) == 0:
            continue

        print(f"Analisi in corso per {cp}...")

        # Esegue il comando fadtk tramite subprocess.
        # Grazie all'impostazione di os.environ all'inizio, questo child-process NON vedrà la GPU.
        cmd = ["fadtk", "vggish", ref_dir, cp_dir]
        result = subprocess.run(cmd, capture_output=True, text=True)

        # FADTK stampa il risultato nell'output, cerchiamo il numero finale
        try:
            out_lines = result.stdout.strip().split("\n")
            fad_score = float(out_lines[-1].split()[-1])
            results[cp] = fad_score
            print(f"   [ FAD per {cp} ]: {fad_score:.4f}")
        except Exception as e:
            print(
                f"   [!] Errore parsing FAD per {cp}. Output raw stdout: {result.stdout.strip()[-100:]}"
            )
            if result.stderr:
                print(
                    f"       --- DETTAGLIO ERRORE PYTHON ---\n{result.stderr.strip()}\n       -------------------------------"
                )

    print("\n" + "=" * 60)
    print("RISULTATI FINALI FAD (Più basso è meglio!)")
    print("=" * 60)
    for cp, score in sorted(results.items(), key=lambda item: item[1]):
        print(f" {cp:<15} | FAD: {score:.4f}")


if __name__ == "__main__":
    main()
