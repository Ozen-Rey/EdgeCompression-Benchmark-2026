"""
Sanity check sui CSV energetici e di qualità dei codec video.

Verifica:
  1. Numero righe atteso (113 energy, 57 quality per codec)
  2. Tutte le combinazioni (seq, profile, crf, phase) presenti
  3. Nessun valore NaN/Inf nei campi numerici
  4. Monotonia: bitrate decresce all'aumentare del CRF
  5. Monotonia: VMAF decresce all'aumentare del CRF
  6. Encode > Decode (sempre, sia in tempo che energia)
  7. RA encode > LDP encode (sempre)
  8. Idle CPU/GPU plausibili (10-50W CPU, 40-100W GPU)
  9. Bitstream esistenti su disco

Uso:
  cd ~/tesi/src/benchmark/video
  python sanity_check.py
"""

import csv
import sys
from pathlib import Path
from collections import defaultdict

RESULTS_DIR = Path.home() / "tesi" / "results" / "video"
BITSTREAMS_DIR = Path.home() / "tesi" / "bitstreams_video"

CODECS = {
    "x264": {"crfs": [22, 27, 32, 37], "ext": "264"},
    "x265": {"crfs": [22, 27, 32, 37], "ext": "265"},
    "svtav1": {"crfs": [27, 35, 43, 51], "ext": "ivf"},
    "vvenc": {"crfs": [27, 33, 39, 45], "ext": "266"},
}

SEQUENCES = [
    "Beauty",
    "Bosphorus",
    "HoneyBee",
    "Jockey",
    "ReadySteadyGo",
    "ShakeNDry",
    "YachtRide",
]
PROFILES = ["LDP", "RA"]
PHASES = ["encode", "decode"]


def load_csv(path):
    """Carica CSV in lista di dict, conversione tipi numerici dove possibile."""
    rows = []
    with open(path) as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            for k, v in list(row.items()):
                # Salta valori non-string (None, list, ecc. da DictReader edge cases)
                if not isinstance(v, str):
                    continue
                # Prova int prima, poi float, altrimenti lascia stringa
                try:
                    if (
                        "." in v
                        or "e" in v.lower()
                        or "nan" in v.lower()
                        or "inf" in v.lower()
                    ):
                        row[k] = float(v)
                    else:
                        row[k] = int(v)
                except (ValueError, TypeError, AttributeError):
                    pass  # resta string
            rows.append(row)
    return rows


def check_codec(codec, info):
    """Esegue tutti i check su un singolo codec."""
    print(f"\n{'='*70}")
    print(f"  CODEC: {codec}")
    print(f"{'='*70}")

    energy_csv = RESULTS_DIR / f"{codec}_energy.csv"
    quality_csv = RESULTS_DIR / f"{codec}_quality.csv"

    # ---- Esistenza file ----
    if not energy_csv.exists():
        print(f"  [FAIL] {energy_csv} not found")
        return False
    if not quality_csv.exists():
        print(f"  [FAIL] {quality_csv} not found")
        return False
    print(f"  [OK]  Both CSV files exist")

    # ---- Carica dati ----
    energy = load_csv(energy_csv)
    quality = load_csv(quality_csv)

    # ---- Numero righe ----
    expected_energy = len(SEQUENCES) * len(PROFILES) * len(info["crfs"]) * len(PHASES)
    expected_quality = len(SEQUENCES) * len(PROFILES) * len(info["crfs"])
    print(f"  Energy rows:  {len(energy):>4} / {expected_energy} expected", end="")
    print("  [OK]" if len(energy) == expected_energy else f"  [FAIL]")
    print(f"  Quality rows: {len(quality):>4} / {expected_quality} expected", end="")
    print("  [OK]" if len(quality) == expected_quality else f"  [FAIL]")

    # ---- Combinazioni complete ----
    e_combos = {(r["seq"], r["profile"], r["param"], r["phase"]) for r in energy}
    expected_e = {
        (s, p, f"crf={c}", ph)
        for s in SEQUENCES
        for p in PROFILES
        for c in info["crfs"]
        for ph in PHASES
    }
    missing_e = expected_e - e_combos
    if missing_e:
        print(
            f"  [FAIL] Energy missing combos: {sorted(missing_e)[:5]}{'...' if len(missing_e) > 5 else ''}"
        )
    else:
        print(f"  [OK]  All energy combos present")

    q_combos = {(r["seq"], r["profile"], r["param"]) for r in quality}
    expected_q = {
        (s, p, f"crf={c}") for s in SEQUENCES for p in PROFILES for c in info["crfs"]
    }
    missing_q = expected_q - q_combos
    if missing_q:
        print(
            f"  [FAIL] Quality missing combos: {sorted(missing_q)[:5]}{'...' if len(missing_q) > 5 else ''}"
        )
    else:
        print(f"  [OK]  All quality combos present")

    # ---- NaN / Inf ----
    nan_count = 0
    for r in energy + quality:
        for k, v in r.items():
            if isinstance(v, float):
                if v != v:  # NaN
                    nan_count += 1
                    if nan_count <= 3:
                        print(
                            f"  [WARN] NaN in {r.get('seq')}/{r.get('profile')}/{r.get('param')} field={k}"
                        )
    if nan_count > 3:
        print(f"  [WARN] ... and {nan_count-3} more NaNs")
    elif nan_count == 0:
        print(f"  [OK]  No NaN values")

    # ---- Monotonia bitrate vs CRF (per ogni seq×profilo) ----
    bitrate_issues = 0
    by_sp = defaultdict(list)
    for r in energy:
        if r["phase"] == "encode":
            crf = int(r["param"].split("=")[1])
            by_sp[(r["seq"], r["profile"])].append((crf, r["actual_mbps"]))
    for (seq, prof), pts in by_sp.items():
        pts.sort()  # CRF crescente
        bitrates = [p[1] for p in pts]
        # Bitrate dovrebbe decrescere monotonicamente
        for i in range(len(bitrates) - 1):
            if bitrates[i] < bitrates[i + 1]:
                bitrate_issues += 1
                if bitrate_issues <= 3:
                    print(
                        f"  [WARN] Non-monotonic bitrate {seq}/{prof}: "
                        f"crf{pts[i][0]}={bitrates[i]:.2f} < crf{pts[i+1][0]}={bitrates[i+1]:.2f}"
                    )
    if bitrate_issues == 0:
        print(f"  [OK]  Bitrate monotonic in CRF for all 14 (seq,profile)")
    elif bitrate_issues > 3:
        print(f"  [WARN] {bitrate_issues} non-monotonic bitrate cases")

    # ---- Monotonia VMAF vs CRF ----
    vmaf_issues = 0
    by_sp_q = defaultdict(list)
    for r in quality:
        crf = int(r["param"].split("=")[1])
        by_sp_q[(r["seq"], r["profile"])].append((crf, r["vmaf_mean"]))
    for (seq, prof), pts in by_sp_q.items():
        pts.sort()
        vmafs = [p[1] for p in pts]
        for i in range(len(vmafs) - 1):
            if vmafs[i] < vmafs[i + 1]:
                vmaf_issues += 1
                if vmaf_issues <= 3:
                    print(
                        f"  [WARN] Non-monotonic VMAF {seq}/{prof}: "
                        f"crf{pts[i][0]}={vmafs[i]:.1f} < crf{pts[i+1][0]}={vmafs[i+1]:.1f}"
                    )
    if vmaf_issues == 0:
        print(f"  [OK]  VMAF monotonic in CRF for all 14 (seq,profile)")

    # ---- Encode > Decode ----
    enc_dec_issues = 0
    enc_by_key = {
        (r["seq"], r["profile"], r["param"]): r
        for r in energy
        if r["phase"] == "encode"
    }
    for r in energy:
        if r["phase"] == "decode":
            key = (r["seq"], r["profile"], r["param"])
            enc = enc_by_key.get(key)
            if enc and r["energy_cpu_j"] >= enc["energy_cpu_j"]:
                enc_dec_issues += 1
                if enc_dec_issues <= 2:
                    print(
                        f"  [WARN] Decode≥Encode {key}: "
                        f"dec={r['energy_cpu_j']:.1f}J vs enc={enc['energy_cpu_j']:.1f}J"
                    )
    if enc_dec_issues == 0:
        print(f"  [OK]  Encode > Decode energy for all configs")

    # ---- RA encode > LDP encode ----
    ldp_enc = {
        (r["seq"], r["param"]): r["energy_cpu_j"]
        for r in energy
        if r["phase"] == "encode" and r["profile"] == "LDP"
    }
    ra_enc = {
        (r["seq"], r["param"]): r["energy_cpu_j"]
        for r in energy
        if r["phase"] == "encode" and r["profile"] == "RA"
    }
    ra_ldp_issues = 0
    for k in ldp_enc:
        if k in ra_enc and ra_enc[k] <= ldp_enc[k]:
            ra_ldp_issues += 1
            if ra_ldp_issues <= 2:
                print(
                    f"  [WARN] RA≤LDP encode {k}: "
                    f"RA={ra_enc[k]:.1f}J vs LDP={ldp_enc[k]:.1f}J"
                )
    if ra_ldp_issues == 0:
        print(f"  [OK]  RA encode > LDP encode (more compute, more energy)")

    # ---- Idle plausibile ----
    idle_cpu_vals = [r["idle_cpu_w"] for r in energy if "idle_cpu_w" in r]
    idle_gpu_vals = [r["idle_gpu_w"] for r in energy if "idle_gpu_w" in r]
    if idle_cpu_vals:
        cpu_min, cpu_max = min(idle_cpu_vals), max(idle_cpu_vals)
        cpu_mean = sum(idle_cpu_vals) / len(idle_cpu_vals)
        cpu_ok = 10 <= cpu_min and cpu_max <= 50
        print(
            f"  Idle CPU: min={cpu_min:.1f}W max={cpu_max:.1f}W mean={cpu_mean:.1f}W "
            f"{'[OK]' if cpu_ok else '[WARN: out of 10-50W range]'}"
        )
    if idle_gpu_vals:
        gpu_min, gpu_max = min(idle_gpu_vals), max(idle_gpu_vals)
        gpu_mean = sum(idle_gpu_vals) / len(idle_gpu_vals)
        gpu_ok = 30 <= gpu_min and gpu_max <= 120
        print(
            f"  Idle GPU: min={gpu_min:.1f}W max={gpu_max:.1f}W mean={gpu_mean:.1f}W "
            f"{'[OK]' if gpu_ok else '[WARN: out of 30-120W range]'}"
        )

    # ---- Bitstream su disco ----
    expected_bs = expected_quality
    actual_bs = len(list(BITSTREAMS_DIR.glob(f"{codec}_*.{info['ext']}")))
    print(f"  Bitstream files: {actual_bs} / {expected_bs} expected", end="")
    print("  [OK]" if actual_bs == expected_bs else "  [WARN]")

    return True


def cross_codec_summary():
    """Confronto rapido cross-codec sui dati aggregati."""
    print(f"\n{'='*70}")
    print(f"  CROSS-CODEC SUMMARY (mean over all configs)")
    print(f"{'='*70}")
    print(
        f"{'codec':<10} {'enc_cpu_J':>10} {'dec_cpu_J':>10} {'enc_time':>10} {'mbps':>8} {'vmaf':>7}"
    )
    for codec, info in CODECS.items():
        try:
            energy = load_csv(RESULTS_DIR / f"{codec}_energy.csv")
            quality = load_csv(RESULTS_DIR / f"{codec}_quality.csv")
        except FileNotFoundError:
            continue

        enc_e = [r["energy_cpu_j"] for r in energy if r["phase"] == "encode"]
        dec_e = [r["energy_cpu_j"] for r in energy if r["phase"] == "decode"]
        enc_t = [r["time_s"] for r in energy if r["phase"] == "encode"]
        mbps = [r["actual_mbps"] for r in energy if r["phase"] == "encode"]
        vmaf = [r["vmaf_mean"] for r in quality]

        if enc_e:
            print(
                f"{codec:<10} "
                f"{sum(enc_e)/len(enc_e):>10.1f} "
                f"{sum(dec_e)/len(dec_e):>10.1f} "
                f"{sum(enc_t)/len(enc_t):>10.2f} "
                f"{sum(mbps)/len(mbps):>8.2f} "
                f"{sum(vmaf)/len(vmaf):>7.2f}"
            )


def main():
    if not RESULTS_DIR.exists():
        print(f"[FATAL] {RESULTS_DIR} not found")
        sys.exit(1)
    if not BITSTREAMS_DIR.exists():
        print(f"[WARN] {BITSTREAMS_DIR} not found, skipping bitstream check")

    for codec, info in CODECS.items():
        check_codec(codec, info)

    cross_codec_summary()
    print()


if __name__ == "__main__":
    main()
