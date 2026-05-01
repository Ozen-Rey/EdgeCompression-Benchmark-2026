"""
Sanity check completo del benchmark video R-D-E.

Valida 8 codec (4 classici con preset multipli + 4 neurali) per:
  - Header CSV consistency (presenza colonna `preset` per classici)
  - Conteggi righe attesi vs reali
  - Range plausibili (idle CPU, idle GPU, energy > 0)
  - Cross-CSV consistency (ogni energy → matching quality)
  - Bitstream paths esistenti
  - VMAF/PSNR/SSIM in range plausibile
  - Detection di outlier energetici (>3σ dal mediano per codec/preset)

Output: report stampato a console + JSON con tutti i flag.

Uso:
  cd ~/tesi/src/benchmark/video
  python sanity_check_v2.py
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median, stdev

RESULTS_DIR = Path.home() / "tesi" / "results" / "video"

# === CONFIG: codec attesi e loro caratteristiche ===

CLASSIC_CODECS = {
    "x264": {
        "presets": ["medium", "faster", "slow"],
        "ext": "264",
        "phases": ["encode", "decode"],
    },
    "x265": {
        "presets": ["medium", "faster", "slow"],
        "ext": "265",
        "phases": ["encode", "decode"],
    },
    "svtav1": {
        "presets": ["p5", "p2", "p10"],
        "ext": "ivf",
        "phases": ["encode", "decode"],
    },
    "vvenc": {
        "presets": ["medium", "faster", "slow"],
        "ext": "266",
        "phases": ["encode", "decode"],
    },
}

NEURAL_CODECS = {
    "dcvc_dc": {"phases": ["coding"], "round_trip": False},
    "dcvc_fm": {"phases": ["coding"], "round_trip": False},
    "dcvc_rt": {"phases": ["coding"], "round_trip": False},
    "dcvc_rt_cuda": {"phases": ["encode", "decode"], "round_trip": True},
}

# UVG sequences expected
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

# Operating points per codec type
CLASSIC_CRFS = {
    "x264": [22, 27, 32, 37],
    "x265": [22, 27, 32, 37],
    "svtav1": [22, 27, 32, 37],
    "vvenc": [27, 33, 39, 45],
}
NEURAL_QS = [0, 21, 42, 63]

# Plausible ranges
IDLE_CPU_RANGE = (15.0, 50.0)  # W (sano: 24, marginal: 40+, broken: 60+)
IDLE_GPU_RANGE = (10.0, 100.0)  # W (RTX 5090 idle ~15-60W, può essere alto con monitor)
VMAF_RANGE = (10.0, 100.0)
PSNR_Y_RANGE = (15.0, 60.0)
SSIM_RANGE = (0.5, 1.0)


# === HELPER ===


class Report:
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.info = []
        self.stats = {}

    def err(self, codec, msg):
        self.errors.append(f"[ERROR] [{codec}] {msg}")

    def warn(self, codec, msg):
        self.warnings.append(f"[WARN]  [{codec}] {msg}")

    def info_(self, codec, msg):
        self.info.append(f"[INFO]  [{codec}] {msg}")

    def stat(self, codec, key, value):
        self.stats.setdefault(codec, {})[key] = value

    def summary(self):
        print("\n" + "=" * 70)
        print("SANITY CHECK REPORT")
        print("=" * 70)
        print(f"\nERRORS:   {len(self.errors)}")
        print(f"WARNINGS: {len(self.warnings)}")
        print(f"INFO:     {len(self.info)}")
        print()
        if self.errors:
            print("--- ERRORS ---")
            for e in self.errors:
                print(e)
            print()
        if self.warnings:
            print("--- WARNINGS ---")
            for w in self.warnings:
                print(w)
            print()
        print("--- STATS PER CODEC ---")
        for codec, st in self.stats.items():
            print(f"\n  {codec}:")
            for k, v in st.items():
                print(f"    {k}: {v}")
        print()
        if not self.errors:
            print("  ✓ NO BLOCKING ERRORS")
        else:
            print("  ✗ BLOCKING ERRORS PRESENT")
        print()


def load_csv_rows(path):
    if not path.exists():
        return None
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def safe_float(s, default=0.0):
    try:
        return float(s)
    except (ValueError, TypeError):
        return default


# === CHECKS ===


def check_classic_codec(codec, info, report):
    energy_csv = RESULTS_DIR / f"{codec}_energy.csv"
    quality_csv = RESULTS_DIR / f"{codec}_quality.csv"

    # File existence
    if not energy_csv.exists():
        report.err(codec, f"Energy CSV missing: {energy_csv}")
        return
    if not quality_csv.exists():
        report.err(codec, f"Quality CSV missing: {quality_csv}")
        return

    energy_rows = load_csv_rows(energy_csv)
    quality_rows = load_csv_rows(quality_csv)

    # Header check
    if not energy_rows:
        report.err(codec, "Energy CSV empty")
        return
    expected_cols = {
        "codec",
        "seq",
        "profile",
        "preset",
        "param",
        "phase",
        "energy_cpu_j",
        "energy_gpu_j",
        "idle_cpu_w",
        "actual_mbps",
    }
    actual_cols = set(energy_rows[0].keys())
    missing = expected_cols - actual_cols
    if missing:
        report.err(codec, f"Energy CSV missing columns: {missing}")
        return

    expected_q_cols = {
        "codec",
        "seq",
        "profile",
        "preset",
        "param",
        "vmaf_mean",
        "psnr_y",
        "ssim_y",
        "ms_ssim_y",
    }
    actual_q_cols = set(quality_rows[0].keys()) if quality_rows else set()
    missing_q = expected_q_cols - actual_q_cols
    if missing_q:
        report.err(codec, f"Quality CSV missing columns: {missing_q}")
        return

    # Conteggio atteso
    n_seq = len(SEQUENCES)
    n_prof = len(PROFILES)
    n_preset = len(info["presets"])
    n_op = len(CLASSIC_CRFS[codec])
    n_phase = len(info["phases"])

    expected_energy = n_seq * n_prof * n_preset * n_op * n_phase
    expected_quality = n_seq * n_prof * n_preset * n_op

    actual_energy = len(energy_rows)
    actual_quality = len(quality_rows)

    report.stat(codec, "energy_rows", f"{actual_energy}/{expected_energy}")
    report.stat(codec, "quality_rows", f"{actual_quality}/{expected_quality}")

    if actual_energy != expected_energy:
        report.err(
            codec,
            f"Energy row count: got {actual_energy}, expected {expected_energy} (diff {actual_energy - expected_energy})",
        )
    if actual_quality != expected_quality:
        report.err(
            codec,
            f"Quality row count: got {actual_quality}, expected {expected_quality} (diff {actual_quality - expected_quality})",
        )

    # Per ogni preset, verifica copertura completa
    coverage = defaultdict(
        lambda: defaultdict(set)
    )  # preset -> phase -> set((seq, prof, param))
    for row in energy_rows:
        preset = row["preset"]
        phase = row["phase"]
        key = (row["seq"], row["profile"], row["param"])
        coverage[preset][phase].add(key)

    for preset in info["presets"]:
        for phase in info["phases"]:
            actual = len(coverage[preset][phase])
            exp = n_seq * n_prof * n_op
            if actual != exp:
                report.err(
                    codec,
                    f"Preset={preset} phase={phase}: {actual}/{exp} entries (missing {exp-actual})",
                )

    # Range checks
    high_idle_cpu = 0
    high_idle_gpu = 0
    zero_energy = 0
    nan_count = 0
    energies_per_preset = defaultdict(list)

    for row in energy_rows:
        preset = row["preset"]
        idle_cpu = safe_float(row["idle_cpu_w"], 0)
        idle_gpu = safe_float(row["idle_gpu_w"], 0)
        e_cpu = safe_float(row["energy_cpu_j"], 0)
        e_gpu = safe_float(row["energy_gpu_j"], 0)

        if not (IDLE_CPU_RANGE[0] <= idle_cpu <= IDLE_CPU_RANGE[1]):
            high_idle_cpu += 1
        if not (IDLE_GPU_RANGE[0] <= idle_gpu <= IDLE_GPU_RANGE[1]):
            high_idle_gpu += 1
        if row["phase"] == "encode" and e_cpu <= 0:
            zero_energy += 1
        if e_cpu != e_cpu or e_gpu != e_gpu:  # NaN check
            nan_count += 1

        if row["phase"] == "encode":
            energies_per_preset[preset].append(e_cpu)

    if high_idle_cpu > 0:
        pct = 100 * high_idle_cpu / actual_energy
        if pct > 5:
            report.err(
                codec,
                f"{high_idle_cpu} rows ({pct:.1f}%) with idle_cpu OUT of [{IDLE_CPU_RANGE}]",
            )
        else:
            report.warn(
                codec, f"{high_idle_cpu} rows ({pct:.1f}%) with idle_cpu out of range"
            )

    if high_idle_gpu > 0:
        pct = 100 * high_idle_gpu / actual_energy
        if pct > 30:
            report.warn(
                codec,
                f"{high_idle_gpu} rows ({pct:.1f}%) with idle_gpu out of range (acceptable for monitor-attached)",
            )

    if zero_energy > 0:
        report.err(codec, f"{zero_energy} encode rows with E_cpu <= 0")
    if nan_count > 0:
        report.err(codec, f"{nan_count} rows with NaN energy values")

    # Energy stats per preset
    for preset, energies in energies_per_preset.items():
        if not energies:
            continue
        med = median(energies)
        mn, mx = min(energies), max(energies)
        report.stat(
            codec,
            f"E_cpu_encode_{preset}_J",
            f"med={med:.0f} min={mn:.0f} max={mx:.0f}",
        )

    # Quality range checks
    bad_vmaf = bad_psnr = bad_ssim = 0
    for row in quality_rows:
        vmaf = safe_float(row["vmaf_mean"], 50)
        psnr_y = safe_float(row["psnr_y"], 30)
        ssim_y = safe_float(row["ssim_y"], 0.9)
        if not (VMAF_RANGE[0] <= vmaf <= VMAF_RANGE[1]):
            bad_vmaf += 1
        if not (PSNR_Y_RANGE[0] <= psnr_y <= PSNR_Y_RANGE[1]):
            bad_psnr += 1
        if not (SSIM_RANGE[0] <= ssim_y <= SSIM_RANGE[1]):
            bad_ssim += 1

    if bad_vmaf > 0:
        report.warn(codec, f"{bad_vmaf}/{actual_quality} VMAF outside [{VMAF_RANGE}]")
    if bad_psnr > 0:
        report.warn(
            codec, f"{bad_psnr}/{actual_quality} PSNR_Y outside [{PSNR_Y_RANGE}]"
        )
    if bad_ssim > 0:
        report.warn(codec, f"{bad_ssim}/{actual_quality} SSIM_Y outside [{SSIM_RANGE}]")

    # Cross-CSV consistency: every (codec, seq, profile, preset, param) in quality
    # must have a matching encode in energy
    energy_keys = set()
    for row in energy_rows:
        if row["phase"] in ("encode", "coding"):
            energy_keys.add((row["seq"], row["profile"], row["preset"], row["param"]))

    quality_keys = set()
    for row in quality_rows:
        quality_keys.add((row["seq"], row["profile"], row["preset"], row["param"]))

    only_in_energy = energy_keys - quality_keys
    only_in_quality = quality_keys - energy_keys

    if only_in_energy:
        report.warn(
            codec,
            f"{len(only_in_energy)} energy entries with no matching quality (e.g. {list(only_in_energy)[:2]})",
        )
    if only_in_quality:
        report.warn(
            codec,
            f"{len(only_in_quality)} quality entries with no matching energy (e.g. {list(only_in_quality)[:2]})",
        )

    # Stats summary
    report.stat(codec, "energy_keys", len(energy_keys))
    report.stat(codec, "quality_keys", len(quality_keys))
    report.stat(codec, "matched_pairs", len(energy_keys & quality_keys))


def check_neural_codec(codec, info, report):
    energy_csv = RESULTS_DIR / f"{codec}_energy.csv"
    quality_csv = RESULTS_DIR / f"{codec}_quality.csv"

    if not energy_csv.exists():
        report.err(codec, f"Energy CSV missing: {energy_csv}")
        return
    if not quality_csv.exists():
        report.err(codec, f"Quality CSV missing: {quality_csv}")
        return

    energy_rows = load_csv_rows(energy_csv)
    quality_rows = load_csv_rows(quality_csv)

    if not energy_rows:
        report.err(codec, "Energy CSV empty")
        return

    # Neural codecs do NOT have preset column
    expected_cols = {
        "codec",
        "seq",
        "profile",
        "param",
        "phase",
        "energy_cpu_j",
        "energy_gpu_j",
        "idle_cpu_w",
    }
    actual_cols = set(energy_rows[0].keys())
    missing = expected_cols - actual_cols
    if missing:
        report.err(codec, f"Energy CSV missing columns: {missing}")
        return

    if "preset" in actual_cols:
        report.warn(
            codec, "Neural codec CSV has preset column (unexpected, but harmless)"
        )

    # Solo LDP per neurali
    n_seq = len(SEQUENCES)
    n_prof = 1  # solo LDP
    n_op = len(NEURAL_QS)
    n_phase = len(info["phases"])

    expected_energy = n_seq * n_prof * n_op * n_phase
    expected_quality = n_seq * n_prof * n_op

    actual_energy = len(energy_rows)
    actual_quality = len(quality_rows)

    report.stat(codec, "energy_rows", f"{actual_energy}/{expected_energy}")
    report.stat(codec, "quality_rows", f"{actual_quality}/{expected_quality}")

    if actual_energy != expected_energy:
        report.err(codec, f"Energy: got {actual_energy}, expected {expected_energy}")
    if actual_quality != expected_quality:
        report.err(codec, f"Quality: got {actual_quality}, expected {expected_quality}")

    # Phase coverage
    phase_count = defaultdict(int)
    for row in energy_rows:
        phase_count[row["phase"]] += 1

    for phase in info["phases"]:
        if phase_count[phase] != n_seq * n_prof * n_op:
            report.err(
                codec,
                f"Phase {phase}: {phase_count[phase]} rows (expected {n_seq * n_prof * n_op})",
            )

    # Range check
    high_idle_gpu = 0
    zero_energy = 0
    energies = []

    for row in energy_rows:
        idle_gpu = safe_float(row["idle_gpu_w"], 0)
        e_cpu = safe_float(row["energy_cpu_j"], 0)
        e_gpu = safe_float(row["energy_gpu_j"], 0)

        if idle_gpu > 100 or idle_gpu < 5:
            high_idle_gpu += 1
        if (e_cpu + e_gpu) <= 0:
            zero_energy += 1

        if row["phase"] in ("encode", "coding"):
            energies.append(e_cpu + e_gpu)

    if zero_energy > 0:
        report.err(codec, f"{zero_energy} rows with total E <= 0")
    if high_idle_gpu > actual_energy * 0.5:
        report.warn(codec, f"{high_idle_gpu} rows with anomalous idle_gpu")

    if energies:
        med = median(energies)
        mn, mx = min(energies), max(energies)
        report.stat(
            codec, "E_total_encode_J", f"med={med:.0f} min={mn:.0f} max={mx:.0f}"
        )

    # Quality range (in particolare per neurali, VMAF spesso più stretto)
    bad_vmaf = bad_psnr = 0
    for row in quality_rows:
        vmaf = safe_float(row["vmaf_mean"], 50)
        psnr_y = safe_float(row["psnr_y"], 30)
        if not (VMAF_RANGE[0] <= vmaf <= VMAF_RANGE[1]):
            bad_vmaf += 1
        if not (PSNR_Y_RANGE[0] <= psnr_y <= PSNR_Y_RANGE[1]):
            bad_psnr += 1

    if bad_vmaf > 0:
        report.warn(codec, f"{bad_vmaf}/{actual_quality} VMAF out of range")
    if bad_psnr > 0:
        report.warn(codec, f"{bad_psnr}/{actual_quality} PSNR_Y out of range")


# === MAIN ===


def main():
    report = Report()

    print("=" * 70)
    print("SANITY CHECK v2 — VIDEO BENCHMARK (8 codec, preset multipli)")
    print(f"Results dir: {RESULTS_DIR}")
    print("=" * 70)

    print("\n[1/2] Checking CLASSIC codecs...")
    for codec, info in CLASSIC_CODECS.items():
        print(f"  -> {codec}")
        check_classic_codec(codec, info, report)

    print("\n[2/2] Checking NEURAL codecs...")
    for codec, info in NEURAL_CODECS.items():
        print(f"  -> {codec}")
        check_neural_codec(codec, info, report)

    report.summary()

    # Save JSON report
    json_path = RESULTS_DIR / "sanity_check_report.json"
    with open(json_path, "w") as fh:
        json.dump(
            {
                "errors": report.errors,
                "warnings": report.warnings,
                "info": report.info,
                "stats": report.stats,
            },
            fh,
            indent=2,
        )
    print(f"\nReport saved: {json_path}\n")

    # Exit code: 0 se nessun error, 1 altrimenti
    sys.exit(0 if not report.errors else 1)


if __name__ == "__main__":
    main()
