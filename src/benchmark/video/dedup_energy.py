"""
Deduplica i CSV energy mantenendo l'ULTIMA riga per ogni chiave logica
(seq, profile, param, phase). Salva un backup .bak prima di sovrascrivere.

Uso:
  python dedup_energy.py
"""

import csv
import shutil
from pathlib import Path

RESULTS_DIR = Path.home() / "tesi" / "results" / "video"
KEY_FIELDS = ("seq", "profile", "param", "phase")
CODECS = ["x264", "x265", "svtav1", "vvenc"]


def dedup(csv_path):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"  [SKIP] {csv_path.name} not found")
        return

    # Leggi tutto
    with open(csv_path) as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames
        rows = list(reader)

    if fieldnames is None:
        print(f"  [FAIL] {csv_path.name}: empty or malformed CSV (no header)")
        return

    n_in = len(rows)

    # Tieni l'ULTIMA occorrenza per ogni chiave
    # (dict in Python 3.7+ mantiene insertion order, e l'ultima sovrascrive)
    by_key = {}
    for row in rows:
        key = tuple(row[k] for k in KEY_FIELDS)
        by_key[key] = row

    deduped = list(by_key.values())
    n_out = len(deduped)
    n_dropped = n_in - n_out

    if n_dropped == 0:
        print(f"  [OK] {csv_path.name}: {n_in} rows, no duplicates")
        return

    # Backup prima di sovrascrivere
    backup = csv_path.with_suffix(".csv.bak")
    shutil.copy2(csv_path, backup)
    print(f"  [BACKUP] {backup.name}")

    # Riscrivi
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(deduped)

    print(
        f"  [DEDUP] {csv_path.name}: {n_in} → {n_out} rows ({n_dropped} duplicates removed)"
    )


def main():
    print(f"Deduplication of energy CSVs in {RESULTS_DIR}")
    print(f"Key: {KEY_FIELDS}")
    print(f"Strategy: keep LAST occurrence per key")
    print()

    for codec in CODECS:
        print(f"--- {codec} ---")
        dedup(RESULTS_DIR / f"{codec}_energy.csv")
        print()


if __name__ == "__main__":
    main()
