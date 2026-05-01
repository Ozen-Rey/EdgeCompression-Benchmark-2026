"""
Migration script: aggiunge colonna `preset` ai CSV energy + quality esistenti
dei codec classici. Migra solo i CSV dei 4 classici (x264, x265, svtav1, vvenc),
NON tocca i neurali (che non hanno preset).

Preset originali (hardcoded nei run scripts pre-migration):
  - x264: medium
  - x265: medium
  - svtav1: p5
  - vvenc: medium

Schema CSV pre-migration:
  codec,seq,profile,param,phase,n_frames,time_s,energy_cpu_j,...
Schema CSV post-migration:
  codec,seq,profile,preset,param,phase,n_frames,time_s,energy_cpu_j,...

Esegue su entrambi energy e quality CSV.
Backup automatico (.bak) prima della modifica.

USO:
  python migrate_csv_preset.py
"""

import shutil
from pathlib import Path
import pandas as pd

RESULTS = Path.home() / "tesi" / "results" / "video"

# Preset originali per ogni codec classico
ORIGINAL_PRESETS = {
    "x264": "medium",
    "x265": "medium",
    "svtav1": "p5",
    "vvenc": "medium",
}


def migrate_csv(csv_path, preset_value):
    """Aggiunge colonna 'preset' al CSV se non già presente."""
    if not csv_path.exists():
        print(f"  [SKIP] {csv_path.name} non esiste")
        return False

    # Backup
    backup_path = csv_path.with_suffix(csv_path.suffix + ".bak")
    if not backup_path.exists():
        shutil.copy(csv_path, backup_path)
        print(f"  [BACKUP] {backup_path.name}")

    df = pd.read_csv(csv_path)

    if "preset" in df.columns:
        print(f"  [SKIP] {csv_path.name} ha già colonna 'preset'")
        return False

    # Inserisce colonna preset dopo profile
    profile_idx = df.columns.get_loc("profile")
    df.insert(profile_idx + 1, "preset", preset_value)

    df.to_csv(csv_path, index=False)
    print(
        f"  [OK] {csv_path.name}: aggiunto preset='{preset_value}' "
        f"({len(df)} righe)"
    )
    return True


def main():
    print("=" * 70)
    print("CSV MIGRATION: adding 'preset' column to classic codecs")
    print("=" * 70)

    for codec, preset in ORIGINAL_PRESETS.items():
        print(f"\n[{codec}] preset='{preset}'")

        # Energy CSV
        energy_path = RESULTS / f"{codec}_energy.csv"
        migrate_csv(energy_path, preset)

        # Quality CSV
        quality_path = RESULTS / f"{codec}_quality.csv"
        migrate_csv(quality_path, preset)

    print("\n" + "=" * 70)
    print("Migration complete.")
    print("Backups saved as <file>.bak")
    print("=" * 70)


if __name__ == "__main__":
    main()
