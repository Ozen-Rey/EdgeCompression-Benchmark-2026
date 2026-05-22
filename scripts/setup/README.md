# Router-Only Setup

The setup script prepares the router development environment. It does not
install or reproduce the full benchmark stack.

Run the interactive setup from the repository root:

```powershell
python scripts/setup/setup_router.py
```

The default answer for every action prompt is `No`. Use `--yes` only when you
want to accept router-only actions automatically:

```powershell
python scripts/setup/setup_router.py --yes --with-tests
```

Supported options:

```powershell
python scripts/setup/setup_router.py --dry-run
python scripts/setup/setup_router.py --venv .venv
python scripts/setup/setup_router.py --no-venv
python scripts/setup/setup_router.py --with-tests
python scripts/setup/setup_router.py --report-out setup_router_report.json
```

Allowed actions, with confirmation:

- create a Python virtual environment;
- upgrade `pip` inside that environment;
- install this repository as `python -m pip install -e ".[test]"`;
- run router CLI smoke checks;
- run a lightweight router-only pytest subset.

Non-goals:

- no external codec installation;
- no dataset downloads;
- no checkpoint downloads;
- no system package installation;
- no benchmark execution;
- no writes under `results/`.

Clean Arch validation showed that router dependencies must be explicit.
`Pillow` is a router Python dependency because the current router import path
includes content-aware image support through `PIL`. This setup remains
router-only: benchmark stacks, codec binaries, datasets and checkpoints stay
out of scope.

The read-only environment doctor reports the local router environment and
optional benchmark/execution tools:

```powershell
python scripts/setup/doctor.py --report-out environment_doctor_report.json
```

Missing optional tools such as `ffmpeg`, `cjxl`, `nvidia-smi`, VMAF or ViSQOL do
not make the doctor fail. They are only relevant for benchmark or execution
workflows outside this router setup.

Automated environment validation can be run after setup to collect
cross-platform router setup/replay evidence:

```bash
python scripts/setup/validate_router_environment.py \
  --label arch_laptop_3080 \
  --out-dir validation_runs/arch_laptop_3080_v0463
```

Windows PowerShell:

```powershell
python scripts/setup/validate_router_environment.py `
  --label windows_workstation `
  --out-dir validation_runs/windows_workstation_v0463
```

The validation script writes a manifest, text summary, doctor report, CLI help
captures, `pip freeze`, and fixed audio/video/image fixture router reports
under `--out-dir` only. `validation_runs/` is local output and should not be
committed. This validates router portability and deterministic replay on fixed
R-D-E rows; it does not reproduce the benchmark.

It also writes full and sanitized platform fingerprints. The fingerprint
documents the validation context (OS, CPU, RAM when available, optional GPU
details, Python, Git and external tool visibility) so router reproducibility is
not confused with measurement reproducibility. It does not show that energy
measurements are hardware-invariant. The sanitized JSON replaces home/repo
paths and omits host, network and serial identifiers; use the sanitized version
for sharing.
