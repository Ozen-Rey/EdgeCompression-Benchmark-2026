# Router Setup

The setup script prepares the router development environment. It does not
install or reproduce the full benchmark stack.

## Scope

This setup is intentionally router-only. It can prepare a Python environment,
install the repository in editable mode, run router CLI smoke checks, and run a
small router-only test subset. It does not attempt to install the benchmark
execution stack.

The benchmark depends on external codecs, datasets, checkpoints, quality tools,
codec builds, and hardware-specific energy measurement. Those components have
their own licenses, hardware assumptions, and installation procedures.
This repository does not redistribute or relicense those third-party
components.

Clean Arch validation revealed that router Python dependencies must be declared
explicitly. `Pillow` is a router dependency because the current router import
path includes content-aware image support through `PIL`. Router setup remains
router-only: the benchmark stack, codec binaries, datasets and checkpoints are
still out of scope.

## Interactive Setup

From the repository root:

```powershell
python scripts/setup/setup_router.py
```

Every prompt defaults to `No`:

```text
Virtual environment .venv not found. Create it? [y/N]
Install Python package in editable mode with test dependencies? [y/N]
Run router smoke checks? [y/N]
Run lightweight test subset? [y/N]
```

Use `--dry-run` to see the planned actions without changing anything:

```powershell
python scripts/setup/setup_router.py --dry-run
```

Use `--yes` only when you want to accept router-only actions automatically:

```powershell
python scripts/setup/setup_router.py --yes --with-tests
```

Other useful options:

```powershell
python scripts/setup/setup_router.py --venv .venv
python scripts/setup/setup_router.py --no-venv
python scripts/setup/setup_router.py --with-tests
python scripts/setup/setup_router.py --strict
python scripts/setup/setup_router.py --report-out setup_router_report.json
```

## Allowed Actions

With explicit confirmation, the setup script may:

- create a Python virtual environment;
- upgrade `pip` inside the selected environment;
- run `python -m pip install -e ".[test]"`;
- run router CLI `--help` smoke checks;
- run a lightweight router-only pytest subset.

## Non-Goals

The setup script does not:

- install external codecs;
- install system packages;
- download datasets;
- download checkpoints;
- reproduce the benchmark;
- run benchmark scripts;
- write benchmark outputs under `results/`;
- guarantee identical energy measurements across machines.

Benchmark and execution setup remains separate and domain-specific. See the
benchmark and external-codec documentation for those workflows.
Users are responsible for obtaining external datasets, codec binaries, models,
checkpoints and metric tools under their upstream terms.

## Environment Doctor

The doctor is read-only:

```powershell
python scripts/setup/doctor.py --report-out environment_doctor_report.json
```

It reports:

- OS and Python information;
- active virtual environment metadata;
- router import status and version;
- router CLI help checks;
- Python dependency visibility;
- `Pillow` / `PIL` availability for the router CLI import path;
- optional external tools such as `ffmpeg`, `ffprobe`, `cjxl`, `djxl`, and
  `nvidia-smi`.

Optional external tools are labeled as benchmark/execution dependencies. Missing
tools do not make the doctor fail, because the router development environment
does not require them.

## Environment Validation Report

After a machine has been prepared with `setup_router.py`, the validation
script can collect portable router setup evidence:

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

The script is read-only with respect to the repository source and benchmark
artifacts. It does not install packages, create virtual environments, download
datasets or checkpoints, run benchmarks, require external codec binaries, or
write under `results/`. It writes only inside `--out-dir`.

`validation_runs/` is local evidence output and should not be committed. The
generated manifest and summary demonstrate that the router can be imported,
its CLIs can start, and the fixed audio/video/image R-D-E fixtures can be
replayed on the current platform. They do not replace benchmark reproduction
and do not validate hardware-invariant energy measurements.

The validation run also writes `platform_fingerprint_full.json` and
`platform_fingerprint_sanitized.json`. The fingerprint documents the platform
context used for router validation: OS, CPU, RAM when available, optional GPU
details, Python, Git and external tool visibility. This helps distinguish
router reproducibility from measurement reproducibility. It does not prove that
energy measurements are hardware-invariant. The sanitized fingerprint replaces
home/repository paths and omits host, network and serial identifiers; it is the
version intended for sharing.
