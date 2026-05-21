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

The read-only environment doctor reports the local router environment and
optional benchmark/execution tools:

```powershell
python scripts/setup/doctor.py --report-out environment_doctor_report.json
```

Missing optional tools such as `ffmpeg`, `cjxl`, `nvidia-smi`, VMAF or ViSQOL do
not make the doctor fail. They are only relevant for benchmark or execution
workflows outside this router setup.
