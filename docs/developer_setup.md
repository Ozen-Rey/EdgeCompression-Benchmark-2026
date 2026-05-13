# Developer Setup

This repository supports editable installs for local development:

```powershell
python -m pip install -e .
python -m pytest -q
```

The editable install keeps the existing `src.router` module path stable while
making imports independent of the current working directory. Router CLIs should
continue to be invoked with module execution:

```powershell
python -m src.router.rde_router --help
python -m src.router.external_codec_spec --help
python -m src.router.external_codec_probe --help
python -m src.router.external_codec_dry_run --help
python -m src.router.external_codec_benchmark --help
python -m src.router.external_codec_rde_exporter --help
```

During the structural refactor, keep legacy direct-script import fallbacks in
place unless a module is fully covered by module-execution tests. This preserves
existing workflows while new code can rely on package imports.
