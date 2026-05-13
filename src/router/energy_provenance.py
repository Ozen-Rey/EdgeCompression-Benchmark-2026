import sys as _sys
from pathlib import Path as _Path

if __package__ in {None, ""}:
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from src.router.adaptation import energy_provenance as _module
from src.router.adaptation.energy_provenance import *  # noqa: F401,F403

_sys.modules[__name__] = _module
