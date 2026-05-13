import sys as _sys
from pathlib import Path as _Path

if __package__ in {None, ""}:
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from src.router.adaptation import content_classifier_model as _module
from src.router.adaptation.content_classifier_model import *  # noqa: F401,F403


if __name__ == "__main__":  # pragma: no cover
    _module.main()
else:
    _sys.modules[__name__] = _module
