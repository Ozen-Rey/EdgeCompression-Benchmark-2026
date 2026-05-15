import sys
from pathlib import Path
from typing import List, Optional

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.router.cli import build_router_arg_parser
from src.router.context import RouterContext
from src.router.core.router_config import expand_argv_with_config
from src.router.pipeline import run_router


def main(argv: Optional[List[str]] = None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    original_argv = list(argv)

    argv, router_config_report = expand_argv_with_config(argv)

    expanded_argv = list(argv)

    parser = build_router_arg_parser()
    args = parser.parse_args(argv)
    router_context = RouterContext()

    run_router(
        args=args,
        router_context=router_context,
        original_argv=original_argv,
        expanded_argv=expanded_argv,
        router_config_report=router_config_report,
    )


if __name__ == "__main__":
    try:
        main()
    except ValueError as exc:
        print("\n=== R-D-E Router: infeasible request ===")
        print(str(exc))
        print()
        print("No codec/configuration can satisfy the current constraints.")
        print("Try one of the following:")
        print("  - relax --max-rate")
        print("  - lower --quality-floor or --near-quality-floor")
        print("  - disable --simulate-no-cuda if CUDA codecs are actually available")
        print("  - enable more codecs in the admissible pool")
        raise SystemExit(2)
