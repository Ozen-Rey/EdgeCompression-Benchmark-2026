import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_router_config(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Router config not found: {p}")

    with p.open("r", encoding="utf-8") as f:
        config = json.load(f)

    if not isinstance(config, dict):
        raise ValueError("Router config root must be a JSON object.")

    return config


def _as_list(value):
    if value is None:
        return None
    if isinstance(value, list):
        return value
    return [value]


def _add_value(args: List[str], flag: str, value: Any) -> None:
    if value is None:
        return

    if isinstance(value, bool):
        if value:
            args.append(flag)
        return

    if isinstance(value, list):
        args.extend([flag, ",".join(str(v) for v in value)])
        return

    args.extend([flag, str(value)])


def _get(config: Dict[str, Any], *keys, default=None):
    cur = config
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def config_to_cli_args(config: Dict[str, Any]) -> List[str]:
    args: List[str] = []

    # Root / data
    _add_value(args, "--domain", config.get("domain"))
    _add_value(args, "--csv", _get(config, "data", "csv"))
    _add_value(args, "--input", _get(config, "data", "input"))
    _add_value(args, "--output", _get(config, "data", "output"))
    _add_value(args, "--out", _get(config, "data", "out_report"))
    _add_value(args, "--out-dir", _get(config, "data", "out_dir"))

    # Columns
    _add_value(args, "--codec-col", _get(config, "columns", "codec"))
    _add_value(args, "--config-col", _get(config, "columns", "config"))
    _add_value(args, "--rate-col", _get(config, "columns", "rate"))
    _add_value(args, "--quality-col", _get(config, "columns", "quality"))
    _add_value(args, "--energy-col", _get(config, "columns", "energy"))
    _add_value(args, "--time-col", _get(config, "columns", "time"))

    # Selection
    selection = config.get("selection", {})
    _add_value(args, "--available-codecs", selection.get("available_codecs"))
    _add_value(args, "--exclude-codecs", selection.get("exclude_codecs"))
    _add_value(args, "--exclude-neural", selection.get("exclude_neural"))
    _add_value(args, "--aggregate-by-config", selection.get("aggregate_by_config"))
    _add_value(args, "--auto-weights", selection.get("auto_weights"))
    _add_value(args, "--all-profiles", selection.get("all_profiles"))
    _add_value(args, "--safe-mode", selection.get("safe_mode"))
    _add_value(args, "--quality-constraint-stat", selection.get("quality_constraint_stat"))
    _add_value(args, "--quality-target", selection.get("quality_target"))
    _add_value(args, "--quality-floor", selection.get("quality_floor"))
    _add_value(args, "--near-quality-floor", selection.get("near_quality_floor"))
    _add_value(args, "--allow-degraded-fallback", selection.get("allow_degraded_fallback"))
    _add_value(args, "--max-rate", selection.get("max_rate"))
    _add_value(args, "--max-energy", selection.get("max_energy"))
    _add_value(args, "--max-time-ms", selection.get("max_time_ms"))
    _add_value(args, "--strict-time", selection.get("strict_time"))

    # System
    system = config.get("system", {})
    _add_value(args, "--system-aware", system.get("system_aware"))
    _add_value(args, "--simulate-no-cuda", system.get("simulate_no_cuda"))
    _add_value(args, "--capability-aware", system.get("capability_aware"))
    _add_value(args, "--strict-executables", system.get("strict_executables"))

    # System feature extraction
    system_features = config.get("system_features", {})
    _add_value(args, "--system-features", system_features.get("enabled"))
    _add_value(args, "--system-probe-level", system_features.get("probe_level"))
    _add_value(args, "--system-feature-cache-ttl-s", system_features.get("cache_ttl_s"))
    _add_value(args, "--system-feature-cpu-interval-s", system_features.get("cpu_interval_s"))

    # Context policy
    context = config.get("context", {})
    _add_value(args, "--power-mode", context.get("power_mode"))
    _add_value(args, "--battery-percent", context.get("battery_percent"))
    _add_value(args, "--thermal-state", context.get("thermal_state"))
    _add_value(args, "--network-profile", context.get("network_profile"))
    _add_value(args, "--system-load", context.get("system_load"))

    # Normalization
    normalization = config.get("normalization", {})
    _add_value(args, "--normalization-mode", normalization.get("mode"))
    _add_value(args, "--normalization-file", normalization.get("file"))

    # Calibration
    calibration = config.get("calibration", {})
    _add_value(args, "--calibration-file", calibration.get("file"))

    # Quality thresholds
    quality_thresholds = config.get("quality_thresholds", {})
    _add_value(args, "--quality-thresholds-file", quality_thresholds.get("file"))

    # External codec/backend registry
    registry = config.get("codec_registry", {})
    _add_value(args, "--codec-registry-file", registry.get("file"))

    # Execution
    execution = config.get("execution", {})
    _add_value(args, "--generate-command", execution.get("generate_command"))
    _add_value(args, "--execute", execution.get("execute"))

    # Top-k
    topk = config.get("topk", {})
    _add_value(args, "--export-topk", topk.get("export"))
    _add_value(args, "--top-k", topk.get("k"))

    return args


def expand_argv_with_config(argv: List[str]) -> Tuple[List[str], Dict[str, Any]]:
    argv = list(argv)
    config_path = None
    cleaned: List[str] = []

    i = 0
    while i < len(argv):
        item = argv[i]

        if item == "--config":
            if i + 1 >= len(argv):
                raise ValueError("--config requires a path.")
            config_path = argv[i + 1]
            i += 2
            continue

        if item.startswith("--config="):
            config_path = item.split("=", 1)[1]
            i += 1
            continue

        cleaned.append(item)
        i += 1

    if not config_path:
        return cleaned, {
            "enabled": False,
        }

    config = load_router_config(config_path)
    config_args = config_to_cli_args(config)

    expanded = config_args + cleaned

    return expanded, {
        "enabled": True,
        "source": config_path,
        "experiment_name": config.get("experiment_name"),
        "config_args": config_args,
        "cli_override_args": cleaned,
        "raw_config": config,
    }
