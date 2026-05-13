import importlib.util
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional


def normalize_codec_name(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return "".join(ch for ch in text.lower() if ch.isalnum())


CODEC_CAPABILITIES: Dict[str, Dict[str, Any]] = {
    "jpeg": {
        "canonical": "JPEG",
        "family": "classical",
        "requires_cuda": False,
        "requires_torch": False,
        "requires_executables": [],
        "execution_supported": True,
        "execution_backend": "python_pillow",
        "resource_profile": {
            "cpu_load": "low",
            "memory": "low",
            "gpu": "none",
            "latency": "low",
            "energy": "low",
            "batch_friendly": True,
            "interactive_ok": True,
        },
    },
    "jxl": {
        "canonical": "JXL",
        "family": "classical",
        "requires_cuda": False,
        "requires_torch": False,
        "requires_executables": ["cjxl"],
        "execution_supported": True,
        "execution_backend": "cjxl",
        "resource_profile": {
            "cpu_load": "medium",
            "memory": "low",
            "gpu": "none",
            "latency": "medium",
            "energy": "medium",
            "batch_friendly": True,
            "interactive_ok": True,
        },
    },
    "jpegxl": {
        "canonical": "JXL",
        "family": "classical",
        "requires_cuda": False,
        "requires_torch": False,
        "requires_executables": ["cjxl"],
        "execution_supported": True,
        "execution_backend": "cjxl",
        "resource_profile": {
            "cpu_load": "medium",
            "memory": "low",
            "gpu": "none",
            "latency": "medium",
            "energy": "medium",
            "batch_friendly": True,
            "interactive_ok": True,
        },
    },
    "hevc": {
        "canonical": "HEVC",
        "family": "classical",
        "requires_cuda": False,
        "requires_torch": False,
        "requires_executables": ["ffmpeg"],
        "execution_supported": True,
        "execution_backend": "ffmpeg_hevc_intra",
        "resource_profile": {
            "cpu_load": "high",
            "memory": "medium",
            "gpu": "none",
            "latency": "high",
            "energy": "high",
            "batch_friendly": True,
            "interactive_ok": False,
        },
    },
    "jpegai": {
        "canonical": "JPEG_AI",
        "family": "neural",
        "requires_cuda": True,
        "requires_torch": False,
        "requires_executables": [],
        "execution_supported": False,
        "execution_backend": None,
        "resource_profile": {
            "cpu_load": "medium",
            "memory": "medium",
            "gpu": "medium",
            "latency": "high",
            "energy": "high",
            "batch_friendly": False,
            "interactive_ok": False,
        },
    },
    "balle": {
        "canonical": "Ballé",
        "family": "neural",
        "requires_cuda": True,
        "requires_torch": True,
        "requires_executables": [],
        "execution_supported": False,
        "execution_backend": None,
        "resource_profile": {
            "cpu_load": "medium",
            "memory": "high",
            "gpu": "high",
            "latency": "high",
            "energy": "very-high",
            "batch_friendly": False,
            "interactive_ok": False,
        },
    },
    "cheng": {
        "canonical": "Cheng",
        "family": "neural",
        "requires_cuda": True,
        "requires_torch": True,
        "requires_executables": [],
        "execution_supported": False,
        "execution_backend": None,
        "resource_profile": {
            "cpu_load": "medium",
            "memory": "high",
            "gpu": "high",
            "latency": "high",
            "energy": "very-high",
            "batch_friendly": False,
            "interactive_ok": False,
        },
    },
    "elic": {
        "canonical": "ELIC",
        "family": "neural",
        "requires_cuda": True,
        "requires_torch": True,
        "requires_executables": [],
        "execution_supported": False,
        "execution_backend": None,
        "resource_profile": {
            "cpu_load": "medium",
            "memory": "high",
            "gpu": "high",
            "latency": "high",
            "energy": "very-high",
            "batch_friendly": False,
            "interactive_ok": False,
        },
    },
    "tcm": {
        "canonical": "TCM",
        "family": "neural",
        "requires_cuda": True,
        "requires_torch": True,
        "requires_executables": [],
        "execution_supported": False,
        "execution_backend": None,
        "resource_profile": {
            "cpu_load": "medium",
            "memory": "high",
            "gpu": "high",
            "latency": "high",
            "energy": "very-high",
            "batch_friendly": False,
            "interactive_ok": False,
        },
    },
    "dcae": {
        "canonical": "DCAE",
        "family": "neural",
        "requires_cuda": True,
        "requires_torch": True,
        "requires_executables": [],
        "execution_supported": False,
        "execution_backend": None,
        "resource_profile": {
            "cpu_load": "medium",
            "memory": "high",
            "gpu": "high",
            "latency": "high",
            "energy": "very-high",
            "batch_friendly": False,
            "interactive_ok": False,
        },
    },
}


EXTERNAL_CODEC_CAPABILITIES: Dict[str, Dict[str, Any]] = {}


def load_external_codec_registry(path: str) -> Dict[str, Any]:
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(f"Codec registry not found: {p}")

    with p.open("r", encoding="utf-8") as f:
        registry = json.load(f)

    if not isinstance(registry, dict):
        raise ValueError("Codec registry root must be a JSON object.")

    codecs = registry.get("codecs", {})
    if not isinstance(codecs, dict):
        raise ValueError("Codec registry must contain a 'codecs' object.")

    EXTERNAL_CODEC_CAPABILITIES.clear()

    for codec_name, capability in codecs.items():
        if not isinstance(capability, dict):
            raise ValueError(f"Invalid codec capability for {codec_name}.")

        canonical = capability.get("canonical", codec_name)

        names = [codec_name, canonical]
        names.extend(capability.get("aliases", []))

        for name in names:
            key = normalize_codec_name(str(name))
            cap_copy = dict(capability)
            cap_copy["canonical"] = canonical
            cap_copy["_external_registry"] = True
            cap_copy["_registry_source"] = str(p)
            EXTERNAL_CODEC_CAPABILITIES[key] = cap_copy

    return {
        "enabled": True,
        "source": str(p),
        "version": registry.get("version"),
        "domain": registry.get("domain"),
        "num_codecs": len(codecs),
        "codecs": sorted(codecs.keys()),
    }


def get_codec_capability(codec_name: str) -> Dict[str, Any]:
    normalized = normalize_codec_name(codec_name)

    if normalized in EXTERNAL_CODEC_CAPABILITIES:
        return EXTERNAL_CODEC_CAPABILITIES[normalized]

    for key, capability in EXTERNAL_CODEC_CAPABILITIES.items():
        if key in normalized:
            return capability

    if normalized in CODEC_CAPABILITIES:
        return CODEC_CAPABILITIES[normalized]

    for key, capability in CODEC_CAPABILITIES.items():
        if key in normalized:
            return capability

    return {
        "canonical": codec_name,
        "family": "unknown",
        "requires_cuda": False,
        "prefers_cuda": False,
        "cpu_ok": True,
        "requires_torch": False,
        "requires_executables": [],
        "execution_supported": False,
        "execution_backend": None,
        "benchmark_only": True,
        "unknown": True,
    }


def get_codec_family(codec_name: str) -> str:
    cap = get_codec_capability(codec_name)
    return str(cap.get("family", "unknown"))


def is_neural_codec(codec_name: str) -> bool:
    return get_codec_family(codec_name) == "neural"


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _executable_available(system_state: Dict[str, Any], exe_name: str) -> bool:
    executables = system_state.get("executables", {})
    return bool(executables.get(exe_name, {}).get("available", False))


def _get_executable_path(system_state: Dict[str, Any], exe_name: str) -> Optional[str]:
    executables = system_state.get("executables", {})
    path = executables.get(exe_name, {}).get("path")
    return path


def evaluate_codec_compatibility(
    codec_name: str,
    system_state: Dict[str, Any],
    strict_executables: bool = False,
    simulate_no_cuda: bool = False,
) -> Dict[str, Any]:
    cap = get_codec_capability(codec_name)

    cuda_available = bool(system_state.get("cuda", {}).get("available", False))
    if simulate_no_cuda:
        cuda_available = False

    torch_info = system_state.get("cuda", {}).get("torch", {})
    torch_available = bool(torch_info.get("torch_available", False))

    executables = system_state.get("executables", {})

    compatible = True
    reasons: List[str] = []
    warnings: List[str] = []

    if cap.get("requires_cuda", False) and not cuda_available:
        compatible = False
        reasons.append("requires_cuda_but_cuda_unavailable")

    if cap.get("requires_torch", False) and not torch_available:
        compatible = False
        reasons.append("requires_torch_but_torch_unavailable")

    missing_execs = []
    for exe in cap.get("requires_executables", []):
        exe_info = executables.get(exe, {})
        if not exe_info.get("available", False):
            missing_execs.append(exe)

    if missing_execs:
        msg = "missing_executables:" + ",".join(missing_execs)
        if strict_executables:
            compatible = False
            reasons.append(msg)
        else:
            warnings.append(msg)

    return {
        "codec": codec_name,
        "canonical": cap.get("canonical", codec_name),
        "family": cap.get("family", "unknown"),
        "compatible": compatible,
        "requires_cuda": cap.get("requires_cuda", False),
        "requires_torch": cap.get("requires_torch", False),
        "requires_executables": cap.get("requires_executables", []),
        "execution_supported": cap.get("execution_supported", False),
        "execution_backend": cap.get("execution_backend"),
        "reasons": reasons,
        "warnings": warnings,
        "strict_executables": strict_executables,
    }


def filter_points_by_capabilities(
    points,
    system_state: Dict[str, Any],
    strict_executables: bool = False,
    simulate_no_cuda: bool = False,
):
    filtered = []
    excluded = []
    compatibility_by_codec: Dict[str, Dict[str, Any]] = {}

    for p in points:
        if p.codec not in compatibility_by_codec:
            compatibility_by_codec[p.codec] = evaluate_codec_compatibility(
                codec_name=p.codec,
                system_state=system_state,
                strict_executables=strict_executables,
                simulate_no_cuda=simulate_no_cuda,
            )

        compatibility = compatibility_by_codec[p.codec]

        if compatibility["compatible"]:
            filtered.append(p)
        else:
            excluded.append(
                {
                    "codec": p.codec,
                    "config": p.config,
                    "reasons": compatibility["reasons"],
                }
            )

    report = {
        "enabled": True,
        "strict_executables": strict_executables,
        "simulate_no_cuda": simulate_no_cuda,
        "num_before_capability_filtering": len(points),
        "num_after_capability_filtering": len(filtered),
        "num_excluded_by_capabilities": len(excluded),
        "compatibility_by_codec": compatibility_by_codec,
        "excluded": excluded,
    }

    return filtered, report


def _parse_config_value(config: str, key: str) -> Optional[str]:
    pattern = rf"{re.escape(key)}\s*=\s*([0-9.]+)"
    match = re.search(pattern, config, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    return None


def _default_output_path(input_path: str, codec: str) -> str:
    path = Path(input_path)
    codec_norm = normalize_codec_name(codec)

    if "jpeg" == codec_norm:
        suffix = ".jpg"
    elif "jxl" in codec_norm or "jpegxl" in codec_norm:
        suffix = ".jxl"
    elif "hevc" in codec_norm:
        suffix = ".hevc"
    else:
        suffix = ".bin"

    return str(path.with_suffix(suffix))


def _allowed_suffixes_for_codec(canonical: str) -> list[str]:
    if canonical == "JPEG":
        return [".jpg", ".jpeg"]
    if canonical == "JXL":
        return [".jxl"]
    if canonical == "HEVC":
        return [".mp4", ".mkv", ".hevc", ".h265"]
    return [".bin"]


def _preferred_suffix_for_codec(canonical: str) -> str:
    return _allowed_suffixes_for_codec(canonical)[0]


def _coerce_output_extension(
    output_path: Optional[str],
    input_path: str,
    canonical: str,
    warnings: list[str],
) -> str:
    if not output_path:
        return _default_output_path(input_path, canonical)

    path = Path(output_path)
    suffix = path.suffix.lower()
    allowed = _allowed_suffixes_for_codec(canonical)

    if suffix in allowed:
        return str(path)

    corrected = str(path.with_suffix(_preferred_suffix_for_codec(canonical)))

    warnings.append(
        f"output_extension_corrected:{path.suffix}->{Path(corrected).suffix}"
    )

    return corrected


def _config_params_from_external_registry(cap: Dict[str, Any], config: str) -> Dict[str, str]:
    config_map = cap.get("config_map", {})

    if config in config_map:
        return {
            str(k): str(v)
            for k, v in config_map[config].items()
        }

    params: Dict[str, str] = {}
    for part in str(config).split(","):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        params[key.strip()] = value.strip()

    return params


def _build_external_command(
    cap: Dict[str, Any],
    config: str,
    input_path: str,
    output_path: str,
    system_state: Dict[str, Any],
    plan: Dict[str, Any],
) -> Optional[List[str]]:
    template = cap.get("command_template")

    if not isinstance(template, list) or not template:
        plan["reasons"].append("external_command_missing_template")
        return None

    params = _config_params_from_external_registry(cap, config)

    placeholders: Dict[str, str] = {
        "input": input_path,
        "output": output_path,
    }

    for exe in cap.get("requires_executables", []):
        exe_path = _get_executable_path(system_state, exe)
        if exe_path is None:
            plan["reasons"].append(f"missing_executable:{exe}")
            return None
        placeholders[str(exe)] = exe_path

    placeholders.update(params)

    command: List[str] = []

    for item in template:
        text = str(item)

        try:
            resolved = text.format(**placeholders)
        except KeyError as exc:
            plan["reasons"].append(f"missing_command_placeholder:{exc.args[0]}")
            return None

        command.append(resolved)

    return command


def build_execution_plan(
    codec_name: str,
    config: str,
    input_path: Optional[str],
    output_path: Optional[str],
    system_state: Dict[str, Any],
    requested: bool = False,
) -> Dict[str, Any]:
    cap = get_codec_capability(codec_name)

    plan: Dict[str, Any] = {
        "requested": requested,
        "codec": codec_name,
        "canonical": cap.get("canonical", codec_name),
        "config": config,
        "execution_supported": cap.get("execution_supported", False),
        "execution_backend": cap.get("execution_backend"),
        "can_execute": False,
        "command": None,
        "input": input_path,
        "output": output_path,
        "reasons": [],
        "warnings": [],
    }

    if not requested:
        return plan

    if not cap.get("execution_supported", False):
        plan["reasons"].append("execution_backend_not_registered")
        return plan

    if not input_path:
        plan["reasons"].append("input_path_not_provided")
        return plan

    if not Path(input_path).exists():
        plan["warnings"].append("input_path_does_not_exist_yet")

    canonical = cap.get("canonical", codec_name)

    if cap.get("execution_backend") == "external_command":
        output_extension = cap.get("output_extension")

        if output_extension:
            path = (
                Path(output_path)
                if output_path
                else Path(_default_output_path(input_path, canonical))
            )
            if path.suffix.lower() != str(output_extension).lower():
                corrected = str(path.with_suffix(str(output_extension)))
                plan["warnings"].append(
                    f"output_extension_corrected:{path.suffix}->{Path(corrected).suffix}"
                )
                output_path = corrected
            else:
                output_path = str(path)
        else:
            output_path = _coerce_output_extension(
                output_path=output_path,
                input_path=input_path,
                canonical=canonical,
                warnings=plan["warnings"],
            )

        plan["output"] = output_path

        command = _build_external_command(
            cap=cap,
            config=config,
            input_path=input_path,
            output_path=output_path,
            system_state=system_state,
            plan=plan,
        )

        if command is None:
            return plan

        plan["command"] = command
        plan["can_execute"] = True
        return plan

    output_path = _coerce_output_extension(
        output_path=output_path,
        input_path=input_path,
        canonical=canonical,
        warnings=plan["warnings"],
    )

    plan["output"] = output_path

    if canonical == "JPEG":
        quality = _parse_config_value(config, "q") or "85"

        if not _module_available("PIL"):
            plan["reasons"].append("missing_python_module:PIL")
            return plan

        plan["command"] = [
            sys.executable,
            "-m",
            "src.router.codecs.simple_image_encoder",
            "--codec",
            "jpeg",
            "--input",
            input_path,
            "--output",
            output_path,
            "--quality",
            str(int(float(quality))),
        ]
        plan["can_execute"] = True
        return plan

    if canonical == "JXL":
        distance = _parse_config_value(config, "d") or "1.0"

        cjxl_path = _get_executable_path(system_state, "cjxl")
        if cjxl_path is None:
            plan["reasons"].append("missing_executable:cjxl")
            return plan

        plan["command"] = [
            cjxl_path,
            input_path,
            output_path,
            "-d",
            str(distance),
        ]
        plan["can_execute"] = True
        return plan

    if canonical == "HEVC":
        crf = _parse_config_value(config, "crf") or "25"

        ffmpeg_path = _get_executable_path(system_state, "ffmpeg")
        if ffmpeg_path is None:
            plan["reasons"].append("missing_executable:ffmpeg")
            return plan

        plan["command"] = [
            ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            input_path,
            "-frames:v",
            "1",
            "-vf",
            "format=yuv420p",
            "-c:v",
            "libx265",
            "-preset",
            "medium",
            "-x265-params",
            "keyint=1:min-keyint=1:no-scenecut=1",
            "-crf",
            str(crf),
            "-tag:v",
            "hvc1",
            output_path,
        ]
        plan["can_execute"] = True
        return plan

    plan["reasons"].append("no_command_generator_for_codec")
    return plan
