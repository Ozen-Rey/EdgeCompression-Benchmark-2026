import json
from pathlib import Path
from typing import Any, Dict, Optional


DEFAULT_QUALITY_THRESHOLDS: Dict[str, Any] = {
    "image": {
        "default_metric": "ssimulacra2",
        "higher_is_better": True,
        "targets": {
            "preview": 50.0,
            "normal": 50.0,
            "high": 80.0,
            "very-high": 90.0,
        },
    },
    "video": {
        "default_metric": "vmaf",
        "higher_is_better": True,
        "targets": {
            "preview": 70.0,
            "normal": 80.0,
            "high": 90.0,
            "very-high": 95.0,
        },
    },
    "audio": {
        "default_metric": "visqol",
        "higher_is_better": True,
        "targets": {
            "preview": 3.0,
            "normal": 3.5,
            "high": 4.0,
            "very-high": 4.5,
        },
    },
}


def load_quality_thresholds(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return DEFAULT_QUALITY_THRESHOLDS

    p = Path(path)

    if not p.exists():
        return DEFAULT_QUALITY_THRESHOLDS

    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("quality_thresholds.json non valido: root non è un oggetto JSON.")

    return data


def resolve_quality_floor(
    *,
    domain: str,
    quality_metric: str,
    quality_target: str,
    user_quality_floor: Optional[float],
    thresholds_file: Optional[str],
) -> Dict[str, Any]:
    thresholds = load_quality_thresholds(thresholds_file)

    domain_key = str(domain).strip().lower()
    target_key = str(quality_target).strip().lower()

    domain_cfg = thresholds.get(domain_key)

    if domain_cfg is None:
        domain_cfg = DEFAULT_QUALITY_THRESHOLDS["image"]
        domain_source = "fallback:image"
    else:
        domain_source = domain_key

    targets = domain_cfg.get("targets", {})
    if target_key not in targets:
        raise ValueError(
            f"quality-target non valido per domain={domain_key}: {quality_target}. "
            f"Target disponibili: {sorted(targets.keys())}"
        )

    target_floor = float(targets[target_key])
    user_floor = float(user_quality_floor) if user_quality_floor is not None else None

    if user_floor is None:
        effective_floor = target_floor
        policy = "target_floor"
    else:
        effective_floor = max(target_floor, user_floor)
        policy = "max(target_floor,user_floor)"

    thresholds_path = str(thresholds_file) if thresholds_file else None
    thresholds_file_exists = bool(thresholds_file and Path(thresholds_file).exists())

    return {
        "enabled": True,
        "source": thresholds_path,
        "source_exists": thresholds_file_exists,
        "domain": domain_key,
        "domain_source": domain_source,
        "quality_metric": quality_metric,
        "quality_target": target_key,
        "target_floor": target_floor,
        "user_quality_floor": user_floor,
        "effective_quality_floor": effective_floor,
        "higher_is_better": bool(domain_cfg.get("higher_is_better", True)),
        "policy": policy,
    }
