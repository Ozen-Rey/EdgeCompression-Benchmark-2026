"""Codec executable fingerprints for calibration bundle provenance."""

from __future__ import annotations

import hashlib
import importlib
import subprocess
from pathlib import Path
from typing import Any

from src.router.codecs.codec_capabilities import get_codec_capability
from src.router.adaptation.system_probe import _find_executable


class CodecFingerprintError(ValueError):
    """Raised when codec fingerprints cannot be matched safely."""


class CalibrationStalenessError(CodecFingerprintError):
    """Raised when a bundle was built with stale codec executables."""


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    file_path = Path(path)
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _version_from_python_pillow() -> str | None:
    try:
        pil = importlib.import_module("PIL")
    except Exception:
        return None
    return str(getattr(pil, "__version__", None) or "unknown")


def _version_from_binary(path: str | Path, backend: str) -> str | None:
    binary = str(path)
    if backend == "ffmpeg_hevc_intra":
        command = [binary, "-version"]
    else:
        command = [binary, "--version"]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except Exception:
        return None

    output = (result.stdout or result.stderr or "").strip()
    if not output:
        return None
    return output.splitlines()[0].strip()


def _required_executable_for_codec(codec: str, backend: str | None) -> str | None:
    capability = get_codec_capability(codec)
    executables = capability.get("requires_executables", [])
    if executables:
        return str(executables[0])

    if backend == "cjxl":
        return "cjxl"
    if backend == "ffmpeg_hevc_intra":
        return "ffmpeg"
    return None


def fingerprint_codec(codec: str) -> dict[str, Any]:
    """Build a fingerprint for the current backend used by a codec."""

    capability = get_codec_capability(codec)
    canonical = str(capability.get("canonical", codec))
    backend = capability.get("execution_backend")

    if backend == "python_pillow":
        version = _version_from_python_pillow()
        fingerprint = {
            "backend": backend,
            "version": version,
            "binary_path": None,
            "binary_sha256": None,
            "available": version is not None,
        }
        if version is None:
            fingerprint["reason"] = "python_pillow_unavailable"
        return fingerprint

    executable = _required_executable_for_codec(canonical, backend)
    if executable is None:
        fingerprint = {
            "backend": backend,
            "version": None,
            "binary_path": None,
            "binary_sha256": None,
            "available": bool(capability.get("execution_supported", False)),
        }
        if not fingerprint["available"]:
            fingerprint["reason"] = "codec_execution_not_supported"
        return fingerprint

    executable_info = _find_executable(executable)
    binary_path = executable_info.get("path")
    available = bool(executable_info.get("available", False) and binary_path)

    if not available:
        return {
            "backend": backend,
            "version": None,
            "binary_path": binary_path,
            "binary_sha256": None,
            "available": False,
            "reason": f"missing_executable:{executable}",
        }

    return _fingerprint_binary_path(
        backend=backend,
        binary_path=str(binary_path),
    )


def _fingerprint_binary_path(
    *,
    backend: str | None,
    binary_path: str,
) -> dict[str, Any]:
    path = Path(binary_path)
    if not path.exists() or not path.is_file():
        return {
            "backend": backend,
            "version": None,
            "binary_path": str(path),
            "binary_sha256": None,
            "available": False,
        }

    return {
        "backend": backend,
        "version": _version_from_binary(path, str(backend)),
        "binary_path": str(path),
        "binary_sha256": _sha256_file(path),
        "available": True,
    }


def build_codec_fingerprints_for_manifest(
    accepted_scales: list[dict[str, Any]],
    applied_items: list[dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Fingerprint codecs represented by applied calibration or accepted scales."""

    codecs = sorted({
        str(item.get("codec"))
        for item in accepted_scales
        if item.get("codec") is not None
    } | {
        str(item.get("codec"))
        for item in (applied_items or [])
        if item.get("codec") is not None
    })

    return {
        codec: fingerprint_codec(codec)
        for codec in codecs
    }


def validate_codec_fingerprints(
    expected_fingerprints: Any,
) -> dict[str, Any]:
    """Validate manifest codec fingerprints against the current environment."""

    if expected_fingerprints is None:
        return {
            "enabled": False,
            "validated": False,
            "reason": "manifest_without_codec_fingerprints",
            "validated_codecs": [],
            "mismatches": [],
        }

    if not isinstance(expected_fingerprints, dict):
        raise CodecFingerprintError(
            "Calibration bundle codec_fingerprints must be an object."
        )

    if not expected_fingerprints:
        return {
            "enabled": False,
            "validated": False,
            "reason": "empty_codec_fingerprints",
            "validated_codecs": [],
            "mismatches": [],
        }

    mismatches: list[dict[str, Any]] = []
    validated_codecs: list[str] = []

    for codec, expected in sorted(expected_fingerprints.items()):
        if not isinstance(expected, dict):
            raise CodecFingerprintError(
                f"Calibration bundle codec fingerprint for {codec} must be an object."
            )

        current = _current_fingerprint_for_expected(str(codec), expected)
        validated_codecs.append(str(codec))
        mismatches.extend(_compare_fingerprint(str(codec), expected, current))

    report = {
        "enabled": True,
        "validated": not mismatches,
        "validated_codecs": validated_codecs,
        "mismatches": mismatches,
    }

    if mismatches:
        details = "; ".join(
            f"{item['codec']}:{item['field']}" for item in mismatches
        )
        raise CalibrationStalenessError(
            "Calibration bundle codec fingerprint mismatch: "
            f"{details}"
        )

    return report


def _current_fingerprint_for_expected(
    codec: str,
    expected: dict[str, Any],
) -> dict[str, Any]:
    backend = expected.get("backend")
    binary_path = expected.get("binary_path")

    if binary_path:
        return _fingerprint_binary_path(
            backend=str(backend) if backend is not None else None,
            binary_path=str(binary_path),
        )

    return fingerprint_codec(codec)


def _compare_fingerprint(
    codec: str,
    expected: dict[str, Any],
    current: dict[str, Any],
) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []

    if expected.get("backend") != current.get("backend"):
        mismatches.append(_mismatch(codec, "backend", expected, current))

    expected_available = bool(expected.get("available", False))
    current_available = bool(current.get("available", False))
    if expected_available and not current_available:
        mismatches.append(_mismatch(codec, "available", expected, current))
        return mismatches

    if expected.get("version") != current.get("version"):
        mismatches.append(_mismatch(codec, "version", expected, current))

    expected_hash = expected.get("binary_sha256")
    if expected_hash is not None and expected_hash != current.get("binary_sha256"):
        mismatches.append(_mismatch(codec, "binary_sha256", expected, current))

    return mismatches


def _mismatch(
    codec: str,
    field: str,
    expected: dict[str, Any],
    current: dict[str, Any],
) -> dict[str, Any]:
    return {
        "codec": codec,
        "field": field,
        "expected": expected.get(field),
        "current": current.get(field),
    }
