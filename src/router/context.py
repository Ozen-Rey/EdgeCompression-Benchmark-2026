"""Internal router run context for orchestration/report metadata."""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class RouterContext:
    """Carries internal metadata that should not live on CLI args."""

    run_manifest: Optional[Dict[str, Any]] = None
    calibration_bundle_report: Optional[Dict[str, Any]] = None
    calibration_bundle_validation_report: Optional[Dict[str, Any]] = None
    external_codecs_report: Optional[Dict[str, Any]] = None
    normalization_consistency_report: Optional[Dict[str, Any]] = None
    energy_provenance_report: Optional[Dict[str, Any]] = None
    energy_provenance_summary: Optional[Dict[str, Any]] = None
    energy_provenance_compatibility: Optional[Dict[str, Any]] = None
    energy_tier_policy: Optional[Dict[str, Any]] = None
