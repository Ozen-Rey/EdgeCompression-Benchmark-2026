from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RDEProfile:
    name: str
    w_e: float
    w_r: float
    w_d: float
    min_quality: Optional[float] = None


PROFILES = {
    "balanced": RDEProfile(
        name="balanced",
        w_e=0.33,
        w_r=0.33,
        w_d=0.34,
        min_quality=50.0,
    ),
    "energy-limited": RDEProfile(
        name="energy-limited",
        w_e=0.60,
        w_r=0.25,
        w_d=0.15,
        min_quality=40.0,
    ),
    "bandwidth-limited": RDEProfile(
        name="bandwidth-limited",
        w_e=0.20,
        w_r=0.60,
        w_d=0.20,
        min_quality=30.0,
    ),
    "quality-first": RDEProfile(
        name="quality-first",
        w_e=0.15,
        w_r=0.25,
        w_d=0.60,
        min_quality=80.0,
    ),
}


def get_profile(name: str) -> RDEProfile:
    key = name.strip().lower().replace("_", "-")

    if key not in PROFILES:
        available = ", ".join(PROFILES.keys())
        raise ValueError(f"Profilo sconosciuto: {name}. Profili disponibili: {available}")

    return PROFILES[key]


def available_profiles() -> list[str]:
    return list(PROFILES.keys())