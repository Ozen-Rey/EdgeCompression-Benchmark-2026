import csv
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


PolicyPair = Tuple[str, str]


def load_content_policy_rules(path: str) -> Dict[Tuple[str, str], PolicyPair]:
    """
    Loads rules produced by content_metadata_policy.py.

    Expected CSV columns:
      policy_key, policy_value, selected_codec, selected_config

    Example:
      dataset, tecnick, JPEG, q=85
    """
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(f"Content policy rules file not found: {p}")

    rules: Dict[Tuple[str, str], PolicyPair] = {}

    with p.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        required = {
            "policy_key",
            "policy_value",
            "selected_codec",
            "selected_config",
        }

        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                "Content policy rules CSV missing columns: "
                + ", ".join(sorted(missing))
            )

        for row in reader:
            policy_key = str(row.get("policy_key", "")).strip()
            policy_value = str(row.get("policy_value", "")).strip()
            codec = str(row.get("selected_codec", "")).strip()
            config = str(row.get("selected_config", "")).strip()

            if not policy_key or not policy_value or not codec or not config:
                continue

            rules[(policy_key, policy_value)] = (codec, config)

    if not rules:
        raise ValueError(f"No content policy rules loaded from: {p}")

    return rules


def build_content_policy_report(
    *,
    enabled: bool,
    mode: str,
    rules_file: Optional[str],
    policy_key: str,
    policy_value: Optional[str],
    fallback: str = "router",
) -> Dict[str, Any]:
    """
    Resolves a source-aware / metadata-aware content policy suggestion.

    mode:
      report-only -> produce suggestion only
      apply       -> router may use the suggestion if feasible

    fallback:
      router -> if suggestion is unavailable/infeasible, keep normal router decision
    """
    if mode not in {"report-only", "apply"}:
        raise ValueError("content policy mode must be 'report-only' or 'apply'.")

    if fallback not in {"router"}:
        raise ValueError("content policy fallback must currently be 'router'.")

    report: Dict[str, Any] = {
        "enabled": enabled,
        "mode": mode,
        "applied": False,
        "rules_file": rules_file,
        "rules_loaded": False,
        "policy_key": policy_key,
        "policy_value": policy_value,
        "fallback": fallback,
        "suggestion": None,
        "warnings": [],
        "reasons": [],
    }

    if not enabled:
        report["reasons"].append("content_policy_disabled")
        return report

    if not policy_value:
        report["reasons"].append("missing_policy_value")
        report["warnings"].append(
            "No content source/context was provided; content policy cannot suggest a candidate."
        )
        return report

    if not rules_file:
        report["reasons"].append("missing_rules_file")
        report["warnings"].append("Content policy enabled but no rules file was provided.")
        return report

    rules = load_content_policy_rules(rules_file)
    report["rules_loaded"] = True
    report["num_rules"] = len(rules)

    key = (policy_key, str(policy_value))

    if key not in rules:
        report["reasons"].append("no_rule_for_policy_value")
        report["warnings"].append(
            f"No content policy rule found for {policy_key}={policy_value}."
        )
        return report

    codec, config = rules[key]

    report["suggestion"] = {
        "codec": codec,
        "config": config,
        "source": "content_policy_rules",
        "matched_key": policy_key,
        "matched_value": policy_value,
    }

    return report


def get_content_policy_preferred_candidate(
    report: Dict[str, Any],
) -> Optional[PolicyPair]:
    suggestion = report.get("suggestion")

    if not report.get("enabled", False):
        return None

    if report.get("mode") != "apply":
        return None

    if not suggestion:
        return None

    return str(suggestion["codec"]), str(suggestion["config"])
