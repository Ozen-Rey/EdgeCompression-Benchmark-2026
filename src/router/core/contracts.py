"""Stable contract identifiers for the R-D-E router.

A contract identifier is a stable, citable string that names the schema and
the procedural pipeline used to move data through the router. It covers the
full chain:

    DomainSpec + DatasetManifest + measurements CSV
        --> ingestion (validation + R-D-E CSV emission)
        --> router decision (admissibility + ranking + receipt)

The identifier is intentionally a string constant, not a frozen API. Future
versions of the contract (e.g. ``rde_manifest_v2``) may introduce richer
measurement semantics; they will coexist with v1 rather than replace it. The
identifier lets external documents (thesis, paper, supplementary material)
cite "the v1 contract" unambiguously, without committing the router to
freezing internal interfaces.

The constant is intentionally placed in a dedicated module rather than next
to any specific onboarding script, because the contract spans the whole
ingestion-to-decision chain and not a single CLI entry point.
"""

from __future__ import annotations


ONBOARDING_CONTRACT_ID: str = "rde_manifest_v1"
"""Stable identifier of the current onboarding contract.

A consumer (test, doc, report) that needs to assert "the system implements
the v1 onboarding contract" reads this constant. A change to this constant
is a deliberate signal that the contract has evolved and that downstream
documentation must be revisited.
"""


__all__ = ["ONBOARDING_CONTRACT_ID"]
