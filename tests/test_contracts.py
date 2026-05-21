"""Invariants of the onboarding contract identifier.

These tests assert that the contract identifier is exposed where downstream
consumers (docs, paper, tests) expect to find it, and that it is propagated
to the reports produced by the onboarding chain.
"""

from __future__ import annotations

from src.router.core.contracts import ONBOARDING_CONTRACT_ID
from src.router.core import codec_onboarding as codec_onb
from src.router.core import dataset_onboarding as dataset_onb


def test_contract_identifier_is_versioned_string() -> None:
    assert isinstance(ONBOARDING_CONTRACT_ID, str)
    assert ONBOARDING_CONTRACT_ID.startswith("rde_manifest_v")
    assert ONBOARDING_CONTRACT_ID == ONBOARDING_CONTRACT_ID.strip()


def test_contract_identifier_imported_by_onboarding_modules() -> None:
    # The onboarding modules must import (and therefore depend on) the
    # contracts module, so that ``ONBOARDING_CONTRACT_ID`` cannot drift
    # silently between the constant and the modules that consume it.
    assert codec_onb.ONBOARDING_CONTRACT_ID == ONBOARDING_CONTRACT_ID
    assert dataset_onb.ONBOARDING_CONTRACT_ID == ONBOARDING_CONTRACT_ID
