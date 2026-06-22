"""Shared pytest fixtures for the EIA test suite."""

import os
import pytest


@pytest.fixture(autouse=True)
def _clear_aws_profile_for_unit_tests(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> None:
    """Clear AWS_PROFILE for unit tests so boto3 doesn't try to load a named
    profile that may not exist in the local ~/.aws/config.

    Unit tests mock all boto3 calls after construction, so the profile is
    irrelevant. Integration tests (marked with @pytest.mark.integration) are
    excluded — they run against real AWS and may rely on the profile.
    """
    if request.node.get_closest_marker("integration") is None:
        monkeypatch.delenv("AWS_PROFILE", raising=False)
