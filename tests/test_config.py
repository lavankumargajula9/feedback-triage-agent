"""Tests for triage.config — model-role profiles and API key resolution.

Covers U1 requirements:
- R30: dev/measurement decoupling with centralized model roles.
- R22: judge model must differ from draft (pipeline/baseline) model in the
  measurement profile, asserted in code at load time.
"""

import pytest

from triage.config import (
    DEV,
    MEASUREMENT,
    ROLE_BASELINE,
    ROLE_JUDGE,
    ROLE_PIPELINE,
    ConfigError,
    ModelRoles,
    get_api_key,
    get_model,
    validate_measurement_roles,
)


class TestHappyPath:
    def test_measurement_pipeline_model(self):
        assert get_model(ROLE_PIPELINE, MEASUREMENT) == "claude-sonnet-5"

    def test_measurement_baseline_model(self):
        assert get_model(ROLE_BASELINE, MEASUREMENT) == "claude-sonnet-5"

    def test_measurement_judge_model(self):
        assert get_model(ROLE_JUDGE, MEASUREMENT) == "claude-opus-5"

    def test_dev_profile_uses_cheap_model_for_all_roles(self):
        for role in (ROLE_PIPELINE, ROLE_BASELINE, ROLE_JUDGE):
            assert get_model(role, DEV) == "claude-haiku-4-5"

    def test_measurement_judge_differs_from_draft(self):
        # R22, asserted from shared config rather than convention.
        assert get_model(ROLE_JUDGE, MEASUREMENT) != get_model(ROLE_PIPELINE, MEASUREMENT)
        assert get_model(ROLE_JUDGE, MEASUREMENT) != get_model(ROLE_BASELINE, MEASUREMENT)


class TestErrors:
    def test_judge_equal_to_pipeline_raises(self):
        bad = ModelRoles(
            pipeline="claude-opus-5",
            baseline="claude-opus-5",
            judge="claude-opus-5",
        )
        with pytest.raises(ConfigError, match="judge"):
            validate_measurement_roles(bad)

    def test_judge_equal_to_baseline_raises(self):
        bad = ModelRoles(
            pipeline="claude-opus-5",
            baseline="claude-fable-5",
            judge="claude-fable-5",
        )
        with pytest.raises(ConfigError, match="judge"):
            validate_measurement_roles(bad)

    def test_valid_measurement_roles_pass(self):
        good = ModelRoles(
            pipeline="claude-opus-5",
            baseline="claude-opus-5",
            judge="claude-fable-5",
        )
        validate_measurement_roles(good)  # must not raise

    def test_unknown_role_raises(self):
        with pytest.raises(ConfigError, match="role"):
            get_model("nonexistent-role", MEASUREMENT)

    def test_unknown_profile_raises(self):
        with pytest.raises(ConfigError, match="profile"):
            get_model(ROLE_PIPELINE, "nonexistent-profile")


class TestApiKey:
    def test_missing_key_names_env_var(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY"):
            get_api_key()

    def test_present_key_returned(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-value")
        assert get_api_key() == "sk-test-value"

    def test_empty_key_treated_as_missing(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "")
        with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY"):
            get_api_key()
