"""Centralized model-role configuration (R30, R22).

All model IDs used anywhere in the system are declared here, keyed by role
(pipeline, baseline, judge) and profile (dev, measurement):

- ``dev``: cheap model for development iteration (or mocks upstream of this).
- ``measurement``: the models used for measured runs. The judge model must
  differ from the draft (pipeline/baseline) model — asserted in code at import
  time (R22), not left to convention.

The Anthropic API key is resolved from the environment only (ANTHROPIC_API_KEY);
it is never stored in this repository.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# Role names
ROLE_PIPELINE = "pipeline"
ROLE_BASELINE = "baseline"
ROLE_JUDGE = "judge"

# Profile names
DEV = "dev"
MEASUREMENT = "measurement"

API_KEY_ENV_VAR = "ANTHROPIC_API_KEY"


class ConfigError(Exception):
    """Raised for invalid model-role configuration or missing credentials."""


@dataclass(frozen=True)
class ModelRoles:
    """Model IDs for each role within one profile."""

    pipeline: str
    baseline: str
    judge: str

    def for_role(self, role: str) -> str:
        try:
            return {
                ROLE_PIPELINE: self.pipeline,
                ROLE_BASELINE: self.baseline,
                ROLE_JUDGE: self.judge,
            }[role]
        except KeyError:
            raise ConfigError(
                f"Unknown model role {role!r}; expected one of: "
                f"{ROLE_PIPELINE!r}, {ROLE_BASELINE!r}, {ROLE_JUDGE!r}"
            ) from None


PROFILES: dict[str, ModelRoles] = {
    # Development: cheap model (or mocks) for iteration; never used for
    # measured results.
    DEV: ModelRoles(
        pipeline="claude-haiku-4-5",
        baseline="claude-haiku-4-5",
        judge="claude-haiku-4-5",
    ),
    # Measurement: models whose outputs are reported. Judge is deliberately a
    # different model from the draft models it grades (R22).
    MEASUREMENT: ModelRoles(
        pipeline="claude-sonnet-5",
        baseline="claude-sonnet-5",
        judge="claude-opus-5",
    ),
}


def validate_measurement_roles(roles: ModelRoles) -> None:
    """Assert the judge model differs from both draft models (R22).

    Raises ConfigError if the judge model equals the pipeline or baseline
    model, which would let the judge grade its own outputs.
    """
    if roles.judge == roles.pipeline:
        raise ConfigError(
            f"Measurement judge model ({roles.judge!r}) must differ from the "
            f"pipeline draft model ({roles.pipeline!r}) so the judge never "
            "grades its own outputs (R22)."
        )
    if roles.judge == roles.baseline:
        raise ConfigError(
            f"Measurement judge model ({roles.judge!r}) must differ from the "
            f"baseline draft model ({roles.baseline!r}) so the judge never "
            "grades its own outputs (R22)."
        )


def get_model(role: str, profile: str) -> str:
    """Return the model ID for ``role`` under ``profile``."""
    try:
        roles = PROFILES[profile]
    except KeyError:
        raise ConfigError(
            f"Unknown profile {profile!r}; expected one of: {sorted(PROFILES)}"
        ) from None
    return roles.for_role(role)


def get_api_key() -> str:
    """Return the Anthropic API key from the environment.

    Raises ConfigError naming the environment variable when it is absent or
    empty, rather than surfacing a bare stack trace deep in a client call.
    """
    key = os.environ.get(API_KEY_ENV_VAR, "")
    if not key:
        raise ConfigError(
            f"No Anthropic API key found: set the {API_KEY_ENV_VAR} environment "
            "variable (e.g. `set ANTHROPIC_API_KEY=sk-...` on Windows or "
            "`export ANTHROPIC_API_KEY=sk-...` on POSIX shells)."
        )
    return key


# Load-time assertion (R22): fail at import if the measurement profile is
# misconfigured, so no measured run can start with judge == draft model.
validate_measurement_roles(PROFILES[MEASUREMENT])
