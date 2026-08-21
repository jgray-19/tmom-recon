"""Load and validate committed external magnet-strength fixtures."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
SUPPORTED_STRENGTH_SUFFIXES = (
    ".k0",
    ".k1",
    ".k2",
    ".dk0l",
    ".dk1l",
    ".dk2l",
)


@dataclass(frozen=True)
class ExternalStrengthFixture:
    """A versioned plain-data hand-off from an external fitting workflow."""

    machine: str
    metadata: dict[str, Any]
    strengths: dict[str, float]

    @property
    def fingerprint(self) -> str:
        """Return the canonical SHA-256 fingerprint of the strength mapping."""
        return strength_fingerprint(self.strengths)


def strength_fingerprint(strengths: dict[str, float]) -> str:
    """Fingerprint sorted names and full-precision floating-point values."""
    payload = "\n".join(f"{name}={float(value):.17e}" for name, value in sorted(strengths.items()))
    return hashlib.sha256(payload.encode()).hexdigest()


def load_external_strength_fixture(path: Path) -> ExternalStrengthFixture:
    """Load one fixture and reject incomplete or ambiguous provenance."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"{path}: schema_version must be {SCHEMA_VERSION}, "
            f"got {payload.get('schema_version')!r}"
        )

    machine = payload.get("machine")
    metadata = payload.get("metadata")
    strengths = payload.get("strengths")
    if not isinstance(machine, str) or not machine:
        raise ValueError(f"{path}: machine must be a non-empty string")
    if not isinstance(metadata, dict):
        raise ValueError(f"{path}: metadata must be an object")
    required_metadata = {
        "generated_utc",
        "generator",
        "pymadng_utils_revision",
        "sequence",
        "perturbations",
        "strength_semantics",
        "strength_fingerprint_sha256",
    }
    missing_metadata = required_metadata.difference(metadata)
    if missing_metadata:
        raise ValueError(f"{path}: missing metadata {sorted(missing_metadata)}")
    if not isinstance(strengths, dict) or not strengths:
        raise ValueError(f"{path}: strengths must be a non-empty object")

    parsed: dict[str, float] = {}
    for name, value in strengths.items():
        if not isinstance(name, str) or not name.endswith(SUPPORTED_STRENGTH_SUFFIXES):
            raise ValueError(f"{path}: unsupported strength key {name!r}")
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"{path}: strength {name!r} is not numeric")
        parsed[name] = float(value)
        if not math.isfinite(parsed[name]):
            raise ValueError(f"{path}: strength {name!r} is not finite")

    expected_fingerprint = metadata["strength_fingerprint_sha256"]
    actual_fingerprint = strength_fingerprint(parsed)
    if expected_fingerprint != actual_fingerprint:
        raise ValueError(
            f"{path}: strength fingerprint mismatch; expected "
            f"{expected_fingerprint}, got {actual_fingerprint}"
        )
    return ExternalStrengthFixture(machine, metadata, parsed)


__all__ = [
    "ExternalStrengthFixture",
    "SCHEMA_VERSION",
    "load_external_strength_fixture",
    "strength_fingerprint",
]
