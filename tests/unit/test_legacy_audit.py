"""Schema and deletion-policy checks for the committed legacy audit."""

from __future__ import annotations

import json
from pathlib import Path

AUDIT = Path(__file__).parents[2] / "docs" / "legacy_test_audit.jsonl"
REQUIRED = {
    "schema_version",
    "baseline_commit",
    "legacy_nodeid",
    "source",
    "machine",
    "scenario",
    "error_model",
    "noise_model",
    "truth_source",
    "asserted_planes",
    "physical_rows",
    "assertions",
    "canonical_replacement",
    "same_input_comparison",
    "status",
    "deletion_basis",
    "decision_reason",
}


def _rows() -> list[dict]:
    return [json.loads(line) for line in AUDIT.read_text(encoding="utf-8").splitlines()]


def test_legacy_audit_has_one_valid_row_per_baseline_node() -> None:
    rows = _rows()
    assert len(rows) == 180
    assert len({row["legacy_nodeid"] for row in rows}) == 180
    assert {row["baseline_commit"] for row in rows} == {"a2ee8ed"}
    for row in rows:
        assert row.keys() >= REQUIRED
        assert row["schema_version"] == 1
        assert row["status"] in {"retained", "eligible_to_delete", "deleted"}
        assert row["assertions"]


def test_every_deleted_legacy_node_has_a_valid_basis() -> None:
    for row in _rows():
        if row["status"] != "deleted":
            continue
        comparison = row["same_input_comparison"]
        if row["deletion_basis"] == "invalid_test_oracle":
            assert comparison["input_match"] == "invalid_test_oracle"
            assert comparison["legacy"]["failure_classification"] == "untrusted_xsuite_twiss"
            continue
        assert row["canonical_replacement"]
        assert comparison["input_match"] == "exact"
        assert comparison["legacy"]["outcome"] == "passed"
        assert comparison["canonical"]["outcome"] == "passed"
        assert comparison["both_planes"] is True
        assert comparison["finite_rows"] is True
        assert comparison["missing_rows"] is True
        assert comparison["limit_relation"] in {"equal", "stricter"}
