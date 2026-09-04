from __future__ import annotations

import importlib
import pathlib
import sys

import pytest

_SCRIPT = pathlib.Path(__file__).with_name("audit_execution_horizon_aggregate_risk.py")
sys.path.insert(0, str(_SCRIPT.parent))
audit = importlib.import_module("audit_execution_horizon_aggregate_risk")


def test_audit_split_role_supports_four_way_and_legacy_names() -> None:
    assert audit._audit_split_role("dev_audit") == "development_audit"  # noqa: SLF001
    assert audit._audit_split_role("validation") == "validation"  # noqa: SLF001
    with pytest.raises(ValueError, match="legacy 'validation' or four-way 'dev_audit'"):
        audit._audit_split_role("calibration")  # noqa: SLF001
