#!/usr/bin/env python3
"""Run the paper-informed seven-stage pipeline with terminal cleanup v2 defaults.

The implementation deliberately reuses the tested hierarchical_v1 engine while
binding a new algorithm/schema identity and output root. Historical v1 runs and
their contract remain reproducible and are never overwritten.
"""

from __future__ import annotations

import sys
from pathlib import Path

import qwen_experiment_action_hierarchical_v1 as engine
import qwen_hierarchical_v1_contract as contract


ROOT = Path(__file__).resolve().parent.parent
STAGE_SCHEMA_ID = "resistance_7stage_no_battery_v2"
ALGORITHM_ID = "qwen_experiment_action_hierarchical_v2"
ALGORITHM_SCHEMA_VERSION = "qwen_experiment_action_hierarchical_v2.v1"
DEFAULT_SCHEMA = ROOT / "configs" / "action_schemas" / "resistance_7stage_no_battery_v2.json"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / ALGORITHM_ID


def bind_v2_identity() -> None:
    """Bind v2 metadata into the shared, tested engine for this process only."""
    contract.STAGE_SCHEMA_ID = STAGE_SCHEMA_ID
    engine.STAGE_SCHEMA_ID = STAGE_SCHEMA_ID
    engine.ALGORITHM_ID = ALGORITHM_ID
    engine.ALGORITHM_SCHEMA_VERSION = ALGORITHM_SCHEMA_VERSION
    engine.DEFAULT_SCHEMA = DEFAULT_SCHEMA
    engine.DEFAULT_OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT


def normalized_argv(argv: list[str] | None = None) -> list[str]:
    values = list(sys.argv[1:] if argv is None else argv)
    if "--reduce-recovery-policy" not in values:
        values.extend(["--reduce-recovery-policy", "local_partial"])
    return values


def main(argv: list[str] | None = None) -> int:
    bind_v2_identity()
    return engine.main(normalized_argv(argv))


if __name__ == "__main__":
    raise SystemExit(main())
