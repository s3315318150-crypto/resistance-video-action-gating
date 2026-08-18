#!/usr/bin/env python3
"""Formal seven-stage Temporal Guard screenshot-compatible entrypoint."""

from __future__ import annotations

import sys
from pathlib import Path

import qwen_experiment_action_hierarchical_v1 as engine
import qwen_hierarchical_v1_contract as contract
import qwen_hierarchical_v2_screenshot_guard_reduce as reduce_engine


ROOT = Path(__file__).resolve().parent.parent
STAGE_SCHEMA_ID = "resistance_7stage_no_battery_v2"
ALGORITHM_ID = "qwen_experiment_action_hierarchical_v2_screenshot_guard"
ALGORITHM_SCHEMA_VERSION = "qwen_experiment_action_hierarchical_v2_screenshot_guard.v1"
DEFAULT_SCHEMA = ROOT / "configs" / "action_schemas" / "resistance_7stage_no_battery_v2.json"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / ALGORITHM_ID
STAGES = (
    "circuit_wiring",
    "measurement_1",
    "recording_1",
    "circuit_rewiring",
    "measurement_2",
    "recording_2",
    "material_cleanup",
)


def bind_screenshot_guard() -> None:
    contract.STAGE_SCHEMA_ID = STAGE_SCHEMA_ID
    contract.STAGES = STAGES
    engine.STAGE_SCHEMA_ID = STAGE_SCHEMA_ID
    engine.ALGORITHM_ID = ALGORITHM_ID
    engine.ALGORITHM_SCHEMA_VERSION = ALGORITHM_SCHEMA_VERSION
    engine.DEFAULT_SCHEMA = DEFAULT_SCHEMA
    engine.DEFAULT_OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT
    engine.salvage_reduce_response = reduce_engine.salvage_reduce_response_with_screenshot_guard
    engine.select_events = reduce_engine.select_events_with_screenshot_guard


def normalized_argv(argv: list[str] | None = None) -> list[str]:
    values = list(sys.argv[1:] if argv is None else argv)
    if "--reduce-recovery-policy" not in values:
        values.extend(["--reduce-recovery-policy", "local_partial"])
    return values


def main(argv: list[str] | None = None) -> int:
    bind_screenshot_guard()
    return engine.main(normalized_argv(argv))


if __name__ == "__main__":
    raise SystemExit(main())
