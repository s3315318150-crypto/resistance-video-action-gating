#!/usr/bin/env python3
"""Seven-stage screenshot guard with bounded current-run frame requests."""

from __future__ import annotations

import sys
from pathlib import Path

import qwen_experiment_action_hierarchical_v1 as engine
import qwen_hierarchical_v1_contract as contract
import qwen_hierarchical_v2_screenshot_guard_reduce as reduce_engine
import qwen_hierarchical_v2_segment_frame_agent as frame_agent


ROOT = Path(__file__).resolve().parent.parent
STAGE_SCHEMA_ID = "resistance_7stage_no_battery_v2"
ALGORITHM_ID = "qwen_experiment_action_hierarchical_v2_screenshot_guard_agent"
ALGORITHM_SCHEMA_VERSION = "qwen_experiment_action_hierarchical_v2_screenshot_guard_agent.v1"
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
_ORIGINAL_RUN_MAP = engine._run_map
_ORIGINAL_RUN_REDUCE = engine._run_reduce


def bind_screenshot_guard_agent() -> None:
    contract.STAGE_SCHEMA_ID = STAGE_SCHEMA_ID
    contract.STAGES = STAGES
    engine.STAGE_SCHEMA_ID = STAGE_SCHEMA_ID
    engine.ALGORITHM_ID = ALGORITHM_ID
    engine.ALGORITHM_SCHEMA_VERSION = ALGORITHM_SCHEMA_VERSION
    engine.DEFAULT_SCHEMA = DEFAULT_SCHEMA
    engine.DEFAULT_OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT
    engine.salvage_reduce_response = reduce_engine.salvage_reduce_response_with_screenshot_guard
    engine.select_events = reduce_engine.select_events_with_screenshot_guard
    engine._run_map = _ORIGINAL_RUN_MAP
    engine._run_reduce = _ORIGINAL_RUN_REDUCE
    frame_agent.install(engine)


def normalized_argv(argv: list[str] | None = None) -> list[str]:
    values = list(sys.argv[1:] if argv is None else argv)
    if "--reduce-recovery-policy" not in values:
        values.extend(["--reduce-recovery-policy", "local_partial"])
    if "--max-model-edge" not in values:
        values.extend(["--max-model-edge", "1280"])
    if "--jpeg-quality" not in values:
        values.extend(["--jpeg-quality", "90"])
    return values


def main(argv: list[str] | None = None) -> int:
    bind_screenshot_guard_agent()
    return engine.main(normalized_argv(argv))


if __name__ == "__main__":
    raise SystemExit(main())
