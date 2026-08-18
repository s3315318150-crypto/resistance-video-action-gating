#!/usr/bin/env python3
"""Run v2 Temporal Guard with a behavior-tolerant stage decoder."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import qwen_experiment_action_hierarchical_v1 as engine
import qwen_hierarchical_v1_contract as contract
from qwen_hierarchical_v2_behavior_tolerant_reduce import assign_seven_stages_behavior_tolerant
from qwen_hierarchical_v2_temporal_guard_reduce import (
    salvage_reduce_response_with_temporal_guard,
    select_events_with_temporal_guard,
)


ROOT = Path(__file__).resolve().parent.parent
STAGE_SCHEMA_ID = "resistance_7stage_no_battery_v2"
ALGORITHM_ID = "qwen_experiment_action_hierarchical_v2_behavior_tolerant"
ALGORITHM_SCHEMA_VERSION = "qwen_experiment_action_hierarchical_v2_behavior_tolerant.v1"
DEFAULT_SCHEMA = ROOT / "configs" / "action_schemas" / "resistance_7stage_no_battery_v2.json"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / ALGORITHM_ID

_ORIGINAL_STAGE_SCHEMA_ID = contract.STAGE_SCHEMA_ID
_ORIGINAL_ENGINE_VALUES = {
    "STAGE_SCHEMA_ID": engine.STAGE_SCHEMA_ID,
    "ALGORITHM_ID": engine.ALGORITHM_ID,
    "ALGORITHM_SCHEMA_VERSION": engine.ALGORITHM_SCHEMA_VERSION,
    "DEFAULT_SCHEMA": engine.DEFAULT_SCHEMA,
    "DEFAULT_OUTPUT_ROOT": engine.DEFAULT_OUTPUT_ROOT,
    "salvage_reduce_response": engine.salvage_reduce_response,
    "select_events": engine.select_events,
    "assign_seven_stages": engine.assign_seven_stages,
    "analyze_prepared_video": engine.analyze_prepared_video,
}


def analyze_prepared_video_behavior_tolerant(
    prepared: dict[str, Any], client: Any, schema: dict[str, Any], args: Any
) -> dict[str, Any]:
    result = _ORIGINAL_ENGINE_VALUES["analyze_prepared_video"](prepared, client, schema, args)
    terminal_id = result.get("reduce", {}).get("selection", {}).get("terminal_cleanup_event_id")
    state = assign_seven_stages_behavior_tolerant(
        list(result.get("reduce", {}).get("accepted_events", [])),
        terminal_id if isinstance(terminal_id, str) else None,
    )
    result["stage_decoder"] = state["decoder"]
    result["auxiliary_events"] = state["auxiliary_events"]
    result["downstream_hints"] = {
        "batched_recording": bool(state["decoder"].get("batched_recording")),
    }
    contract.write_json_atomic(prepared["video_dir"] / "result.json", result)
    return result


def bind_behavior_tolerant() -> None:
    contract.STAGE_SCHEMA_ID = STAGE_SCHEMA_ID
    engine.STAGE_SCHEMA_ID = STAGE_SCHEMA_ID
    engine.ALGORITHM_ID = ALGORITHM_ID
    engine.ALGORITHM_SCHEMA_VERSION = ALGORITHM_SCHEMA_VERSION
    engine.DEFAULT_SCHEMA = DEFAULT_SCHEMA
    engine.DEFAULT_OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT
    engine.salvage_reduce_response = salvage_reduce_response_with_temporal_guard
    engine.select_events = select_events_with_temporal_guard
    engine.assign_seven_stages = assign_seven_stages_behavior_tolerant
    engine.analyze_prepared_video = analyze_prepared_video_behavior_tolerant


def restore_original_bindings() -> None:
    contract.STAGE_SCHEMA_ID = _ORIGINAL_STAGE_SCHEMA_ID
    for name, value in _ORIGINAL_ENGINE_VALUES.items():
        setattr(engine, name, value)


def normalized_argv(argv: list[str] | None = None) -> list[str]:
    values = list(sys.argv[1:] if argv is None else argv)
    if "--reduce-recovery-policy" not in values:
        values.extend(["--reduce-recovery-policy", "local_partial"])
    return values


def main(argv: list[str] | None = None) -> int:
    bind_behavior_tolerant()
    try:
        return engine.main(normalized_argv(argv))
    finally:
        restore_original_bindings()


if __name__ == "__main__":
    raise SystemExit(main())
