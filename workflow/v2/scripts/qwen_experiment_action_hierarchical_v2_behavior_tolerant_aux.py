#!/usr/bin/env python3
"""Add non-competing auxiliary actions to behavior-tolerant v2."""

from __future__ import annotations

import sys
from pathlib import Path

import qwen_experiment_action_hierarchical_v1 as engine
import qwen_hierarchical_v1_contract as contract
import qwen_experiment_action_hierarchical_v2_behavior_tolerant as behavior
from qwen_hierarchical_v2_behavior_tolerant_aux import (
    BASE_ACTIONS,
    build_map_prompt_auxiliary,
    build_reduce_prompt_auxiliary,
    deduplicate_map_events_auxiliary,
    find_temporal_conflicts_auxiliary,
    normalize_map_events_auxiliary,
    select_events_auxiliary,
    validate_map_response_auxiliary,
)


ROOT = Path(__file__).resolve().parent.parent
STAGE_SCHEMA_ID = "resistance_7stage_no_battery_v2"
ALGORITHM_ID = "qwen_experiment_action_hierarchical_v2_behavior_tolerant_aux"
ALGORITHM_SCHEMA_VERSION = "qwen_experiment_action_hierarchical_v2_behavior_tolerant_aux.v1"
DEFAULT_SCHEMA = ROOT / "configs" / "action_schemas" / "resistance_7stage_no_battery_v2_behavior_tolerant_aux.json"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / ALGORITHM_ID


def bind_behavior_tolerant_aux() -> None:
    behavior.bind_behavior_tolerant()
    contract.BASE_ACTIONS = BASE_ACTIONS
    engine.ALGORITHM_ID = ALGORITHM_ID
    engine.ALGORITHM_SCHEMA_VERSION = ALGORITHM_SCHEMA_VERSION
    engine.DEFAULT_SCHEMA = DEFAULT_SCHEMA
    engine.DEFAULT_OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT
    engine.validate_map_response = validate_map_response_auxiliary
    engine.normalize_map_events = normalize_map_events_auxiliary
    engine.build_map_prompt = build_map_prompt_auxiliary
    engine.build_reduce_prompt = build_reduce_prompt_auxiliary
    engine.deduplicate_map_events = deduplicate_map_events_auxiliary
    engine.find_temporal_conflicts = find_temporal_conflicts_auxiliary
    engine.select_events = select_events_auxiliary


def restore_original_bindings() -> None:
    contract.BASE_ACTIONS = (
        "wiring_action",
        "measurement_action",
        "writing_action",
        "cleanup_action",
        "uncertain",
    )
    behavior.restore_original_bindings()
    # behavior only owns its v2 patches; reset auxiliary hooks explicitly.
    import qwen_hierarchical_v1_prompts as prompts
    import qwen_hierarchical_v1_reduce as reduce

    engine.validate_map_response = contract.validate_map_response
    engine.normalize_map_events = contract.normalize_map_events
    engine.build_map_prompt = prompts.build_map_prompt
    engine.build_reduce_prompt = prompts.build_reduce_prompt
    engine.deduplicate_map_events = reduce.deduplicate_map_events
    engine.find_temporal_conflicts = reduce.find_temporal_conflicts


def normalized_argv(argv: list[str] | None = None) -> list[str]:
    values = list(sys.argv[1:] if argv is None else argv)
    if "--reduce-recovery-policy" not in values:
        values.extend(["--reduce-recovery-policy", "local_partial"])
    return values


def main(argv: list[str] | None = None) -> int:
    bind_behavior_tolerant_aux()
    try:
        return engine.main(normalized_argv(argv))
    finally:
        restore_original_bindings()


if __name__ == "__main__":
    raise SystemExit(main())
