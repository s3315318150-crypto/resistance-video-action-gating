#!/usr/bin/env python3
"""Add local Map-window boundary review to behavior-tolerant auxiliary v2."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import qwen_experiment_action_hierarchical_v1 as engine
import qwen_hierarchical_v1_contract as contract
import qwen_experiment_action_hierarchical_v2_behavior_tolerant as behavior
import qwen_experiment_action_hierarchical_v2_behavior_tolerant_aux as auxiliary
from qwen_hierarchical_v2_behavior_tolerant_boundary import run_boundary_bridge_review


ROOT = Path(__file__).resolve().parent.parent
ALGORITHM_ID = "qwen_experiment_action_hierarchical_v2_behavior_tolerant_boundary"
ALGORITHM_SCHEMA_VERSION = "qwen_experiment_action_hierarchical_v2_behavior_tolerant_boundary.v1"
DEFAULT_SCHEMA = ROOT / "configs" / "action_schemas" / "resistance_7stage_no_battery_v2_behavior_tolerant_aux.json"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / ALGORITHM_ID

_ORIGINAL_RUN_MAP = engine._run_map


def run_map_with_boundary_review(
    prepared: dict[str, Any], client: Any, args: Any
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    events, window_results, review = _ORIGINAL_RUN_MAP(prepared, client, args)
    events, _bridge_reviews, bridge_review = run_boundary_bridge_review(
        prepared, events, window_results, client, args
    )
    return events, window_results, sorted(set(review + bridge_review))


def analyze_prepared_video_boundary(
    prepared: dict[str, Any], client: Any, schema: dict[str, Any], args: Any
) -> dict[str, Any]:
    result = behavior.analyze_prepared_video_behavior_tolerant(prepared, client, schema, args)
    reviews = list(prepared.get("_boundary_bridge_reviews", []))
    result["boundary_bridge_reviews"] = reviews
    result["map"]["boundary_bridge_review"] = {
        "candidate_count": len(reviews),
        "accepted_extension_count": sum(1 for item in reviews if item.get("accepted_extension")),
        "original_events_preserved": True,
    }
    contract.write_json_atomic(prepared["video_dir"] / "result.json", result)
    return result


def bind_behavior_tolerant_boundary() -> None:
    auxiliary.bind_behavior_tolerant_aux()
    engine.ALGORITHM_ID = ALGORITHM_ID
    engine.ALGORITHM_SCHEMA_VERSION = ALGORITHM_SCHEMA_VERSION
    engine.DEFAULT_SCHEMA = DEFAULT_SCHEMA
    engine.DEFAULT_OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT
    engine._run_map = run_map_with_boundary_review
    engine.analyze_prepared_video = analyze_prepared_video_boundary


def restore_original_bindings() -> None:
    engine._run_map = _ORIGINAL_RUN_MAP
    auxiliary.restore_original_bindings()


def normalized_argv(argv: list[str] | None = None) -> list[str]:
    values = list(sys.argv[1:] if argv is None else argv)
    if "--reduce-recovery-policy" not in values:
        values.extend(["--reduce-recovery-policy", "local_partial"])
    return values


def main(argv: list[str] | None = None) -> int:
    bind_behavior_tolerant_boundary()
    try:
        return engine.main(normalized_argv(argv))
    finally:
        restore_original_bindings()


if __name__ == "__main__":
    raise SystemExit(main())
