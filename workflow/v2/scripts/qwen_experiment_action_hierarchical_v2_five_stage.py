#!/usr/bin/env python3
"""Five-stage hierarchical action segmentation.

This entrypoint reuses the tested Map/Reduce engine while changing only the
stage schema and local state assignment. Measurement and writing actions are
both retained as evidence inside the corresponding recording stage. The
seven-stage v2 entrypoint remains available and is never overwritten.
"""

from __future__ import annotations

import sys
from pathlib import Path

import qwen_experiment_action_hierarchical_v1 as engine
import qwen_hierarchical_v1_contract as contract
import qwen_hierarchical_v1_reduce as reduce_engine


ROOT = Path(__file__).resolve().parent.parent
STAGE_SCHEMA_ID = "resistance_5stage_measurement_recording_v1"
ALGORITHM_ID = "qwen_experiment_action_hierarchical_v2_five_stage"
ALGORITHM_SCHEMA_VERSION = "qwen_experiment_action_hierarchical_v2_five_stage.v1"
DEFAULT_SCHEMA = ROOT / "configs" / "action_schemas" / "resistance_5stage_measurement_recording_v1.json"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / ALGORITHM_ID
STAGES = (
    "circuit_wiring",
    "recording_1",
    "circuit_rewiring",
    "recording_2",
    "material_cleanup",
)


def _merge_five_stage_runs(intervals: list[dict], max_gap_seconds: float = 4.1) -> list[dict]:
    """Merge all evidence belonging to one cycle and expose sub-action types."""
    ordered = sorted(
        intervals,
        key=lambda item: (float(item["start_seconds"]), float(item["end_seconds"]), str(item.get("event_id"))),
    )
    runs: list[dict] = []
    for interval in ordered:
        stage = str(interval["stage"])
        same_cycle = stage in {"recording_1", "recording_2"}
        within_gap = (
            runs
            and float(interval["start_seconds"])
            <= float(runs[-1]["end_seconds"]) + max_gap_seconds
        )
        if runs and runs[-1]["stage"] == stage and (same_cycle or within_gap):
            run = runs[-1]
            run["end_seconds"] = max(float(run["end_seconds"]), float(interval["end_seconds"]))
            if int(interval["end_frame_number"]) >= int(run["end_frame_number"]):
                run["end_frame_number"] = int(interval["end_frame_number"])
                run["end_frame_id"] = interval["end_frame_id"]
            run["event_ids"].append(interval["event_id"])
            run["evidence_items"].append(interval.get("evidence", ""))
            action = interval.get("base_action_type")
            if isinstance(action, str) and action not in run["base_action_types"]:
                run["base_action_types"].append(action)
            run["confidence"] = max(float(run["confidence"]), float(interval.get("confidence") or 0.0))
            run["observed_subintervals"].append(
                {
                    "event_id": interval["event_id"],
                    "action_type": action,
                    "start_seconds": interval["start_seconds"],
                    "end_seconds": interval["end_seconds"],
                    "start_frame_id": interval["start_frame_id"],
                    "end_frame_id": interval["end_frame_id"],
                    "evidence": interval.get("evidence", ""),
                }
            )
            continue
        action = interval.get("base_action_type")
        runs.append(
            {
                "stage": stage,
                "start_seconds": float(interval["start_seconds"]),
                "end_seconds": float(interval["end_seconds"]),
                "start_frame_number": int(interval["start_frame_number"]),
                "end_frame_number": int(interval["end_frame_number"]),
                "start_frame_id": interval["start_frame_id"],
                "end_frame_id": interval["end_frame_id"],
                "event_ids": [interval["event_id"]],
                "evidence_items": [interval.get("evidence", "")],
                "base_action_types": [action] if isinstance(action, str) else [],
                "observed_subintervals": [
                    {
                        "event_id": interval["event_id"],
                        "action_type": action,
                        "start_seconds": interval["start_seconds"],
                        "end_seconds": interval["end_seconds"],
                        "start_frame_id": interval["start_frame_id"],
                        "end_frame_id": interval["end_frame_id"],
                        "evidence": interval.get("evidence", ""),
                    }
                ],
                "confidence": float(interval.get("confidence") or 0.0),
            }
        )
    for index, run in enumerate(runs, start=1):
        run["run_id"] = f"stage_run_{index:03d}"
        run["evidence"] = "；".join(dict.fromkeys(str(item) for item in run["evidence_items"] if item))
        merged = run["stage"] in {"recording_1", "recording_2"}
        semantics = "measurement_and_recording_cycle" if merged else "direct_action_interval"
        run["stage_semantics"] = semantics
        run["cycle_index"] = int(run["stage"].rsplit("_", 1)[1]) if merged else None
        run["measurement_subintervals"] = [
            item for item in run["observed_subintervals"]
            if item.get("action_type") == "measurement_action"
        ]
        run["writing_subintervals"] = [
            item for item in run["observed_subintervals"]
            if item.get("action_type") == "writing_action"
        ]
        run["merged_measurement_recording"] = merged
        run["contains_measurement_evidence"] = "measurement_action" in run["base_action_types"]
        run["contains_writing_evidence"] = "writing_action" in run["base_action_types"]
        # Compatibility aliases for consumers written before the canonical
        # five-stage run contract was introduced.
        run["stage_window_semantics"] = semantics
        run["merged_stage"] = merged
        run["merged_stage_semantics"] = semantics if merged else None
        del run["evidence_items"]
    return runs


def bind_five_stage_identity() -> None:
    contract.STAGE_SCHEMA_ID = STAGE_SCHEMA_ID
    contract.STAGES = STAGES
    engine.STAGE_SCHEMA_ID = STAGE_SCHEMA_ID
    engine.ALGORITHM_ID = ALGORITHM_ID
    engine.ALGORITHM_SCHEMA_VERSION = ALGORITHM_SCHEMA_VERSION
    engine.DEFAULT_SCHEMA = DEFAULT_SCHEMA
    engine.DEFAULT_OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT
    engine.assign_seven_stages = reduce_engine.assign_five_stages
    engine.merge_observed_stage_runs = _merge_five_stage_runs


def normalized_argv(argv: list[str] | None = None) -> list[str]:
    values = list(sys.argv[1:] if argv is None else argv)
    if "--reduce-recovery-policy" not in values:
        values.extend(["--reduce-recovery-policy", "local_partial"])
    return values


def main(argv: list[str] | None = None) -> int:
    bind_five_stage_identity()
    return engine.main(normalized_argv(argv))


if __name__ == "__main__":
    raise SystemExit(main())
