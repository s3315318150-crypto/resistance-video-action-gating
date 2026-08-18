#!/usr/bin/env python3
"""Add bounded supplemental activity frames to the behavior-tolerant pipeline."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import qwen_experiment_action_hierarchical_v1 as engine
import qwen_hierarchical_v1_contract as contract
import qwen_experiment_action_hierarchical_v2_behavior_tolerant_boundary as boundary
from qwen_hierarchical_v2_behavior_tolerant_sampling import (
    scan_activity_compensated,
    select_supplemental_timestamps,
    timestamps_to_frame_numbers,
)


ROOT = Path(__file__).resolve().parent.parent
ALGORITHM_ID = "qwen_experiment_action_hierarchical_v2_behavior_tolerant_adaptive"
ALGORITHM_SCHEMA_VERSION = "qwen_experiment_action_hierarchical_v2_behavior_tolerant_adaptive.v1"
DEFAULT_SCHEMA = ROOT / "configs" / "action_schemas" / "resistance_7stage_no_battery_v2_behavior_tolerant_aux.json"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / ALGORITHM_ID

_ORIGINAL_PREPARE_VIDEO = engine.prepare_video


def prepare_video_adaptive(provenance: dict[str, Any], video_dir: Path, args: Any) -> dict[str, Any]:
    prepared = _ORIGINAL_PREPARE_VIDEO(provenance, video_dir, args)
    source = Path(str(prepared["manifest"]["source_video"]))
    activity = scan_activity_compensated(source, float(prepared["fixed_start"]), float(prepared["fixed_end"]))
    diagnostics: list[dict[str, Any]] = []
    for window in prepared["prepared_windows"]:
        base_frames = list(window["frames"])
        base_timestamps = [float(frame["timestamp_seconds"]) for frame in base_frames]
        start, end = (float(value) for value in window["window_seconds"])
        extra_timestamps, diagnostic = select_supplemental_timestamps(
            start, end, base_timestamps, activity
        )
        extra_numbers = timestamps_to_frame_numbers(
            extra_timestamps, float(prepared["fps"]), int(prepared["frame_count"])
        )
        engine._extract_source_frames(
            prepared["manifest"],
            extra_numbers,
            prepared["frames_dir"],
            args.max_model_edge,
            prepared["frame_registry"],
        )
        all_numbers = sorted({int(frame["frame_number"]) for frame in base_frames} | set(extra_numbers))
        window["frames"] = [prepared["frame_registry"][number] for number in all_numbers]
        diagnostic = {
            "window_id": str(window["window_id"]),
            **diagnostic,
            "base_frames_preserved": all(
                int(frame["frame_number"]) in all_numbers for frame in base_frames
            ),
            "selected_frame_count": len(all_numbers),
        }
        diagnostics.append(diagnostic)
        input_path = Path(str(window["input_path"]))
        input_record = contract.read_json(input_path)
        input_record["sampling"] = {
            **dict(input_record.get("sampling", {})),
            "dynamic_supplement": diagnostic,
        }
        input_record["input_frames"] = window["frames"]
        contract.write_json_atomic(input_path, input_record)
        prompt = engine.build_map_prompt(str(prepared["video_id"]), window, window["frames"])
        engine._write_text(Path(str(window["prompt_path"])), prompt)
    prepared["_dynamic_supplement"] = {
        "enabled": True,
        "full_interval_scan_count": 1,
        "activity_sample_count": len(activity),
        "windows": diagnostics,
        "all_uniform_base_frames_preserved": all(item["base_frames_preserved"] for item in diagnostics),
        "maximum_extra_fraction_per_window": 0.25,
    }
    prepared["source_record"]["dynamic_supplement"] = prepared["_dynamic_supplement"]
    prepared["source_record"]["window_frame_reference_count"] = sum(
        len(window["frames"]) for window in prepared["prepared_windows"]
    )
    prepared["source_record"]["unique_source_frame_count"] = len(prepared["frame_registry"])
    prepared["source_record"]["overlap_reference_savings"] = (
        prepared["source_record"]["window_frame_reference_count"] - len(prepared["frame_registry"])
    )
    contract.write_json_atomic(prepared["video_dir"] / "source.json", prepared["source_record"])
    return prepared


def analyze_prepared_video_adaptive(
    prepared: dict[str, Any], client: Any, schema: dict[str, Any], args: Any
) -> dict[str, Any]:
    result = boundary.analyze_prepared_video_boundary(prepared, client, schema, args)
    result["sampling"]["dynamic_supplement"] = prepared.get("_dynamic_supplement", {"enabled": False})
    contract.write_json_atomic(prepared["video_dir"] / "result.json", result)
    return result


def bind_behavior_tolerant_adaptive() -> None:
    boundary.bind_behavior_tolerant_boundary()
    engine.ALGORITHM_ID = ALGORITHM_ID
    engine.ALGORITHM_SCHEMA_VERSION = ALGORITHM_SCHEMA_VERSION
    engine.DEFAULT_SCHEMA = DEFAULT_SCHEMA
    engine.DEFAULT_OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT
    engine.prepare_video = prepare_video_adaptive
    engine.analyze_prepared_video = analyze_prepared_video_adaptive


def restore_original_bindings() -> None:
    engine.prepare_video = _ORIGINAL_PREPARE_VIDEO
    boundary.restore_original_bindings()


def normalized_argv(argv: list[str] | None = None) -> list[str]:
    values = list(sys.argv[1:] if argv is None else argv)
    if "--reduce-recovery-policy" not in values:
        values.extend(["--reduce-recovery-policy", "local_partial"])
    return values


def main(argv: list[str] | None = None) -> int:
    bind_behavior_tolerant_adaptive()
    try:
        return engine.main(normalized_argv(argv))
    finally:
        restore_original_bindings()


if __name__ == "__main__":
    raise SystemExit(main())
