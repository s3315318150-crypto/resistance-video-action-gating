#!/usr/bin/env python3
"""Run experimental hierarchical v3 without changing the stable v1/v2 engines."""

from __future__ import annotations

import json
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import qwen_experiment_action_hierarchical_v1 as engine
import qwen_hierarchical_v1_contract as contract
import qwen_hierarchical_v1_reduce as base_reduce
import qwen_hierarchical_v3_contract as v3_contract
from qwen_hierarchical_v3_prompts import (
    build_cleanup_confirmation_prompt,
    build_endpoint_cleanup_binary_prompt,
    build_map_prompt,
    build_measurement_binary_prompt,
    build_reduce_prompt,
    build_reverse_boundary_prompt,
)
from qwen_hierarchical_v3_reduce import assign_seven_stages_v3, anomalous_events_from_assigned
from qwen_hierarchical_v3_sampling import adaptive_frame_numbers, motion_consistency_score, scan_activity


ROOT = Path(__file__).resolve().parent.parent
STAGE_SCHEMA_ID = "resistance_7stage_no_battery_v3"
ALGORITHM_ID = "qwen_experiment_action_hierarchical_v3"
ALGORITHM_SCHEMA_VERSION = "qwen_experiment_action_hierarchical_v3.v1"
DEFAULT_SCHEMA = ROOT / "configs" / "action_schemas" / "resistance_7stage_no_battery_v3.json"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / ALGORITHM_ID

COMPLETED_CLEANUP_PATTERNS = (
    re.compile(r"(拆|拔|收|整理|清)\S{0,4}(完|毕|好|空|干净|彻底|结束)"),
    re.compile(r"(放|移|推)\S{0,6}(桌子|桌面)\S{0,4}(左上)"),
    re.compile(r"(实验|操作)\S{0,4}(结束|完成)"),
)

_ORIGINALS = {
    "prepare_video": engine.prepare_video,
    "run_map": engine._run_map,
    "run_reduce": engine._run_reduce,
    "refine_boundaries": engine._refine_boundaries,
    "assign_seven_stages": engine.assign_seven_stages,
    "analyze_prepared_video": engine.analyze_prepared_video,
    "validate_map_response": engine.validate_map_response,
    "normalize_map_events": engine.normalize_map_events,
    "build_map_prompt": engine.build_map_prompt,
    "build_reduce_prompt": engine.build_reduce_prompt,
    "deduplicate_map_events": engine.deduplicate_map_events,
    "find_temporal_conflicts": engine.find_temporal_conflicts,
    "select_events": engine.select_events,
}


def _sampling_summary(diagnostic: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in diagnostic.items() if key != "activity_samples"}


def prepare_video_v3(provenance: dict[str, Any], video_dir: Path, args: Any) -> dict[str, Any]:
    """Replace uniform Map frames with percentile-normalized bucketed TCS."""
    prepared = _ORIGINALS["prepare_video"](provenance, video_dir, args)
    uniform_baseline_count = len(prepared["frame_registry"])
    source = Path(str(prepared["manifest"]["source_video"]))
    full_activity = scan_activity(source, float(prepared["fixed_start"]), float(prepared["fixed_end"]))
    activity_by_window: dict[str, list[dict[str, float]]] = {}
    diagnostics: list[dict[str, Any]] = []
    for window in prepared["prepared_windows"]:
        window_id = str(window["window_id"])
        old_frames = list(window["frames"])
        budget = len(old_frames)
        if budget < 2:
            diagnostic: dict[str, Any] = {
                "strategy": "short_window_original_frame_preserved",
                "frame_budget": budget,
                "selected_frame_count": budget,
                "activity_samples": [],
            }
            frame_numbers = [int(item["frame_number"]) for item in old_frames]
        else:
            start, end = (float(value) for value in window["window_seconds"])
            window_activity = [
                item
                for item in full_activity
                if start - 1e-9 <= float(item["timestamp_seconds"]) <= end + 1e-9
            ]
            frame_numbers, diagnostic = adaptive_frame_numbers(
                source,
                start,
                end,
                float(prepared["fps"]),
                int(prepared["frame_count"]),
                budget,
                activity_samples=window_activity,
            )
            engine._extract_source_frames(
                prepared["manifest"],
                frame_numbers,
                prepared["frames_dir"],
                args.max_model_edge,
                prepared["frame_registry"],
            )
        frames = [prepared["frame_registry"][number] for number in frame_numbers]
        window["frames"] = frames
        activity_by_window[window_id] = list(diagnostic.get("activity_samples", []))
        summary = {"window_id": window_id, **_sampling_summary(diagnostic)}
        window["tcs_sampling"] = summary
        diagnostics.append(summary)
        input_path = Path(str(window["input_path"]))
        input_record = contract.read_json(input_path)
        input_record["sampling"] = {
            **dict(input_record.get("sampling", {})),
            **summary,
            "uniform_sampling_replaced": True,
        }
        input_record["input_frames"] = frames
        contract.write_json_atomic(input_path, input_record)
        prompt = build_map_prompt(str(prepared["video_id"]), window, frames)
        engine._write_text(Path(str(window["prompt_path"])), prompt)
    prepared["_v3_activity_by_window"] = activity_by_window
    selected_frame_numbers = {
        int(frame["frame_number"])
        for window in prepared["prepared_windows"]
        for frame in window["frames"]
    }
    discarded_uniform_frames = 0
    for frame_number in list(prepared["frame_registry"]):
        if frame_number in selected_frame_numbers:
            continue
        frame_path = Path(str(prepared["frame_registry"][frame_number]["path"]))
        frame_path.unlink(missing_ok=True)
        prepared["frame_registry"].pop(frame_number)
        discarded_uniform_frames += 1
    prepared["source_record"]["tcs_sampling"] = {
        "strategy": "percentile_bucketed_tcs_with_low_motion_reserve",
        "full_locked_interval_activity_scan_count": 1,
        "full_locked_interval_activity_sample_count": len(full_activity),
        "same_per_window_image_budget_as_uniform_sampling": True,
        "windows": diagnostics,
        "uniform_baseline_extracted_frame_count": uniform_baseline_count,
        "discarded_uniform_baseline_frame_count": discarded_uniform_frames,
    }
    prepared["source_record"]["window_frame_reference_count"] = sum(
        len(item["frames"]) for item in prepared["prepared_windows"]
    )
    prepared["source_record"]["model_selected_unique_frame_count"] = len(
        selected_frame_numbers
    )
    prepared["source_record"]["unique_source_frame_count"] = len(prepared["frame_registry"])
    prepared["source_record"]["overlap_reference_savings"] = (
        prepared["source_record"]["window_frame_reference_count"] - len(selected_frame_numbers)
    )
    contract.write_json_atomic(prepared["video_dir"] / "source.json", prepared["source_record"])
    return prepared


def _nearest_activity(samples: list[dict[str, float]], timestamp: float) -> float:
    if not samples:
        return 0.5
    nearest = min(samples, key=lambda item: abs(float(item["timestamp_seconds"]) - timestamp))
    return min(1.0, max(0.0, float(nearest.get("activity_score", 0.5))))


def _valid_confidence(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and 0.0 <= float(value) <= 1.0
    )


def _measurement_binary_frames(window: dict[str, Any]) -> list[dict[str, Any]]:
    frames = list(window["frames"])
    timestamps = list(
        window.get("tcs_sampling", {}).get("measurement_candidate_timestamps_seconds", [])
    )
    if not timestamps:
        return frames
    selected: dict[str, dict[str, Any]] = {}
    for timestamp in timestamps:
        frame = min(frames, key=lambda item: abs(float(item["timestamp_seconds"]) - float(timestamp)))
        selected[str(frame["image_id"])] = frame
    return sorted(selected.values(), key=lambda item: int(item["frame_number"]))


def _validate_measurement_binary(
    value: Any,
    window_id: str,
    frames: list[dict[str, Any]],
) -> list[str]:
    if not isinstance(value, dict):
        return ["measurement_binary_not_object"]
    errors: list[str] = []
    known = {str(frame["image_id"]): index for index, frame in enumerate(frames)}
    if value.get("window_id") != window_id:
        errors.append("window_id_mismatch")
    observed = value.get("measurement_observed")
    if observed not in {"yes", "no"}:
        errors.append("measurement_observed_invalid")
    observations = value.get("observations")
    if not isinstance(observations, list) or len(observations) > 8:
        errors.append("observations_invalid")
        observations = []
    if observed == "yes" and not observations:
        errors.append("yes_without_observation")
    if observed == "no" and observations:
        errors.append("no_with_observation")
    decision_evidence_ids = value.get("decision_evidence_frame_ids")
    if (
        not isinstance(decision_evidence_ids, list)
        or not decision_evidence_ids
        or any(not isinstance(item, str) or item not in known for item in decision_evidence_ids)
    ):
        errors.append("decision_evidence_frame_ids_invalid")
    if not isinstance(value.get("decision_evidence"), str) or not str(value.get("decision_evidence", "")).strip():
        errors.append("decision_evidence_invalid")
    for index, item in enumerate(observations):
        prefix = f"observation_{index}"
        if not isinstance(item, dict):
            errors.append(prefix + "_not_object")
            continue
        first_id = item.get("first_frame_id")
        last_id = item.get("last_frame_id")
        representative_id = item.get("representative_frame_id")
        first_valid = isinstance(first_id, str) and first_id in known
        last_valid = isinstance(last_id, str) and last_id in known
        representative_valid = isinstance(representative_id, str) and representative_id in known
        if not first_valid:
            errors.append(prefix + "_first_frame_invalid")
        if not last_valid:
            errors.append(prefix + "_last_frame_invalid")
        if not representative_valid:
            errors.append(prefix + "_representative_frame_invalid")
        if first_valid and last_valid and known[str(first_id)] > known[str(last_id)]:
            errors.append(prefix + "_frame_order_invalid")
        if (
            first_valid
            and last_valid
            and representative_valid
            and not known[str(first_id)] <= known[str(representative_id)] <= known[str(last_id)]
        ):
            errors.append(prefix + "_representative_outside_interval")
        if not isinstance(item.get("evidence"), str) or not str(item.get("evidence", "")).strip():
            errors.append(prefix + "_evidence_invalid")
        if not _valid_confidence(item.get("confidence")):
            errors.append(prefix + "_confidence_invalid")
    if not _valid_confidence(value.get("confidence")):
        errors.append("confidence_invalid")
    return sorted(set(errors))


def _measurement_binary_events(
    value: dict[str, Any],
    window_id: str,
    frames: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if value.get("measurement_observed") != "yes":
        return []
    by_id = {str(frame["image_id"]): frame for frame in frames}
    events: list[dict[str, Any]] = []
    for index, observation in enumerate(value.get("observations", []), start=1):
        first = by_id[str(observation["first_frame_id"])]
        last = by_id[str(observation["last_frame_id"])]
        representative = by_id[str(observation["representative_frame_id"])]
        events.append(
            {
                "source_event_id": f"{window_id}_measurement_binary_e{index:02d}",
                "window_id": window_id,
                "action_type": "measurement_action",
                "first_frame_id": first["image_id"],
                "last_frame_id": last["image_id"],
                "representative_frame_id": representative["image_id"],
                "first_frame_number": int(first["frame_number"]),
                "last_frame_number": int(last["frame_number"]),
                "representative_frame_number": int(representative["frame_number"]),
                "first_seconds": float(first["timestamp_seconds"]),
                "last_seconds": float(last["timestamp_seconds"]),
                "representative_seconds": float(representative["timestamp_seconds"]),
                "evidence": str(observation["evidence"]),
                "confidence": float(observation["confidence"]),
                "independent_binary_confirmation": "measurement",
            }
        )
    return events


def _run_measurement_binary(
    prepared: dict[str, Any],
    window: dict[str, Any],
    client: Any,
    args: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    window_id = str(window["window_id"])
    frames = _measurement_binary_frames(window)
    prompt = build_measurement_binary_prompt(str(prepared["video_id"]), window, frames)
    output_dir = prepared["video_dir"] / "map" / "windows" / window_id / "measurement_binary"
    contract.write_json_atomic(
        output_dir / "input.json",
        {"window_id": window_id, "input_frames": frames, "frame_selection": "5s_anchors_plus_low_motion_budget"},
    )
    engine._write_text(output_dir / "prompt.txt", prompt)
    attempts: list[dict[str, Any]] = []
    parsed: dict[str, Any] | None = None
    errors: list[str] = []
    for attempt_index in range(args.max_attempts):
        current_prompt = prompt
        if attempt_index:
            current_prompt += "\n\n上次输出格式错误：" + ", ".join(errors) + "。请严格按原 JSON 重答。"
        raw = engine._attempt_qwen(client, current_prompt, frames, args.map_max_tokens)
        candidate = raw.get("parsed_result")
        parsed = candidate if isinstance(candidate, dict) else None
        errors = _validate_measurement_binary(parsed, window_id, frames)
        attempts.append({"attempt_index": attempt_index + 1, "qwen": raw, "validation_errors": errors})
        if not errors:
            break
    events = _measurement_binary_events(parsed, window_id, frames) if parsed is not None and not errors else []
    result = {
        "window_id": window_id,
        "valid": not errors,
        "validation_errors": errors,
        "attempts": attempts,
        "parsed_result": parsed,
        "normalized_events": events,
    }
    contract.write_json_atomic(output_dir / "result.json", result)
    return events, result, ([f"measurement_binary_invalid:{window_id}:{','.join(errors)}"] if errors else [])


def _validate_endpoint_cleanup_binary(value: Any, frames: list[dict[str, Any]]) -> list[str]:
    if not isinstance(value, dict):
        return ["endpoint_cleanup_binary_not_object"]
    errors: list[str] = []
    known = {str(frame["image_id"]): index for index, frame in enumerate(frames)}
    completed = value.get("cleanup_completed")
    if completed not in {"yes", "no"}:
        errors.append("cleanup_completed_invalid")
    if value.get("experiment_activity_continues_afterward") not in {"yes", "no"}:
        errors.append("experiment_activity_continues_afterward_invalid")
    frame_fields = ("first_cleanup_frame_id", "last_cleanup_frame_id", "representative_frame_id")
    ids = [value.get(field) for field in frame_fields]
    if completed == "yes" and any(not isinstance(item, str) or item not in known for item in ids):
        errors.append("cleanup_frame_ids_invalid")
    if completed == "no" and any(item is not None for item in ids):
        errors.append("no_with_cleanup_frame_ids")
    if completed == "yes" and all(isinstance(item, str) and item in known for item in ids):
        first_id, last_id, representative_id = (str(item) for item in ids)
        if known[first_id] > known[last_id]:
            errors.append("cleanup_frame_order_invalid")
        if not known[first_id] <= known[representative_id] <= known[last_id]:
            errors.append("cleanup_representative_outside_interval")
    evidence_ids = value.get("evidence_frame_ids")
    if not isinstance(evidence_ids, list) or not evidence_ids or any(item not in known for item in evidence_ids):
        errors.append("evidence_frame_ids_invalid")
    if not isinstance(value.get("evidence"), str) or not str(value.get("evidence", "")).strip():
        errors.append("evidence_invalid")
    if not _valid_confidence(value.get("confidence")):
        errors.append("confidence_invalid")
    return sorted(set(errors))


def _run_endpoint_cleanup_binary(
    prepared: dict[str, Any],
    client: Any,
    args: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    start = max(float(prepared["fixed_start"]), float(prepared["fixed_end"]) - 45.0)
    numbers = engine._frame_numbers_for_range(
        start,
        float(prepared["fixed_end"]),
        2.0,
        float(prepared["fps"]),
        int(prepared["frame_count"]),
    )
    engine._extract_source_frames(
        prepared["manifest"], numbers, prepared["frames_dir"], args.max_model_edge, prepared["frame_registry"]
    )
    frames = [prepared["frame_registry"][number] for number in numbers]
    prompt = build_endpoint_cleanup_binary_prompt(str(prepared["video_id"]), frames)
    output_dir = prepared["video_dir"] / "map" / "endpoint_cleanup_binary"
    contract.write_json_atomic(
        output_dir / "input.json",
        {"range_seconds": [start, float(prepared["fixed_end"])], "sample_interval_seconds": 2.0, "input_frames": frames},
    )
    engine._write_text(output_dir / "prompt.txt", prompt)
    attempts: list[dict[str, Any]] = []
    parsed: dict[str, Any] | None = None
    errors: list[str] = []
    for attempt_index in range(args.max_attempts):
        current_prompt = prompt
        if attempt_index:
            current_prompt += "\n\n上次输出格式错误：" + ", ".join(errors) + "。请严格按原 JSON 重答。"
        raw = engine._attempt_qwen(client, current_prompt, frames, args.map_max_tokens)
        candidate = raw.get("parsed_result")
        parsed = candidate if isinstance(candidate, dict) else None
        errors = _validate_endpoint_cleanup_binary(parsed, frames)
        attempts.append({"attempt_index": attempt_index + 1, "qwen": raw, "validation_errors": errors})
        if not errors:
            break
    events: list[dict[str, Any]] = []
    if (
        parsed is not None
        and not errors
        and parsed.get("cleanup_completed") == "yes"
        and parsed.get("experiment_activity_continues_afterward") == "no"
    ):
        by_id = {str(frame["image_id"]): frame for frame in frames}
        first = by_id[str(parsed["first_cleanup_frame_id"])]
        last = by_id[str(parsed["last_cleanup_frame_id"])]
        representative = by_id[str(parsed["representative_frame_id"])]
        events.append(
            {
                "source_event_id": "endpoint_cleanup_binary_e01",
                "window_id": "endpoint_cleanup",
                "action_type": "cleanup_action",
                "first_frame_id": first["image_id"],
                "last_frame_id": last["image_id"],
                "representative_frame_id": representative["image_id"],
                "first_frame_number": int(first["frame_number"]),
                "last_frame_number": int(last["frame_number"]),
                "representative_frame_number": int(representative["frame_number"]),
                "first_seconds": float(first["timestamp_seconds"]),
                "last_seconds": float(last["timestamp_seconds"]),
                "representative_seconds": float(representative["timestamp_seconds"]),
                "evidence": str(parsed["evidence"]),
                "confidence": float(parsed["confidence"]),
                "independent_binary_confirmation": "endpoint_cleanup",
            }
        )
    result = {
        "valid": not errors,
        "validation_errors": errors,
        "attempts": attempts,
        "parsed_result": parsed,
        "normalized_events": events,
    }
    contract.write_json_atomic(output_dir / "result.json", result)
    review = [f"endpoint_cleanup_binary_invalid:{','.join(errors)}"] if errors else []
    return events, result, review


def run_map_v3(prepared: dict[str, Any], client: Any, args: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    events, window_results, review = _ORIGINALS["run_map"](prepared, client, args)
    results_by_window = {str(item["window_id"]): item for item in window_results}
    measurement_binary_results: list[dict[str, Any]] = []
    for window in prepared["prepared_windows"]:
        binary_events, binary_result, binary_review = _run_measurement_binary(prepared, window, client, args)
        measurement_binary_results.append(binary_result)
        events.extend(binary_events)
        review.extend(binary_review)
        window_result = results_by_window[str(window["window_id"])]
        window_result["measurement_binary"] = binary_result
        contract.write_json_atomic(
            prepared["video_dir"] / "map" / "windows" / str(window["window_id"]) / "result.json",
            window_result,
        )
    cleanup_events, cleanup_result, cleanup_review = _run_endpoint_cleanup_binary(prepared, client, args)
    events.extend(cleanup_events)
    review.extend(cleanup_review)
    prepared["_v3_measurement_binary_results"] = measurement_binary_results
    prepared["_v3_endpoint_cleanup_binary"] = cleanup_result
    by_window = prepared.get("_v3_activity_by_window", {})
    for event in events:
        normalized_motion = _nearest_activity(
            list(by_window.get(str(event["window_id"]), [])),
            float(event["representative_seconds"]),
        )
        event["cv_motion_diagnostic"] = {
            "normalized_motion": round(normalized_motion, 6),
            "action_consistency_score": round(
                motion_consistency_score(str(event["action_type"]), normalized_motion), 6
            ),
            "role": "diagnostic_only_not_a_decision_threshold",
        }
    return events, window_results, review


def deduplicate_map_events_v3(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Do not merge overlapping auxiliary events that have different subtypes."""
    prepared: list[dict[str, Any]] = []
    for event in events:
        item = dict(event)
        if item.get("action_type") == "auxiliary_action":
            item["action_type"] = f"auxiliary_action::{item.get('auxiliary_subtype', 'unknown_manipulation')}"
        prepared.append(item)
    groups = _ORIGINALS["deduplicate_map_events"](prepared)
    for group in groups:
        action = str(group.get("action_type", ""))
        if action.startswith("auxiliary_action::"):
            group["action_type"] = "auxiliary_action"
            group["auxiliary_subtype"] = action.split("::", 1)[1]
    return groups


def find_temporal_conflicts_v3(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _ORIGINALS["find_temporal_conflicts"](
        [event for event in events if event.get("action_type") != "auxiliary_action"]
    )


def select_events_v3(
    events: list[dict[str, Any]],
    reduce_result: dict[str, Any] | None,
    preserve_equal_confidence: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Resolve main-action conflicts while retaining accepted auxiliary diagnostics."""
    if reduce_result is None:
        return _ORIGINALS["select_events"](events, None, preserve_equal_confidence)
    main_events = [event for event in events if event.get("action_type") != "auxiliary_action"]
    auxiliary_events = [event for event in events if event.get("action_type") == "auxiliary_action"]
    main_ids = {str(event["event_id"]) for event in main_events}
    accepted_ids = {
        event_id for event_id in reduce_result.get("accepted_event_ids", []) if isinstance(event_id, str)
    }
    main_result = {
        **reduce_result,
        "accepted_event_ids": [event_id for event_id in reduce_result.get("accepted_event_ids", []) if event_id in main_ids],
        "rejected_events": [
            item
            for item in reduce_result.get("rejected_events", [])
            if isinstance(item, dict) and item.get("event_id") in main_ids
        ],
        "conflicts": [
            item
            for item in reduce_result.get("conflicts", [])
            if isinstance(item, dict)
            and isinstance(item.get("event_ids"), list)
            and all(event_id in main_ids for event_id in item["event_ids"])
        ],
    }
    selected_main, selection = _ORIGINALS["select_events"](
        main_events,
        main_result,
        preserve_equal_confidence,
    )
    selected_auxiliary = [event for event in auxiliary_events if str(event["event_id"]) in accepted_ids]
    selected = sorted(
        [*selected_main, *selected_auxiliary],
        key=lambda item: (int(item["representative_frame_number"]), int(item["first_frame_number"])),
    )
    selection["rejected_events"] = list(reduce_result.get("rejected_events", []))
    selection["conflicts"] = list(reduce_result.get("conflicts", []))
    selection["accepted_auxiliary_event_ids"] = [str(event["event_id"]) for event in selected_auxiliary]
    return selected, selection


def _cleanup_frame_numbers(prepared: dict[str, Any], event: dict[str, Any]) -> list[int]:
    fps = float(prepared["fps"])
    frame_count = int(prepared["frame_count"])
    timestamps = [
        float(event["first_seconds"]) - 3.0,
        float(event["first_seconds"]),
        float(event["representative_seconds"]),
        float(event["last_seconds"]),
        float(event["last_seconds"]) + 3.0,
        float(event["last_seconds"]) + 6.0,
    ]
    numbers: list[int] = []
    for timestamp in timestamps:
        clipped = min(float(prepared["fixed_end"]), max(float(prepared["fixed_start"]), timestamp))
        number = min(frame_count - 1, max(0, int(round(clipped * fps))))
        if number not in numbers:
            numbers.append(number)
    return sorted(numbers)


def _validate_cleanup_confirmation(value: Any, event_id: str, frames: list[dict[str, Any]]) -> list[str]:
    if not isinstance(value, dict):
        return ["cleanup_confirmation_not_object"]
    errors: list[str] = []
    choices = {"yes", "no", "uncertain"}
    if value.get("event_id") != event_id:
        errors.append("event_id_mismatch")
    for field in (
        "completed_cleanup",
        "multiple_wires_disconnected",
        "instrument_returned_upper_left",
        "seat_change_or_person_change",
        "experiment_activity_continues_afterward",
    ):
        if value.get(field) not in choices:
            errors.append(f"{field}_invalid")
    known = {str(frame["image_id"]) for frame in frames}
    evidence_ids = value.get("evidence_frame_ids")
    if not isinstance(evidence_ids, list) or any(item not in known for item in evidence_ids):
        errors.append("evidence_frame_ids_invalid")
    if not isinstance(value.get("evidence"), str) or not value.get("evidence", "").strip():
        errors.append("evidence_invalid")
    confidence = value.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0.0 <= float(confidence) <= 1.0:
        errors.append("confidence_invalid")
    return sorted(set(errors))


def _restore_after_unconfirmed_cleanup(
    canonical_events: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    reduce_result: dict[str, Any],
    terminal: dict[str, Any],
    confirmation_reason: str,
    args: Any,
) -> list[dict[str, Any]]:
    terminal_id = str(terminal["event_id"])
    ignored_items = list(reduce_result.get("ignored_noise_events", []))
    recovery = reduce_result.setdefault("recovery", {})
    ignored_items.extend(recovery.get("ignored_noise_events", []))
    restored_ids = {
        str(event["event_id"])
        for event in ignored_items
        if isinstance(event, dict) and isinstance(event.get("event_id"), str)
    }
    prior_effective = reduce_result.get("effective_parsed_result")
    if isinstance(prior_effective, dict):
        restored_ids.update(
            str(item["event_id"])
            for item in prior_effective.get("rejected_events", [])
            if isinstance(item, dict)
            and isinstance(item.get("event_id"), str)
            and item.get("reason") == "post_terminal_cleanup"
        )
    restored_ids.discard(terminal_id)
    accepted_ids = {str(event["event_id"]) for event in selected} | restored_ids
    accepted_ids.add(terminal_id)
    ordered_accepted = [str(event["event_id"]) for event in canonical_events if str(event["event_id"]) in accepted_ids]
    prior_rejections = {
        str(item.get("event_id")): item
        for item in (prior_effective.get("rejected_events", []) if isinstance(prior_effective, dict) else [])
        if isinstance(item, dict) and isinstance(item.get("event_id"), str)
    }
    rejected: list[dict[str, Any]] = []
    for event in canonical_events:
        event_id = str(event["event_id"])
        if event_id in accepted_ids:
            continue
        rejected.append(
            prior_rejections.get(
                event_id,
                {
                    "event_id": event_id,
                    "reason": "other",
                    "explanation": "多帧整理复核后保留原有局部隔离决定。",
                },
            )
        )
    restored_effective = {
        "accepted_event_ids": ordered_accepted,
        "rejected_events": rejected,
        "conflicts": list(prior_effective.get("conflicts", [])) if isinstance(prior_effective, dict) else [],
        "terminal_cleanup_event_id": None,
        "confidence": float(prior_effective.get("confidence", 0.0)) if isinstance(prior_effective, dict) else 0.0,
        "uncertainty": confirmation_reason[:160],
        "ignored_noise_events": [],
    }
    restored, selection = select_events_v3(
        canonical_events,
        restored_effective,
        preserve_equal_confidence=args.reduce_recovery_policy == "local_partial",
    )
    selection["needs_review"] = True
    selection["terminal_cleanup_event_id"] = None
    reduce_result["effective_parsed_result"] = restored_effective
    reduce_result["selection"] = selection
    reduce_result["accepted_events"] = restored
    reduce_result["ignored_noise_events"] = []
    recovery["applied"] = True
    recovery["ignored_noise_events"] = []
    repairs = [
        item
        for item in recovery.setdefault("repairs", [])
        if not str(item.get("reason", "")).startswith(base_reduce.HARD_STOP_REASON_PREFIX)
    ]
    repairs.append(
        {
            "reason": "Terminal cleanup demoted after multi-frame visual confirmation",
            "event_id": terminal_id,
            "restored_event_ids": sorted(restored_ids),
        }
    )
    recovery["repairs"] = repairs
    return restored


def _record_confirmed_post_terminal_noise(
    canonical_events: list[dict[str, Any]],
    result: dict[str, Any],
    terminal: dict[str, Any],
) -> None:
    existing = {
        str(event.get("event_id"))
        for event in result.get("ignored_noise_events", [])
        if isinstance(event, dict)
    }
    effective = result.get("effective_parsed_result")
    post_terminal_ids = {
        str(item["event_id"])
        for item in (effective.get("rejected_events", []) if isinstance(effective, dict) else [])
        if isinstance(item, dict)
        and isinstance(item.get("event_id"), str)
        and item.get("reason") == "post_terminal_cleanup"
    }
    additions = [
        {
            **event,
            "label": "ignored_noise_post_experiment",
            "ignored_label": "ignored_noise_post_experiment",
            "terminal_cleanup_event_id": terminal["event_id"],
            "hard_stop_trigger_keywords": ["multiframe_visual_confirmation"],
        }
        for event in canonical_events
        if str(event["event_id"]) in post_terminal_ids and str(event["event_id"]) not in existing
    ]
    if not additions:
        return
    result.setdefault("ignored_noise_events", []).extend(additions)
    result.setdefault("recovery", {}).setdefault("ignored_noise_events", []).extend(additions)


def _promote_endpoint_cleanup_candidate(
    canonical_events: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    result: dict[str, Any],
    args: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    candidates = [
        event
        for event in canonical_events
        if event.get("action_type") == "cleanup_action"
        and (
            event.get("independent_binary_confirmation") == "endpoint_cleanup"
            or "endpoint_cleanup_binary_e01" in event.get("source_event_ids", [])
        )
    ]
    if not candidates:
        return selected, None
    terminal = min(candidates, key=lambda item: int(item["first_frame_number"]))
    terminal_id = str(terminal["event_id"])
    terminal_start = int(terminal["first_frame_number"])
    accepted_ids = {str(event["event_id"]) for event in selected}
    accepted_ids.add(terminal_id)
    prior_effective = result.get("effective_parsed_result")
    prior_rejections = {
        str(item.get("event_id")): item
        for item in (prior_effective.get("rejected_events", []) if isinstance(prior_effective, dict) else [])
        if isinstance(item, dict) and isinstance(item.get("event_id"), str)
    }
    rejected: list[dict[str, Any]] = []
    for event in canonical_events:
        event_id = str(event["event_id"])
        if event_id == terminal_id:
            continue
        if int(event["last_frame_number"]) >= terminal_start:
            accepted_ids.discard(event_id)
            rejected.append(
                {
                    "event_id": event_id,
                    "reason": "post_terminal_cleanup",
                    "explanation": "独立末尾整理二分类确认后，先隔离与整理开始重叠或更晚的事件。",
                }
            )
        elif event_id not in accepted_ids:
            rejected.append(
                prior_rejections.get(
                    event_id,
                    {"event_id": event_id, "reason": "other", "explanation": "保留 Reduce 的既有局部选择。"},
                )
            )
    effective = {
        "accepted_event_ids": [
            str(event["event_id"])
            for event in canonical_events
            if str(event["event_id"]) in accepted_ids
        ],
        "rejected_events": rejected,
        "conflicts": list(prior_effective.get("conflicts", [])) if isinstance(prior_effective, dict) else [],
        "terminal_cleanup_event_id": terminal_id,
        "confidence": float(terminal.get("confidence", 0.0)),
        "uncertainty": "",
        "ignored_noise_events": [],
    }
    promoted, selection = select_events_v3(
        canonical_events,
        effective,
        preserve_equal_confidence=args.reduce_recovery_policy == "local_partial",
    )
    result["effective_parsed_result"] = effective
    result["selection"] = selection
    result["selection"]["terminal_cleanup_event_id"] = terminal_id
    result["accepted_events"] = promoted
    result.setdefault("recovery", {}).setdefault("repairs", []).append(
        {
            "reason": "endpoint_cleanup_binary_promoted_to_terminal_candidate",
            "event_id": terminal_id,
        }
    )
    return promoted, terminal


def run_reduce_v3(
    prepared: dict[str, Any],
    map_events: list[dict[str, Any]],
    client: Any,
    args: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    selected, result, review = _ORIGINALS["run_reduce"](prepared, map_events, client, args)
    terminal_id = result.get("selection", {}).get("terminal_cleanup_event_id")
    canonical_events = deduplicate_map_events_v3(map_events)
    terminal = next((event for event in canonical_events if str(event["event_id"]) == terminal_id), None)
    if not isinstance(terminal_id, str) or terminal is None:
        selected, terminal = _promote_endpoint_cleanup_candidate(
            canonical_events,
            selected,
            result,
            args,
        )
        terminal_id = str(terminal["event_id"]) if terminal is not None else None
    if not isinstance(terminal_id, str) or terminal is None:
        result["cleanup_confirmation"] = {"required": False, "reason": "no_terminal_cleanup_candidate"}
        contract.write_json_atomic(prepared["video_dir"] / "reduce" / "result.json", result)
        return selected, result, review

    frame_numbers = _cleanup_frame_numbers(prepared, terminal)
    engine._extract_source_frames(
        prepared["manifest"],
        frame_numbers,
        prepared["frames_dir"],
        args.max_model_edge,
        prepared["frame_registry"],
    )
    frames = [prepared["frame_registry"][number] for number in frame_numbers]
    prompt = build_cleanup_confirmation_prompt(str(prepared["video_id"]), terminal, frames)
    confirmation_dir = prepared["video_dir"] / "reduce" / "cleanup_confirmation"
    contract.write_json_atomic(
        confirmation_dir / "input.json",
        {"terminal_cleanup_candidate": terminal, "input_frames": frames},
    )
    engine._write_text(confirmation_dir / "prompt.txt", prompt)
    raw = engine._attempt_qwen(client, prompt, frames, args.boundary_max_tokens)
    parsed = raw.get("parsed_result")
    errors = _validate_cleanup_confirmation(parsed, terminal_id, frames)
    evidence = str(terminal.get("evidence", ""))
    semantic_matches = [pattern.pattern for pattern in COMPLETED_CLEANUP_PATTERNS if pattern.search(evidence)]
    confirmed = (
        not errors
        and isinstance(parsed, dict)
        and parsed.get("completed_cleanup") == "yes"
        and parsed.get("experiment_activity_continues_afterward") != "yes"
    )
    confirmation = {
        "required": True,
        "terminal_cleanup_candidate_event_id": terminal_id,
        "valid": not errors,
        "confirmed": confirmed,
        "validation_errors": errors,
        "qwen": raw,
        "semantic_text_matches_diagnostic_only": semantic_matches,
        "decision_rule": "completed_cleanup=yes AND experiment_activity_continues_afterward!=yes",
    }
    contract.write_json_atomic(confirmation_dir / "result.json", confirmation)
    result["cleanup_confirmation"] = confirmation
    if not confirmed:
        observed = parsed.get("completed_cleanup") if isinstance(parsed, dict) else "transport_or_parse_error"
        reason = f"cleanup_confirmation_not_confirmed:{observed}"
        selected = _restore_after_unconfirmed_cleanup(
            canonical_events,
            selected,
            result,
            terminal,
            reason,
            args,
        )
        review.append(reason)
    else:
        _record_confirmed_post_terminal_noise(canonical_events, result, terminal)
    contract.write_json_atomic(prepared["video_dir"] / "reduce" / "result.json", result)
    return selected, result, sorted(set(review))


def _reverse_boundary_observation(raw: dict[str, Any], boundary_id: str, frames: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, list[str]]:
    parsed = raw.get("parsed_result")
    errors = contract.validate_boundary_response(parsed if isinstance(parsed, dict) else None, boundary_id, frames)
    if errors or not isinstance(parsed, dict) or parsed.get("decision") != "observed":
        return None, errors or ["reverse_boundary_uncertain"]
    by_id = {str(frame["image_id"]): frame for frame in frames}
    return {
        "last_from_frame_id": parsed["last_from_frame_id"],
        "first_to_frame_id": parsed["first_to_frame_id"],
        "last_from_seconds": float(by_id[str(parsed["last_from_frame_id"])]["timestamp_seconds"]),
        "first_to_seconds": float(by_id[str(parsed["first_to_frame_id"])]["timestamp_seconds"]),
        "selected_seconds": float(by_id[str(parsed["first_to_frame_id"])]["timestamp_seconds"]),
        "evidence": parsed["evidence"],
        "confidence": float(parsed["confidence"]),
    }, []


def refine_boundaries_v3(
    prepared: dict[str, Any],
    candidates: list[dict[str, Any]],
    client: Any,
    stage_labels: dict[str, str],
    args: Any,
) -> tuple[list[dict[str, Any]], list[str]]:
    refined, review = _ORIGINALS["refine_boundaries"](prepared, candidates, client, stage_labels, args)
    for boundary in refined:
        if boundary.get("from_stage") != "circuit_wiring" or boundary.get("to_stage") != "measurement_1":
            continue
        passes = boundary.get("passes", {})
        selected_pass = passes.get("dense_half_second") or passes.get("one_fps") or {}
        frames = list(selected_pass.get("input_frames", []))
        if len(frames) < 2:
            boundary["dual_confirmation"] = {"required": True, "valid": False, "errors": ["input_frames_missing"]}
            boundary["needs_review"] = True
            review.append(f"critical_boundary_reverse_confirmation_failed:{boundary['boundary_id']}")
            continue
        prompt = build_reverse_boundary_prompt(str(prepared["video_id"]), boundary, frames)
        reverse_dir = prepared["video_dir"] / "boundaries" / str(boundary["boundary_id"]) / "reverse_confirmation"
        contract.write_json_atomic(reverse_dir / "input.json", {"boundary": boundary, "input_frames": frames})
        engine._write_text(reverse_dir / "prompt.txt", prompt)
        raw = engine._attempt_qwen(client, prompt, frames, args.boundary_max_tokens)
        observed, errors = _reverse_boundary_observation(raw, str(boundary["boundary_id"]), frames)
        dual = {"required": True, "valid": observed is not None, "errors": errors, "qwen": raw, "observed": observed}
        if observed is not None:
            standard_seconds = float(boundary["selected_seconds"])
            reverse_seconds = float(observed["selected_seconds"])
            difference = abs(standard_seconds - reverse_seconds)
            dual["difference_seconds"] = difference
            if difference > 3.0:
                interval = sorted([standard_seconds, reverse_seconds])
                boundary["boundary_uncertainty_seconds"] = interval
                boundary["selected_seconds"] = interval[0]
                boundary["needs_review"] = True
                boundary["source"] = str(boundary["source"]) + "+dual_prompt_disagreement"
                review.append(f"critical_boundary_dual_prompt_disagreement:{boundary['boundary_id']}")
        else:
            boundary["needs_review"] = True
            review.append(f"critical_boundary_reverse_confirmation_failed:{boundary['boundary_id']}")
        boundary["dual_confirmation"] = dual
        contract.write_json_atomic(reverse_dir / "result.json", dual)
        contract.write_json_atomic(
            prepared["video_dir"] / "boundaries" / str(boundary["boundary_id"]) / "result.json",
            boundary,
        )
    return refined, sorted(set(review))


def _sample_meter_window(start: float, end: float) -> list[float]:
    inner_start = start + 2.0 if end - start > 4.0 else start
    inner_end = end - 2.0 if end - start > 4.0 else end
    values: list[float] = []
    cursor = inner_start
    while cursor <= inner_end + 1e-9:
        values.append(round(cursor, 3))
        cursor += 3.0
    if not values:
        values = [round((start + end) / 2.0, 3)]
    return values


def build_meter_reading_windows(result: dict[str, Any]) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    for run in result.get("observed_stage_runs", []):
        stage = str(run.get("stage"))
        if stage not in {"measurement_1", "measurement_2"}:
            continue
        start = float(run["start_seconds"])
        end = float(run["end_seconds"])
        windows.append(
            {
                "measurement_event": stage,
                "time_range_seconds": [start, end],
                "suggested_sample_times": _sample_meter_window(start, end),
                "stage_confidence": float(run.get("confidence", 0.0)),
                "source_event_ids": list(run.get("event_ids", [])),
            }
        )
    return windows


def analyze_prepared_video_v3(prepared: dict[str, Any], client: Any, schema: dict[str, Any], args: Any) -> dict[str, Any]:
    result = _ORIGINALS["analyze_prepared_video"](prepared, client, schema, args)
    state = assign_seven_stages_v3(
        list(result.get("reduce", {}).get("accepted_events", [])),
        result.get("reduce", {}).get("selection", {}).get("terminal_cleanup_event_id"),
    )
    result["stage_decoder"] = state["decoder"]
    result["anomalous_events"] = anomalous_events_from_assigned(list(result.get("assigned_events", [])))
    result["downstream_hints"] = {
        "meter_reading_windows": build_meter_reading_windows(result),
        "batched_recording": bool(state["decoder"].get("batched_recording")),
    }
    measurement_results = list(prepared.get("_v3_measurement_binary_results", []))
    result["map"]["measurement_binary"] = {
        "window_count": len(measurement_results),
        "valid_window_count": sum(1 for item in measurement_results if item.get("valid")),
        "observed_window_count": sum(
            1
            for item in measurement_results
            if isinstance(item.get("parsed_result"), dict)
            and item["parsed_result"].get("measurement_observed") == "yes"
        ),
    }
    result["map"]["endpoint_cleanup_binary"] = prepared.get("_v3_endpoint_cleanup_binary", {})
    result["sampling"]["map_strategy"] = "percentile_bucketed_tcs_with_low_motion_reserve_equal_budget"
    contract.write_json_atomic(prepared["video_dir"] / "result.json", result)
    return result


def bind_v3_identity() -> None:
    contract.STAGE_SCHEMA_ID = STAGE_SCHEMA_ID
    contract.BASE_ACTIONS = v3_contract.BASE_ACTIONS
    engine.STAGE_SCHEMA_ID = STAGE_SCHEMA_ID
    engine.ALGORITHM_ID = ALGORITHM_ID
    engine.ALGORITHM_SCHEMA_VERSION = ALGORITHM_SCHEMA_VERSION
    engine.DEFAULT_SCHEMA = DEFAULT_SCHEMA
    engine.DEFAULT_OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT
    engine.validate_map_response = v3_contract.validate_map_response
    engine.normalize_map_events = v3_contract.normalize_map_events
    engine.build_map_prompt = build_map_prompt
    engine.build_reduce_prompt = build_reduce_prompt
    engine.deduplicate_map_events = deduplicate_map_events_v3
    engine.find_temporal_conflicts = find_temporal_conflicts_v3
    engine.select_events = select_events_v3
    engine.prepare_video = prepare_video_v3
    engine._run_map = run_map_v3
    engine._run_reduce = run_reduce_v3
    engine._refine_boundaries = refine_boundaries_v3
    engine.assign_seven_stages = assign_seven_stages_v3
    engine.analyze_prepared_video = analyze_prepared_video_v3


def restore_v1_bindings() -> None:
    contract.STAGE_SCHEMA_ID = "resistance_7stage_no_battery_v1"
    contract.BASE_ACTIONS = (
        "wiring_action",
        "measurement_action",
        "writing_action",
        "cleanup_action",
        "uncertain",
    )
    engine.STAGE_SCHEMA_ID = "resistance_7stage_no_battery_v1"
    engine.ALGORITHM_ID = "qwen_experiment_action_hierarchical_v1"
    engine.ALGORITHM_SCHEMA_VERSION = "qwen_experiment_action_hierarchical_v1.v1"
    engine.DEFAULT_SCHEMA = ROOT / "configs" / "action_schemas" / "resistance_7stage_no_battery_v1.json"
    engine.DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "qwen_experiment_action_hierarchical_v1"
    engine.validate_map_response = _ORIGINALS["validate_map_response"]
    engine.normalize_map_events = _ORIGINALS["normalize_map_events"]
    engine.build_map_prompt = _ORIGINALS["build_map_prompt"]
    engine.build_reduce_prompt = _ORIGINALS["build_reduce_prompt"]
    engine.deduplicate_map_events = _ORIGINALS["deduplicate_map_events"]
    engine.find_temporal_conflicts = _ORIGINALS["find_temporal_conflicts"]
    engine.select_events = _ORIGINALS["select_events"]
    engine.prepare_video = _ORIGINALS["prepare_video"]
    engine._run_map = _ORIGINALS["run_map"]
    engine._run_reduce = _ORIGINALS["run_reduce"]
    engine._refine_boundaries = _ORIGINALS["refine_boundaries"]
    engine.assign_seven_stages = _ORIGINALS["assign_seven_stages"]
    engine.analyze_prepared_video = _ORIGINALS["analyze_prepared_video"]


def normalized_argv(argv: list[str] | None = None) -> list[str]:
    values = list(sys.argv[1:] if argv is None else argv)
    if "--reduce-recovery-policy" not in values:
        values.extend(["--reduce-recovery-policy", "local_partial"])
    return values


def main(argv: list[str] | None = None) -> int:
    bind_v3_identity()
    try:
        return engine.main(normalized_argv(argv))
    finally:
        restore_v1_bindings()


if __name__ == "__main__":
    raise SystemExit(main())
