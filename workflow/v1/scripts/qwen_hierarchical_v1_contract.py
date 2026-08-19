#!/usr/bin/env python3
"""Contracts and validators for the independent hierarchical_v1 pipeline."""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any


STAGE_SCHEMA_ID = "resistance_7stage_no_battery_v1"
STAGES = (
    "circuit_wiring",
    "measurement_1",
    "recording_1",
    "circuit_rewiring",
    "measurement_2",
    "recording_2",
    "material_cleanup",
)
BASE_ACTIONS = (
    "wiring_action",
    "measurement_action",
    "writing_action",
    "cleanup_action",
    "uncertain",
)
MAP_DECISIONS = {"observed", "no_action_observed", "uncertain"}
REDUCE_REJECTION_REASONS = {
    "duplicate",
    "conflicts_with_stronger_evidence",
    "insufficient_visual_evidence",
    "outside_locked_segment",
    "post_terminal_cleanup",
    "other",
}
BOUNDARY_DECISIONS = {"observed", "uncertain"}


def utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    # Windows PowerShell may write a UTF-8 BOM; accept it for source manifests
    # and temporary summaries without weakening the object-shape contract.
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"json_root_not_object:{path}")
    return value


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        prefix=path.name + ".",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    )
    temporary_path = Path(handle.name)
    try:
        with handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def create_run_directory(output_root: Path, run_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}", run_id):
        raise ValueError("run_id_must_use_ascii_letters_digits_dot_dash_or_underscore")
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / run_id
    run_dir.mkdir(parents=False, exist_ok=False)
    return run_dir


def load_stage_schema(path: Path) -> dict[str, Any]:
    schema = read_json(path)
    if schema.get("stage_schema_id") != STAGE_SCHEMA_ID:
        raise ValueError("stage_schema_id_mismatch")
    stages = schema.get("stages")
    if not isinstance(stages, list) or len(stages) != len(STAGES) or any(not isinstance(item, dict) for item in stages):
        raise ValueError("stages_shape_invalid")
    if tuple(item.get("id") for item in stages) != STAGES:
        raise ValueError("stage_order_mismatch")
    base_actions = schema.get("base_actions")
    if not isinstance(base_actions, list) or len(base_actions) != len(BASE_ACTIONS) or any(not isinstance(item, dict) for item in base_actions):
        raise ValueError("base_actions_shape_invalid")
    if tuple(item.get("id") for item in base_actions) != BASE_ACTIONS:
        raise ValueError("base_action_order_mismatch")
    if any(not isinstance(item.get("label_zh"), str) or not item.get("label_zh", "").strip() for item in stages):
        raise ValueError("stage_label_invalid")
    if any(item.get("order") != index for index, item in enumerate(stages, start=1)):
        raise ValueError("stage_numeric_order_invalid")
    if any(item.get("required_base_action") not in BASE_ACTIONS[:-1] for item in stages):
        raise ValueError("stage_required_base_action_invalid")
    cleanup = stages[-1]
    if cleanup.get("id") != "material_cleanup" or cleanup.get("absorbing_terminal") is not True:
        raise ValueError("cleanup_absorbing_terminal_required")
    if any(item.get("id") == "battery_change" for item in stages):
        raise ValueError("battery_stage_forbidden")
    return schema


def _finite_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


def valid_confidence(value: Any) -> bool:
    return _finite_number(value) and 0.0 <= float(value) <= 1.0


def _known_string_id(value: Any, known: dict[str, Any]) -> bool:
    return isinstance(value, str) and value in known


def select_source_records(
    summary: dict[str, Any],
    requested_video_ids: set[str] | None = None,
    allow_invalid_source_segments: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records = summary.get("records")
    if not isinstance(records, list):
        raise ValueError("source_summary_records_invalid")
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    found_ids: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            rejected.append({"source_record_index": index, "reason": "record_not_object"})
            continue
        video_id = record.get("source_video_id")
        if not isinstance(video_id, str) or not video_id:
            rejected.append({"source_record_index": index, "reason": "source_video_id_invalid"})
            continue
        if requested_video_ids and video_id not in requested_video_ids:
            continue
        found_ids.add(video_id)
        segment = record.get("segment")
        manifest = record.get("source_manifest")
        errors: list[str] = []
        if not isinstance(segment, dict):
            errors.append("segment_not_object")
            segment = {}
        start = segment.get("start_seconds")
        end = segment.get("end_seconds")
        if not _finite_number(start) or not _finite_number(end) or float(start) < 0 or float(start) >= float(end):
            errors.append("segment_bounds_invalid")
        source_errors = segment.get("segment_errors", [])
        if not isinstance(source_errors, list):
            errors.append("segment_errors_invalid")
            source_errors = ["segment_errors_invalid"]
        if segment.get("segment_valid") is not True:
            errors.append("source_segment_invalid")
        if source_errors:
            errors.append("source_segment_has_errors")
        if not isinstance(manifest, str) or not manifest:
            errors.append("source_manifest_invalid")
        has_usable_bounds = "segment_bounds_invalid" not in errors and "segment_not_object" not in errors
        accepted_despite_invalid = bool(errors) and allow_invalid_source_segments and has_usable_bounds and isinstance(manifest, str) and bool(manifest)
        provenance = {
            "source_video_id": video_id,
            "source_manifest": manifest,
            "source_segment": {
                "start_seconds": float(start) if _finite_number(start) else None,
                "end_seconds": float(end) if _finite_number(end) else None,
                "segment_valid": segment.get("segment_valid") is True,
                "segment_errors": list(source_errors),
                "end_reason": segment.get("end_reason"),
                "cleanup_seconds": segment.get("cleanup_seconds"),
            },
            "source_contract_errors": sorted(set(errors)),
            "accepted_despite_invalid_source": accepted_despite_invalid,
            "needs_review": bool(errors),
        }
        if not errors or accepted_despite_invalid:
            accepted.append(provenance)
        else:
            rejected.append({**provenance, "reason": "source_contract_rejected"})
    if requested_video_ids:
        for missing_id in sorted(requested_video_ids - found_ids):
            rejected.append({"source_video_id": missing_id, "reason": "requested_video_not_found"})
    return accepted, rejected


def build_overlapping_windows(
    start_seconds: float,
    end_seconds: float,
    window_seconds: float = 60.0,
    overlap_seconds: float = 10.0,
) -> list[dict[str, Any]]:
    if not all(_finite_number(item) for item in (start_seconds, end_seconds, window_seconds, overlap_seconds)):
        raise ValueError("window_parameters_must_be_finite")
    if start_seconds >= end_seconds:
        raise ValueError("window_range_invalid")
    if window_seconds <= 0 or overlap_seconds < 0 or overlap_seconds >= window_seconds:
        raise ValueError("window_geometry_invalid")
    stride = window_seconds - overlap_seconds
    windows: list[dict[str, Any]] = []
    cursor = float(start_seconds)
    index = 0
    while cursor < end_seconds - 1e-9:
        window_end = min(float(end_seconds), cursor + float(window_seconds))
        windows.append(
            {
                "window_id": f"w{index:03d}",
                "window_index": index,
                "window_seconds": [round(cursor, 6), round(window_end, 6)],
                "window_length_seconds": round(window_end - cursor, 6),
            }
        )
        if window_end >= end_seconds - 1e-9:
            break
        cursor += stride
        index += 1
    return windows


def sample_timestamps(start_seconds: float, end_seconds: float, interval_seconds: float) -> list[float]:
    if not _finite_number(interval_seconds) or interval_seconds <= 0:
        raise ValueError("sample_interval_invalid")
    if start_seconds > end_seconds:
        raise ValueError("sample_range_invalid")
    count = int(math.floor((end_seconds - start_seconds) / interval_seconds + 1e-9))
    values = [round(start_seconds + index * interval_seconds, 9) for index in range(count + 1)]
    if not values or values[-1] < end_seconds - 1e-6:
        values.append(round(end_seconds, 9))
    return values


def source_frame_id(frame_number: int) -> str:
    if frame_number < 0:
        raise ValueError("frame_number_negative")
    return f"frame_{frame_number:08d}"


def validate_map_response(
    value: dict[str, Any] | None,
    window_id: str,
    frames: list[dict[str, Any]],
) -> list[str]:
    if value is None:
        return ["map_response_not_parsed"]
    errors: list[str] = []
    frame_order = {str(frame["image_id"]): index for index, frame in enumerate(frames)}
    if value.get("window_id") != window_id:
        errors.append("window_id_mismatch")
    decision = value.get("decision")
    if not isinstance(decision, str) or decision not in MAP_DECISIONS:
        errors.append("decision_invalid")
    observations = value.get("observations")
    if not isinstance(observations, list) or len(observations) > 64:
        errors.append("observations_invalid")
        observations = []
    non_uncertain_count = 0
    previous_first_index = -1
    for index, observation in enumerate(observations):
        prefix = f"observation_{index}"
        if not isinstance(observation, dict):
            errors.append(prefix + "_not_object")
            continue
        action = observation.get("action_type")
        first_id = observation.get("first_frame_id")
        last_id = observation.get("last_frame_id")
        representative_id = observation.get("representative_frame_id")
        if action not in BASE_ACTIONS:
            errors.append(prefix + "_action_invalid")
        elif action != "uncertain":
            non_uncertain_count += 1
        first_valid = _known_string_id(first_id, frame_order)
        last_valid = _known_string_id(last_id, frame_order)
        representative_valid = _known_string_id(representative_id, frame_order)
        if not first_valid:
            errors.append(prefix + "_first_frame_invalid")
        if not last_valid:
            errors.append(prefix + "_last_frame_invalid")
        if not representative_valid:
            errors.append(prefix + "_representative_frame_invalid")
        if first_valid and last_valid and frame_order[first_id] > frame_order[last_id]:
            errors.append(prefix + "_frame_order_invalid")
        if first_valid and frame_order[first_id] < previous_first_index:
            errors.append(prefix + "_observation_order_invalid")
        if first_valid:
            previous_first_index = max(previous_first_index, frame_order[first_id])
        if (
            first_valid
            and last_valid
            and representative_valid
            and not frame_order[first_id] <= frame_order[representative_id] <= frame_order[last_id]
        ):
            errors.append(prefix + "_representative_outside_interval")
        evidence = observation.get("evidence")
        if not isinstance(evidence, str) or not evidence.strip() or len(evidence) > 160:
            errors.append(prefix + "_evidence_invalid")
        if not valid_confidence(observation.get("confidence")):
            errors.append(prefix + "_confidence_invalid")
    if decision == "observed" and non_uncertain_count == 0:
        errors.append("observed_without_supported_action")
    if decision == "no_action_observed" and observations:
        errors.append("no_action_with_observations")
    if decision == "uncertain" and not isinstance(value.get("uncertainty"), str):
        errors.append("uncertainty_invalid")
    if not valid_confidence(value.get("confidence")):
        errors.append("confidence_invalid")
    return sorted(set(errors))


def normalize_map_events(
    value: dict[str, Any],
    window_id: str,
    frames: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_id = {str(frame["image_id"]): frame for frame in frames}
    events: list[dict[str, Any]] = []
    for index, observation in enumerate(value.get("observations", []), start=1):
        first = by_id[str(observation["first_frame_id"])]
        last = by_id[str(observation["last_frame_id"])]
        representative = by_id[str(observation["representative_frame_id"])]
        events.append(
            {
                "source_event_id": f"{window_id}_e{index:02d}",
                "window_id": window_id,
                "action_type": observation["action_type"],
                "first_frame_id": first["image_id"],
                "last_frame_id": last["image_id"],
                "representative_frame_id": representative["image_id"],
                "first_frame_number": int(first["frame_number"]),
                "last_frame_number": int(last["frame_number"]),
                "representative_frame_number": int(representative["frame_number"]),
                "first_seconds": float(first["timestamp_seconds"]),
                "last_seconds": float(last["timestamp_seconds"]),
                "representative_seconds": float(representative["timestamp_seconds"]),
                "evidence": observation["evidence"],
                "confidence": float(observation["confidence"]),
            }
        )
    return events


def validate_reduce_response(value: dict[str, Any] | None, events: list[dict[str, Any]]) -> list[str]:
    if value is None:
        return ["reduce_response_not_parsed"]
    errors: list[str] = []
    event_by_id = {str(event["event_id"]): event for event in events}
    accepted = value.get("accepted_event_ids")
    if not isinstance(accepted, list) or any(not _known_string_id(event_id, event_by_id) for event_id in accepted):
        errors.append("accepted_event_ids_invalid")
        accepted = []
    if len(accepted) != len(set(accepted)):
        errors.append("accepted_event_ids_duplicate")
    rejected = value.get("rejected_events")
    rejected_ids: list[str] = []
    if not isinstance(rejected, list):
        errors.append("rejected_events_invalid")
        rejected = []
    for index, item in enumerate(rejected):
        if not isinstance(item, dict):
            errors.append(f"rejected_{index}_not_object")
            continue
        event_id = item.get("event_id")
        rejected_ids.append(event_id) if isinstance(event_id, str) else None
        if not _known_string_id(event_id, event_by_id):
            errors.append(f"rejected_{index}_event_id_invalid")
        reason = item.get("reason")
        if not isinstance(reason, str) or reason not in REDUCE_REJECTION_REASONS:
            errors.append(f"rejected_{index}_reason_invalid")
        if not isinstance(item.get("explanation"), str) or not item.get("explanation", "").strip():
            errors.append(f"rejected_{index}_explanation_invalid")
    if len(rejected_ids) != len(set(rejected_ids)):
        errors.append("rejected_event_ids_duplicate")
    if set(accepted) & set(rejected_ids):
        errors.append("event_both_accepted_and_rejected")
    if set(accepted) | set(rejected_ids) != set(event_by_id):
        errors.append("reduce_decision_not_exhaustive")
    terminal_id = value.get("terminal_cleanup_event_id")
    if terminal_id is not None:
        terminal_event = event_by_id.get(terminal_id) if isinstance(terminal_id, str) else None
        if terminal_event is None or terminal_id not in accepted:
            errors.append("terminal_cleanup_event_not_accepted")
        elif terminal_event.get("action_type") != "cleanup_action":
            errors.append("terminal_cleanup_event_wrong_action")
        elif terminal_id in accepted:
            terminal_frame = int(terminal_event.get("first_frame_number", -1))
            accepted_after_terminal = [
                event_id
                for event_id in accepted
                if event_id != terminal_id
                and int(event_by_id[event_id].get("last_frame_number", -1)) >= terminal_frame
            ]
            if accepted_after_terminal:
                errors.append("accepted_event_after_terminal_cleanup")
    conflicts = value.get("conflicts")
    if not isinstance(conflicts, list):
        errors.append("conflicts_invalid")
    else:
        for index, conflict in enumerate(conflicts):
            if not isinstance(conflict, dict):
                errors.append(f"conflict_{index}_not_object")
                continue
            event_ids = conflict.get("event_ids")
            if not isinstance(event_ids, list) or len(event_ids) < 2 or any(not _known_string_id(item, event_by_id) for item in event_ids):
                errors.append(f"conflict_{index}_event_ids_invalid")
            if not isinstance(conflict.get("resolution"), str) or not conflict.get("resolution", "").strip():
                errors.append(f"conflict_{index}_resolution_invalid")
    if not valid_confidence(value.get("confidence")):
        errors.append("confidence_invalid")
    return sorted(set(errors))


def validate_boundary_response(
    value: dict[str, Any] | None,
    boundary_id: str,
    frames: list[dict[str, Any]],
) -> list[str]:
    if value is None:
        return ["boundary_response_not_parsed"]
    errors: list[str] = []
    order = {str(frame["image_id"]): index for index, frame in enumerate(frames)}
    if value.get("boundary_id") != boundary_id:
        errors.append("boundary_id_mismatch")
    decision = value.get("decision")
    if not isinstance(decision, str) or decision not in BOUNDARY_DECISIONS:
        errors.append("decision_invalid")
    last_from = value.get("last_from_frame_id")
    first_to = value.get("first_to_frame_id")
    if decision == "observed":
        last_valid = _known_string_id(last_from, order)
        first_valid = _known_string_id(first_to, order)
        if not last_valid:
            errors.append("last_from_frame_invalid")
        if not first_valid:
            errors.append("first_to_frame_invalid")
        if last_valid and first_valid and order[last_from] >= order[first_to]:
            errors.append("boundary_frame_order_invalid")
    elif last_from is not None or first_to is not None:
        errors.append("uncertain_boundary_must_use_null_frames")
    if not isinstance(value.get("evidence"), str) or not value.get("evidence", "").strip():
        errors.append("evidence_invalid")
    if not isinstance(value.get("uncertainty"), str):
        errors.append("uncertainty_invalid")
    if not valid_confidence(value.get("confidence")):
        errors.append("confidence_invalid")
    return sorted(set(errors))
