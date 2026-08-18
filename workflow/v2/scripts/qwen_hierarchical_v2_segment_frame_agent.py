#!/usr/bin/env python3
"""Bounded current-run frame requests for missing measurement transitions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import cv2

from qwen_hierarchical_v1_contract import (
    normalize_map_events,
    validate_map_response,
    write_json_atomic,
)
from qwen_hierarchical_v1_prompts import build_map_prompt, build_map_retry_prompt
from qwen_hierarchical_v1_reduce import deduplicate_map_events


AGENT_VERSION = "segment_frame_sampling_agent.v1"
SUPPLEMENTAL_EVENT_SOURCE = "segment_frame_agent_current_run"
SUPPLEMENTAL_MERGE_POLICY = "base_reduce_then_insert_target_measurement"
ROUND_1_INTERVAL_SECONDS = 0.5
ROUND_2_INTERVAL_SECONDS = 0.25
ROUND_1_MAX_EDGE = 1280
ROUND_2_MAX_EDGE = 1920
SUPPLEMENTAL_JPEG_QUALITY = 92
MAX_REQUESTS = 4
MAX_FRAMES = 64
MAX_EXPERIMENT_CYCLES = 2
MAX_ROUND_1_WINDOW_SECONDS = 8.0
MAX_ROUND_2_WINDOW_SECONDS = 4.0


def _seconds(event: dict[str, Any], field: str) -> float:
    return float(event[field])


def _mark_supplemental_event(
    event: dict[str, Any],
    request: dict[str, Any],
) -> dict[str, Any]:
    return {
        **event,
        "segment_frame_agent_source": SUPPLEMENTAL_EVENT_SOURCE,
        "segment_frame_agent_request_id": str(request["request_id"]),
        "segment_frame_agent_cycle_index": int(request["cycle_index"]),
    }


def plan_frame_requests(
    map_events: list[dict[str, Any]],
    fixed_start: float,
    fixed_end: float,
) -> list[dict[str, Any]]:
    """Find wiring-to-writing transitions without direct measurement evidence."""
    canonical = deduplicate_map_events(map_events)
    ordered = sorted(
        (event for event in canonical if event.get("action_type") != "uncertain"),
        key=lambda event: (
            int(event["first_frame_number"]),
            int(event["last_frame_number"]),
            str(event["event_id"]),
        ),
    )
    writings = [event for event in ordered if event.get("action_type") == "writing_action"]
    requests: list[dict[str, Any]] = []
    previous_writing_end = float(fixed_start)
    for cycle_index, writing in enumerate(writings, start=1):
        writing_start = _seconds(writing, "first_seconds")
        wiring_candidates = [
            event
            for event in ordered
            if event.get("action_type") == "wiring_action"
            and _seconds(event, "last_seconds") <= writing_start
            and _seconds(event, "last_seconds") >= previous_writing_end
        ]
        if not wiring_candidates:
            previous_writing_end = max(previous_writing_end, _seconds(writing, "last_seconds"))
            continue
        wiring = max(wiring_candidates, key=lambda event: _seconds(event, "last_seconds"))
        wiring_end = _seconds(wiring, "last_seconds")
        measurement_present = any(
            event.get("action_type") == "measurement_action"
            and _seconds(event, "last_seconds") >= wiring_end
            and _seconds(event, "first_seconds") <= writing_start
            for event in ordered
        )
        if not measurement_present:
            end = min(float(fixed_end), writing_start + 0.5)
            start = max(float(fixed_start), wiring_end - 0.5, end - MAX_ROUND_1_WINDOW_SECONDS)
            if end > start:
                requests.append(
                    {
                        "request_id": f"measurement_gap_{len(requests) + 1:03d}",
                        "target_action": "measurement_action",
                        "cycle_index": cycle_index,
                        "start_seconds": round(start, 6),
                        "end_seconds": round(end, 6),
                        "selected_by": "measurement_missing_between_wiring_and_recording",
                        "source_event_ids": [str(wiring["event_id"]), str(writing["event_id"])],
                    }
                )
        previous_writing_end = max(previous_writing_end, _seconds(writing, "last_seconds"))
    return requests[:MAX_EXPERIMENT_CYCLES]


def _observed_stages(engine: Any, map_events: list[dict[str, Any]]) -> list[str]:
    canonical = deduplicate_map_events(map_events)
    state = engine.assign_seven_stages(canonical, None)
    return list(
        dict.fromkeys(
            str(item["stage"])
            for item in state.get("observed_stage_intervals", [])
            if isinstance(item, dict) and isinstance(item.get("stage"), str)
        )
    )


def _limit_evenly(values: list[int], limit: int) -> list[int]:
    if len(values) <= limit:
        return values
    if limit <= 1:
        return [values[len(values) // 2]]
    indexes = {round(index * (len(values) - 1) / (limit - 1)) for index in range(limit)}
    return [values[index] for index in sorted(indexes)]


def _quality(path: Path) -> dict[str, float]:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return {"laplacian_variance": 0.0, "mean_luminance": 0.0}
    return {
        "laplacian_variance": round(float(cv2.Laplacian(image, cv2.CV_64F).var()), 3),
        "mean_luminance": round(float(image.mean()), 3),
    }


def _run_request(
    *,
    engine: Any,
    prepared: dict[str, Any],
    client: Any,
    args: Any,
    request: dict[str, Any],
    round_number: int,
    interval_seconds: float,
    max_model_edge: int,
    frame_budget: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], int]:
    request_id = f"{request['request_id']}_round_{round_number}"
    start = float(request["start_seconds"])
    end = float(request["end_seconds"])
    if round_number == 2:
        start = max(start, end - MAX_ROUND_2_WINDOW_SECONDS)
    frame_numbers = engine._frame_numbers_for_range(
        start,
        end,
        interval_seconds,
        prepared["fps"],
        prepared["frame_count"],
    )
    frame_numbers = _limit_evenly(frame_numbers, max(1, frame_budget))
    registry: dict[int, dict[str, Any]] = {}
    request_dir = prepared["video_dir"] / "segment_frame_agent" / request_id
    engine._extract_source_frames(
        prepared["manifest"],
        frame_numbers,
        request_dir / "frames",
        max_model_edge,
        registry,
        SUPPLEMENTAL_JPEG_QUALITY,
    )
    frames = []
    for number in frame_numbers:
        frame = dict(registry[number])
        frame["image_group_id"] = request_id
        frame["quality"] = _quality(Path(str(frame["path"])))
        frames.append(frame)
    window = {"window_id": request_id, "window_seconds": [start, end]}
    base_prompt = build_map_prompt(prepared["video_id"], window, frames)
    attempts: list[dict[str, Any]] = []
    parsed: dict[str, Any] | None = None
    errors: list[str] = []
    for attempt_index in range(args.max_attempts):
        prompt = base_prompt if attempt_index == 0 else build_map_retry_prompt(base_prompt, errors)
        raw = engine._attempt_qwen(client, prompt, frames, args.map_max_tokens)
        candidate = raw.get("parsed_result")
        parsed = candidate if isinstance(candidate, dict) else None
        errors = validate_map_response(parsed, request_id, frames)
        attempts.append(
            {
                "attempt_index": attempt_index + 1,
                "qwen": raw,
                "validation_errors": errors,
            }
        )
        if not errors:
            break
    events = normalize_map_events(parsed, request_id, frames) if parsed is not None and not errors else []
    record = {
        **request,
        "request_id": request_id,
        "round_number": round_number,
        "sample_interval_seconds": interval_seconds,
        "max_model_edge": max_model_edge,
        "jpeg_quality": SUPPLEMENTAL_JPEG_QUALITY,
        "input_frames": frames,
        "attempts": attempts,
        "valid": not errors,
        "validation_errors": errors,
        "normalized_events": events,
    }
    write_json_atomic(
        request_dir / "input.json",
        {key: value for key, value in record.items() if key not in {"attempts", "normalized_events"}},
    )
    write_json_atomic(request_dir / "result.json", record)
    return events, record, len(frames)


def run_segment_frame_agent(
    *,
    engine: Any,
    prepared: dict[str, Any],
    client: Any,
    args: Any,
    map_events: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    requests = plan_frame_requests(map_events, prepared["fixed_start"], prepared["fixed_end"])
    observed_stages = _observed_stages(engine, map_events)
    supplemental_events: list[dict[str, Any]] = []
    executions: list[dict[str, Any]] = []
    remaining_frames = MAX_FRAMES
    remaining_requests = MAX_REQUESTS
    for request in requests:
        if remaining_frames <= 0 or remaining_requests <= 0:
            break
        events, record, used = _run_request(
            engine=engine,
            prepared=prepared,
            client=client,
            args=args,
            request=request,
            round_number=1,
            interval_seconds=ROUND_1_INTERVAL_SECONDS,
            max_model_edge=max(ROUND_1_MAX_EDGE, int(args.max_model_edge)),
            frame_budget=min(17, remaining_frames),
        )
        target_events = [
            event
            for event in events
            if event.get("action_type") == request["target_action"]
        ]
        supplemental_events.extend(
            _mark_supplemental_event(event, request) for event in target_events
        )
        executions.append(record)
        remaining_frames -= used
        remaining_requests -= 1
        if target_events:
            continue
        if remaining_frames <= 0 or remaining_requests <= 0:
            continue
        events, record, used = _run_request(
            engine=engine,
            prepared=prepared,
            client=client,
            args=args,
            request=request,
            round_number=2,
            interval_seconds=ROUND_2_INTERVAL_SECONDS,
            max_model_edge=ROUND_2_MAX_EDGE,
            frame_budget=min(15, remaining_frames),
        )
        supplemental_events.extend(
            _mark_supplemental_event(event, request)
            for event in events
            if event.get("action_type") == request["target_action"]
        )
        executions.append(record)
        remaining_frames -= used
        remaining_requests -= 1

    report = {
        "schema_version": "segment_frame_sampling_agent.v1",
        "agent_version": AGENT_VERSION,
        "status": "completed",
        "selection_basis": "current_video_observed_situation_only",
        "observed_stages": observed_stages,
        "selected_skills": (
            [
                {
                    "rubric_ids": [],
                    "skill_id": "segment.missing_measurement_resampling",
                    "parameters": {
                        "round_1_interval_seconds": ROUND_1_INTERVAL_SECONDS,
                        "round_2_interval_seconds": ROUND_2_INTERVAL_SECONDS,
                        "round_1_max_edge": max(ROUND_1_MAX_EDGE, int(args.max_model_edge)),
                        "round_2_max_edge": ROUND_2_MAX_EDGE,
                        "max_requests": MAX_REQUESTS,
                        "max_frames": MAX_FRAMES,
                    },
                    "selected_by": "current_map_wiring_to_writing_gap_without_measurement",
                }
            ]
            if requests
            else []
        ),
        "planned_requests": requests,
        "executed_requests": executions,
        "supplemental_event_count": len(supplemental_events),
        "supplemental_merge_policy": SUPPLEMENTAL_MERGE_POLICY,
        "supplemental_frame_count": MAX_FRAMES - remaining_frames,
        "frame_request_count": len(executions),
        "qwen_request_count": sum(len(item.get("attempts", [])) for item in executions),
        "video_id_used_for_routing": False,
        "historical_artifacts_used": False,
        "fixed_video_roi_used": False,
    }
    write_json_atomic(prepared["video_dir"] / "segment_frame_agent" / "report.json", report)
    return supplemental_events, report


def install(engine: Any) -> Callable[..., tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]]:
    """Wrap Map and Reduce while keeping supplemental evidence additive."""
    original_map = engine._run_map
    original_reduce = engine._run_reduce

    def run_map_with_agent(prepared: dict[str, Any], client: Any, args: Any):
        events, windows, reviews = original_map(prepared, client, args)
        supplemental, report = run_segment_frame_agent(
            engine=engine,
            prepared=prepared,
            client=client,
            args=args,
            map_events=events,
        )
        reviews = list(reviews)
        if report["planned_requests"] and not supplemental:
            reviews.append("segment_frame_agent_no_new_action")
        return events + supplemental, windows, reviews

    def run_reduce_with_agent(
        prepared: dict[str, Any],
        map_events: list[dict[str, Any]],
        client: Any,
        args: Any,
    ):
        base_events = [
            event
            for event in map_events
            if event.get("segment_frame_agent_source") != SUPPLEMENTAL_EVENT_SOURCE
        ]
        supplemental_events = [
            event
            for event in map_events
            if event.get("segment_frame_agent_source") == SUPPLEMENTAL_EVENT_SOURCE
        ]
        selected, result, reviews = original_reduce(prepared, base_events, client, args)
        merged, merge_record = merge_supplemental_measurements(
            selected,
            supplemental_events,
            result.get("selection", {}).get("terminal_cleanup_event_id"),
        )
        result = dict(result)
        result["accepted_events"] = merged
        result["segment_frame_agent_merge"] = merge_record
        selection = dict(result.get("selection", {}))
        selection["segment_frame_agent_inserted_event_ids"] = merge_record["inserted_event_ids"]
        result["selection"] = selection
        write_json_atomic(prepared["video_dir"] / "reduce" / "result.json", result)
        return merged, result, reviews

    engine._run_map = run_map_with_agent
    engine._run_reduce = run_reduce_with_agent
    return original_map


def merge_supplemental_measurements(
    base_selected_events: list[dict[str, Any]],
    supplemental_events: list[dict[str, Any]],
    terminal_cleanup_event_id: str | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Insert current-run measurements without reselecting base Map events."""
    canonical = deduplicate_map_events(
        [
            event
            for event in supplemental_events
            if event.get("action_type") == "measurement_action"
        ]
    )
    terminal = next(
        (
            event
            for event in base_selected_events
            if str(event.get("event_id")) == terminal_cleanup_event_id
        ),
        None,
    )
    terminal_start = int(terminal["first_frame_number"]) if terminal is not None else None
    existing_measurements = [
        event
        for event in base_selected_events
        if event.get("action_type") == "measurement_action"
    ]
    inserted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for event in canonical:
        if terminal_start is not None and int(event["representative_frame_number"]) >= terminal_start:
            rejected.append(
                {
                    "source_event_ids": list(event.get("source_event_ids", [])),
                    "reason": "measurement_not_before_terminal_cleanup",
                }
            )
            continue
        duplicate = next(
            (
                existing
                for existing in existing_measurements + inserted
                if int(existing["first_frame_number"]) <= int(event["last_frame_number"])
                and int(event["first_frame_number"]) <= int(existing["last_frame_number"])
            ),
            None,
        )
        if duplicate is not None:
            rejected.append(
                {
                    "source_event_ids": list(event.get("source_event_ids", [])),
                    "reason": "overlaps_existing_measurement",
                    "existing_event_id": str(duplicate["event_id"]),
                }
            )
            continue
        inserted.append(
            {
                **event,
                "event_id": f"agent_evt_{len(inserted) + 1:04d}",
                "selection_source": SUPPLEMENTAL_MERGE_POLICY,
            }
        )

    merged = sorted(
        [*base_selected_events, *inserted],
        key=lambda event: (
            int(event["representative_frame_number"]),
            int(event["first_frame_number"]),
            str(event["event_id"]),
        ),
    )
    return merged, {
        "policy": SUPPLEMENTAL_MERGE_POLICY,
        "base_accepted_event_ids": [str(event["event_id"]) for event in base_selected_events],
        "inserted_event_ids": [str(event["event_id"]) for event in inserted],
        "rejected_supplemental_events": rejected,
        "base_events_preserved": all(event in merged for event in base_selected_events),
    }
