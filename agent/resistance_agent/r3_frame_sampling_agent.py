"""Bounded, current-video-only adaptive frame sampling for Rubric 3.

The existing R3 OpenCV implementation remains the classifier. This module
adds a small evidence agent around it: run the normal 5 fps scan, inspect only
that scan's diagnostics, and request independent phase-shifted 5 fps bursts
when the switch or wiring evidence is weak or temporally borderline.
"""

from __future__ import annotations

import json
import math
from copy import deepcopy
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import cv2

try:
    from .opencv_switch_overlap import (
        analyze_opencv_switch_overlap,
        fuse_same_frame_records,
    )
    from .opencv_switch_state import (
        _annotate_closed_persistence,
        _smooth_states,
        cluster_threshold,
    )
except ImportError:
    from opencv_switch_overlap import (  # type: ignore
        analyze_opencv_switch_overlap,
        fuse_same_frame_records,
    )
    from opencv_switch_state import (  # type: ignore
        _annotate_closed_persistence,
        _smooth_states,
        cluster_threshold,
    )


AGENT_VERSION = "r3_frame_sampling_agent.v1"
SCHEMA_VERSION = "resistance_agent_r3_frame_sampling_agent.v1"
BASE_SAMPLING_FPS = 5.0
SUPPLEMENTAL_SAMPLING_FPS = 5.0
PHASE_OFFSET_SECONDS = 0.1
BURST_RADIUS_SECONDS = 0.8
STAGE_EDGE_CONTEXT_SECONDS = 1.0
LOW_SWITCH_COVERAGE = 0.18
THRESHOLD_MARGIN = 0.06
NEAR_WIRING_SECONDS = 0.8
DEFAULT_MAX_ROUNDS = 2
DEFAULT_MAX_REQUESTS_PER_ROUND = 3
DEFAULT_MAX_SUPPLEMENTAL_FRAMES = 64
ROI_MODE = "dynamic_current_frame_switch_and_plug"
FUSION_POLICY = "same_frame_closed_and_wiring_active"
FALLBACK_ANCHOR_REASONS = {"stage_boundary_ambiguity"}


Analyzer = Callable[..., dict[str, Any]]


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _frame_number(item: dict[str, Any]) -> int | None:
    value = item.get("frame_number")
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _timestamp(item: dict[str, Any]) -> float:
    return _number(item.get("timestamp_seconds"))


def _stage(item: dict[str, Any]) -> str:
    return str(item.get("stage") or "circuit_wiring")


def _window_for_anchor(
    windows: list[dict[str, Any]], anchor: dict[str, Any]
) -> dict[str, Any] | None:
    timestamp = _number(anchor.get("timestamp_seconds"), -1.0)
    window_id = str(anchor.get("window_id") or "")
    stage = str(anchor.get("stage") or "")
    for window in windows:
        if window_id and str(window.get("window_id") or "") == window_id:
            return window
    for window in windows:
        if stage and str(window.get("stage") or "") != stage:
            continue
        if _number(window.get("start_seconds")) <= timestamp <= _number(
            window.get("end_seconds")
        ):
            return window
    return None


def _near_any(timestamp: float, values: list[float], radius: float) -> bool:
    return any(abs(timestamp - candidate) <= radius for candidate in values)


def assess_evidence(report: dict[str, Any]) -> dict[str, Any]:
    """Summarize evidence quality without changing the binary decision."""
    frames = [item for item in report.get("frames") or [] if isinstance(item, dict)]
    observations = [
        item
        for item in report.get("switch_state_observations") or []
        if isinstance(item, dict)
    ]
    transitions = [
        item for item in report.get("real_plug_transitions") or [] if isinstance(item, dict)
    ]
    sample_count = int(report.get("sample_count") or len(frames) or 0)
    switch_count = int(report.get("switch_tracked_observation_count") or len(observations))
    coverage = _number(
        report.get("switch_coverage"), switch_count / max(sample_count, 1)
    )
    wiring_frames = [item for item in frames if item.get("wiring_active") is True]
    missing_switch = [item for item in wiring_frames if item.get("switch_visible") is not True]
    borderline_closed = [
        item
        for item in observations
        if item.get("state") == "closed"
        and 1 <= int(item.get("closed_persistence_count") or 0) < 3
    ]
    active_times = [_timestamp(item) for item in wiring_frames]
    threshold = _number(report.get("switch_state_threshold"), math.nan)
    threshold_margin = []
    if math.isfinite(threshold) and active_times:
        for item in observations:
            score = _number(item.get("smoothed_bridge_score"), math.nan)
            if not math.isfinite(score):
                continue
            if abs(score - threshold) <= THRESHOLD_MARGIN and _near_any(
                _timestamp(item), active_times, NEAR_WIRING_SECONDS
            ):
                threshold_margin.append(item)

    reasons: list[str] = []
    if switch_count == 0:
        reasons.append("no_switch_observation")
    elif coverage < LOW_SWITCH_COVERAGE:
        reasons.append("low_switch_coverage")
    if not transitions:
        reasons.append("no_wiring_activity_observed")
    if missing_switch:
        reasons.append("switch_not_visible_during_wiring")
    if borderline_closed:
        reasons.append("closed_persistence_boundary")
    if threshold_margin:
        reasons.append("state_threshold_margin")

    decision = str(report.get("decision") or "pass")
    return {
        "decision": decision,
        "sample_count": sample_count,
        "switch_observation_count": switch_count,
        "switch_coverage": round(coverage, 4),
        "real_plug_transition_count": len(transitions),
        "wiring_active_frame_count": len(wiring_frames),
        "wiring_active_without_switch_count": len(missing_switch),
        "borderline_closed_observation_count": len(borderline_closed),
        "threshold_margin_observation_count": len(threshold_margin),
        "reasons": reasons,
        "additional_frames_recommended": decision != "fail" and bool(reasons),
    }


def _longest_visibility_gap(
    frames: list[dict[str, Any]], window: dict[str, Any]
) -> float:
    window_id = str(window.get("window_id") or "")
    stage = str(window.get("stage") or "")
    values = sorted(
        [
            item
            for item in frames
            if (
                str(item.get("window_id") or "") == window_id
                or (
                    _stage(item) == stage
                    and _number(window.get("start_seconds"))
                    <= _timestamp(item)
                    <= _number(window.get("end_seconds"))
                )
            )
        ],
        key=_timestamp,
    )
    if not values:
        return (
            _number(window.get("start_seconds")) + _number(window.get("end_seconds"))
        ) / 2.0
    longest: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    for item in values:
        if item.get("switch_visible") is True:
            if len(current) > len(longest):
                longest = current
            current = []
        else:
            current.append(item)
    if len(current) > len(longest):
        longest = current
    if longest:
        return _timestamp(longest[len(longest) // 2])
    return _timestamp(values[len(values) // 2])


def _candidate_anchors(
    report: dict[str, Any],
    windows: list[dict[str, Any]],
    round_number: int,
) -> list[dict[str, Any]]:
    frames = [item for item in report.get("frames") or [] if isinstance(item, dict)]
    observations = [
        item
        for item in report.get("switch_state_observations") or []
        if isinstance(item, dict)
    ]
    transitions = [
        item for item in report.get("real_plug_transitions") or [] if isinstance(item, dict)
    ]
    active_times = [_timestamp(item) for item in frames if item.get("wiring_active") is True]
    threshold = _number(report.get("switch_state_threshold"), math.nan)
    anchors: list[dict[str, Any]] = []

    def add(item: dict[str, Any], reason: str, priority: int) -> None:
        anchors.append(
            {
                "timestamp_seconds": _timestamp(item),
                "frame_number": _frame_number(item),
                "window_id": str(item.get("window_id") or ""),
                "stage": _stage(item),
                "reason": reason,
                "priority": priority,
            }
        )

    for item in frames:
        if item.get("wiring_active") is True and item.get("switch_visible") is not True:
            add(item, "switch_not_visible_during_wiring", 100)
    for item in observations:
        persistence = int(item.get("closed_persistence_count") or 0)
        if item.get("state") == "closed" and 1 <= persistence < 3:
            add(item, "closed_persistence_boundary", 95)
        score = _number(item.get("smoothed_bridge_score"), math.nan)
        if (
            math.isfinite(score)
            and math.isfinite(threshold)
            and abs(score - threshold) <= THRESHOLD_MARGIN
            and _near_any(_timestamp(item), active_times, NEAR_WIRING_SECONDS)
        ):
            add(item, "state_threshold_margin", 90)
    for item in transitions:
        parent = _window_for_anchor(windows, item)
        if parent is None:
            continue
        timestamp = _timestamp(item)
        edge_distance = min(
            abs(timestamp - _number(parent.get("start_seconds"))),
            abs(timestamp - _number(parent.get("end_seconds"))),
        )
        if edge_distance <= STAGE_EDGE_CONTEXT_SECONDS:
            add(item, "stage_edge_activity", 85)

    quality = assess_evidence(report)
    if "low_switch_coverage" in quality["reasons"] or "no_switch_observation" in quality[
        "reasons"
    ]:
        for window in windows:
            anchors.append(
                {
                    "timestamp_seconds": _longest_visibility_gap(frames, window),
                    "frame_number": None,
                    "window_id": str(window.get("window_id") or ""),
                    "stage": str(window.get("stage") or "circuit_wiring"),
                    "reason": (
                        "no_switch_observation"
                        if quality["switch_observation_count"] == 0
                        else "low_switch_coverage"
                    ),
                    "priority": 70,
                }
            )
    if "no_wiring_activity_observed" in quality["reasons"]:
        for window in windows:
            anchors.append(
                {
                    "timestamp_seconds": (
                        _number(window.get("start_seconds"))
                        + _number(window.get("end_seconds"))
                    )
                    / 2.0,
                    "frame_number": None,
                    "window_id": str(window.get("window_id") or ""),
                    "stage": str(window.get("stage") or "circuit_wiring"),
                    "reason": "no_wiring_activity_observed",
                    "priority": 60,
                }
            )
    if round_number > 1:
        for window in windows:
            for edge in ("start_seconds", "end_seconds"):
                anchors.append(
                    {
                        "timestamp_seconds": _number(window.get(edge)),
                        "frame_number": None,
                        "window_id": str(window.get("window_id") or ""),
                        "stage": str(window.get("stage") or "circuit_wiring"),
                        "reason": "stage_boundary_ambiguity",
                        "priority": 55,
                    }
                )

    anchors.sort(key=lambda item: (-int(item["priority"]), item["timestamp_seconds"]))
    deduplicated: list[dict[str, Any]] = []
    for anchor in anchors:
        if any(
            existing["stage"] == anchor["stage"]
            and existing["reason"] == anchor["reason"]
            and abs(existing["timestamp_seconds"] - anchor["timestamp_seconds"]) < 0.45
            for existing in deduplicated
        ):
            continue
        deduplicated.append(anchor)
    return deduplicated


def _temporal_coverage_order(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Order one trigger reason by midpoint, then progressively wider coverage."""
    ordered = sorted(items, key=lambda item: _number(item.get("timestamp_seconds")))
    if len(ordered) <= 2:
        return ordered
    remaining = list(range(len(ordered)))
    selected_indexes: list[int] = []
    midpoint = len(ordered) // 2
    selected_indexes.append(midpoint)
    remaining.remove(midpoint)
    while remaining:
        index = max(
            remaining,
            key=lambda candidate: (
                min(abs(candidate - selected) for selected in selected_indexes),
                -candidate,
            ),
        )
        selected_indexes.append(index)
        remaining.remove(index)
    return [ordered[index] for index in selected_indexes]


def _reason_diverse_anchors(anchors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Give each direct diagnostic reason one request before repeating a reason."""
    by_reason: dict[str, list[dict[str, Any]]] = defaultdict(list)
    fallback: list[dict[str, Any]] = []
    for anchor in anchors:
        reason = str(anchor.get("reason") or "")
        if reason in FALLBACK_ANCHOR_REASONS:
            fallback.append(anchor)
        else:
            by_reason[reason].append(anchor)
    for reason in by_reason:
        by_reason[reason] = _temporal_coverage_order(by_reason[reason])

    reason_order = sorted(
        by_reason,
        key=lambda reason: (
            -max(int(item.get("priority") or 0) for item in by_reason[reason]),
            min(_number(item.get("timestamp_seconds")) for item in by_reason[reason]),
            reason,
        ),
    )
    selected: list[dict[str, Any]] = []
    for reason in reason_order:
        selected.append(by_reason[reason].pop(0))

    # reason_order already carries priority; each bucket keeps its midpoint/
    # endpoint coverage order instead of being sorted back to the early frames.
    remaining = [item for reason in reason_order for item in by_reason[reason]]
    fallback.sort(key=lambda item: (-int(item["priority"]), item["timestamp_seconds"]))
    remaining.extend(fallback)
    selected.extend(remaining)
    return selected


def _aligned_phase_bounds(
    parent: dict[str, Any],
    anchor_seconds: float,
    request_type: str,
    duration_seconds: float,
) -> tuple[float, float]:
    parent_start = _number(parent.get("start_seconds"))
    parent_end = _number(parent.get("end_seconds"))
    # The original R3 decision is only valid during wiring stages. Boundary
    # searches move toward the edge but never score frames outside that stage.
    allowed_start = max(0.0, parent_start)
    allowed_end = min(duration_seconds, parent_end)
    interval = 1.0 / SUPPLEMENTAL_SAMPLING_FPS
    phase_origin = parent_start + PHASE_OFFSET_SECONDS
    desired_start = anchor_seconds - BURST_RADIUS_SECONDS
    step = math.floor((desired_start - phase_origin) / interval)
    start = phase_origin + step * interval
    if start < allowed_start:
        start = phase_origin + math.ceil((allowed_start - phase_origin) / interval) * interval
    end = min(allowed_end, start + BURST_RADIUS_SECONDS * 2.0)
    minimum_span = 2.0 / SUPPLEMENTAL_SAMPLING_FPS
    if end - start < minimum_span:
        desired_start = max(allowed_start, allowed_end - BURST_RADIUS_SECONDS * 2.0)
        step = math.ceil((desired_start - phase_origin) / interval)
        start = max(allowed_start, phase_origin + step * interval)
        end = allowed_end
    return round(max(0.0, start), 6), round(max(start, end), 6)


def _sample_frame_numbers(
    start_seconds: float,
    end_seconds: float,
    source_fps: float,
    frame_count: int,
) -> list[int]:
    return sorted(
        {
            frame_number
            for _, frame_number in _sample_points(
                start_seconds, end_seconds, source_fps, frame_count
            )
        }
    )


def _sample_points(
    start_seconds: float,
    end_seconds: float,
    source_fps: float,
    frame_count: int,
) -> list[tuple[float, int]]:
    if source_fps <= 0.0 or frame_count <= 0:
        return []
    count = int(
        math.floor((end_seconds - start_seconds) * SUPPLEMENTAL_SAMPLING_FPS + 1e-9)
    ) + 1
    points: list[tuple[float, int]] = []
    seen: set[int] = set()
    for index in range(max(count, 1)):
        timestamp = start_seconds + index / SUPPLEMENTAL_SAMPLING_FPS
        frame_number = min(
            frame_count - 1,
            max(0, int(round(timestamp * source_fps))),
        )
        if frame_number in seen:
            continue
        seen.add(frame_number)
        points.append((round(timestamp, 6), frame_number))
    return points


def _fit_request_window(
    *,
    start_seconds: float,
    end_seconds: float,
    anchor_seconds: float,
    source_fps: float,
    frame_count: int,
    known_frame_numbers: set[int],
    supplemental_frame_numbers: set[int],
    remaining_frame_budget: int,
) -> tuple[float, float, list[int], list[int]] | None:
    """Keep baseline context while preventing overlap between supplemental scans."""
    points = _sample_points(start_seconds, end_seconds, source_fps, frame_count)
    runs: list[list[tuple[float, int]]] = []
    current: list[tuple[float, int]] = []
    for point in points:
        if point[1] in supplemental_frame_numbers:
            if current:
                runs.append(current)
                current = []
        else:
            current.append(point)
    if current:
        runs.append(current)

    candidates: list[tuple[tuple[float, float, float, float], tuple[float, float, list[int], list[int]]]] = []
    for run in runs:
        for left in range(len(run)):
            for right in range(left, len(run)):
                segment = run[left : right + 1]
                all_frames = [frame_number for _, frame_number in segment]
                new_frames = [
                    frame_number
                    for frame_number in all_frames
                    if frame_number not in known_frame_numbers
                ]
                if not new_frames or len(new_frames) > remaining_frame_budget:
                    continue
                segment_midpoint = (segment[0][0] + segment[-1][0]) / 2.0
                score = (
                    float(len(new_frames)),
                    float(len(all_frames)),
                    -abs(segment_midpoint - anchor_seconds),
                    -segment[0][0],
                )
                candidates.append(
                    (
                        score,
                        (segment[0][0], segment[-1][0], all_frames, new_frames),
                    )
                )
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def plan_frame_requests(
    *,
    report: dict[str, Any],
    candidate_windows: list[dict[str, Any]],
    round_number: int,
    duration_seconds: float,
    source_fps: float,
    frame_count: int,
    known_frame_numbers: set[int] | None = None,
    supplemental_frame_numbers: set[int] | None = None,
    max_requests: int = DEFAULT_MAX_REQUESTS_PER_ROUND,
    remaining_frame_budget: int = DEFAULT_MAX_SUPPLEMENTAL_FRAMES,
) -> dict[str, Any]:
    """Create one deterministic request plan from current-run visual evidence."""
    known = set(known_frame_numbers or set())
    supplemental = set(supplemental_frame_numbers or set())
    quality = assess_evidence(report)
    requests: list[dict[str, Any]] = []
    if quality["additional_frames_recommended"]:
        anchors = _reason_diverse_anchors(
            _candidate_anchors(report, candidate_windows, round_number)
        )
        for anchor in anchors:
            parent = _window_for_anchor(candidate_windows, anchor)
            if parent is None:
                continue
            if anchor["reason"] in {"stage_edge_activity", "stage_boundary_ambiguity"}:
                request_type = "expand_within_stage"
            elif anchor["reason"] in {"low_switch_coverage", "no_switch_observation"}:
                request_type = "seek_clearer_frame"
            else:
                request_type = "neighbor_burst"
            start, end = _aligned_phase_bounds(
                parent, anchor["timestamp_seconds"], request_type, duration_seconds
            )
            if end <= start:
                continue
            fitted = _fit_request_window(
                start_seconds=start,
                end_seconds=end,
                anchor_seconds=_number(anchor.get("timestamp_seconds")),
                source_fps=source_fps,
                frame_count=frame_count,
                known_frame_numbers=known,
                supplemental_frame_numbers=supplemental,
                remaining_frame_budget=remaining_frame_budget,
            )
            if fitted is None:
                continue
            start, end, all_frames, new_frames = fitted
            request_number = len(requests) + 1
            request_id = f"round_{round_number:02d}_request_{request_number:02d}"
            candidate_window = {
                "window_id": f"adaptive_{request_id}",
                "stage": str(parent.get("stage") or "circuit_wiring"),
                "start_seconds": start,
                "end_seconds": end,
                "source_event_ids": list(parent.get("source_event_ids") or []),
                "source_confidence": parent.get("source_confidence"),
            }
            requests.append(
                {
                    "request_id": request_id,
                    "request_type": request_type,
                    "reason": anchor["reason"],
                    "selected_by": "current_run_visual_evidence_quality",
                    "anchor_seconds": round(anchor["timestamp_seconds"], 6),
                    "source_frame_number": anchor.get("frame_number"),
                    "source_window_id": str(parent.get("window_id") or ""),
                    "stage": candidate_window["stage"],
                    "sampling_fps": SUPPLEMENTAL_SAMPLING_FPS,
                    "phase_offset_seconds": PHASE_OFFSET_SECONDS,
                    "candidate_window": candidate_window,
                    "scoring_window": {
                        "start_seconds": _number(parent.get("start_seconds")),
                        "end_seconds": _number(parent.get("end_seconds")),
                    },
                    "outside_stage_frames_scored": False,
                    "expected_frame_numbers": all_frames,
                    "expected_new_frame_numbers": new_frames,
                    "expected_new_frame_count": len(new_frames),
                    "baseline_context_frame_count": len(all_frames) - len(new_frames),
                    "roi_mode": ROI_MODE,
                    "fusion_policy": FUSION_POLICY,
                }
            )
            known.update(all_frames)
            supplemental.update(all_frames)
            remaining_frame_budget -= len(new_frames)
            if len(requests) >= max_requests or remaining_frame_budget <= 0:
                break
    return {
        "schema_version": "resistance_agent_r3_frame_request_plan.v1",
        "agent_version": AGENT_VERSION,
        "round_number": round_number,
        "selection_basis": "current_video_observed_situation_only",
        "selection_strategy": "reason_diverse_priority_then_temporal_coverage",
        "evidence_quality": quality,
        "requests": requests,
        "request_count": len(requests),
        "expected_new_frame_count": sum(
            int(item["expected_new_frame_count"]) for item in requests
        ),
        "selected_reason_counts": {
            reason: sum(1 for item in requests if item["reason"] == reason)
            for reason in sorted({str(item["reason"]) for item in requests})
        },
        "duplicate_frame_decode_prevented": True,
        "duplicate_frame_decode_scope": "between_supplemental_requests",
        "video_id_used_for_routing": False,
        "historical_artifacts_used": False,
        "fixed_video_roi_used": False,
        "excel_accessed": False,
    }


def _aggregate_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    frames: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    overlaps: list[dict[str, Any]] = []
    for report in reports:
        frames.extend(item for item in report.get("frames") or [] if isinstance(item, dict))
        observations.extend(
            item
            for item in report.get("switch_state_observations") or []
            if isinstance(item, dict)
        )
        transitions.extend(
            item
            for item in report.get("real_plug_transitions") or []
            if isinstance(item, dict)
        )
        overlaps.extend(
            item for item in report.get("same_frame_overlaps") or [] if isinstance(item, dict)
        )
    observations = deepcopy(observations)
    for item in observations:
        if "bridge_score" not in item:
            item["bridge_score"] = _number(item.get("smoothed_bridge_score"))
    threshold, centers = cluster_threshold(observations)
    _smooth_states(observations, threshold)
    _annotate_closed_persistence(observations)
    frames, overlaps = fuse_same_frame_records(frames, observations, transitions)
    physical_frames = {
        (_stage(item), number)
        for item in frames
        if (number := _frame_number(item)) is not None
    }
    visible_frames = {
        (_stage(item), number)
        for item in frames
        if item.get("switch_visible") is True
        and (number := _frame_number(item)) is not None
    }
    return {
        "decision": "fail" if overlaps else "pass",
        "sample_count": len(physical_frames),
        "switch_tracked_observation_count": len(visible_frames),
        "switch_coverage": len(visible_frames) / max(len(physical_frames), 1),
        "switch_state_threshold": round(float(threshold), 4),
        "switch_state_threshold_source": "shared_current_run_evidence",
        "switch_state_cluster_centers": [round(float(value), 4) for value in centers],
        "frames": frames,
        "switch_state_observations": observations,
        "real_plug_transitions": transitions,
        "same_frame_overlaps": overlaps,
        "shared_threshold_fusion": True,
    }


def _export_evidence_frames(
    video_path: Path,
    reports: list[tuple[str, str | None, dict[str, Any]]],
    combined: dict[str, Any],
    requests: list[dict[str, Any]],
    output_dir: Path,
) -> list[dict[str, Any]]:
    by_frame: dict[int, dict[str, Any]] = {}
    requested_numbers = {
        number
        for request in requests
        for number in request.get("expected_frame_numbers") or []
        if isinstance(number, int)
    }
    for request in requests:
        number = request.get("source_frame_number")
        if isinstance(number, int):
            requested_numbers.add(number)
    request_reasons = {
        str(request.get("request_id") or ""): str(request.get("reason") or "")
        for request in requests
    }
    for phase, request_id, report in reports:
        for item in report.get("frames") or []:
            if not isinstance(item, dict):
                continue
            number = _frame_number(item)
            if number is None:
                continue
            if phase != "baseline" or item.get("same_frame_overlap") is True:
                requested_numbers.add(number)
            record = by_frame.setdefault(
                number,
                {
                    "frame_number": number,
                    "timestamp_seconds": _timestamp(item),
                    "stages": set(),
                    "evidence_phases": set(),
                    "request_ids": set(),
                    "local_switch_states": set(),
                    "shared_switch_states": set(),
                    "wiring_active": False,
                    "switch_visible": False,
                    "same_frame_overlap": False,
                    "switch_roi_paths": set(),
                },
            )
            record["stages"].add(_stage(item))
            record["evidence_phases"].add(phase)
            if request_id:
                record["request_ids"].add(request_id)
            if item.get("switch_state"):
                record["local_switch_states"].add(str(item["switch_state"]))
            crop = item.get("switch_crop_path")
            if isinstance(crop, str) and crop:
                record["switch_roi_paths"].add(crop)

    # Local scans can use different provisional thresholds. Export the state
    # recomputed by the final shared-threshold reducer as the authoritative one.
    for item in combined.get("frames") or []:
        if not isinstance(item, dict):
            continue
        number = _frame_number(item)
        if number is None or number not in by_frame:
            continue
        record = by_frame[number]
        if item.get("switch_state"):
            record["shared_switch_states"].add(str(item["switch_state"]))
        record["wiring_active"] = record["wiring_active"] or bool(
            item.get("wiring_active")
        )
        record["switch_visible"] = record["switch_visible"] or bool(
            item.get("switch_visible")
        )
        record["same_frame_overlap"] = record["same_frame_overlap"] or bool(
            item.get("same_frame_overlap")
        )
        crop = item.get("switch_crop_path")
        if isinstance(crop, str) and crop:
            record["switch_roi_paths"].add(crop)

    selected = sorted(number for number in requested_numbers if number in by_frame)
    frames_dir = output_dir / "evidence_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"unable to open video for R3 evidence export: {video_path}")
    exported: dict[int, str] = {}
    try:
        for number in selected:
            capture.set(cv2.CAP_PROP_POS_FRAMES, number)
            ok, image = capture.read()
            if not ok or image is None:
                continue
            path = frames_dir / f"frame_{number:08d}.jpg"
            if cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 94]):
                exported[number] = str(path.resolve())
    finally:
        capture.release()

    manifest: list[dict[str, Any]] = []
    for image_group, number in enumerate(selected, start=1):
        record = by_frame[number]
        stages = sorted(record["stages"])
        request_ids = sorted(record["request_ids"])
        manifest.append(
            {
                "image_group": image_group,
                "frame_id": f"r3_frame_{number:08d}",
                "image_group_id": f"r3_{stages[0] if stages else 'wiring'}_{number:08d}",
                "frame_number": number,
                "timestamp_seconds": round(float(record["timestamp_seconds"]), 6),
                "frame_path": exported.get(number),
                "stages": stages,
                "evidence_phases": sorted(record["evidence_phases"]),
                "request_ids": request_ids,
                "trigger_reasons": sorted(
                    {
                        request_reasons[request_id]
                        for request_id in request_ids
                        if request_reasons.get(request_id)
                    }
                ),
                "switch_visible": bool(record["switch_visible"]),
                "switch_states": sorted(record["shared_switch_states"]),
                "local_switch_states": sorted(record["local_switch_states"]),
                "state_source": "shared_current_run_threshold",
                "wiring_active": bool(record["wiring_active"]),
                "same_frame_overlap": bool(record["same_frame_overlap"]),
                "switch_roi_paths": sorted(record["switch_roi_paths"]),
                "fixed_video_roi_used": False,
            }
        )
    return manifest


def _video_metadata(video_path: Path) -> tuple[float, int, float]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"unable to open video: {video_path}")
    try:
        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if source_fps <= 0.0 or frame_count <= 0:
        raise RuntimeError("video metadata is invalid")
    return source_fps, frame_count, frame_count / source_fps


def run_r3_frame_sampling_agent(
    *,
    video_path: Path,
    candidate_windows: list[dict[str, Any]],
    output_dir: Path,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    max_requests_per_round: int = DEFAULT_MAX_REQUESTS_PER_ROUND,
    max_supplemental_frames: int = DEFAULT_MAX_SUPPLEMENTAL_FRAMES,
    analyzer: Analyzer | None = None,
) -> dict[str, Any]:
    """Run baseline R3 and bounded supplemental scans, then keep one binary result."""
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    if not candidate_windows:
        raise ValueError("candidate_windows must not be empty")
    if not 1 <= max_rounds <= 2:
        raise ValueError("max_rounds must be 1 or 2")
    if not 1 <= max_requests_per_round <= 3:
        raise ValueError("max_requests_per_round must be between 1 and 3")
    if not 1 <= max_supplemental_frames <= 96:
        raise ValueError("max_supplemental_frames must be between 1 and 96")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "r3_frame_sampling_agent_report.json"
    if report_path.exists():
        raise FileExistsError(f"refusing to reuse previous Agent result: {report_path}")

    source_fps, frame_count, duration_seconds = _video_metadata(video_path)
    execute = analyzer or analyze_opencv_switch_overlap
    baseline = execute(
        video_path=video_path,
        candidate_windows=candidate_windows,
        output_dir=output_dir / "baseline_5fps",
        sampling_fps=BASE_SAMPLING_FPS,
        roi_mode=ROI_MODE,
        fusion_policy=FUSION_POLICY,
    )
    reports: list[tuple[str, str | None, dict[str, Any]]] = [
        ("baseline", None, baseline)
    ]
    plans: list[dict[str, Any]] = []
    all_requests: list[dict[str, Any]] = []
    known_frames = {
        number
        for item in baseline.get("frames") or []
        if isinstance(item, dict) and (number := _frame_number(item)) is not None
    }
    supplemental_frames: set[int] = set()
    remaining_budget = max_supplemental_frames
    stop_reason = "maximum_rounds_reached"

    if str(baseline.get("decision")) == "fail":
        stop_reason = "baseline_counterexample_confirmed"
    else:
        for round_number in range(1, max_rounds + 1):
            combined = _aggregate_reports([item[2] for item in reports])
            plan = plan_frame_requests(
                report=combined,
                candidate_windows=candidate_windows,
                round_number=round_number,
                duration_seconds=duration_seconds,
                source_fps=source_fps,
                frame_count=frame_count,
                known_frame_numbers=known_frames,
                supplemental_frame_numbers=supplemental_frames,
                max_requests=max_requests_per_round,
                remaining_frame_budget=remaining_budget,
            )
            plans.append(plan)
            write_json(output_dir / "plans" / f"round_{round_number:02d}.json", plan)
            if not plan["requests"]:
                stop_reason = "evidence_sufficient_or_no_new_frames"
                break
            round_new_frames = 0
            for request in plan["requests"]:
                request_id = str(request["request_id"])
                supplemental = execute(
                    video_path=video_path,
                    candidate_windows=[request["candidate_window"]],
                    output_dir=output_dir / "supplemental" / request_id,
                    sampling_fps=SUPPLEMENTAL_SAMPLING_FPS,
                    roi_mode=ROI_MODE,
                    fusion_policy=FUSION_POLICY,
                )
                actual_frames = {
                    number
                    for item in supplemental.get("frames") or []
                    if isinstance(item, dict)
                    and (number := _frame_number(item)) is not None
                }
                new_frames = actual_frames - known_frames
                known_frames.update(actual_frames)
                supplemental_frames.update(actual_frames)
                round_new_frames += len(new_frames)
                remaining_budget = max(0, remaining_budget - len(new_frames))
                request["actual_frame_count"] = len(actual_frames)
                request["actual_new_frame_count"] = len(new_frames)
                request["actual_new_frame_numbers"] = sorted(new_frames)
                request["report_path"] = supplemental.get("report_path")
                reports.append(("supplemental", request_id, supplemental))
                all_requests.append(request)
                if remaining_budget <= 0:
                    stop_reason = "supplemental_frame_budget_exhausted"
                    break
            combined_after_round = _aggregate_reports([item[2] for item in reports])
            if combined_after_round["decision"] == "fail":
                stop_reason = "shared_threshold_counterexample_confirmed"
                break
            if remaining_budget <= 0:
                break
            if round_new_frames == 0:
                stop_reason = "no_new_frames"
                break

    report_values = [item[2] for item in reports]
    combined = _aggregate_reports(report_values)
    final_quality = assess_evidence(combined)
    decision = str(combined["decision"])
    if decision == "fail":
        overlap_scores = []
        for item in combined.get("same_frame_overlaps") or []:
            identity = _number(item.get("switch_identity_score"), 0.55)
            transition_scores = [
                _number(transition.get("confidence"), 0.55)
                for transition in (
                    item.get("wiring_active_transitions")
                    or item.get("plug_transitions")
                    or []
                )
                if isinstance(transition, dict)
            ]
            overlap_scores.append(min(identity, max(transition_scores or [0.55])))
        confidence = max(overlap_scores or [0.55])
        reason = "shared_threshold_same_frame_persistent_closed_switch_and_wiring_active"
    else:
        confidence = max(_number(item.get("confidence"), 0.55) for item in report_values)
        if final_quality["reasons"]:
            confidence = min(confidence, 0.55)
            reason = "no_counterexample_after_bounded_sampling_with_low_evidence_quality"
        else:
            reason = "no_counterexample_after_bounded_adaptive_sampling"
    evidence_frames = _export_evidence_frames(
        video_path, reports, combined, all_requests, output_dir
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "agent_version": AGENT_VERSION,
        "decision": decision,
        "predicted_score": 1 if decision == "pass" else 0,
        "confidence": round(float(max(0.0, min(confidence, 0.99))), 4),
        "reason": reason,
        "stop_reason": stop_reason,
        "source_video_path": str(video_path.resolve()),
        "candidate_windows": candidate_windows,
        "sampling_policy": {
            "baseline_sampling_fps": BASE_SAMPLING_FPS,
            "supplemental_sampling_fps": SUPPLEMENTAL_SAMPLING_FPS,
            "phase_offset_seconds": PHASE_OFFSET_SECONDS,
            "supplemental_scans_are_independent": True,
            "cross_scan_persistence_fusion": False,
            "shared_current_run_state_threshold": True,
            "outside_stage_frames_scored": False,
            "duplicate_supplemental_frames_decoded": False,
            "max_rounds": max_rounds,
            "max_requests_per_round": max_requests_per_round,
            "max_supplemental_frames": max_supplemental_frames,
        },
        "initial_evidence_quality": assess_evidence(baseline),
        "final_evidence_quality": final_quality,
        "request_rounds": plans,
        "requests": all_requests,
        "request_count": len(all_requests),
        "supplemental_actual_new_frame_count": sum(
            int(item.get("actual_new_frame_count") or 0) for item in all_requests
        ),
        "evidence_frames": evidence_frames,
        "evidence_frame_count": len(evidence_frames),
        "baseline_report_path": baseline.get("report_path"),
        "supplemental_report_paths": [
            item[2].get("report_path") for item in reports if item[0] == "supplemental"
        ],
        "original_algorithm_version": baseline.get("implementation_version"),
        "original_algorithm_fingerprint": baseline.get("implementation_fingerprint"),
        "shared_threshold_fusion": {
            "enabled": True,
            "threshold": combined.get("switch_state_threshold"),
            "cluster_centers": combined.get("switch_state_cluster_centers"),
            "source": combined.get("switch_state_threshold_source"),
        },
        "selection_basis": "current_video_observed_situation_only",
        "video_id_used_for_routing": False,
        "historical_artifacts_used": False,
        "fixed_video_roi_used": False,
        "excel_accessed": False,
        "ground_truth_sent_to_model": False,
        "qwen_used_for_decision": False,
        "final_result_is_binary": decision in {"pass", "fail"},
    }
    write_json(report_path, result)
    reopened = read_json(report_path)
    evidence_paths = [
        Path(item["frame_path"])
        for item in reopened.get("evidence_frames") or []
        if isinstance(item, dict) and isinstance(item.get("frame_path"), str)
    ]
    evidence_items = [
        item for item in reopened.get("evidence_frames") or [] if isinstance(item, dict)
    ]
    if (
        reopened.get("decision") != decision
        or reopened.get("final_result_is_binary") is not True
        or reopened.get("request_count") != len(reopened.get("requests") or [])
        or int(reopened.get("supplemental_actual_new_frame_count") or 0)
        > max_supplemental_frames
        or reopened.get("evidence_frame_count")
        != len(reopened.get("evidence_frames") or [])
        or len(evidence_paths) != len(evidence_items)
        or not all(path.is_file() for path in evidence_paths)
        or reopened.get("video_id_used_for_routing") is not False
        or reopened.get("historical_artifacts_used") is not False
        or reopened.get("fixed_video_roi_used") is not False
    ):
        raise RuntimeError("R3 frame sampling Agent report failed reopen verification")
    return {**result, "report_path": str(report_path.resolve())}
