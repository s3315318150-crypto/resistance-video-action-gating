"""Current-run adaptive frame sampling for rubric 1.

The sampler only consumes the frames and stage windows produced by the current
run.  It uses a cheap scout to rank temporal evidence, then asks the existing
R1 decoder for dense transition frames and dynamic ROI sheets.  The rubric
reducer remains responsible for the final binary decision.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any


AGENT_VERSION = "r1_frame_sampling_agent.v2"
SCHEMA_VERSION = "resistance_agent_r1_frame_sampling.v1"
SCOUT_FPS = 2.0
TRANSITION_FPS = 5.0
TRANSITION_RADIUS_SECONDS = 1.0
DEFAULT_STABLE_PER_STAGE = 2
DEFAULT_RECOVERY_PER_STAGE = 1
DEFAULT_TRANSITION_ANCHORS = 4
DEFAULT_SUPPLEMENTAL_FRAMES = 12


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _timestamp(item: dict[str, Any]) -> float:
    return _number(item.get("timestamp_seconds"))


def _stage_key(item: dict[str, Any]) -> tuple[str, str, str]:
    """Keep each current-run coarse window in its own selection bucket.

    A long action stage is deliberately split into adjacent coarse windows.  A
    stage/run-only key collapses those windows and lets later sharp frames
    displace an early wiring change, so the window identifier is part of the
    local sampling key.
    """
    return (
        str(item.get("stage") or "broad_search"),
        str(item.get("stage_run") or "1"),
        str(item.get("window_id") or "window_1"),
    )


def _stage_run_key(item: dict[str, Any]) -> tuple[str, str]:
    return (
        str(item.get("stage") or "broad_search"),
        str(item.get("stage_run") or "1"),
    )


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction))))
    return ordered[index]


def _mad(values: list[float], center: float) -> float:
    return _percentile([abs(value - center) for value in values], 0.5)


def _candidate_count(item: dict[str, Any]) -> int:
    candidates = item.get("device_localizations")
    return len(candidates) if isinstance(candidates, list) else 0


def _layout_change(previous: dict[str, Any] | None, current: dict[str, Any]) -> float:
    if previous is None:
        return 0.0
    old = previous.get("device_localizations") or []
    new = current.get("device_localizations") or []
    old_ids = {
        str(item.get("track_id") or item.get("candidate_id"))
        for item in old
        if isinstance(item, dict)
    }
    new_ids = {
        str(item.get("track_id") or item.get("candidate_id"))
        for item in new
        if isinstance(item, dict)
    }
    if not old_ids and not new_ids:
        return float(abs(len(new) - len(old)))
    union = old_ids | new_ids
    return 1.0 - len(old_ids & new_ids) / max(1, len(union))


def enrich_scout_frames(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add normalized, identity-independent evidence scores to scout frames."""
    ordered = sorted((dict(item) for item in frames if isinstance(item, dict)), key=_timestamp)
    motion_values = [_number(item.get("motion_score")) for item in ordered]
    sharpness_values = [_number(item.get("sharpness")) for item in ordered]
    coverage_values = [float(_candidate_count(item)) for item in ordered]
    motion_center = _percentile(motion_values, 0.5)
    motion_scale = max(_mad(motion_values, motion_center), 1e-6)
    sharpness_scale = max(_percentile(sharpness_values, 0.9), 1e-6)
    coverage_scale = max(_percentile(coverage_values, 0.9), 1.0)
    previous: dict[str, Any] | None = None
    previous_window_id: str | None = None
    output: list[dict[str, Any]] = []
    for item in ordered:
        window_id = str(item.get("window_id") or "")
        if previous_window_id is not None and window_id != previous_window_id:
            previous = None
        motion = _number(item.get("motion_score"))
        sharpness = _number(item.get("sharpness"))
        coverage = float(_candidate_count(item))
        change = _layout_change(previous, item)
        enriched = {
            **item,
            "scout_motion": round(motion, 4),
            "scout_sharpness": round(sharpness, 4),
            "scout_layout_change": round(change, 4),
            "scout_transition_score": round(
                max(0.0, (motion - motion_center) / motion_scale) + change,
                4,
            ),
            "scout_view_score": round(
                0.55 * min(1.0, sharpness / sharpness_scale)
                + 0.45 * min(1.0, coverage / coverage_scale),
                4,
            ),
        }
        output.append(enriched)
        previous = item
        previous_window_id = window_id
    return output


def _separated(items: list[dict[str, Any]], minimum_seconds: float) -> list[dict[str, Any]]:
    chosen: list[dict[str, Any]] = []
    for item in items:
        if all(abs(_timestamp(item) - _timestamp(other)) >= minimum_seconds for other in chosen):
            chosen.append(item)
    return chosen


def _stable_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    motion = [_number(item.get("scout_motion")) for item in rows]
    threshold = _percentile(motion, 0.45)
    low_motion = [item for item in rows if _number(item.get("scout_motion")) <= threshold]
    if not low_motion:
        low_motion = [min(rows, key=lambda item: _number(item.get("scout_motion")))]
    plateaus: list[list[dict[str, Any]]] = []
    for item in low_motion:
        if not plateaus or _timestamp(item) - _timestamp(plateaus[-1][-1]) > 0.76:
            plateaus.append([item])
        else:
            plateaus[-1].append(item)
    representatives: list[dict[str, Any]] = []
    for plateau in plateaus:
        representatives.append(
            max(
                plateau,
                key=lambda item: (
                    _number(item.get("scout_view_score")),
                    _timestamp(item),
                ),
            )
        )
    return representatives


def _transition_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(rows) < 2:
        return []
    scores = [_number(item.get("scout_transition_score")) for item in rows]
    threshold = _percentile(scores, 0.7)
    candidates: list[dict[str, Any]] = []
    for index, item in enumerate(rows):
        score = _number(item.get("scout_transition_score"))
        left = scores[index - 1] if index else -1.0
        right = scores[index + 1] if index + 1 < len(scores) else -1.0
        if score >= threshold and score >= left and score >= right and score > 0.0:
            candidates.append(item)
    if not candidates:
        candidates = [max(rows, key=lambda item: _number(item.get("scout_transition_score")))]
    return sorted(candidates, key=lambda item: _number(item.get("scout_transition_score")), reverse=True)


def _recovery_candidates(rows: list[dict[str, Any]], excluded: set[tuple[str, int]]) -> list[dict[str, Any]]:
    return sorted(
        [
            item
            for item in rows
            if (str(item.get("window_id")), int(round(_timestamp(item) * 1000))) not in excluded
        ],
        key=lambda item: (_number(item.get("scout_view_score")), _number(item.get("scout_sharpness"))),
        reverse=True,
    )


def _stage_diverse_transition_anchors(
    candidates: list[dict[str, Any]], limit: int
) -> list[dict[str, Any]]:
    ranked = sorted(
        candidates,
        key=lambda item: _number(item.get("scout_transition_score")),
        reverse=True,
    )
    by_stage_run: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in ranked:
        by_stage_run[_stage_run_key(item)].append(item)
    # Reserve the strongest late-half transition in the first coarse window of
    # every wiring run.  Initial placement peaks occur near the window start;
    # the late-half peak is more likely to show the first completed connection
    # state, which must not be displaced by stronger motion much later.
    first_window_heads: list[dict[str, Any]] = []
    for items in by_stage_run.values():
        first_window_id = min(
            {str(item.get("window_id") or "") for item in items},
            key=lambda window_id: min(
                _number(item.get("start_seconds"), _timestamp(item))
                for item in items
                if str(item.get("window_id") or "") == window_id
            ),
        )
        window_items = [
            item for item in items if str(item.get("window_id") or "") == first_window_id
        ]
        window_start = min(_number(item.get("start_seconds"), _timestamp(item)) for item in window_items)
        window_end = max(_number(item.get("end_seconds"), _timestamp(item)) for item in window_items)
        midpoint = (window_start + window_end) / 2.0
        late_items = [
            item
            for item in window_items
            if midpoint <= _timestamp(item) <= window_end + 1e-6
        ]
        pool = late_items or window_items
        first_window_heads.append(
            max(pool, key=lambda item: _number(item.get("scout_transition_score")))
        )
    chosen = _separated(
        sorted(
            first_window_heads,
            key=lambda item: _number(item.get("scout_transition_score")),
            reverse=True,
        ),
        1.5,
    )[:limit]
    for item in ranked:
        if len(chosen) >= limit:
            break
        if item in chosen:
            continue
        if all(abs(_timestamp(item) - _timestamp(other)) >= 1.5 for other in chosen):
            chosen.append(item)
    return chosen


def select_initial_evidence(
    frames: list[dict[str, Any]],
    *,
    stable_per_stage_run: int = DEFAULT_STABLE_PER_STAGE,
    recovery_per_stage_run: int = DEFAULT_RECOVERY_PER_STAGE,
    max_transition_anchors: int = DEFAULT_TRANSITION_ANCHORS,
) -> dict[str, Any]:
    """Select diverse current-run evidence without video-specific branches."""
    if stable_per_stage_run <= 0 or recovery_per_stage_run < 0 or max_transition_anchors <= 0:
        raise ValueError("frame selection limits must be positive")
    enriched = enrich_scout_frames(frames)
    by_stage: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in enriched:
        by_stage[_stage_key(item)].append(item)
    stable: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    recovery: list[dict[str, Any]] = []
    for rows in by_stage.values():
        rows.sort(key=_timestamp)
        stage_transitions = _transition_candidates(rows)
        transition_times = [_timestamp(item) for item in stage_transitions]
        stable_candidates = _stable_candidates(rows)
        stable_candidates.sort(
            key=lambda item: (
                any(0.2 <= _timestamp(item) - point <= 3.0 for point in transition_times),
                _number(item.get("scout_view_score")),
                _timestamp(item),
            ),
            reverse=True,
        )
        stable.extend(stable_candidates[:stable_per_stage_run])
        transitions.extend(stage_transitions)
    transition_anchors = _stage_diverse_transition_anchors(
        transitions, max_transition_anchors
    )
    selected_keys = {
        (str(item.get("window_id")), int(round(_timestamp(item) * 1000)))
        for item in stable
    }
    for rows in by_stage.values():
        recovery.extend(_recovery_candidates(rows, selected_keys)[:recovery_per_stage_run])
    selected: list[dict[str, Any]] = []
    for item in stable:
        selected.append({**item, "frame_agent_role": "stable_topology", "temporal_role": "stable_candidate", "evidence_phase": "coarse_scan", "selection_reason": "low_motion_plateau"})
    for item in recovery:
        key = (str(item.get("window_id")), int(round(_timestamp(item) * 1000)))
        if key in selected_keys:
            continue
        selected.append({**item, "frame_agent_role": "view_recovery", "temporal_role": "process_scan", "evidence_phase": "coarse_scan", "selection_reason": "current_frame_view_score"})
        selected_keys.add(key)
    for item in transition_anchors:
        selected.append({**item, "frame_agent_role": "connection_transition", "temporal_role": "transition_anchor", "evidence_phase": "coarse_scan", "selection_reason": "motion_or_dynamic_candidate_change"})
    selected.sort(key=_timestamp)
    for group, item in enumerate(selected, start=1):
        item["image_group"] = group
    return {
        "schema_version": SCHEMA_VERSION,
        "agent_version": AGENT_VERSION,
        "selection_basis": "current_video_observed_situation_only",
        "stable_frames": [item for item in selected if item.get("frame_agent_role") == "stable_topology"],
        "transition_anchors": [item for item in selected if item.get("frame_agent_role") == "connection_transition"],
        "recovery_frames": [item for item in selected if item.get("frame_agent_role") == "view_recovery"],
        "selected_frames": selected,
        "scanned_frame_count": len(enriched),
        "selected_frame_count": len(selected),
        "video_id_used_for_routing": False,
        "historical_artifacts_used": False,
        "fixed_video_roi_used": False,
    }


def transition_burst_samples(
    anchors: list[dict[str, Any]],
    *,
    duration_seconds: float,
    fps: float = TRANSITION_FPS,
    radius_seconds: float = TRANSITION_RADIUS_SECONDS,
    phase_offset_seconds: float = 0.0,
) -> list[dict[str, Any]]:
    """Create adjacent before/during/after samples around current-run peaks."""
    if fps <= 0 or radius_seconds <= 0:
        raise ValueError("transition sampling parameters must be positive")
    output: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for index, anchor in enumerate(anchors, start=1):
        center = _timestamp(anchor)
        start = max(_number(anchor.get("start_seconds")), center - radius_seconds)
        end = min(
            duration_seconds,
            _number(anchor.get("review_end_seconds"), _number(anchor.get("end_seconds"), duration_seconds)),
            center + radius_seconds,
        )
        cursor = min(end, start + max(0.0, phase_offset_seconds))
        while cursor <= end + 1e-6:
            timestamp = round(cursor, 3)
            key = (str(anchor.get("window_id") or ""), int(round(timestamp * 1000)))
            if key not in seen:
                seen.add(key)
                relative = timestamp - center
                position = "before" if relative < -0.25 else "after" if relative > 0.25 else "during"
                output.append(
                    {
                        **{key: value for key, value in anchor.items() if not key.endswith("_path")},
                        "window_id": f"r1_transition_{index:02d}",
                        "source_window_id": str(anchor.get("window_id") or ""),
                        "timestamp_seconds": timestamp,
                        "frame_agent_role": "connection_transition",
                        "transition_position": position,
                        "transition_anchor_seconds": round(center, 3),
                        "temporal_role": "dense_direct_confirmation",
                        "evidence_phase": "dense_confirmation",
                        "selection_reason": "current_run_transition_burst",
                    }
                )
            cursor += 1.0 / fps
    return output


def _terminal_identifier(item: dict[str, Any], device: str) -> str:
    """Return a visible current-observation terminal label, if one exists."""
    field = "ammeter_terminal_id" if device == "ammeter" else "battery_terminal_id"
    value = item.get(field) or item.get("terminal_label")
    if not isinstance(value, str):
        return ""
    value = value.strip().lower()
    if not value or value in {"unknown", "unclear", "none", "empty"}:
        return ""
    return value


def _direct_terminal_profile(observation: dict[str, Any]) -> dict[str, Any]:
    """Count unique direct endpoint IDs, not mirrored terminal descriptions."""
    ammeter_ids: set[str] = set()
    battery_ids: set[str] = set()
    direct_rows = 0
    terminals = observation.get("terminal_evidence")
    if isinstance(terminals, list):
        for item in terminals:
            if not isinstance(item, dict):
                continue
            if str(item.get("path_relation") or "").strip().lower() != "direct":
                continue
            device = str(item.get("device") or "").strip().lower()
            far_endpoint = str(item.get("far_endpoint") or "").strip().lower()
            if device == "ammeter" and "battery" in far_endpoint:
                identifier = _terminal_identifier(item, "ammeter")
                if identifier:
                    ammeter_ids.add(identifier)
                    direct_rows += 1
            elif device == "battery_holder" and "ammeter" in far_endpoint:
                identifier = _terminal_identifier(item, "battery_holder")
                if identifier:
                    battery_ids.add(identifier)
                    direct_rows += 1
    return {
        "ammeter_terminal_ids": ammeter_ids,
        "battery_terminal_ids": battery_ids,
        "direct_rows": direct_rows,
        "two_by_two": len(ammeter_ids) >= 2 and len(battery_ids) >= 2,
    }


def _observation_needs_supplement(observation: dict[str, Any]) -> str | None:
    direct_state = observation.get("direct_across_state") in {"candidate", "confirmed"}
    profile = _direct_terminal_profile(observation)
    # One direct ammeter-to-battery edge is normal series wiring.  Mirrored
    # ammeter/battery rows for that same edge must not be treated as two edges.
    direct = direct_state or bool(profile["two_by_two"])
    if direct:
        return "suspected_direct_connection"
    if observation.get("hands_or_plugs") == "occluded" or observation.get("topology_visibility") in {"partial", "insufficient"}:
        return "occluded_view"
    return None


def _supplement_priority(observation: dict[str, Any], frame: dict[str, Any]) -> tuple[float, float, float]:
    """Rank current observations so later strong evidence cannot be crowded out."""
    profile = _direct_terminal_profile(observation)
    state = str(observation.get("direct_across_state") or "").strip().lower()
    state_score = {"confirmed": 4.0, "candidate": 3.0}.get(state, 0.0)
    endpoint_score = 2.0 if profile["two_by_two"] else 0.0
    visibility_score = 1.0 if observation.get("topology_visibility") == "sufficient" else 0.0
    return (
        state_score + endpoint_score + visibility_score,
        _number(frame.get("scout_transition_score")),
        _number(frame.get("scout_view_score")),
    )


def plan_supplemental_round(
    observations: list[dict[str, Any]],
    frames: list[dict[str, Any]],
    *,
    duration_seconds: float,
    max_frames: int = DEFAULT_SUPPLEMENTAL_FRAMES,
    fps: float = TRANSITION_FPS,
    radius_seconds: float = TRANSITION_RADIUS_SECONDS,
) -> dict[str, Any]:
    """Plan at most one evidence round from the first Qwen observations."""
    by_group = {int(item.get("image_group")): item for item in frames if isinstance(item.get("image_group"), int)}
    reasons: dict[str, list[tuple[tuple[float, float, float], dict[str, Any]]]] = defaultdict(list)
    for observation in observations:
        if not isinstance(observation, dict):
            continue
        reason = _observation_needs_supplement(observation)
        frame = by_group.get(int(observation.get("image_group") or -1))
        if reason and frame is not None:
            reasons[reason].append((_supplement_priority(observation, frame), frame))
    # If no stable observation was obtained, use the best current stable frame.
    stable_supported = any(
        item.get("stable_state") is True and item.get("hands_or_plugs") == "hands_away"
        for item in observations
        if isinstance(item, dict)
    )
    if not stable_supported:
        candidates = [item for item in frames if item.get("frame_agent_role") == "stable_topology"]
        if candidates:
            best = max(candidates, key=lambda item: _number(item.get("scout_view_score")))
            reasons["missing_stable_state"].append(((0.0, 0.0, _number(best.get("scout_view_score"))), best))
    anchors: list[dict[str, Any]] = []
    ranked_reasons: list[tuple[tuple[float, float, float], str, dict[str, Any]]] = []
    for reason, entries in reasons.items():
        for priority, frame in entries:
            ranked_reasons.append((priority, reason, frame))
    ranked_reasons.sort(key=lambda item: item[0], reverse=True)
    seen_anchor_keys: set[tuple[str, int]] = set()
    for priority, reason, item in ranked_reasons:
        key = (str(item.get("window_id") or ""), int(round(_timestamp(item) * 1000)))
        if key in seen_anchor_keys:
            continue
        seen_anchor_keys.add(key)
        anchors.append({**item, "supplemental_reason": reason, "supplemental_priority": list(priority)})
        # Two anchors are enough for one bounded supplemental round, while the
        # ranking above ensures the strongest current evidence gets the slot.
        if len(anchors) >= 2:
            break
    burst = transition_burst_samples(
        anchors,
        duration_seconds=duration_seconds,
        fps=fps,
        radius_seconds=radius_seconds,
        phase_offset_seconds=0.1,
    )
    for item in burst:
        reason = str(item.get("supplemental_reason") or "")
        if reason == "missing_stable_state":
            item["frame_agent_role"] = "stable_topology"
            item["temporal_role"] = "stable_candidate"
            item["evidence_phase"] = "supplemental_round"
        elif reason == "occluded_view":
            item["frame_agent_role"] = "view_recovery"
            item["temporal_role"] = "process_scan"
            item["evidence_phase"] = "supplemental_round"
        item["window_id"] = str(item["window_id"]).replace("r1_transition_", "r1_supplemental_")
    burst = burst[:max(0, max_frames)]
    return {
        "schema_version": "resistance_agent_r1_supplemental_round.v1",
        "round_number": 1 if burst else 0,
        "selection_basis": "current_run_qwen_observations_only",
        "reasons": sorted(reasons),
        "frames": burst,
        "frame_count": len(burst),
        "max_rounds": 1,
        "video_id_used_for_routing": False,
        "historical_artifacts_used": False,
        "fixed_video_roi_used": False,
    }
