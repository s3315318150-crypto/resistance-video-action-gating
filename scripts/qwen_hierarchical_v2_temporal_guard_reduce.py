#!/usr/bin/env python3
"""Temporal rejection guard layered on the unchanged v2 reducer."""

from __future__ import annotations

from typing import Any

import qwen_hierarchical_v1_reduce as base_reduce


GUARDED_REJECTION_REASONS = {"duplicate", "conflicts_with_stronger_evidence"}


def _overlaps(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return int(left["first_frame_number"]) <= int(right["last_frame_number"]) and int(
        right["first_frame_number"]
    ) <= int(left["last_frame_number"])


def _has_legal_rejection_witness(
    event: dict[str, Any],
    reason: str,
    originally_accepted_events: list[dict[str, Any]],
) -> bool:
    for witness in originally_accepted_events:
        if not _overlaps(event, witness):
            continue
        same_action = event.get("action_type") == witness.get("action_type")
        if reason == "duplicate" and same_action:
            return True
        if reason == "conflicts_with_stronger_evidence" and not same_action:
            return True
    return False


def _actual_conflicts(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "event_ids": list(conflict["event_ids"]),
            "resolution": "本地仅将源帧时间真正重叠且动作标签不同的事件视为冲突。",
        }
        for conflict in base_reduce.find_temporal_conflicts(events)
    ]


def salvage_reduce_response_with_temporal_guard(
    events: list[dict[str, Any]],
    reduce_result: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Restore normal pre-terminal events rejected without an overlapping witness."""
    repaired, repairs = base_reduce.salvage_reduce_response(events, reduce_result)
    if repaired is None:
        return None, repairs

    event_by_id = {str(event["event_id"]): event for event in events}
    accepted_ids = {
        event_id
        for event_id in repaired.get("accepted_event_ids", [])
        if isinstance(event_id, str) and event_id in event_by_id
    }
    originally_accepted_events = [event_by_id[event_id] for event_id in accepted_ids]
    terminal_id = repaired.get("terminal_cleanup_event_id")
    terminal = event_by_id.get(terminal_id) if isinstance(terminal_id, str) else None
    terminal_start = int(terminal["first_frame_number"]) if terminal is not None else None

    restored_ids: list[str] = []
    retained_rejections: list[dict[str, Any]] = []
    for rejection in repaired.get("rejected_events", []):
        event_id = rejection.get("event_id") if isinstance(rejection, dict) else None
        event = event_by_id.get(event_id) if isinstance(event_id, str) else None
        reason = rejection.get("reason") if isinstance(rejection, dict) else None
        entirely_before_terminal = (
            event is not None
            and (terminal_start is None or int(event["last_frame_number"]) < terminal_start)
        )
        should_guard = event is not None and reason in GUARDED_REJECTION_REASONS and entirely_before_terminal
        if should_guard and not _has_legal_rejection_witness(
            event,
            str(reason),
            originally_accepted_events,
        ):
            accepted_ids.add(str(event_id))
            restored_ids.append(str(event_id))
            repairs.append(
                {
                    "reason": "non_overlapping_rejection_restored",
                    "event_id": str(event_id),
                    "original_rejection_reason": str(reason),
                }
            )
            continue
        retained_rejections.append(rejection)

    ordered_ids = [str(event["event_id"]) for event in events]
    repaired["accepted_event_ids"] = [event_id for event_id in ordered_ids if event_id in accepted_ids]
    rejection_by_id = {
        str(item["event_id"]): item
        for item in retained_rejections
        if isinstance(item, dict) and isinstance(item.get("event_id"), str)
    }
    repaired["rejected_events"] = [
        rejection_by_id[event_id]
        for event_id in ordered_ids
        if event_id in rejection_by_id and event_id not in accepted_ids
    ]
    repaired["conflicts"] = _actual_conflicts(events)
    repaired["temporal_guard"] = {
        "policy": "restore_preterminal_duplicate_or_conflict_rejection_without_overlapping_accepted_witness",
        "restored_event_ids": restored_ids,
        "restored_event_count": len(restored_ids),
        "terminal_cleanup_event_id": terminal_id,
    }
    if restored_ids:
        note = f"本地时序保护恢复了 {len(restored_ids)} 个被非重叠事件错误覆盖的动作。"
        uncertainty = str(repaired.get("uncertainty", "")).strip()
        repaired["uncertainty"] = f"{uncertainty} {note}".strip()[:160]
    return repaired, repairs


def select_events_with_temporal_guard(
    events: list[dict[str, Any]],
    reduce_result: dict[str, Any] | None,
    preserve_equal_confidence: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected, selection = base_reduce.select_events(
        events,
        reduce_result,
        preserve_equal_confidence=preserve_equal_confidence,
    )
    if reduce_result is not None:
        guard = reduce_result.get("temporal_guard")
        selection["mode"] = "qwen_global_reduce_with_temporal_rejection_guard"
        selection["temporal_guard"] = guard if isinstance(guard, dict) else {
            "policy": "temporal_rejection_guard_not_applied",
            "restored_event_ids": [],
            "restored_event_count": 0,
        }
        if selection["temporal_guard"].get("restored_event_count", 0):
            selection["needs_review"] = True
    return selected, selection
