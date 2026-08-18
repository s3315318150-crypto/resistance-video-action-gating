#!/usr/bin/env python3
"""Screenshot-compatible Temporal Guard reducer for the seven-stage pipeline.

The archived screenshot result used a small, deterministic repair layer on
top of the v1 Map/Reduce contract.  This module keeps that behavior explicit:
rejected pre-terminal events are restored only when no accepted event with a
real source-frame overlap can justify the rejection, and the first accepted
cleanup remains an absorbing terminal state.
"""

from __future__ import annotations

import math
from typing import Any

import qwen_hierarchical_v1_reduce as base_reduce


GUARDED_REJECTION_REASONS = {"duplicate", "conflicts_with_stronger_evidence"}


def _overlaps(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return int(left["first_frame_number"]) <= int(right["last_frame_number"]) and int(
        right["first_frame_number"]
    ) <= int(left["last_frame_number"])


def _actual_conflicts(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    conflicts = []
    for conflict in base_reduce.find_temporal_conflicts(events):
        conflicts.append(
            {
                "event_ids": list(conflict["event_ids"]),
                "resolution": "only source-frame-overlapping events with different labels are conflicts",
            }
        )
    return conflicts


def _valid_reduce_result(events: list[dict[str, Any]], result: dict[str, Any] | None) -> bool:
    if not isinstance(result, dict) or not isinstance(result.get("accepted_event_ids"), list):
        return False
    known = {str(event["event_id"]) for event in events}
    return all(isinstance(item, str) and item in known for item in result["accepted_event_ids"])


def salvage_reduce_response_with_screenshot_guard(
    events: list[dict[str, Any]],
    reduce_result: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Repair the screenshot-v2 contract without the newer cleanup demotion."""
    if not _valid_reduce_result(events, reduce_result):
        return None, []
    assert isinstance(reduce_result, dict)
    event_by_id = {str(event["event_id"]): event for event in events}
    ordered_ids = [str(event["event_id"]) for event in events]
    accepted_ids = [
        str(event_id)
        for event_id in reduce_result["accepted_event_ids"]
        if str(event_id) in event_by_id
    ]
    accepted_set = set(accepted_ids)
    repairs: list[dict[str, Any]] = []

    rejected_by_id: dict[str, dict[str, Any]] = {}
    raw_rejected = reduce_result.get("rejected_events", [])
    if isinstance(raw_rejected, list):
        for item in raw_rejected:
            if not isinstance(item, dict):
                continue
            event_id = item.get("event_id")
            if not isinstance(event_id, str) or event_id not in event_by_id or event_id in accepted_set:
                continue
            reason = item.get("reason")
            if reason not in {
                "duplicate",
                "conflicts_with_stronger_evidence",
                "insufficient_visual_evidence",
                "outside_locked_segment",
                "post_terminal_cleanup",
                "other",
            }:
                reason = "other"
                repairs.append({"reason": "invalid_rejection_reason_normalized", "event_id": event_id})
            explanation = item.get("explanation")
            if not isinstance(explanation, str) or not explanation.strip():
                explanation = "模型未提供有效拒绝说明。"
            rejected_by_id[event_id] = {
                "event_id": event_id,
                "reason": reason,
                "explanation": explanation[:160],
            }

    originally_accepted = [event_by_id[event_id] for event_id in accepted_ids]
    restored_ids: list[str] = []
    terminal_id = reduce_result.get("terminal_cleanup_event_id")
    terminal = (
        event_by_id.get(terminal_id)
        if isinstance(terminal_id, str) and terminal_id in accepted_set
        else None
    )
    terminal_start = int(terminal["first_frame_number"]) if terminal is not None else None
    for event_id, rejection in list(rejected_by_id.items()):
        event = event_by_id[event_id]
        before_terminal = terminal_start is None or int(event["last_frame_number"]) < terminal_start
        reason = rejection["reason"]
        witness = any(
            _overlaps(event, accepted_event)
            and (
                (reason == "duplicate" and event.get("action_type") == accepted_event.get("action_type"))
                or (
                    reason == "conflicts_with_stronger_evidence"
                    and event.get("action_type") != accepted_event.get("action_type")
                )
            )
            for accepted_event in originally_accepted
        )
        if before_terminal and reason in GUARDED_REJECTION_REASONS and not witness:
            accepted_set.add(event_id)
            restored_ids.append(event_id)
            del rejected_by_id[event_id]
            repairs.append(
                {
                    "reason": "non_overlapping_rejection_restored",
                    "event_id": event_id,
                    "original_rejection_reason": reason,
                }
            )

    # The screenshot behavior treats the first accepted cleanup as terminal;
    # it does not demote that cleanup when later Map windows contain actions.
    cleanup_events = [
        event_by_id[event_id]
        for event_id in accepted_set
        if event_by_id[event_id].get("action_type") == "cleanup_action"
    ]
    if terminal is not None and terminal.get("action_type") != "cleanup_action":
        repairs.append({"reason": "invalid_terminal_cleanup_demoted", "event_id": terminal_id})
        terminal = None
        terminal_id = None
    if terminal is None and cleanup_events:
        terminal = min(cleanup_events, key=lambda event: int(event["first_frame_number"]))
        terminal_id = str(terminal["event_id"])
        repairs.append({"reason": "terminal_cleanup_promoted_by_cleanup_barrier", "event_id": terminal_id})

    ignored_noise_events: list[dict[str, Any]] = []
    if terminal is not None and terminal_id is not None:
        terminal_start = int(terminal["first_frame_number"])
        for event in events:
            event_id = str(event["event_id"])
            if event_id == terminal_id or int(event["last_frame_number"]) < terminal_start:
                continue
            accepted_set.discard(event_id)
            rejected_by_id[event_id] = {
                "event_id": event_id,
                "reason": "post_terminal_cleanup",
                "explanation": "最终整理作为不可逆终态，之后事件按录像噪声忽略。",
            }
            ignored_noise_events.append(
                {
                    **event,
                    "label": "ignored_noise_post_experiment",
                    "ignored_label": "ignored_noise_post_experiment",
                    "terminal_cleanup_event_id": terminal_id,
                }
            )
        if ignored_noise_events:
            repairs.append(
                {
                    "reason": "terminal_cleanup_hard_stop",
                    "event_id": terminal_id,
                    "ignored_event_ids": [str(item["event_id"]) for item in ignored_noise_events],
                }
            )

    # Preserve every current-run event in either accepted or rejected output.
    for event_id in ordered_ids:
        if event_id not in accepted_set and event_id not in rejected_by_id:
            rejected_by_id[event_id] = {
                "event_id": event_id,
                "reason": "other",
                "explanation": "事件未被 Reduce 明确选择，保留为诊断拒绝。",
            }
            repairs.append({"reason": "omitted_event_locally_quarantined", "event_id": event_id})

    raw_confidence = reduce_result.get("confidence", 0.0)
    confidence = float(raw_confidence) if isinstance(raw_confidence, (int, float)) and not isinstance(raw_confidence, bool) else 0.0
    confidence = min(1.0, max(0.0, confidence)) if math.isfinite(confidence) else 0.0
    uncertainty = reduce_result.get("uncertainty")
    uncertainty_text = uncertainty.strip() if isinstance(uncertainty, str) else ""
    if restored_ids:
        uncertainty_text = f"{uncertainty_text} screenshot temporal guard restored {len(restored_ids)} event(s).".strip()
    if ignored_noise_events:
        uncertainty_text = f"{uncertainty_text} cleanup remains terminal.".strip()
    repaired = {
        "accepted_event_ids": [event_id for event_id in ordered_ids if event_id in accepted_set],
        "rejected_events": [rejected_by_id[event_id] for event_id in ordered_ids if event_id in rejected_by_id],
        "conflicts": _actual_conflicts(events),
        "terminal_cleanup_event_id": terminal_id,
        "confidence": confidence,
        "uncertainty": uncertainty_text[:200],
        "ignored_noise_events": ignored_noise_events,
        "temporal_guard": {
            "policy": "restore_preterminal_rejection_without_real_overlap_and_keep_cleanup_terminal",
            "restored_event_ids": restored_ids,
            "restored_event_count": len(restored_ids),
            "terminal_cleanup_event_id": terminal_id,
        },
    }
    return repaired, repairs


def select_events_with_screenshot_guard(
    events: list[dict[str, Any]],
    reduce_result: dict[str, Any] | None,
    preserve_equal_confidence: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected, selection = base_reduce.select_events(
        events,
        reduce_result,
        preserve_equal_confidence=preserve_equal_confidence,
    )
    selection["mode"] = "qwen_global_reduce_with_screenshot_temporal_guard"
    guard = reduce_result.get("temporal_guard") if isinstance(reduce_result, dict) else None
    selection["temporal_guard"] = guard if isinstance(guard, dict) else {
        "policy": "screenshot_temporal_guard_not_applied",
        "restored_event_ids": [],
        "restored_event_count": 0,
    }
    return selected, selection
