#!/usr/bin/env python3
"""Deterministic event reduction and seven-stage state machine."""

from __future__ import annotations

import math
import re
from typing import Any

from qwen_hierarchical_v1_contract import STAGES


ENVIRONMENT_HARD_STOP_KEYWORDS = ("换座位", "换人", "人脸", "脸", "抬头", "闲聊", "聊天")
COMPLETED_CLEANUP_HARD_STOP_KEYWORDS = (
    "整理完",
    "整理完毕",
    "拆完",
    "全拆",
    "收完",
    "放回桌子的左上角",
    "放回桌子左上角",
    "放到桌子的左上角",
    "放到桌子左上角",
    "移到桌子的左上角",
    "移到桌子左上角",
    "桌子的左上角",
    "桌子左上角",
    "左上角",
    "清空",
)
HARD_STOP_KEYWORDS = ENVIRONMENT_HARD_STOP_KEYWORDS + COMPLETED_CLEANUP_HARD_STOP_KEYWORDS
HARD_STOP_REASON_PREFIX = "Hard stop triggered by"


def hard_stop_trigger_keywords(event: dict[str, Any]) -> list[str]:
    """Return explicit, non-negated hard-stop terms found in Qwen evidence."""
    evidence = event.get("evidence")
    if not isinstance(evidence, str):
        return []
    negation = r"(?:尚未|还未|还没|未见|未观察到|未看到|未出现|未发生|未|没有观察到|没有看到|没有出现|没有发生|没有|无明显|无明确|无|并非|不是)"
    matched: list[str] = []
    # Prefer the most specific term so "整理完毕" is not also logged as "整理完",
    # and "人脸" is not also logged as the generic "脸".
    keywords = sorted(HARD_STOP_KEYWORDS, key=len, reverse=True)
    for keyword in keywords:
        if any(keyword in existing for existing in matched):
            continue
        for match in re.finditer(re.escape(keyword), evidence):
            context = evidence[max(0, match.start() - 18) : match.end()]
            if re.search(negation + r"[^，。；,;]{0,12}" + re.escape(keyword) + r"$", context):
                continue
            matched.append(keyword)
            break
    return [keyword for keyword in HARD_STOP_KEYWORDS if keyword in matched]


def is_hard_stop_triggered(event: dict[str, Any]) -> bool:
    """Return True only when Qwen's evidence explicitly names a hard-stop visual cue."""
    return bool(hard_stop_trigger_keywords(event))


def salvage_reduce_response(
    events: list[dict[str, Any]],
    reduce_result: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Recover known event decisions without turning one semantic error into a video-level abstention."""
    if not isinstance(reduce_result, dict):
        return None, []
    raw_accepted = reduce_result.get("accepted_event_ids")
    if not isinstance(raw_accepted, list):
        return None, []

    event_by_id = {str(event["event_id"]): event for event in events}
    known_ids = set(event_by_id)
    accepted: list[str] = []
    repairs: list[dict[str, Any]] = []
    ignored_noise_events: list[dict[str, Any]] = []
    for raw_id in raw_accepted:
        if not isinstance(raw_id, str) or raw_id not in known_ids:
            repairs.append({"reason": "unknown_accepted_event_removed", "event_id": raw_id})
            continue
        if raw_id in accepted:
            repairs.append({"reason": "duplicate_accepted_event_removed", "event_id": raw_id})
            continue
        accepted.append(raw_id)

    rejected_by_id: dict[str, dict[str, Any]] = {}
    raw_rejected = reduce_result.get("rejected_events")
    if isinstance(raw_rejected, list):
        for item in raw_rejected:
            if not isinstance(item, dict):
                repairs.append({"reason": "malformed_rejection_removed"})
                continue
            event_id = item.get("event_id")
            if not isinstance(event_id, str) or event_id not in known_ids:
                repairs.append({"reason": "unknown_rejected_event_removed", "event_id": event_id})
                continue
            if event_id in accepted:
                repairs.append({"reason": "accepted_event_wins_duplicate_rejection", "event_id": event_id})
                continue
            if event_id in rejected_by_id:
                repairs.append({"reason": "duplicate_rejected_event_removed", "event_id": event_id})
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
                explanation = "本地宽松恢复保留了拒绝决定，但模型未提供有效说明。"
                repairs.append({"reason": "missing_rejection_explanation_filled", "event_id": event_id})
            rejected_by_id[event_id] = {
                "event_id": event_id,
                "reason": reason,
                "explanation": explanation[:120],
            }

    for event_id in event_by_id:
        if event_id in accepted or event_id in rejected_by_id:
            continue
        rejected_by_id[event_id] = {
            "event_id": event_id,
            "reason": "other",
            "explanation": "模型未对该事件作出完整决定，本地仅隔离这一事件。",
        }
        repairs.append({"reason": "omitted_event_locally_quarantined", "event_id": event_id})

    terminal_id = reduce_result.get("terminal_cleanup_event_id")
    if terminal_id is not None:
        terminal = event_by_id.get(terminal_id) if isinstance(terminal_id, str) else None
        terminal_valid = (
            terminal is not None
            and terminal_id in accepted
            and terminal.get("action_type") == "cleanup_action"
        )
        if not terminal_valid:
            repairs.append({"reason": "invalid_terminal_cleanup_demoted", "event_id": terminal_id})
            terminal_id = None

    if terminal_id is None:
        accepted_cleanup_events = [
            event_by_id[event_id]
            for event_id in accepted
            if event_by_id[event_id].get("action_type") == "cleanup_action"
        ]
        if accepted_cleanup_events:
            terminal = min(accepted_cleanup_events, key=lambda event: int(event["first_frame_number"]))
            terminal_id = str(terminal["event_id"])
            repairs.append(
                {
                    "reason": "terminal_cleanup_promoted_by_cleanup_barrier",
                    "event_id": terminal_id,
                }
            )

    if terminal_id is not None:
        terminal = event_by_id[terminal_id]
        terminal_start = int(terminal["first_frame_number"])
        post_terminal_candidates = [
            event
            for event in events
            if str(event["event_id"]) != terminal_id
            and int(event["last_frame_number"]) >= terminal_start
        ]
        trigger_event_matches = [
            (event, hard_stop_trigger_keywords(event))
            for event in [terminal, *post_terminal_candidates]
        ]
        trigger_event_matches = [item for item in trigger_event_matches if item[1]]
        trigger_events = [item[0] for item in trigger_event_matches]
        trigger_keywords = list(
            dict.fromkeys(
                keyword
                for _event, keywords in trigger_event_matches
                for keyword in keywords
            )
        )
        if not trigger_keywords:
            trigger_keywords = ["cleanup_action"]
            trigger_events = [terminal]
        hard_stop_reason = f"{HARD_STOP_REASON_PREFIX} [{', '.join(trigger_keywords)}]"
        ignored_ids = {str(event["event_id"]) for event in post_terminal_candidates}
        accepted = [event_id for event_id in accepted if event_id not in ignored_ids]
        for event in post_terminal_candidates:
            event_id = str(event["event_id"])
            rejected_by_id[event_id] = {
                "event_id": event_id,
                "reason": "post_terminal_cleanup",
                "explanation": "最终整理是不可逆终态，之后的动作按实验结束后的录像噪声忽略。",
            }
            ignored_noise_events.append(
                {
                    **event,
                    "label": "ignored_noise_post_experiment",
                    "ignored_label": "ignored_noise_post_experiment",
                    "terminal_cleanup_event_id": terminal_id,
                    "hard_stop_trigger_event_ids": [str(item["event_id"]) for item in trigger_events],
                    "hard_stop_trigger_keywords": trigger_keywords,
                }
            )
        if post_terminal_candidates:
            repairs.append(
                {
                    "reason": hard_stop_reason,
                    "event_id": terminal_id,
                    "trigger_event_ids": [str(item["event_id"]) for item in trigger_events],
                    "trigger_keywords": trigger_keywords,
                    "ignored_event_ids": sorted(ignored_ids),
                }
            )

    conflicts: list[dict[str, Any]] = []
    raw_conflicts = reduce_result.get("conflicts")
    if isinstance(raw_conflicts, list):
        for conflict in raw_conflicts:
            if not isinstance(conflict, dict):
                continue
            event_ids = conflict.get("event_ids")
            resolution = conflict.get("resolution")
            if (
                isinstance(event_ids, list)
                and len(event_ids) >= 2
                and all(isinstance(item, str) and item in known_ids for item in event_ids)
                and isinstance(resolution, str)
                and resolution.strip()
            ):
                conflicts.append({"event_ids": event_ids, "resolution": resolution[:160]})

    confidence = reduce_result.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(float(confidence)):
        confidence = 0.0
        repairs.append({"reason": "invalid_confidence_replaced"})
    else:
        confidence = min(1.0, max(0.0, float(confidence)))
    original_uncertainty = reduce_result.get("uncertainty")
    uncertainty = original_uncertainty.strip() if isinstance(original_uncertainty, str) else ""
    if ignored_noise_events:
        recovery_note = "最终整理已作为不可逆终态锁定；后续事件作为录像噪声忽略。"
    elif repairs:
        recovery_note = "本地采用事件级宽松恢复；未确认最终整理时不会截断后续动作。"
    else:
        recovery_note = ""
    uncertainty = f"{uncertainty} {recovery_note}".strip()[:160]

    repaired = {
        "accepted_event_ids": accepted,
        "rejected_events": [rejected_by_id[event_id] for event_id in event_by_id if event_id in rejected_by_id],
        "conflicts": conflicts,
        "terminal_cleanup_event_id": terminal_id,
        "confidence": confidence,
        "uncertainty": uncertainty,
        "ignored_noise_events": ignored_noise_events,
    }
    return repaired, repairs


def _overlaps(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return int(left["first_frame_number"]) <= int(right["last_frame_number"]) and int(right["first_frame_number"]) <= int(left["last_frame_number"])


def deduplicate_map_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge same-action overlap created by neighboring Map windows."""
    groups: list[dict[str, Any]] = []
    ordered = sorted(
        events,
        key=lambda item: (
            int(item["first_frame_number"]),
            int(item["last_frame_number"]),
            str(item["action_type"]),
            str(item["source_event_id"]),
        ),
    )
    for event in ordered:
        matching = next(
            (
                group
                for group in groups
                if group["action_type"] == event["action_type"] and _overlaps(group, event)
            ),
            None,
        )
        if matching is None:
            groups.append(
                {
                    **event,
                    "source_event_ids": [str(event["source_event_id"])],
                    "window_ids": [str(event["window_id"])],
                    "evidence_items": [
                        {
                            "source_event_id": str(event["source_event_id"]),
                            "window_id": str(event["window_id"]),
                            "evidence": str(event["evidence"]),
                            "confidence": float(event["confidence"]),
                        }
                    ],
                }
            )
            continue
        if int(event["first_frame_number"]) < int(matching["first_frame_number"]):
            matching["first_frame_number"] = int(event["first_frame_number"])
            matching["first_frame_id"] = event["first_frame_id"]
            matching["first_seconds"] = float(event["first_seconds"])
        if int(event["last_frame_number"]) > int(matching["last_frame_number"]):
            matching["last_frame_number"] = int(event["last_frame_number"])
            matching["last_frame_id"] = event["last_frame_id"]
            matching["last_seconds"] = float(event["last_seconds"])
        if float(event["confidence"]) > float(matching["confidence"]):
            matching["representative_frame_number"] = int(event["representative_frame_number"])
            matching["representative_frame_id"] = event["representative_frame_id"]
            matching["representative_seconds"] = float(event["representative_seconds"])
            matching["evidence"] = event["evidence"]
            matching["confidence"] = float(event["confidence"])
        matching["source_event_ids"].append(str(event["source_event_id"]))
        if str(event["window_id"]) not in matching["window_ids"]:
            matching["window_ids"].append(str(event["window_id"]))
        matching["evidence_items"].append(
            {
                "source_event_id": str(event["source_event_id"]),
                "window_id": str(event["window_id"]),
                "evidence": str(event["evidence"]),
                "confidence": float(event["confidence"]),
            }
        )
    groups.sort(
        key=lambda item: (
            int(item["first_frame_number"]),
            int(item["last_frame_number"]),
            str(item["action_type"]),
        )
    )
    for index, group in enumerate(groups, start=1):
        group["event_id"] = f"evt_{index:04d}"
        group["source_event_ids"] = sorted(set(group["source_event_ids"]))
        group["window_ids"] = sorted(set(group["window_ids"]))
    return groups


def find_temporal_conflicts(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    conflicts: list[dict[str, Any]] = []
    actionable = [event for event in events if event.get("action_type") != "uncertain"]
    for index, left in enumerate(actionable):
        for right in actionable[index + 1 :]:
            if int(right["first_frame_number"]) > int(left["last_frame_number"]):
                break
            if left["action_type"] != right["action_type"] and _overlaps(left, right):
                conflicts.append(
                    {
                        "event_ids": [left["event_id"], right["event_id"]],
                        "reason": "different_actions_overlap_in_source_frames",
                        "overlap_frame_numbers": [
                            max(int(left["first_frame_number"]), int(right["first_frame_number"])),
                            min(int(left["last_frame_number"]), int(right["last_frame_number"])),
                        ],
                    }
                )
    return conflicts


def resolve_accepted_conflicts(
    events: list[dict[str, Any]],
    preserve_equal_confidence: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply a conservative local repair when Reduce accepts incompatible events."""
    kept: list[dict[str, Any]] = []
    repairs: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    for event in sorted(events, key=lambda item: (int(item["first_frame_number"]), str(item["event_id"]))):
        overlaps = [
            previous
            for previous in kept
            if previous.get("action_type") != "uncertain"
            and event.get("action_type") != "uncertain"
            and previous.get("action_type") != event.get("action_type")
            and _overlaps(previous, event)
        ]
        if not overlaps:
            kept.append(event)
            continue
        contenders = overlaps + [event]
        best_confidence = max(float(item.get("confidence", 0.0)) for item in contenders)
        strongest = [item for item in contenders if float(item.get("confidence", 0.0)) == best_confidence]
        if len(strongest) != 1:
            if preserve_equal_confidence:
                winner = min(
                    strongest,
                    key=lambda item: (
                        int(item["last_frame_number"]) - int(item["first_frame_number"]),
                        int(item["representative_frame_number"]),
                        str(item["event_id"]),
                    ),
                )
                for contender in contenders:
                    if contender is winner:
                        continue
                    if contender in kept:
                        kept.remove(contender)
                    repairs.append(
                        {
                            "event_id": str(contender["event_id"]),
                            "reason": "equal_confidence_tie_resolved_by_temporal_specificity",
                            "kept_event_id": str(winner["event_id"]),
                        }
                    )
                if winner is event and winner not in kept:
                    kept.append(winner)
                continue
            for contender in contenders:
                if contender in kept:
                    kept.remove(contender)
            unresolved.append(
                {
                    "event_ids": [str(item["event_id"]) for item in contenders],
                    "reason": "equal_confidence_conflicting_events_quarantined",
                }
            )
            continue
        winner = strongest[0]
        losers = [item for item in contenders if item is not winner]
        for loser in losers:
            if loser in kept:
                kept.remove(loser)
            repairs.append(
                {
                    "event_id": str(loser["event_id"]),
                    "reason": "local_conflict_weaker_than_accepted_event",
                    "kept_event_id": str(winner["event_id"]),
                }
            )
        if winner is event:
            kept.append(event)
    kept.sort(key=lambda item: (int(item["representative_frame_number"]), int(item["first_frame_number"])))
    return kept, repairs, unresolved


def select_events(
    events: list[dict[str, Any]],
    reduce_result: dict[str, Any] | None,
    preserve_equal_confidence: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_id = {str(event["event_id"]): event for event in events}
    if reduce_result is None:
        return [], {
            "mode": "local_fallback_quarantine_all",
            "terminal_cleanup_event_id": None,
            "rejected_events": [
                {
                    "event_id": str(event["event_id"]),
                    "reason": "reduce_contract_invalid_quarantined",
                }
                for event in events
            ],
            "local_conflict_repairs": [],
            "unresolved_conflicts": find_temporal_conflicts(events),
            "needs_review": True,
        }
    accepted = [by_id[event_id] for event_id in reduce_result["accepted_event_ids"]]
    accepted, repairs, unresolved = resolve_accepted_conflicts(
        accepted,
        preserve_equal_confidence=preserve_equal_confidence,
    )
    terminal_id = reduce_result.get("terminal_cleanup_event_id")
    if not isinstance(terminal_id, str) or terminal_id not in {str(event["event_id"]) for event in accepted}:
        terminal_id = None
    return accepted, {
        "mode": "qwen_global_reduce",
        "terminal_cleanup_event_id": terminal_id,
        "rejected_events": list(reduce_result.get("rejected_events", [])),
        "conflicts": list(reduce_result.get("conflicts", [])),
        "local_conflict_repairs": repairs,
        "unresolved_conflicts": unresolved,
        "confidence": reduce_result.get("confidence"),
        "uncertainty": reduce_result.get("uncertainty", ""),
        "needs_review": bool(reduce_result.get("uncertainty")) or bool(repairs) or bool(unresolved),
    }


def assign_seven_stages(
    accepted_events: list[dict[str, Any]],
    terminal_cleanup_event_id: str | None,
) -> dict[str, Any]:
    """Assign first/second labels without allowing stage-order backtracking."""
    phase = 0
    assigned: list[dict[str, Any]] = []
    discarded_after_terminal: list[str] = []
    review_reasons: list[str] = []
    terminal_reached = False
    events = sorted(
        accepted_events,
        key=lambda item: (
            int(item["representative_frame_number"]),
            int(item["first_frame_number"]),
            str(item["event_id"]),
        ),
    )
    for event in events:
        event_id = str(event["event_id"])
        action = str(event["action_type"])
        item = dict(event)
        item["stage"] = None
        item["assignment_reason"] = "unclassified_action"
        if terminal_reached:
            discarded_after_terminal.append(event_id)
            continue
        if action == "cleanup_action":
            if event_id == terminal_cleanup_event_id:
                item["stage"] = "material_cleanup"
                item["assignment_reason"] = "global_reduce_confirmed_terminal_cleanup"
                phase = 6
                terminal_reached = True
            else:
                item["assignment_reason"] = "cleanup_not_confirmed_terminal"
                review_reasons.append(f"nonterminal_cleanup:{event_id}")
            assigned.append(item)
            continue
        if action == "uncertain":
            item["assignment_reason"] = "map_action_uncertain"
            review_reasons.append(f"uncertain_event:{event_id}")
            assigned.append(item)
            continue
        if action == "wiring_action":
            if phase == 0:
                item["stage"] = "circuit_wiring"
                item["assignment_reason"] = "wiring_before_first_record"
            elif phase == 1:
                # A missing first recording is common with sparse sampling.
                # A later, clearly visible wiring run is the only local cue
                # available for recovering the second cycle.
                item["stage"] = "circuit_rewiring"
                item["assignment_reason"] = "rewiring_after_measurement_recording_missing"
                phase = 3
                review_reasons.append(f"recording_1_not_observed_before_rewiring:{event_id}")
            elif phase == 2 or phase == 3:
                item["stage"] = "circuit_rewiring"
                item["assignment_reason"] = "wiring_after_first_record"
                phase = 3
            else:
                item["assignment_reason"] = "wiring_would_break_monotonic_stage_order"
        elif action == "measurement_action":
            if phase == 0 or phase == 1:
                item["stage"] = "measurement_1"
                item["assignment_reason"] = "measurement_before_first_record"
                phase = 1
            elif phase == 3 or phase == 4:
                item["stage"] = "measurement_2"
                item["assignment_reason"] = "measurement_after_rewiring"
                phase = 4
            else:
                item["assignment_reason"] = "measurement_lacks_required_cycle_context"
        elif action == "writing_action":
            if phase <= 2:
                item["stage"] = "recording_1"
                item["assignment_reason"] = "first_writing_run_before_rewiring"
                phase = 2
            elif phase in (3, 4, 5):
                item["stage"] = "recording_2"
                item["assignment_reason"] = "writing_after_rewiring"
                phase = 5
        if item["stage"] is None:
            review_reasons.append(f"unclassified_transition:{event_id}:{item['assignment_reason']}")
        assigned.append(item)
    observed_intervals = [
        {
            "event_id": item["event_id"],
            "stage": item["stage"],
            "start_frame_id": item["first_frame_id"],
            "end_frame_id": item["last_frame_id"],
            "start_frame_number": item["first_frame_number"],
            "end_frame_number": item["last_frame_number"],
            "start_seconds": item["first_seconds"],
            "end_seconds": item["last_seconds"],
            "evidence": item["evidence"],
            "confidence": item["confidence"],
            "assignment_reason": item["assignment_reason"],
        }
        for item in assigned
        if item.get("stage") is not None
    ]
    observed_stages = {str(item["stage"]) for item in observed_intervals}
    return {
        "assigned_events": assigned,
        "observed_stage_intervals": observed_intervals,
        "missing_stages": [stage for stage in STAGES if stage not in observed_stages],
        "review_reasons": sorted(set(review_reasons)),
        "analysis_termination": {
            "terminal_cleanup_reached": terminal_reached,
            "terminal_cleanup_event_id": terminal_cleanup_event_id if terminal_reached else None,
            "discarded_after_terminal_event_ids": discarded_after_terminal,
            "discarded_after_terminal_count": len(discarded_after_terminal),
        },
    }


def build_boundary_candidates(observed_stage_intervals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(
        observed_stage_intervals,
        key=lambda item: (int(item["start_frame_number"]), int(item["end_frame_number"])),
    )
    stage_runs: list[dict[str, Any]] = []
    for interval in ordered:
        if stage_runs and stage_runs[-1]["stage"] == interval["stage"]:
            previous = stage_runs[-1]
            if int(interval["end_frame_number"]) > int(previous["end_frame_number"]):
                previous["end_frame_number"] = interval["end_frame_number"]
                previous["end_frame_id"] = interval["end_frame_id"]
                previous["end_seconds"] = interval["end_seconds"]
            previous["supporting_event_ids"].append(interval["event_id"])
        else:
            stage_runs.append(
                {
                    **interval,
                    "supporting_event_ids": [interval["event_id"]],
                }
            )
    boundaries: list[dict[str, Any]] = []
    for index, (left, right) in enumerate(zip(stage_runs, stage_runs[1:]), start=1):
        boundaries.append(
            {
                "boundary_id": f"b{index:03d}",
                "from_stage": left["stage"],
                "to_stage": right["stage"],
                "coarse_last_from_frame_id": left["end_frame_id"],
                "coarse_first_to_frame_id": right["start_frame_id"],
                "coarse_last_from_seconds": float(left["end_seconds"]),
                "coarse_first_to_seconds": float(right["start_seconds"]),
                "coarse_selected_seconds": float(right["start_seconds"]),
                "coarse_order_valid": int(left["end_frame_number"]) < int(right["start_frame_number"]),
                "from_event_ids": list(left["supporting_event_ids"]),
                "to_event_ids": list(right["supporting_event_ids"]),
            }
        )
    return boundaries


def merge_observed_stage_runs(
    observed_stage_intervals: list[dict[str, Any]],
    max_gap_seconds: float = 4.1,
) -> list[dict[str, Any]]:
    """Make a readable stage table without filling long evidence gaps."""
    runs: list[dict[str, Any]] = []
    for interval in sorted(observed_stage_intervals, key=lambda item: (float(item["start_seconds"]), float(item["end_seconds"]))):
        if (
            runs
            and runs[-1]["stage"] == interval["stage"]
            and float(interval["start_seconds"]) <= float(runs[-1]["end_seconds"]) + max_gap_seconds
        ):
            run = runs[-1]
            run["end_seconds"] = max(float(run["end_seconds"]), float(interval["end_seconds"]))
            run["end_frame_number"] = max(int(run["end_frame_number"]), int(interval["end_frame_number"]))
            run["end_frame_id"] = interval["end_frame_id"] if int(interval["end_frame_number"]) >= int(run["end_frame_number"]) else run["end_frame_id"]
            run["event_ids"].append(interval["event_id"])
            run["evidence_items"].append(interval["evidence"])
            run["confidence"] = max(float(run["confidence"]), float(interval["confidence"]))
        else:
            runs.append(
                {
                    "stage": interval["stage"],
                    "start_seconds": float(interval["start_seconds"]),
                    "end_seconds": float(interval["end_seconds"]),
                    "start_frame_number": int(interval["start_frame_number"]),
                    "end_frame_number": int(interval["end_frame_number"]),
                    "start_frame_id": interval["start_frame_id"],
                    "end_frame_id": interval["end_frame_id"],
                    "event_ids": [interval["event_id"]],
                    "evidence_items": [interval["evidence"]],
                    "confidence": float(interval["confidence"]),
                }
            )
    for index, run in enumerate(runs, start=1):
        run["run_id"] = f"stage_run_{index:03d}"
        run["evidence"] = "；".join(dict.fromkeys(str(item) for item in run["evidence_items"]))
    return runs


def build_evidence_timeline(
    locked_start_seconds: float,
    locked_end_seconds: float,
    observed_stage_intervals: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    timeline: list[dict[str, Any]] = []
    review: list[str] = []
    cursor = float(locked_start_seconds)
    ordered = sorted(observed_stage_intervals, key=lambda item: (float(item["start_seconds"]), float(item["end_seconds"])))
    for interval in ordered:
        start = max(float(locked_start_seconds), float(interval["start_seconds"]))
        end = min(float(locked_end_seconds), float(interval["end_seconds"]))
        if end < start:
            review.append(f"interval_outside_locked_segment:{interval['event_id']}")
            continue
        if start < cursor - 1e-9:
            review.append(f"overlapping_observed_intervals:{interval['event_id']}")
            start = cursor
        if start > cursor + 1e-9:
            timeline.append(
                {
                    "kind": "unclassified",
                    "stage": None,
                    "start_seconds": cursor,
                    "end_seconds": start,
                    "reason": "no_direct_action_evidence",
                }
            )
        timeline.append(
            {
                "kind": "observed_stage",
                "stage": interval["stage"],
                "start_seconds": start,
                "end_seconds": max(start, end),
                "event_id": interval["event_id"],
            }
        )
        cursor = max(cursor, end)
    if cursor < locked_end_seconds - 1e-9:
        timeline.append(
            {
                "kind": "unclassified",
                "stage": None,
                "start_seconds": cursor,
                "end_seconds": float(locked_end_seconds),
                "reason": "no_direct_action_evidence",
            }
        )
    return timeline, sorted(set(review))
