#!/usr/bin/env python3
"""Weighted graph decoder for the experimental hierarchical v3 pipeline."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from qwen_hierarchical_v1_contract import STAGES


CORRECTION_TERMS = ("插紧", "松动", "接触不良", "纠正", "修正", "重新插好", "调整", "检查接线")
FORMAL_REWIRING_TERMS = ("改接", "换接", "另一端", "改变接法", "重新配置", "一节", "两节", "电池", "第二次")


def _duration(event: dict[str, Any]) -> float:
    return max(0.0, float(event.get("last_seconds", 0.0)) - float(event.get("first_seconds", 0.0)))


def _contains(evidence: str, terms: tuple[str, ...]) -> bool:
    return any(term in evidence for term in terms)


def _ordered(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        events,
        key=lambda item: (
            int(item["representative_frame_number"]),
            int(item["first_frame_number"]),
            str(item["event_id"]),
        ),
    )


def _assigned_item(
    event: dict[str, Any],
    stage: str | None,
    reason: str,
    **diagnostics: Any,
) -> dict[str, Any]:
    return {**event, "stage": stage, "assignment_reason": reason, **diagnostics}


def _advance(node: dict[str, Any], item: dict[str, Any], phase: int, delta: float, **updates: Any) -> dict[str, Any]:
    candidate = deepcopy(node)
    candidate["assigned"].append(item)
    candidate["phase"] = phase
    candidate["score"] += delta
    candidate.update(updates)
    return candidate


def _beam_key(node: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(node["phase"]),
        bool(node["recording_1_observed"]),
        bool(node["recording_2_observed"]),
        bool(node["batched_recording"]),
        round(float(node.get("formal_context_until", -1.0)), 1),
    )


def _prune(nodes: list[dict[str, Any]], width: int) -> list[dict[str, Any]]:
    best_by_state: dict[tuple[Any, ...], dict[str, Any]] = {}
    for node in nodes:
        key = _beam_key(node)
        previous = best_by_state.get(key)
        if previous is None or float(node["score"]) > float(previous["score"]):
            best_by_state[key] = node
    return sorted(best_by_state.values(), key=lambda item: float(item["score"]), reverse=True)[:width]


def _anomaly(item: dict[str, Any]) -> dict[str, Any]:
    action = str(item.get("action_type"))
    reason = str(item.get("assignment_reason", "unclassified_action"))
    interpretations = {
        "auxiliary_action": "七阶段之外的可见行为，保留供诊断或专项评分使用。",
        "uncertain": "Map 保留了证据不足或动作类别冲突的片段。",
        "cleanup_action": "整理候选未通过多帧终态确认，因此未截断实验。",
        "wiring_action": "接线动作与当前最优阶段路径不一致，可能是纠错或额外操作。",
        "measurement_action": "测量动作缺少当前路径所需的轮次上下文。",
        "writing_action": "书写动作缺少当前路径所需的轮次上下文。",
    }
    return {
        "event_id": item.get("event_id"),
        "action_type": action,
        "auxiliary_subtype": item.get("auxiliary_subtype"),
        "time_range_seconds": [item.get("first_seconds"), item.get("last_seconds")],
        "reason": reason,
        "interpretation": interpretations.get(action, "事件未进入七阶段主时间线。"),
        "impact": "不阻塞七阶段二分类下游；保留为诊断和候选检索信号。",
        "evidence": item.get("evidence", ""),
        "confidence": item.get("confidence"),
    }


def anomalous_events_from_assigned(assigned_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_anomaly(item) for item in assigned_events if item.get("stage") is None]


def assign_seven_stages_v3(
    accepted_events: list[dict[str, Any]],
    terminal_cleanup_event_id: str | None,
    beam_width: int = 6,
) -> dict[str, Any]:
    """Decode seven stages while allowing a penalized correction loop after measurement 1."""
    if beam_width < 2:
        raise ValueError("beam_width_must_be_at_least_two")
    nodes: list[dict[str, Any]] = [
        {
            "phase": 0,
            "score": 0.0,
            "assigned": [],
            "recording_1_observed": False,
            "recording_2_observed": False,
            "batched_recording": False,
            "last_measurement_duration": None,
            "formal_context_until": -1.0,
            "terminal_reached": False,
            "discarded_after_terminal": [],
        }
    ]

    for event in _ordered(accepted_events):
        event_id = str(event["event_id"])
        action = str(event["action_type"])
        evidence = str(event.get("evidence", ""))
        event_time = float(event.get("representative_seconds", event.get("first_seconds", 0.0)))
        expanded: list[dict[str, Any]] = []
        for node in nodes:
            if node["terminal_reached"]:
                candidate = deepcopy(node)
                candidate["discarded_after_terminal"].append(event_id)
                expanded.append(candidate)
                continue
            phase = int(node["phase"])
            if action == "auxiliary_action":
                subtype = event.get("auxiliary_subtype")
                updates: dict[str, Any] = {}
                if subtype == "battery_configuration_change":
                    updates["formal_context_until"] = event_time + 20.0
                expanded.append(
                    _advance(
                        node,
                        _assigned_item(event, None, f"auxiliary_action:{subtype or 'unknown'}"),
                        phase,
                        0.0,
                        **updates,
                    )
                )
                continue
            if action == "uncertain":
                expanded.append(_advance(node, _assigned_item(event, None, "map_action_uncertain"), phase, -0.1))
                continue
            if action == "cleanup_action":
                if event_id == terminal_cleanup_event_id:
                    expanded.append(
                        _advance(
                            node,
                            _assigned_item(event, "material_cleanup", "multiframe_confirmed_terminal_cleanup"),
                            6,
                            1.2,
                            terminal_reached=True,
                        )
                    )
                else:
                    expanded.append(
                        _advance(node, _assigned_item(event, None, "cleanup_not_multiframe_confirmed"), phase, -0.2)
                    )
                continue
            if action == "wiring_action":
                if phase == 0:
                    expanded.append(
                        _advance(node, _assigned_item(event, "circuit_wiring", "wiring_before_first_measurement"), 0, 0.7)
                    )
                elif phase == 1:
                    measurement_duration = node.get("last_measurement_duration")
                    short_measurement = measurement_duration is not None and float(measurement_duration) < 5.0
                    correction_cue = _contains(evidence, CORRECTION_TERMS)
                    formal_cue = _contains(evidence, FORMAL_REWIRING_TERMS) or event_time <= float(node["formal_context_until"])
                    correction_delta = 0.1 + (1.0 if short_measurement else -0.2) + (1.1 if correction_cue else 0.0) - (0.8 if formal_cue else 0.0)
                    formal_delta = -0.2 + (0.8 if not short_measurement else -0.4) + (1.2 if formal_cue else 0.0) - (0.7 if correction_cue else 0.0)
                    expanded.append(
                        _advance(
                            node,
                            _assigned_item(event, "circuit_wiring", "correction_loop_after_first_measurement", decoder_hypothesis="correction_loop"),
                            1,
                            correction_delta,
                        )
                    )
                    expanded.append(
                        _advance(
                            node,
                            _assigned_item(event, "circuit_rewiring", "formal_rewiring_hypothesis_before_first_recording", decoder_hypothesis="second_cycle"),
                            3,
                            formal_delta,
                        )
                    )
                elif phase in (2, 3):
                    expanded.append(
                        _advance(node, _assigned_item(event, "circuit_rewiring", "wiring_after_first_recording"), 3, 0.9)
                    )
                else:
                    expanded.append(_advance(node, _assigned_item(event, None, "wiring_after_second_cycle"), phase, -0.5))
                continue
            if action == "measurement_action":
                duration = _duration(event)
                if phase in (0, 1, 2):
                    expanded.append(
                        _advance(
                            node,
                            _assigned_item(event, "measurement_1", "measurement_before_formal_rewiring"),
                            1 if phase != 2 else 2,
                            0.7,
                            last_measurement_duration=duration,
                        )
                    )
                elif phase in (3, 4):
                    expanded.append(
                        _advance(
                            node,
                            _assigned_item(event, "measurement_2", "measurement_after_formal_rewiring"),
                            4,
                            0.9,
                            last_measurement_duration=duration,
                        )
                    )
                elif phase == 5:
                    expanded.append(
                        _advance(
                            node,
                            _assigned_item(
                                event,
                                "measurement_2",
                                "penalized_return_to_measurement_2_after_recording_2",
                                decoder_transition="recording_2_to_measurement_2",
                                transition_penalty=-0.2,
                            ),
                            4,
                            -0.2,
                            last_measurement_duration=duration,
                        )
                    )
                else:
                    expanded.append(_advance(node, _assigned_item(event, None, "measurement_after_recording_2"), phase, -0.4))
                continue
            if action == "writing_action":
                if phase in (0, 1, 2):
                    expanded.append(
                        _advance(
                            node,
                            _assigned_item(event, "recording_1", "first_writing_before_formal_rewiring"),
                            2,
                            0.8,
                            recording_1_observed=True,
                        )
                    )
                elif phase == 3:
                    expanded.append(
                        _advance(
                            node,
                            _assigned_item(
                                event,
                                None,
                                "pending_writing_before_measurement_2",
                                pending_stage="recording_2",
                            ),
                            3,
                            -0.25,
                        )
                    )
                elif phase in (4, 5):
                    batched = not bool(node["recording_1_observed"])
                    repeated = bool(node["recording_2_observed"])
                    expanded.append(
                        _advance(
                            node,
                            _assigned_item(
                                event,
                                "recording_2",
                                "repeated_recording_2_after_measurement_return"
                                if repeated
                                else ("batched_writing_after_two_measurements" if batched else "writing_after_second_measurement"),
                                batched_recording=batched,
                                recording_search_aliases=["recording_1", "recording_2"] if batched else ["recording_2"],
                            ),
                            5,
                            0.7 if repeated else 0.8,
                            batched_recording=bool(node["batched_recording"]) or batched,
                            recording_2_observed=True,
                        )
                    )
                continue
            expanded.append(_advance(node, _assigned_item(event, None, "unsupported_action_type"), phase, -0.5))
        nodes = _prune(expanded, beam_width)

    best = max(nodes, key=lambda item: float(item["score"]))
    assigned = list(best["assigned"])
    observed_intervals: list[dict[str, Any]] = []
    for item in assigned:
        if item.get("stage") is None:
            continue
        interval = {
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
        for key in (
            "decoder_hypothesis",
            "batched_recording",
            "recording_search_aliases",
            "decoder_transition",
            "transition_penalty",
            "pending_stage",
        ):
            if key in item:
                interval[key] = item[key]
        observed_intervals.append(interval)
    observed_stages = {str(item["stage"]) for item in observed_intervals}
    anomalies = anomalous_events_from_assigned(assigned)
    # Anomalies and batched writing are diagnostic signals, not abstention gates.
    review_reasons: list[str] = []
    terminal_reached = bool(best["terminal_reached"])
    alternatives = [
        {
            "phase": int(node["phase"]),
            "score": round(float(node["score"]), 6),
            "recording_1_observed": bool(node["recording_1_observed"]),
            "recording_2_observed": bool(node["recording_2_observed"]),
            "batched_recording": bool(node["batched_recording"]),
        }
        for node in sorted(nodes, key=lambda item: float(item["score"]), reverse=True)
    ]
    return {
        "assigned_events": assigned,
        "observed_stage_intervals": observed_intervals,
        "missing_stages": [stage for stage in STAGES if stage not in observed_stages],
        "review_reasons": sorted(set(review_reasons)),
        "anomalous_events": anomalies,
        "decoder": {
            "type": "weighted_directed_graph_beam_search",
            "beam_width": beam_width,
            "selected_score": round(float(best["score"]), 6),
            "alternative_final_states": alternatives,
            "batched_recording": bool(best["batched_recording"]),
        },
        "analysis_termination": {
            "terminal_cleanup_reached": terminal_reached,
            "terminal_cleanup_event_id": terminal_cleanup_event_id if terminal_reached else None,
            "discarded_after_terminal_event_ids": list(best["discarded_after_terminal"]),
            "discarded_after_terminal_count": len(best["discarded_after_terminal"]),
        },
    }
