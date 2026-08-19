#!/usr/bin/env python3
"""Penalized graph decoder for the v2 behavior-tolerant variants."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from qwen_hierarchical_v1_contract import STAGES


CORRECTION_TERMS = (
    "插紧",
    "松动",
    "接触不良",
    "纠正",
    "修正",
    "重新插好",
    "调整接触",
    "检查接线",
)
FORMAL_REWIRING_TERMS = (
    "改接",
    "换接",
    "另一端",
    "改变接法",
    "重新配置",
    "一节",
    "两节",
    "电池配置",
    "第二次",
)


def _duration(event: dict[str, Any]) -> float:
    return max(0.0, float(event.get("last_seconds", 0.0)) - float(event.get("first_seconds", 0.0)))


def _confidence(event: dict[str, Any]) -> float:
    value = event.get("confidence", 0.0)
    return min(1.0, max(0.0, float(value))) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0


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


def _advance(
    node: dict[str, Any],
    item: dict[str, Any] | None,
    phase: int,
    delta: float,
    **updates: Any,
) -> dict[str, Any]:
    candidate = deepcopy(node)
    if item is not None:
        candidate["assigned"].append(item)
        if item.get("stage") is not None:
            candidate["direct_assignment_count"] += 0 if item.get("inferred_stage") else 1
            candidate["inferred_assignment_count"] += 1 if item.get("inferred_stage") else 0
    candidate["phase"] = phase
    candidate["score"] += delta
    candidate.update(updates)
    return candidate


def _beam_key(node: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(node["phase"]),
        bool(node["measurement_1_observed"]),
        bool(node["measurement_2_observed"]),
        bool(node["recording_1_observed"]),
        bool(node["recording_2_observed"]),
        bool(node["batched_recording"]),
        round(float(node["formal_context_until"]), 1),
        bool(node["terminal_reached"]),
    )


def _node_rank(node: dict[str, Any]) -> tuple[Any, ...]:
    event_ids = tuple(str(item.get("event_id", "")) for item in node["assigned"])
    return (
        round(float(node["score"]), 9),
        int(node["direct_assignment_count"]),
        -int(node["inferred_assignment_count"]),
        tuple(reversed(event_ids)),
    )


def _prune(nodes: list[dict[str, Any]], width: int) -> list[dict[str, Any]]:
    best_by_state: dict[tuple[Any, ...], dict[str, Any]] = {}
    for node in nodes:
        key = _beam_key(node)
        previous = best_by_state.get(key)
        if previous is None or _node_rank(node) > _node_rank(previous):
            best_by_state[key] = node
    return sorted(best_by_state.values(), key=_node_rank, reverse=True)[:width]


def _future_context(events: list[dict[str, Any]], index: int, horizon_seconds: float = 30.0) -> dict[str, bool]:
    current = events[index]
    end = float(current.get("last_seconds", current.get("representative_seconds", 0.0)))
    context = {"measurement": False, "writing": False}
    for later in events[index + 1 :]:
        start = float(later.get("first_seconds", later.get("representative_seconds", 0.0)))
        if start - end > horizon_seconds:
            break
        action = str(later.get("action_type"))
        if action == "measurement_action":
            context["measurement"] = True
        elif action == "writing_action":
            context["writing"] = True
    return context


def _transition_score(event: dict[str, Any], base: float) -> float:
    return base + 0.2 * _confidence(event)


def _interval(item: dict[str, Any]) -> dict[str, Any]:
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
        "decoder_transition",
        "transition_penalty",
        "batched_recording",
        "recording_search_aliases",
        "inferred_stage",
        "measurement_2_observed",
        "fallback_source",
    ):
        if key in item:
            interval[key] = item[key]
    return interval


def assign_seven_stages_behavior_tolerant(
    accepted_events: list[dict[str, Any]],
    terminal_cleanup_event_id: str | None,
    beam_width: int = 6,
) -> dict[str, Any]:
    """Decode seven stages with penalized local loops and deterministic ties."""
    if beam_width < 2:
        raise ValueError("beam_width_must_be_at_least_two")
    events = _ordered(accepted_events)
    nodes: list[dict[str, Any]] = [
        {
            "phase": 0,
            "score": 0.0,
            "assigned": [],
            "measurement_1_observed": False,
            "measurement_2_observed": False,
            "recording_1_observed": False,
            "recording_2_observed": False,
            "batched_recording": False,
            "last_measurement_duration": None,
            "formal_context_until": -1.0,
            "terminal_reached": False,
            "discarded_after_terminal": [],
            "direct_assignment_count": 0,
            "inferred_assignment_count": 0,
        }
    ]

    for index, event in enumerate(events):
        event_id = str(event["event_id"])
        action = str(event["action_type"])
        evidence = str(event.get("evidence", ""))
        event_time = float(event.get("representative_seconds", event.get("first_seconds", 0.0)))
        future = _future_context(events, index)
        expanded: list[dict[str, Any]] = []
        for node in nodes:
            phase = int(node["phase"])
            if node["terminal_reached"]:
                candidate = deepcopy(node)
                candidate["discarded_after_terminal"].append(event_id)
                expanded.append(candidate)
                continue

            if action == "auxiliary_action":
                subtype = str(event.get("auxiliary_subtype", "other_action"))
                updates: dict[str, Any] = {}
                if subtype == "battery_configuration_change":
                    updates["formal_context_until"] = event_time + 20.0
                expanded.append(
                    _advance(
                        node,
                        _assigned_item(event, None, f"auxiliary_action:{subtype}"),
                        phase,
                        0.0,
                        **updates,
                    )
                )
                continue

            if action == "uncertain":
                expanded.append(_advance(node, _assigned_item(event, None, "map_action_uncertain"), phase, -0.05))
                continue

            if action == "cleanup_action":
                if event_id == terminal_cleanup_event_id:
                    expanded.append(
                        _advance(
                            node,
                            _assigned_item(event, "material_cleanup", "global_reduce_confirmed_terminal_cleanup"),
                            6,
                            _transition_score(event, 1.0),
                            terminal_reached=True,
                        )
                    )
                else:
                    expanded.append(
                        _advance(node, _assigned_item(event, None, "cleanup_not_confirmed_terminal"), phase, -0.15)
                    )
                continue

            if action == "wiring_action":
                if phase == 0:
                    expanded.append(
                        _advance(
                            node,
                            _assigned_item(event, "circuit_wiring", "wiring_before_first_measurement"),
                            0,
                            _transition_score(event, 0.55),
                        )
                    )
                elif phase == 1:
                    measurement_duration = float(node.get("last_measurement_duration") or 0.0)
                    shortness = max(0.0, min(1.0, 1.0 - measurement_duration / 12.0))
                    longness = 1.0 - shortness
                    correction_cue = _contains(evidence, CORRECTION_TERMS)
                    formal_cue = _contains(evidence, FORMAL_REWIRING_TERMS) or event_time <= float(node["formal_context_until"])
                    correction_delta = (
                        0.2
                        + 0.35 * shortness
                        + (0.45 if correction_cue else 0.0)
                        + (0.2 if future["measurement"] else 0.0)
                        - (0.45 if formal_cue else 0.0)
                    )
                    formal_delta = (
                        0.05
                        + 0.25 * longness
                        + (0.55 if formal_cue else 0.0)
                        + (0.2 if future["writing"] else 0.0)
                        - (0.35 if correction_cue else 0.0)
                    )
                    expanded.append(
                        _advance(
                            node,
                            _assigned_item(
                                event,
                                "circuit_wiring",
                                "correction_loop_after_first_measurement",
                                decoder_hypothesis="correction_loop",
                            ),
                            1,
                            _transition_score(event, correction_delta),
                        )
                    )
                    expanded.append(
                        _advance(
                            node,
                            _assigned_item(
                                event,
                                "circuit_rewiring",
                                "formal_rewiring_hypothesis_before_first_recording",
                                decoder_hypothesis="second_cycle",
                            ),
                            3,
                            _transition_score(event, formal_delta),
                        )
                    )
                elif phase in (2, 3):
                    expanded.append(
                        _advance(
                            node,
                            _assigned_item(event, "circuit_rewiring", "wiring_after_first_recording"),
                            3,
                            _transition_score(event, 0.8),
                        )
                    )
                else:
                    expanded.append(
                        _advance(node, _assigned_item(event, None, "wiring_after_second_cycle"), phase, -0.3)
                    )
                continue

            if action == "measurement_action":
                duration = _duration(event)
                if phase in (0, 1):
                    expanded.append(
                        _advance(
                            node,
                            _assigned_item(event, "measurement_1", "measurement_before_first_recording"),
                            1,
                            _transition_score(event, 0.7),
                            measurement_1_observed=True,
                            last_measurement_duration=duration,
                        )
                    )
                elif phase == 2:
                    expanded.append(
                        _advance(
                            node,
                            _assigned_item(event, "measurement_1", "additional_first_cycle_measurement"),
                            2,
                            _transition_score(event, 0.35),
                            measurement_1_observed=True,
                            last_measurement_duration=duration,
                        )
                    )
                elif phase in (3, 4):
                    expanded.append(
                        _advance(
                            node,
                            _assigned_item(event, "measurement_2", "measurement_after_formal_rewiring"),
                            4,
                            _transition_score(event, 0.8),
                            measurement_2_observed=True,
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
                                transition_penalty=-0.25,
                            ),
                            4,
                            -0.25 + 0.1 * _confidence(event),
                            measurement_2_observed=True,
                            last_measurement_duration=duration,
                        )
                    )
                else:
                    expanded.append(
                        _advance(node, _assigned_item(event, None, "measurement_after_terminal_phase"), phase, -0.4)
                    )
                continue

            if action == "writing_action":
                if phase in (0, 1, 2):
                    expanded.append(
                        _advance(
                            node,
                            _assigned_item(event, "recording_1", "first_writing_before_formal_rewiring"),
                            2,
                            _transition_score(event, 0.75),
                            recording_1_observed=True,
                        )
                    )
                elif phase == 3:
                    expanded.append(
                        _advance(
                            node,
                            _assigned_item(
                                event,
                                "recording_2",
                                "legacy_rewiring_then_writing_fallback",
                                inferred_stage=True,
                                measurement_2_observed=False,
                                fallback_source="legacy_v2_sequence_rule",
                                recording_search_aliases=["recording_2"],
                            ),
                            5,
                            _transition_score(event, 0.45),
                            recording_2_observed=True,
                        )
                    )
                elif phase in (4, 5):
                    batched = (
                        bool(node["measurement_1_observed"])
                        and bool(node["measurement_2_observed"])
                        and not bool(node["recording_1_observed"])
                    )
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
                                inferred_stage=False,
                                measurement_2_observed=True,
                            ),
                            5,
                            _transition_score(event, 0.7 if repeated else 0.8),
                            batched_recording=bool(node["batched_recording"]) or batched,
                            recording_2_observed=True,
                        )
                    )
                continue

            expanded.append(_advance(node, _assigned_item(event, None, "unsupported_action_type"), phase, -0.5))
        nodes = _prune(expanded, beam_width)

    best = max(nodes, key=_node_rank)
    assigned = list(best["assigned"])
    observed_intervals = [_interval(item) for item in assigned if item.get("stage") is not None]
    observed_stages = {str(item["stage"]) for item in observed_intervals}
    auxiliary_events = [item for item in assigned if item.get("action_type") == "auxiliary_action"]
    alternatives = [
        {
            "phase": int(node["phase"]),
            "score": round(float(node["score"]), 6),
            "direct_assignment_count": int(node["direct_assignment_count"]),
            "inferred_assignment_count": int(node["inferred_assignment_count"]),
            "batched_recording": bool(node["batched_recording"]),
        }
        for node in sorted(nodes, key=_node_rank, reverse=True)
    ]
    return {
        "assigned_events": assigned,
        "observed_stage_intervals": observed_intervals,
        "missing_stages": [stage for stage in STAGES if stage not in observed_stages],
        "review_reasons": [],
        "auxiliary_events": auxiliary_events,
        "decoder": {
            "type": "penalized_directed_graph_beam_search",
            "beam_width": beam_width,
            "selected_score": round(float(best["score"]), 6),
            "alternative_final_states": alternatives,
            "batched_recording": bool(best["batched_recording"]),
        },
        "analysis_termination": {
            "terminal_cleanup_reached": bool(best["terminal_reached"]),
            "terminal_cleanup_event_id": terminal_cleanup_event_id if best["terminal_reached"] else None,
            "discarded_after_terminal_event_ids": list(best["discarded_after_terminal"]),
            "discarded_after_terminal_count": len(best["discarded_after_terminal"]),
        },
    }
