#!/usr/bin/env python3
"""Deterministic reducer for resistance rubric 8.

The battery holder has three electrical taps::

    T0 -- cell 1 -- T1 -- cell 2 -- T2

Rubric 8 does not look for physical cell removal.  It looks for a completed
lead relocation from the two-cell pair ``T0-T2`` to either one-cell pair
``T0-T1`` or ``T1-T2`` while the switch is open, followed by a closed switch
state.  Evidence is never combined across episodes.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any


SCHEMA_VERSION = "resistance_disconnect_battery_sequence_core.v1"
TERMINALS = ("T0", "T1", "T2")
TWO_CELL_PAIR = ("T0", "T2")
ONE_CELL_PAIRS = {("T0", "T1"), ("T1", "T2")}
DECISIONS = {"pass", "fail"}

_TERMINAL_ALIASES = {
    "t0": "T0",
    "terminal0": "T0",
    "terminal_0": "T0",
    "leftouter": "T0",
    "left_outer": "T0",
    "leftend": "T0",
    "left_end": "T0",
    "t1": "T1",
    "terminal1": "T1",
    "terminal_1": "T1",
    "middle": "T1",
    "middletap": "T1",
    "middle_tap": "T1",
    "center": "T1",
    "centre": "T1",
    "t2": "T2",
    "terminal2": "T2",
    "terminal_2": "T2",
    "rightouter": "T2",
    "right_outer": "T2",
    "rightend": "T2",
    "right_end": "T2",
}

_OPEN_STATES = {"open", "opened", "off", "disconnected"}
_CLOSED_STATES = {"closed", "close", "on", "connected"}
_OPEN_ACTIONS = {"open", "opened", "opening"}
_CLOSE_ACTIONS = {"close", "closed", "closing"}
_TERMINAL_ACTIONS = {
    "disconnect",
    "insert",
    "lead_move",
    "move",
    "reconnect",
    "relocate",
    "remove",
    "terminal_relocation",
}


def _token(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"[\s-]+", "_", value.strip().lower())


def normalize_terminal(value: Any) -> str | None:
    """Return ``T0``, ``T1`` or ``T2`` for a supported terminal label."""

    token = _token(value)
    if token in {"t0", "t1", "t2"}:
        return token.upper()
    compact = token.replace("_", "")
    return _TERMINAL_ALIASES.get(token) or _TERMINAL_ALIASES.get(compact)


def normalize_terminals(value: Any) -> tuple[str, ...] | None:
    """Normalize a terminal collection while rejecting unknowns/duplicates."""

    if isinstance(value, str) or not isinstance(value, Sequence):
        return None
    normalized = [normalize_terminal(item) for item in value]
    if not normalized or any(item is None for item in normalized):
        return None
    terminals = tuple(sorted(normalized, key=TERMINALS.index))  # type: ignore[arg-type]
    if len(set(terminals)) != len(terminals):
        return None
    return terminals


def effective_series_cells(terminals: Any) -> int | None:
    """Return the electrically selected cell count for a stable terminal pair."""

    pair = normalize_terminals(terminals)
    if pair == TWO_CELL_PAIR:
        return 2
    if pair in ONE_CELL_PAIRS:
        return 1
    return None


def classify_relocation(before: Any, after: Any) -> dict[str, Any]:
    """Classify exactly the required completed two-cell to one-cell change."""

    before_pair = normalize_terminals(before)
    after_pair = normalize_terminals(after)
    before_cells = effective_series_cells(before_pair)
    after_cells = effective_series_cells(after_pair)
    completed = before_pair == TWO_CELL_PAIR and after_pair in ONE_CELL_PAIRS

    moved_from: str | None = None
    moved_to: str | None = None
    fixed_terminal: str | None = None
    if completed and before_pair is not None and after_pair is not None:
        moved_from = next(iter(set(before_pair) - set(after_pair)))
        moved_to = next(iter(set(after_pair) - set(before_pair)))
        fixed_terminal = next(iter(set(before_pair) & set(after_pair)))

    if completed:
        reason_code = "completed_outer_to_middle_relocation"
    elif before_pair is None or after_pair is None:
        reason_code = "invalid_terminal_pair"
    elif before_cells != 2:
        reason_code = "before_state_is_not_two_cells"
    elif after_cells != 1:
        reason_code = "after_state_is_not_one_cell"
    else:
        reason_code = "not_required_relocation"

    return {
        "completed": completed,
        "reason_code": reason_code,
        "before_connection": list(before_pair) if before_pair is not None else None,
        "after_connection": list(after_pair) if after_pair is not None else None,
        "effective_cells_before": before_cells,
        "effective_cells_after": after_cells,
        "moved_lead": {
            "from": moved_from,
            "to": moved_to,
            "fixed_terminal": fixed_terminal,
        }
        if completed
        else None,
    }


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _confidence(value: Any, default: float = 0.5) -> float:
    number = _finite_number(value)
    if number is None:
        return default
    return min(1.0, max(0.0, number))


def _observation_ref(observation: Mapping[str, Any], index: int, timestamp: float) -> dict[str, Any]:
    result: dict[str, Any] = {
        "observation_index": index,
        "timestamp_seconds": timestamp,
    }
    frame_id = observation.get("frame_id") or observation.get("image_id")
    if isinstance(frame_id, str) and frame_id.strip():
        result["frame_id"] = frame_id.strip()
    return result


def _switch_state(observation: Mapping[str, Any]) -> str | None:
    state = _token(observation.get("switch_state"))
    if state in _OPEN_STATES:
        return "open"
    if state in _CLOSED_STATES:
        return "closed"

    action = _token(observation.get("switch_action"))
    completed = observation.get("switch_action_completed", observation.get("action_completed", True))
    if completed is not False and action in _OPEN_ACTIONS:
        return "open"
    if completed is not False and action in _CLOSE_ACTIONS:
        return "closed"
    return None


def _terminal_value(observation: Mapping[str, Any]) -> Any:
    for key in ("battery_terminals", "connected_terminals", "terminal_connections"):
        if key in observation:
            return observation[key]
    return None


def _terminal_action_started(observation: Mapping[str, Any]) -> bool:
    action = _token(observation.get("terminal_action") or observation.get("battery_action"))
    return action in _TERMINAL_ACTIONS and observation.get("direct_battery_contact") is True


def _terminal_state_stable(observation: Mapping[str, Any]) -> bool:
    for key in ("terminal_state_stable", "battery_state_stable", "stable"):
        if key in observation:
            return observation[key] is True
    return True


def _result(
    decision: str,
    episode_id: str,
    reason_code: str,
    reason: str,
    confidence: float,
    *,
    ordered_chain: dict[str, Any] | None,
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    if decision not in DECISIONS:
        raise ValueError(f"invalid binary decision: {decision}")
    return {
        "schema_version": SCHEMA_VERSION,
        "criterion": "switch_open_then_T0-T2_to_one_cell_terminal_relocation_then_switch_closed",
        "episode_id": episode_id,
        "decision": decision,
        "predicted_score": 1 if decision == "pass" else 0,
        "confidence": round(_confidence(confidence), 3),
        "reason_code": reason_code,
        "reason": reason,
        "ordered_chain": ordered_chain,
        "diagnostics": diagnostics,
    }


def _failure(
    episode_id: str,
    reason_code: str,
    reason: str,
    confidence: float,
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    return _result(
        "fail",
        episode_id,
        reason_code,
        reason,
        confidence,
        ordered_chain=None,
        diagnostics=diagnostics,
    )


def evaluate_episode(episode: Any) -> dict[str, Any]:
    """Evaluate one episode without importing evidence from another episode.

    Expected observations are time ordered dictionaries.  A switch observation
    uses ``switch_state`` (``open``/``closed``).  A battery observation uses
    ``battery_terminals`` with one or two values from ``T0``/``T1``/``T2``.
    One-terminal observations and explicit direct-contact terminal actions can
    mark the start of a relocation; only a stable two-terminal after-state can
    complete it.
    """

    if not isinstance(episode, Mapping):
        return _failure(
            "unknown_episode",
            "invalid_episode",
            "Episode must be a mapping.",
            1.0,
            {"validation_errors": ["episode_not_mapping"], "observation_count": 0},
        )

    episode_id = str(episode.get("episode_id") or "unknown_episode")
    observations = episode.get("observations")
    if not isinstance(observations, list):
        return _failure(
            episode_id,
            "invalid_episode",
            "Episode observations must be a list.",
            1.0,
            {"validation_errors": ["observations_not_list"], "observation_count": 0},
        )

    diagnostics: dict[str, Any] = {
        "observation_count": len(observations),
        "validation_errors": [],
        "ignored_observations": [],
        "switch_events": [],
        "stable_terminal_states": [],
        "relocations": [],
        "violations": [],
    }
    normalized: list[tuple[int, float, Mapping[str, Any]]] = []
    previous_timestamp: float | None = None
    switch_states_by_time: dict[float, set[str]] = {}
    terminal_pairs_by_time: dict[float, set[tuple[str, ...]]] = {}

    episode_start = _finite_number(episode.get("start_seconds"))
    episode_end = _finite_number(episode.get("end_seconds"))
    if (episode_start is None) != (episode.get("start_seconds") is None):
        diagnostics["validation_errors"].append("invalid_episode_start")
    if (episode_end is None) != (episode.get("end_seconds") is None):
        diagnostics["validation_errors"].append("invalid_episode_end")
    if episode_start is not None and episode_end is not None and episode_start > episode_end:
        diagnostics["validation_errors"].append("episode_start_after_end")

    for index, observation in enumerate(observations):
        if not isinstance(observation, Mapping):
            diagnostics["validation_errors"].append(f"observation_{index}_not_mapping")
            continue
        timestamp = _finite_number(observation.get("timestamp_seconds"))
        if timestamp is None:
            diagnostics["validation_errors"].append(f"observation_{index}_invalid_timestamp")
            continue
        if previous_timestamp is not None and timestamp < previous_timestamp:
            diagnostics["validation_errors"].append(f"observation_{index}_non_monotonic_timestamp")
        previous_timestamp = timestamp
        if episode_start is not None and timestamp < episode_start:
            diagnostics["validation_errors"].append(f"observation_{index}_before_episode")
        if episode_end is not None and timestamp > episode_end:
            diagnostics["validation_errors"].append(f"observation_{index}_after_episode")

        normalized.append((index, timestamp, observation))
        state = _switch_state(observation)
        if state is not None:
            switch_states_by_time.setdefault(timestamp, set()).add(state)

        if _token(observation.get("battery_object")) not in {"rejected", "not_battery"}:
            terminals = normalize_terminals(_terminal_value(observation))
            if terminals is not None and len(terminals) == 2 and _terminal_state_stable(observation):
                terminal_pairs_by_time.setdefault(timestamp, set()).add(terminals)

    for timestamp, states in switch_states_by_time.items():
        if len(states) > 1:
            diagnostics["validation_errors"].append(
                f"conflicting_switch_states_at_{timestamp:g}"
            )
    for timestamp, pairs in terminal_pairs_by_time.items():
        if len(pairs) > 1:
            diagnostics["validation_errors"].append(
                f"conflicting_terminal_states_at_{timestamp:g}"
            )

    if diagnostics["validation_errors"]:
        return _failure(
            episode_id,
            "invalid_observation_timeline",
            "Observation timing or same-frame state evidence is invalid.",
            1.0,
            diagnostics,
        )

    switch_events: list[dict[str, Any]] = []
    outer_baseline: dict[str, Any] | None = None
    change_start: dict[str, Any] | None = None
    change_end_action: dict[str, Any] | None = None
    relocations: list[dict[str, Any]] = []

    for index, timestamp, observation in normalized:
        ref = _observation_ref(observation, index, timestamp)
        state = _switch_state(observation)
        if state is not None:
            switch_event = {
                **ref,
                "state": state,
                "confidence": _confidence(observation.get("confidence")),
            }
            switch_events.append(switch_event)
            diagnostics["switch_events"].append(switch_event)

        if _token(observation.get("battery_object")) in {"rejected", "not_battery"}:
            diagnostics["ignored_observations"].append(
                {**ref, "reason": "battery_object_rejected"}
            )
            continue

        raw_terminal_value = _terminal_value(observation)
        terminals = normalize_terminals(raw_terminal_value)
        if raw_terminal_value is not None and terminals is None:
            diagnostics["ignored_observations"].append(
                {**ref, "reason": "invalid_terminal_labels"}
            )

        if outer_baseline is not None and change_start is None and _terminal_action_started(observation):
            change_start = {
                **ref,
                "confidence": _confidence(observation.get("confidence")),
                "source": "direct_terminal_action",
            }

        if (
            outer_baseline is not None
            and change_start is not None
            and observation.get("terminal_rewire_completed") is True
        ):
            change_end_action = {
                **ref,
                "confidence": _confidence(observation.get("confidence")),
                "source": "explicit_reconnect_completion",
            }

        if terminals is not None and len(terminals) != 2:
            if outer_baseline is not None and terminals != tuple(outer_baseline["terminals"]):
                if change_start is None:
                    change_start = {
                        **ref,
                        "confidence": _confidence(observation.get("confidence")),
                        "source": "incomplete_terminal_state",
                    }
            continue

        if terminals is None or not _terminal_state_stable(observation):
            continue

        cells = effective_series_cells(terminals)
        terminal_state = {
            **ref,
            "terminals": list(terminals),
            "effective_cells": cells,
            "confidence": _confidence(observation.get("confidence")),
        }
        diagnostics["stable_terminal_states"].append(terminal_state)

        if terminals == TWO_CELL_PAIR:
            # Seeing the original stable state again means a pending attempt was
            # abandoned; this observation becomes the new before-state.
            outer_baseline = terminal_state
            change_start = None
            change_end_action = None
            continue

        if terminals in ONE_CELL_PAIRS and outer_baseline is not None:
            start = change_start or {
                **ref,
                "confidence": terminal_state["confidence"],
                "source": "direct_stable_state_transition",
            }
            relocation = classify_relocation(outer_baseline["terminals"], terminals)
            completion = (
                change_end_action
                if change_end_action is not None
                and float(start["timestamp_seconds"]) <= float(change_end_action["timestamp_seconds"]) <= timestamp
                else terminal_state
            )
            relocation.update(
                {
                    "before_evidence": outer_baseline,
                    "change_start_evidence": start,
                    "after_evidence": terminal_state,
                    "completion_evidence": completion,
                    "change_start_seconds": start["timestamp_seconds"],
                    "change_end_seconds": completion["timestamp_seconds"],
                }
            )
            relocations.append(relocation)
            diagnostics["relocations"].append(relocation)
            outer_baseline = None
            change_start = None
            change_end_action = None

    if not relocations:
        terminal_confidence = max(
            (float(item["confidence"]) for item in diagnostics["stable_terminal_states"]),
            default=0.25,
        )
        return _failure(
            episode_id,
            "no_completed_two_to_one_relocation",
            "No stable T0-T2 to T0-T1/T1-T2 terminal relocation was completed.",
            terminal_confidence,
            diagnostics,
        )

    chains: list[dict[str, Any]] = []
    for relocation_index, relocation in enumerate(relocations):
        change_start_seconds = float(relocation["change_start_seconds"])
        change_end_seconds = float(relocation["change_end_seconds"])
        prior_switch = [
            event for event in switch_events if float(event["timestamp_seconds"]) < change_start_seconds
        ]
        open_event = prior_switch[-1] if prior_switch and prior_switch[-1]["state"] == "open" else None

        if open_event is None:
            violation = {
                "relocation_index": relocation_index,
                "reason_code": "switch_not_open_before_relocation",
                "change_start_seconds": change_start_seconds,
            }
            diagnostics["violations"].append(violation)
            continue

        closed_during = [
            event
            for event in switch_events
            if event["state"] == "closed"
            and float(open_event["timestamp_seconds"])
            < float(event["timestamp_seconds"])
            <= change_end_seconds
        ]
        if closed_during:
            diagnostics["violations"].append(
                {
                    "relocation_index": relocation_index,
                    "reason_code": "switch_closed_during_relocation",
                    "evidence": closed_during,
                }
            )
            continue

        close_event = next(
            (
                event
                for event in switch_events
                if event["state"] == "closed"
                and float(event["timestamp_seconds"]) > change_end_seconds
            ),
            None,
        )
        if close_event is None:
            diagnostics["violations"].append(
                {
                    "relocation_index": relocation_index,
                    "reason_code": "switch_not_closed_after_relocation",
                    "change_end_seconds": change_end_seconds,
                }
            )
            continue

        chain_confidence = min(
            float(open_event["confidence"]),
            float(relocation["before_evidence"]["confidence"]),
            float(relocation["after_evidence"]["confidence"]),
            float(close_event["confidence"]),
        )
        chains.append(
            {
                "relocation_index": relocation_index,
                "switch_open": open_event,
                "terminal_relocation": relocation,
                "switch_close": close_event,
                "confidence": round(chain_confidence, 3),
            }
        )

    if diagnostics["violations"]:
        reason_code = str(diagnostics["violations"][0]["reason_code"])
        reason_by_code = {
            "switch_not_open_before_relocation": "The switch was not observed open before the terminal relocation began.",
            "switch_closed_during_relocation": "The switch was observed closed before the terminal relocation completed.",
            "switch_not_closed_after_relocation": "The switch was not observed closed after the terminal relocation completed.",
        }
        confidence_by_code = {
            "switch_not_open_before_relocation": 0.8,
            "switch_closed_during_relocation": 0.95,
            "switch_not_closed_after_relocation": 0.75,
        }
        return _failure(
            episode_id,
            reason_code,
            reason_by_code[reason_code],
            confidence_by_code[reason_code],
            diagnostics,
        )

    best_chain = max(chains, key=lambda item: float(item["confidence"]))
    return _result(
        "pass",
        episode_id,
        "ordered_sequence_confirmed",
        "The switch was open before and during the completed two-cell to one-cell terminal relocation, then closed afterwards.",
        float(best_chain["confidence"]),
        ordered_chain=best_chain,
        diagnostics=diagnostics,
    )


def aggregate_episodes(episodes: Any, video_id: Any = None) -> dict[str, Any]:
    """Aggregate independent episodes without joining their event timelines."""

    episode_values = episodes if isinstance(episodes, list) else []
    results = [evaluate_episode(item) for item in episode_values]
    passed = [item for item in results if item["decision"] == "pass"]
    decision = "pass" if passed else "fail"
    confidence = (
        max(float(item["confidence"]) for item in passed)
        if passed
        else max((float(item["confidence"]) for item in results), default=0.25)
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "video_id": None if video_id is None else str(video_id),
        "decision": decision,
        "predicted_score": 1 if decision == "pass" else 0,
        "confidence": round(confidence, 3),
        "reason_code": "at_least_one_episode_passed" if passed else "no_episode_passed",
        "passing_episode_ids": [item["episode_id"] for item in passed],
        "episodes": results,
        "diagnostics": {
            "aggregation": "independent_episode_results_only",
            "episode_count": len(results),
            "cross_episode_evidence_fusion": False,
        },
    }
