#!/usr/bin/env python3
"""Additional Map contracts for hierarchical v3 auxiliary actions."""

from __future__ import annotations

from typing import Any

import qwen_hierarchical_v1_contract as base


STAGE_SCHEMA_ID = "resistance_7stage_no_battery_v3"
BASE_ACTIONS = (
    "wiring_action",
    "measurement_action",
    "writing_action",
    "cleanup_action",
    "auxiliary_action",
    "uncertain",
)
AUXILIARY_SUBTYPES = {
    "battery_configuration_change",
    "seat_change",
    "social_interruption",
    "teacher_intervention",
    "off_task",
    "unknown_manipulation",
}


def validate_map_response(
    value: dict[str, Any] | None,
    window_id: str,
    frames: list[dict[str, Any]],
) -> list[str]:
    errors = list(base.validate_map_response(value, window_id, frames))
    if not isinstance(value, dict) or not isinstance(value.get("observations"), list):
        return sorted(set(errors))
    for index, observation in enumerate(value["observations"]):
        if not isinstance(observation, dict):
            continue
        subtype = observation.get("auxiliary_subtype")
        if observation.get("action_type") == "auxiliary_action":
            if subtype not in AUXILIARY_SUBTYPES:
                errors.append(f"observation_{index}_auxiliary_subtype_invalid")
        elif subtype is not None:
            errors.append(f"observation_{index}_unexpected_auxiliary_subtype")
    return sorted(set(errors))


def normalize_map_events(
    value: dict[str, Any],
    window_id: str,
    frames: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    events = base.normalize_map_events(value, window_id, frames)
    for event, observation in zip(events, value.get("observations", [])):
        if event.get("action_type") == "auxiliary_action":
            event["auxiliary_subtype"] = observation.get("auxiliary_subtype")
    return events
