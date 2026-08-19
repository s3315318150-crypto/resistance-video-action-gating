#!/usr/bin/env python3
"""Auxiliary-action contracts and conflict isolation for behavior-tolerant v2."""

from __future__ import annotations

from typing import Any

import qwen_hierarchical_v1_contract as contract
import qwen_hierarchical_v1_prompts as prompts
from qwen_hierarchical_v1_reduce import deduplicate_map_events, find_temporal_conflicts
from qwen_hierarchical_v2_temporal_guard_reduce import select_events_with_temporal_guard


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
    "teacher_intervention",
    "seat_change",
    "conversation",
    "phone_use",
    "off_task_behavior",
    "other_action",
}


def validate_map_response_auxiliary(
    value: dict[str, Any] | None,
    window_id: str,
    frames: list[dict[str, Any]],
) -> list[str]:
    errors = list(contract.validate_map_response(value, window_id, frames))
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


def normalize_map_events_auxiliary(
    value: dict[str, Any],
    window_id: str,
    frames: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    events = contract.normalize_map_events(value, window_id, frames)
    for event, observation in zip(events, value.get("observations", []), strict=True):
        if event.get("action_type") == "auxiliary_action":
            event["auxiliary_subtype"] = observation["auxiliary_subtype"]
    return events


def build_map_prompt_auxiliary(video_id: str, window: dict[str, Any], frames: list[dict[str, Any]]) -> str:
    prompt = prompts.build_map_prompt(video_id, window, frames)
    prompt = prompt.replace(
        "- 本任务没有“换电池”基础动作或阶段。不要输出 battery_change，也不要因看到电池盒就推断发生换电池。",
        "- 七阶段中没有换电池阶段。只有直接看到电池盒接线端或接入配置发生变化时，才输出 auxiliary_action/battery_configuration_change；仅看到电池盒不能推断变化。",
    )
    prompt = prompt.replace(
        "5. `uncertain`：画面遮挡、抽帧间隔跨过动作，或证据不足以可靠区分上述动作。",
        """5. `auxiliary_action`：直接看到不属于七阶段但需要保留的行为。必须填写 `auxiliary_subtype`：电池接入配置变化=`battery_configuration_change`；老师介入=`teacher_intervention`；换座位或换人=`seat_change`；闲聊=`conversation`；使用手机=`phone_use`；其他离题行为=`off_task_behavior`；其他可见但无法归入主动作的行为=`other_action`。\n6. `uncertain`：画面遮挡、抽帧间隔跨过动作，或证据不足以可靠区分上述动作。""",
    )
    prompt = prompt.replace(
        "人坐着不动、交谈、寻找材料、身体遮挡、画外动作或仅凭实验常规推测，都不能补造动作。",
        "人坐着不动、身体遮挡和画外动作不能补造；直接可见的交谈、换座位、教师介入、手机或离题行为应报告为 auxiliary_action。",
    )
    prompt = prompt.replace(
        '"action_type": "wiring_action" | "measurement_action" | "writing_action" | "cleanup_action" | "uncertain",',
        '"action_type": "wiring_action" | "measurement_action" | "writing_action" | "cleanup_action" | "auxiliary_action" | "uncertain",\n      "auxiliary_subtype": "battery_configuration_change" | "teacher_intervention" | "seat_change" | "conversation" | "phone_use" | "off_task_behavior" | "other_action" | null,',
    )
    prompt += (
        "\n\n辅助动作只保存直接观察，不替代同时发生的主动作。若同一时间既在接线又有老师介入，"
        "应分别输出 wiring_action 和 auxiliary_action。换座位或闲聊不能单独输出 cleanup_action。"
    )
    return prompt


def build_reduce_prompt_auxiliary(video_id: str, events: list[dict[str, Any]]) -> str:
    prompt = prompts.build_reduce_prompt(video_id, events)
    return prompt.replace(
        "不要在这里编号第一次/第二次，也不要补造没有事件支持的动作。",
        "不要在这里编号第一次/第二次，也不要补造没有事件支持的动作。auxiliary_action 必须保留为诊断事件，且不与同时发生的主动作互斥。",
    )


def deduplicate_map_events_auxiliary(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for event in events:
        item = dict(event)
        if item.get("action_type") == "auxiliary_action":
            item["action_type"] = f"auxiliary_action::{item.get('auxiliary_subtype', 'other_action')}"
        prepared.append(item)
    groups = deduplicate_map_events(prepared)
    for group in groups:
        action = str(group.get("action_type", ""))
        if action.startswith("auxiliary_action::"):
            group["action_type"] = "auxiliary_action"
            group["auxiliary_subtype"] = action.split("::", 1)[1]
    return groups


def find_temporal_conflicts_auxiliary(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return find_temporal_conflicts([event for event in events if event.get("action_type") != "auxiliary_action"])


def select_events_auxiliary(
    events: list[dict[str, Any]],
    reduce_result: dict[str, Any] | None,
    preserve_equal_confidence: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    main_events = [event for event in events if event.get("action_type") != "auxiliary_action"]
    auxiliary_events = [event for event in events if event.get("action_type") == "auxiliary_action"]
    if reduce_result is None:
        selected_main, selection = select_events_with_temporal_guard(
            main_events, None, preserve_equal_confidence
        )
    else:
        main_ids = {str(event["event_id"]) for event in main_events}
        main_result = {
            **reduce_result,
            "accepted_event_ids": [
                event_id for event_id in reduce_result.get("accepted_event_ids", []) if event_id in main_ids
            ],
            "rejected_events": [
                item
                for item in reduce_result.get("rejected_events", [])
                if isinstance(item, dict) and item.get("event_id") in main_ids
            ],
            "conflicts": [
                item
                for item in reduce_result.get("conflicts", [])
                if isinstance(item, dict)
                and isinstance(item.get("event_ids"), list)
                and all(event_id in main_ids for event_id in item["event_ids"])
            ],
        }
        selected_main, selection = select_events_with_temporal_guard(
            main_events, main_result, preserve_equal_confidence
        )
    terminal_id = selection.get("terminal_cleanup_event_id")
    terminal_event = next(
        (event for event in selected_main if str(event.get("event_id")) == terminal_id),
        None,
    )
    if terminal_event is None:
        selected_auxiliary = auxiliary_events
        post_terminal_auxiliary: list[dict[str, Any]] = []
    else:
        terminal_start = int(terminal_event["first_frame_number"])
        selected_auxiliary = [
            event for event in auxiliary_events if int(event["last_frame_number"]) < terminal_start
        ]
        post_terminal_auxiliary = [
            event for event in auxiliary_events if int(event["last_frame_number"]) >= terminal_start
        ]
    selected = sorted(
        [*selected_main, *selected_auxiliary],
        key=lambda item: (
            int(item["representative_frame_number"]),
            int(item["first_frame_number"]),
            str(item["event_id"]),
        ),
    )
    selection["accepted_auxiliary_event_ids"] = [str(event["event_id"]) for event in selected_auxiliary]
    selection["post_terminal_auxiliary_event_ids"] = [str(event["event_id"]) for event in post_terminal_auxiliary]
    selection["auxiliary_events_preserved_outside_main_conflicts"] = True
    return selected, selection
