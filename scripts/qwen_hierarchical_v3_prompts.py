#!/usr/bin/env python3
"""Prompt extensions for hierarchical v3."""

from __future__ import annotations

import json
from typing import Any

import qwen_hierarchical_v1_prompts as base


def build_map_prompt(video_id: str, window: dict[str, Any], frames: list[dict[str, Any]]) -> str:
    prompt = base.build_map_prompt(video_id, window, frames)
    prompt = prompt.replace(
        "- 本任务没有“换电池”基础动作或阶段。不要输出 battery_change，也不要因看到电池盒就推断发生换电池。",
        "- 七阶段中没有 battery_change 阶段。只有直接看到电池盒端子或接入节数发生改变时，才输出 auxiliary_action/battery_configuration_change；不能仅因看到电池盒就推断发生变化。",
    )
    prompt = prompt.replace(
        "5. `uncertain`：画面遮挡、抽帧间隔跨过动作，或证据不足以可靠区分上述动作。",
        """5. `auxiliary_action`：直接看到不属于七阶段但需要保留的行为。必须填写 `auxiliary_subtype`：换接电池盒端子或改变电池配置=`battery_configuration_change`；换座位/换人=`seat_change`；抬头闲聊=`social_interruption`；老师介入=`teacher_intervention`；喝水、玩手机等=`off_task`；其他无法归入主动作的器材操作=`unknown_manipulation`。换电池不再强行标成 wiring_action。\n6. `uncertain`：画面遮挡、抽帧间隔跨过动作，或证据不足以可靠区分上述动作。""",
    )
    prompt = prompt.replace(
        "输出 cleanup_action 就表示这段整理完成后实验结束。",
        "输出 cleanup_action 只表示这是最终整理候选；程序还会用动作前、中、后多帧复核是否真正结束。",
    )
    prompt = prompt.replace(
        '"action_type": "wiring_action" | "measurement_action" | "writing_action" | "cleanup_action" | "uncertain",',
        '"action_type": "wiring_action" | "measurement_action" | "writing_action" | "cleanup_action" | "auxiliary_action" | "uncertain",\n      "auxiliary_subtype": "battery_configuration_change" | "seat_change" | "social_interruption" | "teacher_intervention" | "off_task" | "unknown_manipulation" | null,',
    )
    prompt = prompt.replace(
        "人坐着不动、交谈、寻找材料、身体遮挡、画外动作或仅凭实验常规推测，都不能补造动作。",
        "人坐着不动或画外动作不能补造；直接可见的交谈、换座位、教师介入和离题活动应报告为 auxiliary_action。",
    )
    return prompt


def build_reduce_prompt(video_id: str, events: list[dict[str, Any]]) -> str:
    prompt = base.build_reduce_prompt(video_id, events)
    prompt = prompt.replace(
        "不要在这里编号第一次/第二次，也不要补造没有事件支持的动作。",
        "不要在这里编号第一次/第二次，也不要补造没有事件支持的动作。auxiliary_action 应保留为诊断事件，但不能被解释成七阶段。",
    )
    prompt = prompt.replace(
        "6. 首个已接受的 `cleanup_action` 是不可逆的实验结束屏障，必须填写为 `terminal_cleanup_event_id`。不要因为它之后又出现书写、碰导线、老师指导或其他动作而撤销终态；这些后续事件都是实验结束后的录像噪声。",
        "6. 首个可信 `cleanup_action` 只能作为待复核的终态候选，填写为 `terminal_cleanup_event_id`。程序随后会把动作前、中、后多帧重新发给视觉模型；未通过视觉复核时会撤销候选并恢复后续事件。",
    )
    prompt = prompt.replace(
        "7. 一旦出现 terminal cleanup，与它重叠或更晚的其他事件必须以 `post_terminal_cleanup` 拒绝。环境结束词、彻底整理词和“橙红色仪器放到桌子左上角”可以作为补充解释，但不是锁定 cleanup_action 的必要条件。",
        "7. 对 terminal cleanup 候选，与它重叠或更晚的其他事件暂以 `post_terminal_cleanup` 拒绝；这些事件仍会保留，若多帧复核不通过，程序必须恢复它们。文字关键词和正则只用于解释，不能代替原图复核。",
    )
    auxiliary = compact_auxiliary_events(events)
    prompt = prompt.replace(
        "候选事件：",
        f"辅助事件及 subtype（仅作诊断，不与同时发生的主动作互斥）：\n{auxiliary}\n\n候选事件：",
        1,
    )
    return prompt


def build_cleanup_confirmation_prompt(
    video_id: str,
    event: dict[str, Any],
    frames: list[dict[str, Any]],
) -> str:
    frame_ids = [str(frame["image_id"]) for frame in frames]
    return f"""你正在复核伏安法测电阻视频 `{video_id}` 的最终整理候选 `{event['event_id']}`。

图片按时间先后排列，合法 FRAME ID 为：{', '.join(frame_ids)}。必须比较整理前、动作中、动作结束和之后的画面，不能只凭单张图判断。

只有以下任一完成态成立且之后不再继续实验，才回答 completed_cleanup=yes：
1. 多根导线已经拆下，器材明显完成集中归拢或桌面实验区已清理；
2. 橙红色目标仪器已经放回桌子左上角；
3. 学生明确换座位/换人，且实验器材不再继续操作。

仅移动单件仪器、拔下一根导线、暂时推到一边、纠错改线或后续继续测量/书写，必须回答 no。看不清则回答 uncertain。

只输出一个 JSON：
{{
  "event_id": "{event['event_id']}",
  "completed_cleanup": "yes" | "no" | "uncertain",
  "multiple_wires_disconnected": "yes" | "no" | "uncertain",
  "instrument_returned_upper_left": "yes" | "no" | "uncertain",
  "seat_change_or_person_change": "yes" | "no" | "uncertain",
  "experiment_activity_continues_afterward": "yes" | "no" | "uncertain",
  "evidence_frame_ids": ["{frame_ids[0]}"],
  "evidence": "不超过180字，只描述直接可见变化",
  "confidence": 0.0
}}"""


def build_reverse_boundary_prompt(
    video_id: str,
    boundary: dict[str, Any],
    frames: list[dict[str, Any]],
) -> str:
    frame_ids = [str(frame["image_id"]) for frame in frames]
    return f"""复核视频 `{video_id}` 的关键边界 `{boundary['boundary_id']}`。请反向寻找：最早在哪张图已经出现开关操作、观察仪表或读取示数，并且此前最后一张图仍在插接、拔出或调整导线？

合法 FRAME ID：{', '.join(frame_ids)}。不要根据实验流程猜测；看不清就返回 uncertain。

只输出 JSON：
{{
  "boundary_id": "{boundary['boundary_id']}",
  "decision": "observed" | "uncertain",
  "last_from_frame_id": "{frame_ids[0]}" | null,
  "first_to_frame_id": "{frame_ids[1] if len(frame_ids) > 1 else frame_ids[0]}" | null,
  "evidence": "不超过160字",
  "confidence": 0.0,
  "uncertainty": "不超过120字；无则为空字符串"
}}"""


def compact_auxiliary_events(events: list[dict[str, Any]]) -> str:
    return json.dumps(
        [
            {
                "event_id": event.get("event_id"),
                "subtype": event.get("auxiliary_subtype"),
                "evidence": event.get("evidence"),
            }
            for event in events
            if event.get("action_type") == "auxiliary_action"
        ],
        ensure_ascii=False,
    )
