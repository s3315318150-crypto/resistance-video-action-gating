#!/usr/bin/env python3
"""Prompt builders for hierarchical_v1 Map, Reduce, and boundary review."""

from __future__ import annotations

import json
from typing import Any


def _frame_identity_rules(image_ids: list[str]) -> str:
    return f"""图片身份规则：
1. 图片按本提示中的顺序提供，唯一合法 FRAME ID 为：{", ".join(image_ids)}。
2. 每张图底部新增黑色信息条，白字格式为 `FRAME ID=<id> | VIDEO T=<视频相对秒数>s`。信息条不覆盖原视频像素。
3. `VIDEO T=75.0s` 是视频相对时间，绝不是 `frame_00000075`；摄像机原画面中的日期时间也不是视频相对时间。
4. JSON 中的 frame_id 必须从合法 FRAME ID 列表或图片信息条原样复制。不能把秒数、图片序号或日期时间改写成 frame_id。
5. 时间只用于检查先后关系。动作类别必须来自可见像素，最终秒数由程序按源 frame_number / fps 计算。"""


def build_map_prompt(
    video_id: str,
    window: dict[str, Any],
    frames: list[dict[str, Any]],
) -> str:
    image_ids = [str(frame["image_id"]) for frame in frames]
    start, end = window["window_seconds"]
    return f"""你是一名严谨的中学物理实验视频判读员。视频内容是“伏安法测电阻”，视频名为 `{video_id}`。

你现在执行分层算法的 Map 步骤，只分析窗口 `{window['window_id']}`，范围为视频相对时间 {start:.3f}s–{end:.3f}s。相邻窗口会有 10 秒重叠，所以同一动作可能在别的窗口再次出现；不要猜测其他窗口的内容，也不要决定全局阶段。

{_frame_identity_rules(image_ids)}

实验画面背景：
- 桌上可见橙红色的电池盒、开关、电表等器材，颜色只能辅助定位，必须结合形状、导线连接、手部动作和前后帧判断。
- 实验可能包含摆放器材、接线、操作开关和观察电表、在纸上记录、修改线路、再次测量和最终整理。
- 本任务没有“换电池”基础动作或阶段。不要输出 battery_change，也不要因看到电池盒就推断发生换电池。
- 你只识别下列基础可见动作，不要在本步骤判断第一次或第二次：

基础动作定义：
1. `wiring_action`：手在插接、拔出、移动或调整导线，连接或重新连接电表、开关、待测电阻等线路，或者为了接线摆放相关器材。一张图若清楚看到手持导线正在接触接线柱，也可以作为证据；单纯手停在线路附近不够。
2. `measurement_action`：线路准备后，操作开关、手指停在开关操作位置、观察电流表或电压表、靠近表盘读取示数。若一张清晰图显示学生面对已连接电路读取仪表，且相邻图没有接线或书写，也可以报告测量，不要求必须同时看清开关闭合和表针变化。
3. `writing_action`：可见笔尖接触记录纸并处于填写、书写或计算姿态。优先使用相邻图片的位置变化确认；若只有一张图但笔尖、纸面和书写姿态都很清楚，也可以报告，不再把“连续多帧”作为硬门槛。仅拿着笔、笔悬空或纸面完全被遮挡仍不算。
4. `cleanup_action`：可见一段连续的最终整理过程，例如持续拆卸线路、拔下多根导线、卷起或集中收拢导线和器材。输出 cleanup_action 就表示这段整理完成后实验结束。仅拔下一根导线、暂时移动一个电表、为了继续实验而改线，必须归为 `wiring_action`，不能输出 cleanup_action。
5. `uncertain`：画面遮挡、抽帧间隔跨过动作，或证据不足以可靠区分上述动作。

判定方法：
1. 逐帧比较手、笔、导线、开关、电表和记录纸的位置变化。只能报告本窗口中直接可见的动作。
2. 一个 observation 表示一段连续可见证据。`first_frame_id` 是最早能支持该动作的帧，`last_frame_id` 是最后仍能支持该动作的帧，`representative_frame_id` 必须位于两者之间并最能说明动作。
3. 同一种动作中间若被另一种动作明确打断，应拆成两个 observation；相邻几帧持续同一动作应合为一个 observation。
4. 人坐着不动、交谈、寻找材料、身体遮挡、画外动作或仅凭实验常规推测，都不能补造动作。
5. 不输出七阶段名称，不输出 `circuit_wiring`、`measurement_1`、`recording_1`、`circuit_rewiring`、`measurement_2`、`recording_2` 或 `material_cleanup`。
6. 看见至少一个可靠的非 uncertain 动作时 decision=`observed`；没有上述动作时为 `no_action_observed` 且 observations=[]；证据主要不可判时为 `uncertain`。
7. 硬截断视觉描述分为两类：一是明确看到学生换座位、换人、人脸入镜、抬头闲聊或聊天；二是明确看到器材已经完成终态整理，例如“整理完”“整理完毕”“拆完”“全拆”“收完”“桌面清空”，或者明确看到橙红色仪器被放回、放到或移到桌子的左上角。此实验没有器材盒，不得描述“收进器材盒”。出现任一类时，必须在对应 observation 的 `evidence` 中直接写出画面实际支持的完成状态或“桌子左上角”。只有移动单件仪器、拔下一根导线、中途改线或尚未完成的收拢，不得写“整理完毕”“桌面清空”或“已放到桌子左上角”。若只看到这些结束特征而没有其他实验动作，仍输出一个 `uncertain` observation 保存该时间范围和证据，并令 decision=`uncertain`。

只输出一个合法 JSON 对象，不要 Markdown、解释前缀或代码围栏：
{{
  "window_id": "{window['window_id']}",
  "decision": "observed" | "no_action_observed" | "uncertain",
  "observations": [
    {{
      "action_type": "wiring_action" | "measurement_action" | "writing_action" | "cleanup_action" | "uncertain",
      "first_frame_id": "{image_ids[0]}",
      "last_frame_id": "{image_ids[0]}",
      "representative_frame_id": "{image_ids[0]}",
      "evidence": "不超过160字，只描述相邻画面中直接可见的变化",
      "confidence": 0.0
    }}
  ],
  "confidence": 0.0,
  "uncertainty": "不超过120字；无则为空字符串"
}}"""


def build_map_retry_prompt(base_prompt: str, errors: list[str]) -> str:
    guidance: list[str] = []
    if any(error.startswith("observation_") and error.endswith("_confidence_invalid") for error in errors):
        guidance.append(
            "每一个 observations 数组元素都必须单独包含 confidence 字段，值必须是 0.0 到 1.0 的 JSON 数字；"
            "不能只返回顶层 confidence，也不能省略、写成字符串或 null。"
        )
    detail = "" if not guidance else "具体修复要求：" + "".join(guidance)
    return (
        base_prompt
        + "\n\n上一版回答未通过本地 Map 契约校验，错误代码为："
        + "、".join(sorted(set(errors)))
        + "。"
        + detail
        + "请重新查看图片，只使用合法基础动作和真实 FRAME ID，输出完整 JSON。"
    )


def build_reduce_prompt(video_id: str, events: list[dict[str, Any]]) -> str:
    compact_events = [
        {
            "event_id": event["event_id"],
            "source_event_ids": event.get("source_event_ids", []),
            "window_ids": event.get("window_ids", []),
            "action_type": event["action_type"],
            "first_frame_id": event["first_frame_id"],
            "last_frame_id": event["last_frame_id"],
            "representative_frame_id": event["representative_frame_id"],
            "first_seconds_local": event["first_seconds"],
            "last_seconds_local": event["last_seconds"],
            "evidence": event["evidence"],
            "confidence": event["confidence"],
        }
        for event in events
    ]
    return f"""你是分层视频分析算法的全局 Reduce 审核员。视频是“伏安法测电阻”，视频名为 `{video_id}`。

下面的事件由多个有 10 秒重叠的一分钟窗口独立产生，并已由程序做第一轮重复合并。你不看原图，不重新识别动作，只根据事件的直接证据、时间关系、来源窗口和置信度完成全局一致性选择。

重要限制：
1. 只能引用下面真实存在的 `event_id`，不得新建事件、frame_id 或时间。
2. 每个事件必须且只能被放入 `accepted_event_ids` 或 `rejected_events`，不能遗漏、重复或同时接受和拒绝。
3. 相邻窗口对同一动作可能重复描述；保留证据更直接、时间更连贯的事件，重复或弱证据事件可拒绝。
4. 不要把预期实验顺序当作证据，不要在这里编号第一次/第二次，也不要补造没有事件支持的动作。
5. 时间范围重叠不自动等于冲突，因为两秒抽帧可能把相邻动作包在同一范围。只有代表帧相同且动作互斥、证据描述自相矛盾，或一个事件明显是误判时才列为冲突。
6. 首个已接受的 `cleanup_action` 是不可逆的实验结束屏障，必须填写为 `terminal_cleanup_event_id`。不要因为它之后又出现书写、碰导线、老师指导或其他动作而撤销终态；这些后续事件都是实验结束后的录像噪声。
7. 一旦出现 terminal cleanup，与它重叠或更晚的其他事件必须以 `post_terminal_cleanup` 拒绝。环境结束词、彻底整理词和“橙红色仪器放到桌子左上角”可以作为补充解释，但不是锁定 cleanup_action 的必要条件。
8. 只有候选事件中完全没有可信的 `cleanup_action` 时，`terminal_cleanup_event_id` 才填写 null。
9. 字段中的秒数由本地程序从源帧换算，只帮助比较时间；你的输出不得返回或修改秒数。

候选事件：
{json.dumps(compact_events, ensure_ascii=False, indent=2)}

只输出一个合法 JSON 对象，不要 Markdown：
{{
  "accepted_event_ids": ["evt_0001"],
  "rejected_events": [
    {{
      "event_id": "evt_0002",
      "reason": "duplicate" | "conflicts_with_stronger_evidence" | "insufficient_visual_evidence" | "outside_locked_segment" | "post_terminal_cleanup" | "other",
      "explanation": "不超过120字"
    }}
  ],
  "conflicts": [
    {{"event_ids": ["evt_0001", "evt_0002"], "resolution": "不超过160字，说明保留或拒绝依据"}}
  ],
  "terminal_cleanup_event_id": "evt_0001" | null,
  "confidence": 0.0,
  "uncertainty": "不超过160字；无则为空字符串"
}}"""


def build_reduce_retry_prompt(base_prompt: str, errors: list[str]) -> str:
    guidance: list[str] = []
    if "accepted_event_after_terminal_cleanup" in errors:
        guidance.append(
            "你把某个 cleanup 设为 terminal 后又接受了与其重叠或更晚的事件。"
            "cleanup_action 是不可逆终态，必须保留为 terminal_cleanup_event_id；"
            "将所有与其重叠或更晚的其他事件拒绝为 post_terminal_cleanup，不得因后续动作撤销最终整理。"
        )
    if any(error.startswith("terminal_cleanup_event_") for error in errors):
        guidance.append("terminal_cleanup_event_id 只能引用已接受的 cleanup_action；不能确认时请填写 null。")
    if "reduce_decision_not_exhaustive" in errors:
        guidance.append("逐项核对候选列表，确保每个 event_id 恰好出现一次。")
    detail = "" if not guidance else "具体修复要求：" + "".join(guidance)
    return (
        base_prompt
        + "\n\n上一版回答未通过本地 Reduce 契约校验，错误代码为："
        + "、".join(sorted(set(errors)))
        + "。"
        + detail
        + "请对每个候选 event_id 作且只作一次接受或拒绝决定；不得发明 ID，并输出完整 JSON。"
    )


def build_boundary_prompt(
    video_id: str,
    boundary: dict[str, Any],
    frames: list[dict[str, Any]],
    stage_labels: dict[str, str],
) -> str:
    image_ids = [str(frame["image_id"]) for frame in frames]
    from_stage = str(boundary["from_stage"])
    to_stage = str(boundary["to_stage"])
    return f"""你是“伏安法测电阻”视频动作边界复核员。视频名为 `{video_id}`。现在只复核边界 `{boundary['boundary_id']}`：从 `{from_stage}`（{stage_labels[from_stage]}）转为 `{to_stage}`（{stage_labels[to_stage]}）。

{_frame_identity_rules(image_ids)}

复核规则：
1. 图片是候选边界附近的连续等间隔抽帧。只判断指定的两个阶段，不重新分割整段视频。
2. `last_from_frame_id` 选择最后一张仍有直接证据支持前一阶段的图片；`first_to_frame_id` 选择第一张已有直接证据支持后一阶段的图片。
3. 两个 ID 必须来自合法列表，且 last_from 严格早于 first_to。程序将两帧之间作为边界不确定区间，并把 first_to 的源帧时间作为操作性边界。
4. 若抽帧中看不到前一阶段或后一阶段、动作被遮挡、两种动作无法分开，decision=`uncertain`，两个 frame_id 都填 null，不得猜测。
5. 不要根据实验标准流程补造动作。第二次测量和第二次记录仍必须有指定阶段的直接可见动作证据。

只输出一个合法 JSON 对象，不要 Markdown：
{{
  "boundary_id": "{boundary['boundary_id']}",
  "decision": "observed" | "uncertain",
  "last_from_frame_id": "{image_ids[0]}" | null,
  "first_to_frame_id": "{image_ids[1] if len(image_ids) > 1 else image_ids[0]}" | null,
  "evidence": "不超过160字，引用真实 FRAME ID 描述可见变化",
  "confidence": 0.0,
  "uncertainty": "不超过120字；无则为空字符串"
}}"""


def build_boundary_retry_prompt(base_prompt: str, errors: list[str]) -> str:
    return (
        base_prompt
        + "\n\n上一版回答未通过本地边界契约校验，错误代码为："
        + "、".join(sorted(set(errors)))
        + "。请只引用候选图片的真实 FRAME ID；不能确认时两个 ID 都用 null，并输出完整 JSON。"
    )
