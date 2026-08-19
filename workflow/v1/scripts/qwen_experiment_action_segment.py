#!/usr/bin/env python3
"""Classify actions inside previously locked volt-ampere experiment windows.

The workflow is deliberately coarse-to-fine:
1. Qwen classifies five-second images into the high-level experiment stages.
2. Each proposed transition is rechecked in a local one-frame-per-second window.

The input experiment start/end window is read-only.  This script never revises
the earlier experiment-boundary decision.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
try:
    from openai import OpenAI
except ModuleNotFoundError:
    from qwen_experiment_segment_judge import OpenAI

import qwen_experiment_segment_judge as qwen_base


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = ROOT / "outputs" / "marker_filter"
DEFAULT_SEGMENT_SUMMARY = ROOT / "outputs" / "experiment_boundary"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "action_segments"

STAGES = (
    "circuit_wiring",
    "measurement_1",
    "recording_1",
    "circuit_rewiring",
    "measurement_2",
    "recording_2",
    "material_cleanup",
)
LEGACY_IGNORED_MISSING_STAGES = {"battery_replacement"}
STAGE_ORDER = {stage: index for index, stage in enumerate(STAGES)}
STAGE_LABELS = {
    "circuit_wiring": "连线",
    "measurement_1": "第一次测量",
    "recording_1": "第一次记录",
    "circuit_rewiring": "重新连线",
    "measurement_2": "第二次测量",
    "recording_2": "第二次记录",
    "material_cleanup": "整理材料",
}
FINE_DECISIONS = {"first_stable_target", "uncertain", "not_observed"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def format_clock(seconds: Any) -> str:
    if not isinstance(seconds, (int, float)):
        return "未观察到"
    total = max(0.0, float(seconds))
    minutes = int(total // 60)
    remainder = total - minutes * 60
    if abs(remainder - round(remainder)) < 0.0005:
        second_text = f"{int(round(remainder)):02d}"
    else:
        second_text = f"{remainder:06.3f}"
    return f"{minutes:02d}:{second_text}"


def unique_timestamps(start_seconds: float, end_seconds: float, interval_seconds: float) -> list[float]:
    if end_seconds < start_seconds:
        raise ValueError("end_seconds must not precede start_seconds")
    values = [float(start_seconds)]
    candidate = start_seconds + interval_seconds
    while candidate < end_seconds - 0.01:
        values.append(candidate)
        candidate += interval_seconds
    if end_seconds - values[-1] > 0.01:
        values.append(float(end_seconds))
    return values


def extract_window(
    manifest: dict[str, Any],
    output_dir: Path,
    start_seconds: float,
    end_seconds: float,
    interval_seconds: float,
    longest_edge: int,
    id_prefix: str,
) -> list[dict[str, Any]]:
    source = Path(str(manifest["source_video"]))
    metadata = manifest["video_metadata"]
    fps = float(metadata["fps"])
    frame_count = int(metadata["frame_count"])
    maximum_seconds = (frame_count - 1) / fps
    start_seconds = max(0.0, min(float(start_seconds), maximum_seconds))
    end_seconds = max(start_seconds, min(float(end_seconds), maximum_seconds))
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open source video: {source}")
    records: list[dict[str, Any]] = []
    try:
        for index, requested_seconds in enumerate(unique_timestamps(start_seconds, end_seconds, interval_seconds), start=1):
            frame_number = min(frame_count - 1, max(0, int(round(requested_seconds * fps))))
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError(f"Unable to read frame {frame_number} from {source}")
            image_id = f"{id_prefix}_{index:03d}"
            timestamp_seconds = round(frame_number / fps, 3)
            image = qwen_base.resize_for_model(frame, longest_edge)
            image = qwen_base.add_relative_timestamp_banner(image, image_id, timestamp_seconds)
            image_path = output_dir / f"{image_id}_{frame_number:08d}_{timestamp_seconds:010.3f}s.jpg"
            if not cv2.imwrite(str(image_path), image, [cv2.IMWRITE_JPEG_QUALITY, 84]):
                raise RuntimeError(f"Unable to write image: {image_path}")
            records.append({
                "image_id": image_id,
                "frame_number": frame_number,
                "timestamp_seconds": timestamp_seconds,
                "path": str(image_path.resolve()),
                "visible_banner": f"FRAME ID={image_id} | VIDEO T={timestamp_seconds:.1f}s",
            })
    finally:
        capture.release()
    return records


def build_coarse_prompt(image_ids: list[str]) -> str:
    stage_lines = "\n".join(
        f"* `{stage}`（{STAGE_LABELS[stage]}）：{definition}"
        for stage, definition in (
            ("circuit_wiring", "摆放器材、插接导线、连接电池盒、开关、电表或待测电阻。"),
            ("measurement_1", "电路已准备好后观察仪表、闭合电路后测量；不是书写记录。"),
            ("recording_1", "在纸上持续记录读数、计算或填写表格。"),
            ("circuit_rewiring", "在首次测量前或之后，明确拆开、修改或重新插接实验线路以继续实验。它不是整理结束，也不是单独的滑动变阻器调节。"),
            ("measurement_2", "重新连线后，再次观察仪表并测量。"),
            ("recording_2", "第二轮测量后的持续书写、计算或填写表格。"),
            ("material_cleanup", "开始拆下导线、断开线路、收拢并归位器材。"),
        )
    )
    return f"""你是一名严谨的中学物理实验视频判读员。以下图片全部来自同一名学生已经确认开始、尚未结束的一次“伏安法测电阻”实验。图片按时间顺序排列，唯一可用的图片 ID 为：{", ".join(image_ids)}。

你的任务不是重新判断实验开始或结束，也不得移动给定实验窗口。请只把窗口内部的主要活动粗分为高层阶段。

图片底部左侧可能出现 `FRAME ID=coarse_001 | VIDEO T=80.0s`：
* `FRAME ID` 是唯一可填写到 JSON 的图片 ID，必须从上方列表或图片中的 `FRAME ID=` 原样复制。
* `VIDEO T` 是视频相对时间，不是帧号，也不是右下角摄像机日期时间。绝不能把 `VIDEO T=55.0s` 写成 `coarse_055`。
* 时间标签仅帮助核对顺序，任何阶段都必须由实际可见动作支持。

阶段定义：
{stage_lines}

规则：
1. 可能没有第二次测量、第二次记录或整理材料；缺失时必须写入 missing_stages，不能按预期流程补造。
2. 常见顺序是：连线 -> 第一次测量 -> 第一次记录 -> 重新连线 -> 第二次测量 -> 第二次记录 -> 整理材料。可跳过阶段。`circuit_rewiring` 可以在首次测量前出现，也可以在后续实验中重复出现；其他规范阶段不可重复或倒序。若镜头晃动、学生离座或遮挡导致第二次测量不可见，但之后明确看到第二次记录，可输出 `recording_2` 并把缺失测量的区间写入 uncertain_intervals；不得虚构 `measurement_2`。
3. `material_cleanup` 只能是最后一个阶段。若之后仍有重新接线、测量、写数据或其他继续实验的动作，前面的拆线/移动器材必须分类为 `circuit_rewiring` 或放入 uncertain_intervals，绝不能写为整理材料。
4. `measurement_2` 必须在明确重新连线之后，且需要看到重新观察仪表或测量的动作。若后段只是拆线、收拢或归位器材，应为 `material_cleanup`，不要为了凑第二次测量而猜测。
5. 仅拿笔、看纸、短暂停顿、拿其他材料、遮挡、镜头变化，不足以单独形成“记录”。无法判断的时段写入 uncertain_intervals，不要硬分类。
6. 不得把滑动变阻器调节设为阶段，也不得臆测仪表读数、开关状态或电路接法。特别是不能在 evidence 中称“调节滑动变阻器”。
7. 每一阶段 start_frame_id 是该阶段第一个有稳定可见证据的粗抽帧；end_frame_id 是进入下一阶段前最后一个仍属该阶段的粗抽帧。相邻阶段之间的精确边界稍后会复核。
8. 只输出观察到的主要阶段。coarse_segments 至少 1 条、最多 12 条；只有 `circuit_rewiring` 可以出现多次，每次都必须有明确重新接线证据。

只输出一个合法 JSON 对象：
{{
  "coarse_segments": [
    {{
      "stage": "circuit_wiring",
      "start_frame_id": "coarse_001",
      "end_frame_id": "coarse_006",
      "evidence": "不超过50字，只描述可见动作",
      "confidence": 0.0
    }}
  ],
  "missing_stages": [],
  "uncertain_intervals": [
    {{"start_frame_id": "coarse_010", "end_frame_id": "coarse_011", "reason": "不超过40字"}}
  ],
  "overall_evidence": "不超过100字，按时间概括实际可见活动",
  "confidence": 0.0,
  "needs_local_refinement": true,
  "uncertainty": "不超过60字；无则为空字符串"
}}"""


def build_refinement_prompt(
    image_ids: list[str],
    from_stage: str,
    to_stage: str,
) -> str:
    return f"""你正在精定位一处伏安法测电阻实验的动作转折。以下图片按每秒一张、按时间顺序提供，唯一图片 ID 为：{", ".join(image_ids)}。

已知粗分类认为活动可能从“{STAGE_LABELS[from_stage]}（{from_stage}）”转为“{STAGE_LABELS[to_stage]}（{to_stage}）”。请找出新阶段最早稳定成立的图片。

图片底部左侧的 `FRAME ID=` 是唯一可返回的图片 ID；`VIDEO T=` 只是相对秒数，绝不能据此编造图片 ID。不能仅依时间标签猜测，必须看见目标阶段的动作。

判定标准：
* `circuit_wiring` 到 `measurement_1`：连线基本完成后，开始稳定观察仪表或进行测量。
* `measurement_1` 到 `recording_1`：出现持续、明确的在记录纸上书写/填写，不是短暂拿笔或看纸。
* `recording_1` 到 `circuit_rewiring`：第一次记录后明确拆开、修改或重新插接实验线路以继续实验。
* `circuit_rewiring` 到 `measurement_2`：重新连线后，重新观察仪表或开始第二次测量。
* `measurement_2` 到 `recording_2`：第二轮测量后持续记录。
* 到 `material_cleanup`：第一次明确开始拆线、断开线路或收拢归位器材。

转折证据规则：
1. before_frame_id 必须是 transition_frame_id 之前的一张图片，并且只展示前一阶段 `{from_stage}` 的主要动作。
2. transition_frame_id 必须展示新阶段 `{to_stage}` 的稳定、直接证据。它不能只是拿笔、手部遮挡、停顿或准备下一步。
3. 若局部窗口第一张已经是新阶段、却看不到前一阶段，说明转折发生得更早；不得把第一张当作精确转折，返回 null 和 uncertain。
4. 若局部图片无法确认转折，transition_frame_id 必须为 null，decision 为 uncertain 或 not_observed。不要为了完整流程而猜测。

只输出一个合法 JSON 对象：
{{
  "transition_frame_id": "fine_001" | null,
  "decision": "first_stable_target" | "uncertain" | "not_observed",
  "before_frame_id": "fine_001" | null,
  "before_stage": "circuit_wiring" | null,
  "target_stage": "measurement_1" | null,
  "before_visible_evidence": "不超过50字，只描述 before_frame_id 的前一阶段动作",
  "target_visible_evidence": "不超过50字，只描述 transition_frame_id 的目标阶段动作",
  "confidence": 0.0,
  "uncertainty": "不超过60字；无则为空字符串"
}}"""


def build_chunk_coarse_prompt(image_ids: list[str], chunk_index: int, chunk_count: int) -> str:
    """Classify one contiguous 1 fps fragment without assuming its start stage."""

    stage_lines = "\n".join(
        f"* `{stage}`（{STAGE_LABELS[stage]}）：{definition}"
        for stage, definition in (
            ("circuit_wiring", "摆放器材、插接导线、连接电池盒、开关、电表或待测电阻。"),
            ("measurement_1", "电路已准备好后观察仪表、闭合电路后测量；不是书写记录。"),
            ("recording_1", "在纸上持续记录读数、计算或填写表格。"),
            ("circuit_rewiring", "明确拆开、修改或重新插接实验线路以继续实验。"),
            ("measurement_2", "重新连线后，再次观察仪表并测量。"),
            ("recording_2", "第二轮测量后的持续书写、计算或填写表格。"),
            ("material_cleanup", "开始拆下导线、断开线路、收拢并归位器材。"),
        )
    )
    return f"""你正在判读伏安法测电阻实验的一段连续视频片段。这是整个实验窗口中的第 {chunk_index}/{chunk_count} 个时间块，不保证从实验开始，也不保证包含完整七阶段。图片按每秒一张、按时间顺序提供，唯一图片 ID 为：{", ".join(image_ids)}。

只依据可见动作，把这个时间块内实际观察到的连续主活动标为下列阶段之一：
{stage_lines}

规则：
1. 不要因预期流程补造未出现的阶段；时间块可以从任意阶段开始或结束。
2. `recording_1` / `recording_2` 必须是持续、明确的书写、填写或计算；拿笔、看纸或短暂停顿不足以成立。
3. 无法判断时写入 uncertain_intervals，不得把仪表读数、接线正确性或开关状态写入证据。
4. start_frame_id 和 end_frame_id 必须是本块图片列表中的原始 ID。
5. 若本块全部静止、遮挡或无法由可见动作判断阶段，允许 `coarse_segments` 为 []；此时 `missing_stages` 必须列出全部七个阶段，并用至少一个覆盖本块的 `uncertain_intervals` 说明原因。不得用空列表回避本应可见的明确书写、连线或整理动作。
6. 只输出一个合法 JSON 对象，不要 Markdown：
{{
  "coarse_segments": [
    {{"stage": "recording_1", "start_frame_id": "coarse_001", "end_frame_id": "coarse_006", "evidence": "不超过50字，只描述可见动作", "confidence": 0.0}}
  ],
  "missing_stages": [],
  "uncertain_intervals": [
    {{"start_frame_id": "coarse_010", "end_frame_id": "coarse_011", "reason": "不超过40字"}}
  ],
  "overall_evidence": "不超过100字，只概括本时间块可见活动",
  "confidence": 0.0,
  "needs_local_refinement": true,
  "uncertainty": "不超过60字；无则为空字符串"
}}"""


def call_qwen(client: OpenAI, prompt: str, frames: list[dict[str, Any]], max_tokens: int) -> dict[str, Any]:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for frame in frames:
        content.append({"type": "image_url", "image_url": {"url": qwen_base.image_data_url(Path(frame["path"]))}})
    completion = client.chat.completions.create(
        model=qwen_base.MODEL,
        messages=[{"role": "user", "content": content}],
        max_tokens=max_tokens,
        temperature=0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    choice = completion.choices[0]
    raw = choice.message.content or ""
    result: dict[str, Any] = {
        "finish_reason": choice.finish_reason or "unknown",
        "raw_model_content": raw,
        "parsed": False,
    }
    if not isinstance(raw, str) or not raw.strip():
        result["parse_error"] = "Qwen did not return text content"
        return result
    try:
        result["parsed_result"] = qwen_base.parse_json(raw)
        result["parsed"] = True
    except (json.JSONDecodeError, ValueError) as exc:
        result["parse_error"] = str(exc)
    return result


def valid_confidence(value: Any) -> bool:
    return isinstance(value, (int, float)) and 0.0 <= float(value) <= 1.0


def stage_sequence_errors(stages: list[str]) -> list[str]:
    """Validate real action transitions while allowing repeated circuit rewiring."""
    allowed_next = {
        "circuit_wiring": {"circuit_rewiring", "measurement_1", "material_cleanup"},
        "circuit_rewiring": {"circuit_rewiring", "measurement_1", "measurement_2", "recording_2", "material_cleanup"},
        "measurement_1": {"recording_1", "circuit_rewiring", "material_cleanup"},
        "recording_1": {"circuit_rewiring", "material_cleanup"},
        "measurement_2": {"recording_2", "circuit_rewiring", "material_cleanup"},
        "recording_2": {"circuit_rewiring", "material_cleanup"},
        "material_cleanup": set(),
    }
    errors: list[str] = []
    seen: set[str] = set()
    for index, stage in enumerate(stages):
        if stage != "circuit_rewiring" and stage in seen:
            errors.append("coarse_stage_duplicate")
        if index > 0 and stage not in allowed_next.get(stages[index - 1], set()):
            errors.append("coarse_stage_transition_invalid")
        if stage == "measurement_2":
            if "measurement_1" not in seen:
                errors.append("measurement_2_without_measurement_1")
            if "circuit_rewiring" not in seen:
                errors.append("measurement_2_without_reconfiguration")
        seen.add(stage)
    if "material_cleanup" in stages and stages[-1] != "material_cleanup":
        errors.append("material_cleanup_not_final_stage")
    return errors


def validate_coarse(value: dict[str, Any], frames: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    by_id = {str(frame["image_id"]): index for index, frame in enumerate(frames)}
    segments = value.get("coarse_segments")
    if not isinstance(segments, list) or not 1 <= len(segments) <= 12:
        errors.append("coarse_segments_invalid")
    else:
        previous_end = -1
        for index, segment in enumerate(segments):
            if not isinstance(segment, dict):
                errors.append(f"coarse_segment_{index}_not_object")
                continue
            stage = segment.get("stage")
            start_id, end_id = segment.get("start_frame_id"), segment.get("end_frame_id")
            if stage not in STAGES:
                errors.append(f"coarse_segment_{index}_stage_invalid")
                continue
            if start_id not in by_id or end_id not in by_id:
                errors.append(f"coarse_segment_{index}_frame_id_invalid")
                continue
            if by_id[start_id] > by_id[end_id]:
                errors.append(f"coarse_segment_{index}_end_before_start")
            if by_id[start_id] <= previous_end:
                errors.append("coarse_segments_overlap_or_unsorted")
            previous_end = max(previous_end, by_id[end_id])
            if not isinstance(segment.get("evidence"), str) or not segment["evidence"].strip():
                errors.append(f"coarse_segment_{index}_evidence_invalid")
            if not valid_confidence(segment.get("confidence")):
                errors.append(f"coarse_segment_{index}_confidence_invalid")
        if isinstance(segments, list):
            stages = [segment.get("stage") for segment in segments if isinstance(segment, dict)]
            if all(stage in STAGES for stage in stages):
                errors.extend(stage_sequence_errors(stages))
    missing = value.get("missing_stages")
    if not isinstance(missing, list) or any(item not in STAGES for item in missing):
        errors.append("missing_stages_invalid")
    elif isinstance(segments, list):
        present = {segment.get("stage") for segment in segments if isinstance(segment, dict)}
        if present & set(missing):
            errors.append("present_stage_listed_missing")
    uncertain = value.get("uncertain_intervals")
    if not isinstance(uncertain, list):
        errors.append("uncertain_intervals_invalid")
    elif any(
        not isinstance(item, dict)
        or item.get("start_frame_id") not in by_id
        or item.get("end_frame_id") not in by_id
        or not isinstance(item.get("reason"), str)
        or not item["reason"].strip()
        for item in uncertain
    ):
        errors.append("uncertain_interval_entry_invalid")
    if not isinstance(value.get("overall_evidence"), str) or not value["overall_evidence"].strip():
        errors.append("overall_evidence_invalid")
    if not valid_confidence(value.get("confidence")):
        errors.append("overall_confidence_invalid")
    if not isinstance(value.get("needs_local_refinement"), bool):
        errors.append("needs_local_refinement_invalid")
    if not isinstance(value.get("uncertainty"), str):
        errors.append("uncertainty_invalid")
    return sorted(set(errors))


def normalize_terminal_cleanup(
    value: dict[str, Any],
    frames: list[dict[str, Any]],
    enabled: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Keep the first cleanup segment and exclude later coarse-stage output.

    The original parsed model response is retained separately in the run record.
    This normalization is only the locally consumed action prefix and makes the
    terminal policy auditable without rewriting the model response.
    """
    segments = value.get("coarse_segments")
    default = {
        "enabled": enabled,
        "reached": False,
        "reason": "material_cleanup_not_observed" if enabled else "terminal_cleanup_disabled",
        "discarded_segment_count": 0,
        "not_evaluated_after_cleanup": [],
    }
    if not enabled or not isinstance(segments, list):
        return value, default
    by_id = {str(frame.get("image_id")): frame for frame in frames if isinstance(frame, dict)}
    for index, segment in enumerate(segments):
        if not isinstance(segment, dict) or segment.get("stage") != "material_cleanup":
            continue
        start = by_id.get(str(segment.get("start_frame_id")))
        end = by_id.get(str(segment.get("end_frame_id")))
        if start is None or end is None:
            return value, {**default, "reason": "material_cleanup_frame_unresolved"}
        normalized = dict(value)
        normalized["coarse_segments"] = [dict(item) for item in segments[: index + 1] if isinstance(item, dict)]
        legacy_missing: list[str] = []
        missing = value.get("missing_stages")
        if isinstance(missing, list):
            legacy_missing = [item for item in missing if item in LEGACY_IGNORED_MISSING_STAGES]
            normalized["missing_stages"] = [item for item in missing if item not in LEGACY_IGNORED_MISSING_STAGES]
        cutoff_ids = {
            str(frame.get("image_id"))
            for frame in frames
            if isinstance(frame, dict)
            and isinstance(frame.get("timestamp_seconds"), (int, float))
            and math.isfinite(float(frame["timestamp_seconds"]))
            and float(frame["timestamp_seconds"]) <= float(end["timestamp_seconds"])
        }
        intervals = value.get("uncertain_intervals")
        if isinstance(intervals, list):
            normalized["uncertain_intervals"] = [
                dict(item)
                for item in intervals
                if isinstance(item, dict)
                and str(item.get("start_frame_id")) in cutoff_ids
                and str(item.get("end_frame_id")) in cutoff_ids
            ]
        discarded = [item for item in segments[index + 1 :] if isinstance(item, dict)]
        return normalized, {
            "enabled": True,
            "reached": True,
            "reason": "material_cleanup_is_terminal_action_boundary",
            "cleanup_start_seconds": float(start["timestamp_seconds"]),
            "cleanup_end_seconds": float(end["timestamp_seconds"]),
            "retrieval_cutoff_seconds": float(start["timestamp_seconds"]),
            "discarded_segment_count": len(discarded),
            "legacy_missing_stage_labels_ignored": legacy_missing,
            "not_evaluated_after_cleanup": [
                str(item.get("stage")) for item in discarded if isinstance(item.get("stage"), str)
            ],
        }
    return value, default


def validate_refinement(
    value: dict[str, Any],
    frames: list[dict[str, Any]],
    expected_from_stage: str,
    expected_to_stage: str,
) -> list[str]:
    errors: list[str] = []
    by_id = {str(frame["image_id"]): index for index, frame in enumerate(frames)}
    ids = set(by_id)
    transition_id = value.get("transition_frame_id")
    before_id = value.get("before_frame_id")
    decision = value.get("decision")
    if transition_id is not None and transition_id not in ids:
        errors.append("transition_frame_id_invalid")
    if before_id is not None and before_id not in ids:
        errors.append("before_frame_id_invalid")
    if decision not in FINE_DECISIONS:
        errors.append("decision_invalid")
    if transition_id is None and decision == "first_stable_target":
        errors.append("target_decision_without_frame")
    if transition_id is not None and decision != "first_stable_target":
        errors.append("frame_without_target_decision")
    if transition_id is not None:
        if before_id is None:
            errors.append("transition_without_before_evidence")
        elif before_id in by_id and transition_id in by_id and by_id[before_id] >= by_id[transition_id]:
            errors.append("before_not_before_transition")
        if transition_id in by_id and by_id[transition_id] == 0:
            errors.append("transition_at_local_window_start")
        if value.get("before_stage") != expected_from_stage:
            errors.append("before_stage_mismatch")
        if value.get("target_stage") != expected_to_stage:
            errors.append("target_stage_mismatch")
    if not isinstance(value.get("before_visible_evidence"), str) or not value["before_visible_evidence"].strip():
        errors.append("before_evidence_invalid")
    if not isinstance(value.get("target_visible_evidence"), str) or not value["target_visible_evidence"].strip():
        errors.append("target_evidence_invalid")
    if transition_id is not None and isinstance(value.get("target_visible_evidence"), str):
        target_evidence = value["target_visible_evidence"]
        if expected_to_stage in {"measurement_1", "measurement_2"} and any(
            word in target_evidence for word in ("书写", "记录", "填写", "写字")
        ):
            errors.append("target_evidence_conflicts_with_measurement")
    if not valid_confidence(value.get("confidence")):
        errors.append("confidence_invalid")
    if not isinstance(value.get("uncertainty"), str):
        errors.append("uncertainty_invalid")
    return sorted(set(errors))


def frame_by_id(frames: list[dict[str, Any]], image_id: Any) -> dict[str, Any] | None:
    if not isinstance(image_id, str):
        return None
    return next((frame for frame in frames if frame["image_id"] == image_id), None)


def run_refinement_attempt(
    client: OpenAI,
    manifest: dict[str, Any],
    output_dir: Path,
    from_stage: str,
    to_stage: str,
    fine_fps: float,
    longest_edge: int,
    max_tokens: int,
    local_start: float,
    local_end: float,
) -> dict[str, Any]:
    frames = extract_window(
        manifest,
        output_dir,
        local_start,
        local_end,
        1.0 / fine_fps,
        longest_edge,
        "fine",
    )
    raw = call_qwen(client, build_refinement_prompt([frame["image_id"] for frame in frames], from_stage, to_stage), frames, max_tokens)
    parsed = raw.get("parsed_result")
    errors = (
        validate_refinement(parsed, frames, from_stage, to_stage)
        if isinstance(parsed, dict) else ["qwen_response_not_parsed"]
    )
    selected = frame_by_id(frames, parsed.get("transition_frame_id")) if isinstance(parsed, dict) else None
    if selected is None:
        errors.append("transition_not_observed")
    return {
        "local_window_seconds": [local_start, local_end],
        "input_frames": frames,
        "qwen": raw,
        "selected_seconds": selected.get("timestamp_seconds") if selected else None,
        "refinement_valid": not errors,
        "refinement_errors": sorted(set(errors)),
    }


def refine_transition(
    client: OpenAI,
    manifest: dict[str, Any],
    output_dir: Path,
    fixed_start: float,
    fixed_end: float,
    candidate_seconds: float,
    from_stage: str,
    to_stage: str,
    fine_fps: float,
    fine_margin_seconds: float,
    longest_edge: int,
    max_tokens: int,
    max_attempts: int,
) -> dict[str, Any]:
    local_start = max(fixed_start, candidate_seconds - fine_margin_seconds)
    local_end = min(fixed_end, candidate_seconds + fine_margin_seconds)
    attempts: list[dict[str, Any]] = []
    chosen: dict[str, Any] | None = None
    for attempt_index in range(1, max_attempts + 1):
        backward_seconds = fine_margin_seconds * (2 ** (attempt_index - 1))
        attempt_start = max(fixed_start, candidate_seconds - backward_seconds)
        chosen = run_refinement_attempt(
            client, manifest, output_dir / f"attempt_{attempt_index:02d}", from_stage, to_stage, fine_fps,
            longest_edge, max_tokens, attempt_start, local_end,
        )
        attempts.append(chosen)
        if "transition_at_local_window_start" not in chosen["refinement_errors"] or attempt_start <= fixed_start + 0.01:
            break
    if (
        len(attempts) < max_attempts
        and "target_evidence_conflicts_with_measurement" in chosen["refinement_errors"]
        and local_end < fixed_end - 0.01
    ):
        expanded_end = min(fixed_end, candidate_seconds + 2 * fine_margin_seconds)
        if expanded_end > local_end + 0.01:
            chosen = run_refinement_attempt(
                client, manifest, output_dir / f"attempt_{len(attempts) + 1:02d}_forward", from_stage, to_stage,
                fine_fps, longest_edge, max_tokens, local_start, expanded_end,
            )
            attempts.append(chosen)
    if chosen is None:
        raise RuntimeError("No refinement attempt was performed")
    return {
        "from_stage": from_stage,
        "to_stage": to_stage,
        "candidate_seconds": candidate_seconds,
        "attempts": attempts,
        **chosen,
    }


def compile_actions(
    coarse: dict[str, Any],
    coarse_frames: list[dict[str, Any]],
    refinements: list[dict[str, Any]],
    fixed_start: float,
    fixed_end: float,
) -> tuple[list[dict[str, Any]], list[str], bool]:
    segments = coarse["coarse_segments"]
    starts: list[float] = [fixed_start]
    abstention_reasons: list[str] = []
    for index, segment in enumerate(segments[1:], start=1):
        coarse_frame = frame_by_id(coarse_frames, segment.get("start_frame_id"))
        coarse_seconds = float(coarse_frame["timestamp_seconds"]) if coarse_frame else fixed_end
        refinement = refinements[index - 1] if index - 1 < len(refinements) else None
        selected_seconds = refinement.get("selected_seconds") if isinstance(refinement, dict) else None
        chosen = float(selected_seconds) if isinstance(selected_seconds, (int, float)) else coarse_seconds
        if refinement and not refinement.get("refinement_valid"):
            abstention_reasons.append(f"boundary_{index}_coarse_fallback")
        if chosen <= starts[-1] + 0.001 or chosen >= fixed_end - 0.001:
            abstention_reasons.append(f"boundary_{index}_outside_valid_range")
            chosen = coarse_seconds
        if chosen <= starts[-1] + 0.001 or chosen >= fixed_end - 0.001:
            abstention_reasons.append(f"boundary_{index}_unusable")
            chosen = min(fixed_end - 0.001, max(starts[-1] + 0.001, coarse_seconds))
        starts.append(round(chosen, 3))
    actions: list[dict[str, Any]] = []
    for index, segment in enumerate(segments):
        start_seconds = starts[index]
        end_seconds = starts[index + 1] if index + 1 < len(starts) else fixed_end
        if end_seconds <= start_seconds:
            abstention_reasons.append(f"action_{index}_non_positive_duration")
            continue
        confidence = float(segment["confidence"])
        if confidence < 0.65:
            abstention_reasons.append(f"action_{index}_low_coarse_confidence")
        refinement = refinements[index - 1] if index > 0 and index - 1 < len(refinements) else None
        actions.append({
            "stage": segment["stage"],
            "stage_label": STAGE_LABELS[segment["stage"]],
            "start_seconds": round(start_seconds, 3),
            "end_seconds": round(end_seconds, 3),
            "coarse_start_frame_id": segment["start_frame_id"],
            "coarse_end_frame_id": segment["end_frame_id"],
            "coarse_evidence": segment["evidence"],
            "coarse_confidence": confidence,
            "boundary_source": "fixed_experiment_start" if index == 0 else (
                "local_refinement" if refinement and refinement.get("refinement_valid") else "coarse_fallback"
            ),
        })
    refinement_incomplete = len(segments) > 1 and len(refinements) < len(segments) - 1
    evidence_insufficient = bool(abstention_reasons or coarse.get("uncertain_intervals") or refinement_incomplete)
    return actions, sorted(set(abstention_reasons)), evidence_insufficient


def manifest_by_video_id(input_dir: Path) -> dict[str, tuple[Path, dict[str, Any]]]:
    result: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in sorted(input_dir.glob("*.marker_filter.json")):
        manifest = read_json(path)
        video_id = str(manifest.get("source_video_id", ""))
        if video_id:
            result[video_id] = (path, manifest)
    return result


def load_source_records(segment_source: Path) -> list[dict[str, Any]]:
    """Read either a normal summary JSON or a directory of per-video results."""
    if segment_source.is_dir():
        candidates = sorted(path for path in segment_source.glob("*.json") if path.name != "summary.json")
        records: list[dict[str, Any]] = []
        for path in candidates:
            value = read_json(path)
            if value.get("source_video_id"):
                records.append(value)
        return records
    return [record for record in read_json(segment_source).get("records", []) if isinstance(record, dict)]


def validate_chunk_coarse(value: dict[str, Any], frames: list[dict[str, Any]]) -> list[str]:
    """Validate a chunk without imposing whole-video stage coverage or order."""

    sequence_errors = {
        "coarse_segments_invalid",
        "coarse_stage_duplicate",
        "coarse_stage_transition_invalid",
        "measurement_2_without_measurement_1",
        "measurement_2_without_reconfiguration",
        "material_cleanup_not_final_stage",
    }
    errors = [error for error in validate_coarse(value, frames) if error not in sequence_errors]
    segments = value.get("coarse_segments")
    if segments == []:
        if set(value.get("missing_stages", [])) != set(STAGES):
            errors.append("empty_chunk_missing_stages_incomplete")
        intervals = value.get("uncertain_intervals")
        if not isinstance(intervals, list) or not intervals:
            errors.append("empty_chunk_uncertainty_required")
    elif not isinstance(segments, list) or not segments:
        errors.append("coarse_segments_invalid")
    return sorted(set(errors))


def merge_chunk_coarse_results(
    chunk_results: list[tuple[list[dict[str, Any]], dict[str, Any]]],
    all_frames: list[dict[str, Any]],
) -> dict[str, Any]:
    by_id = {str(frame["image_id"]): index for index, frame in enumerate(all_frames)}
    merged_segments: list[dict[str, Any]] = []
    uncertain_intervals: list[dict[str, Any]] = []
    evidence: list[str] = []
    confidences: list[float] = []
    uncertainty: list[str] = []
    needs_refinement = False
    for chunk_frames, result in chunk_results:
        local_ids = {str(frame["image_id"]) for frame in chunk_frames}
        for segment in result["coarse_segments"]:
            start_id, end_id = str(segment["start_frame_id"]), str(segment["end_frame_id"])
            if start_id not in local_ids or end_id not in local_ids:
                continue
            item = dict(segment)
            if merged_segments:
                previous = merged_segments[-1]
                previous_end = by_id[str(previous["end_frame_id"])]
                current_start = by_id[start_id]
                if previous["stage"] == item["stage"] and current_start <= previous_end + 2:
                    previous["end_frame_id"] = item["end_frame_id"]
                    previous["confidence"] = round(
                        (float(previous["confidence"]) + float(item["confidence"])) / 2.0,
                        6,
                    )
                    previous["evidence"] = str(previous["evidence"])
                    continue
            merged_segments.append(item)
        for interval in result.get("uncertain_intervals", []):
            if str(interval.get("start_frame_id")) in local_ids and str(interval.get("end_frame_id")) in local_ids:
                uncertain_intervals.append(dict(interval))
        if isinstance(result.get("overall_evidence"), str) and result["overall_evidence"].strip():
            evidence.append(result["overall_evidence"].strip())
        if valid_confidence(result.get("confidence")):
            confidences.append(float(result["confidence"]))
        if isinstance(result.get("uncertainty"), str) and result["uncertainty"].strip():
            uncertainty.append(result["uncertainty"].strip())
        needs_refinement = needs_refinement or bool(result.get("needs_local_refinement"))
    present = {str(segment["stage"]) for segment in merged_segments}
    return {
        "coarse_segments": merged_segments,
        "missing_stages": sorted(set(STAGES) - present, key=STAGE_ORDER.get),
        "uncertain_intervals": uncertain_intervals,
        "overall_evidence": " | ".join(evidence)[:100] or "分块阶段观察未形成可用摘要",
        "confidence": round(float(np.mean(confidences)), 6) if confidences else 0.0,
        "needs_local_refinement": needs_refinement,
        "uncertainty": " | ".join(uncertainty)[:60],
    }


def call_chunked_coarse(
    client: OpenAI,
    frames: list[dict[str, Any]],
    chunk_size: int,
    max_tokens: int,
) -> dict[str, Any]:
    chunks = [frames[index : index + chunk_size] for index in range(0, len(frames), chunk_size)]
    raw_chunks: list[dict[str, Any]] = []
    valid_results: list[tuple[list[dict[str, Any]], dict[str, Any]]] = []
    for index, chunk_frames in enumerate(chunks, start=1):
        raw = call_qwen(
            client,
            build_chunk_coarse_prompt([str(frame["image_id"]) for frame in chunk_frames], index, len(chunks)),
            chunk_frames,
            max_tokens,
        )
        parsed = raw.get("parsed_result")
        errors = validate_chunk_coarse(parsed, chunk_frames) if isinstance(parsed, dict) else ["qwen_response_not_parsed"]
        raw_chunks.append(
            {
                "chunk_index": index,
                "frame_id_range": [chunk_frames[0]["image_id"], chunk_frames[-1]["image_id"]],
                "qwen": raw,
                "valid": not errors,
                "errors": errors,
            }
        )
        if errors:
            return {
                "mode": "chunked_1fps",
                "chunk_size": chunk_size,
                "chunks": raw_chunks,
                "parsed_result": None,
                "error": "chunk_result_invalid",
            }
        valid_results.append((chunk_frames, parsed))
    merged = merge_chunk_coarse_results(valid_results, frames)
    return {
        "mode": "chunked_1fps",
        "chunk_size": chunk_size,
        "chunks": raw_chunks,
        "parsed_result": merged,
    }


def render_markdown(record: dict[str, Any]) -> str:
    lines = [
        f"# 动作分段：{record.get('source_video_id', '')}",
        "",
        f"固定实验区间：{format_clock(record['fixed_experiment_window_seconds'][0])}–{format_clock(record['fixed_experiment_window_seconds'][1])}",
        "",
        "| 阶段 | 开始 | 结束 | 边界来源 | 粗分类证据 |",
        "|---|---:|---:|---|---|",
    ]
    for action in record.get("actions", []):
        lines.append(
            f"| {action['stage_label']} | {format_clock(action['start_seconds'])} | {format_clock(action['end_seconds'])} | {action['boundary_source']} | {action['coarse_evidence']} |"
        )
    terminal_cleanup = record.get("terminal_cleanup")
    if isinstance(terminal_cleanup, dict) and terminal_cleanup.get("reached") is True:
        lines.extend([
            "",
            "实际分析截止：" + format_clock(terminal_cleanup.get("cleanup_end_seconds")) + "（整理材料后不再使用后续阶段）",
            "整理后未评估阶段：" + ("、".join(STAGE_LABELS.get(stage, stage) for stage in record.get("not_evaluated_after_cleanup", [])) or "无"),
        ])
    missing = record.get("missing_stages", [])
    lines.extend([
        "",
        "缺失阶段：" + ("、".join(STAGE_LABELS[stage] for stage in missing) if missing else "无"),
        "",
        "证据不足：" + ("是" if record.get("evidence_insufficient") else "否"),
        "",
        "说明：阶段开始/结束均在已锁定的实验窗口内；本报告不会改写原始实验开始和结束结果。",
        "",
    ])
    return "\n".join(lines)


def process_video(
    client: OpenAI,
    source_record: dict[str, Any],
    manifest_path: Path,
    manifest: dict[str, Any],
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    video_id = str(source_record["source_video_id"])
    segment = source_record["segment"]
    fixed_start = float(segment["start_seconds"])
    fixed_end = float(segment["end_seconds"])
    video_dir = output_dir / qwen_base.slug(video_id)
    coarse_frames = extract_window(
        manifest,
        video_dir / "coarse_inputs",
        fixed_start,
        fixed_end,
        args.coarse_interval_seconds,
        args.max_model_edge,
        "coarse",
    )
    record: dict[str, Any] = {
        "schema_version": "qwen_experiment_action_segment.v1",
        "generated_at": utc_now(),
        "source_video_id": video_id,
        "source_manifest": str(manifest_path.resolve()),
        "source_segment_summary": str(args.segment_summary.resolve()),
        "fixed_experiment_window_seconds": [fixed_start, fixed_end],
        "sampling": {
            "coarse_interval_seconds": args.coarse_interval_seconds,
            "fine_fps": args.fine_fps,
            "fine_margin_seconds": args.fine_margin_seconds,
            "max_model_edge": args.max_model_edge,
            "visible_banner": "bottom-left FRAME ID=<id> | VIDEO T=<video-relative-seconds>s",
        },
        "coarse_input_frames": coarse_frames,
    }
    coarse_raw = (
        call_chunked_coarse(client, coarse_frames, args.coarse_chunk_size, args.max_tokens)
        if args.coarse_chunk_size > 0
        else call_qwen(client, build_coarse_prompt([frame["image_id"] for frame in coarse_frames]), coarse_frames, args.max_tokens)
    )
    record["coarse_qwen"] = coarse_raw
    raw_coarse = coarse_raw.get("parsed_result")
    coarse, terminal_cleanup = (
        normalize_terminal_cleanup(raw_coarse, coarse_frames, not args.continue_after_cleanup)
        if isinstance(raw_coarse, dict)
        else (raw_coarse, {
            "enabled": not args.continue_after_cleanup,
            "reached": False,
            "reason": "qwen_response_not_parsed",
            "discarded_segment_count": 0,
            "not_evaluated_after_cleanup": [],
        })
    )
    record["terminal_cleanup"] = terminal_cleanup
    if terminal_cleanup.get("reached") is True:
        record["analysis_window_seconds"] = [fixed_start, terminal_cleanup["cleanup_end_seconds"]]
        record["retrieval_cutoff_seconds"] = terminal_cleanup["retrieval_cutoff_seconds"]
        record["coarse_normalized_for_terminal_cleanup"] = coarse
    else:
        record["analysis_window_seconds"] = [fixed_start, fixed_end]
    coarse_errors = (
        validate_chunk_coarse(coarse, coarse_frames)
        if isinstance(coarse, dict) and args.coarse_chunk_size > 0
        else (validate_coarse(coarse, coarse_frames) if isinstance(coarse, dict) else ["qwen_response_not_parsed"])
    )
    record["coarse_valid"] = not coarse_errors
    record["coarse_errors"] = coarse_errors
    if coarse_errors:
        record.update({
            "status": "coarse_result_invalid",
            "actions": [],
            "missing_stages": [],
            "not_evaluated_after_cleanup": sorted(
                set(terminal_cleanup.get("not_evaluated_after_cleanup", [])),
                key=STAGE_ORDER.get,
            ),
            "evidence_insufficient": True,
            "evidence_status": "insufficient",
        })
        record["analysis_termination"] = {
            "reached": terminal_cleanup.get("reached") is True,
            "reason": terminal_cleanup.get("reason"),
            "terminal_stage": "material_cleanup" if terminal_cleanup.get("reached") is True else None,
            "cutoff_seconds": terminal_cleanup.get("retrieval_cutoff_seconds"),
            "terminal_stage_end_seconds": terminal_cleanup.get("cleanup_end_seconds"),
            "discarded_post_terminal_count": terminal_cleanup.get("discarded_segment_count", 0),
            "post_terminal_not_evaluated": terminal_cleanup.get("reached") is True,
        }
        return record

    refinements: list[dict[str, Any]] = []
    coarse_segments = coarse["coarse_segments"]
    analysis_end = float(record["analysis_window_seconds"][1])
    if not args.skip_refinement:
        for index, next_segment in enumerate(coarse_segments[1:], start=1):
            candidate_frame = frame_by_id(coarse_frames, next_segment["start_frame_id"])
            if candidate_frame is None:
                continue
            refinement = refine_transition(
                client,
                manifest,
                video_dir / "fine_inputs" / f"{index:02d}_{coarse_segments[index - 1]['stage']}_to_{next_segment['stage']}",
                fixed_start,
                analysis_end,
                float(candidate_frame["timestamp_seconds"]),
                coarse_segments[index - 1]["stage"],
                next_segment["stage"],
                args.fine_fps,
                args.fine_margin_seconds,
                args.max_model_edge,
                args.max_tokens,
                args.max_refinement_attempts,
            )
            refinements.append(refinement)
    record["boundary_refinements"] = refinements
    actions, abstention_reasons, evidence_insufficient = compile_actions(coarse, coarse_frames, refinements, fixed_start, analysis_end)
    not_evaluated_after_cleanup = set(terminal_cleanup.get("not_evaluated_after_cleanup", []))
    missing_stages = sorted(
        (set(STAGES) - {segment["stage"] for segment in coarse_segments}) - not_evaluated_after_cleanup,
        key=STAGE_ORDER.get,
    )
    record.update({
        "status": "completed_evidence_insufficient" if evidence_insufficient else "completed",
        "actions": actions,
        "missing_stages": missing_stages,
        "not_evaluated_after_cleanup": sorted(not_evaluated_after_cleanup, key=STAGE_ORDER.get),
        "uncertain_intervals": coarse.get("uncertain_intervals", []),
        "overall_evidence": coarse.get("overall_evidence"),
        "coarse_confidence": coarse.get("confidence"),
        "evidence_insufficient": evidence_insufficient,
        "evidence_status": "insufficient" if evidence_insufficient else "usable",
        "abstention_reasons": abstention_reasons,
    })
    record["analysis_termination"] = {
        "reached": terminal_cleanup.get("reached") is True,
        "reason": terminal_cleanup.get("reason"),
        "terminal_stage": "material_cleanup" if terminal_cleanup.get("reached") is True else None,
        "cutoff_seconds": terminal_cleanup.get("retrieval_cutoff_seconds"),
        "terminal_stage_end_seconds": terminal_cleanup.get("cleanup_end_seconds"),
        "discarded_post_terminal_count": terminal_cleanup.get("discarded_segment_count", 0),
        "post_terminal_not_evaluated": terminal_cleanup.get("reached") is True,
    }
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--segment-summary", type=Path, default=DEFAULT_SEGMENT_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--video-id", action="append", default=[])
    parser.add_argument("--coarse-interval-seconds", type=float, default=5.0)
    parser.add_argument("--coarse-chunk-size", type=int, default=0)
    parser.add_argument("--fine-fps", type=float, default=1.0)
    parser.add_argument("--fine-margin-seconds", type=float, default=10.0)
    parser.add_argument("--max-model-edge", type=int, default=640)
    parser.add_argument("--max-tokens", type=int, default=1800)
    parser.add_argument("--max-refinement-attempts", type=int, default=3)
    parser.add_argument("--skip-refinement", action="store_true")
    parser.add_argument("--continue-after-cleanup", action="store_true")
    args = parser.parse_args(argv)
    if min(
        args.coarse_interval_seconds, args.fine_fps, args.fine_margin_seconds,
        args.max_model_edge, args.max_tokens, args.max_refinement_attempts,
    ) <= 0:
        parser.error("sampling, resolution, and token parameters must be positive")
    if args.coarse_chunk_size < 0:
        parser.error("coarse chunk size must be zero or positive")
    source_records = [
        record for record in load_source_records(args.segment_summary)
        if isinstance(record, dict)
        and isinstance(record.get("segment"), dict)
        and record["segment"].get("segment_valid") is True
        and isinstance(record["segment"].get("start_seconds"), (int, float))
        and isinstance(record["segment"].get("end_seconds"), (int, float))
    ]
    if args.video_id:
        requested = set(args.video_id)
        source_records = [record for record in source_records if record.get("source_video_id") in requested]
    if not source_records:
        parser.error("No valid fixed experiment segments selected")
    manifests = manifest_by_video_id(args.input_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    qwen_base.require_qwen_configuration()
    client = OpenAI(
        base_url=qwen_base.API_BASE_URL,
        api_key=qwen_base.API_TOKEN,
        timeout=180,
        max_retries=0,
    )
    records: list[dict[str, Any]] = []
    for source_record in source_records:
        video_id = str(source_record["source_video_id"])
        if video_id not in manifests:
            record = {
                "source_video_id": video_id,
                "status": "source_manifest_not_found",
                "actions": [],
                "evidence_insufficient": True,
                "evidence_status": "insufficient",
            }
        else:
            manifest_path, manifest = manifests[video_id]
            try:
                record = process_video(client, source_record, manifest_path, manifest, args.output_dir, args)
            except Exception as exc:
                record = {
                    "source_video_id": video_id,
                    "status": "processing_failed",
                    "error_type": type(exc).__name__,
                    "error": qwen_base.safe_error_message(exc),
                    "actions": [],
                    "evidence_insufficient": True,
                    "evidence_status": "insufficient",
                }
        video_dir = args.output_dir / qwen_base.slug(video_id)
        write_json(video_dir / "result.json", record)
        if "fixed_experiment_window_seconds" in record:
            (video_dir / "action_report.md").write_text(render_markdown(record), encoding="utf-8")
        records.append(record)
        print(json.dumps({
            "video": video_id,
            "status": record["status"],
            "action_count": len(record.get("actions", [])),
            "evidence_insufficient": record.get("evidence_insufficient"),
        }, ensure_ascii=False), flush=True)
    final_summary = {
        "schema_version": "qwen_experiment_action_segment.v1",
        "generated_at": utc_now(),
        "source_segment_summary": str(args.segment_summary.resolve()),
        "config": {
            "coarse_interval_seconds": args.coarse_interval_seconds,
            "coarse_chunk_size": args.coarse_chunk_size,
            "fine_fps": args.fine_fps,
            "fine_margin_seconds": args.fine_margin_seconds,
            "max_model_edge": args.max_model_edge,
            "max_tokens": args.max_tokens,
            "max_refinement_attempts": args.max_refinement_attempts,
            "skip_refinement": args.skip_refinement,
            "continue_after_cleanup": args.continue_after_cleanup,
        },
        "records": records,
    }
    write_json(args.output_dir / "summary.json", final_summary)
    print(f"summary={(args.output_dir / 'summary.json').resolve()}")
    fatal_statuses = {"source_manifest_not_found", "processing_failed", "coarse_result_invalid"}
    return 1 if any(record.get("status") in fatal_statuses for record in records) else 0


if __name__ == "__main__":
    raise SystemExit(main())
