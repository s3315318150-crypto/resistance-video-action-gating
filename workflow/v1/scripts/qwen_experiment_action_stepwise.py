#!/usr/bin/env python3
"""Find experiment actions sequentially in two-second frames.

Each pass handles one current action only.  It first scans a 60-second window
for a possible exit, then separately verifies the local transition.  Keeping
these two visual questions separate prevents an approximate scan from being
mistaken for a precise stage boundary.  Locked experiment start/end times are
read-only.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI

import qwen_experiment_action_segment as action_base
import qwen_experiment_segment_judge as qwen_base


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = ROOT / "outputs" / "marker_filter"
DEFAULT_SEGMENT_SOURCE = ROOT / "outputs" / "experiment_boundary"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "action_segments_stepwise"

STAGES = (
    "circuit_wiring",
    "measurement_1",
    "recording_1",
    "circuit_rewiring",
    "measurement_2",
    "recording_2",
    "material_cleanup",
)
STAGE_LABELS = {
    "circuit_wiring": "连线",
    "measurement_1": "第一次测量",
    "recording_1": "第一次记录",
    "circuit_rewiring": "重新连线",
    "measurement_2": "第二次测量",
    "recording_2": "第二次记录",
    "material_cleanup": "整理材料",
}
SCAN_DECISIONS = {"current_stage_continues", "transition_possible", "uncertain", "current_stage_not_observed"}
REFINEMENT_DECISIONS = {"transition_observed", "uncertain"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def allowed_next_stages(current_stage: str) -> tuple[str, ...]:
    index = STAGES.index(current_stage)
    return STAGES[index + 1:]


def stage_definition(stage: str) -> str:
    definitions = {
        "circuit_wiring": "摆放器材、插接导线、连接电池盒、开关、电表或待测电阻。",
        "measurement_1": "电路准备后观察仪表、闭合电路后读数或测量；不是书写。",
        "recording_1": "在纸上持续书写、填写表格或计算第一轮数据。",
        "circuit_rewiring": "明确拆开、修改或重新插接实验线路以继续实验；不是最终整理。",
        "measurement_2": "重新配置之后，再次观察仪表并测量；不是书写或收纳。",
        "recording_2": "第二轮测量后的持续书写、填写或计算。",
        "material_cleanup": "开始拆下导线、断开线路、收拢并归位器材，且之后不再继续实验。",
    }
    return definitions[stage]


def terminal_cleanup_reached(
    next_stage: str,
    refinement: dict[str, Any],
    continue_after_cleanup: bool,
    minimum_confidence: float,
) -> bool:
    """Return whether a locally confirmed cleanup should end stage search."""
    return (
        next_stage == "material_cleanup"
        and not continue_after_cleanup
        and valid_confidence(refinement.get("confidence"))
        and float(refinement["confidence"]) >= minimum_confidence
    )


def build_scan_prompt(current_stage: str, image_ids: list[str], is_final_window: bool) -> str:
    next_options = allowed_next_stages(current_stage)
    next_values = " | ".join(f'"{stage}"' for stage in next_options) if next_options else "null"
    next_display = "、".join(f"{STAGE_LABELS[stage]}（{stage}）" for stage in next_options) if next_options else "无；整理材料之后不应继续实验"
    return f"""你正在逐步分析一段“伏安法测电阻”实验视频。现在是第一轮“窗口扫描”：只判断当前动作“{STAGE_LABELS[current_stage]}（{current_stage}）”在这个一分钟窗口内有没有可能结束。不要给出精确的最后一帧或下一动作第一帧；那会由下一轮独立复核。

本窗口图片按时间顺序、每 2 秒一张。唯一可用图片 ID 为：{", ".join(image_ids)}。
当前动作定义：{stage_definition(current_stage)}
若当前动作结束，后续只可能进入：{next_display}。

图片底部左侧有 `FRAME ID=step_001 | VIDEO T=75.0s`：
* 只能把 `FRAME ID=` 后完整的 ID 原样填入 JSON；`VIDEO T` 是相对时间，绝不是 ID。
* 不能根据时间数字编造 `step_075`，也不能把摄像机右下角日期时间当视频时间。

扫描规则：
1. 若没有任何足以说明离开当前动作的直接可见迹象，decision 必须是 current_stage_continues。不要因窗口结束、短暂停顿、手遮挡或坐姿变化猜测切换。
2. 若窗口中出现可能从当前动作转为其他动作的迹象，decision 为 transition_possible。transition_range_start_frame_id 和 transition_range_end_frame_id 仅圈出“可能发生转折”的宽松范围，前者不得晚于后者；它们不是精确边界。
3. possible_next_stage 只是复核时优先检查的候选。只有直接可见证据才可填写，不能把“手在桌面附近”猜成测量或记录。
4. 重新连线、第二次测量、第二次记录可能根本没有发生。整理材料只能是最后动作；若之后仍会连接、测量或记录，不得把它当整理。一旦后续复核确认进入整理材料，系统会停止查看该视频后续帧。
5. 画面模糊、遮挡、学生离座，且不能判断是否离开当前动作时，decision 必须为 uncertain。{ '本窗口已到固定实验结束时间。' if is_final_window else '窗口结束不等于实验或当前动作结束，后面仍有视频。'}

只输出一个合法 JSON 对象：
{{
  "current_stage": "{current_stage}",
  "decision": "current_stage_continues" | "transition_possible" | "uncertain" | "current_stage_not_observed",
  "transition_range_start_frame_id": "step_001" | null,
  "transition_range_end_frame_id": "step_001" | null,
  "possible_next_stage": {next_values} | null,
  "scan_evidence": "不超过80字，只描述扫描中直接可见的变化或持续证据",
  "confidence": 0.0,
  "uncertainty": "不超过80字；无则为空字符串"
}}"""


def build_refinement_prompt(current_stage: str, image_ids: list[str]) -> str:
    next_options = allowed_next_stages(current_stage)
    next_values = " | ".join(f'"{stage}"' for stage in next_options) if next_options else "null"
    next_display = "、".join(f"{STAGE_LABELS[stage]}（{stage}）" for stage in next_options) if next_options else "无"
    return f"""你正在逐步分析一段“伏安法测电阻”实验视频。现在是第二轮“转折复核”：已知前一轮在这组相邻抽帧附近发现当前动作“{STAGE_LABELS[current_stage]}（{current_stage}）”可能结束。请只确认精确边界，不要重新给整段视频分阶段。

图片按时间顺序、每 2 秒一张。唯一可用图片 ID 为：{", ".join(image_ids)}。
当前动作定义：{stage_definition(current_stage)}
后续候选动作：{next_display}。

图片底部左侧有 `FRAME ID=step_001 | VIDEO T=75.0s`：只能把 `FRAME ID=` 后完整 ID 原样填入 JSON；不得用 VIDEO T 或摄像机日期时间编造 ID。

复核规则：
1. 只有同时找到“最后一张明确仍属于当前动作的图片”和“其后第一张明确属于下一动作的图片”时，才使用 transition_observed。前者必须严格早于后者。
2. last_current_evidence 只能描述当前动作的直接可见证据；first_next_evidence 只能描述 next_stage 的直接可见证据。描述与阶段名称不一致，必须选择 uncertain。
3. next_stage 必须是当前动作之后最早直接可见的动作。若后面既出现测量又出现记录，先可见测量时必须选测量，不能跳到记录。只有中间动作完全没有直接可见证据时才可跳过，并写入 skipped_stages。
4. 测量需要看见观察电表、读数、闭合电路后的实际测量；记录需要看见持续书写、填写或计算；重新连线需要明确拆接、修改或插接导线；整理材料需要拆线、收拢或归位且之后不再实验。若选择整理材料，本地程序会在这张首次确认帧停止后续阶段搜索，因此证据必须直接、清楚。
5. 模糊、遮挡、停顿、离座或证据不足时，输出 uncertain，不要猜测边界。

只输出一个合法 JSON 对象：
{{
  "current_stage": "{current_stage}",
  "decision": "transition_observed" | "uncertain",
  "last_current_frame_id": "step_001" | null,
  "last_current_evidence": "不超过60字，只描述该图中当前动作的可见证据",
  "next_stage": {next_values} | null,
  "first_next_frame_id": "step_001" | null,
  "first_next_evidence": "不超过60字，只描述该图中下一动作的可见证据",
  "skipped_stages": [],
  "confidence": 0.0,
  "uncertainty": "不超过80字；无则为空字符串"
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
    result: dict[str, Any] = {"finish_reason": choice.finish_reason or "unknown", "raw_model_content": raw, "parsed": False}
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


def evidence_supports_stage(stage: str, evidence: str) -> bool:
    if not evidence.strip():
        return False
    if stage == "circuit_wiring":
        return "未进行连线" not in evidence and any(word in evidence for word in ("连线", "接线", "导线", "插接", "连接"))
    if stage in {"measurement_1", "measurement_2"}:
        return (
            any(word in evidence for word in ("测量", "观察", "电表", "仪表", "读数", "闭合开关"))
            and not any(word in evidence for word in ("书写", "记录", "填写"))
            and not any(word in evidence for word in ("准备", "即将", "将要", "待测量", "拆卸", "断开电路", "重新连线", "插拔导线"))
        )
    if stage in {"recording_1", "recording_2"}:
        return any(word in evidence for word in ("书写", "记录", "填写", "写字", "纸上"))
    if stage == "circuit_rewiring":
        return any(word in evidence for word in ("重新", "拆", "导线", "接线", "插接"))
    if stage == "material_cleanup":
        return any(word in evidence for word in ("整理", "收拢", "归位", "拆", "断开"))
    return False


def validate_scan(value: dict[str, Any], current_stage: str, frames: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    by_id = {str(frame["image_id"]): index for index, frame in enumerate(frames)}
    ids = set(by_id)
    decision = value.get("decision")
    range_start = value.get("transition_range_start_frame_id")
    range_end = value.get("transition_range_end_frame_id")
    possible_next = value.get("possible_next_stage")
    if value.get("current_stage") != current_stage:
        errors.append("current_stage_mismatch")
    if decision not in SCAN_DECISIONS:
        errors.append("scan_decision_invalid")
    if range_start is not None and range_start not in ids:
        errors.append("scan_range_start_frame_id_invalid")
    if range_end is not None and range_end not in ids:
        errors.append("scan_range_end_frame_id_invalid")
    if range_start in by_id and range_end in by_id and by_id[range_start] > by_id[range_end]:
        errors.append("scan_range_not_forward")
    if not isinstance(value.get("scan_evidence"), str) or not value["scan_evidence"].strip():
        errors.append("scan_evidence_invalid")
    if not valid_confidence(value.get("confidence")):
        errors.append("confidence_invalid")
    if not isinstance(value.get("uncertainty"), str):
        errors.append("uncertainty_invalid")
    if decision == "transition_possible":
        if range_start is None or range_end is None:
            errors.append("transition_possible_without_range")
        if possible_next not in allowed_next_stages(current_stage):
            errors.append("possible_next_stage_invalid")
    elif range_start is not None or range_end is not None or possible_next is not None:
        errors.append("non_transition_scan_with_transition_fields")
    return sorted(set(errors))


def validate_refinement(value: dict[str, Any], current_stage: str, frames: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    by_id = {str(frame["image_id"]): index for index, frame in enumerate(frames)}
    ids = set(by_id)
    decision = value.get("decision")
    last_id = value.get("last_current_frame_id")
    next_stage = value.get("next_stage")
    first_next_id = value.get("first_next_frame_id")
    skipped = value.get("skipped_stages")
    if value.get("current_stage") != current_stage:
        errors.append("current_stage_mismatch")
    if decision not in REFINEMENT_DECISIONS:
        errors.append("decision_invalid")
    if last_id is not None and last_id not in ids:
        errors.append("last_current_frame_id_invalid")
    if first_next_id is not None and first_next_id not in ids:
        errors.append("first_next_frame_id_invalid")
    if not isinstance(value.get("last_current_evidence"), str) or not value["last_current_evidence"].strip():
        errors.append("last_current_evidence_invalid")
    if not isinstance(value.get("first_next_evidence"), str):
        errors.append("first_next_evidence_invalid")
    if not isinstance(skipped, list) or any(item not in STAGES for item in skipped):
        errors.append("skipped_stages_invalid")
    if not valid_confidence(value.get("confidence")):
        errors.append("confidence_invalid")
    if not isinstance(value.get("uncertainty"), str):
        errors.append("uncertainty_invalid")
    if decision == "transition_observed":
        if last_id is None or first_next_id is None:
            errors.append("transition_missing_boundary_frame")
        if next_stage not in allowed_next_stages(current_stage):
            errors.append("next_stage_invalid")
        if last_id in by_id and first_next_id in by_id and by_id[last_id] >= by_id[first_next_id]:
            errors.append("transition_not_forward")
        if not value.get("first_next_evidence", "").strip():
            errors.append("transition_missing_next_evidence")
        if isinstance(value.get("last_current_evidence"), str) and not evidence_supports_stage(current_stage, value["last_current_evidence"]):
            errors.append("last_current_evidence_does_not_support_stage")
        if next_stage in STAGES and isinstance(value.get("first_next_evidence"), str) and not evidence_supports_stage(next_stage, value["first_next_evidence"]):
            errors.append("next_evidence_does_not_support_stage")
    else:
        if next_stage is not None or first_next_id is not None:
            errors.append("non_transition_with_next_stage")
    return sorted(set(errors))


def frame_by_id(frames: list[dict[str, Any]], image_id: Any) -> dict[str, Any] | None:
    return next((frame for frame in frames if frame["image_id"] == image_id), None) if isinstance(image_id, str) else None


def locked_segment(record: dict[str, Any]) -> dict[str, Any] | None:
    """Accept both the segment-judge schema and the prior action-segment schema."""
    segment = record.get("segment")
    if isinstance(segment, dict):
        return segment
    window = record.get("fixed_experiment_window_seconds")
    if isinstance(window, list) and len(window) == 2 and all(isinstance(item, (int, float)) for item in window):
        return {"start_seconds": float(window[0]), "end_seconds": float(window[1]), "segment_valid": True}
    return None


def render_markdown(record: dict[str, Any]) -> str:
    lines = [
        f"# 逐阶段动作搜索：{record.get('source_video_id', '')}",
        "",
        f"固定实验区间：{action_base.format_clock(record['fixed_experiment_window_seconds'][0])}–{action_base.format_clock(record['fixed_experiment_window_seconds'][1])}",
        "",
        "| 阶段 | 开始 | 最后确认帧 | 下一阶段首次确认帧 | 状态 |",
        "|---|---:|---:|---:|---|",
    ]
    terminal_cleanup = record.get("terminal_cleanup")
    if isinstance(terminal_cleanup, dict) and terminal_cleanup.get("reached") is True:
        lines.extend([
            "",
            "实际分析截止：" + action_base.format_clock(terminal_cleanup.get("start_seconds")) + "（首次确认进入整理材料；后续帧未再读取）",
        ])
    for stage in record.get("stage_results", []):
        lines.append(
            f"| {STAGE_LABELS[stage['stage']]} | {action_base.format_clock(stage['start_seconds'])} | {action_base.format_clock(stage.get('last_current_seconds'))} | {action_base.format_clock(stage.get('next_start_seconds'))} | {stage['status']} |"
        )
    lines.extend([
        "",
        "未观察到阶段：" + ("、".join(STAGE_LABELS[item] for item in record.get("missing_stages", [])) or "无"),
        "",
        "因整理材料终止而未评估：" + (
            "、".join(STAGE_LABELS[item] for item in record.get("not_evaluated_after_cleanup", []))
            or "无"
        ),
        "",
        "证据不足：" + ("是" if record.get("evidence_insufficient") else "否"),
        "",
        "说明：每一轮只检查一个当前动作，在 60 秒窗口内每 2 秒抽帧。阶段的结束是最后确认属于该动作的图片时间；下一阶段开始是下一动作首次确认图片时间。两者相差最多一个抽帧间隔时，之间属于转折间隔。",
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
    segment = locked_segment(source_record)
    if segment is None:
        raise ValueError("source record has no locked experiment segment")
    fixed_start = float(segment["start_seconds"])
    fixed_end = float(segment["end_seconds"])
    video_dir = output_dir / qwen_base.slug(video_id)
    current_stage = STAGES[0]
    stage_start = fixed_start
    window_start = fixed_start
    stage_results: list[dict[str, Any]] = []
    all_steps: list[dict[str, Any]] = []
    missing: list[str] = []
    not_evaluated_after_cleanup: list[str] = []
    status = "completed"
    evidence_insufficient = False
    sequence_index = 0
    terminal_cleanup: dict[str, Any] | None = None
    while window_start < fixed_end - 0.001:
        window_end = min(fixed_end, window_start + args.window_seconds)
        frames = action_base.extract_window(
            manifest,
            video_dir / "windows" / f"{sequence_index:02d}_{current_stage}_{window_start:010.3f}",
            window_start,
            window_end,
            args.sample_interval_seconds,
            args.max_model_edge,
            "step",
        )
        scan_prompt = build_scan_prompt(
            current_stage,
            [frame["image_id"] for frame in frames],
            window_end >= fixed_end - 0.01,
        )
        scan_attempts: list[dict[str, Any]] = []
        scan_raw: dict[str, Any] = {}
        scan_errors: list[str] = []
        scan: dict[str, Any] | None = None
        for attempt_index in range(args.max_step_attempts):
            scan_raw = call_qwen(client, scan_prompt, frames, args.max_tokens)
            parsed_value = scan_raw.get("parsed_result")
            scan = parsed_value if isinstance(parsed_value, dict) else None
            scan_errors = validate_scan(scan, current_stage, frames) if scan is not None else ["qwen_response_not_parsed"]
            scan_attempts.append({"attempt_index": attempt_index + 1, "qwen": scan_raw, "validation_errors": scan_errors})
            if not scan_errors:
                break
            scan_prompt += "\n\n上一版扫描回答未通过本地校验，错误为：" + "、".join(scan_errors) + "。请不要输出精确边界，只重新输出完整扫描 JSON。"
        step = {
            "step_index": sequence_index,
            "current_stage": current_stage,
            "window_seconds": [window_start, window_end],
            "input_frames": frames,
            "scan": scan_raw,
            "scan_attempts": scan_attempts,
            "scan_valid": not scan_errors,
            "scan_errors": scan_errors,
        }
        all_steps.append(step)
        if scan_errors:
            status = "stopped_invalid_step"
            evidence_insufficient = True
            break
        scan_decision = scan["decision"]
        if scan_decision == "current_stage_continues":
            if window_end >= fixed_end - 0.01:
                stage_results.append({
                    "stage": current_stage,
                    "start_seconds": stage_start,
                    "last_current_seconds": frames[-1].get("timestamp_seconds") if frames else fixed_end,
                    "next_start_seconds": None,
                    "status": "continued_to_fixed_experiment_end",
                    "last_current_evidence": scan.get("scan_evidence"),
                })
                break
            window_start = window_end
            sequence_index += 1
            continue

        if scan_decision == "transition_possible":
            range_start = frame_by_id(frames, scan.get("transition_range_start_frame_id"))
            range_end = frame_by_id(frames, scan.get("transition_range_end_frame_id"))
            if range_start is None or range_end is None:
                status = "stopped_invalid_step"
                evidence_insufficient = True
                step["refinement_errors"] = ["scan_range_frame_not_found"]
                break
            start_index = frames.index(range_start)
            end_index = frames.index(range_end)
            refinement_frames = frames[max(0, start_index - 4):min(len(frames), end_index + 5)]
            refinement_prompt = build_refinement_prompt(current_stage, [frame["image_id"] for frame in refinement_frames])
            refinement_attempts: list[dict[str, Any]] = []
            refinement_raw: dict[str, Any] = {}
            refinement_errors: list[str] = []
            refinement: dict[str, Any] | None = None
            for attempt_index in range(args.max_step_attempts):
                refinement_raw = call_qwen(client, refinement_prompt, refinement_frames, args.max_tokens)
                parsed_value = refinement_raw.get("parsed_result")
                refinement = parsed_value if isinstance(parsed_value, dict) else None
                refinement_errors = validate_refinement(refinement, current_stage, refinement_frames) if refinement is not None else ["qwen_response_not_parsed"]
                refinement_attempts.append({"attempt_index": attempt_index + 1, "qwen": refinement_raw, "validation_errors": refinement_errors})
                if not refinement_errors:
                    break
                refinement_prompt += "\n\n上一版转折复核未通过本地校验，错误为：" + "、".join(refinement_errors) + "。请严格按图片时间顺序重新输出完整 JSON，不要解释。"
            step.update({
                "refinement_frames": refinement_frames,
                "refinement": refinement_raw,
                "refinement_attempts": refinement_attempts,
                "refinement_valid": not refinement_errors,
                "refinement_errors": refinement_errors,
            })
            if refinement_errors:
                status = "stopped_invalid_step"
                evidence_insufficient = True
                break
            if refinement["decision"] != "transition_observed":
                status = "stopped_uncertain"
                evidence_insufficient = True
                stage_results.append({
                    "stage": current_stage,
                    "start_seconds": stage_start,
                    "last_current_seconds": None,
                    "next_start_seconds": None,
                    "status": "uncertain",
                    "last_current_evidence": scan.get("scan_evidence"),
                    "uncertainty": refinement.get("uncertainty") or "转折范围内没有同时确认前后两个动作。",
                })
                break
            last_frame = frame_by_id(refinement_frames, refinement.get("last_current_frame_id"))
            next_frame = frame_by_id(refinement_frames, refinement.get("first_next_frame_id"))
            next_stage = str(refinement["next_stage"])
            stage_results.append({
                "stage": current_stage,
                "start_seconds": stage_start,
                "last_current_seconds": last_frame.get("timestamp_seconds") if last_frame else None,
                "next_start_seconds": next_frame.get("timestamp_seconds") if next_frame else None,
                "status": "transition_observed",
                "last_current_evidence": refinement.get("last_current_evidence"),
                "next_stage": next_stage,
                "first_next_evidence": refinement.get("first_next_evidence"),
            })
            current_index = STAGES.index(current_stage)
            next_index = STAGES.index(next_stage)
            canonical_skipped = list(STAGES[current_index + 1:next_index])
            step["model_skipped_stages"] = refinement.get("skipped_stages", [])
            step["canonical_skipped_stages"] = canonical_skipped
            for skipped in canonical_skipped:
                if skipped not in missing:
                    missing.append(skipped)
            if terminal_cleanup_reached(
                next_stage,
                refinement,
                bool(getattr(args, "continue_after_cleanup", False)),
                float(getattr(args, "terminal_cleanup_min_confidence", 0.8)),
            ):
                cleanup_start = float(next_frame["timestamp_seconds"])
                for skipped in canonical_skipped:
                    if skipped == "material_cleanup":
                        continue
                    if skipped in missing:
                        missing.remove(skipped)
                    if skipped not in not_evaluated_after_cleanup:
                        not_evaluated_after_cleanup.append(skipped)
                stage_results.append({
                    "stage": "material_cleanup",
                    "start_seconds": cleanup_start,
                    "last_current_seconds": cleanup_start,
                    "next_start_seconds": None,
                    "status": "terminal_cleanup_observed",
                    "last_current_evidence": refinement.get("first_next_evidence"),
                })
                terminal_cleanup = {
                    "reached": True,
                    "start_seconds": cleanup_start,
                    "first_frame_id": next_frame.get("image_id"),
                    "evidence": refinement.get("first_next_evidence"),
                    "minimum_confidence": float(getattr(args, "terminal_cleanup_min_confidence", 0.8)),
                    "reason": "confirmed_material_cleanup",
                    "post_cleanup_frames_not_sampled": True,
                }
                status = "completed_at_terminal_cleanup"
                break
            current_stage = next_stage
            stage_start = float(next_frame["timestamp_seconds"])
            window_start = stage_start
            sequence_index += 1
            continue
        status = "stopped_uncertain" if scan_decision == "uncertain" else "stopped_current_stage_not_observed"
        evidence_insufficient = True
        stage_results.append({
            "stage": current_stage,
            "start_seconds": stage_start,
            "last_current_seconds": None,
            "next_start_seconds": None,
            "status": scan_decision,
            "last_current_evidence": scan.get("scan_evidence"),
            "uncertainty": scan.get("uncertainty"),
        })
        break
    observed = {item["stage"] for item in stage_results}
    for stage in STAGES:
        if stage not in observed and stage not in missing and stage not in not_evaluated_after_cleanup:
            missing.append(stage)
    if terminal_cleanup:
        not_evaluated_after_cleanup = [
            stage for stage in STAGES
            if stage in not_evaluated_after_cleanup and stage != "material_cleanup"
        ]
    record = {
        "schema_version": "qwen_experiment_action_stepwise.v2",
        "generated_at": utc_now(),
        "source_video_id": video_id,
        "source_manifest": str(manifest_path.resolve()),
        "source_segment_source": str(args.segment_source.resolve()),
        "fixed_experiment_window_seconds": [fixed_start, fixed_end],
        "analysis_window_seconds": [fixed_start, terminal_cleanup["start_seconds"] if terminal_cleanup else fixed_end],
        "sampling": {
            "window_seconds": args.window_seconds,
            "sample_interval_seconds": args.sample_interval_seconds,
            "max_model_edge": args.max_model_edge,
            "visible_banner": "bottom-left FRAME ID=<id> | VIDEO T=<video-relative-seconds>s",
        },
        "status": status,
        "stage_results": stage_results,
        "missing_stages": missing,
        "not_evaluated_after_cleanup": not_evaluated_after_cleanup,
        "evidence_insufficient": evidence_insufficient,
        "evidence_status": "insufficient" if evidence_insufficient else "usable",
        "step_runs": all_steps,
        "terminal_cleanup": terminal_cleanup or {
            "reached": False,
            "enabled": not bool(getattr(args, "continue_after_cleanup", False)),
        },
        "analysis_termination": {
            "reached": bool(terminal_cleanup),
            "reason": terminal_cleanup.get("reason") if terminal_cleanup else "terminal_cleanup_not_observed",
            "terminal_stage": "material_cleanup" if terminal_cleanup else None,
            "cutoff_seconds": terminal_cleanup.get("start_seconds") if terminal_cleanup else None,
            "post_cleanup_frames_not_sampled": bool(terminal_cleanup and terminal_cleanup.get("post_cleanup_frames_not_sampled")),
        },
    }
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--segment-source", type=Path, default=DEFAULT_SEGMENT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--video-id", action="append", default=[])
    parser.add_argument("--window-seconds", type=float, default=60.0)
    parser.add_argument("--sample-interval-seconds", type=float, default=2.0)
    parser.add_argument("--max-model-edge", type=int, default=640)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--max-step-attempts", type=int, default=2)
    parser.add_argument("--terminal-cleanup-min-confidence", type=float, default=0.8)
    parser.add_argument("--continue-after-cleanup", action="store_true")
    args = parser.parse_args(argv)
    if min(
        args.window_seconds,
        args.sample_interval_seconds,
        args.max_model_edge,
        args.max_tokens,
        args.max_step_attempts,
        args.terminal_cleanup_min_confidence,
    ) <= 0 or args.terminal_cleanup_min_confidence > 1:
        parser.error("window, sample interval, resolution, and token parameters must be positive")
    source_records = []
    for record in action_base.load_source_records(args.segment_source):
        segment = locked_segment(record)
        if (
            segment is not None
            and segment.get("segment_valid") is True
            and isinstance(segment.get("start_seconds"), (int, float))
            and isinstance(segment.get("end_seconds"), (int, float))
        ):
            source_records.append(record)
    if args.video_id:
        source_records = [record for record in source_records if record.get("source_video_id") in set(args.video_id)]
    if not source_records:
        parser.error("No valid locked experiment segments selected")
    manifests = action_base.manifest_by_video_id(args.input_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    qwen_base.require_qwen_configuration()
    client = OpenAI(base_url=qwen_base.API_BASE_URL, api_key=qwen_base.API_TOKEN, timeout=180, max_retries=0)
    records: list[dict[str, Any]] = []
    for source_record in source_records:
        video_id = str(source_record["source_video_id"])
        if video_id not in manifests:
            record = {"source_video_id": video_id, "status": "source_manifest_not_found", "evidence_insufficient": True, "evidence_status": "insufficient", "stage_results": []}
        else:
            path, manifest = manifests[video_id]
            try:
                record = process_video(client, source_record, path, manifest, args.output_dir, args)
            except Exception as exc:
                record = {
                    "source_video_id": video_id,
                    "status": "processing_failed",
                    "error_type": type(exc).__name__,
                    "error": qwen_base.safe_error_message(exc),
                    "evidence_insufficient": True,
                    "evidence_status": "insufficient",
                    "stage_results": [],
                }
        video_dir = args.output_dir / qwen_base.slug(video_id)
        write_json(video_dir / "result.json", record)
        if "fixed_experiment_window_seconds" in record:
            (video_dir / "stepwise_report.md").write_text(render_markdown(record), encoding="utf-8")
        records.append(record)
        print(json.dumps({
            "video": video_id,
            "status": record["status"],
            "stage_count": len(record.get("stage_results", [])),
            "evidence_insufficient": record.get("evidence_insufficient"),
        }, ensure_ascii=False), flush=True)
    summary = {
        "schema_version": "qwen_experiment_action_stepwise.v2",
        "generated_at": utc_now(),
        "config": {
            "window_seconds": args.window_seconds,
            "sample_interval_seconds": args.sample_interval_seconds,
            "max_model_edge": args.max_model_edge,
            "max_tokens": args.max_tokens,
            "max_step_attempts": args.max_step_attempts,
            "terminal_cleanup_min_confidence": args.terminal_cleanup_min_confidence,
            "continue_after_cleanup": args.continue_after_cleanup,
        },
        "source_segment_source": str(args.segment_source.resolve()),
        "records": records,
    }
    write_json(args.output_dir / "summary.json", summary)
    print(f"summary={(args.output_dir / 'summary.json').resolve()}")
    fatal_statuses = {"source_manifest_not_found", "processing_failed"}
    return 1 if any(record.get("status") in fatal_statuses for record in records) else 0


if __name__ == "__main__":
    raise SystemExit(main())
