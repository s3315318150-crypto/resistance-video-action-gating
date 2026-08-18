#!/usr/bin/env python3
"""Classify locked experiment intervals one fixed minute at a time.

The start/end interval is read-only.  It is split into non-overlapping
60-second windows; every window is sampled every two seconds at the requested
model resolution and sent to Qwen independently.  Minute labels are kept
separate from the later merged report so each exact evidence item remains
evidence behind each label.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI

import qwen_experiment_action_segment as action_base
import qwen_experiment_action_stepwise as stage_base
import qwen_experiment_segment_judge as qwen_base


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = ROOT / "outputs" / "marker_filter"
DEFAULT_SEGMENT_SOURCE = ROOT / "outputs" / "experiment_boundary" / "summary.json"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "action_minutes"

DECISIONS = {"observed", "no_stage_observed", "uncertain"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def locked_segment(record: dict[str, Any]) -> dict[str, Any] | None:
    segment = record.get("segment")
    if isinstance(segment, dict):
        return segment
    window = record.get("fixed_experiment_window_seconds")
    if isinstance(window, list) and len(window) == 2 and all(isinstance(item, (int, float)) for item in window):
        return {"start_seconds": float(window[0]), "end_seconds": float(window[1]), "segment_valid": True}
    return None


def build_prompt(
    minute_index: int,
    window_start: float,
    window_end: float,
    image_ids: list[str],
    sample_interval_seconds: float,
) -> str:
    stage_lines = "\n".join(
        f"- {stage}: {stage_base.STAGE_LABELS[stage]}；{stage_base.stage_definition(stage)}"
        for stage in stage_base.STAGES
    )
    return f"""你是中学物理实验视频判读员。视频内容是“伏安法测电阻”实验。

现在只分析固定的第 {minute_index + 1} 个一分钟窗口：视频相对时间 {window_start:.3f}s–{window_end:.3f}s。本窗口与其他分钟独立判断，不知道也不需要推测前后分钟发生了什么。
本窗口图片严格按时间顺序提供，每 {sample_interval_seconds:g} 秒一张。唯一图片 ID 为：{", ".join(image_ids)}。

动作定义：
{stage_lines}

实验可能按“连线、第一次测量、第一次记录、重新连线、第二次测量、第二次记录、整理材料”进行，但某些阶段可以缺失或在本分钟没有出现。仪器通常为橙红色，但颜色不能作为唯一依据。

判定要求：
1. 只报告本一分钟图片中直接看见的动作，不要根据实验流程补造动作，也不要把上一分钟或下一分钟的动作写进来。
2. 可以按时间顺序报告多个动作。一个动作必须有直接证据：测量要看见电表/仪表读数或闭合开关后的实际测量；记录要看见持续书写、填写或计算；重新连线要看见拆接或插接导线；整理材料要看见拆线、收拢或归位，并且不能只凭短暂移动判断最终结束。
3. “准备测量”“准备记录”“拿起笔”“摆放器材”“可能开始”都不是已经发生的测量或记录；若只能看到准备动作，就写 uncertain 或不列该阶段。
4. 图片底部左侧有 `FRAME ID=<id> | VIDEO T=<relative seconds>s`。必须原样返回真实 FRAME ID；不能使用 VIDEO T 数字代替 ID，不能使用摄像机日期时间。
5. 每个 observation 的 evidence_frame_id 必须属于本窗口，stage 必须使用下列英文枚举：{", ".join(stage_base.STAGES)}。stage_order 按图片时间顺序去重；不要为了流程排序。
6. 如果画面遮挡、动作只出现一张且无法确认、或多个动作无法按时间区分，decision 为 uncertain，并说明原因。不要猜测。

只输出一个合法 JSON 对象，不要 Markdown：
{{
  "minute_index": {minute_index},
  "window_seconds": [{window_start:.3f}, {window_end:.3f}],
  "decision": "observed" | "no_stage_observed" | "uncertain",
  "stage_order": ["circuit_wiring", "measurement_1"],
  "dominant_stage": "circuit_wiring" | null,
  "observations": [
    {{"stage": "circuit_wiring", "evidence_frame_id": "m00_001", "evidence": "不超过60字的直接可见证据", "confidence": 0.0}}
  ],
  "confidence": 0.0,
  "uncertainty": "不超过100字；无则为空字符串"
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
    raw = completion.choices[0].message.content or ""
    result: dict[str, Any] = {"finish_reason": completion.choices[0].finish_reason or "unknown", "raw_model_content": raw, "parsed": False}
    try:
        result["parsed_result"] = qwen_base.parse_json(raw)
        result["parsed"] = True
    except (json.JSONDecodeError, ValueError) as exc:
        result["parse_error"] = str(exc)
    return result


def valid_confidence(value: Any) -> bool:
    return isinstance(value, (int, float)) and 0.0 <= float(value) <= 1.0


def normalize_observations(value: dict[str, Any], frames: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Keep direct observations and retain rejected model labels as audit data."""
    normalized = dict(value)
    by_id = {str(frame["image_id"]) for frame in frames}
    accepted: list[dict[str, Any]] = []
    discarded: list[dict[str, Any]] = []
    observations = value.get("observations")
    if not isinstance(observations, list):
        return normalized, discarded
    for observation in observations:
        if not isinstance(observation, dict):
            discarded.append({"observation": observation, "reason": "not_an_object"})
            continue
        stage = observation.get("stage")
        frame_id = observation.get("evidence_frame_id")
        evidence = observation.get("evidence")
        reason = ""
        if stage not in stage_base.STAGES:
            reason = "stage_invalid"
        elif frame_id not in by_id:
            reason = "frame_id_invalid"
        elif not isinstance(evidence, str) or not evidence.strip():
            reason = "evidence_invalid"
        elif not stage_base.evidence_supports_stage(stage, evidence):
            reason = "evidence_does_not_support_stage"
        if reason:
            discarded.append({"observation": observation, "reason": reason})
        else:
            accepted.append(observation)
    normalized["observations"] = accepted
    normalized["stage_order"] = list(dict.fromkeys(str(item["stage"]) for item in accepted))
    if normalized.get("dominant_stage") not in normalized["stage_order"]:
        normalized["dominant_stage"] = normalized["stage_order"][-1] if normalized["stage_order"] else None
    return normalized, discarded


def validate(value: dict[str, Any] | None, minute_index: int, frames: list[dict[str, Any]]) -> list[str]:
    if value is None:
        return ["qwen_response_not_parsed"]
    errors: list[str] = []
    by_id = {str(frame["image_id"]): index for index, frame in enumerate(frames)}
    decision = value.get("decision")
    observations = value.get("observations")
    stage_order = value.get("stage_order")
    dominant = value.get("dominant_stage")
    if value.get("minute_index") != minute_index:
        errors.append("minute_index_mismatch")
    if decision not in DECISIONS:
        errors.append("decision_invalid")
    if not isinstance(observations, list) or len(observations) > len(stage_base.STAGES):
        errors.append("observations_invalid")
        observations = []
    if not isinstance(stage_order, list) or any(stage not in stage_base.STAGES for stage in stage_order):
        errors.append("stage_order_invalid")
        stage_order = []
    if dominant is not None and dominant not in stage_base.STAGES:
        errors.append("dominant_stage_invalid")
    if not isinstance(value.get("uncertainty"), str):
        errors.append("uncertainty_invalid")
    seen: list[str] = []
    observation_stages: list[str] = []
    for index, observation in enumerate(observations):
        if not isinstance(observation, dict):
            errors.append(f"observation_{index}_not_object")
            continue
        stage = observation.get("stage")
        frame_id = observation.get("evidence_frame_id")
        evidence = observation.get("evidence")
        if stage not in stage_base.STAGES:
            errors.append(f"observation_{index}_stage_invalid")
        if frame_id not in by_id:
            errors.append(f"observation_{index}_frame_invalid")
        if not isinstance(evidence, str) or not evidence.strip():
            errors.append(f"observation_{index}_evidence_invalid")
        if stage in stage_base.STAGES and frame_id in by_id:
            observation_stages.append(stage)
            seen.append(stage)
    if stage_order != list(dict.fromkeys(observation_stages)):
        errors.append("stage_order_observation_mismatch")
    if dominant is not None and dominant not in observation_stages:
        errors.append("dominant_not_observed")
    if decision == "observed" and not observation_stages:
        errors.append("observed_without_observation")
    if decision != "observed" and observation_stages:
        errors.append("non_observed_with_observation")
    if not valid_confidence(value.get("confidence", 0.0)):
        errors.append("confidence_invalid")
    return sorted(set(errors))


def format_clock(seconds: Any) -> str:
    return action_base.format_clock(seconds) if seconds is not None else "未确定"


def render_report(record: dict[str, Any]) -> str:
    lines = [
        f"# 固定一分钟动作判断：{record.get('source_video_id', '')}",
        "",
        f"固定实验区间：{format_clock(record['fixed_experiment_window_seconds'][0])}–{format_clock(record['fixed_experiment_window_seconds'][1])}",
        "",
        "| 分钟窗口 | 判断 | 动作顺序 | 主要动作 | 证据 |",
        "|---|---|---|---|---|",
    ]
    for item in record.get("minute_results", []):
        labels = "、".join(stage_base.STAGE_LABELS.get(stage, stage) for stage in item.get("stage_order", [])) or "无"
        evidence = "；".join(f"{stage_base.STAGE_LABELS.get(obs.get('stage'), obs.get('stage'))}@{obs.get('evidence_frame_id')}: {obs.get('evidence')}" for obs in item.get("observations", []))
        lines.append(f"| {format_clock(item['window_seconds'][0])}–{format_clock(item['window_seconds'][1])} | {item.get('decision')} | {labels} | {stage_base.STAGE_LABELS.get(item.get('dominant_stage'), '无')} | {evidence} |")
    lines.extend(["", "说明：每个一分钟窗口独立判断；本表不改变已锁定的实验开始和结束时间。", ""])
    return "\n".join(lines)


def process_video(client: OpenAI, source_record: dict[str, Any], manifest_path: Path, manifest: dict[str, Any], output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    video_id = str(source_record["source_video_id"])
    segment = locked_segment(source_record)
    if segment is None:
        raise ValueError("source record has no locked experiment segment")
    fixed_start, fixed_end = float(segment["start_seconds"]), float(segment["end_seconds"])
    video_dir = output_dir / qwen_base.slug(video_id)
    minute_results: list[dict[str, Any]] = []
    minute_index = 0
    status = "completed"
    evidence_insufficient = False
    while fixed_start + minute_index * 60.0 < fixed_end - 0.001:
        window_start = fixed_start + minute_index * 60.0
        window_end = min(fixed_end, window_start + 60.0)
        frames = action_base.extract_window(manifest, video_dir / "minutes" / f"{minute_index:02d}_{window_start:010.3f}", window_start, window_end, args.sample_interval_seconds, args.max_model_edge, f"m{minute_index:02d}")
        prompt = build_prompt(
            minute_index,
            window_start,
            window_end,
            [frame["image_id"] for frame in frames],
            args.sample_interval_seconds,
        )
        attempts: list[dict[str, Any]] = []
        parsed: dict[str, Any] | None = None
        discarded_observations: list[dict[str, Any]] = []
        errors: list[str] = []
        raw: dict[str, Any] = {}
        for attempt_index in range(args.max_attempts):
            raw = call_qwen(client, prompt, frames, args.max_tokens)
            candidate = raw.get("parsed_result")
            if isinstance(candidate, dict):
                parsed, discarded_observations = normalize_observations(candidate, frames)
            else:
                parsed, discarded_observations = None, []
            errors = validate(parsed, minute_index, frames)
            attempts.append({"attempt_index": attempt_index + 1, "qwen": raw, "discarded_observations": discarded_observations, "validation_errors": errors})
            if not errors:
                break
            prompt += "\n\n上一版回答未通过本地校验，错误为：" + "、".join(errors) + "。请严格按图片顺序重新输出完整 JSON。"
        item = {"minute_index": minute_index, "window_seconds": [window_start, window_end], "input_frames": frames, "qwen": raw, "attempts": attempts, "discarded_observations": discarded_observations, "valid": not errors, "validation_errors": errors}
        if parsed is not None:
            item.update({key: parsed.get(key) for key in ("decision", "stage_order", "dominant_stage", "observations", "uncertainty")})
        else:
            item.update({"decision": "uncertain", "stage_order": [], "dominant_stage": None, "observations": [], "uncertainty": "Qwen 未返回合法 JSON。"})
        minute_results.append(item)
        if item.get("decision") == "uncertain":
            evidence_insufficient = True
        if errors:
            evidence_insufficient = True
        minute_index += 1
    if evidence_insufficient:
        status = "completed_evidence_insufficient"
    return {
        "schema_version": "qwen_experiment_action_minute.v1",
        "generated_at": utc_now(),
        "source_video_id": video_id,
        "source_manifest": str(manifest_path.resolve()),
        "source_segment_source": str(args.segment_source.resolve()),
        "fixed_experiment_window_seconds": [fixed_start, fixed_end],
        "sampling": {"window_seconds": 60.0, "sample_interval_seconds": args.sample_interval_seconds, "max_model_edge": args.max_model_edge, "visible_banner": "bottom-left FRAME ID=<id> | VIDEO T=<video-relative-seconds>s"},
        "status": status,
        "minute_results": minute_results,
        "evidence_insufficient": evidence_insufficient,
        "evidence_status": "insufficient" if evidence_insufficient else "usable",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--segment-source", type=Path, default=DEFAULT_SEGMENT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--video-id", action="append", default=[])
    parser.add_argument("--sample-interval-seconds", type=float, default=2.0)
    parser.add_argument("--max-model-edge", type=int, default=640)
    parser.add_argument("--max-tokens", type=int, default=1800)
    parser.add_argument("--max-attempts", type=int, default=2)
    args = parser.parse_args(argv)
    if min(args.sample_interval_seconds, args.max_model_edge, args.max_tokens, args.max_attempts) <= 0:
        parser.error("sample interval, resolution, token, and attempt parameters must be positive")
    source_records = []
    for record in action_base.load_source_records(args.segment_source):
        segment = locked_segment(record)
        if segment and segment.get("segment_valid") is True and isinstance(segment.get("start_seconds"), (int, float)) and isinstance(segment.get("end_seconds"), (int, float)):
            source_records.append(record)
    if args.video_id:
        selected = set(args.video_id)
        source_records = [record for record in source_records if record.get("source_video_id") in selected]
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
            record = {"source_video_id": video_id, "status": "source_manifest_not_found", "evidence_insufficient": True, "evidence_status": "insufficient", "minute_results": []}
        else:
            path, manifest = manifests[video_id]
            try:
                record = process_video(client, source_record, path, manifest, args.output_dir, args)
            except Exception as exc:
                record = {"source_video_id": video_id, "status": "processing_failed", "error_type": type(exc).__name__, "error": qwen_base.safe_error_message(exc), "evidence_insufficient": True, "evidence_status": "insufficient", "minute_results": []}
        video_dir = args.output_dir / qwen_base.slug(video_id)
        write_json(video_dir / "result.json", record)
        if "fixed_experiment_window_seconds" in record:
            (video_dir / "minute_report.md").write_text(render_report(record), encoding="utf-8")
        records.append(record)
        print(json.dumps({"video": video_id, "status": record["status"], "minute_count": len(record.get("minute_results", [])), "evidence_insufficient": record.get("evidence_insufficient")}, ensure_ascii=False), flush=True)
    summary = {"schema_version": "qwen_experiment_action_minute.v1", "generated_at": utc_now(), "config": {"window_seconds": 60.0, "sample_interval_seconds": args.sample_interval_seconds, "max_model_edge": args.max_model_edge, "max_tokens": args.max_tokens, "max_attempts": args.max_attempts}, "source_segment_source": str(args.segment_source.resolve()), "records": records}
    write_json(args.output_dir / "summary.json", summary)
    print(f"summary={(args.output_dir / 'summary.json').resolve()}")
    fatal_statuses = {"source_manifest_not_found", "processing_failed"}
    return 1 if any(record.get("status") in fatal_statuses for record in records) else 0


if __name__ == "__main__":
    raise SystemExit(main())
