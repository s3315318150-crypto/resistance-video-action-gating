#!/usr/bin/env python3
"""Merge independent minute-level action evidence into chronological segments."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = ROOT / "outputs" / "action_minutes"
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


def format_clock(seconds: Any) -> str:
    if not isinstance(seconds, (int, float)):
        return "未观察到"
    total = max(0.0, float(seconds))
    minutes = int(total // 60)
    remainder = total - minutes * 60
    return f"{minutes:02d}:{remainder:06.3f}" if abs(remainder - round(remainder)) >= 0.0005 else f"{minutes:02d}:{int(round(remainder)):02d}"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def evidence_events(record: dict[str, Any]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for minute in record.get("minute_results", []):
        if not isinstance(minute, dict) or minute.get("valid") is not True:
            continue
        by_id = {str(frame.get("image_id")): frame for frame in minute.get("input_frames", []) if isinstance(frame, dict)}
        for observation in minute.get("observations", []):
            if not isinstance(observation, dict) or observation.get("stage") not in STAGES:
                continue
            frame = by_id.get(str(observation.get("evidence_frame_id")))
            timestamp = frame.get("timestamp_seconds") if isinstance(frame, dict) else None
            if not isinstance(timestamp, (int, float)):
                continue
            events.append({
                "stage": observation["stage"],
                "timestamp_seconds": float(timestamp),
                "frame_id": observation["evidence_frame_id"],
                "evidence": observation.get("evidence", ""),
                "minute_index": int(minute.get("minute_index", 0)),
            })
    return sorted(events, key=lambda item: (item["timestamp_seconds"], item["minute_index"], item["frame_id"]))


def merge_record(record: dict[str, Any]) -> dict[str, Any]:
    fixed_start, fixed_end = (float(item) for item in record["fixed_experiment_window_seconds"])
    events = evidence_events(record)
    segments: list[dict[str, Any]] = []
    ignored_confirmations: list[dict[str, Any]] = []
    if not events:
        return {"source_video_id": record.get("source_video_id"), "fixed_experiment_window_seconds": [fixed_start, fixed_end], "segments": [], "evidence_insufficient": True, "abstention_reason": "没有任何通过校验的动作证据帧。"}

    current = events[0]
    current_start = fixed_start
    for event in events[1:]:
        if event["stage"] == current["stage"]:
            ignored_confirmations.append(event)
            continue
        if event["timestamp_seconds"] <= current_start:
            ignored_confirmations.append({**event, "ignored_reason": "non_forward_event"})
            continue
        segments.append({
            "stage": current["stage"],
            "stage_label": STAGE_LABELS[current["stage"]],
            "start_seconds": current_start,
            "end_seconds": event["timestamp_seconds"],
            "start_source": "locked_experiment_start" if current_start == fixed_start else current["frame_id"],
            "end_source": event["frame_id"],
            "start_evidence": current["evidence"],
        })
        current = event
        current_start = event["timestamp_seconds"]
    segments.append({
        "stage": current["stage"],
        "stage_label": STAGE_LABELS[current["stage"]],
        "start_seconds": current_start,
        "end_seconds": fixed_end,
        "start_source": "locked_experiment_start" if current_start == fixed_start else current["frame_id"],
        "end_source": "locked_experiment_end",
        "start_evidence": current["evidence"],
    })
    return {
        "source_video_id": record.get("source_video_id"),
        "fixed_experiment_window_seconds": [fixed_start, fixed_end],
        "segments": segments,
        "evidence_events": events,
        "ignored_same_stage_confirmations": ignored_confirmations,
        "evidence_insufficient": False,
        "abstention_reason": None,
        "boundary_note": "边界使用下一动作的首个直接证据帧；动作可能在相邻 2 秒抽帧之间发生。",
    }


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        f"# 按动作分割：{result.get('source_video_id', '')}",
        "",
        f"固定实验区间：{format_clock(result['fixed_experiment_window_seconds'][0])}–{format_clock(result['fixed_experiment_window_seconds'][1])}",
        "",
        "| 动作阶段 | 开始时间 | 结束时间 | 边界依据 |",
        "|---|---:|---:|---|",
    ]
    for segment in result.get("segments", []):
        lines.append(f"| {segment['stage_label']} | {format_clock(segment['start_seconds'])} | {format_clock(segment['end_seconds'])} | {segment['start_source']} -> {segment['end_source']} |")
    lines.extend(["", "说明：每个边界是后一个动作首个直接证据帧；因抽帧间隔为 2 秒，真实切换可能在该帧与前一帧之间。", ""])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    args = parser.parse_args(argv)
    paths = sorted(args.input_dir.glob("*/result.json"))
    if not paths:
        parser.error("No minute-level result.json files found")
    records: list[dict[str, Any]] = []
    for path in paths:
        result = merge_record(read_json(path))
        output_dir = path.parent
        write_json(output_dir / "action_segments.json", result)
        (output_dir / "action_segments.md").write_text(render_markdown(result), encoding="utf-8")
        records.append(result)
        print(json.dumps({"video": result.get("source_video_id"), "segment_count": len(result.get("segments", []))}, ensure_ascii=False), flush=True)
    summary = {"schema_version": "qwen_experiment_action_minute_merge.v1", "generated_at": utc_now(), "source_minute_dir": str(args.input_dir.resolve()), "records": records}
    write_json(args.input_dir / "action_segments_summary.json", summary)
    print(f"summary={(args.input_dir / 'action_segments_summary.json').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
