#!/usr/bin/env python3
"""Local review for actions cut by overlapping Map-window boundaries."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import qwen_experiment_action_hierarchical_v1 as engine
import qwen_hierarchical_v1_contract as contract


REVIEWABLE_ACTIONS = {"wiring_action", "measurement_action", "writing_action", "cleanup_action"}


def _event_touches(event: dict[str, Any], timestamp: float, tolerance: float) -> bool:
    return (
        abs(float(event["first_seconds"]) - timestamp) <= tolerance + 1e-9
        or abs(float(event["last_seconds"]) - timestamp) <= tolerance + 1e-9
        or float(event["first_seconds"]) <= timestamp <= float(event["last_seconds"])
    )


def discover_boundary_bridge_candidates(
    prepared: dict[str, Any],
    window_results: list[dict[str, Any]],
    sample_interval_seconds: float,
) -> list[dict[str, Any]]:
    """Return generic edge-touching or cross-window same-action candidates."""
    fixed_start = float(prepared["fixed_start"])
    fixed_end = float(prepared["fixed_end"])
    tolerance = max(0.5, float(sample_interval_seconds))
    results_by_id = {str(item["window_id"]): item for item in window_results}
    windows = sorted(prepared["prepared_windows"], key=lambda item: float(item["window_seconds"][0]))
    candidates: dict[tuple[float, str], dict[str, Any]] = {}

    def add(timestamp: float, action: str, event_ids: list[str], reason: str) -> None:
        if action not in REVIEWABLE_ACTIONS or timestamp <= fixed_start + 1e-9 or timestamp >= fixed_end - 1e-9:
            return
        key = (round(timestamp, 3), action)
        existing = candidates.get(key)
        if existing is None:
            candidates[key] = {
                "bridge_id": "",
                "action_type": action,
                "boundary_seconds": round(timestamp, 6),
                "review_window_seconds": [
                    round(max(fixed_start, timestamp - 10.0), 6),
                    round(min(fixed_end, timestamp + 10.0), 6),
                ],
                "source_event_ids": sorted(set(event_ids)),
                "trigger_reasons": [reason],
            }
        else:
            existing["source_event_ids"] = sorted(set(existing["source_event_ids"] + event_ids))
            if reason not in existing["trigger_reasons"]:
                existing["trigger_reasons"].append(reason)

    for window in windows:
        window_id = str(window["window_id"])
        result = results_by_id.get(window_id, {})
        events = list(result.get("normalized_events", [])) if result.get("valid") else []
        start, end = (float(value) for value in window["window_seconds"])
        for event in events:
            action = str(event.get("action_type"))
            event_id = str(event.get("source_event_id", event.get("event_id", "")))
            if _event_touches(event, start, tolerance):
                add(start, action, [event_id], "action_touches_window_start")
            if _event_touches(event, end, tolerance):
                add(end, action, [event_id], "action_touches_window_end")

    for left, right in zip(windows, windows[1:]):
        left_events = list(results_by_id.get(str(left["window_id"]), {}).get("normalized_events", []))
        right_events = list(results_by_id.get(str(right["window_id"]), {}).get("normalized_events", []))
        left_end = float(left["window_seconds"][1])
        right_start = float(right["window_seconds"][0])
        for left_event in left_events:
            action = str(left_event.get("action_type"))
            matches = [event for event in right_events if event.get("action_type") == action]
            for right_event in matches:
                if float(right_event["first_seconds"]) <= float(left_event["last_seconds"]) + 2 * tolerance:
                    if _event_touches(left_event, left_end, tolerance):
                        timestamp = left_end
                    elif _event_touches(right_event, right_start, tolerance):
                        timestamp = right_start
                    else:
                        continue
                    add(
                        timestamp,
                        action,
                        [
                            str(left_event.get("source_event_id", "")),
                            str(right_event.get("source_event_id", "")),
                        ],
                        "same_action_reported_across_adjacent_windows",
                    )

    ordered = sorted(candidates.values(), key=lambda item: (item["boundary_seconds"], item["action_type"]))
    for index, candidate in enumerate(ordered, start=1):
        candidate["bridge_id"] = f"boundary_bridge_{index:03d}"
    return ordered


def build_boundary_bridge_prompt(candidate: dict[str, Any], frames: list[dict[str, Any]]) -> str:
    frame_ids = [str(frame["image_id"]) for frame in frames]
    action = str(candidate["action_type"])
    start, end = candidate["review_window_seconds"]
    return f"""你正在复核伏安法测电阻视频中被一分钟窗口边缘截断的动作。

候选 `{candidate['bridge_id']}` 只允许判断 `{action}`，复核范围为视频相对时间 {start:.3f}s–{end:.3f}s。合法 FRAME ID：{', '.join(frame_ids)}。

要求：
1. 逐帧判断该动作在边界前后是否连续，不能改判成其他动作，也不能根据实验顺序补造。
2. observed 时选择最早和最晚仍直接支持 `{action}` 的帧，代表帧必须位于两者之间。
3. 画面仅支持原事件而没有延伸也可以回答 observed；程序只会合并同类动作。
4. 没看到该动作回答 no；遮挡或无法确定边缘回答 uncertain。

只输出一个 JSON：
{{
  "bridge_id": "{candidate['bridge_id']}",
  "decision": "observed" | "no" | "uncertain",
  "action_type": "{action}",
  "first_frame_id": "{frame_ids[0]}" | null,
  "last_frame_id": "{frame_ids[-1]}" | null,
  "representative_frame_id": "{frame_ids[0]}" | null,
  "evidence": "不超过160字，只描述直接可见的同类动作",
  "confidence": 0.0
}}"""


def validate_boundary_bridge_response(
    value: Any, candidate: dict[str, Any], frames: list[dict[str, Any]]
) -> list[str]:
    if not isinstance(value, dict):
        return ["boundary_bridge_not_object"]
    errors: list[str] = []
    known = {str(frame["image_id"]): index for index, frame in enumerate(frames)}
    if value.get("bridge_id") != candidate["bridge_id"]:
        errors.append("bridge_id_mismatch")
    if value.get("action_type") != candidate["action_type"]:
        errors.append("action_type_mismatch")
    decision = value.get("decision")
    if decision not in {"observed", "no", "uncertain"}:
        errors.append("decision_invalid")
    ids = [value.get(name) for name in ("first_frame_id", "last_frame_id", "representative_frame_id")]
    if decision == "observed":
        if any(not isinstance(item, str) or item not in known for item in ids):
            errors.append("observed_frame_ids_invalid")
        else:
            first, last, representative = (str(item) for item in ids)
            if known[first] > known[last]:
                errors.append("frame_order_invalid")
            if not known[first] <= known[representative] <= known[last]:
                errors.append("representative_outside_interval")
    elif any(item is not None for item in ids):
        errors.append("non_observed_with_frame_ids")
    evidence = value.get("evidence")
    if not isinstance(evidence, str) or not evidence.strip() or len(evidence) > 160:
        errors.append("evidence_invalid")
    confidence = value.get("confidence")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not 0.0 <= float(confidence) <= 1.0
    ):
        errors.append("confidence_invalid")
    return sorted(set(errors))


def normalize_boundary_bridge_event(
    value: dict[str, Any], candidate: dict[str, Any], frames: list[dict[str, Any]]
) -> dict[str, Any] | None:
    if value.get("decision") != "observed":
        return None
    by_id = {str(frame["image_id"]): frame for frame in frames}
    first = by_id[str(value["first_frame_id"])]
    last = by_id[str(value["last_frame_id"])]
    representative = by_id[str(value["representative_frame_id"])]
    return {
        "source_event_id": f"{candidate['bridge_id']}_e01",
        "window_id": candidate["bridge_id"],
        "action_type": candidate["action_type"],
        "first_frame_id": first["image_id"],
        "last_frame_id": last["image_id"],
        "representative_frame_id": representative["image_id"],
        "first_frame_number": int(first["frame_number"]),
        "last_frame_number": int(last["frame_number"]),
        "representative_frame_number": int(representative["frame_number"]),
        "first_seconds": float(first["timestamp_seconds"]),
        "last_seconds": float(last["timestamp_seconds"]),
        "representative_seconds": float(representative["timestamp_seconds"]),
        "evidence": str(value["evidence"]),
        "confidence": float(value["confidence"]),
        "boundary_bridge_confirmation": True,
        "boundary_seconds": candidate["boundary_seconds"],
    }


def _frames_for_review(
    prepared: dict[str, Any], candidate: dict[str, Any], interval: float, max_model_edge: int
) -> list[dict[str, Any]]:
    start, end = (float(value) for value in candidate["review_window_seconds"])
    numbers = engine._frame_numbers_for_range(
        start, end, interval, float(prepared["fps"]), int(prepared["frame_count"])
    )
    engine._extract_source_frames(
        prepared["manifest"], numbers, prepared["frames_dir"], max_model_edge, prepared["frame_registry"]
    )
    return [prepared["frame_registry"][number] for number in numbers]


def run_boundary_bridge_review(
    prepared: dict[str, Any],
    map_events: list[dict[str, Any]],
    window_results: list[dict[str, Any]],
    client: Any,
    args: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    candidates = discover_boundary_bridge_candidates(prepared, window_results, args.sample_interval_seconds)
    reviews: list[dict[str, Any]] = []
    review_reasons: list[str] = []
    added_events: list[dict[str, Any]] = []
    for candidate in candidates:
        bridge_dir = prepared["video_dir"] / "map" / "boundary_bridges" / str(candidate["bridge_id"])
        passes: list[dict[str, Any]] = []
        selected_event: dict[str, Any] | None = None
        for interval in (1.0, 0.5):
            frames = _frames_for_review(prepared, candidate, interval, args.max_model_edge)
            prompt = build_boundary_bridge_prompt(candidate, frames)
            engine._write_text(bridge_dir / f"prompt_{interval:g}s.txt", prompt)
            contract.write_json_atomic(
                bridge_dir / f"input_{interval:g}s.json",
                {**candidate, "sample_interval_seconds": interval, "input_frames": frames},
            )
            raw = engine._attempt_qwen(client, prompt, frames, args.map_max_tokens)
            parsed = raw.get("parsed_result") if isinstance(raw.get("parsed_result"), dict) else None
            errors = validate_boundary_bridge_response(parsed, candidate, frames)
            event = normalize_boundary_bridge_event(parsed, candidate, frames) if parsed is not None and not errors else None
            pass_result = {
                "sample_interval_seconds": interval,
                "valid": not errors,
                "validation_errors": errors,
                "qwen": raw,
                "parsed_result": parsed,
                "normalized_event": event,
            }
            passes.append(pass_result)
            if event is not None:
                selected_event = event
                break
            if not errors and isinstance(parsed, dict) and parsed.get("decision") == "no":
                break
        if selected_event is not None:
            added_events.append(selected_event)
        elif any(item["validation_errors"] for item in passes):
            review_reasons.append(f"boundary_bridge_invalid:{candidate['bridge_id']}")
        review = {
            **candidate,
            "passes": passes,
            "accepted_extension": selected_event is not None,
            "accepted_event": selected_event,
            "original_events_preserved": True,
        }
        reviews.append(review)
        contract.write_json_atomic(bridge_dir / "result.json", review)
    prepared["_boundary_bridge_reviews"] = reviews
    return [*map_events, *added_events], reviews, review_reasons
