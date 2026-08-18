#!/usr/bin/env python3
"""Current-run R5 pointer evidence reused for Rubric 4."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

import cv2


AGENT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = AGENT_ROOT.parent
V15_PATH = AGENT_ROOT / "scripts" / "run_qwen_meter_polarity_lenient.py"
ALGORITHM_VERSION = "r4_meter_polarity_v21_r5_direct_meter_pointer"
RUBRIC_ID = 4
POINTER_STATES = {"normal_positive_deflection", "reverse_below_zero", "zero_or_unclear"}
READING_SIGNS = {"positive", "negative", "zero", "unclear"}


def _load_v15() -> Any:
    spec = importlib.util.spec_from_file_location("resistance_agent_meter_polarity_v15", V15_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load v15 implementation: {V15_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


V15 = _load_v15()
DEFAULT_STAGE_MANIFEST = V15.DEFAULT_STAGE_MANIFEST
DEFAULT_REFERENCE_MANIFEST = V15.DEFAULT_REFERENCE_MANIFEST
DEFAULT_DETECTOR_ROOT = V15.DEFAULT_DETECTOR_ROOT


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _confidence(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return round(max(0.0, min(1.0, float(value))), 4)


def _source_record(summary: dict[str, Any], source_video_id: str, video_id: str) -> dict[str, Any]:
    records = summary.get("records")
    if not isinstance(records, list):
        raise ValueError("Temporal Guard records are missing")
    for record in records:
        if not isinstance(record, dict):
            continue
        source = str(record.get("source_video_id") or "")
        if source != source_video_id and not source.startswith(f"{video_id}_"):
            continue
        for key in ("replay_result", "result_path"):
            nested = record.get(key)
            if isinstance(nested, str) and Path(nested).is_file():
                document = read_json(Path(nested))
                if isinstance(document.get("observed_stage_runs"), list):
                    return document
        return record
    raise ValueError(f"Temporal Guard record not found for video {video_id}")


def _stage_runs(record: dict[str, Any]) -> list[dict[str, Any]]:
    raw = record.get("observed_stage_runs")
    return sorted(
        [item for item in raw if isinstance(item, dict)] if isinstance(raw, list) else [],
        key=lambda item: float(item.get("start_seconds") or 0.0),
    )


def _fallback_stage_frames(
    record: dict[str, Any],
    duration: float,
    fps: float,
    stage_mode: str = "measurement_first",
    max_stage_frames: int = 8,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    runs = _stage_runs(record)
    for run in runs:
        stage = str(run.get("stage") or "")
        if stage not in V15.OBSERVATION_STAGES:
            continue
        if stage_mode == "pre_recording_recovery" and stage.startswith("measurement"):
            continue
        start = max(0.0, float(run.get("start_seconds") or 0.0))
        end = min(max(0.0, duration - 0.1), float(run.get("end_seconds") or start))
        phase = (
            "measurement"
            if stage.startswith("measurement")
            or run.get("contains_measurement_evidence") is True
            or run.get("merged_measurement_recording") is True
            else "recording"
        )
        for value in V15.evenly_sample(
            [start + (end - start) * index / 7.0 for index in range(8)], 4
        ):
            timestamp = round(float(value), 3)
            rows.append(
                {
                    "stage": stage,
                    "observation_phase": phase,
                    "timestamp_seconds": timestamp,
                    "frame_number": int(round(timestamp * fps)),
                    "stage_interval_seconds": [round(start, 3), round(end, 3)],
                }
            )
    recording_starts = [
        float(item.get("start_seconds") or 0.0)
        for item in runs
        if str(item.get("stage") or "").startswith("recording")
    ]
    if not any(row["observation_phase"] == "measurement" for row in rows):
        for start in recording_starts[:2]:
            left, right = max(0.0, start - 24.0), max(0.0, start - 2.0)
            for timestamp in (left, left + (right - left) / 3.0, left + 2.0 * (right - left) / 3.0, right):
                rows.append(
                    {
                        "stage": "measurement_recovery",
                        "observation_phase": "measurement",
                        "timestamp_seconds": round(timestamp, 3),
                        "frame_number": int(round(timestamp * fps)),
                        "stage_interval_seconds": [round(left, 3), round(right, 3)],
                    }
                )
    if not rows and stage_mode == "broad_search" and duration > 0:
        right = max(0.0, duration - 0.1)
        for timestamp in V15.evenly_sample(
            [right * index / 11.0 for index in range(12)], max_stage_frames
        ):
            rows.append(
                {
                    "stage": "broad_visual_search",
                    "observation_phase": "measurement",
                    "timestamp_seconds": round(float(timestamp), 3),
                    "frame_number": int(round(float(timestamp) * fps)),
                    "stage_interval_seconds": [0.0, round(right, 3)],
                }
            )
    by_frame: dict[int, dict[str, Any]] = {}
    for row in rows:
        by_frame[int(row["frame_number"])] = row
    ordered = sorted(by_frame.values(), key=lambda item: float(item["timestamp_seconds"]))
    if stage_mode == "measurement_first":
        ordered.sort(key=lambda item: (item.get("observation_phase") != "measurement", float(item["timestamp_seconds"])))
    elif stage_mode == "pre_recording_recovery":
        ordered.sort(key=lambda item: (item.get("stage") != "measurement_recovery", float(item["timestamp_seconds"])))
    return V15.evenly_sample(ordered, max_stage_frames)


def select_stage_frames(
    video_id: str,
    record: dict[str, Any],
    duration: float,
    fps: float,
    stage_manifest_path: Path,
    detector_root: Path,
    allow_video_calibration: bool = True,
    stage_mode: str = "measurement_first",
    max_stage_frames: int = 8,
) -> tuple[list[dict[str, Any]], str]:
    if allow_video_calibration and stage_manifest_path.is_file():
        manifest = read_json(stage_manifest_path)
        try:
            video = V15.select_stage_video(manifest, video_id)
        except ValueError:
            video = None
        if isinstance(video, dict):
            preferred = V15.detector_frame_numbers(detector_root, video_id)
            selected = V15.select_stage_frames(
                video, max_stage_frames, preferred_frame_numbers=preferred
            )
            return [
                {
                    "stage": str(item["stage"]),
                    "observation_phase": str(item["observation_phase"]),
                    "timestamp_seconds": float(item["timestamp_seconds"]),
                    "frame_number": int(item["frame_number"]),
                    "stage_interval_seconds": item.get("stage_interval_seconds"),
                }
                for item in selected
            ], "v15_measurement_recording_manifest"
    return _fallback_stage_frames(
        record,
        duration,
        fps,
        stage_mode=stage_mode,
        max_stage_frames=max_stage_frames,
    ), "temporal_guard_broad_recovery"


def decode_stage_frames(
    video_path: Path,
    selected: list[dict[str, Any]],
    output_dir: Path,
) -> list[dict[str, Any]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"unable to open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    try:
        for row in selected:
            capture.set(cv2.CAP_PROP_POS_MSEC, float(row["timestamp_seconds"]) * 1000.0)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            frame_number = max(0, int(round(capture.get(cv2.CAP_PROP_POS_FRAMES))) - 1)
            if frame_number in seen:
                continue
            seen.add(frame_number)
            actual = frame_number / fps if fps > 0 else float(row["timestamp_seconds"])
            frame_id = f"frame_{frame_number:08d}"
            path = output_dir / "source_frames" / f"{frame_id}_{actual:010.3f}s.jpg"
            V15.write_jpeg(path, frame, quality=94)
            rows.append(
                {
                    **row,
                    "frame_number": frame_number,
                    "frame_id": frame_id,
                    "timestamp_seconds": round(actual, 6),
                    "output_frame_path": str(path.resolve()),
                }
            )
    finally:
        capture.release()
    if not rows:
        raise RuntimeError("no polarity stage frames decoded")
    return rows


def build_groups(
    rows: list[dict[str, Any]],
    video_id: str,
    evidence_dir: Path,
    reference_manifest_path: Path,
    detector_root: Path,
    allow_video_calibration: bool = True,
    dynamic_meter_candidates: bool = True,
    candidate_crops_per_frame: int = 4,
) -> list[dict[str, Any]]:
    reference = read_json(reference_manifest_path) if allow_video_calibration and reference_manifest_path.is_file() else {}
    boxes, size, source_id = V15.reference_boxes(reference, video_id) if allow_video_calibration else ({}, None, None)
    preferred = V15.detector_frame_numbers(detector_root, video_id) if allow_video_calibration else set()
    polarity_boxes = {key: value for key, value in boxes.items() if key in {"ammeter", "voltmeter"}}
    if preferred:
        polarity_boxes, size = {}, None
    groups = V15.build_media_groups(
        rows,
        polarity_boxes,
        size,
        evidence_dir / "media",
        detector_root=detector_root if allow_video_calibration else None,
        video_id=video_id if allow_video_calibration else None,
    )
    frame_by_number = {int(row["frame_number"]): row for row in rows}
    for index, group in enumerate(groups, start=1):
        timestamp = float(group["timestamp_seconds"])
        row = min(rows, key=lambda item: abs(float(item["timestamp_seconds"]) - timestamp))
        group["image_group"] = index
        group["frame_id"] = row["frame_id"]
        group["frame_number"] = row["frame_number"]
        group["reference_source_id"] = source_id
        if not allow_video_calibration and dynamic_meter_candidates:
            try:
                from . import meter_rubrics as meter_module
            except ImportError:
                import meter_rubrics as meter_module  # type: ignore
            exported = meter_module._export_candidates(
                {"frame_path": str(row["output_frame_path"]), "sharpness": group.get("overview_sharpness", 0.0)},
                evidence_dir / "dynamic_meter_candidates",
            )
            for candidate in list(exported.get("candidates") or [])[:candidate_crops_per_frame]:
                if not isinstance(candidate, dict):
                    continue
                candidate_path = candidate.get("enhanced_path") or candidate.get("wide_path")
                if not candidate_path:
                    continue
                group["rois"].append(
                    {
                        "instrument": "meter_candidate",
                        "candidate_label": str(candidate.get("candidate_id") or "dynamic"),
                        "detector_identity_trusted": False,
                        "path": str(candidate_path),
                        "face_path": candidate.get("face_path"),
                        "bbox_xyxy": candidate.get("bbox"),
                    }
                )
            group["dynamic_detector_json"] = None
    return groups


def _group_listing(groups: list[dict[str, Any]]) -> str:
    return ", ".join(
        f"{item['image_group']}={item['frame_id']}@{float(item['timestamp_seconds']):.3f}s/{item['stage']}"
        for item in groups
    )


def endpoint_prompt(groups: list[dict[str, Any]], skill_instruction: str = "") -> str:
    base = V15.prompt_text(groups, sorted({str(roi.get("instrument")) for group in groups for roi in group.get("rois", [])}))
    return (
        "本地可信图组映射为：" + _group_listing(groups) + "。"
        "Skill instruction: " + skill_instruction + "。"
        "每个 evidence_seconds 必须精确取自该映射中的真实时间点，不得生成其他时间。\n" + base
        + "\n输出约束：只填写最终可见观察，不展示思考过程，不重复追踪或自我修正。"
        "ammeter.evidence、voltmeter.evidence、source_positive_evidence 各不超过120个汉字，"
        "observation_summary 不超过80个汉字。导线远端或电源极性发生冲突时直接填 unclear，"
        "不得在 evidence 字段中继续讨论多种可能。"
    )


def pointer_prompt(groups: list[dict[str, Any]], skill_instruction: str = "") -> str:
    return f"""只观察测量和记录阶段每个真实图组中标有 A 和 V 的表盘指针。可信映射：{_group_listing(groups)}。
Skill instruction: {skill_instruction}
同一 image_group 的全景和 ROI 只是一张真实帧的不同视图，只算一票。弧形刻度零位在最左端；指向竖直或右上刻度数字为 normal_positive_deflection；只有越过最左端零刻度向外反打才是 reverse_below_zero；停在零位、遮挡或看不清为 zero_or_unclear。导线或插头颜色完全不参与观察，evidence 中不得描述红线、黑线或其他导线颜色。
逐组独立填写，不分析接线，不输出评分。只返回 JSON：
{{"observations":[{{"image_group":1,"frame_id":"frame_00000000","ammeter_pointer":"normal_positive_deflection","voltmeter_pointer":"zero_or_unclear","confidence":0.0,"evidence":"visible pointer position"}}]}}"""


def reading_prompt(groups: list[dict[str, Any]], skill_instruction: str = "") -> str:
    return f"""逐个真实图组观察 A/V 表盘及记录纸 I/U 最终数值的正负号。可信映射：{_group_listing(groups)}。
Skill instruction: {skill_instruction}
同一 image_group 的不同裁剪只算一票。表针从最左端零位向刻度数字方向为 positive，越过左端向外反打才为 negative。纸面只有紧邻 I/U 最终数值的清楚负号才是 negative；空格、横线、模糊或未绑定符号为 unclear。导线或插头颜色完全不参与观察，evidence 中不得描述红线、黑线或其他导线颜色。不分析拓扑，不输出评分。
只返回 JSON：
{{"observations":[{{"image_group":1,"frame_id":"frame_00000000","ammeter_face_sign":"positive","voltmeter_face_sign":"unclear","recorded_current_sign":"unclear","recorded_voltage_sign":"unclear","confidence":0.0,"evidence":"visible sign"}}]}}"""


def _known_mapping(groups: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(group["image_group"]): group for group in groups}


def validate_frame_observations(
    value: dict[str, Any],
    groups: list[dict[str, Any]],
    kind: str,
) -> dict[str, Any]:
    raw = value.get("observations")
    if not isinstance(raw, list):
        raise ValueError(f"{kind}_observations_missing")
    mapping = _known_mapping(groups)
    parsed: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict) or item.get("image_group") not in mapping:
            raise ValueError(f"{kind}_image_group_invalid")
        group_id = int(item["image_group"])
        trusted = mapping[group_id]
        confidence = _confidence(item.get("confidence"))
        evidence = str(item.get("evidence") or "")
        if any(pattern.search(evidence) for pattern in V15.WIRE_COLOR_REFERENCE_PATTERNS):
            raise ValueError(f"{kind}_wire_color_evidence_forbidden")
        if kind == "pointer":
            if item.get("ammeter_pointer") not in POINTER_STATES or item.get("voltmeter_pointer") not in POINTER_STATES:
                raise ValueError("pointer_state_invalid")
        else:
            for field in ("ammeter_face_sign", "voltmeter_face_sign", "recorded_current_sign", "recorded_voltage_sign"):
                if item.get(field) not in READING_SIGNS:
                    raise ValueError(f"{field}_invalid")
        parsed.append(
            {
                **item,
                "image_group": group_id,
                "frame_id": trusted["frame_id"],
                "model_frame_id": str(item.get("frame_id") or ""),
                "frame_id_corrected_from_group": item.get("frame_id") != trusted["frame_id"],
                "timestamp_seconds": trusted["timestamp_seconds"],
                "confidence": confidence,
                "evidence": evidence,
            }
        )
    if {item["image_group"] for item in parsed} != set(mapping):
        raise ValueError(f"{kind}_image_groups_incomplete")
    return {"observations": sorted(parsed, key=lambda item: item["image_group"])}


def validate_endpoint(value: dict[str, Any], groups: list[dict[str, Any]]) -> dict[str, Any]:
    content = json.dumps(value, ensure_ascii=False)
    parsed, errors = V15.validate_response(content)
    if parsed is None:
        raise ValueError("endpoint_schema_invalid:" + ",".join(errors))
    allowed = {round(float(group["timestamp_seconds"]), 3): group for group in groups}
    support: dict[str, list[dict[str, Any]]] = {}
    for role in ("ammeter", "voltmeter"):
        matched: list[dict[str, Any]] = []
        for timestamp in parsed[role].get("evidence_seconds") or []:
            key = round(float(timestamp), 3)
            group = allowed.get(key)
            if group is None:
                raise ValueError(f"{role}_evidence_timestamp_not_supplied:{timestamp}")
            matched.append(
                {
                    "image_group": group["image_group"],
                    "frame_id": group["frame_id"],
                    "timestamp_seconds": group["timestamp_seconds"],
                }
            )
        support[role] = matched
    return {"base_observation": {key: value[key] for key in V15.RESPONSE_FIELDS}, "endpoint_support": support}


def _media_paths(group: dict[str, Any], kind: str) -> list[Path]:
    paths = [Path(group["overview"])]
    for roi in group.get("rois") or []:
        instrument = str(roi.get("instrument") or "")
        if kind == "pointer" and instrument not in {"ammeter", "voltmeter", "meter_candidate", "topology_context"}:
            continue
        paths.append(Path(roi["path"]))
    return paths


def _call_qwen(
    prompt: str,
    groups: list[dict[str, Any]],
    model_config: dict[str, Any],
    raw_path: Path,
    validator: Callable[[dict[str, Any], list[dict[str, Any]]], dict[str, Any]],
    kind: str,
    execution_fingerprint: str | None = None,
) -> dict[str, Any]:
    if raw_path.is_file():
        cached = read_json(raw_path)
        if (
            cached.get("algorithm_version") == ALGORITHM_VERSION
            and cached.get("execution_fingerprint") == execution_fingerprint
            and isinstance(cached.get("observation"), dict)
        ):
            return validator(cached["observation"], groups)
    base_url = os.getenv("QWEN_API_BASE_URL", str(model_config["base_url"]))
    token = os.getenv("QWEN_API_TOKEN", "EMPTY")
    model = os.getenv("QWEN_MODEL", str(model_config["model"]))
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    media: list[dict[str, Any]] = []
    for group in groups:
        content.append(
            {
                "type": "text",
                "text": f"Image group {group['image_group']}; trusted frame_id={group['frame_id']}; VIDEO T={float(group['timestamp_seconds']):.3f}s.",
            }
        )
        for path in _media_paths(group, kind):
            content.append({"type": "image_url", "image_url": {"url": V15.data_url(path)}})
            media.append({"image_group": group["image_group"], "path": str(path.resolve())})
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 4096,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    attempts: list[dict[str, Any]] = []
    for attempt in range(2):
        text: str | None = None
        finish_reason: Any = None
        request = urllib.request.Request(
            base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                raw = json.loads(response.read().decode("utf-8"))
            choice = raw.get("choices", [{}])[0]
            text = choice.get("message", {}).get("content")
            finish_reason = choice.get("finish_reason")
            if not isinstance(text, str):
                raise ValueError("response_content_not_text")
            value = V15.parse_model_json(text)
            parsed = validator(value, groups)
            attempts.append({"attempt": attempt + 1, "content": text, "finish_reason": finish_reason, "schema_errors": []})
            write_json(
                raw_path,
                {
                    "algorithm_version": ALGORITHM_VERSION,
                    "execution_fingerprint": execution_fingerprint,
                    "kind": kind,
                    "model": model,
                    "base_url": base_url,
                    "prompt": prompt,
                    "media": media,
                    "attempts": attempts,
                    "observation": value,
                },
            )
            return parsed
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ConnectionError, OSError, ValueError, json.JSONDecodeError) as exc:
            failure: dict[str, Any] = {"attempt": attempt + 1, "errors": [f"{type(exc).__name__}:{exc}"]}
            if isinstance(text, str):
                failure["content"] = text
            if "finish_reason" in locals():
                failure["finish_reason"] = finish_reason
            attempts.append(failure)
            if attempt == 0:
                content.append(
                    {
                        "type": "text",
                        "text": (
                            "证据与结构修正：只返回完整合法 JSON；逐组使用给定 image_group，"
                            "时间必须来自可信映射。导线或插头颜色必须完全忽略，不能用于区分、"
                            "跟踪或描述任何端点路径；路径不清楚时填写 unclear。"
                        ),
                    }
                )
                time.sleep(2.0)
    write_json(
        raw_path.with_name(raw_path.stem + "_failed.json"),
        {"algorithm_version": ALGORITHM_VERSION, "kind": kind, "model": model, "base_url": base_url, "media": media, "attempts": attempts, "status": "request_failed"},
    )
    raise RuntimeError(f"Qwen {kind} request failed after targeted retry: {attempts}")


def pointer_consensus(
    observation: dict[str, Any],
    minimum_distinct_frames: int = 2,
    minimum_confidence: float = 0.85,
) -> tuple[dict[str, str], float | None, dict[str, Any]]:
    result: dict[str, str] = {}
    support_report: dict[str, Any] = {}
    confidences: list[float] = []
    for role in ("ammeter", "voltmeter"):
        field = f"{role}_pointer"
        items = list(observation.get("observations") or [])
        normal = [item for item in items if item[field] == "normal_positive_deflection" and item["confidence"] >= minimum_confidence]
        reverse = [item for item in items if item[field] == "reverse_below_zero" and item["confidence"] >= minimum_confidence]
        if len({item["frame_id"] for item in normal}) >= minimum_distinct_frames:
            state, selected = "normal_positive_deflection", normal
        elif len({item["frame_id"] for item in reverse}) >= minimum_distinct_frames:
            state, selected = "reverse_below_zero", reverse
        else:
            state, selected = "zero_or_unclear", []
        result[role] = state
        if selected:
            confidences.append(min(item["confidence"] for item in selected))
        support_report[role] = {
            "state": state,
            "minimum_required_distinct_frames": minimum_distinct_frames,
            "supporting_frame_ids": [item["frame_id"] for item in selected],
            "supporting_image_groups": [item["image_group"] for item in selected],
        }
    return result, min(confidences) if confidences else None, support_report


def reading_consensus(observation: dict[str, Any]) -> dict[str, Any]:
    fields = ("ammeter_face_sign", "voltmeter_face_sign", "recorded_current_sign", "recorded_voltage_sign")
    items = list(observation.get("observations") or [])
    result: dict[str, Any] = {"evidence_seconds": [], "evidence": "frame-bound reading sign consensus"}
    selected_confidences: list[float] = []
    for field in fields:
        negative = [item for item in items if item[field] == "negative" and item["confidence"] >= 0.85]
        positive = [item for item in items if item[field] == "positive" and item["confidence"] >= 0.85]
        zero = [item for item in items if item[field] == "zero" and item["confidence"] >= 0.85]
        selected = negative or positive or zero
        result[field] = "negative" if negative else "positive" if positive else "zero" if zero else "unclear"
        for item in selected:
            result["evidence_seconds"].append(item["timestamp_seconds"])
            selected_confidences.append(item["confidence"])
    result["evidence_seconds"] = sorted(set(result["evidence_seconds"]))
    result["confidence"] = min(selected_confidences) if selected_confidences else 0.0
    return result


def reduce_result(
    endpoint: dict[str, Any],
    pointer: dict[str, Any],
    reading: dict[str, Any],
    groups: list[dict[str, Any]],
    pointer_min_distinct_frames: int = 2,
    pointer_min_confidence: float = 0.85,
) -> dict[str, Any]:
    overrides, pointer_confidence, pointer_support = pointer_consensus(
        pointer,
        minimum_distinct_frames=pointer_min_distinct_frames,
        minimum_confidence=pointer_min_confidence,
    )
    reading_summary = reading_consensus(reading)
    base = endpoint["base_observation"]
    parsed, errors = V15.validate_response(
        json.dumps(base, ensure_ascii=False),
        meter_context_supplied=True,
        pointer_overrides=overrides,
        pointer_observation_confidence=pointer_confidence,
        reading_sign_observation=reading_summary,
    )
    if parsed is None:
        raise ValueError("v15_reducer_failed:" + ",".join(errors))
    decision = str(parsed["result"])
    return {
        "decision": decision,
        "predicted_score": 1 if decision == "pass" else 0,
        "confidence": float(parsed["confidence"]),
        "reason": str(parsed["reason"]),
        "diagnostics": {
            "algorithm_version": ALGORITHM_VERSION,
            "fail_trigger": parsed["fail_trigger"],
            "ammeter": parsed["ammeter"],
            "voltmeter": parsed["voltmeter"],
            "source_polarity_state": parsed["source_polarity_state"],
            "source_positive_end_identity": parsed["source_positive_end_identity"],
            "source_negative_end_identity": parsed["source_negative_end_identity"],
            "source_series_bridge_state": parsed["source_series_bridge_state"],
            "source_positive_evidence": parsed["source_positive_evidence"],
            "endpoint_support": endpoint["endpoint_support"],
            "pointer_observations": pointer["observations"],
            "pointer_consensus": pointer_support,
            "reading_sign_observations": reading["observations"],
            "reading_sign_consensus": reading_summary,
            "pointer_state_overrides": parsed["pointer_state_overrides"],
            "negative_reading_applied": parsed["negative_reading_applied"],
            "image_group_mapping": [
                {
                    "image_group": group["image_group"],
                    "frame_id": group["frame_id"],
                    "timestamp_seconds": group["timestamp_seconds"],
                    "stage": group["stage"],
                    "overview": group["overview"],
                    "rois": group.get("rois", []),
                }
                for group in groups
            ],
        },
    }


def reduce_r5_pointer_result(r5_result: dict[str, Any], r5_result_path: Path | None = None) -> dict[str, Any]:
    """Map the current run's direct meter observation to the R4 binary result."""
    decision = r5_result.get("decision")
    expected_score = 1 if decision == "pass" else 0
    if (
        r5_result.get("schema_version") != "resistance_agent_rubric_result.v2"
        or r5_result.get("rubric_id") != 5
        or decision not in {"pass", "fail"}
        or r5_result.get("predicted_score") != expected_score
        or r5_result.get("execution_mode") != "execute_visual_evidence"
    ):
        raise ValueError("current-run R5 result is invalid")

    source_diagnostics = r5_result.get("diagnostics")
    if not isinstance(source_diagnostics, dict):
        source_diagnostics = {}
    r5_reason = str(r5_result.get("reason") or "unspecified")
    path_value = str(r5_result_path.resolve()) if r5_result_path is not None else None
    return {
        "decision": decision,
        "predicted_score": expected_score,
        "confidence": _confidence(r5_result.get("confidence")),
        "reason": f"current_run_r5_direct_meter_pointer:{r5_reason}",
        "diagnostics": {
            "algorithm_version": ALGORITHM_VERSION,
            "decision_basis": "current_run_r5_direct_meter_pointer",
            "source_rubric_id": 5,
            "r5_decision": decision,
            "r5_reason": r5_reason,
            "r5_confidence": _confidence(r5_result.get("confidence")),
            "r5_result_path": path_value,
            "r5_result_sha256": sha256(r5_result_path) if r5_result_path is not None else None,
            "r5_source_artifact": r5_result.get("source_artifact"),
            "needle_states": source_diagnostics.get("needle_states", {}),
            "effective_needle_states": source_diagnostics.get(
                "effective_needle_states", source_diagnostics.get("needle_states", {})
            ),
            "identity_observations": source_diagnostics.get("identity_observations", {}),
            "evidence_timepoints_seconds": source_diagnostics.get("evidence_timepoints_seconds", []),
            "original_frame_paths": source_diagnostics.get("original_frame_paths", []),
            "roi_paths": source_diagnostics.get("roi_paths", []),
            "endpoint_topology_used": False,
            "wire_color_used": False,
            "historical_artifacts_used": False,
        },
    }


def run_polarity_rubric(
    video_path: Path,
    source_video_id: str,
    video_id: str,
    run_dir: Path,
    model_config: dict[str, Any],
    action_summary_path: Path | None = None,
    fallback_action_summary_path: Path | None = None,
    stage_manifest_path: Path | None = None,
    reference_manifest_path: Path | None = None,
    detector_root: Path | None = None,
    allow_video_calibration: bool = False,
    allow_historical_fallback: bool = False,
    skill_plan: dict[str, Any] | None = None,
    r5_result_path: Path | None = None,
) -> dict[str, Any]:
    try:
        from .skills import EXECUTOR_REGISTRY, execution_for_rubric
    except ImportError:
        from skills import EXECUTOR_REGISTRY, execution_for_rubric  # type: ignore
    execution = (
        execution_for_rubric(skill_plan, 4)
        if skill_plan
        else {
            "skill_id": "polarity.explicit_measurement_dynamic_roi",
            "parameters": dict(
                EXECUTOR_REGISTRY["polarity.explicit_measurement_dynamic_roi"].defaults
            ),
            "execution_fingerprint": None,
        }
    )
    parameters = execution["parameters"]
    if r5_result_path is None:
        raise ValueError("current-run R5 result is required for R4 execute")
    if r5_result_path is not None:
        expected_parent = (run_dir / "rubrics").resolve()
        resolved_r5_path = r5_result_path.resolve()
        if resolved_r5_path.parent != expected_parent or resolved_r5_path.name != "rubric_5.json":
            raise ValueError("R5 result must come from the current run rubric directory")
        r5_result = read_json(resolved_r5_path)
        if (
            str(r5_result.get("video_id")) != str(video_id)
            or r5_result.get("source_video_id") != source_video_id
            or r5_result.get("routing_policy") != "live_situation_skills.v1"
        ):
            raise ValueError("R5 result identity or routing provenance is invalid")
        source_digest = sha256(video_path)
        result = reduce_r5_pointer_result(r5_result, resolved_r5_path)
        report = {
            "schema_version": "resistance_agent_polarity_from_r5_evidence.v1",
            "algorithm_version": ALGORITHM_VERSION,
            "video_id": video_id,
            "source_video_id": source_video_id,
            "source_video_path": str(video_path.resolve()),
            "source_video_sha256": source_digest,
            "selection_source": "current_run_r5_visual_evidence",
            "r5_result_path": str(resolved_r5_path),
            "r5_result_sha256": sha256(resolved_r5_path),
            "r5_source_artifact": r5_result.get("source_artifact"),
            "rubric_4": result,
            "excel_accessed": False,
            "ground_truth_sent_to_model": False,
            "wiring_stage_accessed": False,
            "endpoint_topology_used": False,
            "wire_color_used": False,
            "source_video_unchanged": sha256(video_path) == source_digest,
            "selection_checkpoint_reused": False,
            "allow_video_calibration": False,
            "fixed_video_roi_used": False,
            "historical_fallback_used": False,
            "routing_policy": (skill_plan or {}).get("routing_policy"),
            "skill_selection": (skill_plan or {}).get("skills", []),
            "skill_execution": execution,
        }
        report_path = run_dir / "polarity_rubric" / "polarity_evidence_report.json"
        write_json(report_path, report)
        reopened = read_json(report_path)
        if reopened.get("rubric_4", {}).get("decision") != result["decision"]:
            raise ValueError("R4-from-R5 report verification failed")
        return {"rubric_4": result, "report_path": str(report_path.resolve())}

    action_path = action_summary_path if action_summary_path and action_summary_path.is_file() else (
        fallback_action_summary_path if allow_historical_fallback else None
    )
    if action_path is None or not action_path.is_file():
        raise ValueError("action summary is required")
    record = _source_record(read_json(action_path), source_video_id, video_id)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"unable to open video: {video_path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    duration = frame_count / fps if fps > 0 else 0.0
    evidence_dir = run_dir / "polarity_rubric"
    stage_path = stage_manifest_path or DEFAULT_STAGE_MANIFEST
    reference_path = reference_manifest_path or DEFAULT_REFERENCE_MANIFEST
    detector_path = detector_root or DEFAULT_DETECTOR_ROOT
    selected, selection_source = select_stage_frames(
        video_id,
        record,
        duration,
        fps,
        stage_path,
        detector_path,
        allow_video_calibration=allow_video_calibration,
        stage_mode=str(parameters["stage_mode"]),
        max_stage_frames=int(parameters["max_stage_frames"]),
    )
    source_digest = sha256(video_path)
    checkpoint_path = evidence_dir / "evidence_pre_qwen.json"
    checkpoint = read_json(checkpoint_path) if checkpoint_path.is_file() else {}
    checkpoint_valid = (
        checkpoint.get("algorithm_version") == ALGORITHM_VERSION
        and checkpoint.get("source_video_sha256") == source_digest
        and checkpoint.get("allow_video_calibration", True) == allow_video_calibration
        and checkpoint.get("routing_policy") == (skill_plan or {}).get("routing_policy")
        and checkpoint.get("execution_fingerprint") == execution["execution_fingerprint"]
        and isinstance(checkpoint.get("groups"), list)
    )
    if checkpoint_valid:
        groups = list(checkpoint["groups"])
    else:
        rows = decode_stage_frames(video_path, selected, evidence_dir)
        groups = build_groups(
            rows,
            video_id,
            evidence_dir,
            reference_path,
            detector_path,
            allow_video_calibration=allow_video_calibration,
            dynamic_meter_candidates=bool(parameters["dynamic_meter_candidates"]),
            candidate_crops_per_frame=int(parameters["candidate_crops_per_frame"]),
        )
        write_json(
            checkpoint_path,
            {
                "algorithm_version": ALGORITHM_VERSION,
                "source_video_sha256": source_digest,
                "allow_video_calibration": allow_video_calibration,
                "routing_policy": (skill_plan or {}).get("routing_policy"),
                "execution_fingerprint": execution["execution_fingerprint"],
                "skill_execution": execution,
                "selection_source": selection_source,
                "selected_stage_frames": selected,
                "groups": groups,
            },
        )
    measurement_groups = [group for group in groups if str(group.get("observation_phase")) == "measurement"]
    if not measurement_groups:
        measurement_groups = groups
    skill_instruction = str(parameters["prompt_instruction"])
    endpoint = _call_qwen(endpoint_prompt(groups, skill_instruction), groups, model_config, evidence_dir / "qwen" / "endpoint.json", validate_endpoint, "endpoint", execution["execution_fingerprint"])
    pointer = _call_qwen(
        pointer_prompt(groups, skill_instruction),
        groups,
        model_config,
        evidence_dir / "qwen" / "pointer.json",
        lambda value, rows: validate_frame_observations(value, rows, "pointer"),
        "pointer",
        execution["execution_fingerprint"],
    )
    reading = _call_qwen(
        reading_prompt(groups, skill_instruction),
        groups,
        model_config,
        evidence_dir / "qwen" / "reading.json",
        lambda value, rows: validate_frame_observations(value, rows, "reading"),
        "reading",
        execution["execution_fingerprint"],
    )
    result = reduce_result(
        endpoint,
        pointer,
        reading,
        groups,
        pointer_min_distinct_frames=int(parameters["pointer_min_distinct_frames"]),
        pointer_min_confidence=float(parameters["pointer_min_confidence"]),
    )
    report = {
        "schema_version": "resistance_agent_polarity_evidence.v1",
        "algorithm_version": ALGORITHM_VERSION,
        "video_id": video_id,
        "source_video_id": source_video_id,
        "source_video_path": str(video_path.resolve()),
        "source_video_sha256": source_digest,
        "selection_source": selection_source,
        "stage_manifest_path": str(stage_path.resolve()),
        "action_summary_path": str(action_path.resolve()),
        "groups": groups,
        "endpoint_observation": endpoint,
        "pointer_observation": pointer,
        "reading_sign_observation": reading,
        "rubric_4": result,
        "excel_accessed": False,
        "ground_truth_sent_to_model": False,
        "wiring_stage_accessed": False,
        "source_video_unchanged": sha256(video_path) == source_digest,
        "selection_checkpoint_reused": checkpoint_valid,
        "allow_video_calibration": allow_video_calibration,
        "fixed_video_roi_used": bool(allow_video_calibration),
        "historical_fallback_used": bool(allow_historical_fallback and action_path == fallback_action_summary_path),
        "routing_policy": (skill_plan or {}).get("routing_policy"),
        "skill_selection": (skill_plan or {}).get("skills", []),
        "skill_execution": execution,
    }
    report_path = evidence_dir / "polarity_evidence_report.json"
    write_json(report_path, report)
    return {"rubric_4": result, "report_path": str(report_path.resolve())}
