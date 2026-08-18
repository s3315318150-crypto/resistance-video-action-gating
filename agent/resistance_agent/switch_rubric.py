#!/usr/bin/env python3
"""Real-video evidence acquisition and binary reduction for Rubric 3."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import http.client
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ALGORITHM_VERSION = "r3_opencv_same_frame_overlap_v3"
WIRING_STAGES = {"circuit_wiring", "circuit_rewiring"}
BLADE_CONTACT_STATES = {"inside_jaws", "gap_visible", "unclear"}
WIRING_ACTIONS = {
    "terminal_connection_or_disconnection",
    "other_handling",
    "none",
    "unclear",
}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_json_object(content: str) -> dict[str, Any]:
    stripped = content.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.I | re.S)
    if fenced:
        stripped = fenced.group(1).strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(stripped[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("model response must be a JSON object")
    return value


def _source_record(
    summary: dict[str, Any],
    source_video_id: str,
    video_id: str,
    allowed_root: Path | None = None,
) -> dict[str, Any]:
    records = summary.get("records")
    if not isinstance(records, list):
        raise ValueError("action summary records are missing")
    for record in records:
        if not isinstance(record, dict):
            continue
        source = str(record.get("source_video_id") or "")
        if source != source_video_id and (
            allowed_root is not None or not source.startswith(f"{video_id}_")
        ):
            continue
        replay_result = record.get("replay_result")
        if allowed_root is not None and isinstance(replay_result, str) and replay_result:
            raise ValueError("replay_result is forbidden in live switch evidence")
        for key in ("replay_result", "result_path"):
            nested_path = record.get(key)
            if isinstance(nested_path, str) and nested_path:
                resolved = Path(nested_path).resolve()
                if allowed_root is not None and not resolved.is_relative_to(allowed_root.resolve()):
                    raise ValueError(f"live switch stage artifact is outside the current run: {resolved}")
                if not resolved.is_file():
                    continue
                nested = read_json(resolved)
                if _stage_runs(nested):
                    return nested
        return record
    raise ValueError(f"Temporal Guard record not found for video {video_id}")


def _boundary_record(
    summary: dict[str, Any],
    source_video_id: str,
    video_id: str,
    allowed_root: Path | None = None,
) -> dict[str, Any] | None:
    records = summary.get("records")
    if not isinstance(records, list):
        return None
    for record in records:
        if not isinstance(record, dict):
            continue
        source = str(record.get("source_video_id") or "")
        if source != source_video_id and (
            allowed_root is not None or not source.startswith(f"{video_id}_")
        ):
            continue
        nested_path = record.get("result_path")
        if isinstance(nested_path, str) and nested_path:
            resolved = Path(nested_path).resolve()
            if allowed_root is not None and not resolved.is_relative_to(allowed_root.resolve()):
                raise ValueError(f"live switch boundary artifact is outside the current run: {resolved}")
            if not resolved.is_file():
                continue
            nested = read_json(resolved)
            runs = nested.get("source_observed_stage_runs")
            if isinstance(runs, list):
                return {"observed_stage_runs": runs}
        runs = record.get("source_observed_stage_runs")
        if isinstance(runs, list):
            return {"observed_stage_runs": runs}
    return None


def _stage_runs(record: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("observed_stage_runs", "source_observed_stage_runs", "observed_stage_intervals"):
        raw = record.get(key)
        if isinstance(raw, list):
            return [item for item in raw if isinstance(item, dict)]
    return []


def candidate_windows(
    record: dict[str, Any],
    duration_seconds: float,
    window_mode: str = "all_wiring_runs",
) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    for index, item in enumerate(_stage_runs(record), start=1):
        stage = str(item.get("stage") or item.get("label") or "")
        if stage not in WIRING_STAGES or (
            window_mode == "initial_wiring_only" and stage != "circuit_wiring"
        ):
            continue
        try:
            start = max(0.0, min(float(item["start_seconds"]), duration_seconds))
            end = max(start, min(float(item["end_seconds"]), duration_seconds))
        except (KeyError, TypeError, ValueError):
            continue
        if end - start < 0.4:
            continue
        windows.append(
            {
                "window_id": f"{stage}_{index:03d}",
                "stage": stage,
                "start_seconds": round(start, 3),
                "end_seconds": round(end, 3),
                "source_event_ids": list(item.get("event_ids") or []),
                "source_confidence": item.get("confidence"),
            }
        )
    if window_mode == "initial_wiring_only" and windows:
        windows = windows[:1]
    if not windows and duration_seconds > 0:
        fallback_end = (
            duration_seconds
            if window_mode == "broad_search"
            else min(duration_seconds, max(12.0, duration_seconds * 0.45))
        )
        windows.append(
            {
                "window_id": "wiring_duration_fallback_001",
                "stage": "circuit_wiring",
                "start_seconds": 0.0,
                "end_seconds": round(fallback_end, 3),
                "source_event_ids": [],
                "source_confidence": None,
            }
        )
    return windows


def sampling_timestamps(
    windows: list[dict[str, Any]],
    sampling_fps: float | None = None,
    max_samples_per_window: int | None = None,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    seen: set[tuple[str, float]] = set()
    for window in windows:
        start = float(window["start_seconds"])
        end = float(window["end_seconds"])
        duration = end - start
        interval = (
            1.0 / max(float(sampling_fps), 0.01)
            if sampling_fps is not None
            else 1.0 if duration <= 12.0 else 2.0 if duration <= 40.0 else 4.0
        )
        count = max(2, int(math.floor(duration / interval)) + 1)
        values = [start + index * interval for index in range(count)]
        if not values or end - values[-1] > 0.25:
            values.append(end)
        if max_samples_per_window and len(values) > max_samples_per_window:
            indexes = {
                round(index * (len(values) - 1) / (max_samples_per_window - 1))
                for index in range(max_samples_per_window)
            }
            values = [values[index] for index in sorted(indexes)]
        for timestamp in values:
            timestamp = round(min(end, timestamp), 3)
            key = (str(window["window_id"]), timestamp)
            if key in seen:
                continue
            seen.add(key)
            samples.append(
                {
                    "window_id": window["window_id"],
                    "stage": window["stage"],
                    "timestamp_seconds": timestamp,
                    "window_start_seconds": start,
                    "window_end_seconds": end,
                    "evidence_phase": "coarse_scan",
                }
            )
    return samples


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 94]):
        raise RuntimeError(f"unable to write image: {path}")


def _enhance(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    light, a, b = cv2.split(lab)
    light = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(light)
    enhanced = cv2.cvtColor(cv2.merge((light, a, b)), cv2.COLOR_LAB2BGR)
    blurred = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
    return cv2.addWeighted(enhanced, 1.45, blurred, -0.45, 0)


def _orange_candidate_boxes(image: np.ndarray, limit: int = 4) -> list[list[int]]:
    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # A high saturation seed keeps hands and red leads from joining nearby equipment
    # into a single contour. The ROI still includes surrounding contacts after padding.
    mask = cv2.inRange(hsv, np.array([0, 130, 65]), np.array([28, 255, 255]))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    frame_area = float(height * width)
    candidates: list[tuple[float, list[int]]] = []
    for contour in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area < frame_area * 0.0004 or area > frame_area * 0.04 or w < 35 or h < 18:
            continue
        aspect = w / max(h, 1)
        if aspect < 1.1 or aspect > 6.0:
            continue
        pad_x = int(w * 0.30)
        pad_top = int(h * 0.65)
        pad_bottom = int(h * 0.35)
        left = max(0, x - pad_x)
        top = max(0, y - pad_top)
        right = min(width, x + w + pad_x)
        bottom = min(height, y + h + pad_bottom)
        expanded_area = (right - left) * (bottom - top)
        if expanded_area > frame_area * 0.15:
            continue
        orange_ratio = float(cv2.countNonZero(mask[y : y + h, x : x + w])) / max(area, 1)
        crop_hsv = hsv[top:bottom, left:right]
        crop_gray = gray[top:bottom, left:right]
        crop_area = max(crop_gray.size, 1)
        green_ratio = float(
            cv2.countNonZero(cv2.inRange(crop_hsv, np.array([30, 70, 35]), np.array([100, 255, 255])))
        ) / crop_area
        dark_ratio = float(np.count_nonzero(crop_gray < 75)) / crop_area
        horizontal = max(0.0, 1.0 - abs(aspect - 1.8) / 1.8)
        size_score = min(area / (frame_area * 0.012), 1.0)
        dark_structure = max(0.0, min(1.0, dark_ratio / 0.08, (0.34 - dark_ratio) / 0.16))
        green_penalty = min(green_ratio / 0.10, 1.0)
        score = (
            orange_ratio * 0.38
            + horizontal * 0.24
            + size_score * 0.10
            + dark_structure * 0.28
            - green_penalty * 0.25
        )
        candidates.append((score, [left, top, right, bottom]))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return [box for _, box in candidates[:limit]]


def _decode_and_export(
    video_path: Path,
    samples: list[dict[str, Any]],
    evidence_dir: Path,
) -> list[dict[str, Any]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"unable to open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        capture.release()
        raise RuntimeError("video FPS is invalid")
    output: list[dict[str, Any]] = []
    try:
        requested_by_frame: dict[int, tuple[int, dict[str, Any]]] = {}
        for original_index, sample in enumerate(samples, start=1):
            frame_number = int(round(float(sample["timestamp_seconds"]) * fps))
            requested_by_frame.setdefault(frame_number, (original_index, sample))
        requested = sorted(
            (frame_number, index, sample)
            for frame_number, (index, sample) in requested_by_frame.items()
        )
        if not requested:
            raise RuntimeError("no R3 sampling timestamps were supplied")
        capture.set(cv2.CAP_PROP_POS_FRAMES, requested[0][0])
        current_frame = requested[0][0]
        target_index = 0
        previous_gray_by_window: dict[str, np.ndarray] = {}
        while target_index < len(requested):
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            target_frame, index, sample = requested[target_index]
            if current_frame < target_frame:
                current_frame += 1
                continue
            if current_frame > target_frame:
                target_index += 1
                continue
            actual = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
            stem = f"frame_{index:04d}_{actual:09.3f}s"
            frame_path = evidence_dir / "frames" / f"{stem}.jpg"
            enhanced_path = evidence_dir / "frames_enhanced" / f"{stem}_enhanced.jpg"
            _write_image(frame_path, frame)
            _write_image(enhanced_path, _enhance(frame))
            full_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(full_gray, (320, 180), interpolation=cv2.INTER_AREA)
            window_id = str(sample["window_id"])
            previous_gray = previous_gray_by_window.get(window_id)
            motion = 0.0 if previous_gray is None else float(np.mean(cv2.absdiff(gray, previous_gray)))
            previous_gray_by_window[window_id] = gray
            roi_records: list[dict[str, Any]] = []
            for roi_index, box in enumerate(_orange_candidate_boxes(frame), start=1):
                left, top, right, bottom = box
                crop = frame[top:bottom, left:right]
                if crop.size == 0:
                    continue
                roi_path = evidence_dir / "switch_roi" / f"{stem}_candidate_{roi_index:02d}.jpg"
                roi_enhanced = evidence_dir / "switch_roi_enhanced" / f"{stem}_candidate_{roi_index:02d}_enhanced.jpg"
                _write_image(roi_path, crop)
                _write_image(roi_enhanced, _enhance(crop))
                roi_records.append(
                    {
                        "candidate_index": roi_index,
                        "box_xyxy": box,
                        "roi_path": str(roi_path.resolve()),
                        "enhanced_path": str(roi_enhanced.resolve()),
                    }
                )
            output.append(
                {
                    **sample,
                    "sample_index": index,
                    "timestamp_seconds": round(actual, 3),
                    "frame_number": target_frame,
                    "frame_path": str(frame_path.resolve()),
                    "enhanced_frame_path": str(enhanced_path.resolve()),
                    "sharpness": round(float(cv2.Laplacian(full_gray, cv2.CV_64F).var()), 3),
                    "motion_score": round(motion, 3),
                    "switch_candidates": roi_records,
                }
            )
            target_index += 1
            current_frame += 1
    finally:
        capture.release()
    if not output:
        raise RuntimeError("no R3 candidate frames were decoded")
    return output


def _even_indices(length: int, limit: int) -> list[int]:
    if length <= limit:
        return list(range(length))
    return sorted({int(round(value)) for value in np.linspace(0, length - 1, limit)})


def select_frames(records: list[dict[str, Any]], per_window_limit: int = 16) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    window_ids = list(dict.fromkeys(str(item["window_id"]) for item in records))
    for window_id in window_ids:
        group = [item for item in records if str(item["window_id"]) == window_id]
        backbone_limit = max(2, per_window_limit - 6)
        indexes = set(_even_indices(len(group), backbone_limit))
        for index in sorted(range(len(group)), key=lambda value: float(group[value]["motion_score"]), reverse=True)[:3]:
            indexes.add(index)
        for index in sorted(
            range(len(group)),
            key=lambda value: float(group[value].get("sharpness") or 0.0),
            reverse=True,
        )[:3]:
            indexes.add(index)
        if len(indexes) > per_window_limit:
            mandatory = set(_even_indices(len(group), backbone_limit))
            ranked = [
                index
                for index in sorted(
                    indexes,
                    key=lambda value: (
                        float(group[value]["motion_score"]),
                        float(group[value].get("sharpness") or 0.0),
                    ),
                    reverse=True,
                )
                if index not in mandatory
            ]
            indexes = mandatory | set(ranked[: max(0, per_window_limit - len(mandatory))])
        selected.extend(group[index] for index in sorted(indexes))
    for image_group, item in enumerate(selected, start=1):
        item["image_group"] = image_group
    return selected


def image_data_url(path: Path, max_edge: int, jpeg_quality: int = 88) -> str:
    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"unable to read model image: {path}")
    height, width = image.shape[:2]
    scale = min(1.0, float(max_edge) / max(height, width))
    if scale < 1.0:
        image = cv2.resize(
            image,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    if not ok:
        raise RuntimeError(f"unable to encode model image: {path}")
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def _prompt(
    frames: list[dict[str, Any]],
    dense_confirmation: bool = False,
    skill_instruction: str = "",
) -> str:
    groups = ", ".join(
        f"{item['image_group']}={item['timestamp_seconds']:.3f}s/{item['stage']}" for item in frames
    )
    confirmation_instruction = (
        "These are dense adjacent frames around a coarse possible contact event. Compare them as one temporal "
        "sequence and describe only the blade-to-fixed-contact geometry in each frame."
        if dense_confirmation
        else "Compare adjacent supplied frames when they show the same switch."
    )
    return f"""You are a visual observer for a school resistance-measurement experiment.
Inspect only visible evidence. The relevant switch is a separate orange knife-switch base with a metal blade
between two raised contacts. Do not confuse meters, battery holders, terminal blocks, or loose plugs with the switch.
 {confirmation_instruction} {skill_instruction}
For each image group, report the blade/contact geometry and whether the hands visibly connect or disconnect a wire at
any circuit terminal. First identify the pivot end where the blade is permanently hinged; then trace the long metal
blade to its opposite FREE TIP and judge only whether that free tip enters the fixed receiving jaws. The two upright
metal supports around the blade are not two contacts bridged by the blade. A blade can look horizontal or low while its
free tip remains outside the jaws; that is gap_visible, not inside_jaws. Do not infer hidden state from experiment order
or expected procedure. The local program,
not you, will assign the rubric decision. Groups: {groups}.

Return exactly one JSON object:
{{
  "observations": [
    {{
      "image_group": 1,
      "blade_contact": "inside_jaws|gap_visible|unclear",
      "wiring_action": "terminal_connection_or_disconnection|other_handling|none|unclear",
      "switch_visible": true,
      "confidence": 0.0,
      "evidence": "short direct visual observation"
    }}
  ]
}}
Include exactly one observation for every supplied image_group. Preserve the exact numeric image_group values listed
in Groups; do not renumber them from 1. Use inside_jaws only when the FREE blade tip is visibly inserted into the slot
between the receiving jaws. The black handle direction, a horizontal blade, or contact at the permanent pivot never
proves closure. Use gap_visible when the receiving slot is empty or any visible air gap separates it from the free tip.
Use unclear
when the relevant blade tip/contact is hidden or the crop is not the knife switch. Use
terminal_connection_or_disconnection only for visible insertion, removal, or manipulation of a wire/plug at a terminal."""


def validate_observation(value: dict[str, Any], expected_groups: set[int]) -> list[str]:
    errors: list[str] = []
    observations = value.get("observations")
    if not isinstance(observations, list):
        return ["observations_missing"]
    groups: list[int] = []
    for item in observations:
        if not isinstance(item, dict):
            errors.append("observation_not_object")
            continue
        group = item.get("image_group")
        if isinstance(group, bool) or not isinstance(group, int):
            errors.append("image_group_invalid")
        else:
            groups.append(group)
        if item.get("blade_contact") not in BLADE_CONTACT_STATES:
            errors.append("blade_contact_invalid")
        if item.get("wiring_action") not in WIRING_ACTIONS:
            errors.append("wiring_action_invalid")
        if not isinstance(item.get("switch_visible"), bool):
            errors.append("switch_visible_invalid")
        confidence = item.get("confidence")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= float(confidence) <= 1:
            errors.append("confidence_invalid")
        if not isinstance(item.get("evidence"), str):
            errors.append("evidence_invalid")
    if set(groups) != expected_groups or len(groups) != len(expected_groups):
        errors.append("image_groups_mismatch")
    return sorted(set(errors))


def _call_qwen_batch(
    frames: list[dict[str, Any]],
    model_config: dict[str, Any],
    raw_path: Path,
    dense_confirmation: bool = False,
    skill_instruction: str = "",
    execution_fingerprint: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base_url = os.getenv("QWEN_API_BASE_URL", str(model_config["base_url"]))
    token = os.getenv("QWEN_API_TOKEN", "EMPTY")
    model = os.getenv("QWEN_MODEL", str(model_config["model"]))
    source_groups = [int(item["image_group"]) for item in frames]
    if raw_path.is_file():
        try:
            cached = read_json(raw_path)
            mapping = cached.get("batch_group_mapping")
            observations = cached.get("observation", {}).get("observations")
            expected_mapping = {
                str(local_group): source_group
                for local_group, source_group in enumerate(source_groups, start=1)
            }
            source_to_local = {
                source_group: local_group
                for local_group, source_group in enumerate(source_groups, start=1)
            }
            localized = [
                {**item, "image_group": source_to_local.get(item.get("image_group"))}
                for item in observations or []
                if isinstance(item, dict)
            ]
            if (
                cached.get("algorithm_version") == ALGORITHM_VERSION
                and cached.get("execution_fingerprint") == execution_fingerprint
                and mapping == expected_mapping
                and cached.get("fallback_used") is not True
                and isinstance(observations, list)
                and not validate_observation(
                    {"observations": localized},
                    set(range(1, len(source_groups) + 1)),
                )
            ):
                return observations, cached
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    local_frames = [
        {**item, "image_group": local_group}
        for local_group, item in enumerate(frames, start=1)
    ]
    expected = set(range(1, len(local_frames) + 1))
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": _prompt(
                local_frames,
                dense_confirmation=dense_confirmation,
                skill_instruction=skill_instruction,
            ),
        }
    ]
    media: list[dict[str, Any]] = []
    for item, source_group in zip(local_frames, source_groups):
        group = int(item["image_group"])
        panorama = Path(item["enhanced_frame_path"])
        content.append({"type": "text", "text": f"Image group {group}: enhanced panorama."})
        content.append({"type": "image_url", "image_url": {"url": image_data_url(panorama, 1152)}})
        media.append(
            {
                "image_group": source_group,
                "local_image_group": group,
                "role": "panorama",
                "path": str(panorama),
            }
        )
        for candidate in item.get("switch_candidates", [])[:2]:
            roi = Path(candidate["enhanced_path"])
            content.append(
                {
                    "type": "text",
                    "text": f"Image group {group}: orange-object candidate crop; verify whether it is the knife switch.",
                }
            )
            content.append({"type": "image_url", "image_url": {"url": image_data_url(roi, 768)}})
            media.append(
                {
                    "image_group": source_group,
                    "local_image_group": group,
                    "role": "candidate_roi",
                    "path": str(roi),
                }
            )
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 1800,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    attempts: list[dict[str, Any]] = []
    for attempt in range(2):
        if attempt:
            payload["messages"][0]["content"].append(
                {
                    "type": "text",
                    "text": (
                        "Schema correction: return exactly one valid JSON object and one observation per supplied "
                        f"image_group. Use these exact IDs, without renumbering: {sorted(expected)}."
                    ),
                }
            )
        request = urllib.request.Request(
            base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            http.client.RemoteDisconnected,
            TimeoutError,
        ) as exc:
            attempts.append(
                {
                    "attempt": attempt + 1,
                    "errors": [f"transport_error:{type(exc).__name__}:{getattr(exc, 'code', '')}"],
                }
            )
            if attempt == 0:
                time.sleep(2.0)
                continue
            break
        choices = raw.get("choices") if isinstance(raw, dict) else None
        choice = choices[0] if isinstance(choices, list) and choices else None
        message = choice.get("message") if isinstance(choice, dict) else None
        text = message.get("content") if isinstance(message, dict) else None
        if not isinstance(text, str):
            errors = ["response_content_not_text"]
            parsed = None
        else:
            try:
                parsed = parse_json_object(text)
                errors = validate_observation(parsed, expected)
            except (ValueError, json.JSONDecodeError) as exc:
                parsed = None
                errors = [f"parse_error:{type(exc).__name__}"]
        attempts.append(
            {
                "attempt": attempt + 1,
                "finish_reason": choice.get("finish_reason") if isinstance(choice, dict) else None,
                "content": text,
                "parsed": parsed if not errors else None,
                "schema_errors": errors,
            }
        )
        if not errors and parsed is not None:
            remapped_observations = []
            for observation in parsed["observations"]:
                local_group = int(observation["image_group"])
                remapped_observations.append(
                    {
                        **observation,
                        "image_group": source_groups[local_group - 1],
                        "local_image_group": local_group,
                    }
                )
            artifact = {
                "algorithm_version": ALGORITHM_VERSION,
                "execution_fingerprint": execution_fingerprint,
                "model": model,
                "base_url": base_url,
                "batch_group_mapping": {
                    str(local_group): source_group
                    for local_group, source_group in enumerate(source_groups, start=1)
                },
                "media": media,
                "attempts": [
                    {
                        key: value
                        for key, value in attempt_record.items()
                        if key != "parsed"
                    }
                    for attempt_record in attempts
                ],
                "observation": {"observations": remapped_observations},
            }
            write_json(raw_path, artifact)
            return remapped_observations, artifact
    fallback = [
        {
            "image_group": source_group,
            "local_image_group": local_group,
            "blade_contact": "unclear",
            "wiring_action": "unclear",
            "switch_visible": False,
            "confidence": 0.2,
            "evidence": "Qwen batch unavailable after one targeted retry.",
        }
        for local_group, source_group in enumerate(source_groups, start=1)
    ]
    artifact = {
        "algorithm_version": ALGORITHM_VERSION,
        "execution_fingerprint": execution_fingerprint,
        "model": model,
        "base_url": base_url,
        "batch_group_mapping": {
            str(local_group): source_group
            for local_group, source_group in enumerate(source_groups, start=1)
        },
        "media": media,
        "attempts": attempts,
        "observation": {"observations": fallback},
        "fallback_used": True,
    }
    write_json(raw_path, artifact)
    return fallback, artifact


def call_qwen(
    frames: list[dict[str, Any]],
    model_config: dict[str, Any],
    evidence_dir: Path,
    batch_size: int = 4,
    dense_confirmation: bool = False,
    skill_instruction: str = "",
    execution_fingerprint: str | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    observations: list[dict[str, Any]] = []
    artifacts: list[str] = []
    for batch_index, start in enumerate(range(0, len(frames), batch_size), start=1):
        batch = frames[start : start + batch_size]
        raw_path = evidence_dir / "qwen_batches" / f"batch_{batch_index:03d}.json"
        values, _ = _call_qwen_batch(
            batch,
            model_config,
            raw_path,
            dense_confirmation=dense_confirmation,
            skill_instruction=skill_instruction,
            execution_fingerprint=execution_fingerprint,
        )
        observations.extend(values)
        artifacts.append(str(raw_path.resolve()))
    return observations, artifacts


def call_qwen_sequence(
    frames: list[dict[str, Any]],
    model_config: dict[str, Any],
    evidence_dir: Path,
    maximum_frames: int = 9,
    skill_instruction: str = "",
    execution_fingerprint: str | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Keep one trigger window in one model request so temporal geometry stays coherent."""
    observations: list[dict[str, Any]] = []
    artifacts: list[str] = []
    trigger_ids = list(dict.fromkeys(int(item["trigger_index"]) for item in frames))
    for trigger_id in trigger_ids:
        sequence = [item for item in frames if int(item["trigger_index"]) == trigger_id]
        for part_index, start in enumerate(range(0, len(sequence), maximum_frames), start=1):
            batch = sequence[start : start + maximum_frames]
            raw_path = evidence_dir / "qwen_batches" / f"trigger_{trigger_id:03d}_part_{part_index:02d}.json"
            values, _ = _call_qwen_batch(
                batch,
                model_config,
                raw_path,
                dense_confirmation=True,
                skill_instruction=skill_instruction,
                execution_fingerprint=execution_fingerprint,
            )
            observations.extend(values)
            artifacts.append(str(raw_path.resolve()))
    return observations, artifacts


def dense_confirmation_samples(
    observations: list[dict[str, Any]],
    frames: list[dict[str, Any]],
    windows: list[dict[str, Any]] | None = None,
    radius_seconds: float = 2.0,
    interval_seconds: float = 0.5,
) -> list[dict[str, Any]]:
    """Build stage-bounded adjacent-frame samples around coarse contact candidates."""
    by_group = {int(item["image_group"]): item for item in frames}
    by_window = {
        str(item["window_id"]): item
        for item in (windows or [])
        if isinstance(item, dict) and "window_id" in item
    }
    triggers: list[dict[str, Any]] = []
    for item in observations:
        group = item.get("image_group")
        frame = by_group.get(group) if isinstance(group, int) else None
        if (
            frame is None
            or item.get("blade_contact") != "inside_jaws"
            or float(item.get("confidence") or 0.0) < 0.55
        ):
            continue
        triggers.append(frame)

    samples: list[dict[str, Any]] = []
    seen: set[tuple[str, float]] = set()
    for trigger_index, frame in enumerate(triggers, start=1):
        center = float(frame["timestamp_seconds"])
        window = by_window.get(str(frame["window_id"]), {})
        window_start = float(
            window.get("start_seconds", frame.get("window_start_seconds", 0.0))
        )
        window_end = float(
            window.get("end_seconds", frame.get("window_end_seconds", center + radius_seconds))
        )
        lower = max(window_start, center - radius_seconds)
        upper = min(window_end, center + radius_seconds)
        count = int(math.floor((upper - lower) / interval_seconds)) + 1
        for sample_index in range(count):
            timestamp = round(lower + sample_index * interval_seconds, 3)
            key = (str(frame["window_id"]), timestamp)
            if key in seen:
                continue
            seen.add(key)
            samples.append(
                {
                    "window_id": frame["window_id"],
                    "stage": frame["stage"],
                    "timestamp_seconds": timestamp,
                    "evidence_phase": "dense_confirmation",
                    "trigger_image_group": frame["image_group"],
                    "trigger_index": trigger_index,
                }
            )
    return samples


def recurring_dense_confirmation_samples(
    windows: list[dict[str, Any]],
    interval_seconds: float = 4.0,
) -> list[dict[str, Any]]:
    """Add a deterministic second sampling phase independent of coarse VLM labels."""
    samples: list[dict[str, Any]] = []
    for trigger_index, window in enumerate(windows, start=1):
        start = float(window["start_seconds"])
        end = float(window["end_seconds"])
        timestamp = start
        while timestamp <= end + 1e-6:
            samples.append(
                {
                    "window_id": window["window_id"],
                    "stage": window["stage"],
                    "timestamp_seconds": round(min(timestamp, end), 3),
                    "evidence_phase": "dense_confirmation",
                    "trigger_image_group": None,
                    "trigger_index": trigger_index,
                    "trigger_source": "deterministic_wiring_scan",
                }
            )
            timestamp += interval_seconds
        if samples and samples[-1]["window_id"] == window["window_id"]:
            if end - float(samples[-1]["timestamp_seconds"]) > 0.25:
                samples.append(
                    {
                        "window_id": window["window_id"],
                        "stage": window["stage"],
                        "timestamp_seconds": round(end, 3),
                        "evidence_phase": "dense_confirmation",
                        "trigger_image_group": None,
                        "trigger_index": trigger_index,
                        "trigger_source": "deterministic_wiring_scan",
                    }
                )
    return samples


def select_dense_confirmation_frames(
    records: list[dict[str, Any]],
    targeted_per_window_limit: int = 24,
) -> list[dict[str, Any]]:
    deterministic = [
        item
        for item in records
        if item.get("trigger_source") == "deterministic_wiring_scan"
    ]
    targeted = [
        item
        for item in records
        if item.get("trigger_source") != "deterministic_wiring_scan"
    ]
    targeted_selected = (
        select_frames(targeted, per_window_limit=targeted_per_window_limit) if targeted else []
    )
    selected_by_time = {
        (str(item["window_id"]), float(item["timestamp_seconds"])): item
        for item in deterministic
    }
    for item in targeted_selected:
        selected_by_time[(str(item["window_id"]), float(item["timestamp_seconds"]))] = item
    return sorted(
        selected_by_time.values(),
        key=lambda item: (str(item["window_id"]), float(item["timestamp_seconds"])),
    )


def _confirmed_seated_sequences(
    observations: list[dict[str, Any]],
    maximum_gap_seconds: float = 1.01,
) -> list[list[dict[str, Any]]]:
    ordered_dense = sorted(
        (item for item in observations if item.get("evidence_phase") == "dense_confirmation"),
        key=lambda item: (str(item["window_id"]), float(item["timestamp_seconds"])),
    )
    sequences: list[list[dict[str, Any]]] = []
    by_window: dict[str, list[dict[str, Any]]] = {}
    for item in ordered_dense:
        by_window.setdefault(str(item["window_id"]), []).append(item)
    for window_items in by_window.values():
        for first, second in zip(window_items, window_items[1:]):
            delta = float(second["timestamp_seconds"]) - float(first["timestamp_seconds"])
            both_seated = all(
                item.get("blade_contact") == "inside_jaws"
                and item.get("switch_visible") is True
                and float(item.get("confidence") or 0.0) >= 0.55
                and item.get("model_fallback_used") is not True
                for item in (first, second)
            )
            if both_seated and delta <= maximum_gap_seconds:
                sequences.append([first, second])
    return sequences


def reduce_results(
    observations: list[dict[str, Any]],
    frames: list[dict[str, Any]],
    neighbor_seconds: float = 2.1,
) -> dict[str, Any]:
    by_group = {int(item["image_group"]): item for item in frames}
    normalized_by_time: dict[tuple[str, float], dict[str, Any]] = {}
    for item in observations:
        group = item.get("image_group")
        frame = by_group.get(group) if isinstance(group, int) else None
        if frame is None:
            continue
        normalized = {
            **item,
            "timestamp_seconds": frame["timestamp_seconds"],
            "window_id": frame["window_id"],
            "stage": frame["stage"],
            "evidence_phase": frame.get("evidence_phase", "synthetic_or_legacy"),
            "model_fallback_used": frame.get("model_fallback_used", False),
            "frame_path": frame["frame_path"],
            "enhanced_frame_path": frame["enhanced_frame_path"],
            "switch_candidates": frame.get("switch_candidates", []),
        }
        key = (str(normalized["window_id"]), round(float(normalized["timestamp_seconds"]), 3))
        existing = normalized_by_time.get(key)
        if existing is None or (
            normalized["evidence_phase"] == "dense_confirmation"
            and existing.get("evidence_phase") != "dense_confirmation"
        ):
            normalized_by_time[key] = normalized
    normalized = sorted(
        normalized_by_time.values(),
        key=lambda item: (str(item["window_id"]), float(item["timestamp_seconds"])),
    )
    seated_sequences = _confirmed_seated_sequences(normalized)
    confirmed_seated = [item for sequence in seated_sequences for item in sequence]
    actions = [
        item
        for item in normalized
        if item.get("wiring_action") == "terminal_connection_or_disconnection"
        and float(item.get("confidence") or 0.0) >= 0.55
    ]
    confirmed_counterexamples: list[dict[str, Any]] = []
    for seated_item in confirmed_seated:
        for action_item in actions:
            delta = abs(float(seated_item["timestamp_seconds"]) - float(action_item["timestamp_seconds"]))
            if seated_item["window_id"] != action_item["window_id"] or delta > neighbor_seconds:
                continue
            confirmed_counterexamples.append(
                {
                    "seated_observation": seated_item,
                    "wiring_observation": action_item,
                    "delta_seconds": round(delta, 3),
                }
            )
    confirmed_keys = {
        (str(item["window_id"]), round(float(item["timestamp_seconds"]), 3))
        for item in confirmed_seated
    }
    suppressed_isolated_counterexamples = [
        item
        for item in normalized
        if item.get("blade_contact") == "inside_jaws"
        and (str(item["window_id"]), round(float(item["timestamp_seconds"]), 3)) not in confirmed_keys
        and any(
            action["window_id"] == item["window_id"]
            and abs(float(action["timestamp_seconds"]) - float(item["timestamp_seconds"])) <= neighbor_seconds
            for action in actions
        )
    ]
    if confirmed_counterexamples:
        supporting_confidences = [
            min(
                float(item["seated_observation"].get("confidence") or 0.0),
                float(item["wiring_observation"].get("confidence") or 0.0),
            )
            for item in confirmed_counterexamples
        ]
        confidence = round(max(supporting_confidences or [0.55]), 3)
        decision = "fail"
        reason = "confirmed_seated_blade_overlaps_terminal_wiring"
    else:
        action_items = [item for item in normalized if item.get("wiring_action") == "terminal_connection_or_disconnection"]
        separated_actions = [item for item in action_items if item.get("blade_contact") == "gap_visible"]
        visible = [item for item in normalized if item.get("switch_visible")]
        observation_ratio = len(visible) / max(len(normalized), 1)
        separated_ratio = len(separated_actions) / max(len(action_items), 1) if action_items else 0.0
        mean_confidence = (
            sum(float(item.get("confidence") or 0.0) for item in normalized) / max(len(normalized), 1)
        )
        confidence = round(
            max(
                0.51,
                min(
                    0.95,
                    0.35 + 0.25 * observation_ratio + 0.25 * separated_ratio + 0.15 * mean_confidence,
                ),
            ),
            3,
        )
        decision = "pass"
        reason = "no_confirmed_seated_blade_during_terminal_wiring"
    return {
        "decision": decision,
        "predicted_score": 1 if decision == "pass" else 0,
        "confidence": confidence,
        "reason": reason,
        "diagnostics": {
            "observation_count": len(normalized),
            "terminal_wiring_observation_count": len(actions),
            "seated_observation_count": sum(
                1 for item in normalized if item.get("blade_contact") == "inside_jaws"
            ),
            "confirmed_seated_sequence_count": len(seated_sequences),
            "confirmed_seated_sequences": seated_sequences,
            "confirmed_counterexamples": confirmed_counterexamples,
            "suppressed_isolated_counterexamples": suppressed_isolated_counterexamples,
            "counterexamples": confirmed_counterexamples,
            "observations": normalized,
            "tie_break": "fail_only_on_temporally_confirmed_seated_blade_near_terminal_wiring;otherwise_pass",
        },
    }


def _run_switch_rubric_legacy_vlm(
    video_path: Path,
    source_video_id: str,
    video_id: str,
    run_dir: Path,
    model_config: dict[str, Any],
    action_summary_path: Path | None = None,
    boundary_summary_path: Path | None = None,
    fallback_action_summary_path: Path | None = None,
    allow_historical_fallback: bool = False,
    skill_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        from .skills import EXECUTOR_REGISTRY, execution_for_rubric
    except ImportError:
        from skills import EXECUTOR_REGISTRY, execution_for_rubric  # type: ignore
    execution = (
        execution_for_rubric(skill_plan, 3)
        if skill_plan
        else {
            "skill_id": "switch.multi_stage_dense",
            "parameters": dict(EXECUTOR_REGISTRY["switch.multi_stage_dense"].defaults),
            "execution_fingerprint": None,
        }
    )
    parameters = execution["parameters"]
    fallback = fallback_action_summary_path
    action_path = action_summary_path if action_summary_path and action_summary_path.is_file() else (
        fallback if allow_historical_fallback and fallback is not None else None
    )
    if action_path is None or not action_path.is_file():
        raise ValueError("current live action summary is required")
    record = _source_record(read_json(action_path), source_video_id, video_id)
    boundary_record = None
    if boundary_summary_path and boundary_summary_path.is_file():
        boundary_record = _boundary_record(read_json(boundary_summary_path), source_video_id, video_id)
        if boundary_record is not None:
            record = boundary_record
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"unable to open video: {video_path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    duration = frame_count / fps if fps > 0 else 0.0
    evidence_dir = run_dir / "switch_rubric"
    source_digest = sha256(video_path)
    checkpoint_path = evidence_dir / "selected_frames_pre_qwen.json"
    checkpoint = read_json(checkpoint_path) if checkpoint_path.is_file() else {}
    selected = checkpoint.get("selected_frames") if isinstance(checkpoint, dict) else None
    checkpoint_valid = (
        checkpoint.get("algorithm_version") == ALGORITHM_VERSION
        and checkpoint.get("source_video_sha256") == source_digest
        and checkpoint.get("routing_policy") == (skill_plan or {}).get("routing_policy")
        and checkpoint.get("execution_fingerprint") == execution["execution_fingerprint"]
        and isinstance(selected, list)
        and bool(selected)
        and all(
            isinstance(item, dict)
            and isinstance(item.get("frame_path"), str)
            and Path(item["frame_path"]).is_file()
            for item in selected
        )
    )
    if checkpoint_valid:
        windows = list(
            checkpoint.get("candidate_windows")
            or candidate_windows(record, duration, str(parameters["window_mode"]))
        )
        sample_count = int(checkpoint.get("sample_count") or len(selected))
    else:
        windows = candidate_windows(record, duration, str(parameters["window_mode"]))
        decoded = _decode_and_export(
            video_path,
            sampling_timestamps(
                windows,
                sampling_fps=float(parameters["sampling_fps"]),
                max_samples_per_window=int(parameters["max_samples_per_window"]),
            ),
            evidence_dir,
        )
        selected = select_frames(decoded, per_window_limit=int(parameters["per_window_frame_limit"]))
        sample_count = len(decoded)
        write_json(
            checkpoint_path,
            {
                "algorithm_version": ALGORITHM_VERSION,
                "source_video_sha256": source_digest,
                "routing_policy": (skill_plan or {}).get("routing_policy"),
                "execution_fingerprint": execution["execution_fingerprint"],
                "skill_execution": execution,
                "candidate_windows": windows,
                "sample_count": sample_count,
                "selected_frames": selected,
            },
        )
    coarse_observations, coarse_qwen_artifacts = call_qwen(
        selected,
        model_config,
        evidence_dir,
        skill_instruction=str(parameters["prompt_instruction"]),
        execution_fingerprint=execution["execution_fingerprint"],
    )
    targeted_dense_samples = dense_confirmation_samples(
        coarse_observations,
        selected,
        windows=windows,
        interval_seconds=1.0 / max(float(parameters["dense_sampling_fps"]), 0.01),
    )
    deterministic_dense_samples = recurring_dense_confirmation_samples(
        windows,
        interval_seconds=float(parameters["deterministic_scan_interval_seconds"]),
    )
    if not parameters["dense_confirmation"]:
        targeted_dense_samples = []
        deterministic_dense_samples = []
    dense_by_time: dict[tuple[str, float], dict[str, Any]] = {
        (str(item["window_id"]), float(item["timestamp_seconds"])): item
        for item in deterministic_dense_samples
    }
    for item in targeted_dense_samples:
        dense_by_time[(str(item["window_id"]), float(item["timestamp_seconds"]))] = item
    dense_samples = sorted(
        dense_by_time.values(),
        key=lambda item: (str(item["window_id"]), float(item["timestamp_seconds"])),
    )
    dense_selected: list[dict[str, Any]] = []
    dense_observations: list[dict[str, Any]] = []
    dense_qwen_artifacts: list[str] = []
    refinement_samples: list[dict[str, Any]] = []
    refinement_selected: list[dict[str, Any]] = []
    refinement_observations: list[dict[str, Any]] = []
    refinement_qwen_artifacts: list[str] = []
    if dense_samples:
        dense_dir = evidence_dir / "dense_confirmation"
        dense_decoded = _decode_and_export(video_path, dense_samples, dense_dir)
        dense_selected = select_dense_confirmation_frames(dense_decoded)
        for image_group, item in enumerate(dense_selected, start=len(selected) + 1):
            item["image_group"] = image_group
        dense_observations, dense_qwen_artifacts = call_qwen_sequence(
            dense_selected,
            model_config,
            dense_dir,
            skill_instruction=str(parameters["prompt_instruction"]),
            execution_fingerprint=execution["execution_fingerprint"],
        )
        fallback_groups: set[int] = set()
        for artifact_path in dense_qwen_artifacts:
            artifact = read_json(Path(artifact_path))
            if artifact.get("fallback_used") is True:
                fallback_groups.update(
                    int(item["image_group"])
                    for item in artifact.get("media", [])
                    if isinstance(item, dict) and isinstance(item.get("image_group"), int)
                )
        for item in dense_selected:
            item["model_fallback_used"] = int(item["image_group"]) in fallback_groups
        refinement_samples = dense_confirmation_samples(
            dense_observations,
            dense_selected,
            windows=windows,
            interval_seconds=1.0 / max(float(parameters["dense_sampling_fps"]), 0.01),
        )
        for item in refinement_samples:
            item["trigger_source"] = "dense_observation_refinement"
        existing_times = {
            (str(item["window_id"]), float(item["timestamp_seconds"]))
            for item in dense_selected
        }
        refinement_samples = [
            item
            for item in refinement_samples
            if (str(item["window_id"]), float(item["timestamp_seconds"])) not in existing_times
        ]
        if refinement_samples:
            refinement_dir = evidence_dir / "dense_refinement"
            refinement_selected = _decode_and_export(
                video_path,
                refinement_samples,
                refinement_dir,
            )
            for image_group, item in enumerate(
                refinement_selected,
                start=len(selected) + len(dense_selected) + 1,
            ):
                item["image_group"] = image_group
            refinement_observations, refinement_qwen_artifacts = call_qwen_sequence(
                refinement_selected,
                model_config,
                refinement_dir,
                skill_instruction=str(parameters["prompt_instruction"]),
                execution_fingerprint=execution["execution_fingerprint"],
            )
            refinement_fallback_groups: set[int] = set()
            for artifact_path in refinement_qwen_artifacts:
                artifact = read_json(Path(artifact_path))
                if artifact.get("fallback_used") is True:
                    refinement_fallback_groups.update(
                        int(item["image_group"])
                        for item in artifact.get("media", [])
                        if isinstance(item, dict) and isinstance(item.get("image_group"), int)
                    )
            for item in refinement_selected:
                item["model_fallback_used"] = (
                    int(item["image_group"]) in refinement_fallback_groups
                )
    all_frames = selected + dense_selected + refinement_selected
    observations = coarse_observations + dense_observations + refinement_observations
    rubric_3 = reduce_results(
        observations,
        all_frames,
        neighbor_seconds=float(parameters["neighbor_seconds"]),
    )
    report = {
        "schema_version": "resistance_agent_switch_evidence.v1",
        "algorithm_version": ALGORITHM_VERSION,
        "video_id": video_id,
        "source_video_id": source_video_id,
        "source_video_path": str(video_path.resolve()),
        "source_video_sha256": source_digest,
        "action_summary_path": str(action_path.resolve()),
        "boundary_summary_path": str(boundary_summary_path.resolve()) if boundary_summary_path else None,
        "boundary_stage_runs_used": boundary_record is not None,
        "candidate_windows": windows,
        "sample_count": sample_count,
        "selected_frame_count": len(selected),
        "selected_frames": selected,
        "coarse_qwen_batch_artifacts": coarse_qwen_artifacts,
        "dense_confirmation_trigger_count": len(
            {int(item["trigger_index"]) for item in dense_samples}
        ),
        "targeted_dense_sample_count": len(targeted_dense_samples),
        "deterministic_dense_sample_count": len(deterministic_dense_samples),
        "dense_confirmation_sample_count": len(dense_samples),
        "dense_confirmation_frames": dense_selected,
        "dense_qwen_batch_artifacts": dense_qwen_artifacts,
        "dense_refinement_sample_count": len(refinement_samples),
        "dense_refinement_frames": refinement_selected,
        "dense_refinement_qwen_batch_artifacts": refinement_qwen_artifacts,
        "qwen_batch_artifacts": (
            coarse_qwen_artifacts + dense_qwen_artifacts + refinement_qwen_artifacts
        ),
        "coarse_qwen_observations": coarse_observations,
        "dense_qwen_observations": dense_observations,
        "dense_refinement_qwen_observations": refinement_observations,
        "qwen_observations": observations,
        "rubric_3": rubric_3,
        "human_review_used": False,
        "excel_accessed": False,
        "ground_truth_sent_to_model": False,
        "selection_checkpoint_reused": checkpoint_valid,
        "routing_policy": (skill_plan or {}).get("routing_policy"),
        "skill_selection": (skill_plan or {}).get("skills", []),
        "skill_execution": execution,
        "historical_fallback_used": bool(
            allow_historical_fallback and fallback is not None and action_path == fallback
        ),
    }
    report_path = evidence_dir / "switch_evidence_report.json"
    write_json(report_path, report)
    return {"rubric_3": rubric_3, "report_path": str(report_path.resolve())}


def run_switch_rubric(
    video_path: Path,
    source_video_id: str,
    video_id: str,
    run_dir: Path,
    model_config: dict[str, Any],
    action_summary_path: Path | None = None,
    boundary_summary_path: Path | None = None,
    fallback_action_summary_path: Path | None = None,
    allow_historical_fallback: bool = False,
    skill_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate R3 using only same-frame OpenCV state and plug evidence."""
    del model_config
    try:
        from .opencv_switch_overlap import analyze_opencv_switch_overlap
        from .skills import EXECUTOR_REGISTRY, execution_for_rubric
    except ImportError:
        from opencv_switch_overlap import analyze_opencv_switch_overlap  # type: ignore
        from skills import EXECUTOR_REGISTRY, execution_for_rubric  # type: ignore

    execution = (
        execution_for_rubric(skill_plan, 3)
        if skill_plan
        else {
            "skill_id": "switch.multi_stage_dense",
            "parameters": dict(EXECUTOR_REGISTRY["switch.multi_stage_dense"].defaults),
            "execution_fingerprint": None,
        }
    )
    parameters = execution["parameters"]
    fallback = fallback_action_summary_path
    action_path = (
        action_summary_path
        if action_summary_path and action_summary_path.is_file()
        else (fallback if allow_historical_fallback and fallback is not None else None)
    )
    if action_path is None or not action_path.is_file():
        raise ValueError("current live action summary is required")
    record = _source_record(
        read_json(action_path), source_video_id, video_id, allowed_root=run_dir
    )
    boundary_record = None
    if boundary_summary_path and boundary_summary_path.is_file():
        boundary_record = _boundary_record(
            read_json(boundary_summary_path),
            source_video_id,
            video_id,
            allowed_root=run_dir,
        )
        if boundary_record is not None:
            record = boundary_record
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"unable to open video: {video_path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    duration = frame_count / fps if fps > 0 else 0.0
    windows = candidate_windows(record, duration, str(parameters["window_mode"]))
    evidence_dir = run_dir / "switch_rubric"
    opencv = analyze_opencv_switch_overlap(
        video_path=video_path,
        candidate_windows=windows,
        output_dir=evidence_dir / "opencv_same_frame_overlap",
        sampling_fps=float(parameters["sampling_fps"]),
        roi_mode=str(parameters["roi_mode"]),
        fusion_policy=str(parameters["fusion_policy"]),
    )
    diagnostics = {
        "algorithm_version": ALGORITHM_VERSION,
        "implementation_version": opencv.get(
            "implementation_version", "r3_opencv_same_frame_overlap_v3"
        ),
        "implementation_fingerprint": opencv.get("implementation_fingerprint"),
        "decision_source": opencv["decision_source"],
        "sampling_fps": opencv["sampling_fps"],
        "roi_mode": opencv["roi_mode"],
        "fusion_policy": opencv["fusion_policy"],
        "execution_fingerprint": execution["execution_fingerprint"],
        "same_frame_overlap_count": opencv["same_frame_overlap_count"],
        "same_frame_overlaps": opencv["same_frame_overlaps"],
        "real_plug_transition_count": opencv["real_plug_transition_count"],
        "real_plug_transitions": opencv["real_plug_transitions"],
        "wiring_active_frame_count": opencv.get("wiring_active_frame_count", 0),
        "wiring_active_interval_count": opencv.get("wiring_active_interval_count", 0),
        "switch_tracked_observation_count": opencv[
            "switch_tracked_observation_count"
        ],
        "switch_state_cluster_centers": opencv["switch_state_cluster_centers"],
        "switch_state_threshold": opencv["switch_state_threshold"],
        "frames": opencv["frames"],
        "switch_min_closed_persistence_observations": opencv.get(
            "switch_min_closed_persistence_observations", 3
        ),
        "switch_persistent_closed_observation_count": opencv.get(
            "switch_persistent_closed_observation_count", 0
        ),
        "tie_break": "fail_only_when_at_least_three_temporally_contiguous_closed_observations_support_closed_and_wiring_active_is_true_on_the_same_real_5fps_frame;otherwise_pass",
    }
    rubric_3 = {
        "decision": opencv["decision"],
        "predicted_score": opencv["predicted_score"],
        "confidence": opencv["confidence"],
        "reason": opencv["reason"],
        "diagnostics": diagnostics,
    }
    report = {
        "schema_version": "resistance_agent_switch_evidence.v3",
        "algorithm_version": ALGORITHM_VERSION,
        "implementation_version": opencv.get(
            "implementation_version", "r3_opencv_same_frame_overlap_v3"
        ),
        "implementation_fingerprint": opencv.get("implementation_fingerprint"),
        "video_id": video_id,
        "source_video_id": source_video_id,
        "source_video_path": str(video_path.resolve()),
        "source_video_sha256": sha256(video_path),
        "action_summary_path": str(action_path.resolve()),
        "boundary_summary_path": (
            str(boundary_summary_path.resolve()) if boundary_summary_path else None
        ),
        "boundary_stage_runs_used": boundary_record is not None,
        "candidate_windows": windows,
        "sample_count": opencv["sample_count"],
        "selected_frame_count": opencv["switch_tracked_observation_count"],
        "dense_confirmation_sample_count": 0,
        "dense_refinement_sample_count": 0,
        "qwen_batch_artifacts": [],
        "qwen_observations": [],
        "qwen_used_for_decision": False,
        "opencv_report_path": opencv["report_path"],
        "opencv_switch_state_method": opencv["switch_state_method"],
        "opencv_plug_motion_method": opencv["plug_motion_method"],
        "rubric_3": rubric_3,
        "human_review_used": False,
        "excel_accessed": False,
        "ground_truth_sent_to_model": False,
        "selection_checkpoint_reused": False,
        "routing_policy": (skill_plan or {}).get("routing_policy"),
        "skill_selection": (skill_plan or {}).get("skills", []),
        "skill_execution": {
            **execution,
            "implementation_version": execution.get(
                "implementation_version", "r3_opencv_same_frame_overlap_v3"
            ),
            "effective_parameters": {
                "sampling_fps": float(parameters["sampling_fps"]),
                "roi_mode": str(parameters["roi_mode"]),
                "fusion_policy": str(parameters["fusion_policy"]),
            },
        },
        "historical_fallback_used": bool(
            allow_historical_fallback and fallback is not None and action_path == fallback
        ),
        "fixed_video_roi_used": False,
        "video_id_used_for_routing": False,
    }
    report_path = evidence_dir / "switch_evidence_report.json"
    write_json(report_path, report)
    reopened = read_json(report_path)
    if (
        reopened.get("rubric_3", {}).get("decision") != rubric_3["decision"]
        or reopened.get("qwen_used_for_decision") is not False
    ):
        raise RuntimeError("OpenCV switch evidence report failed reopen verification")
    return {"rubric_3": rubric_3, "report_path": str(report_path.resolve())}


run_switch_rubric.supports_boundary_summary = True
