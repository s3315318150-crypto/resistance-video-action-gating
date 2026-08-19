#!/usr/bin/env python3
"""Real-video evidence acquisition for Rubrics 0, 2, 4, and 8."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ALGORITHM_VERSION = "r02_temporal_visual_r8_live_cv_coarse_dense_v4"
CLEANUP_REVIEW_VERSION = "r0_pairwise_workspace_review_v1"
CLEANUP_CENTRAL_ROI = (0.18, 0.38, 0.86, 0.96)
DEFAULT_R8_SCRIPT = PROJECT_ROOT / "agent" / "scripts" / "run_rubric8_specialized.py"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_json_object(content: str) -> dict[str, Any]:
    text = content.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S | re.I)
    if fenced:
        text = fenced.group(1)
    elif not text.startswith("{"):
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("response does not contain a JSON object")
        text = text[start : end + 1]
    try:
        value = json.loads(text)
    except json.JSONDecodeError as json_error:
        try:
            import yaml

            value = yaml.safe_load(text)
        except Exception:
            raise json_error
    if not isinstance(value, dict):
        raise ValueError("response JSON root is not an object")
    return value


def response_token_budget(image_group_count: int) -> int:
    return max(2400, min(7000, 1600 + max(0, image_group_count) * 700))


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


def _boundary_record(summary: dict[str, Any], source_video_id: str, video_id: str) -> dict[str, Any] | None:
    records = summary.get("records")
    if not isinstance(records, list):
        return None
    for record in records:
        if not isinstance(record, dict):
            continue
        source = str(record.get("source_video_id") or "")
        if source != source_video_id and not source.startswith(f"{video_id}_"):
            continue
        runs = record.get("source_observed_stage_runs") or record.get("observed_stage_runs")
        if isinstance(runs, list):
            return {"observed_stage_runs": runs}
        nested = record.get("result_path")
        if isinstance(nested, str) and Path(nested).is_file():
            document = read_json(Path(nested))
            runs = document.get("source_observed_stage_runs") or document.get("observed_stage_runs")
            if isinstance(runs, list):
                return {"observed_stage_runs": runs}
    return None


def _stage_runs(record: dict[str, Any]) -> list[dict[str, Any]]:
    raw = record.get("observed_stage_runs") or record.get("source_observed_stage_runs")
    return sorted(
        [item for item in raw if isinstance(item, dict)] if isinstance(raw, list) else [],
        key=lambda item: float(item.get("start_seconds") or 0.0),
    )


def _is_merged_measurement_recording(run: dict[str, Any]) -> bool:
    return str(run.get("stage")) in {"recording_1", "recording_2"} and (
        run.get("stage_semantics") == "measurement_and_recording_cycle"
        or run.get("stage_window_semantics") == "measurement_and_recording_cycle"
        or run.get("merged_stage_semantics") == "measurement_and_recording_cycle"
        or run.get("merged_measurement_recording") is True
        or run.get("merged_stage") is True
    )


def _subaction_intervals(run: dict[str, Any], action_type: str) -> list[tuple[float, float]]:
    field = "measurement_subintervals" if action_type == "measurement_action" else "writing_subintervals"
    raw = run.get(field)
    explicit_field = isinstance(raw, list)
    if not explicit_field:
        raw = run.get("observed_subintervals")
    intervals: list[tuple[float, float]] = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        if not explicit_field and item.get("action_type") != action_type:
            continue
        try:
            start = float(item["start_seconds"])
            end = float(item["end_seconds"])
        except (KeyError, TypeError, ValueError):
            continue
        if start < end:
            intervals.append((start, end))
    return intervals


def _linspace(start: float, end: float, count: int) -> list[float]:
    if end <= start + 1e-6:
        return [round(start, 3)]
    return sorted({round(float(value), 3) for value in np.linspace(start, end, count)})


def candidate_times(
    record: dict[str, Any],
    duration: float,
    skill_parameters: dict[int, dict[str, Any]] | None = None,
) -> dict[int, list[dict[str, Any]]]:
    runs = _stage_runs(record)
    decodable_end = max(0.0, duration - 0.1)

    def rows_for(
        stages: set[str],
        count: int,
        margin_before: float = 0.0,
        margin_after: float = 0.0,
        subaction_type: str | None = None,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for index, run in enumerate(runs, start=1):
            stage = str(run.get("stage") or "")
            if stage not in stages:
                continue
            intervals = (
                _subaction_intervals(run, subaction_type)
                if subaction_type and _is_merged_measurement_recording(run)
                else []
            )
            if not intervals:
                intervals = [
                    (
                        float(run.get("start_seconds") or 0.0),
                        float(run.get("end_seconds") or run.get("start_seconds") or 0.0),
                    )
                ]
            for interval_index, (raw_start, raw_end) in enumerate(intervals, start=1):
                start = max(0.0, raw_start - margin_before)
                end = min(decodable_end, raw_end + margin_after)
                for timestamp in _linspace(start, end, count):
                    rows.append(
                        {
                            "stage": stage,
                            "stage_run": index,
                            "timestamp_seconds": timestamp,
                            "subaction_type": subaction_type if len(intervals) > 0 and subaction_type else None,
                            "subinterval_index": interval_index if subaction_type else None,
                        }
                    )
        return rows

    parameters = skill_parameters or {}
    p0 = parameters.get(0, {})
    p2 = parameters.get(2, {})
    p8 = parameters.get(8, {})
    cleanup_count = int(p0.get("sample_count", 5))
    cleanup = (
        []
        if p0.get("time_mode") == "video_tail"
        else rows_for({"material_cleanup"}, cleanup_count, 2.0, 1.0)
    )
    if not cleanup:
        cleanup = [
            {"stage": "terminal_scan_fallback", "stage_run": 0, "timestamp_seconds": value}
            for value in _linspace(
                max(0.0, decodable_end - 20.0), decodable_end, cleanup_count
            )
        ]
    stable_count = int(p2.get("sample_count", 6))
    stable_stages = (
        {"recording_1", "recording_2"}
        if p2.get("time_mode") == "recording_context"
        else {"measurement_1", "measurement_2"}
    )
    stable = rows_for(
        stable_stages,
        stable_count,
        0.0 if stable_stages == {"recording_1", "recording_2"} else 2.0,
        0.0,
        "measurement_action" if stable_stages == {"recording_1", "recording_2"} else None,
    )
    if not stable and stable_stages == {"measurement_1", "measurement_2"}:
        stable = rows_for(
            {"recording_1", "recording_2"},
            stable_count,
            0.0,
            0.0,
            "measurement_action",
        )
    if not stable:
        stable = rows_for({"circuit_wiring", "circuit_rewiring"}, stable_count, 0.0, 3.0)
    rewiring_count = int(p8.get("sample_count", 9))
    time_mode = p8.get("time_mode")
    rewiring_stages = (
        {"circuit_wiring", "circuit_rewiring"}
        if time_mode == "broad_transition_search"
        else {"circuit_wiring"}
        if time_mode == "wiring_transition"
        else {"circuit_rewiring"}
    )
    rewiring = rows_for(rewiring_stages, rewiring_count, 4.0, 4.0)
    if not rewiring:
        rewiring = rows_for({"circuit_wiring"}, rewiring_count, 0.0, 2.0)
    if not rewiring and time_mode == "broad_transition_search":
        rewiring = [
            {"stage": "broad_transition_search", "stage_run": 0, "timestamp_seconds": value}
            for value in _linspace(0.0, decodable_end, rewiring_count)
        ]
    return {0: cleanup, 2: stable, 4: stable, 8: rewiring}


def _enhance(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    light, a, b = cv2.split(lab)
    light = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(light)
    return cv2.cvtColor(cv2.merge((light, a, b)), cv2.COLOR_LAB2BGR)


def _write_jpeg(path: Path, image: np.ndarray, quality: int = 93) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image, [int(cv2.IMWRITE_JPEG_QUALITY), quality]):
        raise RuntimeError(f"unable to write evidence image: {path}")


def _sample_times(start: float, end: float, sample_fps: float) -> list[float]:
    if end < start or sample_fps <= 0:
        return []
    count = max(1, int(np.floor((end - start) * sample_fps)) + 1)
    values = [round(start + index / sample_fps, 3) for index in range(count)]
    if not values or end - values[-1] > 0.05:
        values.append(round(end, 3))
    return values


def _box_iou(left: list[float] | None, right: list[float] | None) -> float:
    if left is None or right is None:
        return 0.0
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return intersection / max(left_area + right_area - intersection, 1e-9)


def dynamic_battery_candidate_boxes(frame: np.ndarray, limit: int = 3) -> list[list[float]]:
    """Find orange/red apparatus candidates without video-specific coordinates."""
    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    warm = cv2.inRange(hsv, np.array([0, 75, 45]), np.array([35, 255, 255]))
    warm = cv2.morphologyEx(
        warm,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (11, 7)),
        iterations=2,
    )
    warm = cv2.morphologyEx(warm, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    grouped = cv2.morphologyEx(
        warm,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (max(15, width // 55), max(7, height // 100)),
        ),
        iterations=2,
    )
    frame_area = float(height * width)
    ranked: list[tuple[float, list[float]]] = []
    cell_mask = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([34, 75, 40]), np.array([105, 255, 255])),
        cv2.inRange(hsv, np.array([16, 85, 55]), np.array([38, 255, 255])),
    )
    cell_mask = cv2.morphologyEx(
        cell_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5)),
        iterations=2,
    )
    cell_seeds: list[list[int]] = []
    for contour in cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        x, y, w, h = cv2.boundingRect(contour)
        area = float(w * h)
        aspect = w / max(h, 1)
        if frame_area * 0.00025 <= area <= frame_area * 0.035 and 0.8 <= aspect <= 6.0:
            cell_seeds.append([x, y, x + w, y + h])
    for left_index, left_seed in enumerate(cell_seeds):
        for right_seed in cell_seeds[left_index + 1 :]:
            first, second = sorted((left_seed, right_seed), key=lambda box: box[0])
            center_delta = abs((first[1] + first[3]) - (second[1] + second[3])) / (2.0 * height)
            horizontal_gap = (second[0] - first[2]) / width
            if center_delta > 0.10 or not -0.04 <= horizontal_gap <= 0.24:
                continue
            x1, y1 = min(first[0], second[0]), min(first[1], second[1])
            x2, y2 = max(first[2], second[2]), max(first[3], second[3])
            pair_width, pair_height = x2 - x1, y2 - y1
            if pair_width < width * 0.08:
                continue
            candidate_box = [
                max(0, x1 - int(pair_width * 0.18)),
                max(0, y1 - int(pair_height * 0.70)),
                min(width, x2 + int(pair_width * 0.18)),
                min(height, y2 + int(pair_height * 0.70)),
            ]
            cx1, cy1, cx2, cy2 = candidate_box
            candidate_warm = float(np.mean(warm[cy1:cy2, cx1:cx2] > 0))
            candidate_gray = cv2.cvtColor(frame[cy1:cy2, cx1:cx2], cv2.COLOR_BGR2GRAY)
            candidate_edges = float(np.mean(cv2.Canny(candidate_gray, 60, 160) > 0))
            first_area = max(1.0, float((first[2] - first[0]) * (first[3] - first[1])))
            second_area = max(1.0, float((second[2] - second[0]) * (second[3] - second[1])))
            similarity = min(first_area, second_area) / max(first_area, second_area)
            alignment = max(0.0, 1.0 - center_delta / 0.10)
            ranked.append(
                (
                    1.0
                    + 0.30 * alignment
                    + 0.20 * similarity
                    + 0.30 * min(1.0, candidate_warm / 0.12)
                    + 0.25 * min(1.0, candidate_edges / 0.10),
                    [
                        cx1 / width,
                        cy1 / height,
                        cx2 / width,
                        cy2 / height,
                    ],
                )
            )
    contours = list(cv2.findContours(warm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])
    contours.extend(cv2.findContours(grouped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = float(w * h)
        if area < frame_area * 0.0007 or area > frame_area * 0.16 or w < 30 or h < 18:
            continue
        aspect = w / max(h, 1)
        if not 0.7 <= aspect <= 8.0:
            continue
        pad_x, pad_y = int(w * 0.28), int(h * 0.40)
        left, top = max(0, x - pad_x), max(0, y - pad_y)
        right, bottom = min(width, x + w + pad_x), min(height, y + h + pad_y)
        orange_ratio = float(cv2.countNonZero(warm[y : y + h, x : x + w])) / max(area, 1.0)
        size_score = min(1.0, area / (frame_area * 0.035))
        horizontal_score = max(0.0, 1.0 - abs(aspect - 2.2) / 2.5)
        crop = frame[top:bottom, left:right]
        crop_hsv = hsv[top:bottom, left:right]
        crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        edge_ratio = float(np.mean(cv2.Canny(crop_gray, 60, 160) > 0))
        green_ratio = float(
            np.mean(cv2.inRange(crop_hsv, np.array([35, 65, 35]), np.array([100, 255, 255])) > 0)
        )
        yellow_ratio = float(
            np.mean(cv2.inRange(crop_hsv, np.array([18, 80, 50]), np.array([38, 255, 255])) > 0)
        )
        texture_score = min(1.0, edge_ratio / 0.10)
        cell_color_score = min(1.0, (green_ratio + yellow_ratio) / 0.10)
        touches_border = int(left == 0 or top == 0 or right == width or bottom == height)
        score = (
            0.24 * orange_ratio
            + 0.18 * size_score
            + 0.16 * horizontal_score
            + 0.24 * texture_score
            + 0.18 * cell_color_score
            - 0.25 * touches_border
        )
        ranked.append(
            (
                score,
                [left / width, top / height, right / width, bottom / height],
            )
        )
    ranked.sort(key=lambda item: item[0], reverse=True)
    selected: list[list[float]] = []
    for _, box in ranked:
        if any(_box_iou(box, existing) >= 0.72 for existing in selected):
            continue
        selected.append(box)
        if len(selected) >= limit:
            break
    return selected


def _crop_normalized(frame: np.ndarray, box: list[float]) -> np.ndarray:
    height, width = frame.shape[:2]
    left = max(0, min(width - 1, int(round(box[0] * width))))
    top = max(0, min(height - 1, int(round(box[1] * height))))
    right = max(left + 1, min(width, int(round(box[2] * width))))
    bottom = max(top + 1, min(height, int(round(box[3] * height))))
    return frame[top:bottom, left:right]


def r8_episode_candidates(
    record: dict[str, Any],
    duration: float,
    parameters: dict[str, Any],
) -> list[dict[str, Any]]:
    """Recover live rewire episodes from current stage runs only."""
    runs = _stage_runs(record)
    supplied = [
        [float(item["start_seconds"]), float(item["end_seconds"])]
        for item in runs
        if item.get("stage") == "circuit_rewiring"
    ]
    candidates = [
        {"core_interval_seconds": interval, "candidate_source": "stage_circuit_rewiring"}
        for interval in supplied
    ]
    accepted = list(supplied)
    stage_gap_recovery_allowed = not any(
        _is_merged_measurement_recording(item) for item in runs
    )
    for stage in ("recording_1", "measurement_1") if stage_gap_recovery_allowed else ():
        values = sorted(
            [
                [float(item["start_seconds"]), float(item["end_seconds"])]
                for item in runs
                if item.get("stage") == stage
            ]
        )
        for before, after in zip(values, values[1:]):
            gap = [before[1], after[0]]
            gap_seconds = gap[1] - gap[0]
            if not 0.5 <= gap_seconds <= 45.0:
                continue
            if any(max(gap[0], interval[0]) <= min(gap[1], interval[1]) for interval in accepted):
                continue
            candidates.append(
                {
                    "core_interval_seconds": gap,
                    "candidate_source": "repeated_stage_gap_recovery",
                    "recovery_stage": stage,
                }
            )
            accepted.append(gap)
    time_mode = str(parameters.get("time_mode") or "rewiring_recovery")
    if not candidates and time_mode == "wiring_transition":
        initial = next((item for item in runs if item.get("stage") == "circuit_wiring"), None)
        if initial is not None:
            candidates.append(
                {
                    "core_interval_seconds": [
                        float(initial["start_seconds"]),
                        float(initial["end_seconds"]),
                    ],
                    "candidate_source": "stage_circuit_wiring",
                }
            )
    if not candidates and time_mode == "broad_transition_search":
        candidates.append(
            {
                "core_interval_seconds": [0.0, max(0.0, duration - 0.1)],
                "candidate_source": "whole_video_broad_search",
            }
        )
    cleanup_starts = [
        float(item["start_seconds"])
        for item in runs
        if item.get("stage") == "material_cleanup"
    ]
    cleanup_cutoff = min(cleanup_starts) if cleanup_starts else duration
    margin = float(parameters.get("episode_margin_seconds", 10.0))
    ordered_runs = sorted(runs, key=lambda item: float(item.get("start_seconds") or 0.0))
    for index, candidate in enumerate(candidates, start=1):
        core_start, core_end = candidate["core_interval_seconds"]
        successor = next(
            (
                item
                for item in ordered_runs
                if float(item.get("start_seconds") or 0.0) >= core_end
                and item.get("stage") in {"measurement_2", "recording_2", "recording_1"}
            ),
            None,
        )
        successor_end = float(successor.get("start_seconds") or core_end) + 3.0 if successor else core_end
        expanded_end = min(duration, cleanup_cutoff, max(core_end + margin, successor_end))
        candidate["episode_id"] = f"live_r8_episode_{index:02d}"
        candidate["expanded_interval_seconds"] = [
            round(max(0.0, core_start - margin), 3),
            round(max(core_start, expanded_end), 3),
        ]
    return candidates


def _scan_r8_frames(
    video_path: Path,
    requests: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"unable to open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    targets: dict[int, dict[str, Any]] = {}
    for request in requests:
        frame_number = int(round(float(request["timestamp_seconds"]) * fps))
        existing = targets.setdefault(frame_number, {**request, "sampling_origins": []})
        existing["sampling_origins"].append(str(request.get("sampling_origin") or "unknown"))
    ordered = sorted(targets.items())
    if not ordered:
        capture.release()
        return []
    capture.set(cv2.CAP_PROP_POS_FRAMES, ordered[0][0])
    current = ordered[0][0]
    target_index = 0
    previous_gray_by_episode: dict[str, np.ndarray] = {}
    previous_roi_by_episode: dict[str, np.ndarray] = {}
    previous_box_by_episode: dict[str, list[float]] = {}
    records: list[dict[str, Any]] = []
    try:
        while target_index < len(ordered):
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            target_frame, request = ordered[target_index]
            if current < target_frame:
                current += 1
                continue
            if current > target_frame:
                target_index += 1
                continue
            episode_id = str(request["episode_id"])
            full_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(full_gray, (160, 90), interpolation=cv2.INTER_AREA)
            previous_gray = previous_gray_by_episode.get(episode_id)
            motion = 0.0 if previous_gray is None else float(np.mean(cv2.absdiff(small, previous_gray)))
            previous_gray_by_episode[episode_id] = small
            boxes = dynamic_battery_candidate_boxes(frame)
            previous_box = previous_box_by_episode.get(episode_id)
            box = max(
                boxes,
                key=lambda value: (value[2] - value[0]) * (value[3] - value[1]),
                default=None,
            )
            if box is None:
                box = previous_box
            if box is not None:
                previous_box_by_episode[episode_id] = box
                roi = _crop_normalized(frame, box)
                roi_gray = cv2.resize(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (120, 72))
                previous_roi = previous_roi_by_episode.get(episode_id)
                battery_motion = 0.0 if previous_roi is None else float(np.mean(cv2.absdiff(roi_gray, previous_roi)))
                previous_roi_by_episode[episode_id] = roi_gray
            else:
                battery_motion = 0.0
            records.append(
                {
                    **request,
                    "frame_number": current,
                    "timestamp_seconds": round(current / fps, 3),
                    "sharpness": round(float(cv2.Laplacian(full_gray, cv2.CV_64F).var()), 3),
                    "motion_score": round(motion, 3),
                    "battery_motion_score": round(battery_motion, 3),
                    "dynamic_battery_roi_xyxy_normalized": box,
                    "dynamic_battery_candidate_boxes": boxes,
                }
            )
            target_index += 1
            current += 1
    finally:
        capture.release()
    return records


def build_r8_cv_candidate_plan(
    video_path: Path,
    record: dict[str, Any],
    duration: float,
    parameters: dict[str, Any],
    manifest_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Apply the mature automatic OpenCV coarse-to-dense sampling policy."""
    episodes = r8_episode_candidates(record, duration, parameters)
    coarse_fps = float(parameters.get("coarse_sampling_fps", 2.0))
    core_fps = float(parameters.get("core_sampling_fps", 5.0))
    transition_fps = float(parameters.get("transition_sampling_fps", 10.0))
    maximum = int(parameters.get("sample_count", 12))
    initial_requests: list[dict[str, Any]] = []
    for episode in episodes:
        expanded_start, expanded_end = episode["expanded_interval_seconds"]
        core_start, core_end = episode["core_interval_seconds"]
        for timestamp in _sample_times(expanded_start, expanded_end, coarse_fps):
            initial_requests.append({**episode, "timestamp_seconds": timestamp, "sampling_origin": "coarse_2fps"})
        for timestamp in _sample_times(core_start, core_end, core_fps):
            initial_requests.append({**episode, "timestamp_seconds": timestamp, "sampling_origin": "core_5fps"})
    initial = _scan_r8_frames(video_path, initial_requests)
    dense_requests: list[dict[str, Any]] = []
    for episode in episodes:
        group = [item for item in initial if item["episode_id"] == episode["episode_id"]]
        anchors = sorted(group, key=lambda item: (item["battery_motion_score"], item["motion_score"]), reverse=True)[:4]
        expanded_start, expanded_end = episode["expanded_interval_seconds"]
        for anchor in anchors:
            left = max(expanded_start, float(anchor["timestamp_seconds"]) - 1.5)
            right = min(expanded_end, float(anchor["timestamp_seconds"]) + 1.5)
            for timestamp in _sample_times(left, right, transition_fps):
                dense_requests.append({**episode, "timestamp_seconds": timestamp, "sampling_origin": "transition_10fps"})
    dense = _scan_r8_frames(video_path, dense_requests)
    combined_by_key = {
        (str(item["episode_id"]), int(item["frame_number"])): item
        for item in initial + dense
    }
    selected: list[dict[str, Any]] = []
    per_episode = max(4, int(np.ceil(maximum / max(len(episodes), 1))))
    for episode in episodes:
        group = sorted(
            [item for item in combined_by_key.values() if item["episode_id"] == episode["episode_id"]],
            key=lambda item: float(item["timestamp_seconds"]),
        )
        if not group:
            continue
        core_start, core_end = episode["core_interval_seconds"]
        expanded_start, expanded_end = episode["expanded_interval_seconds"]
        anchors = [expanded_start, core_start, core_end, min(expanded_end, core_end + 1.0), expanded_end]
        chosen = {
            int(min(group, key=lambda item: abs(float(item["timestamp_seconds"]) - anchor))["frame_number"])
            for anchor in anchors
        }
        for metric, count in (("battery_motion_score", 4), ("motion_score", 3), ("sharpness", 2)):
            chosen.update(
                int(item["frame_number"])
                for item in sorted(group, key=lambda item: float(item[metric]), reverse=True)[:count]
            )
        episode_selected = [item for item in group if int(item["frame_number"]) in chosen]
        if len(episode_selected) > per_episode:
            required = {
                int(min(episode_selected, key=lambda item: abs(float(item["timestamp_seconds"]) - anchor))["frame_number"])
                for anchor in (expanded_start, core_start, core_end, expanded_end)
            }
            ranked = sorted(
                episode_selected,
                key=lambda item: (
                    int(item["frame_number"]) in required,
                    float(item["battery_motion_score"]),
                    float(item["motion_score"]),
                    float(item["sharpness"]),
                ),
                reverse=True,
            )[:per_episode]
            episode_selected = sorted(ranked, key=lambda item: float(item["timestamp_seconds"]))
        selected.extend(episode_selected)
    if len(selected) > maximum:
        indexes = np.linspace(0, len(selected) - 1, maximum).round().astype(int)
        selected = [selected[int(index)] for index in indexes]
    plan = []
    for item in sorted(selected, key=lambda row: float(row["timestamp_seconds"])):
        plan.append(
            {
                **item,
                "stage": "circuit_rewiring",
                "stage_run": int(str(item["episode_id"]).rsplit("_", 1)[-1]),
                "selection_policy": "uniform_temporal_backbone_plus_battery_motion_global_motion_sharpness",
            }
        )
    if manifest_path is not None:
        write_json(
            manifest_path,
            {
                "algorithm_version": ALGORITHM_VERSION,
                "selection_basis": "current_video_opencv_evidence_only",
                "episodes": episodes,
                "sampling_policy": {
                    "coarse_fps": coarse_fps,
                    "core_fps": core_fps,
                    "transition_fps": transition_fps,
                },
                "initial_scanned_count": len(initial),
                "dense_scanned_count": len(dense),
                "selected_count": len(plan),
                "selected_frames": plan,
                "video_id_used_for_routing": False,
                "fixed_video_roi_used": False,
            },
        )
    return plan


def decode_evidence(
    video_path: Path,
    plans: dict[int, list[dict[str, Any]]],
    evidence_dir: Path,
    video_id: str,
    allow_video_calibration: bool = False,
) -> dict[int, list[dict[str, Any]]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"unable to open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    output: dict[int, list[dict[str, Any]]] = {}
    try:
        for rubric_id, requests in plans.items():
            rows: list[dict[str, Any]] = []
            seen: set[int] = set()
            targets_by_frame: dict[int, dict[str, Any]] = {}
            for request in requests:
                frame_number = int(round(float(request["timestamp_seconds"]) * fps))
                targets_by_frame.setdefault(frame_number, request)
            targets = sorted(targets_by_frame.items())
            if not targets:
                output[rubric_id] = rows
                continue
            capture.set(cv2.CAP_PROP_POS_FRAMES, targets[0][0])
            current = targets[0][0]
            target_index = 0
            while target_index < len(targets):
                ok, frame = capture.read()
                if not ok or frame is None:
                    break
                frame_number, request = targets[target_index]
                if current < frame_number:
                    current += 1
                    continue
                if current > frame_number:
                    target_index += 1
                    continue
                if frame_number in seen:
                    target_index += 1
                    continue
                seen.add(frame_number)
                actual = frame_number / fps if fps > 0 else float(request["timestamp_seconds"])
                frame_id = f"frame_{frame_number:08d}"
                stem = f"r{rubric_id}_{frame_id}_{actual:010.3f}s"
                panorama_path = evidence_dir / "frames" / f"{stem}.jpg"
                enhanced_path = evidence_dir / "enhanced" / f"{stem}_enhanced.jpg"
                _write_jpeg(panorama_path, frame)
                _write_jpeg(enhanced_path, _enhance(frame))
                height, width = frame.shape[:2]
                role_views: dict[str, Any] = {}
                if rubric_id == 8:
                    candidate_boxes = list(request.get("dynamic_battery_candidate_boxes") or [])
                    primary_box = request.get("dynamic_battery_roi_xyxy_normalized")
                    if isinstance(primary_box, list) and len(primary_box) == 4:
                        candidate_boxes = [primary_box] + [
                            box for box in candidate_boxes if box != primary_box
                        ]
                    battery_candidates: list[dict[str, Any]] = []
                    for candidate_index, box in enumerate(candidate_boxes[:3], start=1):
                        if not isinstance(box, list) or len(box) != 4:
                            continue
                        crop = _crop_normalized(frame, [float(value) for value in box])
                        if not crop.size:
                            continue
                        crop = _enhance(crop)
                        if max(crop.shape[:2]) < 1000:
                            scale = min(3.0, 1000.0 / max(crop.shape[:2]))
                            crop = cv2.resize(
                                crop,
                                None,
                                fx=scale,
                                fy=scale,
                                interpolation=cv2.INTER_CUBIC,
                            )
                        candidate_path = (
                            evidence_dir
                            / "battery_rois"
                            / f"{stem}_candidate_{candidate_index:02d}.jpg"
                        )
                        _write_jpeg(candidate_path, crop, 95)
                        battery_candidates.append(
                            {
                                "normalized_xyxy": [float(value) for value in box],
                                "image_path": str(candidate_path.resolve()),
                            }
                        )
                    role_views["battery_candidates"] = battery_candidates
                rows.append(
                    {
                        **request,
                        "image_group": len(rows) + 1,
                        "frame_id": frame_id,
                        "frame_number": frame_number,
                        "timestamp_seconds": round(actual, 6),
                        "panorama_path": str(panorama_path.resolve()),
                        "enhanced_path": str(enhanced_path.resolve()),
                        "role_views": role_views,
                    }
                )
                target_index += 1
                current += 1
            output[rubric_id] = rows
    finally:
        capture.release()
    return output


def image_data_url(
    path: Path,
    max_edge: int = 1120,
    quality: int = 82,
    lossless: bool = False,
) -> str:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"unable to decode model image: {path}")
    height, width = image.shape[:2]
    scale = min(1.0, float(max_edge) / max(height, width))
    if scale < 1.0:
        image = cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA)
    extension = ".png" if lossless else ".jpg"
    options = (
        [int(cv2.IMWRITE_PNG_COMPRESSION), 3]
        if lossless
        else [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    )
    ok, encoded = cv2.imencode(extension, image, options)
    if not ok:
        raise ValueError(f"unable to encode model image: {path}")
    media_type = "image/png" if lossless else "image/jpeg"
    return f"data:{media_type};base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def _frame_list(rows: list[dict[str, Any]]) -> str:
    return ", ".join(
        f"group {index}={row['frame_id']}@{float(row['timestamp_seconds']):.3f}s stage={row['stage']}"
        for index, row in enumerate(rows, start=1)
    )


def prompt_for(
    rubric_id: int, rows: list[dict[str, Any]], skill_instruction: str = ""
) -> str:
    frames = _frame_list(rows)
    shared = f"""You inspect real frames from one middle-school resistance experiment video. Frames: {frames}.
Skill instruction: {skill_instruction}
Each image group is one real timestamp. Use visible pixels and temporal changes only. Do not infer hidden details from expected experiment procedure. Do not output pass/fail. Return exactly one JSON object with one observation for every image group."""
    if rubric_id == 0:
        return shared + """
For each group classify organizing_action as visible, not_visible, or unclear. Organizing includes gathering/coiling wires, moving apparatus into a compact orderly group, returning apparatus, or clearing the central work area. Merely connecting or measuring is not organizing.
JSON: {"observations":[{"image_group":1,"frame_id":"frame_00000000","organizing_action":"visible","workspace_change":"direct visible change","confidence":0.0,"evidence":"visible pixels"}],"overall_evidence":"summary"}"""
    if rubric_id == 2:
        return shared + """
Inspect the stable circuit topology. For each group classify voltmeter_relation as parallel_across_resistor, same_node_or_not_parallel, or unclear. parallel_across_resistor requires the two voltmeter leads to reach the two different endpoints/nodes of the fixed resistor under test. A wire may continue outside a crop, but do not invent its endpoint. Also report stable_state.
JSON: {"observations":[{"image_group":1,"frame_id":"frame_00000000","voltmeter_visible":true,"resistor_visible":true,"voltmeter_relation":"parallel_across_resistor","stable_state":true,"confidence":0.0,"evidence":"visible endpoints/path"}],"overall_evidence":"summary"}"""
    if rubric_id == 4:
        return shared + """
Each group contains panorama, enhanced panorama, then available ammeter/voltmeter crops. Determine visible terminal polarity for each meter. State is correct, reversed, or unclear. A leftward/negative deflection is direct reversal evidence; image vertical is not meter zero. Do not use wire color alone unless its battery-side polarity is also visible.
JSON: {"observations":[{"image_group":1,"frame_id":"frame_00000000","ammeter_polarity":"correct","voltmeter_polarity":"correct","negative_deflection":false,"confidence":0.0,"evidence":"visible terminal/needle evidence"}],"overall_evidence":"summary"}"""
    if rubric_id == 8:
        return shared + """
Track the rewiring episode. For each group report switch_state=open, closed, or unclear; battery_action=none, touching_or_rewiring, configuration_changed, or unclear. Then classify the most likely visual sequence without judging compliance: open_then_change_then_closed, closed_during_change, change_with_switch_unclear, no_battery_change, or unclear. The order must come from group numbers.
JSON: {"observations":[{"image_group":1,"frame_id":"frame_00000000","switch_state":"open","battery_action":"touching_or_rewiring","confidence":0.0,"evidence":"visible state/action"}],"sequence":{"state":"open_then_change_then_closed","open_group_ids":[1],"change_group_ids":[2],"closed_group_ids":[3],"confidence":0.0,"evidence":"visible temporal order"}}"""
    raise ValueError(f"unsupported rubric: {rubric_id}")


def validate_observation(rubric_id: int, value: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    observations = value.get("observations")
    if not isinstance(observations, list):
        raise ValueError("observations_missing")
    expected = {index: row["frame_id"] for index, row in enumerate(rows, start=1)}
    parsed: list[dict[str, Any]] = []
    enums: dict[int, tuple[str, set[str]]] = {
        0: ("organizing_action", {"visible", "not_visible", "unclear"}),
        2: ("voltmeter_relation", {"parallel_across_resistor", "same_node_or_not_parallel", "unclear"}),
        4: ("ammeter_polarity", {"correct", "reversed", "unclear"}),
        8: ("switch_state", {"open", "closed", "unclear"}),
    }
    field, allowed = enums[rubric_id]
    for item in observations:
        if not isinstance(item, dict) or item.get("image_group") not in expected:
            raise ValueError("image_group_invalid")
        group = int(item["image_group"])
        if item.get(field) not in allowed:
            raise ValueError(f"{field}_invalid")
        normalized = dict(item)
        normalized.update(
            {
                "image_group": group,
                "frame_id": expected[group],
                "model_frame_id": str(item.get("frame_id") or ""),
                "frame_id_corrected_from_group": item.get("frame_id") != expected[group],
                "confidence": _confidence(item.get("confidence")),
                "evidence": str(item.get("evidence") or ""),
            }
        )
        if rubric_id == 2:
            normalized["voltmeter_visible"] = bool(item.get("voltmeter_visible"))
            normalized["resistor_visible"] = bool(item.get("resistor_visible"))
            normalized["stable_state"] = bool(item.get("stable_state"))
        elif rubric_id == 4:
            if item.get("voltmeter_polarity") not in {"correct", "reversed", "unclear"}:
                raise ValueError("voltmeter_polarity_invalid")
            normalized["negative_deflection"] = bool(item.get("negative_deflection"))
        elif rubric_id == 8:
            if item.get("battery_action") not in {"none", "touching_or_rewiring", "configuration_changed", "unclear"}:
                raise ValueError("battery_action_invalid")
        parsed.append(normalized)
    if {item["image_group"] for item in parsed} != set(expected):
        raise ValueError("image_groups_incomplete")
    parsed.sort(key=lambda item: item["image_group"])
    result: dict[str, Any] = {"observations": parsed, "overall_evidence": str(value.get("overall_evidence") or "")}
    if rubric_id == 8:
        sequence = value.get("sequence")
        if not isinstance(sequence, dict) or sequence.get("state") not in {
            "open_then_change_then_closed", "closed_during_change", "change_with_switch_unclear", "no_battery_change", "unclear"
        }:
            raise ValueError("sequence_invalid")
        known = set(expected)
        result["sequence"] = {
            "state": sequence["state"],
            "open_group_ids": [int(item) for item in sequence.get("open_group_ids") or [] if item in known],
            "change_group_ids": [int(item) for item in sequence.get("change_group_ids") or [] if item in known],
            "closed_group_ids": [int(item) for item in sequence.get("closed_group_ids") or [] if item in known],
            "confidence": _confidence(sequence.get("confidence")),
            "evidence": str(sequence.get("evidence") or ""),
        }
    return result


def _media_groups(rubric_id: int, rows: list[dict[str, Any]]) -> list[list[Path]]:
    groups: list[list[Path]] = []
    for row in rows:
        paths = (
            [Path(row["panorama_path"])]
            if rubric_id == 2
            else [Path(row["panorama_path"]), Path(row["enhanced_path"])]
        )
        if rubric_id == 4:
            for role in ("ammeter", "voltmeter"):
                view = row.get("role_views", {}).get(role)
                if isinstance(view, dict) and view.get("image_path"):
                    paths.append(Path(view["image_path"]))
        if rubric_id == 2:
            topology = row.get("role_views", {}).get("joint_topology")
            if isinstance(topology, dict):
                if topology.get("enhanced_path"):
                    paths.append(Path(topology["enhanced_path"]))
            for role in ("voltmeter_candidates", "resistor_candidates"):
                views = row.get("role_views", {}).get(role, [])
                view = views[0] if isinstance(views, list) and views else None
                if isinstance(view, dict) and view.get("enhanced_path"):
                    paths.append(Path(view["enhanced_path"]))
        if rubric_id == 8:
            for view in row.get("role_views", {}).get("battery_candidates", [])[:3]:
                if isinstance(view, dict) and view.get("image_path"):
                    paths.append(Path(view["image_path"]))
        groups.append(paths)
    return groups


def cleanup_workspace_metrics(rows: list[dict[str, Any]], review_dir: Path | None = None) -> dict[str, Any]:
    """Measure whether colored apparatus/wires leave the central work area."""
    per_frame: list[dict[str, Any]] = []
    x1, y1, x2, y2 = CLEANUP_CENTRAL_ROI
    for row in rows:
        image = cv2.imread(str(row["panorama_path"]), cv2.IMREAD_COLOR)
        if image is None:
            continue
        height, width = image.shape[:2]
        box = [int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)]
        left, top, right, bottom = box
        crop = image[top:bottom, left:right]
        if not crop.size:
            continue
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        occupied = (saturation > 60) & (value > 45)
        roi_path: str | None = None
        if review_dir is not None:
            target = review_dir / "central_rois" / f"{row['frame_id']}_central.jpg"
            _write_jpeg(target, crop, 95)
            roi_path = str(target.resolve())
        per_frame.append(
            {
                "image_group": int(row["image_group"]),
                "frame_id": str(row["frame_id"]),
                "timestamp_seconds": float(row["timestamp_seconds"]),
                "central_saturated_occupancy": round(float(np.mean(occupied)), 6),
                "central_roi_xyxy_normalized": list(CLEANUP_CENTRAL_ROI),
                "central_roi_path": roi_path,
            }
        )
    values = [float(item["central_saturated_occupancy"]) for item in per_frame]
    if len(values) < 2:
        return {
            "method": "central_hsv_saturated_occupancy_transition",
            "per_frame": per_frame,
            "supports_cleanup_transition": False,
            "strong_cleanup_transition": False,
            "reason": "fewer_than_two_decodable_frames",
        }
    early_peak = max(values[: min(2, len(values))])
    tail = values[-1]
    drop = early_peak - tail
    supports = early_peak >= 0.15 and tail <= 0.18 and drop >= 0.10
    strong = early_peak >= 0.24 and tail <= 0.05 and drop >= 0.18
    return {
        "method": "central_hsv_saturated_occupancy_transition",
        "per_frame": per_frame,
        "early_peak": round(early_peak, 6),
        "tail": round(tail, 6),
        "early_to_tail_drop": round(drop, 6),
        "supports_cleanup_transition": supports,
        "strong_cleanup_transition": strong,
        "thresholds": {
            "support": {"minimum_early_peak": 0.15, "maximum_tail": 0.18, "minimum_drop": 0.10},
            "strong": {"minimum_early_peak": 0.24, "maximum_tail": 0.05, "minimum_drop": 0.18},
        },
    }


def validate_cleanup_sequence_review(value: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    allowed = {"visible", "not_visible", "unclear"}
    state = value.get("organizing_sequence")
    if state not in allowed:
        raise ValueError("organizing_sequence_invalid")
    expected = {int(row["image_group"]): str(row["frame_id"]) for row in rows}
    raw_groups = value.get("evidence_group_ids") or []
    if not isinstance(raw_groups, list) or any(not isinstance(item, int) or item not in expected for item in raw_groups):
        raise ValueError("evidence_group_ids_invalid")
    details: dict[str, str] = {}
    for field in ("wire_gathering", "apparatus_moved_to_edge", "central_area_cleared"):
        if value.get(field) not in allowed:
            raise ValueError(f"{field}_invalid")
        details[field] = str(value[field])
    groups = sorted(set(raw_groups))
    return {
        "organizing_sequence": str(state),
        **details,
        "evidence_group_ids": groups,
        "evidence_frame_ids": [expected[group] for group in groups],
        "confidence": _confidence(value.get("confidence")),
        "evidence": str(value.get("evidence") or ""),
    }


def _call_qwen(
    rubric_id: int,
    rows: list[dict[str, Any]],
    model_config: dict[str, Any],
    raw_path: Path,
    skill_instruction: str = "",
    execution_fingerprint: str | None = None,
) -> dict[str, Any]:
    input_frame_ids = [str(row["frame_id"]) for row in rows]
    if raw_path.is_file():
        cached = read_json(raw_path)
        observation = cached.get("observation")
        if (
            isinstance(observation, dict)
            and cached.get("algorithm_version") == ALGORITHM_VERSION
            and cached.get("execution_fingerprint") == execution_fingerprint
            and cached.get("input_frame_ids") == input_frame_ids
        ):
            return validate_observation(rubric_id, observation, rows)
    if rubric_id == 2 and len(rows) > 1:
        merged_items: list[dict[str, Any]] = []
        summaries: list[str] = []
        batch_paths: list[str] = []
        for batch_index, start in enumerate(range(0, len(rows), 1), start=1):
            batch_rows = rows[start : start + 1]
            batch_path = raw_path.with_name(f"{raw_path.stem}_batch_{batch_index:02d}.json")
            batch = _call_qwen(
                rubric_id,
                batch_rows,
                model_config,
                batch_path,
                skill_instruction=skill_instruction,
                execution_fingerprint=execution_fingerprint,
            )
            for item in batch["observations"]:
                merged_items.append({**item, "image_group": int(item["image_group"]) + start})
            summaries.append(str(batch.get("overall_evidence") or ""))
            batch_paths.append(str(batch_path.resolve()))
        merged = validate_observation(
            rubric_id,
            {
                "observations": merged_items,
                "overall_evidence": " | ".join(value for value in summaries if value),
            },
            rows,
        )
        write_json(
            raw_path,
            {
                "algorithm_version": ALGORITHM_VERSION,
                "execution_fingerprint": execution_fingerprint,
                "input_frame_ids": input_frame_ids,
                "rubric_id": rubric_id,
                "batch_size": 1,
                "batch_paths": batch_paths,
                "observation": merged,
            },
        )
        return merged
    base_url = os.getenv("QWEN_API_BASE_URL", str(model_config["base_url"]))
    token = os.getenv("QWEN_API_TOKEN", "EMPTY")
    model = os.getenv("QWEN_MODEL", str(model_config["model"]))
    prompt = prompt_for(rubric_id, rows, skill_instruction)
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    media: list[dict[str, Any]] = []
    media_groups = _media_groups(rubric_id, rows)
    for group, paths in enumerate(media_groups, start=1):
        content.append({"type": "text", "text": f"Image group {group}."})
        for path in paths:
            max_edge = 4096 if rubric_id == 2 else 1120
            quality = 100 if rubric_id == 2 else 82
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_data_url(
                            path,
                            max_edge=max_edge,
                            quality=quality,
                            lossless=rubric_id == 2,
                        )
                    },
                }
            )
            media.append({"image_group": group, "path": str(path.resolve())})
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": response_token_budget(len(media_groups)),
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    attempts: list[dict[str, Any]] = []
    for attempt in range(2):
        text: str | None = None
        request = urllib.request.Request(
            base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                raw = json.loads(response.read().decode("utf-8"))
            message = raw.get("choices", [{}])[0].get("message", {})
            text = message.get("content")
            if not isinstance(text, str):
                raise ValueError("response_content_not_text")
            parsed = validate_observation(rubric_id, parse_json_object(text), rows)
            attempts.append({"attempt": attempt + 1, "content": text, "schema_errors": []})
            write_json(
                raw_path,
                {
                    "algorithm_version": ALGORITHM_VERSION,
                    "execution_fingerprint": execution_fingerprint,
                    "input_frame_ids": input_frame_ids,
                    "rubric_id": rubric_id,
                    "model": model,
                    "base_url": base_url,
                    "prompt": prompt,
                    "media": media,
                    "attempts": attempts,
                    "observation": parsed,
                },
            )
            return parsed
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ConnectionError, OSError, ValueError, json.JSONDecodeError) as exc:
            attempt_record: dict[str, Any] = {
                "attempt": attempt + 1,
                "errors": [f"{type(exc).__name__}:{exc}"],
            }
            if text is not None:
                attempt_record["content"] = text
            attempts.append(attempt_record)
            if attempt == 0:
                content.append(
                    {
                        "type": "text",
                        "text": (
                            "Schema correction: start directly with { and return one complete legal JSON object "
                            "for every supplied image group. Omit analysis, Markdown, and code fences."
                        ),
                    }
                )
                time.sleep(2.0)
    write_json(
        raw_path.with_name(raw_path.stem + "_failed.json"),
        {
            "algorithm_version": ALGORITHM_VERSION,
            "execution_fingerprint": execution_fingerprint,
            "rubric_id": rubric_id,
            "model": model,
            "base_url": base_url,
            "media": media,
            "attempts": attempts,
            "status": "request_failed",
        },
    )
    raise RuntimeError(f"Qwen rubric {rubric_id} request failed after targeted retry: {attempts}")


def _call_cleanup_sequence_review(
    rows: list[dict[str, Any]],
    model_config: dict[str, Any],
    raw_path: Path,
    workspace_metrics: dict[str, Any],
    skill_instruction: str = "",
    execution_fingerprint: str | None = None,
) -> dict[str, Any]:
    if raw_path.is_file():
        cached = read_json(raw_path)
        observation = cached.get("observation")
        if (
            isinstance(observation, dict)
            and cached.get("algorithm_version") == CLEANUP_REVIEW_VERSION
            and cached.get("execution_fingerprint") == execution_fingerprint
        ):
            return validate_cleanup_sequence_review(observation, rows)
    base_url = os.getenv("QWEN_API_BASE_URL", str(model_config["base_url"]))
    token = os.getenv("QWEN_API_TOKEN", "EMPTY")
    model = os.getenv("QWEN_MODEL", str(model_config["model"]))
    frames = _frame_list(rows)
    prompt = f"""You compare an ordered sequence of real frames from one middle-school resistance experiment video. Frames: {frames}.
Skill instruction: {skill_instruction}
Judge the sequence as a whole, not each frame in isolation. Use visible temporal changes only. An organizing sequence is visible when hands gather or coil leads, apparatus moves into a compact edge group, or an occupied central work area becomes visibly clear. A visible wire loop being gathered is organizing unless the sequence directly shows its plug being inserted into a terminal for continued measurement. Static final placement alone is insufficient without an ordered visible change. Do not output pass/fail.
Return exactly one JSON object:
{{"organizing_sequence":"visible|not_visible|unclear","wire_gathering":"visible|not_visible|unclear","apparatus_moved_to_edge":"visible|not_visible|unclear","central_area_cleared":"visible|not_visible|unclear","evidence_group_ids":[1],"confidence":0.0,"evidence":"describe the visible before-to-after change"}}"""
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    media: list[dict[str, Any]] = []
    for group, row in enumerate(rows, start=1):
        path = Path(row["panorama_path"])
        content.append({"type": "text", "text": f"Ordered image group {group}."})
        content.append({"type": "image_url", "image_url": {"url": image_data_url(path)}})
        media.append({"image_group": group, "path": str(path.resolve())})
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 1200,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    attempts: list[dict[str, Any]] = []
    for attempt in range(2):
        request = urllib.request.Request(
            base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                raw = json.loads(response.read().decode("utf-8"))
            text = raw.get("choices", [{}])[0].get("message", {}).get("content")
            if not isinstance(text, str):
                raise ValueError("response_content_not_text")
            parsed = validate_cleanup_sequence_review(parse_json_object(text), rows)
            attempts.append({"attempt": attempt + 1, "content": text, "schema_errors": []})
            write_json(
                raw_path,
                {
                    "algorithm_version": CLEANUP_REVIEW_VERSION,
                    "execution_fingerprint": execution_fingerprint,
                    "model": model,
                    "base_url": base_url,
                    "prompt": prompt,
                    "media": media,
                    "workspace_metrics": workspace_metrics,
                    "attempts": attempts,
                    "observation": parsed,
                },
            )
            return parsed
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ConnectionError, OSError, ValueError, json.JSONDecodeError) as exc:
            attempts.append({"attempt": attempt + 1, "errors": [f"{type(exc).__name__}:{exc}"]})
            if attempt == 0:
                content.append({"type": "text", "text": "Schema correction: return one complete legal JSON object and cite only supplied image-group numbers."})
                time.sleep(2.0)
    write_json(
        raw_path.with_name(raw_path.stem + "_failed.json"),
        {
            "algorithm_version": CLEANUP_REVIEW_VERSION,
            "execution_fingerprint": execution_fingerprint,
            "model": model,
            "base_url": base_url,
            "media": media,
            "workspace_metrics": workspace_metrics,
            "attempts": attempts,
            "status": "request_failed",
        },
    )
    raise RuntimeError(f"Qwen cleanup sequence review failed after targeted retry: {attempts}")


def fuse_cleanup_review(
    primary_result: dict[str, Any],
    sequence_review: dict[str, Any] | None,
    workspace_metrics: dict[str, Any],
) -> dict[str, Any]:
    if primary_result.get("decision") != "fail":
        return primary_result
    review_visible = bool(sequence_review and sequence_review.get("organizing_sequence") == "visible")
    cv_support = bool(workspace_metrics.get("supports_cleanup_transition"))
    cv_strong = bool(workspace_metrics.get("strong_cleanup_transition"))
    recovered = (review_visible and cv_support) or cv_strong
    diagnostics = dict(primary_result.get("diagnostics") or {})
    diagnostics["cleanup_failure_review"] = {
        "sequence_review": sequence_review,
        "workspace_metrics": workspace_metrics,
        "fusion_rule": "sequence_visible_and_workspace_transition_or_strong_workspace_clearance",
        "recovered": recovered,
    }
    if not recovered:
        return {**primary_result, "diagnostics": diagnostics}
    confidence = 0.88 if cv_strong else max(0.62, _confidence(sequence_review.get("confidence") if sequence_review else 0.0))
    reason = (
        "organizing_action_recovered_by_strong_workspace_clearance"
        if cv_strong and not review_visible
        else "organizing_action_recovered_by_sequence_and_workspace_change"
    )
    return _result("pass", confidence, reason, diagnostics)


def _result(decision: str, confidence: float, reason: str, diagnostics: dict[str, Any]) -> dict[str, Any]:
    return {
        "decision": decision,
        "predicted_score": 1 if decision == "pass" else 0,
        "confidence": round(max(0.0, min(1.0, confidence)), 4),
        "reason": reason,
        "diagnostics": diagnostics,
    }


def reduce_rubric(
    rubric_id: int,
    observation: dict[str, Any],
    rows: list[dict[str, Any]],
    fusion_parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    items = list(observation.get("observations") or [])
    parameters = fusion_parameters or {}
    frame_by_group = {row["image_group"]: row for row in rows}

    def evidence(groups: list[int]) -> list[dict[str, Any]]:
        return [
            {
                "image_group": group,
                "frame_id": frame_by_group[group]["frame_id"],
                "timestamp_seconds": frame_by_group[group]["timestamp_seconds"],
                "panorama_path": frame_by_group[group]["panorama_path"],
                "role_views": frame_by_group[group].get("role_views", {}),
            }
            for group in groups
            if group in frame_by_group
        ]

    if rubric_id == 0:
        visible = [item for item in items if item["organizing_action"] == "visible"]
        if len(visible) >= int(parameters.get("minimum_visible_actions", 1)):
            groups = [item["image_group"] for item in visible]
            return _result("pass", max(0.55, max(item["confidence"] for item in visible)), "organizing_action_visible", {"supporting_evidence": evidence(groups), "model_observations": items})
        confidence = max([item["confidence"] for item in items] or [0.0])
        return _result("fail", max(0.5, confidence), "no_organizing_action_visible_binary_tie_break", {"supporting_evidence": evidence([item["image_group"] for item in items]), "model_observations": items})

    if rubric_id == 2:
        parallel = [item for item in items if item["voltmeter_relation"] == "parallel_across_resistor" and item.get("stable_state")]
        if len(parallel) >= int(parameters.get("minimum_parallel_support", 1)):
            groups = [item["image_group"] for item in parallel]
            return _result("pass", max(0.52, max(item["confidence"] for item in parallel)), "stable_voltmeter_parallel_across_resistor", {"supporting_evidence": evidence(groups), "model_observations": items})
        wrong = [item for item in items if item["voltmeter_relation"] == "same_node_or_not_parallel"]
        groups = [item["image_group"] for item in (wrong or items)]
        return _result("fail", max(0.5, max([item["confidence"] for item in (wrong or items)] or [0.0])), "parallel_topology_not_visually_supported_binary_tie_break", {"supporting_evidence": evidence(groups), "model_observations": items})

    if rubric_id == 4:
        reversed_items = [
            item for item in items
            if item["ammeter_polarity"] == "reversed" or item["voltmeter_polarity"] == "reversed" or item.get("negative_deflection")
        ]
        if reversed_items:
            groups = [item["image_group"] for item in reversed_items]
            return _result("fail", max(0.58, max(item["confidence"] for item in reversed_items)), "visible_reversed_meter_polarity", {"supporting_evidence": evidence(groups), "model_observations": items})
        correct = [item for item in items if item["ammeter_polarity"] == "correct" or item["voltmeter_polarity"] == "correct"]
        groups = [item["image_group"] for item in (correct or items)]
        return _result("pass", max(0.5, max([item["confidence"] for item in (correct or items)] or [0.0])), "no_visible_reversal_lenient_binary_rule", {"supporting_evidence": evidence(groups), "model_observations": items})

    sequence = observation["sequence"]
    state = sequence["state"]
    groups = sorted(set(sequence["open_group_ids"] + sequence["change_group_ids"] + sequence["closed_group_ids"]))
    sequence_confident = float(sequence.get("confidence") or 0.0) >= float(
        parameters.get("minimum_sequence_confidence", 0.5)
    )
    if state == "open_then_change_then_closed" and sequence_confident:
        return _result("pass", max(0.55, sequence["confidence"]), "visible_open_change_close_sequence", {"supporting_evidence": evidence(groups), "sequence": sequence, "model_observations": items})
    if state == "no_battery_change":
        return _result("pass", max(0.5, sequence["confidence"]), "no_battery_change_observed_conditional_pass", {"supporting_evidence": evidence(groups or [item["image_group"] for item in items]), "sequence": sequence, "model_observations": items})
    if state == "closed_during_change":
        return _result("fail", max(0.58, sequence["confidence"]), "switch_visibly_closed_during_battery_change", {"supporting_evidence": evidence(groups), "sequence": sequence, "model_observations": items})
    return _result("fail", max(0.5, sequence["confidence"]), "battery_change_sequence_not_confirmed_binary_tie_break", {"supporting_evidence": evidence(groups or [item["image_group"] for item in items]), "sequence": sequence, "model_observations": items})


def _specialized_rubric8_result(document: dict[str, Any], result_path: Path) -> dict[str, Any]:
    decision = document.get("decision")
    predicted = document.get("predicted_score")
    if decision not in {"pass", "fail"} or predicted != (1 if decision == "pass" else 0):
        raise ValueError("specialized Rubric 8 result is not a valid binary artifact")
    diagnostics = document.get("diagnostics")
    if not isinstance(diagnostics, dict):
        diagnostics = {}
    return _result(
        decision,
        _confidence(document.get("confidence")),
        str(document.get("reason") or "specialized_battery_sequence_result"),
        {
            "specialized_algorithm": "resistance_disconnect_battery_sequence_v3_dynamic_roi",
            "specialized_result_path": str(result_path.resolve()),
            "episodes": document.get("episodes") if isinstance(document.get("episodes"), list) else [],
            "aggregate": diagnostics,
        },
    )


def run_specialized_rubric8(
    video_path: Path,
    video_id: str,
    output_dir: Path,
    model_config: dict[str, Any],
    skill_parameters: dict[str, Any],
    script_path: Path = DEFAULT_R8_SCRIPT,
    action_summary_path: Path | None = None,
) -> dict[str, Any]:
    result_path = output_dir / f"video_{video_id}" / "result.json"
    requested_time_mode = str(skill_parameters.get("time_mode", "rewiring_recovery"))
    source_size_bytes = video_path.stat().st_size
    if result_path.is_file():
        existing = read_json(result_path)
        if (
            existing.get("source_video_size_bytes") == source_size_bytes
            and existing.get("algorithm_id") == "resistance_disconnect_battery_sequence_v3_dynamic_roi"
            and existing.get("dynamic_r8_execution") is True
            and existing.get("time_mode") == requested_time_mode
        ):
            return _specialized_rubric8_result(existing, result_path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"incomplete specialized Rubric 8 output exists: {output_dir}")
    if action_summary_path is None:
        raise ValueError("current-run Rubric 8 action summary is required")
    for required in (script_path, action_summary_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    token_env = str(model_config.get("api_key_env") or "QWEN_API_TOKEN")
    base_url_env = str(model_config.get("base_url_env") or "QWEN_API_BASE_URL")
    model_env = str(model_config.get("model_env") or "QWEN_MODEL")
    base_url = str(model_config.get("base_url") or os.getenv(base_url_env, "")).strip()
    token = os.getenv(token_env, "").strip()
    model = str(model_config.get("model") or os.getenv(model_env, "qwen")).strip() or "qwen"
    if not base_url or not token:
        raise RuntimeError(f"{base_url_env} and {token_env} are required")
    command = [
        sys.executable,
        str(script_path.resolve()),
        "--action-summary",
        str(action_summary_path.resolve()),
        "--output-root",
        str(output_dir.resolve()),
        "--video-ids",
        video_id,
        "--source-video",
        str(video_path.resolve()),
        "--api-base-url",
        base_url,
        "--api-token",
        token,
        "--model",
        model,
        "--coarse-fps",
        str(skill_parameters["coarse_fps"]),
        "--core-fps",
        str(skill_parameters["core_fps"]),
        "--transition-fps",
        str(skill_parameters["transition_fps"]),
        "--dynamic-roi-min-confidence",
        str(skill_parameters["dynamic_roi_min_confidence"]),
        "--time-mode",
        requested_time_mode,
    ]
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        timeout=1800,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()[-2000:]
        raise RuntimeError(f"specialized Rubric 8 runner exited {completed.returncode}: {detail}")
    if not result_path.is_file():
        raise RuntimeError(f"specialized Rubric 8 result was not generated: {result_path}")
    document = read_json(result_path)
    if document.get("video_id") != video_id:
        raise ValueError("specialized Rubric 8 video id mismatch")
    if document.get("source_video_size_bytes") != source_size_bytes:
        raise ValueError("specialized Rubric 8 source size mismatch")
    return _specialized_rubric8_result(document, result_path)


def run_remaining_rubrics(
    video_path: Path,
    source_video_id: str,
    video_id: str,
    run_dir: Path,
    model_config: dict[str, Any],
    action_summary_path: Path | None = None,
    boundary_summary_path: Path | None = None,
    fallback_action_summary_path: Path | None = None,
    rubric8_script_path: Path | None = None,
    rubric8_action_summary_path: Path | None = None,
    allow_video_calibration: bool = False,
    enable_specialized_r8: bool = True,
    allow_historical_fallback: bool = False,
    skill_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        from .skills import EXECUTOR_REGISTRY, execution_for_rubric
    except ImportError:
        from skills import EXECUTOR_REGISTRY, execution_for_rubric  # type: ignore
    fallback_skill_ids = {
        0: "cleanup.explicit_stage",
        2: "voltmeter.parallel_endpoint_adaptive",
        8: "battery.recovery_episode",
    }
    executions = {
        rubric_id: (
            execution_for_rubric(skill_plan, rubric_id)
            if skill_plan
            else {
                "skill_id": skill_id,
                "parameters": dict(EXECUTOR_REGISTRY[skill_id].defaults),
                "execution_fingerprint": None,
            }
        )
        for rubric_id, skill_id in fallback_skill_ids.items()
    }
    execution_fingerprints = {
        str(rubric_id): execution["execution_fingerprint"]
        for rubric_id, execution in executions.items()
    }
    fallback = fallback_action_summary_path
    action_path = action_summary_path if action_summary_path and action_summary_path.is_file() else (
        fallback if allow_historical_fallback and fallback is not None else None
    )
    if action_path is None or not action_path.is_file():
        raise ValueError("current live action summary is required")
    record = _source_record(read_json(action_path), source_video_id, video_id)
    boundary_used = False
    if boundary_summary_path and boundary_summary_path.is_file():
        boundary = _boundary_record(read_json(boundary_summary_path), source_video_id, video_id)
        if boundary is not None:
            record, boundary_used = boundary, True
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"unable to open video: {video_path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    duration = frame_count / fps if fps > 0 else 0.0
    rubric_ids = (0, 2)
    evidence_dir = run_dir / "remaining_rubrics"
    all_plans = candidate_times(
        record,
        duration,
        {rubric_id: execution["parameters"] for rubric_id, execution in executions.items()},
    )
    # R2 uses a separate native-resolution current-run sampler.  The legacy
    # generic plan remains available for replay/regression, but live execution
    # must bind to dynamic observation/recording-cycle evidence.
    try:
        from .r2_frame_sampling_agent import build_r2_candidate_plan
    except ImportError:
        from r2_frame_sampling_agent import build_r2_candidate_plan  # type: ignore
    if str(executions[2].get("skill_id", "")).startswith("voltmeter."):
        all_plans[2] = build_r2_candidate_plan(
            video_path,
            record,
            duration,
            executions[2]["parameters"],
            evidence_dir / "rubric2_live_frame_manifest.json",
        )
    all_plans[8] = build_r8_cv_candidate_plan(
        video_path,
        record,
        duration,
        executions[8]["parameters"],
        evidence_dir / "rubric8_live_cv_selection.json",
    )
    plans = {rubric_id: all_plans[rubric_id] for rubric_id in rubric_ids}
    source_digest = sha256(video_path)
    checkpoint_path = evidence_dir / "evidence_pre_qwen.json"
    checkpoint = read_json(checkpoint_path) if checkpoint_path.is_file() else {}
    raw_evidence = checkpoint.get("rubrics") if isinstance(checkpoint, dict) else None
    checkpoint_valid = (
        checkpoint.get("algorithm_version") == ALGORITHM_VERSION
        and checkpoint.get("source_video_sha256") == source_digest
        and checkpoint.get("allow_video_calibration", True) == allow_video_calibration
        and checkpoint.get("routing_policy") == (skill_plan or {}).get("routing_policy")
        and checkpoint.get("execution_fingerprints") == execution_fingerprints
        and isinstance(raw_evidence, dict)
        and all(str(rubric_id) in raw_evidence for rubric_id in rubric_ids)
    )
    if checkpoint_valid:
        evidence = {rubric_id: list(raw_evidence[str(rubric_id)]) for rubric_id in rubric_ids}
    else:
        evidence = decode_evidence(
            video_path,
            {0: plans[0]},
            evidence_dir,
            video_id,
            allow_video_calibration=allow_video_calibration,
        )
        if str(executions[2].get("skill_id", "")).startswith("voltmeter."):
            try:
                from .r2_frame_sampling_agent import decode_r2_evidence
            except ImportError:
                from r2_frame_sampling_agent import decode_r2_evidence  # type: ignore
            evidence[2] = decode_r2_evidence(video_path, plans[2], evidence_dir / "rubric2_live")
        else:
            evidence[2] = decode_evidence(
                video_path,
                {2: plans[2]},
                evidence_dir,
                video_id,
                allow_video_calibration=allow_video_calibration,
            )[2]
        write_json(
            checkpoint_path,
            {
                "algorithm_version": ALGORITHM_VERSION,
                "source_video_sha256": source_digest,
                "allow_video_calibration": allow_video_calibration,
                "routing_policy": (skill_plan or {}).get("routing_policy"),
                "execution_fingerprints": execution_fingerprints,
                "skill_executions": {str(key): value for key, value in executions.items()},
                "candidate_plans": plans,
                "rubrics": {str(key): value for key, value in evidence.items()},
            },
        )
    results: dict[int, dict[str, Any]] = {}
    reports: dict[str, Any] = {}
    for rubric_id in rubric_ids:
        rows = evidence[rubric_id]
        if not rows:
            results[rubric_id] = _result("fail", 0.5, "no_decodable_candidate_frames_binary_tie_break", {"candidate_plan": plans[rubric_id]})
            reports[str(rubric_id)] = {"frames": [], "result": results[rubric_id]}
            continue
        execution = executions[rubric_id]
        observation = _call_qwen(
            rubric_id,
            rows,
            model_config,
            evidence_dir / "qwen" / f"rubric_{rubric_id}.json",
            skill_instruction=str(execution["parameters"]["prompt_instruction"]),
            execution_fingerprint=execution["execution_fingerprint"],
        )
        results[rubric_id] = reduce_rubric(
            rubric_id,
            observation,
            rows,
            fusion_parameters=execution["parameters"],
        )
        if rubric_id == 2 and str(execution.get("skill_id", "")).startswith("voltmeter."):
            diagnostics = dict(results[rubric_id].get("diagnostics") or {})
            diagnostics["r2_frame_agent"] = {
                "algorithm_version": "r2_frame_sampling_agent.v1",
                "native_frame_count": len(rows),
                "native_dimensions": sorted(
                    {
                        f"{int(row.get('native_width', 0))}x{int(row.get('native_height', 0))}"
                        for row in rows
                        if row.get("native_width") and row.get("native_height")
                    }
                ),
                "frame_ids": [str(row.get("frame_id")) for row in rows],
                "manifest_path": str((evidence_dir / "rubric2_live_frame_manifest.json").resolve()),
                "roi_mode": execution["parameters"].get("roi_mode"),
                "model_max_edge": execution["parameters"].get("model_max_edge"),
            }
            results[rubric_id]["diagnostics"] = diagnostics
        if rubric_id == 0 and results[rubric_id]["decision"] == "fail":
            workspace_metrics = cleanup_workspace_metrics(rows, evidence_dir / "cleanup_review")
            sequence_review: dict[str, Any] | None = None
            review_error: str | None = None
            try:
                sequence_review = _call_cleanup_sequence_review(
                    rows,
                    model_config,
                    evidence_dir / "qwen" / "rubric_0_cleanup_review.json",
                    workspace_metrics,
                    skill_instruction=str(execution["parameters"]["prompt_instruction"]),
                    execution_fingerprint=execution["execution_fingerprint"],
                )
            except (OSError, RuntimeError, ValueError, KeyError) as exc:
                review_error = f"{type(exc).__name__}: {exc}"
            results[rubric_id] = fuse_cleanup_review(results[rubric_id], sequence_review, workspace_metrics)
            if review_error:
                results[rubric_id].setdefault("diagnostics", {}).setdefault("cleanup_failure_review", {})["sequence_review_error"] = review_error
        reports[str(rubric_id)] = {"frames": rows, "observation": observation, "result": results[rubric_id]}
    specialized_dir = evidence_dir / "rubric8_specialized"
    specialized_error: Exception | None = None
    specialized_requested = (
        enable_specialized_r8
        and executions[8]["skill_id"]
        in {
            "battery.recovery_episode",
            "battery.wiring_transition",
            "battery.broad_transition_search",
        }
    )
    if specialized_requested:
        try:
            results[8] = run_specialized_rubric8(
                video_path=video_path,
                video_id=video_id,
                output_dir=specialized_dir,
                model_config=model_config,
                skill_parameters=executions[8]["parameters"],
                script_path=rubric8_script_path or DEFAULT_R8_SCRIPT,
                action_summary_path=rubric8_action_summary_path or action_path,
            )
            reports["8"] = {
                "algorithm": "resistance_disconnect_battery_sequence_v3_dynamic_roi",
                "result": results[8],
                "fallback_used": False,
            }
        except (OSError, RuntimeError, ValueError, KeyError, subprocess.SubprocessError) as exc:
            specialized_error = exc
    if not specialized_requested or specialized_error is not None:
        exc = specialized_error
        fallback_rows = decode_evidence(
            video_path,
            {8: all_plans[8]},
            evidence_dir / "rubric8_fallback",
            video_id,
            allow_video_calibration=allow_video_calibration,
        )[8]
        if fallback_rows:
            observation = _call_qwen(
                8,
                fallback_rows,
                model_config,
                evidence_dir / "qwen" / "rubric_8_fallback.json",
                skill_instruction=str(executions[8]["parameters"]["prompt_instruction"]),
                execution_fingerprint=executions[8]["execution_fingerprint"],
            )
            results[8] = reduce_rubric(
                8,
                observation,
                fallback_rows,
                fusion_parameters=executions[8]["parameters"],
            )
            if exc is not None:
                results[8].setdefault("diagnostics", {})["specialized_runner_error"] = f"{type(exc).__name__}: {exc}"
            reports["8"] = {
                "frames": fallback_rows,
                "observation": observation,
                "result": results[8],
                "fallback_used": True,
            }
        else:
            results[8] = _result(
                "fail",
                0.5,
                "no_decodable_rubric8_frames_binary_tie_break",
                {"specialized_runner_error": f"{type(exc).__name__}: {exc}" if exc is not None else "live_generic_skill"},
            )
            reports["8"] = {"frames": [], "result": results[8], "fallback_used": True}
    report = {
        "schema_version": "resistance_agent_remaining_evidence.v1",
        "algorithm_version": ALGORITHM_VERSION,
        "video_id": video_id,
        "source_video_id": source_video_id,
        "source_video_path": str(video_path.resolve()),
        "source_video_sha256": source_digest,
        "action_summary_path": str(action_path.resolve()),
        "boundary_summary_path": str(boundary_summary_path.resolve()) if boundary_summary_path else None,
        "boundary_stage_runs_used": boundary_used,
        "rubrics": reports,
        **{f"rubric_{rubric_id}": result for rubric_id, result in results.items()},
        "excel_accessed": False,
        "ground_truth_sent_to_model": False,
        "source_video_unchanged": sha256(video_path) == source_digest,
        "allow_video_calibration": allow_video_calibration,
        "enable_specialized_r8": enable_specialized_r8,
        "historical_fallback_used": bool(
            allow_historical_fallback and fallback is not None and action_path == fallback
        ),
        "routing_policy": (skill_plan or {}).get("routing_policy"),
        "skill_selection": (skill_plan or {}).get("skills", []),
        "skill_executions": {str(key): value for key, value in executions.items()},
        "selection_checkpoint_reused": checkpoint_valid,
    }
    report_path = evidence_dir / "remaining_evidence_report.json"
    write_json(report_path, report)
    return {**{f"rubric_{rubric_id}": result for rubric_id, result in results.items()}, "report_path": str(report_path.resolve())}


run_remaining_rubrics.supports_boundary_summary = True
