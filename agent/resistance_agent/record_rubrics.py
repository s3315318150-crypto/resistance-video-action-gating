#!/usr/bin/env python3
"""Cycle-bound real-video evidence acquisition for Rubrics 7 and 9."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ALGORITHM_VERSION = "r79_cycle_aware_digit_review_v16"
POST_WRITE_REVEAL_SECONDS = 6.0
PAPER_SELECTION_OFFSETS_SECONDS = (-0.5, 1.5, 2.5, 4.5)
PAPER_DENSE_SELECTION_OFFSETS_SECONDS = (2.3, 2.7)
PAPER_ANCHOR_NEIGHBORHOOD_SECONDS = 0.21
PAPER_MODEL_GROUP_LIMIT = 6
PAPER_FIELD_MIN_CONFIDENCE = 0.70
PAPER_FIELD_MIN_DISTINCT_FRAMES = 2
PAPER_DYNAMIC_VIEW_LIMIT = 4
PAPER_MODEL_VIEW_LIMIT = 2
PAPER_MODEL_MAX_EDGE = 2048
PAPER_MODEL_JPEG_QUALITY = 95
PAPER_DIGIT_REVIEW_MAX_EDGE = 2560
PAPER_DIGIT_REVIEW_QUALITY = 95
# The second recording boundary can be reported a few seconds early while the
# camera is still moving. Keep a short in-recording tail so the last three
# model frames come from the stable post-rewire meter view.
METER_POST_RECORDING_SECONDS = 9.0
FIELD_NAMES = ("u1", "i1", "u2", "i2")
# The recording sheet is normally held in the lower half of the view. These
# overlapping views are fallbacks when a white sheet merges into the tabletop
# and contour detection cannot isolate its border.
PAPER_SEARCH_ROIS: tuple[tuple[float, float, float, float], ...] = (
    (0.60, 0.42, 1.00, 1.00),
    (0.30, 0.42, 0.78, 1.00),
    (0.00, 0.42, 0.48, 1.00),
)


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
    return max(1800, min(6000, 1200 + max(0, image_group_count) * 650))


def normalize_decimal(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        match = re.fullmatch(r"\s*(?:about|approx|approximately|约|≈|~)?\s*(\d+(?:\.\d+)?)\s*", value, re.I)
        if not match:
            return None
        number = float(match.group(1))
    else:
        return None
    return number if math.isfinite(number) and number >= 0.0 else None


def one_division_tolerance(role: str, selected_range: Any) -> float:
    numeric_range = normalize_decimal(selected_range)
    if numeric_range is None:
        numeric_range = 3.0 if role == "voltmeter" else 0.6
    floor = 0.05 if role == "voltmeter" else 0.01
    return round(max(floor, numeric_range / 30.0 * 1.25), 6)


def compare_value(paper: Any, meter: Any, role: str, selected_range: Any) -> dict[str, Any]:
    paper_value, meter_value = normalize_decimal(paper), normalize_decimal(meter)
    tolerance = one_division_tolerance(role, selected_range)
    difference = None if paper_value is None or meter_value is None else abs(paper_value - meter_value)
    return {
        "paper_value": paper_value,
        "meter_value": meter_value,
        "absolute_difference": None if difference is None else round(difference, 6),
        "tolerance": tolerance,
        "matched": bool(difference is not None and difference <= tolerance + 1e-9),
    }


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
            nested_path = record.get(key)
            if isinstance(nested_path, str) and Path(nested_path).is_file():
                nested = read_json(Path(nested_path))
                if isinstance(nested.get("observed_stage_runs"), list):
                    return nested
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
        nested_path = record.get("result_path")
        if isinstance(nested_path, str) and Path(nested_path).is_file():
            nested = read_json(Path(nested_path))
            runs = nested.get("source_observed_stage_runs") or nested.get("observed_stage_runs")
            if isinstance(runs, list):
                return {"observed_stage_runs": runs}
    return None


def _broad_cycle_windows(
    runs: list[dict[str, Any]], duration_seconds: float
) -> dict[int, dict[str, Any]]:
    """Build identity-independent current-run recovery windows."""
    cleanup_starts = [
        float(item.get("start_seconds") or duration_seconds)
        for item in runs
        if item.get("stage") == "material_cleanup"
    ]
    active_end = min(cleanup_starts) if cleanup_starts else duration_seconds
    active_starts = [
        float(item.get("start_seconds") or 0.0)
        for item in runs
        if item.get("stage") != "material_cleanup"
    ]
    active_start = min(active_starts) if active_starts else 0.0
    if active_end <= active_start:
        return {}

    windows: dict[int, dict[str, Any]] = {}
    for cycle in (1, 2):
        measurement = next(
            (item for item in runs if item.get("stage") == f"measurement_{cycle}"),
            None,
        )
        if measurement is not None:
            paper_start = min(active_end, float(measurement.get("end_seconds") or active_start))
            paper_end = min(active_end, paper_start + 24.0)
            meter_start = max(active_start, float(measurement.get("start_seconds") or paper_start - 12.0))
            meter_end = max(meter_start, paper_start)
            source_ids = list(measurement.get("event_ids") or [])
        else:
            rewiring = [
                item for item in runs if item.get("stage") == "circuit_rewiring"
            ]
            if cycle == 2 and rewiring:
                paper_start = min(active_end, float(rewiring[-1].get("end_seconds") or active_start))
                paper_end = min(active_end, paper_start + 24.0)
                source_ids = list(rewiring[-1].get("event_ids") or [])
            else:
                span = active_end - active_start
                left_fraction, right_fraction = ((0.35, 0.60) if cycle == 1 else (0.60, 0.90))
                paper_start = active_start + span * left_fraction
                paper_end = active_start + span * right_fraction
                source_ids = []
            meter_end = paper_start
            meter_start = max(active_start, meter_end - 12.0)
        if paper_end <= paper_start:
            continue
        windows[cycle] = {
            "cycle": cycle,
            "recording_stage": f"recording_{cycle}",
            "recording_start_seconds": round(paper_start, 3),
            "recording_end_seconds": round(paper_end, 3),
            "paper_window_seconds": [round(paper_start, 3), round(paper_end, 3)],
            "meter_window_seconds": [round(meter_start, 3), round(meter_end, 3)],
            "meter_window_source": f"current_run_broad_search_{cycle}",
            "source_event_ids": source_ids,
            "broad_search": True,
        }
    return windows


def cycle_windows(
    record: dict[str, Any],
    duration_seconds: float,
    cycle_mode: str = "all_observed_cycles",
) -> dict[int, dict[str, Any]]:
    raw_runs = record.get("observed_stage_runs")
    runs = sorted(
        [item for item in raw_runs if isinstance(item, dict)] if isinstance(raw_runs, list) else [],
        key=lambda item: float(item.get("start_seconds") or 0.0),
    )
    windows: dict[int, dict[str, Any]] = {}

    def is_merged(run: dict[str, Any]) -> bool:
        return str(run.get("stage")) in {"recording_1", "recording_2"} and (
            run.get("stage_semantics") == "measurement_and_recording_cycle"
            or run.get("stage_window_semantics") == "measurement_and_recording_cycle"
            or run.get("merged_stage_semantics") == "measurement_and_recording_cycle"
            or run.get("merged_measurement_recording") is True
            or run.get("merged_stage") is True
        )

    def subintervals(run: dict[str, Any], action_type: str) -> list[dict[str, Any]]:
        field = "measurement_subintervals" if action_type == "measurement_action" else "writing_subintervals"
        raw = run.get(field)
        explicit_field = isinstance(raw, list)
        if not explicit_field:
            raw = run.get("observed_subintervals")
        result: list[dict[str, Any]] = []
        for item in raw if isinstance(raw, list) else []:
            if not isinstance(item, dict):
                continue
            if not explicit_field and item.get("action_type") != action_type:
                continue
            try:
                start = max(0.0, min(duration_seconds, float(item["start_seconds"])))
                end = max(start, min(duration_seconds, float(item["end_seconds"])))
            except (KeyError, TypeError, ValueError):
                continue
            if end - start >= 0.2:
                result.append({**item, "start_seconds": start, "end_seconds": end})
        return result

    for cycle in (1, 2):
        stage = f"recording_{cycle}"
        matches = [item for item in runs if item.get("stage") == stage]
        if not matches:
            continue
        merged = any(is_merged(item) for item in matches)
        start = max(0.0, min(float(item.get("start_seconds") or 0.0) for item in matches))
        end = min(duration_seconds, max(float(item.get("end_seconds") or start) for item in matches))
        later = [
            float(item.get("start_seconds") or duration_seconds)
            for item in runs
            if str(item.get("stage") or "").startswith("recording_")
            and str(item.get("stage")) != stage
            and float(item.get("start_seconds") or 0.0) > end + 1e-6
        ]
        next_start = min(later) if later else duration_seconds
        source_event_ids = sorted(
            {
                str(event_id)
                for item in matches
                for event_id in (item.get("event_ids") or [])
                if isinstance(event_id, str)
            }
        )

        if merged:
            measurement_parts = [part for item in matches for part in subintervals(item, "measurement_action")]
            writing_parts = [part for item in matches for part in subintervals(item, "writing_action")]
            if measurement_parts:
                meter_start = min(float(item["start_seconds"]) for item in measurement_parts)
                meter_end = max(float(item["end_seconds"]) for item in measurement_parts)
                meter_source = f"{stage}_measurement_subinterval"
            else:
                meter_start, meter_end = start, end
                meter_source = f"{stage}_merged_cycle_fallback"
            if writing_parts:
                paper_start = min(float(item["start_seconds"]) for item in writing_parts)
                paper_end = max(float(item["end_seconds"]) for item in writing_parts)
                paper_source = f"{stage}_writing_subinterval"
            else:
                paper_start, paper_end = start, end
                paper_source = f"{stage}_merged_cycle_fallback"
            later_stage_starts = [
                float(item.get("start_seconds") or duration_seconds)
                for item in runs
                if str(item.get("stage")) != stage
                and float(item.get("start_seconds") or 0.0) > paper_end + 1e-6
            ]
            next_stage_start = min(later_stage_starts) if later_stage_starts else duration_seconds
            reveal_limit = max(paper_end, next_stage_start - 1.0)
            reveal_end = min(
                duration_seconds,
                paper_end + POST_WRITE_REVEAL_SECONDS,
                reveal_limit,
                next_start,
            )
            windows[cycle] = {
                "cycle": cycle,
                "recording_stage": stage,
                "recording_start_seconds": round(paper_start, 3),
                "recording_end_seconds": round(paper_end, 3),
                "paper_window_seconds": [round(paper_start, 3), round(reveal_end, 3)],
                "meter_window_seconds": [round(meter_start, 3), round(meter_end, 3)],
                "meter_window_source": meter_source,
                "paper_window_source": paper_source,
                "merged_measurement_recording": True,
                "measurement_subintervals": measurement_parts,
                "writing_subintervals": writing_parts,
                "source_event_ids": source_event_ids,
            }
            continue

        # Legacy seven-stage input keeps its established pre-recording behavior.
        match = matches[0]
        start = max(0.0, float(match.get("start_seconds") or 0.0))
        end = min(duration_seconds, float(match.get("end_seconds") or start))
        reveal_end = min(duration_seconds, end + POST_WRITE_REVEAL_SECONDS, next_start)
        preceding = [item for item in runs if float(item.get("end_seconds") or 0.0) <= start + 1e-6]
        explicit = [item for item in preceding if item.get("stage") == f"measurement_{cycle}"]
        if explicit:
            anchor = explicit[-1]
            meter_start = max(0.0, float(anchor.get("start_seconds") or start - 12.0) - 2.0)
        else:
            meter_start = max(0.0, start - 12.0)
            if cycle == 2:
                rewiring = [item for item in preceding if item.get("stage") == "circuit_rewiring"]
                if rewiring:
                    meter_start = max(meter_start, float(rewiring[-1].get("end_seconds") or meter_start))
        windows[cycle] = {
            "cycle": cycle,
            "recording_stage": stage,
            "recording_start_seconds": round(start, 3),
            "recording_end_seconds": round(end, 3),
            "paper_window_seconds": [round(start, 3), round(reveal_end, 3)],
            "meter_window_seconds": [round(meter_start, 3), round(start, 3)],
            "meter_window_source": f"measurement_{cycle}" if explicit else f"derived_pre_recording_{cycle}",
            "source_event_ids": source_event_ids,
        }
    if cycle_mode == "broad_cycle_search":
        recovered = _broad_cycle_windows(runs, duration_seconds)
        for cycle, value in recovered.items():
            windows.setdefault(cycle, value)
    return windows


def paper_timestamps(window: dict[str, Any], maximum: int = 10) -> list[float]:
    start, end = (float(value) for value in window["paper_window_seconds"])
    if end <= start:
        return [start]
    recording_end = float(window.get("recording_end_seconds") or end)
    targets = [recording_end + offset for offset in PAPER_SELECTION_OFFSETS_SECONDS]
    # Exact anchors avoid missing a brief sharp reveal between a coarse uniform
    # grid. Adjacent frames around the two post-write anchors support blur and
    # occlusion ranking without becoming independent model votes.
    values = list(targets)
    for offset in (2.5, 4.5):
        values.extend((recording_end + offset - 0.2, recording_end + offset + 0.2))
    bounded = sorted({round(min(end, max(start, float(value))), 3) for value in values})
    return bounded[:maximum]


def meter_timestamps(
    window: dict[str, Any],
    maximum: int = 5,
) -> list[float]:
    start, end = (float(value) for value in window["meter_window_seconds"])
    recording_end = float(window.get("recording_end_seconds") or end)
    extend_into_recording = window.get("meter_window_source") == "derived_pre_recording_2"
    evidence_end = min(recording_end, end + METER_POST_RECORDING_SECONDS) if extend_into_recording else end
    if evidence_end <= start:
        return [start]
    margin_end = max(start, evidence_end - 0.5)
    lookback = 8.0 if window.get("meter_window_source") == "derived_pre_recording_1" else 5.0
    values = list(np.linspace(max(start, margin_end - lookback), margin_end, maximum))
    return sorted({round(float(value), 3) for value in values})


def _enhance(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    light, channel_a, channel_b = cv2.split(lab)
    light = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(light)
    enhanced = cv2.cvtColor(cv2.merge((light, channel_a, channel_b)), cv2.COLOR_LAB2BGR)
    blurred = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
    return cv2.addWeighted(enhanced, 1.35, blurred, -0.35, 0)


def _enhance_ink(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 35, 35)
    gray = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
    blurred = cv2.GaussianBlur(gray, (0, 0), 1.1)
    sharpened = cv2.addWeighted(gray, 1.65, blurred, -0.65, 0)
    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)


def pointer_geometry_from_face(image: np.ndarray, calibration: dict[str, Any]) -> dict[str, Any] | None:
    """Read a calibrated pointer angle without using paper values or labels."""
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 160)
    lines = cv2.HoughLinesP(
        edges,
        1,
        math.pi / 1800.0,
        70,
        minLineLength=max(80, round(min(height, width) * 0.07)),
        maxLineGap=25,
    )
    if lines is None:
        return None
    x_min, x_max = (float(value) for value in calibration["line_midpoint_x_norm"])
    y_max = float(calibration["line_midpoint_y_max"])
    angle_min, angle_max = (float(value) for value in calibration["line_abs_angle_degrees"])
    candidates: list[tuple[float, float, list[int]]] = []
    for raw in lines:
        x1, y1, x2, y2 = (int(value) for value in raw.reshape(-1)[:4])
        midpoint_x = (x1 + x2) / 2.0 / width
        midpoint_y = (y1 + y2) / 2.0 / height
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        length = math.hypot(x2 - x1, y2 - y1)
        if x_min <= midpoint_x <= x_max and midpoint_y <= y_max and angle_min <= abs(angle) <= angle_max:
            if angle > 0.0:
                angle -= 180.0
            candidates.append((length, angle, [x1, y1, x2, y2]))
    if not candidates:
        return None
    length, pointer_angle, line_xyxy = max(candidates, key=lambda item: item[0])
    pivot_x, pivot_y = (float(value) for value in calibration["pivot_norm"])
    pivot = (pivot_x * width, pivot_y * height)
    anchor_angles: list[float] = []
    for point_x, point_y in calibration["anchor_points_norm"]:
        anchor_angles.append(
            math.degrees(math.atan2(float(point_y) * height - pivot[1], float(point_x) * width - pivot[0]))
        )
    anchor_values = [float(value) for value in calibration["anchor_values"]]
    angle_span = anchor_angles[1] - anchor_angles[0]
    if abs(angle_span) < 1e-6:
        return None
    value_raw = anchor_values[0] + (pointer_angle - anchor_angles[0]) / angle_span * (
        anchor_values[1] - anchor_values[0]
    )
    selected_range = float(calibration["selected_range"])
    if not -0.1 <= value_raw <= selected_range + 0.1:
        return None
    minor_division = float(calibration["minor_division"])
    value = round(round(value_raw / minor_division) * minor_division, 6)
    return {
        "status": "pointer_scale_candidate",
        "selected_range": selected_range,
        "value": value,
        "value_raw": round(value_raw, 6),
        "minor_division": minor_division,
        "pointer_angle_degrees": round(pointer_angle, 6),
        "anchor_angles_degrees": [round(value, 6) for value in anchor_angles],
        "line_xyxy": line_xyxy,
        "line_length_pixels": round(length, 3),
        "confidence": 0.9,
    }


def _paper_candidates(frame: np.ndarray) -> list[dict[str, Any]]:
    height, width = frame.shape[:2]
    scale = min(1.0, 1200.0 / max(height, width))
    small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 0, 110]), np.array([179, 75, 255]))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output: list[dict[str, Any]] = []
    image_area = float(small.shape[0] * small.shape[1])
    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        area = box_width * box_height
        if area < image_area * 0.006 or area > image_area * 0.55:
            continue
        aspect = box_width / max(1.0, float(box_height))
        if aspect < 0.45 or aspect > 4.5:
            continue
        crop = small[y : y + box_height, x : x + box_width]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        edge_ratio = float(np.mean(cv2.Canny(gray, 70, 150) > 0))
        bright_ratio = float(np.mean(gray > 145))
        rectangularity = float(cv2.contourArea(contour)) / max(1.0, area)
        score = 0.38 * bright_ratio + 0.30 * min(1.0, edge_ratio / 0.09) + 0.20 * rectangularity + 0.12 * min(1.0, area / (image_area * 0.12))
        pad_x, pad_y = int(box_width * 0.12), int(box_height * 0.18)
        box = [
            max(0, int(round((x - pad_x) / scale))),
            max(0, int(round((y - pad_y) / scale))),
            min(width, int(round((x + box_width + pad_x) / scale))),
            min(height, int(round((y + box_height + pad_y) / scale))),
        ]
        output.append(
            {
                "bbox_xyxy": box,
                "score": round(score, 6),
                "edge_ratio": round(edge_ratio, 6),
                "bright_ratio": round(bright_ratio, 6),
                "detector": "light_contour",
            }
        )
    output.extend(_paper_writing_context_candidates(frame))
    output.extend(_paper_text_context_candidates(frame))
    return sorted(output, key=lambda item: float(item["score"]), reverse=True)[:12]


def _candidate_context_score(image: np.ndarray) -> dict[str, float]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blackhat = cv2.morphologyEx(
        gray,
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25)),
    )
    white_ratio = float(np.mean((hsv[:, :, 1] < 90) & (hsv[:, :, 2] > 105)))
    saturated_ratio = float(np.mean(hsv[:, :, 1] > 125))
    text_ratio = float(np.mean(blackhat > 14))
    edge_ratio = float(np.mean(cv2.Canny(gray, 70, 150) > 0))
    score = (
        0.34 * min(1.0, text_ratio / 0.08)
        + 0.30 * white_ratio
        + 0.20 * min(1.0, edge_ratio / 0.10)
        - 0.18 * saturated_ratio
    )
    return {
        "score": round(float(score), 6),
        "text_ratio": round(text_ratio, 6),
        "edge_ratio": round(edge_ratio, 6),
        "bright_ratio": round(white_ratio, 6),
        "saturated_ratio": round(saturated_ratio, 6),
    }


def _paper_writing_context_candidates(frame: np.ndarray) -> list[dict[str, Any]]:
    """Use current-frame skin components to keep the sheet near a writing hand."""
    height, width = frame.shape[:2]
    scale = min(1.0, 1200.0 / max(height, width))
    small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ycrcb = cv2.cvtColor(small, cv2.COLOR_BGR2YCrCb)
    skin = cv2.inRange(
        ycrcb,
        np.array([40, 132, 75], dtype=np.uint8),
        np.array([255, 180, 135], dtype=np.uint8),
    )
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    contours, _ = cv2.findContours(skin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = float(small.shape[0] * small.shape[1])
    output: list[dict[str, Any]] = []
    for contour in contours:
        relative_area = float(cv2.contourArea(contour)) / image_area
        if not 0.003 <= relative_area <= 0.12:
            continue
        x, y, box_width, box_height = cv2.boundingRect(contour)
        pad_x = max(int(small.shape[1] * 0.08), int(box_width * 0.65))
        pad_y = max(int(small.shape[0] * 0.08), int(box_height * 0.55))
        left = max(0, x - pad_x)
        top = max(0, y - pad_y)
        right = min(small.shape[1], x + box_width + pad_x)
        bottom = min(small.shape[0], y + box_height + pad_y)
        crop = small[top:bottom, left:right]
        if crop.size == 0:
            continue
        metrics = _candidate_context_score(crop)
        metrics["score"] = round(metrics["score"] + min(0.12, relative_area), 6)
        output.append(
            {
                "bbox_xyxy": [
                    int(round(left / scale)),
                    int(round(top / scale)),
                    int(round(right / scale)),
                    int(round(bottom / scale)),
                ],
                "detector": "writing_hand_context",
                "skin_component_area_ratio": round(relative_area, 6),
                **metrics,
            }
        )
    return sorted(output, key=lambda item: float(item["score"]), reverse=True)[:3]


def _paper_text_context_candidates(frame: np.ndarray) -> list[dict[str, Any]]:
    """Find dark local strokes on a light, low-saturation current-frame area."""
    height, width = frame.shape[:2]
    scale = min(1.0, 1200.0 / max(height, width))
    small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    blackhat = cv2.morphologyEx(
        gray,
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25)),
    )
    mask = ((blackhat > 14) & (hsv[:, :, 1] < 140) & (hsv[:, :, 2] > 70)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5)),
        iterations=2,
    )
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 7)))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = float(small.shape[0] * small.shape[1])
    output: list[dict[str, Any]] = []
    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        area = box_width * box_height
        if not image_area * 0.00035 <= area <= image_area * 0.035:
            continue
        center_x, center_y = x + box_width / 2.0, y + box_height / 2.0
        target_width = max(int(small.shape[1] * 0.16), int(box_width * 2.2))
        target_height = max(int(small.shape[0] * 0.18), int(box_height * 2.2))
        left = max(0, int(center_x - target_width / 2.0))
        top = max(0, int(center_y - target_height / 2.0))
        right = min(small.shape[1], left + target_width)
        bottom = min(small.shape[0], top + target_height)
        crop = small[top:bottom, left:right]
        if crop.size == 0:
            continue
        metrics = _candidate_context_score(crop)
        if metrics["bright_ratio"] < 0.20:
            continue
        output.append(
            {
                "bbox_xyxy": [
                    int(round(left / scale)),
                    int(round(top / scale)),
                    int(round(right / scale)),
                    int(round(bottom / scale)),
                ],
                "detector": "local_text_context",
                **metrics,
            }
        )
    return sorted(output, key=lambda item: float(item["score"]), reverse=True)[:6]


def _box_iou(left: list[int], right: list[int]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    left_area = max(0, left[2] - left[0]) * max(0, left[3] - left[1])
    right_area = max(0, right[2] - right[0]) * max(0, right[3] - right[1])
    union = left_area + right_area - intersection
    return float(intersection) / union if union else 0.0


def _write_jpeg(path: Path, image: np.ndarray, quality: int = 94) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image, [int(cv2.IMWRITE_JPEG_QUALITY), quality]):
        raise RuntimeError(f"unable to write image: {path}")


def _dynamic_paper_field_view(
    frame: np.ndarray,
    candidate: dict[str, Any],
    evidence_dir: Path,
    stem: str,
) -> dict[str, Any] | None:
    box = candidate.get("bbox_xyxy")
    if not isinstance(box, list) or len(box) != 4:
        return None
    left, top, right, bottom = (int(value) for value in box)
    crop = frame[top:bottom, left:right]
    if crop.size == 0:
        return None
    prepared = _enhance(crop)
    ink = _enhance_ink(crop)
    max_edge = max(prepared.shape[:2])
    if max_edge < 1800:
        resize_scale = min(3.0, 1800.0 / max_edge)
        prepared = cv2.resize(
            prepared,
            None,
            fx=resize_scale,
            fy=resize_scale,
            interpolation=cv2.INTER_CUBIC,
        )
        ink = cv2.resize(
            ink,
            None,
            fx=resize_scale,
            fy=resize_scale,
            interpolation=cv2.INTER_CUBIC,
        )
    path = evidence_dir / "paper_field_rois" / f"{stem}_dynamic_fields.jpg"
    ink_path = evidence_dir / "paper_field_rois" / f"{stem}_dynamic_fields_ink.jpg"
    _write_jpeg(path, prepared, 96)
    _write_jpeg(ink_path, ink, 96)
    return {
        "bbox_xyxy": [left, top, right, bottom],
        "roi_path": str(path.resolve()),
        "ink_roi_path": str(ink_path.resolve()),
        "sharpness": round(
            float(
                cv2.Laplacian(
                    cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F
                ).var()
            ),
            6,
        ),
        "dynamic": True,
        "candidate_score": float(candidate.get("score") or 0.0),
        "detector": str(candidate.get("detector") or "unknown"),
    }


def _dynamic_paper_field_views(
    frame: np.ndarray,
    candidates: list[dict[str, Any]],
    evidence_dir: Path,
    stem: str,
    limit: int = PAPER_DYNAMIC_VIEW_LIMIT,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for detector in (
        "writing_hand_context",
        "local_text_context",
        "light_contour",
        "unknown",
    ):
        detector_rows = sorted(
            [
                item
                for item in candidates
                if str(item.get("detector") or "unknown") == detector
            ],
            key=lambda item: float(item.get("score") or 0.0),
            reverse=True,
        )
        detector_limit = 2 if detector != "light_contour" else 1
        if detector == "writing_hand_context" and len(detector_rows) > 1:
            smallest = min(
                detector_rows,
                key=lambda item: float(item.get("skin_component_area_ratio") or 1.0),
            )
            detector_rows = [detector_rows[0], smallest] + [
                item for item in detector_rows[1:] if item is not smallest
            ]
        for item in detector_rows:
            box = item.get("bbox_xyxy")
            if not isinstance(box, list) or len(box) != 4:
                continue
            if any(_box_iou(box, value["bbox_xyxy"]) > 0.82 for value in selected):
                continue
            selected.append(item)
            if len([value for value in selected if value.get("detector") == detector]) >= detector_limit:
                break
        if len(selected) >= limit:
            break
    views: list[dict[str, Any]] = []
    for index, item in enumerate(selected[:limit], start=1):
        view = _dynamic_paper_field_view(
            frame, item, evidence_dir, f"{stem}_view_{index:02d}"
        )
        if view is not None:
            views.append(view)
    return views


def _decode_evidence(
    video_path: Path,
    windows: dict[int, dict[str, Any]],
    evidence_dir: Path,
    paper_max_samples: int = 10,
    meter_max_samples: int = 5,
    dynamic_paper_candidates: bool = True,
    dynamic_meter_candidates: bool = True,
    candidate_crops_per_frame: int = 4,
) -> dict[int, dict[str, list[dict[str, Any]]]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"unable to open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    records: dict[int, dict[str, list[dict[str, Any]]]] = {}
    try:
        for cycle, window in windows.items():
            paper_rows: list[dict[str, Any]] = []
            for timestamp in paper_timestamps(window, maximum=paper_max_samples):
                capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
                ok, frame = capture.read()
                if not ok or frame is None:
                    continue
                frame_number = int(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                actual = frame_number / fps if fps > 0 else timestamp
                stem = f"cycle_{cycle}_paper_{frame_number:08d}_{actual:010.3f}s"
                panorama_path = evidence_dir / "paper_frames" / f"{stem}.jpg"
                _write_jpeg(panorama_path, frame, 92)
                candidates = _paper_candidates(frame) if dynamic_paper_candidates else []
                candidate_rows: list[dict[str, Any]] = []
                for index, candidate in enumerate(candidates, start=1):
                    left, top, right, bottom = candidate["bbox_xyxy"]
                    crop = frame[top:bottom, left:right]
                    if crop.size == 0:
                        continue
                    enhanced = _enhance(crop)
                    max_edge = max(enhanced.shape[:2])
                    if max_edge < 1800:
                        scale = min(3.0, 1800.0 / max_edge)
                        enhanced = cv2.resize(enhanced, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                    path = evidence_dir / "paper_rois" / f"{stem}_candidate_{index:02d}.jpg"
                    _write_jpeg(path, enhanced, 95)
                    candidate_rows.append({**candidate, "roi_path": str(path.resolve())})
                search_rows: list[dict[str, Any]] = []
                height, width = frame.shape[:2]
                for index, normalized in enumerate(
                    PAPER_SEARCH_ROIS if dynamic_paper_candidates else (), start=1
                ):
                    x1, y1, x2, y2 = normalized
                    box = [int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)]
                    left, top, right, bottom = box
                    crop = frame[top:bottom, left:right]
                    if crop.size == 0:
                        continue
                    enhanced = _enhance(crop)
                    path = evidence_dir / "paper_search_rois" / f"{stem}_search_{index:02d}.jpg"
                    _write_jpeg(path, enhanced, 95)
                    search_rows.append(
                        {
                            "bbox_xyxy": box,
                            "normalized_xyxy": list(normalized),
                            "roi_path": str(path.resolve()),
                        }
                    )
                field_views = (
                    _dynamic_paper_field_views(frame, candidate_rows, evidence_dir, stem)
                    if candidate_rows
                    else []
                )
                field_row = field_views[0] if field_views else None
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                paper_rows.append(
                    {
                        "cycle": cycle,
                        "frame_id": f"frame_{frame_number:08d}",
                        "frame_number": frame_number,
                        "timestamp_seconds": round(actual, 6),
                        "panorama_path": str(panorama_path.resolve()),
                        "paper_candidates": candidate_rows,
                        "paper_search_views": search_rows,
                        "paper_calibrated_view": None,
                        "paper_field_view": field_row,
                        "paper_field_views": field_views,
                        "sharpness": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 6),
                    }
                )
            meter_rows: list[dict[str, Any]] = []
            for timestamp in meter_timestamps(
                window,
                maximum=meter_max_samples,
            ):
                capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
                ok, frame = capture.read()
                if not ok or frame is None:
                    continue
                frame_number = int(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                actual = frame_number / fps if fps > 0 else timestamp
                height, width = frame.shape[:2]
                crop = frame
                crop = _enhance(crop)
                path = evidence_dir / "meter_frames" / f"cycle_{cycle}_meter_{frame_number:08d}_{actual:010.3f}s.jpg"
                _write_jpeg(path, crop, 94)
                role_views: dict[str, dict[str, Any]] = {}
                face_views: dict[str, dict[str, Any]] = {}
                dynamic_candidates: list[dict[str, Any]] = []
                if dynamic_meter_candidates:
                    # Detect meter candidates from the current frame. The
                    # detector supplies frame-tied crops, so no video-specific
                    # coordinate or geometry is imported into live execute.
                    try:
                        from . import meter_rubrics as meter_module
                    except ImportError:
                        import meter_rubrics as meter_module  # type: ignore
                    exported = meter_module._export_candidates(
                        {"frame_path": str(path), "sharpness": float(cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())},
                        evidence_dir / "dynamic_meter_candidates",
                    )
                    dynamic_candidates = [
                        item
                        for item in exported.get("candidates", [])
                        if isinstance(item, dict)
                    ][:candidate_crops_per_frame]
                    for candidate in dynamic_candidates:
                        role = str(candidate.get("role_hint") or "")
                        if role not in {"ammeter", "voltmeter"}:
                            continue
                        role_views.setdefault(
                            role,
                            {
                                "image_path": candidate.get("enhanced_path") or candidate.get("wide_path"),
                                "candidate_id": candidate.get("candidate_id"),
                                "dynamic": True,
                            },
                        )
                meter_rows.append(
                    {
                        "cycle": cycle,
                        "frame_id": f"frame_{frame_number:08d}",
                        "frame_number": frame_number,
                        "timestamp_seconds": round(actual, 6),
                        "meter_roi_normalized_xyxy": None,
                        "image_path": str(path.resolve()),
                        "role_views": role_views,
                        "face_views": face_views,
                        "dynamic_meter_candidates": dynamic_candidates,
                    }
                )
            records[cycle] = {"paper": paper_rows, "meter": meter_rows}
    finally:
        capture.release()
    return records


def image_data_url(
    path: Path,
    max_edge: int = 960,
    quality: int = 80,
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


def _paper_prompt(
    cycle: int, rows: list[dict[str, Any]], skill_instruction: str = ""
) -> str:
    frame_list = ", ".join(f"{index}={row['frame_id']}@{row['timestamp_seconds']:.3f}s" for index, row in enumerate(rows, start=1))
    target = "U1/I1" if cycle == 1 else "U2/I2"
    return f"""Read handwritten {target} from recording_{cycle}. Frames: {frame_list}.
Skill instruction: {skill_instruction}
Each image group contains one panorama and several current-frame candidate crops. Candidate crops can come from light contours, local text, or the area beside a visible writing hand. Color and grayscale high-contrast renderings of one candidate are the same pixels and remain one temporal vote.
Use visible ink in each image group independently. Keep U1/I1 and U2/I2 separate; for cycle 2 inspect the second row.
Transcribe every visible digit after the decimal point. Check the color and high-contrast views together before deciding whether a decimal has one or two digits; do not merge adjacent digits into a different single digit.
Find the small loose white recording sheet, not the white experiment table or meter face. A crop may include only part of the sheet.
Set paper_visible=false only when no view in that image group shows the recording sheet. All views in one group remain one vote.
Never infer that a hidden value stayed unchanged from another group. A field is visible only when its label and digits can be read in that group.
For cycle 1, report only the row whose visible labels are U1 and I1. For cycle 2, report only the lower row whose visible labels are U2 and I2; do not copy U1/I1 digits into U2/I2.
Return exactly this JSON shape with one observation per supplied image group:
{{
  "observations": [
    {{
      "image_group": 1,
      "frame_id": "frame_00000000",
      "paper_visible": true,
      "u1": {{"label_visible": true, "raw_text": "U1=1.5V", "value": 1.5, "confidence": 0.0}},
      "i1": {{"label_visible": true, "raw_text": "I1=0.22A", "value": 0.22, "confidence": 0.0}},
      "u2": {{"label_visible": false, "raw_text": null, "value": null, "confidence": 0.0}},
      "i2": {{"label_visible": false, "raw_text": null, "value": null, "confidence": 0.0}},
      "evidence": "visible ink"
    }}
  ],
  "evidence": "summary"
}}"""


def _meter_prompt(
    cycle: int, rows: list[dict[str, Any]], skill_instruction: str = ""
) -> str:
    frame_list = ", ".join(f"{index}={row['frame_id']}@{row['timestamp_seconds']:.3f}s" for index, row in enumerate(rows, start=1))
    return f"""Read the analog ammeter and voltmeter immediately before recording_{cycle}. Frames: {frame_list}.
Skill instruction: {skill_instruction}
Each image group is one real frame. Its views are ordered as: broad meter view, broad ammeter crop, enlarged ammeter face when available, broad voltmeter crop, enlarged voltmeter face when available.
Use visible pixels only and compare adjacent frames. Read occupied-terminal range and needle value.
The ammeter has outer 0-3 and inner 0-0.6 scales; the middle 0.6 terminal selects the inner scale. The voltmeter has outer 0-15 and inner 0-3 scales; the middle 3 terminal selects the inner scale.
The printed scale zero is at the left end of its arc, not at image vertical. Read the visible adjacent labeled marks, count the intervening minor divisions, and calculate the needle value from those marks.
For every visible needle, first identify the adjacent printed major values on its left and right, count the minor intervals between them, then count how many intervals the needle is past the left major value. Calculate the final value from those counts.
Count minor divisions instead of rounding to a major mark: on the selected inner scales one minor division is normally 0.02 A or 0.1 V. Keep a halfway reading between 0 and 0.2 as 0.10 rather than rounding it to 0.2.
Do not treat a wire, reflection, printed tick, or meter-case edge as the needle. Prefer repeated energized readings; ignore an out-of-view frame instead of converting it to zero.
All selected_range and value fields must be JSON numbers or null. The nulls below describe the schema and are not expected readings. Return exactly this JSON shape; do not output pass/fail:
{{
  "per_frame": [
    {{"image_group": 1, "frame_id": "frame_00000000", "energized": true,
      "ammeter": {{"visible": true, "selected_range": null, "left_major": null, "right_major": null, "minor_intervals_between": null, "minor_intervals_after_left": null, "value": null, "confidence": 0.0, "evidence": "visible needle and terminals"}},
      "voltmeter": {{"visible": true, "selected_range": null, "left_major": null, "right_major": null, "minor_intervals_between": null, "minor_intervals_after_left": null, "value": null, "confidence": 0.0, "evidence": "visible needle and terminals"}}}}
  ],
  "consensus": {{
    "ammeter": {{"selected_range": null, "value": null, "confidence": 0.0, "supporting_frame_ids": ["frame_00000000"], "evidence": "stable visual reading and division calculation"}},
    "voltmeter": {{"selected_range": null, "value": null, "confidence": 0.0, "supporting_frame_ids": ["frame_00000000"], "evidence": "stable visual reading and division calculation"}}
  }},
  "evidence": "visible reading"
}}"""


def _paper_digit_review_prompt(
    field: str, rows: list[dict[str, Any]], skill_instruction: str = ""
) -> str:
    frame_list = ", ".join(
        f"{index}={row['frame_id']}@{row['timestamp_seconds']:.3f}s"
        for index, row in enumerate(rows, start=1)
    )
    label = field.upper()
    return f"""Independently re-read only handwritten {label}. Frames: {frame_list}.
Skill instruction: {skill_instruction}
Each image group is one real frame and contains color-enhanced and high-contrast views of the same paper pixels. Treat all views in one image group as one vote.
Use visible ink only. Do not use experiment expectations, meter readings, another field, or an earlier OCR answer to choose the value.
Transcribe every digit after the decimal point. For the final digit, explicitly compare its strokes: a handwritten 0 normally forms a closed oval without an inward hook, while a handwritten 6 normally has a loop plus an inward curl or upper entry stroke. Also consider 1, 2, 3, 4, 5, 7, 8, and 9 when their strokes fit better. Do not assume the answer is 0 or 6.
Set label_visible=false when the {label} label and its complete numeric value cannot be read in that image group. Preserve trailing zeroes in decimal_digits and raw_text even though JSON value is numeric.
Return exactly this JSON shape with one observation per supplied image group:
{{
  "observations": [
    {{
      "image_group": 1,
      "frame_id": "frame_00000000",
      "field": "{field}",
      "label_visible": true,
      "raw_text": "{label}=0.10A",
      "value": 0.10,
      "decimal_digits": "10",
      "final_digit_shape": "closed oval without inward hook",
      "final_digit_closed_loop": true,
      "final_digit_inward_hook": false,
      "confidence": 0.0,
      "evidence": "visible stroke description"
    }}
  ],
  "evidence": "independent visual re-read"
}}"""


def _validate_field(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name}_not_object")
    confidence = value.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        confidence = 0.0
    return {
        "label_visible": bool(value.get("label_visible")),
        "raw_text": str(value["raw_text"]) if value.get("raw_text") is not None else None,
        "value": normalize_decimal(value.get("value")),
        "confidence": round(max(0.0, min(1.0, float(confidence))), 4),
    }


def validate_paper_observation(value: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    observations = value.get("observations")
    if not isinstance(observations, list):
        raise ValueError("paper_observations_missing")
    expected = {index: row["frame_id"] for index, row in enumerate(rows, start=1)}
    parsed: list[dict[str, Any]] = []
    for item in observations:
        if not isinstance(item, dict) or item.get("image_group") not in expected:
            raise ValueError("paper_image_group_invalid")
        group = int(item["image_group"])
        parsed.append(
            {
                "image_group": group,
                "frame_id": expected[group],
                "model_frame_id": str(item.get("frame_id") or ""),
                "frame_id_corrected_from_group": item.get("frame_id") != expected[group],
                "paper_visible": bool(item.get("paper_visible")),
                **{name: _validate_field(item.get(name), name) for name in FIELD_NAMES},
                "evidence": str(item.get("evidence") or ""),
            }
        )
    if {item["image_group"] for item in parsed} != set(expected):
        raise ValueError("paper_image_groups_incomplete")
    return {"observations": parsed, "evidence": str(value.get("evidence") or "")}


def validate_meter_observation(value: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    consensus = value.get("consensus")
    if not isinstance(consensus, dict):
        raise ValueError("meter_consensus_missing")
    parsed_consensus: dict[str, Any] = {}
    known_ids = {row["frame_id"] for row in rows}
    for role in ("ammeter", "voltmeter"):
        item = consensus.get(role)
        if not isinstance(item, dict):
            raise ValueError(f"meter_{role}_missing")
        confidence = item.get("confidence")
        parsed_consensus[role] = {
            "selected_range": normalize_decimal(item.get("selected_range")),
            "value": normalize_decimal(item.get("value")),
            "confidence": round(max(0.0, min(1.0, float(confidence or 0.0))), 4),
            "supporting_frame_ids": [str(frame_id) for frame_id in item.get("supporting_frame_ids") or [] if str(frame_id) in known_ids],
            "evidence": str(item.get("evidence") or ""),
        }
    return {"per_frame": value.get("per_frame") if isinstance(value.get("per_frame"), list) else [], "consensus": parsed_consensus, "evidence": str(value.get("evidence") or "")}


def validate_paper_digit_review(
    value: dict[str, Any],
    rows: list[dict[str, Any]],
    field: str,
) -> dict[str, Any]:
    observations = value.get("observations")
    if not isinstance(observations, list):
        raise ValueError("paper_digit_review_observations_missing")
    expected = {index: row["frame_id"] for index, row in enumerate(rows, start=1)}
    parsed: list[dict[str, Any]] = []
    for item in observations:
        if not isinstance(item, dict) or item.get("image_group") not in expected:
            raise ValueError("paper_digit_review_image_group_invalid")
        if str(item.get("field") or "").lower() != field:
            raise ValueError("paper_digit_review_field_mismatch")
        group = int(item["image_group"])
        confidence = item.get("confidence")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            confidence = 0.0
        decimal_digits = item.get("decimal_digits")
        if decimal_digits is not None and not re.fullmatch(r"\d+", str(decimal_digits)):
            decimal_digits = None
        parsed.append(
            {
                "image_group": group,
                "frame_id": expected[group],
                "model_frame_id": str(item.get("frame_id") or ""),
                "frame_id_corrected_from_group": item.get("frame_id") != expected[group],
                "field": field,
                "label_visible": bool(item.get("label_visible")),
                "raw_text": str(item["raw_text"]) if item.get("raw_text") is not None else None,
                "value": normalize_decimal(item.get("value")),
                "decimal_digits": str(decimal_digits) if decimal_digits is not None else None,
                "final_digit_shape": str(item.get("final_digit_shape") or ""),
                "final_digit_closed_loop": (
                    item.get("final_digit_closed_loop")
                    if isinstance(item.get("final_digit_closed_loop"), bool)
                    else None
                ),
                "final_digit_inward_hook": (
                    item.get("final_digit_inward_hook")
                    if isinstance(item.get("final_digit_inward_hook"), bool)
                    else None
                ),
                "confidence": round(max(0.0, min(1.0, float(confidence))), 4),
                "evidence": str(item.get("evidence") or ""),
            }
        )
    if {item["image_group"] for item in parsed} != set(expected):
        raise ValueError("paper_digit_review_image_groups_incomplete")
    return {"field": field, "observations": parsed, "evidence": str(value.get("evidence") or "")}


def _call_qwen(
    prompt: str,
    media_groups: list[list[Path]],
    model_config: dict[str, Any],
    raw_path: Path,
    validator: Any,
    rows: list[dict[str, Any]],
    image_max_edge: int = 960,
    execution_fingerprint: str | None = None,
    image_quality: int = 80,
    lossless: bool = False,
) -> dict[str, Any]:
    if raw_path.is_file():
        cached = read_json(raw_path)
        observation = cached.get("observation")
        if (
            isinstance(observation, dict)
            and cached.get("algorithm_version") == ALGORITHM_VERSION
            and cached.get("execution_fingerprint") == execution_fingerprint
        ):
            return validator(observation, rows)
    base_url = os.getenv("QWEN_API_BASE_URL", str(model_config["base_url"]))
    token = os.getenv("QWEN_API_TOKEN", "EMPTY")
    model = os.getenv("QWEN_MODEL", str(model_config["model"]))

    def build_content(
        max_edge: int, quality: int, use_lossless: bool
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        media: list[dict[str, Any]] = []
        for group, paths in enumerate(media_groups, start=1):
            content.append({"type": "text", "text": f"Image group {group}."})
            for path in paths:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_url(
                                path,
                                max_edge=max_edge,
                                quality=quality,
                                lossless=use_lossless,
                            )
                        },
                    }
                )
                media.append({"image_group": group, "path": str(path.resolve())})
        return content, media

    transport = {
        "max_edge": image_max_edge,
        "quality": image_quality,
        "lossless": lossless,
    }
    content, media = build_content(**{
        "max_edge": transport["max_edge"],
        "quality": transport["quality"],
        "use_lossless": transport["lossless"],
    })
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
            choice = raw.get("choices", [{}])[0]
            message = choice.get("message") if isinstance(choice, dict) else None
            text = message.get("content") if isinstance(message, dict) else None
            if not isinstance(text, str):
                raise ValueError("response_content_not_text")
            parsed = validator(parse_json_object(text), rows)
            attempts.append(
                {
                    "attempt": attempt + 1,
                    "content": text,
                    "schema_errors": [],
                    "transport": dict(transport),
                }
            )
            artifact = {
                "algorithm_version": ALGORITHM_VERSION,
                "execution_fingerprint": execution_fingerprint,
                "model": model,
                "base_url": base_url,
                "media": media,
                "attempts": attempts,
                "image_transport": dict(transport),
                "observation": parsed,
            }
            write_json(raw_path, artifact)
            raw_path.with_name(raw_path.stem + "_failed.json").unlink(missing_ok=True)
            return parsed
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ConnectionError, OSError, ValueError, json.JSONDecodeError) as exc:
            attempt_record: dict[str, Any] = {
                "attempt": attempt + 1,
                "errors": [f"{type(exc).__name__}:{exc}"],
                "transport": dict(transport),
            }
            if text is not None:
                attempt_record["content"] = text
            if attempt == 0:
                status_code = exc.code if isinstance(exc, urllib.error.HTTPError) else None
                if status_code in {413, 500, 502, 503, 504}:
                    transport = {
                        "max_edge": min(image_max_edge, 1800 if lossless else 1536),
                        "quality": min(image_quality, 90),
                        "lossless": False,
                    }
                    content, media = build_content(
                        int(transport["max_edge"]),
                        int(transport["quality"]),
                        bool(transport["lossless"]),
                    )
                    payload["messages"][0]["content"] = content
                    attempt_record["next_transport"] = dict(transport)
                else:
                    payload["messages"][0]["content"].append(
                        {
                            "type": "text",
                            "text": (
                                "Schema correction: start directly with { and return one complete legal JSON object "
                                "using exactly the requested image groups and frame_ids. Omit analysis, Markdown, and code fences."
                            ),
                        }
                    )
            attempts.append(attempt_record)
            if attempt == 0:
                time.sleep(2.0)
    failure_path = raw_path.with_name(raw_path.stem + "_failed.json")
    write_json(
        failure_path,
        {
            "algorithm_version": ALGORITHM_VERSION,
            "execution_fingerprint": execution_fingerprint,
            "model": model,
            "base_url": base_url,
            "media": media,
            "attempts": attempts,
            "image_transport": dict(transport),
            "observation": None,
            "status": "request_failed",
        },
    )
    raise RuntimeError(f"Qwen request failed after targeted retry: {attempts}")


def _call_qwen_panorama_locator(
    frame_path: Path,
    model_config: dict[str, Any],
    raw_path: Path,
    execution_fingerprint: str | None,
) -> dict[str, Any]:
    try:
        from .skills import dynamic_meter_reading
    except ImportError:
        from skills import dynamic_meter_reading  # type: ignore

    source_digest = sha256(frame_path)
    if raw_path.is_file():
        cached = read_json(raw_path)
        locations = cached.get("locations")
        if (
            isinstance(locations, dict)
            and cached.get("algorithm_version") == ALGORITHM_VERSION
            and cached.get("execution_fingerprint") == execution_fingerprint
            and cached.get("source_image_sha256") == source_digest
        ):
            return dynamic_meter_reading.validate_panorama_locations(locations)
    failed_path = raw_path.with_name(raw_path.stem + "_failed.json")
    if failed_path.is_file():
        cached_failure = read_json(failed_path)
        if (
            cached_failure.get("algorithm_version") == ALGORITHM_VERSION
            and cached_failure.get("execution_fingerprint") == execution_fingerprint
            and cached_failure.get("source_image_sha256") == source_digest
        ):
            raise RuntimeError("cached Qwen panorama meter location failure")

    base_url = os.getenv("QWEN_API_BASE_URL", str(model_config["base_url"]))
    token = os.getenv("QWEN_API_TOKEN", "EMPTY")
    model = os.getenv("QWEN_MODEL", str(model_config["model"]))
    content: list[dict[str, Any]] = [
        {"type": "text", "text": dynamic_meter_reading.panorama_location_prompt()},
        {"type": "image_url", "image_url": {"url": image_data_url(frame_path, max_edge=1600)}},
    ]
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 1000,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    attempts: list[dict[str, Any]] = []
    for attempt in range(2):
        response_text: str | None = None
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
            message = choice.get("message") if isinstance(choice, dict) else None
            response_text = message.get("content") if isinstance(message, dict) else None
            if not isinstance(response_text, str):
                raise ValueError("response_content_not_text")
            locations = dynamic_meter_reading.validate_panorama_locations(
                parse_json_object(response_text)
            )
            attempts.append({"attempt": attempt + 1, "content": response_text, "errors": []})
            write_json(
                raw_path,
                {
                    "algorithm_version": ALGORITHM_VERSION,
                    "execution_fingerprint": execution_fingerprint,
                    "source_image": str(frame_path.resolve()),
                    "source_image_sha256": source_digest,
                    "model": model,
                    "base_url": base_url,
                    "attempts": attempts,
                    "locations": locations,
                    "selection_basis": "current_frame_visible_A_V_glyph_and_analog_arc",
                    "video_id_used_for_routing": False,
                    "historical_artifacts_used": False,
                    "fixed_video_roi_used": False,
                },
            )
            return locations
        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            TimeoutError,
            ConnectionError,
            OSError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            record: dict[str, Any] = {
                "attempt": attempt + 1,
                "errors": [f"{type(exc).__name__}:{exc}"],
            }
            if response_text is not None:
                record["content"] = response_text
            attempts.append(record)
            if attempt == 0:
                payload["messages"][0]["content"].append(
                    {
                        "type": "text",
                        "text": "Return both distinct visible analog meters only, with legal 0..1000 boxes and no Markdown.",
                    }
                )
                time.sleep(2.0)
    write_json(
        failed_path,
        {
            "algorithm_version": ALGORITHM_VERSION,
            "execution_fingerprint": execution_fingerprint,
            "source_image": str(frame_path.resolve()),
            "source_image_sha256": source_digest,
            "model": model,
            "base_url": base_url,
            "attempts": attempts,
            "locations": None,
            "status": "panorama_location_failed",
            "video_id_used_for_routing": False,
            "historical_artifacts_used": False,
            "fixed_video_roi_used": False,
        },
    )
    raise RuntimeError("Qwen panorama meter location failed after targeted retry")


def _ground_adaptive_meter_roles(
    cycle: int,
    rows: list[dict[str, Any]],
    model_config: dict[str, Any],
    evidence_dir: Path,
    execution_fingerprint: str | None,
    maximum_attempts: int = 2,
) -> list[dict[str, Any]]:
    try:
        from .skills import dynamic_meter_reading
    except ImportError:
        from skills import dynamic_meter_reading  # type: ignore

    adaptive_rows = [
        row
        for row in rows
        if row.get("window_source") == "record_meter"
        and isinstance(row.get("image_path"), str)
        and Path(row["image_path"]).is_file()
    ]
    by_round: dict[int, list[dict[str, Any]]] = {}
    for row in adaptive_rows:
        by_round.setdefault(int(row.get("adaptive_request_number") or 0), []).append(row)
    selected: list[dict[str, Any]] = []
    for request_number in sorted(by_round, reverse=True):
        selected.append(
            max(
                by_round[request_number],
                key=lambda item: float(item.get("timestamp_seconds") or 0.0),
            )
        )
    for row in sorted(
        adaptive_rows,
        key=lambda item: (
            float(item.get("sharpness") or 0.0),
            float(item.get("timestamp_seconds") or 0.0),
        ),
        reverse=True,
    ):
        if row not in selected:
            selected.append(row)

    audit: list[dict[str, Any]] = []
    for row in selected[: max(0, maximum_attempts)]:
        frame_id = str(row.get("frame_id") or "frame_unknown")
        frame_path = Path(row["image_path"])
        raw_path = evidence_dir / "qwen" / f"cycle_{cycle}_{frame_id}_panorama_locator.json"
        try:
            locations = _call_qwen_panorama_locator(
                frame_path, model_config, raw_path, execution_fingerprint
            )
            crops = dynamic_meter_reading.export_panorama_crops(
                frame_path,
                locations,
                evidence_dir / "panorama_meter_crops" / f"cycle_{cycle}" / frame_id,
                frame_id,
            )
            role_views = row.setdefault("role_views", {})
            for role, crop in crops.items():
                role_views[role] = {
                    **crop,
                    "dynamic": True,
                    "source": "qwen_panorama_location",
                }
            row["panorama_location"] = {
                "status": "both_roles_grounded",
                "locations": locations,
                "raw_path": str(raw_path.resolve()),
                "selection_basis": "current_frame_visible_A_V_glyph_and_analog_arc",
                "video_id_used_for_routing": False,
                "historical_artifacts_used": False,
                "fixed_video_roi_used": False,
            }
            audit.append(
                {
                    "frame_id": frame_id,
                    "status": "both_roles_grounded",
                    "raw_path": str(raw_path.resolve()),
                    "roles": sorted(crops),
                }
            )
        except (OSError, RuntimeError, ValueError) as exc:
            audit.append(
                {
                    "frame_id": frame_id,
                    "status": "location_failed",
                    "error": f"{type(exc).__name__}:{exc}",
                    "raw_path": str(raw_path.resolve()),
                }
            )
    return audit


def _paper_media(rows: list[dict[str, Any]], window: dict[str, Any]) -> list[list[Path]]:
    # Sample around completion and the post-write reveal rather than blindly
    # taking the first/middle/last frames. Multiple views remain one vote.
    ordered = sorted(rows, key=lambda item: float(item["timestamp_seconds"]))
    if len(ordered) <= len(PAPER_SELECTION_OFFSETS_SECONDS):
        selected = ordered
    else:
        recording_end = float(window["recording_end_seconds"])
        selected = []
        selection_offsets = tuple(
            sorted(set(PAPER_SELECTION_OFFSETS_SECONDS + PAPER_DENSE_SELECTION_OFFSETS_SECONDS))
        )
        for offset in selection_offsets:
            target = recording_end + offset
            if offset in PAPER_DENSE_SELECTION_OFFSETS_SECONDS:
                nearest = min(ordered, key=lambda item: abs(float(item["timestamp_seconds"]) - target))
                if nearest not in selected:
                    selected.append(nearest)
                continue
            neighbors = [
                item
                for item in ordered
                if abs(float(item["timestamp_seconds"]) - target) <= PAPER_ANCHOR_NEIGHBORHOOD_SECONDS
            ]
            if neighbors:
                nearest = max(
                    neighbors,
                    key=lambda item: (
                        float((item.get("paper_field_view") or {}).get("sharpness") or 0.0),
                        float(item.get("sharpness") or 0.0),
                        -abs(float(item["timestamp_seconds"]) - target),
                    ),
                )
            else:
                nearest = min(ordered, key=lambda item: abs(float(item["timestamp_seconds"]) - target))
            if nearest not in selected:
                selected.append(nearest)
        if len(selected) < len(selection_offsets):
            remaining = [row for row in ordered if row not in selected]
            remaining.sort(key=lambda item: (float(item.get("sharpness") or 0.0), float(item["timestamp_seconds"])), reverse=True)
            selected.extend(remaining[: len(selection_offsets) - len(selected)])
    supplemental = [
        item
        for item in ordered
        if isinstance(item.get("adaptive_request_number"), int)
        and isinstance(item.get("paper_field_view"), dict)
    ]
    supplemental.sort(
        key=lambda item: (
            float((item.get("paper_field_view") or {}).get("sharpness") or 0.0),
            float(item.get("timestamp_seconds") or 0.0),
        ),
        reverse=True,
    )
    if supplemental:
        combined = supplemental[:2] + selected
        deduplicated: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in combined:
            frame_id = str(item.get("frame_id") or "")
            if frame_id and frame_id not in seen:
                seen.add(frame_id)
                deduplicated.append(item)
        selected = deduplicated[:PAPER_MODEL_GROUP_LIMIT]
    selected.sort(key=lambda item: float(item["timestamp_seconds"]))
    groups: list[list[Path]] = []
    for row in selected:
        paths = [Path(row["panorama_path"])]
        field_views = [
            item
            for item in row.get("paper_field_views", [])
            if isinstance(item, dict) and item.get("roi_path")
        ]
        field_view = row.get("paper_field_view")
        if not field_views and isinstance(field_view, dict) and field_view.get("roi_path"):
            field_views = [field_view]
        if field_views:
            for item in field_views[:PAPER_MODEL_VIEW_LIMIT]:
                paths.append(Path(item["roi_path"]))
                if item.get("ink_roi_path"):
                    paths.append(Path(item["ink_roi_path"]))
        else:
            calibrated = row.get("paper_calibrated_view")
            if isinstance(calibrated, dict) and calibrated.get("roi_path"):
                paths.append(Path(calibrated["roi_path"]))
            else:
                candidates = list(row.get("paper_candidates") or [])
                if candidates:
                    paths.append(Path(candidates[0]["roi_path"]))
            search_views = list(row.get("paper_search_views") or [])
            if search_views:
                middle = search_views[len(search_views) // 2]
                if isinstance(middle, dict) and middle.get("roi_path"):
                    paths.append(Path(middle["roi_path"]))
        groups.append(paths)
    rows[:] = selected
    return groups


def _paper_digit_review_rows(
    field: str,
    paper_observation: dict[str, Any],
    rows: list[dict[str, Any]],
    maximum: int = 3,
) -> list[dict[str, Any]]:
    visible_ids = {
        item["frame_id"]
        for item in paper_observation.get("observations", [])
        if isinstance(item, dict)
        and item.get("paper_visible")
        and isinstance(item.get(field), dict)
        and item[field].get("label_visible")
        and normalize_decimal(item[field].get("value")) is not None
    }
    candidates = [
        row
        for row in rows
        if row.get("frame_id") in visible_ids
        and (
            isinstance(row.get("paper_field_view"), dict)
            or any(
                isinstance(item, dict) for item in row.get("paper_field_views", [])
            )
        )
    ]
    candidates.sort(
        key=lambda item: (
            float((item.get("paper_field_view") or {}).get("sharpness") or 0.0),
            float(item.get("timestamp_seconds") or 0.0),
        ),
        reverse=True,
    )
    selected = candidates[:maximum]
    selected.sort(key=lambda item: float(item.get("timestamp_seconds") or 0.0))
    return selected


def _paper_digit_review_media(rows: list[dict[str, Any]]) -> list[list[Path]]:
    groups: list[list[Path]] = []
    for row in rows:
        field_views = [
            item for item in row.get("paper_field_views", []) if isinstance(item, dict)
        ]
        field_view = row.get("paper_field_view")
        if not field_views and isinstance(field_view, dict):
            field_views = [field_view]
        paths = [
            Path(path)
            for item in field_views[:PAPER_MODEL_VIEW_LIMIT]
            for path in (item.get("roi_path"), item.get("ink_roi_path"))
            if isinstance(path, str) and path
        ]
        if paths:
            groups.append(paths)
    return groups


def reduce_paper_digit_review(
    field: str,
    review: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    row_by_id = {row["frame_id"]: row for row in rows}
    buckets: dict[float, list[dict[str, Any]]] = {}
    for item in review.get("observations", []):
        if not isinstance(item, dict) or not item.get("label_visible"):
            continue
        numeric = normalize_decimal(item.get("value"))
        frame_id = item.get("frame_id")
        if numeric is None or frame_id not in row_by_id:
            continue
        candidate = {
            "value": numeric,
            "confidence": float(item.get("confidence") or 0.0),
            "frame_id": frame_id,
            "timestamp_seconds": row_by_id[frame_id]["timestamp_seconds"],
            "raw_text": item.get("raw_text"),
            "decimal_digits": item.get("decimal_digits"),
            "final_digit_shape": item.get("final_digit_shape"),
            "final_digit_closed_loop": item.get("final_digit_closed_loop"),
            "final_digit_inward_hook": item.get("final_digit_inward_hook"),
            "evidence": item.get("evidence"),
        }
        buckets.setdefault(round(numeric, 4), []).append(candidate)
    if not buckets:
        return {
            "field": field,
            "status": "missing",
            "value": None,
            "confidence": 0.25,
            "support_frame_count": 0,
            "support": [],
            "evidence": review.get("evidence"),
        }
    ranked = sorted(
        buckets.items(),
        key=lambda pair: (
            len({item["frame_id"] for item in pair[1]}),
            sum(item["confidence"] for item in pair[1]) / len(pair[1]),
            max(item["timestamp_seconds"] for item in pair[1]),
        ),
        reverse=True,
    )
    winner_value, support = ranked[0]
    support_frame_count = len({item["frame_id"] for item in support})
    confidence = sum(item["confidence"] for item in support) / len(support)
    if support_frame_count == 1:
        confidence = min(confidence, 0.64)
    conflict = len(ranked) > 1 and len({item["frame_id"] for item in ranked[1][1]}) >= support_frame_count
    return {
        "field": field,
        "status": "conflict" if conflict else "read",
        "value": winner_value,
        "confidence": round(confidence, 4),
        "support_frame_count": support_frame_count,
        "support": support,
        "alternatives": [
            {"value": value, "support_frame_count": len({item['frame_id'] for item in items})}
            for value, items in ranked[1:]
        ],
        "evidence": review.get("evidence"),
    }


def fuse_paper_digit_review(
    paper: dict[str, Any],
    field: str,
    review_reduced: dict[str, Any],
) -> dict[str, Any]:
    fused = {**paper, "fields": {name: dict(value) for name, value in paper["fields"].items()}}
    before = dict(fused["fields"].get(field) or {})
    accepted = (
        review_reduced.get("status") == "read"
        and review_reduced.get("value") is not None
        and int(review_reduced.get("support_frame_count") or 0) >= 2
        and float(review_reduced.get("confidence") or 0.0) >= 0.7
    )
    review_record = {
        "trigger": "same_cycle_paper_meter_mismatch_or_paper_conflict",
        "accepted": accepted,
        "paper_before_review": before,
        "independent_visual_review": review_reduced,
    }
    if accepted:
        fused["fields"][field] = {
            "status": "read",
            "value": review_reduced["value"],
            "confidence": review_reduced["confidence"],
            "support_frame_count": review_reduced["support_frame_count"],
            "support": review_reduced["support"],
            "alternatives": review_reduced.get("alternatives", []),
            "source": "independent_digit_stroke_review",
        }
    fused.setdefault("digit_reviews", {})[field] = review_record
    return fused


def paper_fields_requiring_digit_review(cycle: int, result: dict[str, Any]) -> list[str]:
    diagnostics = result.get("diagnostics") if isinstance(result.get("diagnostics"), dict) else {}
    comparison = diagnostics.get("comparison") if isinstance(diagnostics.get("comparison"), dict) else {}
    fields: list[str] = []
    for role, field in (("voltage", f"u{cycle}"), ("current", f"i{cycle}")):
        item = comparison.get(role)
        if not isinstance(item, dict) or not item.get("matched"):
            fields.append(field)
    return fields


def extract_legacy_four_field_observations(
    observations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Recover labeled U1/I1/U2/I2 text without merging the two rows."""
    patterns = {
        "u1": re.compile(r"\bU\s*1\s*=\s*(\d+(?:\.\d+)?)", re.I),
        "i1": re.compile(r"\bI\s*1\s*=\s*(\d+(?:\.\d+)?)", re.I),
        "u2": re.compile(r"\bU\s*2\s*=\s*(\d+(?:\.\d+)?)", re.I),
        "i2": re.compile(r"\bI\s*2\s*=\s*(\d+(?:\.\d+)?)", re.I),
    }
    output: list[dict[str, Any]] = []
    for source in observations:
        if not isinstance(source, dict):
            continue
        outcome = source.get("outcome") if isinstance(source.get("outcome"), dict) else {}
        result = outcome.get("result") if isinstance(outcome.get("result"), dict) else {}
        texts = [str(result.get("reason") or "")]
        visible = result.get("visible_handwriting")
        if isinstance(visible, list):
            texts.extend(str(item) for item in visible)
        for name in ("u_field", "i_field"):
            field = result.get(name)
            if isinstance(field, dict):
                texts.extend(str(field.get(key) or "") for key in ("raw_text", "label"))
        combined = " ; ".join(texts)
        fields: dict[str, Any] = {}
        for name, pattern in patterns.items():
            match = pattern.search(combined)
            fields[name] = {
                "label_visible": bool(match),
                "raw_text": match.group(0) if match else None,
                "value": float(match.group(1)) if match else None,
                "confidence": 0.78 if match else 0.0,
            }
        output.append(
            {
                "frame_id": str(result.get("frame_id") or source.get("frame_id") or ""),
                "timestamp_seconds": float(source.get("timestamp_seconds") or 0.0),
                "paper_visible": bool(result.get("paper_visible")),
                **fields,
                "evidence": "labeled values recovered from frozen visual model text",
            }
        )
    return output


def _meter_media(rows: list[dict[str, Any]], window: dict[str, Any] | None = None) -> list[list[Path]]:
    rows.sort(key=lambda item: float(item["timestamp_seconds"]))
    adaptive_rows = [
        row for row in rows if row.get("window_source") == "record_meter"
    ]
    if adaptive_rows:
        baseline_rows = [row for row in rows if row not in adaptive_rows]
        if isinstance(window, dict) and window.get("meter_window_source") == "derived_pre_recording_1":
            baseline_pool = baseline_rows[:3]
        else:
            baseline_pool = baseline_rows[-3:]
        baseline_selected = max(
            baseline_pool,
            key=lambda item: (
                len(item.get("role_views") or {}),
                float(item.get("sharpness") or 0.0),
            ),
            default=None,
        )
        adaptive_selected = sorted(
            adaptive_rows,
            key=lambda item: (
                bool(item.get("panorama_location")),
                len(item.get("dynamic_meter_candidates") or []),
                float(item.get("sharpness") or 0.0),
            ),
            reverse=True,
        )[:3]
        selected = adaptive_selected
        if baseline_selected is not None:
            selected.append(baseline_selected)
        rows[:] = sorted(
            selected, key=lambda item: float(item["timestamp_seconds"])
        )
    elif len(rows) > 3:
        if isinstance(window, dict) and window.get("meter_window_source") == "derived_pre_recording_1":
            rows[:] = rows[:3]
        else:
            rows[:] = rows[-3:]
    groups: list[list[Path]] = []
    for row in rows:
        paths = [Path(row["image_path"])]
        role_views = row.get("role_views") if isinstance(row.get("role_views"), dict) else {}
        face_views = row.get("face_views") if isinstance(row.get("face_views"), dict) else {}
        for role in ("ammeter", "voltmeter"):
            view = role_views.get(role)
            if isinstance(view, dict) and view.get("image_path"):
                paths.append(Path(view["image_path"]))
            face = face_views.get(role)
            if isinstance(face, dict) and face.get("image_path"):
                paths.append(Path(face["image_path"]))
        for candidate in list(row.get("dynamic_meter_candidates") or [])[:3]:
            if isinstance(candidate, dict):
                candidate_path = candidate.get("enhanced_path") or candidate.get("wide_path")
                if candidate_path:
                    paths.append(Path(candidate_path))
        groups.append(paths)
    return groups


def _range_overlap_seconds(
    candidate: tuple[float, float], previous: list[tuple[float, float]]
) -> float:
    start, end = candidate
    return sum(max(0.0, min(end, old_end) - max(start, old_start)) for old_start, old_end in previous)


def _second_round_meter_range(
    window: dict[str, Any],
    previous: list[tuple[float, float]],
    duration_seconds: float,
) -> tuple[float, float] | None:
    """Choose a broad current-cycle range that avoids the adjacent first round."""
    meter_window = window.get("meter_window_seconds")
    recording_start = float(window.get("recording_start_seconds") or 0.0)
    recording_end = float(window.get("recording_end_seconds") or recording_start)
    segments: list[tuple[float, float, int]] = []
    if isinstance(meter_window, list) and len(meter_window) == 2:
        meter_start = max(0.0, float(meter_window[0]))
        meter_end = min(duration_seconds, float(meter_window[1]))
        if meter_end - meter_start >= 0.1:
            segments.append((meter_start, meter_end, 0))
    recording_start = max(0.0, recording_start)
    recording_end = min(duration_seconds, recording_end)
    if recording_end - recording_start >= 0.1:
        segments.append((recording_start, recording_end, 1))

    candidates: list[tuple[float, float, int]] = []
    for segment_start, segment_end, priority in segments:
        width = min(4.0, segment_end - segment_start)
        starts = {
            segment_start,
            max(segment_start, segment_end - width),
            max(segment_start, min(segment_end - width, recording_start - width)),
        }
        for start in starts:
            end = min(segment_end, start + width)
            if end - start >= 0.1:
                candidates.append((start, end, priority))
    if not candidates:
        return None
    start, end, _priority = min(
        candidates,
        key=lambda item: (
            _range_overlap_seconds((item[0], item[1]), previous),
            -round(item[1] - item[0], 6),
            item[2],
            abs(item[1] - recording_start),
        ),
    )
    return round(start, 3), round(end, 3)


def fuse_meter_geometry(observation: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    fused = {
        **observation,
        "consensus": {role: dict(value) for role, value in observation.get("consensus", {}).items()},
    }
    geometry_fusion: dict[str, Any] = {}
    for role in ("ammeter", "voltmeter"):
        candidates: list[dict[str, Any]] = []
        for row in rows:
            face_views = row.get("face_views") if isinstance(row.get("face_views"), dict) else {}
            face = face_views.get(role) if isinstance(face_views, dict) else None
            geometry = face.get("geometry") if isinstance(face, dict) else None
            if isinstance(geometry, dict) and geometry.get("status") == "pointer_scale_candidate":
                candidates.append({"frame_id": row["frame_id"], **geometry})
        if len(candidates) < 2:
            continue
        raw_values = [float(item["value_raw"]) for item in candidates]
        minor_division = float(candidates[0]["minor_division"])
        spread = max(raw_values) - min(raw_values)
        if spread > minor_division * 1.25:
            geometry_fusion[role] = {
                "status": "conflict",
                "spread": round(spread, 6),
                "candidates": candidates,
            }
            continue
        median_raw = float(np.median(np.array(raw_values, dtype=np.float64)))
        value = round(round(median_raw / minor_division) * minor_division, 6)
        before = dict(fused["consensus"].get(role) or {})
        fused["consensus"][role] = {
            "selected_range": float(candidates[0]["selected_range"]),
            "value": value,
            "confidence": round(min(float(item["confidence"]) for item in candidates), 4),
            "supporting_frame_ids": [item["frame_id"] for item in candidates],
            "evidence": "Stable calibrated OpenCV pointer angles across distinct frames.",
        }
        geometry_fusion[role] = {
            "status": "fused",
            "qwen_consensus_before_geometry": before,
            "median_value_raw": round(median_raw, 6),
            "value": value,
            "spread": round(spread, 6),
            "candidates": candidates,
        }
    if geometry_fusion:
        fused["geometry_fusion"] = geometry_fusion
    return fused


def assess_cycle_meter_evidence(
    cycle: int,
    meter_observation: dict[str, Any],
    meter_rows: list[dict[str, Any]],
    min_confidence: float = 0.70,
    min_distinct_frames: int = 2,
) -> dict[str, Any]:
    """Decide whether this cycle needs more frames from meter evidence only."""
    row_frame_ids = [
        str(row.get("frame_id"))
        for row in meter_rows
        if isinstance(row, dict) and row.get("frame_id")
    ]
    known_frame_ids = set(row_frame_ids)
    group_frame_ids = {
        index: frame_id for index, frame_id in enumerate(row_frame_ids, start=1)
    }

    def canonical_frame_id(item: dict[str, Any]) -> str | None:
        group = item.get("image_group")
        if isinstance(group, str) and group.strip().isdigit():
            group = int(group.strip())
        if type(group) is int and group in group_frame_ids:
            return group_frame_ids[group]
        frame_id = str(item.get("frame_id") or "")
        return frame_id if frame_id in known_frame_ids else None

    consensus = (
        meter_observation.get("consensus")
        if isinstance(meter_observation.get("consensus"), dict)
        else {}
    )
    per_frame = [
        item
        for item in meter_observation.get("per_frame", [])
        if isinstance(item, dict)
    ]
    role_reports: dict[str, Any] = {}
    request_reasons: list[str] = []
    stable_by_role: dict[str, set[str]] = {}

    for role in ("ammeter", "voltmeter"):
        role_consensus = consensus.get(role)
        role_consensus = role_consensus if isinstance(role_consensus, dict) else {}
        selected_range = normalize_decimal(role_consensus.get("selected_range"))
        consensus_value = normalize_decimal(role_consensus.get("value"))
        confidence = float(role_consensus.get("confidence") or 0.0)
        confidence = max(0.0, min(1.0, confidence))

        visible_frame_ids: set[str] = set()
        readings_by_frame: dict[str, list[tuple[float, float]]] = {}
        for item in per_frame:
            frame_id = canonical_frame_id(item)
            reading = item.get(role)
            if frame_id is None or not isinstance(reading, dict):
                continue
            if reading.get("visible"):
                visible_frame_ids.add(frame_id)
            frame_range = normalize_decimal(reading.get("selected_range"))
            frame_value = normalize_decimal(reading.get("value"))
            if frame_range is None or frame_value is None:
                continue
            readings_by_frame.setdefault(frame_id, []).append(
                (frame_range, frame_value)
            )

        supporting_frame_ids = {
            str(frame_id)
            for frame_id in role_consensus.get("supporting_frame_ids", [])
            if str(frame_id) in known_frame_ids
        }
        if not supporting_frame_ids:
            supporting_frame_ids = set(readings_by_frame)

        observed_ranges = sorted(
            {
                round(frame_range, 6)
                for readings in readings_by_frame.values()
                for frame_range, _value in readings
            }
        )
        observed_values = sorted(
            {
                round(frame_value, 6)
                for readings in readings_by_frame.values()
                for _frame_range, frame_value in readings
            }
        )
        range_conflict = len(observed_ranges) > 1
        tolerance_range = selected_range or (
            observed_ranges[0] if observed_ranges else None
        )
        reading_tolerance = one_division_tolerance(role, tolerance_range)
        reading_spread = (
            max(observed_values) - min(observed_values)
            if len(observed_values) > 1
            else 0.0
        )
        reading_conflict = reading_spread > reading_tolerance + 1e-9

        stable_frame_ids: set[str] = set()
        if consensus_value is not None and consensus_value > 1e-9:
            for frame_id, readings in readings_by_frame.items():
                if any(
                    frame_value > 1e-9
                    and abs(frame_value - consensus_value) <= reading_tolerance + 1e-9
                    and (
                        selected_range is None
                        or abs(frame_range - selected_range) <= 1e-9
                    )
                    for frame_range, frame_value in readings
                ):
                    stable_frame_ids.add(frame_id)

        role_reasons: list[str] = []
        missing = selected_range is None or consensus_value is None
        if missing:
            role_reasons.append(f"{role}_missing")
        else:
            if not stable_frame_ids:
                role_reasons.append(f"{role}_no_stable_deflection")
            if len(supporting_frame_ids) < min_distinct_frames:
                role_reasons.append(f"{role}_single_frame_support")
            if range_conflict:
                role_reasons.append(f"{role}_range_conflict")
            if reading_conflict:
                role_reasons.append(f"{role}_reading_conflict")
            if confidence < min_confidence:
                role_reasons.append(f"{role}_low_confidence")

        request_reasons.extend(role_reasons)
        stable_by_role[role] = stable_frame_ids
        role_reports[role] = {
            "selected_range": selected_range,
            "value": consensus_value,
            "confidence": round(confidence, 4),
            "visible_frame_ids": sorted(visible_frame_ids),
            "supporting_frame_ids": sorted(supporting_frame_ids),
            "distinct_frame_support": len(supporting_frame_ids),
            "stable_deflection_frame_ids": sorted(stable_frame_ids),
            "observed_ranges": observed_ranges,
            "observed_values": observed_values,
            "reading_spread": round(reading_spread, 6),
            "reading_tolerance": reading_tolerance,
            "request_reasons": role_reasons,
        }

    stable_dual_meter_frame_ids = sorted(
        stable_by_role["ammeter"] & stable_by_role["voltmeter"]
    )
    if not stable_dual_meter_frame_ids:
        request_reasons.append("no_stable_dual_meter_frames")

    dynamic_roi_frame_ids = sorted(
        {
            str(row.get("frame_id"))
            for row in meter_rows
            if isinstance(row, dict)
            and row.get("frame_id")
            and (
                bool(row.get("dynamic_meter_candidates"))
                or bool(row.get("role_views"))
                or bool(row.get("face_views"))
            )
        }
    )
    return {
        "schema_version": "resistance_agent_cycle_meter_evidence_quality.v1",
        "cycle": cycle,
        "request_more_frames": bool(request_reasons),
        "request_reasons": list(dict.fromkeys(request_reasons)),
        "roles": role_reports,
        "stable_dual_meter_frame_ids": stable_dual_meter_frame_ids,
        "dynamic_roi_frame_ids": dynamic_roi_frame_ids,
        "meter_values_only": True,
        "paper_values_used": False,
        "excel_accessed": False,
        "video_id_used_for_routing": False,
        "historical_artifacts_used": False,
        "fixed_video_roi_used": False,
    }


def reduce_paper_cycle(
    cycle: int,
    observation: dict[str, Any],
    rows: list[dict[str, Any]],
    digit_consensus_min_support: int = 1,
) -> dict[str, Any]:
    targets = (f"u{cycle}", f"i{cycle}")
    row_by_id = {row["frame_id"]: row for row in rows}
    output: dict[str, Any] = {}
    for field in targets:
        candidates: list[dict[str, Any]] = []
        for item in observation.get("observations", []):
            value = item.get(field) if isinstance(item, dict) else None
            if not isinstance(value, dict) or not item.get("paper_visible"):
                continue
            numeric = normalize_decimal(value.get("value"))
            if numeric is None or not value.get("label_visible"):
                continue
            candidates.append(
                {
                    "value": numeric,
                    "confidence": float(value.get("confidence") or 0.0),
                    "frame_id": item["frame_id"],
                    "timestamp_seconds": row_by_id[item["frame_id"]]["timestamp_seconds"],
                    "raw_text": value.get("raw_text"),
                }
            )
        if not candidates:
            output[field] = {"status": "missing", "value": None, "confidence": 0.25, "support": []}
            continue
        buckets: dict[float, list[dict[str, Any]]] = {}
        for candidate in candidates:
            buckets.setdefault(round(float(candidate["value"]), 4), []).append(candidate)
        ranked = sorted(
            buckets.items(),
            key=lambda pair: (
                len({item["frame_id"] for item in pair[1]}),
                sum(item["confidence"] for item in pair[1]) / len(pair[1]),
                max(item["timestamp_seconds"] for item in pair[1]),
            ),
            reverse=True,
        )
        winner_value, support = ranked[0]
        distinct_support = len({item["frame_id"] for item in support})
        conflicting = len(ranked) > 1 and len({item["frame_id"] for item in ranked[1][1]}) >= distinct_support
        confidence = sum(item["confidence"] for item in support) / len(support)
        if distinct_support == 1:
            confidence = min(confidence, 0.64)
        output[field] = {
            "status": (
                "conflict"
                if conflicting or distinct_support < digit_consensus_min_support
                else "read"
            ),
            "value": winner_value,
            "confidence": round(confidence, 4),
            "support_frame_count": distinct_support,
            "support": support,
            "alternatives": [{"value": value, "support_frame_count": len({item['frame_id'] for item in items})} for value, items in ranked[1:]],
        }
    return {
        "cycle": cycle,
        "target_fields": list(targets),
        "fields": output,
        "evidence": observation.get("evidence"),
    }


def assess_record_evidence(
    cycle: int,
    paper: dict[str, Any],
    observation: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    min_confidence: float = PAPER_FIELD_MIN_CONFIDENCE,
    min_distinct_frames: int = PAPER_FIELD_MIN_DISTINCT_FRAMES,
) -> dict[str, Any]:
    """Assess paper evidence without consulting meter values or labels."""
    targets = (f"u{cycle}", f"i{cycle}")
    visible_groups = {
        str(item.get("frame_id"))
        for item in observation.get("observations", [])
        if isinstance(item, dict) and item.get("paper_visible")
    }
    field_reports: dict[str, Any] = {}
    reasons: list[str] = []
    fields_needing_review: list[str] = []
    if not visible_groups:
        reasons.append("paper_not_found")
    for field in targets:
        item = paper.get("fields", {}).get(field)
        item = item if isinstance(item, dict) else {}
        status = str(item.get("status") or "missing")
        support_count = int(item.get("support_frame_count") or 0)
        confidence = float(item.get("confidence") or 0.0)
        field_reasons: list[str] = []
        if status == "missing" or item.get("value") is None:
            field_reasons.append("field_missing")
        if status == "conflict":
            field_reasons.append("digit_conflict")
        if item.get("value") is not None and support_count < min_distinct_frames:
            field_reasons.append("single_frame_support")
        if item.get("value") is not None and confidence < min_confidence:
            field_reasons.append("low_confidence")
        if field_reasons:
            fields_needing_review.append(field)
            reasons.extend(field_reasons)
        field_reports[field] = {
            "status": status,
            "value": item.get("value"),
            "confidence": round(confidence, 4),
            "distinct_frame_support": support_count,
            "supporting_frame_ids": sorted(
                {
                    str(value.get("frame_id"))
                    for value in item.get("support", [])
                    if isinstance(value, dict) and value.get("frame_id")
                }
            ),
            "request_reasons": sorted(set(field_reasons)),
        }
    return {
        "schema_version": "resistance_agent_record_evidence_quality.v1",
        "cycle": cycle,
        "target_fields": field_reports,
        "paper_visible_frame_ids": sorted(visible_groups),
        "dynamic_field_roi_frame_count": sum(
            isinstance(row.get("paper_field_view"), dict)
            or any(
                isinstance(item, dict) for item in row.get("paper_field_views", [])
            )
            for row in rows
        ),
        "request_more_frames": bool(reasons),
        "request_reasons": sorted(set(reasons)),
        "fields_needing_review": fields_needing_review,
        "meter_values_used": False,
        "excel_accessed": False,
    }


def paper_fields_requiring_visual_review(quality: dict[str, Any]) -> list[str]:
    values = quality.get("fields_needing_review")
    return [str(value) for value in values] if isinstance(values, list) else []


def _adaptive_record_result_paths(run_dir: Path, cycle: int | None = None) -> list[Path]:
    root = run_dir / "adaptive_evidence" / "record_paper"
    pattern = f"cycle_{cycle}/request_*/result.json" if cycle else "cycle_*/request_*/result.json"
    return sorted(root.glob(pattern))


def _adaptive_record_rows(
    run_dir: Path, cycle: int, source_digest: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in _adaptive_record_result_paths(run_dir, cycle):
        result = read_json(path)
        if (
            result.get("evidence_profile") != "record_paper"
            or result.get("cycle") != cycle
            or result.get("source_video_sha256") != source_digest
            or result.get("video_id_used_for_routing") is not False
            or result.get("historical_artifacts_used") is not False
            or result.get("fixed_video_roi_used") is not False
        ):
            continue
        for row in result.get("paper_rows", []):
            if isinstance(row, dict) and row.get("frame_id"):
                rows.append(row)
    return rows


def _merge_paper_rows(
    baseline: list[dict[str, Any]], supplemental: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for row in baseline + supplemental:
        frame_id = row.get("frame_id")
        if not isinstance(frame_id, str) or not frame_id:
            continue
        current = merged.get(frame_id)
        if current is None or (
            (
                isinstance(row.get("paper_field_view"), dict)
                or any(
                    isinstance(item, dict)
                    for item in row.get("paper_field_views", [])
                )
            )
            and not (
                isinstance(current.get("paper_field_view"), dict)
                or any(
                    isinstance(item, dict)
                    for item in current.get("paper_field_views", [])
                )
            )
        ):
            merged[frame_id] = row
    return sorted(merged.values(), key=lambda item: float(item.get("timestamp_seconds") or 0.0))


def _adaptive_record_meter_result_paths(
    run_dir: Path, cycle: int | None = None
) -> list[Path]:
    root = run_dir / "adaptive_evidence" / "record_meter"
    pattern = (
        f"cycle_{cycle}/request_*/result.json"
        if cycle
        else "cycle_*/request_*/result.json"
    )
    return sorted(root.glob(pattern))


def _adaptive_record_meter_rows(
    run_dir: Path, cycle: int, source_digest: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in _adaptive_record_meter_result_paths(run_dir, cycle):
        result = read_json(path)
        if (
            result.get("evidence_profile") != "record_meter"
            or result.get("cycle") != cycle
            or result.get("source_video_sha256") != source_digest
            or result.get("video_id_used_for_routing") is not False
            or result.get("historical_artifacts_used") is not False
            or result.get("fixed_video_roi_used") is not False
            or result.get("paper_values_used") is not False
            or result.get("excel_accessed") is not False
        ):
            continue
        for row in result.get("meter_rows", []):
            if (
                isinstance(row, dict)
                and row.get("frame_id")
                and row.get("window_source") == "record_meter"
                and row.get("source_video_sha256") == source_digest
            ):
                rows.append(row)
    return rows


def _merge_meter_rows(
    baseline: list[dict[str, Any]], supplemental: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for row in baseline + supplemental:
        frame_id = row.get("frame_id")
        if not isinstance(frame_id, str) or not frame_id:
            continue
        current = merged.get(frame_id)
        if current is None or row.get("window_source") == "record_meter":
            merged[frame_id] = row
    return sorted(
        merged.values(),
        key=lambda item: float(item.get("timestamp_seconds") or 0.0),
    )


def _adaptive_record_meter_digest(run_dir: Path) -> str | None:
    paths = _adaptive_record_meter_result_paths(run_dir)
    if not paths:
        return None
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(run_dir).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _meter_rows_visual_digest(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: str(item.get("frame_id") or "")):
        digest.update(str(row.get("frame_id") or "").encode("utf-8"))
        role_views = row.get("role_views") if isinstance(row.get("role_views"), dict) else {}
        for role in sorted(role_views):
            view = role_views[role]
            if not isinstance(view, dict):
                continue
            digest.update(role.encode("ascii"))
            digest.update(str(view.get("source") or "").encode("utf-8"))
            path_value = view.get("image_path")
            if isinstance(path_value, str):
                path = Path(path_value)
                digest.update(str(path.resolve()).encode("utf-8"))
                if path.is_file():
                    digest.update(sha256(path).encode("ascii"))
    return digest.hexdigest()


def _adaptive_record_digest(run_dir: Path) -> str | None:
    paths = _adaptive_record_result_paths(run_dir)
    if not paths:
        return None
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(run_dir).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _record_execution_fingerprint(
    base_fingerprint: str | None, adaptive_digest: str | None
) -> str | None:
    if adaptive_digest is None:
        return base_fingerprint
    payload = json.dumps(
        {"base": base_fingerprint, "adaptive_record_digest": adaptive_digest},
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _clip_range_to_current_intervals(
    start: float,
    end: float,
    intervals: list[tuple[float, float]],
) -> tuple[float, float] | None:
    overlaps = [
        (max(start, allowed_start), min(end, allowed_end))
        for allowed_start, allowed_end in intervals
        if min(end, allowed_end) - max(start, allowed_start) >= 0.1
    ]
    if not overlaps:
        return None
    return max(overlaps, key=lambda value: value[1] - value[0])


def _build_meter_adaptive_recommendation(
    cycle_reports: dict[str, Any],
    run_dir: Path,
    duration_seconds: float,
    parameters: dict[str, Any],
    max_rounds: int,
) -> dict[str, Any] | None:
    reason_priority = (
        "ammeter_missing",
        "voltmeter_missing",
        "ammeter_no_stable_deflection",
        "voltmeter_no_stable_deflection",
        "ammeter_single_frame_support",
        "voltmeter_single_frame_support",
        "ammeter_range_conflict",
        "voltmeter_range_conflict",
        "ammeter_reading_conflict",
        "voltmeter_reading_conflict",
        "ammeter_low_confidence",
        "voltmeter_low_confidence",
        "no_stable_dual_meter_frames",
    )
    for cycle in (1, 2):
        report = cycle_reports.get(str(cycle))
        if not isinstance(report, dict):
            continue
        quality = report.get("meter_quality")
        if not isinstance(quality, dict) or not quality.get("request_more_frames"):
            continue
        completed_requests = _adaptive_record_meter_result_paths(run_dir, cycle)
        completed_rounds = len(completed_requests)
        if completed_rounds >= max_rounds:
            continue
        reasons = [str(value) for value in quality.get("request_reasons", [])]
        reason = next(
            (value for value in reason_priority if value in reasons),
            "no_stable_dual_meter_frames",
        )
        roles = quality.get("roles") if isinstance(quality.get("roles"), dict) else {}
        target_roles = [
            role
            for role in ("ammeter", "voltmeter")
            if isinstance(roles.get(role), dict)
            and bool(roles[role].get("request_reasons"))
        ]
        if not target_roles:
            target_roles = ["ammeter", "voltmeter"]
        anchors = sorted(
            {
                str(frame_id)
                for role in target_roles
                for frame_id in (
                    list((roles.get(role) or {}).get("supporting_frame_ids") or [])
                    + list((roles.get(role) or {}).get("stable_deflection_frame_ids") or [])
                )
                if isinstance(frame_id, str) and frame_id
            }
        )
        rows = [
            item for item in report.get("meter_frames", []) if isinstance(item, dict)
        ]
        anchor_rows = [row for row in rows if row.get("frame_id") in anchors]
        window = report.get("window") if isinstance(report.get("window"), dict) else {}
        if anchor_rows:
            center = sum(
                float(row.get("timestamp_seconds") or 0.0) for row in anchor_rows
            ) / len(anchor_rows)
        elif rows:
            center = sum(
                float(row.get("timestamp_seconds") or 0.0) for row in rows
            ) / len(rows)
        else:
            meter_window = window.get("meter_window_seconds")
            if isinstance(meter_window, list) and len(meter_window) == 2:
                center = (float(meter_window[0]) + float(meter_window[1])) / 2.0
            else:
                center = float(window.get("recording_start_seconds") or 0.0)
        interval = float(parameters.get("adaptive_interval_seconds", 0.2))
        max_frames = min(20, int(parameters.get("adaptive_max_frames", 20)))
        search_mode = (
            "current_run_meter_search"
            if window.get("broad_search")
            else "adjacent_meter_dense"
        )
        if completed_rounds == 0:
            radius = min(1.0, max(0.1, (max_frames - 1) * interval / 2.0))
            start = max(0.0, center - radius)
            end = min(duration_seconds, center + radius)
        else:
            previous_ranges: list[tuple[float, float]] = []
            for result_path in completed_requests:
                request_path = result_path.with_name("request.json")
                if not request_path.is_file():
                    continue
                request = read_json(request_path)
                for value in request.get("time_ranges") or []:
                    if not isinstance(value, dict):
                        continue
                    try:
                        previous_ranges.append(
                            (float(value["start_seconds"]), float(value["end_seconds"]))
                        )
                    except (KeyError, TypeError, ValueError):
                        continue
            broad_range = _second_round_meter_range(
                window, previous_ranges, duration_seconds
            )
            if broad_range is None:
                continue
            start, end = broad_range
            interval = max(0.25, interval)
            search_mode = "current_run_meter_search"
        try:
            from .adaptive_record_meter_evidence import current_cycle_meter_intervals
        except ImportError:
            from adaptive_record_meter_evidence import current_cycle_meter_intervals  # type: ignore
        allowed_intervals = current_cycle_meter_intervals(
            run_dir, cycle, duration_seconds
        )
        if allowed_intervals:
            clipped = _clip_range_to_current_intervals(
                start, end, allowed_intervals
            )
            if clipped is None:
                continue
            start, end = clipped
        if end - start < 0.1:
            continue
        template = {
            "rubric_ids": [7 if cycle == 1 else 9],
            "evidence_profile": "record_meter",
            "cycle": cycle,
            "reason": reason,
            "target_roles": target_roles,
            "anchor_frame_ids": anchors,
            "search_mode": search_mode,
            "time_ranges": [
                {
                    "start_seconds": round(start, 3),
                    "end_seconds": round(end, 3),
                }
            ],
            "interval_seconds": interval,
            "max_frames": max_frames,
            "roi_mode": "dynamic_meter_candidates",
            "view": "meter_pair",
        }
        return {
            "adaptive_evidence_recommended": True,
            "adaptive_evidence_reasons": reasons,
            "adaptive_request_template": template,
        }
    return None


def build_record_adaptive_recommendation(
    cycle_reports: dict[str, Any],
    run_dir: Path,
    duration_seconds: float,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    if not bool(parameters.get("adaptive_enabled", True)):
        return {
            "adaptive_evidence_recommended": False,
            "adaptive_evidence_reasons": [],
            "adaptive_request_template": None,
        }
    max_rounds = int(parameters.get("adaptive_max_rounds", 2))
    meter_recommendation = _build_meter_adaptive_recommendation(
        cycle_reports, run_dir, duration_seconds, parameters, max_rounds
    )
    if meter_recommendation is not None:
        return meter_recommendation
    reason_priority = (
        "paper_not_found",
        "field_missing",
        "digit_conflict",
        "single_frame_support",
        "low_confidence",
    )
    for cycle in (1, 2):
        report = cycle_reports.get(str(cycle))
        if not isinstance(report, dict):
            continue
        quality = report.get("paper_quality")
        if not isinstance(quality, dict) or not quality.get("request_more_frames"):
            continue
        existing_rounds = len(_adaptive_record_result_paths(run_dir, cycle))
        if existing_rounds >= max_rounds:
            continue
        reasons = [str(value) for value in quality.get("request_reasons", [])]
        reason = next((value for value in reason_priority if value in reasons), "low_confidence")
        rows = [item for item in report.get("paper_frames", []) if isinstance(item, dict)]
        anchors = sorted(
            {
                frame_id
                for field in quality.get("target_fields", {}).values()
                if isinstance(field, dict)
                for frame_id in field.get("supporting_frame_ids", [])
                if isinstance(frame_id, str) and frame_id
            }
        )
        anchor_rows = [row for row in rows if row.get("frame_id") in anchors]
        window = report.get("window") if isinstance(report.get("window"), dict) else {}
        if anchor_rows:
            center = sum(float(row.get("timestamp_seconds") or 0.0) for row in anchor_rows) / len(anchor_rows)
        else:
            center = float(window.get("recording_end_seconds") or 0.0)
        if reason == "paper_not_found":
            start = max(0.0, center - 1.0)
            end = min(duration_seconds, start + 4.0)
            interval = 0.5
            search_mode = "post_write_reveal"
        else:
            start = max(0.0, center - 1.0)
            end = min(duration_seconds, center + 1.0)
            interval = float(parameters.get("adaptive_interval_seconds", 0.2))
            search_mode = "adjacent_dense"
        if window.get("broad_search"):
            search_mode = "current_run_broad_writing_search"
        if end - start < 0.1:
            continue
        rubric_id = 7 if cycle == 1 else 9
        template = {
            "rubric_ids": [rubric_id],
            "evidence_profile": "record_paper",
            "cycle": cycle,
            "reason": reason,
            "target_fields": list(quality.get("fields_needing_review") or []),
            "anchor_frame_ids": anchors,
            "search_mode": search_mode,
            "time_ranges": [
                {
                    "start_seconds": round(start, 3),
                    "end_seconds": round(end, 3),
                }
            ],
            "interval_seconds": interval,
            "max_frames": int(parameters.get("adaptive_max_frames", 20)),
            "roi_mode": "dynamic_paper_tracking",
            "view": "paper_fields",
        }
        return {
            "adaptive_evidence_recommended": True,
            "adaptive_evidence_reasons": reasons,
            "adaptive_request_template": template,
        }
    return {
        "adaptive_evidence_recommended": False,
        "adaptive_evidence_reasons": [],
        "adaptive_request_template": None,
    }


def reduce_cycle_result(cycle: int, paper: dict[str, Any], meter: dict[str, Any], window: dict[str, Any]) -> dict[str, Any]:
    u_field, i_field = paper["fields"][f"u{cycle}"], paper["fields"][f"i{cycle}"]
    voltmeter, ammeter = meter["consensus"]["voltmeter"], meter["consensus"]["ammeter"]
    voltage = compare_value(u_field.get("value"), voltmeter.get("value"), "voltmeter", voltmeter.get("selected_range"))
    current = compare_value(i_field.get("value"), ammeter.get("value"), "ammeter", ammeter.get("selected_range"))
    paper_complete = all(field.get("status") == "read" and field.get("value") is not None for field in (u_field, i_field))
    meter_complete = all(item.get("value") is not None for item in (voltmeter, ammeter))
    passed = paper_complete and meter_complete and voltage["matched"] and current["matched"]
    direct_confidences = [float(u_field.get("confidence") or 0.0), float(i_field.get("confidence") or 0.0), float(voltmeter.get("confidence") or 0.0), float(ammeter.get("confidence") or 0.0)]
    confidence = min(direct_confidences) if passed else max(0.55, max(direct_confidences))
    return {
        "decision": "pass" if passed else "fail",
        "predicted_score": 1 if passed else 0,
        "confidence": round(min(1.0, confidence), 4),
        "reason": "cycle_bound_paper_values_match_same_cycle_meters" if passed else "cycle_bound_values_missing_conflicting_or_outside_meter_tolerance",
        "diagnostics": {
            "cycle": cycle,
            "recording_stage": window["recording_stage"],
            "paper_window_seconds": window["paper_window_seconds"],
            "meter_window_seconds": window["meter_window_seconds"],
            "meter_window_source": window["meter_window_source"],
            "paper": paper,
            "meters": meter,
            "comparison": {"voltage": voltage, "current": current},
            "tie_break": "same_cycle_then_distinct_frame_support_then_average_confidence_then_later_completed_ink",
        },
    }


def run_record_rubrics(
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
        execution_for_rubric(skill_plan, 7)
        if skill_plan
        else {
            "skill_id": "record.two_cycle_consistency",
            "parameters": dict(EXECUTOR_REGISTRY["record.two_cycle_consistency"].defaults),
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
    windows = cycle_windows(record, duration, str(parameters["cycle_mode"]))
    if parameters["cycle_mode"] == "first_observed_cycle" and windows:
        first_cycle = min(windows)
        windows = {first_cycle: windows[first_cycle]}
    evidence_dir = run_dir / "record_rubrics"
    source_digest = sha256(video_path)
    checkpoint_path = evidence_dir / "evidence_pre_qwen.json"
    checkpoint = read_json(checkpoint_path) if checkpoint_path.is_file() else {}
    evidence = checkpoint.get("cycles") if isinstance(checkpoint, dict) else None
    checkpoint_valid = (
        checkpoint.get("algorithm_version") == ALGORITHM_VERSION
        and checkpoint.get("source_video_sha256") == source_digest
        and checkpoint.get("routing_policy") == (skill_plan or {}).get("routing_policy")
        and checkpoint.get("execution_fingerprint") == execution["execution_fingerprint"]
        and isinstance(evidence, dict)
        and all(str(cycle) in evidence for cycle in windows)
    )
    if not checkpoint_valid:
        decoded = _decode_evidence(
            video_path,
            windows,
            evidence_dir,
            paper_max_samples=int(parameters["paper_max_samples"]),
            meter_max_samples=int(parameters["meter_max_samples"]),
            dynamic_paper_candidates=bool(parameters["dynamic_paper_candidates"]),
            dynamic_meter_candidates=bool(parameters["dynamic_meter_candidates"]),
            candidate_crops_per_frame=int(parameters["candidate_crops_per_frame"]),
        )
        evidence = {str(cycle): value for cycle, value in decoded.items()}
        write_json(
            checkpoint_path,
            {
                "algorithm_version": ALGORITHM_VERSION,
                "source_video_sha256": source_digest,
                "routing_policy": (skill_plan or {}).get("routing_policy"),
                "execution_fingerprint": execution["execution_fingerprint"],
                "skill_execution": execution,
                "cycle_windows": windows,
                "cycles": evidence,
            },
        )
    adaptive_digest = _adaptive_record_digest(run_dir)
    adaptive_meter_digest = _adaptive_record_meter_digest(run_dir)
    paper_execution_fingerprint = _record_execution_fingerprint(
        execution["execution_fingerprint"], adaptive_digest
    )
    meter_execution_fingerprint = _record_execution_fingerprint(
        execution["execution_fingerprint"], adaptive_meter_digest
    )
    effective_execution_fingerprint = _record_execution_fingerprint(
        paper_execution_fingerprint, adaptive_meter_digest
    )
    results: dict[int, dict[str, Any]] = {}
    cycle_reports: dict[str, Any] = {}
    for cycle in (1, 2):
        window = windows.get(cycle)
        cycle_evidence = evidence.get(str(cycle)) if isinstance(evidence, dict) else None
        if window is None or not isinstance(cycle_evidence, dict):
            missing = {
                "decision": "fail",
                "predicted_score": 0,
                "confidence": 0.72,
                "reason": f"recording_{cycle}_stage_missing_binary_tie_break_fail",
                "diagnostics": {"cycle": cycle, "recording_stage": f"recording_{cycle}", "cycle_stage_missing": True},
            }
            results[7 if cycle == 1 else 9] = missing
            cycle_reports[str(cycle)] = {"result": missing}
            continue
        paper_rows = _merge_paper_rows(
            list(cycle_evidence.get("paper") or []),
            _adaptive_record_rows(run_dir, cycle, source_digest),
        )
        meter_rows = _merge_meter_rows(
            list(cycle_evidence.get("meter") or []),
            _adaptive_record_meter_rows(run_dir, cycle, source_digest),
        )
        if not paper_rows or not meter_rows:
            raise RuntimeError(f"cycle_{cycle}_evidence_frames_missing")
        paper_media = _paper_media(paper_rows, window)
        paper_observation = _call_qwen(
            _paper_prompt(cycle, paper_rows, str(parameters["prompt_instruction"])),
            paper_media,
            model_config,
            evidence_dir / "qwen" / f"cycle_{cycle}_paper.json",
            validate_paper_observation,
            paper_rows,
            PAPER_MODEL_MAX_EDGE,
            paper_execution_fingerprint,
            PAPER_MODEL_JPEG_QUALITY,
        )
        panorama_location_audit = _ground_adaptive_meter_roles(
            cycle,
            meter_rows,
            model_config,
            evidence_dir,
            meter_execution_fingerprint,
        )
        cycle_meter_execution_fingerprint = _record_execution_fingerprint(
            meter_execution_fingerprint,
            _meter_rows_visual_digest(meter_rows),
        )
        meter_media = _meter_media(meter_rows, window)
        meter_observation = _call_qwen(
            _meter_prompt(cycle, meter_rows, str(parameters["prompt_instruction"])),
            meter_media,
            model_config,
            evidence_dir / "qwen" / f"cycle_{cycle}_meters.json",
            validate_meter_observation,
            meter_rows,
            execution_fingerprint=cycle_meter_execution_fingerprint,
        )
        meter_observation = fuse_meter_geometry(meter_observation, meter_rows)
        meter_quality = assess_cycle_meter_evidence(
            cycle, meter_observation, meter_rows
        )
        paper_reduced = reduce_paper_cycle(
            cycle,
            paper_observation,
            paper_rows,
            digit_consensus_min_support=int(parameters["digit_consensus_min_support"]),
        )
        paper_quality = assess_record_evidence(
            cycle, paper_reduced, paper_observation, paper_rows
        )
        digit_review_observations: dict[str, Any] = {}
        for field in paper_fields_requiring_visual_review(paper_quality):
            review_rows = _paper_digit_review_rows(field, paper_observation, paper_rows)
            review_media = _paper_digit_review_media(review_rows)
            if len(review_rows) < 2 or len(review_media) != len(review_rows):
                continue
            review_observation = _call_qwen(
                _paper_digit_review_prompt(
                    field, review_rows, str(parameters["prompt_instruction"])
                ),
                review_media,
                model_config,
                evidence_dir / "qwen" / f"cycle_{cycle}_paper_{field}_digit_review.json",
                lambda value, rows, target=field: validate_paper_digit_review(value, rows, target),
                review_rows,
                PAPER_DIGIT_REVIEW_MAX_EDGE,
                paper_execution_fingerprint,
                PAPER_DIGIT_REVIEW_QUALITY,
                True,
            )
            review_reduced = reduce_paper_digit_review(field, review_observation, review_rows)
            paper_reduced = fuse_paper_digit_review(paper_reduced, field, review_reduced)
            digit_review_observations[field] = review_observation
        if digit_review_observations:
            paper_quality = assess_record_evidence(
                cycle, paper_reduced, paper_observation, paper_rows
            )
        result = reduce_cycle_result(cycle, paper_reduced, meter_observation, window)
        rubric_id = 7 if cycle == 1 else 9
        results[rubric_id] = result
        cycle_reports[str(cycle)] = {
            "window": window,
            "paper_frames": paper_rows,
            "meter_frames": meter_rows,
            "paper_observation": paper_observation,
            "paper_reduced": paper_reduced,
            "paper_digit_review_observations": digit_review_observations,
            "paper_quality": paper_quality,
            "meter_observation": meter_observation,
            "meter_quality": meter_quality,
            "panorama_location_audit": panorama_location_audit,
            "meter_execution_fingerprint": cycle_meter_execution_fingerprint,
            "result": result,
        }
    adaptive = build_record_adaptive_recommendation(
        cycle_reports, run_dir, duration, parameters
    )
    report = {
        "schema_version": "resistance_agent_record_evidence.v1",
        "algorithm_version": ALGORITHM_VERSION,
        "video_id": video_id,
        "source_video_id": source_video_id,
        "source_video_path": str(video_path.resolve()),
        "source_video_sha256": source_digest,
        "action_summary_path": str(action_path.resolve()),
        "boundary_summary_path": str(boundary_summary_path.resolve()) if boundary_summary_path else None,
        "boundary_stage_runs_used": boundary_used,
        "cycle_windows": windows,
        "cycles": cycle_reports,
        "rubric_7": results[7],
        "rubric_9": results[9],
        "excel_accessed": False,
        "ground_truth_sent_to_model": False,
        "paper_values_sent_to_meter_model": False,
        "source_video_unchanged": sha256(video_path) == source_digest,
        "selection_checkpoint_reused": checkpoint_valid,
        "adaptive_record_digest": adaptive_digest,
        "adaptive_record_meter_digest": adaptive_meter_digest,
        "effective_execution_fingerprint": effective_execution_fingerprint,
        "paper_execution_fingerprint": paper_execution_fingerprint,
        "meter_execution_fingerprint": meter_execution_fingerprint,
        "fixed_video_roi_used": False,
        "historical_fallback_used": bool(
            allow_historical_fallback and fallback is not None and action_path == fallback
        ),
        "routing_policy": (skill_plan or {}).get("routing_policy"),
        "skill_selection": (skill_plan or {}).get("skills", []),
        "skill_execution": execution,
        **adaptive,
    }
    report_path = evidence_dir / "record_evidence_report.json"
    write_json(report_path, report)
    return {
        "rubric_7": results[7],
        "rubric_9": results[9],
        "report_path": str(report_path.resolve()),
        **adaptive,
    }


run_record_rubrics.supports_boundary_summary = True
