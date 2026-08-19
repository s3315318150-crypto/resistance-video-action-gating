#!/usr/bin/env python3
"""Cycle-bound real-video evidence acquisition for Rubrics 7 and 9."""

from __future__ import annotations

import base64
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
ALGORITHM_VERSION = "r79_cycle_aware_digit_review_v15_meter_gate"
POST_WRITE_REVEAL_SECONDS = 6.0
PAPER_SELECTION_OFFSETS_SECONDS = (-0.5, 1.5, 2.5, 4.5)
PAPER_DENSE_SELECTION_OFFSETS_SECONDS = (2.3, 2.7)
PAPER_ANCHOR_NEIGHBORHOOD_SECONDS = 0.21
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
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("response JSON root is not an object")
    return value


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


def _current_run_nested_record(record: dict[str, Any], allowed_root: Path) -> dict[str, Any] | None:
    if record.get("replay_result"):
        raise ValueError("replay_result is forbidden in live R7/R9")
    nested_value = record.get("result_path")
    if not isinstance(nested_value, str) or not nested_value:
        return None
    nested_path = Path(nested_value).resolve()
    if not nested_path.is_relative_to(allowed_root.resolve()):
        raise ValueError("R7/R9 stage artifact is outside the current run")
    return read_json(nested_path) if nested_path.is_file() else None


def _source_record(
    summary: dict[str, Any], source_video_id: str, video_id: str, allowed_root: Path
) -> dict[str, Any]:
    direct_runs = summary.get("source_observed_stage_runs") or summary.get("observed_stage_runs")
    if isinstance(direct_runs, list):
        return {"observed_stage_runs": direct_runs}
    records = summary.get("records")
    if not isinstance(records, list):
        raise ValueError("Temporal Guard records are missing")
    for record in records:
        if not isinstance(record, dict):
            continue
        source = str(record.get("source_video_id") or "")
        if source != source_video_id:
            continue
        nested = _current_run_nested_record(record, allowed_root)
        if nested is not None and isinstance(nested.get("observed_stage_runs"), list):
            return nested
        return record
    raise ValueError(f"Temporal Guard record not found for video {video_id}")


def _boundary_record(
    summary: dict[str, Any], source_video_id: str, video_id: str, allowed_root: Path
) -> dict[str, Any] | None:
    direct_runs = summary.get("source_observed_stage_runs") or summary.get("observed_stage_runs")
    if isinstance(direct_runs, list):
        return {"observed_stage_runs": direct_runs}
    records = summary.get("records")
    if not isinstance(records, list):
        return None
    for record in records:
        if not isinstance(record, dict):
            continue
        source = str(record.get("source_video_id") or "")
        if source != source_video_id:
            continue
        runs = record.get("source_observed_stage_runs") or record.get("observed_stage_runs")
        if isinstance(runs, list):
            return {"observed_stage_runs": runs}
        nested = _current_run_nested_record(record, allowed_root)
        if nested is not None:
            runs = nested.get("source_observed_stage_runs") or nested.get("observed_stage_runs")
            if isinstance(runs, list):
                return {"observed_stage_runs": runs}
    return None


def cycle_windows(record: dict[str, Any], duration_seconds: float) -> dict[int, dict[str, Any]]:
    raw_runs = record.get("observed_stage_runs")
    runs = sorted(
        [item for item in raw_runs if isinstance(item, dict)] if isinstance(raw_runs, list) else [],
        key=lambda item: float(item.get("start_seconds") or 0.0),
    )
    windows: dict[int, dict[str, Any]] = {}
    for cycle in (1, 2):
        stage = f"recording_{cycle}"
        match = next((item for item in runs if item.get("stage") == stage), None)
        if match is None:
            continue
        start = max(0.0, float(match.get("start_seconds") or 0.0))
        end = min(duration_seconds, float(match.get("end_seconds") or start))
        later = [
            float(item.get("start_seconds") or duration_seconds)
            for item in runs
            if str(item.get("stage") or "").startswith("recording_")
            and float(item.get("start_seconds") or 0.0) > end + 1e-6
        ]
        next_start = min(later) if later else duration_seconds
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
            "source_event_ids": list(match.get("event_ids") or []),
        }
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
        output.append({"bbox_xyxy": box, "score": round(score, 6), "edge_ratio": round(edge_ratio, 6), "bright_ratio": round(bright_ratio, 6)})
    return sorted(output, key=lambda item: float(item["score"]), reverse=True)[:4]


def _write_jpeg(path: Path, image: np.ndarray, quality: int = 94) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image, [int(cv2.IMWRITE_JPEG_QUALITY), quality]):
        raise RuntimeError(f"unable to write image: {path}")


def _decode_evidence(
    video_path: Path,
    windows: dict[int, dict[str, Any]],
    evidence_dir: Path,
    video_id: str,
    allow_video_calibration: bool = True,
) -> dict[int, dict[str, list[dict[str, Any]]]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"unable to open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    records: dict[int, dict[str, list[dict[str, Any]]]] = {}
    try:
        for cycle, window in windows.items():
            paper_rows: list[dict[str, Any]] = []
            for timestamp in paper_timestamps(window):
                capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
                ok, frame = capture.read()
                if not ok or frame is None:
                    continue
                frame_number = int(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                actual = frame_number / fps if fps > 0 else timestamp
                stem = f"cycle_{cycle}_paper_{frame_number:08d}_{actual:010.3f}s"
                panorama_path = evidence_dir / "paper_frames" / f"{stem}.jpg"
                _write_jpeg(panorama_path, frame, 92)
                candidates = _paper_candidates(frame)
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
                for index, normalized in enumerate(PAPER_SEARCH_ROIS, start=1):
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
                        "paper_field_view": None,
                        "sharpness": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 6),
                    }
                )
            meter_rows: list[dict[str, Any]] = []
            for timestamp in meter_timestamps(window):
                capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
                ok, frame = capture.read()
                if not ok or frame is None:
                    continue
                frame_number = int(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                actual = frame_number / fps if fps > 0 else timestamp
                height, width = frame.shape[:2]
                crop = frame
                if crop.size == 0:
                    crop = frame
                crop = _enhance(crop)
                path = evidence_dir / "meter_frames" / f"cycle_{cycle}_meter_{frame_number:08d}_{actual:010.3f}s.jpg"
                _write_jpeg(path, crop, 94)
                role_views: dict[str, dict[str, Any]] = {}
                face_views: dict[str, dict[str, Any]] = {}
                dynamic_candidates: list[dict[str, Any]] = []
                try:
                    from . import meter_rubrics as meter_module
                except ImportError:
                    import meter_rubrics as meter_module  # type: ignore
                exported = meter_module._export_candidates(
                    {"frame_path": str(path), "sharpness": float(cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())},
                    evidence_dir / "dynamic_meter_candidates",
                )
                dynamic_candidates = [item for item in exported.get("candidates", []) if isinstance(item, dict)][:4]
                for candidate in dynamic_candidates:
                    role = str(candidate.get("role_hint") or "")
                    if role in {"ammeter", "voltmeter"}:
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


def image_data_url(path: Path, max_edge: int = 960, quality: int = 80) -> str:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"unable to decode model image: {path}")
    height, width = image.shape[:2]
    scale = min(1.0, float(max_edge) / max(height, width))
    if scale < 1.0:
        image = cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise ValueError(f"unable to encode model image: {path}")
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def _paper_prompt(cycle: int, rows: list[dict[str, Any]]) -> str:
    frame_list = ", ".join(f"{index}={row['frame_id']}@{row['timestamp_seconds']:.3f}s" for index, row in enumerate(rows, start=1))
    target = "U1/I1" if cycle == 1 else "U2/I2"
    return f"""Read handwritten {target} from recording_{cycle}. Frames: {frame_list}.
Each image group contains three views of one real frame: panorama and two local renderings. When a handwritten-field crop is available, the last two views are color-enhanced and grayscale high-contrast renderings of exactly the same pixels.
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


def _meter_prompt(cycle: int, rows: list[dict[str, Any]]) -> str:
    frame_list = ", ".join(f"{index}={row['frame_id']}@{row['timestamp_seconds']:.3f}s" for index, row in enumerate(rows, start=1))
    return f"""Read the analog ammeter and voltmeter immediately before recording_{cycle}. Frames: {frame_list}.
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


def _paper_digit_review_prompt(field: str, rows: list[dict[str, Any]]) -> str:
    frame_list = ", ".join(
        f"{index}={row['frame_id']}@{row['timestamp_seconds']:.3f}s"
        for index, row in enumerate(rows, start=1)
    )
    label = field.upper()
    return f"""Independently re-read only handwritten {label}. Frames: {frame_list}.
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
) -> dict[str, Any]:
    if raw_path.is_file():
        cached = read_json(raw_path)
        observation = cached.get("observation")
        if isinstance(observation, dict) and cached.get("algorithm_version") == ALGORITHM_VERSION:
            return validator(observation, rows)
    base_url = os.getenv("QWEN_API_BASE_URL", str(model_config["base_url"]))
    token = os.getenv("QWEN_API_TOKEN", "EMPTY")
    model = os.getenv("QWEN_MODEL", str(model_config["model"]))
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    media: list[dict[str, Any]] = []
    for group, paths in enumerate(media_groups, start=1):
        content.append({"type": "text", "text": f"Image group {group}."})
        for path in paths:
            content.append({"type": "image_url", "image_url": {"url": image_data_url(path, max_edge=image_max_edge)}})
            media.append({"image_group": group, "path": str(path.resolve())})
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 1800,
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
            choice = raw.get("choices", [{}])[0]
            message = choice.get("message") if isinstance(choice, dict) else None
            text = message.get("content") if isinstance(message, dict) else None
            if not isinstance(text, str):
                raise ValueError("response_content_not_text")
            parsed = validator(parse_json_object(text), rows)
            attempts.append({"attempt": attempt + 1, "content": text, "schema_errors": []})
            artifact = {
                "algorithm_version": ALGORITHM_VERSION,
                "model": model,
                "base_url": base_url,
                "media": media,
                "attempts": attempts,
                "observation": parsed,
            }
            write_json(raw_path, artifact)
            return parsed
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ConnectionError, OSError, ValueError, json.JSONDecodeError) as exc:
            attempts.append({"attempt": attempt + 1, "errors": [f"{type(exc).__name__}:{exc}"]})
            if attempt == 0:
                payload["messages"][0]["content"].append({"type": "text", "text": "Schema correction: return one complete legal JSON object using exactly the requested image groups and frame_ids."})
                time.sleep(2.0)
    failure_path = raw_path.with_name(raw_path.stem + "_failed.json")
    write_json(
        failure_path,
        {
            "algorithm_version": ALGORITHM_VERSION,
            "model": model,
            "base_url": base_url,
            "media": media,
            "attempts": attempts,
            "observation": None,
            "status": "request_failed",
        },
    )
    raise RuntimeError(f"Qwen request failed after targeted retry: {attempts}")


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
    selected.sort(key=lambda item: float(item["timestamp_seconds"]))
    groups: list[list[Path]] = []
    for row in selected:
        paths = [Path(row["panorama_path"])]
        field_view = row.get("paper_field_view")
        if isinstance(field_view, dict) and field_view.get("roi_path"):
            paths.append(Path(field_view["roi_path"]))
            if field_view.get("ink_roi_path"):
                paths.append(Path(field_view["ink_roi_path"]))
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
        if row.get("frame_id") in visible_ids and isinstance(row.get("paper_field_view"), dict)
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
        field_view = row.get("paper_field_view")
        if not isinstance(field_view, dict):
            continue
        paths = [
            Path(path)
            for path in (field_view.get("roi_path"), field_view.get("ink_roi_path"))
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
    if len(rows) > 3:
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


def reduce_paper_cycle(cycle: int, observation: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
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
            "status": "conflict" if conflicting else "read",
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


def _current_meter_prerequisites(run_dir: Path) -> dict[str, dict[str, Any]]:
    """Load only the current run's R4/R5/R6 binary results for R7/R9 gating."""
    prerequisites: dict[str, dict[str, Any]] = {}
    for rubric_id in (4, 5, 6):
        path = run_dir / "rubrics" / f"rubric_{rubric_id}.json"
        if not path.is_file():
            continue
        try:
            value = read_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(value, dict) and value.get("decision") in {"pass", "fail"}:
            prerequisites[str(rubric_id)] = value
    return prerequisites


def apply_meter_prerequisite_gate(
    result: dict[str, Any],
    cycle: int,
    prerequisites: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Require correct terminals, normal deflection, and a suitable range."""
    labels = {
        "4": "polarity_terminals",
        "5": "normal_pointer_deflection",
        "6": "suitable_meter_range",
    }
    failed = [labels[key] for key in ("4", "5", "6") if prerequisites.get(key, {}).get("decision") == "fail"]
    gate = {
        "required": [labels["4"], labels["5"], labels["6"]],
        "rubric_decisions": {
            labels[key]: prerequisites[key].get("decision")
            for key in ("4", "5", "6")
            if key in prerequisites
        },
        "missing_rubrics": [f"rubric_{key}" for key in ("4", "5", "6") if key not in prerequisites],
        "failed_items": failed,
        "cycle": cycle,
        "rule": "R7/R9 = record_match AND R4 AND R5 AND R6; explicit prerequisite fail forces fail",
    }
    diagnostics = dict(result.get("diagnostics") or {})
    diagnostics["meter_prerequisite_gate"] = gate
    if not failed:
        return {**result, "diagnostics": diagnostics}
    return {
        **result,
        "decision": "fail",
        "predicted_score": 0,
        "reason": "meter_prerequisite_failed:" + ",".join(failed),
        "diagnostics": diagnostics,
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
    allow_video_calibration: bool = False,
    allow_historical_fallback: bool = False,
    skill_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if allow_video_calibration or allow_historical_fallback or fallback_action_summary_path is not None:
        raise ValueError("live R7/R9 forbid fixed calibration and historical fallback")
    action_path = action_summary_path if action_summary_path and action_summary_path.is_file() else None
    if action_path is None or not action_path.is_file():
        raise ValueError("current live action summary is required")
    record = _source_record(read_json(action_path), source_video_id, video_id, run_dir)
    boundary_used = False
    if boundary_summary_path and boundary_summary_path.is_file():
        boundary = _boundary_record(read_json(boundary_summary_path), source_video_id, video_id, run_dir)
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
    windows = cycle_windows(record, duration)
    evidence_dir = run_dir / "record_rubrics"
    checkpoint_path = evidence_dir / "evidence_pre_qwen.json"
    checkpoint = read_json(checkpoint_path) if checkpoint_path.is_file() else {}
    evidence = checkpoint.get("cycles") if isinstance(checkpoint, dict) else None
    checkpoint_valid = (
        checkpoint.get("algorithm_version") == ALGORITHM_VERSION
        and checkpoint.get("allow_video_calibration", True) == allow_video_calibration
        and checkpoint.get("routing_policy") == (skill_plan or {}).get("routing_policy")
        and isinstance(evidence, dict)
        and all(str(cycle) in evidence for cycle in windows)
    )
    if not checkpoint_valid:
        decoded = _decode_evidence(
            video_path,
            windows,
            evidence_dir,
            video_id,
            allow_video_calibration=allow_video_calibration,
        )
        evidence = {str(cycle): value for cycle, value in decoded.items()}
        write_json(
            checkpoint_path,
            {
                "algorithm_version": ALGORITHM_VERSION,
                "allow_video_calibration": allow_video_calibration,
                "routing_policy": (skill_plan or {}).get("routing_policy"),
                "cycle_windows": windows,
                "cycles": evidence,
            },
        )
    results: dict[int, dict[str, Any]] = {}
    cycle_reports: dict[str, Any] = {}
    meter_prerequisites = _current_meter_prerequisites(run_dir)
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
        paper_rows = list(cycle_evidence.get("paper") or [])
        meter_rows = list(cycle_evidence.get("meter") or [])
        if not paper_rows or not meter_rows:
            raise RuntimeError(f"cycle_{cycle}_evidence_frames_missing")
        paper_media = _paper_media(paper_rows, window)
        paper_observation = _call_qwen(
            _paper_prompt(cycle, paper_rows),
            paper_media,
            model_config,
            evidence_dir / "qwen" / f"cycle_{cycle}_paper.json",
            validate_paper_observation,
            paper_rows,
            1440,
        )
        meter_media = _meter_media(meter_rows, window)
        meter_observation = _call_qwen(
            _meter_prompt(cycle, meter_rows),
            meter_media,
            model_config,
            evidence_dir / "qwen" / f"cycle_{cycle}_meters.json",
            validate_meter_observation,
            meter_rows,
        )
        meter_observation = fuse_meter_geometry(meter_observation, meter_rows)
        paper_reduced = reduce_paper_cycle(cycle, paper_observation, paper_rows)
        result = reduce_cycle_result(cycle, paper_reduced, meter_observation, window)
        digit_review_observations: dict[str, Any] = {}
        for field in paper_fields_requiring_digit_review(cycle, result):
            review_rows = _paper_digit_review_rows(field, paper_observation, paper_rows)
            review_media = _paper_digit_review_media(review_rows)
            if len(review_rows) < 2 or len(review_media) != len(review_rows):
                continue
            review_observation = _call_qwen(
                _paper_digit_review_prompt(field, review_rows),
                review_media,
                model_config,
                evidence_dir / "qwen" / f"cycle_{cycle}_paper_{field}_digit_review.json",
                lambda value, rows, target=field: validate_paper_digit_review(value, rows, target),
                review_rows,
                1800,
            )
            review_reduced = reduce_paper_digit_review(field, review_observation, review_rows)
            paper_reduced = fuse_paper_digit_review(paper_reduced, field, review_reduced)
            digit_review_observations[field] = review_observation
        if digit_review_observations:
            result = reduce_cycle_result(cycle, paper_reduced, meter_observation, window)
        result = apply_meter_prerequisite_gate(result, cycle, meter_prerequisites)
        rubric_id = 7 if cycle == 1 else 9
        results[rubric_id] = result
        cycle_reports[str(cycle)] = {
            "window": window,
            "paper_frames": paper_rows,
            "meter_frames": meter_rows,
            "paper_observation": paper_observation,
            "paper_reduced": paper_reduced,
            "paper_digit_review_observations": digit_review_observations,
            "meter_observation": meter_observation,
            "result": result,
        }
    report = {
        "schema_version": "resistance_agent_record_evidence.v1",
        "algorithm_version": ALGORITHM_VERSION,
        "video_id": video_id,
        "source_video_id": source_video_id,
        "source_video_path": str(video_path.resolve()),
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
        "source_video_unchanged": True,
        "selection_checkpoint_reused": checkpoint_valid,
        "allow_video_calibration": allow_video_calibration,
        "fixed_video_roi_used": bool(allow_video_calibration),
        "historical_fallback_used": False,
        "routing_policy": (skill_plan or {}).get("routing_policy"),
        "skill_selection": (skill_plan or {}).get("skills", []),
    }
    report_path = evidence_dir / "record_evidence_report.json"
    write_json(report_path, report)
    return {"rubric_7": results[7], "rubric_9": results[9], "report_path": str(report_path.resolve())}


run_record_rubrics.supports_boundary_summary = True
