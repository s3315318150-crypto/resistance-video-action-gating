"""High-resolution, current-run frame acquisition for Rubric 2.

The agent answers only the evidence-acquisition part of R2.  It does not use
video identity, fixed coordinates, historical windows, or Excel labels.  A
measurement stage and a recording stage are treated as one observation/
recording cycle because either stage can be the only reliable segmentation
anchor.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np


AGENT_VERSION = "r2_frame_sampling_agent.v1"
STAGES = {
    "circuit_wiring",
    "measurement_1",
    "recording_1",
    "circuit_rewiring",
    "measurement_2",
    "recording_2",
    "material_cleanup",
}
DEFAULT_MAX_GROUPS_PER_CYCLE = 10
DEFAULT_INITIAL_MARGIN_SECONDS = 3.0
DEFAULT_MAX_MARGIN_SECONDS = 8.0


class R2FrameAgentError(ValueError):
    """Raised when the current-run R2 evidence contract is invalid."""


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _number(value: Any, field: str, default: float | None = None) -> float:
    if value is None and default is not None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise R2FrameAgentError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise R2FrameAgentError(f"{field} must be finite")
    return result


def _stage_runs(record: dict[str, Any]) -> list[dict[str, Any]]:
    raw = record.get("observed_stage_runs") or record.get("source_observed_stage_runs")
    if not isinstance(raw, list):
        return []
    runs: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict) or item.get("stage") not in STAGES:
            continue
        try:
            start = _number(item.get("start_seconds"), "start_seconds")
            end = _number(item.get("end_seconds"), "end_seconds")
        except R2FrameAgentError:
            continue
        if start >= end:
            continue
        runs.append({**item, "start_seconds": start, "end_seconds": end})
    return sorted(runs, key=lambda item: (item["start_seconds"], item["end_seconds"]))


def _cycle_index(stage: str) -> int | None:
    if stage.endswith("_1"):
        return 1
    if stage.endswith("_2"):
        return 2
    return None


def build_observation_recording_cycles(
    record: dict[str, Any],
    duration: float,
    initial_margin_seconds: float = DEFAULT_INITIAL_MARGIN_SECONDS,
    max_margin_seconds: float = DEFAULT_MAX_MARGIN_SECONDS,
) -> list[dict[str, Any]]:
    """Build cycles when either measurement or recording is present.

    A missing adjacent label is deliberately represented in the result rather
    than treated as evidence that the action did not happen.
    """
    runs = _stage_runs(record)
    grouped: dict[int, list[dict[str, Any]]] = {1: [], 2: []}
    for run in runs:
        cycle = _cycle_index(str(run["stage"]))
        if cycle in grouped and str(run["stage"]).startswith(("measurement_", "recording_")):
            grouped[cycle].append(run)
    cycles: list[dict[str, Any]] = []
    for cycle_id in (1, 2):
        anchors = grouped[cycle_id]
        if not anchors:
            continue
        start = min(float(item["start_seconds"]) for item in anchors)
        end = max(float(item["end_seconds"]) for item in anchors)
        stages = sorted({str(item["stage"]) for item in anchors})
        has_measurement = f"measurement_{cycle_id}" in stages
        has_recording = f"recording_{cycle_id}" in stages
        barriers = [
            item
            for item in runs
            if item not in anchors
            and (
                item["stage"] in {"circuit_wiring", "circuit_rewiring", "material_cleanup"}
                or (
                    str(item["stage"]).startswith(("measurement_", "recording_"))
                    and _cycle_index(str(item["stage"])) != cycle_id
                )
            )
        ]
        left_candidates = [float(item["end_seconds"]) for item in barriers if float(item["end_seconds"]) <= start]
        right_candidates = [float(item["start_seconds"]) for item in barriers if float(item["start_seconds"]) >= end]
        left_limit = max(left_candidates, default=0.0)
        right_limit = min(right_candidates, default=duration)
        # The first window is deliberately modest.  The expanded window is
        # used only when candidate quality is insufficient in the first pass.
        initial = [max(left_limit, start - initial_margin_seconds), min(right_limit, end + initial_margin_seconds)]
        expanded = [max(left_limit, start - max_margin_seconds), min(right_limit, end + max_margin_seconds)]
        cycles.append(
            {
                "cycle_id": cycle_id,
                "anchor_stages_detected": stages,
                "measurement_detected": has_measurement,
                "recording_detected": has_recording,
                "missing_adjacent_action_may_be_unsegmented": not (has_measurement and has_recording),
                "initial_window_seconds": [round(value, 6) for value in initial],
                "expanded_window_seconds": [round(value, 6) for value in expanded],
                "boundary_limits_seconds": [round(left_limit, 6), round(right_limit, 6)],
            }
        )
    if cycles:
        return cycles
    return [
        {
            "cycle_id": 0,
            "anchor_stages_detected": [],
            "measurement_detected": False,
            "recording_detected": False,
            "missing_adjacent_action_may_be_unsegmented": True,
            "initial_window_seconds": [0.0, max(0.0, duration)],
            "expanded_window_seconds": [0.0, max(0.0, duration)],
            "window_source": "broad_search",
        }
    ]


def _box_iou(left: list[float], right: list[float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_left = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    area_right = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return inter / max(area_left + area_right - inter, 1e-9)


def _dedupe_boxes(boxes: list[tuple[float, list[float]]], limit: int) -> list[list[float]]:
    selected: list[list[float]] = []
    for _, box in sorted(boxes, key=lambda item: item[0], reverse=True):
        if any(_box_iou(box, existing) >= 0.65 for existing in selected):
            continue
        selected.append(box)
        if len(selected) >= limit:
            break
    return selected


def detect_dynamic_object_boxes(frame: np.ndarray) -> dict[str, Any]:
    """Find meter-like circular faces and resistor-like rectangles per frame."""
    height, width = frame.shape[:2]
    scale = min(1.0, 960.0 / max(height, width))
    small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else frame
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    sh, sw = gray.shape[:2]
    meters: list[tuple[float, list[float]]] = []
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(20, int(min(sw, sh) * 0.08)),
        param1=90,
        param2=28,
        minRadius=max(12, int(min(sw, sh) * 0.025)),
        maxRadius=max(20, int(min(sw, sh) * 0.18)),
    )
    if circles is not None:
        for cx, cy, radius in np.round(circles[0]).astype(int):
            x1, y1 = max(0, cx - int(radius * 1.35)), max(0, cy - int(radius * 1.35))
            x2, y2 = min(sw, cx + int(radius * 1.35)), min(sh, cy + int(radius * 1.35))
            area = (x2 - x1) * (y2 - y1)
            if area <= 0:
                continue
            meters.append((float(radius) / max(min(sw, sh), 1), [x1 / sw, y1 / sh, x2 / sw, y2 / sh]))
    # Contour fallback catches square/rectangular analog meters when the face
    # is partly occluded and Hough cannot close a circle.
    edges = cv2.Canny(gray, 50, 150)
    contours = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        aspect = w / max(h, 1)
        if sw * sh * 0.002 <= area <= sw * sh * 0.20 and 0.55 <= aspect <= 1.8:
            rectangularity = cv2.contourArea(contour) / max(area, 1)
            if rectangularity >= 0.22:
                meters.append((rectangularity, [x / sw, y / sh, (x + w) / sw, (y + h) / sh]))

    resistor_candidates: list[tuple[float, list[float]]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        aspect = max(w, h) / max(min(w, h), 1)
        if sw * sh * 0.00025 <= area <= sw * sh * 0.10 and 1.25 <= aspect <= 12.0:
            rectangularity = cv2.contourArea(contour) / max(area, 1)
            if rectangularity >= 0.18:
                resistor_candidates.append(
                    (0.5 * rectangularity + 0.5 * min(1.0, aspect / 5.0), [x / sw, y / sh, (x + w) / sw, (y + h) / sh])
                )
    meter_boxes = _dedupe_boxes(meters, 3)
    resistor_boxes = _dedupe_boxes(resistor_candidates, 4)
    all_boxes = meter_boxes + resistor_boxes
    if all_boxes:
        x1 = max(0.0, min(box[0] for box in all_boxes) - 0.08)
        y1 = max(0.0, min(box[1] for box in all_boxes) - 0.12)
        x2 = min(1.0, max(box[2] for box in all_boxes) + 0.08)
        y2 = min(1.0, max(box[3] for box in all_boxes) + 0.12)
        joint = [x1, y1, x2, y2]
    else:
        joint = [0.0, 0.0, 1.0, 1.0]
    return {
        "meter_candidate_boxes": meter_boxes,
        "resistor_candidate_boxes": resistor_boxes,
        "joint_topology_box": joint,
        "object_candidate_count": len(meter_boxes) + len(resistor_boxes),
    }


def _quality(frame: np.ndarray, previous_small: np.ndarray | None) -> tuple[float, float, float, np.ndarray]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    small = cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA)
    motion = 0.0 if previous_small is None else float(np.mean(cv2.absdiff(small, previous_small)))
    exposure = float(np.mean((gray > 12) & (gray < 245)))
    score = math.log1p(max(0.0, sharpness)) / 12.0 + 0.25 * exposure - 0.015 * motion
    return score, sharpness, motion, small


def _scan_window(capture: cv2.VideoCapture, fps: float, start: float, end: float) -> list[dict[str, Any]]:
    first = max(0, int(math.floor(start * fps)))
    last = max(first, int(math.ceil(end * fps)))
    capture.set(cv2.CAP_PROP_POS_FRAMES, first)
    current = first
    previous_small: np.ndarray | None = None
    records: list[dict[str, Any]] = []
    while current <= last:
        ok, frame = capture.read()
        if not ok or frame is None:
            break
        score, sharpness, motion, previous_small = _quality(frame, previous_small)
        boxes = detect_dynamic_object_boxes(frame)
        records.append(
            {
                "frame_number": current,
                "timestamp_seconds": round(current / fps, 6),
                "quality_score": round(score, 6),
                "sharpness": round(sharpness, 4),
                "motion_score": round(motion, 4),
                **boxes,
            }
        )
        current += 1
    return records


def _select_temporally_spread(records: list[dict[str, Any]], max_groups: int) -> list[dict[str, Any]]:
    if not records:
        return []
    max_groups = max(1, int(max_groups))
    if len(records) <= max_groups:
        return records
    selected: list[dict[str, Any]] = []
    for index in np.linspace(0, len(records) - 1, max_groups).round().astype(int):
        selected.append(records[int(index)])
    # Replace low-quality anchors with the best frame in the same temporal bin.
    for index in range(max_groups):
        left = int(round(index * len(records) / max_groups))
        right = int(round((index + 1) * len(records) / max_groups))
        pool = records[left:max(left + 1, right)]
        if pool and pool:
            best = max(pool, key=lambda item: float(item["quality_score"]) + 0.12 * min(1.0, item["object_candidate_count"] / 3.0))
            selected[index] = best
    return sorted({int(item["frame_number"]): item for item in selected}.values(), key=lambda item: item["frame_number"])


def build_r2_candidate_plan(
    video_path: Path,
    record: dict[str, Any],
    duration: float,
    parameters: dict[str, Any] | None = None,
    manifest_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Scan current observation/recording cycles and return native frame plans."""
    parameters = parameters or {}
    max_groups = int(parameters.get("max_groups_per_cycle", DEFAULT_MAX_GROUPS_PER_CYCLE))
    cycles = build_observation_recording_cycles(
        record,
        duration,
        float(parameters.get("initial_margin_seconds", DEFAULT_INITIAL_MARGIN_SECONDS)),
        float(parameters.get("max_margin_seconds", DEFAULT_MAX_MARGIN_SECONDS)),
    )
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise R2FrameAgentError(f"unable to open current video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if fps <= 0.0:
        capture.release()
        raise R2FrameAgentError("current video FPS is invalid")
    selected: list[dict[str, Any]] = []
    cycle_reports: list[dict[str, Any]] = []
    try:
        for cycle in cycles:
            start, end = cycle["initial_window_seconds"]
            scanned = _scan_window(capture, fps, start, end)
            chosen = _select_temporally_spread(scanned, max_groups)
            # If all initial frames are poor, expand only this cycle.  This is
            # evidence-driven and does not load an old time window.
            expanded = False
            if not chosen or max(float(item["quality_score"]) for item in chosen) < float(parameters.get("quality_expand_threshold", 1.0)):
                expanded = True
                expanded_scan = _scan_window(capture, fps, *cycle["expanded_window_seconds"])
                chosen = _select_temporally_spread(expanded_scan, max_groups)
                scanned = expanded_scan
            for item in chosen:
                selected.append(
                    {
                        **item,
                        "cycle_id": cycle["cycle_id"],
                        "stage": "observation_recording_cycle",
                        "sampling_origin": "expanded_full_frame_scan" if expanded else "initial_full_frame_scan",
                    }
                )
            cycle_reports.append(
                {
                    **cycle,
                    "scanned_frame_count": len(scanned),
                    "selected_frame_count": len(chosen),
                    "expanded_scan_used": expanded,
                }
            )
    finally:
        capture.release()
    selected.sort(key=lambda item: (int(item["cycle_id"]), int(item["frame_number"])))
    for group, item in enumerate(selected, start=1):
        item["image_group"] = group
        item["selection_policy"] = "native_4k_temporal_spread_quality_and_dynamic_object_visibility"
    if manifest_path is not None:
        _write_json(
            manifest_path,
            {
                "schema_version": "r2_frame_sampling_manifest.v1",
                "algorithm_version": AGENT_VERSION,
                "selection_basis": "current_video_observed_situation_only",
                "source_video_path": str(video_path.resolve()),
                "source_video_dimensions": "native_decode",
                "source_fps": fps,
                "cycles": cycle_reports,
                "selected_frames": selected,
                "max_groups_per_cycle": max_groups,
                "video_id_used_for_routing": False,
                "historical_artifacts_used": False,
                "fixed_video_roi_used": False,
            },
        )
    return selected


def _crop(frame: np.ndarray, box: list[float]) -> np.ndarray:
    height, width = frame.shape[:2]
    left = max(0, min(width - 1, int(round(box[0] * width))))
    top = max(0, min(height - 1, int(round(box[1] * height))))
    right = max(left + 1, min(width, int(round(box[2] * width))))
    bottom = max(top + 1, min(height, int(round(box[3] * height))))
    return frame[top:bottom, left:right]


def _enhance(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    enhanced = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    blur = cv2.GaussianBlur(enhanced, (0, 0), 1.1)
    return cv2.addWeighted(enhanced, 1.18, blur, -0.18, 0)


def _write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image, [int(cv2.IMWRITE_PNG_COMPRESSION), 3]):
        raise R2FrameAgentError(f"unable to write evidence image: {path}")


def decode_r2_evidence(video_path: Path, plans: list[dict[str, Any]], evidence_dir: Path) -> list[dict[str, Any]]:
    """Decode selected native frames and bind all ROI views to frame_number."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise R2FrameAgentError(f"unable to open current video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    targets = sorted({int(item["frame_number"]): item for item in plans}.items())
    rows: list[dict[str, Any]] = []
    try:
        for frame_number, request in targets:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            frame_id = f"frame_{frame_number:08d}"
            stem = f"r2_{frame_id}_{frame_number / fps:010.3f}s"
            native_path = evidence_dir / "native_4k" / f"{stem}.png"
            enhanced_path = evidence_dir / "native_4k_enhanced" / f"{stem}.png"
            _write_png(native_path, frame)
            _write_png(enhanced_path, _enhance(frame))
            views: dict[str, Any] = {}
            joint_box = request.get("joint_topology_box") or [0.0, 0.0, 1.0, 1.0]
            joint = _crop(frame, [float(value) for value in joint_box])
            joint_path = evidence_dir / "joint_topology_native" / f"{stem}.png"
            joint_enhanced_path = evidence_dir / "joint_topology_enhanced" / f"{stem}.png"
            _write_png(joint_path, joint)
            _write_png(joint_enhanced_path, _enhance(joint))
            views["joint_topology"] = {
                "normalized_xyxy": joint_box,
                "native_path": str(joint_path.resolve()),
                "enhanced_path": str(joint_enhanced_path.resolve()),
            }
            for role, boxes in (
                ("voltmeter_candidates", request.get("meter_candidate_boxes") or []),
                ("resistor_candidates", request.get("resistor_candidate_boxes") or []),
            ):
                output: list[dict[str, Any]] = []
                for index, box in enumerate(boxes[:4], start=1):
                    crop = _crop(frame, [float(value) for value in box])
                    native = evidence_dir / role / f"{stem}_{index:02d}.png"
                    enhanced = evidence_dir / f"{role}_enhanced" / f"{stem}_{index:02d}.png"
                    _write_png(native, crop)
                    _write_png(enhanced, _enhance(crop))
                    output.append({"normalized_xyxy": box, "native_path": str(native.resolve()), "enhanced_path": str(enhanced.resolve())})
                views[role] = output
            rows.append(
                {
                    **request,
                    "frame_id": frame_id,
                    "frame_number": frame_number,
                    "timestamp_seconds": round(frame_number / fps, 6),
                    "panorama_path": str(native_path.resolve()),
                    "enhanced_path": str(enhanced_path.resolve()),
                    "role_views": views,
                    "image_group": len(rows) + 1,
                    "native_width": int(frame.shape[1]),
                    "native_height": int(frame.shape[0]),
                    "model_max_edge": 4096,
                    "model_encoding": "png_lossless",
                }
            )
    finally:
        capture.release()
    return rows
