#!/usr/bin/env python3
"""Temporal-consistency sampling for hierarchical v3."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _read_at(capture: cv2.VideoCapture, timestamp: float) -> np.ndarray | None:
    capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp) * 1000.0)
    ok, frame = capture.read()
    if not ok or frame is None:
        return None
    height, width = frame.shape[:2]
    if width > 240:
        scale = 240.0 / width
        frame = cv2.resize(frame, (240, max(1, round(height * scale))), interpolation=cv2.INTER_AREA)
    return frame


def scan_activity(
    video: Path,
    start_seconds: float,
    end_seconds: float,
    scan_interval_seconds: float = 0.5,
) -> list[dict[str, float]]:
    """Return low-resolution frame-difference and HSV-change activity scores."""
    if scan_interval_seconds <= 0 or end_seconds <= start_seconds:
        raise ValueError("invalid_activity_scan_range")
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"video_open_failed:{video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if not math.isfinite(fps) or fps <= 0 or frame_count <= 0:
        capture.release()
        raise RuntimeError(f"video_metadata_invalid:{video}")
    target_frames: list[int] = []
    current = start_seconds
    while current <= end_seconds + 1e-6:
        frame_number = min(frame_count - 1, max(0, int(round(min(current, end_seconds) * fps))))
        if not target_frames or frame_number != target_frames[-1]:
            target_frames.append(frame_number)
        current += scan_interval_seconds
    samples: list[dict[str, float]] = []
    previous_gray: np.ndarray | None = None
    previous_hist: np.ndarray | None = None
    try:
        next_frame = target_frames[0]
        capture.set(cv2.CAP_PROP_POS_FRAMES, next_frame)
        for frame_number in target_frames:
            while next_frame < frame_number:
                if not capture.grab():
                    raise RuntimeError(f"activity_scan_advance_failed:{frame_number}")
                next_frame += 1
            ok, frame = capture.read()
            next_frame = frame_number + 1
            if not ok or frame is None:
                raise RuntimeError(f"activity_scan_read_failed:{frame_number}")
            height, width = frame.shape[:2]
            if width > 240:
                scale = 240.0 / width
                frame = cv2.resize(frame, (240, max(1, round(height * scale))), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [18, 16], [0, 180, 0, 256])
            cv2.normalize(hist, hist)
            frame_difference = 0.0
            histogram_change = 0.0
            if previous_gray is not None:
                if previous_gray.shape != gray.shape:
                    previous_gray = cv2.resize(previous_gray, (gray.shape[1], gray.shape[0]))
                frame_difference = float(cv2.absdiff(gray, previous_gray).mean()) / 255.0
            if previous_hist is not None:
                histogram_change = float(cv2.compareHist(previous_hist, hist, cv2.HISTCMP_BHATTACHARYYA))
            activity = min(1.0, 3.5 * frame_difference + 1.5 * histogram_change)
            samples.append(
                {
                    "timestamp_seconds": round(frame_number / fps, 6),
                    "frame_difference": round(frame_difference, 6),
                    "histogram_change": round(histogram_change, 6),
                    "activity_score": round(activity, 6),
                }
            )
            previous_gray = gray
            previous_hist = hist
    finally:
        capture.release()
    return samples


def select_timestamps(
    start_seconds: float,
    end_seconds: float,
    budget: int,
    activity_samples: list[dict[str, float]],
    anchor_interval_seconds: float = 5.0,
    minimum_dense_gap_seconds: float = 0.5,
) -> list[float]:
    """Keep global anchors and spend the remaining budget on activity peaks."""
    if budget < 2 or end_seconds <= start_seconds:
        raise ValueError("invalid_tcs_budget")
    anchors: list[float] = []
    current = start_seconds
    while current <= end_seconds + 1e-6:
        anchors.append(round(min(current, end_seconds), 6))
        current += anchor_interval_seconds
    if anchors[-1] < end_seconds - 1e-6:
        anchors.append(round(end_seconds, 6))
    anchors = sorted(set(anchors))
    if len(anchors) > budget:
        indexes = np.linspace(0, len(anchors) - 1, budget).round().astype(int)
        return sorted({anchors[int(index)] for index in indexes})

    selected = list(anchors)
    ranked = sorted(
        activity_samples,
        key=lambda item: (-float(item.get("activity_score", 0.0)), float(item.get("timestamp_seconds", 0.0))),
    )
    for sample in ranked:
        timestamp = float(sample.get("timestamp_seconds", start_seconds))
        if not start_seconds <= timestamp <= end_seconds:
            continue
        if any(abs(timestamp - existing) < minimum_dense_gap_seconds - 1e-9 for existing in selected):
            continue
        selected.append(round(timestamp, 6))
        if len(selected) >= budget:
            break
    if len(selected) < budget:
        fallback_count = max(budget * 3, 2)
        for timestamp in np.linspace(start_seconds, end_seconds, fallback_count):
            value = round(float(timestamp), 6)
            if any(abs(value - existing) < minimum_dense_gap_seconds - 1e-9 for existing in selected):
                continue
            selected.append(value)
            if len(selected) >= budget:
                break
    return sorted(selected[:budget])


def timestamps_to_frame_numbers(
    timestamps: list[float],
    fps: float,
    frame_count: int,
) -> list[int]:
    values: list[int] = []
    seen: set[int] = set()
    for timestamp in timestamps:
        number = min(frame_count - 1, max(0, int(round(timestamp * fps))))
        if number not in seen:
            values.append(number)
            seen.add(number)
    return values


def adaptive_frame_numbers(
    video: Path,
    start_seconds: float,
    end_seconds: float,
    fps: float,
    frame_count: int,
    budget: int,
) -> tuple[list[int], dict[str, Any]]:
    activity = scan_activity(video, start_seconds, end_seconds)
    timestamps = select_timestamps(start_seconds, end_seconds, budget, activity)
    numbers = timestamps_to_frame_numbers(timestamps, fps, frame_count)
    return numbers, {
        "strategy": "fixed_5s_anchors_plus_activity_peaks",
        "frame_budget": budget,
        "selected_frame_count": len(numbers),
        "scan_interval_seconds": 0.5,
        "anchor_interval_seconds": 5.0,
        "minimum_dense_gap_seconds": 0.5,
        "activity_sample_count": len(activity),
        "selected_timestamps_seconds": [round(number / fps, 6) for number in numbers],
        "peak_activity_scores": sorted(
            (float(item["activity_score"]) for item in activity), reverse=True
        )[:8],
        "activity_samples": activity,
    }


def motion_consistency_score(action_type: str, normalized_motion: float) -> float:
    centers = {
        "wiring_action": 0.65,
        "measurement_action": 0.25,
        "writing_action": 0.35,
        "cleanup_action": 0.75,
        "auxiliary_action": 0.5,
        "uncertain": 0.5,
    }
    center = centers.get(action_type, 0.5)
    return max(0.0, 1.0 - abs(float(normalized_motion) - center) / 0.75)


def finite_score(value: float) -> float:
    return float(value) if math.isfinite(float(value)) else 0.5
