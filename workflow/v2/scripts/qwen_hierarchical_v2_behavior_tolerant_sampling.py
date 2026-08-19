#!/usr/bin/env python3
"""Supplemental activity sampling that always preserves uniform base frames."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _resize_scan_frame(frame: np.ndarray, max_width: int = 240) -> np.ndarray:
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame
    scale = max_width / float(width)
    return cv2.resize(frame, (max_width, max(1, round(height * scale))), interpolation=cv2.INTER_AREA)


def _compensated_difference(previous: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    previous_float = previous.astype(np.float32)
    current_float = current.astype(np.float32)
    shift_x = 0.0
    shift_y = 0.0
    try:
        (shift_x, shift_y), response = cv2.phaseCorrelate(previous_float, current_float)
        if not math.isfinite(response) or response < 0.05 or not all(math.isfinite(value) for value in (shift_x, shift_y)):
            shift_x = shift_y = 0.0
    except cv2.error:
        shift_x = shift_y = 0.0
    transform = np.float32([[1.0, 0.0, shift_x], [0.0, 1.0, shift_y]])
    aligned = cv2.warpAffine(
        previous,
        transform,
        (current.shape[1], current.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    difference = float(cv2.absdiff(current, aligned).mean()) / 255.0
    motion = math.hypot(shift_x, shift_y) / max(1.0, math.hypot(current.shape[1], current.shape[0]))
    return difference, motion


def scan_activity_compensated(
    video: Path,
    start_seconds: float,
    end_seconds: float,
    interval_seconds: float = 0.5,
) -> list[dict[str, float]]:
    if interval_seconds <= 0 or end_seconds <= start_seconds:
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
    timestamp = start_seconds
    while timestamp <= end_seconds + 1e-6:
        number = min(frame_count - 1, max(0, int(round(min(timestamp, end_seconds) * fps))))
        if not target_frames or target_frames[-1] != number:
            target_frames.append(number)
        timestamp += interval_seconds
    samples: list[dict[str, float]] = []
    previous_gray: np.ndarray | None = None
    previous_hist: np.ndarray | None = None
    try:
        current_number = target_frames[0]
        capture.set(cv2.CAP_PROP_POS_FRAMES, current_number)
        for frame_number in target_frames:
            while current_number < frame_number:
                if not capture.grab():
                    raise RuntimeError(f"activity_scan_advance_failed:{frame_number}")
                current_number += 1
            ok, frame = capture.read()
            current_number = frame_number + 1
            if not ok or frame is None:
                raise RuntimeError(f"activity_scan_read_failed:{frame_number}")
            frame = _resize_scan_frame(frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [18, 16], [0, 180, 0, 256])
            cv2.normalize(hist, hist)
            compensated_difference = 0.0
            global_camera_motion = 0.0
            histogram_change = 0.0
            if previous_gray is not None:
                compensated_difference, global_camera_motion = _compensated_difference(previous_gray, gray)
            if previous_hist is not None:
                histogram_change = float(cv2.compareHist(previous_hist, hist, cv2.HISTCMP_BHATTACHARYYA))
            raw_score = max(0.0, 3.5 * compensated_difference + 1.5 * histogram_change)
            samples.append(
                {
                    "timestamp_seconds": round(frame_number / fps, 6),
                    "compensated_frame_difference": round(compensated_difference, 6),
                    "histogram_change": round(histogram_change, 6),
                    "global_camera_motion": round(global_camera_motion, 6),
                    "raw_activity_score": round(raw_score, 6),
                }
            )
            previous_gray = gray
            previous_hist = hist
    finally:
        capture.release()
    return samples


def percentile_normalize(samples: list[dict[str, float]]) -> list[dict[str, float]]:
    if not samples:
        return []
    raw = np.asarray([float(item["raw_activity_score"]) for item in samples], dtype=np.float64)
    low = float(np.percentile(raw, 20.0))
    high = float(np.percentile(raw, 90.0))
    if high - low <= 1e-9 and float(raw.max()) - float(raw.min()) > 1e-9:
        low, high = float(raw.min()), float(raw.max())
    if high - low <= 1e-9:
        normalized = np.full_like(raw, 0.5)
    else:
        normalized = np.clip((raw - low) / (high - low), 0.0, 1.0)
    return [
        {
            **item,
            "activity_percentile_score": round(float(score), 6),
            "normalization_p20": round(low, 6),
            "normalization_p90": round(high, 6),
        }
        for item, score in zip(samples, normalized, strict=True)
    ]


def select_supplemental_timestamps(
    start_seconds: float,
    end_seconds: float,
    base_timestamps: list[float],
    activity_samples: list[dict[str, float]],
    maximum_fraction: float = 0.25,
    bucket_seconds: float = 10.0,
    minimum_base_gap_seconds: float = 0.5,
) -> tuple[list[float], dict[str, Any]]:
    if not 0.0 <= maximum_fraction <= 0.25:
        raise ValueError("maximum_fraction_outside_contract")
    budget = int(math.floor(len(base_timestamps) * maximum_fraction + 1e-9))
    normalized = percentile_normalize(
        [
            sample
            for sample in activity_samples
            if start_seconds - 1e-9 <= float(sample["timestamp_seconds"]) <= end_seconds + 1e-9
        ]
    )
    buckets: dict[int, list[dict[str, float]]] = {}
    for sample in normalized:
        timestamp = float(sample["timestamp_seconds"])
        if any(abs(timestamp - base) < minimum_base_gap_seconds - 1e-9 for base in base_timestamps):
            continue
        bucket = int(max(0.0, timestamp - start_seconds) // bucket_seconds)
        buckets.setdefault(bucket, []).append(sample)
    peaks: list[dict[str, float]] = []
    for bucket, samples in sorted(buckets.items()):
        median = float(np.median([float(item["activity_percentile_score"]) for item in samples]))
        peak = max(
            samples,
            key=lambda item: (float(item["activity_percentile_score"]), -float(item["timestamp_seconds"])),
        )
        if float(peak["activity_percentile_score"]) + 1e-9 >= median:
            peaks.append({**peak, "time_bucket": bucket, "bucket_median_score": round(median, 6)})
    selected_peaks = sorted(
        peaks,
        key=lambda item: (-float(item["activity_percentile_score"]), int(item["time_bucket"])),
    )[:budget]
    selected = sorted(float(item["timestamp_seconds"]) for item in selected_peaks)
    return selected, {
        "strategy": "uniform_base_plus_percentile_bucketed_activity_supplement",
        "base_frame_count": len(base_timestamps),
        "maximum_extra_fraction": maximum_fraction,
        "extra_frame_budget": budget,
        "extra_frame_count": len(selected),
        "scan_interval_seconds": 0.5,
        "time_bucket_seconds": bucket_seconds,
        "at_most_one_extra_per_time_bucket": True,
        "minimum_base_gap_seconds": minimum_base_gap_seconds,
        "base_timestamps_seconds": [round(value, 6) for value in base_timestamps],
        "extra_timestamps_seconds": [round(value, 6) for value in selected],
        "selected_peaks": selected_peaks,
        "camera_motion_compensation": "phase_correlation_translation_before_frame_difference",
    }


def timestamps_to_frame_numbers(timestamps: list[float], fps: float, frame_count: int) -> list[int]:
    values: list[int] = []
    seen: set[int] = set()
    for timestamp in timestamps:
        number = min(frame_count - 1, max(0, int(round(timestamp * fps))))
        if number not in seen:
            seen.add(number)
            values.append(number)
    return values
