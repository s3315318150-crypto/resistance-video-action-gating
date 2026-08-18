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
    """Return unbounded low-resolution frame-difference and HSV-change scores."""
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
            raw_activity = max(0.0, 3.5 * frame_difference + 1.5 * histogram_change)
            samples.append(
                {
                    "timestamp_seconds": round(frame_number / fps, 6),
                    "frame_difference": round(frame_difference, 6),
                    "histogram_change": round(histogram_change, 6),
                    "raw_activity_score": round(raw_activity, 6),
                    # Selection replaces this compatibility field with a
                    # window-normalized score.
                    "activity_score": round(raw_activity, 6),
                }
            )
            previous_gray = gray
            previous_hist = hist
    finally:
        capture.release()
    return samples


def percentile_normalize_activity(
    activity_samples: list[dict[str, float]],
    low_percentile: float = 20.0,
    high_percentile: float = 90.0,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    """Normalize motion per window without clipping most active samples to one."""
    if not activity_samples:
        return [], {"percentile_low": 0.0, "percentile_high": 0.0}
    raw = np.asarray(
        [float(item.get("raw_activity_score", item.get("activity_score", 0.0))) for item in activity_samples],
        dtype=np.float64,
    )
    low = float(np.percentile(raw, low_percentile))
    high = float(np.percentile(raw, high_percentile))
    if high - low <= 1e-9 and float(raw.max()) - float(raw.min()) > 1e-9:
        low = float(raw.min())
        high = float(raw.max())
    if high - low <= 1e-9:
        normalized = np.full_like(raw, 0.5)
    else:
        normalized = np.clip((raw - low) / (high - low), 0.0, 1.0)
    result: list[dict[str, float]] = []
    for item, score in zip(activity_samples, normalized, strict=True):
        result.append(
            {
                **item,
                "raw_activity_score": round(
                    float(item.get("raw_activity_score", item.get("activity_score", 0.0))), 6
                ),
                "activity_score": round(float(score), 6),
            }
        )
    return result, {
        "percentile_low": round(low, 6),
        "percentile_high": round(high, 6),
        "normalization_low_percentile": float(low_percentile),
        "normalization_high_percentile": float(high_percentile),
    }


def _bucketed_candidates(
    samples: list[dict[str, float]],
    start_seconds: float,
    end_seconds: float,
    bucket_count: int,
    descending: bool,
) -> list[dict[str, float]]:
    buckets: list[list[dict[str, float]]] = [[] for _ in range(max(1, bucket_count))]
    duration = max(1e-9, end_seconds - start_seconds)
    for sample in samples:
        timestamp = float(sample["timestamp_seconds"])
        ratio = min(1.0 - 1e-12, max(0.0, (timestamp - start_seconds) / duration))
        buckets[min(len(buckets) - 1, int(ratio * len(buckets)))].append(sample)
    for bucket in buckets:
        bucket.sort(
            key=lambda item: (
                -float(item.get("activity_score", 0.0)) if descending else float(item.get("activity_score", 0.0)),
                float(item["timestamp_seconds"]),
            )
        )
    ordered: list[dict[str, float]] = []
    depth = 0
    while any(depth < len(bucket) for bucket in buckets):
        for bucket in buckets:
            if depth < len(bucket):
                ordered.append(bucket[depth])
        depth += 1
    return ordered


def select_timestamps(
    start_seconds: float,
    end_seconds: float,
    budget: int,
    activity_samples: list[dict[str, float]],
    anchor_interval_seconds: float = 5.0,
    minimum_dense_gap_seconds: float = 0.5,
    high_motion_fraction: float = 0.6,
    return_diagnostics: bool = False,
) -> list[float] | tuple[list[float], dict[str, Any]]:
    """Keep anchors, then distribute high- and low-motion samples over time."""
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
        values = sorted({anchors[int(index)] for index in indexes})
        diagnostic = {
            "anchor_selected_count": len(values),
            "high_motion_selected_count": 0,
            "low_motion_selected_count": 0,
            "anchor_timestamps_seconds": values,
            "high_motion_timestamps_seconds": [],
            "low_motion_timestamps_seconds": [],
            "measurement_candidate_timestamps_seconds": values,
            "activity_samples": [],
            "percentile_low": 0.0,
            "percentile_high": 0.0,
        }
        return (values, diagnostic) if return_diagnostics else values

    selected = list(anchors)
    normalized, normalization = percentile_normalize_activity(
        [
            item
            for item in activity_samples
            if start_seconds - 1e-9 <= float(item.get("timestamp_seconds", -1.0)) <= end_seconds + 1e-9
        ]
    )
    remaining = budget - len(selected)
    high_target = min(remaining, max(0, int(round(remaining * high_motion_fraction))))
    low_target = remaining - high_target
    bucket_count = min(8, max(1, int(math.ceil((end_seconds - start_seconds) / 10.0))))
    high_pool = [item for item in normalized if float(item["activity_score"]) >= 0.5] or normalized
    low_pool = [item for item in normalized if float(item["activity_score"]) < 0.5] or normalized
    high_ranked = _bucketed_candidates(high_pool, start_seconds, end_seconds, bucket_count, True)
    low_ranked = _bucketed_candidates(low_pool, start_seconds, end_seconds, bucket_count, False)
    selected_by_role: dict[str, list[float]] = {"high_motion": [], "low_motion": []}

    def add_candidates(candidates: list[dict[str, float]], target: int, role: str) -> None:
        for sample in candidates:
            if len(selected_by_role[role]) >= target or len(selected) >= budget:
                break
            timestamp = round(float(sample["timestamp_seconds"]), 6)
            if any(abs(timestamp - existing) < minimum_dense_gap_seconds - 1e-9 for existing in selected):
                continue
            selected.append(timestamp)
            selected_by_role[role].append(timestamp)

    add_candidates(high_ranked, high_target, "high_motion")
    add_candidates(low_ranked, low_target, "low_motion")
    if len(selected) < budget:
        combined = _bucketed_candidates(normalized, start_seconds, end_seconds, bucket_count, True)
        add_candidates(combined, budget, "high_motion")
    if len(selected) < budget:
        fallback_count = max(budget * 3, 2)
        for timestamp in np.linspace(start_seconds, end_seconds, fallback_count):
            value = round(float(timestamp), 6)
            if any(abs(value - existing) < minimum_dense_gap_seconds - 1e-9 for existing in selected):
                continue
            selected.append(value)
            if len(selected) >= budget:
                break
    values = sorted(selected[:budget])
    diagnostic = {
        **normalization,
        "time_bucket_count": bucket_count,
        "high_motion_budget_fraction": high_motion_fraction,
        "low_motion_budget_fraction": 1.0 - high_motion_fraction,
        "anchor_selected_count": len(anchors),
        "high_motion_selected_count": len(selected_by_role["high_motion"]),
        "low_motion_selected_count": len(selected_by_role["low_motion"]),
        "anchor_timestamps_seconds": anchors,
        "high_motion_timestamps_seconds": selected_by_role["high_motion"],
        "low_motion_timestamps_seconds": selected_by_role["low_motion"],
        "measurement_candidate_timestamps_seconds": sorted(set(anchors + selected_by_role["low_motion"])),
        "activity_samples": normalized,
    }
    return (values, diagnostic) if return_diagnostics else values


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
    activity_samples: list[dict[str, float]] | None = None,
) -> tuple[list[int], dict[str, Any]]:
    activity = activity_samples
    if activity is None:
        activity = scan_activity(video, start_seconds, end_seconds)
    timestamps, selection = select_timestamps(
        start_seconds,
        end_seconds,
        budget,
        activity,
        return_diagnostics=True,
    )
    numbers = timestamps_to_frame_numbers(timestamps, fps, frame_count)
    return numbers, {
        "strategy": "percentile_bucketed_tcs_with_low_motion_reserve",
        "frame_budget": budget,
        "selected_frame_count": len(numbers),
        "scan_interval_seconds": 0.5,
        "anchor_interval_seconds": 5.0,
        "minimum_dense_gap_seconds": 0.5,
        "activity_sample_count": len(activity),
        "selected_timestamps_seconds": [round(number / fps, 6) for number in numbers],
        "peak_activity_scores": sorted(
            (float(item["activity_score"]) for item in selection["activity_samples"]), reverse=True
        )[:8],
        **selection,
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
