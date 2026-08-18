#!/usr/bin/env python3
"""Locate the last experiment pass using an orange instrument as a marker.

The recording convention is:

* the orange-red instrument is in the upper-left corner before an experiment;
* it is removed to start the experiment; and
* it is put back when the experiment is over.

Some recordings contain a restart (put back, then removed again).  This
script treats each ``present -> absent -> present`` sequence as one pass and
selects the pass with the latest start.  It is intentionally label-blind and
only emits a time/frame manifest; the source video is never modified.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


SCHEMA_VERSION = "marker_filter.v1"
DEFAULT_ROI = (0.0, 0.0, 0.25, 0.22)


@dataclass(frozen=True)
class MarkerSample:
    frame_number: int
    timestamp_seconds: float
    score: float
    warm_fraction: float
    largest_component_fraction: float
    state: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "frame_number": self.frame_number,
            "timestamp_seconds": self.timestamp_seconds,
            "marker_score": self.score,
            "warm_fraction": self.warm_fraction,
            "largest_component_fraction": self.largest_component_fraction,
            "instrument_state": self.state,
        }


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def validate_roi(values: Iterable[float]) -> tuple[float, float, float, float]:
    roi = tuple(float(value) for value in values)
    if len(roi) != 4:
        raise ValueError("ROI must contain x1 y1 x2 y2")
    x1, y1, x2, y2 = roi
    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        raise ValueError("ROI must satisfy 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1")
    return roi


def resize_for_analysis(frame: np.ndarray, width: int) -> np.ndarray:
    if width <= 0 or frame.shape[1] <= width:
        return frame
    scale = width / frame.shape[1]
    return cv2.resize(frame, (width, max(1, int(round(frame.shape[0] * scale)))), interpolation=cv2.INTER_AREA)


def marker_features(frame: np.ndarray, roi: tuple[float, float, float, float]) -> tuple[float, float, float]:
    """Return a robust orange/red occupancy score for the marker ROI.

    Thin red wires can cross the ROI.  Combining total warm-color occupancy
    with the largest connected component makes a solid instrument score much
    higher than a few disconnected wire pixels.
    """

    height, width = frame.shape[:2]
    x1, y1, x2, y2 = roi
    left, top = int(math.floor(x1 * width)), int(math.floor(y1 * height))
    right, bottom = int(math.ceil(x2 * width)), int(math.ceil(y2 * height))
    crop = frame[max(0, top) : min(height, bottom), max(0, left) : min(width, right)]
    if crop.size == 0:
        return 0.0, 0.0, 0.0

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    warm = (
        (((hue <= 30) | (hue >= 170)) & (saturation >= 75) & (value >= 45))
    ).astype(np.uint8) * 255
    # Keep broad device faces while removing isolated compression/noise pixels.
    kernel = np.ones((3, 3), dtype=np.uint8)
    warm = cv2.morphologyEx(warm, cv2.MORPH_OPEN, kernel)
    warm = cv2.morphologyEx(warm, cv2.MORPH_CLOSE, kernel)
    warm_fraction = float(np.count_nonzero(warm)) / float(warm.size)

    component_fraction = 0.0
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(warm, connectivity=8)
    if component_count > 1:
        largest_area = int(stats[1:, cv2.CC_STAT_AREA].max())
        component_fraction = largest_area / float(warm.size)

    # The score is deliberately not a fixed absolute threshold.  Thresholds
    # are inferred from the video's low/high score clusters below.
    score = clamp(0.62 * warm_fraction + 0.38 * component_fraction, 0.0, 1.0)
    return score, warm_fraction, component_fraction


def moving_median(values: list[float], window: int) -> list[float]:
    if not values:
        return []
    window = max(1, int(window))
    if window % 2 == 0:
        window += 1
    radius = window // 2
    result: list[float] = []
    for index in range(len(values)):
        start, end = max(0, index - radius), min(len(values), index + radius + 1)
        result.append(float(np.median(values[start:end])))
    return result


def infer_thresholds(scores: list[float]) -> dict[str, float | bool]:
    """Fit two deterministic 1-D clusters and return hysteresis thresholds."""

    if not scores:
        return {
            "low_center": 0.0,
            "high_center": 0.0,
            "separation": 0.0,
            "absent_threshold": 0.0,
            "present_threshold": 0.0,
            "bimodal": False,
        }

    low, high = float(min(scores)), float(max(scores))
    for _ in range(20):
        midpoint = (low + high) / 2.0
        lower = [score for score in scores if score <= midpoint]
        upper = [score for score in scores if score > midpoint]
        if not lower or not upper:
            break
        next_low = float(sum(lower) / len(lower))
        next_high = float(sum(upper) / len(upper))
        if abs(next_low - low) + abs(next_high - high) < 1e-8:
            low, high = next_low, next_high
            break
        low, high = next_low, next_high

    separation = max(0.0, high - low)
    # A very small separation means the marker is not observable in this
    # recording.  Keep the thresholds usable but mark the result non-bimodal.
    bimodal = separation >= max(0.025, 0.08 * max(high, 1e-6))
    if not bimodal:
        midpoint = float(np.median(scores))
        return {
            "low_center": round(low, 6),
            "high_center": round(high, 6),
            "separation": round(separation, 6),
            "absent_threshold": round(midpoint, 6),
            "present_threshold": round(midpoint, 6),
            "bimodal": False,
        }

    return {
        "low_center": round(low, 6),
        "high_center": round(high, 6),
        "separation": round(separation, 6),
        "absent_threshold": round(low + 0.35 * separation, 6),
        "present_threshold": round(low + 0.65 * separation, 6),
        "bimodal": True,
    }


def classify_states(
    scores: list[float],
    thresholds: dict[str, float | bool],
    minimum_state_samples: int,
) -> list[str]:
    """Apply hysteresis and require a state to persist before switching."""

    if not scores:
        return []
    absent_threshold = float(thresholds["absent_threshold"])
    present_threshold = float(thresholds["present_threshold"])
    midpoint = (absent_threshold + present_threshold) / 2.0
    state = "present" if scores[0] >= midpoint else "absent"
    states = [state]
    pending: str | None = None
    pending_count = 0
    minimum_state_samples = max(1, int(minimum_state_samples))

    for score in scores[1:]:
        candidate: str | None = None
        if state == "present" and score <= absent_threshold:
            candidate = "absent"
        elif state == "absent" and score >= present_threshold:
            candidate = "present"

        if candidate is None:
            pending = None
            pending_count = 0
        elif candidate == pending:
            pending_count += 1
        else:
            pending = candidate
            pending_count = 1

        if pending is not None and pending_count >= minimum_state_samples:
            state = pending
            pending = None
            pending_count = 0
        states.append(state)
    return states


def find_cycles(
    samples: list[MarkerSample],
    duration_seconds: float,
    frame_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return all marker transitions and experiment passes."""

    if not samples:
        return [], []
    transitions: list[dict[str, Any]] = []
    cycles: list[dict[str, Any]] = []
    active_start: MarkerSample | None = None
    previous = samples[0]

    for current in samples[1:]:
        if current.state == previous.state:
            previous = current
            continue
        transition_type = f"{previous.state}_to_{current.state}"
        transitions.append(
            {
                "type": transition_type,
                "frame_number": current.frame_number,
                "timestamp_seconds": current.timestamp_seconds,
                "score": current.score,
            }
        )
        if previous.state == "present" and current.state == "absent":
            active_start = current
        elif previous.state == "absent" and current.state == "present":
            if active_start is None and samples[0].state == "absent":
                # A clip may begin after removal.  Only infer a zero-second
                # start when a later put-back is actually observed; an
                # all-absent clip is otherwise reported as not found.
                active_start = MarkerSample(0, 0.0, previous.score, previous.warm_fraction, previous.largest_component_fraction, "absent")
            if active_start is None:
                previous = current
                continue
            cycles.append(
                {
                    "start_frame": active_start.frame_number,
                    "end_frame_exclusive": current.frame_number,
                    "start_seconds": active_start.timestamp_seconds,
                    "end_seconds_exclusive": current.timestamp_seconds,
                    "duration_seconds": round(max(0.0, current.timestamp_seconds - active_start.timestamp_seconds), 3),
                    "complete": True,
                }
            )
            active_start = None
        previous = current

    if active_start is not None:
        end_frame = max(active_start.frame_number + 1, frame_count)
        cycles.append(
            {
                "start_frame": active_start.frame_number,
                "end_frame_exclusive": end_frame,
                "start_seconds": active_start.timestamp_seconds,
                "end_seconds_exclusive": round(max(active_start.timestamp_seconds, duration_seconds), 3),
                "duration_seconds": round(max(0.0, duration_seconds - active_start.timestamp_seconds), 3),
                "complete": False,
            }
        )
    return transitions, cycles


def scan_video(
    video_path: Path,
    roi: tuple[float, float, float, float],
    sample_fps: float,
    analysis_width: int,
    smooth_seconds: float,
    minimum_state_seconds: float,
) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(round(float(capture.get(cv2.CAP_PROP_FRAME_COUNT))))
        width = int(round(float(capture.get(cv2.CAP_PROP_FRAME_WIDTH))))
        height = int(round(float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
            raise RuntimeError("Video metadata is invalid")
        sample_fps = clamp(float(sample_fps), 0.05, fps)
        sample_step = max(1, int(round(fps / sample_fps)))
        raw: list[tuple[int, float, float, float, float]] = []
        frame_number = 0
        while True:
            if not capture.grab():
                break
            if frame_number % sample_step == 0:
                ok, frame = capture.retrieve()
                if not ok or frame is None:
                    break
                analysis = resize_for_analysis(frame, analysis_width)
                score, warm_fraction, component_fraction = marker_features(analysis, roi)
                raw.append((frame_number, frame_number / fps, score, warm_fraction, component_fraction))
            frame_number += 1
    finally:
        capture.release()

    if not raw:
        raise RuntimeError("No frames were sampled")
    raw_scores = [record[2] for record in raw]
    window = max(1, int(round(float(smooth_seconds) * sample_fps)))
    smoothed_scores = moving_median(raw_scores, window)
    thresholds = infer_thresholds(smoothed_scores)
    min_samples = max(1, int(math.ceil(float(minimum_state_seconds) * sample_fps)))
    states = classify_states(smoothed_scores, thresholds, min_samples)
    samples = [
        MarkerSample(
            frame_number=record[0],
            timestamp_seconds=round(record[1], 3),
            score=round(smoothed_scores[index], 6),
            warm_fraction=round(record[3], 6),
            largest_component_fraction=round(record[4], 6),
            state=states[index],
        )
        for index, record in enumerate(raw)
    ]
    duration_seconds = frame_count / fps
    transitions, cycles = find_cycles(samples, duration_seconds, frame_count)
    selected = max(cycles, key=lambda cycle: int(cycle["start_frame"])) if cycles else None
    source_stat = video_path.stat()
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "method": "upper_left_orange_instrument_state",
        "source_video": str(video_path.resolve()),
        "source_video_id": video_path.name,
        "source_video_fingerprint": {
            "size_bytes": source_stat.st_size,
            "mtime_utc": datetime.fromtimestamp(source_stat.st_mtime, timezone.utc).replace(microsecond=0).isoformat(),
            "sha256_not_computed": True,
        },
        "video_metadata": {
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "duration_seconds": round(duration_seconds, 3),
        },
        "roi_normalized_xyxy": [round(value, 6) for value in roi],
        "detector_config": {
            "sample_fps": sample_fps,
            "analysis_width": analysis_width,
            "smooth_seconds": smooth_seconds,
            "minimum_state_seconds": minimum_state_seconds,
            "minimum_state_samples": min_samples,
            "hue_ranges": [[0, 30], [170, 179]],
            "minimum_saturation": 75,
            "minimum_value": 45,
        },
        "thresholds": thresholds,
        "transitions": transitions,
        "cycles": cycles,
        "selected_cycle": selected,
        "selection_status": "selected" if selected else "not_found",
        "selection_note": (
            "Latest present->absent->present pass selected; an incomplete final pass is kept when it starts latest."
            if selected
            else "No stable instrument removal was detected. Inspect the ROI or provide a manual interval."
        ),
        "samples": [sample.as_dict() for sample in samples],
    }
    if selected:
        result["selected_time_range_seconds"] = [selected["start_seconds"], selected["end_seconds_exclusive"]]
        result["selected_frame_range_inclusive"] = [selected["start_frame"], max(selected["start_frame"], selected["end_frame_exclusive"] - 1)]
    else:
        result["selected_time_range_seconds"] = None
        result["selected_frame_range_inclusive"] = None
    return result


def emit_frames(video_path: Path, result: dict[str, Any], output_dir: Path, frame_sampling_fps: float) -> list[dict[str, Any]]:
    selected = result.get("selected_cycle")
    if not isinstance(selected, dict):
        return []
    metadata = result["video_metadata"]
    fps = float(metadata["fps"])
    start_frame = int(selected["start_frame"])
    end_frame = int(selected["end_frame_exclusive"])
    step = max(1, int(round(fps / clamp(float(frame_sampling_fps), 0.05, fps))))
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to reopen video for frame emission: {video_path}")
    emitted: list[dict[str, Any]] = []
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_number = start_frame
        while frame_number < end_frame:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            if (frame_number - start_frame) % step == 0:
                timestamp = frame_number / fps
                output_path = output_dir / f"frame_{frame_number:08d}_{timestamp:010.3f}s.jpg"
                if not cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise RuntimeError(f"Unable to write frame: {output_path}")
                emitted.append(
                    {
                        "frame_number": frame_number,
                        "timestamp_seconds": round(timestamp, 3),
                        "output_path": str(output_path.resolve()),
                    }
                )
            frame_number += 1
    finally:
        capture.release()
    return emitted


def process_one(
    video_path: Path,
    output_dir: Path,
    roi: tuple[float, float, float, float],
    sample_fps: float,
    analysis_width: int,
    smooth_seconds: float,
    minimum_state_seconds: float,
    emit_frame_fps: float | None,
) -> Path:
    result = scan_video(video_path, roi, sample_fps, analysis_width, smooth_seconds, minimum_state_seconds)
    if emit_frame_fps is not None:
        frames_dir = output_dir / f"{video_path.stem}_frames"
        emitted = emit_frames(video_path, result, frames_dir, emit_frame_fps)
        result["emitted_frames"] = emitted
        result["emitted_frame_count"] = len(emitted)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{video_path.stem}.marker_filter.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "source_video": str(video_path),
        "output_manifest": str(output_path.resolve()),
        "selection_status": result["selection_status"],
        "selected_time_range_seconds": result["selected_time_range_seconds"],
        "cycles_detected": len(result["cycles"]),
        "emitted_frame_count": result.get("emitted_frame_count", 0),
    }, ensure_ascii=False))
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos", nargs="+", type=Path, help="One or more input videos")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/marker_filtered"))
    parser.add_argument("--roi", nargs=4, type=float, default=DEFAULT_ROI, metavar=("X1", "Y1", "X2", "Y2"), help="Upper-left marker ROI in normalized coordinates")
    parser.add_argument("--sample-fps", type=float, default=1.0, help="Marker scan rate (default: 1 fps)")
    parser.add_argument("--analysis-width", type=int, default=480, help="Width used for color analysis")
    parser.add_argument("--smooth-seconds", type=float, default=3.0, help="Median smoothing window")
    parser.add_argument("--minimum-state-seconds", type=float, default=3.0, help="Required duration before accepting a state change")
    parser.add_argument("--emit-frames", type=float, metavar="FPS", help="Also export selected source frames at this FPS")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        roi = validate_roi(args.roi)
        if args.analysis_width <= 0:
            raise ValueError("analysis-width must be positive")
        if args.sample_fps <= 0 or args.smooth_seconds < 0 or args.minimum_state_seconds < 0:
            raise ValueError("sample-fps, smooth-seconds, and minimum-state-seconds must be non-negative (sample-fps > 0)")
        for video in args.videos:
            if not video.is_file():
                raise FileNotFoundError(video)
            process_one(video, args.output_dir, roi, args.sample_fps, args.analysis_width, args.smooth_seconds, args.minimum_state_seconds, args.emit_frames)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
