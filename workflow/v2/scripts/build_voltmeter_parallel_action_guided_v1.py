#!/usr/bin/env python3
"""Build action-guided visual packets for the voltmeter-parallel rubric.

The only analysis inputs are the fixed action-segmentation summary and the
original resistance-experiment MP4 files.  Action stages narrow where to look;
only recording-stage successors are eligible, and the stages do not establish
any circuit-topology conclusion.  The output is a
label-blind evidence packet that can be checked by ``preflight_qwen_request``
without making a model request.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2


ROOT = Path(__file__).resolve().parent.parent
ACTION_SUMMARY_PATH = (
    ROOT
    / "outputs"
    / "action_minutes"
    / "action_segments_summary.json"
)
DEFAULT_SOURCE_DIR = ROOT / "data" / "videos"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "voltmeter_parallel_action_guided_v1"

RUBRIC_ID = "resistance.voltmeter_parallel_connection"
RUBRIC_NAME = "电压表正确并联在待测电阻两端"
WIRING_STAGES = {"circuit_wiring", "circuit_rewiring"}
PRE_WIRING_SECONDS = 20.0
POST_SUCCESSOR_SECONDS = 15.0
POST_WIRING_START_GUARD_SECONDS = 1.0
SUCCESSOR_END_GUARD_SECONDS = 1.0
MIN_STABLE_SUCCESSOR_SECONDS = 4.0
SAMPLE_FPS = 1.0
MIN_BACKUP_SEPARATION_SECONDS = 3.0
PANORAMA_EDGE = 1280
DETAIL_EDGE = 1600
PANORAMA_QUALITY = 84
DETAIL_QUALITY = 88
MAX_PACKET_IMAGES = 8
MAX_SINGLE_IMAGE_BYTES = 2 * 1024 * 1024
TARGET_SINGLE_IMAGE_BYTES = 1500 * 1024
MAX_TOTAL_IMAGE_BYTES = 10 * 1024 * 1024
MAX_ESTIMATED_BASE64_BYTES = 14 * 1024 * 1024
MAX_SOURCE_ROI_AREA_RATIO = 0.80
MIN_CONTEXT_FRAME_COUNT = 2
MAX_CONTEXT_FRAME_COUNT = MAX_PACKET_IMAGES - 4
MAX_SAMPLE_FPS = 4.0
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_slug(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "item"


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def rounded(value: float, places: int = 6) -> float:
    return round(float(value), places)


def stage_is_recording(stage: Any) -> bool:
    return isinstance(stage, str) and stage.startswith("recording")


def clean_segment(raw: Any, index: int) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"segments[{index}] must be an object")
    stage = raw.get("stage")
    start = raw.get("start_seconds")
    end = raw.get("end_seconds")
    if not isinstance(stage, str) or not stage:
        raise ValueError(f"segments[{index}].stage is invalid")
    if not finite(start) or not finite(end) or float(end) < float(start):
        raise ValueError(f"segments[{index}] has invalid timing")
    result = {
        "segment_index": index,
        "stage": stage,
        "start_seconds": float(start),
        "end_seconds": float(end),
    }
    for key in ("stage_label", "start_source", "end_source", "start_evidence"):
        if isinstance(raw.get(key), str):
            result[key] = raw[key]
    return result


def action_events(record: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Create wiring-to-recording pairs for recording-stage topology observation."""
    video_id = record.get("source_video_id")
    window = record.get("fixed_experiment_window_seconds")
    raw_segments = record.get("segments")
    if not isinstance(video_id, str) or not video_id:
        raise ValueError("record.source_video_id is invalid")
    if (
        not isinstance(window, list)
        or len(window) != 2
        or not finite(window[0])
        or not finite(window[1])
        or float(window[1]) <= float(window[0])
    ):
        raise ValueError(f"{video_id}: fixed_experiment_window_seconds is invalid")
    if not isinstance(raw_segments, list):
        raise ValueError(f"{video_id}: segments is invalid")

    experiment_start, experiment_end = float(window[0]), float(window[1])
    segments = [clean_segment(item, index) for index, item in enumerate(raw_segments)]
    for previous, current in zip(segments, segments[1:]):
        if current["start_seconds"] < previous["end_seconds"] - 1e-6:
            raise ValueError(
                f"{video_id}: segments {previous['segment_index']} and {current['segment_index']} overlap"
            )
    events: list[dict[str, Any]] = []
    ignored_wiring: list[dict[str, Any]] = []
    for index, wiring in enumerate(segments):
        if wiring["stage"] not in WIRING_STAGES:
            continue
        successor = segments[index + 1] if index + 1 < len(segments) else None
        if successor is None or not stage_is_recording(successor["stage"]):
            ignored_wiring.append(
                {
                    "segment_index": wiring["segment_index"],
                    "stage": wiring["stage"],
                    "end_seconds": wiring["end_seconds"],
                    "reason": "not_immediately_followed_by_recording",
                }
            )
            continue
        retrieval_window = [
            wiring["end_seconds"] - PRE_WIRING_SECONDS,
            min(successor["end_seconds"], successor["start_seconds"] + POST_SUCCESSOR_SECONDS),
        ]
        stable_start = max(wiring["end_seconds"], successor["start_seconds"]) + POST_WIRING_START_GUARD_SECONDS
        stable_end = min(
            successor["end_seconds"] - SUCCESSOR_END_GUARD_SECONDS,
            successor["start_seconds"] + POST_SUCCESSOR_SECONDS,
            experiment_end,
        )
        stable_start = clamp(stable_start, experiment_start, experiment_end)
        stable_end = clamp(stable_end, experiment_start, experiment_end)
        if stable_end - stable_start < MIN_STABLE_SUCCESSOR_SECONDS:
            ignored_wiring.append(
                {
                    "segment_index": wiring["segment_index"],
                    "stage": wiring["stage"],
                    "end_seconds": wiring["end_seconds"],
                    "reason": "insufficient_post_wiring_stability",
                    "stable_successor_window_seconds": [rounded(stable_start), rounded(stable_end)],
                    "minimum_stable_successor_seconds": MIN_STABLE_SUCCESSOR_SECONDS,
                }
            )
            continue
        events.append(
            {
                "event_index": len(events) + 1,
                "event_id": f"event_{len(events) + 1:02d}_{wiring['stage']}_to_{successor['stage']}",
                "wiring_segment": wiring,
                "successor_segment": successor,
                "experiment_window_seconds": [experiment_start, experiment_end],
                "retrieval_window_seconds": [
                    clamp(retrieval_window[0], experiment_start, experiment_end),
                    clamp(retrieval_window[1], experiment_start, experiment_end),
                ],
                "stable_successor_window_seconds": [stable_start, stable_end],
                "event_window_seconds": [stable_start, stable_end],
            }
        )
    return events, ignored_wiring


def source_video_candidates(source_dir: Path) -> list[Path]:
    return sorted(
        path.resolve()
        for path in source_dir.iterdir()
        if path.is_file() and path.suffix.casefold() in VIDEO_SUFFIXES
    )


def find_source_video(source_video_id: str, candidates: list[Path]) -> tuple[Path, str]:
    requested_name = Path(source_video_id).name.casefold()
    exact = [path for path in candidates if path.name.casefold() == requested_name]
    if len(exact) == 1:
        return exact[0], "exact_source_video_id"
    requested_stem = Path(source_video_id).stem.casefold()
    stem_matches = [path for path in candidates if path.stem.casefold() == requested_stem]
    if len(stem_matches) == 1:
        return stem_matches[0], "unique_source_stem"
    details = [str(path) for path in (exact or stem_matches or candidates)]
    raise ValueError(f"Could not resolve one source video for {source_video_id}: {details}")


def video_metadata(video_path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open source video: {video_path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(round(float(capture.get(cv2.CAP_PROP_FRAME_COUNT))))
        width = int(round(float(capture.get(cv2.CAP_PROP_FRAME_WIDTH))))
        height = int(round(float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    finally:
        capture.release()
    if fps <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(f"Source video metadata is incomplete: {video_path}")
    return {
        "path": str(video_path.resolve()),
        "fps": rounded(fps),
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_seconds": rounded(frame_count / fps) if frame_count > 0 else None,
        "direct_original_video_decode": True,
    }


def sample_timestamps(
    start_seconds: float,
    end_seconds: float,
    sampling_fps: float = SAMPLE_FPS,
) -> list[float]:
    if not finite(sampling_fps) or float(sampling_fps) <= 0:
        raise ValueError("sampling_fps must be positive and finite")
    rate = float(sampling_fps)
    first = math.ceil(start_seconds * rate - 1e-9)
    last = math.floor(end_seconds * rate + 1e-9)
    values = [rounded(index / rate, 6) for index in range(first, last + 1)]
    if values:
        return values
    return [rounded((start_seconds + end_seconds) / 2.0, 3)]


def analysis_image(image: Any) -> tuple[Any, float, float]:
    """Return a small grayscale frame plus sharpness and a hand-like occupancy proxy."""
    small = image
    height, width = image.shape[:2]
    edge = max(height, width)
    if edge > 480:
        scale = 480.0 / edge
        small = cv2.resize(
            image,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    thumbnail = cv2.resize(gray, (320, 180), interpolation=cv2.INTER_AREA)
    sharpness = float(cv2.Laplacian(thumbnail, cv2.CV_64F).var())

    ycrcb = cv2.cvtColor(small, cv2.COLOR_BGR2YCrCb)
    mask = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
    mask_height, mask_width = mask.shape[:2]
    y1, y2 = round(mask_height * 0.16), round(mask_height * 0.92)
    x1, x2 = round(mask_width * 0.08), round(mask_width * 0.92)
    central = mask[y1:y2, x1:x2]
    skin_like_fraction = (
        float(cv2.countNonZero(central)) / float(max(1, central.shape[0] * central.shape[1]))
    )
    return thumbnail, sharpness, skin_like_fraction


def decoded_frame_number(capture: Any, timestamp_seconds: float, fps: float) -> int:
    position = float(capture.get(cv2.CAP_PROP_POS_FRAMES))
    if math.isfinite(position) and position > 0:
        return max(0, int(round(position)) - 1)
    return max(0, int(round(timestamp_seconds * fps)))


def decode_samples(video_path: Path, timestamps: list[float], source_fps: float) -> list[dict[str, Any]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open source video: {video_path}")
    samples: list[dict[str, Any]] = []
    try:
        for timestamp_seconds in timestamps:
            capture.set(cv2.CAP_PROP_POS_MSEC, float(timestamp_seconds) * 1000.0)
            ok, image = capture.read()
            if not ok or image is None or image.size == 0:
                continue
            thumbnail, sharpness, skin_like_fraction = analysis_image(image)
            samples.append(
                {
                    "source_timestamp_seconds": float(timestamp_seconds),
                    "source_frame_number": decoded_frame_number(capture, float(timestamp_seconds), source_fps),
                    "sharpness_raw": sharpness,
                    "skin_like_fraction": skin_like_fraction,
                    "thumbnail": thumbnail,
                }
            )
    finally:
        capture.release()
    return samples


def thumbnail_motion(left: Any, right: Any) -> tuple[float, float]:
    difference = cv2.absdiff(left, right)
    mean_absolute_difference = float(cv2.mean(difference)[0])
    _, changed = cv2.threshold(difference, 18, 255, cv2.THRESH_BINARY)
    changed_fraction = float(cv2.countNonZero(changed)) / float(max(1, changed.shape[0] * changed.shape[1]))
    return mean_absolute_difference, changed_fraction


def assign_selection_scores(samples: list[dict[str, Any]], wiring_end_seconds: float) -> None:
    if not samples:
        return
    log_sharpness = [math.log1p(max(0.0, float(sample["sharpness_raw"]))) for sample in samples]
    low, high = min(log_sharpness), max(log_sharpness)
    for index, sample in enumerate(samples):
        neighbor_motion: list[tuple[float, float]] = []
        if index > 0:
            neighbor_motion.append(thumbnail_motion(sample["thumbnail"], samples[index - 1]["thumbnail"]))
        if index + 1 < len(samples):
            neighbor_motion.append(thumbnail_motion(sample["thumbnail"], samples[index + 1]["thumbnail"]))
        if neighbor_motion:
            motion_mad = sum(item[0] for item in neighbor_motion) / len(neighbor_motion)
            changed_fraction = sum(item[1] for item in neighbor_motion) / len(neighbor_motion)
        else:
            motion_mad = 0.0
            changed_fraction = 0.0
        sharpness_normalized = 0.5 if high - low < 1e-9 else (log_sharpness[index] - low) / (high - low)
        stability_score = clamp(1.0 - motion_mad / 40.0, 0.0, 1.0)
        low_motion_score = clamp(1.0 - changed_fraction, 0.0, 1.0)
        low_occlusion_proxy_score = clamp(
            1.0 - (0.65 * changed_fraction + 0.35 * float(sample["skin_like_fraction"])),
            0.0,
            1.0,
        )
        after_wiring_end = float(sample["source_timestamp_seconds"]) >= wiring_end_seconds
        selection_score = (
            0.38 * sharpness_normalized
            + 0.23 * stability_score
            + 0.20 * low_motion_score
            + 0.19 * low_occlusion_proxy_score
        )
        sample["selection_features"] = {
            "sharpness_normalized": rounded(sharpness_normalized),
            "stability_score": rounded(stability_score),
            "low_motion_score": rounded(low_motion_score),
            "low_occlusion_proxy_score": rounded(low_occlusion_proxy_score),
            "after_wiring_end": after_wiring_end,
            "temporal_motion_mad": rounded(motion_mad),
            "temporal_changed_fraction": rounded(changed_fraction),
            "skin_like_fraction_proxy": rounded(float(sample["skin_like_fraction"])),
        }
        sample["selection_score"] = rounded(selection_score)


def thumbnail_mad(left: dict[str, Any], right: dict[str, Any]) -> float:
    return float(cv2.mean(cv2.absdiff(left["thumbnail"], right["thumbnail"]))[0])


def choose_context_samples(
    samples: list[dict[str, Any]],
    context_frame_count: int,
) -> list[dict[str, Any]]:
    if context_frame_count < MIN_CONTEXT_FRAME_COUNT:
        raise ValueError("At least two post-wiring stable overview contexts are required")
    if len(samples) < context_frame_count:
        raise ValueError("Not enough post-wiring stable samples for the requested overview contexts")
    if not all(bool(item["selection_features"].get("after_wiring_end")) for item in samples):
        raise ValueError("Candidate samples must all be after the wiring boundary")
    ranked = sorted(
        samples,
        key=lambda item: (
            float(item["selection_score"]),
            bool(item["selection_features"]["after_wiring_end"]),
            float(item["source_timestamp_seconds"]),
        ),
        reverse=True,
    )
    selected = [ranked[0]]
    while len(selected) < context_frame_count:
        remaining = [item for item in ranked if item not in selected]
        separated = [
            item
            for item in remaining
            if all(
                abs(float(item["source_timestamp_seconds"]) - float(existing["source_timestamp_seconds"]))
                >= MIN_BACKUP_SEPARATION_SECONDS
                for existing in selected
            )
        ]
        diverse = [
            item
            for item in separated
            if all(thumbnail_mad(existing, item) > 3.0 for existing in selected)
        ]
        selected.append((diverse or separated or remaining)[0])
    primary = selected[0]
    for index, context in enumerate(selected[1:], start=2):
        context["context_thumbnail_mad_from_primary"] = rounded(thumbnail_mad(primary, context))
        context["context_selection_ordinal"] = index
    return selected


def decode_frame_at(video_path: Path, timestamp_seconds: float, source_fps: float) -> tuple[Any, int]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open source video: {video_path}")
    try:
        capture.set(cv2.CAP_PROP_POS_MSEC, float(timestamp_seconds) * 1000.0)
        ok, image = capture.read()
        if not ok or image is None or image.size == 0:
            raise RuntimeError(f"Unable to decode source video frame at {timestamp_seconds:.3f}s: {video_path}")
        return image, decoded_frame_number(capture, timestamp_seconds, source_fps)
    finally:
        capture.release()


def resize_edge(image: Any, maximum_edge: int) -> Any:
    height, width = image.shape[:2]
    edge = max(height, width)
    if edge <= maximum_edge:
        return image.copy()
    scale = maximum_edge / float(edge)
    return cv2.resize(
        image,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def write_jpeg(image: Any, path: Path, maximum_edge: int, preferred_quality: int) -> dict[str, Any]:
    """Write within the local preflight budget while preserving the requested edge when possible."""
    encoded_bytes: bytes | None = None
    prepared: Any | None = None
    selected_quality: int | None = None
    for edge_scale in (1.0, 0.90, 0.80, 0.70):
        edge = max(640, round(maximum_edge * edge_scale))
        candidate_image = resize_edge(image, edge)
        for quality in range(preferred_quality, 45, -6):
            ok, encoded = cv2.imencode(".jpg", candidate_image, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if not ok:
                continue
            payload = encoded.tobytes()
            encoded_bytes = payload
            prepared = candidate_image
            selected_quality = quality
            if len(payload) <= TARGET_SINGLE_IMAGE_BYTES:
                break
        if encoded_bytes is not None and len(encoded_bytes) <= TARGET_SINGLE_IMAGE_BYTES:
            break
    if encoded_bytes is None or prepared is None or selected_quality is None:
        raise RuntimeError(f"Unable to JPEG encode image: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded_bytes)
    height, width = prepared.shape[:2]
    return {
        "path": str(path.resolve()),
        "width": int(width),
        "height": int(height),
        "file_size_bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
        "jpeg_quality": selected_quality,
    }


def bbox_area_ratio(bbox: list[int], width: int, height: int) -> float:
    return max(0.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])) / float(max(1, width * height))


def unverified_region_specs(width: int, height: int) -> list[tuple[str, list[int]]]:
    """Four wide, overlapping tiles preserve context without pretending to identify components."""
    crop_width = max(2, round(width * 0.56))
    crop_height = max(2, round(height * 0.62))
    positions = [
        (0, 0),
        (width - crop_width, 0),
        (0, height - crop_height),
        (width - crop_width, height - crop_height),
    ]
    return [
        (f"unverified_visual_region_{index:02d}", [int(x), int(y), int(x + crop_width), int(y + crop_height)])
        for index, (x, y) in enumerate(positions, start=1)
    ]


def sample_summary(sample: dict[str, Any], role: str) -> dict[str, Any]:
    return {
        "candidate_id": role,
        "candidate_role": role,
        "source_frame_number": int(sample["source_frame_number"]),
        "source_timestamp_seconds": rounded(float(sample["source_timestamp_seconds"]), 3),
        "selection_score": rounded(float(sample["selection_score"])),
        "selection_features": sample["selection_features"],
        "selection_status": "automatically_ranked_visual_candidate_not_topology_judgment",
    }


def local_validate_packet(packet: dict[str, Any]) -> dict[str, Any]:
    """Check the fields and image budgets consumed by preflight_qwen_request.py."""
    errors: list[dict[str, Any]] = []
    selection_errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    candidates = packet.get("selected_candidates")
    model_media = packet.get("model_media")
    if not isinstance(candidates, list) or not candidates:
        errors.append({"code": "selected_candidates_missing"})
        candidates = []
    if not isinstance(model_media, list) or not model_media:
        errors.append({"code": "model_media_missing"})
        model_media = []
    if len(model_media) > MAX_PACKET_IMAGES:
        errors.append({"code": "too_many_model_media", "count": len(model_media), "maximum": MAX_PACKET_IMAGES})

    references: list[str] = []
    for candidate in candidates:
        frame = candidate.get("frame") if isinstance(candidate, dict) else None
        frame_path = frame.get("path") if isinstance(frame, dict) else None
        if not isinstance(frame_path, str) or not frame_path:
            errors.append({"code": "candidate_frame_path_missing", "candidate_id": candidate.get("candidate_id") if isinstance(candidate, dict) else None})
            continue
        references.append(frame_path)
        rois = candidate.get("rois", []) if isinstance(candidate, dict) else []
        if not isinstance(rois, list):
            errors.append({"code": "candidate_rois_invalid", "candidate_id": candidate.get("candidate_id")})
            continue
        for roi in rois:
            if not isinstance(roi, dict) or not isinstance(roi.get("crop_path"), str):
                errors.append({"code": "roi_crop_path_missing", "candidate_id": candidate.get("candidate_id")})
                continue
            references.append(roi["crop_path"])
            ratio = roi.get("source_area_ratio")
            if not finite(ratio) or float(ratio) > MAX_SOURCE_ROI_AREA_RATIO:
                errors.append(
                    {
                        "code": "roi_source_area_invalid",
                        "candidate_id": candidate.get("candidate_id"),
                        "source_area_ratio": ratio,
                    }
                )

    media_paths = [item.get("path") for item in model_media if isinstance(item, dict) and isinstance(item.get("path"), str)]
    if set(media_paths) != set(references):
        errors.append(
            {
                "code": "model_media_selected_candidate_path_mismatch",
                "model_media_paths": len(set(media_paths)),
                "selected_candidate_paths": len(set(references)),
            }
        )

    inspected: list[dict[str, Any]] = []
    total_bytes = 0
    for raw_path in sorted(set(references)):
        path = Path(raw_path)
        entry: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
        if not path.is_file():
            errors.append({"code": "media_missing", "path": str(path)})
            inspected.append(entry)
            continue
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            errors.append({"code": "media_decode_failed", "path": str(path)})
            inspected.append(entry)
            continue
        byte_count = int(path.stat().st_size)
        height, width = image.shape[:2]
        entry.update({"decoded": True, "file_size_bytes": byte_count, "width": int(width), "height": int(height)})
        inspected.append(entry)
        total_bytes += byte_count
        if byte_count > MAX_SINGLE_IMAGE_BYTES:
            errors.append({"code": "single_media_too_large", "path": str(path), "file_size_bytes": byte_count})
        if max(width, height) > DETAIL_EDGE:
            errors.append({"code": "media_edge_too_large", "path": str(path), "edge": max(width, height)})
    estimated_base64_bytes = sum(((item.get("file_size_bytes", 0) + 2) // 3) * 4 for item in inspected)
    if total_bytes > MAX_TOTAL_IMAGE_BYTES:
        errors.append({"code": "total_media_payload_too_large", "bytes": total_bytes})
    if estimated_base64_bytes > MAX_ESTIMATED_BASE64_BYTES:
        errors.append({"code": "estimated_base64_payload_too_large", "bytes": estimated_base64_bytes})
    if len(set(references)) != len(references):
        warnings.append({"code": "duplicate_media_reference"})

    required_false_flags = (
        "qwen_called",
        "excel_accessed",
        "labels_accessed",
        "historical_predictions_accessed",
        "score_computed",
    )
    for key in required_false_flags:
        if packet.get(key) is not False:
            errors.append({"code": "scope_flag_invalid", "flag": key, "value": packet.get(key)})
    timing = packet.get("event_timing")
    stable_window = timing.get("stable_successor_window_seconds") if isinstance(timing, dict) else None
    wiring = timing.get("wiring_segment") if isinstance(timing, dict) else None
    successor = timing.get("successor_segment") if isinstance(timing, dict) else None
    if (
        not isinstance(stable_window, list)
        or len(stable_window) != 2
        or not finite(stable_window[0])
        or not finite(stable_window[1])
        or float(stable_window[1]) <= float(stable_window[0])
        or not isinstance(wiring, dict)
        or not isinstance(successor, dict)
        or not stage_is_recording(successor.get("stage"))
        or not finite(wiring.get("end_seconds"))
        or not finite(successor.get("end_seconds"))
    ):
        selection_errors.append({"code": "stable_successor_window_invalid"})
    else:
        stable_start, stable_end = float(stable_window[0]), float(stable_window[1])
        if stable_start < float(wiring["end_seconds"]) + POST_WIRING_START_GUARD_SECONDS - 1e-6:
            selection_errors.append({"code": "stable_window_starts_before_post_wiring_guard"})
        if stable_end > float(successor["end_seconds"]) - SUCCESSOR_END_GUARD_SECONDS + 1e-6:
            selection_errors.append({"code": "stable_window_ends_after_successor_guard"})
        if stable_end - stable_start < MIN_STABLE_SUCCESSOR_SECONDS - 1e-6:
            selection_errors.append({"code": "stable_window_too_short"})
        for candidate in candidates:
            timestamp = candidate.get("source_timestamp_seconds") if isinstance(candidate, dict) else None
            if not finite(timestamp) or not stable_start - 1e-6 <= float(timestamp) <= stable_end + 1e-6:
                selection_errors.append(
                    {"code": "candidate_outside_stable_successor_window", "candidate_id": candidate.get("candidate_id") if isinstance(candidate, dict) else None}
                )
        for item in model_media:
            timestamp = item.get("source_timestamp_seconds") if isinstance(item, dict) else None
            if not finite(timestamp) or not stable_start - 1e-6 <= float(timestamp) <= stable_end + 1e-6:
                selection_errors.append({"code": "model_media_outside_stable_successor_window"})
    return {
        "valid": not errors and not selection_errors,
        "transport_valid": not errors,
        "evidence_selection_valid": not selection_errors,
        "errors": [*errors, *selection_errors],
        "warnings": warnings,
        "image_count": len(inspected),
        "total_jpeg_bytes": total_bytes,
        "estimated_base64_bytes": estimated_base64_bytes,
        "checked_against": "preflight_qwen_request.py media contract plus post-wiring stable-window selection contract",
    }


def build_packet(
    event: dict[str, Any],
    source_video: Path,
    source_meta: dict[str, Any],
    source_video_id: str,
    source_discovery: str,
    samples: list[dict[str, Any]],
    output_root: Path,
    sampling_fps: float,
    context_frame_count: int,
    context_edge: int,
) -> dict[str, Any]:
    assign_selection_scores(samples, float(event["wiring_segment"]["end_seconds"]))
    context_samples = choose_context_samples(samples, context_frame_count)

    event_root = output_root / safe_slug(source_video_id) / "events" / safe_slug(event["event_id"])
    media_root = event_root / "model_media"
    event_id = str(event["event_id"])

    context_entries: list[dict[str, Any]] = []
    for index, context in enumerate(context_samples, start=1):
        image, frame_number = decode_frame_at(
            source_video,
            float(context["source_timestamp_seconds"]),
            float(source_meta["fps"]),
        )
        if index == 1:
            candidate_role = "primary"
            media_role = "context_primary"
            filename = f"{index:02d}_context_primary.jpg"
        elif context_frame_count == 2 and index == 2:
            candidate_role = "backup"
            media_role = "context_backup"
            filename = f"{index:02d}_context_backup.jpg"
        else:
            candidate_role = f"context_{index:02d}"
            media_role = "context_supporting"
            filename = f"{index:02d}_context_supporting.jpg"
        frame = write_jpeg(image, media_root / filename, context_edge, PANORAMA_QUALITY)
        frame.update(
            {
                "media_role": media_role,
                "event_id": event_id,
                "source_frame_number": frame_number,
                "source_timestamp_seconds": rounded(float(context["source_timestamp_seconds"]), 3),
                "source_original_width": int(image.shape[1]),
                "source_original_height": int(image.shape[0]),
                "derived_from_original_video": True,
            }
        )
        context_entries.append(
            {
                "sample": context,
                "image": image,
                "frame_number": frame_number,
                "frame": frame,
                "candidate_role": candidate_role,
            }
        )

    primary_entry = context_entries[0]
    primary_image = primary_entry["image"]
    primary_frame_number = int(primary_entry["frame_number"])
    primary_frame = primary_entry["frame"]

    rois: list[dict[str, Any]] = []
    detail_media: list[dict[str, Any]] = []
    source_height, source_width = primary_image.shape[:2]
    for index, (role, bbox) in enumerate(
        unverified_region_specs(source_width, source_height),
        start=context_frame_count + 1,
    ):
        crop = primary_image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        if crop.size == 0:
            raise RuntimeError(f"Empty generated ROI {role} for {event_id}")
        media = write_jpeg(crop, media_root / f"{index:02d}_{role}.jpg", DETAIL_EDGE, DETAIL_QUALITY)
        media.update(
            {
                "media_role": "unverified_visual_region",
                "detail_role": role,
                "event_id": event_id,
                "source_frame_number": primary_frame_number,
                "source_timestamp_seconds": rounded(float(primary["source_timestamp_seconds"]), 3),
                "source_bbox_xyxy": bbox,
                "source_frame_width": source_width,
                "source_frame_height": source_height,
                "source_area_ratio": rounded(bbox_area_ratio(bbox, source_width, source_height)),
                "semantic_identity": "unverified",
                "derived_from_original_video": True,
            }
        )
        detail_media.append(media)
        rois.append(
            {
                "role": role,
                "semantic_identity": "unverified_visual_region",
                "status": "unverified",
                "crop_path": media["path"],
                "enhanced_or_rectified_path": None,
                "bbox_xyxy": bbox,
                "source_frame_width": source_width,
                "source_frame_height": source_height,
                "source_area_ratio": media["source_area_ratio"],
                "region_selection_policy": "fixed_overlapping_geometric_coverage_no_component_identity_claim",
            }
        )

    selected_candidates: list[dict[str, Any]] = []
    for index, context_entry in enumerate(context_entries):
        candidate = sample_summary(context_entry["sample"], str(context_entry["candidate_role"]))
        candidate["source_frame_number"] = int(context_entry["frame_number"])
        candidate["frame"] = context_entry["frame"]
        candidate["rois"] = rois if index == 0 else []
        selected_candidates.append(candidate)
    media = [entry["frame"] for entry in context_entries] + detail_media

    packet: dict[str, Any] = {
        "schema_version": "1.0",
        "artifact_type": "voltmeter_parallel_action_guided_event_packet_v1",
        "experiment_type": "resistance_voltmeter_parallel_connection_action_guided_v1",
        "rubric_id": RUBRIC_ID,
        "rubric_name": RUBRIC_NAME,
        "evaluation_scope": "recording_stage_only",
        "packet_status": "ready_for_preflight",
        "source_action_summary_path": str(ACTION_SUMMARY_PATH.resolve()),
        "source_video": source_meta,
        "source_video_id": source_video_id,
        "source_video_discovery": source_discovery,
        "event_id": event_id,
        "event_timing": {
            "experiment_window_seconds": [rounded(item) for item in event["experiment_window_seconds"]],
            "wiring_segment": event["wiring_segment"],
            "successor_segment": event["successor_segment"],
            "retrieval_window_seconds": [rounded(item) for item in event["retrieval_window_seconds"]],
            "stable_successor_window_seconds": [rounded(item) for item in event["stable_successor_window_seconds"]],
            "event_window_seconds": [rounded(item) for item in event["event_window_seconds"]],
        },
        "event_window_seconds": [rounded(item) for item in event["event_window_seconds"]],
        "sampling": {
            "source": "direct_source_resistance_mp4_decode",
            "sampling_fps": sampling_fps,
            "requested_timestamp_count": len(sample_timestamps(*event["event_window_seconds"], sampling_fps)),
            "decoded_timestamp_count": len(samples),
            "context_frame_count": context_frame_count,
            "context_max_edge": context_edge,
            "selection_window_is_post_wiring_stable_hard_gate": True,
            "post_wiring_start_guard_seconds": POST_WIRING_START_GUARD_SECONDS,
            "successor_end_guard_seconds": SUCCESSOR_END_GUARD_SECONDS,
            "selection_policy": {
                "sharpness": 0.38,
                "stability": 0.23,
                "low_motion": 0.20,
                "low_occlusion_proxy": 0.19,
            },
        },
        "selected_candidates": selected_candidates,
        "model_media": [{"display_order": index, **item} for index, item in enumerate(media, start=1)],
        "visual_region_policy": {
            "detail_crop_count": len(rois),
            "detail_crop_identity": "unverified_visual_regions",
            "coverage_strategy": "four_overlapping_wide_geometric_tiles_from_post_wiring_primary_context",
            "automatic_component_identity_used": False,
            "automatic_topology_decision_used": False,
        },
        "evidence_contract": {
            "action_stages_are_retrieval_priors_only": True,
            "requires_same_stable_event_for_topology_assessment": True,
            "all_model_media_lies_within_post_wiring_stable_successor_window": True,
            "requires_direct_evidence_of_two_voltmeter_lead_endpoints": True,
            "requires_direct_evidence_of_two_target_resistor_terminals": True,
            "unverified_visual_regions_are_not_component_labels": True,
        },
        "scope_flags": {
            "source_action_summary_only": True,
            "source_resistance_mp4_only": True,
            "old_predictions_read": False,
            "excel_read": False,
            "labels_read": False,
            "qwen_called": False,
            "topology_score_computed": False,
        },
        "source_read_only": True,
        "qwen_called": False,
        "excel_accessed": False,
        "labels_accessed": False,
        "historical_predictions_accessed": False,
        "score_computed": False,
    }
    packet["local_validation"] = local_validate_packet(packet)
    write_json(event_root / "event_packet_manifest.json", packet)
    write_json(event_root / "local_validation.json", packet["local_validation"])
    return packet


def build_video(
    record: dict[str, Any],
    source_videos: list[Path],
    output_root: Path,
    sampling_fps: float,
    context_frame_count: int,
    context_edge: int,
) -> dict[str, Any]:
    source_video_id = str(record["source_video_id"])
    source_video, source_discovery = find_source_video(source_video_id, source_videos)
    source_meta = video_metadata(source_video)
    experiment_window = record.get("fixed_experiment_window_seconds")
    if (
        not isinstance(experiment_window, list)
        or len(experiment_window) != 2
        or not finite(experiment_window[1])
        or not finite(source_meta.get("duration_seconds"))
        or float(experiment_window[1]) > float(source_meta["duration_seconds"]) + 1e-3
    ):
        raise ValueError(f"{source_video_id}: experiment window exceeds source video duration")
    events, ignored_wiring = action_events(record)
    built_events: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for event in events:
        timestamps = sample_timestamps(*event["event_window_seconds"], sampling_fps)
        try:
            samples = decode_samples(source_video, timestamps, float(source_meta["fps"]))
            packet = build_packet(
                event,
                source_video,
                source_meta,
                source_video_id,
                source_discovery,
                samples,
                output_root,
                sampling_fps,
                context_frame_count,
                context_edge,
            )
            event_root = output_root / safe_slug(source_video_id) / "events" / safe_slug(event["event_id"])
            built_events.append(
                {
                    "event_id": event["event_id"],
                    "event_window_seconds": packet["event_window_seconds"],
                    "packet_path": str((event_root / "event_packet_manifest.json").resolve()),
                    "local_validation_path": str((event_root / "local_validation.json").resolve()),
                    "local_validation_valid": packet["local_validation"]["valid"],
                    "transport_valid": packet["local_validation"]["transport_valid"],
                    "evidence_selection_valid": packet["local_validation"]["evidence_selection_valid"],
                    "model_media_count": len(packet["model_media"]),
                }
            )
        except Exception as exc:  # Keep the per-event failure observable in the final report.
            errors.append(
                {
                    "event_id": event["event_id"],
                    "event_window_seconds": event["event_window_seconds"],
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
    video_root = output_root / safe_slug(source_video_id)
    proposal = {
        "schema_version": "1.0",
        "artifact_type": "voltmeter_parallel_action_guided_event_proposals_v1",
        "source_action_summary_path": str(ACTION_SUMMARY_PATH.resolve()),
        "source_video_id": source_video_id,
        "source_video": source_meta,
        "source_video_discovery": source_discovery,
        "candidate_event_count": len(events),
        "built_event_count": len(built_events),
        "ignored_wiring_segments": ignored_wiring,
        "events": built_events,
        "errors": errors,
        "qwen_called": False,
        "excel_accessed": False,
        "labels_accessed": False,
        "historical_predictions_accessed": False,
        "score_computed": False,
    }
    write_json(video_root / "event_proposals.json", proposal)
    return proposal


def load_records(summary_path: Path) -> list[dict[str, Any]]:
    value = read_json(summary_path)
    if not isinstance(value, dict) or not isinstance(value.get("records"), list):
        raise ValueError("Action summary must be an object with a records array")
    records = value["records"]
    if not records:
        raise ValueError("Action summary has no records")
    if not all(isinstance(record, dict) for record in records):
        raise ValueError("Action summary records must all be objects")
    video_ids = [record.get("source_video_id") for record in records]
    if not all(isinstance(video_id, str) and video_id for video_id in video_ids):
        raise ValueError("Action summary records must have non-empty unique source_video_id values")
    if len(set(video_ids)) != len(video_ids):
        raise ValueError("Action summary has duplicate source_video_id values")
    return records


def main() -> int:
    global ACTION_SUMMARY_PATH
    parser = argparse.ArgumentParser(
        description="Build action-guided, label-blind voltmeter parallel-connection evidence packets."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--summary",
        type=Path,
        default=ACTION_SUMMARY_PATH,
        help="Merged seven-stage action summary JSON.",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="Directory containing source MP4 files.",
    )
    parser.add_argument(
        "--video",
        action="append",
        dest="video_ids",
        help="Build only this source video ID; repeatable for a bounded batch.",
    )
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=SAMPLE_FPS,
        help=f"Stable-window frame sampling rate (0 < value <= {MAX_SAMPLE_FPS:g}); default: {SAMPLE_FPS:g}.",
    )
    parser.add_argument(
        "--context-frame-count",
        type=int,
        default=MIN_CONTEXT_FRAME_COUNT,
        help=(
            "Number of overview frames retained before four fixed detail tiles "
            f"({MIN_CONTEXT_FRAME_COUNT}-{MAX_CONTEXT_FRAME_COUNT}); default: {MIN_CONTEXT_FRAME_COUNT}."
        ),
    )
    parser.add_argument(
        "--context-edge",
        type=int,
        default=PANORAMA_EDGE,
        help=f"Maximum overview image edge ({PANORAMA_EDGE}-{DETAIL_EDGE}); default: {PANORAMA_EDGE}.",
    )
    args = parser.parse_args()

    if not finite(args.sample_fps) or not 0 < float(args.sample_fps) <= MAX_SAMPLE_FPS:
        parser.error(f"--sample-fps must be finite and in (0, {MAX_SAMPLE_FPS:g}]")
    if not MIN_CONTEXT_FRAME_COUNT <= args.context_frame_count <= MAX_CONTEXT_FRAME_COUNT:
        parser.error(
            f"--context-frame-count must be in [{MIN_CONTEXT_FRAME_COUNT}, {MAX_CONTEXT_FRAME_COUNT}]"
        )
    if not PANORAMA_EDGE <= args.context_edge <= DETAIL_EDGE:
        parser.error(f"--context-edge must be in [{PANORAMA_EDGE}, {DETAIL_EDGE}]")

    output_root = args.output_dir.expanduser().resolve()
    if output_root.exists():
        raise SystemExit(f"Refusing to overwrite existing output directory: {output_root}")
    summary_path = args.summary.expanduser().resolve()
    source_dir = args.source_dir.expanduser().resolve()
    if not summary_path.is_file():
        raise SystemExit(f"Required action summary is missing: {summary_path}")
    if not source_dir.is_dir():
        raise SystemExit(f"Source video directory is missing: {source_dir}")
    ACTION_SUMMARY_PATH = summary_path
    source_videos = source_video_candidates(source_dir)
    if not source_videos:
        raise SystemExit(f"No supported top-level video files found under {source_dir}")

    records = load_records(summary_path)
    requested_video_ids = set(args.video_ids or [])
    known_video_ids = {str(record["source_video_id"]) for record in records}
    unknown_video_ids = sorted(requested_video_ids - known_video_ids)
    if unknown_video_ids:
        parser.error(f"Unknown --video source_video_id values: {unknown_video_ids}")
    if requested_video_ids:
        records = [record for record in records if str(record["source_video_id"]) in requested_video_ids]
    # Validate all source mappings and action pairs before writing any derived output.
    preflight_plan: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for record in records:
        source_video_id = record.get("source_video_id")
        if not isinstance(source_video_id, str):
            raise SystemExit("Action summary record has no source_video_id")
        find_source_video(source_video_id, source_videos)
        events, _ = action_events(record)
        preflight_plan.append((record, events))
    if not any(events for _, events in preflight_plan):
        raise SystemExit("No wiring-to-recording action pairs found in the action summary")

    output_root.mkdir(parents=True, exist_ok=False)
    video_reports = [
        build_video(
            record,
            source_videos,
            output_root,
            float(args.sample_fps),
            args.context_frame_count,
            args.context_edge,
        )
        for record, _ in preflight_plan
    ]
    packet_count = sum(int(item["built_event_count"]) for item in video_reports)
    candidate_event_count = sum(int(item["candidate_event_count"]) for item in video_reports)
    error_count = sum(len(item["errors"]) for item in video_reports)
    invalid_validation_count = sum(
        1
        for item in video_reports
        for event in item["events"]
        if event.get("local_validation_valid") is not True
    )
    selection_validation_failure_count = sum(
        1
        for item in video_reports
        for event in item["events"]
        if event.get("evidence_selection_valid") is not True
    )
    report = {
        "schema_version": "1.0",
        "artifact_type": "voltmeter_parallel_action_guided_build_report_v1",
        "generated_at": utc_now(),
        "rubric_id": RUBRIC_ID,
        "rubric_name": RUBRIC_NAME,
        "evaluation_scope": "recording_stage_only",
        "source_action_summary_path": str(ACTION_SUMMARY_PATH.resolve()),
        "source_resistance_videos": [str(path) for path in source_videos],
        "selected_source_video_ids": [str(record["source_video_id"]) for record in records],
        "evidence_acquisition": {
            "sampling_fps": float(args.sample_fps),
            "context_frame_count": args.context_frame_count,
            "context_max_edge": args.context_edge,
            "detail_crop_count": 4,
        },
        "video_count": len(video_reports),
        "candidate_event_count": candidate_event_count,
        "packet_count": packet_count,
        "event_error_count": error_count,
        "invalid_local_validation_count": invalid_validation_count,
        "selection_validation_failure_count": selection_validation_failure_count,
        "videos": video_reports,
        "scope_flags": {
            "action_summary_only": True,
            "source_resistance_mp4_only": True,
            "old_predictions_read": False,
            "excel_read": False,
            "labels_read": False,
            "qwen_called": False,
            "topology_score_computed": False,
        },
        "transport_build_valid": error_count == 0 and invalid_validation_count == 0,
        "evidence_selection_build_valid": error_count == 0 and selection_validation_failure_count == 0,
        "build_valid": (
            error_count == 0
            and invalid_validation_count == 0
            and selection_validation_failure_count == 0
            and packet_count == candidate_event_count
        ),
    }
    write_json(output_root / "build_run_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["build_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
