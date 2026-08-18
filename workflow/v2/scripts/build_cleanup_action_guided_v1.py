#!/usr/bin/env python3
"""Build local, action-guided evidence packets for cleanup-action observation.

The seven-stage action segmentation is used only to retrieve a bounded part of
the source video. If no ``material_cleanup`` segment exists, the builder instead
creates a complete, chunked scan of the terminal segment. It does not decide
whether the switch was open, whether the circuit was disconnected, whether
cleanup was completed, or whether the rubric is satisfied. It decodes ordered
original-video frames and writes packets for a later visual judge. No model,
workbook, labels, or historical predictions are read.
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
DEFAULT_ACTION_SUMMARY = (
    ROOT
    / "outputs"
    / "action_minutes"
    / "action_segments_summary.json"
)
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "cleanup_action_guided_v1"

RUBRIC_ID = "resistance.cleanup_and_return"
RUBRIC_NAME = "拆除整理动作观察"
RUBRIC_REQUIREMENT = "从 material_cleanup 候选段提取拆除或收拢动作的可视证据；若完全没有该候选段，则完整扫描末段以检索整理动作。本构建步骤不作合规、结果或评分判断。"
CLEANUP_STAGE = "material_cleanup"
MATERIAL_CLEANUP_CANDIDATE_KIND = "material_cleanup_candidate"
TERMINAL_SEGMENT_FALLBACK_KIND = "terminal_segment_fallback"
TERMINAL_FALLBACK_RETRIEVAL_REASON = "no_material_cleanup_segment"
TAIL_SCAN_ROLE = "tail_scan"
LEGACY_V1_EVIDENCE_ROLES = {"initial_reference", "before", "during", "after", "return"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

DEFAULT_BEFORE_SECONDS = 8.0
DEFAULT_MAX_IMAGES = 8
DEFAULT_TAIL_SCAN_INTERVAL_SECONDS = 1.0
DEFAULT_TAIL_SCAN_MAX_IMAGES = 12
TAIL_SCAN_MAX_SAMPLE_GAP_SECONDS = 1.25
PANORAMA_EDGE = 1280
JPEG_QUALITY = 86
MIN_JPEG_QUALITY = 50
MAX_SINGLE_IMAGE_BYTES = 2 * 1024 * 1024
MAX_TOTAL_IMAGE_BYTES = 10 * 1024 * 1024
MAX_ESTIMATED_BASE64_BYTES = 14 * 1024 * 1024
FRAME_EPSILON = 0.5
TERMINAL_SEGMENT_ALIGNMENT_TOLERANCE_SECONDS = 0.05


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


def rounded(value: Any, places: int = 6) -> float:
    return round(float(value), places)


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
    if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(f"Source video metadata is incomplete: {video_path}")
    duration = max(0.0, (frame_count - 1) / fps)
    stat = video_path.stat()
    return {
        "path": str(video_path.resolve()),
        "fps": rounded(fps),
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_seconds": rounded(duration),
        "fingerprint": {
            "size_bytes": int(stat.st_size),
            "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).replace(microsecond=0).isoformat(),
        },
        "direct_original_video_decode": True,
    }


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
    result: dict[str, Any] = {
        "segment_index": index,
        "stage": stage,
        "start_seconds": rounded(start, 3),
        "end_seconds": rounded(end, 3),
    }
    for key in ("stage_label", "start_source", "end_source", "start_evidence", "evidence"):
        if isinstance(raw.get(key), str):
            result[key] = raw[key]
    return result


def normalized_segments(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize both minute-merge ``segments`` and action-segment ``actions``."""
    raw_segments = record.get("segments")
    if raw_segments is None:
        raw_segments = record.get("actions")
    if not isinstance(raw_segments, list):
        raise ValueError(f"{record.get('source_video_id', '<unknown>')}: segments/actions is missing")
    segments = [clean_segment(item, index) for index, item in enumerate(raw_segments)]
    previous_end: float | None = None
    for segment in segments:
        start = float(segment["start_seconds"])
        if previous_end is not None and start < previous_end - 1e-6:
            raise ValueError(f"{record.get('source_video_id', '<unknown>')}: action segments overlap")
        previous_end = float(segment["end_seconds"])
    return segments


def cleanup_candidates(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Return every material-cleanup action candidate, including mid-experiment ones."""
    return [segment for segment in normalized_segments(record) if segment["stage"] == CLEANUP_STAGE]


def terminal_segment_for_fallback(record: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    """Return the last segment only when it actually reaches the experiment end."""
    segments = normalized_segments(record)
    window = record.get("fixed_experiment_window_seconds")
    if not segments:
        return None, "terminal_segment_missing"
    if not isinstance(window, list) or len(window) != 2 or not all(finite(item) for item in window):
        return None, "fixed_experiment_window_invalid"
    terminal = segments[-1]
    start = float(terminal["start_seconds"])
    end = float(terminal["end_seconds"])
    window_start, window_end = float(window[0]), float(window[1])
    if end <= start + 1e-6:
        return None, "terminal_segment_empty"
    if start < window_start - TERMINAL_SEGMENT_ALIGNMENT_TOLERANCE_SECONDS or end > window_end + TERMINAL_SEGMENT_ALIGNMENT_TOLERANCE_SECONDS:
        return None, "terminal_segment_outside_fixed_experiment_window"
    if abs(end - window_end) > TERMINAL_SEGMENT_ALIGNMENT_TOLERANCE_SECONDS:
        return None, "terminal_segment_does_not_reach_fixed_experiment_end"
    return terminal, None


def plan_terminal_scan_samples(
    start_seconds: float,
    end_seconds: float,
    interval_seconds: float = DEFAULT_TAIL_SCAN_INTERVAL_SECONDS,
) -> list[dict[str, Any]]:
    """Sample every half-open interval in a terminal segment exactly once.

    Each request represents one ``[start + i * interval, min(...))`` bucket.
    The final partial bucket is retained, so the scan covers the complete
    terminal segment without asking for a frame after the experiment window.
    """
    if not all(finite(value) for value in (start_seconds, end_seconds, interval_seconds)):
        raise ValueError("terminal scan timing values must be finite")
    if end_seconds <= start_seconds or interval_seconds <= 0:
        raise ValueError("terminal scan interval is invalid")
    bucket_count = int(math.ceil((float(end_seconds) - float(start_seconds)) / float(interval_seconds) - 1e-9))
    samples: list[dict[str, Any]] = []
    for bucket_index in range(bucket_count):
        bucket_start = float(start_seconds) + bucket_index * float(interval_seconds)
        bucket_end = min(float(end_seconds), bucket_start + float(interval_seconds))
        if bucket_end <= bucket_start + 1e-9:
            continue
        samples.append(
            {
                "role": TAIL_SCAN_ROLE,
                "subrole": f"bucket_{bucket_index:03d}",
                "planned_timestamp_seconds": rounded((bucket_start + bucket_end) / 2.0, 6),
                "selection_basis": "terminal_segment_full_coverage_scan",
                "selection_status": "terminal_segment_retrieval_candidate_not_compliance_judgment",
                "required": True,
                "tail_scan_bucket_index": bucket_index,
                "tail_scan_bucket_start_seconds": rounded(bucket_start, 6),
                "tail_scan_bucket_end_seconds": rounded(bucket_end, 6),
            }
        )
    if not samples:
        raise ValueError("terminal scan has no sampling buckets")
    return samples


def chunk_terminal_scan_samples(samples: list[dict[str, Any]], max_images: int) -> list[list[dict[str, Any]]]:
    """Split a full scan into bounded chunks with one overlapping bucket."""
    if max_images < 2:
        raise ValueError("terminal scan max_images must be at least 2")
    if not samples:
        raise ValueError("terminal scan samples are missing")
    chunks: list[list[dict[str, Any]]] = []
    start = 0
    while start < len(samples):
        chunk = [dict(item) for item in samples[start : start + max_images]]
        if not chunk:
            break
        overlap = start > 0
        for item_index, item in enumerate(chunk):
            item["tail_scan_is_overlap"] = overlap and item_index == 0
        chunks.append(chunk)
        end = start + len(chunk)
        if end >= len(samples):
            break
        start = end - 1
    return chunks


def terminal_scan_coverage(
    requests: list[dict[str, Any]],
    decoded: list[dict[str, Any]],
) -> dict[str, Any]:
    """Verify one decoded frame per requested terminal time bucket."""
    expected: dict[int, dict[str, Any]] = {}
    errors: list[str] = []
    for request in requests:
        index = request.get("tail_scan_bucket_index")
        if not isinstance(index, int) or isinstance(index, bool) or index in expected:
            errors.append("tail_scan_planned_bucket_invalid_or_duplicate")
            continue
        expected[index] = request
    by_bucket: dict[int, list[dict[str, Any]]] = {}
    for item in decoded:
        index = item.get("tail_scan_bucket_index")
        if not isinstance(index, int) or isinstance(index, bool):
            errors.append("tail_scan_decoded_bucket_invalid")
            continue
        by_bucket.setdefault(index, []).append(item)
    missing = sorted(set(expected) - set(by_bucket))
    unexpected = sorted(set(by_bucket) - set(expected))
    duplicate = sorted(index for index, values in by_bucket.items() if len(values) != 1)
    if missing:
        errors.append("tail_scan_bucket_decode_missing")
    if unexpected:
        errors.append("tail_scan_bucket_decode_unexpected")
    if duplicate:
        errors.append("tail_scan_bucket_decode_duplicate")

    source_frames: list[int] = []
    source_timestamps: list[float] = []
    for index in sorted(set(expected) & set(by_bucket)):
        values = by_bucket[index]
        if len(values) != 1:
            continue
        item = values[0]
        start = item.get("tail_scan_bucket_start_seconds")
        end = item.get("tail_scan_bucket_end_seconds")
        timestamp = item.get("source_timestamp_seconds")
        frame_number = item.get("source_frame_number")
        if not all(finite(value) for value in (start, end, timestamp, frame_number)):
            errors.append("tail_scan_decoded_timing_invalid")
            continue
        # Timestamps are persisted to milliseconds, so allow only that storage
        # precision at a bucket edge, never a whole sampling interval.
        if float(timestamp) < float(start) - 0.0015 or float(timestamp) > float(end) + 0.0015:
            errors.append("tail_scan_decoded_frame_outside_bucket")
        source_frames.append(int(frame_number))
        source_timestamps.append(float(timestamp))
    if len(source_frames) != len(set(source_frames)):
        errors.append("tail_scan_source_frame_reused_within_chunk")
    if any(later <= earlier for earlier, later in zip(source_timestamps, source_timestamps[1:])):
        errors.append("tail_scan_decoded_timestamps_not_strictly_increasing")
    return {
        "coverage_complete": not errors,
        "coverage_errors": sorted(set(errors)),
        "planned_sample_count": len(requests),
        "decoded_sample_count": len(decoded),
        "missing_bucket_indices": missing,
        "unexpected_bucket_indices": unexpected,
        "duplicate_bucket_indices": duplicate,
    }


def clamp_time(value: float, lower: float, upper: float) -> float:
    return max(lower, min(float(value), upper))


def plan_role_times(
    experiment_start: float,
    experiment_end: float,
    cleanup_start: float,
    cleanup_end: float,
    fps: float,
    before_seconds: float = DEFAULT_BEFORE_SECONDS,
    max_images: int = DEFAULT_MAX_IMAGES,
) -> list[dict[str, Any]]:
    """Plan pre-action context plus a dense, bounded sample of the action."""
    if not all(finite(item) for item in (experiment_start, experiment_end, cleanup_start, cleanup_end, fps)):
        raise ValueError("cleanup timing values must be finite")
    if fps <= 0 or experiment_end <= experiment_start or cleanup_end < cleanup_start:
        raise ValueError("invalid experiment or cleanup timing")
    if max_images < 2:
        raise ValueError("max_images must be at least 2")
    if cleanup_start < experiment_start - 1e-6 or cleanup_end > experiment_end + 1e-6:
        raise ValueError("cleanup segment lies outside the fixed experiment window")
    frame_period = 1.0 / fps
    cleanup_duration = max(0.0, cleanup_end - cleanup_start)
    requests: list[dict[str, Any]] = []

    # Reserve at least one frame for the action itself. With a larger budget,
    # retain both broad context and a frame immediately before the segment.
    before_budget = min(2, max_images - 1)
    before_time = cleanup_start - min(max(before_seconds, frame_period), max(frame_period, cleanup_start - experiment_start))
    if before_budget >= 1 and before_time < cleanup_start - frame_period * 0.25:
        requests.append(
            {
                "role": "before",
                "subrole": "context",
                "planned_timestamp_seconds": rounded(before_time, 3),
                "selection_basis": "pre_cleanup_context",
                "required": True,
            }
        )

    # A broad cleanup segment can start a few seconds before the first actual
    # unplugging motion. Keep a near-boundary view as immediate pre-action
    # context when the packet budget permits it.
    near_before_time = max(experiment_start, cleanup_start - max(frame_period, min(1.5, before_seconds / 4.0)))
    if (
        before_budget >= 2
        and near_before_time < cleanup_start - frame_period * 0.25
        and all(abs(near_before_time - float(item["planned_timestamp_seconds"])) >= frame_period * 0.5 for item in requests)
    ):
        requests.append(
            {
                "role": "before",
                "subrole": "near_boundary",
                "planned_timestamp_seconds": rounded(near_before_time, 3),
                "selection_basis": "near_cleanup_boundary_context",
                "required": True,
            }
        )

    before_count = sum(1 for item in requests if item["role"] == "before")
    during_count = max(1, max_images - before_count)
    if cleanup_duration <= frame_period * 1.2:
        fractions = [0.5]
    else:
        # Uniform coverage catches short unplugging and grouping motions inside
        # a coarse stage interval without treating the interval as a verdict.
        fractions = [
            (index + 0.5) / during_count
            for index in range(during_count)
        ]
    subroles = ("onset", "early", "mid", "late", "tail", "final")
    for index, fraction in enumerate(fractions):
        requests.append(
            {
                "role": "during",
                "planned_timestamp_seconds": rounded(cleanup_start + cleanup_duration * fraction, 3),
                "subrole": subroles[index] if index < len(subroles) else f"sample_{index + 1:02d}",
                "selection_basis": "cleanup_interval_uniform_action_sample",
                "required": True,
            }
        )

    return requests


def decoded_frame_number(capture: cv2.VideoCapture, fallback: int) -> int:
    position = float(capture.get(cv2.CAP_PROP_POS_FRAMES))
    if math.isfinite(position) and position > 0:
        return max(0, int(round(position)) - 1)
    return fallback


def decode_requests(
    video_path: Path,
    requests: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open source video: {video_path}")
    fps = float(metadata["fps"])
    frame_count = int(metadata["frame_count"])
    decoded: list[dict[str, Any]] = []
    try:
        for request in requests:
            planned_time = clamp_time(float(request["planned_timestamp_seconds"]), 0.0, float(metadata["duration_seconds"]))
            target_frame = max(0, min(frame_count - 1, int(round(planned_time * fps))))
            capture.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ok, image = capture.read()
            if not ok or image is None or image.size == 0:
                continue
            frame_number = decoded_frame_number(capture, target_frame)
            frame_number = max(0, min(frame_count - 1, frame_number))
            decoded.append(
                {
                    **request,
                    "source_frame_number": int(frame_number),
                    "source_timestamp_seconds": rounded(frame_number / fps, 3),
                    "image": image,
                }
            )
    finally:
        capture.release()
    return decoded


def resize_edge(image: Any, maximum_edge: int) -> Any:
    height, width = image.shape[:2]
    edge = max(height, width)
    if edge <= maximum_edge:
        return image.copy()
    scale = maximum_edge / float(edge)
    return cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA)


def write_jpeg(image: Any, path: Path) -> dict[str, Any]:
    encoded_bytes: bytes | None = None
    prepared: Any | None = None
    selected_quality: int | None = None
    for edge_scale in (1.0, 0.90, 0.80, 0.70):
        candidate = resize_edge(image, max(640, round(PANORAMA_EDGE * edge_scale)))
        for quality in range(JPEG_QUALITY, MIN_JPEG_QUALITY - 1, -6):
            ok, encoded = cv2.imencode(".jpg", candidate, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if not ok:
                continue
            payload = encoded.tobytes()
            encoded_bytes, prepared, selected_quality = payload, candidate, quality
            if len(payload) <= MAX_SINGLE_IMAGE_BYTES:
                break
        if encoded_bytes is not None and len(encoded_bytes) <= MAX_SINGLE_IMAGE_BYTES:
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
        "jpeg_quality": int(selected_quality),
    }


def candidate_record(item: dict[str, Any], frame_meta: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    record = {
        "candidate_id": candidate_id,
        "evidence_role": item["role"],
        "subrole": item.get("subrole"),
        "planned_timestamp_seconds": rounded(item["planned_timestamp_seconds"], 3),
        "source_timestamp_seconds": rounded(item["source_timestamp_seconds"], 3),
        "source_frame_number": int(item["source_frame_number"]),
        "selection_basis": item["selection_basis"],
        "selection_status": item.get("selection_status", "action_stage_retrieval_candidate_not_compliance_judgment"),
        "frame": frame_meta,
    }
    for key in (
        "tail_scan_bucket_index",
        "tail_scan_bucket_start_seconds",
        "tail_scan_bucket_end_seconds",
        "tail_scan_is_overlap",
    ):
        if key in item:
            record[key] = item[key]
    return record


def build_media(decoded: list[dict[str, Any]], event_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    media_root = event_root / "model_media"
    candidates: list[dict[str, Any]] = []
    media: list[dict[str, Any]] = []
    for index, item in enumerate(decoded, start=1):
        role = safe_slug(item["role"])
        raw_subrole = item.get("subrole")
        subrole = safe_slug(raw_subrole) if isinstance(raw_subrole, str) and raw_subrole else ""
        suffix = f"_{subrole}" if subrole else ""
        path = media_root / f"{index:02d}_{role}{suffix}_frame_{int(item['source_frame_number']):08d}.jpg"
        frame_meta = write_jpeg(item["image"], path)
        candidate = candidate_record(item, frame_meta, f"candidate_{index:02d}_{role}")
        candidates.append(candidate)
        media.append(
            {
                "display_order": index,
                "media_role": item["role"],
                "detail_role": None,
                "path": frame_meta["path"],
                "sha256": frame_meta["sha256"],
                "file_size_bytes": frame_meta["file_size_bytes"],
                "source_frame_number": int(item["source_frame_number"]),
                "source_timestamp_seconds": rounded(item["source_timestamp_seconds"], 3),
                "candidate_id": candidate["candidate_id"],
                "evidence_role": item["role"],
                "subrole": item.get("subrole"),
                "derived_from_original_video": True,
            }
        )
    return candidates, media


def validate_packet(packet: dict[str, Any]) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    is_terminal_fallback = packet.get("candidate_kind") == TERMINAL_SEGMENT_FALLBACK_KIND
    is_legacy_v1 = packet.get("artifact_type") == "resistance_cleanup_action_guided_event_packet_v1"
    candidates = packet.get("selected_candidates")
    media = packet.get("model_media")
    if not isinstance(candidates, list) or not candidates:
        errors.append({"code": "selected_candidates_missing"})
        candidates = []
    if not isinstance(media, list) or not media:
        errors.append({"code": "model_media_missing"})
        media = []
    sampling = packet.get("sampling")
    default_max_images = DEFAULT_TAIL_SCAN_MAX_IMAGES if is_terminal_fallback else DEFAULT_MAX_IMAGES
    minimum_max_images = 1 if is_terminal_fallback else 2
    max_images = sampling.get("max_images") if isinstance(sampling, dict) else default_max_images
    if is_legacy_v1 and max_images is None:
        max_images = DEFAULT_MAX_IMAGES
    if not isinstance(max_images, int) or isinstance(max_images, bool) or max_images < minimum_max_images:
        errors.append({"code": "sampling_max_images_invalid", "value": max_images})
        max_images = default_max_images
    if len(media) > max_images:
        errors.append({"code": "too_many_model_media", "count": len(media), "maximum": max_images})

    source_meta = packet.get("source_video")
    fps = source_meta.get("fps") if isinstance(source_meta, dict) else None
    frame_count = source_meta.get("frame_count") if isinstance(source_meta, dict) else None
    duration = source_meta.get("duration_seconds") if isinstance(source_meta, dict) else None
    if not finite(fps) or not finite(frame_count) or not finite(duration):
        errors.append({"code": "source_metadata_invalid"})

    cleanup = packet.get("cleanup_segment")
    terminal = packet.get("terminal_segment")
    tail_scan = packet.get("tail_scan")
    cstart = cleanup.get("start_seconds") if isinstance(cleanup, dict) else None
    cend = cleanup.get("end_seconds") if isinstance(cleanup, dict) else None
    terminal_start = terminal.get("start_seconds") if isinstance(terminal, dict) else None
    terminal_end = terminal.get("end_seconds") if isinstance(terminal, dict) else None
    expected_roles = {TAIL_SCAN_ROLE} if is_terminal_fallback else {"before", "during"}
    allowed_roles = LEGACY_V1_EVIDENCE_ROLES if is_legacy_v1 else expected_roles
    tail_planned: dict[int, dict[str, Any]] = {}
    if is_terminal_fallback:
        if not isinstance(terminal, dict) or not finite(terminal_start) or not finite(terminal_end) or float(terminal_end) <= float(terminal_start):
            errors.append({"code": "terminal_segment_metadata_invalid"})
        if not isinstance(tail_scan, dict):
            errors.append({"code": "tail_scan_metadata_missing"})
            tail_scan = {}
        scan_id = tail_scan.get("scan_id")
        chunk_index = tail_scan.get("chunk_index")
        chunk_count = tail_scan.get("chunk_count")
        sample_interval = tail_scan.get("sample_interval_seconds")
        max_sample_gap = tail_scan.get("max_sample_gap_seconds")
        if not isinstance(scan_id, str) or not scan_id:
            errors.append({"code": "tail_scan_id_invalid"})
        if not isinstance(chunk_index, int) or isinstance(chunk_index, bool) or not isinstance(chunk_count, int) or isinstance(chunk_count, bool) or chunk_index < 1 or chunk_count < chunk_index:
            errors.append({"code": "tail_scan_chunk_metadata_invalid"})
        if (
            not finite(sample_interval)
            or float(sample_interval) <= 0
            or float(sample_interval) > TAIL_SCAN_MAX_SAMPLE_GAP_SECONDS + 1e-6
            or not finite(max_sample_gap)
            or float(max_sample_gap) < float(sample_interval)
            or float(max_sample_gap) > TAIL_SCAN_MAX_SAMPLE_GAP_SECONDS + 1e-6
        ):
            errors.append({"code": "tail_scan_sampling_metadata_invalid"})
        if tail_scan.get("required_full_coverage") is not True:
            errors.append({"code": "tail_scan_full_coverage_not_required"})
        planned_samples = tail_scan.get("planned_samples")
        if not isinstance(planned_samples, list) or not planned_samples:
            errors.append({"code": "tail_scan_planned_samples_missing"})
            planned_samples = []
        for item in planned_samples:
            if not isinstance(item, dict):
                errors.append({"code": "tail_scan_planned_sample_invalid"})
                continue
            bucket_index = item.get("bucket_index")
            bucket_start = item.get("bucket_start_seconds")
            bucket_end = item.get("bucket_end_seconds")
            planned_timestamp = item.get("planned_timestamp_seconds")
            if (
                not isinstance(bucket_index, int)
                or isinstance(bucket_index, bool)
                or bucket_index in tail_planned
                or not all(finite(value) for value in (bucket_start, bucket_end, planned_timestamp))
                or float(bucket_end) <= float(bucket_start)
                or not float(bucket_start) - 1e-6 <= float(planned_timestamp) <= float(bucket_end) + 1e-6
            ):
                errors.append({"code": "tail_scan_planned_sample_invalid"})
                continue
            tail_planned[bucket_index] = item
        planned_count = tail_scan.get("planned_sample_count")
        if not isinstance(planned_count, int) or isinstance(planned_count, bool) or planned_count != len(tail_planned):
            errors.append({"code": "tail_scan_planned_count_mismatch"})
        full_bucket_count = tail_scan.get("full_scan_bucket_count")
        if not isinstance(full_bucket_count, int) or isinstance(full_bucket_count, bool) or full_bucket_count < 1:
            errors.append({"code": "tail_scan_full_bucket_count_invalid"})
        if all(finite(value) for value in (terminal_start, terminal_end, sample_interval)) and float(sample_interval) > 0:
            try:
                expected_full_samples = plan_terminal_scan_samples(
                    float(terminal_start),
                    float(terminal_end),
                    float(sample_interval),
                )
            except ValueError:
                expected_full_samples = []
                errors.append({"code": "tail_scan_canonical_plan_invalid"})
            expected_by_bucket = {
                int(item["tail_scan_bucket_index"]): item
                for item in expected_full_samples
            }
            if isinstance(full_bucket_count, int) and full_bucket_count != len(expected_by_bucket):
                errors.append({"code": "tail_scan_full_bucket_count_mismatch"})
            for bucket_index, item in tail_planned.items():
                expected = expected_by_bucket.get(bucket_index)
                if expected is None or any(
                    abs(float(item[key]) - float(expected[expected_key])) > 0.0015
                    for key, expected_key in (
                        ("bucket_start_seconds", "tail_scan_bucket_start_seconds"),
                        ("bucket_end_seconds", "tail_scan_bucket_end_seconds"),
                        ("planned_timestamp_seconds", "planned_timestamp_seconds"),
                    )
                ):
                    errors.append({"code": "tail_scan_planned_sample_not_canonical", "bucket_index": bucket_index})
        window = packet.get("fixed_experiment_window_seconds")
        if (
            not isinstance(window, list)
            or len(window) != 2
            or not all(finite(value) for value in window)
            or not finite(terminal_start)
            or not finite(terminal_end)
            or float(terminal_start) < float(window[0]) - TERMINAL_SEGMENT_ALIGNMENT_TOLERANCE_SECONDS
            or abs(float(terminal_end) - float(window[1])) > TERMINAL_SEGMENT_ALIGNMENT_TOLERANCE_SECONDS
        ):
            errors.append({"code": "terminal_segment_not_aligned_to_fixed_experiment_end"})
    else:
        if not isinstance(cleanup, dict) or not finite(cstart) or not finite(cend):
            errors.append({"code": "cleanup_segment_metadata_invalid"})
    roles = {str(item.get("evidence_role")) for item in candidates if isinstance(item, dict)}
    missing_roles = sorted(expected_roles - roles)
    if missing_roles:
        errors.append({"code": "required_evidence_roles_missing", "roles": missing_roles})
    unexpected_roles = sorted(roles - allowed_roles)
    if unexpected_roles:
        errors.append({"code": "unexpected_evidence_roles", "roles": unexpected_roles})

    references: list[str] = []
    frame_numbers: list[int] = []
    role_timestamps: dict[str, list[float]] = {role: [] for role in expected_roles}
    tail_candidates: dict[int, dict[str, Any]] = {}
    total_bytes = 0
    for candidate in candidates:
        if not isinstance(candidate, dict):
            errors.append({"code": "candidate_not_object"})
            continue
        path_value = candidate.get("frame", {}).get("path") if isinstance(candidate.get("frame"), dict) else None
        if not isinstance(path_value, str) or not path_value:
            errors.append({"code": "candidate_frame_path_missing", "candidate_id": candidate.get("candidate_id")})
            continue
        path = Path(path_value)
        references.append(str(path))
        if not path.is_file():
            errors.append({"code": "media_missing", "path": str(path)})
            continue
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            errors.append({"code": "media_decode_failed", "path": str(path)})
        expected_hash = candidate["frame"].get("sha256")
        if not isinstance(expected_hash, str) or sha256_file(path) != expected_hash:
            errors.append({"code": "media_sha256_mismatch", "path": str(path)})
        byte_count = int(path.stat().st_size)
        total_bytes += byte_count
        if byte_count > MAX_SINGLE_IMAGE_BYTES:
            errors.append({"code": "single_media_too_large", "path": str(path), "file_size_bytes": byte_count})
        if finite(candidate.get("source_frame_number")):
            frame_numbers.append(int(candidate["source_frame_number"]))
        role = candidate.get("evidence_role")
        timestamp = candidate.get("source_timestamp_seconds")
        if not is_terminal_fallback and finite(timestamp) and finite(cstart) and finite(cend):
            value = float(timestamp)
            if role == "before" and value >= float(cstart) - 1e-6:
                errors.append({"code": "before_not_before_cleanup", "timestamp_seconds": value})
            if role == "during" and not float(cstart) - 1e-6 <= value <= float(cend) + 1e-6:
                errors.append({"code": "during_outside_cleanup_interval", "timestamp_seconds": value})
            if role in role_timestamps:
                role_timestamps[role].append(value)
        elif not is_terminal_fallback and role in {"before", "during"}:
            errors.append({"code": "candidate_timestamp_invalid", "role": role})
        if is_terminal_fallback and role == TAIL_SCAN_ROLE:
            bucket_index = candidate.get("tail_scan_bucket_index")
            bucket_start = candidate.get("tail_scan_bucket_start_seconds")
            bucket_end = candidate.get("tail_scan_bucket_end_seconds")
            planned_timestamp = candidate.get("planned_timestamp_seconds")
            if (
                not isinstance(bucket_index, int)
                or isinstance(bucket_index, bool)
                or bucket_index in tail_candidates
                or not all(finite(value) for value in (bucket_start, bucket_end, planned_timestamp, timestamp))
            ):
                errors.append({"code": "tail_scan_candidate_metadata_invalid", "candidate_id": candidate.get("candidate_id")})
                continue
            planned = tail_planned.get(bucket_index)
            if planned is None:
                errors.append({"code": "tail_scan_candidate_bucket_unplanned", "bucket_index": bucket_index})
            else:
                if (
                    abs(float(bucket_start) - float(planned["bucket_start_seconds"])) > 0.0015
                    or abs(float(bucket_end) - float(planned["bucket_end_seconds"])) > 0.0015
                    or abs(float(planned_timestamp) - float(planned["planned_timestamp_seconds"])) > 0.0015
                ):
                    errors.append({"code": "tail_scan_candidate_plan_mismatch", "bucket_index": bucket_index})
            if float(timestamp) < float(bucket_start) - 0.0015 or float(timestamp) > float(bucket_end) + 0.0015:
                errors.append({"code": "tail_scan_candidate_frame_outside_bucket", "bucket_index": bucket_index})
            tail_candidates[bucket_index] = candidate
            role_timestamps[TAIL_SCAN_ROLE].append(float(timestamp))
    if not is_terminal_fallback and role_timestamps["before"] and role_timestamps["during"] and max(role_timestamps["before"]) >= min(role_timestamps["during"]) - 1e-6:
        errors.append({"code": "before_during_order_invalid"})
    if len(set(references)) != len(references):
        warnings.append({"code": "duplicate_media_reference"})
    if len(set(frame_numbers)) != len(frame_numbers):
        target = errors if is_terminal_fallback else warnings
        target.append({"code": "duplicate_source_frame_reference"})
    if is_terminal_fallback:
        missing_buckets = sorted(set(tail_planned) - set(tail_candidates))
        unexpected_buckets = sorted(set(tail_candidates) - set(tail_planned))
        if missing_buckets:
            errors.append({"code": "tail_scan_candidate_bucket_missing", "bucket_indices": missing_buckets})
        if unexpected_buckets:
            errors.append({"code": "tail_scan_candidate_bucket_unexpected", "bucket_indices": unexpected_buckets})
        ordered_tail_timestamps = [
            float(tail_candidates[index]["source_timestamp_seconds"])
            for index in sorted(tail_candidates)
        ]
        if any(later <= earlier for earlier, later in zip(ordered_tail_timestamps, ordered_tail_timestamps[1:])):
            errors.append({"code": "tail_scan_candidate_timestamps_not_strictly_increasing"})
        if tail_scan.get("decoded_sample_count") != len(candidates):
            errors.append({"code": "tail_scan_decoded_count_mismatch"})
        if tail_scan.get("coverage_complete") is not True:
            errors.append({"code": "tail_scan_coverage_incomplete", "coverage_errors": tail_scan.get("coverage_errors")})
        declared_new = tail_scan.get("new_bucket_indices")
        declared_overlap = tail_scan.get("overlap_bucket_indices")
        if not isinstance(declared_new, list) or not all(isinstance(item, int) and not isinstance(item, bool) for item in declared_new):
            errors.append({"code": "tail_scan_new_bucket_indices_invalid"})
        if not isinstance(declared_overlap, list) or not all(isinstance(item, int) and not isinstance(item, bool) for item in declared_overlap):
            errors.append({"code": "tail_scan_overlap_bucket_indices_invalid"})
        if isinstance(declared_new, list) and isinstance(declared_overlap, list):
            expected_chunk_buckets = set(tail_planned)
            if set(declared_new) | set(declared_overlap) != expected_chunk_buckets or set(declared_new) & set(declared_overlap):
                errors.append({"code": "tail_scan_chunk_bucket_partition_invalid"})
    if total_bytes > MAX_TOTAL_IMAGE_BYTES:
        errors.append({"code": "total_media_payload_too_large", "bytes": total_bytes})
    estimated_base64 = sum(((int(Path(path).stat().st_size) + 2) // 3) * 4 for path in set(references) if Path(path).is_file())
    if estimated_base64 > MAX_ESTIMATED_BASE64_BYTES:
        errors.append({"code": "estimated_base64_payload_too_large", "bytes": estimated_base64})

    media_paths = [item.get("path") for item in media if isinstance(item, dict) and isinstance(item.get("path"), str)]
    if set(media_paths) != set(references):
        errors.append({"code": "model_media_candidate_path_mismatch"})
    media_roles = {str(item.get("evidence_role")) for item in media if isinstance(item, dict)}
    unexpected_media_roles = sorted(media_roles - allowed_roles)
    if unexpected_media_roles:
        errors.append({"code": "unexpected_model_media_roles", "roles": unexpected_media_roles})
    if any(packet.get(flag) is not False for flag in ("qwen_called", "excel_accessed", "labels_accessed", "historical_predictions_accessed", "score_computed")):
        errors.append({"code": "scope_flag_invalid"})
    evidence_selection_codes = {
        "required_evidence_roles_missing",
        "unexpected_evidence_roles",
        "unexpected_model_media_roles",
        "before_not_before_cleanup",
        "during_outside_cleanup_interval",
        "before_during_order_invalid",
        "tail_scan_candidate_metadata_invalid",
        "tail_scan_candidate_bucket_unplanned",
        "tail_scan_candidate_plan_mismatch",
        "tail_scan_candidate_frame_outside_bucket",
        "tail_scan_candidate_bucket_missing",
        "tail_scan_candidate_bucket_unexpected",
        "tail_scan_candidate_timestamps_not_strictly_increasing",
        "tail_scan_coverage_incomplete",
        "tail_scan_chunk_bucket_partition_invalid",
        "tail_scan_planned_count_mismatch",
        "tail_scan_full_bucket_count_invalid",
        "tail_scan_canonical_plan_invalid",
        "tail_scan_full_bucket_count_mismatch",
        "tail_scan_planned_sample_not_canonical",
        "terminal_segment_not_aligned_to_fixed_experiment_end",
    }
    return {
        "valid": not errors,
        "transport_valid": not any(item.get("code", "").startswith(("media_", "single_", "total_", "estimated_", "model_media", "too_many")) for item in errors),
        "evidence_selection_valid": not any(item.get("code", "") in evidence_selection_codes for item in errors),
        "errors": errors,
        "warnings": warnings,
        "image_count": len(references),
        "total_jpeg_bytes": total_bytes,
        "estimated_base64_bytes": estimated_base64,
        "checked_against": "cleanup action-guided packet contract v3",
    }


def build_packet(
    record: dict[str, Any],
    source_video: Path,
    source_discovery: str,
    source_meta: dict[str, Any],
    cleanup: dict[str, Any],
    summary_path: Path,
    output_root: Path,
    before_seconds: float,
    max_images: int,
    event_index: int,
) -> dict[str, Any]:
    window = record["fixed_experiment_window_seconds"]
    requests = plan_role_times(
        float(window[0]),
        float(window[1]),
        float(cleanup["start_seconds"]),
        float(cleanup["end_seconds"]),
        float(source_meta["fps"]),
        before_seconds,
        max_images,
    )
    decoded = decode_requests(source_video, requests, source_meta)
    # Keep role order and discard optional decoded frames if the packet budget
    # would otherwise be exceeded. Required roles are never silently dropped.
    if len(decoded) > max_images:
        required = [item for item in decoded if item.get("required")]
        optional = [item for item in decoded if not item.get("required")]
        decoded = required + optional[: max(0, max_images - len(required))]
        decoded.sort(key=lambda item: float(item["planned_timestamp_seconds"]))
    retrieval_window = [
        min((float(item["planned_timestamp_seconds"]) for item in decoded), default=float(cleanup["start_seconds"])),
        max((float(item["planned_timestamp_seconds"]) for item in decoded), default=float(cleanup["end_seconds"])),
    ]
    event_id = f"event_{event_index:02d}_material_cleanup"
    event_root = output_root / safe_slug(str(record["source_video_id"])) / "events" / event_id
    candidates, media = build_media(decoded, event_root)
    selected_roles = {str(item.get("evidence_role")) for item in candidates}
    packet_status = (
        "ready_for_preflight"
        if {"before", "during"}.issubset(selected_roles)
        else "evidence_insufficient"
    )
    packet: dict[str, Any] = {
        "schema_version": "1.0",
        "artifact_type": "resistance_cleanup_action_guided_event_packet_v3",
        "experiment_type": "resistance_cleanup_action_detection_v3",
        "rubric_id": RUBRIC_ID,
        "rubric_name": RUBRIC_NAME,
        "rubric_requirement": RUBRIC_REQUIREMENT,
        "packet_status": packet_status,
        "source_action_summary_path": str(summary_path.resolve()),
        "source_video": source_meta,
        "source_video_id": str(record["source_video_id"]),
        "source_video_discovery": source_discovery,
        "event_id": event_id,
        "candidate_kind": MATERIAL_CLEANUP_CANDIDATE_KIND,
        "retrieval_reason": "material_cleanup_segment",
        "fixed_experiment_window_seconds": [rounded(item, 3) for item in window],
        "event_window_seconds": [rounded(item, 3) for item in retrieval_window],
        "cleanup_interval_seconds": [rounded(cleanup["start_seconds"], 3), rounded(cleanup["end_seconds"], 3)],
        "cleanup_segment": cleanup,
        "cleanup_candidate_stage": CLEANUP_STAGE,
        "sampling": {
            "source": "direct_source_resistance_mp4_decode",
            "planned_role_count": len(requests),
            "decoded_role_count": len(decoded),
            "before_context_seconds": before_seconds,
            "max_images": max_images,
            "role_order": ["before", "during"],
            "action_stage_is_retrieval_prior_only": True,
        },
        "selected_candidates": candidates,
        "model_media": media,
        "evidence_contract": {
            "action_stage_is_retrieval_prior_only": True,
            "requires_before_then_during_order": True,
            "retrieval_observations": [
                "pre_cleanup_context_visible",
                "cleanup_hand_or_wire_removal_action_visible",
                "cleanup_equipment_or_lead_organizing_action_visible",
            ],
            "purpose": "action_candidate_retrieval_only",
        },
        "scope_flags": {
            "source_action_summary_only": True,
            "source_resistance_mp4_only": True,
            "old_predictions_read": False,
            "excel_read": False,
            "labels_read": False,
            "qwen_called": False,
            "score_computed": False,
        },
        "source_read_only": True,
        "qwen_called": False,
        "excel_accessed": False,
        "labels_accessed": False,
        "historical_predictions_accessed": False,
        "score_computed": False,
    }
    packet["local_validation"] = validate_packet(packet)
    write_json(event_root / "event_packet_manifest.json", packet)
    write_json(event_root / "local_validation.json", packet["local_validation"])
    return packet


def build_terminal_scan_packet(
    record: dict[str, Any],
    source_video: Path,
    source_discovery: str,
    source_meta: dict[str, Any],
    terminal_segment: dict[str, Any],
    requests: list[dict[str, Any]],
    full_bucket_count: int,
    scan_id: str,
    chunk_index: int,
    chunk_count: int,
    summary_path: Path,
    output_root: Path,
    tail_scan_interval_seconds: float,
    tail_scan_max_images: int,
) -> dict[str, Any]:
    """Build one bounded, auditable chunk of a complete terminal scan."""
    decoded = decode_requests(source_video, requests, source_meta)
    coverage = terminal_scan_coverage(requests, decoded)
    event_id = f"terminal_scan_{chunk_index:02d}_of_{chunk_count:02d}"
    event_root = output_root / safe_slug(str(record["source_video_id"])) / "events" / event_id
    candidates, media = build_media(decoded, event_root)
    planned_samples = [
        {
            "bucket_index": int(item["tail_scan_bucket_index"]),
            "bucket_start_seconds": rounded(item["tail_scan_bucket_start_seconds"], 6),
            "bucket_end_seconds": rounded(item["tail_scan_bucket_end_seconds"], 6),
            "planned_timestamp_seconds": rounded(item["planned_timestamp_seconds"], 6),
            "is_overlap": bool(item.get("tail_scan_is_overlap")),
        }
        for item in requests
    ]
    overlap_bucket_indices = [item["bucket_index"] for item in planned_samples if item["is_overlap"]]
    new_bucket_indices = [item["bucket_index"] for item in planned_samples if not item["is_overlap"]]
    scan_range = [
        min(float(item["bucket_start_seconds"]) for item in planned_samples),
        max(float(item["bucket_end_seconds"]) for item in planned_samples),
    ]
    new_samples = [item for item in planned_samples if not item["is_overlap"]]
    new_scan_range = [
        min(float(item["bucket_start_seconds"]) for item in new_samples),
        max(float(item["bucket_end_seconds"]) for item in new_samples),
    ]
    packet_status = "ready_for_preflight" if coverage["coverage_complete"] and len(decoded) == len(requests) else "evidence_insufficient"
    packet: dict[str, Any] = {
        "schema_version": "1.0",
        "artifact_type": "resistance_cleanup_action_guided_event_packet_v3",
        "experiment_type": "resistance_cleanup_action_detection_v3",
        "rubric_id": RUBRIC_ID,
        "rubric_name": RUBRIC_NAME,
        "rubric_requirement": RUBRIC_REQUIREMENT,
        "packet_status": packet_status,
        "source_action_summary_path": str(summary_path.resolve()),
        "source_video": source_meta,
        "source_video_id": str(record["source_video_id"]),
        "source_video_discovery": source_discovery,
        "event_id": event_id,
        "candidate_kind": TERMINAL_SEGMENT_FALLBACK_KIND,
        "retrieval_reason": TERMINAL_FALLBACK_RETRIEVAL_REASON,
        "fixed_experiment_window_seconds": [rounded(item, 3) for item in record["fixed_experiment_window_seconds"]],
        "event_window_seconds": [rounded(item, 6) for item in scan_range],
        "terminal_segment": terminal_segment,
        "tail_scan": {
            "scan_id": scan_id,
            "chunk_index": chunk_index,
            "chunk_count": chunk_count,
            "terminal_segment_seconds": [rounded(terminal_segment["start_seconds"], 6), rounded(terminal_segment["end_seconds"], 6)],
            "full_scan_bucket_count": full_bucket_count,
            "scan_range_seconds": [rounded(item, 6) for item in scan_range],
            "new_scan_range_seconds": [rounded(item, 6) for item in new_scan_range],
            "sample_interval_seconds": rounded(tail_scan_interval_seconds, 6),
            "max_sample_gap_seconds": TAIL_SCAN_MAX_SAMPLE_GAP_SECONDS,
            "required_full_coverage": True,
            "planned_sample_count": coverage["planned_sample_count"],
            "decoded_sample_count": coverage["decoded_sample_count"],
            "planned_samples": planned_samples,
            "new_bucket_indices": new_bucket_indices,
            "overlap_bucket_indices": overlap_bucket_indices,
            "coverage_complete": coverage["coverage_complete"],
            "coverage_errors": coverage["coverage_errors"],
            "missing_bucket_indices": coverage["missing_bucket_indices"],
            "unexpected_bucket_indices": coverage["unexpected_bucket_indices"],
            "duplicate_bucket_indices": coverage["duplicate_bucket_indices"],
        },
        "sampling": {
            "source": "direct_source_resistance_mp4_decode",
            "planned_role_count": len(requests),
            "decoded_role_count": len(decoded),
            "max_images": tail_scan_max_images,
            "role_order": [TAIL_SCAN_ROLE],
            "terminal_segment_is_retrieval_prior_only": True,
            "full_terminal_segment_scan": True,
        },
        "selected_candidates": candidates,
        "model_media": media,
        "evidence_contract": {
            "stage_is_retrieval_prior_only": True,
            "terminal_stage_is_not_a_cleanup_verdict": True,
            "requires_full_terminal_scan_coverage_for_no": True,
            "purpose": "fallback_organizing_action_retrieval_only",
        },
        "scope_flags": {
            "action_summary_only": True,
            "source_resistance_mp4_only": True,
            "old_predictions_read": False,
            "excel_read": False,
            "labels_read": False,
            "qwen_called": False,
            "score_computed": False,
        },
        "source_read_only": True,
        "qwen_called": False,
        "excel_accessed": False,
        "labels_accessed": False,
        "historical_predictions_accessed": False,
        "score_computed": False,
    }
    packet["local_validation"] = validate_packet(packet)
    write_json(event_root / "event_packet_manifest.json", packet)
    write_json(event_root / "local_validation.json", packet["local_validation"])
    return packet


def load_records(summary_path: Path) -> list[dict[str, Any]]:
    value = read_json(summary_path)
    if not isinstance(value, dict) or not isinstance(value.get("records"), list):
        raise ValueError("Action summary must be an object with a records array")
    records = value["records"]
    if not records or not all(isinstance(record, dict) for record in records):
        raise ValueError("Action summary records must be non-empty objects")
    ids = [record.get("source_video_id") for record in records]
    if not all(isinstance(item, str) and item for item in ids) or len(set(ids)) != len(ids):
        raise ValueError("Action summary records must have unique source_video_id values")
    return records


def validate_record_window(record: dict[str, Any], source_meta: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    window = record.get("fixed_experiment_window_seconds")
    if not isinstance(window, list) or len(window) != 2 or not all(finite(item) for item in window) or float(window[1]) <= float(window[0]):
        return ["fixed_experiment_window_invalid"]
    if float(window[0]) < -1e-6 or float(window[1]) > float(source_meta["duration_seconds"]) + 1e-3:
        errors.append("fixed_experiment_window_outside_source_video")
    return errors


def build_video(
    record: dict[str, Any],
    summary_path: Path,
    source_videos: list[Path],
    output_root: Path,
    before_seconds: float,
    max_images: int,
    tail_scan_interval_seconds: float = DEFAULT_TAIL_SCAN_INTERVAL_SECONDS,
    tail_scan_max_images: int = DEFAULT_TAIL_SCAN_MAX_IMAGES,
) -> dict[str, Any]:
    video_id = str(record["source_video_id"])
    source_video, discovery = find_source_video(video_id, source_videos)
    source_meta = video_metadata(source_video)
    errors = validate_record_window(record, source_meta)
    if errors:
        return {"source_video_id": video_id, "status": "invalid_record", "errors": errors, "packets": [], "skipped_segments": []}
    candidates = cleanup_candidates(record)
    result: dict[str, Any] = {
        "source_video_id": video_id,
        "source_video": source_meta,
        "status": "candidate_found" if candidates else "terminal_segment_fallback_pending",
        "packets": [],
        "skipped_segments": [],
        "errors": [],
        "material_cleanup_candidate_count": len(candidates),
        "terminal_fallback_packet_count": 0,
        "expected_packet_count": len(candidates),
    }
    if not candidates:
        terminal_segment, unavailable_reason = terminal_segment_for_fallback(record)
        if terminal_segment is None:
            result["status"] = "terminal_segment_fallback_unavailable"
            result["skipped_segments"].append({
                "reason": unavailable_reason or "terminal_segment_fallback_unavailable",
                "retrieval_reason": TERMINAL_FALLBACK_RETRIEVAL_REASON,
            })
            result["terminal_scan"] = {
                "available": False,
                "reason": unavailable_reason or "terminal_segment_fallback_unavailable",
                "required_full_coverage": True,
            }
            result["expected_packet_count"] = 0
            return result
        try:
            samples = plan_terminal_scan_samples(
                float(terminal_segment["start_seconds"]),
                float(terminal_segment["end_seconds"]),
                tail_scan_interval_seconds,
            )
            chunks = chunk_terminal_scan_samples(samples, tail_scan_max_images)
        except Exception as exc:
            result["status"] = "terminal_segment_fallback_unavailable"
            result["skipped_segments"].append({
                "reason": "terminal_scan_planning_failed",
                "error_type": type(exc).__name__,
                "message": str(exc),
                "retrieval_reason": TERMINAL_FALLBACK_RETRIEVAL_REASON,
            })
            result["terminal_scan"] = {
                "available": False,
                "reason": "terminal_scan_planning_failed",
                "required_full_coverage": True,
            }
            result["expected_packet_count"] = 0
            return result
        scan_id = "terminal_scan_01"
        result["status"] = "terminal_segment_fallback_candidate_found"
        result["terminal_fallback_packet_count"] = len(chunks)
        result["expected_packet_count"] = len(chunks)
        result["terminal_scan"] = {
            "available": True,
            "scan_id": scan_id,
            "terminal_segment": terminal_segment,
            "planned_bucket_count": len(samples),
            "chunk_count": len(chunks),
            "sample_interval_seconds": rounded(tail_scan_interval_seconds, 6),
            "max_images_per_chunk": tail_scan_max_images,
            "required_full_coverage": True,
        }
        for chunk_index, requests in enumerate(chunks, start=1):
            try:
                packet = build_terminal_scan_packet(
                    record,
                    source_video,
                    discovery,
                    source_meta,
                    terminal_segment,
                    requests,
                    len(samples),
                    scan_id,
                    chunk_index,
                    len(chunks),
                    summary_path,
                    output_root,
                    tail_scan_interval_seconds,
                    tail_scan_max_images,
                )
                event_root = output_root / safe_slug(video_id) / "events" / str(packet["event_id"])
                result["packets"].append(
                    {
                        "event_id": packet["event_id"],
                        "candidate_kind": packet["candidate_kind"],
                        "event_window_seconds": packet["event_window_seconds"],
                        "packet_status": packet["packet_status"],
                        "packet_path": str((event_root / "event_packet_manifest.json").resolve()),
                        "local_validation_path": str((event_root / "local_validation.json").resolve()),
                        "local_validation_valid": packet["local_validation"]["valid"],
                        "transport_valid": packet["local_validation"]["transport_valid"],
                        "evidence_selection_valid": packet["local_validation"]["evidence_selection_valid"],
                        "model_media_count": len(packet["model_media"]),
                        "tail_scan_coverage_complete": packet["tail_scan"]["coverage_complete"],
                    }
                )
            except Exception as exc:
                result["errors"].append({"chunk_index": chunk_index, "error_type": type(exc).__name__, "message": str(exc)})
        result["terminal_scan"]["coverage_complete"] = (
            len(result["packets"]) == len(chunks)
            and all(packet.get("tail_scan_coverage_complete") is True for packet in result["packets"])
        )
        if result["errors"] and not result["packets"]:
            result["status"] = "build_failed"
        elif result["errors"]:
            result["status"] = "terminal_segment_fallback_incomplete"
        return result
    window = record["fixed_experiment_window_seconds"]
    for event_index, cleanup in enumerate(candidates, start=1):
        if float(cleanup["start_seconds"]) < float(window[0]) - 1e-6 or float(cleanup["end_seconds"]) > float(window[1]) + 1e-6:
            result["errors"].append({
                "event_index": event_index,
                "error": "cleanup_interval_outside_fixed_experiment_window",
            })
            continue
        try:
            packet = build_packet(
                record,
                source_video,
                discovery,
                source_meta,
                cleanup,
                summary_path,
                output_root,
                before_seconds,
                max_images,
                event_index,
            )
            event_root = output_root / safe_slug(video_id) / "events" / str(packet["event_id"])
            result["packets"].append(
                {
                    "event_id": packet["event_id"],
                    "candidate_kind": packet["candidate_kind"],
                    "event_window_seconds": packet["event_window_seconds"],
                    "packet_status": packet["packet_status"],
                    "packet_path": str((event_root / "event_packet_manifest.json").resolve()),
                    "local_validation_path": str((event_root / "local_validation.json").resolve()),
                    "local_validation_valid": packet["local_validation"]["valid"],
                    "transport_valid": packet["local_validation"]["transport_valid"],
                    "evidence_selection_valid": packet["local_validation"]["evidence_selection_valid"],
                    "model_media_count": len(packet["model_media"]),
                }
            )
        except Exception as exc:
            result["errors"].append({"event_index": event_index, "error_type": type(exc).__name__, "message": str(exc)})
    if result["errors"] and not result["packets"]:
        result["status"] = "build_failed"
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build action-guided cleanup-action evidence packets.")
    parser.add_argument("--summary", type=Path, default=DEFAULT_ACTION_SUMMARY)
    parser.add_argument("--source-dir", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--video", action="append", dest="video_ids", help="Build only this source video ID; repeatable.")
    parser.add_argument("--before-seconds", type=float, default=DEFAULT_BEFORE_SECONDS)
    parser.add_argument("--max-images", type=int, default=DEFAULT_MAX_IMAGES)
    parser.add_argument("--tail-scan-interval-seconds", type=float, default=DEFAULT_TAIL_SCAN_INTERVAL_SECONDS)
    parser.add_argument("--tail-scan-max-images", type=int, default=DEFAULT_TAIL_SCAN_MAX_IMAGES)
    args = parser.parse_args(argv)
    if args.before_seconds <= 0 or args.max_images < 2 or args.tail_scan_interval_seconds <= 0 or args.tail_scan_max_images < 2:
        parser.error("before seconds and tail-scan interval must be positive; max-images and tail-scan-max-images must be at least 2")
    summary_path = args.summary.expanduser().resolve()
    source_dir = args.source_dir.expanduser().resolve()
    output_root = args.output_dir.expanduser().resolve()
    if not summary_path.is_file():
        parser.error(f"Action summary is missing: {summary_path}")
    if output_root.exists():
        parser.error(f"Refusing to overwrite existing output directory: {output_root}")
    source_videos = source_video_candidates(source_dir)
    if not source_videos:
        parser.error(f"No supported top-level video files found under {source_dir}")
    try:
        records = load_records(summary_path)
    except Exception as exc:
        parser.error(str(exc))
    requested = set(args.video_ids or [])
    known = {str(item["source_video_id"]) for item in records}
    unknown = sorted(requested - known)
    if unknown:
        parser.error(f"Unknown --video source_video_id values: {unknown}")
    if requested:
        records = [item for item in records if str(item["source_video_id"]) in requested]
    # Validate segment shape before creating any derivative output.
    for record in records:
        normalized_segments(record)
    output_root.mkdir(parents=True, exist_ok=False)
    video_reports: list[dict[str, Any]] = []
    for record in records:
        try:
            video_reports.append(
                build_video(
                    record,
                    summary_path,
                    source_videos,
                    output_root,
                    args.before_seconds,
                    args.max_images,
                    args.tail_scan_interval_seconds,
                    args.tail_scan_max_images,
                )
            )
        except Exception as exc:
            video_reports.append({"source_video_id": record.get("source_video_id"), "status": "build_failed", "packets": [], "skipped_segments": [], "errors": [{"error_type": type(exc).__name__, "message": str(exc)}], "expected_packet_count": 0})
    cleanup_candidate_count = sum(len(cleanup_candidates(record)) for record in records)
    fallback_packet_count = sum(int(item.get("terminal_fallback_packet_count", 0)) for item in video_reports)
    expected_packet_count = sum(int(item.get("expected_packet_count", 0)) for item in video_reports)
    fallback_video_count = sum(1 for item in video_reports if item.get("terminal_fallback_packet_count", 0))
    fallback_unavailable_count = sum(1 for item in video_reports if item.get("status") == "terminal_segment_fallback_unavailable")
    packet_count = sum(len(item.get("packets", [])) for item in video_reports)
    error_count = sum(len(item.get("errors", [])) for item in video_reports)
    invalid_count = sum(1 for item in video_reports for packet in item.get("packets", []) if packet.get("local_validation_valid") is not True)
    report = {
        "schema_version": "1.0",
        "artifact_type": "resistance_cleanup_action_guided_build_report_v3",
        "generated_at": utc_now(),
        "rubric_id": RUBRIC_ID,
        "rubric_name": RUBRIC_NAME,
        "source_action_summary_path": str(summary_path),
        "source_resistance_videos": [str(path) for path in source_videos],
        "selected_source_video_ids": [str(item["source_video_id"]) for item in records],
        "video_count": len(video_reports),
        "candidate_event_count": cleanup_candidate_count,
        "material_cleanup_candidate_count": cleanup_candidate_count,
        "terminal_fallback_video_count": fallback_video_count,
        "terminal_fallback_packet_count": fallback_packet_count,
        "terminal_fallback_unavailable_count": fallback_unavailable_count,
        "expected_packet_count": expected_packet_count,
        "packet_count": packet_count,
        "event_error_count": error_count,
        "invalid_local_validation_count": invalid_count,
        "videos": video_reports,
        "scope_flags": {
            "action_summary_only": True,
            "source_resistance_mp4_only": True,
            "old_predictions_read": False,
            "excel_read": False,
            "labels_read": False,
            "qwen_called": False,
            "score_computed": False,
        },
        "terminal_fallback_policy": {
            "trigger": "no_material_cleanup_segment",
            "sample_interval_seconds": args.tail_scan_interval_seconds,
            "max_images_per_chunk": args.tail_scan_max_images,
            "required_full_coverage": True,
        },
        "transport_build_valid": error_count == 0 and invalid_count == 0,
        "build_valid": error_count == 0 and invalid_count == 0 and packet_count == expected_packet_count,
    }
    write_json(output_root / "build_run_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["build_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
