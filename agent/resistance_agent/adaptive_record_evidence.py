"""Bounded current-run adaptive paper evidence for Rubrics 7 and 9."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import cv2


EVIDENCE_PROFILE = "record_paper"
ALLOWED_REASONS = {
    "paper_not_found",
    "writing_occlusion",
    "field_missing",
    "low_confidence",
    "single_frame_support",
    "digit_conflict",
    "row_identity_conflict",
    "recording_stage_missing",
}
ALLOWED_SEARCH_MODES = {
    "adjacent_dense",
    "post_write_reveal",
    "recording_stage_coverage",
    "current_run_broad_writing_search",
}
MIN_INTERVAL_SECONDS = 0.1
MAX_INTERVAL_SECONDS = 0.5
MAX_RANGE_SECONDS = 4.0
MAX_TOTAL_SECONDS = 6.0
MAX_RANGES = 3
MAX_FRAMES = 20
MAX_REQUEST_ROUNDS = 2
MAX_NEW_FRAMES_PER_CYCLE = 32


class AdaptiveRecordEvidenceError(ValueError):
    """Raised when a record-paper request violates the live contract."""


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AdaptiveRecordEvidenceError(f"{field} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise AdaptiveRecordEvidenceError(f"{field} must be finite")
    return number


def _video_info(state: dict[str, Any]) -> tuple[Path, float, int, str]:
    video = state.get("video") if isinstance(state.get("video"), dict) else {}
    path = Path(str(video.get("path") or "")).resolve()
    if not path.is_file():
        raise AdaptiveRecordEvidenceError("current run video is missing")
    digest = _sha256(path)
    if digest != video.get("sha256"):
        raise AdaptiveRecordEvidenceError("current run source video changed")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise AdaptiveRecordEvidenceError("unable to open current run video")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if fps <= 0.0 or frame_count <= 0:
        raise AdaptiveRecordEvidenceError("current run video metadata is invalid")
    return path, fps, frame_count, digest


def _normalize_targets(cycle: int, target_fields: Any) -> list[str]:
    allowed = {f"u{cycle}", f"i{cycle}"}
    if not isinstance(target_fields, list) or not target_fields:
        return sorted(allowed)
    normalized: list[str] = []
    for value in target_fields:
        if not isinstance(value, str) or value not in allowed:
            raise AdaptiveRecordEvidenceError(
                f"target_fields must contain only {sorted(allowed)}"
            )
        if value not in normalized:
            normalized.append(value)
    return normalized


def _stage_runs(run_dir: Path) -> list[dict[str, Any]]:
    try:
        from .adaptive_evidence import current_stage_runs
    except ImportError:  # pragma: no cover
        from adaptive_evidence import current_stage_runs  # type: ignore
    return current_stage_runs(run_dir)


def _validate_ranges(
    raw_ranges: Any,
    stages: list[dict[str, Any]],
    cycle: int,
    search_mode: str,
    duration: float,
) -> list[dict[str, float]]:
    if not isinstance(raw_ranges, list) or not raw_ranges or len(raw_ranges) > MAX_RANGES:
        raise AdaptiveRecordEvidenceError(
            f"time_ranges must contain 1-{MAX_RANGES} items"
        )
    recording = [item for item in stages if item.get("stage") == f"recording_{cycle}"]
    active = [item for item in stages if item.get("stage") != "material_cleanup"]
    if not recording and search_mode != "current_run_broad_writing_search":
        raise AdaptiveRecordEvidenceError(
            f"recording_{cycle} is missing; use current_run_broad_writing_search"
        )
    if not active:
        raise AdaptiveRecordEvidenceError("current run has no observed active stages")
    ranges: list[dict[str, float]] = []
    total = 0.0
    for index, raw in enumerate(raw_ranges):
        if not isinstance(raw, dict):
            raise AdaptiveRecordEvidenceError(f"time_ranges[{index}] must be an object")
        start = _number(raw.get("start_seconds"), f"time_ranges[{index}].start_seconds")
        end = _number(raw.get("end_seconds"), f"time_ranges[{index}].end_seconds")
        if not 0.0 <= start < end <= duration:
            raise AdaptiveRecordEvidenceError(
                f"time_ranges[{index}] must be inside the current video"
            )
        length = end - start
        if length > MAX_RANGE_SECONDS:
            raise AdaptiveRecordEvidenceError(
                f"time_ranges[{index}] exceeds {MAX_RANGE_SECONDS}s"
            )
        if recording:
            supported = any(
                start >= max(0.0, float(item["start_seconds"]) - 1.0)
                and end <= min(duration, float(item["end_seconds"]) + 6.0)
                for item in recording
            )
        else:
            active_start = min(float(item["start_seconds"]) for item in active)
            active_end = max(float(item["end_seconds"]) for item in active)
            supported = start >= active_start and end <= active_end
        if not supported:
            raise AdaptiveRecordEvidenceError(
                f"time_ranges[{index}] is outside the current cycle evidence area"
            )
        total += length
        ranges.append(
            {"start_seconds": round(start, 6), "end_seconds": round(end, 6)}
        )
    if total > MAX_TOTAL_SECONDS:
        raise AdaptiveRecordEvidenceError(
            f"total requested duration exceeds {MAX_TOTAL_SECONDS}s"
        )
    return ranges


def _sample_numbers(
    ranges: list[dict[str, float]], fps: float, frame_count: int, interval: float
) -> list[int]:
    maximum = max(0, frame_count - 1)
    numbers: set[int] = set()
    for value in ranges:
        start, end = value["start_seconds"], value["end_seconds"]
        count = int(math.floor((end - start) / interval + 1e-9)) + 1
        for index in range(count):
            timestamp = min(end, start + index * interval)
            numbers.add(min(maximum, max(0, int(round(timestamp * fps)))))
        numbers.add(min(maximum, max(0, int(round(end * fps)))))
    return sorted(numbers)


def _known_frame_numbers(run_dir: Path) -> set[int]:
    known: set[int] = set()
    for path in run_dir.rglob("*.json"):
        try:
            value = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError, UnicodeError):
            continue

        def visit(item: Any) -> None:
            if isinstance(item, dict):
                number = item.get("frame_number")
                if isinstance(number, int) and not isinstance(number, bool):
                    known.add(number)
                for child in item.values():
                    visit(child)
            elif isinstance(item, list):
                for child in item:
                    visit(child)

        visit(value)
    return known


def _export_paper_row(
    *,
    frame: Any,
    frame_number: int,
    timestamp: float,
    request_dir: Path,
    request_number: int,
    cycle: int,
    target_fields: list[str],
    source_digest: str,
) -> dict[str, Any]:
    try:
        from . import record_rubrics
    except ImportError:  # pragma: no cover
        import record_rubrics  # type: ignore

    stem = f"cycle_{cycle}_adaptive_{request_number:02d}_{frame_number:08d}_{timestamp:010.3f}s"
    panorama = request_dir / "frames" / f"{stem}.jpg"
    panorama.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(panorama), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95]):
        raise AdaptiveRecordEvidenceError(f"unable to write frame {frame_number}")
    candidate_rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(record_rubrics._paper_candidates(frame), start=1):
        left, top, right, bottom = (int(value) for value in candidate["bbox_xyxy"])
        crop = frame[top:bottom, left:right]
        if crop.size == 0:
            continue
        prepared = record_rubrics._enhance(crop)
        path = request_dir / "paper_rois" / f"{stem}_candidate_{index:02d}.jpg"
        record_rubrics._write_jpeg(path, prepared, 96)
        candidate_rows.append({**candidate, "roi_path": str(path.resolve())})
    field_views: list[dict[str, Any]] = []
    if candidate_rows:
        field_views = record_rubrics._dynamic_paper_field_views(
            frame, candidate_rows, request_dir, stem
        )
    field_view = field_views[0] if field_views else None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return {
        "cycle": cycle,
        "frame_id": f"frame_{frame_number:08d}",
        "image_group_id": f"record_paper_{cycle}_{request_number:02d}_{frame_number:08d}",
        "frame_number": frame_number,
        "timestamp_seconds": round(timestamp, 6),
        "panorama_path": str(panorama.resolve()),
        "paper_candidates": candidate_rows,
        "paper_search_views": [],
        "paper_calibrated_view": None,
        "paper_field_view": field_view,
        "paper_field_views": field_views,
        "sharpness": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 6),
        "adaptive_request_number": request_number,
        "adaptive_target_fields": target_fields,
        "window_source": EVIDENCE_PROFILE,
        "source_video_sha256": source_digest,
    }


def request_additional_record_evidence(
    *,
    run_dir: Path,
    state: dict[str, Any],
    rubric_ids: Any,
    reason: str,
    time_ranges: Any,
    cycle: int | None,
    target_fields: Any,
    anchor_frame_ids: Any,
    search_mode: str | None,
    interval_seconds: float = 0.2,
    max_frames: int = 20,
    roi_mode: str = "dynamic_paper_tracking",
    view: str = "paper_fields",
) -> dict[str, Any]:
    if state.get("mode") != "execute":
        raise AdaptiveRecordEvidenceError(
            "adaptive record evidence is only valid in execute mode"
        )
    if cycle not in {1, 2}:
        raise AdaptiveRecordEvidenceError("cycle must be 1 or 2")
    expected_rubric = 7 if cycle == 1 else 9
    if rubric_ids != [expected_rubric]:
        raise AdaptiveRecordEvidenceError(
            f"cycle {cycle} requires rubric_ids=[{expected_rubric}]"
        )
    if reason not in ALLOWED_REASONS:
        raise AdaptiveRecordEvidenceError(
            f"reason must be one of {sorted(ALLOWED_REASONS)}"
        )
    if search_mode not in ALLOWED_SEARCH_MODES:
        raise AdaptiveRecordEvidenceError(
            f"search_mode must be one of {sorted(ALLOWED_SEARCH_MODES)}"
        )
    if roi_mode != "dynamic_paper_tracking" or view not in {
        "paper_full",
        "paper_fields",
    }:
        raise AdaptiveRecordEvidenceError(
            "record paper requests require dynamic_paper_tracking and a paper view"
        )
    targets = _normalize_targets(cycle, target_fields)
    anchors = []
    if anchor_frame_ids is not None:
        if not isinstance(anchor_frame_ids, list):
            raise AdaptiveRecordEvidenceError("anchor_frame_ids must be an array")
        for value in anchor_frame_ids:
            if not isinstance(value, str) or not value.startswith("frame_"):
                raise AdaptiveRecordEvidenceError("anchor_frame_ids must contain frame IDs")
            if value not in anchors:
                anchors.append(value)
    interval = _number(interval_seconds, "interval_seconds")
    if not MIN_INTERVAL_SECONDS <= interval <= MAX_INTERVAL_SECONDS:
        raise AdaptiveRecordEvidenceError(
            f"interval_seconds must be between {MIN_INTERVAL_SECONDS} and {MAX_INTERVAL_SECONDS}"
        )
    if type(max_frames) is not int or not 1 <= max_frames <= MAX_FRAMES:
        raise AdaptiveRecordEvidenceError(
            f"max_frames must be an integer from 1 to {MAX_FRAMES}"
        )

    request_root = run_dir / "adaptive_evidence" / EVIDENCE_PROFILE / f"cycle_{cycle}"
    previous = sorted(request_root.glob("request_*/request.json"))
    if len(previous) >= MAX_REQUEST_ROUNDS:
        raise AdaptiveRecordEvidenceError(
            f"adaptive record request limit is {MAX_REQUEST_ROUNDS} rounds per cycle"
        )
    request_number = len(previous) + 1
    video_path, fps, frame_count, digest = _video_info(state)
    duration = frame_count / fps
    stages = _stage_runs(run_dir)
    ranges = _validate_ranges(time_ranges, stages, cycle, search_mode, duration)
    requested = _sample_numbers(ranges, fps, frame_count, interval)
    if len(requested) > max_frames:
        raise AdaptiveRecordEvidenceError(
            f"request would produce {len(requested)} frames; increase interval or lower range"
        )
    known = _known_frame_numbers(run_dir)
    new_numbers = [number for number in requested if number not in known]
    previous_new_frames = 0
    for path in sorted(request_root.glob("request_*/result.json")):
        try:
            previous_result = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError, UnicodeError):
            continue
        previous_new_frames += int(
            (previous_result.get("sampling") or {}).get("new_frame_count") or 0
        )
    remaining_budget = MAX_NEW_FRAMES_PER_CYCLE - previous_new_frames
    if len(new_numbers) > remaining_budget:
        raise AdaptiveRecordEvidenceError(
            f"cycle {cycle} adaptive frame budget has {remaining_budget} frames remaining"
        )
    request_dir = request_root / f"request_{request_number:02d}"
    request = {
        "schema_version": "resistance_agent_record_adaptive_request.v1",
        "run_id": state.get("run_id"),
        "evidence_profile": EVIDENCE_PROFILE,
        "request_number": request_number,
        "rubric_ids": rubric_ids,
        "cycle": cycle,
        "reason": reason,
        "target_fields": targets,
        "anchor_frame_ids": anchors,
        "search_mode": search_mode,
        "time_ranges": ranges,
        "sampling": {
            "requested_interval_seconds": round(interval, 6),
            "max_frames": max_frames,
            "requested_frame_count": len(requested),
            "new_frame_count": len(new_numbers),
            "cycle_new_frame_count_after_request": previous_new_frames + len(new_numbers),
            "cycle_new_frame_limit": MAX_NEW_FRAMES_PER_CYCLE,
        },
        "roi_mode": roi_mode,
        "view": view,
        "source_video_sha256": digest,
        "stage_runs_used": stages,
        "selection_basis": "current_video_observed_situation_only",
    }
    _write_json(request_dir / "request.json", request)

    rows: list[dict[str, Any]] = []
    capture = cv2.VideoCapture(str(video_path))
    try:
        for number in new_numbers:
            capture.set(cv2.CAP_PROP_POS_FRAMES, number)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            actual = int(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            rows.append(
                _export_paper_row(
                    frame=frame,
                    frame_number=actual,
                    timestamp=actual / fps,
                    request_dir=request_dir,
                    request_number=request_number,
                    cycle=cycle,
                    target_fields=targets,
                    source_digest=digest,
                )
            )
    finally:
        capture.release()
    selected_count = sum(bool(row.get("paper_field_views")) for row in rows)
    result = {
        "schema_version": "resistance_agent_record_adaptive_evidence.v1",
        "status": "additional_evidence_ready" if rows else "no_new_frames",
        "run_id": state.get("run_id"),
        "evidence_profile": EVIDENCE_PROFILE,
        "request_number": request_number,
        "rubric_ids": rubric_ids,
        "cycle": cycle,
        "reason": reason,
        "target_fields": targets,
        "anchor_frame_ids": anchors,
        "search_mode": search_mode,
        "sampling": request["sampling"],
        "roi_mode": roi_mode,
        "view": view,
        "source_video_sha256": digest,
        "deduplicated_frame_count": len(requested) - len(new_numbers),
        "frame_count": len(rows),
        "selected_frame_count": selected_count,
        "paper_rows": rows,
        "next_tool": "run_rubric_bundle" if rows else None,
        "next_arguments": {"rubric_ids": [7, 9]} if rows else None,
        "selection_basis": "current_video_observed_situation_only",
        "video_id_used_for_routing": False,
        "historical_artifacts_used": False,
        "fixed_video_roi_used": False,
        "excel_accessed": False,
        "ground_truth_sent_to_model": False,
        "source_video_unchanged": _sha256(video_path) == digest,
    }
    _write_json(request_dir / "result.json", result)
    return {
        **result,
        "request_path": str((request_dir / "request.json").resolve()),
        "result_path": str((request_dir / "result.json").resolve()),
    }
