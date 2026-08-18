"""Bounded current-run adaptive meter evidence for Rubrics 7 and 9."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import cv2


EVIDENCE_PROFILE = "record_meter"
ALLOWED_REASONS = {
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
}
ALLOWED_SEARCH_MODES = {"adjacent_meter_dense", "current_run_meter_search"}
MIN_INTERVAL_SECONDS = 0.1
MAX_INTERVAL_SECONDS = 0.5
MAX_RANGE_SECONDS = 4.0
MAX_TOTAL_SECONDS = 6.0
MAX_RANGES = 3
MAX_FRAMES = 20
MAX_REQUEST_ROUNDS = 2
MAX_NEW_FRAMES_PER_CYCLE = 32
MAX_CANDIDATES_PER_FRAME = 4


class AdaptiveRecordMeterEvidenceError(ValueError):
    """Raised when a record-meter request violates the live contract."""


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
        raise AdaptiveRecordMeterEvidenceError(f"{field} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise AdaptiveRecordMeterEvidenceError(f"{field} must be finite")
    return number


def _video_info(state: dict[str, Any]) -> tuple[Path, float, int, str]:
    video = state.get("video") if isinstance(state.get("video"), dict) else {}
    path = Path(str(video.get("path") or "")).resolve()
    if not path.is_file():
        raise AdaptiveRecordMeterEvidenceError("current run video is missing")
    digest = _sha256(path)
    if digest != video.get("sha256"):
        raise AdaptiveRecordMeterEvidenceError("current run source video changed")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise AdaptiveRecordMeterEvidenceError("unable to open current run video")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if fps <= 0.0 or frame_count <= 0:
        raise AdaptiveRecordMeterEvidenceError("current run video metadata is invalid")
    return path, fps, frame_count, digest


def _stage_runs(run_dir: Path) -> list[dict[str, Any]]:
    try:
        from .adaptive_evidence import current_stage_runs
    except ImportError:  # pragma: no cover
        from adaptive_evidence import current_stage_runs  # type: ignore
    return current_stage_runs(run_dir)


def _normalize_target_roles(target_roles: Any) -> list[str]:
    allowed = {"ammeter", "voltmeter"}
    if not isinstance(target_roles, list) or not target_roles:
        return sorted(allowed)
    normalized: list[str] = []
    for value in target_roles:
        if not isinstance(value, str) or value not in allowed:
            raise AdaptiveRecordMeterEvidenceError(
                f"target_roles must contain only {sorted(allowed)}"
            )
        if value not in normalized:
            normalized.append(value)
    return normalized


def cycle_meter_intervals(
    stages: list[dict[str, Any]], cycle: int, duration: float
) -> list[tuple[float, float]]:
    measurements = [
        item for item in stages if item.get("stage") == f"measurement_{cycle}"
    ]
    recordings = [
        item for item in stages if item.get("stage") == f"recording_{cycle}"
    ]
    intervals: list[tuple[float, float]] = []
    if measurements:
        for stage in measurements + recordings:
            intervals.append(
                (
                    max(0.0, float(stage["start_seconds"]) - 0.5),
                    min(duration, float(stage["end_seconds"]) + 0.5),
                )
            )
        return intervals
    rewiring_ends = [
        float(item["end_seconds"])
        for item in stages
        if item.get("stage") == "circuit_rewiring"
    ]
    for stage in recordings:
        recording_start = float(stage["start_seconds"])
        start = max(0.0, recording_start - 12.0)
        if cycle == 2:
            preceding_rewiring = [
                value for value in rewiring_ends if value <= recording_start
            ]
            if preceding_rewiring:
                start = max(start, max(preceding_rewiring))
        intervals.append(
            (start, min(duration, float(stage["end_seconds"]) + 6.0))
        )
    return intervals


def current_cycle_meter_intervals(
    run_dir: Path, cycle: int, duration: float
) -> list[tuple[float, float]]:
    return cycle_meter_intervals(_stage_runs(run_dir), cycle, duration)


def _validate_ranges(
    raw_ranges: Any,
    stages: list[dict[str, Any]],
    cycle: int,
    duration: float,
) -> list[dict[str, float]]:
    if not isinstance(raw_ranges, list) or not raw_ranges or len(raw_ranges) > MAX_RANGES:
        raise AdaptiveRecordMeterEvidenceError(
            f"time_ranges must contain 1-{MAX_RANGES} items"
        )
    measurements = [
        item for item in stages if item.get("stage") == f"measurement_{cycle}"
    ]
    recordings = [
        item for item in stages if item.get("stage") == f"recording_{cycle}"
    ]
    if not measurements and not recordings:
        raise AdaptiveRecordMeterEvidenceError(
            f"current run has no observed meter stage for cycle {cycle}"
        )
    allowed_intervals = cycle_meter_intervals(stages, cycle, duration)

    ranges: list[dict[str, float]] = []
    total = 0.0
    for index, raw in enumerate(raw_ranges):
        if not isinstance(raw, dict):
            raise AdaptiveRecordMeterEvidenceError(
                f"time_ranges[{index}] must be an object"
            )
        start = _number(raw.get("start_seconds"), f"time_ranges[{index}].start_seconds")
        end = _number(raw.get("end_seconds"), f"time_ranges[{index}].end_seconds")
        if not 0.0 <= start < end <= duration:
            raise AdaptiveRecordMeterEvidenceError(
                f"time_ranges[{index}] must be inside the current video"
            )
        length = end - start
        if length > MAX_RANGE_SECONDS:
            raise AdaptiveRecordMeterEvidenceError(
                f"time_ranges[{index}] exceeds {MAX_RANGE_SECONDS}s"
            )
        supported = any(
            start >= allowed_start and end <= allowed_end
            for allowed_start, allowed_end in allowed_intervals
        )
        if not supported:
            raise AdaptiveRecordMeterEvidenceError(
                f"time_ranges[{index}] is outside current cycle {cycle} meter stages"
            )
        total += length
        ranges.append(
            {"start_seconds": round(start, 6), "end_seconds": round(end, 6)}
        )
    if total > MAX_TOTAL_SECONDS:
        raise AdaptiveRecordMeterEvidenceError(
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


def _export_meter_row(
    *,
    frame: Any,
    frame_number: int,
    timestamp: float,
    request_dir: Path,
    request_number: int,
    cycle: int,
    target_roles: list[str],
    source_digest: str,
) -> dict[str, Any]:
    try:
        from . import meter_rubrics
    except ImportError:  # pragma: no cover
        import meter_rubrics  # type: ignore

    stem = (
        f"cycle_{cycle}_adaptive_meter_{request_number:02d}_"
        f"{frame_number:08d}_{timestamp:010.3f}s"
    )
    panorama = request_dir / "frames" / f"{stem}.jpg"
    panorama.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(panorama), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95]):
        raise AdaptiveRecordMeterEvidenceError(
            f"unable to write frame {frame_number}"
        )
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    exported = meter_rubrics._export_candidates(
        {
            "frame_id": f"frame_{frame_number:08d}",
            "frame_number": frame_number,
            "timestamp_seconds": round(timestamp, 6),
            "frame_path": str(panorama.resolve()),
            "sharpness": sharpness,
            "window_source": EVIDENCE_PROFILE,
            "window_priority": -1,
        },
        request_dir / "dynamic_meter_candidates",
    )
    candidates = [
        item
        for item in exported.get("candidates", [])
        if isinstance(item, dict)
    ][:MAX_CANDIDATES_PER_FRAME]
    role_views: dict[str, dict[str, Any]] = {}
    face_views: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        role = str(candidate.get("role_hint") or "")
        if role not in {"ammeter", "voltmeter"}:
            continue
        role_views.setdefault(
            role,
            {
                "image_path": candidate.get("enhanced_path")
                or candidate.get("wide_path"),
                "candidate_id": candidate.get("candidate_id"),
                "dynamic": True,
            },
        )
        if candidate.get("face_path"):
            face_views.setdefault(
                role,
                {
                    "image_path": candidate.get("face_path"),
                    "candidate_id": candidate.get("candidate_id"),
                    "dynamic": True,
                    "geometry": None,
                },
            )
    return {
        "cycle": cycle,
        "frame_id": f"frame_{frame_number:08d}",
        "image_group_id": (
            f"record_meter_{cycle}_{request_number:02d}_{frame_number:08d}"
        ),
        "frame_number": frame_number,
        "timestamp_seconds": round(timestamp, 6),
        "meter_roi_normalized_xyxy": None,
        "image_path": str(panorama.resolve()),
        "role_views": role_views,
        "face_views": face_views,
        "dynamic_meter_candidates": candidates,
        "sharpness": round(sharpness, 6),
        "adaptive_request_number": request_number,
        "adaptive_target_roles": target_roles,
        "window_source": EVIDENCE_PROFILE,
        "window_priority": -1,
        "source_video_sha256": source_digest,
    }


def request_additional_record_meter_evidence(
    *,
    run_dir: Path,
    state: dict[str, Any],
    rubric_ids: Any,
    reason: str,
    time_ranges: Any,
    cycle: int | None,
    target_roles: Any,
    anchor_frame_ids: Any,
    search_mode: str | None,
    interval_seconds: float = 0.2,
    max_frames: int = 20,
    roi_mode: str = "dynamic_meter_candidates",
    view: str = "meter_pair",
) -> dict[str, Any]:
    if state.get("mode") != "execute":
        raise AdaptiveRecordMeterEvidenceError(
            "adaptive record meter evidence is only valid in execute mode"
        )
    if cycle not in {1, 2}:
        raise AdaptiveRecordMeterEvidenceError("cycle must be 1 or 2")
    expected_rubric = 7 if cycle == 1 else 9
    if rubric_ids != [expected_rubric]:
        raise AdaptiveRecordMeterEvidenceError(
            f"cycle {cycle} requires rubric_ids=[{expected_rubric}]"
        )
    if reason not in ALLOWED_REASONS:
        raise AdaptiveRecordMeterEvidenceError(
            f"reason must be one of {sorted(ALLOWED_REASONS)}"
        )
    if search_mode not in ALLOWED_SEARCH_MODES:
        raise AdaptiveRecordMeterEvidenceError(
            f"search_mode must be one of {sorted(ALLOWED_SEARCH_MODES)}"
        )
    if roi_mode != "dynamic_meter_candidates" or view != "meter_pair":
        raise AdaptiveRecordMeterEvidenceError(
            "record meter requests require dynamic_meter_candidates and meter_pair"
        )
    roles = _normalize_target_roles(target_roles)
    anchors: list[str] = []
    if anchor_frame_ids is not None:
        if not isinstance(anchor_frame_ids, list):
            raise AdaptiveRecordMeterEvidenceError(
                "anchor_frame_ids must be an array"
            )
        for value in anchor_frame_ids:
            if not isinstance(value, str) or not value.startswith("frame_"):
                raise AdaptiveRecordMeterEvidenceError(
                    "anchor_frame_ids must contain frame IDs"
                )
            if value not in anchors:
                anchors.append(value)
    interval = _number(interval_seconds, "interval_seconds")
    if not MIN_INTERVAL_SECONDS <= interval <= MAX_INTERVAL_SECONDS:
        raise AdaptiveRecordMeterEvidenceError(
            f"interval_seconds must be between {MIN_INTERVAL_SECONDS} and {MAX_INTERVAL_SECONDS}"
        )
    if type(max_frames) is not int or not 1 <= max_frames <= MAX_FRAMES:
        raise AdaptiveRecordMeterEvidenceError(
            f"max_frames must be an integer from 1 to {MAX_FRAMES}"
        )

    request_root = run_dir / "adaptive_evidence" / EVIDENCE_PROFILE / f"cycle_{cycle}"
    previous = sorted(request_root.glob("request_*/request.json"))
    if len(previous) >= MAX_REQUEST_ROUNDS:
        raise AdaptiveRecordMeterEvidenceError(
            f"adaptive record meter request limit is {MAX_REQUEST_ROUNDS} rounds per cycle"
        )
    request_number = len(previous) + 1
    video_path, fps, frame_count, digest = _video_info(state)
    duration = frame_count / fps
    stages = _stage_runs(run_dir)
    ranges = _validate_ranges(time_ranges, stages, cycle, duration)
    requested = _sample_numbers(ranges, fps, frame_count, interval)
    if len(requested) > max_frames:
        raise AdaptiveRecordMeterEvidenceError(
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
        raise AdaptiveRecordMeterEvidenceError(
            f"cycle {cycle} adaptive meter frame budget has {remaining_budget} frames remaining"
        )

    request = {
        "schema_version": "resistance_agent_record_meter_adaptive_request.v1",
        "run_id": state.get("run_id"),
        "evidence_profile": EVIDENCE_PROFILE,
        "request_number": request_number,
        "rubric_ids": rubric_ids,
        "cycle": cycle,
        "reason": reason,
        "target_roles": roles,
        "anchor_frame_ids": anchors,
        "search_mode": search_mode,
        "time_ranges": ranges,
        "sampling": {
            "requested_interval_seconds": round(interval, 6),
            "max_frames": max_frames,
            "requested_frame_count": len(requested),
            "new_frame_count": len(new_numbers),
            "cycle_new_frame_count_after_request": previous_new_frames
            + len(new_numbers),
            "cycle_new_frame_limit": MAX_NEW_FRAMES_PER_CYCLE,
        },
        "roi_mode": roi_mode,
        "view": view,
        "source_video_sha256": digest,
        "stage_runs_used": stages,
        "selection_basis": "current_video_observed_situation_only",
    }
    if not new_numbers:
        request_path = request_root / "last_no_new_request.json"
        result_path = request_root / "last_no_new_result.json"
        result = {
            "schema_version": "resistance_agent_record_meter_adaptive_evidence.v1",
            "status": "no_new_frames",
            "run_id": state.get("run_id"),
            "evidence_profile": EVIDENCE_PROFILE,
            "request_number": request_number,
            "rubric_ids": rubric_ids,
            "cycle": cycle,
            "reason": reason,
            "target_roles": roles,
            "anchor_frame_ids": anchors,
            "search_mode": search_mode,
            "sampling": request["sampling"],
            "roi_mode": roi_mode,
            "view": view,
            "source_video_sha256": digest,
            "deduplicated_frame_count": len(requested),
            "frame_count": 0,
            "selected_frame_count": 0,
            "meter_rows": [],
            "next_tool": None,
            "next_arguments": None,
            "selection_basis": "current_video_observed_situation_only",
            "video_id_used_for_routing": False,
            "historical_artifacts_used": False,
            "fixed_video_roi_used": False,
            "paper_values_used": False,
            "excel_accessed": False,
            "ground_truth_sent_to_model": False,
            "source_video_unchanged": _sha256(video_path) == digest,
            "round_consumed": False,
        }
        _write_json(request_path, request)
        _write_json(result_path, result)
        return {
            **result,
            "request_path": str(request_path.resolve()),
            "result_path": str(result_path.resolve()),
        }

    request_dir = request_root / f"request_{request_number:02d}"
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
                _export_meter_row(
                    frame=frame,
                    frame_number=actual,
                    timestamp=actual / fps,
                    request_dir=request_dir,
                    request_number=request_number,
                    cycle=cycle,
                    target_roles=roles,
                    source_digest=digest,
                )
            )
    finally:
        capture.release()

    selected_count = sum(bool(row.get("dynamic_meter_candidates")) for row in rows)
    result = {
        "schema_version": "resistance_agent_record_meter_adaptive_evidence.v1",
        "status": "additional_evidence_ready" if rows else "no_new_frames",
        "run_id": state.get("run_id"),
        "evidence_profile": EVIDENCE_PROFILE,
        "request_number": request_number,
        "rubric_ids": rubric_ids,
        "cycle": cycle,
        "reason": reason,
        "target_roles": roles,
        "anchor_frame_ids": anchors,
        "search_mode": search_mode,
        "sampling": request["sampling"],
        "roi_mode": roi_mode,
        "view": view,
        "source_video_sha256": digest,
        "deduplicated_frame_count": len(requested) - len(new_numbers),
        "frame_count": len(rows),
        "selected_frame_count": selected_count,
        "meter_rows": rows,
        "next_tool": "run_rubric_bundle" if rows else None,
        "next_arguments": {"rubric_ids": [7, 9]} if rows else None,
        "selection_basis": "current_video_observed_situation_only",
        "video_id_used_for_routing": False,
        "historical_artifacts_used": False,
        "fixed_video_roi_used": False,
        "paper_values_used": False,
        "excel_accessed": False,
        "ground_truth_sent_to_model": False,
        "source_video_unchanged": _sha256(video_path) == digest,
        "round_consumed": True,
    }
    _write_json(request_dir / "result.json", result)
    return {
        **result,
        "request_path": str((request_dir / "request.json").resolve()),
        "result_path": str((request_dir / "result.json").resolve()),
    }
