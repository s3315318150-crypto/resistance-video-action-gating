"""Bounded, current-run-only requests for additional visual evidence.

The scheduler may request more frames after an observation is weak, but this
module owns the limits, stage containment, source integrity and frame export.
The first implementation serves R5/R6 meter evidence and deliberately keeps
the request format extensible for other Rubrics.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import cv2


ALLOWED_RUBRICS = {5, 6}
ALLOWED_REASONS = {
    "meter_pointer_occluded",
    "meter_identity_conflict",
    "pointer_state_conflict",
    "low_confidence",
    "adjacent_state_change",
    "other",
}
MIN_INTERVAL_SECONDS = 0.1
MAX_INTERVAL_SECONDS = 1.0
MAX_RANGE_SECONDS = 4.0
MAX_TOTAL_SECONDS = 8.0
MAX_RANGES = 3
MAX_FRAMES = 32
MAX_REQUEST_ROUNDS = 2
STAGE_NAMES = {
    "circuit_wiring",
    "measurement_1",
    "recording_1",
    "circuit_rewiring",
    "measurement_2",
    "recording_2",
    "material_cleanup",
}
METER_STAGE_NAMES = STAGE_NAMES - {"material_cleanup"}


class AdaptiveEvidenceError(ValueError):
    """Raised when an adaptive evidence request violates its contract."""


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AdaptiveEvidenceError(f"{field} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise AdaptiveEvidenceError(f"{field} must be finite")
    return number


def _walk_stage_runs(value: Any, output: list[dict[str, Any]]) -> None:
    if isinstance(value, dict):
        stage = value.get("stage")
        if (
            isinstance(stage, str)
            and stage in STAGE_NAMES
            and "start_seconds" in value
            and "end_seconds" in value
        ):
            try:
                start = _finite_number(value["start_seconds"], "start_seconds")
                end = _finite_number(value["end_seconds"], "end_seconds")
            except AdaptiveEvidenceError:
                start = end = -1.0
            if 0.0 <= start < end:
                output.append(
                    {
                        "stage": stage,
                        "start_seconds": round(start, 6),
                        "end_seconds": round(end, 6),
                    }
                )
        for child in value.values():
            _walk_stage_runs(child, output)
    elif isinstance(value, list):
        for child in value:
            _walk_stage_runs(child, output)


def current_stage_runs(run_dir: Path) -> list[dict[str, Any]]:
    """Read only JSON artifacts under this run directory."""
    found: dict[tuple[str, float, float], dict[str, Any]] = {}
    for path in sorted(run_dir.rglob("*.json")):
        if "adaptive_evidence" in path.parts:
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError, UnicodeError):
            continue
        records: list[dict[str, Any]] = []
        _walk_stage_runs(value, records)
        for item in records:
            key = (item["stage"], item["start_seconds"], item["end_seconds"])
            found[key] = item
    return sorted(found.values(), key=lambda item: (item["start_seconds"], item["end_seconds"]))


def _known_frame_numbers(root: Path) -> set[int]:
    known: set[int] = set()
    for path in root.rglob("*.json"):
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


def _request_ranges(
    raw_ranges: Any,
    stages: list[dict[str, Any]],
    duration: float,
) -> list[dict[str, float]]:
    if not isinstance(raw_ranges, list) or not raw_ranges or len(raw_ranges) > MAX_RANGES:
        raise AdaptiveEvidenceError(f"time_ranges must contain 1-{MAX_RANGES} items")
    if not stages:
        raise AdaptiveEvidenceError("current run has no observed stage intervals")
    ranges: list[dict[str, float]] = []
    total = 0.0
    for index, raw in enumerate(raw_ranges):
        if not isinstance(raw, dict):
            raise AdaptiveEvidenceError(f"time_ranges[{index}] must be an object")
        start = _finite_number(raw.get("start_seconds"), f"time_ranges[{index}].start_seconds")
        end = _finite_number(raw.get("end_seconds"), f"time_ranges[{index}].end_seconds")
        if not 0.0 <= start < end <= duration:
            raise AdaptiveEvidenceError(f"time_ranges[{index}] must be inside the current video")
        length = end - start
        if length > MAX_RANGE_SECONDS:
            raise AdaptiveEvidenceError(f"time_ranges[{index}] exceeds {MAX_RANGE_SECONDS}s")
        supported = any(
            start >= max(0.0, float(stage["start_seconds"]) - 2.0)
            and end <= min(duration, float(stage["end_seconds"]) + 2.0)
            for stage in stages
        )
        if not supported:
            raise AdaptiveEvidenceError(
                f"time_ranges[{index}] is outside current observed stages"
            )
        total += length
        ranges.append({"start_seconds": round(start, 6), "end_seconds": round(end, 6)})
    if total > MAX_TOTAL_SECONDS:
        raise AdaptiveEvidenceError(f"total requested duration exceeds {MAX_TOTAL_SECONDS}s")
    return ranges


def _sample_frame_numbers(
    ranges: list[dict[str, float]], fps: float, frame_count: int, interval: float
) -> list[int]:
    max_frame = max(0, frame_count - 1)
    values: set[int] = set()
    for item in ranges:
        start = item["start_seconds"]
        end = item["end_seconds"]
        count = int(math.floor((end - start) / interval + 1e-9)) + 1
        for offset in range(count):
            timestamp = min(end, start + offset * interval)
            values.add(min(max_frame, max(0, int(round(timestamp * fps)))))
        values.add(min(max_frame, max(0, int(round(end * fps)))))
    return sorted(values)


def request_additional_evidence(
    *,
    run_dir: Path,
    state: dict[str, Any],
    rubric_ids: Any,
    reason: str,
    time_ranges: Any,
    interval_seconds: float = 0.2,
    max_frames: int = 24,
    roi_mode: str = "dynamic_meter_candidates",
    view: str = "meter_pair",
    evidence_profile: str | None = None,
    cycle: int | None = None,
    target_fields: Any = None,
    target_roles: Any = None,
    anchor_frame_ids: Any = None,
    search_mode: str | None = None,
) -> dict[str, Any]:
    """Execute one bounded adaptive request against the current run video."""
    if evidence_profile not in {None, "meter_pair"}:
        raise AdaptiveEvidenceError("evidence_profile must be meter_pair")
    if state.get("mode") != "execute":
        raise AdaptiveEvidenceError("adaptive evidence is only valid in execute mode")
    if not isinstance(rubric_ids, list) or not rubric_ids:
        raise AdaptiveEvidenceError("rubric_ids must be a non-empty array")
    normalized: list[int] = []
    for rubric_id in rubric_ids:
        if type(rubric_id) is not int or rubric_id not in ALLOWED_RUBRICS:
            raise AdaptiveEvidenceError("MVP adaptive evidence supports only R5 and R6")
        if rubric_id not in normalized:
            normalized.append(rubric_id)
    if reason not in ALLOWED_REASONS:
        raise AdaptiveEvidenceError(f"reason must be one of {sorted(ALLOWED_REASONS)}")
    interval = _finite_number(interval_seconds, "interval_seconds")
    if not MIN_INTERVAL_SECONDS <= interval <= MAX_INTERVAL_SECONDS:
        raise AdaptiveEvidenceError(
            f"interval_seconds must be between {MIN_INTERVAL_SECONDS} and {MAX_INTERVAL_SECONDS}"
        )
    if type(max_frames) is not int or not 1 <= max_frames <= MAX_FRAMES:
        raise AdaptiveEvidenceError(f"max_frames must be an integer from 1 to {MAX_FRAMES}")
    if roi_mode != "dynamic_meter_candidates":
        raise AdaptiveEvidenceError("fixed ROI is not allowed; use dynamic_meter_candidates")
    if view != "meter_pair":
        raise AdaptiveEvidenceError("MVP view must be meter_pair")

    adaptive_root = run_dir / "adaptive_evidence"
    adaptive_root.mkdir(parents=True, exist_ok=True)
    previous = sorted(adaptive_root.glob("request_*/request.json"))
    if len(previous) >= MAX_REQUEST_ROUNDS:
        raise AdaptiveEvidenceError(f"adaptive request limit is {MAX_REQUEST_ROUNDS} rounds")
    request_number = len(previous) + 1

    video = state.get("video") if isinstance(state.get("video"), dict) else {}
    video_path = Path(str(video.get("path") or "")).resolve()
    if not video_path.is_file():
        raise AdaptiveEvidenceError("current run video is missing")
    expected_hash = video.get("sha256")
    digest_builder = hashlib.sha256()
    with video_path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest_builder.update(block)
    digest = digest_builder.hexdigest()
    if digest != expected_hash:
        raise AdaptiveEvidenceError("current run source video changed")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise AdaptiveEvidenceError("unable to open current run video")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if fps <= 0.0 or frame_count <= 0:
        raise AdaptiveEvidenceError("current run video metadata is invalid")
    duration = frame_count / fps
    stages = [item for item in current_stage_runs(run_dir) if item["stage"] in METER_STAGE_NAMES]
    ranges = _request_ranges(time_ranges, stages, duration)
    frame_numbers = _sample_frame_numbers(ranges, fps, frame_count, interval)
    if len(frame_numbers) > max_frames:
        raise AdaptiveEvidenceError(
            f"request would produce {len(frame_numbers)} frames; increase interval or lower range"
        )
    known = _known_frame_numbers(run_dir)
    new_frame_numbers = [number for number in frame_numbers if number not in known]
    request_dir = adaptive_root / f"request_{request_number:02d}"
    frames_dir = request_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    request_record = {
        "schema_version": "resistance_agent_adaptive_request.v1",
        "request_number": request_number,
        "run_id": state.get("run_id"),
        "rubric_ids": normalized,
        "reason": reason,
        "time_ranges": ranges,
        "sampling": {
            "requested_interval_seconds": round(interval, 6),
            "max_frames": max_frames,
            "actual_frame_count": len(new_frame_numbers),
        },
        "roi_mode": roi_mode,
        "view": view,
        "source_video_sha256": digest,
        "stage_runs_used": stages,
    }
    request_dir.mkdir(parents=True, exist_ok=True)
    (request_dir / "request.json").write_text(
        json.dumps(request_record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    frames: list[dict[str, Any]] = []
    capture = cv2.VideoCapture(str(video_path))
    try:
        for frame_number in new_frame_numbers:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ok, image = capture.read()
            if not ok or image is None:
                continue
            actual = int(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            timestamp = actual / fps
            path = frames_dir / f"frame_{actual:08d}_{timestamp:010.3f}s.jpg"
            if not cv2.imwrite(str(path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 95]):
                continue
            frames.append(
                {
                    "frame_id": f"frame_{actual:08d}",
                    "image_group_id": f"adaptive_{request_number:02d}_{actual:08d}",
                    "frame_number": actual,
                    "timestamp_seconds": round(timestamp, 6),
                    "frame_path": str(path.resolve()),
                    "window_source": f"adaptive:{reason}",
                    "window_priority": 0,
                    "source_video_sha256": digest,
                    "width": int(image.shape[1]),
                    "height": int(image.shape[0]),
                }
            )
    finally:
        capture.release()
    if frames:
        try:
            try:
                from .meter_rubrics import _export_candidates, _select_frame_records
                from .skills import dynamic_meter_reading
            except ImportError:
                from meter_rubrics import _export_candidates, _select_frame_records  # type: ignore
                from skills import dynamic_meter_reading  # type: ignore

            analyzed = [_export_candidates(item, request_dir) for item in frames]
            identity = dynamic_meter_reading.prepare_frames(analyzed)
            selected = _select_frame_records(analyzed, limit=min(6, len(analyzed)))
        except (ImportError, OSError, RuntimeError, ValueError, cv2.error) as exc:
            raise AdaptiveEvidenceError(f"dynamic meter ROI failed: {type(exc).__name__}:{exc}") from exc
    else:
        identity = {"skill_version": "dynamic_meter_reading.v3", "tracks": []}
        selected = []

    result = {
        "schema_version": "resistance_agent_adaptive_evidence.v1",
        "status": "additional_evidence_ready" if selected else "no_new_frames",
        "request_number": request_number,
        "run_id": state.get("run_id"),
        "rubric_ids": normalized,
        "reason": reason,
        "sampling": request_record["sampling"],
        "roi_mode": roi_mode,
        "view": view,
        "source_video_sha256": digest,
        "deduplicated_frame_count": len(frame_numbers) - len(new_frame_numbers),
        "frame_count": len(frames),
        "selected_frame_count": len(selected),
        "frames": frames,
        "selected_frames": selected,
        "dynamic_meter_identity": identity,
        "next_tool": "run_rubric_bundle" if selected else None,
        "next_arguments": {"rubric_ids": [5, 6]} if selected else None,
        "historical_artifacts_used": False,
        "video_id_used_for_routing": False,
        "fixed_video_roi_used": False,
    }
    (request_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {**result, "request_path": str((request_dir / "request.json").resolve()), "result_path": str((request_dir / "result.json").resolve())}
