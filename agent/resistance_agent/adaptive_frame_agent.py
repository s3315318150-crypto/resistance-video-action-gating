"""Current-run adaptive frame extraction with dynamic meter verification.

This module is an evidence producer, not a rubric scorer. It samples the
observed stage windows, runs the existing dynamic OpenCV candidate detector,
asks Qwen whether the selected frame groups visibly contain the requested
instrument roles, and requests bounded adjacent frames when the view is weak.
It never routes by video identity or reads historical artifacts.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import cv2


STAGE_NAMES = {
    "circuit_wiring",
    "measurement_1",
    "recording_1",
    "circuit_rewiring",
    "measurement_2",
    "recording_2",
    "material_cleanup",
}
STAGE_METADATA_FIELDS = {
    "stage_semantics",
    "stage_window_semantics",
    "merged_stage_semantics",
    "merged_measurement_recording",
    "merged_stage",
    "cycle_index",
    "base_action_types",
    "observed_subintervals",
    "measurement_subintervals",
    "writing_subintervals",
    "contains_measurement_evidence",
    "contains_writing_evidence",
}
FRAME_AGENT_VERSION = "adaptive_frame_agent.v2"
MAX_ROUNDS = 2
MAX_FRAMES_PER_ROUND = 32
MAX_CV_FRAMES_PER_ROUND = 16
DEFAULT_INTERVAL_SECONDS = 0.5
ADJACENT_RADIUS_SECONDS = 1.25


class FrameAgentError(ValueError):
    """Raised when current-run frame extraction cannot satisfy its contract."""


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise FrameAgentError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise FrameAgentError(f"{field} must be finite")
    return float(value)


def _walk_stage_runs(value: Any, output: list[dict[str, Any]]) -> None:
    if isinstance(value, dict):
        stage = value.get("stage")
        if stage in STAGE_NAMES and "start_seconds" in value and "end_seconds" in value:
            try:
                start = _number(value.get("start_seconds"), "start_seconds")
                end = _number(value.get("end_seconds"), "end_seconds")
            except FrameAgentError:
                start = end = -1.0
            if 0.0 <= start < end:
                normalized = {
                    "stage": stage,
                    "start_seconds": round(start, 6),
                    "end_seconds": round(end, 6),
                }
                for field in STAGE_METADATA_FIELDS:
                    if field in value:
                        normalized[field] = value[field]
                output.append(normalized)
        for child in value.values():
            _walk_stage_runs(child, output)
    elif isinstance(value, list):
        for child in value:
            _walk_stage_runs(child, output)


def _stage_runs(run_dir: Path) -> list[dict[str, Any]]:
    """Read stage intervals only from JSON produced inside this current run."""
    found: dict[tuple[str, float, float], dict[str, Any]] = {}
    for path in sorted(run_dir.glob("*.json")):
        try:
            value = _read_json(path)
        except (OSError, ValueError, UnicodeError):
            continue
        records: list[dict[str, Any]] = []
        _walk_stage_runs(value, records)
        for normalized in records:
            found[(normalized["stage"], normalized["start_seconds"], normalized["end_seconds"])] = normalized
    return sorted(found.values(), key=lambda item: (item["start_seconds"], item["end_seconds"]))


def _video_info(state: dict[str, Any]) -> tuple[Path, float, int, str]:
    video = state.get("video") if isinstance(state.get("video"), dict) else {}
    path = Path(str(video.get("path") or "")).resolve()
    if not path.is_file():
        raise FrameAgentError("current run video is missing")
    digest = _sha256(path)
    if digest != video.get("sha256"):
        raise FrameAgentError("current run source video changed")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FrameAgentError("unable to open current run video")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if fps <= 0.0 or frame_count <= 0:
        raise FrameAgentError("current run video metadata is invalid")
    return path, fps, frame_count, digest


def _windows(stages: list[dict[str, Any]], duration: float) -> list[dict[str, Any]]:
    relevant = [item for item in stages if item["stage"] != "material_cleanup"]
    if not relevant:
        return [{"source": "broad_search", "start_seconds": 0.0, "end_seconds": round(duration, 6)}]
    windows: list[dict[str, Any]] = []

    def add(source: str, start: float, end: float, priority: int) -> None:
        start = max(0.0, float(start) - 0.5)
        end = min(duration, float(end) + 0.5)
        if end <= start:
            return
        key = (round(start, 6), round(end, 6))
        if any(
            item["start_seconds"] == key[0]
            and item["end_seconds"] == key[1]
            and item["source"] == source
            for item in windows
        ):
            return
        windows.append(
            {
                "source": source,
                "start_seconds": key[0],
                "end_seconds": key[1],
                "priority": priority,
            }
        )

    def subintervals(item: dict[str, Any], action_type: str) -> list[dict[str, Any]]:
        field = "measurement_subintervals" if action_type == "measurement_action" else "writing_subintervals"
        raw = item.get(field)
        if not isinstance(raw, list):
            raw = item.get("observed_subintervals")
        output: list[dict[str, Any]] = []
        for value in raw if isinstance(raw, list) else []:
            if not isinstance(value, dict):
                continue
            if value.get("action_type") not in {None, action_type} and field not in item:
                continue
            try:
                start = _number(value.get("start_seconds"), "subinterval.start_seconds")
                end = _number(value.get("end_seconds"), "subinterval.end_seconds")
            except FrameAgentError:
                continue
            if 0.0 <= start < end:
                output.append({"start_seconds": start, "end_seconds": end})
        return output

    for item in relevant:
        stage = str(item["stage"])
        merged = stage in {"recording_1", "recording_2"} and (
            item.get("stage_semantics") == "measurement_and_recording_cycle"
            or item.get("stage_window_semantics") == "measurement_and_recording_cycle"
            or item.get("merged_stage_semantics") == "measurement_and_recording_cycle"
            or item.get("merged_measurement_recording") is True
            or item.get("merged_stage") is True
        )
        measurements = subintervals(item, "measurement_action") if merged else []
        if measurements:
            for subinterval in measurements:
                add(
                    f"{stage}.measurement_action",
                    subinterval["start_seconds"],
                    subinterval["end_seconds"],
                    0,
                )
            add(stage, float(item["start_seconds"]), float(item["end_seconds"]), 1)
            continue
        priority = 0 if stage.startswith("measurement_") or merged else 1 if stage.startswith("recording_") else 2
        add(stage, float(item["start_seconds"]), float(item["end_seconds"]), priority)
    return sorted(windows, key=lambda item: (int(item["priority"]), item["start_seconds"], item["source"]))


def _sample_numbers(windows: list[dict[str, Any]], fps: float, frame_count: int, interval: float) -> list[int]:
    maximum = max(0, frame_count - 1)
    values: set[int] = set()
    for window in windows:
        start, end = float(window["start_seconds"]), float(window["end_seconds"])
        count = int(math.floor((end - start) / interval + 1e-9)) + 1
        for index in range(count):
            timestamp = min(end, start + index * interval)
            values.add(min(maximum, max(0, int(round(timestamp * fps)))))
        values.add(min(maximum, max(0, int(round(end * fps)))))
    return sorted(values)


def _limit_numbers(values: list[int], limit: int = MAX_FRAMES_PER_ROUND) -> list[int]:
    if len(values) <= limit:
        return values
    if limit == 1:
        return [values[len(values) // 2]]
    indices = {int(round(index * (len(values) - 1) / (limit - 1))) for index in range(limit)}
    return [values[index] for index in sorted(indices)]


def _known_frame_numbers(run_dir: Path) -> set[int]:
    numbers: set[int] = set()
    for path in run_dir.rglob("*.json"):
        if "frame_agent" in path.parts:
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError, UnicodeError):
            continue

        def visit(item: Any) -> None:
            if isinstance(item, dict):
                number = item.get("frame_number")
                if isinstance(number, int) and not isinstance(number, bool):
                    numbers.add(number)
                for child in item.values():
                    visit(child)
            elif isinstance(item, list):
                for child in item:
                    visit(child)

        visit(value)
    return numbers


def _decode_frames(
    video_path: Path,
    frame_numbers: list[int],
    fps: float,
    output_dir: Path,
    digest: str,
    source_by_number: dict[int, str],
    image_group_prefix: str,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FrameAgentError("unable to open current run video")
    frames: list[dict[str, Any]] = []
    try:
        for number in frame_numbers:
            capture.set(cv2.CAP_PROP_POS_FRAMES, number)
            ok, image = capture.read()
            if not ok or image is None:
                continue
            actual = int(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            timestamp = actual / fps
            path = output_dir / f"frame_{actual:08d}_{timestamp:010.3f}s.jpg"
            if not cv2.imwrite(str(path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 95]):
                continue
            frames.append(
                {
                    "frame_id": f"frame_{actual:08d}",
                    "image_group_id": f"{image_group_prefix}_{actual:08d}",
                    "frame_number": actual,
                    "timestamp_seconds": round(timestamp, 6),
                    "frame_path": str(path.resolve()),
                    "window_source": source_by_number.get(number, "broad_search"),
                    "source_video_sha256": digest,
                    "width": int(image.shape[1]),
                    "height": int(image.shape[0]),
                    "sharpness": round(float(cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()), 6),
                }
            )
    finally:
        capture.release()
    return frames


def _preselect_for_cv(frames: list[dict[str, Any]], limit: int = MAX_CV_FRAMES_PER_ROUND) -> list[dict[str, Any]]:
    if len(frames) <= limit:
        return frames
    selected: list[dict[str, Any]] = []
    sources = dict.fromkeys(str(item.get("window_source") or "broad_search") for item in frames)
    for source in sources:
        candidates = [item for item in frames if str(item.get("window_source") or "broad_search") == source]
        if candidates:
            selected.append(max(candidates, key=lambda item: float(item.get("sharpness") or 0.0)))
    ranked = sorted(frames, key=lambda item: float(item.get("sharpness") or 0.0), reverse=True)
    for frame in ranked:
        if frame in selected:
            continue
        timestamp = float(frame.get("timestamp_seconds") or 0.0)
        if all(abs(timestamp - float(existing.get("timestamp_seconds") or 0.0)) >= 0.4 for existing in selected):
            selected.append(frame)
        if len(selected) >= limit:
            break
    for frame in ranked:
        if frame not in selected:
            selected.append(frame)
        if len(selected) >= limit:
            break
    return sorted(selected[:limit], key=lambda item: float(item.get("timestamp_seconds") or 0.0))


def _qwen_roles(observation: dict[str, Any]) -> set[str]:
    roles: set[str] = set()
    for item in observation.get("observations") or []:
        if not isinstance(item, dict):
            continue
        role = item.get("identity")
        if role in {"ammeter", "voltmeter"} and float(item.get("confidence") or 0.0) >= 0.45:
            roles.add(str(role))
    return roles


def _run_round(
    *,
    run_dir: Path,
    video_path: Path,
    fps: float,
    frame_count: int,
    digest: str,
    model_config: dict[str, Any],
    windows: list[dict[str, Any]],
    round_number: int,
    interval: float,
    known: set[int],
) -> dict[str, Any]:
    all_requested = _sample_numbers(windows, fps, frame_count, interval)
    requested = _limit_numbers(all_requested)
    new_numbers = [number for number in requested if number not in known]
    round_dir = run_dir / "frame_agent" / f"round_{round_number:02d}"
    def window_priority(window: dict[str, Any]) -> tuple[int, float]:
        source = str(window.get("source") or "")
        if isinstance(window.get("priority"), int):
            priority = int(window["priority"])
        elif source.startswith("measurement_"):
            priority = 0
        elif source.startswith("recording_"):
            priority = 1
        elif "wiring" in source:
            priority = 2
        elif source == "adaptive_adjacent":
            priority = 0
        else:
            priority = 3
        return priority, float(window.get("start_seconds") or 0.0)

    source_by_number: dict[int, str] = {}
    for number in new_numbers:
        timestamp = number / fps
        tolerance = 0.5 / fps + 1e-6
        matching = [
            window
            for window in windows
            if float(window["start_seconds"]) - tolerance <= timestamp <= float(window["end_seconds"]) + tolerance
        ]
        selected_window = min(matching, key=window_priority) if matching else {"source": "broad_search"}
        source_by_number[number] = str(selected_window["source"])
    frames = _decode_frames(
        video_path,
        new_numbers,
        fps,
        round_dir / "frames",
        digest,
        source_by_number,
        f"round_{round_number:02d}",
    )
    known.update(item["frame_number"] for item in frames)
    try:
        from .meter_rubrics import _call_qwen, _export_candidates, _select_frame_records
        from .skills import dynamic_meter_reading
    except ImportError:
        from meter_rubrics import _call_qwen, _export_candidates, _select_frame_records  # type: ignore
        from skills import dynamic_meter_reading  # type: ignore

    cv_frames = _preselect_for_cv(frames)
    analyzed = [_export_candidates(item, round_dir) for item in cv_frames]
    identity = dynamic_meter_reading.prepare_frames(analyzed, render_overlays=False)
    selected = _select_frame_records(analyzed, limit=min(4, len(analyzed)))
    qwen_observation: dict[str, Any] | None = None
    qwen_error: str | None = None
    if selected:
        try:
            qwen_observation = _call_qwen(
                selected,
                model_config,
                round_dir / "qwen" / "frame_agent_raw.json",
                skill_instruction=(
                    "This is frame selection only. Decide whether the supplied groups visibly contain "
                    "a usable complete ammeter and/or voltmeter. Do not score any rubric. "
                    "Mark low visibility or an incomplete crop as uncertain and use only visible pixels."
                ),
                candidate_crops_per_frame=2,
                execution_fingerprint=FRAME_AGENT_VERSION,
            )
        except (OSError, RuntimeError, ValueError, KeyError) as exc:
            qwen_error = f"{type(exc).__name__}:{exc}"
    roles = sorted(_qwen_roles(qwen_observation or {}))
    useful = bool(selected) and bool(roles)
    needs_more = not {"ammeter", "voltmeter"}.issubset(roles)
    result = {
        "schema_version": "resistance_agent_frame_agent_round.v1",
        "agent_version": FRAME_AGENT_VERSION,
        "round_number": round_number,
        "windows": windows,
        "sampling": {
            "interval_seconds": interval,
            "candidate_frame_count_before_limit": len(all_requested),
            "requested_frame_count": len(requested),
            "new_frame_count": len(new_numbers),
            "decoded_frame_count": len(frames),
            "opencv_frame_count": len(cv_frames),
        },
        "frames": frames,
        "selected_frames": selected,
        "dynamic_meter_identity": identity,
        "qwen_observation": qwen_observation,
        "qwen_error": qwen_error,
        "visible_roles": roles,
        "frame_useful": useful,
        "meter_pair_complete": not needs_more,
        "needs_more_frames": needs_more,
        "historical_artifacts_used": False,
        "video_id_used_for_routing": False,
        "fixed_video_roi_used": False,
    }
    _write_json(round_dir / "result.json", result)
    return result


def run_adaptive_frame_agent(
    *,
    run_dir: Path,
    state: dict[str, Any],
    model_config: dict[str, Any],
    max_rounds: int = MAX_ROUNDS,
    initial_interval_seconds: float = DEFAULT_INTERVAL_SECONDS,
) -> dict[str, Any]:
    """Extract current-run frame groups and self-request adjacent evidence."""
    if state.get("mode") != "execute":
        raise FrameAgentError("frame agent is only valid in execute mode")
    if type(max_rounds) is not int or not 1 <= max_rounds <= MAX_ROUNDS:
        raise FrameAgentError(f"max_rounds must be an integer from 1 to {MAX_ROUNDS}")
    if not 0.1 <= float(initial_interval_seconds) <= 2.0:
        raise FrameAgentError("initial_interval_seconds must be between 0.1 and 2.0")
    video_path, fps, frame_count, digest = _video_info(state)
    duration = frame_count / fps
    stages = _stage_runs(run_dir)
    windows = _windows(stages, duration)
    known = _known_frame_numbers(run_dir)
    rounds: list[dict[str, Any]] = []
    interval = float(initial_interval_seconds)
    for round_number in range(1, max_rounds + 1):
        result = _run_round(
            run_dir=run_dir,
            video_path=video_path,
            fps=fps,
            frame_count=frame_count,
            digest=digest,
            model_config=model_config,
            windows=windows,
            round_number=round_number,
            interval=interval,
            known=known,
        )
        rounds.append(result)
        if not result["needs_more_frames"] or round_number >= max_rounds:
            break
        anchors = [
            float(item["timestamp_seconds"])
            for item in result.get("selected_frames") or []
            if isinstance(item, dict) and isinstance(item.get("timestamp_seconds"), (int, float))
        ]
        if not anchors:
            break
        windows = [
            {
                "source": "adaptive_adjacent",
                "start_seconds": round(max(0.0, timestamp - ADJACENT_RADIUS_SECONDS), 6),
                "end_seconds": round(min(duration, timestamp + ADJACENT_RADIUS_SECONDS), 6),
            }
            for timestamp in anchors[:3]
        ]
        interval = max(0.1, interval / 2.0)

    final_round = rounds[-1] if rounds else {}
    request_limit_reached = bool(final_round.get("needs_more_frames")) and len(rounds) >= max_rounds
    pair_complete = bool(final_round.get("meter_pair_complete"))
    report = {
        "schema_version": "resistance_agent_frame_agent.v1",
        "agent_version": FRAME_AGENT_VERSION,
        "status": "frame_evidence_ready" if pair_complete else "frame_evidence_partial",
        "run_id": state.get("run_id"),
        "source_video_sha256": digest,
        "observed_stage_runs": stages,
        "round_count": len(rounds),
        "rounds": rounds,
        "selected_frames": final_round.get("selected_frames") or [],
        "visible_roles": final_round.get("visible_roles") or [],
        "frame_useful": bool(final_round.get("frame_useful")),
        "meter_pair_complete": pair_complete,
        "needs_more_frames": bool(final_round.get("needs_more_frames")),
        "request_limit_reached": request_limit_reached,
        "next_tool": "run_rubric_bundle" if final_round.get("frame_useful") else None,
        "selection_basis": "current_video_observed_stage_and_current_frame_visual_evidence_only",
        "video_id_used_for_routing": False,
        "historical_artifacts_used": False,
        "fixed_video_roi_used": False,
    }
    report_path = run_dir / "frame_agent" / "frame_agent_report.json"
    _write_json(report_path, report)
    return {**report, "report_path": str(report_path.resolve())}
