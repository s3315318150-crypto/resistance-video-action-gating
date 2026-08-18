#!/usr/bin/env python3
"""Real-video evidence acquisition and binary reduction for Rubrics 5 and 6."""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from .skills import dynamic_meter_reading as DYNAMIC_METER
    from .skills import closed_stable_r6_cv_v3 as CLOSED_STABLE_R6
    from .skills import closed_stable_stage_producer as CLOSED_STABLE_PRODUCER
except ImportError:
    from skills import dynamic_meter_reading as DYNAMIC_METER  # type: ignore
    from skills import closed_stable_r6_cv_v3 as CLOSED_STABLE_R6  # type: ignore
    from skills import closed_stable_stage_producer as CLOSED_STABLE_PRODUCER  # type: ignore


AGENT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = AGENT_ROOT.parent
SCRIPTS_ROOT = AGENT_ROOT / "scripts"
ALGORITHM_VERSION = "r56_temporal_meter_v5_live_closed_stable_cv_v3"
NEEDLE_STATES = {"normal_rightward", "zero", "reverse", "overrange", "uncertain"}
ENERGIZED_STATES = {"energized", "deenergized", "unclear"}
METER_IDENTITIES = {"ammeter", "voltmeter", "unknown"}
TERMINAL_VISIBILITY = {"connected", "not_connected", "uncertain"}
RANGE_CLASSES = {"appropriate", "too_low", "too_high", "unknown"}
POINTER_SCALE_POSITIONS = {"near_zero", "low", "mid", "high", "near_full", "uncertain"}
TERMINAL_OCCUPANCY = {"occupied", "empty", "uncertain"}


def _load_script(name: str, filename: str) -> Any:
    path = SCRIPTS_ROOT / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load script module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


METER_V4 = _load_script("resistance_agent_meter_v4", "detect_colored_meters_v4.py")
POINTER_CV = _load_script("resistance_agent_pointer_cv", "measure_pointer_angle_opencv.py")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_record(summary: dict[str, Any], source_video_id: str, video_id: str) -> dict[str, Any]:
    records = summary.get("records")
    if not isinstance(records, list):
        raise ValueError("action summary records are missing")
    for record in records:
        if not isinstance(record, dict):
            continue
        source = str(record.get("source_video_id") or "")
        if source == source_video_id or source.startswith(f"{video_id}_"):
            for key in ("replay_result", "result_path"):
                nested_path = record.get(key)
                if isinstance(nested_path, str) and Path(nested_path).is_file():
                    nested = read_json(Path(nested_path))
                    if isinstance(nested.get("observed_stage_runs"), list):
                        return nested
            return record
    raise ValueError(f"Temporal Guard record not found for video {video_id}")


def _boundary_record(summary: dict[str, Any], source_video_id: str, video_id: str) -> dict[str, Any] | None:
    records = summary.get("records")
    if not isinstance(records, list):
        return None
    for record in records:
        if not isinstance(record, dict):
            continue
        source = str(record.get("source_video_id") or "")
        if source != source_video_id and not source.startswith(f"{video_id}_"):
            continue
        nested_path = record.get("result_path")
        if isinstance(nested_path, str) and Path(nested_path).is_file():
            nested = read_json(Path(nested_path))
            runs = nested.get("source_observed_stage_runs")
            if isinstance(runs, list):
                return {"observed_stage_runs": runs}
        runs = record.get("source_observed_stage_runs")
        if isinstance(runs, list):
            return {"observed_stage_runs": runs}
    return None


def candidate_windows(
    record: dict[str, Any],
    duration_seconds: float,
    window_mode: str = "measurement_first",
) -> list[dict[str, Any]]:
    raw_runs = record.get("observed_stage_runs")
    runs = [item for item in raw_runs if isinstance(item, dict)] if isinstance(raw_runs, list) else []
    measurement = [item for item in runs if str(item.get("stage") or "").startswith("measurement_")]
    if window_mode == "pre_recording_recovery":
        measurement = []
    recording = [
        item for item in runs
        if str(item.get("stage")) in {"recording_1", "recording_2"}
    ]
    wiring = [item for item in runs if str(item.get("stage")) == "circuit_wiring"]
    windows: list[dict[str, Any]] = []

    def add(
        start: float,
        end: float,
        source: str,
        priority: int,
        cycle_index: int | None = None,
    ) -> None:
        start = max(0.0, min(float(start), duration_seconds))
        end = max(start, min(float(end), duration_seconds))
        if end - start < 0.5:
            return
        key = (round(start, 2), round(end, 2))
        if any((item["start_seconds"], item["end_seconds"]) == key for item in windows):
            return
        windows.append(
            {
                "start_seconds": key[0],
                "end_seconds": key[1],
                "source": source,
                "priority": priority,
                "cycle_index": cycle_index,
            }
        )

    def is_merged(item: dict[str, Any]) -> bool:
        return str(item.get("stage")) in {"recording_1", "recording_2"} and (
            item.get("stage_semantics") == "measurement_and_recording_cycle"
            or item.get("stage_window_semantics") == "measurement_and_recording_cycle"
            or item.get("merged_stage_semantics") == "measurement_and_recording_cycle"
            or item.get("merged_measurement_recording") is True
            or item.get("merged_stage") is True
        )

    def measurement_subintervals(item: dict[str, Any]) -> list[dict[str, Any]]:
        raw = item.get("measurement_subintervals")
        explicit_field = isinstance(raw, list)
        if not explicit_field:
            raw = item.get("observed_subintervals")
        output: list[dict[str, Any]] = []
        for subinterval in raw if isinstance(raw, list) else []:
            if not isinstance(subinterval, dict):
                continue
            if not explicit_field and subinterval.get("action_type") != "measurement_action":
                continue
            try:
                start = float(subinterval["start_seconds"])
                end = float(subinterval["end_seconds"])
            except (KeyError, TypeError, ValueError):
                continue
            if start < end:
                output.append({"start_seconds": start, "end_seconds": end})
        return output

    for item in measurement:
        cycle = int(str(item.get("stage") or "measurement_1").rsplit("_", 1)[1])
        add(
            float(item["start_seconds"]) - 2.0,
            float(item["end_seconds"]) + 2.0,
            str(item.get("stage") or "measurement"),
            0,
            cycle,
        )
    if window_mode != "pre_recording_recovery":
        for item in recording:
            if not is_merged(item):
                continue
            stage = str(item["stage"])
            cycle = int(stage.rsplit("_", 1)[1])
            subintervals = measurement_subintervals(item)
            if subintervals:
                for index, subinterval in enumerate(subintervals, start=1):
                    add(
                        subinterval["start_seconds"] - 1.0,
                        subinterval["end_seconds"] + 1.0,
                        f"{stage}_measurement_subinterval_{index}",
                        0,
                        cycle,
                    )
            else:
                add(
                    float(item["start_seconds"]),
                    float(item["end_seconds"]),
                    f"{stage}_merged_measurement_fallback",
                    0,
                    cycle,
                )
    for item in recording:
        if window_mode != "pre_recording_recovery" and is_merged(item):
            continue
        stage = str(item["stage"])
        cycle = int(stage.rsplit("_", 1)[1])
        rec_start = float(item["start_seconds"])
        rec_end = float(item["end_seconds"])
        add(
            rec_start - 15.0,
            min(rec_end, rec_start + 24.0),
            f"{stage}_measurement_neighborhood",
            1,
            cycle,
        )
        add(
            rec_start - 60.0,
            min(rec_end, rec_start + 30.0),
            f"{stage}_broad_recovery",
            2,
            cycle,
        )
    if wiring and recording:
        last = max((item for item in wiring if float(item["end_seconds"]) <= float(recording[0]["start_seconds"]) + 2.0),
                   key=lambda item: float(item["end_seconds"]), default=max(wiring, key=lambda item: float(item["end_seconds"])))
        rec_start = float(min(recording, key=lambda item: float(item["start_seconds"]))["start_seconds"])
        add(max(float(last["start_seconds"]), float(last["end_seconds"]) - 12.0), rec_start + 2.0, "wiring_to_recording_transition", 2)
    if not windows and wiring:
        last = max(wiring, key=lambda item: float(item["end_seconds"]))
        add(float(last["end_seconds"]) - 12.0, float(last["end_seconds"]) + 4.0, "wiring_tail_fallback", 3)
    if not windows:
        add(max(0.0, duration_seconds * 0.45), min(duration_seconds, duration_seconds * 0.70), "duration_fallback", 4)
    return sorted(windows, key=lambda item: (int(item["priority"]), item["start_seconds"]))


def sampling_timestamps(windows: list[dict[str, Any]], max_samples: int = 28) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    seen: set[float] = set()
    queues: list[tuple[int, dict[str, Any], list[float]]] = []
    for window_index, window in enumerate(windows, start=1):
        start, end = float(window["start_seconds"]), float(window["end_seconds"])
        priority = int(window["priority"])
        step = 1.5 if priority == 0 else 3.0 if priority == 1 else 5.0
        count = max(2, int(math.ceil((end - start) / step)) + 1)
        queues.append((window_index, window, [round(float(value), 3) for value in np.linspace(start, end, count)]))
    for priority in sorted({int(window["priority"]) for window in windows}):
        active = [item for item in queues if int(item[1]["priority"]) == priority]
        offset = 0
        while any(offset < len(values) for _, _, values in active):
            for window_index, window, values in active:
                if offset >= len(values):
                    continue
                timestamp = values[offset]
                if timestamp in seen:
                    continue
                seen.add(timestamp)
                points.append(
                    {
                        "timestamp_seconds": timestamp,
                        "window_index": window_index,
                        "window_source": window["source"],
                        "window_priority": window["priority"],
                        "cycle_index": window.get("cycle_index"),
                    }
                )
                if len(points) >= max_samples:
                    return points
            offset += 1
    return points


def _decode_frames(video_path: Path, samples: list[dict[str, Any]], frames_dir: Path) -> list[dict[str, Any]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"unable to open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frame = max(0, frame_count - 1)
    records: list[dict[str, Any]] = []
    frames_dir.mkdir(parents=True, exist_ok=True)
    try:
        for item in samples:
            requested = min(max_frame, max(0, int(round(float(item["timestamp_seconds"]) * fps))))
            capture.set(cv2.CAP_PROP_POS_FRAMES, requested)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            frame_number = int(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            timestamp = frame_number / fps if fps > 0 else float(item["timestamp_seconds"])
            path = frames_dir / f"frame_{frame_number:08d}_{timestamp:010.3f}s.jpg"
            if not cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95]):
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            records.append(
                {
                    **item,
                    "timestamp_seconds": round(timestamp, 6),
                    "frame_number": frame_number,
                    "frame_path": str(path.resolve()),
                    "width": int(frame.shape[1]),
                    "height": int(frame.shape[0]),
                    "sharpness": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 6),
                }
            )
    finally:
        capture.release()
    if not records:
        raise RuntimeError("no candidate frames could be decoded")
    return records


def _box_crop(image: np.ndarray, box: list[Any], padding: float = 0.0) -> np.ndarray | None:
    if not isinstance(box, list) or len(box) != 4:
        return None
    x, y, width, height = (int(value) for value in box)
    pad_x, pad_y = int(round(width * padding)), int(round(height * padding))
    image_height, image_width = image.shape[:2]
    x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
    x1, y1 = min(image_width, x + width + pad_x), min(image_height, y + height + pad_y)
    return image[y0:y1, x0:x1] if x1 > x0 and y1 > y0 else None


def _enhance(image: np.ndarray) -> np.ndarray:
    scale = min(3.0, max(1.0, 900.0 / max(image.shape[:2])))
    enlarged = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    lab = cv2.cvtColor(enlarged, cv2.COLOR_BGR2LAB)
    light, a, b = cv2.split(lab)
    light = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(light)
    enhanced = cv2.cvtColor(cv2.merge((light, a, b)), cv2.COLOR_LAB2BGR)
    blurred = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
    return cv2.addWeighted(enhanced, 1.45, blurred, -0.45, 0)


def _candidate_quality(candidate: dict[str, Any], frame: dict[str, Any]) -> float:
    face = candidate.get("face") or {}
    pointer = candidate.get("pointer") or {}
    return (
        0.45 * float(candidate.get("score") or 0.0)
        + 0.25 * float(face.get("dial_likeness") or 0.0)
        + 0.15 * float(face.get("structure_score") or 0.0)
        + 0.10 * float(pointer.get("confidence") or 0.0)
        + 0.05 * min(1.0, float(frame.get("sharpness") or 0.0) / 250.0)
        - 0.03 * float(frame.get("window_priority") or 0.0)
    )


def _export_candidates(frame: dict[str, Any], evidence_dir: Path) -> dict[str, Any]:
    frame_path = Path(frame["frame_path"])
    cv_dir = evidence_dir / "opencv"
    original = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if original is None:
        return {**frame, "detection": {"valid": False, "errors": ["frame_decode_failed"]}, "candidates": []}
    detection_path = cv_dir / "detection_frames" / frame_path.name
    detection_path.parent.mkdir(parents=True, exist_ok=True)
    scale = min(1.0, 1600.0 / float(original.shape[1]))
    detection_image = (
        original
        if scale == 1.0
        else cv2.resize(original, (int(round(original.shape[1] * scale)), int(round(original.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    )
    cv2.imwrite(str(detection_path), detection_image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    detection = METER_V4.analyze_image(
        detection_path,
        debug_path=cv_dir / "debug" / f"{frame_path.stem}_v4.jpg",
        roi_dir=cv_dir / "roi" / frame_path.stem,
    )
    detection = {
        **detection,
        "source_image_width": int(original.shape[1]),
        "source_image_height": int(original.shape[0]),
        "candidate_coordinate_space": "source_image_pixels",
    }
    image = original
    candidates: list[dict[str, Any]] = []
    if image is not None:
        wide_dir = evidence_dir / "meter_rois" / frame_path.stem
        wide_dir.mkdir(parents=True, exist_ok=True)
        for index, raw in enumerate(detection.get("candidates") or [], start=1):
            if not isinstance(raw, dict):
                continue
            def source_box(box: list[Any]) -> list[int]:
                return [int(round(float(value) / scale)) for value in box]

            face_box = source_box(list((raw.get("face") or {}).get("bbox") or []))
            wide_box = source_box(list(raw.get("bbox") or []))
            raw_terminal = raw.get("terminal_anchor") or {}
            terminal_box = source_box(list(raw_terminal.get("bbox") or []))
            face = _box_crop(image, face_box, padding=0.08)
            wide = _box_crop(image, wide_box, padding=0.22)
            terminal = _box_crop(image, terminal_box, padding=0.30)
            if face is None or wide is None:
                continue
            candidate_id = f"candidate_{index:02d}"
            face_path = wide_dir / f"{candidate_id}_face.jpg"
            wide_path = wide_dir / f"{candidate_id}_wide.jpg"
            enhanced_path = wide_dir / f"{candidate_id}_enhanced.jpg"
            terminal_path = wide_dir / f"{candidate_id}_terminal.jpg"
            cv2.imwrite(str(face_path), face, [int(cv2.IMWRITE_JPEG_QUALITY), 96])
            cv2.imwrite(str(wide_path), wide, [int(cv2.IMWRITE_JPEG_QUALITY), 96])
            cv2.imwrite(str(enhanced_path), _enhance(wide), [int(cv2.IMWRITE_JPEG_QUALITY), 94])
            if terminal is not None:
                cv2.imwrite(str(terminal_path), _enhance(terminal), [int(cv2.IMWRITE_JPEG_QUALITY), 96])
            candidate = {
                "candidate_id": candidate_id,
                "role_hint": raw.get("role"),
                "detector_source": raw.get("detector_source"),
                "score": raw.get("score"),
                "bbox": wide_box,
                "face_bbox": face_box,
                "terminal_anchor": {**raw_terminal, "bbox": terminal_box},
                "detector_pointer": raw.get("pointer"),
                "pointer_measurement": {
                    "valid": bool((raw.get("pointer") or {}).get("detected")),
                    "pointer": raw.get("pointer"),
                    "classification": {
                        "needle_state": "uncertain",
                        "reason": "zero_or_full_scale_calibration_missing",
                    },
                },
                "face_path": str(face_path.resolve()),
                "wide_path": str(wide_path.resolve()),
                "enhanced_path": str(enhanced_path.resolve()),
                "terminal_path": str(terminal_path.resolve()) if terminal is not None else None,
                "quality": 0.0,
                "opencv_diagnostics": {
                    "candidate_reason": raw.get("candidate_reason"),
                    "evidence_insufficient_reason": raw.get("evidence_insufficient_reason"),
                    "face": raw.get("face"),
                },
            }
            candidate["quality"] = round(_candidate_quality(raw, frame), 6)
            candidates.append(candidate)
    return {**frame, "detection": detection, "candidates": candidates}


def _frame_selection_score(item: dict[str, Any]) -> float:
    candidates = item.get("model_candidates") or item.get("candidates") or []
    selection = item.get("candidate_selection") if isinstance(item.get("candidate_selection"), dict) else {}
    pointer_support = sum(bool((candidate.get("detector_pointer") or {}).get("detected")) for candidate in candidates)
    return (
        max((float(candidate.get("selection_score") or candidate.get("quality") or 0.0) for candidate in candidates), default=0.0)
        + (0.16 if selection.get("status") == "two_distinct_faces" else 0.0)
        + min(0.08, 0.02 * pointer_support)
        - 0.025 * int(item.get("window_priority") or 0)
    )


def _select_frame_records(records: list[dict[str, Any]], limit: int = 4) -> list[dict[str, Any]]:
    ranked = sorted(
        records,
        key=lambda item: (
            -_frame_selection_score(item),
            int(item.get("window_priority") or 0),
            -float(item.get("sharpness") or 0.0),
        ),
    )
    selected: list[dict[str, Any]] = []
    # Explicit measurement stages are strongest. Broad recovery windows need
    # temporal coverage because a clear de-energized frame often outranks the
    # shorter energized measurement moment on pure image quality.
    sources = dict.fromkeys(str(item.get("window_source") or "") for item in records)
    for source in sources:
        source_items = [item for item in records if item.get("window_source") == source and item["candidates"]]
        if not source_items:
            continue
        if source.startswith("measurement_"):
            anchors = [max(source_items, key=_frame_selection_score)]
        elif source == "recording_1_broad_recovery":
            ordered = sorted(source_items, key=lambda item: float(item["timestamp_seconds"]))
            anchor_indices = {0, len(ordered) // 3, (2 * len(ordered)) // 3, len(ordered) - 1}
            anchors = [ordered[index] for index in sorted(anchor_indices)]
        else:
            anchors = [max(source_items, key=_frame_selection_score)]
        for best in anchors:
            if best not in selected:
                selected.append(best)
            if len(selected) >= limit:
                return selected
    for item in ranked:
        if item in selected:
            continue
        if item["candidates"] and all(
            abs(float(item["timestamp_seconds"]) - float(existing["timestamp_seconds"])) >= 2.0
            for existing in selected
        ):
            selected.append(item)
        if len(selected) >= limit:
            break
    return selected or ranked[:limit]


def _load_adaptive_selected(run_dir: Path, source_digest: str) -> list[dict[str, Any]]:
    """Load only current-run adaptive frames matching the current video hash."""
    root = run_dir / "adaptive_evidence"
    if not root.is_dir():
        return []
    frames: list[dict[str, Any]] = []
    for result_path in sorted(root.glob("request_*/result.json")):
        try:
            result = read_json(result_path)
        except (OSError, ValueError):
            continue
        if result.get("source_video_sha256") != source_digest:
            continue
        selected = result.get("selected_frames")
        if not isinstance(selected, list):
            continue
        for item in selected:
            if (
                isinstance(item, dict)
                and item.get("source_video_sha256") == source_digest
                and isinstance(item.get("frame_path"), str)
                and Path(item["frame_path"]).is_file()
            ):
                frames.append(item)
    unique: dict[int, dict[str, Any]] = {}
    for item in frames:
        frame_number = item.get("frame_number")
        if isinstance(frame_number, int):
            unique[frame_number] = item
    return [unique[key] for key in sorted(unique)]


def _add_selected_pointer_diagnostics(frames: list[dict[str, Any]]) -> None:
    for frame in frames:
        for candidate in sorted(
            frame.get("model_candidates") or frame.get("candidates") or [],
            key=lambda item: float(item.get("selection_score") or item.get("quality") or 0.0),
            reverse=True,
        )[:2]:
            try:
                candidate["pointer_measurement"] = POINTER_CV.measure_image(
                    Path(candidate["face_path"]),
                    calibration=None,
                    debug_path=Path(candidate["face_path"]).with_name(
                        Path(candidate["face_path"]).stem + "_pointer_debug.jpg"
                    ),
                )
            except Exception as exc:
                candidate["pointer_measurement"] = {
                    "valid": False,
                    "errors": [f"pointer_cv_failed:{type(exc).__name__}"],
                }


def image_data_url(path: Path, max_edge: int = 1600, jpeg_quality: int = 88) -> str:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"unable to decode model image: {path}")
    height, width = image.shape[:2]
    scale = min(1.0, float(max_edge) / max(height, width))
    if scale < 1.0:
        image = cv2.resize(
            image,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    ok, payload = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    if not ok:
        raise ValueError(f"unable to encode model image: {path}")
    encoded = base64.b64encode(payload.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def parse_json_object(content: str) -> dict[str, Any]:
    text = content.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("Qwen response is not a JSON object")
    return value


def _prompt(skill_instruction: str = "") -> str:
    return """You are observing analog ammeter and voltmeter evidence from one short measurement window in a middle-school resistance experiment.

Skill instruction: """ + skill_instruction + """

Images are grouped by time. Each group begins with one original panorama. Every selected physical meter candidate then contributes a face crop, terminal crop when visible, and an enhanced wide crop from that same frame. Spatially overlapping A/V hypotheses have already been merged, so one physical face appears only once. Candidate role hints are not guaranteed. Use the visible A/V marking, terminal color/layout and panorama context to establish identity. Use only visible pixels. Do not infer from student identity, expected workflow, hidden wires, or ground truth.

For each visible meter observation:
- circuit_state is energized, deenergized, or unclear for that image group. Use deenergized when the switch is visibly open, a probe is visibly disconnected, or the circuit is visibly incomplete. A zero pointer in a deenergized group is expected and must not be treated as an abnormal measurement.
- identity is ammeter, voltmeter, or unknown.
- pointer_state is normal_rightward, zero, reverse, overrange, or uncertain.
- normal_rightward means visibly rightward and inside the usable scale, not at either stop.
- pointer_scale_position is near_zero, low, mid, high, near_full, or uncertain. Judge the physical needle position on the shared printed arc, independently of which numeric scale is selected: near_zero is the leftmost 0-10%, low is 10-30%, mid is 30-70%, high is 70-90%, and near_full is the rightmost 90-100%.
- terminal_occupancy_left_middle_right contains exactly three values: occupied, empty, or uncertain. Establish the meter's upright front-facing orientation from the panorama before applying left/middle/right.
- For an upright ammeter, left is common, middle is 0.6 A, and right is 3 A. For an upright voltmeter, left is common, middle is 3 V, and right is 15 V. The selected range is the occupied positive terminal, not the largest printed label. If the right terminal is visibly empty, do not select the high range.
- selected_range_label copies only a visibly legible terminal/range marking such as 0.6, 3, 3V, or 15V; otherwise null.
- plugged_terminal_visible is connected, not_connected, or uncertain.
- range_assessment is appropriate, too_low, too_high, or unknown. Use too_low only with a near_full/overrange needle on a visibly connected small range. Use too_high only with a near_zero/low needle on a visibly connected large range. A mid/high in-scale needle is not too_high.

The response is observation-only. Do not output pass/fail or a score. Return exactly one JSON object and no Markdown:
{
  "measurement_active": true,
  "observations": [
    {
      "image_group": 1,
      "circuit_state": "energized",
      "identity": "ammeter",
      "pointer_state": "normal_rightward",
      "pointer_scale_position": "mid",
      "terminal_occupancy_left_middle_right": ["occupied", "occupied", "empty"],
      "selected_range_label": "0.6",
      "plugged_terminal_visible": "connected",
      "range_assessment": "appropriate",
      "confidence": 0.0,
      "evidence": "visible-only evidence"
    }
  ],
  "overall_confidence": 0.0,
  "evidence": "brief visible-only summary"
}"""


def validate_observation(value: dict[str, Any], group_count: int) -> list[str]:
    errors: list[str] = []
    if set(value) != {"measurement_active", "observations", "overall_confidence", "evidence"}:
        errors.append("response_fields_not_exact")
    if not isinstance(value.get("measurement_active"), bool):
        errors.append("measurement_active_invalid")
    confidence = value.get("overall_confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0.0 <= float(confidence) <= 1.0:
        errors.append("overall_confidence_invalid")
    if not isinstance(value.get("evidence"), str):
        errors.append("overall_evidence_invalid")
    observations = value.get("observations")
    if not isinstance(observations, list):
        return sorted(set(errors + ["observations_invalid"]))
    required = {
        "image_group",
        "circuit_state",
        "identity",
        "pointer_state",
        "pointer_scale_position",
        "terminal_occupancy_left_middle_right",
        "selected_range_label",
        "plugged_terminal_visible",
        "range_assessment",
        "confidence",
        "evidence",
    }
    for item in observations:
        if not isinstance(item, dict) or set(item) != required:
            errors.append("observation_fields_invalid")
            continue
        if not isinstance(item.get("image_group"), int) or not 1 <= item["image_group"] <= group_count:
            errors.append("image_group_invalid")
        if item.get("circuit_state") not in ENERGIZED_STATES:
            errors.append("circuit_state_invalid")
        if item.get("identity") not in METER_IDENTITIES:
            errors.append("identity_invalid")
        if item.get("pointer_state") not in NEEDLE_STATES:
            errors.append("pointer_state_invalid")
        if item.get("pointer_scale_position") not in POINTER_SCALE_POSITIONS:
            errors.append("pointer_scale_position_invalid")
        occupancy = item.get("terminal_occupancy_left_middle_right")
        if (
            not isinstance(occupancy, list)
            or len(occupancy) != 3
            or any(value not in TERMINAL_OCCUPANCY for value in occupancy)
        ):
            errors.append("terminal_occupancy_invalid")
        if item.get("selected_range_label") is not None and not isinstance(item.get("selected_range_label"), str):
            errors.append("selected_range_label_invalid")
        if item.get("plugged_terminal_visible") not in TERMINAL_VISIBILITY:
            errors.append("plugged_terminal_visible_invalid")
        if item.get("range_assessment") not in RANGE_CLASSES:
            errors.append("range_assessment_invalid")
        item_confidence = item.get("confidence")
        if isinstance(item_confidence, bool) or not isinstance(item_confidence, (int, float)) or not 0.0 <= float(item_confidence) <= 1.0:
            errors.append("observation_confidence_invalid")
        if not isinstance(item.get("evidence"), str):
            errors.append("observation_evidence_invalid")
    return sorted(set(errors))


def _call_qwen(
    frames: list[dict[str, Any]],
    model_config: dict[str, Any],
    raw_path: Path,
    skill_instruction: str = "",
    candidate_crops_per_frame: int = 2,
    execution_fingerprint: str | None = None,
) -> dict[str, Any]:
    base_url = os.getenv("QWEN_API_BASE_URL", str(model_config["base_url"]))
    token = os.getenv("QWEN_API_TOKEN", "EMPTY")
    model = os.getenv("QWEN_MODEL", str(model_config["model"]))
    content: list[dict[str, Any]] = [{"type": "text", "text": _prompt(skill_instruction)}]
    media_manifest: list[dict[str, Any]] = []
    for group_index, frame in enumerate(frames, start=1):
        panorama = Path(frame["frame_path"])
        content.append({"type": "text", "text": f"Image group {group_index}: panorama."})
        content.append({"type": "image_url", "image_url": {"url": image_data_url(panorama)}})
        media_manifest.append(
            {"group": group_index, "role": "panorama", "path": str(panorama), "model_max_edge": 1600}
        )
        candidates = (frame.get("model_candidates") or [])[:candidate_crops_per_frame]
        for candidate_index, candidate in enumerate(candidates, start=1):
            paths = [
                path
                for key in ("face_path", "terminal_path", "enhanced_path")
                if isinstance(candidate.get(key), str)
                for path in [Path(candidate[key])]
                if path.is_file()
            ]
            track_hint = candidate.get("track_identity_hint") or candidate.get("identity_hint") or "unknown"
            for view_index, crop in enumerate(paths, start=1):
                view_name = ("face", "terminal", "enhanced_wide")[view_index - 1] if len(paths) == 3 else crop.stem.rsplit("_", 1)[-1]
                content.append(
                    {
                        "type": "text",
                        "text": (
                            f"Image group {group_index}: physical candidate {candidate_index} "
                            f"({candidate.get('candidate_id')}); view={view_name}; "
                            f"temporal identity hint={track_hint}, verify from pixels."
                        ),
                    }
                )
                content.append({"type": "image_url", "image_url": {"url": image_data_url(crop)}})
                media_manifest.append(
                    {
                        "group": group_index,
                        "frame_id": frame.get("frame_id") or f"frame_{int(frame.get('frame_number') or 0):08d}",
                        "candidate_id": candidate.get("candidate_id"),
                        "role": view_name,
                        "identity_hint": track_hint,
                        "path": str(crop),
                    }
                )
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 2400,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
        method="POST",
    )
    attempts: list[dict[str, Any]] = []
    for attempt in range(2):
        if attempt:
            payload["messages"][0]["content"].append(
                {
                    "type": "text",
                    "text": "Schema correction: return exactly the requested keys and enum strings as one JSON object.",
                }
            )
            request = urllib.request.Request(
                base_url.rstrip("/") + "/chat/completions",
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
                method="POST",
            )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            attempts.append(
                {
                    "attempt": attempt + 1,
                    "errors": [f"transport_error:{type(exc).__name__}:{getattr(exc, 'code', '')}"],
                }
            )
            if attempt == 0:
                time.sleep(2.0)
                continue
            write_json(
                raw_path,
                {
                    "algorithm_version": ALGORITHM_VERSION,
                    "execution_fingerprint": execution_fingerprint,
                    "model": model,
                    "base_url": base_url,
                    "attempts": attempts,
                    "media": media_manifest,
                },
            )
            raise RuntimeError(f"Qwen request failed after one retry: {type(exc).__name__}") from exc
        choices = raw.get("choices") if isinstance(raw, dict) else None
        if not isinstance(choices, list) or not choices:
            attempts.append({"attempt": attempt + 1, "errors": ["response_has_no_choices"]})
            continue
        choice = choices[0]
        message = choice.get("message") if isinstance(choice, dict) else None
        text = message.get("content") if isinstance(message, dict) else None
        if not isinstance(text, str):
            attempts.append({"attempt": attempt + 1, "errors": ["response_content_not_text"]})
            continue
        try:
            parsed = parse_json_object(text)
            errors = validate_observation(parsed, len(frames))
        except (ValueError, json.JSONDecodeError) as exc:
            parsed = None
            errors = [f"parse_error:{type(exc).__name__}"]
        attempts.append(
            {
                "attempt": attempt + 1,
                "finish_reason": choice.get("finish_reason"),
                "content": text,
                "parsed": parsed if not errors else None,
                "schema_errors": errors,
            }
        )
        if not errors and parsed is not None:
            write_json(
                raw_path,
                {
                    "algorithm_version": ALGORITHM_VERSION,
                    "execution_fingerprint": execution_fingerprint,
                    "model": model,
                    "base_url": base_url,
                    "attempts": attempts,
                    "media": media_manifest,
                },
            )
            return parsed
    write_json(
        raw_path,
        {
            "algorithm_version": ALGORITHM_VERSION,
            "execution_fingerprint": execution_fingerprint,
            "model": model,
            "base_url": base_url,
            "attempts": attempts,
            "media": media_manifest,
        },
    )
    raise ValueError("Qwen schema invalid after one correction")


def _weighted_state(observations: list[dict[str, Any]], identity: str) -> tuple[str, float, list[dict[str, Any]]]:
    matching = [item for item in observations if item.get("identity") == identity]
    weights: Counter[str] = Counter()
    for item in matching:
        weights[str(item["pointer_state"])] += max(0.05, float(item["confidence"]))
    if not weights:
        return "uncertain", 0.0, matching
    state, score = max(weights.items(), key=lambda item: (item[1], item[0] == "normal_rightward"))
    total = sum(weights.values())
    return state, min(1.0, score / max(total, 1e-6)), matching


def _terminal_supports_selected_range(item: dict[str, Any]) -> bool:
    occupancy = item.get("terminal_occupancy_left_middle_right")
    if not isinstance(occupancy, list) or len(occupancy) != 3:
        return False
    label = re.sub(r"\s+", "", str(item.get("selected_range_label") or "").lower())
    identity = item.get("identity")
    if identity == "ammeter":
        expected_index = 1 if label in {"0.6", "0.6a"} else 2 if label in {"3", "3a"} else None
    elif identity == "voltmeter":
        expected_index = 1 if label in {"3", "3v"} else 2 if label in {"15", "15v"} else None
    else:
        expected_index = None
    return expected_index is not None and occupancy[expected_index] == "occupied"


def _stable_abnormal(observations: list[dict[str, Any]], identity: str) -> list[dict[str, Any]]:
    abnormal = [
        item
        for item in observations
        if item.get("identity") == identity
        and item.get("circuit_state") != "deenergized"
        and item.get("pointer_state") in {"zero", "reverse", "overrange"}
        and float(item.get("confidence") or 0.0) >= 0.65
    ]
    groups = {int(item.get("image_group") or 0) for item in abnormal}
    if len(groups) >= 2 or any(float(item.get("confidence") or 0.0) >= 0.9 for item in abnormal):
        return abnormal
    return []


def load_signed_pointer_evidence(
    path: Path,
    video_id: str,
    source_video_id: str,
) -> dict[str, Any]:
    report = read_json(path)
    if (
        report.get("artifact_type") != "meter_polarity_measurement_recording_binary_result"
        or report.get("rubric_id") != "resistance.meter_polarity_lenient_v15_apparatus_priors"
        or str(report.get("video_id")) != str(video_id)
        or report.get("excel_accessed") is not False
        or report.get("labels_accessed") is not False
    ):
        raise ValueError("signed pointer evidence identity or provenance is invalid")

    manifest_path = Path(str(report.get("input_manifest") or ""))
    if not manifest_path.is_file():
        raise ValueError("signed pointer input manifest is missing")
    manifest = read_json(manifest_path)
    if str(manifest.get("video_id")) != str(video_id):
        raise ValueError("signed pointer input manifest video mismatch")
    stage_manifest_path = Path(str(manifest.get("stage_manifest") or ""))
    if not stage_manifest_path.is_file() or sha256(stage_manifest_path) != manifest.get("stage_manifest_sha256"):
        raise ValueError("signed pointer stage manifest is missing or changed")
    stage_manifest = read_json(stage_manifest_path)
    stage_record = next(
        (
            item
            for item in stage_manifest.get("videos", [])
            if isinstance(item, dict) and str(item.get("video_id")) == str(video_id)
        ),
        None,
    )
    if not isinstance(stage_record, dict) or Path(str(stage_record.get("source_video") or "")).name != source_video_id:
        raise ValueError("signed pointer source video identity mismatch")

    overrides: dict[str, Any] = {}
    raw_overrides = report.get("pointer_state_overrides")
    for identity in ("ammeter", "voltmeter"):
        pointer = raw_overrides.get(identity) if isinstance(raw_overrides, dict) else None
        meter = report.get(identity)
        if not isinstance(pointer, dict) or not isinstance(meter, dict):
            continue
        integrated = pointer.get("integrated_observation")
        focused = pointer.get("focused_observation")
        reverse_is_consistent = (
            integrated == "reverse_below_zero"
            and focused != "normal_positive_deflection"
            and report.get("fail_trigger") == f"{identity}:reversed"
            and meter.get("violation_type") == "reversed"
        )
        if not reverse_is_consistent:
            continue
        evidence_seconds = [
            float(value)
            for value in meter.get("evidence_seconds", [])
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ]
        overrides[identity] = {
            "pointer_state": "reverse",
            "confidence": round(min(float(report.get("confidence") or 0.0), float(meter.get("confidence") or 0.0)), 4),
            "evidence_seconds": evidence_seconds,
            "evidence": str(meter.get("evidence") or ""),
            "integrated_observation": integrated,
            "focused_observation": focused,
            "fail_trigger": report.get("fail_trigger"),
        }

    evidence_times = {
        timestamp
        for item in overrides.values()
        for timestamp in item.get("evidence_seconds", [])
    }
    evidence_frames: list[dict[str, Any]] = []
    for group in manifest.get("groups", []):
        if not isinstance(group, dict) or float(group.get("timestamp_seconds") or -1.0) not in evidence_times:
            continue
        evidence_frames.append(
            {
                "timestamp_seconds": float(group["timestamp_seconds"]),
                "stage": group.get("stage"),
                "source_stage_frame": group.get("source_stage_frame"),
                "source_stage_frame_sha256": group.get("source_stage_frame_sha256"),
                "overview": group.get("overview"),
                "overview_sha256": group.get("overview_sha256"),
                "rois": group.get("rois", []),
            }
        )
    return {
        "algorithm": "r4_meter_polarity_lenient_v15_apparatus_priors",
        "source_path": str(path.resolve()),
        "source_sha256": sha256(path),
        "input_manifest": str(manifest_path.resolve()),
        "input_manifest_sha256": sha256(manifest_path),
        "video_id": str(video_id),
        "source_video_id": source_video_id,
        "overrides": overrides,
        "evidence_frames": evidence_frames,
        "excel_accessed": False,
        "labels_accessed": False,
    }


def reduce_results(
    qwen: dict[str, Any],
    frames: list[dict[str, Any]],
    signed_pointer_evidence: dict[str, Any] | None = None,
    allow_single_visible_meter: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    observations = [item for item in qwen.get("observations", []) if isinstance(item, dict)]
    active_observations = [item for item in observations if item.get("circuit_state") != "deenergized"]
    ammeter_state, ammeter_consensus, ammeter_obs = _weighted_state(active_observations, "ammeter")
    voltmeter_state, voltmeter_consensus, voltmeter_obs = _weighted_state(active_observations, "voltmeter")
    normal_count = sum(item.get("pointer_state") == "normal_rightward" and float(item.get("confidence") or 0.0) >= 0.45 for item in active_observations)
    stable_abnormal = _stable_abnormal(active_observations, "ammeter") + _stable_abnormal(active_observations, "voltmeter")
    active = any(item.get("circuit_state") == "energized" for item in observations)
    signed_overrides = (
        signed_pointer_evidence.get("overrides", {})
        if isinstance(signed_pointer_evidence, dict)
        else {}
    )
    reverse_overrides = {
        identity: item
        for identity, item in signed_overrides.items()
        if isinstance(item, dict) and item.get("pointer_state") == "reverse"
    }
    if reverse_overrides:
        r5_decision, r5_reason = "fail", "frame_bound_signed_pointer_evidence_confirms_reverse_deflection"
    elif ammeter_state == "normal_rightward" and voltmeter_state == "normal_rightward":
        r5_decision, r5_reason = "pass", "both_meter_identities_show_in_scale_rightward_deflection"
    elif stable_abnormal:
        r5_decision, r5_reason = "fail", "visible_abnormal_pointer_state_in_measurement_window"
    elif allow_single_visible_meter and active and normal_count > 0:
        r5_decision, r5_reason = "pass", "visible_measurement_window_has_normal_pointer_deflection;other_meter_low_visibility"
    else:
        r5_decision, r5_reason = "fail", "no_normal_pointer_deflection_found_after_temporal_and_roi_search"
    r5_confidence = float(qwen.get("overall_confidence") or 0.0)
    if {ammeter_state, voltmeter_state} == {"uncertain"}:
        r5_confidence *= 0.65
    else:
        r5_confidence = max(r5_confidence, 0.5 * (ammeter_consensus + voltmeter_consensus))
    if reverse_overrides:
        r5_confidence = max(
            r5_confidence,
            max(float(item.get("confidence") or 0.0) for item in reverse_overrides.values()),
        )

    explicit_bad_range = [
        item for item in active_observations
        if item.get("range_assessment") in {"too_low", "too_high"}
        and float(item.get("confidence") or 0.0) >= 0.65
        and item.get("plugged_terminal_visible") == "connected"
        and _terminal_supports_selected_range(item)
        and (
            (
                item.get("range_assessment") == "too_low"
                and (
                    item.get("pointer_state") == "overrange"
                    or item.get("pointer_scale_position") == "near_full"
                )
            )
            or (
                item.get("range_assessment") == "too_high"
                and item.get("pointer_scale_position") in {"near_zero", "low"}
            )
        )
    ]
    explicit_good_range = [
        item for item in active_observations
        if item.get("range_assessment") == "appropriate"
        and item.get("pointer_scale_position") in {"low", "mid", "high"}
        and float(item.get("confidence") or 0.0) >= 0.45
    ]
    overrange = [item for item in active_observations if item.get("pointer_state") == "overrange"]
    if explicit_bad_range or overrange:
        r6_decision, r6_reason = "fail", "visible_range_mismatch_or_overrange"
    elif explicit_good_range:
        r6_decision, r6_reason = "pass", "visible_connected_range_and_pointer_position_are_appropriate"
    elif r5_decision == "pass":
        r6_decision, r6_reason = "pass", "in_scale_pointer_deflection_without_visible_range_mismatch"
    else:
        r6_decision, r6_reason = "fail", "range_not_shown_appropriate_after_temporal_and_roi_search"
    r6_confidence = float(qwen.get("overall_confidence") or 0.0)
    if not explicit_bad_range and not explicit_good_range:
        r6_confidence *= 0.72

    common = {
        "candidate_windows": [
            {
                "timestamp_seconds": frame["timestamp_seconds"],
                "frame_number": frame["frame_number"],
                "frame_path": frame["frame_path"],
                "window_source": frame["window_source"],
                "opencv_candidate_count": len(frame["candidates"]),
                "candidates": frame["candidates"],
            }
            for frame in frames
        ],
        "qwen_observation": qwen,
        "signed_pointer_evidence": signed_pointer_evidence,
        "evidence_timepoints_seconds": [frame["timestamp_seconds"] for frame in frames],
        "original_frame_paths": [frame["frame_path"] for frame in frames],
        "roi_paths": [
            path
            for frame in frames
            for candidate in frame["candidates"]
            for path in (
                candidate.get("face_path"),
                candidate.get("wide_path"),
                candidate.get("enhanced_path"),
                candidate.get("terminal_path"),
            )
            if isinstance(path, str)
        ],
        "opencv_is_diagnostic_only": True,
    }
    r5 = {
        "decision": r5_decision,
        "predicted_score": 1 if r5_decision == "pass" else 0,
        "confidence": round(max(0.0, min(1.0, r5_confidence)), 4),
        "reason": r5_reason,
        "diagnostics": {
            **common,
            "needle_states": {"ammeter": ammeter_state, "voltmeter": voltmeter_state},
            "effective_needle_states": {
                "ammeter": (signed_overrides.get("ammeter") or {}).get("pointer_state", ammeter_state),
                "voltmeter": (signed_overrides.get("voltmeter") or {}).get("pointer_state", voltmeter_state),
            },
            "identity_observations": {"ammeter": ammeter_obs, "voltmeter": voltmeter_obs},
        },
    }
    r6 = {
        "decision": r6_decision,
        "predicted_score": 1 if r6_decision == "pass" else 0,
        "confidence": round(max(0.0, min(1.0, r6_confidence)), 4),
        "reason": r6_reason,
        "diagnostics": {
            **common,
            "explicit_bad_range_observations": explicit_bad_range,
            "explicit_good_range_observations": explicit_good_range,
            "range_tie_break": (
                "a mismatch requires consistent visible terminal occupancy plus a near-zero/low or "
                "near-full/overrange pointer; otherwise an in-scale measurement supports pass"
            ),
        },
    }
    return r5, r6


def run_meter_rubrics(
    video_path: Path,
    source_video_id: str,
    video_id: str,
    run_dir: Path,
    model_config: dict[str, Any],
    action_summary_path: Path | None = None,
    boundary_summary_path: Path | None = None,
    fallback_action_summary_path: Path | None = None,
    signed_pointer_evidence_path: Path | None = None,
    allow_historical_fallback: bool = False,
    skill_plan: dict[str, Any] | None = None,
    closed_stable_ammeter_search_path: Path | None = None,
    closed_stable_voltmeter_search_path: Path | None = None,
    closed_stable_runtime_calibration_path: Path | None = None,
    closed_stable_stage_producer_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        from .skills import EXECUTOR_REGISTRY, execution_for_rubric
    except ImportError:
        from skills import EXECUTOR_REGISTRY, execution_for_rubric  # type: ignore
    execution = (
        execution_for_rubric(skill_plan, 5)
        if skill_plan
        else {
            "skill_id": "meter.explicit_measurement",
            "parameters": dict(EXECUTOR_REGISTRY["meter.explicit_measurement"].defaults),
            "execution_fingerprint": None,
        }
    )
    parameters = execution["parameters"]
    fallback = fallback_action_summary_path
    action_path = action_summary_path if action_summary_path and action_summary_path.is_file() else (
        fallback if allow_historical_fallback and fallback is not None else None
    )
    if action_path is None or not action_path.is_file():
        raise ValueError("current live action summary is required")
    summary = read_json(action_path)
    record = _source_record(summary, source_video_id, video_id)
    boundary_record = None
    if boundary_summary_path and boundary_summary_path.is_file():
        boundary_record = _boundary_record(read_json(boundary_summary_path), source_video_id, video_id)
        if boundary_record is not None:
            record = boundary_record
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"unable to open video: {video_path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    duration = frame_count / fps if fps > 0 else 0.0
    evidence_dir = run_dir / "meter_rubrics"
    source_digest = sha256(video_path)
    checkpoint_path = evidence_dir / "selected_frames_pre_qwen.json"
    checkpoint = read_json(checkpoint_path) if checkpoint_path.is_file() else {}
    selected = checkpoint.get("selected_frames") if isinstance(checkpoint, dict) else None
    checkpoint_valid = (
        checkpoint.get("algorithm_version") == ALGORITHM_VERSION
        and checkpoint.get("source_video_sha256") == source_digest
        and checkpoint.get("routing_policy") == (skill_plan or {}).get("routing_policy")
        and checkpoint.get("execution_fingerprint") == execution["execution_fingerprint"]
        and isinstance(selected, list)
        and bool(selected)
        and all(
            isinstance(item, dict)
            and isinstance(item.get("frame_path"), str)
            and Path(item["frame_path"]).is_file()
            for item in selected
        )
    )
    if checkpoint_valid:
        windows = checkpoint.get("candidate_windows") or candidate_windows(
            record, duration, str(parameters["window_mode"])
        )
        sample_count = int(checkpoint.get("sample_count") or len(selected))
        dynamic_identity = checkpoint.get("dynamic_meter_identity") or {
            "skill_version": DYNAMIC_METER.SKILL_VERSION,
            "tracks": [],
        }
    else:
        windows = candidate_windows(record, duration, str(parameters["window_mode"]))
        frames = _decode_frames(
            video_path,
            sampling_timestamps(windows, max_samples=int(parameters["max_samples"])),
            evidence_dir / "frames",
        )
        if parameters["dynamic_meter_candidates"]:
            analyzed = [_export_candidates(item, evidence_dir) for item in frames]
            dynamic_identity = DYNAMIC_METER.prepare_frames(analyzed)
        else:
            analyzed = [
                {
                    **item,
                    "detection": {"valid": False, "errors": ["disabled_by_skill"]},
                    "candidates": [],
                    "model_candidates": [],
                }
                for item in frames
            ]
            dynamic_identity = {
                "skill_version": DYNAMIC_METER.SKILL_VERSION,
                "tracks": [],
            }
        selected = _select_frame_records(analyzed, limit=int(parameters["selected_frame_limit"]))
        _add_selected_pointer_diagnostics(selected)
        sample_count = len(frames)
        write_json(
            checkpoint_path,
            {
                "algorithm_version": ALGORITHM_VERSION,
                "source_video_sha256": source_digest,
                "routing_policy": (skill_plan or {}).get("routing_policy"),
                "execution_fingerprint": execution["execution_fingerprint"],
                "skill_execution": execution,
                "candidate_windows": windows,
                "sample_count": sample_count,
                "dynamic_meter_identity": dynamic_identity,
                "selected_frames": selected,
            },
        )
    adaptive_selected = _load_adaptive_selected(run_dir, source_digest)
    if adaptive_selected:
        existing_frame_numbers = {
            int(item.get("frame_number"))
            for item in selected
            if isinstance(item, dict) and isinstance(item.get("frame_number"), int)
        }
        adaptive_selected = [
            item
            for item in adaptive_selected
            if int(item.get("frame_number")) not in existing_frame_numbers
        ]
        selected = list(selected) + adaptive_selected
    qwen = _call_qwen(
        selected,
        model_config,
        evidence_dir / "qwen_raw_response.json",
        skill_instruction=str(parameters["prompt_instruction"]),
        candidate_crops_per_frame=int(parameters["candidate_crops_per_frame"]),
        execution_fingerprint=execution["execution_fingerprint"],
    )
    signed_pointer_evidence = None
    if signed_pointer_evidence_path is not None:
        signed_pointer_evidence = load_signed_pointer_evidence(
            signed_pointer_evidence_path,
            video_id,
            source_video_id,
        )
    r5, r6 = reduce_results(
        qwen,
        selected,
        signed_pointer_evidence,
        allow_single_visible_meter=bool(parameters["allow_single_visible_meter"]),
    )
    qwen_r6 = r6
    closed_stable_inputs = [
        closed_stable_ammeter_search_path,
        closed_stable_voltmeter_search_path,
        closed_stable_runtime_calibration_path,
    ]
    stage_producer: dict[str, Any] | None = None
    producer_config = closed_stable_stage_producer_config or {}
    if producer_config.get("enabled") is True:
        # Execute mode must either use this run's stage search or its Qwen
        # result. Configured batch evidence is reserved for explicit replay.
        closed_stable_inputs = [None, None, None]
        try:
            stage_producer = CLOSED_STABLE_PRODUCER.run_current_video_search(
                video_path=video_path,
                video_id=video_id,
                temporal_record=record,
                duration_seconds=duration,
                output_root=evidence_dir / "closed_stable_stage_producer",
                config=producer_config,
            )
            live_result_path = Path(str(stage_producer["result_path"]))
            live_calibration_path = Path(str(producer_config["runtime_calibration"]))
            closed_stable_inputs = [live_result_path, live_result_path, live_calibration_path]
        except (OSError, RuntimeError, ValueError, KeyError, subprocess.SubprocessError) as exc:
            producer_error = f"{type(exc).__name__}:{exc}"
            stage_producer = {
                "skill_version": CLOSED_STABLE_PRODUCER.SKILL_VERSION,
                "status": "failed_fallback_to_current_qwen_binary_reducer",
                "error": producer_error,
                "qwen_called": False,
                "excel_accessed": False,
            }
            r6.setdefault("diagnostics", {})["closed_stable_cv_v3"] = {
                "status": "current_video_stage_producer_failed",
                "skill_version": CLOSED_STABLE_R6.SKILL_VERSION,
                "fallback": "current_run_qwen_binary_result",
                "error": producer_error,
            }
    closed_stable_applied = False
    if any(path is not None for path in closed_stable_inputs):
        if not all(path is not None for path in closed_stable_inputs):
            raise ValueError("closed-stable CV V3 requires all three stage-geometry inputs")
        closed_stable = CLOSED_STABLE_R6.evaluate_paths(
            closed_stable_inputs[0],
            closed_stable_inputs[1],
            closed_stable_inputs[2],
            video_id=video_id,
            source_video_id=source_video_id,
        )
        if closed_stable is not None:
            closed_stable["diagnostics"]["parallel_qwen_fallback"] = {
                "decision": qwen_r6["decision"],
                "confidence": qwen_r6["confidence"],
                "reason": qwen_r6["reason"],
            }
            r6 = closed_stable
            closed_stable_applied = True
        else:
            r6["diagnostics"]["closed_stable_cv_v3"] = {
                "status": "current_video_stage_evidence_missing",
                "skill_version": CLOSED_STABLE_R6.SKILL_VERSION,
            }
    report = {
        "schema_version": "resistance_agent_meter_evidence.v1",
        "algorithm_version": ALGORITHM_VERSION,
        "video_id": video_id,
        "source_video_id": source_video_id,
        "source_video_path": str(video_path.resolve()),
        "source_video_sha256": source_digest,
        "action_summary_path": str(action_path.resolve()),
        "boundary_summary_path": str(boundary_summary_path.resolve()) if boundary_summary_path else None,
        "boundary_stage_runs_used": boundary_record is not None,
        "candidate_windows": windows,
        "sample_count": sample_count,
        "selected_frame_count": len(selected),
        "selected_frames": selected,
        "adaptive_evidence_frame_count": len(adaptive_selected),
        "adaptive_evidence_used": bool(adaptive_selected),
        "dynamic_meter_identity": dynamic_identity,
        "qwen_observation": qwen,
        "signed_pointer_evidence": signed_pointer_evidence,
        "closed_stable_cv_v3_applied": closed_stable_applied,
        "closed_stable_cv_v3_skill_version": CLOSED_STABLE_R6.SKILL_VERSION,
        "closed_stable_stage_producer": stage_producer,
        "rubric_5": r5,
        "rubric_6": r6,
        "excel_accessed": False,
        "ground_truth_sent_to_model": False,
        "selection_checkpoint_reused": checkpoint_valid,
        "routing_policy": (skill_plan or {}).get("routing_policy"),
        "skill_selection": (skill_plan or {}).get("skills", []),
        "skill_execution": execution,
        "historical_fallback_used": bool(
            allow_historical_fallback and fallback is not None and action_path == fallback
        ),
    }
    report_path = evidence_dir / "meter_evidence_report.json"
    write_json(report_path, report)
    return {"rubric_5": r5, "rubric_6": r6, "report_path": str(report_path.resolve())}


run_meter_rubrics.supports_boundary_summary = True
run_meter_rubrics.supports_signed_pointer_evidence = True
run_meter_rubrics.supports_closed_stable_cv_v3 = True
run_meter_rubrics.supports_closed_stable_stage_producer = True
