#!/usr/bin/env python3
"""Real-video evidence acquisition for rubric 1: ammeter in the series circuit."""

from __future__ import annotations

import base64
import hashlib
import http.client
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


AGENT_ROOT = Path(__file__).resolve().parent.parent
ALGORITHM_VERSION = "r1_occlusion_aware_dynamic_topology_v8_activity_context"
CONNECTION_STATES = {"seated", "touching", "held_near", "empty", "unclear"}
PATH_VISIBILITY = {"continuous", "partial", "hidden", "unclear"}
PATH_RELATIONS = {"direct", "via_component", "occluded_likely_direct", "no_connection", "unclear"}
FINAL_TOPOLOGY_STATES = {"single_series_loop", "explicit_nonseries", "unclear"}
DIRECT_ACROSS_STATES = {"confirmed", "rejected", "candidate", "unclear"}
HAND_STATES = {"hands_away", "handling_seated", "holding_near", "occluded", "unclear"}
ACTIVITY_CONTEXTS = {"wiring_action", "measurement_action", "writing_action", "unclear"}
VISIBILITY_STATES = {"sufficient", "partial", "insufficient", "unclear"}
CORE_DEVICES = {"ammeter", "battery_holder", "fixed_resistor", "switch"}
TERMINAL_DEVICES = CORE_DEVICES | {"voltmeter", "rheostat", "other", "unclear"}
FAR_ENDPOINTS = TERMINAL_DEVICES | {"out_of_frame"}
METER_IDENTITIES = {"ammeter", "voltmeter", "unknown"}
DEVICE_IDENTITIES = CORE_DEVICES | {"voltmeter", "unknown"}
IDENTITY_BASES = {"A", "V", "green_terminal_panel", "red_terminal_panel", "combined", "unclear"}
ROI_SOURCES = {"dynamic_detection", "temporal_tracking"}
OBSERVATION_STAGE_PREFIXES = ("measurement", "observation")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_fingerprint(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def parse_json_object(content: str) -> dict[str, Any]:
    text = content.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S | re.I)
    if fenced:
        text = fenced.group(1)
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("response does not contain a JSON object")
        text = text[start : end + 1]
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("response JSON root is not an object")
    return value


def _source_record(summary: dict[str, Any], source_video_id: str, video_id: str) -> dict[str, Any]:
    records = summary.get("records")
    if not isinstance(records, list):
        raise ValueError("Temporal Guard summary has no records")
    matches = [
        item
        for item in records
        if isinstance(item, dict)
        and str(item.get("source_video_id") or item.get("video_id") or "")
        in {source_video_id, video_id}
    ]
    if len(matches) != 1:
        matches = [
            item
            for item in records
            if isinstance(item, dict)
            and str(item.get("source_video_id") or "").startswith(f"{video_id}_")
        ]
    if len(matches) != 1:
        raise ValueError(f"expected one Temporal Guard record for video {video_id}, found {len(matches)}")
    return matches[0]


def _boundary_record(
    summary: dict[str, Any],
    source_video_id: str,
    video_id: str,
    allowed_root: Path | None = None,
) -> dict[str, Any] | None:
    records = summary.get("records")
    if not isinstance(records, list):
        return None
    for item in records:
        if not isinstance(item, dict):
            continue
        identity = str(item.get("source_video_id") or item.get("video_id") or "")
        if identity not in {source_video_id, video_id} and not identity.startswith(f"{video_id}_"):
            continue
        runs = item.get("source_observed_stage_runs") or item.get("observed_stage_runs")
        if isinstance(runs, list) and runs:
            return {**item, "observed_stage_runs": runs}
        if allowed_root is None:
            continue
        nested_value = item.get("result_path") or item.get("refined_result_path")
        if not isinstance(nested_value, str) or not nested_value:
            continue
        nested_path = Path(nested_value).resolve()
        if not nested_path.is_relative_to(allowed_root.resolve()) or not nested_path.is_file():
            continue
        nested = read_json(nested_path)
        nested_runs = nested.get("source_observed_stage_runs") or nested.get(
            "observed_stage_runs"
        )
        if isinstance(nested_runs, list) and nested_runs:
            return {**nested, "observed_stage_runs": nested_runs}
    return None


def _stage_runs(record: dict[str, Any]) -> list[dict[str, Any]]:
    runs = record.get("observed_stage_runs")
    if not isinstance(runs, list):
        return []
    return sorted(
        [item for item in runs if isinstance(item, dict)],
        key=lambda item: float(item.get("start_seconds") or 0.0),
    )


def _is_observation_stage(frame_or_stage: dict[str, Any] | str) -> bool:
    """Identify measurement/observation windows without relying on video identity."""
    if isinstance(frame_or_stage, dict):
        stage = str(frame_or_stage.get("stage") or "")
    else:
        stage = str(frame_or_stage or "")
    return stage.startswith(OBSERVATION_STAGE_PREFIXES)


def _is_scored_observation_context(
    observation: dict[str, Any], frame: dict[str, Any]
) -> bool:
    """Require visible measurement activity inside a synthesized recovery window."""
    stage = str(frame.get("stage") or "")
    if stage == "observation_recovery":
        return observation.get("activity_context") == "measurement_action"
    return _is_observation_stage(stage)


def candidate_windows(
    record: dict[str, Any],
    duration_seconds: float,
    window_mode: str = "all_wiring_runs",
    coarse_window_seconds: float = 16.0,
) -> list[dict[str, Any]]:
    """Return current-run wiring and measurement windows for monotonic R1 scoring."""
    if coarse_window_seconds <= 0:
        raise ValueError("coarse_window_seconds must be positive")
    stage_runs = _stage_runs(record)
    has_initial_wiring = any(
        str(item.get("stage") or "") == "circuit_wiring" for item in stage_runs
    )
    windows: list[dict[str, Any]] = []
    stage_counts: dict[str, int] = defaultdict(int)
    for item in stage_runs:
        stage = str(item.get("stage") or "")
        is_wiring_stage = stage in {"circuit_wiring", "circuit_rewiring"}
        if not (is_wiring_stage or _is_observation_stage(stage)):
            continue
        # "initial_wiring_only" limits only the wiring phase. Measurement and
        # observation windows remain mandatory because a later visible
        # topology violation is an irreversible R1 failure.
        if window_mode == "initial_wiring_only" and stage == "circuit_rewiring":
            continue
        start = max(0.0, float(item.get("start_seconds") or 0.0))
        end = min(duration_seconds, float(item.get("end_seconds") or start))
        if end < start:
            continue
        stage_counts[stage] += 1
        stage_run = stage_counts[stage]
        chunk_index = 0
        cursor = start
        while cursor <= end + 1e-6:
            chunk_index += 1
            chunk_end = min(end, cursor + coarse_window_seconds)
            final_chunk = chunk_end >= end - 1e-6
            windows.append(
                {
                    "window_id": f"{stage}_{stage_run:03d}_w{chunk_index:03d}",
                    "stage": stage,
                    "stage_run": stage_run,
                    "coarse_window_index": chunk_index,
                    "is_stage_final_chunk": final_chunk,
                    "start_seconds": round(cursor, 3),
                    "end_seconds": round(chunk_end, 3),
                    "review_end_seconds": round(
                        min(duration_seconds, chunk_end + (6.0 if final_chunk else 0.0)), 3
                    ),
                    "source_event_ids": list(item.get("event_ids") or []),
                    "source_confidence": item.get("confidence"),
                }
            )
            if final_chunk:
                break
            cursor = chunk_end
    if has_initial_wiring:
        observation_runs = [item for item in stage_runs if _is_observation_stage(item)]
        recording_runs = [
            item
            for item in stage_runs
            if str(item.get("stage") or "").startswith("recording")
        ]
        for recovery_index, recording in enumerate(recording_runs, start=1):
            recording_start = max(
                0.0,
                min(duration_seconds, float(recording.get("start_seconds") or 0.0)),
            )
            recovery_start = max(0.0, recording_start - 12.0)
            if recording_start <= recovery_start:
                continue
            has_prior_observation = any(
                float(item.get("start_seconds") or 0.0) <= recording_start
                and float(item.get("end_seconds") or 0.0) >= recovery_start
                for item in observation_runs
            )
            if has_prior_observation:
                continue
            windows.append(
                {
                    "window_id": f"observation_recovery_{recovery_index:03d}_w001",
                    "stage": "observation_recovery",
                    "stage_run": recovery_index,
                    "coarse_window_index": 1,
                    "is_stage_final_chunk": True,
                    "start_seconds": round(recovery_start, 3),
                    "end_seconds": round(recording_start, 3),
                    "review_end_seconds": round(recording_start, 3),
                    "source_event_ids": list(recording.get("event_ids") or []),
                    "source_confidence": recording.get("confidence"),
                }
            )
        windows.sort(
            key=lambda item: (
                float(item["start_seconds"]),
                float(item["end_seconds"]),
                str(item["window_id"]),
            )
        )
    if window_mode == "initial_wiring_only" and windows:
        first_wiring_run = next(
            (
                item["stage_run"]
                for item in windows
                if item["stage"] == "circuit_wiring"
            ),
            None,
        )
        windows = [
            item
            for item in windows
            if _is_observation_stage(item)
            or (
                item["stage"] == "circuit_wiring"
                and item["stage_run"] == first_wiring_run
            )
        ]
    # Missing initial wiring means the temporal segmenter did not give R1 a
    # complete search interval. Use the same current-video broad search for
    # every such run instead of trusting a later rewiring fragment.
    if not has_initial_wiring:
        windows = []
    if not windows:
        bounds = record.get("effective_experiment_interval_seconds") or record.get(
            "locked_experiment_interval_seconds"
        )
        broad_start, broad_end = 0.0, duration_seconds
        if (
            isinstance(bounds, list)
            and len(bounds) == 2
            and all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in bounds)
        ):
            broad_start = max(0.0, min(duration_seconds, float(bounds[0])))
            broad_end = max(broad_start, min(duration_seconds, float(bounds[1])))
        cursor = broad_start
        chunk_index = 0
        while cursor <= broad_end + 1e-6:
            chunk_index += 1
            chunk_end = min(broad_end, cursor + coarse_window_seconds)
            windows.append(
                {
                    "window_id": f"broad_search_w{chunk_index:03d}",
                    "stage": "broad_search",
                    "stage_run": chunk_index,
                    "coarse_window_index": chunk_index,
                    "is_stage_final_chunk": chunk_end >= broad_end - 1e-6,
                    "start_seconds": round(cursor, 3),
                    "end_seconds": round(chunk_end, 3),
                    "review_end_seconds": round(chunk_end, 3),
                    "source_event_ids": [],
                    "source_confidence": None,
                }
            )
            if chunk_end >= broad_end - 1e-6:
                break
            cursor = chunk_end
    return windows


def sampling_timestamps(
    windows: list[dict[str, Any]],
    interval_seconds: float = 5.0,
    max_samples_per_window: int | None = None,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for window in windows:
        start = float(window["start_seconds"])
        end = float(window["end_seconds"])
        review_end = float(window.get("review_end_seconds", end))
        values: list[float] = []
        cursor = start
        while cursor <= review_end + 1e-6:
            values.append(cursor)
            cursor += interval_seconds
        values.extend([end - 4.0, end - 2.0, end, end + 2.0, end + 4.0, review_end])
        normalized = sorted({round(min(review_end, max(start, value)), 3) for value in values})
        if max_samples_per_window and len(normalized) > max_samples_per_window:
            indexes = {
                round(index * (len(normalized) - 1) / (max_samples_per_window - 1))
                for index in range(max_samples_per_window)
            }
            normalized = [normalized[index] for index in sorted(indexes)]
        for timestamp in normalized:
            key = (str(window["window_id"]), int(round(timestamp * 1000)))
            if key in seen:
                continue
            seen.add(key)
            samples.append(
                {
                    **window,
                    "timestamp_seconds": timestamp,
                    "temporal_role": (
                        "stable_candidate"
                        if window.get("is_stage_final_chunk") is True and timestamp >= end - 6.0
                        else "process_scan"
                    ),
                    "evidence_phase": "coarse_scan",
                }
            )
    return samples


def select_coarse_model_frames(
    frames: list[dict[str, Any]], per_window_limit: int = 8
) -> list[dict[str, Any]]:
    """Select an auditable uniform/motion/sharpness subset after a full 2 FPS CV scan."""
    if per_window_limit <= 0:
        raise ValueError("per_window_limit must be positive")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for frame in frames:
        grouped[str(frame["window_id"])].append(frame)
    selected: list[dict[str, Any]] = []
    for window_id in sorted(grouped, key=lambda key: float(grouped[key][0]["start_seconds"])):
        rows = sorted(grouped[window_id], key=lambda item: float(item["timestamp_seconds"]))
        if len(rows) <= per_window_limit:
            chosen = rows
        else:
            priority: list[int] = [0, len(rows) - 1]
            priority.extend(
                round(index * (len(rows) - 1) / max(1, per_window_limit - 1))
                for index in range(per_window_limit)
            )
            priority.append(
                max(range(len(rows)), key=lambda index: float(rows[index].get("motion_score") or 0.0))
            )
            priority.append(
                max(range(len(rows)), key=lambda index: float(rows[index].get("sharpness") or 0.0))
            )
            chosen_indexes: list[int] = []
            # Motion and sharpness are mandatory representatives; replace the
            # least informative uniform interior samples when the limit is full.
            mandatory = priority[-2:]
            for index in priority[:-2]:
                if index not in chosen_indexes:
                    chosen_indexes.append(index)
                if len(chosen_indexes) >= per_window_limit:
                    break
            for index in mandatory:
                if index in chosen_indexes:
                    continue
                replace_at = next(
                    (
                        position
                        for position in range(len(chosen_indexes) - 2, 0, -1)
                        if chosen_indexes[position] not in {0, len(rows) - 1}
                    ),
                    None,
                )
                if replace_at is not None:
                    chosen_indexes[replace_at] = index
            chosen = [rows[index] for index in sorted(set(chosen_indexes))]
        selected.extend(dict(item) for item in chosen)
    for image_group, item in enumerate(selected, start=1):
        item["image_group"] = image_group
        item["model_selection"] = "uniform_motion_sharpness"
    return selected


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 91]):
        raise RuntimeError(f"unable to write image: {path}")


def _enhance(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    light, channel_a, channel_b = cv2.split(lab)
    light = cv2.createCLAHE(clipLimit=1.7, tileGridSize=(8, 8)).apply(light)
    enhanced = cv2.cvtColor(cv2.merge((light, channel_a, channel_b)), cv2.COLOR_LAB2BGR)
    blurred = cv2.GaussianBlur(enhanced, (0, 0), 1.1)
    return cv2.addWeighted(enhanced, 1.25, blurred, -0.25, 0)


def _enhance_model_roi(
    image: np.ndarray, target_long_edge: int = 1400, max_upscale: float = 4.0
) -> np.ndarray:
    """Enlarge a native crop before applying restrained local enhancement."""
    if image.size == 0:
        return image
    height, width = image.shape[:2]
    scale = min(max_upscale, max(1.0, float(target_long_edge) / max(height, width)))
    prepared = (
        cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
        if scale > 1.01
        else image.copy()
    )
    lab = cv2.cvtColor(prepared, cv2.COLOR_BGR2LAB)
    light, channel_a, channel_b = cv2.split(lab)
    light = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8)).apply(light)
    enhanced = cv2.cvtColor(cv2.merge((light, channel_a, channel_b)), cv2.COLOR_LAB2BGR)
    blurred = cv2.GaussianBlur(enhanced, (0, 0), 0.9)
    return cv2.addWeighted(enhanced, 1.18, blurred, -0.18, 0)


def _roi_quality(crop: np.ndarray, frame_area: int, identity: str) -> dict[str, Any]:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    contrast = float(gray.std())
    relative_area = float(crop.shape[0] * crop.shape[1]) / max(1.0, float(frame_area))
    identity_score = 1.0 if identity in {"ammeter", "voltmeter"} else 0.0
    area_score = min(1.0, relative_area / 0.08)
    sharpness_score = min(1.0, math.log1p(max(0.0, sharpness)) / 8.0)
    priority = 0.45 * identity_score + 0.30 * area_score + 0.25 * sharpness_score
    return {
        "native_width": int(crop.shape[1]),
        "native_height": int(crop.shape[0]),
        "native_long_edge": int(max(crop.shape[:2])),
        "sharpness": round(sharpness, 3),
        "contrast": round(contrast, 3),
        "relative_frame_area": round(relative_area, 6),
        "model_view_priority": round(priority, 6),
    }


def _ammeter_context_hint(candidate: dict[str, Any]) -> tuple[int, float, int]:
    """Rank current-frame ammeter evidence ahead of irrelevant meter crops."""
    identity = str(candidate.get("identity") or "unknown")
    diagnostics = candidate.get("identity_diagnostics") or {}
    current_identity = str(diagnostics.get("identity") or "unknown")
    green = diagnostics.get("green_panel") or {}
    terminal_count = max(0, int(green.get("dark_terminal_count") or 0))
    component = green.get("component") or {}
    aspect = float(green.get("aspect_ratio") or 0.0)
    adjacency = float(green.get("orange_adjacency") or 0.0)
    fill_ratio = float(component.get("fill_ratio") or 0.0)
    partial_green_panel = (
        bool(component)
        and terminal_count >= 1
        and 1.2 <= aspect <= 6.5
        and adjacency >= 0.08
        and fill_ratio >= 0.20
    )
    if identity == "ammeter" or current_identity == "ammeter":
        return 3, 1.0, terminal_count
    if green.get("valid") is True or partial_green_panel:
        # A tracked candidate may still carry an older voltmeter label when the
        # current crop contains a partly occluded green ammeter terminal panel.
        return 2, 0.75 if partial_green_panel else 0.9, terminal_count
    if identity == "unknown":
        return 1, 0.0, terminal_count
    return 0, 0.0, terminal_count


def _r1_model_roi_box(candidate: dict[str, Any], image_shape: tuple[int, ...]) -> list[int]:
    """Expand a detected ammeter around its terminals and outgoing leads."""
    height, width = int(image_shape[0]), int(image_shape[1])
    left, top, right, bottom = (int(value) for value in candidate["bbox_xyxy"])
    identity_tier, _, _ = _ammeter_context_hint(candidate)
    if identity_tier < 2:
        return [left, top, right, bottom]

    box_width = max(1, right - left)
    box_height = max(1, bottom - top)
    # Ammeter leads can span nearly the full work surface before reaching a
    # component or a visibly loose plug. Keep a wide current-frame context so
    # the far endpoint is not cropped out of the only enlarged R1 view.
    target_width = min(width, max(box_width, int(round(width * 0.92)), int(box_width * 1.35)))
    target_height = min(
        height,
        max(box_height, int(round(height * 0.72)), int(box_height * 1.25)),
    )
    center_x = (left + right) / 2.0
    center_y = (top + bottom) / 2.0
    expanded_left = int(round(center_x - target_width / 2.0))
    expanded_top = int(round(center_y - target_height / 2.0))
    expanded_left = max(0, min(width - target_width, expanded_left))
    expanded_top = max(0, min(height - target_height, expanded_top))
    return [
        expanded_left,
        expanded_top,
        expanded_left + target_width,
        expanded_top + target_height,
    ]


def _rank_r1_model_roi_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Prefer ammeter terminal/lead context without using video identity."""
    for candidate in candidates:
        identity_tier, hint, terminal_count = _ammeter_context_hint(candidate)
        candidate["model_roi_selection_basis"] = "r1_current_frame_ammeter_context_first"
        candidate["model_roi_selection_features"] = {
            "identity_tier": identity_tier,
            "ammeter_visual_hint": round(hint, 3),
            "green_terminal_count": terminal_count,
        }
    return sorted(
        candidates,
        key=lambda item: (
            int((item.get("model_roi_selection_features") or {}).get("identity_tier") or 0),
            float(
                (item.get("model_roi_selection_features") or {}).get("ammeter_visual_hint")
                or 0.0
            ),
            min(
                2,
                int(
                    (item.get("model_roi_selection_features") or {}).get(
                        "green_terminal_count"
                    )
                    or 0
                ),
            ),
            float((item.get("roi_quality") or {}).get("model_view_priority") or 0.0),
        ),
        reverse=True,
    )


def _orange_device_boxes(image: np.ndarray, limit: int = 8) -> list[list[int]]:
    height, width = image.shape[:2]
    scale = min(1.0, 1200.0 / max(height, width))
    small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 80, 75]), np.array([25, 255, 255]))
    mask |= cv2.inRange(hsv, np.array([165, 75, 65]), np.array([179, 255, 255]))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[tuple[float, list[int]]] = []
    image_area = float(small.shape[0] * small.shape[1])
    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        area = box_width * box_height
        if area < image_area * 0.00025 or area > image_area * 0.28:
            continue
        padding = int(max(box_width, box_height) * 0.45)
        left = max(0, x - padding)
        top = max(0, y - padding)
        right = min(small.shape[1], x + box_width + padding)
        bottom = min(small.shape[0], y + box_height + padding)
        density = cv2.contourArea(contour) / max(1.0, area)
        score = area * (0.5 + density)
        boxes.append(
            (
                score,
                [
                    int(round(left / scale)),
                    int(round(top / scale)),
                    int(round(right / scale)),
                    int(round(bottom / scale)),
                ],
            )
        )
    selected: list[list[int]] = []
    for _, box in sorted(boxes, key=lambda item: item[0], reverse=True):
        if any(_intersection_over_union(box, existing) > 0.55 for existing in selected):
            continue
        selected.append(box)
        if len(selected) >= limit:
            break
    return selected


def _largest_mask_component(mask: np.ndarray) -> dict[str, Any] | None:
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return None
    index = max(range(1, count), key=lambda value: int(stats[value, cv2.CC_STAT_AREA]))
    x, y, width, height, area = (int(value) for value in stats[index])
    return {
        "bbox_xyxy": [x, y, x + width, y + height],
        "width": width,
        "height": height,
        "area": area,
        "fill_ratio": area / float(max(1, width * height)),
    }


def _dark_terminal_count(crop: np.ndarray, bbox: list[int]) -> int:
    left, top, right, bottom = bbox
    panel = crop[max(0, top) : min(crop.shape[0], bottom), max(0, left) : min(crop.shape[1], right)]
    if panel.size == 0:
        return 0
    gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
    dark = cv2.inRange(gray, 0, 90)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    panel_area = float(panel.shape[0] * panel.shape[1])
    count = 0
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        area = float(cv2.contourArea(contour))
        if not 0.002 <= area / max(1.0, panel_area) <= 0.20:
            continue
        aspect = width / max(1.0, float(height))
        if 0.35 <= aspect <= 2.8:
            count += 1
    return count


def classify_partial_meter_identity(crop: np.ndarray) -> dict[str, Any]:
    """Recognize a partial A/V meter panel only inside a dynamic orange candidate."""
    if crop.size == 0:
        return {"identity": "unknown", "identity_basis": "unclear", "reason": "empty_crop"}
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    orange = cv2.inRange(hsv, np.array([8, 75, 65]), np.array([28, 255, 255]))
    green = cv2.inRange(hsv, np.array([35, 55, 35]), np.array([95, 255, 255]))
    red = cv2.inRange(hsv, np.array([0, 100, 55]), np.array([7, 255, 255]))
    red |= cv2.inRange(hsv, np.array([172, 100, 55]), np.array([179, 255, 255]))
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    image_area = float(crop.shape[0] * crop.shape[1])
    orange_ratio = float(np.count_nonzero(orange)) / max(1.0, image_area)

    def assess(mask: np.ndarray) -> dict[str, Any]:
        component = _largest_mask_component(mask)
        if component is None:
            return {"valid": False, "component": None, "dark_terminal_count": 0}
        left, top, right, bottom = component["bbox_xyxy"]
        aspect = component["width"] / max(1.0, float(component["height"]))
        center_y = (top + bottom) / 2.0
        component_mask = np.zeros_like(mask)
        component_mask[top:bottom, left:right] = mask[top:bottom, left:right]
        surround = cv2.dilate(component_mask, np.ones((15, 15), np.uint8))
        orange_adjacency = float(np.count_nonzero((surround > 0) & (orange > 0))) / max(
            1.0, float(component["area"])
        )
        terminal_count = _dark_terminal_count(crop, component["bbox_xyxy"])
        valid = (
            orange_ratio >= 0.015
            and component["area"] / max(1.0, image_area) >= 0.004
            and component["width"] >= 0.18 * crop.shape[1]
            and component["height"] >= 0.04 * crop.shape[0]
            and 1.2 <= aspect <= 6.5
            and component["fill_ratio"] >= 0.25
            and center_y >= 0.25 * crop.shape[0]
            and orange_adjacency >= 0.10
            and terminal_count >= 2
        )
        return {
            "valid": valid,
            "component": component,
            "aspect_ratio": round(aspect, 4),
            "orange_adjacency": round(orange_adjacency, 4),
            "dark_terminal_count": terminal_count,
        }

    green_result = assess(green)
    red_result = assess(red)
    if green_result["valid"] and not red_result["valid"]:
        identity, basis = "ammeter", "green_terminal_panel"
    elif red_result["valid"] and not green_result["valid"]:
        identity, basis = "voltmeter", "red_terminal_panel"
    else:
        identity, basis = "unknown", "unclear"
    return {
        "identity": identity,
        "identity_basis": basis,
        "orange_base_ratio": round(orange_ratio, 6),
        "green_panel": green_result,
        "red_panel": red_result,
        "rule": "orange meter base plus rectangular colored panel plus at least two dark terminal structures",
    }


def _dynamic_device_candidates(image: np.ndarray, boxes: list[list[int]]) -> list[dict[str, Any]]:
    height, width = image.shape[:2]
    candidates: list[dict[str, Any]] = []
    for index, box in enumerate(boxes, start=1):
        left, top, right, bottom = box
        crop = image[top:bottom, left:right]
        identity = classify_partial_meter_identity(crop)
        candidates.append(
            {
                "candidate_id": f"candidate_{index:02d}",
                "bbox_xyxy": box,
                "bbox_xyxy_normalized": [
                    round(left / width, 6),
                    round(top / height, 6),
                    round(right / width, 6),
                    round(bottom / height, 6),
                ],
                "identity": identity["identity"],
                "identity_basis": identity["identity_basis"],
                "roi_source": "dynamic_detection",
                "identity_diagnostics": identity,
            }
        )
    return candidates


def _assign_candidate_tracks(
    candidates: list[dict[str, Any]], previous: list[dict[str, Any]], next_track_id: int
) -> tuple[list[dict[str, Any]], int]:
    used: set[str] = set()
    for candidate in candidates:
        matches = [
            item
            for item in previous
            if str(item.get("track_id")) not in used
            and _intersection_over_union(candidate["bbox_xyxy"], item["bbox_xyxy"]) >= 0.25
        ]
        if matches:
            matched = max(
                matches,
                key=lambda item: _intersection_over_union(candidate["bbox_xyxy"], item["bbox_xyxy"]),
            )
            candidate["track_id"] = matched["track_id"]
            candidate["roi_source"] = "temporal_tracking"
            if candidate.get("identity") == "unknown" and matched.get("identity") in {
                "ammeter",
                "voltmeter",
            }:
                candidate["identity"] = matched["identity"]
                candidate["identity_basis"] = matched["identity_basis"]
                candidate["identity_propagated_from_previous_frame"] = True
            used.add(str(matched["track_id"]))
        else:
            candidate["track_id"] = f"device_track_{next_track_id:03d}"
            next_track_id += 1
    return candidates, next_track_id


def _intersection_over_union(first: list[int], second: list[int]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    first_area = max(1, first[2] - first[0]) * max(1, first[3] - first[1])
    second_area = max(1, second[2] - second[0]) * max(1, second[3] - second[1])
    return intersection / float(first_area + second_area - intersection)


def _roi_sheet(image: np.ndarray, candidates: list[dict[str, Any]]) -> np.ndarray | None:
    if not candidates:
        return None
    tile_width, tile_height = 520, 310
    columns = 2
    rows = int(math.ceil(min(len(candidates), 8) / columns))
    sheet = np.full((rows * tile_height, columns * tile_width, 3), 245, dtype=np.uint8)
    for index, candidate in enumerate(candidates[:8]):
        left, top, right, bottom = candidate["bbox_xyxy"]
        crop = image[top:bottom, left:right]
        if crop.size == 0:
            continue
        available_width, available_height = tile_width - 20, tile_height - 42
        scale = min(available_width / crop.shape[1], available_height / crop.shape[0])
        resized = cv2.resize(
            crop,
            (max(1, int(round(crop.shape[1] * scale))), max(1, int(round(crop.shape[0] * scale)))),
            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
        )
        row, column = divmod(index, columns)
        x = column * tile_width + 10
        y = row * tile_height + 34
        sheet[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
        cv2.putText(
            sheet,
            f"candidate {index + 1}: {candidate['identity']}/{candidate['identity_basis']}",
            (column * tile_width + 12, row * tile_height + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
    return sheet


def _decode_and_export(
    video_path: Path,
    samples: list[dict[str, Any]],
    evidence_dir: Path,
    *,
    roi_target_long_edge: int = 1400,
    max_model_roi_views_per_frame: int = 1,
) -> list[dict[str, Any]]:
    frames_dir = evidence_dir / "frames"
    enhanced_dir = evidence_dir / "frames_enhanced"
    sheet_dir = evidence_dir / "device_roi_sheets"
    roi_dir = evidence_dir / "device_rois"
    native_roi_dir = evidence_dir / "device_rois_native"
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"unable to open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    output: list[dict[str, Any]] = []
    previous_gray: np.ndarray | None = None
    previous_candidates: list[dict[str, Any]] = []
    previous_window_id: str | None = None
    next_track_id = 1
    try:
        for sample_index, sample in enumerate(samples, start=1):
            window_id = str(sample["window_id"])
            if window_id != previous_window_id:
                previous_gray = None
                previous_candidates = []
                previous_window_id = window_id
            timestamp = float(sample["timestamp_seconds"])
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            frame_number = int(round(timestamp * fps)) if fps > 0 else 0
            frame_id = f"frame_{frame_number:08d}"
            stem = f"frame_{sample_index:04d}_{timestamp:09.3f}s"
            frame_path = frames_dir / f"{stem}.jpg"
            enhanced_path = enhanced_dir / f"{stem}_enhanced.jpg"
            enhanced = _enhance(frame)
            _write_image(frame_path, frame)
            _write_image(enhanced_path, enhanced)
            boxes = _orange_device_boxes(frame)
            candidates = _dynamic_device_candidates(frame, boxes)
            candidates, next_track_id = _assign_candidate_tracks(
                candidates, previous_candidates, next_track_id
            )
            frame_area = int(frame.shape[0] * frame.shape[1])
            for candidate in candidates:
                model_roi_box = _r1_model_roi_box(candidate, frame.shape)
                left, top, right, bottom = model_roi_box
                native_crop = frame[top:bottom, left:right]
                roi_path: Path | None = None
                native_roi_path: Path | None = None
                quality: dict[str, Any] = {}
                if native_crop.size:
                    native_roi_path = native_roi_dir / f"{stem}_{candidate['candidate_id']}.jpg"
                    roi_path = roi_dir / f"{stem}_{candidate['candidate_id']}.jpg"
                    prepared = _enhance_model_roi(
                        native_crop, target_long_edge=max(320, int(roi_target_long_edge))
                    )
                    _write_image(native_roi_path, native_crop)
                    _write_image(roi_path, prepared)
                    quality = _roi_quality(native_crop, frame_area, str(candidate.get("identity")))
                candidate.update(
                    {
                        "frame_id": frame_id,
                        "frame_number": frame_number,
                        "timestamp_seconds": round(timestamp, 3),
                        "model_roi_bbox_xyxy": model_roi_box,
                        "model_roi_bbox_xyxy_normalized": [
                            round(left / frame.shape[1], 6),
                            round(top / frame.shape[0], 6),
                            round(right / frame.shape[1], 6),
                            round(bottom / frame.shape[0], 6),
                        ],
                        "model_roi_role": (
                            "ammeter_terminals_leads_and_visible_endpoints"
                            if _ammeter_context_hint(candidate)[0] >= 2
                            else "device_candidate"
                        ),
                        "roi_path": str(roi_path.resolve()) if roi_path else None,
                        "native_roi_path": (
                            str(native_roi_path.resolve()) if native_roi_path else None
                        ),
                        "enhanced_roi_path": str(roi_path.resolve()) if roi_path else None,
                        "roi_quality": quality,
                    }
                )
            ranked_candidates = _rank_r1_model_roi_candidates(candidates)
            selected_candidate_ids = {
                str(item.get("candidate_id"))
                for item in ranked_candidates[: max(0, int(max_model_roi_views_per_frame))]
            }
            for rank, candidate in enumerate(ranked_candidates, start=1):
                candidate["model_roi_rank"] = rank
                candidate["model_roi_selected"] = str(candidate.get("candidate_id")) in selected_candidate_ids
            sheet = _roi_sheet(enhanced, candidates)
            sheet_path: Path | None = None
            if sheet is not None:
                sheet_path = sheet_dir / f"{stem}_candidates.jpg"
                _write_image(sheet_path, sheet)
            gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (320, 180))
            motion = (
                float(np.mean(cv2.absdiff(gray, previous_gray))) if previous_gray is not None else 0.0
            )
            previous_gray = gray
            previous_candidates = candidates
            output.append(
                {
                    **sample,
                    "sample_index": sample_index,
                    "frame_id": frame_id,
                    "frame_number": frame_number,
                    "frame_path": str(frame_path.resolve()),
                    "enhanced_frame_path": str(enhanced_path.resolve()),
                    "device_roi_sheet_path": str(sheet_path.resolve()) if sheet_path else None,
                    "device_candidate_boxes_xyxy": boxes,
                    "device_localizations": candidates,
                    "model_roi_view_limit": max(0, int(max_model_roi_views_per_frame)),
                    "roi_target_long_edge": max(320, int(roi_target_long_edge)),
                    "sharpness": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 3),
                    "motion_score": round(motion, 3),
                }
            )
    finally:
        capture.release()
    for image_group, item in enumerate(output, start=1):
        item["image_group"] = image_group
    return output


def _scan_frame_metadata(
    video_path: Path, samples: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Run the 2 FPS R1 scout without exporting every scanned image."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"unable to open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    output: list[dict[str, Any]] = []
    previous_gray: np.ndarray | None = None
    previous_candidates: list[dict[str, Any]] = []
    previous_window_id: str | None = None
    next_track_id = 1
    try:
        for sample_index, sample in enumerate(samples, start=1):
            window_id = str(sample["window_id"])
            if window_id != previous_window_id:
                previous_gray = None
                previous_candidates = []
                previous_window_id = window_id
            timestamp = float(sample["timestamp_seconds"])
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            boxes = _orange_device_boxes(frame)
            candidates = _dynamic_device_candidates(frame, boxes)
            candidates, next_track_id = _assign_candidate_tracks(
                candidates, previous_candidates, next_track_id
            )
            gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (320, 180))
            motion = (
                float(np.mean(cv2.absdiff(gray, previous_gray)))
                if previous_gray is not None
                else 0.0
            )
            frame_number = int(round(timestamp * fps)) if fps > 0 else 0
            output.append(
                {
                    **sample,
                    "sample_index": sample_index,
                    "frame_id": f"frame_{frame_number:08d}",
                    "frame_number": frame_number,
                    "device_candidate_boxes_xyxy": boxes,
                    "device_localizations": candidates,
                    "sharpness": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 3),
                    "motion_score": round(motion, 3),
                }
            )
            previous_gray = gray
            previous_candidates = candidates
    finally:
        capture.release()
    return output


def image_data_url(path: Path, max_edge: int = 1400, jpeg_quality: int = 88) -> str:
    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"unable to read model image: {path}")
    height, width = image.shape[:2]
    scale = min(1.0, float(max_edge) / max(height, width))
    if scale < 1.0:
        image = cv2.resize(
            image,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    if not ok:
        raise RuntimeError(f"unable to encode model image: {path}")
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def _prompt(
    frames: list[dict[str, Any]],
    dense_confirmation: bool = False,
    skill_instruction: str = "",
) -> str:
    groups = ", ".join(
        f"{item['image_group']}={item['timestamp_seconds']:.3f}s/"
        f"{item.get('stage', 'unknown')}/{item['temporal_role']}"
        for item in frames
    )
    sequence = (
        "These are dense adjacent frames around one possible direct-across event. Compare insertion progress "
        "across the sequence before choosing seated, touching, or held_near."
        if dense_confirmation
        else "Adjacent groups may show one connection changing over time; do not merge different times."
    )
    return f"""You are the visual observer for rubric 1 in a school resistance-measurement video.
 Judge only what the supplied image group directly shows. {sequence} {skill_instruction}

The main final-state question is whether the ammeter, battery holder, fixed resistor, and knife switch form one
series loop. This experiment has no rheostat, so do not require one. The voltmeter may be parallel to the fixed
resistor and is not a member of the series core. The separate process question is whether both ammeter terminals are
simultaneously wired directly to two battery-holder terminals. A single ammeter-to-battery wire is a normal series
edge and is never a violation.

For every visible ammeter or battery terminal, distinguish these states geometrically:
- seated: the metal plug is visibly inserted and retained in the fixed terminal socket;
- touching: the plug tip touches the terminal but insertion/retention is not visible;
- held_near: a hand holds the plug close to the terminal with a visible gap or no visible insertion;
- empty: the socket is visibly empty;
- unclear: the relevant contact is hidden or too blurred.
Do not call touching or held_near seated. Trace a lead to its far endpoint only when the path is continuously visible.
If a path goes through a switch, resistor, meter, or another component, set passes_through_other_device=true. Do not
infer hidden wires from color, experiment order, or expected circuit theory. A candidate ROI sheet is only a set of
orange-object crops; verify device identity against the panorama. A visible A glyph or a rectangular dark-green
terminal panel spatially attached to an orange meter base identifies an ammeter. A visible V glyph or rectangular red
terminal panel on such a base identifies a voltmeter. Green batteries/plugs and red wires/dashed marks are not meter
panels. Color establishes device identity only and never establishes a wiring violation.
For device_localizations, copy a candidate_id or track_id only from the current image group's printed CV hints. Never
reuse a label from another image group. If no current candidate supports an identity, use track_id="unknown" and
identity="unknown" rather than inventing a localization.

Inspect both far ends of the ammeter leads. Add a loose_lead_endpoints row only when the full lead is continuously
visible from a seated ammeter terminal to a banana plug that is visibly outside every socket in this same image group.
If the lead exits the frame, crosses an occlusion, or the far plug cannot be distinguished from a seated plug, do not
report a loose endpoint. A hand pressing the knife switch may still be a stable observation; a hand inserting or
removing a lead at an ammeter or battery terminal makes stable_state=false. In a measurement or observation stage,
still report a directly visible loose ammeter lead when stable_state=false; the active hand changes stability but does
not erase the visible loose endpoint. Set activity_context=wiring_action when a hand inserts, removes, carries, or
repositions a plug or lead at a terminal. Set activity_context=measurement_action only when the visible action is
holding the switch closed, holding a measurement contact steady, or reading the meters without changing terminal
connections. If both appear, choose wiring_action while any terminal connection is being changed.

Use final_topology=single_series_loop only when a stable frame directly supports one loop through all four core
devices and list all four in core_devices_visible. Use explicit_nonseries only for a directly visible stable
topological error. Otherwise use unclear. terminal_evidence may describe terminals on any visible device, not only
the ammeter and battery holder. Use
direct_across_state=confirmed only when two distinct ammeter terminals and two distinct battery terminals are visibly
seated as two direct pairs, with neither path passing through another device. Give stable terminal IDs based on visible
relative position, such as ammeter_left and battery_positive. path_relation must be direct only for an uninterrupted
wire path; use via_component whenever a switch, resistor, voltmeter, or other component lies on the path. Use
occluded_likely_direct only as a derived hypothesis and list the directly visible endpoint facts separately. Groups:
{groups}.

Return exactly one JSON object with one observation for every listed numeric image_group:
{{
  "observations": [
    {{
      "image_group": 1,
      "stable_state": true,
      "hands_or_plugs": "hands_away|handling_seated|holding_near|occluded|unclear",
      "activity_context": "wiring_action|measurement_action|writing_action|unclear",
      "core_devices_visible": ["ammeter", "battery_holder", "fixed_resistor", "switch"],
      "topology_visibility": "sufficient|partial|insufficient|unclear",
      "final_topology": "single_series_loop|explicit_nonseries|unclear",
      "direct_across_state": "confirmed|rejected|candidate|unclear",
      "device_localizations": [
        {{
          "track_id": "visible candidate label or unknown",
          "identity": "ammeter|voltmeter|battery_holder|fixed_resistor|switch|unknown",
          "identity_basis": "A|V|green_terminal_panel|red_terminal_panel|combined|unclear",
          "evidence": "directly visible identity cue"
        }}
      ],
      "path_relation": "direct|via_component|occluded_likely_direct|no_connection|unclear",
      "intermediate_components": [],
      "strict_verification": {{
        "verification": "confirmed|rejected",
        "ammeter_present": true,
        "battery_holder_present": true,
        "two_distinct_ammeter_terminals_connected": true,
        "two_distinct_battery_terminals_connected": true,
        "direct_pairing_visible": true
      }},
      "terminal_evidence": [
        {{
          "device": "ammeter|battery_holder|fixed_resistor|switch|voltmeter|rheostat|other|unclear",
          "terminal_label": "visible label or relative position",
          "ammeter_terminal_id": "ammeter terminal ID or empty string",
          "battery_terminal_id": "battery terminal ID or empty string",
          "connection_state": "seated|touching|held_near|empty|unclear",
          "far_endpoint": "ammeter|battery_holder|fixed_resistor|switch|voltmeter|rheostat|other|out_of_frame|unclear",
          "path_visibility": "continuous|partial|hidden|unclear",
          "path_relation": "direct|via_component|occluded_likely_direct|no_connection|unclear",
          "intermediate_components": [],
          "passes_through_other_device": false,
          "evidence": "short geometric observation"
        }}
      ],
      "loose_lead_endpoints": [
        {{
          "source_device": "ammeter",
          "ammeter_terminal_id": "visible ammeter terminal ID",
          "source_connection_state": "seated|touching|held_near|empty|unclear",
          "far_end_state": "loose_plug",
          "path_visibility": "continuous|partial|hidden|unclear",
          "evidence": "visible uninterrupted lead ending in a loose plug"
        }}
      ],
      "direct_observations": ["facts directly visible in this image group"],
      "derived_observations": ["hypotheses derived from occlusion or temporal continuity"],
      "confidence": 0.0,
      "evidence": "short direct observation"
    }}
  ]
}}
Do not output a rubric score. Preserve every listed image_group exactly and do not use a third final class outside
the enumerated observation fields. Keep the JSON compact: at most five device_localizations, four terminal_evidence
rows, two loose_lead_endpoints, three direct_observations, and one derived_observation per image group; keep each evidence string under 16
words."""


def validate_observation(value: dict[str, Any], expected_groups: set[int]) -> list[str]:
    errors: list[str] = []
    observations = value.get("observations")
    if not isinstance(observations, list):
        return ["observations_missing"]
    groups: list[int] = []
    for item in observations:
        if not isinstance(item, dict):
            errors.append("observation_not_object")
            continue
        group = item.get("image_group")
        if isinstance(group, bool) or not isinstance(group, int):
            errors.append("image_group_invalid")
        else:
            groups.append(group)
        if not isinstance(item.get("stable_state"), bool):
            errors.append("stable_state_invalid")
        if item.get("hands_or_plugs") not in HAND_STATES:
            errors.append("hands_or_plugs_invalid")
        if (
            "activity_context" in item
            and item.get("activity_context") not in ACTIVITY_CONTEXTS
        ):
            errors.append("activity_context_invalid")
        devices = item.get("core_devices_visible")
        if not isinstance(devices, list) or any(device not in DEVICE_IDENTITIES - {"unknown"} for device in devices):
            errors.append("core_devices_visible_invalid")
        if item.get("topology_visibility") not in VISIBILITY_STATES:
            errors.append("topology_visibility_invalid")
        if item.get("final_topology") not in FINAL_TOPOLOGY_STATES:
            errors.append("final_topology_invalid")
        if item.get("direct_across_state") not in DIRECT_ACROSS_STATES:
            errors.append("direct_across_state_invalid")
        localizations = item.get("device_localizations")
        if not isinstance(localizations, list):
            errors.append("device_localizations_invalid")
        else:
            for localization in localizations:
                if not isinstance(localization, dict):
                    errors.append("device_localization_not_object")
                    continue
                if not isinstance(localization.get("track_id"), str):
                    errors.append("device_localization_track_id_invalid")
                if localization.get("identity") not in DEVICE_IDENTITIES:
                    errors.append("device_localization_identity_invalid")
                if localization.get("identity_basis") not in IDENTITY_BASES:
                    errors.append("device_localization_identity_basis_invalid")
                if not isinstance(localization.get("evidence"), str):
                    errors.append("device_localization_evidence_invalid")
        if item.get("path_relation") not in PATH_RELATIONS:
            errors.append("path_relation_invalid")
        intermediate = item.get("intermediate_components")
        if not isinstance(intermediate, list) or any(
            component not in TERMINAL_DEVICES for component in intermediate
        ):
            errors.append("intermediate_components_invalid")
        strict = item.get("strict_verification")
        strict_flags = (
            "ammeter_present",
            "battery_holder_present",
            "two_distinct_ammeter_terminals_connected",
            "two_distinct_battery_terminals_connected",
            "direct_pairing_visible",
        )
        if not isinstance(strict, dict):
            errors.append("strict_verification_invalid")
        else:
            if strict.get("verification") not in {"confirmed", "rejected"}:
                errors.append("strict_verification_state_invalid")
            if any(not isinstance(strict.get(name), bool) for name in strict_flags):
                errors.append("strict_verification_flags_invalid")
            if strict.get("verification") == "confirmed" and not all(
                strict.get(name) is True for name in strict_flags
            ):
                errors.append("strict_verification_inconsistent")
        terminals = item.get("terminal_evidence")
        if not isinstance(terminals, list):
            errors.append("terminal_evidence_invalid")
        else:
            for terminal in terminals:
                if not isinstance(terminal, dict):
                    errors.append("terminal_not_object")
                    continue
                if terminal.get("device") not in TERMINAL_DEVICES:
                    errors.append("terminal_device_invalid")
                if not isinstance(terminal.get("terminal_label"), str):
                    errors.append("terminal_label_invalid")
                if not isinstance(terminal.get("ammeter_terminal_id"), str):
                    errors.append("ammeter_terminal_id_invalid")
                if not isinstance(terminal.get("battery_terminal_id"), str):
                    errors.append("battery_terminal_id_invalid")
                if terminal.get("connection_state") not in CONNECTION_STATES:
                    errors.append("connection_state_invalid")
                if terminal.get("far_endpoint") not in FAR_ENDPOINTS:
                    errors.append("far_endpoint_invalid")
                if terminal.get("path_visibility") not in PATH_VISIBILITY:
                    errors.append("path_visibility_invalid")
                if terminal.get("path_relation") not in PATH_RELATIONS:
                    errors.append("terminal_path_relation_invalid")
                terminal_intermediate = terminal.get("intermediate_components")
                if not isinstance(terminal_intermediate, list) or any(
                    component not in TERMINAL_DEVICES for component in terminal_intermediate
                ):
                    errors.append("terminal_intermediate_components_invalid")
                if terminal.get("passes_through_other_device") not in {True, False, None}:
                    errors.append("passes_through_other_device_invalid")
                if not isinstance(terminal.get("evidence"), str):
                    errors.append("terminal_evidence_text_invalid")
        loose_endpoints = item.get("loose_lead_endpoints")
        if not isinstance(loose_endpoints, list):
            errors.append("loose_lead_endpoints_invalid")
        else:
            for endpoint in loose_endpoints:
                if not isinstance(endpoint, dict):
                    errors.append("loose_lead_endpoint_not_object")
                    continue
                if endpoint.get("source_device") != "ammeter":
                    errors.append("loose_lead_source_device_invalid")
                if not isinstance(endpoint.get("ammeter_terminal_id"), str):
                    errors.append("loose_lead_ammeter_terminal_id_invalid")
                if endpoint.get("source_connection_state") not in CONNECTION_STATES:
                    errors.append("loose_lead_source_connection_state_invalid")
                if endpoint.get("far_end_state") != "loose_plug":
                    errors.append("loose_lead_far_end_state_invalid")
                if endpoint.get("path_visibility") not in PATH_VISIBILITY:
                    errors.append("loose_lead_path_visibility_invalid")
                if not isinstance(endpoint.get("evidence"), str):
                    errors.append("loose_lead_evidence_invalid")
        for field in ("direct_observations", "derived_observations"):
            values = item.get(field)
            if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
                errors.append(f"{field}_invalid")
        if item.get("direct_across_state") == "confirmed" and (
            not isinstance(strict, dict)
            or strict.get("verification") != "confirmed"
            or item.get("path_relation") != "direct"
            or bool(intermediate)
        ):
            errors.append("confirmed_direct_across_inconsistent")
        confidence = item.get("confidence")
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not 0 <= float(confidence) <= 1
        ):
            errors.append("confidence_invalid")
        if not isinstance(item.get("evidence"), str):
            errors.append("evidence_invalid")
    if set(groups) != expected_groups or len(groups) != len(expected_groups):
        errors.append("image_groups_mismatch")
    return sorted(set(errors))


def _normalize_model_observation(item: dict[str, Any]) -> list[str]:
    """Apply only lossless or conservative schema repairs before validation."""
    changes: list[str] = []
    if not isinstance(item.get("loose_lead_endpoints"), list):
        item["loose_lead_endpoints"] = []
        changes.append("defaulted_missing_loose_lead_endpoints")
    if item.get("activity_context") not in ACTIVITY_CONTEXTS:
        item["activity_context"] = "unclear"
        changes.append("defaulted_missing_activity_context")
    aliases = {
        "single_pole_switch": "switch",
        "knife_switch": "switch",
        "resistor": "fixed_resistor",
        "fixed_resistance": "fixed_resistor",
        "battery": "battery_holder",
    }
    devices = item.get("core_devices_visible")
    if isinstance(devices, list):
        normalized = [aliases.get(str(device), str(device)) for device in devices]
        if normalized != devices:
            item["core_devices_visible"] = normalized
            changes.append("normalized_core_device_aliases")
    localizations = item.get("device_localizations")
    if isinstance(localizations, list):
        for localization in localizations:
            if not isinstance(localization, dict):
                continue
            identity = localization.get("identity")
            normalized_identity = aliases.get(str(identity), identity)
            if normalized_identity != identity:
                localization["identity"] = normalized_identity
                changes.append("normalized_device_identity_alias")
    intermediate = item.get("intermediate_components")
    if isinstance(intermediate, list):
        normalized_intermediate = [aliases.get(str(value), str(value)) for value in intermediate]
        if normalized_intermediate != intermediate:
            item["intermediate_components"] = normalized_intermediate
            changes.append("normalized_intermediate_component_aliases")
        if normalized_intermediate and item.get("path_relation") == "direct":
            item["path_relation"] = "via_component"
            changes.append("downgraded_direct_path_with_intermediate_component")
    terminals = item.get("terminal_evidence")
    if isinstance(terminals, list):
        for terminal in terminals:
            if not isinstance(terminal, dict):
                continue
            for field in ("device", "far_endpoint"):
                value = terminal.get(field)
                normalized_value = aliases.get(str(value), value)
                if normalized_value != value:
                    terminal[field] = normalized_value
                    changes.append(f"normalized_terminal_{field}_alias")
            terminal_intermediate = terminal.get("intermediate_components")
            if isinstance(terminal_intermediate, list):
                normalized_terminal_intermediate = [
                    aliases.get(str(value), str(value)) for value in terminal_intermediate
                ]
                if normalized_terminal_intermediate != terminal_intermediate:
                    terminal["intermediate_components"] = normalized_terminal_intermediate
                    changes.append("normalized_terminal_intermediate_aliases")
                if normalized_terminal_intermediate and terminal.get("path_relation") == "direct":
                    terminal["path_relation"] = "via_component"
                    terminal["passes_through_other_device"] = True
                    changes.append("downgraded_terminal_direct_path_with_component")
    strict = item.get("strict_verification")
    strict_flags = (
        "ammeter_present",
        "battery_holder_present",
        "two_distinct_ammeter_terminals_connected",
        "two_distinct_battery_terminals_connected",
        "direct_pairing_visible",
    )
    if (
        isinstance(strict, dict)
        and strict.get("verification") == "confirmed"
        and not all(strict.get(name) is True for name in strict_flags)
    ):
        strict["verification"] = "rejected"
        if item.get("direct_across_state") == "confirmed":
            item["direct_across_state"] = "candidate"
        changes.append("downgraded_incomplete_strict_confirmation")
    if item.get("direct_across_state") == "confirmed" and (
        not isinstance(strict, dict)
        or strict.get("verification") != "confirmed"
        or item.get("path_relation") != "direct"
        or bool(item.get("intermediate_components"))
    ):
        item["direct_across_state"] = "candidate"
        changes.append("downgraded_inconsistent_direct_across_confirmation")
    return sorted(set(changes))


def _bind_observation_to_frame(item: dict[str, Any], frame: dict[str, Any]) -> list[str]:
    """Bind model-localized identities to candidate IDs generated from this exact frame."""
    allowed_track_ids: set[str] = set()
    candidates_by_identity: dict[str, list[str]] = defaultdict(list)
    for candidate in frame.get("device_localizations", []):
        if not isinstance(candidate, dict):
            continue
        for field in ("track_id", "candidate_id"):
            value = candidate.get(field)
            if isinstance(value, str) and value:
                allowed_track_ids.add(value)
        identity = str(candidate.get("identity") or "")
        track_id = str(candidate.get("track_id") or "")
        if identity in DEVICE_IDENTITIES - {"unknown"} and track_id:
            candidates_by_identity[identity].append(track_id)
    bound_identities: set[str] = set()
    unsupported_track_ids: set[str] = set()
    inferred_identities: set[str] = set()
    identities_by_reference: dict[str, set[str]] = defaultdict(set)
    for localization in item.get("device_localizations", []):
        if not isinstance(localization, dict):
            continue
        track_id = str(localization.get("track_id") or "")
        candidate_id = str(localization.get("candidate_id") or "")
        identity = str(localization.get("identity") or "")
        references = {value for value in (track_id, candidate_id) if value}
        if references & allowed_track_ids:
            if identity in DEVICE_IDENTITIES:
                bound_identities.add(identity)
                for reference in references & allowed_track_ids:
                    identities_by_reference[reference].add(identity)
        elif (
            identity in DEVICE_IDENTITIES - {"unknown"}
            and (track_id in {"", "unknown"} or candidate_id in {"", "unknown"})
            and len(candidates_by_identity.get(identity, [])) == 1
        ):
            # The model may omit the printed label. A unique identity cue from
            # this same frame is still a current-run binding, not a cross-frame
            # or video-specific fallback.
            bound_identities.add(identity)
            inferred_identities.add(identity)
        elif track_id and track_id != "unknown":
            unsupported_track_ids.add(track_id)
    conflicting_track_ids = sorted(
        reference
        for reference, identities in identities_by_reference.items()
        if len(identities) > 1
    )
    binding = {
        "frame_id": frame.get("frame_id"),
        "timestamp_seconds": frame.get("timestamp_seconds"),
        "candidate_track_ids": sorted(allowed_track_ids),
        "bound_device_identities": sorted(bound_identities),
        "unsupported_model_track_ids": sorted(unsupported_track_ids),
        "conflicting_track_ids": conflicting_track_ids,
        "identity_inferred_from_unique_current_candidate": sorted(inferred_identities),
        "all_core_devices_frame_bound": CORE_DEVICES.issubset(bound_identities),
        "ammeter_and_battery_frame_bound": {
            "ammeter",
            "battery_holder",
        }.issubset(bound_identities),
    }
    item["frame_binding"] = binding
    return ["rejected_cross_group_or_unknown_track_ids"] if unsupported_track_ids else []


def _frame_bound_support(observation: dict[str, Any], required: set[str]) -> bool:
    binding = observation.get("frame_binding")
    if not isinstance(binding, dict):
        # Synthetic reducer fixtures and explicit replay artifacts predate
        # frame binding; live model observations always receive this field.
        return True
    if binding.get("conflicting_track_ids"):
        return False
    identities = binding.get("bound_device_identities")
    return isinstance(identities, list) and required.issubset(set(identities))


def _partial_observation_objects(content: str) -> list[dict[str, Any]]:
    """Recover complete observation objects when a long JSON response is truncated."""
    marker = content.find('"observations"')
    if marker < 0:
        return []
    start = content.find("[", marker)
    if start < 0:
        return []
    output: list[dict[str, Any]] = []
    index = start + 1
    while index < len(content):
        while index < len(content) and content[index] in " \t\r\n,":
            index += 1
        if index >= len(content) or content[index] != "{":
            break
        object_start = index
        depth = 0
        in_string = False
        escaped = False
        while index < len(content):
            character = content[index]
            if in_string:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == '"':
                    in_string = False
            elif character == '"':
                in_string = True
            elif character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
                if depth == 0:
                    try:
                        value = json.loads(content[object_start : index + 1])
                    except json.JSONDecodeError:
                        return output
                    if isinstance(value, dict):
                        output.append(value)
                    index += 1
                    break
            index += 1
        else:
            break
    return output


def _fallback_observation(local_group: int, reason: str) -> dict[str, Any]:
    return {
        "image_group": local_group,
        "stable_state": False,
        "hands_or_plugs": "unclear",
        "activity_context": "unclear",
        "core_devices_visible": [],
        "topology_visibility": "unclear",
        "final_topology": "unclear",
        "direct_across_state": "unclear",
        "device_localizations": [],
        "path_relation": "unclear",
        "intermediate_components": [],
        "strict_verification": {
            "verification": "rejected",
            "ammeter_present": False,
            "battery_holder_present": False,
            "two_distinct_ammeter_terminals_connected": False,
            "two_distinct_battery_terminals_connected": False,
            "direct_pairing_visible": False,
        },
        "terminal_evidence": [],
        "loose_lead_endpoints": [],
        "direct_observations": [],
        "derived_observations": [reason],
        "confidence": 0.2,
        "evidence": reason,
    }


def _call_qwen_batch(
    frames: list[dict[str, Any]],
    model_config: dict[str, Any],
    raw_path: Path,
    dense_confirmation: bool = False,
    skill_instruction: str = "",
    execution_fingerprint: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base_url = os.getenv("QWEN_API_BASE_URL", str(model_config["base_url"]))
    token = os.getenv("QWEN_API_TOKEN", "EMPTY")
    model = os.getenv("QWEN_MODEL", str(model_config["model"]))
    source_groups = [int(item["image_group"]) for item in frames]
    frame_by_local_group = {
        local_group: frame for local_group, frame in enumerate(frames, start=1)
    }
    if raw_path.is_file():
        try:
            cached = read_json(raw_path)
            mapping = cached.get("batch_group_mapping")
            observations = cached.get("observation", {}).get("observations")
            expected_mapping = {
                str(local_group): source_group
                for local_group, source_group in enumerate(source_groups, start=1)
            }
            source_to_local = {
                source_group: local_group
                for local_group, source_group in enumerate(source_groups, start=1)
            }
            localized = []
            for item in observations or []:
                if not isinstance(item, dict):
                    continue
                local_group = source_to_local.get(item.get("image_group"))
                localized_item = {**item, "image_group": local_group}
                if isinstance(local_group, int) and local_group in frame_by_local_group:
                    _bind_observation_to_frame(
                        localized_item, frame_by_local_group[local_group]
                    )
                localized.append(localized_item)
            if (
                cached.get("algorithm_version") == ALGORITHM_VERSION
                and cached.get("execution_fingerprint") == execution_fingerprint
                and mapping == expected_mapping
                and cached.get("fallback_used") is not True
                and isinstance(observations, list)
                and not validate_observation(
                    {"observations": localized},
                    set(range(1, len(source_groups) + 1)),
                )
            ):
                rebound = [
                    {
                        **item,
                        "image_group": source_groups[int(item["image_group"]) - 1],
                        "local_image_group": int(item["image_group"]),
                    }
                    for item in localized
                ]
                cached["observation"] = {"observations": rebound}
                cached["frame_binding_applied"] = True
                write_json(raw_path, cached)
                return rebound, cached
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    local_frames = [
        {**item, "image_group": local_group}
        for local_group, item in enumerate(frames, start=1)
    ]
    expected = set(range(1, len(local_frames) + 1))
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": _prompt(
                local_frames,
                dense_confirmation=dense_confirmation,
                skill_instruction=skill_instruction,
            ),
        }
    ]
    media: list[dict[str, Any]] = []
    for item, source_group in zip(local_frames, source_groups):
        group = int(item["image_group"])
        panorama = Path(item["enhanced_frame_path"])
        content.append({"type": "text", "text": f"Image group {group}: enhanced panorama."})
        content.append({"type": "image_url", "image_url": {"url": image_data_url(panorama)}})
        dynamic_candidates = [
            {
                key: candidate.get(key)
                for key in (
                    "candidate_id",
                    "track_id",
                    "bbox_xyxy_normalized",
                    "identity",
                    "identity_basis",
                    "roi_source",
                    "model_roi_rank",
                    "model_roi_selected",
                    "model_roi_bbox_xyxy_normalized",
                    "model_roi_role",
                    "model_roi_selection_basis",
                )
            }
            for candidate in item.get("device_localizations", [])
            if isinstance(candidate, dict)
        ]
        content.append(
            {
                "type": "text",
                "text": (
                    f"Image group {group}: current-frame dynamic candidates (CV hints, verify visually): "
                    + json.dumps(dynamic_candidates, ensure_ascii=False)
                ),
            }
        )
        media.append(
            {
                "image_group": source_group,
                "local_image_group": group,
                "role": "enhanced_panorama",
                "path": str(panorama),
                "dynamic_candidates": dynamic_candidates,
            }
        )
        sheet_value = item.get("device_roi_sheet_path")
        if isinstance(sheet_value, str) and Path(sheet_value).is_file():
            sheet = Path(sheet_value)
            content.append(
                {"type": "text", "text": f"Image group {group}: candidate orange-device ROI sheet."}
            )
            content.append({"type": "image_url", "image_url": {"url": image_data_url(sheet, 1200)}})
            media.append(
                {
                    "image_group": source_group,
                    "local_image_group": group,
                    "role": "device_roi_sheet",
                    "path": str(sheet),
                }
            )
        local_roi_candidates = sorted(
            [
                candidate
                for candidate in item.get("device_localizations", [])
                if isinstance(candidate, dict)
                and candidate.get("model_roi_selected") is True
                and isinstance(candidate.get("enhanced_roi_path"), str)
                and Path(str(candidate["enhanced_roi_path"])).is_file()
            ],
            key=lambda candidate: int(candidate.get("model_roi_rank") or 999),
        )
        for candidate in local_roi_candidates:
            local_roi = Path(str(candidate["enhanced_roi_path"]))
            candidate_id = str(candidate.get("candidate_id") or "unknown")
            content.append(
                {
                    "type": "text",
                    "text": (
                        f"Image group {group}: locally enlarged {candidate_id} from this same frame; "
                        "it is an alternate view, not another vote."
                    ),
                }
            )
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_data_url(local_roi, 1600, 92)},
                }
            )
            media.append(
                {
                    "image_group": source_group,
                    "local_image_group": group,
                    "role": "enhanced_device_roi",
                    "candidate_id": candidate_id,
                    "path": str(local_roi),
                }
            )
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 8000 if dense_confirmation else 7000,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    attempts: list[dict[str, Any]] = []
    recovered_local: dict[int, dict[str, Any]] = {}
    for attempt in range(2):
        if attempt:
            payload["messages"][0]["content"].append(
                {
                    "type": "text",
                    "text": (
                        "Schema correction: return exactly one JSON object and one observation for each local "
                        f"image_group {sorted(expected)}. Use only the enumerated values."
                    ),
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
        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            http.client.RemoteDisconnected,
            TimeoutError,
        ) as exc:
            attempts.append(
                {
                    "attempt": attempt + 1,
                    "errors": [f"transport_error:{type(exc).__name__}:{getattr(exc, 'code', '')}"],
                }
            )
            if attempt == 0:
                time.sleep(2.0)
                continue
            break
        choices = raw.get("choices") if isinstance(raw, dict) else None
        choice = choices[0] if isinstance(choices, list) and choices else None
        message = choice.get("message") if isinstance(choice, dict) else None
        text = message.get("content") if isinstance(message, dict) else None
        normalizations: dict[str, list[str]] = {}
        partial_values: list[dict[str, Any]] = []
        if not isinstance(text, str):
            parsed = None
            errors = ["response_content_not_text"]
        else:
            try:
                parsed = parse_json_object(text)
                for observation in parsed.get("observations", []):
                    if isinstance(observation, dict):
                        changes = _normalize_model_observation(observation)
                        group = observation.get("image_group")
                        if isinstance(group, int) and group in frame_by_local_group:
                            changes.extend(
                                _bind_observation_to_frame(
                                    observation, frame_by_local_group[group]
                                )
                            )
                        if changes:
                            normalizations[str(observation.get("image_group"))] = changes
                errors = validate_observation(parsed, expected)
            except (ValueError, json.JSONDecodeError) as exc:
                parsed = None
                errors = [f"parse_error:{type(exc).__name__}"]
                partial_values = _partial_observation_objects(text)
                for observation in partial_values:
                    changes = _normalize_model_observation(observation)
                    group = observation.get("image_group")
                    if isinstance(group, int) and group in frame_by_local_group:
                        changes.extend(
                            _bind_observation_to_frame(
                                observation, frame_by_local_group[group]
                            )
                        )
                    if changes:
                        normalizations[str(observation.get("image_group"))] = changes
        candidates = (
            parsed.get("observations", [])
            if isinstance(parsed, dict) and isinstance(parsed.get("observations"), list)
            else partial_values
        )
        for observation in candidates:
            if not isinstance(observation, dict):
                continue
            local_group = observation.get("image_group")
            if (
                isinstance(local_group, int)
                and not isinstance(local_group, bool)
                and local_group in expected
                and not validate_observation({"observations": [observation]}, {local_group})
            ):
                recovered_local.setdefault(local_group, observation)
        attempts.append(
            {
                "attempt": attempt + 1,
                "finish_reason": choice.get("finish_reason") if isinstance(choice, dict) else None,
                "content": text,
                "schema_errors": errors,
                "normalizations": normalizations,
                "partial_groups_recovered": sorted(recovered_local),
            }
        )
        if not errors and parsed is not None:
            remapped: list[dict[str, Any]] = []
            for observation in parsed["observations"]:
                local_group = int(observation["image_group"])
                remapped.append(
                    {
                        **observation,
                        "image_group": source_groups[local_group - 1],
                        "local_image_group": local_group,
                    }
                )
            artifact = {
                "algorithm_version": ALGORITHM_VERSION,
                "model": model,
                "base_url": base_url,
                "batch_group_mapping": {
                    str(local_group): source_group
                    for local_group, source_group in enumerate(source_groups, start=1)
                },
                "media": media,
                "attempts": attempts,
                "observation": {"observations": remapped},
                "frame_binding_applied": True,
            }
            if execution_fingerprint is not None:
                artifact["execution_fingerprint"] = execution_fingerprint
            write_json(raw_path, artifact)
            return remapped, artifact
    missing_local = sorted(expected - set(recovered_local))
    fallback: list[dict[str, Any]] = []
    for local_group, source_group in enumerate(source_groups, start=1):
        observation = dict(
            recovered_local.get(local_group)
            or _fallback_observation(
                local_group, "Qwen image group unavailable after one targeted retry."
            )
        )
        _bind_observation_to_frame(observation, frame_by_local_group[local_group])
        fallback.append(
            {
                **observation,
                "image_group": source_group,
                "local_image_group": local_group,
            }
        )
    artifact = {
        "algorithm_version": ALGORITHM_VERSION,
        "model": model,
        "base_url": base_url,
        "batch_group_mapping": {
            str(local_group): source_group
            for local_group, source_group in enumerate(source_groups, start=1)
        },
        "media": media,
        "attempts": attempts,
        "observation": {"observations": fallback},
        "fallback_used": bool(missing_local),
        "partial_recovery_used": bool(recovered_local),
        "fallback_image_groups": [source_groups[group - 1] for group in missing_local],
        "frame_binding_applied": True,
    }
    if execution_fingerprint is not None:
        artifact["execution_fingerprint"] = execution_fingerprint
    write_json(raw_path, artifact)
    return fallback, artifact


def call_qwen(
    frames: list[dict[str, Any]],
    model_config: dict[str, Any],
    evidence_dir: Path,
    batch_size: int = 4,
    dense_confirmation: bool = False,
    artifact_prefix: str = "batch",
    skill_instruction: str = "",
    execution_fingerprint: str | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    observations: list[dict[str, Any]] = []
    artifacts: list[str] = []
    for batch_index, start in enumerate(range(0, len(frames), batch_size), start=1):
        batch = frames[start : start + batch_size]
        raw_path = evidence_dir / "qwen_batches" / f"{artifact_prefix}_{batch_index:03d}.json"
        values, _ = _call_qwen_batch(
            batch,
            model_config,
            raw_path,
            dense_confirmation=dense_confirmation,
            skill_instruction=skill_instruction,
            execution_fingerprint=execution_fingerprint,
        )
        observations.extend(values)
        artifacts.append(str(raw_path.resolve()))
    return observations, artifacts


def call_qwen_stage_aware(
    frames: list[dict[str, Any]],
    model_config: dict[str, Any],
    evidence_dir: Path,
    *,
    observation_batch_size: int = 1,
    skill_instruction: str = "",
) -> tuple[list[dict[str, Any]], list[str]]:
    """Give measurement evidence focused calls while batching other context."""
    observation_frames = [item for item in frames if _is_observation_stage(item)]
    context_frames = [item for item in frames if not _is_observation_stage(item)]
    observations: list[dict[str, Any]] = []
    artifacts: list[str] = []
    if context_frames:
        values, paths = call_qwen(
            context_frames,
            model_config,
            evidence_dir,
            artifact_prefix="context",
            skill_instruction=skill_instruction,
            execution_fingerprint=None,
        )
        observations.extend(values)
        artifacts.extend(paths)
    if observation_frames:
        values, paths = call_qwen(
            observation_frames,
            model_config,
            evidence_dir,
            batch_size=observation_batch_size,
            artifact_prefix="observation",
            skill_instruction=skill_instruction,
            execution_fingerprint=None,
        )
        observations.extend(values)
        artifacts.extend(paths)
    observations.sort(key=lambda item: int(item.get("image_group") or 0))
    return observations, artifacts


def dense_confirmation_samples(
    observations: list[dict[str, Any]],
    frames: list[dict[str, Any]],
    radius_seconds: float = 1.5,
    interval_seconds: float = 0.5,
) -> list[dict[str, Any]]:
    by_group = {int(item["image_group"]): item for item in frames}
    triggers: list[dict[str, Any]] = []
    for item in observations:
        if (
            item.get("direct_across_state") not in {"candidate", "confirmed"}
            and not _terminal_direct_pair_support(item)
            and not _occluded_pair_support(item)
        ):
            continue
        frame = by_group.get(int(item.get("image_group") or -1))
        if not frame or frame.get("model_fallback_used") is True:
            continue
        triggers.append(frame)
    output: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for trigger_index, frame in enumerate(triggers, start=1):
        center = float(frame["timestamp_seconds"])
        start = max(float(frame["start_seconds"]), center - radius_seconds)
        end = min(float(frame.get("review_end_seconds", frame["end_seconds"])), center + radius_seconds)
        cursor = start
        while cursor <= end + 1e-6:
            timestamp = round(cursor, 3)
            key = (str(frame["window_id"]), int(round(timestamp * 1000)))
            if key not in seen:
                seen.add(key)
                output.append(
                    {
                        **{key: value for key, value in frame.items() if not key.endswith("_path")},
                        "timestamp_seconds": timestamp,
                        "temporal_role": "dense_direct_confirmation",
                        "evidence_phase": "dense_confirmation",
                        "trigger_index": trigger_index,
                    }
                )
            cursor += interval_seconds
    return output


def _deduplicate(
    observations: list[dict[str, Any]], frames: list[dict[str, Any]]
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    by_group = {int(item["image_group"]): item for item in frames}
    chosen: dict[tuple[str, int], tuple[dict[str, Any], dict[str, Any]]] = {}
    for observation in observations:
        frame = by_group.get(int(observation.get("image_group") or -1))
        if frame is None:
            continue
        key = (str(frame.get("window_id")), int(round(float(frame["timestamp_seconds"]) * 1000)))
        current = chosen.get(key)
        preference = (
            frame.get("evidence_phase") == "dense_confirmation",
            frame.get("model_fallback_used") is not True,
            float(observation.get("confidence") or 0.0),
        )
        if current is None:
            chosen[key] = (observation, frame)
            continue
        old_observation, old_frame = current
        old_preference = (
            old_frame.get("evidence_phase") == "dense_confirmation",
            old_frame.get("model_fallback_used") is not True,
            float(old_observation.get("confidence") or 0.0),
        )
        if preference > old_preference:
            chosen[key] = (observation, frame)
    return sorted(chosen.values(), key=lambda pair: float(pair[1]["timestamp_seconds"]))


def _terminal_frame_support(observation: dict[str, Any]) -> bool:
    if _frame_bound_support(observation, {"ammeter", "battery_holder"}):
        return True
    binding = observation.get("frame_binding")
    devices = observation.get("core_devices_visible")
    if not isinstance(binding, dict) or not isinstance(devices, list):
        return False
    bound = set(binding.get("bound_device_identities") or [])
    return (
        not binding.get("unsupported_model_track_ids")
        and not binding.get("conflicting_track_ids")
        and {"ammeter", "battery_holder"}.issubset(set(devices))
        and bool(bound.intersection({"ammeter", "battery_holder"}))
    )


def _terminal_direct_pair_support(observation: dict[str, Any]) -> bool:
    if not _terminal_frame_support(observation):
        return False
    terminals = observation.get("terminal_evidence")
    if not isinstance(terminals, list):
        return False
    strict = observation.get("strict_verification")
    strict_flags = (
        "ammeter_present",
        "battery_holder_present",
        "two_distinct_ammeter_terminals_connected",
        "two_distinct_battery_terminals_connected",
        "direct_pairing_visible",
    )
    if (
        not isinstance(strict, dict)
        or strict.get("verification") != "confirmed"
        or not all(strict.get(name) is True for name in strict_flags)
        or observation.get("path_relation") != "direct"
        or bool(observation.get("intermediate_components"))
    ):
        return False
    direct_pairs: set[tuple[str, str]] = set()
    aggregate_ammeter_ids: set[str] = set()
    aggregate_battery_ids: set[str] = set()
    for item in terminals:
        if not isinstance(item, dict):
            continue
        direct_terminal = (
            item.get("connection_state") == "seated"
            and item.get("path_visibility") == "continuous"
            and item.get("path_relation") == "direct"
            and not item.get("intermediate_components")
            and item.get("passes_through_other_device") is False
        )
        if direct_terminal and item.get("device") == "ammeter":
            ammeter_terminal = str(item.get("ammeter_terminal_id") or "").strip()
            if ammeter_terminal:
                aggregate_ammeter_ids.add(ammeter_terminal)
        if direct_terminal and item.get("device") == "battery_holder":
            battery_terminal = str(item.get("battery_terminal_id") or "").strip()
            if battery_terminal:
                aggregate_battery_ids.add(battery_terminal)
        if (
            item.get("device") == "ammeter"
            and direct_terminal
            and item.get("far_endpoint") == "battery_holder"
        ):
            ammeter_terminal = str(item.get("ammeter_terminal_id") or "").strip()
            battery_terminal = str(item.get("battery_terminal_id") or "").strip()
            if ammeter_terminal and battery_terminal:
                direct_pairs.add((ammeter_terminal, battery_terminal))
    explicit_pairs = (
        len(direct_pairs) >= 2
        and len({pair[0] for pair in direct_pairs}) >= 2
        and len({pair[1] for pair in direct_pairs}) >= 2
    )
    split_rows = (
        observation.get("direct_across_state") == "confirmed"
        and len(aggregate_ammeter_ids) >= 2
        and len(aggregate_battery_ids) >= 2
    )
    return explicit_pairs or split_rows


def _ammeter_loose_lead_support(observation: dict[str, Any]) -> bool:
    """Return true only for a current-frame, continuously visible loose ammeter lead."""
    if not _frame_bound_support(observation, {"ammeter"}):
        return False
    endpoints = observation.get("loose_lead_endpoints")
    if not isinstance(endpoints, list):
        return False
    for endpoint in endpoints:
        if not isinstance(endpoint, dict):
            continue
        terminal_id = str(endpoint.get("ammeter_terminal_id") or "").strip()
        if (
            endpoint.get("source_device") == "ammeter"
            and terminal_id
            and endpoint.get("source_connection_state") == "seated"
            and endpoint.get("far_end_state") == "loose_plug"
            and endpoint.get("path_visibility") == "continuous"
        ):
            return True
    return False


def _ambiguous_direct_endpoint_claim(observation: dict[str, Any]) -> bool:
    """Detect split endpoint claims that conflict with the reported topology."""
    if (
        _terminal_direct_pair_support(observation)
        or not _frame_bound_support(observation, {"ammeter", "battery_holder"})
    ):
        return False
    strict = observation.get("strict_verification")
    if not isinstance(strict, dict) or not all(
        strict.get(name) is True
        for name in (
            "ammeter_present",
            "battery_holder_present",
            "two_distinct_ammeter_terminals_connected",
            "two_distinct_battery_terminals_connected",
        )
    ):
        return False
    ammeter_ids: set[str] = set()
    battery_ids: set[str] = set()
    terminals = observation.get("terminal_evidence")
    if not isinstance(terminals, list):
        return False
    for item in terminals:
        if not isinstance(item, dict) or item.get("connection_state") != "seated":
            continue
        battery_id = str(item.get("battery_terminal_id") or "").strip()
        if battery_id:
            battery_ids.add(battery_id)
        if (
            item.get("device") == "ammeter"
            and item.get("far_endpoint") == "battery_holder"
            and item.get("path_visibility") == "continuous"
            and item.get("path_relation") == "direct"
            and not item.get("intermediate_components")
            and item.get("passes_through_other_device") is False
        ):
            ammeter_id = str(item.get("ammeter_terminal_id") or "").strip()
            if ammeter_id:
                ammeter_ids.add(ammeter_id)
    return len(ammeter_ids) >= 2 and len(battery_ids) >= 2


def _occluded_pair_support(observation: dict[str, Any]) -> bool:
    """Return true only when both endpoint sets are visible but the middle path is occluded."""
    if not _frame_bound_support(observation, {"ammeter", "battery_holder"}):
        return False
    if (
        observation.get("path_relation") != "occluded_likely_direct"
        or bool(observation.get("intermediate_components"))
    ):
        return False
    strict = observation.get("strict_verification")
    if not isinstance(strict, dict) or not all(
        strict.get(name) is True
        for name in (
            "ammeter_present",
            "battery_holder_present",
            "two_distinct_ammeter_terminals_connected",
            "two_distinct_battery_terminals_connected",
        )
    ):
        return False
    ammeter_ids: set[str] = set()
    battery_ids: set[str] = set()
    for item in observation.get("terminal_evidence", []):
        if not isinstance(item, dict) or item.get("connection_state") != "seated":
            continue
        if item.get("path_relation") == "via_component" or item.get("intermediate_components"):
            return False
        ammeter_id = str(item.get("ammeter_terminal_id") or "").strip()
        battery_id = str(item.get("battery_terminal_id") or "").strip()
        if ammeter_id:
            ammeter_ids.add(ammeter_id)
        if battery_id:
            battery_ids.add(battery_id)
    return len(ammeter_ids) >= 2 and len(battery_ids) >= 2


def _complete_series_support(observation: dict[str, Any]) -> bool:
    devices = observation.get("core_devices_visible")
    if not (
        isinstance(devices, list)
        and CORE_DEVICES.issubset(set(devices))
        and observation.get("final_topology") == "single_series_loop"
        and observation.get("topology_visibility") in {"sufficient", "partial"}
        and not _terminal_direct_pair_support(observation)
        and not _ammeter_loose_lead_support(observation)
        and not _ambiguous_direct_endpoint_claim(observation)
    ):
        return False
    return _frame_bound_support(observation, CORE_DEVICES)


def _adjacent_clusters(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]], max_gap: float
) -> list[list[tuple[dict[str, Any], dict[str, Any]]]]:
    clusters: list[list[tuple[dict[str, Any], dict[str, Any]]]] = []
    for pair in sorted(pairs, key=lambda value: float(value[1]["timestamp_seconds"])):
        if not clusters:
            clusters.append([pair])
            continue
        previous = clusters[-1][-1][1]
        frame = pair[1]
        delta = float(frame["timestamp_seconds"]) - float(previous["timestamp_seconds"])
        if frame.get("window_id") == previous.get("window_id") and 0 < delta <= max_gap:
            clusters[-1].append(pair)
        else:
            clusters.append([pair])
    return clusters


def reduce_results(
    observations: list[dict[str, Any]],
    frames: list[dict[str, Any]],
    compare_latest_stable_topology: bool = True,
    direct_cluster_max_gap_seconds: float = 0.76,
) -> dict[str, Any]:
    pairs = _deduplicate(observations, frames)
    eligible_pairs = [
        pair
        for pair in pairs
        if pair[1].get("temporal_role")
        in {"stable_candidate", "process_scan", "dense_direct_confirmation"}
        and pair[1].get("model_fallback_used") is not True
    ]
    stable_pairs = [
        pair
        for pair in eligible_pairs
        if pair[0].get("stable_state") is True
        and pair[0].get("hands_or_plugs") in {"hands_away", "handling_seated"}
    ]
    final_pass = [
        pair
        for pair in stable_pairs
        if _complete_series_support(pair[0])
        and float(pair[0].get("confidence") or 0.0) >= 0.55
    ]
    observation_fail_candidates = [
        pair
        for pair in eligible_pairs
        if _is_scored_observation_context(pair[0], pair[1])
        and (
            (
                pair[0].get("final_topology") == "explicit_nonseries"
                and _frame_bound_support(pair[0], CORE_DEVICES)
                and pair[0].get("topology_visibility") == "sufficient"
            )
            or _ammeter_loose_lead_support(pair[0])
        )
        and float(pair[0].get("confidence") or 0.0) >= 0.55
    ]
    # Observation-stage topology errors are permanent rubric violations. A
    # later stable pass frame cannot erase an already observed error.
    final_fail_candidates = observation_fail_candidates
    fail_clusters = [
        cluster for cluster in _adjacent_clusters(final_fail_candidates, 6.1) if len(cluster) >= 2
    ]
    strong_single_fail = [
        pair for pair in final_fail_candidates if float(pair[0].get("confidence") or 0.0) >= 0.92
    ]
    conclusive: list[tuple[str, tuple[dict[str, Any], dict[str, Any]]]] = [
        ("pass", pair) for pair in final_pass
    ]
    for cluster in fail_clusters:
        conclusive.append(("fail", cluster[-1]))
    conclusive.extend(("fail", pair) for pair in strong_single_fail)
    if conclusive:
        final_decision, selected_final = max(
            conclusive,
            key=(
                (lambda value: float(value[1][1]["timestamp_seconds"]))
                if compare_latest_stable_topology
                else (lambda value: float(value[1][0].get("confidence") or 0.0))
            ),
        )
        # Fixed evidence levels are intentionally not presented as calibrated
        # model probabilities.
        final_confidence = 0.88 if final_decision == "pass" else 0.92
        final_reason = (
            "latest_stable_frame_supports_single_series_loop"
            if final_decision == "pass"
            else "latest_stable_direct_evidence_shows_nonseries_topology"
        )
    else:
        final_decision = "pass"
        selected_final = None
        final_confidence = 0.55 if pairs else 0.35
        final_reason = "no_direct_stable_nonseries_evidence_binary_tie_break_pass"

    supported_direct = [
        pair
        for pair in pairs
        if pair[1].get("model_fallback_used") is not True
        and _terminal_direct_pair_support(pair[0])
    ]
    # One strict current-frame confirmation is sufficient: direct connection
    # to both battery terminals is an irreversible R1 violation.
    direct_clusters = _adjacent_clusters(supported_direct, direct_cluster_max_gap_seconds)
    occluded_pairs = [
        pair
        for pair in pairs
        if pair[1].get("model_fallback_used") is not True
        and _occluded_pair_support(pair[0])
        and float(pair[0].get("confidence") or 0.0) >= 0.75
    ]
    occluded_coarse = [
        pair for pair in occluded_pairs if pair[1].get("evidence_phase") == "coarse_scan"
    ]
    occluded_dense_clusters = [
        cluster
        for cluster in _adjacent_clusters(
            [
                pair
                for pair in occluded_pairs
                if pair[1].get("evidence_phase") == "dense_confirmation"
            ],
            direct_cluster_max_gap_seconds,
        )
        if len(cluster) >= 2
    ]
    strong_final_series_counterevidence = any(
        pair[0].get("topology_visibility") == "sufficient"
        and float(pair[0].get("confidence") or 0.0) >= 0.75
        for pair in final_pass
    )
    occlusion_confirmations: list[
        tuple[
            tuple[dict[str, Any], dict[str, Any]],
            list[tuple[dict[str, Any], dict[str, Any]]],
        ]
    ] = []
    if not strong_final_series_counterevidence:
        for coarse_pair in occluded_coarse:
            for cluster in occluded_dense_clusters:
                if coarse_pair[1].get("window_id") != cluster[0][1].get("window_id"):
                    continue
                if min(
                    abs(
                        float(pair[1]["timestamp_seconds"])
                        - float(coarse_pair[1]["timestamp_seconds"])
                    )
                    for pair in cluster
                ) <= 4.0:
                    occlusion_confirmations.append((coarse_pair, cluster))
                    break
    temporary_decision = "fail" if direct_clusters or occlusion_confirmations else "pass"
    if direct_clusters:
        temporary_confidence = 0.94
        temporary_reason = (
            "adjacent_frames_confirm_two_seated_direct_pairs"
            if any(len(cluster) >= 2 for cluster in direct_clusters)
            else "single_current_frame_confirms_two_seated_direct_pairs"
        )
    elif occlusion_confirmations:
        temporary_confidence = 0.82
        temporary_reason = "coarse_and_dense_endpoint_evidence_corroborates_occluded_direct_paths"
    else:
        temporary_confidence = 0.72 if occluded_pairs else 0.6
        temporary_reason = "no_adjacent_dense_sequence_confirms_two_seated_direct_pairs"
    suppressed = []
    confirmed_keys = {
        (str(pair[1].get("window_id")), float(pair[1]["timestamp_seconds"]))
        for cluster in direct_clusters
        for pair in cluster
    }
    for observation, frame in pairs:
        ambiguous_endpoint_claim = _ambiguous_direct_endpoint_claim(observation)
        if (
            observation.get("direct_across_state") not in {"candidate", "confirmed"}
            and not _terminal_direct_pair_support(observation)
            and not _occluded_pair_support(observation)
            and not ambiguous_endpoint_claim
        ):
            continue
        key = (str(frame.get("window_id")), float(frame["timestamp_seconds"]))
        if key in confirmed_keys:
            continue
        terminal_states = sorted(
            {
                str(item.get("connection_state"))
                for item in observation.get("terminal_evidence", [])
                if isinstance(item, dict)
            }
        )
        suppressed.append(
            {
                "timestamp_seconds": frame["timestamp_seconds"],
                "window_id": frame.get("window_id"),
                "evidence_phase": frame.get("evidence_phase"),
                "direct_across_state": observation.get("direct_across_state"),
                "terminal_states": terminal_states,
                "terminal_pair_support": _terminal_direct_pair_support(observation),
                "ambiguous_direct_endpoint_claim": ambiguous_endpoint_claim,
                "reason": (
                    "conflicting_split_endpoint_claim_requires_dense_confirmation"
                    if ambiguous_endpoint_claim
                    else "not_part_of_two_frame_seated_direct_pair_cluster"
                ),
            }
        )
    if direct_clusters:
        decision = "fail"
        confidence = temporary_confidence
        reason = "confirmed_process_direct_across_battery"
        decision_branch = "direct_violation"
        decisive_pairs = list(direct_clusters[-1])
        path_relation = "direct"
    elif observation_fail_candidates:
        selected_observation_fail = min(
            observation_fail_candidates,
            key=lambda pair: float(pair[1]["timestamp_seconds"]),
        )
        decision = "fail"
        confidence = max(
            0.7,
            float(selected_observation_fail[0].get("confidence") or 0.0),
        )
        reason = (
            "observation_stage_loose_ammeter_lead"
            if _ammeter_loose_lead_support(selected_observation_fail[0])
            else "observation_stage_nonseries_topology"
        )
        decision_branch = "observation_violation"
        # Keep the first observed error as the decisive evidence. The failure
        # remains true even when later frames support a valid series loop.
        decisive_pairs = [selected_observation_fail]
        path_relation = str(decisive_pairs[0][0].get("path_relation") or "unclear")
    elif final_decision == "fail":
        decision = "fail"
        confidence = final_confidence
        reason = final_reason
        decision_branch = "direct_violation"
        decisive_pairs = [selected_final] if selected_final is not None else []
        path_relation = str(selected_final[0].get("path_relation") or "unclear") if selected_final else "unclear"
    elif occlusion_confirmations:
        decision = "fail"
        confidence = temporary_confidence
        reason = "corroborated_occluded_process_direct_across_battery"
        decision_branch = "occlusion_corroboration"
        coarse_pair, dense_cluster = occlusion_confirmations[-1]
        decisive_pairs = [coarse_pair, *dense_cluster]
        path_relation = "occluded_likely_direct"
    elif temporary_decision == "fail":
        decision = "fail"
        confidence = temporary_confidence
        reason = "confirmed_process_direct_across_battery"
        decision_branch = "direct_violation"
        decisive_pairs = []
        path_relation = "direct"
    else:
        decision = "pass"
        confidence = final_confidence
        reason = final_reason
        decision_branch = "binary_fallback"
        # A binary tie-break is still required, but an arbitrary last frame is
        # not decision evidence when it did not pass the stable-topology checks.
        decisive_pairs = [selected_final] if selected_final is not None else []
        path_relation = (
            "via_component"
            if selected_final is not None and final_decision == "pass"
            else "unclear"
        )

    unique_decisive: list[tuple[dict[str, Any], dict[str, Any]]] = []
    seen_decisive: set[tuple[str, int]] = set()
    for pair in decisive_pairs:
        if pair is None:
            continue
        key = (
            str(pair[1].get("window_id")),
            int(round(float(pair[1]["timestamp_seconds"]) * 1000)),
        )
        if key not in seen_decisive:
            seen_decisive.add(key)
            unique_decisive.append(pair)
    supporting_frame_ids = [
        str(
            frame.get("frame_id")
            or f"frame_{int(frame.get('frame_number') or 0):08d}"
        )
        for _, frame in unique_decisive
    ]
    supporting_timestamps = [
        round(float(frame["timestamp_seconds"]), 3) for _, frame in unique_decisive
    ]
    direct_observations: list[dict[str, Any]] = []
    for observation, frame in unique_decisive:
        visible = observation.get("direct_observations")
        facts = visible if isinstance(visible, list) and visible else [observation.get("evidence")]
        if _ammeter_loose_lead_support(observation):
            facts = [
                *facts,
                *[
                    endpoint.get("evidence")
                    for endpoint in observation.get("loose_lead_endpoints", [])
                    if isinstance(endpoint, dict)
                ],
            ]
        for fact in facts:
            if isinstance(fact, str) and fact.strip():
                direct_observations.append(
                    {
                        "frame_id": str(
                            frame.get("frame_id")
                            or f"frame_{int(frame.get('frame_number') or 0):08d}"
                        ),
                        "timestamp_seconds": round(float(frame["timestamp_seconds"]), 3),
                        "observation": fact,
                    }
                )
    derived_observations: list[dict[str, Any]] = [
        {
            "rule": reason,
            "decision_branch": decision_branch,
            "source": "current_run_evidence_only",
        }
    ]
    for observation, frame in unique_decisive:
        for value in observation.get("derived_observations", []):
            if isinstance(value, str) and value.strip():
                derived_observations.append(
                    {
                        "frame_id": str(
                            frame.get("frame_id")
                            or f"frame_{int(frame.get('frame_number') or 0):08d}"
                        ),
                        "timestamp_seconds": round(float(frame["timestamp_seconds"]), 3),
                        "observation": value,
                    }
                )
    return {
        "decision": decision,
        "predicted_score": 1 if decision == "pass" else 0,
        "binary_score": 1 if decision == "pass" else 0,
        "confidence": round(confidence, 3),
        "final_series_circuit": final_decision,
        "temporary_direct_across_battery": temporary_decision,
        "decision_branch": decision_branch,
        "path_relation": path_relation if path_relation in PATH_RELATIONS else "unclear",
        "direct_observations": direct_observations,
        "derived_observations": derived_observations,
        "supporting_frame_ids": supporting_frame_ids,
        "supporting_timestamps_seconds": supporting_timestamps,
        "reason": reason,
        "diagnostics": {
            "scoring_policy": "monotonic_fail_on_confirmed_direct_across_or_observation_nonseries",
            "confidence_policy": "fixed_identity_independent_evidence_levels_not_calibrated_model_probability",
            "observation_count": len(pairs),
            "decisive_evidence_available": bool(unique_decisive),
            "final_series_circuit": {
                "decision": final_decision,
                "confidence": round(final_confidence, 3),
                "reason": final_reason,
                "selected_timestamp_seconds": (
                    selected_final[1]["timestamp_seconds"] if selected_final is not None else None
                ),
                "stable_pass_support_count": len(final_pass),
                "stable_fail_support_count": len(final_fail_candidates),
                "confirmed_fail_cluster_count": len(fail_clusters),
            },
            "temporary_direct_across_battery": {
                "decision": temporary_decision,
                "confidence": temporary_confidence,
                "reason": temporary_reason,
                "confirmed_sequence_count": len(direct_clusters),
                "confirmed_sequences": [
                    [float(pair[1]["timestamp_seconds"]) for pair in cluster]
                    for cluster in direct_clusters
                ],
                "suppressed_candidates": suppressed,
                "occlusion_confirmation_count": len(occlusion_confirmations),
                "strong_final_series_counterevidence": strong_final_series_counterevidence,
            },
            "observation_stage_violation": {
                "decision": "fail" if observation_fail_candidates else "pass",
                "confirmed_frame_count": len(observation_fail_candidates),
                "loose_ammeter_lead_frame_count": sum(
                    _ammeter_loose_lead_support(pair[0])
                    for pair in observation_fail_candidates
                ),
                "explicit_nonseries_frame_count": sum(
                    pair[0].get("final_topology") == "explicit_nonseries"
                    for pair in observation_fail_candidates
                ),
                "timestamps_seconds": [
                    round(float(pair[1]["timestamp_seconds"]), 3)
                    for pair in observation_fail_candidates
                ],
                "activity_contexts": [
                    str(pair[0].get("activity_context") or "unclear")
                    for pair in observation_fail_candidates
                ],
                "later_pass_can_override": False,
            },
        },
    }


def _fallback_groups(artifact_paths: list[str]) -> set[int]:
    groups: set[int] = set()
    for artifact_path in artifact_paths:
        artifact = read_json(Path(artifact_path))
        if artifact.get("fallback_used") is not True:
            continue
        explicit = artifact.get("fallback_image_groups")
        if isinstance(explicit, list):
            groups.update(
                int(value)
                for value in explicit
                if isinstance(value, int) and not isinstance(value, bool)
            )
        else:
            groups.update(
                int(item["image_group"])
                for item in artifact.get("media", [])
                if isinstance(item, dict) and isinstance(item.get("image_group"), int)
            )
    return groups


def _run_adaptive_series_rubric(
    *,
    video_path: Path,
    source_video_id: str,
    video_id: str,
    evidence_dir: Path,
    duration_seconds: float,
    record: dict[str, Any],
    boundary_record: dict[str, Any] | None,
    action_path: Path,
    boundary_summary_path: Path | None,
    model_config: dict[str, Any],
    execution: dict[str, Any],
    skill_plan: dict[str, Any] | None,
    windows: list[dict[str, Any]],
) -> dict[str, Any]:
    try:
        from . import r1_frame_sampling_agent as frame_agent
    except ImportError:
        import r1_frame_sampling_agent as frame_agent  # type: ignore

    parameters = execution["parameters"]
    scout_samples = sampling_timestamps(
        windows,
        interval_seconds=float(parameters["sampling_interval_seconds"]),
        max_samples_per_window=int(parameters["max_samples_per_window"]),
    )
    scanned = _scan_frame_metadata(video_path, scout_samples)
    if not scanned:
        raise RuntimeError("R1 current-run 2 FPS scout produced no readable frames")
    frame_plan = frame_agent.select_initial_evidence(
        scanned,
        stable_per_stage_run=int(parameters["stable_frames_per_stage_run"]),
        recovery_per_stage_run=int(parameters["view_recovery_frames_per_stage_run"]),
        max_transition_anchors=int(parameters["max_transition_anchors"]),
    )
    frame_agent_dir = evidence_dir / "frame_agent"
    base_samples = [
        item
        for item in frame_plan["selected_frames"]
        if item.get("frame_agent_role") != "connection_transition"
    ]
    selected_frames = _decode_and_export(
        video_path,
        base_samples,
        frame_agent_dir / "selected",
        roi_target_long_edge=int(parameters["roi_target_long_edge"]),
        max_model_roi_views_per_frame=int(parameters["max_model_roi_views_per_frame"]),
    )
    transition_samples = frame_agent.transition_burst_samples(
        frame_plan["transition_anchors"],
        duration_seconds=duration_seconds,
        fps=float(parameters["transition_sampling_fps"]),
        radius_seconds=float(parameters["transition_radius_seconds"]),
    )
    transition_frames = _decode_and_export(
        video_path,
        transition_samples,
        frame_agent_dir / "transition_bursts",
        roi_target_long_edge=int(parameters["roi_target_long_edge"]),
        max_model_roi_views_per_frame=int(parameters["max_model_roi_views_per_frame"]),
    )
    initial_frames = selected_frames + transition_frames
    if not initial_frames:
        raise RuntimeError("R1 adaptive frame Agent selected no readable evidence")
    for image_group, item in enumerate(initial_frames, start=1):
        item["image_group"] = image_group
    initial_observations, initial_artifacts = call_qwen_stage_aware(
        initial_frames,
        model_config,
        frame_agent_dir / "initial_qwen",
        observation_batch_size=int(parameters["observation_model_batch_size"]),
        skill_instruction=str(parameters["prompt_instruction"]),
    )
    initial_fallback = _fallback_groups(initial_artifacts)
    for item in initial_frames:
        item["model_fallback_used"] = int(item["image_group"]) in initial_fallback

    supplemental_plan = {
        "schema_version": "resistance_agent_r1_supplemental_round.v1",
        "round_number": 0,
        "selection_basis": "current_run_qwen_observations_only",
        "reasons": [],
        "frames": [],
        "frame_count": 0,
        "max_rounds": 1,
        "video_id_used_for_routing": False,
        "historical_artifacts_used": False,
        "fixed_video_roi_used": False,
    }
    supplemental_frames: list[dict[str, Any]] = []
    supplemental_observations: list[dict[str, Any]] = []
    supplemental_artifacts: list[str] = []
    if int(parameters["max_supplemental_rounds"]) > 0:
        supplemental_plan = frame_agent.plan_supplemental_round(
            initial_observations,
            initial_frames,
            duration_seconds=duration_seconds,
            max_frames=int(parameters["max_supplemental_frames"]),
            fps=float(parameters["transition_sampling_fps"]),
            radius_seconds=float(parameters["transition_radius_seconds"]),
        )
        if supplemental_plan["frames"]:
            supplemental_frames = _decode_and_export(
                video_path,
                supplemental_plan["frames"],
                frame_agent_dir / "supplemental_round",
                roi_target_long_edge=int(parameters["roi_target_long_edge"]),
                max_model_roi_views_per_frame=int(parameters["max_model_roi_views_per_frame"]),
            )
            existing_numbers = {
                int(item["frame_number"])
                for item in initial_frames
                if isinstance(item.get("frame_number"), int)
            }
            supplemental_frames = [
                item
                for item in supplemental_frames
                if not isinstance(item.get("frame_number"), int)
                or int(item["frame_number"]) not in existing_numbers
            ]
            for image_group, item in enumerate(
                supplemental_frames, start=len(initial_frames) + 1
            ):
                item["image_group"] = image_group
            if supplemental_frames:
                supplemental_observations, supplemental_artifacts = call_qwen_stage_aware(
                    supplemental_frames,
                    model_config,
                    frame_agent_dir / "supplemental_qwen",
                    observation_batch_size=int(parameters["observation_model_batch_size"]),
                    skill_instruction=str(parameters["prompt_instruction"]),
                )
                fallback = _fallback_groups(supplemental_artifacts)
                for item in supplemental_frames:
                    item["model_fallback_used"] = int(item["image_group"]) in fallback
        supplemental_plan["decoded_frame_count"] = len(supplemental_frames)

    frames = initial_frames + supplemental_frames
    observations = initial_observations + supplemental_observations
    rubric_1 = reduce_results(
        observations,
        frames,
        compare_latest_stable_topology=bool(parameters["compare_latest_stable_topology"]),
        direct_cluster_max_gap_seconds=float(parameters["direct_cluster_max_gap_seconds"]),
    )
    plan_live_skills = {
        "selection_basis": (skill_plan or {}).get(
            "selection_basis", "current_video_observed_situation_only"
        ),
        "observed_stages": (skill_plan or {}).get(
            "observed_stages", record.get("observed_stage_runs", [])
        ),
        "selected_skills": (skill_plan or {}).get(
            "selected_skills", (skill_plan or {}).get("skills", [])
        ),
        "video_id_used_for_routing": False,
        "historical_artifacts_used": False,
        "fixed_video_roi_used": False,
    }
    rubric_1["plan_live_skills"] = plan_live_skills
    safe_execution = {
        key: value for key, value in execution.items() if "fingerprint" not in key.lower()
    }
    frame_report = {
        "schema_version": frame_agent.SCHEMA_VERSION,
        "agent_version": frame_agent.AGENT_VERSION,
        "selection_basis": "current_video_observed_situation_only",
        "candidate_windows": windows,
        "scout_sampling_fps": round(
            1.0 / float(parameters["sampling_interval_seconds"]), 3
        ),
        "scanned_frame_count": len(scanned),
        "initial_model_frame_count": len(initial_frames),
        "stable_frame_count": len(frame_plan["stable_frames"]),
        "transition_anchor_count": len(frame_plan["transition_anchors"]),
        "transition_frame_count": len(transition_frames),
        "view_recovery_frame_count": len(frame_plan["recovery_frames"]),
        "supplemental_round": supplemental_plan,
        "supplemental_model_frame_count": len(supplemental_frames),
        "total_model_frame_count": len(frames),
        "local_roi_views_sent": sum(
            1
            for frame in frames
            for candidate in frame.get("device_localizations", [])
            if isinstance(candidate, dict) and candidate.get("model_roi_selected") is True
        ),
        "selected_frames": frames,
        "video_id_used_for_routing": False,
        "historical_artifacts_used": False,
        "fixed_video_roi_used": False,
    }
    frame_report_path = frame_agent_dir / "report.json"
    write_json(frame_report_path, frame_report)
    report = {
        "schema_version": "resistance_agent_series_evidence.v2",
        "algorithm_version": ALGORITHM_VERSION,
        "video_id": video_id,
        "source_video_id": source_video_id,
        "source_video_path": str(video_path.resolve()),
        "action_summary_path": str(action_path.resolve()),
        "boundary_summary_path": (
            str(boundary_summary_path.resolve()) if boundary_summary_path else None
        ),
        "boundary_stage_runs_used": boundary_record is not None,
        "excel_accessed": False,
        "ground_truth_sent_to_model": False,
        "candidate_windows": windows,
        "frame_agent_report_path": str(frame_report_path.resolve()),
        "coarse_scan": {
            "sampling_fps": frame_report["scout_sampling_fps"],
            "scanned_frame_count": len(scanned),
            "model_frame_count": len(frames),
            "selection_policy": "stable_transition_view_recovery",
        },
        "scanned_frames": scanned,
        "selected_frames": initial_frames,
        "coarse_qwen_batch_artifacts": initial_artifacts,
        "dense_confirmation_frames": supplemental_frames,
        "dense_qwen_batch_artifacts": supplemental_artifacts,
        "observations": observations,
        "rubric_1": rubric_1,
        "routing_policy": (skill_plan or {}).get("routing_policy"),
        "skill_selection": (skill_plan or {}).get("skills", []),
        "skill_execution": safe_execution,
        "plan_live_skills": plan_live_skills,
        "historical_fallback_used": False,
        "fixed_video_roi_used": False,
    }
    report_path = evidence_dir / "series_evidence_report.json"
    write_json(report_path, report)
    reopened = read_json(report_path)
    if reopened.get("rubric_1", {}).get("decision") not in {"pass", "fail"}:
        raise RuntimeError("series evidence report verification failed")
    return {"rubric_1": rubric_1, "report_path": str(report_path.resolve())}


def run_series_rubric(
    video_path: Path,
    source_video_id: str,
    video_id: str,
    run_dir: Path,
    model_config: dict[str, Any],
    action_summary_path: Path | None = None,
    boundary_summary_path: Path | None = None,
    skill_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        from .skills import EXECUTOR_REGISTRY, execution_for_rubric
    except ImportError:
        from skills import EXECUTOR_REGISTRY, execution_for_rubric  # type: ignore
    execution = (
        execution_for_rubric(skill_plan, 1)
        if skill_plan
        else {
            "skill_id": "series.adaptive_terminal_sampling",
            "parameters": dict(EXECUTOR_REGISTRY["series.adaptive_terminal_sampling"].defaults),
            "execution_fingerprint": None,
        }
    )
    parameters = execution["parameters"]
    action_path = action_summary_path if action_summary_path and action_summary_path.is_file() else None
    if action_path is None or not action_path.is_file():
        raise ValueError("current live action summary is required")
    record = _source_record(read_json(action_path), source_video_id, video_id)
    boundary_record = None
    if boundary_summary_path and boundary_summary_path.is_file():
        boundary_record = _boundary_record(
            read_json(boundary_summary_path),
            source_video_id,
            video_id,
            allowed_root=boundary_summary_path.parent,
        )
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
    evidence_dir = run_dir / "series_rubric"
    windows = candidate_windows(
        record,
        duration,
        str(parameters["window_mode"]),
        coarse_window_seconds=float(parameters["coarse_window_seconds"]),
    )
    if execution["skill_id"] == "series.adaptive_terminal_sampling":
        return _run_adaptive_series_rubric(
            video_path=video_path,
            source_video_id=source_video_id,
            video_id=video_id,
            evidence_dir=evidence_dir,
            duration_seconds=duration,
            record=record,
            boundary_record=boundary_record,
            action_path=action_path,
            boundary_summary_path=boundary_summary_path,
            model_config=model_config,
            execution=execution,
            skill_plan=skill_plan,
            windows=windows,
        )
    source_digest = sha256(video_path)
    stage_window_fingerprint = _json_fingerprint(
        {
            "observed_stage_runs": record.get("observed_stage_runs", []),
            "effective_experiment_interval_seconds": record.get(
                "effective_experiment_interval_seconds"
            ),
            "locked_experiment_interval_seconds": record.get(
                "locked_experiment_interval_seconds"
            ),
            "candidate_windows": windows,
        }
    )
    evidence_fingerprint = _json_fingerprint(
        {
            "skill_execution_fingerprint": execution["execution_fingerprint"],
            "stage_window_fingerprint": stage_window_fingerprint,
        }
    )
    checkpoint_path = evidence_dir / "selected_frames_pre_qwen.json"
    checkpoint = read_json(checkpoint_path) if checkpoint_path.is_file() else {}
    selected = checkpoint.get("selected_frames") if isinstance(checkpoint, dict) else None
    scanned = checkpoint.get("scanned_frames") if isinstance(checkpoint, dict) else None
    checkpoint_valid = (
        checkpoint.get("algorithm_version") == ALGORITHM_VERSION
        and checkpoint.get("source_video_sha256") == source_digest
        and checkpoint.get("routing_policy") == (skill_plan or {}).get("routing_policy")
        and checkpoint.get("execution_fingerprint") == evidence_fingerprint
        and checkpoint.get("stage_window_fingerprint") == stage_window_fingerprint
        and checkpoint.get("candidate_windows") == windows
        and isinstance(scanned, list)
        and bool(scanned)
        and isinstance(selected, list)
        and bool(selected)
        and all(
            isinstance(item, dict)
            and isinstance(item.get("frame_path"), str)
            and Path(item["frame_path"]).is_file()
            for item in selected
        )
    )
    if not checkpoint_valid:
        scanned = _decode_and_export(
            video_path,
            sampling_timestamps(
                windows,
                interval_seconds=float(parameters["sampling_interval_seconds"]),
                max_samples_per_window=int(parameters["max_samples_per_window"]),
            ),
            evidence_dir,
        )
        selected = select_coarse_model_frames(
            scanned,
            per_window_limit=int(parameters["coarse_model_frame_limit"]),
        )
        if not scanned or not selected:
            raise RuntimeError("R1 current-run 2 FPS scan produced no readable frames")
        write_json(
            checkpoint_path,
            {
                "algorithm_version": ALGORITHM_VERSION,
                "source_video_sha256": source_digest,
                "routing_policy": (skill_plan or {}).get("routing_policy"),
                "execution_fingerprint": evidence_fingerprint,
                "stage_window_fingerprint": stage_window_fingerprint,
                "skill_execution": execution,
                "candidate_windows": windows,
                "scanned_frames": scanned,
                "selected_frames": selected,
            },
        )
    coarse_observations, coarse_artifacts = call_qwen(
        selected,
        model_config,
        evidence_dir,
        skill_instruction=str(parameters["prompt_instruction"]),
        execution_fingerprint=evidence_fingerprint,
    )
    coarse_fallback_groups: set[int] = set()
    for artifact_path in coarse_artifacts:
        artifact = read_json(Path(artifact_path))
        if artifact.get("fallback_used") is True:
            explicit = artifact.get("fallback_image_groups")
            if isinstance(explicit, list):
                coarse_fallback_groups.update(
                    int(value) for value in explicit if isinstance(value, int) and not isinstance(value, bool)
                )
            else:
                coarse_fallback_groups.update(
                    int(item["image_group"])
                    for item in artifact.get("media", [])
                    if isinstance(item, dict) and isinstance(item.get("image_group"), int)
                )
    for item in selected:
        item["model_fallback_used"] = int(item["image_group"]) in coarse_fallback_groups
    dense_samples = (
        dense_confirmation_samples(
            coarse_observations,
            selected,
            radius_seconds=float(parameters["dense_radius_seconds"]),
            interval_seconds=1.0 / max(float(parameters["dense_sampling_fps"]), 0.01),
        )
        if parameters["dense_confirmation"]
        else []
    )
    dense_frames: list[dict[str, Any]] = []
    dense_observations: list[dict[str, Any]] = []
    dense_artifacts: list[str] = []
    if dense_samples:
        dense_dir = evidence_dir / "dense_confirmation"
        dense_frames = _decode_and_export(video_path, dense_samples, dense_dir)
        for image_group, item in enumerate(dense_frames, start=len(selected) + 1):
            item["image_group"] = image_group
        dense_observations, dense_artifacts = call_qwen(
            dense_frames,
            model_config,
            dense_dir,
            batch_size=7,
            dense_confirmation=True,
            artifact_prefix="sequence",
            skill_instruction=str(parameters["prompt_instruction"]),
            execution_fingerprint=evidence_fingerprint,
        )
        fallback_groups: set[int] = set()
        for artifact_path in dense_artifacts:
            artifact = read_json(Path(artifact_path))
            if artifact.get("fallback_used") is True:
                explicit = artifact.get("fallback_image_groups")
                if isinstance(explicit, list):
                    fallback_groups.update(
                        int(value)
                        for value in explicit
                        if isinstance(value, int) and not isinstance(value, bool)
                    )
                else:
                    fallback_groups.update(
                        int(item["image_group"])
                        for item in artifact.get("media", [])
                        if isinstance(item, dict) and isinstance(item.get("image_group"), int)
                    )
        for item in dense_frames:
            item["model_fallback_used"] = int(item["image_group"]) in fallback_groups
    frames = selected + dense_frames
    observations = coarse_observations + dense_observations
    rubric_1 = reduce_results(
        observations,
        frames,
        compare_latest_stable_topology=bool(parameters["compare_latest_stable_topology"]),
        direct_cluster_max_gap_seconds=float(parameters["direct_cluster_max_gap_seconds"]),
    )
    plan_live_skills = {
        "selection_basis": (skill_plan or {}).get(
            "selection_basis", "current_video_observed_situation_only"
        ),
        "observed_stages": (skill_plan or {}).get(
            "observed_stages", record.get("observed_stage_runs", [])
        ),
        "selected_skills": (skill_plan or {}).get(
            "selected_skills", (skill_plan or {}).get("skills", [])
        ),
        "video_id_used_for_routing": bool(
            (skill_plan or {}).get("video_id_used_for_routing", False)
        ),
        "historical_artifacts_used": False,
        "fixed_video_roi_used": False,
    }
    rubric_1["plan_live_skills"] = plan_live_skills
    report = {
        "schema_version": "resistance_agent_series_evidence.v1",
        "algorithm_version": ALGORITHM_VERSION,
        "video_id": video_id,
        "source_video_id": source_video_id,
        "source_video_path": str(video_path.resolve()),
        "source_video_sha256": source_digest,
        "action_summary_path": str(action_path.resolve()),
        "boundary_summary_path": str(boundary_summary_path.resolve()) if boundary_summary_path else None,
        "boundary_stage_runs_used": boundary_record is not None,
        "excel_accessed": False,
        "ground_truth_sent_to_model": False,
        "candidate_windows": windows,
        "stage_window_fingerprint": stage_window_fingerprint,
        "execution_fingerprint": evidence_fingerprint,
        "coarse_scan": {
            "sampling_fps": round(1.0 / float(parameters["sampling_interval_seconds"]), 3),
            "coarse_window_seconds": float(parameters["coarse_window_seconds"]),
            "scanned_frame_count": len(scanned or []),
            "model_frame_count": len(selected),
            "selection_policy": "uniform_motion_sharpness",
        },
        "scanned_frames": scanned,
        "selected_frames": selected,
        "coarse_qwen_batch_artifacts": coarse_artifacts,
        "dense_confirmation_frames": dense_frames,
        "dense_qwen_batch_artifacts": dense_artifacts,
        "observations": observations,
        "rubric_1": rubric_1,
        "routing_policy": (skill_plan or {}).get("routing_policy"),
        "skill_selection": (skill_plan or {}).get("skills", []),
        "skill_execution": execution,
        "plan_live_skills": plan_live_skills,
        "historical_fallback_used": False,
        "fixed_video_roi_used": False,
    }
    report_path = evidence_dir / "series_evidence_report.json"
    write_json(report_path, report)
    reopened = read_json(report_path)
    if reopened.get("rubric_1", {}).get("decision") not in {"pass", "fail"}:
        raise RuntimeError("series evidence report verification failed")
    return {
        "rubric_1": rubric_1,
        "report_path": str(report_path.resolve()),
    }


run_series_rubric.supports_boundary_summary = True
