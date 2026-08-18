"""Fuse OpenCV switch state and real plug displacement on the same frame."""

from __future__ import annotations

import json
import hashlib
from collections import deque
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np

try:
    from .opencv_plug_motion import Base, frame_candidates, transitions_for_triple
    from .opencv_switch_state import (
        analyze_candidate_records,
        component_candidates,
        save_contact_sheet,
    )
except ImportError:
    from opencv_plug_motion import Base, frame_candidates, transitions_for_triple  # type: ignore
    from opencv_switch_state import (  # type: ignore
        analyze_candidate_records,
        component_candidates,
        save_contact_sheet,
    )


ALGORITHM_VERSION = "opencv_same_frame_overlap_v3"
IMPLEMENTATION_VERSION = "r3_opencv_same_frame_overlap_v3"
IMPLEMENTATION_FINGERPRINT = hashlib.sha256(IMPLEMENTATION_VERSION.encode("ascii")).hexdigest()
ROI_MODE = "dynamic_current_frame_switch_and_plug"
FUSION_POLICY = "same_frame_closed_and_wiring_active"
MIN_CLOSED_PERSISTENCE_OBSERVATIONS = 3
BRIDGE_KEYS = {
    "bridge_score",
    "bridge_span",
    "bridge_column_coverage",
    "bridge_dark_ratio",
    "base_value_median",
    "value_cutoff",
}


def _sampling_targets(
    windows: list[dict[str, Any]],
    source_fps: float,
    frame_count: int,
    sampling_fps: float,
    phase_offset_seconds: float = 0.0,
) -> list[dict[str, Any]]:
    if source_fps <= 0 or sampling_fps <= 0:
        raise ValueError("video and sampling FPS must be positive")
    if not np.isfinite(phase_offset_seconds):
        raise ValueError("sampling phase offset must be finite")
    sample_period = 1.0 / sampling_fps
    normalized_phase = phase_offset_seconds % sample_period
    targets: list[dict[str, Any]] = []
    for window_index, window in enumerate(windows, start=1):
        start = max(0.0, float(window["start_seconds"]))
        end = max(start, float(window["end_seconds"]))
        window_id = str(window.get("window_id") or f"window_{window_index:03d}")
        stage = str(window.get("stage") or "wiring_action")
        first_timestamp = start + normalized_phase
        sample_count = (
            int(np.floor((end - first_timestamp) * sampling_fps + 1e-9)) + 1
            if first_timestamp <= end
            else 0
        )
        for sample_index in range(sample_count):
            timestamp = first_timestamp + sample_index / sampling_fps
            frame_number = min(int(round(timestamp * source_fps)), frame_count - 1)
            targets.append(
                {
                    "window_id": window_id,
                    "stage": stage,
                    "timestamp_seconds": round(frame_number / source_fps, 3),
                    "frame_number": frame_number,
                }
            )
    unique: dict[tuple[str, int], dict[str, Any]] = {}
    for target in targets:
        unique[(target["window_id"], target["frame_number"])] = target
    return sorted(
        unique.values(), key=lambda item: (item["frame_number"], item["window_id"])
    )


def _sampled_frames(
    capture: cv2.VideoCapture,
    targets: list[dict[str, Any]],
    analysis_width: int,
) -> Iterator[tuple[dict[str, Any], np.ndarray]]:
    current_frame = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
    for target in targets:
        target_frame = int(target["frame_number"])
        if target_frame < current_frame or target_frame - current_frame > 90:
            capture.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            current_frame = target_frame
        while current_frame < target_frame:
            if not capture.grab():
                raise RuntimeError(f"unable to seek to frame {target_frame}")
            current_frame += 1
        ok, frame = capture.read()
        if not ok or frame is None:
            raise RuntimeError(f"unable to decode frame {target_frame}")
        current_frame += 1
        if frame.shape[1] > analysis_width:
            scale = analysis_width / float(frame.shape[1])
            frame = cv2.resize(
                frame,
                (analysis_width, round(frame.shape[0] * scale)),
                interpolation=cv2.INTER_AREA,
            )
        yield target, frame


def _save_switch_candidates(
    frame: np.ndarray,
    target: dict[str, Any],
    candidate_dir: Path,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    detected = component_candidates(frame, limit=5)
    for candidate_index, candidate in enumerate(detected, start=1):
        crop_path = candidate_dir / (
            f"{target['window_id']}_{target['frame_number']:06d}_{candidate_index:02d}.jpg"
        )
        if not cv2.imwrite(str(crop_path), candidate["crop"], [cv2.IMWRITE_JPEG_QUALITY, 92]):
            raise OSError(f"unable to write switch candidate crop: {crop_path}")
        bridge = {key: candidate[key] for key in BRIDGE_KEYS}
        candidates.append(
            {
                key: value
                for key, value in candidate.items()
                if key not in BRIDGE_KEYS | {"crop"}
            }
            | {"bridge": bridge, "crop_path": str(crop_path.resolve())}
        )
    return {**target, "candidates": candidates}


def _frame_key(item: dict[str, Any]) -> tuple[str, int]:
    return str(item["window_id"]), int(item["frame_number"])


def _closed_state_is_supported(switch: dict[str, Any] | None) -> bool:
    if not switch or switch.get("state") != "closed":
        return False
    persistence = switch.get("closed_persistence_count")
    # Synthetic callers and old reports have no persistence field; preserve
    # their exact same-frame semantics while new runs enforce temporal support.
    if persistence is None:
        return True
    return int(persistence) >= MIN_CLOSED_PERSISTENCE_OBSERVATIONS


def fuse_same_frame_records(
    sampled_frames: list[dict[str, Any]],
    switch_observations: list[dict[str, Any]],
    plug_transitions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply the only failure branch: closed AND active wiring on one real frame."""
    switch_by_frame = {_frame_key(item): item for item in switch_observations}
    transitions_by_frame: dict[tuple[str, int], list[dict[str, Any]]] = {}
    active_by_frame: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for transition in plug_transitions:
        transitions_by_frame.setdefault(_frame_key(transition), []).append(transition)
        support_frames = transition.get("support_frames") or []
        for support in support_frames:
            if isinstance(support, dict) and "window_id" in support and "frame_number" in support:
                active_by_frame.setdefault(_frame_key(support), []).append(transition)
        active_by_frame.setdefault(_frame_key(transition), []).append(transition)
    frames: list[dict[str, Any]] = []
    overlaps: list[dict[str, Any]] = []
    for sampled in sampled_frames:
        key = _frame_key(sampled)
        switch = switch_by_frame.get(key)
        transitions = transitions_by_frame.get(key, [])
        active_transitions = active_by_frame.get(key, [])
        switch_closed = _closed_state_is_supported(switch)
        real_plug_transition = bool(transitions)
        wiring_active = bool(active_transitions)
        overlap = switch_closed and wiring_active
        row = {
            "window_id": sampled["window_id"],
            "stage": sampled["stage"],
            "timestamp_seconds": sampled["timestamp_seconds"],
            "frame_number": sampled["frame_number"],
            "switch_visible": switch is not None,
            "switch_state": switch.get("state") if switch else None,
            "switch_closed_persistence_count": (
                switch.get("closed_persistence_count") if switch else None
            ),
            "switch_closed_persistence_duration_seconds": (
                switch.get("closed_persistence_duration_seconds") if switch else None
            ),
            "switch_state_temporally_supported": switch_closed,
            "switch_bridge_score": switch.get("bridge_score") if switch else None,
            "switch_identity_score": switch.get("identity_score") if switch else None,
            "switch_crop_path": switch.get("crop_path") if switch else None,
            "real_plug_transition": real_plug_transition,
            "plug_transitions": transitions,
            "wiring_active": wiring_active,
            "wiring_active_transitions": active_transitions,
            "same_frame_overlap": overlap,
        }
        frames.append(row)
        if overlap:
            overlaps.append(row)
    return frames, overlaps


def analyze_opencv_switch_overlap(
    video_path: Path,
    candidate_windows: list[dict[str, Any]],
    output_dir: Path,
    sampling_fps: float = 5.0,
    sampling_phase_offset_seconds: float = 0.0,
    analysis_width: int = 960,
    roi_mode: str = ROI_MODE,
    fusion_policy: str = FUSION_POLICY,
) -> dict[str, Any]:
    """Run both OpenCV evidence chains and freeze one binary R3 result."""
    if roi_mode != ROI_MODE:
        raise ValueError(f"unsupported switch ROI mode: {roi_mode!r}")
    if fusion_policy != FUSION_POLICY:
        raise ValueError(f"unsupported switch fusion policy: {fusion_policy!r}")
    if not candidate_windows:
        raise ValueError("at least one current-run candidate window is required")
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir = output_dir / "switch_candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"unable to open video: {video_path}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    targets = _sampling_targets(
        candidate_windows,
        source_fps,
        frame_count,
        sampling_fps,
        sampling_phase_offset_seconds,
    )
    switch_candidate_frames: list[dict[str, Any]] = []
    sampled_summaries: list[dict[str, Any]] = []
    plug_transitions: list[dict[str, Any]] = []
    plug_track_count = 0
    previous_bases: list[Base] = []
    next_track_id = 1
    triple: deque[dict[str, Any]] = deque(maxlen=3)
    previous_window_id: str | None = None
    try:
        for target, frame in _sampled_frames(capture, targets, analysis_width):
            if target["window_id"] != previous_window_id:
                previous_bases = []
                next_track_id = 1
                triple.clear()
                previous_window_id = target["window_id"]
            switch_candidate_frames.append(
                _save_switch_candidates(frame, target, candidate_dir)
            )
            bases, next_track_id, candidates = frame_candidates(
                frame, previous_bases, next_track_id
            )
            previous_bases = bases
            plug_frame = {**target, "candidates": candidates}
            triple.append(plug_frame)
            frame_transition_count = 0
            if len(triple) == 3:
                common_ids = (
                    set(triple[0]["candidates"])
                    & set(triple[1]["candidates"])
                    & set(triple[2]["candidates"])
                )
                plug_track_count += len(common_ids)
                current_transitions = transitions_for_triple(*triple)
                plug_transitions.extend(current_transitions)
                frame_transition_count = len(current_transitions)
            sampled_summaries.append(
                {
                    **target,
                    "dynamic_base_count": len(bases),
                    "lead_candidate_count": sum(
                        item.get("is_lead") is True
                        for rows in candidates.values()
                        for item in rows
                    ),
                    "plug_transition_count_when_centered": frame_transition_count,
                }
            )
    finally:
        capture.release()

    switch = analyze_candidate_records(switch_candidate_frames)
    contact_sheet_path = output_dir / "switch_state_contact_sheet.jpg"
    save_contact_sheet(switch["observations"], contact_sheet_path)
    frames, overlaps = fuse_same_frame_records(
        sampled_summaries, switch["observations"], plug_transitions
    )
    decision = "fail" if overlaps else "pass"
    switch_coverage = len(switch["observations"]) / max(len(targets), 1)
    if overlaps:
        overlap_confidences = [
            min(
                float(item.get("switch_identity_score") or 0.0),
                    max(
                        float(transition.get("confidence") or 0.0)
                        for transition in (
                            item.get("wiring_active_transitions")
                            or item.get("plug_transitions")
                            or []
                        )
                    ),
            )
            for item in overlaps
        ]
        confidence = float(np.clip(max(overlap_confidences), 0.55, 0.99))
    else:
        confidence = float(np.clip(0.58 + 0.34 * switch_coverage, 0.58, 0.92))
    report: dict[str, Any] = {
        "schema_version": "resistance_agent_opencv_switch_overlap.v2",
        "algorithm_version": ALGORITHM_VERSION,
        "implementation_version": IMPLEMENTATION_VERSION,
        "implementation_fingerprint": IMPLEMENTATION_FINGERPRINT,
        "decision": decision,
        "predicted_score": 1 if decision == "pass" else 0,
        "confidence": round(confidence, 4),
        "reason": (
            "same_frame_persistent_closed_switch_and_wiring_active"
            if overlaps
            else "no_same_frame_persistent_closed_switch_and_wiring_active"
        ),
        "decision_source": "opencv_same_frame_overlap",
        "switch_state_method": "opencv_dynamic_orange_temporal_group_seed_bridge_continuity",
        "plug_motion_method": "opencv_dynamic_base_relative_wiring_activity_transition",
        "sampling_fps": sampling_fps,
        "sampling_phase_offset_seconds": sampling_phase_offset_seconds,
        "analysis_width": analysis_width,
        "roi_mode": roi_mode,
        "fusion_policy": fusion_policy,
        "candidate_windows": candidate_windows,
        "sample_count": len(targets),
        "switch_tracked_observation_count": len(switch["observations"]),
        "switch_coverage": round(switch_coverage, 4),
        "switch_state_threshold_source": switch.get("state_threshold_source"),
        "switch_state_threshold": switch["state_threshold"],
        "switch_state_cluster_centers": switch["state_cluster_centers"],
        "switch_open_observation_count": switch.get("open_observation_count", 0),
        "switch_closed_observation_count": switch.get("closed_observation_count", 0),
        "switch_min_closed_persistence_observations": MIN_CLOSED_PERSISTENCE_OBSERVATIONS,
        "switch_persistent_closed_observation_count": sum(
            int(item.get("closed_persistence_count") or 0)
            >= MIN_CLOSED_PERSISTENCE_OBSERVATIONS
            for item in switch["observations"]
            if item.get("state") == "closed"
        ),
        "plug_three_frame_base_track_count": plug_track_count,
        "real_plug_transition_count": len(plug_transitions),
        "real_plug_transitions": plug_transitions,
        "wiring_active_frame_count": sum(
            bool(item.get("wiring_active")) for item in frames
        ),
        "wiring_active_interval_count": len(plug_transitions),
        "same_frame_overlap_count": len(overlaps),
        "same_frame_overlaps": overlaps,
        "frames": frames,
        "switch_state_observations": switch["observations"],
        "switch_state_contact_sheet": (
            str(contact_sheet_path.resolve()) if contact_sheet_path.is_file() else None
        ),
        "source_video_path": str(video_path.resolve()),
        "qwen_used_for_decision": False,
        "human_review_used": False,
        "excel_accessed": False,
        "ground_truth_sent_to_model": False,
        "historical_artifacts_used": False,
        "fixed_video_roi_used": False,
        "video_id_used_for_routing": False,
    }
    report_path = output_dir / "opencv_switch_overlap_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    reopened = json.loads(report_path.read_text(encoding="utf-8"))
    if reopened.get("decision") != decision or reopened.get("sample_count") != len(targets):
        raise RuntimeError("OpenCV overlap report failed reopen verification")
    return {**report, "report_path": str(report_path.resolve())}
