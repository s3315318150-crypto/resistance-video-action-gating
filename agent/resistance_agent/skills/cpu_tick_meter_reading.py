"""CPU printed-tick reading for current-run R5/R6 meter frames."""

from __future__ import annotations

import importlib
import math
import sys
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from statistics import median
from typing import Any

import cv2


SKILL_VERSION = "cpu_tick_meter_reading.v1"
ROLE_RANGES = {
    "ammeter": {"small": 0.6, "large": 3.0, "unit": "A"},
    "voltmeter": {"small": 3.0, "large": 15.0, "unit": "V"},
}
COMPONENT_FILES = (
    "scale_tick_grid_v1.py",
    "scale_tick_grid_batch_v1.py",
    "count_meter_ticks_v1.py",
    "generic_meter_tick_batch_v4_role_glyph.py",
)
DEFAULT_COMPONENT_ROOT = Path(__file__).resolve().parent / "r5_r6_dense_meter_state"


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


@lru_cache(maxsize=2)
def _load_components(root_value: str) -> dict[str, Any]:
    root = Path(root_value).resolve()
    missing = [name for name in COMPONENT_FILES if not (root / name).is_file()]
    if missing:
        raise ValueError(f"CPU tick reader components missing: {', '.join(missing)}")
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    return {
        "single": importlib.import_module("scale_tick_grid_v1"),
        "batch": importlib.import_module("scale_tick_grid_batch_v1"),
        "count": importlib.import_module("count_meter_ticks_v1"),
        "generic": importlib.import_module("generic_meter_tick_batch_v4_role_glyph"),
    }


def active_image_groups(qwen: dict[str, Any], group_count: int) -> list[int]:
    """Select groups with energized or non-deenergized current-run evidence."""
    states: dict[int, set[str]] = {index: set() for index in range(1, group_count + 1)}
    for item in qwen.get("observations") or []:
        if not isinstance(item, dict):
            continue
        group = item.get("image_group")
        state = item.get("circuit_state")
        if isinstance(group, int) and group in states and isinstance(state, str):
            states[group].add(state)
    energized = [group for group, values in states.items() if "energized" in values]
    if energized:
        return energized
    if qwen.get("measurement_active") is True:
        return [group for group, values in states.items() if values != {"deenergized"}]
    return []


def _fine_tick_count(
    grid_consensus: dict[str, Any],
    range_max: float | None,
    unit: str,
    nearest_tick: Any,
) -> dict[str, Any]:
    if grid_consensus.get("status") != "grid_consensus_candidate":
        return {
            "status": "tick_grid_missing",
            "nearest_tick_index": None,
            "reading": None,
            "unit": unit,
        }
    zero = float(grid_consensus["consensus_zero_angle_deg"])
    full = float(grid_consensus["consensus_full_angle_deg"])
    if abs(zero - full) < 1e-6:
        return {
            "status": "tick_grid_degenerate",
            "nearest_tick_index": None,
            "reading": None,
            "unit": unit,
        }
    raw_ticks = [
        (zero - float(item["pointer_angle_deg"])) / (zero - full) * 30.0
        for item in grid_consensus.get("frame_positions") or []
        if isinstance(item, dict) and item.get("pointer_angle_deg") is not None
    ]
    if not raw_ticks:
        return {
            "status": "pointer_positions_missing",
            "nearest_tick_index": None,
            "reading": None,
            "unit": unit,
        }
    raw_tick = float(median(raw_ticks))
    if raw_tick < -0.5:
        pointer_state = "reverse"
        tick = None
    elif raw_tick > 30.5:
        pointer_state = "overrange"
        tick = None
    else:
        tick = min(30, max(0, int(nearest_tick(raw_tick))))
        pointer_state = "zero" if tick == 0 else "normal_rightward"
    smallest = float(range_max) / 30.0 if range_max is not None else None
    reading = tick * smallest if tick is not None and smallest is not None else None
    return {
        "status": "confirmed_tick_reading" if reading is not None else "pointer_state_only",
        "method": "printed_grid_endpoints_then_30_tick_half_up_count",
        "total_divisions": 30,
        "raw_tick_candidates": [round(value, 6) for value in raw_ticks],
        "raw_tick_index": round(raw_tick, 6),
        "nearest_tick_index": tick,
        "pointer_state": pointer_state,
        "range_max_value": range_max,
        "smallest_division": round(smallest, 6) if smallest is not None else None,
        "reading": round(reading, 6) if reading is not None else None,
        "unit": unit,
    }


def _range_assessment(role: str, tick_count: dict[str, Any]) -> str:
    values = ROLE_RANGES[role]
    range_max = tick_count.get("range_max_value")
    raw_tick = tick_count.get("raw_tick_index")
    tick = tick_count.get("nearest_tick_index")
    state = tick_count.get("pointer_state")
    if range_max is None or raw_tick is None:
        return "unknown"
    if state == "overrange":
        return "too_low"
    if math.isclose(float(range_max), float(values["small"]), rel_tol=0.0, abs_tol=1e-6):
        return "too_low" if float(raw_tick) >= 27.0 else "appropriate" if tick not in {None, 0} else "unknown"
    if math.isclose(float(range_max), float(values["large"]), rel_tol=0.0, abs_tol=1e-6):
        return "too_high" if tick is not None and tick <= 9 else "appropriate" if tick is not None else "unknown"
    return "unknown"


def _role_result(
    role: str,
    observations: list[dict[str, Any]],
    grid_frames: list[dict[str, Any]],
    components: dict[str, Any],
) -> dict[str, Any]:
    generic_summary = components["generic"].summarize_role(observations, role)
    matched_grid_frames = [
        item
        for item in grid_frames
        if (item.get("per_frame_position") or {}).get("matched") is True
    ]
    grid_consensus = components["batch"].consensus_grid(matched_grid_frames)
    range_max = generic_summary.get("range_max_value")
    tick_count = _fine_tick_count(
        grid_consensus,
        float(range_max) if range_max is not None else None,
        str(ROLE_RANGES[role]["unit"]),
        components["count"]._nearest_tick,
    )
    calibrated_tick = generic_summary.get("median_tick_index")
    printed_tick = tick_count.get("nearest_tick_index")
    agreement = None
    if calibrated_tick is not None and printed_tick is not None:
        agreement = abs(int(calibrated_tick) - int(printed_tick)) <= 2
    reading_available = tick_count.get("reading") is not None
    if reading_available:
        status = "reading_candidate"
    elif tick_count.get("pointer_state") in {"reverse", "overrange"}:
        status = "pointer_state_candidate"
    else:
        status = "candidate_incomplete"
    used_frames = int(grid_consensus.get("used_frame_count") or 0)
    if used_frames >= 2:
        confidence = 0.88 if agreement is not False else 0.78
    elif used_frames == 1:
        confidence = 0.72 if agreement is not False else 0.64
    else:
        confidence = 0.45
    return {
        "role": role,
        "status": status,
        "confidence": confidence,
        "pointer_state": tick_count.get("pointer_state"),
        "range_assessment": _range_assessment(role, tick_count),
        "reading": tick_count.get("reading"),
        "unit": ROLE_RANGES[role]["unit"],
        "reading_source": "dynamic_printed_grid_30_tick_count" if reading_available else None,
        "generic_calibrated_summary": generic_summary,
        "printed_grid_consensus": grid_consensus,
        "tick_count": tick_count,
        "calibrated_and_printed_tick_agree_within_2": agreement,
        "observations": observations,
        "grid_frames": grid_frames,
        "matched_grid_frame_count": len(matched_grid_frames),
    }


def run_cpu_tick_reader(
    frames: list[dict[str, Any]],
    *,
    baseline_root: Path,
    calibration: str | Path,
    terminal_annotations: str | Path,
    output_dir: Path,
    max_frames: int = 6,
    max_feature_width: int = 2400,
) -> dict[str, Any]:
    """Run the four CPU meter components on current-run selected frames."""
    root = baseline_root.resolve()
    calibration_path = _resolve(root, calibration)
    terminal_dir = _resolve(root, terminal_annotations)
    if not calibration_path.is_file() or not terminal_dir.is_dir():
        raise ValueError("CPU tick reader calibration or terminal annotations missing")
    components = _load_components(str(root))
    output_dir.mkdir(parents=True, exist_ok=True)
    sift = cv2.SIFT_create(nfeatures=14000, contrastThreshold=0.012, edgeThreshold=14)
    face_templates = components["generic"].build_face_templates(calibration_path, terminal_dir, sift)
    terminal_templates = components["generic"].build_terminal_templates(terminal_dir, sift)
    scan_config = components["generic"].ScanConfig()

    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for frame in frames:
        path = str(frame.get("frame_path") or "")
        if path and path not in seen and Path(path).is_file():
            seen.add(path)
            unique.append(frame)
        if len(unique) >= max_frames:
            break

    role_observations: dict[str, list[dict[str, Any]]] = {role: [] for role in ROLE_RANGES}
    role_grids: dict[str, list[dict[str, Any]]] = {role: [] for role in ROLE_RANGES}
    for sequence, frame in enumerate(unique, start=1):
        frame_path = Path(str(frame["frame_path"]))
        group = int(frame.get("image_group") or sequence)
        for role in ROLE_RANGES:
            raw = components["generic"].process_observation(
                frame_path,
                role,
                face_templates,
                terminal_templates,
                sift,
                scan_config,
                output_dir / "generic" / f"group_{group:02d}",
                max_feature_width,
            )
            observation = {key: value for key, value in raw.items() if key != "source_integrity"}
            observation["image_group"] = group
            observation["frame_number"] = frame.get("frame_number")
            role_observations[role].append(observation)
            pointer = observation.get("pointer") or {}
            face_value = observation.get("rectified_face_path")
            if pointer.get("anchor") is None or pointer.get("angle_deg") is None or not face_value:
                continue
            face_path = Path(str(face_value))
            face = cv2.imread(str(face_path), cv2.IMREAD_COLOR)
            if face is None:
                continue
            ticks = components["single"].detect_scale_ticks(face, pointer["anchor"])
            grid = ticks.get("regular_grid") or {}
            position = components["single"].pointer_grid_position(grid, float(pointer["angle_deg"]))
            grid_dir = output_dir / "printed_grid" / role
            grid_dir.mkdir(parents=True, exist_ok=True)
            overlay_path = grid_dir / f"group_{group:02d}_{face_path.stem}_grid.jpg"
            cv2.imwrite(
                str(overlay_path),
                components["single"].draw_overlay(face, ticks, float(pointer["angle_deg"])),
                [int(cv2.IMWRITE_JPEG_QUALITY), 97],
            )
            role_grids[role].append(
                {
                    "image_group": group,
                    "timestamp_seconds": observation.get("timestamp_seconds") or frame.get("timestamp_seconds"),
                    "frame_number": frame.get("frame_number"),
                    "pointer_angle_deg": pointer["angle_deg"],
                    "dynamic_pointer_anchor": pointer["anchor"],
                    "grid": grid,
                    "per_frame_position": position,
                    "range_max_value": (observation.get("range") or {}).get("range_max_value"),
                    "selected_port": (observation.get("range") or {}).get("selected_port"),
                    "overlay_path": str(overlay_path.resolve()),
                }
            )

    roles = {
        role: _role_result(role, role_observations[role], role_grids[role], components)
        for role in ROLE_RANGES
    }
    return {
        "schema_version": "resistance-agent-cpu-tick-meter-reading.v1",
        "skill_version": SKILL_VERSION,
        "status": "completed",
        "selection_basis": "current_run_active_measurement_frames_only",
        "video_id_used_for_routing": False,
        "historical_artifacts_used": False,
        "fixed_video_roi_used": False,
        "component_paths": [
            f"agent/resistance_agent/skills/r5_r6_dense_meter_state/{name}"
            for name in COMPONENT_FILES
        ],
        "input_frame_count": len(unique),
        "input_image_groups": [int(item.get("image_group") or index) for index, item in enumerate(unique, start=1)],
        "roles": roles,
    }


def _compact(evidence: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": evidence.get("status"),
        "skill_version": evidence.get("skill_version"),
        "selection_basis": evidence.get("selection_basis"),
        "video_id_used_for_routing": evidence.get("video_id_used_for_routing"),
        "historical_artifacts_used": evidence.get("historical_artifacts_used"),
        "fixed_video_roi_used": evidence.get("fixed_video_roi_used"),
        "roles": {
            role: {
                key: value.get(key)
                for key in (
                    "status",
                    "confidence",
                    "pointer_state",
                    "range_assessment",
                    "reading",
                    "unit",
                    "reading_source",
                    "calibrated_and_printed_tick_agree_within_2",
                )
            }
            for role, value in (evidence.get("roles") or {}).items()
            if isinstance(value, dict)
        },
    }


def fuse_binary_results(
    rubric_5: dict[str, Any],
    rubric_6: dict[str, Any],
    evidence: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fuse direct CPU tick evidence without introducing a third outcome."""
    r5, r6 = deepcopy(rubric_5), deepcopy(rubric_6)
    compact = _compact(evidence)
    roles = [
        item
        for item in (evidence.get("roles") or {}).values()
        if isinstance(item, dict) and item.get("status") in {"reading_candidate", "pointer_state_candidate"}
    ]
    abnormal = [item for item in roles if item.get("pointer_state") in {"reverse", "overrange"}]
    normal = [item for item in roles if item.get("pointer_state") == "normal_rightward"]
    bad_range = [item for item in roles if item.get("range_assessment") in {"too_low", "too_high"}]
    good_range = [item for item in roles if item.get("range_assessment") == "appropriate"]

    r5.setdefault("diagnostics", {})["cpu_tick_grid"] = compact
    if abnormal:
        r5.update(
            decision="fail",
            predicted_score=0,
            confidence=round(max(float(r5.get("confidence") or 0.0), max(float(item["confidence"]) for item in abnormal)), 4),
            reason="cpu_tick_grid_confirms_abnormal_pointer_position",
        )
        r5["diagnostics"]["cpu_tick_grid_fusion"] = "direct_abnormal_overrides"
    elif normal and r5.get("reason") == "no_normal_pointer_deflection_found_after_temporal_and_roi_search":
        r5.update(
            decision="pass",
            predicted_score=1,
            confidence=round(max(float(r5.get("confidence") or 0.0), max(float(item["confidence"]) for item in normal)), 4),
            reason="cpu_tick_grid_confirms_in_scale_rightward_deflection",
        )
        r5["diagnostics"]["cpu_tick_grid_fusion"] = "direct_normal_resolves_missing_qwen_pointer"
    else:
        r5["diagnostics"]["cpu_tick_grid_fusion"] = "diagnostic_or_agrees_with_existing_result"

    r6.setdefault("diagnostics", {})["cpu_tick_grid"] = compact
    if bad_range:
        r6.update(
            decision="fail",
            predicted_score=0,
            confidence=round(max(float(r6.get("confidence") or 0.0), max(float(item["confidence"]) for item in bad_range)), 4),
            reason="cpu_tick_grid_confirms_range_mismatch",
        )
        r6["diagnostics"]["cpu_tick_grid_fusion"] = "direct_range_mismatch_overrides"
    elif good_range and r6.get("reason") == "range_not_shown_appropriate_after_temporal_and_roi_search":
        r6.update(
            decision="pass",
            predicted_score=1,
            confidence=round(max(float(r6.get("confidence") or 0.0), max(float(item["confidence"]) for item in good_range)), 4),
            reason="cpu_tick_grid_confirms_selected_range_and_pointer_position",
        )
        r6["diagnostics"]["cpu_tick_grid_fusion"] = "direct_range_resolves_missing_qwen_range"
    else:
        r6["diagnostics"]["cpu_tick_grid_fusion"] = "diagnostic_or_agrees_with_existing_result"
    return r5, r6
