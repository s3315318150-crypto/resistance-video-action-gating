"""Situation-driven Rubric 6 classification from stable pointer geometry.

This ports the closed-stable-state CV V3 decision rule into the Agent.  It
consumes stage-level pointer angles and a meter-face calibration; it does not
route by a known prediction, inspect Excel, or call a multimodal model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SKILL_VERSION = "closed_stable_r6_cv_v3.v1"
ROLES = ("ammeter", "voltmeter")
POST_CLOSE_STAGE_PROXIES = {
    "measurement_1",
    "recording_1",
    "measurement_2",
    "recording_2",
}
OVERRANGE_STATE = "overrange_candidate"
REVERSE_STATE = "negative_deflection_candidate"
ZERO_STATE = "zero_band_candidate"
MISSING_STATE = "no_stable_pointer_candidate"
STRONG_REVERSE_RATIO = -0.10


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def classify_pointer_scale_state(
    zero_angle_deg: float,
    full_angle_deg: float,
    pointer_angle_deg: float,
    *,
    sweep_direction: str,
    zero_uncertainty_deg: float = 1.5,
    full_uncertainty_deg: float = 1.5,
    pointer_uncertainty_deg: float = 0.5,
    minimum_normal_ratio: float = 0.05,
) -> dict[str, Any]:
    """Map a pointer angle to an unclamped scale ratio and geometric state."""
    if sweep_direction not in {"decreasing", "increasing"}:
        raise ValueError("sweep_direction must be decreasing or increasing")
    sign = -1.0 if sweep_direction == "decreasing" else 1.0
    sweep = sign * (full_angle_deg - zero_angle_deg)
    position = sign * (pointer_angle_deg - zero_angle_deg)
    if sweep <= 0.0:
        raise ValueError("zero/full angles do not match sweep direction")
    ratio = position / sweep
    zero_band = zero_uncertainty_deg + pointer_uncertainty_deg
    full_band = full_uncertainty_deg + pointer_uncertainty_deg

    if abs(pointer_angle_deg - zero_angle_deg) <= zero_band:
        state = ZERO_STATE
    elif ratio < 0.0:
        state = REVERSE_STATE
    elif abs(pointer_angle_deg - full_angle_deg) <= full_band:
        state = "full_scale_band_candidate"
    elif ratio > 1.0:
        state = OVERRANGE_STATE
    elif ratio < minimum_normal_ratio:
        state = "positive_but_too_small_candidate"
    else:
        state = "normal_positive_deflection_candidate"
    return {
        "state": state,
        "geometric_ratio_unclamped": round(ratio, 6),
        "geometric_percent_unclamped": round(100.0 * ratio, 3),
    }


def _geometry_by_role(runtime_calibration: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for template in runtime_calibration.get("templates", []):
        if not isinstance(template, dict):
            continue
        role = str(template.get("role") or "")
        calibration = template.get("geometry_calibration")
        if role in ROLES and isinstance(calibration, dict):
            result[role] = calibration
    missing = [role for role in ROLES if role not in result]
    if missing:
        raise ValueError(f"runtime calibration is missing roles: {', '.join(missing)}")
    return result


def scan_role_stages(
    search_role: dict[str, Any], role: str, calibration: dict[str, Any]
) -> dict[str, Any]:
    """Classify every stable stage instead of reading only the selected stage."""
    observations: list[dict[str, Any]] = []
    for stage_result in search_role.get("stage_results", []):
        if not isinstance(stage_result, dict):
            continue
        consensus = stage_result.get("temporal_consensus") or {}
        angle = consensus.get("median_angle_deg")
        stable = bool(consensus.get("stable"))
        common = {
            "stage": stage_result.get("stage"),
            "supporting_frame_count": int(consensus.get("supporting_frame_count") or 0),
            "start_seconds": consensus.get("start_seconds"),
            "end_seconds": consensus.get("end_seconds"),
            "evidence_paths": list(consensus.get("evidence_paths") or []),
        }
        if not stable or angle is None:
            observations.append(
                {
                    **common,
                    "state": MISSING_STATE,
                    "pointer_assessable": False,
                    "overrange_candidate": False,
                }
            )
            continue
        classified = classify_pointer_scale_state(
            float(calibration["zero_angle_deg"]),
            float(calibration["full_angle_deg"]),
            float(angle),
            sweep_direction=str(calibration["sweep_direction"]),
            zero_uncertainty_deg=float(calibration.get("zero_uncertainty_deg", 1.5)),
            full_uncertainty_deg=float(calibration.get("full_uncertainty_deg", 1.5)),
        )
        observations.append(
            {
                **common,
                **classified,
                "pointer_angle_deg": float(angle),
                "pointer_assessable": True,
                "overrange_candidate": classified["state"] == OVERRANGE_STATE,
            }
        )

    assessable = [item for item in observations if item["pointer_assessable"]]
    overrange = [item for item in assessable if item["overrange_candidate"]]
    chosen = overrange[0] if overrange else (assessable[0] if assessable else None)
    return {
        "role": role,
        "state": chosen["state"] if chosen else MISSING_STATE,
        "selected_stage": chosen.get("stage") if chosen else None,
        "pointer_angle_deg": chosen.get("pointer_angle_deg") if chosen else None,
        "geometric_ratio_unclamped": chosen.get("geometric_ratio_unclamped") if chosen else None,
        "supporting_frame_count": int(chosen.get("supporting_frame_count") or 0) if chosen else 0,
        "start_seconds": chosen.get("start_seconds") if chosen else None,
        "end_seconds": chosen.get("end_seconds") if chosen else None,
        "evidence_paths": list(chosen.get("evidence_paths") or []) if chosen else [],
        "pointer_assessable": bool(assessable),
        "overrange_candidate": bool(overrange),
        "searched_stages": list(search_role.get("searched_stages") or []),
        "stage_observations": observations,
    }


def _batch_roles(batch: dict[str, Any], role: str) -> dict[str, dict[str, Any]]:
    return {
        str(item["video_id"]): item.get("roles", {}).get(role, {})
        for item in batch.get("videos", [])
        if isinstance(item, dict) and item.get("video_id") is not None
    }


def _batch_video_record(batch: dict[str, Any], video_id: str) -> dict[str, Any] | None:
    return next(
        (
            item
            for item in batch.get("videos", [])
            if isinstance(item, dict) and str(item.get("video_id")) == str(video_id)
        ),
        None,
    )


def build_video_source(
    ammeter_search: dict[str, Any],
    voltmeter_search: dict[str, Any],
    runtime_calibration: dict[str, Any],
    *,
    video_id: str,
    source_video_id: str,
) -> dict[str, Any] | None:
    """Bind stage evidence to the current input; this does not choose an algorithm."""
    ammeter_record = _batch_video_record(ammeter_search, video_id)
    voltmeter_record = _batch_video_record(voltmeter_search, video_id)
    if ammeter_record is None and voltmeter_record is None:
        return None
    source_names = {
        Path(str(item.get("source_video"))).name
        for item in (ammeter_record, voltmeter_record)
        if isinstance(item, dict) and item.get("source_video")
    }
    if source_names and source_video_id not in source_names:
        raise ValueError("closed-stable stage evidence source video mismatch")
    geometry = _geometry_by_role(runtime_calibration)
    ammeter_roles = _batch_roles(ammeter_search, "ammeter")
    voltmeter_roles = _batch_roles(voltmeter_search, "voltmeter")
    return {
        "video_id": str(video_id),
        "source_video_id": source_video_id,
        "ammeter": scan_role_stages(ammeter_roles.get(str(video_id), {}), "ammeter", geometry["ammeter"]),
        "voltmeter": scan_role_stages(voltmeter_roles.get(str(video_id), {}), "voltmeter", geometry["voltmeter"]),
    }


def _role_observation(video: dict[str, Any], role: str) -> dict[str, Any]:
    source = video.get(role) or {}
    return {
        "role": role,
        "state": str(source.get("state") or MISSING_STATE),
        "selected_stage": source.get("selected_stage"),
        "pointer_angle_deg": source.get("pointer_angle_deg"),
        "geometric_ratio_unclamped": source.get("geometric_ratio_unclamped"),
        "supporting_frame_count": int(source.get("supporting_frame_count") or 0),
        "start_seconds": source.get("start_seconds"),
        "end_seconds": source.get("end_seconds"),
        "evidence_paths": list(source.get("evidence_paths") or []),
        "pointer_assessable": bool(source.get("pointer_assessable")),
        "stage_observations": list(source.get("stage_observations") or []),
    }


def _ratio(item: dict[str, Any]) -> float | None:
    value = item.get("geometric_ratio_unclamped")
    return float(value) if value is not None else None


def _is_strong_reverse(item: dict[str, Any]) -> bool:
    ratio = _ratio(item)
    return item.get("state") == REVERSE_STATE and ratio is not None and ratio <= STRONG_REVERSE_RATIO


def _is_zero_like(item: dict[str, Any]) -> bool:
    if item.get("state") == ZERO_STATE:
        return True
    ratio = _ratio(item)
    return item.get("state") == REVERSE_STATE and ratio is not None and ratio > STRONG_REVERSE_RATIO


def classify_role(video: dict[str, Any], role: str) -> dict[str, Any]:
    observation = _role_observation(video, role)
    assessable = [
        item
        for item in observation["stage_observations"]
        if item.get("stage") in POST_CLOSE_STAGE_PROXIES and bool(item.get("pointer_assessable"))
    ]
    if not assessable and observation["pointer_assessable"]:
        assessable = [
            {
                "stage": observation["selected_stage"],
                "state": observation["state"],
                "pointer_angle_deg": observation["pointer_angle_deg"],
                "geometric_ratio_unclamped": observation["geometric_ratio_unclamped"],
                "supporting_frame_count": observation["supporting_frame_count"],
                "start_seconds": observation["start_seconds"],
                "end_seconds": observation["end_seconds"],
                "evidence_paths": observation["evidence_paths"],
                "pointer_assessable": True,
            }
        ]
    overrange = [item for item in assessable if item.get("state") == OVERRANGE_STATE]
    reverse = [item for item in assessable if _is_strong_reverse(item)]
    zero_like = [item for item in assessable if _is_zero_like(item)]
    observation.update(
        {
            "closed_stable_binding": "measurement_or_recording_stable_consensus_proxy",
            "switch_closure_directly_observed": False,
            "assessable_closed_stable_candidates": assessable,
            "assessable_closed_stable_candidate_count": len(assessable),
            "overrange_stage_candidates": overrange,
            "strong_reverse_stage_candidates": reverse,
            "zero_like_stage_candidates": zero_like,
            "normal_or_other_stage_candidates": [item for item in assessable if item not in zero_like],
            "overrange_candidate": bool(overrange),
            "strong_reverse_candidate": bool(reverse),
            "all_assessable_candidates_zero_like": bool(assessable) and len(zero_like) == len(assessable),
        }
    )
    return observation


def classify_video(video: dict[str, Any]) -> dict[str, Any]:
    observations = {role: classify_role(video, role) for role in ROLES}
    overrange_roles = [role for role, item in observations.items() if item["overrange_candidate"]]
    reverse_roles = [role for role, item in observations.items() if item["strong_reverse_candidate"]]
    zero_only_roles = [
        role for role, item in observations.items() if item["all_assessable_candidates_zero_like"]
    ]
    assessable_roles = [
        role for role, item in observations.items() if item["assessable_closed_stable_candidate_count"] > 0
    ]
    missing_roles = [role for role in ROLES if role not in assessable_roles]
    if overrange_roles:
        decision, rule, level = "fail", "explicit_overrange_in_closed_stable_proxy", "high"
    elif reverse_roles:
        decision, rule, level = "fail", "explicit_strong_reverse_in_closed_stable_proxy", "high"
    elif zero_only_roles:
        decision, rule, level = "fail", "all_found_stable_candidates_at_zero", "high"
    elif len(assessable_roles) == 2:
        decision, rule, level = "pass", "both_roles_have_nonzero_legal_candidate", "high"
    elif len(assessable_roles) == 1:
        decision, rule, level = "pass", "one_role_legal_other_missing_fallback", "medium"
    else:
        decision, rule, level = "fail", "no_assessable_closed_stable_pointer_candidate", "low"
    confidence = {"high": 0.95, "medium": 0.72, "low": 0.38}[level]
    return {
        "decision": decision,
        "predicted_score": 1 if decision == "pass" else 0,
        "confidence": confidence,
        "reason": rule,
        "diagnostics": {
            "skill_version": SKILL_VERSION,
            "confidence_level": level,
            "strong_reverse_ratio_threshold": STRONG_REVERSE_RATIO,
            "closed_stable_binding": "measurement_or_recording_stable_consensus_proxy",
            "switch_closure_directly_observed": False,
            "overrange_roles": overrange_roles,
            "strong_reverse_roles": reverse_roles,
            "zero_only_roles": zero_only_roles,
            "assessable_roles": assessable_roles,
            "missing_roles": missing_roles,
            "roles": observations,
            "qwen_called": False,
            "excel_accessed": False,
            "routing_policy": "current-stage geometry; no prediction or Excel routing",
        },
    }


def evaluate_paths(
    ammeter_search_path: Path,
    voltmeter_search_path: Path,
    runtime_calibration_path: Path,
    *,
    video_id: str,
    source_video_id: str,
) -> dict[str, Any] | None:
    for path in (ammeter_search_path, voltmeter_search_path, runtime_calibration_path):
        if not path.is_file():
            raise ValueError(f"closed-stable CV V3 input is missing: {path}")
    video = build_video_source(
        read_json(ammeter_search_path),
        read_json(voltmeter_search_path),
        read_json(runtime_calibration_path),
        video_id=video_id,
        source_video_id=source_video_id,
    )
    if video is None:
        return None
    result = classify_video(video)
    result["diagnostics"]["source_stage_results"] = {
        "ammeter_search": str(ammeter_search_path.resolve()),
        "voltmeter_search": str(voltmeter_search_path.resolve()),
        "runtime_calibration": str(runtime_calibration_path.resolve()),
    }
    return result
