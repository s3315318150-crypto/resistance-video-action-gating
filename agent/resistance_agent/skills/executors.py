"""Executable registry for situation-selected evidence skills."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Any


class SkillExecutionError(ValueError):
    """Raised when a live skill plan cannot be bound to an executor."""


SWITCH_IMPLEMENTATION_VERSION = "r3_opencv_same_frame_overlap_v3"
SWITCH_IMPLEMENTATION_FINGERPRINT = hashlib.sha256(
    SWITCH_IMPLEMENTATION_VERSION.encode("ascii")
).hexdigest()
R3_FRAME_AGENT_VERSION = "r3_frame_sampling_agent.v1"
R3_FRAME_AGENT_FINGERPRINT = hashlib.sha256(
    R3_FRAME_AGENT_VERSION.encode("ascii")
).hexdigest()


@dataclass(frozen=True)
class SkillExecutor:
    skill_id: str
    producer_tool: str
    rubric_ids: tuple[int, ...]
    defaults: dict[str, Any]


def _switch_defaults(**overrides: Any) -> dict[str, Any]:
    values = {
        "window_mode": "all_wiring_runs",
        "sampling_fps": 5.0,
        "roi_mode": "dynamic_current_frame_switch_and_plug",
        "fusion_policy": "same_frame_closed_and_wiring_active",
    }
    values.update(overrides)
    return values


def _switch_adaptive_defaults(**overrides: Any) -> dict[str, Any]:
    values = {
        **_switch_defaults(),
        "max_rounds": 2,
        "max_requests_per_round": 3,
        "max_supplemental_frames": 64,
    }
    values.update(overrides)
    return values


def _series_defaults(**overrides: Any) -> dict[str, Any]:
    values = {
        "window_mode": "all_wiring_runs",
        "coarse_window_seconds": 16.0,
        "sampling_interval_seconds": 0.5,
        "max_samples_per_window": 36,
        "coarse_model_frame_limit": 8,
        "dense_confirmation": True,
        "dense_sampling_fps": 2.0,
        "dense_radius_seconds": 2.0,
        "compare_latest_stable_topology": True,
        "direct_cluster_max_gap_seconds": 0.76,
        "roi_mode": "dynamic_terminal_graph_candidates",
        "prompt_instruction": "Identify partial A/V meters using visible glyph or green/red terminal-panel evidence, recover structured terminal paths, classify terminal-changing wiring_action separately from stable measurement_action, and keep temporary contact separate from the latest stable topology.",
        "fusion_policy": "final_graph_dynamic_roi_process_2fps",
    }
    values.update(overrides)
    return values


def _series_adaptive_defaults(**overrides: Any) -> dict[str, Any]:
    values = {
        **_series_defaults(),
        "stable_frames_per_stage_run": 2,
        "view_recovery_frames_per_stage_run": 1,
        "max_transition_anchors": 4,
        "transition_sampling_fps": 5.0,
        "transition_radius_seconds": 1.0,
        "max_supplemental_rounds": 1,
        "max_supplemental_frames": 12,
        "roi_target_long_edge": 1400,
        "max_model_roi_views_per_frame": 1,
        "observation_model_batch_size": 1,
    }
    values.update(overrides)
    return values


def _meter_defaults(**overrides: Any) -> dict[str, Any]:
    values = {
        "window_mode": "measurement_first",
        "max_samples": 28,
        "selected_frame_limit": 4,
        "dynamic_meter_candidates": True,
        "candidate_crops_per_frame": 2,
        "allow_single_visible_meter": True,
        "roi_mode": "dynamic_meter_candidates",
        "prompt_instruction": "Prioritize energized measurement frames and compare pointer direction, scale position, and occupied range terminals.",
        "fusion_policy": "multi_frame_identity_weighted",
    }
    values.update(overrides)
    return values


def _record_defaults(**overrides: Any) -> dict[str, Any]:
    values = {
        "cycle_mode": "all_observed_cycles",
        "paper_max_samples": 10,
        "meter_max_samples": 7,
        "dynamic_paper_candidates": True,
        "dynamic_meter_candidates": True,
        "candidate_crops_per_frame": 3,
        "digit_consensus_min_support": 1,
        "adaptive_enabled": True,
        "adaptive_max_rounds": 2,
        "adaptive_interval_seconds": 0.2,
        "adaptive_max_frames": 20,
        "roi_mode": "dynamic_paper_and_meter_candidates",
        "prompt_instruction": "Bind every paper value and meter reading to the same observed recording cycle before comparing them.",
        "fusion_policy": "cycle_bound_digit_consensus",
    }
    values.update(overrides)
    return values


def _remaining_defaults(rubric_id: int, **overrides: Any) -> dict[str, Any]:
    base = {
        0: {
            "time_mode": "cleanup_stage_or_tail",
            "sample_count": 6,
            "roi_mode": "dynamic_workspace_center",
            "prompt_instruction": "Compare the ordered tail sequence for a visible transition from active apparatus to cleared workspace.",
            "fusion_policy": "workspace_transition_sequence",
            "minimum_visible_actions": 1,
        },
        2: {
            "time_mode": "measurement_or_recording_context",
            "sample_count": 6,
            "roi_mode": "dynamic_meter_candidates",
            "prompt_instruction": "Judge whether both meters remain stably positioned and usable during the active measurement context.",
            "fusion_policy": "multi_frame_visible_state",
            "minimum_parallel_support": 1,
        },
        8: {
            "time_mode": "wiring_transition",
            "sample_count": 8,
            "coarse_fps": 2.0,
            "core_fps": 5.0,
            "transition_fps": 10.0,
            "dynamic_roi_min_confidence": 0.45,
            "roi_mode": "dynamic_battery_and_ammeter_candidates",
            "prompt_instruction": "Track the battery-to-ammeter connection across the visible wiring transition without using fixed video coordinates.",
            "fusion_policy": "ordered_transition_evidence",
            "minimum_sequence_confidence": 0.5,
        },
    }[rubric_id]
    base.update(overrides)
    return base


def _r2_frame_defaults(**overrides: Any) -> dict[str, Any]:
    values = {
        "window_mode": "observation_recording_cycles",
        "max_groups_per_cycle": 10,
        "initial_margin_seconds": 3.0,
        "max_margin_seconds": 8.0,
        "quality_expand_threshold": 1.0,
        "native_decode": True,
        "roi_mode": "dynamic_current_frame_meter_resistor_topology",
        "model_max_edge": 4096,
        "model_image_quality": 100,
        "prompt_instruction": "Inspect same-frame voltage-meter leads and the two endpoints of the fixed resistor. Use native current-run ROI views; do not infer hidden paths.",
        "fusion_policy": "same_frame_topology_with_temporal_consensus",
    }
    values.update(overrides)
    return values


def _polarity_defaults(**overrides: Any) -> dict[str, Any]:
    values = {
        "stage_mode": "measurement_first",
        "max_stage_frames": 8,
        "dynamic_meter_candidates": True,
        "candidate_crops_per_frame": 4,
        "pointer_min_distinct_frames": 2,
        "pointer_min_confidence": 0.85,
        "roi_mode": "dynamic_meter_candidates",
        "prompt_instruction": "Reuse the current run R5 direct meter observations; do not trace wire colors or infer polarity from video identity.",
        "fusion_policy": "current_run_r5_direct_meter_pointer",
    }
    values.update(overrides)
    return values


def _spec(skill_id: str, producer_tool: str, rubric_ids: tuple[int, ...], defaults: dict[str, Any]) -> SkillExecutor:
    return SkillExecutor(skill_id, producer_tool, rubric_ids, defaults)


EXECUTOR_REGISTRY: dict[str, SkillExecutor] = {
    item.skill_id: item
    for item in (
        _spec(
            "switch.adaptive_frame_sampling",
            "run_switch_rubric",
            (3,),
            _switch_adaptive_defaults(),
        ),
        _spec("switch.multi_stage_dense", "run_switch_rubric", (3,), _switch_defaults()),
        _spec("switch.initial_wiring_dense", "run_switch_rubric", (3,), _switch_defaults(window_mode="initial_wiring_only")),
        _spec("switch.broad_wiring_search", "run_switch_rubric", (3,), _switch_defaults(window_mode="broad_search")),
        _spec("series.adaptive_terminal_sampling", "run_series_rubric", (1,), _series_adaptive_defaults()),
        _spec("series.multi_stage_terminal_graph", "run_series_rubric", (1,), _series_defaults()),
        _spec("series.initial_terminal_graph", "run_series_rubric", (1,), _series_defaults(window_mode="initial_wiring_only")),
        _spec("series.broad_terminal_graph", "run_series_rubric", (1,), _series_defaults(window_mode="broad_search")),
        _spec("meter.explicit_measurement", "run_meter_rubrics", (5, 6), _meter_defaults()),
        _spec("meter.pre_recording_recovery", "run_meter_rubrics", (5, 6), _meter_defaults(window_mode="pre_recording_recovery", max_samples=36, selected_frame_limit=6)),
        _spec("record.two_cycle_consistency", "run_record_rubrics", (7, 9), _record_defaults()),
        _spec("record.single_cycle_consistency", "run_record_rubrics", (7, 9), _record_defaults(cycle_mode="first_observed_cycle")),
        _spec("record.broad_cycle_search", "run_record_rubrics", (7, 9), _record_defaults(cycle_mode="broad_cycle_search", paper_max_samples=12, meter_max_samples=9)),
        _spec("cleanup.explicit_stage", "run_remaining_rubrics", (0,), _remaining_defaults(0)),
        _spec("cleanup.video_tail", "run_remaining_rubrics", (0,), _remaining_defaults(0, time_mode="video_tail", sample_count=8)),
        _spec("stable_meter.explicit_measurement", "run_remaining_rubrics", (2,), _remaining_defaults(2)),
        _spec("stable_meter.recording_context", "run_remaining_rubrics", (2,), _remaining_defaults(2, time_mode="recording_context", sample_count=8)),
        _spec("voltmeter.parallel_endpoint_adaptive", "run_remaining_rubrics", (2,), _r2_frame_defaults()),
        _spec("voltmeter.parallel_endpoint_broad_search", "run_remaining_rubrics", (2,), _r2_frame_defaults(window_mode="broad_search", max_groups_per_cycle=12)),
        _spec("battery.recovery_episode", "run_remaining_rubrics", (8,), _remaining_defaults(8, time_mode="rewiring_recovery", sample_count=10)),
        _spec("battery.wiring_transition", "run_remaining_rubrics", (8,), _remaining_defaults(8)),
        _spec("battery.broad_transition_search", "run_remaining_rubrics", (8,), _remaining_defaults(8, time_mode="broad_transition_search", sample_count=10)),
        _spec("polarity.explicit_measurement_dynamic_roi", "run_polarity_rubric", (4,), _polarity_defaults()),
        _spec("polarity.pre_recording_dynamic_roi", "run_polarity_rubric", (4,), _polarity_defaults(stage_mode="pre_recording_recovery", max_stage_frames=10)),
        _spec("polarity.broad_dynamic_roi_search", "run_polarity_rubric", (4,), _polarity_defaults(stage_mode="broad_search", max_stage_frames=12, pointer_min_confidence=0.8)),
    )
}

ENUM_PARAMETERS: dict[str, set[str]] = {
    "window_mode": {
        "all_wiring_runs",
        "initial_wiring_only",
        "broad_search",
        "measurement_first",
        "pre_recording_recovery",
        "observation_recording_cycles",
    },
    "cycle_mode": {"all_observed_cycles", "first_observed_cycle", "broad_cycle_search"},
    "stage_mode": {"measurement_first", "pre_recording_recovery", "broad_search"},
    "time_mode": {
        "cleanup_stage_or_tail",
        "video_tail",
        "measurement_or_recording_context",
        "recording_context",
        "wiring_transition",
        "rewiring_recovery",
        "broad_transition_search",
    },
    "roi_mode": {
        "dynamic_current_frame_switch_and_plug",
        "dynamic_terminal_graph_candidates",
        "dynamic_meter_candidates",
        "dynamic_paper_and_meter_candidates",
        "dynamic_workspace_center",
        "dynamic_battery_and_ammeter_candidates",
        "dynamic_current_frame_meter_resistor_topology",
    },
    "fusion_policy": {
        "same_frame_closed_and_wiring_active",
        "latest_stable_plus_adjacent_direct_pair",
        "multi_frame_identity_weighted",
        "cycle_bound_digit_consensus",
        "workspace_transition_sequence",
        "multi_frame_visible_state",
        "ordered_transition_evidence",
        "endpoint_pointer_reading_consensus",
        "current_run_r5_direct_meter_pointer",
        "same_frame_topology_with_temporal_consensus",
    },
}

POSITIVE_PARAMETERS = {
    "sampling_fps",
    "max_samples_per_window",
    "per_window_frame_limit",
    "dense_sampling_fps",
    "deterministic_scan_interval_seconds",
    "neighbor_seconds",
    "sampling_interval_seconds",
    "coarse_window_seconds",
    "coarse_model_frame_limit",
    "observation_model_batch_size",
    "dense_radius_seconds",
    "direct_cluster_max_gap_seconds",
    "max_samples",
    "selected_frame_limit",
    "candidate_crops_per_frame",
    "paper_max_samples",
    "meter_max_samples",
    "digit_consensus_min_support",
    "adaptive_max_rounds",
    "adaptive_interval_seconds",
    "adaptive_max_frames",
    "sample_count",
    "coarse_fps",
    "core_fps",
    "transition_fps",
    "minimum_visible_actions",
    "minimum_parallel_support",
    "max_stage_frames",
    "pointer_min_distinct_frames",
    "max_rounds",
    "max_requests_per_round",
    "max_supplemental_frames",
    "stable_frames_per_stage_run",
    "view_recovery_frames_per_stage_run",
    "max_transition_anchors",
    "transition_sampling_fps",
    "transition_radius_seconds",
    "max_supplemental_rounds",
    "max_groups_per_cycle",
    "initial_margin_seconds",
    "max_margin_seconds",
    "quality_expand_threshold",
    "model_max_edge",
    "model_image_quality",
}


def _validate_parameter(skill_id: str, key: str, value: Any) -> None:
    switch_fixed = {
        "sampling_fps": 5.0,
        "roi_mode": "dynamic_current_frame_switch_and_plug",
        "fusion_policy": "same_frame_closed_and_wiring_active",
    }
    if skill_id.startswith("switch.") and key in switch_fixed and value != switch_fixed[key]:
        raise SkillExecutionError(
            f"{skill_id}.{key} must be {switch_fixed[key]!r} for the OpenCV same-frame executor"
        )
    allowed = ENUM_PARAMETERS.get(key)
    if allowed is not None and value not in allowed:
        raise SkillExecutionError(f"invalid value for {skill_id}.{key}: {value!r}")
    if key in POSITIVE_PARAMETERS and float(value) <= 0:
        raise SkillExecutionError(f"{skill_id}.{key} must be positive")
    bounded_integer_parameters = {
        "max_rounds": 2,
        "max_requests_per_round": 3,
        "max_supplemental_frames": 96,
        "adaptive_max_rounds": 2,
        "adaptive_max_frames": 20,
        "max_supplemental_rounds": 1,
    }
    if key in bounded_integer_parameters and int(value) > bounded_integer_parameters[key]:
        raise SkillExecutionError(
            f"{skill_id}.{key} must be at most {bounded_integer_parameters[key]}"
        )
    if key in {"pointer_min_confidence", "minimum_sequence_confidence", "dynamic_roi_min_confidence"} and not 0.0 <= float(value) <= 1.0:
        raise SkillExecutionError(f"{skill_id}.{key} must be between 0 and 1")


def _same_type(expected: Any, value: Any) -> bool:
    if isinstance(expected, bool):
        return isinstance(value, bool)
    if isinstance(expected, int):
        return isinstance(value, int) and not isinstance(value, bool)
    if isinstance(expected, float):
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return isinstance(value, type(expected))


def _bind(item: dict[str, Any]) -> dict[str, Any]:
    skill_id = item.get("skill_id")
    spec = EXECUTOR_REGISTRY.get(skill_id) if isinstance(skill_id, str) else None
    if spec is None:
        raise SkillExecutionError(f"unregistered live skill: {skill_id!r}")
    declared = item.get("rubric_ids")
    if not isinstance(declared, list) or tuple(declared) != spec.rubric_ids:
        raise SkillExecutionError(f"rubric_ids mismatch for {spec.skill_id}")
    raw = item.get("parameters") or {}
    if not isinstance(raw, dict):
        raise SkillExecutionError(f"parameters must be an object for {spec.skill_id}")
    unknown = sorted(set(raw) - set(spec.defaults))
    if unknown:
        raise SkillExecutionError(f"unknown parameters for {spec.skill_id}: {unknown}")
    parameters = deepcopy(spec.defaults)
    for key, value in raw.items():
        if not _same_type(spec.defaults[key], value):
            raise SkillExecutionError(f"invalid type for {spec.skill_id}.{key}")
        _validate_parameter(spec.skill_id, key, value)
        parameters[key] = float(value) if isinstance(spec.defaults[key], float) else value
    payload = {
        "skill_id": spec.skill_id,
        "producer_tool": spec.producer_tool,
        "rubric_ids": list(spec.rubric_ids),
        "selected_by": str(item.get("selected_by") or "unspecified"),
        "parameters": parameters,
    }
    if spec.skill_id == "switch.adaptive_frame_sampling":
        payload["implementation_version"] = R3_FRAME_AGENT_VERSION
        payload["implementation_fingerprint"] = R3_FRAME_AGENT_FINGERPRINT
    elif spec.skill_id.startswith("switch."):
        payload["implementation_version"] = SWITCH_IMPLEMENTATION_VERSION
        payload["implementation_fingerprint"] = SWITCH_IMPLEMENTATION_FINGERPRINT
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("ascii")
    payload["execution_fingerprint"] = hashlib.sha256(encoded).hexdigest()
    return payload


def bind_skill_plan(plan: dict[str, Any]) -> list[dict[str, Any]]:
    raw = plan.get("skills")
    if not isinstance(raw, list):
        raise SkillExecutionError("live skill plan must contain a skills array")
    bound = [_bind(item) for item in raw if isinstance(item, dict)]
    covered = [rubric_id for item in bound for rubric_id in item["rubric_ids"]]
    if sorted(covered) != list(range(10)):
        raise SkillExecutionError(f"live skill coverage must be exactly rubric 0..9, got {sorted(covered)}")
    return bound


def executions_for_rubrics(plan: dict[str, Any], rubric_ids: tuple[int, ...]) -> list[dict[str, Any]]:
    requested = set(rubric_ids)
    return [item for item in bind_skill_plan(plan) if requested.intersection(item["rubric_ids"])]


def execution_for_rubric(plan: dict[str, Any], rubric_id: int) -> dict[str, Any]:
    matches = executions_for_rubrics(plan, (rubric_id,))
    if len(matches) != 1:
        raise SkillExecutionError(f"rubric {rubric_id} must resolve to exactly one live skill")
    return matches[0]


def producer_plan(plan: dict[str, Any], rubric_ids: tuple[int, ...]) -> list[dict[str, Any]]:
    requested = set(rubric_ids)
    grouped: dict[str, set[int]] = {}
    order: list[str] = []
    for execution in bind_skill_plan(plan):
        produced = set(execution["rubric_ids"])
        if not requested.intersection(produced):
            continue
        tool = execution["producer_tool"]
        if tool not in grouped:
            grouped[tool] = set()
            order.append(tool)
        grouped[tool].update(produced)
    return [{"tool": tool, "rubric_ids": sorted(grouped[tool])} for tool in order]
