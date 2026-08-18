"""Select live evidence skills from observed stages, never from video identity."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


LIVE_ROUTING_POLICY = "live_situation_skills.v1"


def _is_merged_measurement_recording(item: dict[str, Any]) -> bool:
    return str(item.get("stage")) in {"recording_1", "recording_2"} and (
        item.get("merged_measurement_recording") is True
        or item.get("merged_stage") is True
        or item.get("stage_semantics") == "measurement_and_recording_cycle"
        or item.get("stage_window_semantics") == "measurement_and_recording_cycle"
        or item.get("merged_stage_semantics") == "measurement_and_recording_cycle"
    )


def _has_direct_measurement_evidence(item: dict[str, Any]) -> bool:
    """Return true only when the current run contains a measurement action."""
    stage = str(item.get("stage"))
    if stage in {"measurement_1", "measurement_2"}:
        return True
    if not _is_merged_measurement_recording(item):
        return False
    if item.get("contains_measurement_evidence") is True:
        return True
    if isinstance(item.get("measurement_subintervals"), list) and item["measurement_subintervals"]:
        return True
    base_actions = item.get("base_action_types")
    if isinstance(base_actions, list) and "measurement_action" in base_actions:
        return True
    subintervals = item.get("observed_subintervals")
    return isinstance(subintervals, list) and any(
        isinstance(subinterval, dict)
        and subinterval.get("action_type") == "measurement_action"
        for subinterval in subintervals
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runs(value: dict[str, Any]) -> list[dict[str, Any]]:
    raw = value.get("source_observed_stage_runs") or value.get("observed_stage_runs")
    if not isinstance(raw, list):
        return []
    return sorted(
        [item for item in raw if isinstance(item, dict) and isinstance(item.get("stage"), str)],
        key=lambda item: float(item.get("start_seconds") or 0.0),
    )


def _record_runs(
    summary: dict[str, Any],
    source_video_id: str,
    allowed_root: Path | None = None,
) -> list[dict[str, Any]]:
    direct = _runs(summary)
    if direct:
        return direct
    records = summary.get("records")
    if not isinstance(records, list):
        return []
    for record in records:
        if not isinstance(record, dict) or record.get("source_video_id") != source_video_id:
            continue
        direct = _runs(record)
        if direct:
            return direct
        replay_result = record.get("replay_result")
        if allowed_root is not None and isinstance(replay_result, str) and replay_result:
            raise ValueError("replay_result is forbidden in a live skill plan")
        for key in ("result_path", "replay_result"):
            nested = record.get(key)
            if not isinstance(nested, str) or not nested:
                continue
            nested_path = Path(nested).resolve()
            if allowed_root is not None and not nested_path.is_relative_to(allowed_root.resolve()):
                raise ValueError(f"live stage artifact is outside the current run: {nested_path}")
            if nested_path.is_file():
                direct = _runs(_read_json(nested_path))
                if direct:
                    return direct
    return []


def _skill(skill_id: str, rubric_ids: list[int], selected_by: str, **parameters: Any) -> dict[str, Any]:
    return {
        "skill_id": skill_id,
        "rubric_ids": rubric_ids,
        "selected_by": selected_by,
        "parameters": parameters,
    }


def select_live_skills(
    *,
    source_video_id: str,
    boundary_summary_path: Path | None,
    action_summary_path: Path | None,
    allowed_root: Path | None = None,
) -> dict[str, Any]:
    """Build a deterministic skill plan from the current run's observed situation."""
    stage_runs: list[dict[str, Any]] = []
    source_path: Path | None = None
    for candidate in (boundary_summary_path, action_summary_path):
        if candidate is None or not candidate.is_file():
            continue
        stage_runs = _record_runs(_read_json(candidate), source_video_id, allowed_root)
        if stage_runs:
            source_path = candidate
            break

    counts = Counter(str(item["stage"]) for item in stage_runs)
    wiring_count = counts["circuit_wiring"]
    rewiring_count = counts["circuit_rewiring"]
    measurement_count = counts["measurement_1"] + counts["measurement_2"]
    recording_count = counts["recording_1"] + counts["recording_2"]
    recording_cycle_ids = sorted(
        {
            str(item.get("stage"))
            for item in stage_runs
            if str(item.get("stage")) in {"recording_1", "recording_2"}
        }
    )
    merged_measurement_recording_count = sum(
        1 for item in stage_runs if _is_merged_measurement_recording(item)
    )
    merged_measurement_evidence_count = sum(
        1
        for item in stage_runs
        if _is_merged_measurement_recording(item) and _has_direct_measurement_evidence(item)
    )
    recording_cycle_count = len(recording_cycle_ids)
    measurement_context_cycle_ids = {
        str(item.get("stage")).replace("measurement_", "recording_")
        for item in stage_runs
        if _has_direct_measurement_evidence(item)
    }
    measurement_context_count = len(measurement_context_cycle_ids)
    cleanup_count = counts["material_cleanup"]
    multi_wiring = rewiring_count > 0
    has_wiring = wiring_count > 0
    has_recording = recording_count > 0

    if multi_wiring:
        switch_selected_by = "circuit rewiring observed; inspect every wiring stage"
        switch_window_mode = "all_wiring_runs"
    elif has_wiring:
        switch_selected_by = "initial wiring observed"
        switch_window_mode = "initial_wiring_only"
    else:
        switch_selected_by = "no wiring stage; broad temporal search"
        switch_window_mode = "broad_search"

    skills = [
        _skill(
            "switch.adaptive_frame_sampling",
            [3],
            switch_selected_by,
            window_mode=switch_window_mode,
            sampling_fps=5.0,
            max_rounds=2,
            max_requests_per_round=3,
            max_supplemental_frames=64,
            roi_mode="dynamic_current_frame_switch_and_plug",
            fusion_policy="same_frame_closed_and_wiring_active",
        ),
        _skill(
            "series.adaptive_terminal_sampling",
            [1],
            "circuit_rewiring observed"
            if multi_wiring
            else "initial wiring observed"
            if has_wiring
            else "no wiring stage; broad topology recovery",
            compare_latest_stable_topology=True,
            window_mode=(
                "all_wiring_runs"
                if multi_wiring
                else "initial_wiring_only"
                if has_wiring
                else "broad_search"
            ),
            stable_frames_per_stage_run=2,
            view_recovery_frames_per_stage_run=1,
            max_transition_anchors=4,
            transition_sampling_fps=5.0,
            transition_radius_seconds=1.0,
            max_supplemental_rounds=1,
            max_supplemental_frames=12,
            roi_target_long_edge=1400,
            max_model_roi_views_per_frame=1,
        ),
        _skill(
            "meter.explicit_measurement" if measurement_context_count else "meter.pre_recording_recovery",
            [5, 6],
            (
                "merged measurement-recording cycle observed"
                if merged_measurement_evidence_count
                else "measurement stage observed"
                if measurement_count
                else "derive energized window before recording"
            ),
            dynamic_meter_candidates=True,
        ),
        _skill(
            "record.two_cycle_consistency"
            if recording_cycle_count >= 2
            else "record.single_cycle_consistency"
            if recording_cycle_count == 1
            else "record.broad_cycle_search",
            [7, 9],
            f"recording cycle count={recording_cycle_count}",
            dynamic_paper_candidates=True,
            dynamic_meter_candidates=True,
        ),
        _skill(
            "cleanup.explicit_stage" if cleanup_count else "cleanup.video_tail",
            [0],
            "material_cleanup observed" if cleanup_count else "no cleanup stage; inspect terminal tail",
        ),
        _skill(
            "voltmeter.parallel_endpoint_adaptive"
            if measurement_context_count or recording_cycle_count
            else "voltmeter.parallel_endpoint_broad_search",
            [2],
            (
                "current measurement/recording cycle; scan native frames and dynamically locate meter, resistor, and endpoints"
                if measurement_context_count
                else "recording anchor may contain an unsegmented measurement; scan its adjacent cycle"
                if recording_cycle_count
                else "no measurement or recording stage; use the same broad current-run search"
            ),
            window_mode="observation_recording_cycles"
            if measurement_context_count or recording_cycle_count
            else "broad_search",
            native_decode=True,
            roi_mode="dynamic_current_frame_meter_resistor_topology",
            fusion_policy="same_frame_topology_with_temporal_consensus",
        ),
        _skill(
            "battery.recovery_episode"
            if multi_wiring
            else "battery.wiring_transition"
            if has_wiring
            else "battery.broad_transition_search",
            [8],
            "circuit_rewiring observed"
            if multi_wiring
            else "initial wiring transition observed"
            if has_wiring
            else "no wiring stage; broad transition search",
            roi_mode="dynamic_battery_and_ammeter_candidates",
        ),
        _skill(
            "polarity.explicit_measurement_dynamic_roi"
            if measurement_context_count
            else "polarity.pre_recording_dynamic_roi"
            if has_recording
            else "polarity.broad_dynamic_roi_search",
            [4],
            "merged measurement-recording cycle observed"
            if merged_measurement_evidence_count
            else "measurement stage observed"
            if measurement_count
            else "recover measurement evidence before recording"
            if has_recording
            else "no measurement or recording stage; broad visual search",
            dynamic_meter_candidates=True,
        ),
    ]
    return {
        "schema_version": "resistance_agent_live_skill_plan.v1",
        "routing_policy": LIVE_ROUTING_POLICY,
        "selection_basis": "current_video_observed_situation_only",
        "observed_stages": stage_runs,
        "selected_skills": skills,
        "video_id_used_for_routing": False,
        "historical_artifacts_used": False,
        "fixed_video_roi_used": False,
        "video_id_routing_allowed": False,
        "historical_result_artifacts_allowed": False,
        "fixed_video_roi_allowed": False,
        "stage_source_path": str(source_path.resolve()) if source_path else None,
        "stage_source_sha256": _sha256(source_path) if source_path else None,
        "stage_counts": dict(sorted(counts.items())),
        "merged_measurement_recording_count": merged_measurement_recording_count,
        "merged_measurement_evidence_count": merged_measurement_evidence_count,
        "recording_cycle_ids": recording_cycle_ids,
        "recording_cycle_count": recording_cycle_count,
        "measurement_context_cycle_ids": sorted(measurement_context_cycle_ids),
        "observed_stage_runs": stage_runs,
        "skills": skills,
    }
