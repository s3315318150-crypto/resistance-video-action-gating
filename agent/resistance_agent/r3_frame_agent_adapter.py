"""Current-run stage adapter for the experimental R3 frame sampling Agent."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import cv2

try:
    from .r3_frame_sampling_agent import (
        AGENT_VERSION,
        BASE_SAMPLING_FPS,
        DEFAULT_MAX_REQUESTS_PER_ROUND,
        DEFAULT_MAX_ROUNDS,
        DEFAULT_MAX_SUPPLEMENTAL_FRAMES,
        FUSION_POLICY,
        ROI_MODE,
        run_r3_frame_sampling_agent,
        write_json,
    )
    from .switch_rubric import candidate_windows
except ImportError:
    from r3_frame_sampling_agent import (  # type: ignore
        AGENT_VERSION,
        BASE_SAMPLING_FPS,
        DEFAULT_MAX_REQUESTS_PER_ROUND,
        DEFAULT_MAX_ROUNDS,
        DEFAULT_MAX_SUPPLEMENTAL_FRAMES,
        FUSION_POLICY,
        ROI_MODE,
        run_r3_frame_sampling_agent,
        write_json,
    )
    from switch_rubric import candidate_windows  # type: ignore


WIRING_STAGES = {"circuit_wiring", "circuit_rewiring"}
FORBIDDEN_HISTORY_CONTENT_KEYS = {
    "excel_ground_truth",
    "fixed_roi",
    "fixed_video_roi",
    "ground_truth",
    "historical_artifact",
    "historical_artifacts",
    "historical_result",
    "historical_time_windows",
    "manual_review_result",
    "previous_prediction",
    "previous_result",
    "replay_result",
    "selected_frames_pre_qwen",
}
FORBIDDEN_TRUE_AUDIT_KEYS = {
    "excel_accessed",
    "fixed_video_roi_used",
    "ground_truth_sent_to_model",
    "historical_artifacts_used",
    "historical_fallback_used",
    "selection_checkpoint_reused",
    "video_id_used_for_routing",
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _direct_stage_runs(value: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("source_observed_stage_runs", "observed_stage_runs", "observed_stage_intervals"):
        raw = value.get(key)
        if isinstance(raw, list):
            return [item for item in raw if isinstance(item, dict)]
    return []


def _contains_forbidden_history(value: Any) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).lower()
            if normalized in FORBIDDEN_HISTORY_CONTENT_KEYS:
                return True
            if normalized in FORBIDDEN_TRUE_AUDIT_KEYS and item is not False:
                return True
            if _contains_forbidden_history(item):
                return True
        return False
    if isinstance(value, list):
        return any(_contains_forbidden_history(item) for item in value)
    return False


def _nearest_run_root(path: Path) -> Path | None:
    resolved = path.resolve()
    for parent in (resolved.parent, *resolved.parents):
        if (parent / "state.json").is_file():
            return parent
    return None


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _resolve_nested_path(raw: str, summary_path: Path) -> Path:
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = summary_path.parent / candidate
    return candidate.resolve()


def _verify_video_association(value: dict[str, Any], video_path: Path) -> bool:
    source = value.get("source_video_id")
    if not isinstance(source, str) or not source:
        return False
    if Path(source).name != video_path.name:
        raise ValueError("stage result belongs to a different current-run video")
    return True


def _load_current_stage_runs(
    stage_summary_path: Path,
    video_path: Path,
) -> tuple[list[dict[str, Any]], Path, Path, bool]:
    summary_path = stage_summary_path.resolve()
    summary = _read_json(summary_path)
    if _contains_forbidden_history(summary):
        raise ValueError("historical or replay fields are forbidden for the R3 frame Agent")

    run_root = _nearest_run_root(summary_path)
    if run_root is not None:
        if not _within(video_path, run_root):
            raise ValueError("video must be the current run copy when a run root is available")
    else:
        common = Path(os.path.commonpath([str(summary_path), str(video_path.resolve())]))
        run_root = common if common.is_dir() else common.parent

    association_verified = _verify_video_association(summary, video_path)
    direct = _direct_stage_runs(summary)
    if direct:
        return direct, summary_path, run_root, association_verified

    records = summary.get("records")
    if not isinstance(records, list):
        return [], summary_path, run_root, association_verified
    rows = [item for item in records if isinstance(item, dict)]
    if len(rows) != 1:
        raise ValueError(
            "stage summary must describe exactly one current run; ID-based record selection is forbidden"
        )
    direct = _direct_stage_runs(rows[0])
    if direct:
        association_verified = _verify_video_association(rows[0], video_path)
        return direct, summary_path, run_root, association_verified
    raw_result_path = rows[0].get("result_path")
    if not isinstance(raw_result_path, str) or not raw_result_path:
        raise ValueError("single-record stage summary has no direct stages or result_path")
    result_path = _resolve_nested_path(raw_result_path, summary_path)
    if not _within(result_path, run_root):
        raise ValueError("nested stage result must stay inside the current run")
    if not result_path.is_file():
        raise FileNotFoundError(result_path)
    result = _read_json(result_path)
    if _contains_forbidden_history(result):
        raise ValueError("nested stage result contains forbidden historical fields")
    association_verified = _verify_video_association(result, video_path)
    return _direct_stage_runs(result), result_path, run_root, association_verified


def _duration(video_path: Path) -> float:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"unable to open video: {video_path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if fps <= 0.0 or frames <= 0:
        raise RuntimeError("video metadata is invalid")
    return frames / fps


def _live_skill_parameters(skill_execution: dict[str, Any]) -> dict[str, Any]:
    if skill_execution.get("skill_id") != "switch.adaptive_frame_sampling":
        raise ValueError("formal R3 frame Agent requires switch.adaptive_frame_sampling")
    parameters = skill_execution.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError("formal R3 frame Agent parameters must be an object")
    allowed = {
        "window_mode",
        "sampling_fps",
        "roi_mode",
        "fusion_policy",
        "max_rounds",
        "max_requests_per_round",
        "max_supplemental_frames",
    }
    unknown = sorted(set(parameters) - allowed)
    if unknown:
        raise ValueError(f"unknown formal R3 frame Agent parameters: {unknown}")
    required = allowed - set(parameters)
    if required:
        raise ValueError(f"missing formal R3 frame Agent parameters: {sorted(required)}")
    if parameters["window_mode"] not in {
        "all_wiring_runs",
        "initial_wiring_only",
        "broad_search",
    }:
        raise ValueError("invalid formal R3 frame Agent window_mode")
    if float(parameters["sampling_fps"]) != BASE_SAMPLING_FPS:
        raise ValueError("formal R3 frame Agent baseline must remain 5 fps")
    if parameters["roi_mode"] != ROI_MODE or parameters["fusion_policy"] != FUSION_POLICY:
        raise ValueError("formal R3 frame Agent must retain dynamic ROI and same-frame fusion")
    limits = {
        "max_rounds": (1, 2),
        "max_requests_per_round": (1, 3),
        "max_supplemental_frames": (1, 96),
    }
    normalized = dict(parameters)
    for key, (minimum, maximum) in limits.items():
        value = parameters[key]
        if type(value) is not int or not minimum <= value <= maximum:
            raise ValueError(f"formal R3 frame Agent {key} must be {minimum}..{maximum}")
        normalized[key] = int(value)
    normalized["sampling_fps"] = float(parameters["sampling_fps"])
    return normalized


def run_r3_frame_agent_live_skill(
    *,
    video_path: Path,
    source_video_id: str,
    video_id: str,
    run_dir: Path,
    stage_summary_path: Path,
    skill_execution: dict[str, Any],
    routing_policy: str,
) -> dict[str, Any]:
    """Run the adaptive Agent as a formal current-run R3 evidence producer."""
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    if not stage_summary_path.is_file():
        raise FileNotFoundError(stage_summary_path)
    parameters = _live_skill_parameters(skill_execution)
    execution_fingerprint = skill_execution.get("execution_fingerprint")
    if (
        not isinstance(execution_fingerprint, str)
        or len(execution_fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in execution_fingerprint)
    ):
        raise ValueError("formal R3 frame Agent execution fingerprint is invalid")
    (
        stage_runs,
        resolved_stage_result_path,
        current_run_root,
        video_association_verified,
    ) = _load_current_stage_runs(stage_summary_path, video_path)
    if current_run_root.resolve() != run_dir.resolve():
        raise ValueError("formal R3 frame Agent stage evidence must belong to the current run")
    windows = candidate_windows(
        {"observed_stage_runs": stage_runs},
        _duration(video_path),
        str(parameters["window_mode"]),
    )
    evidence_dir = (
        run_dir
        / "switch_rubric"
        / "adaptive_frame_agent"
        / execution_fingerprint[:16]
    )
    agent = run_r3_frame_sampling_agent(
        video_path=video_path,
        candidate_windows=windows,
        output_dir=evidence_dir / "agent",
        max_rounds=parameters["max_rounds"],
        max_requests_per_round=parameters["max_requests_per_round"],
        max_supplemental_frames=parameters["max_supplemental_frames"],
    )
    diagnostics = {
        "algorithm_version": AGENT_VERSION,
        "original_algorithm_version": agent.get("original_algorithm_version"),
        "original_algorithm_fingerprint": agent.get("original_algorithm_fingerprint"),
        "execution_fingerprint": execution_fingerprint,
        "candidate_windows": windows,
        "sampling_policy": agent["sampling_policy"],
        "initial_evidence_quality": agent["initial_evidence_quality"],
        "final_evidence_quality": agent["final_evidence_quality"],
        "request_rounds": agent["request_rounds"],
        "requests": agent["requests"],
        "request_count": agent["request_count"],
        "supplemental_actual_new_frame_count": agent[
            "supplemental_actual_new_frame_count"
        ],
        "stop_reason": agent["stop_reason"],
        "shared_threshold_fusion": agent["shared_threshold_fusion"],
        "evidence_frames": agent["evidence_frames"],
        "evidence_frame_count": agent["evidence_frame_count"],
        "agent_report_path": agent["report_path"],
    }
    rubric_3 = {
        "decision": agent["decision"],
        "predicted_score": agent["predicted_score"],
        "confidence": agent["confidence"],
        "reason": agent["reason"],
        "diagnostics": diagnostics,
    }
    report = {
        "schema_version": "resistance_agent_switch_evidence.v4",
        "execution_scope": "formal_execute_evidence_producer",
        "video_id": video_id,
        "source_video_id": source_video_id,
        "source_video_path": str(video_path.resolve()),
        "stage_summary_path": str(stage_summary_path.resolve()),
        "resolved_stage_result_path": str(resolved_stage_result_path),
        "video_association_verified": video_association_verified,
        "selection_basis": "current_video_observed_situation_only",
        "observed_stages": stage_runs,
        "candidate_windows": windows,
        "skill_execution": skill_execution,
        "routing_policy": routing_policy,
        "agent_report_path": agent["report_path"],
        "rubric_3": rubric_3,
        "video_id_used_for_routing": False,
        "historical_artifacts_used": False,
        "fixed_video_roi_used": False,
        "excel_accessed": False,
        "ground_truth_sent_to_model": False,
        "qwen_used_for_decision": False,
    }
    report_path = evidence_dir / "switch_evidence_report.json"
    write_json(report_path, report)
    reopened = _read_json(report_path)
    if (
        reopened.get("rubric_3", {}).get("decision") not in {"pass", "fail"}
        or reopened.get("skill_execution", {}).get("execution_fingerprint")
        != execution_fingerprint
        or reopened.get("video_id_used_for_routing") is not False
        or reopened.get("historical_artifacts_used") is not False
        or reopened.get("fixed_video_roi_used") is not False
        or not Path(str(reopened.get("agent_report_path") or "")).is_file()
    ):
        raise RuntimeError("formal R3 frame Agent evidence report failed verification")
    return {
        "rubric_3": rubric_3,
        "report_path": str(report_path.resolve()),
        "agent_report_path": agent["report_path"],
    }


def run_r3_frame_agent_from_current_stages(
    *,
    video_path: Path,
    stage_summary_path: Path,
    output_dir: Path,
    association_id: str | None = None,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    max_requests_per_round: int = DEFAULT_MAX_REQUESTS_PER_ROUND,
    max_supplemental_frames: int = DEFAULT_MAX_SUPPLEMENTAL_FRAMES,
) -> dict[str, Any]:
    """Run the Agent from one direct current-run stage summary."""
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    if not stage_summary_path.is_file():
        raise FileNotFoundError(stage_summary_path)
    (
        stage_runs,
        resolved_stage_result_path,
        current_run_root,
        video_association_verified,
    ) = _load_current_stage_runs(stage_summary_path, video_path)
    observed = [
        item
        for item in stage_runs
        if str(item.get("stage") or item.get("label") or "") in WIRING_STAGES
    ]
    has_wiring = any(
        str(item.get("stage") or item.get("label") or "") == "circuit_wiring"
        for item in observed
    )
    has_rewiring = any(
        str(item.get("stage") or item.get("label") or "") == "circuit_rewiring"
        for item in observed
    )
    window_mode = (
        "all_wiring_runs" if has_rewiring else "initial_wiring_only" if has_wiring else "broad_search"
    )
    duration_seconds = _duration(video_path)
    windows = candidate_windows(
        {"observed_stage_runs": stage_runs}, duration_seconds, window_mode
    )
    agent_dir = output_dir / "r3_frame_sampling_agent"
    agent = run_r3_frame_sampling_agent(
        video_path=video_path,
        candidate_windows=windows,
        output_dir=agent_dir,
        max_rounds=max_rounds,
        max_requests_per_round=max_requests_per_round,
        max_supplemental_frames=max_supplemental_frames,
    )
    diagnostics = {
        "algorithm_version": AGENT_VERSION,
        "base_sampling_fps": BASE_SAMPLING_FPS,
        "roi_mode": ROI_MODE,
        "fusion_policy": FUSION_POLICY,
        "candidate_windows": windows,
        "request_count": agent["request_count"],
        "supplemental_actual_new_frame_count": agent[
            "supplemental_actual_new_frame_count"
        ],
        "initial_evidence_quality": agent["initial_evidence_quality"],
        "final_evidence_quality": agent["final_evidence_quality"],
        "stop_reason": agent["stop_reason"],
        "agent_report_path": agent["report_path"],
        "evidence_frames": agent["evidence_frames"],
    }
    rubric_3 = {
        "schema_version": "resistance_agent_r3_frame_agent_result.v1",
        "rubric_id": 3,
        "decision": agent["decision"],
        "predicted_score": agent["predicted_score"],
        "confidence": agent["confidence"],
        "reason": agent["reason"],
        "execution_scope": "standalone_experiment",
        "formal_execute_integrated": False,
        "diagnostics": diagnostics,
    }
    rubric_path = output_dir / "rubric_3.json"
    write_json(rubric_path, rubric_3)
    report = {
        "schema_version": "resistance_agent_r3_frame_agent_adapter.v1",
        "association_id": association_id,
        "source_video_path": str(video_path.resolve()),
        "stage_summary_path": str(stage_summary_path.resolve()),
        "resolved_stage_result_path": str(resolved_stage_result_path),
        "current_run_root": str(current_run_root),
        "video_association_verified": video_association_verified,
        "selection_basis": "current_video_observed_situation_only",
        "observed_stages": stage_runs,
        "selected_skills": [
            {
                "rubric_ids": [3],
                "skill_id": AGENT_VERSION,
                "parameters": {
                    "window_mode": window_mode,
                    "baseline_sampling_fps": BASE_SAMPLING_FPS,
                    "supplemental_sampling_fps": BASE_SAMPLING_FPS,
                    "max_rounds": max_rounds,
                    "max_requests_per_round": max_requests_per_round,
                    "max_supplemental_frames": max_supplemental_frames,
                    "roi_mode": ROI_MODE,
                    "fusion_policy": FUSION_POLICY,
                },
                "selected_by": "current run wiring stages and visual evidence quality",
            }
        ],
        "video_id_used_for_routing": False,
        "historical_artifacts_used": False,
        "fixed_video_roi_used": False,
        "excel_accessed": False,
        "ground_truth_sent_to_model": False,
        "rubric_3": rubric_3,
        "rubric_path": str(rubric_path.resolve()),
        "agent_report_path": agent["report_path"],
    }
    report_path = output_dir / "r3_frame_agent_adapter_report.json"
    write_json(report_path, report)
    reopened = _read_json(report_path)
    if (
        reopened.get("rubric_3", {}).get("decision") not in {"pass", "fail"}
        or reopened.get("rubric_3", {}).get("formal_execute_integrated") is not False
        or not Path(str(reopened.get("rubric_path") or "")).is_file()
        or not Path(str(reopened.get("agent_report_path") or "")).is_file()
        or reopened.get("video_id_used_for_routing") is not False
        or reopened.get("historical_artifacts_used") is not False
        or reopened.get("fixed_video_roi_used") is not False
    ):
        raise RuntimeError("R3 frame Agent adapter report failed reopen verification")
    return {**report, "report_path": str(report_path.resolve())}
