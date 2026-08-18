#!/usr/bin/env python3
"""Run Rubric 3 end to end from one video without human review."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AGENT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(AGENT_ROOT / "resistance_agent"))

from toolkit import (  # noqa: E402
    ToolError,
    create_run,
    inspect_video,
    load_config,
    plan_live_skills,
    read_json,
    refine_rubric_boundaries,
    resolve_inside,
    run_full_pipeline,
    run_switch_rubric,
    sanitize_run_id,
    write_json,
)


DEFAULT_CONFIG = AGENT_ROOT / "config_resistance_live_skills_blind.json"
R3_AUTOMATIC_PARAMETERS = {
    "sampling_fps": 5.0,
    "roi_mode": "dynamic_current_frame_switch_and_plug",
    "fusion_policy": "same_frame_closed_and_wiring_active",
}
R3_IMPLEMENTATION_VERSION = "r3_opencv_same_frame_overlap_v3"
R3_IMPLEMENTATION_FINGERPRINT = hashlib.sha256(
    R3_IMPLEMENTATION_VERSION.encode("ascii")
).hexdigest()


def default_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"switch_auto_{stamp}"


def _configure_temporal_guard(config_path: Path) -> None:
    """Configure Qwen for stage discovery, never for the R3 decision."""
    config = load_config(config_path)
    settings = config.get("models", {}).get("qwen", {})
    base_url = str(os.getenv("QWEN_API_BASE_URL") or settings.get("base_url") or "").strip()
    model = str(os.getenv("QWEN_MODEL") or settings.get("model") or "qwen").strip() or "qwen"
    if not base_url:
        raise ToolError("Qwen base URL is missing from the live configuration")
    os.environ["QWEN_API_BASE_URL"] = base_url
    os.environ.setdefault("QWEN_API_TOKEN", "")
    os.environ["QWEN_MODEL"] = model


def _verify_automatic_run(plan: dict[str, Any], report: dict[str, Any]) -> None:
    expected = {
        "selection_basis": "current_video_observed_situation_only",
        "video_id_used_for_routing": False,
        "historical_artifacts_used": False,
        "fixed_video_roi_used": False,
    }
    for key, value in expected.items():
        if plan.get(key) != value:
            raise ToolError(f"automatic-run audit failed: {key}={plan.get(key)!r}")
    if report.get("human_review_used") is not False:
        raise ToolError("automatic-run audit failed: human review was used or not declared")
    if report.get("historical_fallback_used") is not False:
        raise ToolError("automatic-run audit failed: historical switch evidence was used")
    if report.get("excel_accessed") is not False or report.get("ground_truth_sent_to_model") is not False:
        raise ToolError("automatic-run audit failed: label isolation was not preserved")
    if report.get("qwen_used_for_decision") is not False:
        raise ToolError("automatic-run audit failed: Qwen participated in the R3 decision")
    if report.get("fixed_video_roi_used") is not False:
        raise ToolError("automatic-run audit failed: a fixed video ROI was used")
    diagnostics = report.get("rubric_3", {}).get("diagnostics", {})
    if diagnostics.get("decision_source") != "opencv_same_frame_overlap":
        raise ToolError("automatic-run audit failed: decision source is not OpenCV same-frame overlap")
    raw_plan_executions = plan.get("skill_executions")
    if not isinstance(raw_plan_executions, list):
        raise ToolError("automatic-run audit failed: planned executions are missing")
    plan_executions = [
        item
        for item in raw_plan_executions
        if isinstance(item, dict)
        and isinstance(item.get("rubric_ids"), list)
        and 3 in item["rubric_ids"]
    ]
    if len(plan_executions) != 1:
        raise ToolError("automatic-run audit failed: rubric 3 must have exactly one planned execution")
    plan_execution = plan_executions[0]
    report_execution = report.get("skill_execution")
    if not isinstance(report_execution, dict):
        raise ToolError("automatic-run audit failed: switch execution metadata is missing")
    for key in ("skill_id", "producer_tool", "rubric_ids"):
        if report_execution.get(key) != plan_execution.get(key):
            raise ToolError(f"automatic-run audit failed: plan/report {key} differ")
    if report_execution.get("parameters") != plan_execution.get("parameters"):
        raise ToolError("automatic-run audit failed: plan/report complete parameters differ")
    for source_name, parameters in (
        ("plan", plan_execution.get("parameters")),
        ("report", report_execution.get("parameters")),
        ("effective report", report_execution.get("effective_parameters")),
        ("rubric diagnostics", diagnostics),
    ):
        if not isinstance(parameters, dict):
            raise ToolError(f"automatic-run audit failed: {source_name} parameters are missing")
        for key, expected_value in R3_AUTOMATIC_PARAMETERS.items():
            if parameters.get(key) != expected_value:
                raise ToolError(
                    f"automatic-run audit failed: {source_name} {key}="
                    f"{parameters.get(key)!r}, expected {expected_value!r}"
                )
    plan_fingerprint = plan_execution.get("execution_fingerprint")
    if not isinstance(plan_fingerprint, str) or not plan_fingerprint:
        raise ToolError("automatic-run audit failed: planned execution fingerprint is missing")
    if report_execution.get("execution_fingerprint") != plan_fingerprint:
        raise ToolError("automatic-run audit failed: plan/report execution fingerprints differ")
    if diagnostics.get("execution_fingerprint") != plan_fingerprint:
        raise ToolError("automatic-run audit failed: plan/result execution fingerprints differ")
    for source_name, source in (
        ("plan", plan_execution),
        ("report", report_execution),
        ("rubric diagnostics", diagnostics),
    ):
        if source.get("implementation_version") != R3_IMPLEMENTATION_VERSION:
            raise ToolError(
                f"automatic-run audit failed: {source_name} implementation version drifted"
            )
        if source.get("implementation_fingerprint") != R3_IMPLEMENTATION_FINGERPRINT:
            raise ToolError(
                f"automatic-run audit failed: {source_name} implementation fingerprint drifted"
            )


def _counterexample_intervals(report: dict[str, Any]) -> list[dict[str, Any]]:
    diagnostics = report.get("rubric_3", {}).get("diagnostics", {})
    return [
        {
            "window_id": item.get("window_id"),
            "stage": item.get("stage"),
            "start_seconds": float(item["timestamp_seconds"]),
            "end_seconds": float(item["timestamp_seconds"]),
            "observation_count": 1,
            "switch_crop_path": item.get("switch_crop_path"),
            "plug_transitions": item.get("plug_transitions", []),
        }
        for item in diagnostics.get("same_frame_overlaps", [])
        if isinstance(item, dict) and isinstance(item.get("timestamp_seconds"), (int, float))
    ]


def execute_switch_auto(video_ref: str, run_id: str, config_path: Path) -> dict[str, Any]:
    _configure_temporal_guard(config_path)
    inspection = inspect_video(video_ref=video_ref, config_path=config_path)
    created = create_run(
        run_id=run_id,
        video_ref=video_ref,
        mode="execute",
        config_path=config_path,
    )
    run_dir = Path(created["run_dir"])
    pipeline = run_full_pipeline(run_id=run_id, dry_run=False)
    boundary = refine_rubric_boundaries(run_id=run_id, execute=True)
    plan = plan_live_skills(run_id=run_id)
    rubric_run = run_switch_rubric(run_id=run_id, use_fallback_temporal_guard=False)
    report_path = resolve_inside(rubric_run["evidence_report"], AGENT_ROOT)
    report = read_json(report_path)
    if not isinstance(report, dict):
        raise ToolError("switch evidence report is invalid")
    _verify_automatic_run(plan, report)

    rubric = rubric_run["rubric"]
    decision = rubric.get("decision")
    predicted_score = rubric.get("predicted_score")
    if decision not in {"pass", "fail"} or predicted_score != (1 if decision == "pass" else 0):
        raise ToolError("switch result is not a valid binary decision")
    selected_skills = [
        item
        for item in plan.get("skill_executions", [])
        if isinstance(item, dict) and 3 in item.get("rubric_ids", [])
    ]
    diagnostics = report.get("rubric_3", {}).get("diagnostics", {})
    result = {
        "schema_version": "resistance_agent_switch_auto_result.v3",
        "status": "completed",
        "run_id": run_id,
        "video_id": inspection["video_id"],
        "source_video_id": inspection["source_video_id"],
        "source_video_sha256": inspection["sha256"],
        "rubric_id": 3,
        "criterion": "电路连接过程中开关始终保持断开状态",
        "decision": decision,
        "predicted_score": predicted_score,
        "confidence": rubric.get("confidence"),
        "reason": rubric.get("reason"),
        "automatic_method": "fresh Temporal Guard stages + OpenCV dynamic knife-switch state with three-observation persistence + OpenCV base-relative wiring activity + exact same-frame AND",
        "decision_source": "opencv_same_frame_overlap",
        "implementation_version": selected_skills[0].get(
            "implementation_version", "r3_opencv_same_frame_overlap_v3"
        ),
        "implementation_fingerprint": selected_skills[0].get(
            "implementation_fingerprint", diagnostics.get("implementation_fingerprint")
        ),
        "qwen_used_for_decision": False,
        "automatic_parameters": dict(R3_AUTOMATIC_PARAMETERS),
        "execution_fingerprint": selected_skills[0]["execution_fingerprint"],
        "counterexample_intervals": _counterexample_intervals(report),
        "observation_summary": {
            "candidate_window_count": len(report.get("candidate_windows", [])),
            "sample_count": report.get("sample_count"),
            "selected_frame_count": report.get("selected_frame_count"),
            "dense_confirmation_sample_count": report.get("dense_confirmation_sample_count"),
            "dense_refinement_sample_count": report.get("dense_refinement_sample_count"),
            "switch_tracked_observation_count": diagnostics.get("switch_tracked_observation_count"),
            "real_plug_transition_count": diagnostics.get("real_plug_transition_count"),
            "wiring_active_frame_count": diagnostics.get("wiring_active_frame_count"),
            "wiring_active_interval_count": diagnostics.get("wiring_active_interval_count"),
            "same_frame_overlap_count": diagnostics.get("same_frame_overlap_count"),
        },
        "selection_basis": plan["selection_basis"],
        "observed_stages": plan["observed_stages"],
        "selected_skills": selected_skills,
        "video_id_used_for_routing": False,
        "historical_artifacts_used": False,
        "fixed_video_roi_used": False,
        "human_review_used": False,
        "excel_accessed": False,
        "candidate_windows": report.get("candidate_windows", []),
        "evidence_report": str(report_path),
        "rubric_result": rubric["result_path"],
        "pipeline_report": pipeline["run_report"],
        "boundary_summary": boundary["summary_path"],
    }
    result_path = run_dir / "switch_auto_result.json"
    write_json(result_path, result)
    reopened = read_json(result_path)
    if reopened.get("decision") != decision or reopened.get("status") != "completed":
        raise ToolError("switch auto result failed reopen verification")
    return {**result, "result_path": str(result_path.resolve())}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", "--video-ref", dest="video_ref", required=True)
    parser.add_argument("--run-id", default=default_run_id())
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    try:
        run_id = sanitize_run_id(args.run_id)
        config_path = resolve_inside(args.config, AGENT_ROOT)
        result = execute_switch_auto(str(args.video_ref), run_id, config_path)
    except (ToolError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(
            json.dumps(
                {"status": "failed", "error": f"{type(exc).__name__}: {exc}"},
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 1
    print(
        json.dumps(
            {
                "status": result["status"],
                "decision": result["decision"],
                "predicted_score": result["predicted_score"],
                "confidence": result["confidence"],
                "result_path": result["result_path"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
