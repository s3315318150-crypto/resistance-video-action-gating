#!/usr/bin/env python3
"""Whitelisted project tools exposed by the resistance-video MCP server."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable


AGENT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = AGENT_ROOT.parent
ROOT = PROJECT_ROOT
DEFAULT_CONFIG = AGENT_ROOT / "config.json"
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
DECISIONS = {"pass", "fail"}
LIVE_ROUTING_POLICY = "live_situation_skills.v1"
FIVE_STAGE_ALGORITHM_ID = "qwen_experiment_action_hierarchical_v2_five_stage"
FIVE_STAGE_SCHEMA_ID = "resistance_5stage_measurement_recording_v1"
FIVE_STAGE_OUTPUT_SCHEMA_VERSION = "qwen_experiment_action_hierarchical_v2_five_stage.v1"
FIVE_STAGE_CONFIG_SCHEMA_VERSION = "resistance_action_schema.v1"
FIVE_STAGE_NAMES = (
    "circuit_wiring",
    "recording_1",
    "circuit_rewiring",
    "recording_2",
    "material_cleanup",
)
V2_ACTION_VERSION = "v2"
V2_ALGORITHM_ID = "qwen_experiment_action_hierarchical_v2"
V2_OUTPUT_SCHEMA_VERSION = "qwen_experiment_action_hierarchical_v2.v1"
V2_FRAME_AGENT_ACTION_VERSION = "v2-frame-agent"
V2_FRAME_AGENT_ALGORITHM_ID = "qwen_experiment_action_hierarchical_v2_frame_agent"
V2_FRAME_AGENT_OUTPUT_SCHEMA_VERSION = "qwen_experiment_action_hierarchical_v2_frame_agent.v1"
SCREENSHOT_GUARD_ACTION_VERSION = "v2-screenshot-guard"
SCREENSHOT_GUARD_ALGORITHM_ID = "qwen_experiment_action_hierarchical_v2_screenshot_guard"
SCREENSHOT_GUARD_SCHEMA_ID = "resistance_7stage_no_battery_v2"
SCREENSHOT_GUARD_OUTPUT_SCHEMA_VERSION = "qwen_experiment_action_hierarchical_v2_screenshot_guard.v1"
SCREENSHOT_GUARD_AGENT_ACTION_VERSION = "v2-screenshot-guard-agent"
SCREENSHOT_GUARD_AGENT_ALGORITHM_ID = "qwen_experiment_action_hierarchical_v2_screenshot_guard_agent"
SCREENSHOT_GUARD_AGENT_OUTPUT_SCHEMA_VERSION = (
    "qwen_experiment_action_hierarchical_v2_screenshot_guard_agent.v1"
)
SCREENSHOT_GUARD_CONFIG_SCHEMA_VERSION = "resistance_action_schema.v2"
SCREENSHOT_GUARD_STAGE_NAMES = (
    "circuit_wiring",
    "measurement_1",
    "recording_1",
    "circuit_rewiring",
    "measurement_2",
    "recording_2",
    "material_cleanup",
)
FORBIDDEN_LIVE_WORKFLOW_KEYS = frozenset(
    {
        "ammeter_search",
        "best_fusion",
        "detector_root",
        "fallback_action_summary",
        "fixed_roi",
        "fixed_video_roi",
        "freeze_path",
        "ground_truth",
        "reference_manifest",
        "results_root",
        "roi_by_video",
        "rubric8_action_summary",
        "segment_source",
        "specialized_best",
        "stage_manifest",
        "supported_video_ids",
        "v15_best",
        "video_id",
        "video_ids",
        "video_rois",
        "voltmeter_search",
    }
)
REDACTED_KEYS = re.compile(r"(?:api[_-]?key|token|authorization|secret|password)", re.I)
REDACTED_VALUES = re.compile(
    r"(?i)\b(?:sk-[A-Za-z0-9_-]{8,}|bearer\s+[A-Za-z0-9._~+/-]{8,})\b"
)


class ToolError(RuntimeError):
    """A stable error type returned through MCP instead of a traceback."""


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sanitize_run_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip()).strip("._")
    if not cleaned or len(cleaned) > 80:
        raise ToolError("run_id must contain 1-80 safe characters")
    return cleaned


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): "<redacted>" if REDACTED_KEYS.search(str(key)) else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, str):
        return REDACTED_VALUES.sub("<redacted>", value)
    return value


def resolve_inside(
    path: str | Path,
    allowed_root: Path = ROOT,
    must_exist: bool = True,
) -> Path:
    value = Path(path)
    resolved = (value if value.is_absolute() else allowed_root / value).resolve()
    root = allowed_root.resolve()
    if resolved != root and root not in resolved.parents:
        raise ToolError(f"path is outside the project: {resolved}")
    if must_exist and not resolved.exists():
        raise ToolError(f"path does not exist: {resolved}")
    return resolved


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _forbidden_live_workflow_paths(value: Any, path: tuple[str, ...] = ()) -> list[str]:
    violations: list[str] = []
    if isinstance(value, dict):
        for raw_key, item in value.items():
            key = str(raw_key)
            child_path = (*path, key)
            if (
                key in FORBIDDEN_LIVE_WORKFLOW_KEYS
                or re.fullmatch(r"video_\d+(?:_.+)?", key, flags=re.I)
            ):
                violations.append(".".join(child_path))
            violations.extend(_forbidden_live_workflow_paths(item, child_path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            violations.extend(_forbidden_live_workflow_paths(item, (*path, str(index))))
    return violations


def _validate_live_workflow_config(config: dict[str, Any]) -> None:
    workflow = config.get("workflow")
    if not isinstance(workflow, dict):
        raise ToolError("agent config workflow must be an object")
    violations = sorted(set(_forbidden_live_workflow_paths(workflow, ("workflow",))))
    if violations:
        raise ToolError(
            "live workflow contains replay/history-only routing settings: "
            + ", ".join(violations)
        )


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = resolve_inside(path, AGENT_ROOT)
    value = read_json(config_path)
    if not isinstance(value, dict) or value.get("schema_version") != "resistance_agent_config.v1":
        raise ToolError(f"invalid agent config: {config_path}")
    _validate_live_workflow_config(value)
    return value


def _config_path(value: str | Path | None) -> Path:
    return resolve_inside(value or DEFAULT_CONFIG, AGENT_ROOT)


def _configured_video_dirs(config: dict[str, Any]) -> list[Path]:
    values = config.get("video_dirs") or ["."]
    if not isinstance(values, list) or not values:
        raise ToolError("config video_dirs must be a non-empty list")
    output: list[Path] = []
    for value in values:
        path = resolve_inside(str(value), PROJECT_ROOT, must_exist=False)
        if path.is_dir() and path not in output:
            output.append(path)
    return output


def _display_video_id(filename: str) -> str:
    match = re.match(r"^(\d+)(?:_|$)", filename)
    return match.group(1) if match else Path(filename).stem


def _video_paths(config: dict[str, Any]) -> list[tuple[str, Path]]:
    records: list[tuple[str, Path]] = []
    seen: set[Path] = set()
    for directory in _configured_video_dirs(config):
        for path in sorted(directory.iterdir(), key=lambda item: item.name):
            resolved = path.resolve()
            if not path.is_file() or path.suffix.lower() not in VIDEO_SUFFIXES or resolved in seen:
                continue
            seen.add(resolved)
            records.append((_display_video_id(path.name), resolved))
    return records


def discover_videos(config_path: str | Path | None = None) -> dict[str, Any]:
    config = load_config(_config_path(config_path))
    records = [
        {
            "video_id": video_id,
            "source_video_id": path.name,
            "path": str(path),
            "bytes": path.stat().st_size,
        }
        for video_id, path in _video_paths(config)
    ]
    return {"status": "completed", "count": len(records), "videos": records}


@lru_cache(maxsize=32)
def _inspect_video_cached(config_digest: str, config_path: str, video_ref: str) -> dict[str, Any]:
    del config_digest
    explicit_path = Path(video_ref).expanduser()
    if explicit_path.is_file() and explicit_path.suffix.lower() in VIDEO_SUFFIXES:
        resolved = explicit_path.resolve()
        matches = [(_display_video_id(resolved.name), resolved)]
    else:
        config = load_config(config_path)
        matches = [
            (video_id, path)
            for video_id, path in _video_paths(config)
            if video_ref in {video_id, path.name, str(path)}
        ]
    if len(matches) != 1:
        raise ToolError(f"expected one video for {video_ref}, found {len(matches)}")
    video_id, path = matches[0]
    record: dict[str, Any] = {
        "video_id": video_id,
        "source_video_id": path.name,
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }
    try:
        import cv2

        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise ToolError(f"unable to open video {video_ref}")
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        finally:
            capture.release()
        record.update(
            {
                "fps": round(fps, 6),
                "frame_count": frame_count,
                "duration_seconds": round(frame_count / fps, 3) if fps > 0 else None,
                "width": width,
                "height": height,
            }
        )
    except ImportError:
        record["metadata_status"] = "opencv_not_available"
    return record


def inspect_video(
    video_ref: str | None = None,
    video_id: str | None = None,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    selected = str(video_ref if video_ref is not None else video_id or "")
    if not selected:
        raise ToolError("video_ref is required")
    path = _config_path(config_path)
    return dict(_inspect_video_cached(sha256(path), str(path), selected))


def _new_run_dir(run_id: str) -> Path:
    path = AGENT_ROOT / "runs" / sanitize_run_id(run_id)
    if path.exists() and any(path.iterdir()):
        raise ToolError(f"refusing to overwrite non-empty run: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _existing_run_dir(run_id: str) -> Path:
    return resolve_inside(AGENT_ROOT / "runs" / sanitize_run_id(run_id), AGENT_ROOT)


def create_run(
    run_id: str,
    video_ref: str | None = None,
    video_id: str | None = None,
    mode: str = "replay",
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    if mode not in {"replay", "prepare", "execute"}:
        raise ToolError("mode must be replay, prepare, or execute")
    config_file = _config_path(config_path)
    video = inspect_video(video_ref=video_ref, video_id=video_id, config_path=config_file)
    safe_id = sanitize_run_id(run_id)
    existing = AGENT_ROOT / "runs" / safe_id / "state.json"
    if existing.is_file():
        state = read_json(existing)
        if (
            state.get("run_id") == safe_id
            and state.get("video", {}).get("path") == video["path"]
            and state.get("mode") == mode
            and state.get("config_sha256") == sha256(config_file)
        ):
            return {
                "status": "existing",
                "run_id": safe_id,
                "run_dir": str(existing.parent.resolve()),
                "state_path": str(existing.resolve()),
                "video": video,
            }
    run_dir = _new_run_dir(run_id)
    state = {
        "schema_version": "resistance_agent_run.v2",
        "run_id": sanitize_run_id(run_id),
        "video_id": video["video_id"],
        "source_video_id": video["source_video_id"],
        "mode": mode,
        "status": "created",
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "config_path": str(config_file),
        "config_sha256": sha256(config_file),
        "video": video,
        "action_summary": None,
        "boundary_plan": None,
        "boundary_summary": None,
        "rubric_summary": None,
        "rubric_evidence_reports": {},
        "rubric_results": {},
        "frame_agent_report": None,
        "r3_frame_agent_report": None,
        "skill_plan": None,
        "tool_calls": [],
        "final_result": None,
    }
    write_json(run_dir / "state.json", state)
    return {
        "status": "created",
        "run_id": state["run_id"],
        "run_dir": str(run_dir.resolve()),
        "state_path": str((run_dir / "state.json").resolve()),
        "video": video,
    }


def _state(run_id: str) -> tuple[Path, dict[str, Any]]:
    run_dir = _existing_run_dir(run_id)
    state_path = run_dir / "state.json"
    state = read_json(state_path)
    if not isinstance(state, dict):
        raise ToolError(f"invalid state: {state_path}")
    return run_dir, state


def _save_state(run_dir: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = utc_now()
    write_json(run_dir / "state.json", redact(state))


def _state_config(state: dict[str, Any]) -> dict[str, Any]:
    path = resolve_inside(state["config_path"], AGENT_ROOT)
    if sha256(path) != state.get("config_sha256"):
        raise ToolError("agent config changed after run creation")
    return load_config(path)


def _ensure_live_skill_plan(run_dir: Path, state: dict[str, Any]) -> dict[str, Any]:
    if state.get("mode") != "execute":
        raise ToolError("live skills are only available in execute mode")
    try:
        from .skills import SkillExecutionError, bind_skill_plan, select_live_skills
    except ImportError:
        from skills import SkillExecutionError, bind_skill_plan, select_live_skills  # type: ignore

    def optional_path(value: Any) -> Path | None:
        if not isinstance(value, str) or not value:
            return None
        candidate = resolve_inside(value, run_dir, must_exist=False)
        return candidate if candidate.is_file() else None

    plan = select_live_skills(
        source_video_id=str(state.get("source_video_id") or state.get("video_id") or ""),
        boundary_summary_path=optional_path(state.get("boundary_summary")),
        action_summary_path=optional_path(state.get("action_summary")),
        allowed_root=run_dir,
    )
    try:
        plan["skill_executions"] = bind_skill_plan(plan)
    except SkillExecutionError as exc:
        raise ToolError(f"live skill plan is not executable: {exc}") from exc
    existing_path = state.get("skill_plan")
    if isinstance(existing_path, str) and existing_path:
        path = resolve_inside(existing_path, run_dir, must_exist=False)
        if path.is_file():
            existing = read_json(path)
            existing_fingerprints = [
                item.get("execution_fingerprint")
                for item in existing.get("skill_executions", [])
                if isinstance(item, dict)
            ]
            planned_fingerprints = [
                item.get("execution_fingerprint")
                for item in plan["skill_executions"]
                if isinstance(item, dict)
            ]
            if (
                existing.get("routing_policy") == LIVE_ROUTING_POLICY
                and existing.get("stage_source_sha256") == plan.get("stage_source_sha256")
                and isinstance(existing.get("skill_executions"), list)
                and existing_fingerprints == planned_fingerprints
            ):
                return existing
            previous_by_rubric = {
                rubric_id: item.get("execution_fingerprint")
                for item in existing.get("skill_executions", [])
                if isinstance(item, dict)
                for rubric_id in item.get("rubric_ids", [])
                if type(rubric_id) is int
            }
            planned_by_rubric = {
                rubric_id: item.get("execution_fingerprint")
                for item in plan["skill_executions"]
                if isinstance(item, dict)
                for rubric_id in item.get("rubric_ids", [])
                if type(rubric_id) is int
            }
            changed_rubrics = {
                rubric_id
                for rubric_id in set(previous_by_rubric) | set(planned_by_rubric)
                if previous_by_rubric.get(rubric_id) != planned_by_rubric.get(rubric_id)
            }
            # Keep unaffected current-run predictions. A changed R3 executor
            # must not force the other nine visual producers to run again.
            for rubric_id in changed_rubrics:
                state.setdefault("rubric_results", {}).pop(str(rubric_id), None)
            evidence_reports = state.setdefault("rubric_evidence_reports", {})
            for key in list(evidence_reports):
                grouped_ids = {
                    int(part) for part in str(key).split("_") if part.isdigit()
                }
                if grouped_ids.intersection(changed_rubrics):
                    evidence_reports.pop(key, None)
            if 3 in changed_rubrics:
                state["r3_frame_agent_report"] = None
            if changed_rubrics:
                state.pop("rubric_summary", None)
                state["final_result"] = None
                state["status"] = "boundaries_completed"
    plan_path = run_dir / "skills" / "live_skill_plan.json"
    write_json(plan_path, plan)
    state["skill_plan"] = str(plan_path.resolve())
    return plan


def _rubric_skill_selection(plan: dict[str, Any], rubric_ids: tuple[int, ...]) -> list[dict[str, Any]]:
    requested = set(rubric_ids)
    return [
        item
        for item in plan.get("skills", [])
        if isinstance(item, dict) and requested.intersection(item.get("rubric_ids") or [])
    ]


def plan_live_skills(run_id: str) -> dict[str, Any]:
    """Select evidence skills from current Temporal Guard stages without video-ID routing."""
    run_dir, state = _state(run_id)
    previous: dict[str, Any] | None = None
    if isinstance(state.get("skill_plan"), str) and Path(state["skill_plan"]).is_file():
        value = read_json(Path(state["skill_plan"]))
        previous = value if isinstance(value, dict) else None
    plan = _ensure_live_skill_plan(run_dir, state)
    previous_fingerprints = [
        item.get("execution_fingerprint")
        for item in (previous or {}).get("skill_executions", [])
        if isinstance(item, dict)
    ]
    planned_fingerprints = [
        item.get("execution_fingerprint")
        for item in plan.get("skill_executions", [])
        if isinstance(item, dict)
    ]
    existing = bool(
        previous
        and previous.get("routing_policy") == LIVE_ROUTING_POLICY
        and previous.get("stage_source_sha256") == plan.get("stage_source_sha256")
        and isinstance(previous.get("skill_executions"), list)
        and previous_fingerprints == planned_fingerprints
    )
    if not existing:
        state["tool_calls"].append(
            {
                "tool": "plan_live_skills",
                "routing_policy": plan["routing_policy"],
                "skill_ids": [item["skill_id"] for item in plan["skills"]],
                "at": utc_now(),
            }
        )
        _save_state(run_dir, state)
    return {
        "status": "live_skills_planned",
        "run_id": state["run_id"],
        "video_id": state["video_id"],
        "routing_policy": plan["routing_policy"],
        "selection_basis": plan["selection_basis"],
        "video_id_routing_allowed": plan["video_id_routing_allowed"],
        "historical_result_artifacts_allowed": plan["historical_result_artifacts_allowed"],
        "fixed_video_roi_allowed": plan["fixed_video_roi_allowed"],
        "video_id_used_for_routing": plan["video_id_used_for_routing"],
        "historical_artifacts_used": plan["historical_artifacts_used"],
        "fixed_video_roi_used": plan["fixed_video_roi_used"],
        "observed_stages": plan["observed_stages"],
        "selected_skills": plan["selected_skills"],
        "stage_counts": plan["stage_counts"],
        "skills": plan["skills"],
        "skill_executions": plan["skill_executions"],
        "plan_path": state["skill_plan"],
        "idempotent_replay": existing,
    }


def run_adaptive_frame_agent(
    run_id: str,
    max_rounds: int = 2,
    initial_interval_seconds: float = 0.5,
) -> dict[str, Any]:
    """Extract current-run frame groups and request adjacent frames when weak."""
    run_dir, state = _state(run_id)
    if state.get("mode") != "execute":
        raise ToolError("adaptive frame agent is only valid in execute mode")
    config = _state_config(state)
    model_config = config.get("models", {}).get("qwen")
    if not isinstance(model_config, dict):
        raise ToolError("Qwen model configuration is missing")
    try:
        from .adaptive_frame_agent_v3 import FrameAgentError, run_adaptive_frame_agent as execute_agent
    except ImportError:
        from adaptive_frame_agent_v3 import FrameAgentError, run_adaptive_frame_agent as execute_agent  # type: ignore
    try:
        report = execute_agent(
            run_dir=run_dir,
            state=state,
            model_config=model_config,
            max_rounds=max_rounds,
            initial_interval_seconds=initial_interval_seconds,
        )
    except FrameAgentError as exc:
        raise ToolError(str(exc)) from exc
    state["frame_agent_report"] = report["report_path"]
    state["tool_calls"].append(
        {
            "tool": "run_adaptive_frame_agent",
            "report_path": report["report_path"],
            "round_count": report["round_count"],
            "frame_useful": report["frame_useful"],
            "meter_pair_complete": report["meter_pair_complete"],
            "needs_more_frames": report["needs_more_frames"],
            "at": utc_now(),
        }
    )
    _save_state(run_dir, state)
    return {
        "status": report["status"],
        "run_id": run_id,
        "report_path": report["report_path"],
        "round_count": report["round_count"],
        "selected_frame_count": len(report.get("selected_frames") or []),
        "visible_roles": report.get("visible_roles") or [],
        "frame_useful": report["frame_useful"],
        "meter_pair_complete": report["meter_pair_complete"],
        "needs_more_frames": report["needs_more_frames"],
        "request_limit_reached": report["request_limit_reached"],
        "selection_basis": report["selection_basis"],
        "video_id_used_for_routing": False,
        "historical_artifacts_used": False,
        "fixed_video_roi_used": False,
        "source_video_unchanged": True,
    }


def _live_skill_plan(run_dir: Path, state: dict[str, Any]) -> dict[str, Any]:
    """Return the current situation plan and persist it before live evidence runs."""
    plan = _ensure_live_skill_plan(run_dir, state)
    if state.get("skill_plan") != str((run_dir / "skills" / "live_skill_plan.json").resolve()):
        state["skill_plan"] = str((run_dir / "skills" / "live_skill_plan.json").resolve())
    return plan


def _skill_metadata(plan: dict[str, Any], rubric_ids: tuple[int, ...]) -> dict[str, Any]:
    try:
        from .skills import SkillExecutionError, executions_for_rubrics
    except ImportError:
        from skills import SkillExecutionError, executions_for_rubrics  # type: ignore
    if plan.get("selection_basis") == "grouping-only fixture":
        executions = []
    else:
        try:
            executions = executions_for_rubrics(plan, rubric_ids)
        except SkillExecutionError as exc:
            raise ToolError(f"live skill execution resolution failed: {exc}") from exc
    return {
        "routing_policy": plan["routing_policy"],
        "selection_basis": plan["selection_basis"],
        "video_id_used_for_routing": bool(plan.get("video_id_used_for_routing", False)),
        "historical_artifacts_used": bool(plan.get("historical_artifacts_used", False)),
        "fixed_video_roi_used": bool(plan.get("fixed_video_roi_used", False)),
        "skill_selection": _rubric_skill_selection(plan, rubric_ids),
        "skill_executions": executions,
    }


def _reject_historical_live_fallback(use_fallback_temporal_guard: bool) -> None:
    if use_fallback_temporal_guard:
        raise ToolError(
            "historical Temporal Guard fallback is available only to explicit replay/regression tools"
        )


def _command(
    command: list[str],
    timeout_seconds: int,
    cwd: Path = ROOT,
    env_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    environment = os.environ.copy()
    if env_overrides:
        environment.update(env_overrides)
    try:
        process = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            env=environment,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ToolError(f"command timed out after {timeout_seconds}s: {command[0]}") from exc
    result = {
        "command": command,
        "cwd": str(cwd.resolve()),
        "exit_status": process.returncode,
        "stdout": process.stdout[-8000:],
        "stderr": process.stderr[-8000:],
    }
    if process.returncode != 0:
        raise ToolError(json.dumps(redact(result), ensure_ascii=False))
    return result


def _release_root(config: dict[str, Any]) -> Path:
    return resolve_inside(config["workflow"]["release_root"], PROJECT_ROOT)


def _five_stage_contract(config: dict[str, Any]) -> tuple[str, str]:
    workflow = config.get("workflow")
    settings = workflow.get("action_segmentation") if isinstance(workflow, dict) else None
    if not isinstance(settings, dict):
        raise ToolError("workflow.action_segmentation must be configured for five-stage")
    algorithm_id = settings.get("algorithm_id")
    schema_id = settings.get("stage_schema_id")
    stage_count = settings.get("stage_count")
    if algorithm_id != FIVE_STAGE_ALGORITHM_ID:
        raise ToolError(
            "five-stage action algorithm_id must be "
            f"{FIVE_STAGE_ALGORITHM_ID}, got {algorithm_id!r}"
        )
    if schema_id != FIVE_STAGE_SCHEMA_ID:
        raise ToolError(
            f"five-stage stage_schema_id must be {FIVE_STAGE_SCHEMA_ID}, got {schema_id!r}"
        )
    if stage_count != len(FIVE_STAGE_NAMES):
        raise ToolError(f"five-stage stage_count must be {len(FIVE_STAGE_NAMES)}")
    script_value = settings.get("script")
    schema_value = settings.get("schema")
    if not isinstance(script_value, str) or not script_value.strip():
        raise ToolError("five-stage action script must be configured")
    if not isinstance(schema_value, str) or not schema_value.strip():
        raise ToolError("five-stage action schema must be configured")
    script_path = resolve_inside(script_value, PROJECT_ROOT, must_exist=True)
    if script_path.name != "qwen_experiment_action_hierarchical_v2_five_stage.py":
        raise ToolError(f"five-stage action script is not the canonical executor: {script_path}")
    schema_path = resolve_inside(schema_value, PROJECT_ROOT, must_exist=True)
    if schema_path.name != "resistance_5stage_measurement_recording_v1.json":
        raise ToolError(f"five-stage action schema is not canonical: {schema_path}")
    try:
        schema = read_json(schema_path)
    except (OSError, json.JSONDecodeError) as exc:
        raise ToolError(f"five-stage action schema cannot be read: {schema_path}") from exc
    if schema.get("schema_version") != FIVE_STAGE_CONFIG_SCHEMA_VERSION:
        raise ToolError(
            "five-stage action schema version mismatch: "
            f"expected {FIVE_STAGE_CONFIG_SCHEMA_VERSION}, got {schema.get('schema_version')!r}"
        )
    if schema.get("stage_schema_id") != schema_id:
        raise ToolError("five-stage action schema id does not match configured stage_schema_id")
    stages = schema.get("stages")
    stage_ids = [item.get("id") for item in stages] if isinstance(stages, list) else []
    if stage_ids != list(FIVE_STAGE_NAMES):
        raise ToolError("five-stage action schema stages do not match canonical order")
    return str(algorithm_id), str(schema_id)


def _seven_stage_contract(config: dict[str, Any]) -> tuple[str, str, str, bool]:
    workflow = config.get("workflow")
    settings = workflow.get("action_segmentation") if isinstance(workflow, dict) else None
    if not isinstance(settings, dict):
        raise ToolError("workflow.action_segmentation must configure seven-stage action segmentation")
    action_version = _configured_action_version(config)
    contracts = {
        V2_ACTION_VERSION: (
            V2_ALGORITHM_ID,
            "qwen_experiment_action_hierarchical_v2.py",
            V2_OUTPUT_SCHEMA_VERSION,
            False,
        ),
        V2_FRAME_AGENT_ACTION_VERSION: (
            V2_FRAME_AGENT_ALGORITHM_ID,
            "qwen_experiment_action_hierarchical_v2_frame_agent.py",
            V2_FRAME_AGENT_OUTPUT_SCHEMA_VERSION,
            True,
        ),
        SCREENSHOT_GUARD_ACTION_VERSION: (
            SCREENSHOT_GUARD_ALGORITHM_ID,
            "qwen_experiment_action_hierarchical_v2_screenshot_guard.py",
            SCREENSHOT_GUARD_OUTPUT_SCHEMA_VERSION,
            False,
        ),
        SCREENSHOT_GUARD_AGENT_ACTION_VERSION: (
            SCREENSHOT_GUARD_AGENT_ALGORITHM_ID,
            "qwen_experiment_action_hierarchical_v2_screenshot_guard_agent.py",
            SCREENSHOT_GUARD_AGENT_OUTPUT_SCHEMA_VERSION,
            True,
        ),
    }
    if action_version not in contracts:
        raise ToolError(f"unsupported seven-stage action version: {action_version}")
    expected_algorithm_id, expected_script_name, expected_output_schema, uses_frame_agent = contracts[action_version]
    algorithm_id = settings.get("algorithm_id")
    schema_id = settings.get("stage_schema_id")
    if algorithm_id != expected_algorithm_id:
        raise ToolError(
            f"seven-stage algorithm_id must be {expected_algorithm_id}, got {algorithm_id!r}"
        )
    if schema_id != SCREENSHOT_GUARD_SCHEMA_ID:
        raise ToolError(
            f"seven-stage stage_schema_id must be {SCREENSHOT_GUARD_SCHEMA_ID}, got {schema_id!r}"
        )
    if settings.get("stage_count") != len(SCREENSHOT_GUARD_STAGE_NAMES):
        raise ToolError(f"seven-stage stage_count must be {len(SCREENSHOT_GUARD_STAGE_NAMES)}")
    script = resolve_inside(str(settings.get("script") or ""), PROJECT_ROOT, must_exist=True)
    schema_path = resolve_inside(str(settings.get("schema") or ""), PROJECT_ROOT, must_exist=True)
    if script.name != expected_script_name:
        raise ToolError(f"seven-stage script is not the canonical executor: {script}")
    if schema_path.name != "resistance_7stage_no_battery_v2.json":
        raise ToolError(f"seven-stage schema is not canonical: {schema_path}")
    schema = read_json(schema_path)
    if schema.get("schema_version") != SCREENSHOT_GUARD_CONFIG_SCHEMA_VERSION:
        raise ToolError("seven-stage action schema version mismatch")
    if schema.get("stage_schema_id") != schema_id:
        raise ToolError("seven-stage action schema id mismatch")
    stages = schema.get("stages")
    stage_ids = [item.get("id") for item in stages] if isinstance(stages, list) else []
    if stage_ids != list(SCREENSHOT_GUARD_STAGE_NAMES):
        raise ToolError("seven-stage schema stages do not match canonical order")
    return str(algorithm_id), str(schema_id), expected_output_schema, uses_frame_agent


def _validate_segment_frame_agent_report(result_path: Path, run_dir: Path) -> None:
    report_path = result_path.parent / "segment_frame_agent" / "report.json"
    if not report_path.is_file():
        raise ToolError("screenshot guard Agent result is missing segment frame Agent report")
    report = read_json(report_path)
    if report.get("schema_version") != "segment_frame_sampling_agent.v1":
        raise ToolError("segment frame Agent report schema_version mismatch")
    if report.get("status") != "completed":
        raise ToolError("segment frame Agent report is not complete")
    if report.get("selection_basis") != "current_video_observed_situation_only":
        raise ToolError("segment frame Agent did not use current-video situation routing")
    for field in ("video_id_used_for_routing", "historical_artifacts_used", "fixed_video_roi_used"):
        if report.get(field) is not False:
            raise ToolError(f"segment frame Agent anti-overfitting field must be false: {field}")
    if not isinstance(report.get("observed_stages"), list):
        raise ToolError("segment frame Agent report must include observed_stages")
    selected_skills = report.get("selected_skills")
    if not isinstance(selected_skills, list):
        raise ToolError("segment frame Agent report must include selected_skills")
    planned = report.get("planned_requests")
    executed = report.get("executed_requests")
    if not isinstance(planned, list) or not isinstance(executed, list):
        raise ToolError("segment frame Agent report must include request records")
    if len(planned) > 2 or len(executed) > 4:
        raise ToolError("segment frame Agent request budget exceeded")
    supplemental_frames = report.get("supplemental_frame_count")
    if not isinstance(supplemental_frames, int) or not 0 <= supplemental_frames <= 64:
        raise ToolError("segment frame Agent frame budget exceeded")
    if planned and not selected_skills:
        raise ToolError("segment frame Agent planned requests without a selected skill")
    if not planned and selected_skills:
        raise ToolError("segment frame Agent selected a skill without an observed gap")
    for execution in executed:
        if not isinstance(execution, dict):
            raise ToolError("segment frame Agent execution record is invalid")
        request_id = execution.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            raise ToolError("segment frame Agent execution request_id is invalid")
        frames = execution.get("input_frames")
        if not isinstance(frames, list):
            raise ToolError("segment frame Agent execution is missing input_frames")
        for frame in frames:
            if not isinstance(frame, dict) or not isinstance(frame.get("path"), str):
                raise ToolError("segment frame Agent frame record is invalid")
            resolve_inside(frame["path"], run_dir, must_exist=True)


def _validate_seven_stage_action_summary(
    run_dir: Path,
    state: dict[str, Any],
    config: dict[str, Any],
    report: dict[str, Any],
    *,
    dry_run: bool,
) -> str:
    """Accept only current-run seven-stage evidence for the configured executor."""
    expected_algorithm_id, expected_schema_id, expected_output_schema, uses_frame_agent = _seven_stage_contract(config)
    outputs = report.get("outputs")
    if not isinstance(outputs, dict) or not isinstance(outputs.get("action_summary"), str):
        raise ToolError("screenshot guard report must include outputs.action_summary")
    summary_path = resolve_inside(outputs["action_summary"], run_dir, must_exist=not dry_run)
    if dry_run:
        return str(summary_path.resolve())
    summary = read_json(summary_path)
    if summary.get("schema_version") != expected_output_schema:
        raise ToolError("screenshot guard action summary schema_version mismatch")
    if summary.get("algorithm_id") != expected_algorithm_id:
        raise ToolError("screenshot guard action summary algorithm mismatch")
    if summary.get("stage_schema_id") != expected_schema_id:
        raise ToolError("screenshot guard action summary stage schema mismatch")
    if summary.get("status") != "completed":
        raise ToolError(f"screenshot guard action summary is not complete: {summary.get('status')!r}")
    summary_config = summary.get("config")
    if not isinstance(summary_config, dict):
        raise ToolError("screenshot guard summary must include current-run config")
    if summary_config.get("algorithm_id") != expected_algorithm_id:
        raise ToolError("screenshot guard summary config algorithm mismatch")
    if summary_config.get("stage_schema_id") != expected_schema_id:
        raise ToolError("screenshot guard summary config stage schema mismatch")
    if summary_config.get("prepare_only") is not False:
        raise ToolError("screenshot guard execute summary must not be prepare-only")
    segment_source = summary_config.get("segment_source")
    if not isinstance(segment_source, str) or not segment_source:
        raise ToolError("screenshot guard summary must include segment_source")
    segment_path = resolve_inside(segment_source, run_dir, must_exist=True)
    if segment_path.name != "summary.json" or segment_path.parent.name != "experiment_boundary":
        raise ToolError("screenshot guard segment_source must be current-run experiment boundary summary")
    records = summary.get("records")
    if not isinstance(records, list) or len(records) != 1:
        raise ToolError("screenshot guard summary must contain exactly one current-video record")
    record = records[0]
    if not isinstance(record, dict) or record.get("status") not in {"completed", "completed_with_review"}:
        raise ToolError("screenshot guard record is not complete")
    expected_source = str(state.get("source_video_id") or "")
    if expected_source and str(record.get("source_video_id") or "") != expected_source:
        raise ToolError("screenshot guard record source does not match current video")
    result_value = record.get("result_path")
    if not isinstance(result_value, str) or not result_value:
        raise ToolError("screenshot guard record must include result_path")
    result_path = resolve_inside(result_value, run_dir, must_exist=True)
    result = read_json(result_path)
    if result.get("schema_version") != expected_output_schema:
        raise ToolError("screenshot guard action result schema_version mismatch")
    if result.get("algorithm_id") != expected_algorithm_id or result.get("stage_schema_id") != expected_schema_id:
        raise ToolError("screenshot guard action result identity mismatch")
    if str(result.get("source_video_id") or "") != str(record.get("source_video_id") or ""):
        raise ToolError("screenshot guard result source does not match summary record")
    source_manifest = result.get("source_manifest")
    if not isinstance(source_manifest, str) or not source_manifest:
        raise ToolError("screenshot guard result must include source_manifest")
    source_manifest_path = resolve_inside(source_manifest, run_dir, must_exist=True)
    provenance = result.get("source_segment_provenance")
    if not isinstance(provenance, dict):
        raise ToolError("screenshot guard result must include source_segment_provenance")
    if str(provenance.get("source_video_id") or "") != str(record.get("source_video_id") or ""):
        raise ToolError("screenshot guard provenance source does not match current video")
    provenance_manifest = provenance.get("source_manifest")
    if not isinstance(provenance_manifest, str) or not provenance_manifest:
        raise ToolError("screenshot guard provenance must include source_manifest")
    if resolve_inside(provenance_manifest, run_dir, must_exist=True) != source_manifest_path:
        raise ToolError("screenshot guard provenance manifest does not match result")
    stage_runs = result.get("observed_stage_runs")
    if not isinstance(stage_runs, list):
        raise ToolError("screenshot guard result must contain observed_stage_runs")
    stage_ranks = {name: index for index, name in enumerate(SCREENSHOT_GUARD_STAGE_NAMES)}
    observed_stages: set[str] = set()
    last_rank = -1
    previous_end = -1.0
    for stage_run in stage_runs:
        if not isinstance(stage_run, dict) or stage_run.get("stage") not in stage_ranks:
            raise ToolError("screenshot guard result contains a non-canonical stage")
        stage = str(stage_run["stage"])
        rank = stage_ranks[stage]
        if rank < last_rank:
            raise ToolError("screenshot guard stages are out of canonical order")
        try:
            start = float(stage_run["start_seconds"])
            end = float(stage_run["end_seconds"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ToolError("screenshot guard stage interval is invalid") from exc
        if not math.isfinite(start) or not math.isfinite(end) or start < 0.0 or end <= start or start < previous_end:
            raise ToolError("screenshot guard stage intervals are not monotonic")
        last_rank = rank
        previous_end = end
        observed_stages.add(stage)
    expected_missing = [name for name in SCREENSHOT_GUARD_STAGE_NAMES if name not in observed_stages]
    if result.get("missing_stages") != expected_missing or record.get("missing_stages") != expected_missing:
        raise ToolError("screenshot guard missing_stages does not match observed stages")
    if record.get("observed_stage_run_count") != len(stage_runs):
        raise ToolError("screenshot guard observed_stage_run_count mismatch")
    if uses_frame_agent:
        _validate_segment_frame_agent_report(result_path, run_dir)
    return str(summary_path.resolve())


def _validate_five_stage_action_summary(
    run_dir: Path,
    state: dict[str, Any],
    config: dict[str, Any],
    report: dict[str, Any],
    *,
    dry_run: bool,
) -> str:
    """Validate only action artifacts produced inside this run."""
    action_version = _configured_action_version(config)
    if action_version != "five-stage":
        return ""
    expected_algorithm_id, expected_schema_id = _five_stage_contract(config)
    outputs = report.get("outputs")
    if not isinstance(outputs, dict) or not isinstance(outputs.get("action_summary"), str):
        raise ToolError("five-stage pipeline report must include outputs.action_summary")
    summary_path = resolve_inside(outputs["action_summary"], run_dir, must_exist=not dry_run)
    if dry_run:
        return str(summary_path.resolve())
    summary = read_json(summary_path)
    if summary.get("schema_version") != FIVE_STAGE_OUTPUT_SCHEMA_VERSION:
        raise ToolError(
            "five-stage action summary schema_version mismatch: "
            f"expected {FIVE_STAGE_OUTPUT_SCHEMA_VERSION}, got {summary.get('schema_version')!r}"
        )
    if summary.get("algorithm_id") != expected_algorithm_id:
        raise ToolError(
            f"five-stage action summary algorithm mismatch: expected {expected_algorithm_id}, "
            f"got {summary.get('algorithm_id')}"
        )
    if summary.get("stage_schema_id") != expected_schema_id:
        raise ToolError(
            f"five-stage action summary schema mismatch: expected {expected_schema_id}, "
            f"got {summary.get('stage_schema_id')}"
        )
    if summary.get("status") != "completed":
        raise ToolError(f"five-stage action summary is not complete: {summary.get('status')!r}")
    summary_config = summary.get("config")
    if not isinstance(summary_config, dict):
        raise ToolError("five-stage action summary must include current-run config")
    if summary_config.get("algorithm_id") != expected_algorithm_id:
        raise ToolError("five-stage action summary config algorithm mismatch")
    if summary_config.get("stage_schema_id") != expected_schema_id:
        raise ToolError("five-stage action summary config schema mismatch")
    if summary_config.get("prepare_only") is not False:
        raise ToolError("five-stage execute summary must not be prepare-only")
    segment_source = summary_config.get("segment_source")
    if not isinstance(segment_source, str) or not segment_source:
        raise ToolError("five-stage action summary config must include segment_source")
    segment_path = resolve_inside(segment_source, run_dir, must_exist=True)
    if segment_path.name != "summary.json" or segment_path.parent.name != "experiment_boundary":
        raise ToolError("five-stage segment_source must be the current-run experiment boundary summary")
    records = summary.get("records")
    if not isinstance(records, list) or len(records) != 1:
        raise ToolError("five-stage action summary must contain exactly one current-video record")
    expected_source = str(state.get("source_video_id") or "")
    for record in records:
        if not isinstance(record, dict):
            raise ToolError("five-stage action summary record must be an object")
        if record.get("status") not in {"completed", "completed_with_review"}:
            raise ToolError(f"five-stage action record is not complete: {record.get('status')!r}")
        if expected_source and str(record.get("source_video_id") or "") != expected_source:
            raise ToolError("five-stage action record source does not match current video")
        if not isinstance(record.get("missing_stages"), list):
            raise ToolError("five-stage action record must include missing_stages")
        if not isinstance(record.get("observed_stage_run_count"), int) or isinstance(
            record.get("observed_stage_run_count"), bool
        ):
            raise ToolError("five-stage action record must include observed_stage_run_count")
        result_value = record.get("result_path")
        if not isinstance(result_value, str) or not result_value:
            raise ToolError("five-stage action record must include result_path")
        result_path = resolve_inside(result_value, run_dir, must_exist=True)
        result = read_json(result_path)
        if result.get("schema_version") != FIVE_STAGE_OUTPUT_SCHEMA_VERSION:
            raise ToolError("five-stage action result schema_version mismatch")
        if result.get("algorithm_id") != expected_algorithm_id:
            raise ToolError("five-stage action result algorithm mismatch")
        if result.get("stage_schema_id") != expected_schema_id:
            raise ToolError("five-stage action result schema mismatch")
        if str(result.get("source_video_id") or "") != str(record.get("source_video_id") or ""):
            raise ToolError("five-stage action result source does not match its summary record")
        source_manifest = result.get("source_manifest")
        if not isinstance(source_manifest, str) or not source_manifest:
            raise ToolError("five-stage action result must include source_manifest")
        source_manifest_path = resolve_inside(source_manifest, run_dir, must_exist=True)
        if source_manifest_path == run_dir:
            raise ToolError("five-stage source_manifest must be a current-run file")
        provenance = result.get("source_segment_provenance")
        if not isinstance(provenance, dict):
            raise ToolError("five-stage action result must include source_segment_provenance")
        if str(provenance.get("source_video_id") or "") != str(record.get("source_video_id") or ""):
            raise ToolError("five-stage source provenance video does not match current video")
        provenance_manifest = provenance.get("source_manifest")
        if not isinstance(provenance_manifest, str) or not provenance_manifest:
            raise ToolError("five-stage source provenance must include source_manifest")
        provenance_manifest_path = resolve_inside(provenance_manifest, run_dir, must_exist=True)
        if provenance_manifest_path != source_manifest_path:
            raise ToolError("five-stage source provenance manifest does not match result")
        stage_runs = result.get("observed_stage_runs")
        if not isinstance(stage_runs, list):
            raise ToolError("five-stage action result must contain observed_stage_runs")
        observed_stages: set[str] = set()
        last_rank = -1
        previous_end = 0.0
        stage_ranks = {name: index for index, name in enumerate(FIVE_STAGE_NAMES)}
        for stage_run in stage_runs:
            if not isinstance(stage_run, dict) or stage_run.get("stage") not in FIVE_STAGE_NAMES:
                raise ToolError("five-stage action result contains a non-canonical stage")
            stage_name = str(stage_run["stage"])
            rank = stage_ranks[stage_name]
            if rank < last_rank:
                raise ToolError("five-stage action result stages are out of temporal order")
            last_rank = rank
            observed_stages.add(stage_name)
            try:
                start = float(stage_run["start_seconds"])
                end = float(stage_run["end_seconds"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ToolError("five-stage action result stage interval is invalid") from exc
            if not math.isfinite(start) or not math.isfinite(end) or start < 0.0 or end <= start:
                raise ToolError("five-stage action result stage interval is invalid")
            if stage_runs and start < previous_end:
                raise ToolError("five-stage action result stage intervals are out of temporal order")
            previous_end = end
            if stage_name in {"recording_1", "recording_2"}:
                required = {
                    "stage_semantics",
                    "cycle_index",
                    "merged_measurement_recording",
                    "measurement_subintervals",
                    "writing_subintervals",
                    "contains_measurement_evidence",
                    "contains_writing_evidence",
                }
                missing = sorted(key for key in required if key not in stage_run)
                if missing:
                    raise ToolError(
                        "five-stage merged recording run missing fields: " + ", ".join(missing)
                    )
                if stage_run.get("merged_measurement_recording") is not True:
                    raise ToolError("five-stage recording run must be marked merged_measurement_recording")
                if stage_run.get("stage_semantics") != "measurement_and_recording_cycle":
                    raise ToolError("five-stage recording run has invalid stage semantics")
                expected_cycle = 1 if stage_name == "recording_1" else 2
                if stage_run.get("cycle_index") != expected_cycle:
                    raise ToolError("five-stage recording run has invalid cycle index")
                if not isinstance(stage_run.get("measurement_subintervals"), list) or not isinstance(
                    stage_run.get("writing_subintervals"), list
                ):
                    raise ToolError("five-stage subinterval fields must be lists")
                for field, action_type in (
                    ("measurement_subintervals", "measurement_action"),
                    ("writing_subintervals", "writing_action"),
                ):
                    for subinterval in stage_run[field]:
                        if not isinstance(subinterval, dict):
                            raise ToolError("five-stage subinterval must be an object")
                        if subinterval.get("action_type") != action_type:
                            raise ToolError("five-stage subinterval action type is inconsistent")
                        try:
                            sub_start = float(subinterval["start_seconds"])
                            sub_end = float(subinterval["end_seconds"])
                        except (KeyError, TypeError, ValueError) as exc:
                            raise ToolError("five-stage subinterval interval is invalid") from exc
                        if (
                            not math.isfinite(sub_start)
                            or not math.isfinite(sub_end)
                            or sub_start < start
                            or sub_end > end
                            or sub_end <= sub_start
                        ):
                            raise ToolError("five-stage subinterval interval is outside its stage")
                if stage_run["contains_measurement_evidence"] is not bool(
                    stage_run["measurement_subintervals"]
                ):
                    raise ToolError("five-stage measurement evidence flag is inconsistent")
                if stage_run["contains_writing_evidence"] is not bool(stage_run["writing_subintervals"]):
                    raise ToolError("five-stage writing evidence flag is inconsistent")
            elif stage_run.get("merged_measurement_recording") not in {False, None}:
                raise ToolError("non-recording five-stage run cannot be marked merged")
        expected_missing = [name for name in FIVE_STAGE_NAMES if name not in observed_stages]
        actual_missing = result.get("missing_stages")
        if actual_missing != expected_missing:
            raise ToolError("five-stage missing_stages does not match observed stages")
        if record.get("missing_stages") != expected_missing:
            raise ToolError("five-stage record missing_stages does not match result")
        observed_count = record.get("observed_stage_run_count")
        if observed_count != len(stage_runs):
            raise ToolError("five-stage record observed_stage_run_count does not match result")
    return str(summary_path.resolve())


def _verify_source_video(state: dict[str, Any]) -> Path:
    source = resolve_inside(state["video"]["path"], PROJECT_ROOT)
    expected_size = state["video"].get("bytes")
    expected_hash = state["video"].get("sha256")
    if source.stat().st_size != expected_size or sha256(source) != expected_hash:
        raise ToolError(f"source video changed after run creation: {source}")
    return source


def run_action_segmentation(run_id: str, execute: bool = False) -> dict[str, Any]:
    """Reject the removed legacy entrypoint instead of reading historical inputs."""
    del run_id, execute
    raise ToolError("run_action_segmentation is retired; use run_full_pipeline")


def refine_rubric_boundaries(run_id: str, execute: bool = False) -> dict[str, Any]:
    """Plan or run Rubric-guided refinement after Temporal Guard."""
    run_dir, state = _state(run_id)
    config = _state_config(state)
    if state["mode"] == "replay":
        raise ToolError("boundary refinement is only valid in prepare or execute mode")
    if bool(execute) != (state["mode"] == "execute"):
        raise ToolError("execute must match the run mode")
    if not state.get("action_summary"):
        raise ToolError("run_full_pipeline must complete first")
    action_summary = resolve_inside(state["action_summary"], run_dir, must_exist=False)
    release_root = _release_root(config)
    settings = config["workflow"]["boundary_refinement"]
    output_root = run_dir / "boundary_refinement"
    command = [
        sys.executable,
        str(resolve_inside(settings["script"], PROJECT_ROOT)),
        "--action-summary",
        str(action_summary),
        "--output-root",
        str(output_root),
        "--run-id",
        "rubric_boundaries",
        "--max-model-edge",
        str(settings.get("max_model_edge", 640)),
    ]
    if not execute:
        command.append("--prepare-only")
        plan_path = output_root / "boundary_plan.json"
        plan = {
            "schema_version": "resistance_agent_boundary_plan.v1",
            "status": "planned",
            "algorithm": settings.get("algorithm", "Rubric boundary refinement"),
            "source_action_summary": str(action_summary),
            "source_action_summary_exists": action_summary.is_file(),
            "command": command,
            "cwd": str(release_root),
            "qwen_requested": False,
        }
        write_json(plan_path, plan)
        state["boundary_plan"] = str(plan_path.resolve())
        state["status"] = "boundaries_planned"
        state["tool_calls"].append(
            {"tool": "refine_rubric_boundaries", "execute": False, "plan": plan, "at": utc_now()}
        )
        _save_state(run_dir, state)
        return {
            "status": state["status"],
            "plan_path": str(plan_path.resolve()),
            "source_action_summary": str(action_summary),
            "source_action_summary_exists": action_summary.is_file(),
            "command": command,
        }
    if not action_summary.is_file():
        raise ToolError(f"completed action summary is missing: {action_summary}")
    action = read_json(action_summary)
    if action.get("status") != "completed":
        raise ToolError("boundary refinement needs completed Temporal Guard stage results")
    summary_path = output_root / "rubric_boundaries" / "summary.json"
    records = action.get("records")
    current_results: list[tuple[dict[str, Any], Path, dict[str, Any]]] = []
    if isinstance(records, list):
        for record in records:
            if not isinstance(record, dict):
                continue
            result_value = record.get("result_path")
            if not isinstance(result_value, str) or not result_value:
                continue
            result_path = resolve_inside(result_value, run_dir, must_exist=False)
            if result_path.is_file():
                current_results.append((record, result_path, read_json(result_path)))
    if current_results and all(
        isinstance(result.get("observed_stage_runs"), list)
        and not result["observed_stage_runs"]
        for _, _, result in current_results
    ):
        fallback_records = []
        for record, result_path, result in current_results:
            fallback_records.append(
                {
                    "source_video_id": str(
                        result.get("source_video_id") or record.get("source_video_id") or ""
                    ),
                    "status": "completed_broad_search_fallback",
                    "boundary_count": 0,
                    "qwen_call_count": 0,
                    "source_result_path": str(result_path.resolve()),
                    "source_observed_stage_runs": [],
                    "source_observed_stage_intervals": [],
                    "effective_experiment_interval_seconds": result.get(
                        "effective_experiment_interval_seconds"
                    )
                    or result.get("locked_experiment_interval_seconds"),
                    "broad_search_required": True,
                    "fallback_reason": "current_run_stage_detector_returned_no_observed_stages",
                }
            )
        fallback_summary = {
            "schema_version": "resistance_agent_boundary_broad_search_fallback.v1",
            "algorithm_id": "current_run_empty_stage_broad_search",
            "status": "completed",
            "source_action_summary": str(action_summary.resolve()),
            "source_stage_runs_unchanged": True,
            "golden_fixture_used": False,
            "historical_artifacts_used": False,
            "boundary_count": 0,
            "qwen_call_count": 0,
            "records": fallback_records,
        }
        write_json(summary_path, fallback_summary)
        state["boundary_summary"] = str(summary_path.resolve())
        state["status"] = "boundaries_completed"
        state["tool_calls"].append(
            {
                "tool": "refine_rubric_boundaries",
                "execute": True,
                "broad_search_fallback": True,
                "at": utc_now(),
            }
        )


        _save_state(run_dir, state)
        return {
            "status": state["status"],
            "summary_path": str(summary_path.resolve()),
            "source_stage_runs_unchanged": True,
            "boundary_count": 0,
            "broad_search_fallback": True,
            "fallback_reason": "current_run_stage_detector_returned_no_observed_stages",
        }
    if summary_path.is_file():
        summary = read_json(summary_path)
        if (
            summary.get("status") == "completed"
            and summary.get("source_stage_runs_unchanged") is True
            and Path(str(summary.get("source_action_summary") or "")).resolve() == action_summary.resolve()
        ):
            state["boundary_summary"] = str(summary_path.resolve())
            state["status"] = "boundaries_completed"
            state["tool_calls"].append(
                {"tool": "refine_rubric_boundaries", "execute": True, "idempotent_replay": True, "at": utc_now()}
            )
            _save_state(run_dir, state)
            return {
                "status": state["status"],
                "summary_path": str(summary_path.resolve()),
                "source_stage_runs_unchanged": True,
                "boundary_count": summary.get("boundary_count", 0),
                "idempotent_replay": True,
            }
    process = _command(
        command,
        int(config["limits"]["tool_timeout_seconds"]),
        release_root,
        _qwen_pipeline_environment(config, require_token=execute),
    )
    if not summary_path.is_file():
        raise ToolError(f"boundary refinement summary not created: {summary_path}")
    summary = read_json(summary_path)
    if summary.get("source_stage_runs_unchanged") is not True:
        raise ToolError("boundary refinement did not preserve source stage runs")
    state["boundary_summary"] = str(summary_path.resolve())
    state["status"] = "boundaries_completed"
    state["tool_calls"].append(
        {"tool": "refine_rubric_boundaries", "execute": execute, "process": process, "at": utc_now()}
    )
    _save_state(run_dir, state)
    return {
        "status": state["status"],
        "summary_path": str(summary_path.resolve()),
        "source_stage_runs_unchanged": True,
        "boundary_count": summary.get("boundary_count", 0),
        "process": process,
    }


def _meter_adaptive_recommendation(
    rubrics: dict[str, Any],
    evidence_report: str | None,
    duration_seconds: float,
) -> dict[str, Any]:
    reasons: list[str] = []
    for rubric_id in ("5", "6"):
        item = rubrics.get(rubric_id)
        if not isinstance(item, dict):
            continue
        confidence = item.get("confidence")
        if isinstance(confidence, (int, float)) and not isinstance(confidence, bool):
            if float(confidence) < 0.72:
                reasons.append(f"r{rubric_id}_confidence_below_0.72")
        reason = str(item.get("reason") or "").lower()
        if any(token in reason for token in ("uncertain", "low_visibility", "not_visible", "missing")):
            reasons.append(f"r{rubric_id}_visibility_or_conflict")

    report: dict[str, Any] = {}
    if isinstance(evidence_report, str) and evidence_report:
        path = Path(evidence_report)
        if path.is_file():
            value = read_json(path)
            report = value if isinstance(value, dict) else {}
    qwen = report.get("qwen_observation") if isinstance(report.get("qwen_observation"), dict) else {}
    qwen_confidence = qwen.get("overall_confidence")
    if isinstance(qwen_confidence, (int, float)) and not isinstance(qwen_confidence, bool):
        if float(qwen_confidence) < 0.72:
            reasons.append("qwen_overall_confidence_below_0.72")
    selected = [
        item
        for item in report.get("selected_frames", [])
        if isinstance(item, dict) and isinstance(item.get("timestamp_seconds"), (int, float))
    ]
    if selected and any(not (item.get("model_candidates") or item.get("candidates")) for item in selected):
        reasons.append("selected_frame_has_no_meter_candidate")

    reasons = sorted(set(reasons))
    request_template = None
    if reasons and selected:
        anchor = min(
            selected,
            key=lambda item: (
                len(item.get("model_candidates") or item.get("candidates") or []),
                float(item.get("sharpness") or 0.0),
            ),
        )
        timestamp = float(anchor["timestamp_seconds"])
        start = max(0.0, timestamp - 1.0)
        end = min(float(duration_seconds), timestamp + 1.0)
        if end - start >= 0.1:
            request_template = {
                "rubric_ids": [5, 6],
                "reason": "meter_pointer_occluded"
                if "selected_frame_has_no_meter_candidate" in reasons
                else "low_confidence",
                "time_ranges": [
                    {
                        "start_seconds": round(start, 3),
                        "end_seconds": round(end, 3),
                    }
                ],
                "interval_seconds": 0.2,
                "max_frames": 16,
                "roi_mode": "dynamic_meter_candidates",
                "view": "meter_pair",
            }
    return {
        "adaptive_evidence_recommended": bool(request_template),
        "adaptive_evidence_reasons": reasons,
        "adaptive_request_template": request_template,
    }


def request_additional_evidence(
    run_id: str,
    rubric_ids: list[int],
    reason: str,
    time_ranges: list[dict[str, Any]],
    interval_seconds: float = 0.2,
    max_frames: int = 24,
    roi_mode: str = "dynamic_meter_candidates",
    view: str = "meter_pair",
    evidence_profile: str | None = None,
    cycle: int | None = None,
    target_fields: list[str] | None = None,
    target_roles: list[str] | None = None,
    anchor_frame_ids: list[str] | None = None,
    search_mode: str | None = None,
) -> dict[str, Any]:
    """Acquire bounded adjacent frames from the current execute run only."""
    run_dir, state = _state(run_id)
    if state.get("status") == "completed" or state.get("final_result"):
        raise ToolError("finalized predictions are immutable; create a new run for more evidence")
    _verify_source_video(state)
    try:
        from .adaptive_evidence import AdaptiveEvidenceError, request_additional_evidence as acquire
    except ImportError:
        from adaptive_evidence import AdaptiveEvidenceError, request_additional_evidence as acquire  # type: ignore
    try:
        result = acquire(
            run_dir=run_dir,
            state=state,
            rubric_ids=rubric_ids,
            reason=reason,
            time_ranges=time_ranges,
            interval_seconds=interval_seconds,
            max_frames=max_frames,
            roi_mode=roi_mode,
            view=view,
            evidence_profile=evidence_profile,
            cycle=cycle,
            target_fields=target_fields,
            target_roles=target_roles,
            anchor_frame_ids=anchor_frame_ids,
            search_mode=search_mode,
        )
    except AdaptiveEvidenceError as exc:
        raise ToolError(str(exc)) from exc
    archived: list[dict[str, Any]] = []
    invalidated: list[int] = []
    profile = str(result.get("evidence_profile") or "meter_pair")
    record_request = profile in {"record_meter", "record_paper"}
    acquired = (
        int(result.get("frame_count") or 0) > 0
        if record_request
        else int(result.get("selected_frame_count") or 0) > 0
    )
    if acquired:
        archive_dir = Path(result["result_path"]).parent / "prior_results"
        archive_dir.mkdir(parents=True, exist_ok=True)
        rubric_results = state.setdefault("rubric_results", {})
        grouped_rubrics = (7, 9) if record_request else (5, 6)
        report_key = "7_9" if record_request else "5_6"
        report_kind = "record_evidence_report" if record_request else "meter_evidence_report"
        for rubric_id in grouped_rubrics:
            value = rubric_results.pop(str(rubric_id), None)
            if not isinstance(value, str) or not value:
                continue
            source = resolve_inside(value, run_dir, must_exist=False)
            if source.is_file():
                destination = archive_dir / source.name
                shutil.copy2(source, destination)
                archived.append(
                    {
                        "kind": f"rubric_{rubric_id}",
                        "path": str(destination.resolve()),
                        "sha256": sha256(destination),
                    }
                )
            invalidated.append(rubric_id)
        evidence_value = state.setdefault("rubric_evidence_reports", {}).pop(report_key, None)
        if isinstance(evidence_value, str) and evidence_value:
            source = resolve_inside(evidence_value, run_dir, must_exist=False)
            if source.is_file():
                destination = archive_dir / source.name
                shutil.copy2(source, destination)
                archived.append(
                    {
                        "kind": report_kind,
                        "path": str(destination.resolve()),
                        "sha256": sha256(destination),
                    }
                )
        if invalidated:
            state["status"] = "boundaries_completed"
    result["invalidated_rubric_ids"] = invalidated
    result["archived_prior_artifacts"] = archived
    write_json(Path(result["result_path"]), result)
    state.setdefault("adaptive_evidence_requests", []).append(result["result_path"])
    state["tool_calls"].append(
        {
            "tool": "request_additional_evidence",
            "request_number": result["request_number"],
            "rubric_ids": result["rubric_ids"],
            "reason": result["reason"],
            "frame_count": result["frame_count"],
            "selected_frame_count": result["selected_frame_count"],
            "evidence_profile": profile,
            "cycle": result.get("cycle"),
            "at": utc_now(),
        }
    )
    _save_state(run_dir, state)
    return result


def _qwen_pipeline_environment(config: dict[str, Any], *, require_token: bool) -> dict[str, str]:
    models = config.get("models") if isinstance(config.get("models"), dict) else {}
    settings = models.get("qwen") if isinstance(models.get("qwen"), dict) else {}
    base_url = (os.getenv("QWEN_API_BASE_URL") or str(settings.get("base_url") or "")).strip()
    model = (os.getenv("QWEN_MODEL") or str(settings.get("model") or "qwen")).strip() or "qwen"
    token_env = str(settings.get("api_key_env") or "QWEN_API_TOKEN")
    token = os.getenv(token_env, "").strip()
    authentication = str(settings.get("authentication") or "")
    if not token and "EMPTY" in authentication.upper():
        token = "EMPTY"
    if not base_url:
        raise ToolError("Qwen base URL is missing for the full pipeline")
    if require_token and not token:
        raise ToolError(f"{token_env} is required for the full pipeline")
    environment = {
        "QWEN_API_BASE_URL": base_url,
        "QWEN_MODEL": model,
    }
    if token:
        environment["QWEN_API_TOKEN"] = token
    return environment


def _configured_action_version(config: dict[str, Any]) -> str:
    workflow = config.get("workflow")
    full_pipeline = workflow.get("full_pipeline") if isinstance(workflow, dict) else None
    value = full_pipeline.get("action_version") if isinstance(full_pipeline, dict) else None
    if not isinstance(value, str) or not value.strip():
        raise ToolError("workflow.full_pipeline.action_version must be explicitly configured")
    return value.strip()


def _validate_full_pipeline_report(
    run_dir: Path,
    state: dict[str, Any],
    config: dict[str, Any],
    report: dict[str, Any],
    *,
    dry_run: bool,
) -> str:
    action_version = _configured_action_version(config)
    expected_status = "planned" if dry_run else "completed"
    if report.get("status") != expected_status:
        raise ToolError(
            f"full pipeline report status mismatch: expected {expected_status}, "
            f"got {report.get('status')}"
        )
    if bool(report.get("dry_run")) != bool(dry_run):
        raise ToolError("full pipeline report dry_run flag does not match the current mode")
    if report.get("action_version") != action_version:
        raise ToolError(f"full pipeline did not select configured action version {action_version}")
    outputs = report.get("outputs")
    if not isinstance(outputs, dict) or not isinstance(outputs.get("action_summary"), str):
        raise ToolError("full pipeline report must include outputs.action_summary")
    summary_path = resolve_inside(outputs["action_summary"], run_dir, must_exist=not dry_run)
    if action_version == "five-stage":
        return _validate_five_stage_action_summary(
            run_dir,
            state,
            config,
            report,
            dry_run=dry_run,
        )
    if action_version in {
        V2_ACTION_VERSION,
        V2_FRAME_AGENT_ACTION_VERSION,
        SCREENSHOT_GUARD_ACTION_VERSION,
        SCREENSHOT_GUARD_AGENT_ACTION_VERSION,
    }:
        return _validate_seven_stage_action_summary(
            run_dir,
            state,
            config,
            report,
            dry_run=dry_run,
        )
    return str(summary_path.resolve())


def run_full_pipeline(run_id: str, dry_run: bool = False) -> dict[str, Any]:
    """Run or plan the release pipeline against a verified private video copy."""
    run_dir, state = _state(run_id)
    if state["mode"] == "replay":
        raise ToolError("full pipeline is only valid in prepare or execute mode")
    if bool(dry_run) != (state["mode"] == "prepare"):
        raise ToolError("dry_run must match the run mode")
    config = _state_config(state)
    release_root = _release_root(config)
    settings = config["workflow"]["full_pipeline"]
    action_version = _configured_action_version(config)
    input_dir = run_dir / "input_video"
    input_dir.mkdir(exist_ok=True)
    source = _verify_source_video(state)
    linked = input_dir / source.name
    if not linked.exists():
        shutil.copy2(source, linked)
    if linked.is_symlink() or not linked.is_file():
        raise ToolError(f"pipeline input is not an isolated file copy: {linked}")
    if os.path.samefile(source, linked):
        raise ToolError(f"pipeline input must not share source file identity: {linked}")
    if linked.stat().st_size != source.stat().st_size or sha256(linked) != state["video"]["sha256"]:
        raise ToolError(f"pipeline input copy failed verification: {linked}")
    output_root = run_dir / "pipeline"
    report_path = output_root / "pipeline" / "run_report.json"
    if report_path.is_file():
        report = read_json(report_path)
        expected_status = "planned" if dry_run else "completed"
        if report.get("status") == expected_status and bool(report.get("dry_run")) == bool(dry_run):
            action_summary = _validate_full_pipeline_report(
                run_dir,
                state,
                config,
                report,
                dry_run=dry_run,
            )
            state["action_summary"] = action_summary
            state["status"] = "pipeline_planned" if dry_run else "pipeline_completed"
            state["tool_calls"].append(
                {"tool": "run_full_pipeline", "dry_run": dry_run, "idempotent_replay": True, "at": utc_now()}
            )
            _save_state(run_dir, state)
            return {
                "status": state["status"],
                "run_report": str(report_path.resolve()),
                "outputs": report.get("outputs", {}),
                "rubric_specific_artifacts_required": report.get(
                    "rubric_specific_artifacts_required", []
                ),
                "idempotent_replay": True,
            }
    command = [
        sys.executable,
        str(resolve_inside(settings["script"], PROJECT_ROOT)),
        "--video-dir",
        str(input_dir),
        "--output-root",
        str(output_root),
        "--run-id",
        "pipeline",
        "--action-version",
        action_version,
    ]
    if dry_run:
        command.append("--dry-run")
    process = _command(
        command,
        int(config["limits"]["pipeline_timeout_seconds"]),
        release_root,
        _qwen_pipeline_environment(config, require_token=not dry_run),
    )
    _verify_source_video(state)
    report = read_json(report_path)
    action_summary = _validate_full_pipeline_report(
        run_dir,
        state,
        config,
        report,
        dry_run=dry_run,
    )
    state["action_summary"] = action_summary
    state["status"] = "pipeline_planned" if dry_run else "pipeline_completed"
    state["tool_calls"].append(
        {"tool": "run_full_pipeline", "dry_run": dry_run, "process": process, "at": utc_now()}
    )
    _save_state(run_dir, state)
    return {
        "status": state["status"],
        "run_report": str(report_path.resolve()),
        "outputs": report.get("outputs", {}),
        "rubric_specific_artifacts_required": report.get("rubric_specific_artifacts_required", []),
        "source_video_unchanged": True,
        "process": process,
    }


def _load_replay_summary(config: dict[str, Any]) -> dict[str, Any]:
    path = resolve_inside(config["replay"]["all_rubrics_summary"], PROJECT_ROOT)
    summary = read_json(path)
    if summary.get("status") != "completed" or not isinstance(summary.get("rows"), list):
        raise ToolError(f"invalid replay summary: {path}")
    return summary


def load_rubric_result(run_id: str, rubric_id: int) -> dict[str, Any]:
    """Load one frozen result only in explicit replay mode."""
    run_dir, state = _state(run_id)
    if state["mode"] != "replay":
        raise ToolError("frozen rubric results are only available in replay mode")
    if rubric_id not in range(10):
        raise ToolError("rubric_id must be an integer from 0 through 9")
    summary = _load_replay_summary(_state_config(state))
    row = next((item for item in summary["rows"] if str(item.get("video_id")) == state["video_id"]), None)
    if row is None:
        raise ToolError(f"no frozen artifact for video {state['video_id']}")
    evaluation = row.get("evaluations", {}).get(str(rubric_id))
    if not isinstance(evaluation, dict) or evaluation.get("decision") not in DECISIONS:
        raise ToolError(f"invalid rubric {rubric_id} result for video {state['video_id']}")
    result = {
        "schema_version": "resistance_agent_rubric_result.v2",
        "video_id": state["video_id"],
        "source_video_id": state["source_video_id"],
        "rubric_id": rubric_id,
        "decision": evaluation["decision"],
        "predicted_score": evaluation["predicted_score"],
        "confidence": evaluation.get("confidence"),
        "reason": evaluation.get("reason"),
        "source_artifact": evaluation.get("source_artifact"),
        "diagnostics": evaluation.get("diagnostics", {}),
        "execution_mode": "replay_frozen_artifact",
    }
    result_path = run_dir / "rubrics" / f"rubric_{rubric_id}.json"
    write_json(result_path, result)
    state["rubric_results"][str(rubric_id)] = str(result_path.resolve())
    state["status"] = "rubrics_in_progress"
    state["tool_calls"].append({"tool": "load_rubric_result", "rubric_id": rubric_id, "at": utc_now()})
    _save_state(run_dir, state)
    return {
        "video_id": result["video_id"],
        "rubric_id": result["rubric_id"],
        "decision": result["decision"],
        "predicted_score": result["predicted_score"],
        "confidence": result.get("confidence"),
        "reason": result.get("reason"),
        "result_path": str(result_path.resolve()),
    }


def load_rubric_bundle(run_id: str, rubric_ids: list[int]) -> dict[str, Any]:
    """Load multiple frozen replay results through one bounded MCP call."""
    if not isinstance(rubric_ids, list) or not rubric_ids:
        raise ToolError("rubric_ids must be a non-empty list")
    if any(isinstance(item, bool) or not isinstance(item, int) or item not in range(10) for item in rubric_ids):
        raise ToolError("rubric_ids must contain integers from 0 through 9")
    ordered = sorted(set(rubric_ids))
    results = [load_rubric_result(run_id, rubric_id) for rubric_id in ordered]
    run_dir, state = _state(run_id)
    state["tool_calls"].append(
        {"tool": "load_rubric_bundle", "rubric_ids": ordered, "at": utc_now()}
    )
    _save_state(run_dir, state)
    return {
        "video_id": state["video_id"],
        "rubric_ids": ordered,
        "loaded_count": len(results),
        "decisions": {
            str(item["rubric_id"]): item["decision"] for item in results
        },
        "result_paths": [item["result_path"] for item in results],
    }


def run_meter_rubrics(run_id: str, use_fallback_temporal_guard: bool = False) -> dict[str, Any]:
    """Acquire real video evidence for R5/R6 and write two binary artifacts."""
    run_dir, state = _state(run_id)
    if state.get("mode") != "execute":
        raise ToolError("meter rubrics are only valid in execute mode")
    _reject_historical_live_fallback(use_fallback_temporal_guard)
    skill_plan = _live_skill_plan(run_dir, state)
    existing_paths = state.get("rubric_results", {})
    if all(str(rubric_id) in existing_paths for rubric_id in (5, 6)):
        existing: dict[str, Any] = {}
        for rubric_id in (5, 6):
            result_path = resolve_inside(existing_paths[str(rubric_id)], AGENT_ROOT)
            item = read_json(result_path)
            expected_score = 1 if item.get("decision") == "pass" else 0
            if (
                item.get("schema_version") != "resistance_agent_rubric_result.v2"
                or item.get("rubric_id") != rubric_id
                or item.get("video_id") != state.get("video_id")
                or item.get("source_video_id") != state.get("source_video_id")
                or item.get("decision") not in DECISIONS
                or item.get("predicted_score") != expected_score
                or item.get("execution_mode") != "execute_visual_evidence"
                or item.get("routing_policy") != LIVE_ROUTING_POLICY
            ):
                raise ToolError(f"existing rubric {rubric_id} artifact is invalid")
            existing[str(rubric_id)] = {
                "decision": item["decision"],
                "predicted_score": item["predicted_score"],
                "confidence": item.get("confidence"),
                "reason": item.get("reason"),
                "result_path": str(result_path),
            }
        return {
            "status": state.get("status", "meter_rubrics_completed"),
            "video_id": state["video_id"],
            "rubrics": existing,
            "evidence_report": state.get("rubric_summary"),
            **_meter_adaptive_recommendation(
                existing,
                state.get("rubric_summary"),
                float(state.get("video", {}).get("duration_seconds") or 0.0),
            ),
            **_skill_metadata(skill_plan, (5, 6)),
            "source_video_unchanged": True,
            "idempotent_replay": True,
        }
    if state.get("status") not in {
        "boundaries_completed",
        "switch_rubric_completed",
        "series_rubric_completed",
        "meter_rubrics_completed",
    } and not use_fallback_temporal_guard:
        raise ToolError("refine_rubric_boundaries must complete before meter rubrics")
    config = _state_config(state)
    source = _verify_source_video(state)
    input_dir = run_dir / "input_video"
    input_dir.mkdir(exist_ok=True)
    private_copy = input_dir / source.name
    if not private_copy.exists():
        shutil.copy2(source, private_copy)
    if private_copy.is_symlink() or not private_copy.is_file() or os.path.samefile(source, private_copy):
        raise ToolError(f"meter input is not an isolated file copy: {private_copy}")
    if private_copy.stat().st_size != source.stat().st_size or sha256(private_copy) != state["video"]["sha256"]:
        raise ToolError(f"meter input copy failed verification: {private_copy}")

    action_path: Path | None = None
    if isinstance(state.get("action_summary"), str) and state["action_summary"]:
        candidate = resolve_inside(state["action_summary"], AGENT_ROOT, must_exist=False)
        if candidate.is_file():
            action_path = candidate
    boundary_path: Path | None = None
    if isinstance(state.get("boundary_summary"), str) and state["boundary_summary"]:
        candidate = resolve_inside(state["boundary_summary"], AGENT_ROOT, must_exist=False)
        if candidate.is_file():
            boundary_path = candidate
    settings = config["workflow"].get("meter_rubrics", {})
    try:
        try:
            from . import meter_rubrics as meter_module
        except ImportError:
            import meter_rubrics as meter_module  # type: ignore

        meter_arguments = {
            "video_path": private_copy,
            "source_video_id": state["source_video_id"],
            "video_id": state["video_id"],
            "run_dir": run_dir,
            "model_config": config["models"]["qwen"],
            "action_summary_path": action_path,
            "fallback_action_summary_path": None,
            "allow_historical_fallback": False,
            "skill_plan": skill_plan,
        }
        if getattr(meter_module.run_meter_rubrics, "supports_boundary_summary", False):
            meter_arguments["boundary_summary_path"] = boundary_path
        closed_stable = settings.get("closed_stable_cv_v3", {})
        if (
            getattr(meter_module.run_meter_rubrics, "supports_closed_stable_cv_v3", False)
            and isinstance(closed_stable, dict)
            and closed_stable.get("enabled") is True
        ):
            stage_producer = closed_stable.get("stage_producer")
            if (
                getattr(meter_module.run_meter_rubrics, "supports_closed_stable_stage_producer", False)
                and isinstance(stage_producer, dict)
                and stage_producer.get("enabled") is True
            ):
                producer_config = dict(stage_producer)
                producer_root = Path(str(producer_config.get("producer_root") or ""))
                producer_root = (
                    producer_root.resolve()
                    if producer_root.is_absolute()
                    else (PROJECT_ROOT / producer_root).resolve()
                )
                producer_config["producer_root"] = str(producer_root)
                runtime_calibration = Path(str(producer_config.get("runtime_calibration") or ""))
                producer_config["runtime_calibration"] = str(
                    runtime_calibration.resolve()
                    if runtime_calibration.is_absolute()
                    else (producer_root / runtime_calibration).resolve()
                )
                meter_arguments["closed_stable_stage_producer_config"] = producer_config
        evidence = meter_module.run_meter_rubrics(**meter_arguments)
    except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise ToolError(f"meter evidence acquisition failed: {type(exc).__name__}: {exc}") from exc

    paths: dict[str, str] = {}
    compact: dict[str, Any] = {}
    for rubric_id, key in ((5, "rubric_5"), (6, "rubric_6")):
        item = evidence.get(key)
        if not isinstance(item, dict) or item.get("decision") not in DECISIONS:
            raise ToolError(f"meter evidence returned invalid rubric {rubric_id} result")
        predicted = 1 if item["decision"] == "pass" else 0
        if item.get("predicted_score") != predicted:
            raise ToolError(f"meter evidence rubric {rubric_id} score mismatch")
        result = {
            "schema_version": "resistance_agent_rubric_result.v2",
            "video_id": state["video_id"],
            "source_video_id": state["source_video_id"],
            "rubric_id": rubric_id,
            "decision": item["decision"],
            "predicted_score": predicted,
            "confidence": item.get("confidence"),
            "reason": item.get("reason"),
            "source_artifact": evidence.get("report_path"),
            "diagnostics": item.get("diagnostics", {}),
            "execution_mode": "execute_visual_evidence",
            **_skill_metadata(skill_plan, (5, 6)),
        }
        result_path = run_dir / "rubrics" / f"rubric_{rubric_id}.json"
        write_json(result_path, result)
        reopened = read_json(result_path)
        if reopened.get("decision") not in DECISIONS or reopened.get("predicted_score") != predicted:
            raise ToolError(f"rubric {rubric_id} artifact verification failed")
        paths[str(rubric_id)] = str(result_path.resolve())
        compact[str(rubric_id)] = {
            "decision": result["decision"],
            "predicted_score": result["predicted_score"],
            "confidence": result["confidence"],
            "reason": result["reason"],
            "result_path": paths[str(rubric_id)],
        }

    _verify_source_video(state)
    state["rubric_results"].update(paths)
    state["rubric_summary"] = evidence.get("report_path")
    state.setdefault("rubric_evidence_reports", {})["5_6"] = evidence.get("report_path")
    state["status"] = "meter_rubrics_completed"
    state["skill_plan"] = str((run_dir / "skills" / "live_skill_plan.json").resolve())
    adaptive = _meter_adaptive_recommendation(
        compact,
        evidence.get("report_path"),
        float(state.get("video", {}).get("duration_seconds") or 0.0),
    )
    state["tool_calls"].append(
        {
            "tool": "run_meter_rubrics",
            "rubric_ids": [5, 6],
            "evidence_report": evidence.get("report_path"),
            "adaptive_evidence_recommended": adaptive["adaptive_evidence_recommended"],
            "at": utc_now(),
        }
    )
    _save_state(run_dir, state)
    return {
        "status": state["status"],
        "video_id": state["video_id"],
        "rubrics": compact,
        "evidence_report": evidence.get("report_path"),
        **adaptive,
        **_skill_metadata(skill_plan, (5, 6)),
        "source_video_unchanged": True,
    }


def run_record_rubrics(run_id: str, use_fallback_temporal_guard: bool = False) -> dict[str, Any]:
    """Acquire cycle-bound real video evidence for R7/R9."""
    run_dir, state = _state(run_id)
    if state.get("mode") != "execute":
        raise ToolError("record rubrics are only valid in execute mode")
    _reject_historical_live_fallback(use_fallback_temporal_guard)
    skill_plan = _live_skill_plan(run_dir, state)
    existing_paths = state.get("rubric_results", {})
    if all(str(rubric_id) in existing_paths for rubric_id in (7, 9)):
        existing: dict[str, Any] = {}
        for rubric_id in (7, 9):
            result_path = resolve_inside(existing_paths[str(rubric_id)], AGENT_ROOT)
            item = read_json(result_path)
            expected_score = 1 if item.get("decision") == "pass" else 0
            if (
                item.get("schema_version") != "resistance_agent_rubric_result.v2"
                or item.get("rubric_id") != rubric_id
                or item.get("video_id") != state.get("video_id")
                or item.get("source_video_id") != state.get("source_video_id")
                or item.get("decision") not in DECISIONS
                or item.get("predicted_score") != expected_score
                or item.get("execution_mode") != "execute_visual_evidence"
                or item.get("routing_policy") != LIVE_ROUTING_POLICY
            ):
                raise ToolError(f"existing rubric {rubric_id} artifact is invalid")
            existing[str(rubric_id)] = {
                "decision": item["decision"],
                "predicted_score": item["predicted_score"],
                "confidence": item.get("confidence"),
                "reason": item.get("reason"),
                "result_path": str(result_path),
            }
        return {
            "status": state.get("status", "record_rubrics_completed"),
            "video_id": state["video_id"],
            "rubrics": existing,
            "evidence_report": state.get("rubric_evidence_reports", {}).get("7_9"),
            **_skill_metadata(skill_plan, (7, 9)),
            "source_video_unchanged": True,
            "idempotent_replay": True,
        }
    if state.get("status") not in {
        "boundaries_completed",
        "switch_rubric_completed",
        "series_rubric_completed",
        "meter_rubrics_completed",
        "record_rubrics_completed",
    } and not use_fallback_temporal_guard:
        raise ToolError("refine_rubric_boundaries must complete before record rubrics")
    config = _state_config(state)
    source = _verify_source_video(state)
    input_dir = run_dir / "input_video"
    input_dir.mkdir(exist_ok=True)
    private_copy = input_dir / source.name
    if not private_copy.exists():
        shutil.copy2(source, private_copy)
    if private_copy.is_symlink() or not private_copy.is_file() or os.path.samefile(source, private_copy):
        raise ToolError(f"record input is not an isolated file copy: {private_copy}")
    if private_copy.stat().st_size != source.stat().st_size or sha256(private_copy) != state["video"]["sha256"]:
        raise ToolError(f"record input copy failed verification: {private_copy}")

    action_path: Path | None = None
    if isinstance(state.get("action_summary"), str) and state["action_summary"]:
        candidate = resolve_inside(state["action_summary"], AGENT_ROOT, must_exist=False)
        if candidate.is_file():
            action_path = candidate
    boundary_path: Path | None = None
    if isinstance(state.get("boundary_summary"), str) and state["boundary_summary"]:
        candidate = resolve_inside(state["boundary_summary"], AGENT_ROOT, must_exist=False)
        if candidate.is_file():
            boundary_path = candidate
    try:
        try:
            from . import record_rubrics as record_module
        except ImportError:
            import record_rubrics as record_module  # type: ignore

        arguments = {
            "video_path": private_copy,
            "source_video_id": state["source_video_id"],
            "video_id": state["video_id"],
            "run_dir": run_dir,
            "model_config": config["models"]["qwen"],
            "action_summary_path": action_path,
            "fallback_action_summary_path": None,
            "allow_historical_fallback": False,
            "skill_plan": skill_plan,
        }
        if getattr(record_module.run_record_rubrics, "supports_boundary_summary", False):
            arguments["boundary_summary_path"] = boundary_path
        evidence = record_module.run_record_rubrics(**arguments)
    except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise ToolError(f"record evidence acquisition failed: {type(exc).__name__}: {exc}") from exc

    paths: dict[str, str] = {}
    compact: dict[str, Any] = {}
    for rubric_id, key in ((7, "rubric_7"), (9, "rubric_9")):
        item = evidence.get(key)
        if not isinstance(item, dict) or item.get("decision") not in DECISIONS:
            raise ToolError(f"record evidence returned invalid rubric {rubric_id} result")
        predicted = 1 if item["decision"] == "pass" else 0
        if item.get("predicted_score") != predicted:
            raise ToolError(f"record evidence rubric {rubric_id} score mismatch")
        result = {
            "schema_version": "resistance_agent_rubric_result.v2",
            "video_id": state["video_id"],
            "source_video_id": state["source_video_id"],
            "rubric_id": rubric_id,
            "decision": item["decision"],
            "predicted_score": predicted,
            "confidence": item.get("confidence"),
            "reason": item.get("reason"),
            "source_artifact": evidence.get("report_path"),
            "diagnostics": item.get("diagnostics", {}),
            "execution_mode": "execute_visual_evidence",
            **_skill_metadata(skill_plan, (7, 9)),
        }
        result_path = run_dir / "rubrics" / f"rubric_{rubric_id}.json"
        write_json(result_path, result)
        reopened = read_json(result_path)
        if reopened.get("decision") not in DECISIONS or reopened.get("predicted_score") != predicted:
            raise ToolError(f"rubric {rubric_id} artifact verification failed")
        paths[str(rubric_id)] = str(result_path.resolve())
        compact[str(rubric_id)] = {
            "decision": result["decision"],
            "predicted_score": result["predicted_score"],
            "confidence": result["confidence"],
            "reason": result["reason"],
            "result_path": paths[str(rubric_id)],
        }

    _verify_source_video(state)
    state["rubric_results"].update(paths)
    state.setdefault("rubric_evidence_reports", {})["7_9"] = evidence.get("report_path")
    state["status"] = "record_rubrics_completed"
    state["skill_plan"] = str((run_dir / "skills" / "live_skill_plan.json").resolve())
    state["tool_calls"].append(
        {
            "tool": "run_record_rubrics",
            "rubric_ids": [7, 9],
            "evidence_report": evidence.get("report_path"),
            "at": utc_now(),
        }
    )
    _save_state(run_dir, state)
    adaptive = {
        "adaptive_evidence_recommended": bool(
            evidence.get("adaptive_evidence_recommended", False)
        ),
        "adaptive_evidence_reasons": list(
            evidence.get("adaptive_evidence_reasons") or []
        ),
        "adaptive_request_template": evidence.get("adaptive_request_template"),
    }
    return {
        "status": state["status"],
        "video_id": state["video_id"],
        "rubrics": compact,
        "evidence_report": evidence.get("report_path"),
        **adaptive,
        **_skill_metadata(skill_plan, (7, 9)),
        "source_video_unchanged": True,
    }


def run_switch_rubric(run_id: str, use_fallback_temporal_guard: bool = False) -> dict[str, Any]:
    """Acquire real video evidence for R3 and write one binary artifact."""
    run_dir, state = _state(run_id)
    if state.get("mode") != "execute":
        raise ToolError("switch rubric is only valid in execute mode")
    _reject_historical_live_fallback(use_fallback_temporal_guard)
    skill_plan = _live_skill_plan(run_dir, state)
    try:
        try:
            from .skills import SkillExecutionError, execution_for_rubric
        except ImportError:
            from skills import SkillExecutionError, execution_for_rubric  # type: ignore
        execution = execution_for_rubric(skill_plan, 3)
    except SkillExecutionError as exc:
        raise ToolError(f"R3 live skill resolution failed: {exc}") from exc
    existing_path = state.get("rubric_results", {}).get("3")
    if existing_path:
        result_path = resolve_inside(existing_path, AGENT_ROOT)
        item = read_json(result_path)
        expected_score = 1 if item.get("decision") == "pass" else 0
        existing_executions = item.get("skill_executions")
        if not isinstance(existing_executions, list):
            existing_executions = []
        if (
            item.get("schema_version") != "resistance_agent_rubric_result.v2"
            or item.get("rubric_id") != 3
            or item.get("video_id") != state.get("video_id")
            or item.get("source_video_id") != state.get("source_video_id")
            or item.get("decision") not in DECISIONS
            or item.get("predicted_score") != expected_score
            or item.get("execution_mode") != "execute_visual_evidence"
            or item.get("routing_policy") != LIVE_ROUTING_POLICY
            or not any(
                isinstance(candidate, dict)
                and candidate.get("execution_fingerprint")
                == execution.get("execution_fingerprint")
                for candidate in existing_executions
            )
        ):
            raise ToolError("existing rubric 3 artifact is invalid")
        return {
            "status": state.get("status", "switch_rubric_completed"),
            "video_id": state["video_id"],
            "rubric": {
                "decision": item["decision"],
                "predicted_score": item["predicted_score"],
                "confidence": item.get("confidence"),
                "reason": item.get("reason"),
                "result_path": str(result_path),
            },
            "evidence_report": state.get("rubric_evidence_reports", {}).get("3"),
            "r3_frame_agent_report": state.get("r3_frame_agent_report"),
            **_skill_metadata(skill_plan, (3,)),
            "source_video_unchanged": True,
            "idempotent_replay": True,
        }
    if state.get("status") not in {
        "boundaries_completed",
        "switch_rubric_completed",
        "meter_rubrics_completed",
    } and not use_fallback_temporal_guard:
        raise ToolError("refine_rubric_boundaries must complete before switch rubric")
    source = _verify_source_video(state)
    input_dir = run_dir / "input_video"
    input_dir.mkdir(exist_ok=True)
    private_copy = input_dir / source.name
    if not private_copy.exists():
        shutil.copy2(source, private_copy)
    if private_copy.is_symlink() or not private_copy.is_file() or os.path.samefile(source, private_copy):
        raise ToolError(f"switch input is not an isolated file copy: {private_copy}")
    if private_copy.stat().st_size != source.stat().st_size or sha256(private_copy) != state["video"]["sha256"]:
        raise ToolError(f"switch input copy failed verification: {private_copy}")

    action_path: Path | None = None
    if isinstance(state.get("action_summary"), str) and state["action_summary"]:
        candidate = resolve_inside(state["action_summary"], AGENT_ROOT, must_exist=False)
        if candidate.is_file():
            action_path = candidate
    boundary_path: Path | None = None
    if isinstance(state.get("boundary_summary"), str) and state["boundary_summary"]:
        candidate = resolve_inside(state["boundary_summary"], AGENT_ROOT, must_exist=False)
        if candidate.is_file():
            boundary_path = candidate
    try:
        if execution["skill_id"] == "switch.adaptive_frame_sampling":
            try:
                from .r3_frame_agent_adapter import run_r3_frame_agent_live_skill
            except ImportError:
                from r3_frame_agent_adapter import run_r3_frame_agent_live_skill  # type: ignore
            stage_summary_path = boundary_path or action_path
            if stage_summary_path is None:
                raise ValueError("current run stage summary is required for the R3 frame Agent")
            evidence = run_r3_frame_agent_live_skill(
                video_path=private_copy,
                source_video_id=state["source_video_id"],
                video_id=state["video_id"],
                run_dir=run_dir,
                stage_summary_path=stage_summary_path,
                skill_execution=execution,
                routing_policy=LIVE_ROUTING_POLICY,
            )
        else:
            config = _state_config(state)
            try:
                from . import switch_rubric as switch_module
            except ImportError:
                import switch_rubric as switch_module  # type: ignore

            switch_arguments = {
                "video_path": private_copy,
                "source_video_id": state["source_video_id"],
                "video_id": state["video_id"],
                "run_dir": run_dir,
                "model_config": config["models"]["qwen"],
                "action_summary_path": action_path,
                "fallback_action_summary_path": None,
                "allow_historical_fallback": False,
                "skill_plan": skill_plan,
            }
            if getattr(switch_module.run_switch_rubric, "supports_boundary_summary", False):
                switch_arguments["boundary_summary_path"] = boundary_path
            evidence = switch_module.run_switch_rubric(**switch_arguments)
    except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise ToolError(f"switch evidence acquisition failed: {type(exc).__name__}: {exc}") from exc

    item = evidence.get("rubric_3")
    if not isinstance(item, dict) or item.get("decision") not in DECISIONS:
        raise ToolError("switch evidence returned invalid rubric 3 result")
    predicted = 1 if item["decision"] == "pass" else 0
    if item.get("predicted_score") != predicted:
        raise ToolError("switch evidence rubric 3 score mismatch")
    result = {
        "schema_version": "resistance_agent_rubric_result.v2",
        "video_id": state["video_id"],
        "source_video_id": state["source_video_id"],
        "rubric_id": 3,
        "decision": item["decision"],
        "predicted_score": predicted,
        "confidence": item.get("confidence"),
        "reason": item.get("reason"),
        "source_artifact": evidence.get("report_path"),
        "diagnostics": item.get("diagnostics", {}),
        "execution_mode": "execute_visual_evidence",
        **_skill_metadata(skill_plan, (3,)),
    }
    result_path = run_dir / "rubrics" / "rubric_3.json"
    write_json(result_path, result)
    reopened = read_json(result_path)
    if reopened.get("decision") not in DECISIONS or reopened.get("predicted_score") != predicted:
        raise ToolError("rubric 3 artifact verification failed")

    _verify_source_video(state)
    state["rubric_results"]["3"] = str(result_path.resolve())
    state.setdefault("rubric_evidence_reports", {})["3"] = evidence.get("report_path")
    state["r3_frame_agent_report"] = evidence.get("agent_report_path")
    state["skill_plan"] = str((run_dir / "skills" / "live_skill_plan.json").resolve())
    state["status"] = (
        "meter_rubrics_completed"
        if all(str(rubric_id) in state["rubric_results"] for rubric_id in (5, 6))
        else "switch_rubric_completed"
    )
    state["tool_calls"].append(
        {
            "tool": "run_switch_rubric",
            "rubric_ids": [3],
            "evidence_report": evidence.get("report_path"),
            "r3_frame_agent_report": evidence.get("agent_report_path"),
            "skill_id": execution["skill_id"],
            "at": utc_now(),
        }
    )
    _save_state(run_dir, state)
    return {
        "status": state["status"],
        "video_id": state["video_id"],
        "rubric": {
            "decision": result["decision"],
            "predicted_score": result["predicted_score"],
            "confidence": result["confidence"],
            "reason": result["reason"],
            "result_path": str(result_path.resolve()),
        },
        "evidence_report": evidence.get("report_path"),
        "r3_frame_agent_report": evidence.get("agent_report_path"),
        **_skill_metadata(skill_plan, (3,)),
        "source_video_unchanged": True,
    }


def _series_compatibility_fields(item: dict[str, Any]) -> dict[str, Any]:
    diagnostics = item.get("diagnostics")
    diagnostics = diagnostics if isinstance(diagnostics, dict) else {}

    def diagnostic_decision(name: str, default: str) -> str:
        value = item.get(name, diagnostics.get(name))
        if isinstance(value, dict):
            value = value.get("decision")
        return str(value) if value in DECISIONS else default

    def evidence_list(name: str) -> list[Any]:
        value = item.get(name, diagnostics.get(name))
        return list(value) if isinstance(value, list) else []

    decision = str(item["decision"])
    predicted = 1 if decision == "pass" else 0
    temporary = diagnostic_decision("temporary_direct_across_battery", "pass")
    final = diagnostic_decision("final_series_circuit", decision)
    path_relation = item.get("path_relation", diagnostics.get("path_relation"))
    if path_relation is None:
        path_relation = "direct" if temporary == "fail" else "no_connection"
    if path_relation not in {
        "direct",
        "via_component",
        "occluded_likely_direct",
        "no_connection",
        "unclear",
    }:
        raise ToolError("series evidence returned an invalid path_relation")
    decision_branch = item.get("decision_branch", diagnostics.get("decision_branch"))
    if not isinstance(decision_branch, str) or not decision_branch:
        decision_branch = "direct_violation" if decision == "fail" else "binary_fallback"
    return {
        "binary_score": predicted,
        "final_series_circuit": final,
        "temporary_direct_across_battery": temporary,
        "decision_branch": decision_branch,
        "path_relation": path_relation,
        "direct_observations": evidence_list("direct_observations"),
        "derived_observations": evidence_list("derived_observations"),
        "supporting_frame_ids": evidence_list("supporting_frame_ids"),
        "supporting_timestamps_seconds": evidence_list("supporting_timestamps_seconds"),
    }


def run_series_rubric(run_id: str, use_fallback_temporal_guard: bool = False) -> dict[str, Any]:
    """Acquire real video evidence for R1 and write one binary artifact."""
    run_dir, state = _state(run_id)
    if state.get("mode") != "execute":
        raise ToolError("series rubric is only valid in execute mode")
    _reject_historical_live_fallback(use_fallback_temporal_guard)
    skill_plan = _live_skill_plan(run_dir, state)
    existing_path = state.get("rubric_results", {}).get("1")
    recovered_current_run_artifact = False
    if not existing_path:
        current_run_result = run_dir / "rubrics" / "rubric_1.json"
        if current_run_result.is_file():
            existing_path = str(current_run_result)
            recovered_current_run_artifact = True
    if existing_path:
        result_path = resolve_inside(existing_path, run_dir)
        item = read_json(result_path)
        expected_score = 1 if item.get("decision") == "pass" else 0
        if (
            item.get("schema_version") != "resistance_agent_rubric_result.v2"
            or item.get("rubric_id") != 1
            or item.get("video_id") != state.get("video_id")
            or item.get("source_video_id") != state.get("source_video_id")
            or item.get("decision") not in DECISIONS
            or item.get("predicted_score") != expected_score
            or item.get("binary_score", expected_score) != expected_score
            or item.get("execution_mode") != "execute_visual_evidence"
            or item.get("routing_policy") != LIVE_ROUTING_POLICY
        ):
            raise ToolError("existing rubric 1 artifact is invalid")
        compatibility = _series_compatibility_fields(item)
        existing_report = state.get("rubric_evidence_reports", {}).get("1")
        evidence_report = (
            str(resolve_inside(existing_report, run_dir))
            if isinstance(existing_report, str) and existing_report
            else None
        )
        if recovered_current_run_artifact:
            report_candidate = run_dir / "series_rubric" / "series_evidence_report.json"
            if report_candidate.is_file():
                evidence_report = str(report_candidate.resolve())
                state.setdefault("rubric_evidence_reports", {})["1"] = evidence_report
            state.setdefault("rubric_results", {})["1"] = str(result_path)
            state.setdefault("tool_calls", []).append(
                {
                    "tool": "run_series_rubric",
                    "rubric_ids": [1],
                    "idempotent_replay": True,
                    "current_run_artifact_recovered": True,
                    "at": utc_now(),
                }
            )
            _save_state(run_dir, state)
        return {
            "status": state.get("status", "series_rubric_completed"),
            "video_id": state["video_id"],
            "rubric": {
                "decision": item["decision"],
                "predicted_score": item["predicted_score"],
                "confidence": item.get("confidence"),
                "reason": item.get("reason"),
                **compatibility,
                "result_path": str(result_path),
            },
            "evidence_report": evidence_report,
            **_skill_metadata(skill_plan, (1,)),
            "source_video_unchanged": True,
            "idempotent_replay": True,
        }
    boundary_summary = state.get("boundary_summary")
    if (
        not isinstance(boundary_summary, str)
        or not boundary_summary
    ) and not use_fallback_temporal_guard:
        raise ToolError("refine_rubric_boundaries must complete before series rubric")
    config = _state_config(state)
    source = _verify_source_video(state)
    input_dir = run_dir / "input_video"
    input_dir.mkdir(exist_ok=True)
    private_copy = input_dir / source.name
    if not private_copy.exists():
        shutil.copy2(source, private_copy)
    if private_copy.is_symlink() or not private_copy.is_file() or os.path.samefile(source, private_copy):
        raise ToolError(f"series input is not an isolated file copy: {private_copy}")
    if private_copy.stat().st_size != source.stat().st_size or sha256(private_copy) != state["video"]["sha256"]:
        raise ToolError(f"series input copy failed verification: {private_copy}")

    action_path: Path | None = None
    if isinstance(state.get("action_summary"), str) and state["action_summary"]:
        candidate = resolve_inside(state["action_summary"], run_dir, must_exist=False)
        if candidate.is_file():
            action_path = candidate
    boundary_path: Path | None = None
    if isinstance(state.get("boundary_summary"), str) and state["boundary_summary"]:
        candidate = resolve_inside(state["boundary_summary"], run_dir, must_exist=False)
        if candidate.is_file():
            boundary_path = candidate
    try:
        try:
            from . import series_rubric as series_module
        except ImportError:
            import series_rubric as series_module  # type: ignore

        arguments = {
            "video_path": private_copy,
            "source_video_id": state["source_video_id"],
            "video_id": state["video_id"],
            "run_dir": run_dir,
            "model_config": config["models"]["qwen"],
            "action_summary_path": action_path,
            "skill_plan": skill_plan,
        }
        if getattr(series_module.run_series_rubric, "supports_boundary_summary", False):
            arguments["boundary_summary_path"] = boundary_path
        evidence = series_module.run_series_rubric(**arguments)
    except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise ToolError(f"series evidence acquisition failed: {type(exc).__name__}: {exc}") from exc

    evidence_report_value = evidence.get("report_path")
    if not isinstance(evidence_report_value, str) or not evidence_report_value:
        raise ToolError("series evidence returned no current-run report")
    evidence_report = resolve_inside(evidence_report_value, run_dir)
    item = evidence.get("rubric_1")
    if not isinstance(item, dict) or item.get("decision") not in DECISIONS:
        raise ToolError("series evidence returned invalid rubric 1 result")
    predicted = 1 if item["decision"] == "pass" else 0
    if item.get("predicted_score", item.get("binary_score")) != predicted:
        raise ToolError("series evidence rubric 1 score mismatch")
    if "binary_score" in item and item["binary_score"] != predicted:
        raise ToolError("series evidence rubric 1 binary score mismatch")
    compatibility = _series_compatibility_fields(item)
    result = {
        "schema_version": "resistance_agent_rubric_result.v2",
        "video_id": state["video_id"],
        "source_video_id": state["source_video_id"],
        "rubric_id": 1,
        "decision": item["decision"],
        "predicted_score": predicted,
        **compatibility,
        "confidence": item.get("confidence"),
        "reason": item.get("reason"),
        "source_artifact": str(evidence_report),
        "diagnostics": item.get("diagnostics", {}),
        "execution_mode": "execute_visual_evidence",
        **_skill_metadata(skill_plan, (1,)),
        "plan_live_skills": item.get(
            "plan_live_skills",
            {
                "selection_basis": skill_plan["selection_basis"],
                "observed_stages": skill_plan["observed_stages"],
                "selected_skills": skill_plan["selected_skills"],
                "video_id_used_for_routing": False,
                "historical_artifacts_used": False,
                "fixed_video_roi_used": False,
            },
        ),
    }
    result_path = run_dir / "rubrics" / "rubric_1.json"
    write_json(result_path, result)
    reopened = read_json(result_path)
    if reopened.get("decision") not in DECISIONS or reopened.get("predicted_score") != predicted:
        raise ToolError("rubric 1 artifact verification failed")

    _verify_source_video(state)
    state["rubric_results"]["1"] = str(result_path.resolve())
    state.setdefault("rubric_evidence_reports", {})["1"] = str(evidence_report)
    state["skill_plan"] = str((run_dir / "skills" / "live_skill_plan.json").resolve())
    if all(str(rubric_id) in state["rubric_results"] for rubric_id in (1, 3, 5, 6)):
        state["status"] = "meter_rubrics_completed"
    else:
        state["status"] = "series_rubric_completed"
    state["tool_calls"].append(
        {
            "tool": "run_series_rubric",
            "rubric_ids": [1],
            "evidence_report": str(evidence_report),
            "at": utc_now(),
        }
    )
    _save_state(run_dir, state)
    return {
        "status": state["status"],
        "video_id": state["video_id"],
        "rubric": {
            "decision": result["decision"],
            "predicted_score": result["predicted_score"],
            "confidence": result["confidence"],
            "reason": result["reason"],
            **compatibility,
            "result_path": str(result_path.resolve()),
        },
        "evidence_report": str(evidence_report),
        **_skill_metadata(skill_plan, (1,)),
        "source_video_unchanged": True,
    }


def run_remaining_rubrics(run_id: str, use_fallback_temporal_guard: bool = False) -> dict[str, Any]:
    """Acquire real video evidence for R0/R2/R8 and write binary artifacts."""
    run_dir, state = _state(run_id)
    rubric_ids = (0, 2, 8)
    if state.get("mode") != "execute":
        raise ToolError("remaining rubrics are only valid in execute mode")
    _reject_historical_live_fallback(use_fallback_temporal_guard)
    skill_plan = _live_skill_plan(run_dir, state)
    existing_paths = state.get("rubric_results", {})
    if all(str(rubric_id) in existing_paths for rubric_id in rubric_ids):
        existing: dict[str, Any] = {}
        for rubric_id in rubric_ids:
            result_path = resolve_inside(existing_paths[str(rubric_id)], AGENT_ROOT)
            item = read_json(result_path)
            expected_score = 1 if item.get("decision") == "pass" else 0
            if (
                item.get("schema_version") != "resistance_agent_rubric_result.v2"
                or item.get("rubric_id") != rubric_id
                or item.get("video_id") != state.get("video_id")
                or item.get("source_video_id") != state.get("source_video_id")
                or item.get("decision") not in DECISIONS
                or item.get("predicted_score") != expected_score
                or item.get("execution_mode") != "execute_visual_evidence"
                or item.get("routing_policy") != LIVE_ROUTING_POLICY
            ):
                raise ToolError(f"existing rubric {rubric_id} artifact is invalid")
            existing[str(rubric_id)] = {
                "decision": item["decision"],
                "predicted_score": item["predicted_score"],
                "confidence": item.get("confidence"),
                "reason": item.get("reason"),
                "result_path": str(result_path),
            }
        return {
            "status": state.get("status", "remaining_rubrics_completed"),
            "video_id": state["video_id"],
            "rubrics": existing,
            "evidence_report": state.get("rubric_evidence_reports", {}).get("0_2_8"),
            **_skill_metadata(skill_plan, rubric_ids),
            "source_video_unchanged": True,
            "idempotent_replay": True,
        }
    if state.get("status") not in {
        "boundaries_completed",
        "switch_rubric_completed",
        "series_rubric_completed",
        "meter_rubrics_completed",
        "record_rubrics_completed",
        "remaining_rubrics_completed",
    } and not use_fallback_temporal_guard:
        raise ToolError("refine_rubric_boundaries must complete before remaining rubrics")
    config = _state_config(state)
    source = _verify_source_video(state)
    input_dir = run_dir / "input_video"
    input_dir.mkdir(exist_ok=True)
    private_copy = input_dir / source.name
    if not private_copy.exists():
        shutil.copy2(source, private_copy)
    if private_copy.is_symlink() or not private_copy.is_file() or os.path.samefile(source, private_copy):
        raise ToolError(f"remaining-rubric input is not an isolated file copy: {private_copy}")
    if private_copy.stat().st_size != source.stat().st_size or sha256(private_copy) != state["video"]["sha256"]:
        raise ToolError(f"remaining-rubric input copy failed verification: {private_copy}")

    action_path: Path | None = None
    if isinstance(state.get("action_summary"), str) and state["action_summary"]:
        candidate = resolve_inside(state["action_summary"], AGENT_ROOT, must_exist=False)
        if candidate.is_file():
            action_path = candidate
    boundary_path: Path | None = None
    if isinstance(state.get("boundary_summary"), str) and state["boundary_summary"]:
        candidate = resolve_inside(state["boundary_summary"], AGENT_ROOT, must_exist=False)
        if candidate.is_file():
            boundary_path = candidate
    try:
        try:
            from . import remaining_rubrics as remaining_module
        except ImportError:
            import remaining_rubrics as remaining_module  # type: ignore

        arguments = {
            "video_path": private_copy,
            "source_video_id": state["source_video_id"],
            "video_id": state["video_id"],
            "run_dir": run_dir,
            "model_config": config["models"]["qwen"],
            "action_summary_path": action_path,
            "fallback_action_summary_path": None,
            "allow_video_calibration": False,
            "enable_specialized_r8": True,
            "allow_historical_fallback": False,
            "skill_plan": skill_plan,
        }
        if getattr(remaining_module.run_remaining_rubrics, "supports_boundary_summary", False):
            arguments["boundary_summary_path"] = boundary_path
        evidence = remaining_module.run_remaining_rubrics(**arguments)
    except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise ToolError(f"remaining evidence acquisition failed: {type(exc).__name__}: {exc}") from exc

    paths: dict[str, str] = {}
    compact: dict[str, Any] = {}
    for rubric_id in rubric_ids:
        item = evidence.get(f"rubric_{rubric_id}")
        if not isinstance(item, dict) or item.get("decision") not in DECISIONS:
            raise ToolError(f"remaining evidence returned invalid rubric {rubric_id} result")
        predicted = 1 if item["decision"] == "pass" else 0
        if item.get("predicted_score") != predicted:
            raise ToolError(f"remaining evidence rubric {rubric_id} score mismatch")
        result = {
            "schema_version": "resistance_agent_rubric_result.v2",
            "video_id": state["video_id"],
            "source_video_id": state["source_video_id"],
            "rubric_id": rubric_id,
            "decision": item["decision"],
            "predicted_score": predicted,
            "confidence": item.get("confidence"),
            "reason": item.get("reason"),
            "source_artifact": evidence.get(f"rubric_{rubric_id}_report_path") or evidence.get("report_path"),
            "diagnostics": item.get("diagnostics", {}),
            "execution_mode": "execute_visual_evidence",
            **_skill_metadata(skill_plan, rubric_ids),
        }
        result_path = run_dir / "rubrics" / f"rubric_{rubric_id}.json"
        write_json(result_path, result)
        reopened = read_json(result_path)
        if reopened.get("decision") not in DECISIONS or reopened.get("predicted_score") != predicted:
            raise ToolError(f"rubric {rubric_id} artifact verification failed")
        paths[str(rubric_id)] = str(result_path.resolve())
        compact[str(rubric_id)] = {
            "decision": result["decision"],
            "predicted_score": result["predicted_score"],
            "confidence": result["confidence"],
            "reason": result["reason"],
            "result_path": paths[str(rubric_id)],
        }

    _verify_source_video(state)
    state["rubric_results"].update(paths)
    state.setdefault("rubric_evidence_reports", {})["0_2_8"] = evidence.get("report_path")
    if evidence.get("rubric_8_report_path"):
        state["rubric_evidence_reports"]["8"] = evidence["rubric_8_report_path"]
    state["status"] = "remaining_rubrics_completed"
    state["skill_plan"] = str((run_dir / "skills" / "live_skill_plan.json").resolve())
    state["tool_calls"].append(
        {
            "tool": "run_remaining_rubrics",
            "rubric_ids": list(rubric_ids),
            "evidence_report": evidence.get("report_path"),
            "at": utc_now(),
        }
    )
    _save_state(run_dir, state)
    return {
        "status": state["status"],
        "video_id": state["video_id"],
        "rubrics": compact,
        "evidence_report": evidence.get("report_path"),
        **_skill_metadata(skill_plan, rubric_ids),
        "source_video_unchanged": True,
    }


def run_polarity_rubric(run_id: str, use_fallback_temporal_guard: bool = False) -> dict[str, Any]:
    """Derive R4 from the current run's direct R5 meter-pointer evidence."""
    run_dir, state = _state(run_id)
    if state.get("mode") != "execute":
        raise ToolError("polarity rubric is only valid in execute mode")
    _reject_historical_live_fallback(use_fallback_temporal_guard)
    skill_plan = _live_skill_plan(run_dir, state)
    if "5" not in state.get("rubric_results", {}):
        run_meter_rubrics(run_id, use_fallback_temporal_guard=False)
        run_dir, state = _state(run_id)
        skill_plan = _live_skill_plan(run_dir, state)

    r5_result_path = resolve_inside(state["rubric_results"]["5"], AGENT_ROOT)
    if r5_result_path.parent != (run_dir / "rubrics").resolve() or r5_result_path.name != "rubric_5.json":
        raise ToolError("R5 dependency must be the current run rubric_5.json")
    r5_item = read_json(r5_result_path)
    r5_expected_score = 1 if r5_item.get("decision") == "pass" else 0
    if (
        r5_item.get("schema_version") != "resistance_agent_rubric_result.v2"
        or r5_item.get("rubric_id") != 5
        or r5_item.get("video_id") != state.get("video_id")
        or r5_item.get("source_video_id") != state.get("source_video_id")
        or r5_item.get("decision") not in DECISIONS
        or r5_item.get("predicted_score") != r5_expected_score
        or r5_item.get("execution_mode") != "execute_visual_evidence"
        or r5_item.get("routing_policy") != LIVE_ROUTING_POLICY
    ):
        raise ToolError("current run R5 dependency is invalid")
    r5_evidence_path = resolve_inside(r5_item.get("source_artifact"), run_dir)
    expected_r5_evidence_path = (run_dir / "meter_rubrics" / "meter_evidence_report.json").resolve()
    if r5_evidence_path != expected_r5_evidence_path:
        raise ToolError("R5 dependency must point to the current run meter evidence report")
    r5_evidence = read_json(r5_evidence_path)
    if (
        r5_evidence.get("schema_version") != "resistance_agent_meter_evidence.v1"
        or r5_evidence.get("video_id") != state.get("video_id")
        or r5_evidence.get("source_video_id") != state.get("source_video_id")
        or r5_evidence.get("excel_accessed") is not False
        or r5_evidence.get("ground_truth_sent_to_model") is not False
        or r5_evidence.get("historical_fallback_used") is not False
        or r5_evidence.get("routing_policy") != LIVE_ROUTING_POLICY
    ):
        raise ToolError("current run R5 evidence report provenance is invalid")
    r5_result_sha256 = sha256(r5_result_path)

    existing_path = state.get("rubric_results", {}).get("4")
    if existing_path:
        result_path = resolve_inside(existing_path, AGENT_ROOT)
        item = read_json(result_path)
        expected_score = 1 if item.get("decision") == "pass" else 0
        algorithm = item.get("diagnostics", {}).get("algorithm_version")
        if (
            item.get("schema_version") == "resistance_agent_rubric_result.v2"
            and item.get("rubric_id") == 4
            and item.get("video_id") == state.get("video_id")
            and item.get("source_video_id") == state.get("source_video_id")
            and item.get("decision") in DECISIONS
            and item.get("predicted_score") == expected_score
            and item.get("execution_mode") == "execute_visual_evidence"
            and item.get("routing_policy") == LIVE_ROUTING_POLICY
            and algorithm == "r4_meter_polarity_v21_r5_direct_meter_pointer"
            and item.get("diagnostics", {}).get("r5_result_sha256") == r5_result_sha256
        ):
            return {
                "status": state.get("status", "polarity_rubric_completed"),
                "video_id": state["video_id"],
                "rubric": {
                    "decision": item["decision"],
                    "predicted_score": item["predicted_score"],
                    "confidence": item.get("confidence"),
                    "reason": item.get("reason"),
                    "result_path": str(result_path),
                },
                "evidence_report": state.get("rubric_evidence_reports", {}).get("4")
                or state.get("rubric_evidence_reports", {}).get("4_v16"),
                **_skill_metadata(skill_plan, (4,)),
                "source_video_unchanged": True,
                "idempotent_replay": True,
            }
    if state.get("status") not in {
        "boundaries_completed",
        "switch_rubric_completed",
        "series_rubric_completed",
        "meter_rubrics_completed",
        "record_rubrics_completed",
        "remaining_rubrics_completed",
        "polarity_rubric_completed",
    } and not use_fallback_temporal_guard:
        raise ToolError("refine_rubric_boundaries must complete before polarity rubric")
    config = _state_config(state)
    source = _verify_source_video(state)
    input_dir = run_dir / "input_video"
    input_dir.mkdir(exist_ok=True)
    private_copy = input_dir / source.name
    if not private_copy.exists():
        shutil.copy2(source, private_copy)
    if private_copy.is_symlink() or not private_copy.is_file() or os.path.samefile(source, private_copy):
        raise ToolError(f"polarity input is not an isolated file copy: {private_copy}")
    if private_copy.stat().st_size != source.stat().st_size or sha256(private_copy) != state["video"]["sha256"]:
        raise ToolError(f"polarity input copy failed verification: {private_copy}")

    action_path: Path | None = None
    if isinstance(state.get("action_summary"), str) and state["action_summary"]:
        candidate = resolve_inside(state["action_summary"], AGENT_ROOT, must_exist=False)
        if candidate.is_file():
            action_path = candidate
    try:
        try:
            from . import polarity_rubric as polarity_module
        except ImportError:
            import polarity_rubric as polarity_module  # type: ignore

        evidence = polarity_module.run_polarity_rubric(
            video_path=private_copy,
            source_video_id=state["source_video_id"],
            video_id=state["video_id"],
            run_dir=run_dir,
            model_config=config["models"]["qwen"],
            action_summary_path=action_path,
            fallback_action_summary_path=None,
            stage_manifest_path=None,
            reference_manifest_path=None,
            detector_root=None,
            allow_video_calibration=False,
            allow_historical_fallback=False,
            skill_plan=skill_plan,
            r5_result_path=r5_result_path,
        )
    except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise ToolError(f"polarity evidence acquisition failed: {type(exc).__name__}: {exc}") from exc
    item = evidence.get("rubric_4")
    if not isinstance(item, dict) or item.get("decision") not in DECISIONS:
        raise ToolError("polarity evidence returned invalid rubric 4 result")
    predicted = 1 if item["decision"] == "pass" else 0
    if item.get("predicted_score") != predicted:
        raise ToolError("polarity evidence rubric 4 score mismatch")
    result = {
        "schema_version": "resistance_agent_rubric_result.v2",
        "video_id": state["video_id"],
        "source_video_id": state["source_video_id"],
        "rubric_id": 4,
        "decision": item["decision"],
        "predicted_score": predicted,
        "confidence": item.get("confidence"),
        "reason": item.get("reason"),
        "source_artifact": evidence.get("report_path"),
        "diagnostics": item.get("diagnostics", {}),
        "execution_mode": "execute_visual_evidence",
        **_skill_metadata(skill_plan, (4,)),
    }
    result_path = run_dir / "rubrics" / "rubric_4.json"
    write_json(result_path, result)
    reopened = read_json(result_path)
    if reopened.get("decision") not in DECISIONS or reopened.get("predicted_score") != predicted:
        raise ToolError("rubric 4 artifact verification failed")
    _verify_source_video(state)
    state["rubric_results"]["4"] = str(result_path.resolve())
    state.setdefault("rubric_evidence_reports", {})["4"] = evidence.get("report_path")
    state["status"] = "polarity_rubric_completed"
    state["skill_plan"] = str((run_dir / "skills" / "live_skill_plan.json").resolve())
    state["tool_calls"].append(
        {"tool": "run_polarity_rubric", "rubric_ids": [4], "evidence_report": evidence.get("report_path"), "at": utc_now()}
    )
    _save_state(run_dir, state)
    return {
        "status": state["status"],
        "video_id": state["video_id"],
        "rubric": {
            "decision": result["decision"],
            "predicted_score": result["predicted_score"],
            "confidence": result["confidence"],
            "reason": result["reason"],
            "result_path": str(result_path.resolve()),
        },
        "evidence_report": evidence.get("report_path"),
        **_skill_metadata(skill_plan, (4,)),
        "source_video_unchanged": True,
    }


def inspect_run_status(run_id: str) -> dict[str, Any]:
    run_dir, state = _state(run_id)
    missing = [index for index in range(10) if str(index) not in state.get("rubric_results", {})]
    return {
        "run_id": state["run_id"],
        "mode": state["mode"],
        "status": state["status"],
        "video_id": state["video_id"],
        "action_summary": state.get("action_summary"),
        "boundary_plan": state.get("boundary_plan"),
        "boundary_summary": state.get("boundary_summary"),
        "skill_plan": state.get("skill_plan"),
        "frame_agent_report": state.get("frame_agent_report"),
        "r3_frame_agent_report": state.get("r3_frame_agent_report"),
        "routing_policy": LIVE_ROUTING_POLICY if state.get("mode") == "execute" else None,
        "rubric_summary": state.get("rubric_summary"),
        "rubric_evidence_reports": state.get("rubric_evidence_reports", {}),
        "completed_rubrics": 10 - len(missing),
        "missing_rubrics": missing,
        "run_dir": str(run_dir.resolve()),
    }


def validate_run(run_id: str) -> dict[str, Any]:
    run_dir, state = _state(run_id)
    errors: list[str] = []
    decisions: dict[str, str] = {}
    for rubric_id in range(10):
        raw_path = state.get("rubric_results", {}).get(str(rubric_id))
        if not raw_path:
            errors.append(f"rubric_{rubric_id}_missing")
            continue
        try:
            item = read_json(resolve_inside(raw_path, AGENT_ROOT))
        except (OSError, ValueError, json.JSONDecodeError, ToolError) as exc:
            errors.append(f"rubric_{rubric_id}_invalid:{exc}")
            continue
        if item.get("schema_version") != "resistance_agent_rubric_result.v2":
            errors.append(f"rubric_{rubric_id}_schema_mismatch")
        elif item.get("rubric_id") != rubric_id:
            errors.append(f"rubric_{rubric_id}_identity_mismatch")
        elif item.get("video_id") != state["video_id"]:
            errors.append(f"rubric_{rubric_id}_video_mismatch")
        elif item.get("source_video_id") != state["source_video_id"]:
            errors.append(f"rubric_{rubric_id}_source_mismatch")
        elif item.get("decision") not in DECISIONS:
            errors.append(f"rubric_{rubric_id}_nonbinary")
        elif item.get("predicted_score") != (1 if item["decision"] == "pass" else 0):
            errors.append(f"rubric_{rubric_id}_score_mismatch")
        elif state.get("mode") == "execute" and item.get("routing_policy") != LIVE_ROUTING_POLICY:
            errors.append(f"rubric_{rubric_id}_live_routing_policy_mismatch")
        else:
            decisions[str(rubric_id)] = item["decision"]
    report = {
        "schema_version": "resistance_agent_validation.v2",
        "run_id": state["run_id"],
        "video_id": state["video_id"],
        "valid": not errors,
        "binary_count": len(decisions),
        "errors": errors,
        "decisions": decisions,
        "validated_at": utc_now(),
    }
    report_path = run_dir / "validation.json"
    write_json(report_path, report)
    return {**report, "report_path": str(report_path.resolve())}


def finalize_run(run_id: str) -> dict[str, Any]:
    run_dir, state = _state(run_id)
    validation = validate_run(run_id)
    if not validation["valid"]:
        raise ToolError(f"run is incomplete: {validation['errors']}")
    _verify_source_video(state)
    results = [
        read_json(resolve_inside(state["rubric_results"][str(index)], AGENT_ROOT))
        for index in range(10)
    ]
    final = {
        "schema_version": "resistance_agent_final.v2",
        "run_id": state["run_id"],
        "video_id": state["video_id"],
        "source_video_id": state["source_video_id"],
        "mode": state["mode"],
        "status": "completed",
        "routing_policy": LIVE_ROUTING_POLICY if state.get("mode") == "execute" else None,
        "skill_plan": state.get("skill_plan"),
        "results": results,
        "decision_counts": {
            "pass": sum(item["decision"] == "pass" for item in results),
            "fail": sum(item["decision"] == "fail" for item in results),
        },
        "completed_at": utc_now(),
        "source_videos_modified": False,
        "excel_accessed": False,
    }
    final_path = run_dir / "final_result.json"
    write_json(final_path, final)
    state["status"] = "completed"
    state["final_result"] = str(final_path.resolve())
    state["tool_calls"].append({"tool": "finalize_run", "at": utc_now()})
    _save_state(run_dir, state)
    return {
        "status": final["status"],
        "run_id": final["run_id"],
        "video_id": final["video_id"],
        "source_video_id": final["source_video_id"],
        "mode": final["mode"],
        "result_count": len(results),
        "decision_counts": final["decision_counts"],
        "final_result_path": str(final_path.resolve()),
        "validation_path": validation["report_path"],
    }


RUBRIC_PRODUCER_GROUPS: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("run_switch_rubric", (3,)),
    ("run_series_rubric", (1,)),
    ("run_meter_rubrics", (5, 6)),
    ("run_record_rubrics", (7, 9)),
    ("run_remaining_rubrics", (0, 2, 8)),
    ("run_polarity_rubric", (4,)),
)


def _rubric_bundle_plan(
    rubric_ids: list[int], skill_plan: dict[str, Any] | None = None
) -> dict[str, Any]:
    if not isinstance(rubric_ids, list) or not rubric_ids:
        raise ToolError("rubric_ids must be a non-empty array")
    normalized: list[int] = []
    for rubric_id in rubric_ids:
        if type(rubric_id) is not int or rubric_id not in range(10):
            raise ToolError("each rubric_id must be an integer from 0 through 9")
        if rubric_id not in normalized:
            normalized.append(rubric_id)

    requested = set(normalized)
    if skill_plan is None:
        producer_plan = [
            {"tool": tool_name, "rubric_ids": list(produced_ids)}
            for tool_name, produced_ids in RUBRIC_PRODUCER_GROUPS
            if requested.intersection(produced_ids)
        ]
    else:
        try:
            from .skills import SkillExecutionError, producer_plan as registered_producer_plan
        except ImportError:
            from skills import SkillExecutionError, producer_plan as registered_producer_plan  # type: ignore
        try:
            producer_plan = registered_producer_plan(skill_plan, tuple(normalized))
        except SkillExecutionError as exc:
            raise ToolError(f"registered skill dispatch failed: {exc}") from exc
    co_produced = sorted(
        rubric_id
        for item in producer_plan
        for rubric_id in item["rubric_ids"]
    )
    producer_call_count = len(producer_plan)
    return {
        "requested_rubric_ids": normalized,
        "co_produced_rubric_ids": co_produced,
        "producer_plan": producer_plan,
        "per_rubric_call_count": len(normalized),
        "producer_call_count": producer_call_count,
        "saved_call_count": len(normalized) - producer_call_count,
    }


def run_rubric_bundle(
    run_id: str,
    rubric_ids: list[int],
    use_fallback_temporal_guard: bool = False,
) -> dict[str, Any]:
    """Dispatch requested rubrics to the minimum set of evidence producers."""
    run_dir = AGENT_ROOT / "runs" / sanitize_run_id(run_id)
    state_path = run_dir / "state.json"
    if state_path.is_file():
        _, state = _state(run_id)
        if state.get("mode") != "execute":
            raise ToolError("rubric bundle is only valid in execute mode")
        _reject_historical_live_fallback(use_fallback_temporal_guard)
        skill_plan = _live_skill_plan(run_dir, state)
    else:
        # Keep the pure grouping helper usable in unit tests and regression
        # tooling; real MCP execution always has a created execute run.
        skill_plan = {
            "routing_policy": LIVE_ROUTING_POLICY,
            "selection_basis": "grouping-only fixture",
            "skills": [],
        }
    plan = _rubric_bundle_plan(rubric_ids, skill_plan if state_path.is_file() else None)
    producer_calls: list[dict[str, Any]] = []
    video_id: str | None = None
    for item in plan["producer_plan"]:
        tool_name = item["tool"]
        producer = globals().get(tool_name)
        if not callable(producer):
            raise ToolError(f"rubric producer is unavailable: {tool_name}")
        result = producer(
            run_id=run_id,
            use_fallback_temporal_guard=use_fallback_temporal_guard,
        )
        current_video_id = result.get("video_id")
        if video_id is None and isinstance(current_video_id, str):
            video_id = current_video_id
        elif current_video_id != video_id:
            raise ToolError(f"rubric producer returned inconsistent video_id: {tool_name}")
        reused = result.get("idempotent_replay") is True
        producer_calls.append(
            {
                "tool": tool_name,
                "rubric_ids": item["rubric_ids"],
                "execution": "reused" if reused else "executed",
                "status": result.get("status"),
                "evidence_report": result.get("evidence_report"),
                "rubrics": result.get("rubrics"),
                "adaptive_evidence_recommended": result.get(
                    "adaptive_evidence_recommended", False
                ),
                "adaptive_evidence_reasons": result.get("adaptive_evidence_reasons", []),
                "adaptive_request_template": result.get("adaptive_request_template"),
                "source_video_unchanged": result.get("source_video_unchanged") is True,
            }
        )

    return {
        "status": "rubric_bundle_completed",
        "run_id": run_id,
        "video_id": video_id,
        **_skill_metadata(skill_plan, tuple(plan["requested_rubric_ids"])),
        **plan,
        "executed_producers": [
            item["tool"] for item in producer_calls if item["execution"] == "executed"
        ],
        "reused_producers": [
            item["tool"] for item in producer_calls if item["execution"] == "reused"
        ],
        "producer_calls": producer_calls,
        "source_video_unchanged": all(
            item["source_video_unchanged"] for item in producer_calls
        ),
    }


TOOL_REGISTRY: dict[str, Callable[..., dict[str, Any]]] = {
    "discover_videos": discover_videos,
    "inspect_video": inspect_video,
    "create_run": create_run,
    "run_full_pipeline": run_full_pipeline,
    "refine_rubric_boundaries": refine_rubric_boundaries,
    "plan_live_skills": plan_live_skills,
    "run_adaptive_frame_agent": run_adaptive_frame_agent,
    "request_additional_evidence": request_additional_evidence,
    "run_switch_rubric": run_switch_rubric,
    "run_series_rubric": run_series_rubric,
    "run_meter_rubrics": run_meter_rubrics,
    "run_record_rubrics": run_record_rubrics,
    "run_remaining_rubrics": run_remaining_rubrics,
    "run_polarity_rubric": run_polarity_rubric,
    "run_rubric_bundle": run_rubric_bundle,
    "load_rubric_result": load_rubric_result,
    "load_rubric_bundle": load_rubric_bundle,
    "inspect_run_status": inspect_run_status,
    "validate_run": validate_run,
    "finalize_run": finalize_run,
}


def _schema(
    name: str,
    description: str,
    properties: dict[str, Any],
    required: list[str],
) -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "inputSchema": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
    }


RUN_ID = {"type": "string", "pattern": "^[A-Za-z0-9_.-]{1,80}$"}
TOOL_SCHEMAS: list[dict[str, Any]] = [
    _schema("discover_videos", "List configured local videos without modifying them.", {"config_path": {"type": "string"}}, []),
    _schema("inspect_video", "Inspect one local video and return its canonical filename and media metadata.", {"video_ref": {"type": "string"}, "config_path": {"type": "string"}}, ["video_ref"]),
    _schema("create_run", "Create an isolated run and pin its config digest.", {"run_id": RUN_ID, "video_ref": {"type": "string"}, "mode": {"type": "string", "enum": ["replay", "prepare", "execute"]}, "config_path": {"type": "string"}}, ["run_id", "video_ref"]),
    _schema("run_full_pipeline", "Plan or execute marker filtering, experiment boundary detection, hierarchical v2, and wiring config generation for one video.", {"run_id": RUN_ID, "dry_run": {"type": "boolean"}}, ["run_id"]),
    _schema("refine_rubric_boundaries", "Plan or execute Rubric boundary refinement while preserving source stage runs.", {"run_id": RUN_ID, "execute": {"type": "boolean"}}, ["run_id"]),
    _schema("plan_live_skills", "Select evidence Skills from the current run's observed stage situation; never route by video identity or historical result.", {"run_id": RUN_ID}, ["run_id"]),
    _schema(
        "run_adaptive_frame_agent",
        "Extract current-run frame groups with dynamic ROI and ask Qwen for visible instrument quality; request bounded adjacent frames when the view is weak. This tool never scores a rubric.",
        {
            "run_id": RUN_ID,
            "max_rounds": {"type": "integer", "minimum": 1, "maximum": 2},
            "initial_interval_seconds": {"type": "number", "minimum": 0.1, "maximum": 2.0},
        },
        ["run_id"],
    ),
    _schema(
        "request_additional_evidence",
        "Request bounded adjacent current-run frames when R5/R6 meter evidence or R7/R9 meter/paper evidence is weak; the local executor enforces stage, duration, frame-count and dynamic-ROI limits.",
        {
            "run_id": RUN_ID,
            "rubric_ids": {
                "type": "array",
                "items": {"type": "integer", "enum": [5, 6, 7, 9]},
                "minItems": 1,
            },
            "reason": {
                "type": "string",
                "enum": [
                    "meter_pointer_occluded",
                    "meter_identity_conflict",
                    "pointer_state_conflict",
                    "low_confidence",
                    "adjacent_state_change",
                    "paper_not_found",
                    "writing_occlusion",
                    "field_missing",
                    "single_frame_support",
                    "digit_conflict",
                    "row_identity_conflict",
                    "recording_stage_missing",
                    "ammeter_missing",
                    "voltmeter_missing",
                    "ammeter_no_stable_deflection",
                    "voltmeter_no_stable_deflection",
                    "ammeter_single_frame_support",
                    "voltmeter_single_frame_support",
                    "ammeter_range_conflict",
                    "voltmeter_range_conflict",
                    "ammeter_reading_conflict",
                    "voltmeter_reading_conflict",
                    "ammeter_low_confidence",
                    "voltmeter_low_confidence",
                    "no_stable_dual_meter_frames",
                    "other",
                ],
            },
            "time_ranges": {
                "type": "array",
                "minItems": 1,
                "maxItems": 3,
                "items": {
                    "type": "object",
                    "properties": {
                        "start_seconds": {"type": "number"},
                        "end_seconds": {"type": "number"},
                    },
                    "required": ["start_seconds", "end_seconds"],
                    "additionalProperties": False,
                },
            },
            "interval_seconds": {"type": "number", "minimum": 0.1, "maximum": 1.0},
            "max_frames": {"type": "integer", "minimum": 1, "maximum": 32},
            "roi_mode": {"type": "string", "enum": ["dynamic_meter_candidates", "dynamic_paper_tracking"]},
            "view": {"type": "string", "enum": ["meter_pair", "paper_full", "paper_fields"]},
            "evidence_profile": {"type": "string", "enum": ["meter_pair", "record_meter", "record_paper"]},
            "cycle": {"type": "integer", "enum": [1, 2]},
            "target_fields": {
                "type": "array",
                "items": {"type": "string", "enum": ["u1", "i1", "u2", "i2"]},
            },
            "target_roles": {
                "type": "array",
                "items": {"type": "string", "enum": ["ammeter", "voltmeter"]},
            },
            "anchor_frame_ids": {
                "type": "array",
                "items": {"type": "string", "pattern": "^frame_[0-9]+$"},
            },
            "search_mode": {
                "type": "string",
                "enum": ["adjacent_dense", "post_write_reveal", "recording_stage_coverage", "current_run_broad_writing_search", "adjacent_meter_dense", "current_run_meter_search"],
            },
        },
        ["run_id", "rubric_ids", "reason", "time_ranges"],
    ),
    _schema(
        "run_switch_rubric",
        "Use all current Temporal Guard wiring/rewiring windows and pure OpenCV same-frame switch/plug overlap on the real video copy to generate binary R3 evidence.",
        {"run_id": RUN_ID},
        ["run_id"],
    ),
    _schema(
        "run_series_rubric",
        "Use Temporal Guard, OpenCV and Qwen on the real video copy to generate terminal-state and topology evidence for binary R1.",
        {"run_id": RUN_ID},
        ["run_id"],
    ),
    _schema(
        "run_meter_rubrics",
        "Use Temporal Guard, OpenCV and Qwen on the real video copy to generate binary R5/R6 evidence artifacts.",
        {"run_id": RUN_ID},
        ["run_id"],
    ),
    _schema(
        "run_record_rubrics",
        "Use cycle-bound Temporal Guard windows, OpenCV and Qwen on the real video copy to generate binary R7/R9 evidence artifacts.",
        {"run_id": RUN_ID},
        ["run_id"],
    ),
    _schema(
        "run_remaining_rubrics",
        "Use Temporal Guard windows, OpenCV and Qwen on the real video copy to generate binary R0/R2/R8 evidence artifacts.",
        {"run_id": RUN_ID},
        ["run_id"],
    ),
    _schema(
        "run_polarity_rubric",
        "Use v15 endpoint reasoning plus frame-bound pointer/reading observations on the real video copy to generate binary R4 evidence.",
        {"run_id": RUN_ID},
        ["run_id"],
    ),
    _schema(
        "run_rubric_bundle",
        "Run the minimum set of grouped evidence producers once each for the requested binary rubrics.",
        {
            "run_id": RUN_ID,
            "rubric_ids": {
                "type": "array",
                "items": {"type": "integer", "minimum": 0, "maximum": 9},
                "minItems": 1,
                "description": "Requested rubric IDs; duplicates are removed and grouped producers may co-produce related rubrics.",
            },
        },
        ["run_id", "rubric_ids"],
    ),
    _schema("load_rubric_result", "Load one frozen binary rubric result in explicit five-video replay mode.", {"run_id": RUN_ID, "rubric_id": {"type": "integer", "minimum": 0, "maximum": 9}}, ["run_id", "rubric_id"]),
    _schema(
        "load_rubric_bundle",
        "Load the requested frozen binary rubric results in one explicit five-video replay call.",
        {
            "run_id": RUN_ID,
            "rubric_ids": {
                "type": "array",
                "items": {"type": "integer", "minimum": 0, "maximum": 9},
                "minItems": 1,
            },
        },
        ["run_id", "rubric_ids"],
    ),
    _schema("inspect_run_status", "Return current artifacts and missing rubric results.", {"run_id": RUN_ID}, ["run_id"]),
    _schema("validate_run", "Validate that all ten results satisfy the pass/fail contract.", {"run_id": RUN_ID}, ["run_id"]),
    _schema("finalize_run", "Write the final result only after ten binary artifacts validate.", {"run_id": RUN_ID}, ["run_id"]),
]


def call_tool(name: str, arguments: dict[str, Any] | None) -> dict[str, Any]:
    function = TOOL_REGISTRY.get(name)
    if function is None:
        raise ToolError(f"unknown tool: {name}")
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        raise ToolError("tool arguments must be a JSON object")
    try:
        return redact(function(**arguments))
    except TypeError as exc:
        raise ToolError(f"invalid arguments for {name}: {exc}") from exc
