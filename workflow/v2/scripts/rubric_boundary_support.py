#!/usr/bin/env python3
"""Current-run support for Rubric-guided v2 boundary refinement.

This module intentionally contains no replay, golden-label, or video-specific
fallback path. It derives every review interval from the stage runs and profile
configuration supplied by the current action-segmentation run.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import qwen_experiment_action_hierarchical_v1 as engine
import qwen_experiment_action_hierarchical_v2 as v2_entrypoint
import qwen_experiment_segment_judge as qwen_base
import qwen_hierarchical_v1_contract as contract
import qwen_hierarchical_v1_prompts as mature_prompts
import qwen_hierarchical_v1_reduce as mature_reduce


SELECTED_RUBRICS = {0, 3, 5, 7, 9}
BOUNDARY_HALF_WIDTH_SECONDS = 3.0
BOUNDARY_SAMPLE_INTERVAL_SECONDS = 0.5


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"json_root_not_object:{path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


class CountingClient:
    """Count requests while preserving the OpenAI-compatible client surface."""

    def __init__(self, client: Any) -> None:
        self.delegate = client
        self.call_count = 0
        self.image_exposures = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs: Any) -> Any:
        self.call_count += 1
        for message in kwargs.get("messages") or []:
            content = message.get("content") if isinstance(message, dict) else None
            if isinstance(content, list):
                self.image_exposures += sum(
                    isinstance(item, dict) and item.get("type") == "image_url"
                    for item in content
                )
        return self.delegate.chat.completions.create(**kwargs)


def default_engine_args() -> argparse.Namespace:
    return argparse.Namespace(
        max_model_edge=640,
        max_attempts=2,
        boundary_max_tokens=1200,
        boundary_min_confidence=0.72,
    )


def bind_mature_v2_pipeline() -> None:
    v2_entrypoint.bind_v2_identity()
    engine.assign_seven_stages = mature_reduce.assign_seven_stages


def prepare_shell(baseline: dict[str, Any], video_dir: Path) -> dict[str, Any]:
    manifest_path = Path(str(baseline["source_manifest"]))
    manifest = read_json(manifest_path)
    source, fps, frame_count = engine._video_metadata(manifest)
    fixed_start, fixed_end = map(float, baseline["locked_experiment_interval_seconds"])
    source_record = {
        "algorithm_id": "v2-rubric-boundary-current-run-support",
        "stage_schema_id": baseline.get("stage_schema_id", v2_entrypoint.STAGE_SCHEMA_ID),
        "source_video_id": baseline["source_video_id"],
        "source_video": str(source.resolve()),
        "source_manifest": str(manifest_path.resolve()),
        "source_segment_provenance": baseline["source_segment_provenance"],
        "locked_experiment_interval_seconds": [fixed_start, fixed_end],
        "selection_basis": "current_video_observed_situation_only",
        "historical_artifacts_used": False,
        "fixed_video_roi_used": False,
    }
    video_dir.mkdir(parents=True, exist_ok=False)
    write_json(video_dir / "source.json", source_record)
    return {
        "video_id": str(baseline["source_video_id"]),
        "manifest": manifest,
        "manifest_path": manifest_path,
        "video_dir": video_dir,
        "frames_dir": video_dir / "frames" / "source",
        "frame_registry": {},
        "prepared_windows": [],
        "source_record": source_record,
        "fixed_start": fixed_start,
        "fixed_end": fixed_end,
        "fps": fps,
        "frame_count": frame_count,
    }


def frames_for_times(
    prepared: dict[str, Any], times: list[float], max_model_edge: int
) -> list[dict[str, Any]]:
    numbers: list[int] = []
    seen: set[int] = set()
    for timestamp in times:
        number = min(
            prepared["frame_count"] - 1,
            max(0, int(round(float(timestamp) * prepared["fps"]))),
        )
        if number not in seen:
            numbers.append(number)
            seen.add(number)
    engine._extract_source_frames(
        prepared["manifest"],
        sorted(numbers),
        prepared["frames_dir"],
        max_model_edge,
        prepared["frame_registry"],
    )
    return [prepared["frame_registry"][number] for number in sorted(numbers)]


def candidate_boundaries(baseline: dict[str, Any]) -> list[dict[str, Any]]:
    intervals = baseline.get("observed_stage_intervals")
    if not isinstance(intervals, list):
        raise ValueError("baseline_observed_stage_intervals_missing")
    return [
        item
        for item in engine.build_boundary_candidates([dict(value) for value in intervals])
        if item.get("coarse_order_valid") is True
    ]


def _normalize_runs(raw_runs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for index, item in enumerate(raw_runs, start=1):
        try:
            start = float(item["start_seconds"])
            end = float(item["end_seconds"])
        except (KeyError, TypeError, ValueError):
            continue
        stage = str(item.get("stage") or "")
        if stage and end >= start:
            runs.append(
                {
                    "stage": stage,
                    "start_seconds": start,
                    "end_seconds": end,
                    "run_id": str(item.get("run_id") or f"run_{index:03d}"),
                }
            )
    return sorted(runs, key=lambda item: (item["start_seconds"], item["end_seconds"], item["stage"]))


def _runs_for(runs: list[dict[str, Any]], stages: Iterable[str]) -> list[dict[str, Any]]:
    accepted = set(stages)
    return [item for item in runs if item["stage"] in accepted]


def _make_window(
    *,
    role: str,
    start: float,
    end: float,
    interval: float,
    bounds: tuple[float, float],
    priority: int,
    source_stages: list[str],
    source_run_ids: list[str],
) -> dict[str, Any] | None:
    start = max(bounds[0], min(start, bounds[1]))
    end = max(bounds[0], min(end, bounds[1]))
    if end < start or interval <= 0:
        return None
    return {
        "role": role,
        "start_seconds": round(start, 6),
        "end_seconds": round(end, 6),
        "sample_interval_seconds": interval,
        "priority": priority,
        "source_stages": source_stages,
        "source_run_ids": source_run_ids,
    }


def _fallback_windows(
    fallback: dict[str, Any],
    runs: list[dict[str, Any]],
    bounds: tuple[float, float],
) -> list[dict[str, Any]]:
    kind = str(fallback["type"])
    interval = float(fallback["interval_seconds"])
    windows: list[dict[str, Any]] = []
    if kind == "experiment_tail":
        window = _make_window(
            role="terminal_state_search",
            start=bounds[1] - float(fallback["duration_seconds"]),
            end=bounds[1],
            interval=interval,
            bounds=bounds,
            priority=3,
            source_stages=[],
            source_run_ids=[],
        )
        return [window] if window else []
    if kind == "experiment_sparse":
        window = _make_window(
            role="global_sparse_fallback",
            start=bounds[0],
            end=bounds[1],
            interval=interval,
            bounds=bounds,
            priority=4,
            source_stages=[],
            source_run_ids=[],
        )
        return [window] if window else []
    if kind in {"before_stage", "after_stage"}:
        for run in _runs_for(runs, fallback["stages"]):
            if kind == "before_stage":
                start = run["start_seconds"] - float(fallback["lookback_seconds"])
                end = run["start_seconds"] + float(fallback["include_after_seconds"])
                role = "causal_precondition_search"
            else:
                start = run["end_seconds"] - float(fallback["before_end_seconds"])
                end = run["end_seconds"] + float(fallback["after_end_seconds"])
                role = "stable_post_action_state"
            window = _make_window(
                role=role,
                start=start,
                end=end,
                interval=interval,
                bounds=bounds,
                priority=2,
                source_stages=[run["stage"]],
                source_run_ids=[run["run_id"]],
            )
            if window:
                windows.append(window)
        return windows
    if kind == "between_stage_groups":
        left = _runs_for(runs, fallback["left_stages"])
        right = _runs_for(runs, fallback["right_stages"])
        padding = float(fallback["padding_seconds"])
        for left_run in left:
            following = next(
                (item for item in right if item["start_seconds"] >= left_run["end_seconds"]),
                None,
            )
            if following is None:
                continue
            window = _make_window(
                role="inter_cycle_gap_search",
                start=left_run["end_seconds"] - padding,
                end=following["start_seconds"] + padding,
                interval=interval,
                bounds=bounds,
                priority=2,
                source_stages=[left_run["stage"], following["stage"]],
                source_run_ids=[left_run["run_id"], following["run_id"]],
            )
            if window:
                windows.append(window)
        return windows
    raise ValueError(f"unsupported_fallback_type:{kind}")


def _sample_times(window: dict[str, Any]) -> list[float]:
    start = float(window["start_seconds"])
    end = float(window["end_seconds"])
    interval = float(window["sample_interval_seconds"])
    if math.isclose(start, end):
        return [round(start, 6)]
    values = [round(start + index * interval, 6) for index in range(int((end - start) // interval) + 1)]
    if values[-1] < end - 1e-6:
        values.append(round(end, 6))
    return sorted(set(values))


def _uniform_pick(values: list[float], count: int) -> list[float]:
    if count >= len(values):
        return values
    if count <= 1:
        return [values[len(values) // 2]]
    indices = {round(index * (len(values) - 1) / (count - 1)) for index in range(count)}
    return [values[index] for index in sorted(indices)]


def _allocate_budget(windows: list[dict[str, Any]], budget: int) -> list[dict[str, Any]]:
    unique: dict[tuple[Any, ...], dict[str, Any]] = {}
    for item in windows:
        key = (
            item["role"],
            item["start_seconds"],
            item["end_seconds"],
            item["sample_interval_seconds"],
            tuple(item["source_run_ids"]),
        )
        unique.setdefault(key, item)
    ordered = sorted(
        unique.values(),
        key=lambda item: (item["priority"], item["start_seconds"], item["end_seconds"], item["role"]),
    )[:budget]
    raw = [_sample_times(item) for item in ordered]
    counts = [1] * len(ordered)
    remaining = budget - len(ordered)
    while remaining > 0:
        progressed = False
        for index, values in enumerate(raw):
            if remaining and counts[index] < len(values):
                counts[index] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break
    output: list[dict[str, Any]] = []
    for index, item in enumerate(ordered, start=1):
        payload = dict(item)
        payload["window_id"] = f"candidate_{index:03d}"
        payload["planned_sample_times_seconds"] = _uniform_pick(raw[index - 1], counts[index - 1])
        payload["planned_frame_count"] = len(payload["planned_sample_times_seconds"])
        output.append(payload)
    return output


def _profile_plan(
    profile: dict[str, Any],
    runs: list[dict[str, Any]],
    bounds: tuple[float, float],
) -> dict[str, Any]:
    anchors = _runs_for(runs, profile["anchor_stages"])
    before, after = map(float, profile["anchor_padding_seconds"])
    windows: list[dict[str, Any]] = []
    for run in anchors:
        window = _make_window(
            role="direct_stage_evidence",
            start=run["start_seconds"] - before,
            end=run["end_seconds"] + after,
            interval=float(profile["anchor_interval_seconds"]),
            bounds=bounds,
            priority=1,
            source_stages=[run["stage"]],
            source_run_ids=[run["run_id"]],
        )
        if window:
            windows.append(window)
    stage_fallback_added = False
    measurement_anchor = bool(
        {"measurement_1", "measurement_2"}.intersection(profile["anchor_stages"])
        and _runs_for(runs, {"measurement_1", "measurement_2"})
    )
    for fallback in profile.get("fallbacks", []):
        mode = fallback["mode"]
        enabled = (
            mode == "supplementary"
            or (mode == "when_no_anchor" and not anchors)
            or (mode == "when_no_measurement_anchor" and not measurement_anchor)
            or (
                mode == "when_no_anchor_and_no_stage_fallback"
                and not anchors
                and not stage_fallback_added
            )
        )
        if not enabled:
            continue
        added = _fallback_windows(fallback, runs, bounds)
        if added and fallback["type"] != "experiment_sparse":
            stage_fallback_added = True
        windows.extend(added)
    windows = _allocate_budget(windows, int(profile["budget_frames"]))
    return {
        "rubric_id": int(profile["rubric_id"]),
        "key": profile["key"],
        "budget_frames": int(profile["budget_frames"]),
        "planned_frame_count": sum(item["planned_frame_count"] for item in windows),
        "candidate_windows": windows,
    }


def relevant_rubric_intervals(
    baseline: dict[str, Any], profiles: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    runs = _normalize_runs(baseline.get("observed_stage_runs") or [])
    if not runs:
        raise ValueError("no_current_run_observed_stage_runs")
    bounds = (runs[0]["start_seconds"], max(item["end_seconds"] for item in runs))
    plans = [_profile_plan(profile, runs, bounds) for profile in profiles["profiles"]]
    intervals = [
        {
            "source": f"rubric_{plan['rubric_id']}",
            "start_seconds": float(window["start_seconds"]),
            "end_seconds": float(window["end_seconds"]),
            "planned_times": [float(value) for value in window["planned_sample_times_seconds"]],
        }
        for plan in plans
        if plan["rubric_id"] in SELECTED_RUBRICS
        for window in plan["candidate_windows"]
    ]
    return intervals, {
        "schema_version": "ten_rubric_retrieval_plan.current_run.v1",
        "selection_basis": "current_video_observed_situation_only",
        "observed_stage_runs": runs,
        "rubric_plans": plans,
        "video_id_used_for_routing": False,
        "historical_artifacts_used": False,
        "fixed_video_roi_used": False,
    }


def rubric_boundary_range(
    prepared: dict[str, Any],
    boundary: dict[str, Any],
    intervals: list[dict[str, Any]],
) -> tuple[float, float, dict[str, Any]]:
    center = float(boundary["coarse_selected_seconds"])
    context_start = max(float(prepared["fixed_start"]), center - 10.0)
    context_end = min(float(prepared["fixed_end"]), center + 10.0)
    touching = [
        interval
        for interval in intervals
        if float(interval["end_seconds"]) >= context_start
        and float(interval["start_seconds"]) <= context_end
    ]
    planned = [
        float(value)
        for interval in touching
        for value in interval.get("planned_times", [])
        if context_start <= float(value) <= context_end
    ]
    selected_center = min(planned, key=lambda value: abs(value - center)) if planned else center
    start = max(float(prepared["fixed_start"]), selected_center - BOUNDARY_HALF_WIDTH_SECONDS)
    end = min(float(prepared["fixed_end"]), selected_center + BOUNDARY_HALF_WIDTH_SECONDS)
    return start, end, {
        "coarse_center_seconds": center,
        "selected_center_seconds": selected_center,
        "touching_interval_count": len(touching),
        "candidate_time_count": len(planned),
        "fallback_to_coarse_center": not planned,
    }


def run_original_boundary_prompt(
    prepared: dict[str, Any],
    boundary: dict[str, Any],
    range_start: float,
    range_end: float,
    labels: dict[str, str],
    client: Any | None,
    args: argparse.Namespace,
    method: str,
) -> dict[str, Any]:
    times = contract.sample_timestamps(
        range_start, range_end, BOUNDARY_SAMPLE_INTERVAL_SECONDS
    )
    frames = frames_for_times(prepared, times, args.max_model_edge)
    prompt = mature_prompts.build_boundary_prompt("anonymous", boundary, frames, labels)
    output_dir = prepared["video_dir"] / "boundaries" / str(boundary["boundary_id"])
    output_dir.mkdir(parents=True, exist_ok=True)
    engine._write_text(output_dir / "prompt.txt", prompt)
    write_json(
        output_dir / "input.json",
        {
            "boundary": boundary,
            "range_seconds": [range_start, range_end],
            "sample_interval_seconds": BOUNDARY_SAMPLE_INTERVAL_SECONDS,
            "input_frames": frames,
            "retrieval_method": method,
        },
    )
    if client is None:
        result = {
            "pass_id": method,
            "range_seconds": [range_start, range_end],
            "sample_interval_seconds": BOUNDARY_SAMPLE_INTERVAL_SECONDS,
            "input_frames": frames,
            "valid": False,
            "validation_errors": ["prepare_only"],
            "attempts": [],
            "parsed_result": None,
        }
        write_json(output_dir / "result.json", result)
        return result
    attempts: list[dict[str, Any]] = []
    parsed: dict[str, Any] | None = None
    errors: list[str] = []
    for attempt_index in range(args.max_attempts):
        attempt_prompt = (
            prompt
            if attempt_index == 0
            else mature_prompts.build_boundary_retry_prompt(prompt, errors)
        )
        raw = engine._attempt_qwen(client, attempt_prompt, frames, args.boundary_max_tokens)
        candidate = raw.get("parsed_result")
        parsed = candidate if isinstance(candidate, dict) else None
        errors = contract.validate_boundary_response(parsed, str(boundary["boundary_id"]), frames)
        attempts.append(
            {
                "attempt_index": attempt_index + 1,
                "qwen": raw,
                "validation_errors": errors,
            }
        )
        if not errors:
            break
    result = {
        "pass_id": method,
        "range_seconds": [range_start, range_end],
        "sample_interval_seconds": BOUNDARY_SAMPLE_INTERVAL_SECONDS,
        "input_frames": frames,
        "valid": not errors,
        "validation_errors": errors,
        "attempts": attempts,
        "parsed_result": parsed,
    }
    write_json(output_dir / "result.json", result)
    return result
