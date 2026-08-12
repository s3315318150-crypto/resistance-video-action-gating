#!/usr/bin/env python3
"""Build query-driven evidence retrieval plans from saved action-stage runs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILES = EXPERIMENT_ROOT / "configs" / "ten_rubric_retrieval_profiles_v1.json"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def normalize_runs(raw_runs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for index, item in enumerate(raw_runs, start=1):
        try:
            start = float(item["start_seconds"])
            end = float(item["end_seconds"])
        except (KeyError, TypeError, ValueError):
            continue
        stage = str(item.get("stage") or "")
        if not stage or end < start:
            continue
        runs.append(
            {
                "stage": stage,
                "start_seconds": start,
                "end_seconds": end,
                "run_id": str(item.get("run_id") or f"run_{index:03d}"),
                "confidence": item.get("confidence"),
                "event_ids": list(item.get("event_ids") or []),
            }
        )
    return sorted(runs, key=lambda item: (item["start_seconds"], item["end_seconds"], item["stage"]))


def clamp_interval(start: float, end: float, bounds: tuple[float, float]) -> tuple[float, float]:
    lower, upper = bounds
    return max(lower, min(start, upper)), max(lower, min(end, upper))


def make_window(
    *,
    role: str,
    start: float,
    end: float,
    interval: float,
    bounds: tuple[float, float],
    priority: int,
    source_stages: list[str],
    source_run_ids: list[str],
    reason: str,
) -> dict[str, Any] | None:
    start, end = clamp_interval(start, end, bounds)
    if end < start or interval <= 0:
        return None
    return {
        "role": role,
        "start_seconds": round(start, 6),
        "end_seconds": round(end, 6),
        "sample_interval_seconds": float(interval),
        "priority": priority,
        "source_stages": source_stages,
        "source_run_ids": source_run_ids,
        "reason": reason,
    }


def runs_for(runs: list[dict[str, Any]], stages: Iterable[str]) -> list[dict[str, Any]]:
    accepted = set(stages)
    return [item for item in runs if item["stage"] in accepted]


def measurement_anchor_present(runs: list[dict[str, Any]], profile: dict[str, Any]) -> bool:
    measurement = {"measurement_1", "measurement_2"}
    return bool(measurement.intersection(profile["anchor_stages"]) and runs_for(runs, measurement))


def fallback_enabled(
    mode: str,
    *,
    has_anchor: bool,
    has_stage_fallback: bool,
    has_measurement_anchor: bool,
) -> bool:
    return {
        "supplementary": True,
        "when_no_anchor": not has_anchor,
        "when_no_measurement_anchor": not has_measurement_anchor,
        "when_no_anchor_and_no_stage_fallback": not has_anchor and not has_stage_fallback,
    }.get(mode, False)


def fallback_windows(
    fallback: dict[str, Any],
    runs: list[dict[str, Any]],
    bounds: tuple[float, float],
) -> list[dict[str, Any]]:
    kind = fallback["type"]
    interval = float(fallback["interval_seconds"])
    windows: list[dict[str, Any]] = []
    if kind == "experiment_tail":
        duration = float(fallback["duration_seconds"])
        window = make_window(
            role="terminal_state_search",
            start=bounds[1] - duration,
            end=bounds[1],
            interval=interval,
            bounds=bounds,
            priority=3,
            source_stages=[],
            source_run_ids=[],
            reason="No cleanup stage was observed; inspect the bounded experiment tail.",
        )
        return [window] if window else []
    if kind == "experiment_sparse":
        window = make_window(
            role="global_sparse_fallback",
            start=bounds[0],
            end=bounds[1],
            interval=interval,
            bounds=bounds,
            priority=4,
            source_stages=[],
            source_run_ids=[],
            reason="No stage-local candidate was available; preserve sparse global coverage.",
        )
        return [window] if window else []
    if kind in {"before_stage", "after_stage"}:
        selected = runs_for(runs, fallback["stages"])
        for run in selected:
            if kind == "before_stage":
                start = run["start_seconds"] - float(fallback["lookback_seconds"])
                end = run["start_seconds"] + float(fallback["include_after_seconds"])
                role = "causal_precondition_search"
                reason = f"Search backward from {run['stage']} for the latest stable prerequisite state."
            else:
                start = run["end_seconds"] - float(fallback["before_end_seconds"])
                end = run["end_seconds"] + float(fallback["after_end_seconds"])
                role = "stable_post_action_state"
                reason = f"Inspect the stable state around the end of {run['stage']}."
            window = make_window(
                role=role,
                start=start,
                end=end,
                interval=interval,
                bounds=bounds,
                priority=2,
                source_stages=[run["stage"]],
                source_run_ids=[run["run_id"]],
                reason=reason,
            )
            if window:
                windows.append(window)
        return windows
    if kind == "between_stage_groups":
        left = runs_for(runs, fallback["left_stages"])
        right = runs_for(runs, fallback["right_stages"])
        padding = float(fallback["padding_seconds"])
        for left_run in left:
            candidate = next(
                (item for item in right if item["start_seconds"] >= left_run["end_seconds"]),
                None,
            )
            if candidate is None:
                continue
            window = make_window(
                role="inter_cycle_gap_search",
                start=left_run["end_seconds"] - padding,
                end=candidate["start_seconds"] + padding,
                interval=interval,
                bounds=bounds,
                priority=2,
                source_stages=[left_run["stage"], candidate["stage"]],
                source_run_ids=[left_run["run_id"], candidate["run_id"]],
                reason="Search the gap between first-cycle and second-cycle evidence for a short reconfiguration.",
            )
            if window:
                windows.append(window)
        return windows
    raise ValueError(f"unsupported_fallback_type:{kind}")


def raw_sample_times(window: dict[str, Any]) -> list[float]:
    start = float(window["start_seconds"])
    end = float(window["end_seconds"])
    interval = float(window["sample_interval_seconds"])
    if math.isclose(start, end):
        return [round(start, 6)]
    count = max(1, int(math.floor((end - start) / interval)))
    values = [round(start + index * interval, 6) for index in range(count + 1)]
    if values[-1] < end - 1e-6:
        values.append(round(end, 6))
    return sorted(set(values))


def uniform_pick(values: list[float], count: int) -> list[float]:
    if count >= len(values):
        return values
    if count <= 1:
        return [values[len(values) // 2]]
    indices = {
        round(index * (len(values) - 1) / (count - 1))
        for index in range(count)
    }
    return [values[index] for index in sorted(indices)]


def allocate_budget(windows: list[dict[str, Any]], budget: int) -> list[dict[str, Any]]:
    ordered = sorted(
        windows,
        key=lambda item: (
            item["priority"],
            item["start_seconds"],
            item["end_seconds"],
            item["role"],
        ),
    )
    if not ordered or budget <= 0:
        return []
    ordered = ordered[:budget]
    raw = [raw_sample_times(item) for item in ordered]
    counts = [1 for _ in ordered]
    remaining = budget - len(ordered)
    while remaining:
        progressed = False
        for index, values in enumerate(raw):
            if counts[index] < len(values) and remaining:
                counts[index] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break
    output: list[dict[str, Any]] = []
    for index, window in enumerate(ordered, start=1):
        selected = uniform_pick(raw[index - 1], counts[index - 1])
        payload = dict(window)
        payload["window_id"] = f"candidate_{index:03d}"
        payload["raw_frame_count"] = len(raw[index - 1])
        payload["planned_sample_times_seconds"] = selected
        payload["planned_frame_count"] = len(selected)
        output.append(payload)
    return output


def deduplicate_windows(windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for item in windows:
        key = (
            item["role"],
            item["start_seconds"],
            item["end_seconds"],
            item["sample_interval_seconds"],
            tuple(item["source_run_ids"]),
        )
        if key not in seen:
            seen.add(key)
            output.append(item)
    return output


def build_profile_plan(
    profile: dict[str, Any],
    runs: list[dict[str, Any]],
    bounds: tuple[float, float],
) -> dict[str, Any]:
    anchors = runs_for(runs, profile["anchor_stages"])
    before, after = [float(value) for value in profile["anchor_padding_seconds"]]
    windows: list[dict[str, Any]] = []
    for run in anchors:
        window = make_window(
            role="direct_stage_evidence",
            start=run["start_seconds"] - before,
            end=run["end_seconds"] + after,
            interval=float(profile["anchor_interval_seconds"]),
            bounds=bounds,
            priority=1,
            source_stages=[run["stage"]],
            source_run_ids=[run["run_id"]],
            reason=f"Direct evidence window anchored to observed stage {run['stage']}.",
        )
        if window:
            windows.append(window)
    stage_fallback_added = False
    for fallback in profile.get("fallbacks", []):
        enabled = fallback_enabled(
            fallback["mode"],
            has_anchor=bool(anchors),
            has_stage_fallback=stage_fallback_added,
            has_measurement_anchor=measurement_anchor_present(runs, profile),
        )
        if not enabled:
            continue
        added = fallback_windows(fallback, runs, bounds)
        if added and fallback["type"] != "experiment_sparse":
            stage_fallback_added = True
        windows.extend(added)
    windows = allocate_budget(deduplicate_windows(windows), int(profile["budget_frames"]))
    planned_count = sum(item["planned_frame_count"] for item in windows)
    observed_anchor_stages = sorted({item["stage"] for item in anchors})
    return {
        "rubric_id": int(profile["rubric_id"]),
        "key": profile["key"],
        "observation_question": profile["observation_question"],
        "roi_targets": list(profile["roi_targets"]),
        "budget_frames": int(profile["budget_frames"]),
        "planned_frame_count": planned_count,
        "budget_respected": planned_count <= int(profile["budget_frames"]),
        "observed_anchor_stages": observed_anchor_stages,
        "missing_anchor_stages": sorted(set(profile["anchor_stages"]) - set(observed_anchor_stages)),
        "fallback_used": any(item["role"] != "direct_stage_evidence" for item in windows),
        "candidate_windows": windows,
    }


def build_video_plan(record: dict[str, Any], profiles: dict[str, Any]) -> dict[str, Any]:
    runs = normalize_runs(record.get("observed_stage_runs") or [])
    if not runs and record.get("replay_result"):
        runs = normalize_runs(read_json(Path(record["replay_result"])) .get("observed_stage_runs") or [])
    if not runs:
        raise ValueError(f"no_observed_stage_runs:{record.get('source_video_id')}")
    bounds = (runs[0]["start_seconds"], max(item["end_seconds"] for item in runs))
    rubric_plans = [build_profile_plan(profile, runs, bounds) for profile in profiles["profiles"]]
    duration = max(0.0, bounds[1] - bounds[0])
    baseline_per_rubric = int(math.floor(duration / float(profiles["uniform_baseline_interval_seconds"]))) + 1
    baseline_total = baseline_per_rubric * len(rubric_plans)
    planned_total = sum(item["planned_frame_count"] for item in rubric_plans)
    return {
        "schema_version": "night_exploration.ten_rubric_retrieval_plan.v1",
        "source_video_id": record.get("source_video_id"),
        "experiment_interval_seconds": [bounds[0], bounds[1]],
        "observed_stage_runs": runs,
        "rubric_plans": rubric_plans,
        "comparison": {
            "uniform_full_interval_frames_for_ten_rubrics": baseline_total,
            "query_driven_planned_frames_for_ten_rubrics": planned_total,
            "frame_reduction_fraction": round(1.0 - planned_total / baseline_total, 6) if baseline_total else 0.0,
            "all_ten_rubrics_planned": len(rubric_plans) == 10,
            "all_budgets_respected": all(item["budget_respected"] for item in rubric_plans),
        },
    }


def markdown_summary(summary: dict[str, Any]) -> str:
    lines = [
        "# Ten-rubric retrieval planner v1 comparison",
        "",
        "| Video | Rubrics | Uniform frames | Planned frames | Reduction | Budgets |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for item in summary["records"]:
        comparison = item["comparison"]
        lines.append(
            f"| {item['source_video_id']} | {len(item['rubric_plans'])} | "
            f"{comparison['uniform_full_interval_frames_for_ten_rubrics']} | "
            f"{comparison['query_driven_planned_frames_for_ten_rubrics']} | "
            f"{comparison['frame_reduction_fraction']:.1%} | "
            f"{'pass' if comparison['all_budgets_respected'] else 'fail'} |"
        )
    lines.extend(
        [
            "",
            "This comparison measures retrieval cost and structural coverage only. It does not measure rubric accuracy and does not emit pass/fail decisions.",
        ]
    )
    return "\n".join(lines) + "\n"


def run(action_summary: Path, profiles_path: Path, output: Path) -> dict[str, Any]:
    source = read_json(action_summary)
    profiles = read_json(profiles_path)
    records = [build_video_plan(record, profiles) for record in source.get("records", [])]
    if not records:
        raise ValueError("action_summary_has_no_records")
    output.mkdir(parents=True, exist_ok=False)
    for index, item in enumerate(records, start=1):
        write_json(output / f"source_{index:03d}" / "retrieval_plan.json", item)
    summary = {
        "schema_version": "night_exploration.ten_rubric_retrieval_summary.v1",
        "algorithm_id": "ten_rubric_query_driven_retrieval_planner_v1",
        "source_action_summary": str(action_summary.resolve()),
        "profile_path": str(profiles_path.resolve()),
        "record_count": len(records),
        "records": records,
        "aggregate": {
            "all_records_have_ten_rubrics": all(len(item["rubric_plans"]) == 10 for item in records),
            "all_budgets_respected": all(item["comparison"]["all_budgets_respected"] for item in records),
            "uniform_frame_count": sum(item["comparison"]["uniform_full_interval_frames_for_ten_rubrics"] for item in records),
            "planned_frame_count": sum(item["comparison"]["query_driven_planned_frames_for_ten_rubrics"] for item in records),
        },
    }
    baseline = summary["aggregate"]["uniform_frame_count"]
    planned = summary["aggregate"]["planned_frame_count"]
    summary["aggregate"]["frame_reduction_fraction"] = round(1.0 - planned / baseline, 6) if baseline else 0.0
    write_json(output / "comparison.json", summary)
    (output / "comparison.md").write_text(markdown_summary(summary), encoding="utf-8", newline="\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action-summary", required=True, type=Path)
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run(args.action_summary.resolve(), args.profiles.resolve(), args.output.resolve())
    print(
        json.dumps(
            {
                "status": "completed",
                "video_count": result["record_count"],
                "rubrics_per_video": 10,
                "planned_frames": result["aggregate"]["planned_frame_count"],
                "frame_reduction_fraction": result["aggregate"]["frame_reduction_fraction"],
                "output": str(args.output.resolve()),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
