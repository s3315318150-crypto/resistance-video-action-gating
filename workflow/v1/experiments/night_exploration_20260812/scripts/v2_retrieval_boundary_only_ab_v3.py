#!/usr/bin/env python3
"""A/B candidate-frame retrieval for unchanged mature-v2 boundaries only.

Saved v2 Map events, Reduce output, state assignments, and stage runs are reused
verbatim. Retrieval may only choose the uniformly sampled frames sent to the
original v2 boundary prompt. This isolates retrieval quality from event and
state-machine changes.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_SCRIPT = SCRIPT_DIR / "v2_retrieval_only_ab_v1.py"
SPEC = importlib.util.spec_from_file_location("v2_retrieval_only_ab_v1_base_v3", BASE_SCRIPT)
base = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(base)


SCHEMA_VERSION = "night_exploration.v2_retrieval_boundary_only_ab.v3"
ALGORITHM_ID = "qwen_experiment_action_hierarchical_v2_retrieval_boundary_only_ab_v3"
METHODS = ("rubric_guided_boundary", "yes_no_boundary")
BOUNDARY_HALF_WIDTH_SECONDS = 3.0
BOUNDARY_SAMPLE_INTERVAL_SECONDS = 0.5


def candidate_boundaries(baseline: dict[str, Any]) -> list[dict[str, Any]]:
    intervals = baseline.get("observed_stage_intervals")
    if not isinstance(intervals, list):
        raise ValueError("baseline_observed_stage_intervals_missing")
    return [
        item
        for item in base.engine.build_boundary_candidates([dict(value) for value in intervals])
        if item.get("coarse_order_valid") is True
    ]


def golden_boundaries(gold_runs: list[list[Any]]) -> list[dict[str, Any]]:
    collapsed: list[list[Any]] = []
    for run in gold_runs:
        if collapsed and collapsed[-1][0] == run[0]:
            collapsed[-1][2] = run[2]
        else:
            collapsed.append(list(run))
    return [
        {
            "from_stage": left[0],
            "to_stage": right[0],
            "selected_seconds": float(right[1]),
        }
        for left, right in zip(collapsed, collapsed[1:])
    ]


def compare_boundaries(
    actual: list[dict[str, Any]],
    reference: list[dict[str, Any]],
) -> dict[str, Any]:
    by_transition: dict[tuple[str, str], list[float]] = {}
    for item in reference:
        key = (str(item["from_stage"]), str(item["to_stage"]))
        by_transition.setdefault(key, []).append(float(item["selected_seconds"]))
    comparisons: list[dict[str, Any]] = []
    for item in actual:
        key = (str(item["from_stage"]), str(item["to_stage"]))
        candidates = by_transition.get(key, [])
        selected = float(item["selected_seconds"])
        if not candidates:
            comparisons.append(
                {
                    "boundary_id": item["boundary_id"],
                    "transition": list(key),
                    "matched": False,
                }
            )
            continue
        reference_seconds = min(candidates, key=lambda value: abs(value - selected))
        comparisons.append(
            {
                "boundary_id": item["boundary_id"],
                "transition": list(key),
                "matched": True,
                "selected_seconds": selected,
                "reference_seconds": reference_seconds,
                "absolute_error_seconds": abs(selected - reference_seconds),
                "within_2_seconds": abs(selected - reference_seconds) <= 2.0,
            }
        )
    matched = [item for item in comparisons if item["matched"]]
    return {
        "boundary_count": len(actual),
        "matched_boundary_count": len(matched),
        "within_2_seconds_count": sum(item["within_2_seconds"] for item in matched),
        "mean_absolute_error_seconds": (
            sum(item["absolute_error_seconds"] for item in matched) / len(matched)
            if matched
            else None
        ),
        "comparisons": comparisons,
    }


def relevant_rubric_intervals(
    baseline: dict[str, Any],
    profiles: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    intervals, plan = base.rubric_intervals(baseline, profiles)
    return intervals, plan


def _clip_bounds(
    prepared: dict[str, Any],
    center: float,
    half_width: float = BOUNDARY_HALF_WIDTH_SECONDS,
) -> tuple[float, float]:
    start = max(float(prepared["fixed_start"]), center - half_width)
    end = min(float(prepared["fixed_end"]), center + half_width)
    return start, end


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
    start, end = _clip_bounds(prepared, selected_center)
    return start, end, {
        "coarse_center_seconds": center,
        "selected_center_seconds": selected_center,
        "touching_interval_count": len(touching),
        "candidate_time_count": len(planned),
        "fallback_to_coarse_center": not planned,
    }


def boundary_selector_prompt(
    boundary: dict[str, Any],
    clips: list[dict[str, Any]],
    labels: dict[str, str],
) -> str:
    mapping = "\n".join(
        f"- {clip['clip_id']}: {', '.join(frame['image_id'] for frame in clip['frames'])}"
        for clip in clips
    )
    from_stage = str(boundary["from_stage"])
    to_stage = str(boundary["to_stage"])
    return f"""这些图片来自同一个伏安法测电阻视频，分属于边界附近按时间排列且互不重叠的候选片段。

目标转折：从 `{from_stage}`（{labels[from_stage]}）转为 `{to_stage}`（{labels[to_stage]}）。

片段及合法 FRAME ID：
{mapping}

你只负责选择后续交给正式 v2 边界提示词复核的片段，不负责输出最终边界或阶段。对每个片段判断上述两种状态的可见转折是否可能出现在片段内。只依据可见像素，不按实验流程猜测，不输出秒数。

只输出 JSON：
{{"clips":[{{"clip_id":"...","answer":"yes"|"no","target_probability":0.0,"selected_frame_id":"..."|null,"evidence":"..."}}]}}"""


def select_yes_no_boundary_range(
    prepared: dict[str, Any],
    boundary: dict[str, Any],
    labels: dict[str, str],
    client: Any | None,
    args: argparse.Namespace,
) -> tuple[float, float, dict[str, Any]]:
    center = float(boundary["coarse_selected_seconds"])
    context_start = max(float(prepared["fixed_start"]), center - 10.0)
    context_end = min(float(prepared["fixed_end"]), center + 10.0)
    raw_clips = base.binary_support.partition(
        context_start,
        context_end,
        4,
        f"sel_{boundary['boundary_id']}",
    )
    clips: list[dict[str, Any]] = []
    for raw in raw_clips:
        clip = dict(raw)
        clip["frames"] = base.frames_for_times(
            prepared,
            [float(value) for value in clip["sample_seconds"]],
            args.max_model_edge,
        )
        clips.append(clip)
    prompt = boundary_selector_prompt(boundary, clips, labels)
    selector_dir = prepared["video_dir"] / "retrieval" / str(boundary["boundary_id"])
    selector_dir.mkdir(parents=True, exist_ok=True)
    base.engine._write_text(selector_dir / "prompt.txt", prompt)
    base.write_json(selector_dir / "input.json", {"boundary": boundary, "clips": clips})
    if client is None:
        response = {"status": "prepared"}
        selected_clip = min(
            clips,
            key=lambda clip: abs((float(clip["start_seconds"]) + float(clip["end_seconds"])) / 2.0 - center),
        )
    else:
        all_frames = [frame for clip in clips for frame in clip["frames"]]
        response = base.binary_support.QwenClient.call_json(
            client,
            prompt,
            all_frames,
            lambda value, c=clips: base.validate_selector_response(value, c),
        )
        if response.get("status") == "valid":
            scores = {
                str(row["clip_id"]): float(row["target_probability"])
                for row in response["result"]["clips"]
            }
            selected_clip = min(
                clips,
                key=lambda clip: (
                    -scores[str(clip["clip_id"])],
                    abs((float(clip["start_seconds"]) + float(clip["end_seconds"])) / 2.0 - center),
                ),
            )
        else:
            selected_clip = min(
                clips,
                key=lambda clip: abs((float(clip["start_seconds"]) + float(clip["end_seconds"])) / 2.0 - center),
            )
    selected_center = (
        float(selected_clip["start_seconds"]) + float(selected_clip["end_seconds"])
    ) / 2.0
    start, end = _clip_bounds(prepared, selected_center)
    trace = {
        "coarse_center_seconds": center,
        "selected_center_seconds": selected_center,
        "selected_clip_id": selected_clip["clip_id"],
        "qwen": response,
    }
    base.write_json(selector_dir / "result.json", trace)
    return start, end, trace


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
    times = base.contract.sample_timestamps(
        range_start,
        range_end,
        BOUNDARY_SAMPLE_INTERVAL_SECONDS,
    )
    frames = base.frames_for_times(prepared, times, args.max_model_edge)
    prompt = base.mature_prompts.build_boundary_prompt(
        prepared["video_id"], boundary, frames, labels
    )
    output_dir = prepared["video_dir"] / "boundaries" / str(boundary["boundary_id"])
    output_dir.mkdir(parents=True, exist_ok=True)
    base.engine._write_text(output_dir / "prompt.txt", prompt)
    base.write_json(
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
        base.write_json(output_dir / "result.json", result)
        return result
    attempts: list[dict[str, Any]] = []
    parsed: dict[str, Any] | None = None
    errors: list[str] = []
    for attempt_index in range(args.max_attempts):
        attempt_prompt = (
            prompt
            if attempt_index == 0
            else base.mature_prompts.build_boundary_retry_prompt(prompt, errors)
        )
        raw = base.engine._attempt_qwen(
            client,
            attempt_prompt,
            frames,
            args.boundary_max_tokens,
        )
        candidate = raw.get("parsed_result")
        parsed = candidate if isinstance(candidate, dict) else None
        errors = base.contract.validate_boundary_response(
            parsed,
            str(boundary["boundary_id"]),
            frames,
        )
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
    base.write_json(output_dir / "result.json", result)
    return result


def fallback_boundary(boundary: dict[str, Any], saved: dict[str, Any] | None) -> dict[str, Any]:
    if isinstance(saved, dict):
        return {**saved, "source": "saved_v2_boundary_fallback"}
    return {
        **boundary,
        "last_from_frame_id": boundary["coarse_last_from_frame_id"],
        "first_to_frame_id": boundary["coarse_first_to_frame_id"],
        "last_from_seconds": boundary["coarse_last_from_seconds"],
        "first_to_seconds": boundary["coarse_first_to_seconds"],
        "boundary_interval_seconds": [
            boundary["coarse_last_from_seconds"],
            boundary["coarse_first_to_seconds"],
        ],
        "selected_seconds": boundary["coarse_first_to_seconds"],
        "sampling_interval_seconds": 2.0,
        "confidence": None,
        "evidence": "候选帧复核无合法结果，保留原 Map 粗边界。",
        "uncertainty": "boundary_retrieval_failed",
        "needs_review": True,
        "source": "coarse_map_fallback",
    }


def run_method(
    method: str,
    item: dict[str, Any],
    output_root: Path,
    profiles: dict[str, Any],
    gold_runs: list[list[Any]],
    client: Any | None,
    labels: dict[str, str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    baseline = item["baseline"]
    prepared = base.prepare_shell(
        baseline,
        output_root / base.binary_support.safe_slug(item["source_video_id"]),
    )
    boundaries = candidate_boundaries(baseline)
    saved_by_id = {
        str(value["boundary_id"]): value
        for value in baseline.get("boundaries", [])
        if isinstance(value, dict) and isinstance(value.get("boundary_id"), str)
    }
    rubric_intervals, rubric_plan = relevant_rubric_intervals(baseline, profiles)
    refined: list[dict[str, Any]] = []
    retrieval_traces: list[dict[str, Any]] = []
    for boundary in boundaries:
        if method == "rubric_guided_boundary":
            start, end, retrieval = rubric_boundary_range(
                prepared, boundary, rubric_intervals
            )
        elif method == "yes_no_boundary":
            start, end, retrieval = select_yes_no_boundary_range(
                prepared, boundary, labels, client, args
            )
        else:
            raise ValueError(f"unknown_method:{method}")
        boundary_pass = run_original_boundary_prompt(
            prepared,
            boundary,
            start,
            end,
            labels,
            client,
            args,
            method,
        )
        observed = base.engine._observed_boundary_from_pass(
            boundary_pass,
            args.boundary_min_confidence,
        )
        chosen = (
            {**boundary, **observed, "source": f"{method}_original_v2_boundary_prompt"}
            if observed is not None
            else fallback_boundary(boundary, saved_by_id.get(str(boundary["boundary_id"])))
        )
        refined.append(chosen)
        retrieval_traces.append(
            {
                "boundary_id": boundary["boundary_id"],
                "range_seconds": [start, end],
                "retrieval": retrieval,
                "boundary_prompt_valid": boundary_pass["valid"],
                "result_source": chosen["source"],
            }
        )
    refined, rejected = base.engine._enforce_boundary_monotonicity(refined)
    result_payload = {
        "schema_version": SCHEMA_VERSION,
        "algorithm_id": ALGORITHM_ID,
        "source_video_id": item["source_video_id"],
        "map_reduce_state_source": str(item["source_result"]),
        "observed_stage_intervals": baseline.get("observed_stage_intervals", []),
        "observed_stage_runs": baseline.get("observed_stage_runs", []),
        "boundaries": refined,
        "rejected_boundaries": rejected,
        "retrieval_traces": retrieval_traces,
        "rubric_plan": rubric_plan if method == "rubric_guided_boundary" else None,
        "stage_runs_reused_verbatim": True,
    }
    base.write_json(prepared["video_dir"] / "result.json", result_payload)
    return {
        "status": "prepared" if client is None else "completed",
        "source_video_id": item["source_video_id"],
        "result_path": str((prepared["video_dir"] / "result.json").resolve()),
        "observed_stage_runs": baseline.get("observed_stage_runs", []),
        "boundaries": refined,
        "rejected_boundaries": rejected,
        "retrieval_traces": retrieval_traces,
        "vs_saved_v2_boundaries": compare_boundaries(
            refined,
            [value for value in baseline.get("boundaries", []) if isinstance(value, dict)],
        ),
        "vs_golden_boundaries": compare_boundaries(refined, golden_boundaries(gold_runs)),
    }


def aggregate(records: list[dict[str, Any]], method: str) -> dict[str, Any]:
    methods = [record["methods"][method] for record in records]
    matched_saved = sum(item["vs_saved_v2_boundaries"]["matched_boundary_count"] for item in methods)
    matched_gold = sum(item["vs_golden_boundaries"]["matched_boundary_count"] for item in methods)
    return {
        "method": method,
        "video_count": len(methods),
        "stage_runs_identical_to_saved_v2_count": sum(
            item["vs_saved_v2"]["within_2_seconds"] for item in methods
        ),
        "saved_boundary_matched_count": matched_saved,
        "saved_boundary_within_2_seconds_count": sum(
            item["vs_saved_v2_boundaries"]["within_2_seconds_count"] for item in methods
        ),
        "gold_boundary_matched_count": matched_gold,
        "gold_boundary_within_2_seconds_count": sum(
            item["vs_golden_boundaries"]["within_2_seconds_count"] for item in methods
        ),
        "gold_boundary_mean_absolute_error_seconds": (
            sum(
                comparison["absolute_error_seconds"]
                for item in methods
                for comparison in item["vs_golden_boundaries"]["comparisons"]
                if comparison["matched"]
            )
            / matched_gold
            if matched_gold
            else None
        ),
    }


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# v2 boundary-only retrieval A/B",
        "",
        "Saved v2 Map, Reduce, state assignments, and stage runs are reused verbatim.",
        "Only frames sent to the original v2 boundary prompt are retrieved differently.",
        "",
        "| Method | Stage runs identical | Saved boundaries within 2s | Golden boundaries within 2s | Golden boundary MAE |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in result.get("aggregate", []):
        mae = item["gold_boundary_mean_absolute_error_seconds"]
        lines.append(
            f"| {item['method']} | {item['stage_runs_identical_to_saved_v2_count']}/{item['video_count']} | "
            f"{item['saved_boundary_within_2_seconds_count']}/{item['saved_boundary_matched_count']} | "
            f"{item['gold_boundary_within_2_seconds_count']}/{item['gold_boundary_matched_count']} | "
            f"{mae:.3f}s |" if mae is not None else "n/a |"
        )
    lines.extend(
        [
            "",
            f"Qwen calls: {result['qwen_usage']['call_count']}; image exposures: {result['qwen_usage']['image_exposures']}.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    base.bind_mature_v2_pipeline()
    schema = base.contract.load_stage_schema(base.v2_entrypoint.DEFAULT_SCHEMA)
    labels = {str(item["id"]): str(item["label_zh"]) for item in schema["stages"]}
    sources = base.baseline_sources(base.read_json(args.baseline_summary))
    profiles = base.read_json(args.profiles)
    gold = base.gold_by_video(base.read_json(args.gold))
    client = None
    if not args.prepare_only:
        delegate = base.qwen_base.OpenAI(
            base_url=args.endpoint,
            api_key=args.token,
            timeout=args.timeout,
            max_retries=0,
        )
        client = base.CountingClient(delegate)
        client.endpoint = args.endpoint.rstrip("/") + "/chat/completions"
        client.token = args.token
        client.model = args.model
        client.timeout = args.timeout
    engine_args = base.default_engine_args()
    records: list[dict[str, Any]] = []
    for index, item in enumerate(sources, start=1):
        video_id = item["source_video_id"]
        record: dict[str, Any] = {
            "source_video_id": video_id,
            "gold_sent_to_qwen": False,
            "methods": {"baseline_v2": base.baseline_record(item)},
        }
        before_calls = client.call_count if client else 0
        before_images = client.image_exposures if client else 0
        for method in METHODS:
            method_before_calls = client.call_count if client else 0
            method_before_images = client.image_exposures if client else 0
            value = run_method(
                method,
                item,
                args.output / method,
                profiles,
                gold[video_id],
                client,
                labels,
                engine_args,
            )
            value["qwen_calls"] = (client.call_count - method_before_calls) if client else 0
            value["qwen_image_exposures"] = (
                client.image_exposures - method_before_images
            ) if client else 0
            record["methods"][method] = value
        base.attach_comparisons(record, gold[video_id])
        record["qwen_calls"] = (client.call_count - before_calls) if client else 0
        record["qwen_image_exposures"] = (client.image_exposures - before_images) if client else 0
        records.append(record)
        base.write_json(args.output / "records" / f"source_{index:03d}.json", record)
        print(json.dumps({"video": index, "total": len(sources), "source_video_id": video_id}, ensure_ascii=False), flush=True)
    result = {
        "schema_version": SCHEMA_VERSION,
        "algorithm_id": ALGORITHM_ID,
        "status": "prepared" if args.prepare_only else "completed",
        "invariants": {
            "saved_map_events_reused": True,
            "saved_reduce_reused": True,
            "saved_state_assignments_reused": True,
            "saved_stage_runs_reused_verbatim": True,
            "original_v2_boundary_prompt_imported_unchanged": True,
            "gold_sent_to_qwen": False,
            "source_bindings": base.source_bindings(),
        },
        "records": records,
        "aggregate": [aggregate(records, method) for method in METHODS] if not args.prepare_only else [],
        "qwen_usage": {
            "call_count": client.call_count if client else 0,
            "image_exposures": client.image_exposures if client else 0,
        },
    }
    base.write_json(args.output / "comparison.json", result)
    write_markdown(args.output / "comparison.md", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-summary", required=True, type=Path)
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument(
        "--profiles",
        type=Path,
        default=base.REPO_ROOT / "experiments" / "night_exploration_20260812" / "configs" / "ten_rubric_retrieval_profiles_v1.json",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--endpoint", default="https://cossin.ecnu.edu.cn/skill/api/qwen/v1")
    parser.add_argument("--token", default="EMPTY")
    parser.add_argument("--model", default="qwen")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    for field in ("baseline_summary", "gold", "profiles", "output"):
        setattr(args, field, getattr(args, field).resolve())
    result = run(args)
    print(json.dumps({"status": result["status"], "output": str(args.output), "qwen_calls": result["qwen_usage"]["call_count"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
