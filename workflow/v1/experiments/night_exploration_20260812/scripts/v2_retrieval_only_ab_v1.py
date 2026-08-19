#!/usr/bin/env python3
"""A/B supplemental retrieval while executing the unchanged mature v2 logic.

A replays saved v2 results. B adds rubric-guided windows. C uses Yes/No only
to choose windows. Every supplemental window is classified by the original v2
Map prompt; combined events then pass through the original v2 Reduce, state
machine, and boundary refinement functions. Gold is used only after inference.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_SCRIPTS = REPO_ROOT / "scripts"
EXPERIMENT_SCRIPTS = Path(__file__).resolve().parent
for directory in (PROJECT_SCRIPTS, EXPERIMENT_SCRIPTS):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import qwen_experiment_action_hierarchical_v1 as engine
import qwen_experiment_action_hierarchical_v2 as v2_entrypoint
import qwen_experiment_segment_judge as qwen_base
import qwen_hierarchical_v1_contract as contract
import qwen_hierarchical_v1_prompts as mature_prompts
import qwen_hierarchical_v1_reduce as mature_reduce


def load_local_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


retrieval_planner = load_local_module(
    "night_retrieval_planner",
    EXPERIMENT_SCRIPTS / "ten_rubric_retrieval_planner_v1.py",
)
binary_support = load_local_module(
    "night_online_p0_support",
    EXPERIMENT_SCRIPTS / "online_p0_ab_v1.py",
)


SCHEMA_VERSION = "night_exploration.v2_retrieval_only_ab.v1"
ALGORITHM_ID = "qwen_experiment_action_hierarchical_v2_retrieval_only_ab_v1"
SELECTED_RUBRICS = (0, 3, 5, 7, 9)
ACTION_QUERIES = {
    "wiring_action": "画面是否直接出现手持导线插接、拔出或调整接线，或者为接线摆放器材？",
    "measurement_action": "画面是否直接出现操作开关、观察电表或靠近表盘读取示数？",
    "writing_action": "画面是否直接出现笔尖接触记录纸并书写、填写或计算？",
    "cleanup_action": "画面是否直接出现最终连续拆线、集中收拢器材或把橙红色仪器放回桌子左上角？",
}


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_bindings() -> dict[str, Any]:
    paths = {
        "map_reduce_boundary_prompts": Path(mature_prompts.__file__).resolve(),
        "state_machine_and_event_reducer": Path(mature_reduce.__file__).resolve(),
        "pipeline_engine": Path(engine.__file__).resolve(),
        "v2_entrypoint": Path(v2_entrypoint.__file__).resolve(),
        "v2_schema": v2_entrypoint.DEFAULT_SCHEMA.resolve(),
    }
    return {
        key: {"path": str(path), "sha256": sha256(path)}
        for key, path in paths.items()
    }


class CountingClient:
    def __init__(self, client: Any) -> None:
        self.delegate = client
        self.call_count = 0
        self.image_exposures = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs: Any) -> Any:
        self.call_count += 1
        messages = kwargs.get("messages") or []
        for message in messages:
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
        map_max_tokens=2200,
        reduce_max_tokens=2600,
        boundary_max_tokens=1200,
        reduce_recovery_policy="local_partial",
        sample_interval_seconds=2.0,
        window_seconds=60.0,
        overlap_seconds=10.0,
        boundary_context_seconds=10.0,
        dense_boundary_context_seconds=3.0,
        boundary_min_confidence=0.72,
        skip_boundary_refinement=False,
    )


def summary_records(summary: dict[str, Any]) -> list[dict[str, Any]]:
    records = summary.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("baseline_summary_records_missing")
    return [record for record in records if isinstance(record, dict)]


def baseline_sources(summary: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for record in summary_records(summary):
        source_result = Path(str(record.get("source_result") or ""))
        if not source_result.is_file():
            raise FileNotFoundError(f"source_result_missing:{source_result}")
        source_pipeline_result = read_json(source_result)
        # The source_result is the saved mature v2 run. The surrounding replay
        # summary is used only as an anonymous inventory of the five inputs.
        baseline = dict(source_pipeline_result)
        output.append(
            {
                "source_video_id": str(record["source_video_id"]),
                "source_result": source_result.resolve(),
                "reference_result": source_result.resolve(),
                "baseline": baseline,
            }
        )
    return output


def bind_mature_v2_pipeline() -> None:
    """Bind the already-shipped mature v2 logic without copying or editing it."""
    v2_entrypoint.bind_v2_identity()
    engine.assign_seven_stages = mature_reduce.assign_seven_stages


def gold_by_video(payload: dict[str, Any]) -> dict[str, list[list[Any]]]:
    result = {}
    for record in payload.get("records", []):
        result[str(record["source_video_id"])] = [list(run) for run in record["expected_stage_runs"]]
    return result


def stage_runs_as_lists(runs: list[dict[str, Any]]) -> list[list[Any]]:
    return [
        [str(run["stage"]), float(run["start_seconds"]), float(run["end_seconds"])]
        for run in runs
    ]


def compare_runs(actual: list[list[Any]], expected: list[list[Any]]) -> dict[str, Any]:
    same_count = len(actual) == len(expected)
    same_stages = same_count and all(a[0] == e[0] for a, e in zip(actual, expected))
    deltas = []
    if same_stages:
        deltas = [
            {
                "index": index,
                "stage": actual_run[0],
                "start_delta_seconds": round(float(actual_run[1]) - float(expected_run[1]), 6),
                "end_delta_seconds": round(float(actual_run[2]) - float(expected_run[2]), 6),
            }
            for index, (actual_run, expected_run) in enumerate(zip(actual, expected))
        ]
    return {
        "same_stage_count": same_count,
        "same_stage_sequence": same_stages,
        "within_2_seconds": same_stages and all(
            abs(item["start_delta_seconds"]) <= 2.0 and abs(item["end_delta_seconds"]) <= 2.0
            for item in deltas
        ),
        "boundary_deltas": deltas,
    }


def raw_saved_map_events(source_result: Path) -> list[dict[str, Any]]:
    windows_root = source_result.parent / "map" / "windows"
    if not windows_root.is_dir():
        raise FileNotFoundError(f"saved_map_windows_missing:{windows_root}")
    events = []
    for window_dir in sorted(path for path in windows_root.iterdir() if path.is_dir()):
        result_path = window_dir / "result.json"
        result = read_json(result_path)
        if result.get("valid") is True:
            events.extend(dict(event) for event in result.get("normalized_events", []))
    return events


def prepare_shell(
    baseline: dict[str, Any],
    video_dir: Path,
) -> dict[str, Any]:
    manifest_path = Path(str(baseline["source_manifest"]))
    manifest = read_json(manifest_path)
    source, fps, frame_count = engine._video_metadata(manifest)
    fixed = baseline["locked_experiment_interval_seconds"]
    fixed_start, fixed_end = float(fixed[0]), float(fixed[1])
    source_record = {
        "schema_version": SCHEMA_VERSION,
        "algorithm_id": ALGORITHM_ID,
        "stage_schema_id": v2_entrypoint.STAGE_SCHEMA_ID,
        "source_video_id": baseline["source_video_id"],
        "source_video": str(source.resolve()),
        "source_manifest": str(manifest_path.resolve()),
        "source_segment_provenance": baseline["source_segment_provenance"],
        "locked_experiment_interval_seconds": [fixed_start, fixed_end],
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
    prepared: dict[str, Any],
    times: list[float],
    max_model_edge: int,
) -> list[dict[str, Any]]:
    numbers = []
    seen = set()
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


def clip_times(start: float, end: float) -> list[float]:
    return [start + fraction * (end - start) for fraction in (0.2, 0.5, 0.8)]


def selector_prompt(clips: list[dict[str, Any]], action: str) -> str:
    mapping = "\n".join(
        f"- {clip['clip_id']}: {', '.join(frame['image_id'] for frame in clip['frames'])}"
        for clip in clips
    )
    return f"""这些图片来自同一个伏安法测电阻视频，分属于按时间排列且互不重叠的候选片段。

检索问题：{ACTION_QUERIES[action]}

片段及其合法 FRAME ID：
{mapping}

你现在只负责选择后续需要由正式 v2 Map 重新分析的候选片段，不负责输出阶段、时间或最终结论。对每个片段给出画面包含目标动作的概率。只依据可见像素，不按实验流程猜测。

只输出 JSON：
{{"clips":[{{"clip_id":"...","answer":"yes"|"no","target_probability":0.0,"selected_frame_id":"..."|null,"evidence":"..."}}]}}"""


def validate_selector_response(value: dict[str, Any], clips: list[dict[str, Any]]) -> bool:
    rows = value.get("clips")
    if not isinstance(rows, list):
        return False
    expected = {str(clip["clip_id"]) for clip in clips}
    observed = {str(row.get("clip_id")) for row in rows if isinstance(row, dict)}
    if observed != expected or len(rows) != len(clips):
        return False
    frame_ids = {
        str(clip["clip_id"]): {str(frame["image_id"]) for frame in clip["frames"]}
        for clip in clips
    }
    for row in rows:
        if not isinstance(row, dict) or row.get("answer") not in {"yes", "no"}:
            return False
        probability = row.get("target_probability")
        if isinstance(probability, bool) or not isinstance(probability, (int, float)):
            return False
        if not 0.0 <= float(probability) <= 1.0:
            return False
        selected = row.get("selected_frame_id")
        if selected is not None and selected not in frame_ids[str(row["clip_id"])]:
            return False
    return True


def select_yes_no_intervals(
    prepared: dict[str, Any],
    client: Any | None,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    traces = []
    selected = []
    for action in ACTION_QUERIES:
        raw_clips = binary_support.partition(
            prepared["fixed_start"], prepared["fixed_end"], 8, f"sel_{action}"
        )
        clips = []
        for clip in raw_clips:
            item = dict(clip)
            item["frames"] = frames_for_times(prepared, clip["sample_seconds"], args.max_model_edge)
            clips.append(item)
        prompt = selector_prompt(clips, action)
        selector_dir = prepared["video_dir"] / "retrieval" / "yes_no" / action
        selector_dir.mkdir(parents=True, exist_ok=True)
        engine._write_text(selector_dir / "prompt.txt", prompt)
        write_json(selector_dir / "input.json", {"action": action, "clips": clips})
        if client is None:
            response = {"status": "prepared"}
            chosen = clips[:2]
        else:
            all_frames = [frame for clip in clips for frame in clip["frames"]]
            response = binary_support.QwenClient.call_json(
                client,
                prompt,
                all_frames,
                lambda value, c=clips: validate_selector_response(value, c),
            )
            if response["status"] != "valid":
                chosen = []
            else:
                probabilities = {
                    str(row["clip_id"]): float(row["target_probability"])
                    for row in response["result"]["clips"]
                }
                chosen = sorted(
                    clips,
                    key=lambda clip: (
                        -probabilities[clip["clip_id"]],
                        float(clip["start_seconds"]),
                    ),
                )[:2]
        write_json(selector_dir / "result.json", {"action": action, "qwen": response, "selected_clip_ids": [clip["clip_id"] for clip in chosen]})
        traces.append({"action": action, "qwen": response, "selected_clip_ids": [clip["clip_id"] for clip in chosen]})
        selected.extend(
            {
                "source": "yes_no_selector",
                "action_query": action,
                "start_seconds": float(clip["start_seconds"]),
                "end_seconds": float(clip["end_seconds"]),
            }
            for clip in chosen
        )
    return merge_intervals(selected, maximum_length=60.0), traces


def merge_intervals(
    intervals: list[dict[str, Any]],
    maximum_length: float = 60.0,
) -> list[dict[str, Any]]:
    ordered = sorted(intervals, key=lambda item: (float(item["start_seconds"]), float(item["end_seconds"])))
    merged: list[dict[str, Any]] = []
    for item in ordered:
        start = float(item["start_seconds"])
        end = float(item["end_seconds"])
        if not merged or start > float(merged[-1]["end_seconds"]) + 2.0 or end - float(merged[-1]["start_seconds"]) > maximum_length:
            merged.append({
                "start_seconds": start,
                "end_seconds": end,
                "sources": [str(item.get("source") or "unknown")],
                "action_queries": [str(item.get("action_query") or "")],
            })
        else:
            merged[-1]["end_seconds"] = max(float(merged[-1]["end_seconds"]), end)
            merged[-1]["sources"].append(str(item.get("source") or "unknown"))
            merged[-1]["action_queries"].append(str(item.get("action_query") or ""))
    return merged


def rubric_intervals(
    baseline: dict[str, Any],
    profiles: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_record = {
        "source_video_id": baseline["source_video_id"],
        "observed_stage_runs": baseline.get("observed_stage_runs", []),
    }
    plan = retrieval_planner.build_video_plan(source_record, profiles)
    intervals = []
    for rubric in plan["rubric_plans"]:
        if int(rubric["rubric_id"]) not in SELECTED_RUBRICS:
            continue
        for window in rubric["candidate_windows"]:
            intervals.append({
                "source": f"rubric_{rubric['rubric_id']}",
                "action_query": "",
                "start_seconds": float(window["start_seconds"]),
                "end_seconds": float(window["end_seconds"]),
                "planned_times": [float(value) for value in window["planned_sample_times_seconds"]],
            })
    # Preserve profile-level windows and frame budgets. They are already bounded
    # and may intentionally overlap to retrieve different evidence contexts.
    return intervals, plan


def make_map_windows(
    prepared: dict[str, Any],
    intervals: list[dict[str, Any]],
    method: str,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    windows = []
    for index, interval in enumerate(intervals, start=1):
        start = max(prepared["fixed_start"], float(interval["start_seconds"]))
        end = min(prepared["fixed_end"], float(interval["end_seconds"]))
        if end <= start:
            continue
        times = interval.get("planned_times")
        if not isinstance(times, list) or not times:
            times = contract.sample_timestamps(start, end, args.sample_interval_seconds)
        times = [float(value) for value in times if start - 1e-6 <= float(value) <= end + 1e-6]
        frames = frames_for_times(prepared, times, args.max_model_edge)
        if len(frames) < 2:
            frames = frames_for_times(prepared, [start, end], args.max_model_edge)
        window_id = f"supp_{method}_{index:03d}"
        window = {
            "window_id": window_id,
            "window_index": index - 1,
            "window_seconds": [round(start, 6), round(end, 6)],
            "window_length_seconds": round(end - start, 6),
            "frames": frames,
            "retrieval_sources": interval.get("sources", [interval.get("source")]),
            "retrieval_action_queries": interval.get("action_queries", [interval.get("action_query")]),
        }
        input_dir = prepared["video_dir"] / "map" / "windows" / window_id
        write_json(input_dir / "input.json", window)
        engine._write_text(
            input_dir / "prompt.txt",
            mature_prompts.build_map_prompt(prepared["video_id"], window, frames),
        )
        windows.append(window)
    prepared["prepared_windows"] = windows
    return windows


def run_supplemental_map(
    prepared: dict[str, Any],
    client: Any | None,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    if client is None:
        return [], [], []
    return engine._run_map(prepared, client, args)


def analyze_combined(
    prepared: dict[str, Any],
    map_events: list[dict[str, Any]],
    client: Any,
    schema: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    accepted, reduce_result, reduce_review = engine._run_reduce(prepared, map_events, client, args)
    terminal_id = reduce_result["selection"].get("terminal_cleanup_event_id")
    state_result = engine.assign_seven_stages(accepted, terminal_id)
    stage_runs = engine.merge_observed_stage_runs(state_result["observed_stage_intervals"])
    all_candidates = engine.build_boundary_candidates(state_result["observed_stage_intervals"])
    candidates = [item for item in all_candidates if item.get("coarse_order_valid")]
    labels = {str(item["id"]): str(item["label_zh"]) for item in schema["stages"]}
    boundaries, boundary_review = engine._refine_boundaries(
        prepared, candidates, client, labels, args
    )
    boundaries, rejected_boundaries = engine._enforce_boundary_monotonicity(boundaries)
    result = {
        "schema_version": SCHEMA_VERSION,
        "algorithm_id": ALGORITHM_ID,
        "source_video_id": prepared["video_id"],
        "source_locked_interval_seconds": [prepared["fixed_start"], prepared["fixed_end"]],
        "map_event_count_before_deduplication": len(map_events),
        "reduce": reduce_result,
        "assigned_events": state_result["assigned_events"],
        "observed_stage_intervals": state_result["observed_stage_intervals"],
        "observed_stage_runs": stage_runs,
        "boundaries": boundaries,
        "rejected_boundaries": rejected_boundaries,
        "missing_stages": state_result["missing_stages"],
        "review_reasons": sorted(set(reduce_review + state_result["review_reasons"] + boundary_review)),
    }
    write_json(prepared["video_dir"] / "result.json", result)
    return result


def run_method(
    method: str,
    item: dict[str, Any],
    method_root: Path,
    profiles: dict[str, Any],
    client: Any | None,
    schema: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    baseline = item["baseline"]
    prepared = prepare_shell(baseline, method_root / binary_support.safe_slug(item["source_video_id"]))
    base_events = raw_saved_map_events(item["source_result"])
    retrieval_trace: dict[str, Any]
    if method == "rubric_guided":
        intervals, plan = rubric_intervals(baseline, profiles)
        retrieval_trace = {"selected_rubrics": list(SELECTED_RUBRICS), "plan": plan}
    elif method == "yes_no_temporal":
        intervals, traces = select_yes_no_intervals(prepared, client, args)
        retrieval_trace = {"selector_traces": traces}
    else:
        raise ValueError(f"unknown_method:{method}")
    windows = make_map_windows(prepared, intervals, method, args)
    supplemental_events, window_results, map_review = run_supplemental_map(prepared, client, args)
    if client is None:
        result = {
            "status": "prepared",
            "source_video_id": item["source_video_id"],
            "base_event_count": len(base_events),
            "supplemental_window_count": len(windows),
            "supplemental_event_count": 0,
            "retrieval": retrieval_trace,
        }
    else:
        combined = analyze_combined(prepared, base_events + supplemental_events, client, schema, args)
        result = {
            "status": "completed",
            "source_video_id": item["source_video_id"],
            "base_event_count": len(base_events),
            "supplemental_window_count": len(windows),
            "supplemental_event_count": len(supplemental_events),
            "supplemental_map_review": map_review,
            "supplemental_map_windows": window_results,
            "retrieval": retrieval_trace,
            "result_path": str((prepared["video_dir"] / "result.json").resolve()),
            "observed_stage_runs": combined["observed_stage_runs"],
        }
    write_json(prepared["video_dir"] / "method_summary.json", result)
    return result


def baseline_record(item: dict[str, Any]) -> dict[str, Any]:
    baseline = item["baseline"]
    return {
        "status": "saved_baseline",
        "source_video_id": item["source_video_id"],
        "source_result": str(item["source_result"]),
        "reference_result": str(item["reference_result"]),
        "observed_stage_runs": baseline.get("observed_stage_runs", []),
    }


def attach_comparisons(
    record: dict[str, Any],
    gold: list[list[Any]],
) -> None:
    baseline_runs = stage_runs_as_lists(record["methods"]["baseline_v2"]["observed_stage_runs"])
    for method_name, method in record["methods"].items():
        if method.get("status") == "prepared":
            continue
        runs = stage_runs_as_lists(method["observed_stage_runs"])
        method["stage_runs_compact"] = runs
        method["vs_saved_v2"] = compare_runs(runs, baseline_runs)
        method["vs_golden"] = compare_runs(runs, gold)


def aggregate(records: list[dict[str, Any]], method: str) -> dict[str, Any]:
    completed = [record["methods"][method] for record in records if record["methods"][method].get("status") != "prepared"]
    return {
        "method": method,
        "video_count": len(completed),
        "same_sequence_as_saved_v2_count": sum(item["vs_saved_v2"]["same_stage_sequence"] for item in completed),
        "within_2s_of_saved_v2_count": sum(item["vs_saved_v2"]["within_2_seconds"] for item in completed),
        "same_sequence_as_golden_count": sum(item["vs_golden"]["same_stage_sequence"] for item in completed),
        "within_2s_of_golden_count": sum(item["vs_golden"]["within_2_seconds"] for item in completed),
    }


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# v2 retrieval-only online A/B",
        "",
        "The mature v2 Map prompt, Reduce, state machine, terminal barrier, and boundary refinement are imported unchanged.",
        "Gold is local scoring data and is not used for retrieval or prompts.",
        "",
        "| Method | Same sequence as saved v2 | Within 2s of saved v2 | Same sequence as golden | Within 2s of golden |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in result.get("aggregate", []):
        lines.append(
            f"| {item['method']} | {item['same_sequence_as_saved_v2_count']}/{item['video_count']} | "
            f"{item['within_2s_of_saved_v2_count']}/{item['video_count']} | "
            f"{item['same_sequence_as_golden_count']}/{item['video_count']} | "
            f"{item['within_2s_of_golden_count']}/{item['video_count']} |"
        )
    lines.extend(["", f"Qwen calls: {result['qwen_usage']['call_count']}; image exposures: {result['qwen_usage']['image_exposures']}.", ""])
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    bind_mature_v2_pipeline()
    schema = contract.load_stage_schema(v2_entrypoint.DEFAULT_SCHEMA)
    profiles = read_json(args.profiles)
    sources = baseline_sources(read_json(args.baseline_summary))
    gold = gold_by_video(read_json(args.gold))
    missing_gold = [item["source_video_id"] for item in sources if item["source_video_id"] not in gold]
    if missing_gold:
        raise ValueError(f"gold_missing:{missing_gold}")
    client = None
    if not args.prepare_only:
        delegate = qwen_base.OpenAI(
            base_url=args.endpoint,
            api_key=args.token,
            timeout=args.timeout,
            max_retries=0,
        )
        client = CountingClient(delegate)
        # Reuse the tested JSON-call helper for Yes/No selection while all
        # official Map/Reduce/boundary calls continue through the mature engine.
        client.endpoint = args.endpoint.rstrip("/") + "/chat/completions"
        client.token = args.token
        client.model = args.model
        client.timeout = args.timeout
    engine_args = default_engine_args()
    records = []
    for index, item in enumerate(sources, start=1):
        video_id = item["source_video_id"]
        record = {
            "source_video_id": video_id,
            "gold_sent_to_qwen": False,
            "methods": {"baseline_v2": baseline_record(item)},
        }
        for method in ("rubric_guided", "yes_no_temporal"):
            before_calls = client.call_count if client else 0
            before_images = client.image_exposures if client else 0
            method_result = run_method(
                method,
                item,
                args.output / method,
                profiles,
                client,
                schema,
                engine_args,
            )
            method_result["qwen_calls"] = (client.call_count - before_calls) if client else 0
            method_result["qwen_image_exposures"] = (client.image_exposures - before_images) if client else 0
            record["methods"][method] = method_result
        attach_comparisons(record, gold[video_id])
        records.append(record)
        write_json(args.output / "records" / f"source_{index:03d}.json", record)
        print(json.dumps({"video": index, "total": len(sources), "source_video_id": video_id, "status": "prepared" if args.prepare_only else "completed"}, ensure_ascii=False), flush=True)
    result = {
        "schema_version": SCHEMA_VERSION,
        "algorithm_id": ALGORITHM_ID,
        "status": "prepared" if args.prepare_only else "completed",
        "invariants": {
            "mature_v2_logic_imported_unchanged": True,
            "retrieval_changes_only": True,
            "gold_sent_to_qwen": False,
            "selected_rubrics": list(SELECTED_RUBRICS),
            "source_bindings": source_bindings(),
        },
        "records": records,
        "aggregate": [aggregate(records, method) for method in ("baseline_v2", "rubric_guided", "yes_no_temporal")] if not args.prepare_only else [],
        "qwen_usage": {
            "call_count": client.call_count if client else 0,
            "image_exposures": client.image_exposures if client else 0,
        },
    }
    write_json(args.output / "comparison.json", result)
    write_markdown(args.output / "comparison.md", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-summary", required=True, type=Path)
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument(
        "--profiles",
        type=Path,
        default=REPO_ROOT / "experiments" / "night_exploration_20260812" / "configs" / "ten_rubric_retrieval_profiles_v1.json",
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
