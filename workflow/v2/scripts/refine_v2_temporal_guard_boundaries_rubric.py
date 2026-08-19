#!/usr/bin/env python3
"""Refine existing hierarchical-v2 boundaries with rubric-guided frames.

The source v2 stage runs remain immutable. This command selects a bounded
six-second review window for each existing stage transition, samples it every
0.5 seconds, and calls the original v2 boundary prompt. It supports both a
normal hierarchical-v2 summary and the deterministic replay summary used by the
five-video validation, but it never reads golden labels or video-specific
times.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
SCRIPT_ROOT = ROOT / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import rubric_boundary_support as support


DEFAULT_PROFILES = ROOT / "configs" / "rubric_retrieval_profiles_v1.json"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "v2_temporal_guard_rubric_boundary"
SCHEMA_VERSION = "v2_temporal_guard_rubric_boundary.v1"
ALGORITHM_ID = "v2-temporal-guard-rubric-boundary"
FIVE_STAGE_ALGORITHM_ID = "qwen_experiment_action_hierarchical_v2_five_stage"
FRAME_AGENT_ALGORITHM_ID = "qwen_experiment_action_hierarchical_v2_frame_agent"
SCREENSHOT_GUARD_ALGORITHM_ID = "qwen_experiment_action_hierarchical_v2_screenshot_guard"
SCREENSHOT_GUARD_AGENT_ALGORITHM_ID = (
    "qwen_experiment_action_hierarchical_v2_screenshot_guard_agent"
)
FIVE_STAGE_SCHEMA_ID = "resistance_5stage_measurement_recording_v1"
FIVE_STAGE_SCHEMA = (
    ROOT
    / "configs"
    / "action_schemas"
    / "resistance_5stage_measurement_recording_v1.json"
)


def anonymous_video_directory(source_video_id: str) -> str:
    return "video_" + hashlib.sha256(source_video_id.encode("utf-8")).hexdigest()[:12]


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


def resolve_record_path(record: dict[str, Any], summary_path: Path) -> Path:
    raw = record.get("replay_result") or record.get("result_path")
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"action_result_path_missing:{record.get('source_video_id')}")
    path = Path(raw)
    if not path.is_absolute():
        path = summary_path.parent / path
    if not path.is_file():
        raise FileNotFoundError(f"action_result_missing:{path}")
    return path.resolve()


def merge_replay_source(
    record: dict[str, Any],
    replay: dict[str, Any],
    summary_path: Path,
) -> dict[str, Any]:
    source_raw = record.get("source_result")
    if not isinstance(source_raw, str) or not source_raw.strip():
        raise ValueError(f"replay_source_result_missing:{record.get('source_video_id')}")
    source_path = Path(source_raw)
    if not source_path.is_absolute():
        source_path = summary_path.parent / source_path
    source = read_json(source_path.resolve())
    merged = dict(source)
    for field in ("observed_stage_intervals", "observed_stage_runs", "assigned_events"):
        merged[field] = replay.get(field, [])
    merged["temporal_guard_replay_result"] = str(resolve_record_path(record, summary_path))
    return merged


def load_action_records(summary_path: Path) -> list[dict[str, Any]]:
    summary = read_json(summary_path)
    records = summary.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("action_summary_records_missing")
    replay_mode = summary.get("mode") == "stored_successful_map_and_reduce_temporal_guard_replay"
    output: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("action_summary_record_invalid")
        result_path = resolve_record_path(record, summary_path)
        result = read_json(result_path)
        if replay_mode:
            result = merge_replay_source(record, result, summary_path)
        elif result.get("algorithm_id") not in {
            "qwen_experiment_action_hierarchical_v2",
            "qwen_experiment_action_hierarchical_v2_temporal_guard",
            FRAME_AGENT_ALGORITHM_ID,
            SCREENSHOT_GUARD_ALGORITHM_ID,
            SCREENSHOT_GUARD_AGENT_ALGORITHM_ID,
            FIVE_STAGE_ALGORITHM_ID,
        }:
            raise ValueError(
                f"not_hierarchical_v2:{record.get('source_video_id')}:{result.get('algorithm_id')}"
            )
        video_id = str(result.get("source_video_id") or record.get("source_video_id") or "")
        intervals = result.get("observed_stage_intervals")
        runs = result.get("observed_stage_runs")
        if not video_id or not isinstance(intervals, list) or not intervals:
            raise ValueError(f"observed_stage_intervals_missing:{video_id}")
        if not isinstance(runs, list) or not runs:
            raise ValueError(f"observed_stage_runs_missing:{video_id}")
        output.append(
            {
                "source_video_id": video_id,
                "source_result": result_path,
                "baseline": result,
            }
        )
    return output


def stage_labels_for_records(records: list[dict[str, Any]]) -> dict[str, str]:
    """Load labels for one homogeneous action schema without rewriting stages."""
    schema_ids = {
        str(item["baseline"].get("stage_schema_id") or "")
        for item in records
        if isinstance(item.get("baseline"), dict)
    }
    if schema_ids == {FIVE_STAGE_SCHEMA_ID}:
        schema = read_json(FIVE_STAGE_SCHEMA)
    else:
        if FIVE_STAGE_SCHEMA_ID in schema_ids:
            raise ValueError("mixed_action_stage_schemas_not_supported")
        support.bind_mature_v2_pipeline()
        schema = support.contract.load_stage_schema(support.v2_entrypoint.DEFAULT_SCHEMA)
    stages = schema.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ValueError("action_stage_schema_missing")
    return {str(item["id"]): str(item["label_zh"]) for item in stages if isinstance(item, dict)}


def fallback_boundary(boundary: dict[str, Any]) -> dict[str, Any]:
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
        "evidence": "Rubric 边界复核没有合法结果，保留 v2 粗边界。",
        "uncertainty": "rubric_boundary_review_failed",
        "needs_review": True,
        "source": "temporal_guard_v2_coarse_fallback",
    }


def refine_one(
    item: dict[str, Any],
    output_root: Path,
    profiles: dict[str, Any],
    labels: dict[str, str],
    client: Any | None,
    engine_args: argparse.Namespace,
) -> dict[str, Any]:
    baseline = item["baseline"]
    video_dir = output_root / anonymous_video_directory(item["source_video_id"])
    prepared = support.prepare_shell(baseline, video_dir)
    candidates = support.candidate_boundaries(baseline)
    intervals, plan = support.relevant_rubric_intervals(baseline, profiles)
    refined: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    for boundary in candidates:
        start, end, retrieval = support.rubric_boundary_range(prepared, boundary, intervals)
        boundary_pass = support.run_original_boundary_prompt(
            prepared,
            boundary,
            start,
            end,
            labels,
            client,
            engine_args,
            "rubric_guided_boundary",
        )
        observed = support.engine._observed_boundary_from_pass(
            boundary_pass,
            engine_args.boundary_min_confidence,
        )
        chosen = (
            {
                **boundary,
                **observed,
                "source": "rubric_guided_original_v2_boundary_prompt",
            }
            if observed is not None
            else fallback_boundary(boundary)
        )
        refined.append(chosen)
        traces.append(
            {
                "boundary_id": boundary["boundary_id"],
                "from_stage": boundary["from_stage"],
                "to_stage": boundary["to_stage"],
                "review_range_seconds": [start, end],
                "retrieval": retrieval,
                "boundary_prompt_valid": boundary_pass["valid"],
                "result_source": chosen["source"],
            }
        )
    refined, rejected = support.engine._enforce_boundary_monotonicity(refined)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "algorithm_id": ALGORITHM_ID,
        "source_video_id": item["source_video_id"],
        "source_action_result": str(item["source_result"]),
        "source_stage_runs_unchanged": True,
        "source_observed_stage_runs": baseline["observed_stage_runs"],
        "source_observed_stage_intervals": baseline["observed_stage_intervals"],
        "refined_boundaries": refined,
        "rejected_refined_boundaries": rejected,
        "retrieval_traces": traces,
        "rubric_retrieval_plan": plan,
        "boundary_count": len(refined),
        "qwen_call_count": sum(
            len(value.get("attempts", []))
            for value in (
                read_json(path)
                for path in sorted((video_dir / "boundaries").glob("*/result.json"))
            )
        ),
        "status": "prepared" if client is None else "completed",
    }
    write_json(video_dir / "result.json", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action-summary", required=True, type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id")
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES)
    parser.add_argument("--max-model-edge", type=int, default=640)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--boundary-max-tokens", type=int, default=1200)
    parser.add_argument("--boundary-min-confidence", type=float, default=0.72)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--prepare-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_model_edge <= 0 or args.max_attempts <= 0 or args.boundary_max_tokens <= 0:
        raise ValueError("resolution_attempts_and_tokens_must_be_positive")
    if not 0.0 <= args.boundary_min_confidence <= 1.0:
        raise ValueError("boundary_min_confidence_out_of_range")
    action_summary = args.action_summary.resolve()
    profiles = read_json(args.profiles.resolve())
    records = load_action_records(action_summary)
    run_id = args.run_id or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = args.output_root.resolve() / run_id
    if run_dir.exists():
        raise FileExistsError(f"run_directory_exists:{run_dir}")
    run_dir.mkdir(parents=True)

    # The shared contract defaults to v1 until an entrypoint binds its schema
    # identity. Boundary refinement reuses v2 prompts/contracts but never runs
    # Reduce, so the mature v2 binding is the required process-local setup.
    support.bind_mature_v2_pipeline()
    labels = stage_labels_for_records(records)
    client = None
    if not args.prepare_only:
        endpoint = os.getenv("QWEN_API_BASE_URL", "").strip()
        token = os.getenv("QWEN_API_TOKEN", "").strip()
        model = os.getenv("QWEN_MODEL", "qwen").strip() or "qwen"
        if not endpoint or not token:
            raise RuntimeError("QWEN_API_BASE_URL and QWEN_API_TOKEN are required")
        client = support.CountingClient(
            support.qwen_base.OpenAI(
                base_url=endpoint,
                api_key=token,
                timeout=args.timeout,
                max_retries=0,
            )
        )

    engine_args = support.default_engine_args()
    engine_args.max_model_edge = args.max_model_edge
    engine_args.max_attempts = args.max_attempts
    engine_args.boundary_max_tokens = args.boundary_max_tokens
    engine_args.boundary_min_confidence = args.boundary_min_confidence
    outputs = [
        refine_one(item, run_dir, profiles, labels, client, engine_args)
        for item in records
    ]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "algorithm_id": ALGORITHM_ID,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "status": "prepared" if args.prepare_only else "completed",
        "source_action_summary": str(action_summary),
        "source_stage_runs_unchanged": True,
        "golden_fixture_used": False,
        "video_count": len(outputs),
        "boundary_count": sum(item["boundary_count"] for item in outputs),
        "qwen_call_count": sum(item["qwen_call_count"] for item in outputs),
        "records": [
            {
                "source_video_id": item["source_video_id"],
                "status": item["status"],
                "boundary_count": item["boundary_count"],
                "qwen_call_count": item["qwen_call_count"],
                "result_path": str(
                    (
                        run_dir
                        / anonymous_video_directory(item["source_video_id"])
                        / "result.json"
                    ).resolve()
                ),
            }
            for item in outputs
        ],
    }
    write_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
