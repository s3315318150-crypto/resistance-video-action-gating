#!/usr/bin/env python3
"""Boundary-frame retrieval A/B on the confirmed Temporal Guard v2 baseline.

This version refuses a baseline unless every saved stage run exactly matches
the local Temporal Guard v2 golden fixture. Map events, Reduce output, state
assignments, and stage runs are replayed; only boundary-review frames change.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
V3_SCRIPT = SCRIPT_DIR / "v2_retrieval_boundary_only_ab_v3.py"
SPEC = importlib.util.spec_from_file_location("v2_boundary_only_v3_for_tg_v4", V3_SCRIPT)
v3 = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(v3)

import qwen_experiment_action_hierarchical_v2_temporal_guard as temporal_guard_entrypoint
import qwen_hierarchical_v2_temporal_guard_reduce as temporal_guard_reduce


SCHEMA_VERSION = "night_exploration.v2_temporal_guard_boundary_retrieval_ab.v4"
ALGORITHM_ID = "qwen_experiment_action_hierarchical_v2_temporal_guard_boundary_retrieval_ab_v4"
EXPECTED_REPLAY_MODE = "stored_successful_map_and_reduce_temporal_guard_replay"


def bind_target_pipeline() -> None:
    """Bind the exact reducer used by the user's confirmed v2 result."""
    temporal_guard_entrypoint.bind_v2_temporal_guard()
    v3.base.engine.assign_seven_stages = v3.base.mature_reduce.assign_seven_stages


def target_source_bindings() -> dict[str, Any]:
    bindings = v3.base.source_bindings()
    for key, path in {
        "temporal_guard_entrypoint": Path(temporal_guard_entrypoint.__file__).resolve(),
        "temporal_guard_reducer": Path(temporal_guard_reduce.__file__).resolve(),
    }.items():
        bindings[key] = {"path": str(path), "sha256": v3.base.sha256(path)}
    return bindings


def exact_stage_runs(payload: dict[str, Any]) -> list[list[Any]]:
    return v3.base.stage_runs_as_lists(payload.get("observed_stage_runs", []))


def temporal_guard_sources(
    summary: dict[str, Any],
    gold: dict[str, list[list[Any]]],
) -> list[dict[str, Any]]:
    if summary.get("mode") != EXPECTED_REPLAY_MODE:
        raise ValueError(f"wrong_temporal_guard_replay_mode:{summary.get('mode')}")
    records = summary.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("temporal_guard_replay_records_missing")

    sources: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("temporal_guard_replay_record_invalid")
        video_id = str(record.get("source_video_id") or "")
        if video_id not in gold:
            raise ValueError(f"gold_missing:{video_id}")
        source_result = Path(str(record.get("source_result") or ""))
        replay_result = Path(str(record.get("replay_result") or ""))
        if not source_result.is_file():
            raise FileNotFoundError(f"source_result_missing:{source_result}")
        if not replay_result.is_file():
            raise FileNotFoundError(f"replay_result_missing:{replay_result}")

        source_payload = v3.base.read_json(source_result)
        replay_payload = v3.base.read_json(replay_result)
        replay_runs = exact_stage_runs(replay_payload)
        summary_runs = v3.base.stage_runs_as_lists(record.get("observed_stage_runs", []))
        expected_runs = gold[video_id]
        if replay_runs != summary_runs:
            raise ValueError(f"replay_summary_stage_mismatch:{video_id}")
        if replay_runs != expected_runs:
            raise ValueError(f"baseline_not_confirmed_temporal_guard_v2:{video_id}")

        baseline = dict(source_payload)
        baseline.update(
            {
                "schema_version": temporal_guard_entrypoint.ALGORITHM_SCHEMA_VERSION,
                "algorithm_id": temporal_guard_entrypoint.ALGORITHM_ID,
                "observed_stage_intervals": replay_payload.get("observed_stage_intervals", []),
                "observed_stage_runs": replay_payload.get("observed_stage_runs", []),
                "assigned_events": replay_payload.get("assigned_events", []),
                "boundaries": [],
                "replay_effective_reduce_result": replay_payload.get("effective_reduce_result"),
                "replay_selection": replay_payload.get("selection"),
                "temporal_guard_restored_event_count": int(record.get("restored_event_count", 0)),
            }
        )
        sources.append(
            {
                "source_video_id": video_id,
                "source_result": replay_result.resolve(),
                "reference_result": replay_result.resolve(),
                "map_source_result": source_result.resolve(),
                "baseline": baseline,
            }
        )
    return sources


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise FileExistsError(args.output)

    gold = v3.base.gold_by_video(v3.base.read_json(args.gold))
    sources = temporal_guard_sources(v3.base.read_json(args.baseline_summary), gold)
    bind_target_pipeline()
    schema = v3.base.contract.load_stage_schema(temporal_guard_entrypoint.DEFAULT_SCHEMA)
    labels = {str(item["id"]): str(item["label_zh"]) for item in schema["stages"]}
    profiles = v3.base.read_json(args.profiles)

    args.output.mkdir(parents=True)
    client = None
    if not args.prepare_only:
        delegate = v3.base.qwen_base.OpenAI(
            base_url=args.endpoint,
            api_key=args.token,
            timeout=args.timeout,
            max_retries=0,
        )
        client = v3.base.CountingClient(delegate)
        client.endpoint = args.endpoint.rstrip("/") + "/chat/completions"
        client.token = args.token
        client.model = args.model
        client.timeout = args.timeout

    engine_args = v3.base.default_engine_args()
    records: list[dict[str, Any]] = []
    for index, item in enumerate(sources, start=1):
        video_id = item["source_video_id"]
        baseline_method = v3.base.baseline_record(item)
        baseline_method["temporal_guard_restored_event_count"] = item["baseline"][
            "temporal_guard_restored_event_count"
        ]
        record: dict[str, Any] = {
            "source_video_id": video_id,
            "gold_sent_to_qwen": False,
            "baseline_exactly_matches_confirmed_v2": True,
            "methods": {"baseline_v2": baseline_method},
        }
        before_calls = client.call_count if client else 0
        before_images = client.image_exposures if client else 0
        for method in v3.METHODS:
            method_before_calls = client.call_count if client else 0
            method_before_images = client.image_exposures if client else 0
            value = v3.run_method(
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
        v3.base.attach_comparisons(record, gold[video_id])
        record["qwen_calls"] = (client.call_count - before_calls) if client else 0
        record["qwen_image_exposures"] = (client.image_exposures - before_images) if client else 0
        records.append(record)
        v3.base.write_json(args.output / "records" / f"source_{index:03d}.json", record)
        print(
            json.dumps(
                {"video": index, "total": len(sources), "source_video_id": video_id},
                ensure_ascii=False,
            ),
            flush=True,
        )

    result = {
        "schema_version": SCHEMA_VERSION,
        "algorithm_id": ALGORITHM_ID,
        "status": "prepared" if args.prepare_only else "completed",
        "supersedes_invalid_target_claims_from": [
            "outputs/v2_ro_ab_20260812",
            "outputs/v2_iw_ab_20260812",
            "outputs/v2_bo_ab_20260812",
        ],
        "invariants": {
            "confirmed_temporal_guard_v2_baseline": True,
            "all_baseline_stage_runs_exactly_match_gold": True,
            "saved_map_events_reused": True,
            "saved_temporal_guard_reduce_reused": True,
            "saved_state_assignments_reused": True,
            "saved_stage_runs_reused_verbatim": True,
            "original_v2_boundary_prompt_imported_unchanged": True,
            "gold_sent_to_qwen": False,
            "source_bindings": target_source_bindings(),
        },
        "records": records,
        "aggregate": [v3.aggregate(records, method) for method in v3.METHODS]
        if not args.prepare_only
        else [],
        "qwen_usage": {
            "call_count": client.call_count if client else 0,
            "image_exposures": client.image_exposures if client else 0,
        },
    }
    v3.base.write_json(args.output / "comparison.json", result)
    v3.write_markdown(args.output / "comparison.md", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-summary", required=True, type=Path)
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument(
        "--profiles",
        type=Path,
        default=v3.base.REPO_ROOT
        / "experiments"
        / "night_exploration_20260812"
        / "configs"
        / "ten_rubric_retrieval_profiles_v1.json",
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
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(args.output),
                "qwen_calls": result["qwen_usage"]["call_count"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
