#!/usr/bin/env python3
"""Offline golden comparison for the behavior-tolerant stage decoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from qwen_hierarchical_v1_contract import read_json, write_json_atomic
from qwen_hierarchical_v1_reduce import merge_observed_stage_runs
from qwen_hierarchical_v2_behavior_tolerant_reduce import assign_seven_stages_behavior_tolerant


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GOLDEN = ROOT / "tests" / "fixtures" / "v2_behavior_tolerant_golden.json"
VARIANTS = (
    "v2-behavior-tolerant",
    "v2-behavior-tolerant-aux",
    "v2-behavior-tolerant-boundary",
    "v2-behavior-tolerant-adaptive",
)


def compare_runs(
    actual: list[dict[str, Any]], expected: list[list[Any]], tolerance: float
) -> tuple[bool, list[dict[str, Any]]]:
    differences: list[dict[str, Any]] = []
    if len(actual) != len(expected):
        differences.append({"code": "stage_run_count_mismatch", "actual": len(actual), "expected": len(expected)})
    for index, (actual_run, expected_run) in enumerate(zip(actual, expected), start=1):
        expected_stage, expected_start, expected_end = expected_run
        if actual_run.get("stage") != expected_stage:
            differences.append(
                {"code": "stage_mismatch", "index": index, "actual": actual_run.get("stage"), "expected": expected_stage}
            )
        for field, expected_value in (("start_seconds", expected_start), ("end_seconds", expected_end)):
            delta = abs(float(actual_run[field]) - float(expected_value))
            if delta > tolerance + 1e-9:
                differences.append(
                    {"code": "boundary_outside_tolerance", "index": index, "field": field, "delta_seconds": delta}
                )
    return not differences, differences


def run_comparison(replay_root: Path, golden: dict[str, Any], variant: str) -> dict[str, Any]:
    summary = read_json(replay_root / "summary.json")
    golden_by_id = {str(item["source_video_id"]): item for item in golden["records"]}
    records: list[dict[str, Any]] = []
    tolerance = float(golden["boundary_tolerance_seconds"])
    for index, source_record in enumerate(summary["records"], start=1):
        video_id = str(source_record["source_video_id"])
        replay = read_json(replay_root / f"source_{index:03d}" / "result.json")
        state = assign_seven_stages_behavior_tolerant(
            list(replay["accepted_events"]), replay["selection"].get("terminal_cleanup_event_id")
        )
        runs = merge_observed_stage_runs(state["observed_stage_intervals"])
        expected = golden_by_id[video_id]["expected_stage_runs"]
        passed, differences = compare_runs(runs, expected, tolerance)
        records.append(
            {
                "source_video_id": video_id,
                "passed": passed,
                "actual_stage_runs": [
                    [item["stage"], item["start_seconds"], item["end_seconds"]] for item in runs
                ],
                "expected_stage_runs": expected,
                "differences": differences,
            }
        )
    return {
        "schema_version": "v2_behavior_tolerant_comparison.v1",
        "variant": variant,
        "mode": "offline_replay_of_saved_accepted_events",
        "boundary_tolerance_seconds": tolerance,
        "passed": all(item["passed"] for item in records),
        "passed_count": sum(1 for item in records if item["passed"]),
        "record_count": len(records),
        "records": records,
    }


def markdown_report(result: dict[str, Any]) -> str:
    lines = [
        f"# {result['variant']} offline comparison",
        "",
        "| Video | Result | Differences |",
        "|---|---|---|",
    ]
    for item in result["records"]:
        lines.append(
            f"| {item['source_video_id']} | {'pass' if item['passed'] else 'fail'} | "
            f"{json.dumps(item['differences'], ensure_ascii=False)} |"
        )
    lines.extend(["", f"Overall: {result['passed_count']}/{result['record_count']}"])
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_comparison(args.replay_root.resolve(), read_json(args.golden.resolve()), args.variant)
    output = args.output.resolve()
    write_json_atomic(output / "comparison.json", result)
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.md").write_text(markdown_report(result), encoding="utf-8", newline="\n")
    print(f"comparison={output / 'comparison.json'}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
