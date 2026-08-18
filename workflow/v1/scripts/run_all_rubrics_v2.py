#!/usr/bin/env python3
"""Aggregate the ten resistance-video rubrics using the v2 stage gate.

This runner deliberately replays existing, versioned evidence artifacts.  It
does not overwrite historical runs, read the Excel labels, or turn a missing
artifact into an unqualified positive.  Every rubric has a binary decision;
evidence quality and the original diagnostic state remain attached to it.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "all_rubrics_v2"

RUBRICS = (
    (0, "cleanup_and_return", "拆除整理归位"),
    (1, "series_circuit", "电流表串联"),
    (2, "voltmeter_parallel", "电压表并联在待测电阻两端"),
    (3, "switch_open_during_wiring", "接线时开关保持断开"),
    (4, "meter_polarity", "电表正负接线柱正确"),
    (5, "pointer_normal_deflection", "指针正常偏转"),
    (6, "meter_range_appropriate", "电表量程合适"),
    (7, "record_first_measurement", "正确记录第一组数据"),
    (8, "disconnect_before_battery_change", "更换电池前先断开开关"),
    (9, "record_second_measurement", "正确记录第二组数据"),
)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def video_id_from_name(value: str) -> str:
    stem = Path(value).stem
    if not stem:
        raise ValueError(f"Cannot determine video id from {value!r}")
    match = re.match(r"([0-9]+)(?:_|$)", Path(value).name)
    return match.group(1) if match else re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("_.-")


def result(decision: str, source: str, reason: str, confidence: float | None, **diagnostics: Any) -> dict[str, Any]:
    if decision not in {"pass", "fail"}:
        raise ValueError(f"invalid binary decision: {decision}")
    payload: dict[str, Any] = {
        "decision": decision,
        "predicted_score": 1 if decision == "pass" else 0,
        "source_artifact": source,
        "reason": reason,
    }
    if confidence is not None:
        payload["confidence"] = round(float(confidence), 3)
    payload["diagnostics"] = diagnostics
    return payload


def artifact_binary(item: dict[str, Any], *keys: str) -> str:
    """Read a binary decision from a rubric artifact without inventing success."""
    for key in keys:
        value = item.get(key)
        if value in {"pass", "fail"}:
            return str(value)
    if item.get("predicted_score") == 1:
        return "pass"
    return "fail"


def stages_for(action_result: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item
        for item in action_result.get("timeline_segments", [])
        if isinstance(item, dict) and item.get("kind") == "observed_stage"
    ]


def stage_intervals(stages: list[dict[str, Any]]) -> dict[str, list[list[float]]]:
    values: dict[str, list[list[float]]] = {}
    for item in stages:
        name = str(item.get("stage") or "")
        if not name:
            continue
        try:
            interval = [float(item["start_seconds"]), float(item["end_seconds"])]
        except (KeyError, TypeError, ValueError):
            continue
        values.setdefault(name, []).append(interval)
    return values


def load_action_results(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for record in summary.get("records", []):
        source = str(record.get("source_video_id") or "")
        result_path = resolve(str(record["result_path"]))
        document = read_json(result_path)
        output[video_id_from_name(source)] = {
            "source_video": source,
            "result_path": str(result_path.resolve()),
            "summary_record": record,
            "result": document,
            "intervals": stage_intervals(stages_for(document)),
        }
    if not output:
        raise ValueError("v2 action summary has no video records")
    return output


def load_video_map(path: Path, key: str = "video_id") -> dict[str, dict[str, Any]]:
    document = read_json(path)
    values = document.get("video_results") if isinstance(document, dict) else None
    if not isinstance(values, list):
        values = document.get("results") if isinstance(document, dict) else None
    output: dict[str, dict[str, Any]] = {}
    for item in values or []:
        if not isinstance(item, dict):
            continue
        raw = item.get(key) or item.get("video_prefix") or item.get("video_id")
        if raw is None:
            continue
        try:
            vid = video_id_from_name(str(raw))
        except ValueError:
            continue
        output[vid] = item
    return output


def load_episode_map(path: Path) -> dict[str, list[dict[str, Any]]]:
    document = read_json(path)
    output: dict[str, list[dict[str, Any]]] = {}
    for item in document.get("episodes", []):
        if not isinstance(item, dict):
            continue
        vid = str(item.get("video_id") or "")
        if vid:
            output.setdefault(vid, []).append(item)
    return output


def map_manual_switch(path: Path) -> dict[str, dict[str, Any]]:
    document = read_json(path)
    return {str(item["video_id"]): item for item in document.get("results", []) if isinstance(item, dict)}


def map_result_files(root: Path) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for path in root.glob("video_*/result.json"):
        try:
            item = read_json(path)
            vid = str(item.get("video_id") or path.parent.name.removeprefix("video_"))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        output[vid] = item
    return output


def load_ammeter_source_series_results(path: Path) -> dict[str, dict[str, Any]]:
    """Load the dedicated ammeter/source/resistor topology decisions."""
    document = read_json(path)
    if document.get("rubric_id") != "resistance.ammeter_source_short_circuit_v1":
        raise ValueError(f"Unexpected series rubric in {path}")
    output: dict[str, dict[str, Any]] = {}
    for item in document.get("predictions", []):
        if not isinstance(item, dict) or item.get("automated_outcome") != "scored":
            continue
        video_id = video_id_from_name(str(item.get("video_id") or ""))
        decision = item.get("decision")
        score = item.get("predicted_score")
        if decision not in {"是", "不是"} or score not in {0, 1}:
            raise ValueError(f"Invalid binary series result for video {video_id}")
        expected_score = 1 if decision == "是" else 0
        if score != expected_score:
            raise ValueError(f"Inconsistent binary series result for video {video_id}")
        output[video_id] = item
    return output


def load_rubric8_sequence_results(path: Path) -> dict[str, dict[str, Any]]:
    """Load the dedicated terminal-rewire scorer without touching Excel labels."""
    if not path.is_file():
        return {}
    document = read_json(path)
    values = document.get("videos") if isinstance(document, dict) else None
    if not isinstance(values, list):
        values = document.get("records") if isinstance(document, dict) else None
    output: dict[str, dict[str, Any]] = {}
    for item in values or []:
        if not isinstance(item, dict):
            continue
        raw_id = item.get("video_id")
        if raw_id is None:
            continue
        try:
            video_id = video_id_from_name(str(raw_id))
        except ValueError:
            video_id = str(raw_id)
        output[video_id] = item
    return output


def load_battery_results(path: Path) -> dict[str, dict[str, Any]]:
    """Dispatch Rubric 8 summaries by schema instead of directory name."""
    document = read_json(path)
    if isinstance(document.get("videos"), list) or isinstance(document.get("records"), list):
        return load_rubric8_sequence_results(path)

    predictions = document.get("predictions")
    if isinstance(predictions, list):
        return {
            video_id_from_name(item["video_id"]): item
            for item in predictions
            if isinstance(item, dict) and item.get("rubric_stage") == 8
        }
    raise ValueError(f"Unexpected Rubric 8 summary schema in {path}")


def build_evaluations(action: dict[str, Any], artifacts: dict[str, Any]) -> dict[str, dict[str, Any]]:
    intervals = action["intervals"]
    action_path = action["result_path"]
    evaluations: dict[str, dict[str, Any]] = {}

    cleanup = intervals.get("material_cleanup", [])
    evaluations["0"] = result(
        "pass" if cleanup else "fail",
        action_path,
        "v2 观察到最终整理阶段" if cleanup else "v2 未观察到整理阶段",
        0.8 if cleanup else 0.3,
        stage_intervals=cleanup,
        ignored_noise_event_count=len(action["result"].get("ignored_noise_events", [])),
        action_status=action["summary_record"].get("status"),
    )

    episode_items = artifacts["wiring_episodes"].get(action["video_id"], [])
    dedicated_series = artifacts.get("ammeter_source_series", {}).get(action["video_id"])
    if dedicated_series is not None:
        evaluations["1"] = result(
            "pass" if dedicated_series["predicted_score"] == 1 else "fail",
            artifacts["series_source"],
            str(dedicated_series.get("evidence") or "没有可用的电流表、电源和待测电阻拓扑说明"),
            dedicated_series.get("confidence"),
            stage_intervals={"circuit_wiring": intervals.get("circuit_wiring", [])},
            original_decision=dedicated_series.get("decision"),
            original_predicted_score=dedicated_series.get("predicted_score"),
            rubric_id=dedicated_series.get("rubric_id"),
            violation_checks=dedicated_series.get("violation_checks"),
            ammeter_source_relation=dedicated_series.get("ammeter_source_relation"),
            test_resistor_connection=dedicated_series.get("test_resistor_connection"),
            short_circuit_status=dedicated_series.get("short_circuit_status"),
            inference_required=dedicated_series.get("inference_required"),
            binary_policy="ammeter_source_short_circuit_v1",
        )
    else:
        series = next((item.get("series_circuit") for item in episode_items if isinstance(item.get("series_circuit"), dict)), None)
        if series is None:
            series = artifacts["deepseek"].get(action["video_id"], {}).get("series_circuit", {})
        series_decision = str(series.get("decision") or "fail")
        if series_decision not in {"pass", "fail"}:
            series_decision = "fail"
        evaluations["1"] = result(
            series_decision,
            artifacts["wiring_source"],
            str(series.get("reason") or "没有可用串联拓扑结论，按二分类规则归为 fail"),
            series.get("confidence"),
            stage_intervals={"circuit_wiring": intervals.get("circuit_wiring", [])},
            original_decision=series.get("decision"),
            binary_policy="historical_binary_lenient_or_deepseek_replay",
        )

    parallel = next((item.get("voltmeter_parallel") for item in episode_items if isinstance(item.get("voltmeter_parallel"), dict)), None)
    if parallel is None:
        parallel = artifacts["deepseek"].get(action["video_id"], {}).get("voltmeter_parallel", {})
    parallel_decision = str(parallel.get("decision") or "fail")
    if parallel_decision not in {"pass", "fail"}:
        parallel_decision = "fail"
    evaluations["2"] = result(
        parallel_decision,
        artifacts["wiring_source"],
        str(parallel.get("reason") or "没有可用并联结论，按二分类规则归为 fail"),
        parallel.get("confidence"),
        stage_intervals={"circuit_wiring": intervals.get("circuit_wiring", []), "recording": intervals.get("recording_1", [])},
        original_decision=parallel.get("decision"),
        binary_policy="historical_binary_lenient_or_deepseek_replay",
    )

    switch = artifacts["switch"].get(action["video_id"], {})
    switch_decision = artifact_binary(switch, "verdict", "decision", "result")
    evaluations["3"] = result(
        switch_decision,
        artifacts["switch_source"],
        str(switch.get("reason") or "未得到接线期间开关断开证据"),
        0.95 if switch.get("confidence") == "high" else 0.7,
        stage_intervals={"circuit_wiring": intervals.get("circuit_wiring", [])},
        original_verdict=switch.get("verdict"),
        evidence=switch.get("key_evidence", []),
    )

    polarity = artifacts["polarity"].get(action["video_id"], {})
    polarity_decision = artifact_binary(polarity, "result", "decision")
    evaluations["4"] = result(
        polarity_decision,
        artifacts["polarity_source"],
        str(polarity.get("reason") or "未发现正负接线柱正确证据"),
        polarity.get("confidence"),
        stage_intervals={"measurement_1": intervals.get("measurement_1", []), "measurement_2": intervals.get("measurement_2", []), "recording_1": intervals.get("recording_1", []), "recording_2": intervals.get("recording_2", [])},
        original_result=polarity.get("result"),
        focused_pointer_observation=polarity.get("focused_pointer_observation"),
    )

    pointer_decision = artifacts["opencv"].get(action["video_id"], {}).get("rubric_5", {}).get("decision", {})
    pointer_outcome = pointer_decision.get("automated_outcome")
    pointer_pass = pointer_outcome == "scored" and pointer_decision.get("predicted_score") == 1
    evaluations["5"] = result(
        "pass" if pointer_pass else "fail",
        artifacts["opencv_source"],
        "OpenCV 证据显示两块表均稳定正常右偏" if pointer_pass else "指针专用 OpenCV 证据未形成稳定有效的双表正常偏转结论",
        0.8 if pointer_pass else 0.2,
        stage_intervals={"measurement_1": intervals.get("measurement_1", []), "measurement_2": intervals.get("measurement_2", []), "recording_1": intervals.get("recording_1", []), "recording_2": intervals.get("recording_2", [])},
        original_outcome=pointer_outcome,
        original_reason=pointer_decision.get("reason"),
        binary_mapping="abstained_to_fail" if pointer_outcome == "abstained" else "opencv_scored",
    )

    opencv_item = artifacts["opencv"].get(action["video_id"], {})
    opencv_decision = opencv_item.get("rubric_6", {}).get("decision", {})
    range_pass = (
        isinstance(opencv_decision, dict)
        and opencv_decision.get("automated_outcome") == "scored"
        and opencv_decision.get("predicted_score") == 1
    )
    evaluations["6"] = result(
        "pass" if range_pass else "fail",
        artifacts["opencv_source"],
        "量程选择器、端子与读数证据形成一致结论" if range_pass else "量程选择器和端点证据未形成有效一致结论，按二分类默认归为 fail",
        0.8 if range_pass else 0.2,
        stage_intervals={"measurement_1": intervals.get("measurement_1", []), "measurement_2": intervals.get("measurement_2", [])},
        original_outcome=opencv_decision.get("automated_outcome"),
        original_reason=opencv_decision.get("reason"),
        binary_mapping="opencv_scored_or_fail",
    )

    first = artifacts["first_record"].get(action["video_id"], {})
    first_decision = artifact_binary(first, "result", "decision")
    evaluations["7"] = result(
        first_decision,
        artifacts["first_record_source"],
        str(first.get("reason") or "未形成第一组 U/I 记录证据"),
        first.get("confidence"),
        stage_intervals={"recording_1": intervals.get("recording_1", [])},
        original_result=first.get("result"),
        u_value=first.get("u_value"),
        i_value=first.get("i_value"),
        evidence_seconds=first.get("evidence_seconds", []),
    )

    battery = artifacts["battery"].get(action["video_id"], {})
    battery_decision = "pass" if battery.get("decision") == "pass" else "fail"
    battery_reason = battery.get("reason")
    if not battery_reason and isinstance(battery.get("reducer"), dict):
        reducer = battery["reducer"]
        battery_reason = reducer.get("reason")
    if not battery_reason:
        battery_reason = "未观察到完整的断开→端子换接→重新闭合时序"
    evaluations["8"] = result(
        battery_decision,
        battery.get("source_artifact", artifacts["battery_source"]),
        str(battery_reason),
        battery.get("confidence"),
        stage_intervals={"circuit_rewiring": intervals.get("circuit_rewiring", [])},
        original_decision=battery.get("decision"),
        dedicated_sequence_scorer=True,
        sequence_reason_code=battery.get("reason_code"),
        episodes=battery.get("episodes", []),
        uncertainty_note=battery.get("uncertainty_note"),
        binary_policy="terminal_rewire_local_reducer",
    )

    second_intervals = intervals.get("recording_2", [])
    second_evidence = artifacts["second_record"].get(action["video_id"], {})
    second_decision = artifact_binary(second_evidence, "result", "decision")
    evaluations["9"] = result(
        second_decision,
        artifacts["second_record_source"],
        str(second_evidence.get("reason") or "未找到同时绑定第二次测量语境、两块仪表读数和第二组纸面 U/I 数值的有效证据；按二分类规则归为 fail"),
        second_evidence.get("confidence", 0.25 if second_intervals else 0.15),
        stage_intervals={"recording_2": second_intervals, "measurement_2": intervals.get("measurement_2", [])},
        original_evidence_status=second_evidence.get("status", "artifact_not_available"),
        binary_mapping="second_record_artifact_or_fail",
    )
    return evaluations


def markdown(rows: list[dict[str, Any]]) -> str:
    lines = ["# v2 十项评价结果", "", "| 视频 ID | " + " | ".join(name for _, _, name in RUBRICS) + " |", "|---|" + "---|" * len(RUBRICS)]
    for row in rows:
        cells = [row["video_id"]]
        cells.extend(row["evaluations"][str(index)]["decision"] for index, _, _ in RUBRICS)
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(["", "说明：所有主结果统一为 pass/fail；原始 abstained、证据质量和来源保存在逐视频 JSON 的 diagnostics 中。"])
    return "\n".join(lines) + "\n"


def optional_path(value: Any) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return resolve(value)


def source_label(path: Path | None) -> str:
    return str(path.resolve()) if path is not None else "artifact_not_provided"


def load_deepseek_results(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.is_file():
        return {}
    document = read_json(path)
    return {
        str(item.get("video_prefix") or ""): item.get("result", {})
        for item in document.get("records", [])
        if isinstance(item, dict) and isinstance(item.get("result"), dict)
    }


def load_opencv_results(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.is_file():
        return {}
    return {
        video_id_from_name(str(item.get("video_id") or "")): item
        for item in read_json(path).get("videos", [])
        if isinstance(item, dict) and item.get("video_id") is not None
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action-summary", type=Path, required=True)
    parser.add_argument("--artifact-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    action_summary_path = resolve(args.action_summary)
    artifact_config_path = resolve(args.artifact_config)
    output_root = resolve(args.output_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty output directory: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    action_summary = read_json(action_summary_path)
    action_map = load_action_results(action_summary)
    artifact_config = read_json(artifact_config_path)
    configured = artifact_config.get("artifacts", {})
    if not isinstance(configured, dict):
        raise ValueError("artifact_config_artifacts_missing")
    wiring_path = optional_path(configured.get("wiring_episodes"))
    deepseek_path = optional_path(configured.get("deepseek"))
    switch_path = optional_path(configured.get("switch"))
    polarity_root = optional_path(configured.get("polarity_root"))
    first_record_path = optional_path(configured.get("first_record"))
    opencv_path = optional_path(configured.get("opencv"))
    battery_path = optional_path(configured.get("rubric8"))
    second_path = optional_path(configured.get("second_record"))
    series_path = optional_path(configured.get("ammeter_source_series"))
    ammeter_source_series = load_ammeter_source_series_results(series_path) if series_path is not None else {}
    artifacts = {
        "ammeter_source_series": ammeter_source_series,
        "wiring_episodes": load_episode_map(wiring_path) if wiring_path is not None and wiring_path.is_file() else {},
        "deepseek": load_deepseek_results(deepseek_path),
        "switch": map_manual_switch(switch_path) if switch_path is not None and switch_path.is_file() else {},
        "polarity": map_result_files(polarity_root) if polarity_root is not None and polarity_root.is_dir() else {},
        "first_record": load_video_map(first_record_path) if first_record_path is not None and first_record_path.is_file() else {},
        "opencv": load_opencv_results(opencv_path),
        "battery": load_battery_results(battery_path) if battery_path is not None and battery_path.is_file() else {},
        "second_record": load_video_map(second_path) if second_path is not None and second_path.is_file() else {},
        "wiring_source": source_label(wiring_path),
        "series_source": source_label(series_path),
        "switch_source": source_label(switch_path),
        "polarity_source": source_label(polarity_root),
        "first_record_source": source_label(first_record_path),
        "opencv_source": source_label(opencv_path),
        "battery_source": source_label(battery_path),
        "second_record_source": source_label(second_path),
    }
    rows: list[dict[str, Any]] = []
    for vid, action in sorted(action_map.items()):
        evaluations = build_evaluations({"video_id": vid, **action}, artifacts)
        row = {
            "video_id": vid,
            "source_video": action["source_video"],
            "action_result_path": action["result_path"],
            "stage_intervals": action["intervals"],
            "evaluations": evaluations,
        }
        rows.append(row)
        write_json(output_root / f"video_{vid}" / "result.json", row)
    summary = {
        "schema_version": "all_rubrics_v2.v1",
        "algorithm_id": "resistance_7stage_no_battery_v2_ten_rubric_binary_replay",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "action_summary": str(action_summary_path.resolve()),
        "artifact_config": str(artifact_config_path.resolve()),
        "source_artifacts": {key: value for key, value in artifacts.items() if key.endswith("_source")},
        "video_count": len(rows),
        "evaluation_count": len(RUBRICS),
        "record_count": len(rows) * len(RUBRICS),
        "rows": rows,
        "decision_counts": {
            str(index): {
                "pass": sum(row["evaluations"][str(index)]["decision"] == "pass" for row in rows),
                "fail": sum(row["evaluations"][str(index)]["decision"] == "fail" for row in rows),
            }
            for index, _, _ in RUBRICS
        },
        "excel_accessed": False,
        "source_videos_modified": False,
    }
    write_json(output_root / "summary.json", summary)
    (output_root / "summary.md").write_text(markdown(rows), encoding="utf-8")
    print(json.dumps({"status": "completed", "videos": len(rows), "evaluations_per_video": len(RUBRICS), "output": str(output_root.resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
