"""Identity, temporal, and visual robustness checks for the live R3 Agent."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import cv2
import numpy as np

try:
    from .opencv_switch_overlap import analyze_opencv_switch_overlap
    from .r3_frame_agent_adapter import _load_current_stage_runs
    from .r3_frame_sampling_agent import run_r3_frame_sampling_agent
    from .switch_rubric import candidate_windows
except ImportError:
    from opencv_switch_overlap import analyze_opencv_switch_overlap  # type: ignore
    from r3_frame_agent_adapter import _load_current_stage_runs  # type: ignore
    from r3_frame_sampling_agent import run_r3_frame_sampling_agent  # type: ignore
    from switch_rubric import candidate_windows  # type: ignore


SCHEMA_VERSION = "resistance_agent_r3_stress_suite.v1"
WIRING_STAGES = {"circuit_wiring", "circuit_rewiring"}
QUALITY_VARIANTS = {"1080p", "720p", "blur", "brightness", "recompress"}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def video_metadata(video_path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"unable to open video: {video_path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    if fps <= 0.0 or frame_count <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(f"invalid video metadata: {video_path}")
    return {
        "fps": fps,
        "frame_count": frame_count,
        "duration_seconds": frame_count / fps,
        "width": width,
        "height": height,
    }


def load_current_stages(video_path: Path, stage_summary_path: Path) -> list[dict[str, Any]]:
    stages, _, _, _ = _load_current_stage_runs(stage_summary_path, video_path)
    return [dict(item) for item in stages if isinstance(item, dict)]


def write_stage_variant(
    *,
    source_stages: Iterable[dict[str, Any]],
    target_path: Path,
    target_video_name: str,
    duration_seconds: float,
    source_fps: float,
    wiring_shift_seconds: float = 0.0,
) -> Path:
    stages: list[dict[str, Any]] = []
    for raw in source_stages:
        item = dict(raw)
        stage = str(item.get("stage") or item.get("label") or "")
        if stage in WIRING_STAGES:
            start = max(0.0, float(item.get("start_seconds") or 0.0) + wiring_shift_seconds)
            end = min(
                duration_seconds,
                float(item.get("end_seconds") or start) + wiring_shift_seconds,
            )
            if end <= start:
                end = min(duration_seconds, start + max(1.0 / source_fps, 0.001))
            item["start_seconds"] = round(start, 6)
            item["end_seconds"] = round(end, 6)
            item["start_frame_number"] = int(round(start * source_fps))
            item["end_frame_number"] = min(
                int(round(end * source_fps)),
                max(0, int(math.ceil(duration_seconds * source_fps)) - 1),
            )
            item["start_frame_id"] = f"frame_{item['start_frame_number']:08d}"
            item["end_frame_id"] = f"frame_{item['end_frame_number']:08d}"
        stages.append(item)
    write_json(
        target_path,
        {
            "schema_version": "resistance_agent_r3_stress_stage_variant.v1",
            "source_video_id": target_video_name,
            "source_observed_stage_runs": stages,
            "wiring_shift_seconds": wiring_shift_seconds,
            "selection_basis": "current_video_observed_situation_only",
            "video_id_used_for_routing": False,
            "historical_artifacts_used": False,
            "fixed_video_roi_used": False,
        },
    )
    return target_path


def _window_mode(stages: Iterable[dict[str, Any]]) -> str:
    labels = {str(item.get("stage") or item.get("label") or "") for item in stages}
    if "circuit_rewiring" in labels:
        return "all_wiring_runs"
    if "circuit_wiring" in labels:
        return "initial_wiring_only"
    return "broad_search"


def _phase_analyzer(phase_offset_seconds: float) -> Callable[..., dict[str, Any]]:
    def analyze(**kwargs: Any) -> dict[str, Any]:
        return analyze_opencv_switch_overlap(
            **kwargs,
            sampling_phase_offset_seconds=phase_offset_seconds,
        )

    return analyze


def summarize_agent_report(report_path: Path) -> dict[str, Any]:
    report = read_json(report_path)
    baseline_path = Path(str(report.get("baseline_report_path") or "")).resolve()
    if not baseline_path.is_file():
        raise FileNotFoundError(f"baseline report missing: {baseline_path}")
    baseline = read_json(baseline_path)
    initial = report.get("initial_evidence_quality") or {}
    final = report.get("final_evidence_quality") or {}
    artifact_files = [path for path in report_path.parent.rglob("*") if path.is_file()]
    artifact_mtimes = [path.stat().st_mtime for path in artifact_files]
    runtime_seconds = (
        round(max(artifact_mtimes) - min(artifact_mtimes), 3)
        if artifact_mtimes
        else 0.0
    )
    return {
        "decision": report.get("decision"),
        "predicted_score": report.get("predicted_score"),
        "confidence": report.get("confidence"),
        "stop_reason": report.get("stop_reason"),
        "candidate_window_count": len(report.get("candidate_windows") or []),
        "baseline_sample_count": baseline.get("sample_count"),
        "baseline_decision": baseline.get("decision"),
        "baseline_confidence": baseline.get("confidence"),
        "baseline_switch_observation_count": baseline.get("switch_tracked_observation_count"),
        "baseline_switch_coverage": baseline.get("switch_coverage"),
        "baseline_sampling_phase_offset_seconds": baseline.get(
            "sampling_phase_offset_seconds", 0.0
        ),
        "final_sample_count": final.get("sample_count"),
        "final_switch_observation_count": final.get("switch_observation_count"),
        "final_switch_coverage": final.get("switch_coverage"),
        "request_round_count": len(report.get("request_rounds") or []),
        "request_count": report.get("request_count"),
        "supplemental_actual_new_frame_count": report.get(
            "supplemental_actual_new_frame_count"
        ),
        "evidence_frame_count": report.get("evidence_frame_count"),
        "runtime_seconds": runtime_seconds,
        "runtime_source": "agent_output_artifact_mtime_span",
        "initial_reasons": initial.get("reasons") or [],
        "final_reasons": final.get("reasons") or [],
        "video_id_used_for_routing": report.get("video_id_used_for_routing"),
        "historical_artifacts_used": report.get("historical_artifacts_used"),
        "fixed_video_roi_used": report.get("fixed_video_roi_used"),
        "excel_accessed": report.get("excel_accessed"),
        "ground_truth_sent_to_model": report.get("ground_truth_sent_to_model"),
        "report_path": str(report_path.resolve()),
    }


def run_agent_case(
    *,
    case_id: str,
    video_path: Path,
    stage_summary_path: Path,
    output_dir: Path,
    sampling_phase_offset_seconds: float = 0.0,
    max_rounds: int = 2,
    max_requests_per_round: int = 3,
    max_supplemental_frames: int = 64,
) -> dict[str, Any]:
    case_dir = output_dir / case_id
    if case_dir.exists():
        raise FileExistsError(f"refusing to reuse stress case: {case_dir}")
    metadata = video_metadata(video_path)
    stages = load_current_stages(video_path, stage_summary_path)
    window_mode = _window_mode(stages)
    windows = candidate_windows(
        {"observed_stage_runs": stages},
        float(metadata["duration_seconds"]),
        window_mode,
    )
    started = time.perf_counter()
    result = run_r3_frame_sampling_agent(
        video_path=video_path,
        candidate_windows=windows,
        output_dir=case_dir / "agent",
        max_rounds=max_rounds,
        max_requests_per_round=max_requests_per_round,
        max_supplemental_frames=max_supplemental_frames,
        analyzer=_phase_analyzer(sampling_phase_offset_seconds),
    )
    elapsed = time.perf_counter() - started
    summary = summarize_agent_report(Path(result["report_path"]))
    record = {
        "schema_version": SCHEMA_VERSION,
        "case_id": case_id,
        "video_path": str(video_path.resolve()),
        "video_sha256": sha256(video_path),
        "source_video_name": video_path.name,
        "stage_summary_path": str(stage_summary_path.resolve()),
        "window_mode": window_mode,
        "candidate_windows": windows,
        "sampling_phase_offset_seconds": sampling_phase_offset_seconds,
        "runtime_seconds": round(elapsed, 3),
        "metrics": summary,
        "selection_basis": "current_video_observed_situation_only",
        "video_id_used_for_routing": False,
        "historical_artifacts_used": False,
        "fixed_video_roi_used": False,
        "excel_accessed": False,
        "ground_truth_sent_to_model": False,
    }
    write_json(case_dir / "case_result.json", record)
    return record


def run_identity_test(
    *,
    video_path: Path,
    stage_summary_path: Path,
    output_dir: Path,
    random_seed: int = 20260818,
    max_rounds: int = 2,
    max_requests_per_round: int = 3,
    max_supplemental_frames: int = 64,
) -> dict[str, Any]:
    metadata = video_metadata(video_path)
    stages = load_current_stages(video_path, stage_summary_path)
    fixture_dir = output_dir / "identity_fixture"
    fixture_dir.mkdir(parents=True, exist_ok=False)
    original_summary = write_stage_variant(
        source_stages=stages,
        target_path=fixture_dir / "original_stages.json",
        target_video_name=video_path.name,
        duration_seconds=float(metadata["duration_seconds"]),
        source_fps=float(metadata["fps"]),
    )
    rng = random.Random(random_seed)
    alias_name = f"anonymous_{rng.getrandbits(64):016x}{video_path.suffix.lower()}"
    alias_video = fixture_dir / alias_name
    shutil.copy2(video_path, alias_video)
    if sha256(alias_video) != sha256(video_path):
        raise RuntimeError("identity fixture copy hash mismatch")
    alias_summary = write_stage_variant(
        source_stages=stages,
        target_path=fixture_dir / "alias_stages.json",
        target_video_name=alias_name,
        duration_seconds=float(metadata["duration_seconds"]),
        source_fps=float(metadata["fps"]),
    )
    original = run_agent_case(
        case_id="identity_original",
        video_path=video_path,
        stage_summary_path=original_summary,
        output_dir=output_dir,
        max_rounds=max_rounds,
        max_requests_per_round=max_requests_per_round,
        max_supplemental_frames=max_supplemental_frames,
    )
    alias = run_agent_case(
        case_id="identity_alias",
        video_path=alias_video,
        stage_summary_path=alias_summary,
        output_dir=output_dir,
        max_rounds=max_rounds,
        max_requests_per_round=max_requests_per_round,
        max_supplemental_frames=max_supplemental_frames,
    )
    comparable_fields = (
        "decision",
        "predicted_score",
        "baseline_sample_count",
        "final_sample_count",
        "baseline_switch_observation_count",
        "final_switch_observation_count",
        "request_count",
        "supplemental_actual_new_frame_count",
    )
    differences = {
        field: [original["metrics"].get(field), alias["metrics"].get(field)]
        for field in comparable_fields
        if original["metrics"].get(field) != alias["metrics"].get(field)
    }
    result = {
        "schema_version": "resistance_agent_r3_identity_test.v1",
        "source_sha256": sha256(video_path),
        "alias_sha256": sha256(alias_video),
        "alias_name": alias_name,
        "same_content": sha256(video_path) == sha256(alias_video),
        "same_candidate_windows": original["candidate_windows"] == alias["candidate_windows"],
        "same_parameters": True,
        "same_decision": original["metrics"]["decision"] == alias["metrics"]["decision"],
        "metric_differences": differences,
        "passed": not differences
        and original["candidate_windows"] == alias["candidate_windows"],
        "video_id_used_for_routing": False,
        "historical_artifacts_used": False,
        "fixed_video_roi_used": False,
    }
    write_json(output_dir / "identity_test.json", result)
    return result


def run_temporal_test(
    *,
    video_path: Path,
    stage_summary_path: Path,
    output_dir: Path,
    phase_offsets: Iterable[float] = (-0.1, 0.1),
    boundary_shifts: Iterable[float] = (-5.0, -2.0, 2.0, 5.0),
    max_rounds: int = 2,
    max_requests_per_round: int = 3,
    max_supplemental_frames: int = 64,
) -> dict[str, Any]:
    metadata = video_metadata(video_path)
    stages = load_current_stages(video_path, stage_summary_path)
    fixture_dir = output_dir / "temporal_fixtures"
    fixture_dir.mkdir(parents=True, exist_ok=False)
    base_summary = write_stage_variant(
        source_stages=stages,
        target_path=fixture_dir / "base.json",
        target_video_name=video_path.name,
        duration_seconds=float(metadata["duration_seconds"]),
        source_fps=float(metadata["fps"]),
    )
    baseline = run_agent_case(
        case_id="temporal_baseline",
        video_path=video_path,
        stage_summary_path=base_summary,
        output_dir=output_dir,
        max_rounds=max_rounds,
        max_requests_per_round=max_requests_per_round,
        max_supplemental_frames=max_supplemental_frames,
    )
    cases: list[dict[str, Any]] = []
    for offset in phase_offsets:
        case = run_agent_case(
            case_id=f"phase_{offset:+.1f}".replace("+", "plus_").replace("-", "minus_"),
            video_path=video_path,
            stage_summary_path=base_summary,
            output_dir=output_dir,
            sampling_phase_offset_seconds=float(offset),
            max_rounds=max_rounds,
            max_requests_per_round=max_requests_per_round,
            max_supplemental_frames=max_supplemental_frames,
        )
        cases.append({"kind": "phase", "offset_seconds": offset, **case})
    for shift in boundary_shifts:
        shifted_summary = write_stage_variant(
            source_stages=stages,
            target_path=fixture_dir
            / f"boundary_{shift:+.1f}.json".replace("+", "plus_").replace("-", "minus_"),
            target_video_name=video_path.name,
            duration_seconds=float(metadata["duration_seconds"]),
            source_fps=float(metadata["fps"]),
            wiring_shift_seconds=float(shift),
        )
        case = run_agent_case(
            case_id=f"boundary_{shift:+.1f}".replace("+", "plus_").replace("-", "minus_"),
            video_path=video_path,
            stage_summary_path=shifted_summary,
            output_dir=output_dir,
            max_rounds=max_rounds,
            max_requests_per_round=max_requests_per_round,
            max_supplemental_frames=max_supplemental_frames,
        )
        cases.append({"kind": "boundary", "offset_seconds": shift, **case})
    baseline_decision = baseline["metrics"]["decision"]
    result = {
        "schema_version": "resistance_agent_r3_temporal_test.v1",
        "baseline_decision": baseline_decision,
        "cases": cases,
        "decision_flip_count": sum(
            item["metrics"]["decision"] != baseline_decision for item in cases
        ),
        "phase_period_seconds": 0.2,
        "note": "At 5 fps, -0.1s and +0.1s normalize to the same interior sampling phase.",
        "excel_accessed": False,
    }
    write_json(output_dir / "temporal_test.json", result)
    return result


def create_quality_variant(source: Path, target: Path, variant: str) -> dict[str, Any]:
    if variant not in QUALITY_VARIANTS:
        raise ValueError(f"unsupported quality variant: {variant}")
    metadata = video_metadata(source)
    source_width = int(metadata["width"])
    source_height = int(metadata["height"])
    if variant == "1080p":
        scale = min(1.0, 1080.0 / source_height)
    elif variant == "720p":
        scale = min(1.0, 720.0 / source_height)
    else:
        scale = 1.0
    width = max(2, int(round(source_width * scale / 2.0)) * 2)
    height = max(2, int(round(source_height * scale / 2.0)) * 2)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"unable to open source video: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(target),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(metadata["fps"]),
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"unable to open video writer: {target}")
    frame_count = 0
    started = time.perf_counter()
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            if (frame.shape[1], frame.shape[0]) != (width, height):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            if variant == "blur":
                frame = cv2.GaussianBlur(frame, (5, 5), 1.2)
            elif variant == "brightness":
                frame = np.clip(frame.astype(np.int16) + 25, 0, 255).astype(np.uint8)
            writer.write(frame)
            frame_count += 1
    finally:
        capture.release()
        writer.release()
    if frame_count != int(metadata["frame_count"]):
        raise RuntimeError(
            f"quality variant frame count drifted: {frame_count} != {metadata['frame_count']}"
        )
    reopened = video_metadata(target)
    return {
        "variant": variant,
        "path": str(target.resolve()),
        "sha256": sha256(target),
        "bytes": target.stat().st_size,
        "width": reopened["width"],
        "height": reopened["height"],
        "fps": reopened["fps"],
        "frame_count": reopened["frame_count"],
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "audio_preserved": False,
        "source_unchanged": True,
    }


def run_quality_test(
    *,
    video_path: Path,
    stage_summary_path: Path,
    output_dir: Path,
    variants: Iterable[str] = ("1080p", "720p", "blur", "brightness", "recompress"),
    run_agent: bool = True,
    max_rounds: int = 2,
    max_requests_per_round: int = 3,
    max_supplemental_frames: int = 64,
) -> dict[str, Any]:
    metadata = video_metadata(video_path)
    stages = load_current_stages(video_path, stage_summary_path)
    variant_root = output_dir / "quality_variants"
    variant_root.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, Any]] = []
    for variant in variants:
        variant = str(variant)
        target = variant_root / variant / f"quality_{variant}.mp4"
        generated = create_quality_variant(video_path, target, variant)
        summary = write_stage_variant(
            source_stages=stages,
            target_path=target.parent / "stages.json",
            target_video_name=target.name,
            duration_seconds=float(metadata["duration_seconds"]),
            source_fps=float(metadata["fps"]),
        )
        case = None
        if run_agent:
            case = run_agent_case(
                case_id=f"quality_{variant}",
                video_path=target,
                stage_summary_path=summary,
                output_dir=output_dir,
                max_rounds=max_rounds,
                max_requests_per_round=max_requests_per_round,
                max_supplemental_frames=max_supplemental_frames,
            )
        records.append({"generated": generated, "agent_case": case})
    result = {
        "schema_version": "resistance_agent_r3_quality_test.v1",
        "source_video_path": str(video_path.resolve()),
        "source_video_sha256": sha256(video_path),
        "variants": records,
        "excel_accessed": False,
    }
    write_json(output_dir / "quality_test.json", result)
    return result


def aggregate_reports(
    labeled_reports: Iterable[tuple[str, Path]], output_dir: Path
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for label, report_path in labeled_reports:
        metrics = summarize_agent_report(report_path)
        rows.append({"video_label": str(label), **metrics})
    rows.sort(key=lambda item: item["video_label"])
    summary = {
        "schema_version": "resistance_agent_r3_fixed_agent_comparison.v1",
        "evaluation_scope": "development_and_cross_experiment_stability_only",
        "accuracy_claimed": False,
        "video_count": len(rows),
        "decision_change_count": sum(
            item["baseline_decision"] != item["decision"] for item in rows
        ),
        "total_baseline_samples": sum(int(item["baseline_sample_count"] or 0) for item in rows),
        "total_final_samples": sum(int(item["final_sample_count"] or 0) for item in rows),
        "total_requests": sum(int(item["request_count"] or 0) for item in rows),
        "total_supplemental_new_frames": sum(
            int(item["supplemental_actual_new_frame_count"] or 0) for item in rows
        ),
        "total_runtime_seconds": round(
            sum(float(item["runtime_seconds"] or 0.0) for item in rows), 3
        ),
        "rows": rows,
        "excel_accessed": False,
        "ground_truth_read": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "summary.json", summary)
    if rows:
        with (output_dir / "summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return summary
