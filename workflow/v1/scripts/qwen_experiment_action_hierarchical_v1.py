#!/usr/bin/env python3
"""Hierarchical Map/Reduce action segmentation without a battery stage.

This is an independent pipeline.  It reads a locked experiment interval from
the mature start/end output, prepares overlapping Map windows, asks Qwen only
for visible base actions, assigns seven stages locally, and optionally refines
candidate boundaries.  Every run must use a new output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2

import qwen_experiment_segment_judge as qwen_base
from qwen_hierarchical_v1_contract import (
    STAGE_SCHEMA_ID,
    build_overlapping_windows,
    create_run_directory,
    load_stage_schema,
    normalize_map_events,
    read_json,
    sample_timestamps,
    select_source_records,
    source_frame_id,
    utc_now,
    validate_boundary_response,
    validate_map_response,
    validate_reduce_response,
    write_json_atomic,
)
from qwen_hierarchical_v1_prompts import (
    build_boundary_prompt,
    build_boundary_retry_prompt,
    build_map_prompt,
    build_map_retry_prompt,
    build_reduce_prompt,
    build_reduce_retry_prompt,
)
from qwen_hierarchical_v1_reduce import (
    assign_seven_stages,
    build_boundary_candidates,
    build_evidence_timeline,
    deduplicate_map_events,
    find_temporal_conflicts,
    merge_observed_stage_runs,
    salvage_reduce_response,
    select_events,
)


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SCHEMA = ROOT / "configs" / "action_schemas" / "resistance_7stage_no_battery_v1.json"
DEFAULT_SEGMENT_SOURCE = ROOT / "outputs" / "experiment_boundary" / "summary.json"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "qwen_experiment_action_hierarchical_v1"
ALGORITHM_ID = "qwen_experiment_action_hierarchical_v1"
ALGORITHM_SCHEMA_VERSION = "qwen_experiment_action_hierarchical_v1.v1"


def safe_video_directory_name(video_id: str) -> str:
    """Keep per-video output names unique and confined to the run directory."""
    slugged = qwen_base.slug(video_id).strip("._-") or "video"
    digest = hashlib.sha256(video_id.encode("utf-8")).hexdigest()[:12]
    return f"{slugged[:80]}__{digest}"


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _video_metadata(manifest: dict[str, Any]) -> tuple[Path, float, int]:
    source = Path(str(manifest.get("source_video", "")))
    metadata = manifest.get("video_metadata")
    if not source.is_file():
        raise FileNotFoundError(f"source_video_not_found:{source}")
    if not isinstance(metadata, dict):
        raise ValueError("video_metadata_invalid")
    fps = metadata.get("fps")
    frame_count = metadata.get("frame_count")
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(float(fps)) or float(fps) <= 0:
        raise ValueError("video_fps_invalid")
    if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count <= 0:
        raise ValueError("video_frame_count_invalid")
    return source, float(fps), frame_count


def _frame_numbers_for_range(
    start_seconds: float,
    end_seconds: float,
    interval_seconds: float,
    fps: float,
    frame_count: int,
) -> list[int]:
    values: list[int] = []
    seen: set[int] = set()
    for timestamp in sample_timestamps(start_seconds, end_seconds, interval_seconds):
        frame_number = min(frame_count - 1, max(0, int(round(timestamp * fps))))
        if frame_number not in seen:
            values.append(frame_number)
            seen.add(frame_number)
    return values


def _extract_source_frames(
    manifest: dict[str, Any],
    frame_numbers: list[int],
    frames_dir: Path,
    max_model_edge: int,
    registry: dict[int, dict[str, Any]],
) -> None:
    missing = sorted(set(frame_numbers) - set(registry))
    if not missing:
        return
    source, fps, frame_count = _video_metadata(manifest)
    if any(frame_number < 0 or frame_number >= frame_count for frame_number in missing):
        raise ValueError("requested_frame_outside_video")
    frames_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"unable_to_open_source_video:{source}")
    try:
        # Decode each video in one forward pass. Re-seeking a 4K H.264 file for
        # every two-second sample repeatedly decodes the same GOPs and is much
        # slower than advancing between the already sorted target frames.
        current_frame = missing[0]
        capture.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        for frame_number in missing:
            if frame_number < current_frame:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                current_frame = frame_number
            while current_frame < frame_number:
                if not capture.grab():
                    raise RuntimeError(f"unable_to_advance_source_frame:{frame_number}")
                current_frame += 1
            ok, frame = capture.read()
            current_frame = frame_number + 1
            if not ok or frame is None:
                raise RuntimeError(f"unable_to_read_source_frame:{frame_number}")
            image_id = source_frame_id(frame_number)
            timestamp_seconds = frame_number / fps
            image = qwen_base.resize_for_model(frame, max_model_edge)
            image = qwen_base.add_relative_timestamp_banner(image, image_id, timestamp_seconds)
            path = frames_dir / f"{image_id}_{timestamp_seconds:010.3f}s.jpg"
            if path.exists():
                raise FileExistsError(f"refusing_to_overwrite_frame:{path}")
            if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 84]):
                raise RuntimeError(f"unable_to_write_frame:{path}")
            registry[frame_number] = {
                "image_id": image_id,
                "frame_number": frame_number,
                "timestamp_seconds": timestamp_seconds,
                "path": str(path.resolve()),
                "relative_timestamp_banner": f"FRAME ID={image_id} | VIDEO T={timestamp_seconds:.1f}s",
                "banner_position": "new_bottom_information_bar_left_aligned",
                "source_pixels_occluded": False,
            }
    finally:
        capture.release()


def prepare_video(
    provenance: dict[str, Any],
    video_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    video_id = str(provenance["source_video_id"])
    manifest_path = Path(str(provenance["source_manifest"]))
    manifest = read_json(manifest_path)
    if manifest.get("source_video_id") != video_id:
        raise ValueError("source_manifest_video_id_mismatch")
    source, fps, frame_count = _video_metadata(manifest)
    segment = provenance["source_segment"]
    requested_start = float(segment["start_seconds"])
    requested_end = float(segment["end_seconds"])
    last_frame_seconds = (frame_count - 1) / fps
    half_frame_tolerance = 0.5 / fps + 1e-6
    if requested_start < -half_frame_tolerance or requested_end > last_frame_seconds + half_frame_tolerance:
        raise ValueError("locked_segment_outside_real_video_frames")
    start_frame = min(frame_count - 1, max(0, int(round(requested_start * fps))))
    end_frame = min(frame_count - 1, max(0, int(round(requested_end * fps))))
    if start_frame >= end_frame:
        raise ValueError("locked_segment_collapses_after_frame_snap")
    fixed_start = start_frame / fps
    fixed_end = end_frame / fps
    windows = build_overlapping_windows(
        fixed_start,
        fixed_end,
        args.window_seconds,
        args.overlap_seconds,
    )
    window_frame_numbers: dict[str, list[int]] = {}
    all_frame_numbers: set[int] = set()
    for window in windows:
        start, end = window["window_seconds"]
        numbers = _frame_numbers_for_range(
            float(start),
            float(end),
            args.sample_interval_seconds,
            fps,
            frame_count,
        )
        window_frame_numbers[str(window["window_id"])] = numbers
        all_frame_numbers.update(numbers)
    frame_registry: dict[int, dict[str, Any]] = {}
    frames_dir = video_dir / "frames" / "source"
    _extract_source_frames(
        manifest,
        sorted(all_frame_numbers),
        frames_dir,
        args.max_model_edge,
        frame_registry,
    )
    prepared_windows: list[dict[str, Any]] = []
    for window in windows:
        window_id = str(window["window_id"])
        frames = [frame_registry[number] for number in window_frame_numbers[window_id]]
        prompt = build_map_prompt(video_id, window, frames)
        window_dir = video_dir / "map" / "windows" / window_id
        input_record = {
            "schema_version": ALGORITHM_SCHEMA_VERSION,
            "algorithm_id": ALGORITHM_ID,
            "stage_schema_id": STAGE_SCHEMA_ID,
            **window,
            "sampling": {
                "sample_interval_seconds": args.sample_interval_seconds,
                "max_model_edge": args.max_model_edge,
                "timestamp_watermark": True,
                "watermark_position": "new_bottom_information_bar_left_aligned",
            },
            "input_frames": frames,
        }
        write_json_atomic(window_dir / "input.json", input_record)
        _write_text(window_dir / "prompt.txt", prompt)
        prepared_windows.append({**window, "frames": frames, "input_path": str((window_dir / "input.json").resolve()), "prompt_path": str((window_dir / "prompt.txt").resolve())})
    source_record = {
        "schema_version": ALGORITHM_SCHEMA_VERSION,
        "algorithm_id": ALGORITHM_ID,
        "stage_schema_id": STAGE_SCHEMA_ID,
        "source_video_id": video_id,
        "source_video": str(source.resolve()),
        "source_manifest": str(manifest_path.resolve()),
        "source_segment_provenance": provenance,
        "video_metadata": {
            "fps": fps,
            "frame_count": frame_count,
            "last_frame_seconds": last_frame_seconds,
        },
        "requested_locked_interval_seconds": [requested_start, requested_end],
        "locked_experiment_interval_seconds": [fixed_start, fixed_end],
        "locked_interval_frame_snap": {
            "start_frame_number": start_frame,
            "end_frame_number": end_frame,
            "start_adjustment_seconds": fixed_start - requested_start,
            "end_adjustment_seconds": fixed_end - requested_end,
        },
        "window_geometry": {
            "window_seconds": args.window_seconds,
            "overlap_seconds": args.overlap_seconds,
            "stride_seconds": args.window_seconds - args.overlap_seconds,
            "interpretation": "fixed_length_windows_with_adjacent_overlap",
        },
        "window_count": len(windows),
        "window_frame_reference_count": sum(len(item["frames"]) for item in prepared_windows),
        "unique_source_frame_count": len(frame_registry),
        "overlap_reference_savings": sum(len(item["frames"]) for item in prepared_windows) - len(frame_registry),
    }
    write_json_atomic(video_dir / "source.json", source_record)
    return {
        "video_id": video_id,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "video_dir": video_dir,
        "frames_dir": frames_dir,
        "frame_registry": frame_registry,
        "prepared_windows": prepared_windows,
        "source_record": source_record,
        "fixed_start": fixed_start,
        "fixed_end": fixed_end,
        "fps": fps,
        "frame_count": frame_count,
    }


def _call_qwen(
    client: Any,
    prompt: str,
    frames: list[dict[str, Any]],
    max_tokens: int,
) -> dict[str, Any]:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for frame in frames:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": qwen_base.image_data_url(Path(str(frame["path"])))},
            }
        )
    completion = client.chat.completions.create(
        model=qwen_base.MODEL,
        messages=[{"role": "user", "content": content}],
        max_tokens=max_tokens,
        temperature=0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    choice = completion.choices[0]
    raw = choice.message.content or ""
    result: dict[str, Any] = {
        "finish_reason": choice.finish_reason or "unknown",
        "raw_model_content": raw,
        "parsed": False,
    }
    try:
        result["parsed_result"] = qwen_base.parse_json(raw)
        result["parsed"] = True
    except (json.JSONDecodeError, ValueError) as exc:
        result["parse_error"] = str(exc)
    return result


def _attempt_qwen(
    client: Any,
    prompt: str,
    frames: list[dict[str, Any]],
    max_tokens: int,
) -> dict[str, Any]:
    try:
        return _call_qwen(client, prompt, frames, max_tokens)
    except Exception as exc:
        return {
            "finish_reason": "transport_error",
            "parsed": False,
            "transport_error_type": type(exc).__name__,
            "transport_error": str(exc),
        }


def _run_map(
    prepared: dict[str, Any],
    client: Any,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    all_events: list[dict[str, Any]] = []
    window_results: list[dict[str, Any]] = []
    review_reasons: list[str] = []
    for window in prepared["prepared_windows"]:
        window_id = str(window["window_id"])
        frames = list(window["frames"])
        base_prompt = build_map_prompt(prepared["video_id"], window, frames)
        attempts: list[dict[str, Any]] = []
        parsed: dict[str, Any] | None = None
        errors: list[str] = []
        raw: dict[str, Any] = {}
        for attempt_index in range(args.max_attempts):
            prompt = base_prompt if attempt_index == 0 else build_map_retry_prompt(base_prompt, errors)
            raw = _attempt_qwen(client, prompt, frames, args.map_max_tokens)
            candidate = raw.get("parsed_result")
            parsed = candidate if isinstance(candidate, dict) else None
            errors = validate_map_response(parsed, window_id, frames)
            attempts.append(
                {
                    "attempt_index": attempt_index + 1,
                    "qwen": raw,
                    "validation_errors": errors,
                }
            )
            if not errors:
                break
        events = normalize_map_events(parsed, window_id, frames) if parsed is not None and not errors else []
        all_events.extend(events)
        result = {
            "window_id": window_id,
            "window_seconds": window["window_seconds"],
            "valid": not errors,
            "validation_errors": errors,
            "attempts": attempts,
            "parsed_result": parsed,
            "normalized_events": events,
        }
        if errors:
            review_reasons.append(f"map_invalid:{window_id}:{','.join(errors)}")
        if isinstance(parsed, dict) and parsed.get("decision") == "uncertain":
            review_reasons.append(f"map_uncertain:{window_id}")
        window_dir = prepared["video_dir"] / "map" / "windows" / window_id
        write_json_atomic(window_dir / "result.json", result)
        window_results.append(result)
        print(
            json.dumps(
                {
                    "video": prepared["video_id"],
                    "map_window": window_id,
                    "valid": not errors,
                    "event_count": len(events),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    return all_events, window_results, review_reasons


def _run_reduce(
    prepared: dict[str, Any],
    map_events: list[dict[str, Any]],
    client: Any,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    canonical_events = deduplicate_map_events(map_events)
    pre_reduce_conflicts = find_temporal_conflicts(canonical_events)
    reduce_dir = prepared["video_dir"] / "reduce"
    candidate_record = {
        "canonical_events": canonical_events,
        "pre_reduce_temporal_conflicts": pre_reduce_conflicts,
    }
    write_json_atomic(reduce_dir / "candidate_events.json", candidate_record)
    base_prompt = build_reduce_prompt(prepared["video_id"], canonical_events)
    _write_text(reduce_dir / "prompt.txt", base_prompt)
    review_reasons: list[str] = []
    attempts: list[dict[str, Any]] = []
    parsed: dict[str, Any] | None = None
    errors: list[str] = []
    if canonical_events:
        for attempt_index in range(args.max_attempts):
            prompt = base_prompt if attempt_index == 0 else build_reduce_retry_prompt(base_prompt, errors)
            raw = _attempt_qwen(client, prompt, [], args.reduce_max_tokens)
            candidate = raw.get("parsed_result")
            parsed = candidate if isinstance(candidate, dict) else None
            errors = validate_reduce_response(parsed, canonical_events)
            attempts.append(
                {
                    "attempt_index": attempt_index + 1,
                    "qwen": raw,
                    "validation_errors": errors,
                }
            )
            if not errors:
                break
    else:
        parsed = {
            "accepted_event_ids": [],
            "rejected_events": [],
            "conflicts": [],
            "terminal_cleanup_event_id": None,
            "confidence": 0.0,
            "uncertainty": "Map 未产生可用动作事件。",
        }
        errors = []
        review_reasons.append("map_produced_no_events")
    original_errors = list(errors)
    valid_parsed = parsed if parsed is not None and not errors else None
    recovery: dict[str, Any] = {
        "policy": args.reduce_recovery_policy,
        "applied": False,
        "repairs": [],
        "ignored_noise_events": [],
        "validation_errors_after_recovery": [],
    }
    if args.reduce_recovery_policy == "local_partial" and parsed is not None:
        repaired, repairs = salvage_reduce_response(canonical_events, parsed)
        recovery_errors = validate_reduce_response(repaired, canonical_events) if repaired is not None else ["reduce_recovery_unavailable"]
        ignored_noise_events = repaired.get("ignored_noise_events", []) if repaired is not None else []
        recovery_applied = repaired is not None and not recovery_errors and bool(original_errors or repairs or ignored_noise_events)
        recovery.update(
            {
                "applied": recovery_applied,
                "repairs": repairs,
                "ignored_noise_events": ignored_noise_events,
                "validation_errors_after_recovery": recovery_errors,
            }
        )
        if repaired is not None and not recovery_errors and (valid_parsed is None or recovery_applied):
            valid_parsed = repaired
            if recovery_applied:
                review_reasons.append("reduce_locally_recovered")
    selected, selection = select_events(
        canonical_events,
        valid_parsed,
        preserve_equal_confidence=args.reduce_recovery_policy == "local_partial",
    )
    if original_errors and valid_parsed is None:
        review_reasons.append("reduce_invalid:" + ",".join(original_errors))
    elif original_errors:
        review_reasons.append("reduce_model_response_invalid_recovered:" + ",".join(original_errors))
    if pre_reduce_conflicts:
        review_reasons.append("pre_reduce_temporal_conflicts_present")
    if selection.get("needs_review"):
        review_reasons.append("reduce_selection_needs_review")
    result = {
        "valid": valid_parsed is not None,
        "model_response_valid": not original_errors,
        "validation_errors": original_errors,
        "attempts": attempts,
        "parsed_result": parsed,
        "effective_parsed_result": valid_parsed,
        "recovery": recovery,
        "selection": selection,
        "accepted_events": selected,
        "ignored_noise_events": recovery["ignored_noise_events"],
        "pre_reduce_temporal_conflicts": pre_reduce_conflicts,
    }
    write_json_atomic(reduce_dir / "result.json", result)
    return selected, result, review_reasons


def _run_boundary_pass(
    prepared: dict[str, Any],
    boundary: dict[str, Any],
    pass_id: str,
    range_start: float,
    range_end: float,
    sample_interval: float,
    client: Any,
    stage_labels: dict[str, str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    frame_numbers = _frame_numbers_for_range(
        range_start,
        range_end,
        sample_interval,
        prepared["fps"],
        prepared["frame_count"],
    )
    _extract_source_frames(
        prepared["manifest"],
        frame_numbers,
        prepared["frames_dir"],
        args.max_model_edge,
        prepared["frame_registry"],
    )
    frames = [prepared["frame_registry"][number] for number in frame_numbers]
    base_prompt = build_boundary_prompt(prepared["video_id"], boundary, frames, stage_labels)
    pass_dir = prepared["video_dir"] / "boundaries" / str(boundary["boundary_id"]) / pass_id
    write_json_atomic(
        pass_dir / "input.json",
        {
            "boundary": boundary,
            "range_seconds": [range_start, range_end],
            "sample_interval_seconds": sample_interval,
            "input_frames": frames,
        },
    )
    _write_text(pass_dir / "prompt.txt", base_prompt)
    attempts: list[dict[str, Any]] = []
    parsed: dict[str, Any] | None = None
    errors: list[str] = []
    for attempt_index in range(args.max_attempts):
        prompt = base_prompt if attempt_index == 0 else build_boundary_retry_prompt(base_prompt, errors)
        raw = _attempt_qwen(client, prompt, frames, args.boundary_max_tokens)
        candidate = raw.get("parsed_result")
        parsed = candidate if isinstance(candidate, dict) else None
        errors = validate_boundary_response(parsed, str(boundary["boundary_id"]), frames)
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
        "pass_id": pass_id,
        "range_seconds": [range_start, range_end],
        "sample_interval_seconds": sample_interval,
        "input_frames": frames,
        "valid": not errors,
        "validation_errors": errors,
        "attempts": attempts,
        "parsed_result": parsed,
    }
    write_json_atomic(pass_dir / "result.json", result)
    return result


def _observed_boundary_from_pass(
    boundary_pass: dict[str, Any],
    minimum_confidence: float,
) -> dict[str, Any] | None:
    parsed = boundary_pass.get("parsed_result")
    if not boundary_pass.get("valid") or not isinstance(parsed, dict) or parsed.get("decision") != "observed":
        return None
    by_id = {str(frame["image_id"]): frame for frame in boundary_pass["input_frames"]}
    last_from = by_id.get(parsed.get("last_from_frame_id"))
    first_to = by_id.get(parsed.get("first_to_frame_id"))
    if last_from is None or first_to is None:
        return None
    confidence = float(parsed["confidence"])
    uncertainty = str(parsed.get("uncertainty", "")).strip()
    has_uncertainty = uncertainty.lower() not in {"", "无", "none", "null", "n/a", "无不确定性"}
    return {
        "last_from_frame_id": last_from["image_id"],
        "first_to_frame_id": first_to["image_id"],
        "last_from_frame_number": last_from["frame_number"],
        "first_to_frame_number": first_to["frame_number"],
        "last_from_seconds": last_from["timestamp_seconds"],
        "first_to_seconds": first_to["timestamp_seconds"],
        "boundary_interval_seconds": [last_from["timestamp_seconds"], first_to["timestamp_seconds"]],
        "selected_seconds": first_to["timestamp_seconds"],
        "sampling_interval_seconds": boundary_pass["sample_interval_seconds"],
        "confidence": confidence,
        "evidence": parsed.get("evidence", ""),
        "uncertainty": uncertainty,
        "needs_review": confidence < minimum_confidence or has_uncertainty,
    }


def _enforce_boundary_monotonicity(
    boundaries: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    previous_selected = -math.inf
    for boundary in boundaries:
        selected = boundary.get("selected_seconds")
        last_from = boundary.get("last_from_seconds")
        first_to = boundary.get("first_to_seconds")
        valid_numbers = all(
            not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))
            for value in (selected, last_from, first_to)
        )
        reason = None
        if not valid_numbers:
            reason = "boundary_time_not_finite"
        elif float(last_from) >= float(first_to):
            reason = "boundary_interval_not_increasing"
        elif float(selected) <= previous_selected:
            reason = "global_boundary_order_not_increasing"
        if reason is not None:
            rejected.append({**boundary, "global_validation_error": reason, "needs_review": True})
            continue
        accepted.append({**boundary, "global_order_valid": True})
        previous_selected = float(selected)
    return accepted, rejected


def _refine_boundaries(
    prepared: dict[str, Any],
    candidates: list[dict[str, Any]],
    client: Any,
    stage_labels: dict[str, str],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[str]]:
    refined: list[dict[str, Any]] = []
    review_reasons: list[str] = []
    for boundary in candidates:
        center = float(boundary["coarse_selected_seconds"])
        first_start = max(prepared["fixed_start"], center - args.boundary_context_seconds)
        first_end = min(prepared["fixed_end"], center + args.boundary_context_seconds)
        coarse_pass = _run_boundary_pass(
            prepared,
            boundary,
            "pass_1_1fps",
            first_start,
            first_end,
            1.0,
            client,
            stage_labels,
            args,
        )
        coarse_observed = _observed_boundary_from_pass(coarse_pass, args.boundary_min_confidence)
        should_escalate = coarse_observed is None or coarse_observed["needs_review"]
        dense_pass: dict[str, Any] | None = None
        dense_observed: dict[str, Any] | None = None
        if should_escalate:
            dense_center = coarse_observed["selected_seconds"] if coarse_observed is not None else center
            dense_start = max(prepared["fixed_start"], float(dense_center) - args.dense_boundary_context_seconds)
            dense_end = min(prepared["fixed_end"], float(dense_center) + args.dense_boundary_context_seconds)
            dense_pass = _run_boundary_pass(
                prepared,
                boundary,
                "pass_2_0.5s",
                dense_start,
                dense_end,
                0.5,
                client,
                stage_labels,
                args,
            )
            dense_observed = _observed_boundary_from_pass(dense_pass, args.boundary_min_confidence)
        chosen = dense_observed or coarse_observed
        if chosen is None:
            last_from = float(boundary["coarse_last_from_seconds"])
            first_to = float(boundary["coarse_first_to_seconds"])
            chosen = {
                "last_from_frame_id": boundary["coarse_last_from_frame_id"],
                "first_to_frame_id": boundary["coarse_first_to_frame_id"],
                "last_from_seconds": last_from,
                "first_to_seconds": first_to,
                "boundary_interval_seconds": [min(last_from, first_to), max(last_from, first_to)],
                "selected_seconds": first_to,
                "sampling_interval_seconds": args.sample_interval_seconds,
                "confidence": None,
                "evidence": "局部复核未得到合法边界，保留 Map 粗边界。",
                "uncertainty": "boundary_refinement_failed",
                "needs_review": True,
            }
            source = "coarse_map_fallback"
            review_reasons.append(f"boundary_refinement_failed:{boundary['boundary_id']}")
        elif dense_observed is not None:
            source = "local_0.5s_refinement"
            if dense_observed["needs_review"]:
                review_reasons.append(f"boundary_dense_needs_review:{boundary['boundary_id']}")
        else:
            source = "local_1fps_refinement"
            if chosen["needs_review"]:
                review_reasons.append(f"boundary_needs_review:{boundary['boundary_id']}")
        result = {
            **boundary,
            **chosen,
            "source": source,
            "passes": {
                "one_fps": coarse_pass,
                "dense_half_second": dense_pass,
            },
        }
        write_json_atomic(
            prepared["video_dir"] / "boundaries" / str(boundary["boundary_id"]) / "result.json",
            result,
        )
        refined.append(result)
    return refined, review_reasons


def _format_clock(seconds: float | None) -> str:
    if seconds is None:
        return "未观察到"
    minutes = int(seconds // 60)
    remainder = seconds - minutes * 60
    return f"{minutes:02d}:{remainder:06.3f}".rstrip("0").rstrip(".")


def _render_report(result: dict[str, Any], stage_labels: dict[str, str]) -> str:
    start, end = result["locked_experiment_interval_seconds"]
    lines = [
        f"# hierarchical_v1 动作分割：{result['source_video_id']}",
        "",
        f"锁定区间：{_format_clock(start)}–{_format_clock(end)}",
        "",
        "| 阶段 | 可见证据区间 | 证据 |",
        "|---|---|---|",
    ]
    for item in result.get("observed_stage_runs", result.get("observed_stage_intervals", [])):
        label = stage_labels.get(str(item["stage"]), str(item["stage"]))
        lines.append(
            f"| {label} | {_format_clock(float(item['start_seconds']))}–{_format_clock(float(item['end_seconds']))} | {item.get('evidence', '')} |"
        )
    if not result.get("observed_stage_runs", result.get("observed_stage_intervals")):
        lines.append("| 未观察到 | - | Map/Reduce 未产生有效阶段证据 |")
    lines.extend(
        [
            "",
            "| 边界 | 最后前阶段帧 | 第一后阶段帧 | 操作时间 | 来源 |",
            "|---|---|---|---|---|",
        ]
    )
    for item in result.get("boundaries", []):
        transition = f"{stage_labels.get(item['from_stage'], item['from_stage'])} → {stage_labels.get(item['to_stage'], item['to_stage'])}"
        lines.append(
            f"| {transition} | {_format_clock(float(item['last_from_seconds']))} | {_format_clock(float(item['first_to_seconds']))} | {_format_clock(float(item['selected_seconds']))} | {item['source']} |"
        )
    lines.extend(
        [
            "",
            "说明：可见证据区间不自动填满没有直接证据的空档；完整 timeline 中这些空档标为 unclassified。",
            "",
        ]
    )
    return "\n".join(lines)


def analyze_prepared_video(
    prepared: dict[str, Any],
    client: Any,
    schema: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    map_events, window_results, map_review = _run_map(prepared, client, args)
    accepted_events, reduce_result, reduce_review = _run_reduce(prepared, map_events, client, args)
    terminal_id = reduce_result["selection"].get("terminal_cleanup_event_id")
    state_result = assign_seven_stages(accepted_events, terminal_id)
    effective_end = prepared["fixed_end"]
    if isinstance(terminal_id, str):
        terminal_event = next(
            (event for event in accepted_events if str(event.get("event_id")) == terminal_id),
            None,
        )
        if terminal_event is not None:
            effective_end = min(effective_end, float(terminal_event["last_seconds"]))
    stage_runs = merge_observed_stage_runs(state_result["observed_stage_intervals"])
    all_boundary_candidates = build_boundary_candidates(state_result["observed_stage_intervals"])
    invalid_coarse_boundaries = [item for item in all_boundary_candidates if not item.get("coarse_order_valid")]
    boundary_candidates = [item for item in all_boundary_candidates if item.get("coarse_order_valid")]
    boundary_review = [f"coarse_boundary_order_invalid:{item['boundary_id']}" for item in invalid_coarse_boundaries]
    stage_labels = {str(item["id"]): str(item["label_zh"]) for item in schema["stages"]}
    if args.skip_boundary_refinement:
        boundaries = [
            {
                **item,
                "last_from_seconds": item["coarse_last_from_seconds"],
                "first_to_seconds": item["coarse_first_to_seconds"],
                "boundary_interval_seconds": [
                    min(item["coarse_last_from_seconds"], item["coarse_first_to_seconds"]),
                    max(item["coarse_last_from_seconds"], item["coarse_first_to_seconds"]),
                ],
                "selected_seconds": item["coarse_first_to_seconds"],
                "sampling_interval_seconds": args.sample_interval_seconds,
                "source": "coarse_map_only",
                "needs_review": True,
            }
            for item in boundary_candidates
        ]
        if boundaries:
            boundary_review.append("boundary_refinement_skipped")
    else:
        boundaries, refinement_review = _refine_boundaries(
            prepared,
            boundary_candidates,
            client,
            stage_labels,
            args,
        )
        boundary_review.extend(refinement_review)
    boundaries, rejected_refined_boundaries = _enforce_boundary_monotonicity(boundaries)
    rejected_boundaries = [
        {**item, "global_validation_error": "coarse_boundary_order_invalid", "needs_review": True}
        for item in invalid_coarse_boundaries
    ] + rejected_refined_boundaries
    boundary_review.extend(
        f"boundary_global_validation_failed:{item['boundary_id']}:{item['global_validation_error']}"
        for item in rejected_refined_boundaries
    )
    timeline, timeline_review = build_evidence_timeline(
        prepared["fixed_start"],
        effective_end,
        state_result["observed_stage_intervals"],
    )
    provenance = prepared["source_record"]["source_segment_provenance"]
    review_reasons = sorted(
        set(
            map_review
            + reduce_review
            + list(state_result["review_reasons"])
            + boundary_review
            + timeline_review
            + (["source_segment_invalid"] if provenance.get("needs_review") else [])
        )
    )
    ignored_noise_events = reduce_result.get("ignored_noise_events", [])
    analysis_termination = dict(state_result["analysis_termination"])
    if isinstance(terminal_id, str):
        ignored_event_ids = [str(event["event_id"]) for event in ignored_noise_events]
        analysis_termination.update(
            {
                "terminal_cleanup_end_seconds": effective_end,
                "discarded_after_terminal_event_ids": ignored_event_ids,
                "discarded_after_terminal_count": len(ignored_event_ids),
            }
        )
    result = {
        "schema_version": ALGORITHM_SCHEMA_VERSION,
        "algorithm_id": ALGORITHM_ID,
        "stage_schema_id": STAGE_SCHEMA_ID,
        "generated_at": utc_now(),
        "source_video_id": prepared["video_id"],
        "source_manifest": str(prepared["manifest_path"].resolve()),
        "source_segment_provenance": provenance,
        "source_locked_interval_seconds": [prepared["fixed_start"], prepared["fixed_end"]],
        "locked_experiment_interval_seconds": [prepared["fixed_start"], effective_end],
        "effective_experiment_interval_seconds": [prepared["fixed_start"], effective_end],
        "sampling": {
            "map_window_seconds": args.window_seconds,
            "map_overlap_seconds": args.overlap_seconds,
            "map_stride_seconds": args.window_seconds - args.overlap_seconds,
            "map_sample_interval_seconds": args.sample_interval_seconds,
            "max_model_edge": args.max_model_edge,
            "boundary_first_pass_interval_seconds": 1.0,
            "boundary_dense_pass_interval_seconds": 0.5,
        },
        "map": {
            "window_count": len(window_results),
            "valid_window_count": sum(1 for item in window_results if item["valid"]),
            "normalized_event_count": len(map_events),
        },
        "reduce": reduce_result,
        "ignored_noise_events": ignored_noise_events,
        "assigned_events": state_result["assigned_events"],
        "observed_stage_intervals": state_result["observed_stage_intervals"],
        "observed_stage_runs": stage_runs,
        "timeline_segments": timeline,
        "boundaries": boundaries,
        "rejected_boundaries": rejected_boundaries,
        "missing_stages": state_result["missing_stages"],
        "analysis_termination": analysis_termination,
        "needs_review": bool(review_reasons),
        "review_reasons": review_reasons,
        "status": "completed_with_review" if review_reasons else "completed",
    }
    write_json_atomic(prepared["video_dir"] / "result.json", result)
    _write_text(prepared["video_dir"] / "report.md", _render_report(result, stage_labels))
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segment-source", type=Path, default=DEFAULT_SEGMENT_SOURCE)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--video-id", action="append", default=[])
    parser.add_argument("--allow-invalid-source-segments", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-boundary-refinement", action="store_true")
    parser.add_argument("--window-seconds", type=float, default=60.0)
    parser.add_argument("--overlap-seconds", type=float, default=10.0)
    parser.add_argument("--sample-interval-seconds", type=float, default=2.0)
    parser.add_argument("--max-model-edge", type=int, default=640)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument(
        "--reduce-recovery-policy",
        choices=("strict", "local_partial"),
        default="strict",
        help="strict quarantines an invalid Reduce; local_partial preserves non-conflicting known events.",
    )
    parser.add_argument("--map-max-tokens", type=int, default=2200)
    parser.add_argument("--reduce-max-tokens", type=int, default=2600)
    parser.add_argument("--boundary-max-tokens", type=int, default=1200)
    parser.add_argument("--boundary-context-seconds", type=float, default=10.0)
    parser.add_argument("--dense-boundary-context-seconds", type=float, default=3.0)
    parser.add_argument("--boundary-min-confidence", type=float, default=0.72)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    positive_values = (
        args.window_seconds,
        args.sample_interval_seconds,
        args.max_model_edge,
        args.max_attempts,
        args.map_max_tokens,
        args.reduce_max_tokens,
        args.boundary_max_tokens,
        args.boundary_context_seconds,
        args.dense_boundary_context_seconds,
    )
    if any(value <= 0 for value in positive_values):
        parser.error("window, sampling, resolution, attempt, token, and boundary parameters must be positive")
    if args.overlap_seconds < 0 or args.overlap_seconds >= args.window_seconds:
        parser.error("overlap must be non-negative and smaller than window length")
    if not 0.0 <= args.boundary_min_confidence <= 1.0:
        parser.error("boundary minimum confidence must be between 0 and 1")
    schema = load_stage_schema(args.schema)
    source_summary = read_json(args.segment_source)
    selected_ids = set(args.video_id) if args.video_id else None
    accepted, rejected = select_source_records(
        source_summary,
        selected_ids,
        args.allow_invalid_source_segments,
    )
    if not accepted:
        parser.error("no source segments passed the source contract")
    run_id = args.run_id or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    try:
        run_dir = create_run_directory(args.output_root, run_id)
    except (FileExistsError, ValueError) as exc:
        parser.error(str(exc))
    config = {
        "algorithm_id": ALGORITHM_ID,
        "stage_schema_id": STAGE_SCHEMA_ID,
        "segment_source": str(args.segment_source.resolve()),
        "schema_path": str(args.schema.resolve()),
        "prepare_only": args.prepare_only,
        "allow_invalid_source_segments": args.allow_invalid_source_segments,
        "window_seconds": args.window_seconds,
        "overlap_seconds": args.overlap_seconds,
        "stride_seconds": args.window_seconds - args.overlap_seconds,
        "sample_interval_seconds": args.sample_interval_seconds,
        "max_model_edge": args.max_model_edge,
        "max_attempts": args.max_attempts,
        "reduce_recovery_policy": args.reduce_recovery_policy,
        "timestamp_watermark": "new bottom information bar, left aligned FRAME ID | VIDEO T",
    }
    run_manifest = {
        "schema_version": ALGORITHM_SCHEMA_VERSION,
        "algorithm_id": ALGORITHM_ID,
        "stage_schema_id": STAGE_SCHEMA_ID,
        "run_id": run_id,
        "created_at": utc_now(),
        "status": "preparing",
        "config": config,
        "accepted_source_count": len(accepted),
        "rejected_sources": rejected,
        "output_isolation": {
            "run_directory": str(run_dir.resolve()),
            "created_new": True,
            "overwrite_supported": False,
        },
    }
    write_json_atomic(run_dir / "run_manifest.json", run_manifest)
    client = None
    if not args.prepare_only:
        client = qwen_base.OpenAI(
            base_url=qwen_base.API_BASE_URL,
            api_key=qwen_base.API_TOKEN,
            timeout=180,
            max_retries=0,
        )
    summary_records: list[dict[str, Any]] = []
    for provenance in accepted:
        video_id = str(provenance["source_video_id"])
        video_dir = run_dir / safe_video_directory_name(video_id)
        try:
            prepared = prepare_video(provenance, video_dir, args)
            if args.prepare_only:
                result = {
                    "source_video_id": video_id,
                    "status": "prepared",
                    "needs_review": bool(provenance.get("needs_review")),
                    "review_reasons": ["source_segment_invalid"] if provenance.get("needs_review") else [],
                    "locked_experiment_interval_seconds": [prepared["fixed_start"], prepared["fixed_end"]],
                    "window_count": len(prepared["prepared_windows"]),
                    "window_frame_reference_count": prepared["source_record"]["window_frame_reference_count"],
                    "unique_source_frame_count": len(prepared["frame_registry"]),
                    "overlap_reference_savings": prepared["source_record"]["overlap_reference_savings"],
                    "result_path": str((video_dir / "source.json").resolve()),
                }
            else:
                assert client is not None
                analyzed = analyze_prepared_video(prepared, client, schema, args)
                result = {
                    "source_video_id": video_id,
                    "status": analyzed["status"],
                    "needs_review": analyzed["needs_review"],
                    "review_reasons": analyzed["review_reasons"],
                    "locked_experiment_interval_seconds": analyzed["locked_experiment_interval_seconds"],
                    "observed_stage_count": len(analyzed["observed_stage_intervals"]),
                    "observed_stage_run_count": len(analyzed["observed_stage_runs"]),
                    "boundary_count": len(analyzed["boundaries"]),
                    "missing_stages": analyzed["missing_stages"],
                    "result_path": str((video_dir / "result.json").resolve()),
                }
        except Exception as exc:
            result = {
                "source_video_id": video_id,
                "status": "processing_failed",
                "needs_review": True,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "result_path": str((video_dir / "failure.json").resolve()),
            }
            write_json_atomic(video_dir / "failure.json", result)
        summary_records.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    final_status = "prepared" if args.prepare_only else "completed"
    if any(record.get("status") == "processing_failed" for record in summary_records):
        final_status += "_with_failures"
    summary = {
        "schema_version": ALGORITHM_SCHEMA_VERSION,
        "algorithm_id": ALGORITHM_ID,
        "stage_schema_id": STAGE_SCHEMA_ID,
        "run_id": run_id,
        "generated_at": utc_now(),
        "status": final_status,
        "config": config,
        "records": summary_records,
        "rejected_sources": rejected,
    }
    write_json_atomic(run_dir / "summary.json", summary)
    run_manifest["status"] = final_status
    run_manifest["completed_at"] = utc_now()
    run_manifest["summary_path"] = str((run_dir / "summary.json").resolve())
    write_json_atomic(run_dir / "run_manifest.json", run_manifest)
    print(f"summary={(run_dir / 'summary.json').resolve()}", flush=True)
    return 0 if "failures" not in final_status else 1


if __name__ == "__main__":
    raise SystemExit(main())
