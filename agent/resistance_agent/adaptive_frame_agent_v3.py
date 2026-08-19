"""Situation-routed adaptive frame extraction with cumulative evidence.

Version 3 keeps the video-identity-independent contract from v2, but changes
the retry semantics: later rounds search for the missing instrument instead of
only revisiting the already selected timestamps, and observations accumulate
across rounds.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from . import adaptive_frame_agent as base
except ImportError:  # pragma: no cover - direct module execution in tests
    import adaptive_frame_agent as base  # type: ignore


FrameAgentError = base.FrameAgentError
FRAME_AGENT_VERSION = "adaptive_frame_agent.v3"
MAX_ROUNDS = base.MAX_ROUNDS
MAX_FRAMES_PER_ROUND = base.MAX_FRAMES_PER_ROUND
MAX_CV_FRAMES_PER_ROUND = base.MAX_CV_FRAMES_PER_ROUND
DEFAULT_INTERVAL_SECONDS = base.DEFAULT_INTERVAL_SECONDS
ROLE_NAMES = {"ammeter", "voltmeter"}


def _stage_metadata_score(item: dict[str, Any]) -> tuple[int, int]:
    metadata_fields = (
        "stage_semantics",
        "merged_measurement_recording",
        "contains_measurement_evidence",
        "contains_writing_evidence",
        "measurement_subintervals",
        "writing_subintervals",
        "observed_subintervals",
        "base_action_types",
    )
    present = sum(field in item for field in metadata_fields)
    subinterval_count = sum(
        len(item.get(field) or [])
        for field in ("measurement_subintervals", "writing_subintervals", "observed_subintervals")
        if isinstance(item.get(field), list)
    )
    return present, subinterval_count


def _stage_runs(run_dir: Path) -> list[dict[str, Any]]:
    """Find stage intervals anywhere in this run, excluding frame-agent output."""
    found: dict[tuple[str, float, float], dict[str, Any]] = {}
    for path in sorted(run_dir.rglob("*.json")):
        try:
            relative_parts = path.relative_to(run_dir).parts
        except ValueError:
            continue
        if "frame_agent" in relative_parts:
            continue
        try:
            value = base._read_json(path)
        except (OSError, ValueError, UnicodeError):
            continue
        records: list[dict[str, Any]] = []
        base._walk_stage_runs(value, records)
        for item in records:
            key = (item["stage"], item["start_seconds"], item["end_seconds"])
            existing = found.get(key)
            if existing is None or _stage_metadata_score(item) > _stage_metadata_score(existing):
                found[key] = item
    return sorted(found.values(), key=lambda item: (item["start_seconds"], item["end_seconds"], item["stage"]))


def _record_tracks(record: dict[str, Any]) -> set[str]:
    tracks: set[str] = set()
    candidates = record.get("model_candidates") or record.get("candidates") or []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        track = candidate.get("track_id")
        if isinstance(track, str) and track:
            tracks.add(track)
            continue
        bbox = candidate.get("bbox")
        if isinstance(bbox, list) and len(bbox) >= 4:
            try:
                cx = int(round((float(bbox[0]) + float(bbox[2]) / 2.0) / 160.0))
                cy = int(round((float(bbox[1]) + float(bbox[3]) / 2.0) / 90.0))
            except (TypeError, ValueError):
                continue
            tracks.add(f"spatial_{cx}_{cy}")
    return tracks


def _record_identity_hints(record: dict[str, Any]) -> set[str]:
    hints: set[str] = set()
    candidates = record.get("model_candidates") or record.get("candidates") or []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        hint = candidate.get("track_identity_hint") or candidate.get("identity_hint") or candidate.get("role_hint")
        if hint in ROLE_NAMES:
            hints.add(str(hint))
    return hints


def _select_diverse_frame_records(
    records: list[dict[str, Any]],
    *,
    limit: int = 4,
    confirmed_roles: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Greedily select deterministic, stage- and track-diverse image groups."""
    if limit <= 0:
        return []
    try:
        from .meter_rubrics import _frame_selection_score
    except ImportError:  # pragma: no cover
        from meter_rubrics import _frame_selection_score  # type: ignore

    confirmed = confirmed_roles or set()
    remaining = [item for item in records if item.get("candidates")]
    selected: list[dict[str, Any]] = []
    used_tracks: set[str] = set()
    used_sources: set[str] = set()

    while remaining and len(selected) < limit:
        ranked: list[tuple[tuple[float, float, float, float, int], dict[str, Any]]] = []
        for item in remaining:
            tracks = _record_tracks(item)
            hints = _record_identity_hints(item)
            source = str(item.get("window_source") or "broad_search")
            new_track_bonus = 0.18 if not tracks or tracks - used_tracks else 0.0
            new_source_bonus = 0.08 if source not in used_sources else 0.0
            duplicate_confirmed_penalty = 0.24 if hints and hints.issubset(confirmed) else 0.0
            score = float(_frame_selection_score(item)) + new_track_bonus + new_source_bonus - duplicate_confirmed_penalty
            timestamp = float(item.get("timestamp_seconds") or 0.0)
            frame_number = int(item.get("frame_number") or 0)
            key = (
                score,
                float(item.get("sharpness") or 0.0),
                -float(item.get("window_priority") or 0),
                -timestamp,
                -frame_number,
            )
            ranked.append((key, item))
        _, chosen = max(ranked, key=lambda pair: pair[0])
        selected.append(chosen)
        used_tracks.update(_record_tracks(chosen))
        used_sources.add(str(chosen.get("window_source") or "broad_search"))
        remaining.remove(chosen)

    return selected


def _observation_groups(round_result: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    observation = round_result.get("qwen_observation")
    if not isinstance(observation, dict):
        return grouped
    for item in observation.get("observations") or []:
        if not isinstance(item, dict):
            continue
        group = item.get("image_group")
        if isinstance(group, int) and not isinstance(group, bool):
            grouped.setdefault(group, []).append(item)
    return grouped


def _second_round_windows(
    *,
    first_round: dict[str, Any],
    original_windows: list[dict[str, Any]],
    duration: float,
    fps: float,
    known: set[int],
) -> list[dict[str, Any]]:
    """Allocate retry points across adjacent, stage and global exploration."""
    selected = [item for item in first_round.get("selected_frames") or [] if isinstance(item, dict)]
    grouped = _observation_groups(first_round)
    ambiguous: list[float] = []
    for index, frame in enumerate(selected, start=1):
        observations = grouped.get(index, [])
        visible = {
            str(item.get("identity"))
            for item in observations
            if item.get("identity") in ROLE_NAMES and float(item.get("confidence") or 0.0) >= 0.45
        }
        if not visible or "unknown" in {str(item.get("identity")) for item in observations}:
            ambiguous.append(float(frame.get("timestamp_seconds") or 0.0))
    if not ambiguous:
        ambiguous = [float(item.get("timestamp_seconds") or 0.0) for item in selected]

    points: list[dict[str, Any]] = []
    planned_numbers: set[int] = set()

    def add(timestamp: float, source: str) -> None:
        timestamp = min(duration, max(0.0, timestamp))
        number = max(0, int(round(timestamp * fps)))
        if number in known or number in planned_numbers:
            return
        planned_numbers.add(number)
        actual = round(number / fps, 6)
        points.append({"source": source, "start_seconds": actual, "end_seconds": actual})

    offsets = (-1.0, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1.0)
    for anchor in ambiguous:
        for offset in offsets:
            add(anchor + offset, "adaptive_adjacent_missing_role")
            if sum(item["source"] == "adaptive_adjacent_missing_role" for item in points) >= 12:
                break
        if sum(item["source"] == "adaptive_adjacent_missing_role" for item in points) >= 12:
            break

    stage_numbers = base._sample_numbers(original_windows, fps, max(1, int(round(duration * fps))), 0.5)
    for number in base._limit_numbers(stage_numbers, 20):
        add(number / fps, "missing_role_stage_search")
        if sum(item["source"] == "missing_role_stage_search" for item in points) >= 12:
            break

    global_count = 16
    for index in range(global_count):
        timestamp = duration * index / max(1, global_count - 1)
        add(timestamp, "missing_role_global_search")
        if len(points) >= MAX_FRAMES_PER_ROUND:
            break

    if len(points) < MAX_FRAMES_PER_ROUND:
        dense_count = MAX_FRAMES_PER_ROUND * 4
        for index in range(dense_count):
            add(duration * index / max(1, dense_count - 1), "missing_role_global_search")
            if len(points) >= MAX_FRAMES_PER_ROUND:
                break
    return points


def _run_round(
    *,
    run_dir: Path,
    video_path: Path,
    fps: float,
    frame_count: int,
    digest: str,
    model_config: dict[str, Any],
    windows: list[dict[str, Any]],
    round_number: int,
    interval: float,
    known: set[int],
    confirmed_roles: set[str],
) -> dict[str, Any]:
    all_requested = base._sample_numbers(windows, fps, frame_count, interval)
    requested = base._limit_numbers(all_requested)
    new_numbers = [number for number in requested if number not in known]
    round_dir = run_dir / "frame_agent" / f"round_{round_number:02d}"
    source_by_number: dict[int, str] = {}
    for number in new_numbers:
        timestamp = number / fps
        matching = [
            window
            for window in windows
            if float(window["start_seconds"]) - 0.5 / fps <= timestamp <= float(window["end_seconds"]) + 0.5 / fps
        ]
        source_by_number[number] = str((matching[0] if matching else {"source": "broad_search"})["source"])

    frames = base._decode_frames(
        video_path,
        new_numbers,
        fps,
        round_dir / "frames",
        digest,
        source_by_number,
        f"round_{round_number:02d}",
    )
    known.update(item["frame_number"] for item in frames)

    try:
        from .meter_rubrics import _call_qwen, _export_candidates
        from .skills import dynamic_meter_reading
    except ImportError:  # pragma: no cover
        from meter_rubrics import _call_qwen, _export_candidates  # type: ignore
        from skills import dynamic_meter_reading  # type: ignore

    cv_frames = base._preselect_for_cv(frames)
    analyzed = [_export_candidates(item, round_dir) for item in cv_frames]
    identity = dynamic_meter_reading.prepare_frames(analyzed, render_overlays=False)
    selected = _select_diverse_frame_records(analyzed, limit=min(4, len(analyzed)), confirmed_roles=confirmed_roles)
    qwen_observation: dict[str, Any] | None = None
    qwen_error: str | None = None
    missing_before = sorted(ROLE_NAMES - confirmed_roles)
    if selected:
        try:
            qwen_observation = _call_qwen(
                selected,
                model_config,
                round_dir / "qwen" / "frame_agent_raw.json",
                skill_instruction=(
                    "This is frame selection only. Identify every complete visible ammeter and voltmeter. "
                    f"Roles not yet confirmed by earlier current-run evidence: {missing_before}. "
                    "Do not score a rubric and do not infer a role from workflow order. Treat each image group independently."
                ),
                candidate_crops_per_frame=2,
                execution_fingerprint=FRAME_AGENT_VERSION,
            )
        except (OSError, RuntimeError, ValueError, KeyError) as exc:
            qwen_error = f"{type(exc).__name__}:{exc}"
    roles = sorted(base._qwen_roles(qwen_observation or {}))
    cumulative_roles = sorted(confirmed_roles | set(roles))
    result = {
        "schema_version": "resistance_agent_frame_agent_round.v2",
        "agent_version": FRAME_AGENT_VERSION,
        "round_number": round_number,
        "windows": windows,
        "search_strategy": "initial_stage_coverage" if round_number == 1 else "missing_role_adjacent_stage_and_global_search",
        "missing_roles_before_round": missing_before,
        "sampling": {
            "interval_seconds": interval,
            "candidate_frame_count_before_limit": len(all_requested),
            "requested_frame_count": len(requested),
            "new_frame_count": len(new_numbers),
            "decoded_frame_count": len(frames),
            "opencv_frame_count": len(cv_frames),
        },
        "frames": frames,
        "selected_frames": selected,
        "dynamic_meter_identity": identity,
        "qwen_observation": qwen_observation,
        "qwen_error": qwen_error,
        "visible_roles": roles,
        "cumulative_visible_roles": cumulative_roles,
        "frame_useful": bool(selected) and bool(roles),
        "meter_pair_complete": ROLE_NAMES.issubset(cumulative_roles),
        "needs_more_frames": not ROLE_NAMES.issubset(cumulative_roles),
        "historical_artifacts_used": False,
        "video_id_used_for_routing": False,
        "fixed_video_roi_used": False,
    }
    base._write_json(round_dir / "result.json", result)
    return result


def run_adaptive_frame_agent(
    *,
    run_dir: Path,
    state: dict[str, Any],
    model_config: dict[str, Any],
    max_rounds: int = MAX_ROUNDS,
    initial_interval_seconds: float = DEFAULT_INTERVAL_SECONDS,
) -> dict[str, Any]:
    """Extract current-run evidence and retain all confirmed roles across retries."""
    if state.get("mode") != "execute":
        raise FrameAgentError("frame agent is only valid in execute mode")
    if type(max_rounds) is not int or not 1 <= max_rounds <= MAX_ROUNDS:
        raise FrameAgentError(f"max_rounds must be an integer from 1 to {MAX_ROUNDS}")
    if not 0.1 <= float(initial_interval_seconds) <= 2.0:
        raise FrameAgentError("initial_interval_seconds must be between 0.1 and 2.0")

    video_path, fps, frame_count, digest = base._video_info(state)
    duration = frame_count / fps
    stages = _stage_runs(run_dir)
    original_windows = base._windows(stages, duration)
    windows = original_windows
    # Other current-run tools may record frame numbers without exposing their
    # decoded images here. De-duplicate only frames decoded by this agent.
    known: set[int] = set()
    rounds: list[dict[str, Any]] = []
    confirmed_roles: set[str] = set()
    interval = float(initial_interval_seconds)

    for round_number in range(1, max_rounds + 1):
        result = _run_round(
            run_dir=run_dir,
            video_path=video_path,
            fps=fps,
            frame_count=frame_count,
            digest=digest,
            model_config=model_config,
            windows=windows,
            round_number=round_number,
            interval=interval,
            known=known,
            confirmed_roles=confirmed_roles,
        )
        rounds.append(result)
        confirmed_roles.update(result["visible_roles"])
        if ROLE_NAMES.issubset(confirmed_roles) or round_number >= max_rounds:
            break
        windows = _second_round_windows(
            first_round=result,
            original_windows=original_windows,
            duration=duration,
            fps=fps,
            known=known,
        )
        if not windows:
            break
        interval = max(0.1, interval / 2.0)

    pair_complete = ROLE_NAMES.issubset(confirmed_roles)
    cumulative_selected: dict[str, dict[str, Any]] = {}
    evidence_by_role: dict[str, list[dict[str, Any]]] = {role: [] for role in sorted(ROLE_NAMES)}
    for item in rounds:
        selected = item.get("selected_frames") or []
        for frame in selected:
            if isinstance(frame, dict):
                cumulative_selected[str(frame.get("frame_id") or frame.get("frame_number"))] = frame
        for group, observations in _observation_groups(item).items():
            for observation in observations:
                role = observation.get("identity")
                if role in ROLE_NAMES and float(observation.get("confidence") or 0.0) >= 0.45:
                    frame = selected[group - 1] if 0 < group <= len(selected) else {}
                    evidence_by_role[str(role)].append(
                        {
                            "round_number": item["round_number"],
                            "image_group": group,
                            "frame_id": frame.get("frame_id"),
                            "timestamp_seconds": frame.get("timestamp_seconds"),
                            "confidence": observation.get("confidence"),
                            "evidence": observation.get("evidence"),
                        }
                    )

    request_limit_reached = not pair_complete and len(rounds) >= max_rounds
    report = {
        "schema_version": "resistance_agent_frame_agent.v2",
        "agent_version": FRAME_AGENT_VERSION,
        "status": "frame_evidence_ready" if pair_complete else "frame_evidence_partial",
        "run_id": state.get("run_id"),
        "source_video_sha256": digest,
        "observed_stage_runs": stages,
        "round_count": len(rounds),
        "rounds": rounds,
        "selected_frames": list(cumulative_selected.values()),
        "visible_roles": sorted(confirmed_roles),
        "confirmed_roles": sorted(confirmed_roles),
        "missing_roles": sorted(ROLE_NAMES - confirmed_roles),
        "role_evidence": evidence_by_role,
        "frame_useful": bool(confirmed_roles),
        "meter_pair_complete": pair_complete,
        "needs_more_frames": not pair_complete,
        "request_limit_reached": request_limit_reached,
        "next_tool": "run_rubric_bundle" if confirmed_roles else None,
        "selection_basis": "current_video_observed_stage_and_current_frame_visual_evidence_only",
        "video_id_used_for_routing": False,
        "historical_artifacts_used": False,
        "fixed_video_roi_used": False,
    }
    report_path = run_dir / "frame_agent" / "frame_agent_report.json"
    base._write_json(report_path, report)
    return {**report, "report_path": str(report_path.resolve())}
