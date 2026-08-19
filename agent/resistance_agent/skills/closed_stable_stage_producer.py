"""Produce current-run stable pointer candidates for closed-stable R6 CV V3."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


SKILL_VERSION = "closed_stable_stage_producer.v1"
STAGE_ORDER = ("measurement_1", "recording_1", "measurement_2", "recording_2")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_stage_intervals(
    temporal_record: dict[str, Any],
    duration_seconds: float,
    *,
    measurement_lead_seconds: float = 8.0,
) -> list[dict[str, Any]]:
    """Build four ordered stages from direct recordings and pre-recording proxies."""
    raw_runs = temporal_record.get("observed_stage_runs")
    runs = [item for item in raw_runs if isinstance(item, dict)] if isinstance(raw_runs, list) else []
    normalized_runs: list[dict[str, Any]] = []
    for item in runs:
        stage = str(item.get("stage") or "")
        try:
            start = max(0.0, min(duration_seconds, float(item["start_seconds"])))
            end = max(start, min(duration_seconds, float(item["end_seconds"])))
        except (KeyError, TypeError, ValueError):
            continue
        if end - start < 0.5:
            continue
        normalized = {"stage": stage, "start_seconds": start, "end_seconds": end}
        for field in (
            "stage_semantics",
            "stage_window_semantics",
            "merged_stage_semantics",
            "merged_measurement_recording",
            "merged_stage",
            "observed_subintervals",
            "measurement_subintervals",
        ):
            if field in item:
                normalized[field] = item[field]
        normalized_runs.append(normalized)

    direct: dict[str, dict[str, Any]] = {}
    for item in normalized_runs:
        stage = str(item["stage"])
        if stage not in STAGE_ORDER:
            continue
        start = float(item["start_seconds"])
        end = float(item["end_seconds"])
        existing = direct.get(stage)
        if existing is None or end - start > float(existing["end_seconds"]) - float(existing["start_seconds"]):
            direct[stage] = {
                "stage": stage,
                "start_seconds": round(start, 3),
                "end_seconds": round(end, 3),
                "window_type": "temporal_guard_direct_stage",
                "segmentation_claim": True,
            }

    def is_merged_cycle(item: dict[str, Any]) -> bool:
        return str(item.get("stage")) in {"recording_1", "recording_2"} and (
            item.get("stage_semantics") == "measurement_and_recording_cycle"
            or item.get("stage_window_semantics") == "measurement_and_recording_cycle"
            or item.get("merged_stage_semantics") == "measurement_and_recording_cycle"
            or item.get("merged_measurement_recording") is True
            or item.get("merged_stage") is True
        )

    for cycle in (1, 2):
        recording_stage = f"recording_{cycle}"
        measurement_stage = f"measurement_{cycle}"
        merged_runs = [
            item for item in normalized_runs
            if item["stage"] == recording_stage and is_merged_cycle(item)
        ]
        if not merged_runs or measurement_stage in direct:
            continue
        observed: list[tuple[float, float]] = []
        for item in merged_runs:
            raw = item.get("measurement_subintervals")
            explicit_field = isinstance(raw, list)
            if not explicit_field:
                raw = item.get("observed_subintervals")
            for subinterval in raw if isinstance(raw, list) else []:
                if not isinstance(subinterval, dict):
                    continue
                if not explicit_field and subinterval.get("action_type") != "measurement_action":
                    continue
                try:
                    start = max(0.0, min(duration_seconds, float(subinterval["start_seconds"])))
                    end = max(start, min(duration_seconds, float(subinterval["end_seconds"])))
                except (KeyError, TypeError, ValueError):
                    continue
                if end - start >= 0.5:
                    observed.append((start, end))
        if observed:
            start = min(item[0] for item in observed)
            end = max(item[1] for item in observed)
            window_type = "temporal_guard_merged_measurement_subinterval"
        else:
            start = min(float(item["start_seconds"]) for item in merged_runs)
            end = max(float(item["end_seconds"]) for item in merged_runs)
            window_type = "temporal_guard_merged_cycle_fallback"
        if end - start >= 0.5:
            direct[measurement_stage] = {
                "stage": measurement_stage,
                "start_seconds": round(start, 3),
                "end_seconds": round(end, 3),
                "window_type": window_type,
                "segmentation_claim": bool(observed),
                "source_stage": recording_stage,
            }

    for cycle, previous_stage, following_stages in (
        (1, "circuit_wiring", ("circuit_rewiring", "material_cleanup")),
        (2, "circuit_rewiring", ("material_cleanup",)),
    ):
        recording_stage = f"recording_{cycle}"
        measurement_stage = f"measurement_{cycle}"
        if recording_stage in direct:
            continue
        previous_ends = [
            float(item["end_seconds"])
            for item in normalized_runs
            if item["stage"] == previous_stage
        ]
        if not previous_ends:
            continue
        cycle_start = max(previous_ends)
        following_starts = [
            float(item["start_seconds"])
            for item in normalized_runs
            if item["stage"] in following_stages and float(item["start_seconds"]) >= cycle_start
        ]
        cycle_end = min(following_starts) if following_starts else duration_seconds
        if cycle_end - cycle_start < 1.0:
            continue

        if measurement_stage in direct:
            recording_start = max(
                float(direct[measurement_stage]["end_seconds"]),
                cycle_end - min(8.0, cycle_end - cycle_start),
            )
        else:
            recording_width = min(8.0, max(0.5, (cycle_end - cycle_start) / 2.0))
            recording_start = cycle_end - recording_width
            measurement_start = max(cycle_start, recording_start - measurement_lead_seconds)
            if recording_start - measurement_start >= 0.5:
                direct[measurement_stage] = {
                    "stage": measurement_stage,
                    "start_seconds": round(measurement_start, 3),
                    "end_seconds": round(recording_start, 3),
                    "window_type": "inferred_cycle_gap_measurement_proxy",
                    "segmentation_claim": False,
                    "proxy_reason": "recording_stage_missing_from_temporal_guard",
                }
        if cycle_end - recording_start >= 0.5:
            direct[recording_stage] = {
                "stage": recording_stage,
                "start_seconds": round(recording_start, 3),
                "end_seconds": round(cycle_end, 3),
                "window_type": "inferred_cycle_gap_recording_proxy",
                "segmentation_claim": False,
                "proxy_reason": "recording_stage_missing_from_temporal_guard",
            }

    for cycle in (1, 2):
        measurement_stage = f"measurement_{cycle}"
        recording_stage = f"recording_{cycle}"
        if measurement_stage in direct or recording_stage not in direct:
            continue
        recording_start = float(direct[recording_stage]["start_seconds"])
        proxy_start = max(0.0, recording_start - measurement_lead_seconds)
        if recording_start - proxy_start >= 0.5:
            direct[measurement_stage] = {
                "stage": measurement_stage,
                "start_seconds": round(proxy_start, 3),
                "end_seconds": round(recording_start, 3),
                "window_type": "inferred_pre_recording_proxy",
                "segmentation_claim": False,
            }

    intervals = [direct[stage] for stage in STAGE_ORDER if stage in direct]
    if not intervals:
        raise ValueError("Temporal Guard has no measurement or recording stage for closed-stable search")
    return intervals


def build_manifest(
    *,
    video_path: Path,
    video_id: str,
    temporal_record: dict[str, Any],
    duration_seconds: float,
    measurement_lead_seconds: float = 8.0,
    force_full_stage_scan: bool = True,
) -> dict[str, Any]:
    intervals = build_stage_intervals(
        temporal_record,
        duration_seconds,
        measurement_lead_seconds=measurement_lead_seconds,
    )
    if force_full_stage_scan:
        intervals = [
            {
                **item,
                "source_segmentation_claim": bool(item.get("segmentation_claim")),
                # The baseline producer uses this field as an early-stop flag.
                # Suppress it during search, then restore the source semantics.
                "segmentation_claim": False,
                "producer_scan_policy": "force_all_available_stages",
            }
            for item in intervals
        ]
    return {
        "schema_version": "resistance-agent-current-video-four-stage-manifest.v1",
        "skill_version": SKILL_VERSION,
        "stage_order": list(STAGE_ORDER),
        "videos": [
            {
                "video_id": str(video_id),
                "source_video": str(video_path.resolve()),
                "intervals": intervals,
            }
        ],
        "routing_policy": "current Temporal Guard stages; no video-id prediction routing",
        "force_full_stage_scan": force_full_stage_scan,
        "qwen_called": False,
        "excel_accessed": False,
        "score_computed": False,
    }


def _configured_path(root: Path, value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"closed-stable stage producer config is missing {field}")
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _verified_existing(
    summary_path: Path,
    *,
    video_path: Path,
    source_sha256: str,
    manifest_sha256: str,
) -> dict[str, Any] | None:
    if not summary_path.is_file():
        return None
    summary = _read_json(summary_path)
    result_path = Path(str(summary.get("result_path") or ""))
    if (
        summary.get("skill_version") != SKILL_VERSION
        or summary.get("source_video_path") != str(video_path.resolve())
        or summary.get("source_video_sha256") != source_sha256
        or summary.get("manifest_sha256") != manifest_sha256
        or not result_path.is_file()
    ):
        return None
    if summary.get("result_sha256") != _sha256(result_path):
        return None
    payload = _read_json(result_path)
    videos = payload.get("videos")
    if not isinstance(videos, list) or len(videos) != 1:
        return None
    integrity = videos[0].get("source_video_integrity") or {}
    if integrity.get("unchanged") is not True or integrity.get("sha256_after") != source_sha256:
        return None
    return {**summary, "checkpoint_reused": True}


def normalize_full_scan_role(role_payload: dict[str, Any]) -> dict[str, Any]:
    """Rebuild the role summary after early-stop suppression scanned every stage."""
    stage_results = [
        item for item in role_payload.get("stage_results", []) if isinstance(item, dict)
    ]
    stable = [item for item in stage_results if bool((item.get("temporal_consensus") or {}).get("stable"))]
    output = dict(role_payload)
    output["searched_stages"] = [str(item.get("stage")) for item in stage_results]
    output["full_stage_scan_completed"] = True
    output["stopped_after_stable_candidate"] = False
    output["selected_stage"] = stable[-1].get("stage") if stable else None
    output["state"] = (
        "stable_pointer_candidate"
        if stable
        else "no_stable_pointer_candidate_after_all_available_stages"
    )
    output["stable_stage_count"] = len(stable)
    return output


def run_current_video_search(
    *,
    video_path: Path,
    video_id: str,
    temporal_record: dict[str, Any],
    duration_seconds: float,
    output_root: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Run the established CPU arc-to-hub producer on one current video."""
    producer_root = _configured_path(Path.cwd(), config.get("producer_root"), "producer_root")
    script_path = _configured_path(producer_root, config.get("script"), "script")
    calibration_path = _configured_path(producer_root, config.get("calibration"), "calibration")
    terminal_annotations = _configured_path(
        producer_root, config.get("terminal_annotations"), "terminal_annotations"
    )
    for path in (producer_root, script_path, calibration_path, terminal_annotations):
        if not path.exists():
            raise ValueError(f"closed-stable stage producer dependency is missing: {path}")

    source_sha256 = _sha256(video_path)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(
        video_path=video_path,
        video_id=video_id,
        temporal_record=temporal_record,
        duration_seconds=duration_seconds,
        measurement_lead_seconds=float(config.get("measurement_lead_seconds", 8.0)),
        force_full_stage_scan=bool(config.get("force_full_stage_scan", True)),
    )
    manifest_path = output_root / "current_video_four_stage_manifest.json"
    _write_json(manifest_path, manifest)
    manifest_sha256 = _sha256(manifest_path)
    summary_path = output_root / "producer_summary.json"
    existing = _verified_existing(
        summary_path,
        video_path=video_path,
        source_sha256=source_sha256,
        manifest_sha256=manifest_sha256,
    )
    if existing is not None:
        return existing

    attempt = 1
    while (output_root / f"search_{attempt:03d}").exists():
        attempt += 1
    search_dir = output_root / f"search_{attempt:03d}"
    command = [
        sys.executable,
        str(script_path),
        "--manifest",
        str(manifest_path),
        "--calibration",
        str(calibration_path),
        "--terminal-annotations",
        str(terminal_annotations),
        "--output-dir",
        str(search_dir),
        "--roles",
        "ammeter",
        "voltmeter",
        "--coarse-seconds",
        str(float(config.get("coarse_seconds", 1.0))),
        "--dense-seconds",
        str(float(config.get("dense_seconds", 0.1))),
        "--dense-radius-seconds",
        str(float(config.get("dense_radius_seconds", 0.7))),
        "--max-feature-width",
        str(int(config.get("max_feature_width", 2400))),
    ]
    completed = subprocess.run(
        command,
        cwd=producer_root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=float(config.get("timeout_seconds", 3600.0)),
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "closed-stable stage producer failed: "
            f"exit={completed.returncode}; stderr={completed.stderr[-2000:]}"
        )
    result_path = search_dir / "four_stage_coarse_to_fine_results.json"
    if not result_path.is_file():
        raise RuntimeError("closed-stable stage producer did not create its result")
    payload = _read_json(result_path)
    videos = payload.get("videos")
    if not isinstance(videos, list) or len(videos) != 1 or str(videos[0].get("video_id")) != str(video_id):
        raise ValueError("closed-stable stage producer result video binding is invalid")
    integrity = videos[0].get("source_video_integrity") or {}
    source_intervals = {
        str(item["stage"]): item
        for item in manifest["videos"][0]["intervals"]
        if isinstance(item, dict) and item.get("stage")
    }
    roles = videos[0].get("roles") or {}
    for role, role_payload in tuple(roles.items()):
        for stage_result in role_payload.get("stage_results", []):
            source_interval = source_intervals.get(str(stage_result.get("stage")))
            if source_interval is None:
                continue
            result_interval = stage_result.get("source_interval") or {}
            if "source_segmentation_claim" in source_interval:
                result_interval["segmentation_claim"] = source_interval["source_segmentation_claim"]
                result_interval["producer_early_stop_claim_suppressed"] = True
            stage_result["source_interval"] = result_interval
        roles[role] = normalize_full_scan_role(role_payload)
    videos[0]["roles"] = roles
    payload["producer_skill_version"] = SKILL_VERSION
    payload["force_full_stage_scan"] = bool(manifest.get("force_full_stage_scan"))
    _write_json(result_path, payload)
    after_sha256 = _sha256(video_path)
    if after_sha256 != source_sha256 or integrity.get("unchanged") is not True:
        raise RuntimeError("closed-stable stage producer changed the source video")
    summary = {
        "skill_version": SKILL_VERSION,
        "source_video_path": str(video_path.resolve()),
        "source_video_sha256": source_sha256,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": manifest_sha256,
        "result_path": str(result_path.resolve()),
        "result_sha256": _sha256(result_path),
        "command": command,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "exit_status": completed.returncode,
        "checkpoint_reused": False,
        "qwen_called": False,
        "excel_accessed": False,
        "original_video_unchanged": True,
    }
    _write_json(summary_path, summary)
    return summary
