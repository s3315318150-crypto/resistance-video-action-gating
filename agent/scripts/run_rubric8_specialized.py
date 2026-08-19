#!/usr/bin/env python3
"""Run the existing Rubric 8 algorithm against an Agent-owned video copy."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "workflow" / "v2" / "scripts"))

import run_resistance_disconnect_battery_sequence_v1 as workflow  # noqa: E402


def preserve_same_frame_completion(
    summary: Mapping[str, Any], observations: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Prevent a contact overlay from erasing a completion on the same frame."""
    rewire = summary.get("terminal_rewire")
    contact_ids = summary.get("direct_contact_frame_ids")
    if (
        summary.get("battery_object") != "confirmed"
        or not isinstance(rewire, Mapping)
        or rewire.get("completed") is not True
        or not isinstance(contact_ids, list)
    ):
        return observations
    end_id = rewire.get("end_frame_id")
    if not isinstance(end_id, str) or end_id not in contact_ids:
        return observations
    for item in observations:
        if item.get("frame_id") == end_id:
            item["direct_battery_contact"] = True
            item["terminal_action"] = "reconnect"
            item["terminal_rewire_completed"] = True
    return observations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-video", type=Path, required=True)
    args, forwarded = parser.parse_known_args(argv)
    source_video = args.source_video.resolve()
    if not source_video.is_file():
        raise FileNotFoundError(source_video)

    requested_ids: list[str] = []
    for index, value in enumerate(forwarded):
        if value == "--video-ids" and index + 1 < len(forwarded):
            requested_ids = [item.strip() for item in forwarded[index + 1].split(",") if item.strip()]
            break
    source_id = workflow.video_id_from_name(source_video.name)
    if requested_ids != [source_id]:
        raise ValueError("--source-video requires one matching --video-ids value")

    original_discover = workflow.discover_video
    original_convert = workflow.structured_summary_to_observations

    def discover_video(source_name: str) -> Path:
        if workflow.video_id_from_name(source_name) != source_id:
            raise ValueError("action record and Agent video id do not match")
        return source_video

    def convert(summary: Mapping[str, Any], records: list[Any]) -> tuple[list[dict[str, Any]], list[str]]:
        observations, errors = original_convert(summary, records)
        return preserve_same_frame_completion(summary, observations), errors

    workflow.discover_video = discover_video
    workflow.structured_summary_to_observations = convert
    try:
        return workflow.main(forwarded)
    finally:
        workflow.discover_video = original_discover
        workflow.structured_summary_to_observations = original_convert


if __name__ == "__main__":
    raise SystemExit(main())
