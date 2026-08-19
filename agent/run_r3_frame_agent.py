#!/usr/bin/env python3
"""Run the experimental current-video-only R3 frame sampling Agent."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


AGENT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(AGENT_ROOT / "resistance_agent"))

from r3_frame_agent_adapter import run_r3_frame_agent_from_current_stages  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--stage-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--association-id")
    parser.add_argument("--max-rounds", type=int, default=2)
    parser.add_argument("--max-requests-per-round", type=int, default=3)
    parser.add_argument("--max-supplemental-frames", type=int, default=64)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_r3_frame_agent_from_current_stages(
        video_path=args.video.resolve(),
        stage_summary_path=args.stage_summary.resolve(),
        output_dir=args.output_dir.resolve(),
        association_id=args.association_id,
        max_rounds=args.max_rounds,
        max_requests_per_round=args.max_requests_per_round,
        max_supplemental_frames=args.max_supplemental_frames,
    )
    print(
        json.dumps(
            {
                "decision": result["rubric_3"]["decision"],
                "confidence": result["rubric_3"]["confidence"],
                "rubric_path": result["rubric_path"],
                "report_path": result["report_path"],
                "agent_report_path": result["agent_report_path"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
