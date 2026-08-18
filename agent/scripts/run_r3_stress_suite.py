#!/usr/bin/env python3
"""Run isolated R3 identity, temporal, quality, or aggregate checks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


AGENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_ROOT))

from resistance_agent.r3_stress_suite import (  # noqa: E402
    aggregate_reports,
    run_identity_test,
    run_quality_test,
    run_temporal_test,
)


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--stage-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-rounds", type=int, default=2)
    parser.add_argument("--max-requests-per-round", type=int, default=3)
    parser.add_argument("--max-supplemental-frames", type=int, default=64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    identity = subparsers.add_parser("identity")
    add_common(identity)
    identity.add_argument("--random-seed", type=int, default=20260818)

    temporal = subparsers.add_parser("temporal")
    add_common(temporal)
    temporal.add_argument("--phase-offset", type=float, action="append")
    temporal.add_argument("--boundary-shift", type=float, action="append")
    temporal.add_argument("--skip-phase", action="store_true")
    temporal.add_argument("--skip-boundary", action="store_true")

    quality = subparsers.add_parser("quality")
    add_common(quality)
    quality.add_argument(
        "--variant",
        choices=("1080p", "720p", "blur", "brightness", "recompress"),
        action="append",
    )
    quality.add_argument("--generate-only", action="store_true")

    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--agent-report", action="append", required=True)
    aggregate.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def common(args: argparse.Namespace) -> dict:
    return {
        "video_path": args.video.resolve(),
        "stage_summary_path": args.stage_summary.resolve(),
        "output_dir": args.output_dir.resolve(),
        "max_rounds": args.max_rounds,
        "max_requests_per_round": args.max_requests_per_round,
        "max_supplemental_frames": args.max_supplemental_frames,
    }


def parse_labeled_report(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("--agent-report must use LABEL=PATH")
    label, raw_path = value.split("=", 1)
    if not label or not raw_path:
        raise ValueError("--agent-report must use LABEL=PATH")
    path = Path(raw_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return label, path


def main() -> int:
    args = parse_args()
    if args.command == "identity":
        result = run_identity_test(random_seed=args.random_seed, **common(args))
    elif args.command == "temporal":
        values = common(args)
        result = run_temporal_test(
            phase_offsets=(
                () if args.skip_phase else args.phase_offset or (-0.1, 0.1)
            ),
            boundary_shifts=(
                ()
                if args.skip_boundary
                else args.boundary_shift or (-5.0, -2.0, 2.0, 5.0)
            ),
            **values,
        )
    elif args.command == "quality":
        result = run_quality_test(
            variants=args.variant
            or ("1080p", "720p", "blur", "brightness", "recompress"),
            run_agent=not args.generate_only,
            **common(args),
        )
    else:
        result = aggregate_reports(
            [parse_labeled_report(value) for value in args.agent_report],
            args.output_dir.resolve(),
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
