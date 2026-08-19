#!/usr/bin/env python3
"""One-command live R3 run that freezes prediction before any Excel access."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any


AGENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_ROOT))

from resistance_agent import toolkit  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def configure_qwen(config_path: Path) -> None:
    config = toolkit.load_config(config_path)
    qwen = config.get("models", {}).get("qwen", {})
    base_url = str(qwen.get("base_url") or "").strip()
    if base_url:
        os.environ.setdefault("QWEN_API_BASE_URL", base_url)
    os.environ.setdefault("QWEN_API_TOKEN", "EMPTY")


def freeze_prediction(run_id: str) -> dict[str, Any]:
    run_dir, state = toolkit._state(run_id)
    raw = state.get("rubric_results", {}).get("3")
    if not isinstance(raw, str) or not raw:
        raise RuntimeError("formal R3 result is missing")
    source = Path(raw).resolve()
    if source != (run_dir / "rubrics" / "rubric_3.json").resolve():
        raise RuntimeError("R3 result is outside the current run")
    result = read_json(source)
    if (
        result.get("schema_version") != "resistance_agent_rubric_result.v2"
        or result.get("rubric_id") != 3
        or result.get("decision") not in {"pass", "fail"}
        or result.get("predicted_score") not in {0, 1}
        or result.get("video_id_used_for_routing") is not False
        or result.get("historical_artifacts_used") is not False
        or result.get("fixed_video_roi_used") is not False
    ):
        raise RuntimeError("formal R3 result failed freeze validation")
    freeze_dir = run_dir / "frozen_r3"
    freeze_dir.mkdir(parents=True, exist_ok=True)
    target = freeze_dir / "predictions_frozen.json"
    hash_path = freeze_dir / "predictions_frozen.sha256"
    if target.exists() or hash_path.exists():
        if not target.is_file() or not hash_path.is_file():
            raise RuntimeError("partial R3 freeze exists")
        expected = hash_path.read_text(encoding="ascii").split()[0]
        actual = sha256(target)
        if expected != actual:
            raise RuntimeError("existing R3 freeze hash mismatch")
        return {
            "status": "already_frozen",
            "path": str(target.resolve()),
            "sha256": actual,
        }
    shutil.copyfile(source, target)
    frozen_hash = sha256(target)
    hash_path.write_text(f"{frozen_hash}  {target.name}\n", encoding="ascii")
    reopened = read_json(target)
    if reopened.get("decision") != result["decision"] or sha256(source) != frozen_hash:
        raise RuntimeError("R3 freeze reopen verification failed")
    return {
        "status": "frozen",
        "path": str(target.resolve()),
        "sha256": frozen_hash,
        "excel_read": False,
        "ground_truth_read": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-ref", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config", type=Path, default=AGENT_ROOT / "config.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    configure_qwen(config_path)
    inspected = toolkit.inspect_video(args.video_ref, config_path=config_path)
    created = toolkit.create_run(
        args.run_id,
        video_ref=args.video_ref,
        mode="execute",
        config_path=config_path,
    )
    run_dir, state = toolkit._state(args.run_id)
    if not state.get("action_summary"):
        toolkit.run_full_pipeline(args.run_id, dry_run=False)
    run_dir, state = toolkit._state(args.run_id)
    if not state.get("boundary_summary"):
        toolkit.refine_rubric_boundaries(args.run_id, execute=True)
    plan = toolkit.plan_live_skills(args.run_id)
    run_dir, state = toolkit._state(args.run_id)
    if "3" not in state.get("rubric_results", {}):
        produced = toolkit.run_switch_rubric(args.run_id)
    else:
        produced = {
            "status": "switch_rubric_already_completed",
            "rubric": read_json(Path(state["rubric_results"]["3"])),
        }
    frozen = freeze_prediction(args.run_id)
    run_dir, state = toolkit._state(args.run_id)
    payload = {
        "status": "r3_prediction_frozen",
        "run_id": args.run_id,
        "run_dir": str(run_dir.resolve()),
        "video": inspected,
        "create_status": created.get("status"),
        "routing_policy": plan.get("routing_policy"),
        "selection_basis": plan.get("selection_basis"),
        "video_id_used_for_routing": plan.get("video_id_used_for_routing"),
        "historical_artifacts_used": plan.get("historical_artifacts_used"),
        "fixed_video_roi_used": plan.get("fixed_video_roi_used"),
        "rubric": produced.get("rubric"),
        "freeze": frozen,
        "excel_read": False,
        "ground_truth_read": False,
        "source_video_unchanged": sha256(Path(state["video"]["path"]))
        == state["video"]["sha256"],
    }
    output_path = run_dir / "r3_blind_execute_result.json"
    toolkit.write_json(output_path, payload)
    reopened = read_json(output_path)
    if reopened.get("status") != "r3_prediction_frozen":
        raise RuntimeError("blind R3 result failed reopen verification")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
