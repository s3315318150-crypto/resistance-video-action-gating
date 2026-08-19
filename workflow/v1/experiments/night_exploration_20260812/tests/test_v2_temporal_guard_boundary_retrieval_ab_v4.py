from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = EXPERIMENT_ROOT / "scripts" / "v2_temporal_guard_boundary_retrieval_ab_v4.py"
SPEC = importlib.util.spec_from_file_location("v2_temporal_guard_boundary_v4", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class TemporalGuardBoundaryRetrievalV4Tests(unittest.TestCase):
    def make_fixture(self, root: Path, stage: str = "circuit_wiring") -> tuple[dict, dict]:
        source = root / "source.json"
        replay = root / "replay.json"
        run = {
            "stage": stage,
            "start_seconds": 1.0,
            "end_seconds": 2.0,
            "start_frame_number": 30,
            "end_frame_number": 60,
        }
        source.write_text(
            json.dumps(
                {
                    "source_video_id": "anonymous.mp4",
                    "source_manifest": "manifest.json",
                    "source_segment_provenance": {},
                    "locked_experiment_interval_seconds": [0.0, 3.0],
                    "observed_stage_runs": [],
                }
            ),
            encoding="utf-8",
        )
        replay.write_text(
            json.dumps(
                {
                    "observed_stage_runs": [run],
                    "observed_stage_intervals": [],
                    "assigned_events": [],
                    "effective_reduce_result": {},
                    "selection": {},
                }
            ),
            encoding="utf-8",
        )
        summary = {
            "mode": MODULE.EXPECTED_REPLAY_MODE,
            "records": [
                {
                    "source_video_id": "anonymous.mp4",
                    "source_result": str(source),
                    "replay_result": str(replay),
                    "restored_event_count": 2,
                    "observed_stage_runs": [run],
                }
            ],
        }
        gold = {"anonymous.mp4": [[stage, 1.0, 2.0]]}
        return summary, gold

    def test_replay_result_is_the_confirmed_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            summary, gold = self.make_fixture(Path(temporary))
            item = MODULE.temporal_guard_sources(summary, gold)[0]
            self.assertEqual("circuit_wiring", item["baseline"]["observed_stage_runs"][0]["stage"])
            self.assertEqual(2, item["baseline"]["temporal_guard_restored_event_count"])
            self.assertEqual(Path(summary["records"][0]["replay_result"]).resolve(), item["reference_result"])

    def test_nonmatching_baseline_is_rejected_before_inference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            summary, gold = self.make_fixture(Path(temporary))
            gold["anonymous.mp4"] = [["recording_1", 1.0, 2.0]]
            with self.assertRaisesRegex(ValueError, "baseline_not_confirmed_temporal_guard_v2"):
                MODULE.temporal_guard_sources(summary, gold)

    def test_temporal_guard_reducer_is_bound_by_identity(self) -> None:
        MODULE.bind_target_pipeline()
        self.assertIs(
            MODULE.v3.base.engine.salvage_reduce_response,
            MODULE.temporal_guard_reduce.salvage_reduce_response_with_temporal_guard,
        )
        self.assertIs(
            MODULE.v3.base.engine.select_events,
            MODULE.temporal_guard_reduce.select_events_with_temporal_guard,
        )


if __name__ == "__main__":
    unittest.main()
