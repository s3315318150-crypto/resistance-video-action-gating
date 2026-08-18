from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "refine_v2_temporal_guard_boundaries_rubric.py"
SPEC = importlib.util.spec_from_file_location("rubric_boundary_entrypoint", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class RubricBoundaryEntrypointTests(unittest.TestCase):
    def test_normal_temporal_guard_summary_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = root / "result.json"
            result.write_text(
                json.dumps(
                    {
                        "algorithm_id": "qwen_experiment_action_hierarchical_v2_temporal_guard",
                        "source_video_id": "new-video.mp4",
                        "observed_stage_intervals": [{"event_id": "e1"}],
                        "observed_stage_runs": [{"stage": "circuit_wiring"}],
                    }
                ),
                encoding="utf-8",
            )
            summary = root / "summary.json"
            summary.write_text(
                json.dumps(
                    {
                        "records": [
                            {
                                "source_video_id": "new-video.mp4",
                                "result_path": str(result),
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            records = MODULE.load_action_records(summary)
            self.assertEqual("new-video.mp4", records[0]["source_video_id"])

    def test_plain_v2_result_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = root / "result.json"
            result.write_text(
                json.dumps(
                    {
                        "algorithm_id": "qwen_experiment_action_hierarchical_v2",
                        "source_video_id": "new-video.mp4",
                        "observed_stage_intervals": [{"event_id": "e1"}],
                        "observed_stage_runs": [{"stage": "circuit_wiring"}],
                    }
                ),
                encoding="utf-8",
            )
            summary = root / "summary.json"
            summary.write_text(
                json.dumps(
                    {
                        "records": [
                            {
                                "source_video_id": "new-video.mp4",
                                "result_path": str(result),
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "not_temporal_guard_v2"):
                MODULE.load_action_records(summary)

    def test_replay_summary_uses_replayed_stages_and_source_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.json"
            source.write_text(
                json.dumps(
                    {
                        "source_video_id": "new-video.mp4",
                        "source_manifest": "manifest.json",
                        "source_segment_provenance": {},
                        "locked_experiment_interval_seconds": [0.0, 10.0],
                    }
                ),
                encoding="utf-8",
            )
            replay = root / "replay.json"
            replay.write_text(
                json.dumps(
                    {
                        "observed_stage_intervals": [{"event_id": "replayed"}],
                        "observed_stage_runs": [{"stage": "recording_1"}],
                        "assigned_events": [],
                    }
                ),
                encoding="utf-8",
            )
            summary = root / "summary.json"
            summary.write_text(
                json.dumps(
                    {
                        "mode": "stored_successful_map_and_reduce_temporal_guard_replay",
                        "records": [
                            {
                                "source_video_id": "new-video.mp4",
                                "source_result": str(source),
                                "replay_result": str(replay),
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            record = MODULE.load_action_records(summary)[0]
            self.assertEqual("manifest.json", record["baseline"]["source_manifest"])
            self.assertEqual("replayed", record["baseline"]["observed_stage_intervals"][0]["event_id"])
            self.assertEqual("recording_1", record["baseline"]["observed_stage_runs"][0]["stage"])

    def test_cli_has_no_gold_argument(self) -> None:
        actions = MODULE.build_parser()._option_string_actions
        self.assertNotIn("--gold", actions)
        self.assertIn("--action-summary", actions)

    def test_v2_schema_loads_after_explicit_binding(self) -> None:
        MODULE.support.base.bind_mature_v2_pipeline()
        schema = MODULE.support.base.contract.load_stage_schema(
            MODULE.support.base.v2_entrypoint.DEFAULT_SCHEMA
        )
        self.assertEqual("resistance_7stage_no_battery_v2", schema["stage_schema_id"])

    def test_output_directory_is_anonymous_and_stable(self) -> None:
        first = MODULE.anonymous_video_directory("private-name.mp4")
        second = MODULE.anonymous_video_directory("private-name.mp4")
        self.assertEqual(first, second)
        self.assertRegex(first, r"^video_[0-9a-f]{12}$")
        self.assertNotIn("private", first)


if __name__ == "__main__":
    unittest.main()
