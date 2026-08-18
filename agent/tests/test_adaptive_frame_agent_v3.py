from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np


AGENT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AGENT_ROOT / "resistance_agent"))

import adaptive_frame_agent_v3 as agent  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class AdaptiveFrameAgentV3Tests(unittest.TestCase):
    def _video(self, root: Path) -> Path:
        path = root / "current.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 4.0, (96, 64))
        self.assertTrue(writer.isOpened())
        try:
            for index in range(80):
                writer.write(np.full((64, 96, 3), 30 + index % 40, dtype=np.uint8))
        finally:
            writer.release()
        return path

    @staticmethod
    def _observation(identity: str, confidence: float = 0.9) -> dict:
        return {
            "image_group": 1,
            "circuit_state": "energized",
            "identity": identity,
            "pointer_state": "normal_rightward",
            "pointer_scale_position": "mid",
            "terminal_occupancy_left_middle_right": ["occupied", "occupied", "empty"],
            "selected_range_label": "0.6" if identity == "ammeter" else "3",
            "plugged_terminal_visible": "connected",
            "range_assessment": "appropriate",
            "confidence": confidence,
            "evidence": f"visible {identity}",
        }

    def test_second_round_has_adjacent_stage_and_global_search(self) -> None:
        first = {
            "selected_frames": [
                {"timestamp_seconds": 5.0},
                {"timestamp_seconds": 10.0},
            ],
            "qwen_observation": {
                "observations": [
                    {"image_group": 1, "identity": "voltmeter", "confidence": 0.9},
                    {"image_group": 2, "identity": "unknown", "confidence": 0.2},
                ]
            },
        }
        windows = agent._second_round_windows(
            first_round=first,
            original_windows=[{"source": "measurement_1", "start_seconds": 2.0, "end_seconds": 18.0}],
            duration=20.0,
            fps=4.0,
            known={20, 40},
        )
        sources = {item["source"] for item in windows}
        self.assertIn("adaptive_adjacent_missing_role", sources)
        self.assertIn("missing_role_stage_search", sources)
        self.assertIn("missing_role_global_search", sources)
        self.assertLessEqual(len(windows), agent.MAX_FRAMES_PER_ROUND)
        self.assertTrue(all(item["start_seconds"] == item["end_seconds"] for item in windows))

    def test_stage_runs_are_discovered_in_nested_current_run_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nested = root / "pipeline" / "actions"
            nested.mkdir(parents=True)
            (nested / "result.json").write_text(
                json.dumps(
                    {
                        "observed_stage_runs": [
                            {"stage": "measurement_1", "start_seconds": 2.0, "end_seconds": 4.0}
                        ]
                    }
                ),
                encoding="utf-8",
            )
            ignored = root / "frame_agent" / "round_01"
            ignored.mkdir(parents=True)
            (ignored / "result.json").write_text(
                json.dumps(
                    {
                        "observed_stage_runs": [
                            {"stage": "measurement_2", "start_seconds": 8.0, "end_seconds": 10.0}
                        ]
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                [{"stage": "measurement_1", "start_seconds": 2.0, "end_seconds": 4.0}],
                agent._stage_runs(root),
            )

    def test_stage_runs_keep_rich_five_stage_subinterval_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rich = root / "a_action_result.json"
            poor = root / "z_boundary_copy.json"
            stage = {
                "stage": "recording_1",
                "start_seconds": 10.0,
                "end_seconds": 30.0,
                "stage_semantics": "measurement_and_recording_cycle",
                "merged_measurement_recording": True,
                "contains_measurement_evidence": True,
                "contains_writing_evidence": True,
                "measurement_subintervals": [
                    {
                        "action_type": "measurement_action",
                        "start_seconds": 12.0,
                        "end_seconds": 18.0,
                    }
                ],
                "writing_subintervals": [
                    {
                        "action_type": "writing_action",
                        "start_seconds": 22.0,
                        "end_seconds": 30.0,
                    }
                ],
            }
            rich.write_text(json.dumps({"observed_stage_runs": [stage]}), encoding="utf-8")
            poor.write_text(
                json.dumps(
                    {
                        "source_observed_stage_runs": [
                            {"stage": "recording_1", "start_seconds": 10.0, "end_seconds": 30.0}
                        ]
                    }
                ),
                encoding="utf-8",
            )

            stages = agent._stage_runs(root)
            windows = agent.base._windows(stages, duration=40.0)

        self.assertEqual(stage["measurement_subintervals"], stages[0]["measurement_subintervals"])
        self.assertEqual("recording_1.measurement_action", windows[0]["source"])
        self.assertEqual(11.5, windows[0]["start_seconds"])
        self.assertEqual(18.5, windows[0]["end_seconds"])

    def test_diverse_selection_prefers_a_new_meter_track(self) -> None:
        def record(frame: int, quality: float, track: str) -> dict:
            candidate = {"candidate_id": f"candidate_{frame}", "quality": quality, "track_id": track}
            return {
                "frame_number": frame,
                "timestamp_seconds": float(frame),
                "window_source": "measurement_1",
                "sharpness": 10.0,
                "candidates": [candidate],
                "model_candidates": [candidate],
            }

        selected = agent._select_diverse_frame_records(
            [record(1, 1.0, "meter_track_01"), record(2, 0.95, "meter_track_01"), record(3, 0.8, "meter_track_02")],
            limit=2,
        )
        tracks = {next(iter(agent._record_tracks(item))) for item in selected}
        self.assertEqual({"meter_track_01", "meter_track_02"}, tracks)

    def test_roles_accumulate_across_rounds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "run"
            run_dir.mkdir()
            video = self._video(root)
            (run_dir / "stages.json").write_text(
                json.dumps(
                    {
                        "observed_stage_runs": [
                            {"stage": "measurement_1", "start_seconds": 1.0, "end_seconds": 8.0},
                            {"stage": "measurement_2", "start_seconds": 9.0, "end_seconds": 18.0},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            state = {
                "run_id": "cumulative_fixture",
                "mode": "execute",
                "video": {"path": str(video), "sha256": sha256(video)},
            }
            calls = 0

            def fake_export(frame: dict, _root: Path) -> dict:
                candidate = {"candidate_id": "candidate_01", "quality": 0.9, "track_id": "meter_track_01"}
                return {**frame, "candidates": [candidate], "model_candidates": [candidate]}

            def fake_qwen(*_args: object, **_kwargs: object) -> dict:
                nonlocal calls
                calls += 1
                role = "ammeter" if calls == 1 else "voltmeter"
                return {
                    "measurement_active": True,
                    "observations": [self._observation(role)],
                    "overall_confidence": 0.9,
                    "evidence": f"only {role} in this round",
                }

            with mock.patch("meter_rubrics._export_candidates", side_effect=fake_export), mock.patch(
                "skills.dynamic_meter_reading.prepare_frames", return_value={"skill_version": "test", "tracks": []}
            ), mock.patch("meter_rubrics._call_qwen", side_effect=fake_qwen):
                result = agent.run_adaptive_frame_agent(
                    run_dir=run_dir,
                    state=state,
                    model_config={"base_url": "https://example.invalid/v1", "model": "qwen"},
                )

        self.assertEqual(2, result["round_count"])
        self.assertEqual(["ammeter", "voltmeter"], result["visible_roles"])
        self.assertEqual([], result["missing_roles"])
        self.assertTrue(result["meter_pair_complete"])
        self.assertEqual("frame_evidence_ready", result["status"])
        self.assertEqual(1, len(result["role_evidence"]["ammeter"]))
        self.assertEqual(1, len(result["role_evidence"]["voltmeter"]))
        self.assertFalse(result["video_id_used_for_routing"])
        self.assertFalse(result["historical_artifacts_used"])
        self.assertFalse(result["fixed_video_roi_used"])


if __name__ == "__main__":
    unittest.main()
