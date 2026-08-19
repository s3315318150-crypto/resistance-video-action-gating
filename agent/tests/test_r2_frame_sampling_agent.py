from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys
import json
from unittest import mock

import cv2
import numpy as np


AGENT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AGENT_ROOT / "resistance_agent"))

import r2_frame_sampling_agent as agent  # noqa: E402
import remaining_rubrics  # noqa: E402


class R2FrameSamplingAgentTests(unittest.TestCase):
    def _video(self, root: Path) -> Path:
        path = root / "current.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (160, 96))
        self.assertTrue(writer.isOpened())
        try:
            for index in range(90):
                frame = np.full((96, 160, 3), 45, dtype=np.uint8)
                cv2.circle(frame, (48 + index // 30, 45), 20, (220, 220, 220), 2)
                cv2.rectangle(frame, (80, 38), (125, 53), (180, 180, 180), 2)
                writer.write(frame)
        finally:
            writer.release()
        return path

    def test_recording_only_builds_a_cycle(self) -> None:
        cycles = agent.build_observation_recording_cycles(
            {
                "observed_stage_runs": [
                    {"stage": "recording_1", "start_seconds": 0.8, "end_seconds": 1.6}
                ]
            },
            3.0,
        )
        self.assertEqual(1, len(cycles))
        self.assertEqual(["recording_1"], cycles[0]["anchor_stages_detected"])
        self.assertTrue(cycles[0]["missing_adjacent_action_may_be_unsegmented"])
        self.assertTrue(cycles[0]["recording_detected"])
        self.assertFalse(cycles[0]["measurement_detected"])

    def test_measurement_and_recording_merge(self) -> None:
        cycles = agent.build_observation_recording_cycles(
            {
                "observed_stage_runs": [
                    {"stage": "measurement_1", "start_seconds": 1.0, "end_seconds": 1.7},
                    {"stage": "recording_1", "start_seconds": 1.5, "end_seconds": 2.1},
                ]
            },
            4.0,
        )
        self.assertEqual(["measurement_1", "recording_1"], cycles[0]["anchor_stages_detected"])
        self.assertFalse(cycles[0]["missing_adjacent_action_may_be_unsegmented"])

    def test_cycle_expansion_stops_at_rewiring_and_next_cycle(self) -> None:
        cycles = agent.build_observation_recording_cycles(
            {
                "observed_stage_runs": [
                    {"stage": "circuit_wiring", "start_seconds": 0.0, "end_seconds": 8.0},
                    {"stage": "recording_1", "start_seconds": 10.0, "end_seconds": 12.0},
                    {"stage": "circuit_rewiring", "start_seconds": 14.0, "end_seconds": 18.0},
                    {"stage": "measurement_2", "start_seconds": 20.0, "end_seconds": 22.0},
                ]
            },
            30.0,
        )
        self.assertEqual([8.0, 14.0], cycles[0]["boundary_limits_seconds"])
        self.assertEqual([8.0, 14.0], cycles[0]["expanded_window_seconds"])
        self.assertEqual([18.0, 30.0], cycles[1]["boundary_limits_seconds"])

    def test_native_decode_binds_roi_views_to_frame(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = self._video(root)
            plan = agent.build_r2_candidate_plan(
                video,
                {"observed_stage_runs": [{"stage": "measurement_1", "start_seconds": 0.5, "end_seconds": 1.5}]},
                3.0,
                {"max_groups_per_cycle": 2, "quality_expand_threshold": -100.0},
            )
            self.assertGreaterEqual(len(plan), 1)
            rows = agent.decode_r2_evidence(video, plan, root / "evidence")
            self.assertEqual(len(plan), len(rows))
            for row in rows:
                self.assertTrue(Path(row["panorama_path"]).is_file())
                self.assertTrue(Path(row["enhanced_path"]).is_file())
                self.assertEqual(row["frame_id"], f"frame_{row['frame_number']:08d}")
                topology = row["role_views"]["joint_topology"]
                self.assertTrue(Path(topology["native_path"]).is_file())
                self.assertEqual(4096, row["model_max_edge"])

    def test_dynamic_locator_has_no_video_id_argument(self) -> None:
        self.assertNotIn("video_id", agent.detect_dynamic_object_boxes.__code__.co_varnames)

    def test_r2_model_encoding_is_lossless_png(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "view.png"
            self.assertTrue(cv2.imwrite(str(path), np.full((24, 40, 3), 120, dtype=np.uint8)))
            encoded = remaining_rubrics.image_data_url(path, max_edge=4096, lossless=True)
            self.assertTrue(encoded.startswith("data:image/png;base64,"))

    def test_r2_qwen_batches_merge_global_image_groups(self) -> None:
        class Response:
            def __init__(self, payload: dict) -> None:
                self.payload = payload

            def __enter__(self) -> "Response":
                return self

            def __exit__(self, *_: object) -> None:
                return None

            def read(self) -> bytes:
                return json.dumps(self.payload).encode("utf-8")

        def response(count: int) -> Response:
            observations = [
                {
                    "image_group": index,
                    "frame_id": f"model_{index}",
                    "voltmeter_visible": True,
                    "resistor_visible": True,
                    "voltmeter_relation": "parallel_across_resistor",
                    "stable_state": True,
                    "confidence": 0.8,
                    "evidence": "visible",
                }
                for index in range(1, count + 1)
            ]
            content = json.dumps({"observations": observations, "overall_evidence": "batch"})
            return Response({"choices": [{"message": {"content": content}}]})

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "view.png"
            self.assertTrue(cv2.imwrite(str(image), np.full((24, 40, 3), 120, dtype=np.uint8)))
            rows = [
                {
                    "image_group": index,
                    "frame_id": f"frame_{index:08d}",
                    "frame_number": index,
                    "timestamp_seconds": float(index),
                    "stage": "observation_recording_cycle",
                    "panorama_path": str(image),
                    "enhanced_path": str(image),
                    "role_views": {},
                }
                for index in range(1, 4)
            ]
            with mock.patch.object(
                remaining_rubrics.urllib.request,
                "urlopen",
                side_effect=[response(1), response(1), response(1)],
            ):
                result = remaining_rubrics._call_qwen(
                    2,
                    rows,
                    {"base_url": "https://example.invalid/v1", "model": "qwen"},
                    root / "rubric_2.json",
                    execution_fingerprint="fixture",
                )
            self.assertEqual([1, 2, 3], [item["image_group"] for item in result["observations"]])
            self.assertEqual(
                [row["frame_id"] for row in rows],
                [item["frame_id"] for item in result["observations"]],
            )
            self.assertTrue((root / "rubric_2_batch_01.json").is_file())
            self.assertTrue((root / "rubric_2_batch_02.json").is_file())
            self.assertTrue((root / "rubric_2_batch_03.json").is_file())


if __name__ == "__main__":
    unittest.main()
