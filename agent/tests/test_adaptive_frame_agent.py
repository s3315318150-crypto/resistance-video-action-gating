from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import sys

import cv2
import numpy as np

AGENT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AGENT_ROOT / "resistance_agent"))

import adaptive_frame_agent  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class AdaptiveFrameAgentTests(unittest.TestCase):
    def _video(self, root: Path, offset: int = 0) -> Path:
        path = root / f"current_{offset}.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 2.0, (96, 64))
        self.assertTrue(writer.isOpened())
        try:
            for index in range(20):
                writer.write(np.full((64, 96, 3), 30 + offset + index, dtype=np.uint8))
        finally:
            writer.release()
        return path

    def test_frame_number_limit_preserves_endpoints_and_bound(self) -> None:
        limited = adaptive_frame_agent._limit_numbers(list(range(200)))
        self.assertEqual(adaptive_frame_agent.MAX_FRAMES_PER_ROUND, len(limited))
        self.assertEqual((0, 199), (limited[0], limited[-1]))

    def test_cv_preselection_preserves_stage_coverage_and_bound(self) -> None:
        frames = [
            {
                "timestamp_seconds": float(index),
                "window_source": "measurement_1" if index < 20 else "recording_1",
                "sharpness": float(index),
            }
            for index in range(40)
        ]
        selected = adaptive_frame_agent._preselect_for_cv(frames)
        self.assertEqual(adaptive_frame_agent.MAX_CV_FRAMES_PER_ROUND, len(selected))
        self.assertEqual({"measurement_1", "recording_1"}, {item["window_source"] for item in selected})

    def test_agent_requests_adjacent_frames_after_weak_qwen_view(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "run"
            run_dir.mkdir()
            video = self._video(root)
            (run_dir / "stages.json").write_text(
                json.dumps(
                    {
                        "observed_stage_runs": [
                            {"stage": "measurement_1", "start_seconds": 1.0, "end_seconds": 3.0}
                        ]
                    }
                ),
                encoding="utf-8",
            )
            state = {
                "run_id": "frame_agent_fixture",
                "mode": "execute",
                "video": {"path": str(video), "sha256": sha256(video)},
            }
            calls: list[int] = []

            def fake_export(frame: dict, _evidence_dir: Path) -> dict:
                return {
                    **frame,
                    "detection": {"source_image_width": 96, "source_image_height": 64},
                    "candidates": [{"candidate_id": "candidate_01", "quality": 0.9}],
                }

            def fake_qwen(*_args: object, **_kwargs: object) -> dict:
                calls.append(1)
                if len(calls) == 1:
                    return {
                        "measurement_active": True,
                        "observations": [
                            {
                                "image_group": 1,
                                "circuit_state": "unclear",
                                "identity": "unknown",
                                "pointer_state": "uncertain",
                                "pointer_scale_position": "uncertain",
                                "terminal_occupancy_left_middle_right": ["uncertain"] * 3,
                                "selected_range_label": None,
                                "plugged_terminal_visible": "uncertain",
                                "range_assessment": "unknown",
                                "confidence": 0.2,
                                "evidence": "meter face is occluded",
                            }
                        ],
                        "overall_confidence": 0.2,
                        "evidence": "weak first view",
                    }
                return {
                    "measurement_active": True,
                    "observations": [
                        {
                            "image_group": 1,
                            "circuit_state": "energized",
                            "identity": "ammeter",
                            "pointer_state": "normal_rightward",
                            "pointer_scale_position": "mid",
                            "terminal_occupancy_left_middle_right": ["occupied", "occupied", "empty"],
                            "selected_range_label": "0.6",
                            "plugged_terminal_visible": "connected",
                            "range_assessment": "appropriate",
                            "confidence": 0.9,
                            "evidence": "visible A face",
                        },
                        {
                            "image_group": 1,
                            "circuit_state": "energized",
                            "identity": "voltmeter",
                            "pointer_state": "normal_rightward",
                            "pointer_scale_position": "mid",
                            "terminal_occupancy_left_middle_right": ["occupied", "occupied", "empty"],
                            "selected_range_label": "3",
                            "plugged_terminal_visible": "connected",
                            "range_assessment": "appropriate",
                            "confidence": 0.9,
                            "evidence": "visible V face",
                        },
                    ],
                    "overall_confidence": 0.9,
                    "evidence": "both faces visible",
                }

            with mock.patch("meter_rubrics._export_candidates", side_effect=fake_export), mock.patch(
                "skills.dynamic_meter_reading.prepare_frames", return_value={"skill_version": "test", "tracks": []}
            ), mock.patch("meter_rubrics._call_qwen", side_effect=fake_qwen):
                result = adaptive_frame_agent.run_adaptive_frame_agent(
                    run_dir=run_dir,
                    state=state,
                    model_config={"base_url": "https://example.invalid/v1", "model": "qwen"},
                )

            self.assertEqual(2, result["round_count"])
            self.assertTrue(result["frame_useful"])
            self.assertTrue(result["meter_pair_complete"])
            self.assertEqual(["ammeter", "voltmeter"], result["visible_roles"])
            self.assertEqual(2, len(calls))
            self.assertFalse(result["video_id_used_for_routing"])
            self.assertFalse(result["historical_artifacts_used"])
            self.assertFalse(result["fixed_video_roi_used"])
            self.assertTrue(all(item.get("frame_id") and item.get("image_group_id") for item in result["selected_frames"]))
            self.assertTrue(Path(result["report_path"]).is_file())

    def test_same_stage_situation_has_same_routing_flags_for_another_video(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "run"
            run_dir.mkdir()
            video = self._video(root, offset=7)
            (run_dir / "stages.json").write_text(
                json.dumps({"observed_stage_runs": [{"stage": "measurement_1", "start_seconds": 1.0, "end_seconds": 3.0}]}),
                encoding="utf-8",
            )
            state = {"run_id": "different_identity_fixture", "mode": "execute", "video": {"path": str(video), "sha256": sha256(video)}}
            weak = {
                "measurement_active": True,
                "observations": [],
                "overall_confidence": 0.1,
                "evidence": "no complete meter",
            }
            with mock.patch("meter_rubrics._export_candidates", side_effect=lambda frame, _root: {**frame, "candidates": []}), mock.patch(
                "skills.dynamic_meter_reading.prepare_frames", return_value={"skill_version": "test", "tracks": []}
            ), mock.patch("meter_rubrics._call_qwen", return_value=weak):
                result = adaptive_frame_agent.run_adaptive_frame_agent(
                    run_dir=run_dir,
                    state=state,
                    model_config={"base_url": "https://example.invalid/v1", "model": "qwen"},
                    max_rounds=1,
                )
            self.assertEqual("current_video_observed_stage_and_current_frame_visual_evidence_only", result["selection_basis"])
            self.assertEqual("frame_evidence_partial", result["status"])
            self.assertTrue(result["request_limit_reached"])
            self.assertFalse(result["video_id_used_for_routing"])
            self.assertFalse(result["historical_artifacts_used"])
            self.assertFalse(result["fixed_video_roi_used"])


if __name__ == "__main__":
    unittest.main()
