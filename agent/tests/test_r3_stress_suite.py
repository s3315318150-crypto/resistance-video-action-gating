from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

AGENT_ROOT = Path(__file__).resolve().parent.parent
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from resistance_agent import opencv_switch_overlap
from resistance_agent import r3_stress_suite as stress


class R3StressSuiteTests(unittest.TestCase):
    def make_video(self, root: Path, name: str = "source.mp4", frames: int = 12) -> Path:
        path = root / name
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (96, 64)
        )
        self.assertTrue(writer.isOpened())
        for index in range(frames):
            frame = np.full((64, 96, 3), index * 10, dtype=np.uint8)
            cv2.rectangle(frame, (10 + index, 20), (28 + index, 35), (0, 100, 255), -1)
            writer.write(frame)
        writer.release()
        self.assertTrue(path.is_file())
        return path

    def stages(self) -> list[dict]:
        return [
            {
                "stage": "circuit_wiring",
                "start_seconds": 0.1,
                "end_seconds": 0.8,
            },
            {
                "stage": "measurement_1",
                "start_seconds": 0.8,
                "end_seconds": 1.0,
            },
        ]

    def test_sampling_phase_is_periodic_and_stays_inside_window(self) -> None:
        windows = [
            {
                "window_id": "w1",
                "stage": "circuit_wiring",
                "start_seconds": 1.0,
                "end_seconds": 2.0,
            }
        ]
        negative = opencv_switch_overlap._sampling_targets(
            windows, 30.0, 1000, 5.0, -0.1
        )
        positive = opencv_switch_overlap._sampling_targets(
            windows, 30.0, 1000, 5.0, 0.1
        )
        baseline = opencv_switch_overlap._sampling_targets(
            windows, 30.0, 1000, 5.0, 0.0
        )
        self.assertEqual(negative, positive)
        self.assertNotEqual(baseline, positive)
        self.assertTrue(all(1.0 <= item["timestamp_seconds"] <= 2.0 for item in positive))

    def test_stage_variant_rebinds_name_and_shifts_only_wiring(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stages.json"
            stress.write_stage_variant(
                source_stages=self.stages(),
                target_path=path,
                target_video_name="anonymous.mp4",
                duration_seconds=2.0,
                source_fps=10.0,
                wiring_shift_seconds=0.2,
            )
            value = stress.read_json(path)
            self.assertEqual("anonymous.mp4", value["source_video_id"])
            wiring, measurement = value["source_observed_stage_runs"]
            self.assertEqual(0.3, wiring["start_seconds"])
            self.assertEqual(1.0, wiring["end_seconds"])
            self.assertEqual(0.8, measurement["start_seconds"])
            self.assertFalse(value["video_id_used_for_routing"])

    def test_quality_variants_preserve_frame_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.make_video(root)
            for variant in ("720p", "blur", "brightness", "recompress"):
                target = root / variant / "variant.mp4"
                result = stress.create_quality_variant(source, target, variant)
                self.assertEqual(12, result["frame_count"])
                self.assertTrue(target.is_file())
                self.assertFalse(result["audio_preserved"])
            downscaled = stress.create_quality_variant(
                source, root / "1080p" / "variant.mp4", "1080p"
            )
            self.assertEqual(96, downscaled["width"])
            self.assertEqual(64, downscaled["height"])

    def test_identity_test_uses_same_parameters_and_detects_metric_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = self.make_video(root)
            summary = root / "source_stages.json"
            stress.write_json(
                summary,
                {
                    "source_video_id": video.name,
                    "source_observed_stage_runs": self.stages(),
                },
            )
            common_metrics = {
                "decision": "pass",
                "predicted_score": 1,
                "baseline_sample_count": 5,
                "final_sample_count": 7,
                "baseline_switch_observation_count": 2,
                "final_switch_observation_count": 4,
                "request_count": 1,
                "supplemental_actual_new_frame_count": 2,
            }

            def fake_case(**kwargs: object) -> dict:
                return {
                    "candidate_windows": [
                        {
                            "stage": "circuit_wiring",
                            "start_seconds": 0.1,
                            "end_seconds": 0.8,
                        }
                    ],
                    "metrics": dict(common_metrics),
                }

            with mock.patch.object(stress, "run_agent_case", side_effect=fake_case):
                result = stress.run_identity_test(
                    video_path=video,
                    stage_summary_path=summary,
                    output_dir=root / "out",
                )
            self.assertTrue(result["same_content"])
            self.assertTrue(result["same_decision"])
            self.assertTrue(result["passed"])
            self.assertFalse(result["video_id_used_for_routing"])

    def test_aggregate_reports_compares_fixed_and_agent_without_accuracy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline.json"
            stress.write_json(
                baseline,
                {
                    "decision": "pass",
                    "confidence": 0.7,
                    "sample_count": 10,
                    "switch_tracked_observation_count": 4,
                    "switch_coverage": 0.4,
                    "sampling_phase_offset_seconds": 0.0,
                },
            )
            report = root / "agent.json"
            stress.write_json(
                report,
                {
                    "decision": "pass",
                    "predicted_score": 1,
                    "confidence": 0.6,
                    "stop_reason": "maximum_rounds_reached",
                    "candidate_windows": [{}],
                    "baseline_report_path": str(baseline),
                    "initial_evidence_quality": {"reasons": ["low_switch_coverage"]},
                    "final_evidence_quality": {
                        "sample_count": 14,
                        "switch_observation_count": 8,
                        "switch_coverage": 0.57,
                        "reasons": [],
                    },
                    "request_rounds": [{"round_number": 1}],
                    "request_count": 1,
                    "supplemental_actual_new_frame_count": 4,
                    "evidence_frame_count": 8,
                    "video_id_used_for_routing": False,
                    "historical_artifacts_used": False,
                    "fixed_video_roi_used": False,
                    "excel_accessed": False,
                    "ground_truth_sent_to_model": False,
                },
            )
            result = stress.aggregate_reports([("x", report)], root / "aggregate")
            self.assertFalse(result["accuracy_claimed"])
            self.assertEqual(10, result["total_baseline_samples"])
            self.assertEqual(14, result["total_final_samples"])
            self.assertEqual(0, result["decision_change_count"])
            self.assertGreaterEqual(result["total_runtime_seconds"], 0.0)
            self.assertEqual(
                "agent_output_artifact_mtime_span",
                result["rows"][0]["runtime_source"],
            )
            self.assertTrue((root / "aggregate" / "summary.csv").is_file())


if __name__ == "__main__":
    unittest.main()
