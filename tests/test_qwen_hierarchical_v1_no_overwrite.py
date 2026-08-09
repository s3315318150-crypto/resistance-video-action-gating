from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import qwen_experiment_action_hierarchical_v1 as pipeline  # noqa: E402


class NoOverwriteTests(unittest.TestCase):
    def test_video_directory_names_are_unique_and_cannot_escape(self) -> None:
        first = pipeline.safe_video_directory_name("甲.mp4")
        second = pipeline.safe_video_directory_name("乙.mp4")
        traversal = pipeline.safe_video_directory_name("..")
        self.assertNotEqual(first, second)
        self.assertNotIn("\\", first)
        self.assertNotIn("/", first)
        self.assertNotIn("..", traversal)

    def test_qwen_transport_error_is_returned_as_retryable_attempt(self) -> None:
        with mock.patch.object(pipeline, "_call_qwen", side_effect=TimeoutError("gateway timeout")):
            result = pipeline._attempt_qwen(object(), "prompt", [], 100)
        self.assertFalse(result["parsed"])
        self.assertEqual("TimeoutError", result["transport_error_type"])

    def test_boundary_global_order_quarantines_non_monotonic_answer(self) -> None:
        boundaries = [
            {"boundary_id": "b001", "last_from_seconds": 1.0, "first_to_seconds": 2.0, "selected_seconds": 2.0},
            {"boundary_id": "b002", "last_from_seconds": 1.5, "first_to_seconds": 1.8, "selected_seconds": 1.8},
        ]
        accepted, rejected = pipeline._enforce_boundary_monotonicity(boundaries)
        self.assertEqual(["b001"], [item["boundary_id"] for item in accepted])
        self.assertEqual("global_boundary_order_not_increasing", rejected[0]["global_validation_error"])

    def test_nonempty_boundary_uncertainty_requires_review_even_with_high_confidence(self) -> None:
        boundary_pass = {
            "valid": True,
            "sample_interval_seconds": 1.0,
            "input_frames": [
                {"image_id": "f1", "frame_number": 1, "timestamp_seconds": 1.0},
                {"image_id": "f2", "frame_number": 2, "timestamp_seconds": 2.0},
            ],
            "parsed_result": {
                "decision": "observed",
                "last_from_frame_id": "f1",
                "first_to_frame_id": "f2",
                "confidence": 0.95,
                "evidence": "可见变化",
                "uncertainty": "手部遮挡一部分",
            },
        }
        observed = pipeline._observed_boundary_from_pass(boundary_pass, 0.72)
        self.assertIsNotNone(observed)
        self.assertTrue(observed["needs_review"])

    def test_prepare_only_does_not_construct_qwen_and_keeps_existing_sentinel(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_summary = root / "source.json"
            source_summary.write_text(
                '{"records":[{"source_video_id":"demo.mp4","source_manifest":"manifest.json","segment":{"start_seconds":0.0,"end_seconds":2.0,"segment_valid":true,"segment_errors":[]}}]}',
                encoding="utf-8",
            )
            schema = ROOT / "configs" / "action_schemas" / "resistance_7stage_no_battery_v1.json"
            legacy = root / "legacy-output"
            legacy.mkdir()
            sentinel = legacy / "sentinel.txt"
            sentinel.write_text("do not modify", encoding="utf-8")

            def fake_prepare(provenance, video_dir, args):
                return {
                    "fixed_start": 0.0,
                    "fixed_end": 2.0,
                    "prepared_windows": [{"window_id": "w000"}],
                    "source_record": {"window_frame_reference_count": 1, "overlap_reference_savings": 0},
                    "frame_registry": {0: {}},
                }

            with mock.patch.object(pipeline, "prepare_video", side_effect=fake_prepare), mock.patch.object(
                pipeline.qwen_base, "OpenAI", side_effect=AssertionError("prepare-only must not call Qwen")
            ):
                code = pipeline.main(
                    [
                        "--segment-source",
                        str(source_summary),
                        "--schema",
                        str(schema),
                        "--output-root",
                        str(root / "new-output"),
                        "--run-id",
                        "prepare-test",
                        "--prepare-only",
                    ]
                )
            self.assertEqual(0, code)
            self.assertEqual("do not modify", sentinel.read_text(encoding="utf-8"))
            self.assertTrue((root / "new-output" / "prepare-test" / "summary.json").is_file())

    def test_dense_low_confidence_boundary_is_marked_for_review(self) -> None:
        frames = [
            {"image_id": "frame_00000030", "frame_number": 30, "timestamp_seconds": 1.0},
            {"image_id": "frame_00000060", "frame_number": 60, "timestamp_seconds": 2.0},
        ]
        boundary_pass = {
            "valid": True,
            "sample_interval_seconds": 0.5,
            "input_frames": frames,
            "parsed_result": {
                "decision": "observed",
                "last_from_frame_id": "frame_00000030",
                "first_to_frame_id": "frame_00000060",
                "confidence": 0.5,
                "evidence": "低置信边界",
                "uncertainty": "遮挡",
            },
        }
        prepared = {"fixed_start": 0.0, "fixed_end": 10.0, "video_dir": Path("unused")}
        candidate = {
            "boundary_id": "b001",
            "from_stage": "circuit_wiring",
            "to_stage": "recording_1",
            "coarse_last_from_frame_id": "frame_00000030",
            "coarse_first_to_frame_id": "frame_00000060",
            "coarse_last_from_seconds": 1.0,
            "coarse_first_to_seconds": 2.0,
            "coarse_selected_seconds": 2.0,
        }
        args = SimpleNamespace(
            boundary_context_seconds=10.0,
            dense_boundary_context_seconds=3.0,
            boundary_min_confidence=0.72,
            sample_interval_seconds=2.0,
        )
        with mock.patch.object(pipeline, "_run_boundary_pass", side_effect=[boundary_pass, boundary_pass]), mock.patch.object(
            pipeline, "write_json_atomic"
        ):
            boundaries, review = pipeline._refine_boundaries(
                prepared,
                [candidate],
                object(),
                {"circuit_wiring": "连线", "recording_1": "第一次记录"},
                args,
            )
        self.assertTrue(boundaries[0]["needs_review"])
        self.assertIn("boundary_dense_needs_review:b001", review)


if __name__ == "__main__":
    unittest.main()
