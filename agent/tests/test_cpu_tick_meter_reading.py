import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np

# Keep the documented ``python -m unittest discover -s agent\tests`` command
# runnable from the repository root, where ``resistance_agent`` is not on the
# default import path.
AGENT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AGENT_ROOT))

from resistance_agent.skills import cpu_tick_meter_reading as cpu_tick
from resistance_agent import meter_rubrics


class CpuTickMeterReadingTests(unittest.TestCase):
    def test_active_groups_prefer_energized_current_run_frames(self) -> None:
        qwen = {
            "measurement_active": True,
            "observations": [
                {"image_group": 1, "circuit_state": "deenergized"},
                {"image_group": 2, "circuit_state": "energized"},
                {"image_group": 3, "circuit_state": "unclear"},
            ],
        }
        self.assertEqual([2], cpu_tick.active_image_groups(qwen, 3))

    def test_fine_tick_count_uses_30_divisions_and_half_up_rounding(self) -> None:
        result = cpu_tick._fine_tick_count(
            {
                "status": "grid_consensus_candidate",
                "consensus_zero_angle_deg": 120.0,
                "consensus_full_angle_deg": 40.0,
                "frame_positions": [{"pointer_angle_deg": 106.6666667}],
            },
            0.6,
            "A",
            lambda value: int(math.floor(value + 0.5)),
        )
        self.assertEqual(5, result["nearest_tick_index"])
        self.assertAlmostEqual(0.1, result["reading"])
        self.assertEqual("normal_rightward", result["pointer_state"])

    def test_range_assessment_uses_role_specific_small_and_large_ranges(self) -> None:
        small = {
            "range_max_value": 0.6,
            "raw_tick_index": 15.0,
            "nearest_tick_index": 15,
            "pointer_state": "normal_rightward",
        }
        large_low = {
            "range_max_value": 3.0,
            "raw_tick_index": 5.0,
            "nearest_tick_index": 5,
            "pointer_state": "normal_rightward",
        }
        self.assertEqual("appropriate", cpu_tick._range_assessment("ammeter", small))
        self.assertEqual("too_high", cpu_tick._range_assessment("ammeter", large_low))

    def test_unmatched_pointer_position_cannot_become_overrange_consensus(self) -> None:
        captured = []
        components = {
            "generic": SimpleNamespace(
                summarize_role=lambda _observations, role: {
                    "role": role,
                    "median_tick_index": None,
                    "range_max_value": 0.6,
                }
            ),
            "batch": SimpleNamespace(
                consensus_grid=lambda frames: captured.extend(frames)
                or {"status": "no_grid_candidate", "used_frame_count": 0}
            ),
            "count": SimpleNamespace(_nearest_tick=lambda value: int(math.floor(value + 0.5))),
        }
        result = cpu_tick._role_result(
            "ammeter",
            [],
            [
                {
                    "pointer_angle_deg": 47.0,
                    "per_frame_position": {"matched": False},
                    "grid": {"fitted": True, "state": "grid_candidate"},
                }
            ],
            components,
        )
        self.assertEqual([], captured)
        self.assertEqual("candidate_incomplete", result["status"])
        self.assertIsNone(result["pointer_state"])

    def test_dynamic_printed_grid_reading_is_primary_over_fixed_angle_reference(self) -> None:
        components = {
            "generic": SimpleNamespace(
                summarize_role=lambda _observations, role: {
                    "role": role,
                    "median_tick_index": 6,
                    "range_max_value": 3.0,
                }
            ),
            "batch": SimpleNamespace(
                consensus_grid=lambda _frames: {
                    "status": "grid_consensus_candidate",
                    "used_frame_count": 3,
                    "consensus_zero_angle_deg": 116.25,
                    "consensus_full_angle_deg": 40.5,
                    "frame_positions": [
                        {"pointer_angle_deg": 87.18156},
                        {"pointer_angle_deg": 86.151038},
                        {"pointer_angle_deg": 85.940654},
                    ],
                }
            ),
            "count": SimpleNamespace(_nearest_tick=lambda value: int(math.floor(value + 0.5))),
        }
        frames = [
            {"per_frame_position": {"matched": True}}
            for _ in range(3)
        ]
        result = cpu_tick._role_result("voltmeter", [], frames, components)
        self.assertEqual("reading_candidate", result["status"])
        self.assertEqual("dynamic_printed_grid_30_tick_count", result["reading_source"])
        self.assertAlmostEqual(1.2, result["reading"])
        self.assertFalse(result["calibrated_and_printed_tick_agree_within_2"])

    def test_cpu_normal_reading_resolves_missing_qwen_r5_and_r6_evidence(self) -> None:
        r5 = {
            "decision": "fail",
            "predicted_score": 0,
            "confidence": 0.4,
            "reason": "no_normal_pointer_deflection_found_after_temporal_and_roi_search",
            "diagnostics": {},
        }
        r6 = {
            "decision": "fail",
            "predicted_score": 0,
            "confidence": 0.4,
            "reason": "range_not_shown_appropriate_after_temporal_and_roi_search",
            "diagnostics": {},
        }
        evidence = {
            "status": "completed",
            "skill_version": cpu_tick.SKILL_VERSION,
            "roles": {
                "ammeter": {
                    "status": "reading_candidate",
                    "confidence": 0.88,
                    "pointer_state": "normal_rightward",
                    "range_assessment": "appropriate",
                    "reading": 0.3,
                    "unit": "A",
                    "calibrated_and_printed_tick_agree_within_2": True,
                }
            },
        }
        fused_r5, fused_r6 = cpu_tick.fuse_binary_results(r5, r6, evidence)
        self.assertEqual(("pass", 1), (fused_r5["decision"], fused_r5["predicted_score"]))
        self.assertEqual(("pass", 1), (fused_r6["decision"], fused_r6["predicted_score"]))

    def test_cpu_overrange_and_bad_range_force_binary_fail(self) -> None:
        base = {
            "decision": "pass",
            "predicted_score": 1,
            "confidence": 0.6,
            "reason": "qwen_pass",
            "diagnostics": {},
        }
        evidence = {
            "status": "completed",
            "skill_version": cpu_tick.SKILL_VERSION,
            "roles": {
                "ammeter": {
                    "status": "pointer_state_candidate",
                    "confidence": 0.88,
                    "pointer_state": "overrange",
                    "range_assessment": "too_low",
                    "reading": None,
                    "unit": "A",
                    "calibrated_and_printed_tick_agree_within_2": None,
                }
            },
        }
        fused_r5, fused_r6 = cpu_tick.fuse_binary_results(base, base, evidence)
        self.assertEqual("fail", fused_r5["decision"])
        self.assertEqual("cpu_tick_grid_confirms_abnormal_pointer_position", fused_r5["reason"])
        self.assertEqual("fail", fused_r6["decision"])
        self.assertEqual("cpu_tick_grid_confirms_range_mismatch", fused_r6["reason"])

    def test_current_frame_reader_combines_all_four_components(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in cpu_tick.COMPONENT_FILES:
                (root / name).write_text("fixture", encoding="ascii")
            calibration = root / "calibration.json"
            calibration.write_text("{}", encoding="ascii")
            terminals = root / "terminals"
            terminals.mkdir()
            frame_path = root / "frame_00000001_000001.000s.jpg"
            cv2.imwrite(str(frame_path), np.full((520, 640, 3), 230, dtype=np.uint8))

            def process(frame, role, *_args):
                range_max = 0.6 if role == "ammeter" else 3.0
                return {
                    "frame": str(frame),
                    "timestamp_seconds": 1.0,
                    "expected_role": role,
                    "face_localized": True,
                    "rectified_face_path": str(frame),
                    "pointer": {
                        "detected": False,
                        "angle_deg": 80.0,
                        "anchor": [320.0, 410.0],
                        "reasons": ["pointer_line_too_occluded_by_red_lead"],
                    },
                    "tick": {"nearest_tick_index": 15},
                    "range": {"range_max_value": range_max},
                    "source_integrity": {"legacy_hash_field": "must_not_escape"},
                }

            def summarize(observations, role):
                return {
                    "role": role,
                    "median_tick_index": 15,
                    "range_max_value": 0.6 if role == "ammeter" else 3.0,
                    "status": "reading_candidate",
                }

            def consensus(frames):
                return {
                    "status": "grid_consensus_candidate",
                    "used_frame_count": len(frames),
                    "consensus_zero_angle_deg": 120.0,
                    "consensus_full_angle_deg": 40.0,
                    "frame_positions": [
                        {"pointer_angle_deg": item["pointer_angle_deg"]} for item in frames
                    ],
                }

            components = {
                "generic": SimpleNamespace(
                    ScanConfig=lambda: object(),
                    build_face_templates=lambda *_args: {},
                    build_terminal_templates=lambda *_args: {},
                    process_observation=process,
                    summarize_role=summarize,
                ),
                "single": SimpleNamespace(
                    detect_scale_ticks=lambda *_args: {
                        "regular_grid": {
                            "fitted": True,
                            "state": "grid_candidate",
                            "zero_angle_deg": 120.0,
                            "full_angle_deg": 40.0,
                            "total_major_divisions": 15,
                        }
                    },
                    pointer_grid_position=lambda *_args: {"matched": True},
                    draw_overlay=lambda image, *_args: image,
                ),
                "batch": SimpleNamespace(consensus_grid=consensus),
                "count": SimpleNamespace(_nearest_tick=lambda value: int(math.floor(value + 0.5))),
            }
            with mock.patch.object(cpu_tick, "_load_components", return_value=components):
                result = cpu_tick.run_cpu_tick_reader(
                    [{"frame_path": str(frame_path), "frame_number": 1, "image_group": 1}],
                    baseline_root=root,
                    calibration=calibration,
                    terminal_annotations=terminals,
                    output_dir=root / "output",
                )

        self.assertEqual("completed", result["status"])
        self.assertEqual(4, len(result["component_paths"]))
        self.assertFalse(result["video_id_used_for_routing"])
        self.assertFalse(result["historical_artifacts_used"])
        self.assertFalse(result["fixed_video_roi_used"])
        self.assertEqual("reading_candidate", result["roles"]["ammeter"]["status"])
        self.assertNotIn("source_integrity", result["roles"]["ammeter"]["observations"][0])

    def test_meter_rubrics_producer_fuses_cpu_tick_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "run"
            frame_path = root / "frame_00000000_000000.000s.jpg"
            cv2.imwrite(str(frame_path), np.zeros((48, 64, 3), dtype=np.uint8))
            video_path = root / "current.mp4"
            writer = cv2.VideoWriter(
                str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 2.0, (64, 48)
            )
            self.assertTrue(writer.isOpened())
            writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
            writer.release()
            action_result = root / "action_result.json"
            action_result.write_text(
                '{"observed_stage_runs":[{"stage":"measurement_1","start_seconds":0.0,"end_seconds":0.5}]}',
                encoding="ascii",
            )
            action_summary = root / "action_summary.json"
            action_summary.write_text(
                json_text := '{"records":[{"source_video_id":"current.mp4","result_path":"'
                + str(action_result).replace("\\", "\\\\")
                + '"}]}',
                encoding="ascii",
            )
            self.assertIn("current.mp4", json_text)
            checkpoint = {
                "algorithm_version": meter_rubrics.ALGORITHM_VERSION,
                "source_video_sha256": meter_rubrics.sha256(video_path),
                "routing_policy": None,
                "execution_fingerprint": None,
                "candidate_windows": [],
                "sample_count": 1,
                "dynamic_meter_identity": {"tracks": []},
                "selected_frames": [
                    {
                        "frame_path": str(frame_path),
                        "frame_number": 0,
                        "timestamp_seconds": 0.0,
                        "window_source": "measurement_1",
                        "candidates": [],
                    }
                ],
            }
            meter_rubrics.write_json(
                run_dir / "meter_rubrics" / "selected_frames_pre_qwen.json",
                checkpoint,
            )
            cpu_evidence = {
                "status": "completed",
                "skill_version": cpu_tick.SKILL_VERSION,
                "selection_basis": "current_run_active_measurement_frames_only",
                "video_id_used_for_routing": False,
                "historical_artifacts_used": False,
                "fixed_video_roi_used": False,
                "roles": {
                    "ammeter": {
                        "status": "reading_candidate",
                        "confidence": 0.88,
                        "pointer_state": "normal_rightward",
                        "range_assessment": "appropriate",
                        "reading": 0.3,
                        "unit": "A",
                        "reading_source": "dynamic_printed_grid_30_tick_count",
                        "calibrated_and_printed_tick_agree_within_2": True,
                    }
                },
            }
            base_r5 = {
                "decision": "fail",
                "predicted_score": 0,
                "confidence": 0.3,
                "reason": "no_normal_pointer_deflection_found_after_temporal_and_roi_search",
                "diagnostics": {},
            }
            base_r6 = {
                "decision": "fail",
                "predicted_score": 0,
                "confidence": 0.3,
                "reason": "range_not_shown_appropriate_after_temporal_and_roi_search",
                "diagnostics": {},
            }
            qwen = {
                "measurement_active": True,
                "observations": [{"image_group": 1, "circuit_state": "energized"}],
            }
            with mock.patch.object(
                meter_rubrics, "_call_qwen", return_value=qwen
            ), mock.patch.object(
                meter_rubrics, "reduce_results", return_value=(base_r5, base_r6)
            ), mock.patch.object(
                meter_rubrics.CPU_TICK_READER,
                "run_cpu_tick_reader",
                return_value=cpu_evidence,
            ) as cpu_call:
                result = meter_rubrics.run_meter_rubrics(
                    video_path=video_path,
                    source_video_id=video_path.name,
                    video_id="anonymous",
                    run_dir=run_dir,
                    model_config={"base_url": "https://example.invalid/v1", "model": "qwen"},
                    action_summary_path=action_summary,
                    closed_stable_stage_producer_config={
                        "enabled": False,
                        "producer_root": str(root),
                        "calibration": "calibration.json",
                        "terminal_annotations": "terminals",
                        "cpu_tick_grid": {"enabled": True, "max_frames": 1},
                    },
                )

            cpu_call.assert_called_once()
            self.assertEqual("pass", result["rubric_5"]["decision"])
            self.assertEqual("pass", result["rubric_6"]["decision"])
            report = meter_rubrics.read_json(Path(result["report_path"]))
            self.assertEqual("completed", report["cpu_tick_grid_evidence"]["status"])


if __name__ == "__main__":
    unittest.main()
