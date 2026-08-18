from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "resistance_agent"))

import r1_frame_sampling_agent as agent  # noqa: E402
import series_rubric  # noqa: E402


def _frame(second: float, motion: float, sharpness: float, *, count: int = 4) -> dict:
    return {
        "window_id": "broad_search_w001",
        "stage": "broad_search",
        "stage_run": 1,
        "start_seconds": 0.0,
        "end_seconds": 12.0,
        "review_end_seconds": 12.0,
        "timestamp_seconds": second,
        "motion_score": motion,
        "sharpness": sharpness,
        "device_localizations": [
            {"candidate_id": f"device_{index}", "track_id": f"track_{index}"}
            for index in range(count)
        ],
    }


class R1FrameSamplingAgentTests(unittest.TestCase):
    def test_local_roi_enhancement_upscales_without_mutating_native_crop(self) -> None:
        crop = np.full((90, 180, 3), 205, dtype=np.uint8)
        cv2.putText(crop, "A", (55, 70), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (20, 20, 20), 3)
        native = crop.copy()

        enhanced = series_rubric._enhance_model_roi(crop, target_long_edge=720)

        self.assertTrue(np.array_equal(native, crop))
        self.assertEqual(720, max(enhanced.shape[:2]))
        self.assertEqual(2.0, enhanced.shape[1] / enhanced.shape[0])

    def test_decode_exports_native_and_enhanced_ranked_rois(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "current_video.mp4"
            writer = cv2.VideoWriter(
                str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (480, 320)
            )
            self.assertTrue(writer.isOpened())
            try:
                frame = np.full((320, 480, 3), 225, dtype=np.uint8)
                cv2.rectangle(frame, (60, 70), (260, 250), (0, 130, 245), -1)
                cv2.rectangle(frame, (95, 180), (225, 225), (45, 190, 45), -1)
                cv2.circle(frame, (130, 202), 10, (15, 15, 15), -1)
                cv2.circle(frame, (190, 202), 10, (15, 15, 15), -1)
                for _ in range(10):
                    writer.write(frame)
            finally:
                writer.release()
            samples = [
                {
                    "window_id": "circuit_wiring_001_w001",
                    "stage": "circuit_wiring",
                    "stage_run": 1,
                    "start_seconds": 0.0,
                    "end_seconds": 0.9,
                    "review_end_seconds": 0.9,
                    "timestamp_seconds": 0.4,
                    "temporal_role": "stable_candidate",
                    "evidence_phase": "coarse_scan",
                }
            ]

            decoded = series_rubric._decode_and_export(
                video,
                samples,
                root / "evidence",
                roi_target_long_edge=900,
                max_model_roi_views_per_frame=1,
            )

            self.assertEqual(1, len(decoded))
            candidates = decoded[0]["device_localizations"]
            self.assertGreater(len(candidates), 0)
            self.assertEqual(1, sum(item["model_roi_selected"] for item in candidates))
            selected = next(item for item in candidates if item["model_roi_selected"])
            native_path = Path(selected["native_roi_path"])
            enhanced_path = Path(selected["enhanced_roi_path"])
            self.assertTrue(native_path.is_file())
            self.assertTrue(enhanced_path.is_file())
            self.assertEqual(selected["roi_path"], selected["enhanced_roi_path"])
            self.assertGreater(selected["roi_quality"]["sharpness"], 0.0)
            native_image = cv2.imread(str(native_path))
            enhanced_image = cv2.imread(str(enhanced_path))
            self.assertIsNotNone(native_image)
            self.assertIsNotNone(enhanced_image)
            self.assertGreaterEqual(max(enhanced_image.shape[:2]), max(native_image.shape[:2]))
            self.assertEqual("frame_00000004", decoded[0]["frame_id"])
            self.assertEqual(1, decoded[0]["image_group"])

    def test_unique_model_roi_prefers_anonymous_ammeter_context(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "anonymous_current_input.mp4"
            writer = cv2.VideoWriter(
                str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (640, 360)
            )
            self.assertTrue(writer.isOpened())
            try:
                frame = np.full((360, 640, 3), 225, dtype=np.uint8)
                cv2.line(frame, (255, 205), (40, 80), (20, 20, 210), 8)
                cv2.line(frame, (385, 205), (600, 285), (20, 20, 210), 8)
                for _ in range(10):
                    writer.write(frame)
            finally:
                writer.release()

            voltage = {
                "candidate_id": "candidate_voltage",
                "bbox_xyxy": [20, 20, 230, 145],
                "bbox_xyxy_normalized": [0.03125, 0.055556, 0.359375, 0.402778],
                "identity": "voltmeter",
                "identity_basis": "red_terminal_panel",
                "roi_source": "dynamic_detection",
                "identity_diagnostics": {
                    "identity": "voltmeter",
                    "green_panel": {"valid": False, "dark_terminal_count": 0},
                },
            }
            ammeter = {
                "candidate_id": "candidate_ammeter",
                "bbox_xyxy": [250, 145, 390, 250],
                "bbox_xyxy_normalized": [0.390625, 0.402778, 0.609375, 0.694444],
                "identity": "ammeter",
                "identity_basis": "green_terminal_panel",
                "roi_source": "dynamic_detection",
                "identity_diagnostics": {
                    "identity": "ammeter",
                    "green_panel": {"valid": True, "dark_terminal_count": 2},
                },
            }
            samples = [
                {
                    "window_id": "anonymous_wiring_w001",
                    "stage": "circuit_wiring",
                    "stage_run": 1,
                    "start_seconds": 0.0,
                    "end_seconds": 0.9,
                    "review_end_seconds": 0.9,
                    "timestamp_seconds": 0.4,
                    "temporal_role": "stable_candidate",
                    "evidence_phase": "coarse_scan",
                }
            ]

            with mock.patch.object(
                series_rubric,
                "_orange_device_boxes",
                return_value=[voltage["bbox_xyxy"], ammeter["bbox_xyxy"]],
            ), mock.patch.object(
                series_rubric,
                "_dynamic_device_candidates",
                return_value=[dict(voltage), dict(ammeter)],
            ):
                decoded = series_rubric._decode_and_export(
                    video,
                    samples,
                    root / "evidence",
                    max_model_roi_views_per_frame=1,
                )

            candidates = decoded[0]["device_localizations"]
            selected = [item for item in candidates if item["model_roi_selected"]]
            self.assertEqual(["candidate_ammeter"], [item["candidate_id"] for item in selected])
            chosen = selected[0]
            self.assertEqual(
                "ammeter_terminals_leads_and_visible_endpoints", chosen["model_roi_role"]
            )
            self.assertEqual(
                "r1_current_frame_ammeter_context_first",
                chosen["model_roi_selection_basis"],
            )
            left, top, right, bottom = chosen["model_roi_bbox_xyxy"]
            self.assertLessEqual(left, 250)
            self.assertLessEqual(top, 145)
            self.assertGreaterEqual(right, 390)
            self.assertGreaterEqual(bottom, 250)
            self.assertGreaterEqual(right - left, round(640 * 0.92) - 1)
            self.assertLessEqual(left, round(640 * 0.08))
            self.assertGreaterEqual(right, round(640 * 0.92))
            self.assertGreaterEqual(bottom - top, round(360 * 0.72) - 1)

    def test_partial_current_green_panel_beats_sharper_voltmeter_roi(self) -> None:
        candidates = [
            {
                "candidate_id": "clear_voltage",
                "identity": "voltmeter",
                "identity_diagnostics": {
                    "identity": "voltmeter",
                    "green_panel": {"valid": False, "dark_terminal_count": 0},
                },
                "roi_quality": {"model_view_priority": 0.99},
            },
            {
                "candidate_id": "tracked_partial_meter",
                "identity": "voltmeter",
                "identity_diagnostics": {
                    "identity": "unknown",
                    "green_panel": {
                        "valid": False,
                        "dark_terminal_count": 1,
                        "aspect_ratio": 2.4,
                        "orange_adjacency": 0.14,
                        "component": {"fill_ratio": 0.33},
                    },
                },
                "roi_quality": {"model_view_priority": 0.4},
            },
        ]

        ranked = series_rubric._rank_r1_model_roi_candidates(candidates)

        self.assertEqual("tracked_partial_meter", ranked[0]["candidate_id"])
        self.assertEqual(2, ranked[0]["model_roi_selection_features"]["identity_tier"])

    def test_stable_plateau_is_selected_from_current_motion(self) -> None:
        frames = [
            _frame(0.0, 1.0, 20.0),
            _frame(0.5, 8.0, 80.0),
            _frame(1.0, 7.0, 90.0),
            _frame(1.5, 0.2, 120.0),
            _frame(2.0, 0.1, 140.0),
            _frame(2.5, 0.1, 130.0),
            _frame(3.0, 9.0, 100.0),
            _frame(3.5, 0.2, 150.0),
            _frame(4.0, 0.2, 160.0),
        ]
        result = agent.select_initial_evidence(frames, stable_per_stage_run=2, recovery_per_stage_run=0)
        stable = result["stable_frames"]
        self.assertEqual(2, len(stable))
        self.assertTrue(all(item["frame_agent_role"] == "stable_topology" for item in stable))
        self.assertTrue(all(item["temporal_role"] == "stable_candidate" for item in stable))
        self.assertTrue(all(1.5 <= item["timestamp_seconds"] <= 4.0 for item in stable))

    def test_transition_burst_contains_before_during_after(self) -> None:
        anchor = {
            **_frame(5.0, 12.0, 90.0),
            "timestamp_seconds": 5.0,
        }
        samples = agent.transition_burst_samples([anchor], duration_seconds=8.0, fps=5.0, radius_seconds=1.0)
        self.assertEqual(11, len(samples))
        self.assertEqual({"before", "during", "after"}, {item["transition_position"] for item in samples})
        self.assertTrue(all(item["frame_agent_role"] == "connection_transition" for item in samples))
        self.assertTrue(all(item["evidence_phase"] == "dense_confirmation" for item in samples))

    def test_transition_anchor_priority_is_not_reordered_to_earliest_time(self) -> None:
        frames = [
            _frame(0.0, 0.1, 60.0),
            _frame(0.5, 2.0, 60.0),
            _frame(1.0, 0.1, 60.0),
            _frame(4.0, 0.1, 60.0),
            _frame(4.5, 15.0, 60.0, count=1),
            _frame(5.0, 0.1, 60.0),
        ]
        result = agent.select_initial_evidence(
            frames,
            stable_per_stage_run=1,
            recovery_per_stage_run=0,
            max_transition_anchors=1,
        )
        self.assertEqual(4.5, result["transition_anchors"][0]["timestamp_seconds"])

    def test_first_wiring_window_late_transition_is_reserved(self) -> None:
        candidates = [
            {
                **_frame(3.0, 1.0, 60.0),
                "start_seconds": 0.0,
                "end_seconds": 16.0,
                "timestamp_seconds": 3.0,
                "scout_transition_score": 8.0,
            },
            {
                **_frame(12.0, 1.0, 60.0),
                "start_seconds": 0.0,
                "end_seconds": 16.0,
                "timestamp_seconds": 12.0,
                "scout_transition_score": 6.0,
            },
            {
                **_frame(22.0, 1.0, 60.0),
                "window_id": "broad_search_w002",
                "start_seconds": 16.0,
                "end_seconds": 32.0,
                "timestamp_seconds": 22.0,
                "scout_transition_score": 20.0,
            },
        ]

        selected = agent._stage_diverse_transition_anchors(candidates, 2)

        self.assertEqual(
            {12.0, 22.0},
            {item["timestamp_seconds"] for item in selected},
        )

    def test_each_observed_wiring_stage_gets_a_transition_anchor(self) -> None:
        first = [
            {**_frame(float(index), 12.0 if index == 2 else 0.1, 60.0), "stage": "circuit_wiring"}
            for index in range(4)
        ]
        second = [
            {
                **_frame(10.0 + index, 8.0 if index == 2 else 0.1, 60.0),
                "stage": "circuit_rewiring",
                "stage_run": 2,
                "window_id": "circuit_rewiring_w001",
                "start_seconds": 10.0,
                "end_seconds": 14.0,
                "review_end_seconds": 14.0,
            }
            for index in range(4)
        ]
        result = agent.select_initial_evidence(
            first + second,
            stable_per_stage_run=1,
            recovery_per_stage_run=0,
            max_transition_anchors=2,
        )
        self.assertEqual(
            {"circuit_wiring", "circuit_rewiring"},
            {item["stage"] for item in result["transition_anchors"]},
        )

    def test_each_current_coarse_window_gets_stable_coverage(self) -> None:
        first_window = [
            {**_frame(float(index), 0.1 if index != 2 else 9.0, 80.0), "window_id": "broad_search_w001"}
            for index in range(5)
        ]
        second_window = [
            {
                **_frame(10.0 + index, 0.1 if index != 2 else 9.0, 80.0),
                "window_id": "broad_search_w002",
            }
            for index in range(5)
        ]
        result = agent.select_initial_evidence(
            first_window + second_window,
            stable_per_stage_run=1,
            recovery_per_stage_run=0,
            max_transition_anchors=1,
        )
        self.assertEqual(
            {"broad_search_w001", "broad_search_w002"},
            {item["window_id"] for item in result["stable_frames"]},
        )

    def test_view_recovery_uses_clearer_current_frame(self) -> None:
        frames = [
            _frame(0.0, 2.0, 30.0, count=1),
            _frame(0.5, 2.0, 100.0, count=5),
            _frame(1.0, 2.0, 80.0, count=4),
            _frame(1.5, 2.0, 40.0, count=1),
        ]
        result = agent.select_initial_evidence(frames, stable_per_stage_run=1, recovery_per_stage_run=1, max_transition_anchors=1)
        recovery = result["recovery_frames"]
        self.assertEqual(1, len(recovery))
        self.assertIn(recovery[0]["timestamp_seconds"], {1.0, 1.5})
        self.assertEqual("view_recovery", recovery[0]["frame_agent_role"])
        self.assertFalse(result["video_id_used_for_routing"])
        self.assertFalse(result["historical_artifacts_used"])
        self.assertFalse(result["fixed_video_roi_used"])

    def test_supplemental_round_is_single_and_observation_driven(self) -> None:
        frames = [
            {**_frame(3.0, 4.0, 100.0), "image_group": 1, "frame_agent_role": "stable_topology"},
        ]
        observations = [
            {
                "image_group": 1,
                "direct_across_state": "candidate",
                "hands_or_plugs": "occluded",
            }
        ]
        result = agent.plan_supplemental_round(observations, frames, duration_seconds=6.0, max_frames=12)
        self.assertEqual(1, result["round_number"])
        self.assertEqual(1, result["max_rounds"])
        self.assertEqual("current_run_qwen_observations_only", result["selection_basis"])
        self.assertLessEqual(result["frame_count"], 12)
        self.assertNotIn("video_id", result)
        self.assertNotIn("sha256", result)

    def test_mirrored_single_direct_edge_does_not_trigger_double_endpoint(self) -> None:
        observation = {
            "direct_across_state": "unclear",
            "terminal_evidence": [
                {
                    "device": "ammeter",
                    "ammeter_terminal_id": "ammeter_left",
                    "far_endpoint": "battery_holder",
                    "path_relation": "direct",
                },
                {
                    "device": "battery_holder",
                    "battery_terminal_id": "battery_negative",
                    "far_endpoint": "ammeter",
                    "path_relation": "direct",
                },
            ],
        }
        self.assertIsNone(agent._observation_needs_supplement(observation))

    def test_two_by_two_distinct_direct_endpoints_trigger_supplement(self) -> None:
        observation = {
            "direct_across_state": "unclear",
            "terminal_evidence": [
                {"device": "ammeter", "ammeter_terminal_id": "ammeter_left", "far_endpoint": "battery_holder", "path_relation": "direct"},
                {"device": "ammeter", "ammeter_terminal_id": "ammeter_right", "far_endpoint": "battery_holder", "path_relation": "direct"},
                {"device": "battery_holder", "battery_terminal_id": "battery_negative", "far_endpoint": "ammeter", "path_relation": "direct"},
                {"device": "battery_holder", "battery_terminal_id": "battery_positive", "far_endpoint": "ammeter", "path_relation": "direct"},
            ],
        }
        self.assertEqual("suspected_direct_connection", agent._observation_needs_supplement(observation))

    def test_later_strong_candidate_wins_supplement_anchor(self) -> None:
        frames = [
            {**_frame(2.0, 0.2, 80.0), "image_group": 1, "frame_agent_role": "stable_topology"},
            {**_frame(8.0, 5.0, 120.0), "image_group": 2, "frame_agent_role": "connection_transition"},
            {**_frame(12.0, 0.4, 100.0), "image_group": 3, "frame_agent_role": "stable_topology"},
        ]
        observations = [
            {"image_group": 1, "direct_across_state": "candidate"},
            {
                "image_group": 2,
                "direct_across_state": "confirmed",
                "topology_visibility": "sufficient",
            },
        ]
        result = agent.plan_supplemental_round(
            observations,
            frames,
            duration_seconds=16.0,
            max_frames=11,
            fps=5.0,
            radius_seconds=1.0,
        )
        self.assertEqual(11, result["frame_count"])
        self.assertEqual(8.0, result["frames"][5]["transition_anchor_seconds"])

    def test_full_adaptive_execute_writes_binary_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "unseen_current_video.mp4"
            writer = cv2.VideoWriter(
                str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (160, 120)
            )
            self.assertTrue(writer.isOpened())
            try:
                for index in range(60):
                    frame = np.full((120, 160, 3), 225, dtype=np.uint8)
                    x = 15 if index < 20 else 55 if index < 30 else 90
                    cv2.rectangle(frame, (x, 35), (x + 35, 85), (0, 130, 245), -1)
                    writer.write(frame)
            finally:
                writer.release()
            action = root / "action_summary.json"
            action.write_text(
                '{"records":[{"source_video_id":"unseen_current_video.mp4",'
                '"observed_stage_runs":[{"stage":"circuit_wiring",'
                '"start_seconds":0.0,"end_seconds":5.5}]}]}',
                encoding="utf-8",
            )

            def fake_qwen(frames: list[dict], _config: dict, evidence_dir: Path, **_kwargs):
                artifact = evidence_dir / "qwen_batches" / "fixture.json"
                artifact.parent.mkdir(parents=True, exist_ok=True)
                artifact.write_text('{"fallback_used":false}', encoding="utf-8")
                observations = [
                    {
                        "image_group": item["image_group"],
                        "stable_state": True,
                        "hands_or_plugs": "hands_away",
                        "core_devices_visible": [
                            "ammeter",
                            "battery_holder",
                            "fixed_resistor",
                            "switch",
                        ],
                        "topology_visibility": "sufficient",
                        "final_topology": "single_series_loop",
                        "direct_across_state": "rejected",
                        "terminal_evidence": [],
                        "confidence": 0.9,
                    }
                    for item in frames
                ]
                return observations, [str(artifact.resolve())]

            with mock.patch.object(series_rubric, "call_qwen", side_effect=fake_qwen):
                result = series_rubric.run_series_rubric(
                    video,
                    video.name,
                    "unseen_current_video",
                    root / "run",
                    {},
                    action_summary_path=action,
                )
            self.assertEqual("pass", result["rubric_1"]["decision"])
            report = series_rubric.read_json(Path(result["report_path"]))
            frame_report = series_rubric.read_json(Path(report["frame_agent_report_path"]))
            self.assertEqual("series.adaptive_terminal_sampling", report["skill_execution"]["skill_id"])
            self.assertGreater(frame_report["scanned_frame_count"], 0)
            self.assertGreater(frame_report["total_model_frame_count"], 0)
            self.assertFalse(frame_report["video_id_used_for_routing"])
            self.assertFalse(frame_report["historical_artifacts_used"])
            self.assertFalse(frame_report["fixed_video_roi_used"])
            self.assertNotIn("sha256", Path(result["report_path"]).read_text(encoding="utf-8").lower())

    def test_boundary_record_reads_only_explicit_current_artifact_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            current = root / "current_boundary"
            current.mkdir()
            result_path = current / "result.json"
            result_path.write_text(
                '{"source_observed_stage_runs":[{"stage":"circuit_wiring",'
                '"start_seconds":1.0,"end_seconds":6.0}]}',
                encoding="utf-8",
            )
            summary = {
                "records": [
                    {
                        "source_video_id": "unseen.mp4",
                        "result_path": str(result_path),
                    }
                ]
            }
            without_current_root = series_rubric._boundary_record(
                summary, "unseen.mp4", "unseen"
            )
            current_record = series_rubric._boundary_record(
                summary, "unseen.mp4", "unseen", allowed_root=current
            )
            self.assertIsNone(without_current_root)
            self.assertEqual("circuit_wiring", current_record["observed_stage_runs"][0]["stage"])


if __name__ == "__main__":
    unittest.main()
