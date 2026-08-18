from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

AGENT_ROOT = Path(__file__).resolve().parent.parent
import sys

sys.path.insert(0, str(AGENT_ROOT / "resistance_agent"))

import r3_frame_agent_adapter  # noqa: E402
import r3_frame_sampling_agent as agent  # noqa: E402


class R3FrameSamplingAgentTests(unittest.TestCase):
    def _video(self, root: Path) -> Path:
        path = root / "current.mp4"
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (96, 64)
        )
        self.assertTrue(writer.isOpened())
        try:
            for index in range(180):
                frame = np.full((64, 96, 3), 20 + index % 80, dtype=np.uint8)
                writer.write(frame)
        finally:
            writer.release()
        return path

    def _windows(self) -> list[dict]:
        return [
            {
                "window_id": "circuit_wiring_001",
                "stage": "circuit_wiring",
                "start_seconds": 1.0,
                "end_seconds": 5.0,
            }
        ]

    def _weak_report(self) -> dict:
        frames = [
            {
                "window_id": "circuit_wiring_001",
                "stage": "circuit_wiring",
                "timestamp_seconds": 1.0 + index * 0.2,
                "frame_number": 30 + index * 6,
                "switch_visible": index == 0,
                "switch_state": "open" if index == 0 else None,
                "wiring_active": index in {3, 4},
                "same_frame_overlap": False,
            }
            for index in range(10)
        ]
        return {
            "decision": "pass",
            "confidence": 0.74,
            "reason": "no_same_frame_persistent_closed_switch_and_wiring_active",
            "sample_count": len(frames),
            "switch_tracked_observation_count": 1,
            "switch_coverage": 0.1,
            "switch_state_threshold": 0.5,
            "switch_state_observations": [
                {
                    "window_id": "circuit_wiring_001",
                    "stage": "circuit_wiring",
                    "timestamp_seconds": 1.0,
                    "frame_number": 30,
                    "state": "open",
                    "smoothed_bridge_score": 0.5,
                    "closed_persistence_count": 0,
                }
            ],
            "real_plug_transitions": [],
            "frames": frames,
            "same_frame_overlaps": [],
            "implementation_version": "r3_opencv_same_frame_overlap_v3",
            "implementation_fingerprint": "base-fingerprint",
        }

    def _fail_report(self, output_dir: Path) -> dict:
        frames = [
            {
                "window_id": "adaptive_round_01",
                "stage": "circuit_wiring",
                "timestamp_seconds": 1.8 + index * 0.2,
                "frame_number": 54 + index * 6,
                "switch_visible": True,
                "switch_state": "closed",
                "switch_closed_persistence_count": 3,
                "wiring_active": True,
                "same_frame_overlap": True,
                "switch_crop_path": None,
            }
            for index in range(3)
        ]
        path = output_dir / "fake_report.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
        return {
            "decision": "fail",
            "confidence": 0.91,
            "reason": "same_frame_persistent_closed_switch_and_wiring_active",
            "sample_count": 1,
            "switch_tracked_observation_count": 1,
            "switch_coverage": 1.0,
            "switch_state_threshold": 0.5,
            "switch_state_observations": [
                {
                    "window_id": "adaptive_round_01",
                    "stage": "circuit_wiring",
                    "timestamp_seconds": 1.8 + index * 0.2,
                    "frame_number": 54 + index * 6,
                    "bridge_score": 0.9,
                    "smoothed_bridge_score": 0.9,
                    "identity_score": 0.92,
                    "state": "closed",
                    "closed_persistence_count": 3,
                }
                for index in range(3)
            ],
            "real_plug_transitions": [
                {
                    "window_id": "adaptive_round_01",
                    "stage": "circuit_wiring",
                    "timestamp_seconds": 2.0,
                    "frame_number": 60,
                    "confidence": 0.92,
                    "support_frames": [
                        {
                            "window_id": "adaptive_round_01",
                            "frame_number": 54 + index * 6,
                        }
                        for index in range(3)
                    ],
                }
            ],
            "frames": frames,
            "same_frame_overlaps": frames,
            "implementation_version": "r3_opencv_same_frame_overlap_v3",
            "implementation_fingerprint": "base-fingerprint",
            "report_path": str(path.resolve()),
        }

    def test_plan_is_current_evidence_only_and_phase_shifted(self) -> None:
        report = self._weak_report()
        plan = agent.plan_frame_requests(
            report=report,
            candidate_windows=self._windows(),
            round_number=1,
            duration_seconds=6.0,
            source_fps=30.0,
            frame_count=180,
            known_frame_numbers={30, 36, 42, 48, 54, 60, 66, 72, 78, 84},
        )
        self.assertTrue(plan["requests"])
        self.assertEqual("current_video_observed_situation_only", plan["selection_basis"])
        self.assertFalse(plan["video_id_used_for_routing"])
        self.assertFalse(plan["historical_artifacts_used"])
        self.assertFalse(plan["fixed_video_roi_used"])
        first = plan["requests"][0]
        self.assertEqual(5.0, first["sampling_fps"])
        self.assertEqual(0.1, first["phase_offset_seconds"])
        self.assertIn(first["reason"], plan["evidence_quality"]["reasons"])
        self.assertTrue(
            set(first["expected_new_frame_numbers"]).isdisjoint(
                {30, 36, 42, 48, 54, 60, 66, 72, 78, 84}
            )
        )
        requested = [set(item["expected_frame_numbers"]) for item in plan["requests"]]
        for index, frames in enumerate(requested):
            self.assertTrue(all(frames.isdisjoint(other) for other in requested[index + 1 :]))
        self.assertTrue(plan["duplicate_frame_decode_prevented"])

    def test_plan_budget_and_round_limit(self) -> None:
        plan = agent.plan_frame_requests(
            report=self._weak_report(),
            candidate_windows=self._windows(),
            round_number=2,
            duration_seconds=6.0,
            source_fps=30.0,
            frame_count=180,
            max_requests=3,
            remaining_frame_budget=5,
        )
        self.assertLessEqual(plan["request_count"], 3)
        self.assertLessEqual(plan["expected_new_frame_count"], 5)
        self.assertTrue(all(item["request_type"] for item in plan["requests"]))

    def test_plan_preserves_trigger_reason_diversity(self) -> None:
        report = self._weak_report()
        report["switch_state_observations"].append(
            {
                "window_id": "circuit_wiring_001",
                "stage": "circuit_wiring",
                "timestamp_seconds": 4.6,
                "frame_number": 138,
                "state": "closed",
                "smoothed_bridge_score": 0.54,
                "closed_persistence_count": 2,
            }
        )
        plan = agent.plan_frame_requests(
            report=report,
            candidate_windows=self._windows(),
            round_number=2,
            duration_seconds=6.0,
            source_fps=30.0,
            frame_count=180,
            max_requests=3,
        )
        reasons = [item["reason"] for item in plan["requests"]]
        self.assertIn("switch_not_visible_during_wiring", reasons)
        self.assertIn("closed_persistence_boundary", reasons)
        self.assertEqual(
            "reason_diverse_priority_then_temporal_coverage",
            plan["selection_strategy"],
        )

    def test_candidate_windows_stay_inside_parent_wiring_stage(self) -> None:
        plan = agent.plan_frame_requests(
            report=self._weak_report(),
            candidate_windows=self._windows(),
            round_number=2,
            duration_seconds=6.0,
            source_fps=30.0,
            frame_count=180,
            max_requests=3,
        )
        self.assertTrue(plan["requests"])
        for request in plan["requests"]:
            candidate = request["candidate_window"]
            scoring = request["scoring_window"]
            self.assertGreaterEqual(candidate["start_seconds"], scoring["start_seconds"])
            self.assertLessEqual(candidate["end_seconds"], scoring["end_seconds"])
            self.assertFalse(request["outside_stage_frames_scored"])

    def test_partial_baseline_overlap_keeps_new_frames_at_non_integer_ratio(self) -> None:
        baseline_frames = set(agent._sample_frame_numbers(1.0, 5.0, 8.0, 48))
        plan = agent.plan_frame_requests(
            report=self._weak_report(),
            candidate_windows=self._windows(),
            round_number=1,
            duration_seconds=6.0,
            source_fps=8.0,
            frame_count=48,
            known_frame_numbers=baseline_frames,
            supplemental_frame_numbers=set(),
            max_requests=3,
        )
        self.assertTrue(plan["requests"])
        self.assertTrue(
            any(item["baseline_context_frame_count"] > 0 for item in plan["requests"])
        )
        self.assertTrue(
            all(item["expected_new_frame_count"] > 0 for item in plan["requests"])
        )
        requested = [set(item["expected_frame_numbers"]) for item in plan["requests"]]
        for index, frames in enumerate(requested):
            self.assertTrue(all(frames.isdisjoint(other) for other in requested[index + 1 :]))

    def test_long_trigger_episode_uses_temporal_coverage(self) -> None:
        frames = [
            {
                "window_id": "circuit_wiring_001",
                "stage": "circuit_wiring",
                "timestamp_seconds": float(second),
                "frame_number": second * 30,
                "switch_visible": False,
                "wiring_active": True,
            }
            for second in range(1, 30)
        ]
        report = {
            "decision": "pass",
            "sample_count": len(frames),
            "switch_tracked_observation_count": len(frames),
            "switch_coverage": 1.0,
            "switch_state_threshold": 0.8,
            "switch_state_observations": [],
            "real_plug_transitions": [
                {
                    "window_id": "circuit_wiring_001",
                    "stage": "circuit_wiring",
                    "timestamp_seconds": 15.0,
                    "frame_number": 450,
                }
            ],
            "frames": frames,
            "same_frame_overlaps": [],
        }
        plan = agent.plan_frame_requests(
            report=report,
            candidate_windows=[
                {
                    "window_id": "circuit_wiring_001",
                    "stage": "circuit_wiring",
                    "start_seconds": 0.0,
                    "end_seconds": 30.0,
                }
            ],
            round_number=1,
            duration_seconds=30.0,
            source_fps=30.0,
            frame_count=900,
            max_requests=3,
        )
        anchors = [item["anchor_seconds"] for item in plan["requests"]]
        self.assertEqual(3, len(anchors))
        self.assertLess(min(anchors), 5.0)
        self.assertGreater(max(anchors), 25.0)

    def test_exported_states_use_shared_threshold_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = self._video(root)
            reports = [
                (
                    "supplemental",
                    "round_01_request_01",
                    {
                        "frames": [
                            {
                                "window_id": "adaptive_round_01_request_01",
                                "stage": "circuit_wiring",
                                "timestamp_seconds": 2.0,
                                "frame_number": 60,
                                "switch_visible": True,
                                "switch_state": "closed",
                                "wiring_active": True,
                            }
                        ]
                    },
                )
            ]
            combined = {
                "frames": [
                    {
                        "window_id": "adaptive_round_01_request_01",
                        "stage": "circuit_wiring",
                        "timestamp_seconds": 2.0,
                        "frame_number": 60,
                        "switch_visible": True,
                        "switch_state": "open",
                        "wiring_active": True,
                        "same_frame_overlap": False,
                    }
                ]
            }
            exported = agent._export_evidence_frames(
                video,
                reports,
                combined,
                [
                    {
                        "request_id": "round_01_request_01",
                        "reason": "state_threshold_margin",
                        "expected_frame_numbers": [60],
                    }
                ],
                root / "out",
            )
            self.assertEqual(["open"], exported[0]["switch_states"])
            self.assertEqual(["closed"], exported[0]["local_switch_states"])
            self.assertEqual(1, exported[0]["image_group"])
            self.assertEqual(["state_threshold_margin"], exported[0]["trigger_reasons"])
            self.assertTrue(Path(exported[0]["frame_path"]).is_file())

    def test_independent_supplemental_fail_is_binary_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = self._video(root)
            calls: list[str] = []

            def fake_analyzer(**kwargs: object) -> dict:
                output_dir = Path(str(kwargs["output_dir"]))
                calls.append(str(output_dir))
                if "baseline_5fps" in str(output_dir):
                    report = self._weak_report()
                    report["report_path"] = str((output_dir / "report.json").resolve())
                    output_dir.mkdir(parents=True, exist_ok=True)
                    (output_dir / "report.json").write_text("{}", encoding="utf-8")
                    return report
                return self._fail_report(output_dir)

            result = agent.run_r3_frame_sampling_agent(
                video_path=video,
                candidate_windows=self._windows(),
                output_dir=root / "agent_run",
                analyzer=fake_analyzer,
            )
            self.assertEqual("fail", result["decision"])
            self.assertIn("shared_threshold_counterexample_confirmed", result["stop_reason"])
            self.assertTrue(result["final_result_is_binary"])
            self.assertGreaterEqual(len(calls), 2)
            self.assertTrue(Path(result["report_path"]).is_file())
            reopened = json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))
            self.assertEqual("fail", reopened["decision"])
            self.assertTrue(reopened["evidence_frames"])
            self.assertTrue(all("frame_id" in item for item in reopened["evidence_frames"]))

    def test_local_fail_does_not_override_shared_threshold(self) -> None:
        baseline = self._weak_report()
        baseline["switch_state_observations"] = [
            {
                "window_id": "circuit_wiring_001",
                "stage": "circuit_wiring",
                "timestamp_seconds": 1.0 + index * 0.2,
                "frame_number": 30 + index * 6,
                "bridge_score": 0.2,
                "identity_score": 0.9,
                "crop_path": "unused.jpg",
            }
            for index in range(6)
        ]
        supplemental = self._fail_report(Path(tempfile.gettempdir()))
        supplemental["switch_state_observations"] = [
            {
                "window_id": "adaptive_round_01",
                "stage": "circuit_wiring",
                "timestamp_seconds": 2.0 + index * 0.2,
                "frame_number": 60 + index * 6,
                "bridge_score": 0.6,
                "identity_score": 0.9,
                "crop_path": "unused.jpg",
            }
            for index in range(3)
        ]
        supplemental["frames"] = [
            {
                "window_id": "adaptive_round_01",
                "stage": "circuit_wiring",
                "timestamp_seconds": 2.0 + index * 0.2,
                "frame_number": 60 + index * 6,
            }
            for index in range(3)
        ]
        supplemental["real_plug_transitions"] = [
            {
                "window_id": "adaptive_round_01",
                "stage": "circuit_wiring",
                "timestamp_seconds": 2.2,
                "frame_number": 66,
                "confidence": 0.9,
                "support_frames": [
                    {
                        "window_id": "adaptive_round_01",
                        "frame_number": 60 + index * 6,
                    }
                    for index in range(3)
                ],
            }
        ]
        fused = agent._aggregate_reports([baseline, supplemental])
        self.assertEqual("pass", fused["decision"])
        self.assertEqual(0.8, fused["switch_state_threshold"])
        self.assertTrue(fused["shared_threshold_fusion"])

    def test_adapter_rejects_replay_and_does_not_select_by_record_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = self._video(root)
            summary = root / "stages.json"
            summary.write_text(
                json.dumps(
                    {
                        "records": [
                            {
                                "source_video_id": "8",
                                "observed_stage_runs": [
                                    {
                                        "stage": "circuit_wiring",
                                        "start_seconds": 1.0,
                                        "end_seconds": 4.0,
                                    }
                                ],
                            },
                            {
                                "source_video_id": "38",
                                "observed_stage_runs": [
                                    {
                                        "stage": "circuit_wiring",
                                        "start_seconds": 2.0,
                                        "end_seconds": 5.0,
                                    }
                                ],
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                r3_frame_agent_adapter.run_r3_frame_agent_from_current_stages(
                    video_path=video,
                    stage_summary_path=summary,
                    output_dir=root / "out",
                )
            summary.write_text(
                json.dumps(
                    {
                        "observed_stage_runs": [
                            {
                                "stage": "circuit_wiring",
                                "start_seconds": 1.0,
                                "end_seconds": 4.0,
                            }
                        ],
                        "replay_result": "old.json",
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                r3_frame_agent_adapter.run_r3_frame_agent_from_current_stages(
                    video_path=video,
                    stage_summary_path=summary,
                    output_dir=root / "out2",
                )

            for forbidden in (
                {"ground_truth": 0},
                {"historical_artifacts_used": True},
                {"historical_fallback_used": True},
                {"fixed_video_roi_used": True},
                {"selection_checkpoint_reused": True},
                {"excel_accessed": True},
                {"video_id_used_for_routing": True},
                {"ground_truth_sent_to_model": True},
            ):
                summary.write_text(
                    json.dumps({**forbidden, "observed_stage_runs": self._windows()}),
                    encoding="utf-8",
                )
                with self.assertRaises(ValueError):
                    r3_frame_agent_adapter.run_r3_frame_agent_from_current_stages(
                        video_path=video,
                        stage_summary_path=summary,
                        output_dir=root / "forbidden",
                    )

    def test_adapter_uses_broad_search_when_current_stages_are_empty(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = self._video(root)
            summary = root / "stages.json"
            summary.write_text(json.dumps({"observed_stage_runs": []}), encoding="utf-8")
            fake_agent = {
                "decision": "pass",
                "predicted_score": 1,
                "confidence": 0.55,
                "reason": "test",
                "request_count": 0,
                "supplemental_actual_new_frame_count": 0,
                "initial_evidence_quality": {},
                "final_evidence_quality": {},
                "stop_reason": "test",
                "report_path": str(root / "agent.json"),
                "evidence_frames": [],
            }
            (root / "agent.json").write_text("{}", encoding="utf-8")
            with mock.patch.object(
                r3_frame_agent_adapter,
                "run_r3_frame_sampling_agent",
                return_value=fake_agent,
            ) as run_agent:
                result = r3_frame_agent_adapter.run_r3_frame_agent_from_current_stages(
                    video_path=video,
                    stage_summary_path=summary,
                    output_dir=root / "out",
                )
            self.assertEqual(
                "broad_search",
                result["selected_skills"][0]["parameters"]["window_mode"],
            )
            windows = run_agent.call_args.kwargs["candidate_windows"]
            self.assertEqual(0.0, windows[0]["start_seconds"])
            self.assertGreater(windows[0]["end_seconds"], 0.0)

    def test_live_skill_parameters_control_agent_and_write_formal_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory) / "run"
            run_root.mkdir()
            (run_root / "state.json").write_text("{}", encoding="utf-8")
            video = self._video(run_root)
            summary = run_root / "boundary_summary.json"
            summary.write_text(
                json.dumps({"source_observed_stage_runs": self._windows()}),
                encoding="utf-8",
            )
            execution = {
                "skill_id": "switch.adaptive_frame_sampling",
                "rubric_ids": [3],
                "producer_tool": "run_switch_rubric",
                "parameters": {
                    "window_mode": "initial_wiring_only",
                    "sampling_fps": 5.0,
                    "roi_mode": "dynamic_current_frame_switch_and_plug",
                    "fusion_policy": "same_frame_closed_and_wiring_active",
                    "max_rounds": 1,
                    "max_requests_per_round": 2,
                    "max_supplemental_frames": 11,
                },
                "execution_fingerprint": "a" * 64,
            }

            def fake_agent(**kwargs: object) -> dict:
                output_dir = Path(str(kwargs["output_dir"]))
                report_path = output_dir / "r3_frame_sampling_agent_report.json"
                report_path.parent.mkdir(parents=True, exist_ok=True)
                report_path.write_text("{}", encoding="utf-8")
                return {
                    "decision": "pass",
                    "predicted_score": 1,
                    "confidence": 0.7,
                    "reason": "test",
                    "original_algorithm_version": "r3_opencv_same_frame_overlap_v3",
                    "original_algorithm_fingerprint": "base",
                    "sampling_policy": {"outside_stage_frames_scored": False},
                    "initial_evidence_quality": {},
                    "final_evidence_quality": {},
                    "request_rounds": [],
                    "requests": [],
                    "request_count": 0,
                    "supplemental_actual_new_frame_count": 0,
                    "stop_reason": "evidence_sufficient_or_no_new_frames",
                    "shared_threshold_fusion": {"enabled": True, "threshold": 0.8},
                    "evidence_frames": [],
                    "evidence_frame_count": 0,
                    "report_path": str(report_path.resolve()),
                }

            with mock.patch.object(
                r3_frame_agent_adapter,
                "run_r3_frame_sampling_agent",
                side_effect=fake_agent,
            ) as run_agent:
                result = r3_frame_agent_adapter.run_r3_frame_agent_live_skill(
                    video_path=video,
                    source_video_id=video.name,
                    video_id="association-only",
                    run_dir=run_root,
                    stage_summary_path=summary,
                    skill_execution=execution,
                    routing_policy="live_situation_skills.v1",
                )

            self.assertEqual("pass", result["rubric_3"]["decision"])
            self.assertTrue(Path(result["report_path"]).is_file())
            report = json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))
            self.assertEqual("formal_execute_evidence_producer", report["execution_scope"])
            self.assertFalse(report["video_id_used_for_routing"])
            self.assertFalse(report["historical_artifacts_used"])
            self.assertFalse(report["fixed_video_roi_used"])
            self.assertEqual(1, run_agent.call_args.kwargs["max_rounds"])
            self.assertEqual(2, run_agent.call_args.kwargs["max_requests_per_round"])
            self.assertEqual(11, run_agent.call_args.kwargs["max_supplemental_frames"])
            candidate = run_agent.call_args.kwargs["candidate_windows"][0]
            self.assertEqual("circuit_wiring", candidate["stage"])
            self.assertGreaterEqual(candidate["start_seconds"], 1.0)
            self.assertLessEqual(candidate["end_seconds"], 5.0)

    def test_adapter_opens_only_single_nested_current_run_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_root = root / "run"
            run_root.mkdir()
            (run_root / "state.json").write_text("{}", encoding="utf-8")
            video = self._video(run_root)
            nested = run_root / "boundaries" / "one" / "result.json"
            nested.parent.mkdir(parents=True)
            nested.write_text(
                json.dumps({"source_observed_stage_runs": self._windows()}),
                encoding="utf-8",
            )
            summary = run_root / "boundaries" / "summary.json"
            summary.write_text(
                json.dumps(
                    {
                        "records": [
                            {"source_video_id": "association-only", "result_path": str(nested)}
                        ]
                    }
                ),
                encoding="utf-8",
            )
            fake_agent = {
                "decision": "pass",
                "predicted_score": 1,
                "confidence": 0.8,
                "reason": "test",
                "request_count": 0,
                "supplemental_actual_new_frame_count": 0,
                "initial_evidence_quality": {},
                "final_evidence_quality": {},
                "stop_reason": "test",
                "report_path": str(run_root / "agent.json"),
                "evidence_frames": [],
            }
            (run_root / "agent.json").write_text("{}", encoding="utf-8")
            with mock.patch.object(
                r3_frame_agent_adapter,
                "run_r3_frame_sampling_agent",
                return_value=fake_agent,
            ) as run_agent:
                result = r3_frame_agent_adapter.run_r3_frame_agent_from_current_stages(
                    video_path=video,
                    stage_summary_path=summary,
                    output_dir=run_root / "out",
                    association_id="video-name-is-not-routing",
                )
            self.assertEqual(str(nested.resolve()), result["resolved_stage_result_path"])
            self.assertEqual(
                "initial_wiring_only",
                result["selected_skills"][0]["parameters"]["window_mode"],
            )
            self.assertNotIn("source_video_sha256", result)
            self.assertNotIn("stage_summary_sha256", result)
            self.assertFalse(result["rubric_3"]["formal_execute_integrated"])
            self.assertEqual(
                ["circuit_wiring"],
                [item["stage"] for item in run_agent.call_args.kwargs["candidate_windows"]],
            )

            outside = root / "historical_result.json"
            outside.write_text(
                json.dumps({"source_observed_stage_runs": self._windows()}),
                encoding="utf-8",
            )
            summary.write_text(
                json.dumps({"records": [{"result_path": str(outside)}]}),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                r3_frame_agent_adapter.run_r3_frame_agent_from_current_stages(
                    video_path=video,
                    stage_summary_path=summary,
                    output_dir=run_root / "out2",
                )


if __name__ == "__main__":
    unittest.main()
