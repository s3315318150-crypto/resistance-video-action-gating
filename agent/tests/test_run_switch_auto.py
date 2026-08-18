from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


AGENT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AGENT_ROOT))
sys.path.insert(0, str(AGENT_ROOT / "resistance_agent"))

import run_switch_auto  # noqa: E402
import switch_rubric  # noqa: E402


class RunSwitchAutoTests(unittest.TestCase):
    EXECUTION_FINGERPRINT = "f" * 64

    def _plan(self) -> dict[str, object]:
        return {
            "selection_basis": "current_video_observed_situation_only",
            "observed_stages": [
                {"stage": "circuit_wiring", "start_seconds": 2.0, "end_seconds": 8.0}
            ],
            "skill_executions": [
                {
                    "skill_id": "switch.initial_wiring_dense",
                    "producer_tool": "run_switch_rubric",
                    "rubric_ids": [3],
                    "parameters": {
                        "window_mode": "initial_wiring_only",
                        **run_switch_auto.R3_AUTOMATIC_PARAMETERS,
                    },
                    "execution_fingerprint": self.EXECUTION_FINGERPRINT,
                    "implementation_version": run_switch_auto.R3_IMPLEMENTATION_VERSION,
                    "implementation_fingerprint": run_switch_auto.R3_IMPLEMENTATION_FINGERPRINT,
                }
            ],
            "video_id_used_for_routing": False,
            "historical_artifacts_used": False,
            "fixed_video_roi_used": False,
        }

    def _fixtures(self, root: Path, qwen_used: bool = False) -> Path:
        report = root / "report.json"
        report.write_text(
            json.dumps(
                {
                    "human_review_used": False,
                    "historical_fallback_used": False,
                    "excel_accessed": False,
                    "ground_truth_sent_to_model": False,
                    "qwen_used_for_decision": qwen_used,
                    "fixed_video_roi_used": False,
                    "skill_execution": {
                        "skill_id": "switch.initial_wiring_dense",
                        "producer_tool": "run_switch_rubric",
                        "rubric_ids": [3],
                        "parameters": {
                            "window_mode": "initial_wiring_only",
                            **run_switch_auto.R3_AUTOMATIC_PARAMETERS,
                        },
                        "effective_parameters": dict(run_switch_auto.R3_AUTOMATIC_PARAMETERS),
                        "execution_fingerprint": self.EXECUTION_FINGERPRINT,
                        "implementation_version": run_switch_auto.R3_IMPLEMENTATION_VERSION,
                        "implementation_fingerprint": run_switch_auto.R3_IMPLEMENTATION_FINGERPRINT,
                    },
                    "candidate_windows": [
                        {
                            "stage": "circuit_wiring",
                            "start_seconds": 2.0,
                            "end_seconds": 8.0,
                        }
                    ],
                    "rubric_3": {
                        "diagnostics": {
                            "decision_source": "opencv_same_frame_overlap",
                            **run_switch_auto.R3_AUTOMATIC_PARAMETERS,
                        "execution_fingerprint": self.EXECUTION_FINGERPRINT,
                        "implementation_version": run_switch_auto.R3_IMPLEMENTATION_VERSION,
                        "implementation_fingerprint": run_switch_auto.R3_IMPLEMENTATION_FINGERPRINT,
                            "same_frame_overlaps": [
                                {
                                    "window_id": "circuit_wiring_001",
                                    "stage": "circuit_wiring",
                                    "timestamp_seconds": 4.0,
                                    "switch_crop_path": str(root / "switch.jpg"),
                                    "plug_transitions": [{"color": "black"}],
                                },
                            ],
                            "switch_tracked_observation_count": 12,
                            "real_plug_transition_count": 1,
                            "same_frame_overlap_count": 1,
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        return report

    def test_one_command_result_is_binary_and_audited(self) -> None:
        with tempfile.TemporaryDirectory(dir=AGENT_ROOT) as directory:
            root = Path(directory)
            report = self._fixtures(root)
            plan = self._plan()
            with (
                mock.patch.dict(
                    run_switch_auto.os.environ,
                    {"QWEN_API_BASE_URL": "https://example.invalid/v1"},
                    clear=False,
                ),
                mock.patch.object(
                    run_switch_auto,
                    "inspect_video",
                    return_value={
                        "video_id": "sample",
                        "source_video_id": "sample.mp4",
                        "sha256": "a" * 64,
                    },
                ),
                mock.patch.object(
                    run_switch_auto,
                    "create_run",
                    return_value={"run_dir": str(root)},
                ),
                mock.patch.object(
                    run_switch_auto,
                    "run_full_pipeline",
                    return_value={"run_report": str(root / "pipeline.json")},
                ),
                mock.patch.object(
                    run_switch_auto,
                    "refine_rubric_boundaries",
                    return_value={"summary_path": str(root / "boundary.json")},
                ),
                mock.patch.object(run_switch_auto, "plan_live_skills", return_value=plan),
                mock.patch.object(
                    run_switch_auto,
                    "run_switch_rubric",
                    return_value={
                        "evidence_report": str(report),
                        "rubric": {
                            "decision": "pass",
                            "predicted_score": 1,
                            "confidence": 0.83,
                            "reason": "no_confirmed_overlap",
                            "result_path": str(root / "rubric_3.json"),
                        },
                    },
                ),
            ):
                result = run_switch_auto.execute_switch_auto(
                    "sample.mp4", "test_switch_auto", AGENT_ROOT / "config.json"
                )
            self.assertEqual("pass", result["decision"])
            self.assertEqual(1, result["predicted_score"])
            self.assertFalse(result["human_review_used"])
            self.assertFalse(result["video_id_used_for_routing"])
            self.assertFalse(result["qwen_used_for_decision"])
            self.assertEqual("opencv_same_frame_overlap", result["decision_source"])
            self.assertEqual(1, len(result["counterexample_intervals"]))
            self.assertEqual(4.0, result["counterexample_intervals"][0]["start_seconds"])
            self.assertEqual(4.0, result["counterexample_intervals"][0]["end_seconds"])
            self.assertTrue(Path(result["result_path"]).is_file())

    def test_qwen_decision_is_not_accepted_as_a_result(self) -> None:
        with tempfile.TemporaryDirectory(dir=AGENT_ROOT) as directory:
            root = Path(directory)
            report_path = self._fixtures(root, qwen_used=True)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            plan = self._plan()
            with self.assertRaisesRegex(run_switch_auto.ToolError, "Qwen participated"):
                run_switch_auto._verify_automatic_run(plan, report)

    def test_human_review_is_not_accepted_as_a_result(self) -> None:
        with tempfile.TemporaryDirectory(dir=AGENT_ROOT) as directory:
            root = Path(directory)
            report_path = self._fixtures(root)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["human_review_used"] = True
            plan = self._plan()
            with self.assertRaisesRegex(run_switch_auto.ToolError, "human review"):
                run_switch_auto._verify_automatic_run(plan, report)

    def test_plan_report_parameter_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir=AGENT_ROOT) as directory:
            root = Path(directory)
            report_path = self._fixtures(root)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["skill_execution"]["effective_parameters"]["sampling_fps"] = 2.0
            with self.assertRaisesRegex(run_switch_auto.ToolError, "effective report sampling_fps"):
                run_switch_auto._verify_automatic_run(self._plan(), report)

    def test_plan_report_window_mode_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir=AGENT_ROOT) as directory:
            root = Path(directory)
            report_path = self._fixtures(root)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["skill_execution"]["parameters"]["window_mode"] = "broad_search"
            with self.assertRaisesRegex(run_switch_auto.ToolError, "complete parameters differ"):
                run_switch_auto._verify_automatic_run(self._plan(), report)

    def test_broad_search_covers_the_full_video(self) -> None:
        windows = switch_rubric.candidate_windows({}, 120.0, "broad_search")
        self.assertEqual(0.0, windows[0]["start_seconds"])
        self.assertEqual(120.0, windows[0]["end_seconds"])

    def test_decoded_records_keep_each_requested_frame_number(self) -> None:
        with tempfile.TemporaryDirectory(dir=AGENT_ROOT) as directory:
            root = Path(directory)
            video = root / "sample.avi"
            writer = switch_rubric.cv2.VideoWriter(
                str(video),
                switch_rubric.cv2.VideoWriter_fourcc(*"MJPG"),
                5.0,
                (64, 48),
            )
            self.assertTrue(writer.isOpened())
            for value in range(10):
                frame = switch_rubric.np.full((48, 64, 3), value * 10, dtype=switch_rubric.np.uint8)
                writer.write(frame)
            writer.release()
            records = switch_rubric._decode_and_export(
                video,
                [
                    {
                        "window_id": "broad",
                        "stage": "circuit_wiring",
                        "timestamp_seconds": 0.0,
                    },
                    {
                        "window_id": "broad",
                        "stage": "circuit_wiring",
                        "timestamp_seconds": 1.0,
                    },
                ],
                root / "evidence",
            )
        self.assertEqual([0, 5], [item["frame_number"] for item in records])


if __name__ == "__main__":
    unittest.main()
