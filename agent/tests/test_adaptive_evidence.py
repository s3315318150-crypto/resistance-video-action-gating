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

import adaptive_evidence  # noqa: E402
import toolkit  # noqa: E402


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class AdaptiveEvidenceTests(unittest.TestCase):
    def _video(self, root: Path) -> Path:
        path = root / "current.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 2.0, (96, 64))
        self.assertTrue(writer.isOpened())
        try:
            for index in range(12):
                frame = np.full((64, 96, 3), 30 + index * 5, dtype=np.uint8)
                writer.write(frame)
        finally:
            writer.release()
        return path

    def test_request_extracts_current_stage_frames_and_dynamic_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "run"
            run_dir.mkdir()
            video = self._video(root)
            stages = run_dir / "stages.json"
            stages.write_text(
                json.dumps(
                    {
                        "observed_stage_runs": [
                            {"stage": "measurement_1", "start_seconds": 1.0, "end_seconds": 4.0}
                        ]
                    }
                ),
                encoding="utf-8",
            )
            state = {
                "run_id": "adaptive_fixture",
                "mode": "execute",
                "video": {"path": str(video), "sha256": file_sha256(video)},
            }

            def fake_export(item: dict, _evidence_dir: Path) -> dict:
                return {**item, "detection": {"valid": False}, "candidates": []}

            with mock.patch(
                "meter_rubrics._export_candidates",
                side_effect=fake_export,
            ), mock.patch(
                "skills.dynamic_meter_reading.prepare_frames",
                return_value={"skill_version": "dynamic_meter_reading.v3", "tracks": []},
            ):
                result = adaptive_evidence.request_additional_evidence(
                    run_dir=run_dir,
                    state=state,
                    rubric_ids=[5, 6],
                    reason="low_confidence",
                    time_ranges=[{"start_seconds": 1.0, "end_seconds": 2.0}],
                    interval_seconds=0.2,
                    max_frames=8,
                )

            self.assertEqual("additional_evidence_ready", result["status"])
            self.assertEqual(3, result["frame_count"])
            self.assertEqual(3, result["selected_frame_count"])
            self.assertFalse(result["historical_artifacts_used"])
            self.assertFalse(result["video_id_used_for_routing"])
            self.assertFalse(result["fixed_video_roi_used"])
            self.assertTrue(Path(result["result_path"]).is_file())
            self.assertTrue(Path(result["request_path"]).is_file())

    def test_request_rejects_history_outside_current_stage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "run"
            run_dir.mkdir()
            video = self._video(root)
            (run_dir / "stages.json").write_text(
                json.dumps(
                    {
                        "observed_stage_runs": [
                            {"stage": "measurement_1", "start_seconds": 1.0, "end_seconds": 2.0}
                        ]
                    }
                ),
                encoding="utf-8",
            )
            state = {
                "run_id": "adaptive_guard_fixture",
                "mode": "execute",
                "video": {"path": str(video), "sha256": file_sha256(video)},
            }
            with self.assertRaises(adaptive_evidence.AdaptiveEvidenceError):
                adaptive_evidence.request_additional_evidence(
                    run_dir=run_dir,
                    state=state,
                    rubric_ids=[5],
                    reason="low_confidence",
                    time_ranges=[{"start_seconds": 5.0, "end_seconds": 5.5}],
                )

    def test_toolkit_request_archives_and_invalidates_prior_meter_results(self) -> None:
        (toolkit.AGENT_ROOT / "runs").mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=toolkit.AGENT_ROOT / "runs") as directory:
            run_dir = Path(directory)
            rubric_dir = run_dir / "rubrics"
            evidence_dir = run_dir / "meter_rubrics"
            request_dir = run_dir / "adaptive_evidence" / "request_01"
            result_path = request_dir / "result.json"
            rubric_5 = rubric_dir / "rubric_5.json"
            rubric_6 = rubric_dir / "rubric_6.json"
            report = evidence_dir / "meter_evidence_report.json"
            for path, value in (
                (rubric_5, {"decision": "fail"}),
                (rubric_6, {"decision": "fail"}),
                (report, {"status": "old"}),
            ):
                toolkit.write_json(path, value)
            toolkit.write_json(
                run_dir / "state.json",
                {
                    "run_id": "fixture",
                    "mode": "execute",
                    "status": "meter_rubrics_completed",
                    "video": {"path": "fixture.mp4"},
                    "rubric_results": {"5": str(rubric_5), "6": str(rubric_6)},
                    "rubric_evidence_reports": {"5_6": str(report)},
                    "tool_calls": [],
                    "final_result": None,
                },
            )

            def fake_acquire(**_kwargs: object) -> dict:
                result = {
                    "status": "additional_evidence_ready",
                    "request_number": 1,
                    "rubric_ids": [5, 6],
                    "reason": "low_confidence",
                    "frame_count": 4,
                    "selected_frame_count": 2,
                    "result_path": str(result_path),
                    "request_path": str(request_dir / "request.json"),
                    "next_tool": "run_rubric_bundle",
                    "next_arguments": {"rubric_ids": [5, 6]},
                }
                toolkit.write_json(result_path, result)
                return result

            with mock.patch.object(toolkit, "_existing_run_dir", return_value=run_dir), mock.patch.object(
                toolkit, "_verify_source_video", return_value=Path("fixture.mp4")
            ), mock.patch.object(adaptive_evidence, "request_additional_evidence", side_effect=fake_acquire):
                result = toolkit.request_additional_evidence(
                    "fixture",
                    [5, 6],
                    "low_confidence",
                    [{"start_seconds": 1.0, "end_seconds": 2.0}],
                )

            state = toolkit.read_json(run_dir / "state.json")
            self.assertEqual([5, 6], result["invalidated_rubric_ids"])
            self.assertEqual({}, state["rubric_results"])
            self.assertEqual({}, state["rubric_evidence_reports"])
            self.assertEqual("boundaries_completed", state["status"])
            self.assertEqual(3, len(result["archived_prior_artifacts"]))
            self.assertTrue(all(Path(item["path"]).is_file() for item in result["archived_prior_artifacts"]))

    def test_meter_recommendation_builds_bounded_current_frame_request(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "meter_evidence_report.json"
            toolkit.write_json(
                report,
                {
                    "qwen_observation": {"overall_confidence": 0.55},
                    "selected_frames": [
                        {
                            "timestamp_seconds": 12.0,
                            "sharpness": 18.0,
                            "model_candidates": [],
                        }
                    ],
                },
            )
            recommendation = toolkit._meter_adaptive_recommendation(
                {
                    "5": {"confidence": 0.4, "reason": "meter_low_visibility"},
                    "6": {"confidence": 0.65, "reason": "current_qwen_binary"},
                },
                str(report),
                20.0,
            )
            self.assertTrue(recommendation["adaptive_evidence_recommended"])
            request = recommendation["adaptive_request_template"]
            self.assertEqual("meter_pointer_occluded", request["reason"])
            self.assertEqual(
                [{"start_seconds": 11.0, "end_seconds": 13.0}],
                request["time_ranges"],
            )


if __name__ == "__main__":
    unittest.main()
