from __future__ import annotations

import hashlib
import inspect
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

import adaptive_evidence  # noqa: E402
import adaptive_record_evidence  # noqa: E402
import record_rubrics  # noqa: E402
import toolkit  # noqa: E402
import orchestrator  # noqa: E402
from skills import EXECUTOR_REGISTRY  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class AdaptiveRecordEvidenceTests(unittest.TestCase):
    def _video(self, root: Path, name: str = "current.mp4") -> Path:
        path = root / name
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (160, 120)
        )
        self.assertTrue(writer.isOpened())
        try:
            for index in range(30):
                frame = np.full((120, 160, 3), 35, dtype=np.uint8)
                cv2.rectangle(frame, (22, 22), (138, 98), (235, 235, 235), -1)
                cv2.putText(
                    frame,
                    f"U {index % 10}",
                    (38, 66),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (15, 15, 15),
                    2,
                )
                writer.write(frame)
        finally:
            writer.release()
        return path

    def _fixture(self, root: Path, video_name: str = "current.mp4") -> tuple[Path, dict]:
        run_dir = root / "run"
        run_dir.mkdir()
        video = self._video(root, video_name)
        (run_dir / "stages.json").write_text(
            json.dumps(
                {
                    "observed_stage_runs": [
                        {
                            "stage": "recording_1",
                            "start_seconds": 1.0,
                            "end_seconds": 4.0,
                        },
                        {
                            "stage": "measurement_2",
                            "start_seconds": 4.0,
                            "end_seconds": 5.0,
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )
        state = {
            "run_id": "record_fixture",
            "mode": "execute",
            "video_id": "identity_must_not_route",
            "video": {"path": str(video), "sha256": sha256(video)},
        }
        return run_dir, state

    def test_record_profile_extracts_dynamic_paper_views(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir, state = self._fixture(Path(directory))
            candidate = {
                "bbox_xyxy": [20, 20, 140, 100],
                "score": 0.9,
                "edge_ratio": 0.1,
                "bright_ratio": 0.8,
            }
            with mock.patch.object(record_rubrics, "_paper_candidates", return_value=[candidate]):
                result = adaptive_evidence.request_additional_evidence(
                    run_dir=run_dir,
                    state=state,
                    rubric_ids=[7],
                    evidence_profile="record_paper",
                    cycle=1,
                    reason="single_frame_support",
                    target_fields=["u1", "i1"],
                    anchor_frame_ids=["frame_00000008"],
                    search_mode="adjacent_dense",
                    time_ranges=[{"start_seconds": 1.4, "end_seconds": 2.0}],
                    interval_seconds=0.2,
                    max_frames=8,
                    roi_mode="dynamic_paper_tracking",
                    view="paper_fields",
                )

            self.assertEqual("record_paper", result["evidence_profile"])
            self.assertGreater(result["selected_frame_count"], 0)
            first = result["paper_rows"][0]
            self.assertTrue(first["paper_field_view"]["dynamic"])
            self.assertTrue(Path(first["paper_field_view"]["roi_path"]).is_file())
            self.assertTrue(Path(first["paper_field_view"]["ink_roi_path"]).is_file())
            self.assertFalse(result["video_id_used_for_routing"])
            self.assertFalse(result["historical_artifacts_used"])
            self.assertFalse(result["fixed_video_roi_used"])
            self.assertFalse(result["excel_accessed"])

    def test_dynamic_views_keep_small_edge_writing_hand_context(self) -> None:
        frame = np.full((120, 160, 3), 210, dtype=np.uint8)
        candidates = [
            {
                "bbox_xyxy": [0, 0, 100, 90],
                "score": 0.9,
                "detector": "writing_hand_context",
                "skin_component_area_ratio": 0.08,
            },
            {
                "bbox_xyxy": [10, 10, 130, 110],
                "score": 0.8,
                "detector": "writing_hand_context",
                "skin_component_area_ratio": 0.05,
            },
            {
                "bbox_xyxy": [110, 0, 160, 55],
                "score": 0.5,
                "detector": "writing_hand_context",
                "skin_component_area_ratio": 0.01,
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            views = record_rubrics._dynamic_paper_field_views(
                frame, candidates, Path(directory), "edge_hand"
            )
        self.assertEqual([0, 0, 100, 90], views[0]["bbox_xyxy"])
        self.assertEqual([110, 0, 160, 55], views[1]["bbox_xyxy"])

    def test_quality_uses_paper_only_and_clear_two_frame_support_stops(self) -> None:
        paper = {
            "fields": {
                "u1": {
                    "status": "read",
                    "value": 2.4,
                    "confidence": 0.88,
                    "support_frame_count": 2,
                    "support": [{"frame_id": "frame_1"}, {"frame_id": "frame_2"}],
                },
                "i1": {
                    "status": "read",
                    "value": 0.3,
                    "confidence": 0.84,
                    "support_frame_count": 2,
                    "support": [{"frame_id": "frame_1"}, {"frame_id": "frame_2"}],
                },
            }
        }
        observation = {
            "observations": [
                {"frame_id": "frame_1", "paper_visible": True},
                {"frame_id": "frame_2", "paper_visible": True},
            ]
        }
        quality = record_rubrics.assess_record_evidence(
            1,
            paper,
            observation,
            [{"paper_field_view": {}}, {"paper_field_view": {}}],
        )
        self.assertFalse(quality["request_more_frames"])
        self.assertFalse(quality["meter_values_used"])
        self.assertFalse(quality["excel_accessed"])

    def test_single_frame_low_confidence_and_conflict_trigger(self) -> None:
        paper = {
            "fields": {
                "u2": {
                    "status": "read",
                    "value": 2.4,
                    "confidence": 0.62,
                    "support_frame_count": 1,
                    "support": [{"frame_id": "frame_4"}],
                },
                "i2": {
                    "status": "conflict",
                    "value": 0.3,
                    "confidence": 0.8,
                    "support_frame_count": 2,
                    "support": [{"frame_id": "frame_4"}, {"frame_id": "frame_5"}],
                },
            }
        }
        quality = record_rubrics.assess_record_evidence(
            2,
            paper,
            {"observations": [{"frame_id": "frame_4", "paper_visible": True}]},
            [],
        )
        self.assertTrue(quality["request_more_frames"])
        self.assertEqual(
            {"single_frame_support", "low_confidence", "digit_conflict"},
            set(quality["request_reasons"]),
        )

    @unittest.skipUnless(
        (AGENT_ROOT / "evaluations" / "r79_adaptive_frame_agent_qwen_smoke_20260818" / "record_rubrics" / "record_evidence_report.json").is_file(),
        "local regression fixture is not published",
    )
    def test_real_cycle_one_meter_quality_requests_missing_ammeter_and_deflection(self) -> None:
        evaluation = (
            AGENT_ROOT
            / "evaluations"
            / "r79_adaptive_frame_agent_qwen_smoke_20260818"
            / "record_rubrics"
        )
        raw = json.loads(
            (evaluation / "qwen" / "cycle_1_meters.json").read_text(encoding="utf-8")
        )
        report = json.loads(
            (evaluation / "record_evidence_report.json").read_text(encoding="utf-8")
        )
        quality = record_rubrics.assess_cycle_meter_evidence(
            1, raw["observation"], report["cycles"]["1"]["meter_frames"]
        )
        self.assertTrue(quality["request_more_frames"])
        self.assertIn("ammeter_missing", quality["request_reasons"])
        self.assertIn("voltmeter_no_stable_deflection", quality["request_reasons"])
        self.assertIn("no_stable_dual_meter_frames", quality["request_reasons"])
        self.assertEqual([], quality["stable_dual_meter_frame_ids"])

    @unittest.skipUnless(
        (AGENT_ROOT / "evaluations" / "r79_adaptive_frame_agent_qwen_smoke_20260818" / "record_rubrics" / "record_evidence_report.json").is_file(),
        "local regression fixture is not published",
    )
    def test_real_cycle_two_meter_quality_keeps_good_ammeter(self) -> None:
        evaluation = (
            AGENT_ROOT
            / "evaluations"
            / "r79_adaptive_frame_agent_qwen_smoke_20260818"
            / "record_rubrics"
        )
        raw = json.loads(
            (evaluation / "qwen" / "cycle_2_meters.json").read_text(encoding="utf-8")
        )
        report = json.loads(
            (evaluation / "record_evidence_report.json").read_text(encoding="utf-8")
        )
        quality = record_rubrics.assess_cycle_meter_evidence(
            2, raw["observation"], report["cycles"]["2"]["meter_frames"]
        )
        self.assertTrue(quality["request_more_frames"])
        self.assertIn("voltmeter_missing", quality["request_reasons"])
        self.assertEqual([], quality["roles"]["ammeter"]["request_reasons"])
        self.assertEqual(2, quality["roles"]["ammeter"]["distinct_frame_support"])
        self.assertEqual(
            ["frame_00004045", "frame_00004095"],
            quality["roles"]["ammeter"]["stable_deflection_frame_ids"],
        )

    def test_clear_two_meter_two_frame_evidence_stops_requesting(self) -> None:
        rows = [
            {"frame_id": "frame_1", "dynamic_meter_candidates": [{}]},
            {"frame_id": "frame_2", "dynamic_meter_candidates": [{}]},
        ]
        per_frame = []
        for group, frame_id in enumerate(("frame_1", "frame_2"), start=1):
            per_frame.append(
                {
                    "image_group": group,
                    "frame_id": frame_id,
                    "ammeter": {
                        "visible": True,
                        "selected_range": 0.6,
                        "value": 0.2,
                    },
                    "voltmeter": {
                        "visible": True,
                        "selected_range": 3.0,
                        "value": 1.2,
                    },
                }
            )
        observation = {
            "per_frame": per_frame,
            "consensus": {
                "ammeter": {
                    "selected_range": 0.6,
                    "value": 0.2,
                    "confidence": 0.82,
                    "supporting_frame_ids": ["frame_1", "frame_2"],
                },
                "voltmeter": {
                    "selected_range": 3.0,
                    "value": 1.2,
                    "confidence": 0.86,
                    "supporting_frame_ids": ["frame_1", "frame_2"],
                },
            },
        }
        quality = record_rubrics.assess_cycle_meter_evidence(1, observation, rows)
        self.assertFalse(quality["request_more_frames"])
        self.assertEqual(["frame_1", "frame_2"], quality["stable_dual_meter_frame_ids"])

    def test_meter_range_and_reading_conflicts_trigger(self) -> None:
        rows = [{"frame_id": "frame_1"}, {"frame_id": "frame_2"}]
        observation = {
            "per_frame": [
                {
                    "image_group": 1,
                    "ammeter": {"visible": True, "selected_range": 0.6, "value": 0.2},
                    "voltmeter": {"visible": True, "selected_range": 3.0, "value": 1.0},
                },
                {
                    "image_group": 2,
                    "ammeter": {"visible": True, "selected_range": 3.0, "value": 0.2},
                    "voltmeter": {"visible": True, "selected_range": 3.0, "value": 1.5},
                },
            ],
            "consensus": {
                "ammeter": {
                    "selected_range": 0.6,
                    "value": 0.2,
                    "confidence": 0.9,
                    "supporting_frame_ids": ["frame_1", "frame_2"],
                },
                "voltmeter": {
                    "selected_range": 3.0,
                    "value": 1.0,
                    "confidence": 0.9,
                    "supporting_frame_ids": ["frame_1", "frame_2"],
                },
            },
        }
        quality = record_rubrics.assess_cycle_meter_evidence(1, observation, rows)
        self.assertIn("ammeter_range_conflict", quality["request_reasons"])
        self.assertNotIn("ammeter_reading_conflict", quality["request_reasons"])
        self.assertIn("voltmeter_reading_conflict", quality["request_reasons"])
        self.assertNotIn("voltmeter_range_conflict", quality["request_reasons"])

    def test_meter_low_confidence_triggers_below_point_seven(self) -> None:
        rows = [{"frame_id": "frame_1"}, {"frame_id": "frame_2"}]
        observation = {
            "per_frame": [
                {
                    "image_group": group,
                    "ammeter": {"visible": True, "selected_range": 0.6, "value": 0.2},
                    "voltmeter": {"visible": True, "selected_range": 3.0, "value": 1.2},
                }
                for group in (1, 2)
            ],
            "consensus": {
                "ammeter": {
                    "selected_range": 0.6,
                    "value": 0.2,
                    "confidence": 0.69,
                    "supporting_frame_ids": ["frame_1", "frame_2"],
                },
                "voltmeter": {
                    "selected_range": 3.0,
                    "value": 1.2,
                    "confidence": 0.70,
                    "supporting_frame_ids": ["frame_1", "frame_2"],
                },
            },
        }
        quality = record_rubrics.assess_cycle_meter_evidence(1, observation, rows)
        self.assertIn("ammeter_low_confidence", quality["request_reasons"])
        self.assertNotIn("voltmeter_low_confidence", quality["request_reasons"])

    def test_meter_quality_counts_image_group_once_and_has_no_identity_inputs(self) -> None:
        rows = [{"frame_id": "frame_1"}, {"frame_id": "frame_2"}]
        duplicate = {
            "image_group": 1,
            "frame_id": "model_roi_variant",
            "ammeter": {"visible": True, "selected_range": 0.6, "value": 0.2},
            "voltmeter": {"visible": True, "selected_range": 3.0, "value": 1.2},
        }
        observation = {
            "per_frame": [duplicate, dict(duplicate)],
            "consensus": {
                "ammeter": {
                    "selected_range": 0.6,
                    "value": 0.2,
                    "confidence": 0.9,
                    "supporting_frame_ids": ["frame_1", "frame_1"],
                },
                "voltmeter": {
                    "selected_range": 3.0,
                    "value": 1.2,
                    "confidence": 0.9,
                    "supporting_frame_ids": ["frame_1", "frame_1"],
                },
            },
        }
        quality = record_rubrics.assess_cycle_meter_evidence(1, observation, rows)
        self.assertEqual(1, quality["roles"]["ammeter"]["distinct_frame_support"])
        self.assertIn("ammeter_single_frame_support", quality["request_reasons"])
        self.assertFalse(quality["paper_values_used"])
        self.assertFalse(quality["excel_accessed"])
        self.assertEqual(
            {"cycle", "meter_observation", "meter_rows", "min_confidence", "min_distinct_frames"},
            set(inspect.signature(record_rubrics.assess_cycle_meter_evidence).parameters),
        )

    def test_stage_containment_and_two_round_limit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir, state = self._fixture(Path(directory))
            common = {
                "run_dir": run_dir,
                "state": state,
                "rubric_ids": [7],
                "reason": "paper_not_found",
                "cycle": 1,
                "target_fields": ["u1"],
                "anchor_frame_ids": [],
                "search_mode": "adjacent_dense",
                "interval_seconds": 0.5,
                "max_frames": 8,
                "roi_mode": "dynamic_paper_tracking",
                "view": "paper_full",
            }
            missing_cycle = dict(common)
            missing_cycle.update(
                rubric_ids=[9], cycle=2, target_fields=["u2"]
            )
            with self.assertRaises(adaptive_record_evidence.AdaptiveRecordEvidenceError):
                adaptive_record_evidence.request_additional_record_evidence(
                    **missing_cycle,
                    time_ranges=[{"start_seconds": 4.2, "end_seconds": 4.8}],
                )
            adaptive_record_evidence.request_additional_record_evidence(
                **common,
                time_ranges=[{"start_seconds": 1.0, "end_seconds": 1.5}],
            )
            adaptive_record_evidence.request_additional_record_evidence(
                **common,
                time_ranges=[{"start_seconds": 2.0, "end_seconds": 2.5}],
            )
            with self.assertRaises(adaptive_record_evidence.AdaptiveRecordEvidenceError):
                adaptive_record_evidence.request_additional_record_evidence(
                    **common,
                    time_ranges=[{"start_seconds": 3.0, "end_seconds": 3.5}],
                )

    def test_toolkit_record_request_invalidates_grouped_r7_r9(self) -> None:
        (toolkit.AGENT_ROOT / "runs").mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=toolkit.AGENT_ROOT / "runs") as directory:
            run_dir = Path(directory)
            result_path = (
                run_dir
                / "adaptive_evidence"
                / "record_paper"
                / "cycle_1"
                / "request_01"
                / "result.json"
            )
            rubric_7 = run_dir / "rubrics" / "rubric_7.json"
            rubric_9 = run_dir / "rubrics" / "rubric_9.json"
            report = run_dir / "record_rubrics" / "record_evidence_report.json"
            for path in (rubric_7, rubric_9, report):
                toolkit.write_json(path, {"decision": "fail"})
            toolkit.write_json(
                run_dir / "state.json",
                {
                    "run_id": "fixture",
                    "mode": "execute",
                    "status": "record_rubrics_completed",
                    "video": {"path": "fixture.mp4"},
                    "rubric_results": {"7": str(rubric_7), "9": str(rubric_9)},
                    "rubric_evidence_reports": {"7_9": str(report)},
                    "tool_calls": [],
                    "final_result": None,
                },
            )

            def fake_acquire(**_kwargs: object) -> dict:
                value = {
                    "status": "additional_evidence_ready",
                    "evidence_profile": "record_paper",
                    "request_number": 1,
                    "cycle": 1,
                    "rubric_ids": [7],
                    "reason": "single_frame_support",
                    "frame_count": 3,
                    "selected_frame_count": 2,
                    "result_path": str(result_path),
                    "request_path": str(result_path.with_name("request.json")),
                }
                toolkit.write_json(result_path, value)
                return value

            with mock.patch.object(toolkit, "_existing_run_dir", return_value=run_dir), mock.patch.object(
                toolkit, "_verify_source_video", return_value=Path("fixture.mp4")
            ), mock.patch.object(
                adaptive_evidence, "request_additional_evidence", side_effect=fake_acquire
            ):
                result = toolkit.request_additional_evidence(
                    "fixture",
                    [7],
                    "single_frame_support",
                    [{"start_seconds": 1.0, "end_seconds": 2.0}],
                    evidence_profile="record_paper",
                    cycle=1,
                    target_fields=["u1"],
                    search_mode="adjacent_dense",
                    roi_mode="dynamic_paper_tracking",
                    view="paper_fields",
                )

            state = toolkit.read_json(run_dir / "state.json")
            self.assertEqual([7, 9], result["invalidated_rubric_ids"])
            self.assertEqual({}, state["rubric_results"])
            self.assertEqual({}, state["rubric_evidence_reports"])
            self.assertEqual("boundaries_completed", state["status"])
            self.assertEqual(3, len(result["archived_prior_artifacts"]))

    def test_schema_and_record_skill_publish_bounded_adaptive_parameters(self) -> None:
        schema = next(
            item for item in toolkit.TOOL_SCHEMAS
            if item["name"] == "request_additional_evidence"
        )["inputSchema"]["properties"]
        self.assertEqual([5, 6, 7, 9], schema["rubric_ids"]["items"]["enum"])
        self.assertIn("dynamic_paper_tracking", schema["roi_mode"]["enum"])
        self.assertIn("paper_fields", schema["view"]["enum"])
        defaults = EXECUTOR_REGISTRY["record.two_cycle_consistency"].defaults
        self.assertTrue(defaults["adaptive_enabled"])
        self.assertEqual(2, defaults["adaptive_max_rounds"])
        self.assertEqual(0.2, defaults["adaptive_interval_seconds"])
        self.assertEqual(20, defaults["adaptive_max_frames"])

    def test_deterministic_scheduler_consumes_record_request_and_reruns_group(self) -> None:
        bundle_calls = 0

        def fake_invoke(name: str, arguments: dict) -> dict:
            nonlocal bundle_calls
            if name == "inspect_video":
                return {"video_id": "sample"}
            if name == "create_run":
                return {"status": "created"}
            if name == "run_full_pipeline":
                return {
                    "status": "pipeline_completed",
                    "run_report": "run_report.json",
                    "rubric_specific_artifacts_required": [],
                }
            if name == "refine_rubric_boundaries":
                return {"status": "boundaries_completed", "summary_path": "boundaries.json"}
            if name == "plan_live_skills":
                return {"status": "live_skills_planned", "skills": []}
            if name == "run_adaptive_frame_agent":
                return {"status": "frame_evidence_ready"}
            if name == "run_rubric_bundle":
                bundle_calls += 1
                if bundle_calls == 1:
                    return {
                        "status": "rubric_bundle_completed",
                        "producer_calls": [
                            {
                                "adaptive_evidence_recommended": True,
                                "adaptive_request_template": {
                                    "rubric_ids": [7],
                                    "evidence_profile": "record_paper",
                                    "cycle": 1,
                                    "reason": "single_frame_support",
                                    "target_fields": ["u1"],
                                    "anchor_frame_ids": ["frame_10"],
                                    "search_mode": "adjacent_dense",
                                    "time_ranges": [
                                        {"start_seconds": 1.0, "end_seconds": 2.0}
                                    ],
                                    "interval_seconds": 0.2,
                                    "max_frames": 20,
                                    "roi_mode": "dynamic_paper_tracking",
                                    "view": "paper_fields",
                                },
                            }
                        ],
                    }
                return {
                    "status": "rubric_bundle_completed",
                    "producer_calls": [],
                }
            if name == "request_additional_evidence":
                self.assertEqual("record_paper", arguments["evidence_profile"])
                self.assertEqual([7], arguments["rubric_ids"])
                return {"status": "additional_evidence_ready", "frame_count": 3}
            if name == "inspect_run_status":
                return {
                    "status": "rubrics_completed",
                    "video_id": "sample",
                    "run_dir": "fixture",
                    "completed_rubrics": list(range(10)),
                    "missing_rubrics": [],
                }
            if name == "validate_run":
                return {"status": "valid"}
            if name == "finalize_run":
                return {"status": "completed", "result_count": 10}
            raise AssertionError(name)

        with mock.patch.object(orchestrator, "_invoke", side_effect=fake_invoke):
            result = orchestrator.run_deterministic(
                "fixture", "sample.mp4", "execute", toolkit.DEFAULT_CONFIG
            )
        calls = [item for item in result["transcript"] if item["tool"] == "run_rubric_bundle"]
        self.assertEqual(2, len(calls))
        self.assertEqual([7, 9], calls[1]["arguments"]["rubric_ids"])
        self.assertEqual(
            1,
            sum(
                item["tool"] == "request_additional_evidence"
                for item in result["transcript"]
            ),
        )


if __name__ == "__main__":
    unittest.main()
