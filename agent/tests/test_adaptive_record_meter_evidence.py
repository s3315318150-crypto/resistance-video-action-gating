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
import adaptive_record_meter_evidence  # noqa: E402
import orchestrator  # noqa: E402
import record_rubrics  # noqa: E402
import toolkit  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class AdaptiveRecordMeterEvidenceTests(unittest.TestCase):
    def test_adaptive_template_range_is_clipped_to_current_meter_stage(self) -> None:
        clipped = record_rubrics._clip_range_to_current_intervals(
            104.833,
            106.833,
            [(87.5, 106.5), (107.5, 112.5)],
        )
        self.assertEqual((104.833, 106.5), clipped)
        stages = [
            {"stage": "measurement_1", "start_seconds": 88.0, "end_seconds": 106.0},
            {"stage": "recording_1", "start_seconds": 108.0, "end_seconds": 112.0},
        ]
        validated = adaptive_record_meter_evidence._validate_ranges(
            [{"start_seconds": clipped[0], "end_seconds": clipped[1]}],
            stages,
            1,
            184.0,
        )
        self.assertEqual(
            [{"start_seconds": 104.833, "end_seconds": 106.5}], validated
        )

    def _video(self, root: Path) -> Path:
        path = root / "current.mp4"
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (160, 120)
        )
        self.assertTrue(writer.isOpened())
        try:
            for index in range(40):
                frame = np.full((120, 160, 3), 40 + index, dtype=np.uint8)
                cv2.rectangle(frame, (12, 12), (72, 82), (220, 220, 220), 2)
                cv2.rectangle(frame, (88, 12), (148, 82), (220, 220, 220), 2)
                writer.write(frame)
        finally:
            writer.release()
        return path

    def _fixture(self, root: Path) -> tuple[Path, dict]:
        run_dir = root / "run"
        run_dir.mkdir()
        video = self._video(root)
        (run_dir / "stages.json").write_text(
            json.dumps(
                {
                    "observed_stage_runs": [
                        {
                            "stage": "measurement_1",
                            "start_seconds": 1.0,
                            "end_seconds": 3.0,
                        },
                        {
                            "stage": "recording_1",
                            "start_seconds": 3.0,
                            "end_seconds": 4.0,
                        },
                        {
                            "stage": "measurement_2",
                            "start_seconds": 4.0,
                            "end_seconds": 6.0,
                        },
                        {
                            "stage": "recording_2",
                            "start_seconds": 6.0,
                            "end_seconds": 7.0,
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )
        state = {
            "run_id": "record_meter_fixture",
            "mode": "execute",
            "video_id": "must_not_route",
            "video": {"path": str(video), "sha256": sha256(video)},
        }
        return run_dir, state

    @staticmethod
    def _fake_export(item: dict, _evidence_dir: Path) -> dict:
        frame_path = str(Path(item["frame_path"]).resolve())
        return {
            **item,
            "detection": {"valid": True},
            "candidates": [
                {
                    "candidate_id": "candidate_01",
                    "role_hint": "ammeter",
                    "enhanced_path": frame_path,
                    "wide_path": frame_path,
                    "face_path": frame_path,
                    "quality": 0.9,
                }
            ],
        }

    def test_record_meter_extracts_current_cycle_dynamic_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir, state = self._fixture(Path(directory))
            with mock.patch(
                "meter_rubrics._export_candidates", side_effect=self._fake_export
            ):
                result = adaptive_evidence.request_additional_evidence(
                    run_dir=run_dir,
                    state=state,
                    rubric_ids=[7],
                    evidence_profile="record_meter",
                    cycle=1,
                    reason="ammeter_missing",
                    target_roles=["ammeter"],
                    anchor_frame_ids=["frame_00000010"],
                    search_mode="adjacent_meter_dense",
                    time_ranges=[{"start_seconds": 1.2, "end_seconds": 2.0}],
                    interval_seconds=0.2,
                    max_frames=8,
                    roi_mode="dynamic_meter_candidates",
                    view="meter_pair",
                )

            self.assertEqual("record_meter", result["evidence_profile"])
            self.assertEqual(["ammeter"], result["target_roles"])
            self.assertGreater(result["frame_count"], 0)
            self.assertEqual(result["frame_count"], result["selected_frame_count"])
            first = result["meter_rows"][0]
            self.assertEqual("record_meter", first["window_source"])
            self.assertTrue(Path(first["image_path"]).is_file())
            self.assertTrue(first["role_views"]["ammeter"]["dynamic"])
            self.assertFalse(result["video_id_used_for_routing"])
            self.assertFalse(result["historical_artifacts_used"])
            self.assertFalse(result["fixed_video_roi_used"])
            self.assertFalse(result["paper_values_used"])
            self.assertFalse(result["excel_accessed"])
            self.assertNotIn(
                "video_id",
                inspect.signature(
                    adaptive_record_meter_evidence.request_additional_record_meter_evidence
                ).parameters,
            )

    def test_record_meter_rejects_other_cycle_range(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir, state = self._fixture(Path(directory))
            with self.assertRaises(adaptive_evidence.AdaptiveEvidenceError):
                adaptive_evidence.request_additional_evidence(
                    run_dir=run_dir,
                    state=state,
                    rubric_ids=[7],
                    evidence_profile="record_meter",
                    cycle=1,
                    reason="ammeter_missing",
                    target_roles=["ammeter"],
                    search_mode="adjacent_meter_dense",
                    time_ranges=[{"start_seconds": 5.0, "end_seconds": 5.5}],
                )

    def test_record_meter_enforces_two_rounds_per_cycle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir, state = self._fixture(Path(directory))
            common = {
                "run_dir": run_dir,
                "state": state,
                "rubric_ids": [7],
                "evidence_profile": "record_meter",
                "cycle": 1,
                "reason": "ammeter_single_frame_support",
                "target_roles": ["ammeter"],
                "search_mode": "adjacent_meter_dense",
                "interval_seconds": 0.2,
                "max_frames": 4,
                "roi_mode": "dynamic_meter_candidates",
                "view": "meter_pair",
            }
            with mock.patch(
                "meter_rubrics._export_candidates", side_effect=self._fake_export
            ):
                adaptive_evidence.request_additional_evidence(
                    **common,
                    time_ranges=[{"start_seconds": 1.0, "end_seconds": 1.2}],
                )
                adaptive_evidence.request_additional_evidence(
                    **common,
                    time_ranges=[{"start_seconds": 1.4, "end_seconds": 1.6}],
                )
                with self.assertRaises(adaptive_evidence.AdaptiveEvidenceError):
                    adaptive_evidence.request_additional_evidence(
                        **common,
                        time_ranges=[{"start_seconds": 1.8, "end_seconds": 2.0}],
                    )

    def test_zero_new_frames_does_not_consume_round(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir, state = self._fixture(Path(directory))
            toolkit.write_json(
                run_dir / "known.json",
                {"rows": [{"frame_number": 5}, {"frame_number": 6}]},
            )
            common = {
                "run_dir": run_dir,
                "state": state,
                "rubric_ids": [7],
                "evidence_profile": "record_meter",
                "cycle": 1,
                "reason": "ammeter_missing",
                "target_roles": ["ammeter"],
                "search_mode": "adjacent_meter_dense",
                "interval_seconds": 0.2,
                "max_frames": 4,
                "roi_mode": "dynamic_meter_candidates",
                "view": "meter_pair",
            }
            skipped = adaptive_evidence.request_additional_evidence(
                **common,
                time_ranges=[{"start_seconds": 1.0, "end_seconds": 1.2}],
            )
            self.assertEqual("no_new_frames", skipped["status"])
            self.assertFalse(skipped["round_consumed"])
            self.assertFalse(
                (run_dir / "adaptive_evidence" / "record_meter" / "cycle_1" / "request_01").exists()
            )
            with mock.patch(
                "meter_rubrics._export_candidates", side_effect=self._fake_export
            ):
                acquired = adaptive_evidence.request_additional_evidence(
                    **common,
                    time_ranges=[{"start_seconds": 1.4, "end_seconds": 1.6}],
                )
            self.assertEqual(1, acquired["request_number"])
            self.assertTrue(acquired["round_consumed"])

    @unittest.skipUnless(
        (AGENT_ROOT / "evaluations" / "r79_adaptive_frame_agent_qwen_smoke_20260818" / "record_rubrics" / "record_evidence_report.json").is_file(),
        "local regression fixture is not published",
    )
    def test_real_cycle_quality_builds_meter_requests_then_advances_cycle(self) -> None:
        base = (
            AGENT_ROOT
            / "evaluations"
            / "r79_adaptive_frame_agent_qwen_smoke_20260818"
            / "record_rubrics"
        )
        report = json.loads(
            (base / "record_evidence_report.json").read_text(encoding="utf-8")
        )
        cycle_reports: dict[str, dict] = {}
        for cycle in (1, 2):
            raw = json.loads(
                (base / "qwen" / f"cycle_{cycle}_meters.json").read_text(
                    encoding="utf-8"
                )
            )
            current = report["cycles"][str(cycle)]
            cycle_reports[str(cycle)] = {
                "window": current["window"],
                "meter_frames": current["meter_frames"],
                "meter_quality": record_rubrics.assess_cycle_meter_evidence(
                    cycle, raw["observation"], current["meter_frames"]
                ),
            }
        parameters = {
            "adaptive_enabled": True,
            "adaptive_max_rounds": 2,
            "adaptive_interval_seconds": 0.2,
            "adaptive_max_frames": 20,
        }
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            first = record_rubrics.build_record_adaptive_recommendation(
                cycle_reports, run_dir, 200.0, parameters
            )
            template = first["adaptive_request_template"]
            self.assertEqual("record_meter", template["evidence_profile"])
            self.assertEqual(1, template["cycle"])
            self.assertEqual("ammeter_missing", template["reason"])
            self.assertEqual(["ammeter", "voltmeter"], template["target_roles"])
            first_dir = (
                run_dir
                / "adaptive_evidence"
                / "record_meter"
                / "cycle_1"
                / "request_01"
            )
            toolkit.write_json(first_dir / "request.json", template)
            toolkit.write_json(first_dir / "result.json", {"evidence_profile": "record_meter"})
            broad = record_rubrics.build_record_adaptive_recommendation(
                cycle_reports, run_dir, 200.0, parameters
            )["adaptive_request_template"]
            self.assertEqual(1, broad["cycle"])
            self.assertEqual("current_run_meter_search", broad["search_mode"])
            self.assertGreaterEqual(broad["interval_seconds"], 0.25)
            old_range = template["time_ranges"][0]
            new_range = broad["time_ranges"][0]
            overlap = max(
                0.0,
                min(old_range["end_seconds"], new_range["end_seconds"])
                - max(old_range["start_seconds"], new_range["start_seconds"]),
            )
            self.assertEqual(0.0, overlap)
            second_dir = first_dir.parent / "request_02"
            toolkit.write_json(second_dir / "request.json", broad)
            toolkit.write_json(second_dir / "result.json", {"evidence_profile": "record_meter"})
            second = record_rubrics.build_record_adaptive_recommendation(
                cycle_reports, run_dir, 200.0, parameters
            )
            template = second["adaptive_request_template"]
            self.assertEqual(2, template["cycle"])
            self.assertEqual("voltmeter_missing", template["reason"])
            self.assertEqual(["voltmeter"], template["target_roles"])

    def test_supplemental_meter_rows_override_by_frame_and_change_digest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            source_digest = "abc"
            result_path = (
                run_dir
                / "adaptive_evidence"
                / "record_meter"
                / "cycle_1"
                / "request_01"
                / "result.json"
            )
            supplemental = {
                "frame_id": "frame_1",
                "frame_number": 1,
                "timestamp_seconds": 1.0,
                "window_source": "record_meter",
                "source_video_sha256": source_digest,
                "dynamic_meter_candidates": [{"candidate_id": "new"}],
            }
            toolkit.write_json(
                result_path,
                {
                    "evidence_profile": "record_meter",
                    "cycle": 1,
                    "source_video_sha256": source_digest,
                    "video_id_used_for_routing": False,
                    "historical_artifacts_used": False,
                    "fixed_video_roi_used": False,
                    "paper_values_used": False,
                    "excel_accessed": False,
                    "meter_rows": [supplemental],
                },
            )
            loaded = record_rubrics._adaptive_record_meter_rows(
                run_dir, 1, source_digest
            )
            merged = record_rubrics._merge_meter_rows(
                [
                    {
                        "frame_id": "frame_1",
                        "frame_number": 1,
                        "timestamp_seconds": 1.0,
                        "window_source": "baseline",
                    },
                    {
                        "frame_id": "frame_2",
                        "frame_number": 2,
                        "timestamp_seconds": 2.0,
                        "window_source": "baseline",
                    },
                ],
                loaded,
            )
            self.assertEqual(2, len(merged))
            self.assertEqual("record_meter", merged[0]["window_source"])
            digest = record_rubrics._adaptive_record_meter_digest(run_dir)
            self.assertIsNotNone(digest)
            self.assertNotEqual(
                "base",
                record_rubrics._record_execution_fingerprint("base", digest),
            )

    def test_meter_media_combines_adaptive_rows_with_strongest_baseline(self) -> None:
        rows = [
            {
                "frame_id": f"baseline_{index}",
                "timestamp_seconds": float(index),
                "image_path": f"baseline_{index}.jpg",
                "role_views": {},
                "dynamic_meter_candidates": [],
                "sharpness": 10.0 + index,
            }
            for index in range(4)
        ]
        rows.extend(
            {
                "frame_id": f"adaptive_{index}",
                "timestamp_seconds": 5.0 + index,
                "image_path": f"adaptive_{index}.jpg",
                "role_views": {"ammeter": {"image_path": f"a_{index}.jpg"}},
                "dynamic_meter_candidates": [{"enhanced_path": f"a_{index}.jpg"}],
                "sharpness": 100.0 + index,
                "window_source": "record_meter",
            }
            for index in range(3)
        )
        groups = record_rubrics._meter_media(rows)
        self.assertEqual(
            ["baseline_3", "adaptive_0", "adaptive_1", "adaptive_2"],
            [row["frame_id"] for row in rows],
        )
        self.assertEqual("baseline_3.jpg", groups[0][0].name)
        self.assertEqual(
            [row["image_path"] for row in rows],
            [str(group[0]) for group in groups],
        )

    def test_panorama_locator_adds_current_frame_role_crops(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame_path = root / "frame.jpg"
            image = np.full((800, 1000, 3), 210, dtype=np.uint8)
            cv2.rectangle(image, (50, 80), (350, 500), (245, 245, 245), -1)
            cv2.rectangle(image, (600, 60), (900, 500), (245, 245, 245), -1)
            self.assertTrue(cv2.imwrite(str(frame_path), image))
            rows = [
                {
                    "frame_id": "frame_00000020",
                    "frame_number": 20,
                    "timestamp_seconds": 4.0,
                    "image_path": str(frame_path),
                    "window_source": "record_meter",
                    "adaptive_request_number": 2,
                    "role_views": {},
                },
                {
                    "frame_id": "frame_00000010",
                    "frame_number": 10,
                    "timestamp_seconds": 2.0,
                    "image_path": str(frame_path),
                    "window_source": "record_meter",
                    "adaptive_request_number": 1,
                    "role_views": {},
                },
            ]
            locations = {
                "meters": [
                    {
                        "identity": "ammeter",
                        "bbox_normalized_1000": [50, 80, 350, 500],
                        "face_visible": True,
                        "confidence": 0.9,
                        "evidence": "visible A glyph and analog arc",
                    },
                    {
                        "identity": "voltmeter",
                        "bbox_normalized_1000": [600, 60, 900, 500],
                        "face_visible": True,
                        "confidence": 0.9,
                        "evidence": "visible V glyph and analog arc",
                    },
                ],
                "cross_role_iou": 0.0,
                "cross_role_containment": 0.0,
            }
            with mock.patch.object(
                record_rubrics,
                "_call_qwen_panorama_locator",
                return_value=locations,
            ) as locator:
                audit = record_rubrics._ground_adaptive_meter_roles(
                    1,
                    rows,
                    {"base_url": "https://example.invalid", "model": "qwen"},
                    root / "record_rubrics",
                    "fingerprint",
                )
            self.assertEqual(2, locator.call_count)
            self.assertEqual("both_roles_grounded", audit[0]["status"])
            self.assertEqual(2, len(audit))
            for row in rows:
                self.assertEqual({"ammeter", "voltmeter"}, set(row["role_views"]))
                for view in row["role_views"].values():
                    self.assertEqual("qwen_panorama_location", view["source"])
                    self.assertTrue(Path(view["image_path"]).is_file())
            location = rows[0]["panorama_location"]
            self.assertFalse(location["video_id_used_for_routing"])
            self.assertFalse(location["historical_artifacts_used"])
            self.assertFalse(location["fixed_video_roi_used"])
            self.assertNotEqual(
                record_rubrics._meter_rows_visual_digest([rows[0]]),
                record_rubrics._meter_rows_visual_digest([]),
            )

    def test_panorama_locator_ignores_nonadaptive_rows(self) -> None:
        rows = [
            {
                "frame_id": "baseline",
                "timestamp_seconds": 1.0,
                "image_path": "missing.jpg",
                "window_source": "baseline",
            }
        ]
        with mock.patch.object(record_rubrics, "_call_qwen_panorama_locator") as locator:
            audit = record_rubrics._ground_adaptive_meter_roles(
                1,
                rows,
                {"base_url": "https://example.invalid", "model": "qwen"},
                Path("unused"),
                "fingerprint",
            )
        self.assertEqual([], audit)
        locator.assert_not_called()

    def test_toolkit_record_meter_invalidates_only_r7_r9(self) -> None:
        (toolkit.AGENT_ROOT / "runs").mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=toolkit.AGENT_ROOT / "runs") as directory:
            run_dir = Path(directory)
            request_dir = (
                run_dir
                / "adaptive_evidence"
                / "record_meter"
                / "cycle_1"
                / "request_01"
            )
            result_path = request_dir / "result.json"
            rubric_5 = run_dir / "rubrics" / "rubric_5.json"
            rubric_7 = run_dir / "rubrics" / "rubric_7.json"
            rubric_9 = run_dir / "rubrics" / "rubric_9.json"
            report_56 = run_dir / "meter_rubrics" / "meter_evidence_report.json"
            report_79 = run_dir / "record_rubrics" / "record_evidence_report.json"
            for path in (rubric_5, rubric_7, rubric_9, report_56, report_79):
                toolkit.write_json(path, {"decision": "fail"})
            toolkit.write_json(
                run_dir / "state.json",
                {
                    "run_id": "fixture",
                    "mode": "execute",
                    "status": "record_rubrics_completed",
                    "video": {"path": "fixture.mp4"},
                    "rubric_results": {
                        "5": str(rubric_5),
                        "7": str(rubric_7),
                        "9": str(rubric_9),
                    },
                    "rubric_evidence_reports": {
                        "5_6": str(report_56),
                        "7_9": str(report_79),
                    },
                    "tool_calls": [],
                    "final_result": None,
                },
            )

            def fake_acquire(**_kwargs: object) -> dict:
                value = {
                    "status": "additional_evidence_ready",
                    "evidence_profile": "record_meter",
                    "request_number": 1,
                    "cycle": 1,
                    "rubric_ids": [7],
                    "reason": "ammeter_missing",
                    "frame_count": 3,
                    "selected_frame_count": 0,
                    "result_path": str(result_path),
                    "request_path": str(request_dir / "request.json"),
                }
                toolkit.write_json(result_path, value)
                return value

            with mock.patch.object(
                toolkit, "_existing_run_dir", return_value=run_dir
            ), mock.patch.object(
                toolkit, "_verify_source_video", return_value=Path("fixture.mp4")
            ), mock.patch.object(
                adaptive_evidence,
                "request_additional_evidence",
                side_effect=fake_acquire,
            ):
                result = toolkit.request_additional_evidence(
                    "fixture",
                    [7],
                    "ammeter_missing",
                    [{"start_seconds": 1.0, "end_seconds": 2.0}],
                    evidence_profile="record_meter",
                    cycle=1,
                    target_roles=["ammeter"],
                    search_mode="adjacent_meter_dense",
                )

            state = toolkit.read_json(run_dir / "state.json")
            self.assertEqual([7, 9], result["invalidated_rubric_ids"])
            self.assertEqual({"5": str(rubric_5)}, state["rubric_results"])
            self.assertEqual({"5_6": str(report_56)}, state["rubric_evidence_reports"])
            self.assertEqual("boundaries_completed", state["status"])
            self.assertEqual(3, len(result["archived_prior_artifacts"]))

    def test_schema_and_orchestrator_route_record_meter_to_r7_r9(self) -> None:
        schema = next(
            item
            for item in toolkit.TOOL_SCHEMAS
            if item["name"] == "request_additional_evidence"
        )["inputSchema"]["properties"]
        self.assertIn("record_meter", schema["evidence_profile"]["enum"])
        self.assertIn("ammeter_missing", schema["reason"]["enum"])
        self.assertEqual(
            ["ammeter", "voltmeter"], schema["target_roles"]["items"]["enum"]
        )
        self.assertIn("adjacent_meter_dense", schema["search_mode"]["enum"])

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
                return {"status": "boundaries_completed", "summary_path": "b.json"}
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
                                    "evidence_profile": "record_meter",
                                    "cycle": 1,
                                    "reason": "ammeter_missing",
                                    "target_roles": ["ammeter"],
                                    "anchor_frame_ids": [],
                                    "search_mode": "adjacent_meter_dense",
                                    "time_ranges": [
                                        {"start_seconds": 1.0, "end_seconds": 2.0}
                                    ],
                                    "interval_seconds": 0.2,
                                    "max_frames": 20,
                                    "roi_mode": "dynamic_meter_candidates",
                                    "view": "meter_pair",
                                },
                            }
                        ],
                    }
                self.assertEqual([7, 9], arguments["rubric_ids"])
                return {"status": "rubric_bundle_completed", "producer_calls": []}
            if name == "request_additional_evidence":
                self.assertEqual("record_meter", arguments["evidence_profile"])
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
        self.assertEqual("completed", result["final"]["status"])
        self.assertEqual(2, bundle_calls)


if __name__ == "__main__":
    unittest.main()
