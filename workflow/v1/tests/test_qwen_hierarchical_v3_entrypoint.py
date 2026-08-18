from __future__ import annotations

import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import qwen_experiment_action_hierarchical_v3 as v3  # noqa: E402
import qwen_hierarchical_v1_reduce as base_reduce  # noqa: E402


def map_event(source_id: str, action: str, start: float, end: float, subtype: str | None = None) -> dict[str, object]:
    first = int(start * 10)
    last = int(end * 10)
    value: dict[str, object] = {
        "source_event_id": source_id,
        "window_id": "w001",
        "action_type": action,
        "first_frame_id": f"frame_{first:06d}",
        "last_frame_id": f"frame_{last:06d}",
        "representative_frame_id": f"frame_{(first + last) // 2:06d}",
        "first_frame_number": first,
        "last_frame_number": last,
        "representative_frame_number": (first + last) // 2,
        "first_seconds": start,
        "last_seconds": end,
        "representative_seconds": (start + end) / 2.0,
        "evidence": "直接可见动作",
        "confidence": 0.8,
    }
    if subtype is not None:
        value["auxiliary_subtype"] = subtype
    return value


class HierarchicalV3EntrypointTests(unittest.TestCase):
    def tearDown(self) -> None:
        v3.restore_v1_bindings()

    def test_auxiliary_subtypes_are_not_merged_across_same_time_range(self) -> None:
        groups = v3.deduplicate_map_events_v3(
            [
                map_event("a", "auxiliary_action", 1.0, 2.0, "seat_change"),
                map_event("b", "auxiliary_action", 1.0, 2.0, "battery_configuration_change"),
            ]
        )
        self.assertEqual(2, len(groups))
        self.assertEqual({"seat_change", "battery_configuration_change"}, {item["auxiliary_subtype"] for item in groups})

    def test_accepted_auxiliary_event_does_not_compete_with_overlapping_main_action(self) -> None:
        canonical = v3.deduplicate_map_events_v3(
            [
                map_event("main", "wiring_action", 1.0, 3.0),
                map_event("aux", "auxiliary_action", 1.0, 3.0, "teacher_intervention"),
            ]
        )
        response = {
            "accepted_event_ids": [item["event_id"] for item in canonical],
            "rejected_events": [],
            "conflicts": [],
            "terminal_cleanup_event_id": None,
            "confidence": 0.8,
            "uncertainty": "",
        }
        selected, selection = v3.select_events_v3(canonical, response, preserve_equal_confidence=True)
        self.assertEqual(2, len(selected))
        self.assertEqual(1, len(selection["accepted_auxiliary_event_ids"]))
        self.assertEqual([], v3.find_temporal_conflicts_v3(canonical))

    def test_meter_windows_are_derived_from_observed_measurement_runs(self) -> None:
        result = {
            "observed_stage_runs": [
                {"stage": "measurement_1", "start_seconds": 10.0, "end_seconds": 20.0, "confidence": 0.8, "event_ids": ["e1"]},
                {"stage": "recording_1", "start_seconds": 21.0, "end_seconds": 30.0, "confidence": 0.9, "event_ids": ["e2"]},
                {"stage": "measurement_2", "start_seconds": 31.0, "end_seconds": 35.0, "confidence": 0.7, "event_ids": ["e3"]},
            ]
        }
        windows = v3.build_meter_reading_windows(result)
        self.assertEqual(["measurement_1", "measurement_2"], [item["measurement_event"] for item in windows])
        self.assertEqual([12.0, 15.0, 18.0], windows[0]["suggested_sample_times"])
        self.assertEqual([31.0, 34.0], windows[1]["suggested_sample_times"])

    def test_prepare_video_replaces_uniform_frames_without_increasing_model_budget(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "sample.avi"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (96, 64))
            self.assertTrue(writer.isOpened())
            for index in range(80):
                frame = np.zeros((64, 96, 3), dtype=np.uint8)
                frame[:, : 20 + (index % 50)] = 255
                writer.write(frame)
            writer.release()
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "source_video_id": "sample.avi",
                        "source_video": str(video),
                        "video_metadata": {"fps": 10.0, "frame_count": 80},
                    }
                ),
                encoding="utf-8",
            )
            provenance = {
                "source_video_id": "sample.avi",
                "source_manifest": str(manifest),
                "source_segment": {"start_seconds": 0.0, "end_seconds": 7.9},
            }
            args = Namespace(
                window_seconds=6.0,
                overlap_seconds=1.0,
                sample_interval_seconds=2.0,
                max_model_edge=64,
            )
            v3.bind_v3_identity()
            with patch.object(v3, "scan_activity", wraps=v3.scan_activity) as activity_scan:
                prepared = v3.prepare_video_v3(provenance, root / "prepared", args)
            self.assertEqual(1, activity_scan.call_count)
            source_record = prepared["source_record"]
            image_files = list((root / "prepared" / "frames" / "source").glob("*.jpg"))
        self.assertEqual(source_record["unique_source_frame_count"], len(image_files))
        self.assertEqual(source_record["model_selected_unique_frame_count"], len(image_files))
        self.assertEqual(
            source_record["window_frame_reference_count"] - source_record["model_selected_unique_frame_count"],
            source_record["overlap_reference_savings"],
        )

    def test_measurement_binary_yes_becomes_standard_measurement_event(self) -> None:
        frames = [
            {"image_id": "frame_000100", "frame_number": 100, "timestamp_seconds": 10.0},
            {"image_id": "frame_000120", "frame_number": 120, "timestamp_seconds": 12.0},
            {"image_id": "frame_000140", "frame_number": 140, "timestamp_seconds": 14.0},
        ]
        value = {
            "window_id": "w001",
            "measurement_observed": "yes",
            "observations": [
                {
                    "first_frame_id": "frame_000100",
                    "last_frame_id": "frame_000140",
                    "representative_frame_id": "frame_000120",
                    "evidence": "学生操作开关后持续观察电表表盘。",
                    "confidence": 0.86,
                }
            ],
            "decision_evidence_frame_ids": ["frame_000100", "frame_000120"],
            "decision_evidence": "开关操作后学生视线持续朝向电表。",
            "confidence": 0.86,
        }
        self.assertEqual([], v3._validate_measurement_binary(value, "w001", frames))
        events = v3._measurement_binary_events(value, "w001", frames)
        self.assertEqual(1, len(events))
        self.assertEqual("measurement_action", events[0]["action_type"])
        self.assertEqual("measurement", events[0]["independent_binary_confirmation"])
        no_value = {
            "window_id": "w001",
            "measurement_observed": "no",
            "observations": [],
            "decision_evidence_frame_ids": ["frame_000100"],
            "decision_evidence": "全部图片只显示插接导线。",
            "confidence": 0.7,
        }
        self.assertEqual([], v3._validate_measurement_binary(no_value, "w001", frames))
        no_value.pop("decision_evidence")
        self.assertIn("decision_evidence_invalid", v3._validate_measurement_binary(no_value, "w001", frames))

    def test_measurement_bridge_candidates_are_derived_from_event_sequence(self) -> None:
        rewiring = map_event("rewire", "wiring_action", 31.0, 40.0)
        rewiring["evidence"] = "明确换接导线到另一端"
        canonical = v3.deduplicate_map_events_v3(
            [
                map_event("wire", "wiring_action", 0.0, 10.0),
                map_event("measure", "measurement_action", 11.0, 20.0),
                map_event("record", "writing_action", 21.0, 30.0),
                rewiring,
                map_event("later_record", "writing_action", 47.5, 55.0),
                map_event("continued_record", "writing_action", 58.0, 65.0),
            ]
        )
        candidates = v3.discover_measurement_bridge_candidates(canonical, None)
        self.assertEqual(1, len(candidates))
        self.assertEqual([40.0, 47.5], candidates[0]["candidate_range_seconds"])
        self.assertEqual(0.5, candidates[0]["sample_interval_seconds"])
        self.assertEqual(canonical[-2]["event_id"], candidates[0]["writing_event_id"])

    def test_measurement_bridge_uses_first_rewiring_end_for_each_generic_episode(self) -> None:
        first_rewiring = map_event("rewire_a", "wiring_action", 31.0, 40.0)
        first_rewiring["evidence"] = "明确换接导线到另一端"
        same_episode_rewiring = map_event("rewire_b", "wiring_action", 52.0, 55.0)
        same_episode_rewiring["evidence"] = "继续调整并插接实验线路"
        next_episode_rewiring = map_event("rewire_c", "wiring_action", 70.0, 75.0)
        next_episode_rewiring["evidence"] = "再次明确换接导线"
        canonical = v3.deduplicate_map_events_v3(
            [
                map_event("wire", "wiring_action", 0.0, 10.0),
                map_event("measure", "measurement_action", 11.0, 20.0),
                map_event("record", "writing_action", 21.0, 30.0),
                first_rewiring,
                same_episode_rewiring,
                map_event("later_record", "writing_action", 60.0, 65.0),
                next_episode_rewiring,
                map_event("second_later_record", "writing_action", 80.0, 85.0),
            ]
        )

        candidates = v3.discover_measurement_bridge_candidates(canonical, None)

        self.assertEqual(2, len(candidates))
        self.assertEqual([40.0, 60.0], candidates[0]["candidate_range_seconds"])
        self.assertEqual(2, len(candidates[0]["rewiring_event_ids"]))
        self.assertEqual([75.0, 80.0], candidates[1]["candidate_range_seconds"])
        self.assertEqual("first_rewiring_event_end_in_episode", candidates[0]["candidate_start_rule"])

    def test_measurement_bridge_visual_yes_inserts_measurement_2(self) -> None:
        rewiring = map_event("rewire", "wiring_action", 31.0, 40.0)
        rewiring["evidence"] = "明确换接导线到另一端"
        canonical = v3.deduplicate_map_events_v3(
            [
                map_event("wire", "wiring_action", 0.0, 10.0),
                map_event("measure", "measurement_action", 11.0, 20.0),
                map_event("record", "writing_action", 21.0, 30.0),
                rewiring,
                map_event("later_record", "writing_action", 47.5, 55.0),
            ]
        )
        recovered = {
            **map_event("bridge", "measurement_action", 42.0, 45.0),
            "event_id": "measurement_bridge_001",
            "independent_binary_confirmation": "measurement_bridge",
        }
        result = {"selection": {"terminal_cleanup_event_id": None}}
        with patch.object(
            v3,
            "_run_measurement_bridge_candidate",
            return_value=(recovered, {"valid": True, "parsed_result": {"measurement_observed": "yes"}}, []),
        ):
            combined, review = v3._run_measurement_bridge_recovery({}, canonical, result, object(), object())
        state = v3.assign_seven_stages_v3(combined, None)
        stages = {item["event_id"]: item["stage"] for item in state["assigned_events"]}
        self.assertEqual([], review)
        self.assertEqual("measurement_2", stages["measurement_bridge_001"])
        self.assertEqual("recording_2", stages[canonical[-1]["event_id"]])
        self.assertEqual(1, result["measurement_bridge_recovery"]["visual_measurement_recovered_count"])

    def test_measurement_bridge_visual_no_uses_marked_legacy_fallback(self) -> None:
        rewiring = map_event("rewire", "wiring_action", 31.0, 40.0)
        rewiring["evidence"] = "明确换接导线到另一端"
        canonical = v3.deduplicate_map_events_v3(
            [
                map_event("wire", "wiring_action", 0.0, 10.0),
                map_event("measure", "measurement_action", 11.0, 20.0),
                map_event("record", "writing_action", 21.0, 30.0),
                rewiring,
                map_event("later_record", "writing_action", 47.5, 55.0),
            ]
        )
        no_result = {"valid": True, "parsed_result": {"measurement_observed": "no"}}
        result = {"selection": {"terminal_cleanup_event_id": None}}
        with patch.object(
            v3,
            "_run_measurement_bridge_candidate",
            return_value=(None, no_result, []),
        ):
            combined, review = v3._run_measurement_bridge_recovery({}, canonical, result, object(), object())
        state = v3.assign_seven_stages_v3(combined, None)
        fallback = next(item for item in state["assigned_events"] if item["event_id"] == canonical[-1]["event_id"])
        self.assertEqual([], review)
        self.assertEqual("recording_2", fallback["stage"])
        self.assertTrue(fallback["legacy_recording_2_fallback"])
        self.assertTrue(fallback["inferred_stage"])
        self.assertEqual(1, result["measurement_bridge_recovery"]["legacy_fallback_count"])

    def test_endpoint_cleanup_binary_can_create_candidate_without_map_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            registry = {
                number: {
                    "image_id": f"frame_{number:06d}",
                    "frame_number": number,
                    "timestamp_seconds": number / 10.0,
                    "path": str(root / f"frame_{number:06d}.jpg"),
                }
                for number in (0, 20, 40, 60, 80, 100)
            }
            prepared = {
                "video_id": "sample",
                "video_dir": root,
                "manifest": {},
                "frames_dir": root / "frames",
                "frame_registry": registry,
                "fps": 10.0,
                "frame_count": 101,
                "fixed_start": 0.0,
                "fixed_end": 10.0,
            }
            response = {
                "parsed_result": {
                    "cleanup_completed": "yes",
                    "first_cleanup_frame_id": "frame_000060",
                    "last_cleanup_frame_id": "frame_000100",
                    "representative_frame_id": "frame_000080",
                    "experiment_activity_continues_afterward": "no",
                    "evidence_frame_ids": ["frame_000060", "frame_000100"],
                    "evidence": "多根导线拆下，橙红色仪器已放回桌子左上角。",
                    "confidence": 0.91,
                }
            }
            with patch.object(v3.engine, "_extract_source_frames"), patch.object(
                v3.engine, "_attempt_qwen", return_value=response
            ):
                events, result, review = v3._run_endpoint_cleanup_binary(
                    prepared,
                    object(),
                    Namespace(max_model_edge=640, map_max_tokens=1000, max_attempts=1),
                )
        self.assertTrue(result["valid"])
        self.assertEqual([], review)
        self.assertEqual(1, len(events))
        self.assertEqual("cleanup_action", events[0]["action_type"])
        self.assertEqual("endpoint_cleanup", events[0]["independent_binary_confirmation"])

    def test_endpoint_cleanup_binary_candidate_is_promoted_when_reduce_demotes_it(self) -> None:
        raw_events = [
            map_event("wiring", "wiring_action", 0, 5),
            {
                **map_event("endpoint_cleanup_binary_e01", "cleanup_action", 10, 12),
                "independent_binary_confirmation": "endpoint_cleanup",
            },
            map_event("late_writing", "writing_action", 13, 15),
        ]
        canonical = v3.deduplicate_map_events_v3(raw_events)
        wiring, cleanup, late = canonical
        result = {
            "effective_parsed_result": {
                "accepted_event_ids": [wiring["event_id"], late["event_id"]],
                "rejected_events": [
                    {"event_id": cleanup["event_id"], "reason": "other", "explanation": "Reduce 遗漏终态"}
                ],
                "conflicts": [],
                "terminal_cleanup_event_id": None,
                "confidence": 0.5,
                "uncertainty": "",
            },
            "selection": {"terminal_cleanup_event_id": None},
            "accepted_events": [wiring, late],
            "recovery": {"repairs": []},
        }
        selected, terminal = v3._promote_endpoint_cleanup_candidate(
            canonical,
            [wiring, late],
            result,
            Namespace(reduce_recovery_policy="local_partial"),
        )
        self.assertEqual(cleanup["event_id"], terminal["event_id"])
        self.assertEqual(cleanup["event_id"], result["selection"]["terminal_cleanup_event_id"])
        self.assertEqual({wiring["event_id"], cleanup["event_id"]}, {item["event_id"] for item in selected})
        rejected = {item["event_id"]: item["reason"] for item in result["effective_parsed_result"]["rejected_events"]}
        self.assertEqual("post_terminal_cleanup", rejected[late["event_id"]])

    def test_confirmed_cleanup_keeps_terminal_and_noise_barrier(self) -> None:
        raw_events = [map_event("cleanup", "cleanup_action", 10, 12), map_event("wiring", "wiring_action", 13, 16)]
        canonical = v3.deduplicate_map_events_v3(raw_events)
        terminal, later = canonical
        fake_result = {
            "selection": {"terminal_cleanup_event_id": terminal["event_id"], "needs_review": False},
            "effective_parsed_result": {},
            "recovery": {"applied": True, "repairs": [], "ignored_noise_events": [later]},
            "accepted_events": [terminal],
            "ignored_noise_events": [later],
        }
        with tempfile.TemporaryDirectory() as temporary:
            prepared = {
                "video_id": "sample",
                "video_dir": Path(temporary),
                "manifest": {},
                "frames_dir": Path(temporary) / "frames",
                "frame_registry": {},
                "fps": 10.0,
                "frame_count": 1000,
                "fixed_start": 0.0,
                "fixed_end": 30.0,
            }
            for number in v3._cleanup_frame_numbers(prepared, terminal):
                prepared["frame_registry"][number] = {
                    "image_id": f"frame_{number:06d}",
                    "frame_number": number,
                    "timestamp_seconds": number / 10.0,
                    "path": str(Path(temporary) / f"frame_{number:06d}.jpg"),
                }
            evidence_id = next(iter(prepared["frame_registry"].values()))["image_id"]
            response = {
                "parsed_result": {
                    "event_id": terminal["event_id"],
                    "completed_cleanup": "yes",
                    "multiple_wires_disconnected": "yes",
                    "instrument_returned_upper_left": "yes",
                    "seat_change_or_person_change": "no",
                    "experiment_activity_continues_afterward": "no",
                    "evidence_frame_ids": [evidence_id],
                    "evidence": "多根导线拆下且橙红色仪器已回到桌子左上角。",
                    "confidence": 0.9,
                }
            }
            original = v3._ORIGINALS["run_reduce"]
            v3._ORIGINALS["run_reduce"] = lambda *_args: ([terminal], fake_result, [])
            try:
                with patch.object(v3.engine, "_extract_source_frames"), patch.object(v3.engine, "_attempt_qwen", return_value=response):
                    selected, result, review = v3.run_reduce_v3(
                        prepared,
                        raw_events,
                        object(),
                        Namespace(max_model_edge=640, boundary_max_tokens=800, reduce_recovery_policy="local_partial"),
                    )
            finally:
                v3._ORIGINALS["run_reduce"] = original
        self.assertEqual([terminal["event_id"]], [item["event_id"] for item in selected])
        self.assertEqual(terminal["event_id"], result["selection"]["terminal_cleanup_event_id"])
        self.assertEqual([later["event_id"]], [item["event_id"] for item in result["ignored_noise_events"]])
        self.assertTrue(result["cleanup_confirmation"]["confirmed"])
        self.assertEqual([], review)

    def test_critical_boundary_disagreement_records_uncertainty_interval(self) -> None:
        frames = [
            {"image_id": "frame_000100", "timestamp_seconds": 10.0},
            {"image_id": "frame_000110", "timestamp_seconds": 11.0},
            {"image_id": "frame_000150", "timestamp_seconds": 15.0},
        ]
        boundary = {
            "boundary_id": "b001",
            "from_stage": "circuit_wiring",
            "to_stage": "measurement_1",
            "selected_seconds": 15.0,
            "source": "local_1fps_refinement",
            "needs_review": False,
            "passes": {"one_fps": {"input_frames": frames}, "dense_half_second": None},
        }
        response = {
            "parsed_result": {
                "boundary_id": "b001",
                "decision": "observed",
                "last_from_frame_id": "frame_000100",
                "first_to_frame_id": "frame_000110",
                "evidence": "11秒开始观察仪表。",
                "confidence": 0.8,
                "uncertainty": "",
            }
        }
        with tempfile.TemporaryDirectory() as temporary:
            original = v3._ORIGINALS["refine_boundaries"]
            v3._ORIGINALS["refine_boundaries"] = lambda *_args: ([boundary], [])
            try:
                with patch.object(v3.engine, "_attempt_qwen", return_value=response):
                    refined, review = v3.refine_boundaries_v3(
                        {"video_id": "sample", "video_dir": Path(temporary)},
                        [],
                        object(),
                        {},
                        Namespace(boundary_max_tokens=800),
                    )
            finally:
                v3._ORIGINALS["refine_boundaries"] = original
        self.assertEqual([11.0, 15.0], refined[0]["boundary_uncertainty_seconds"])
        self.assertEqual(11.0, refined[0]["selected_seconds"])
        self.assertTrue(refined[0]["needs_review"])
        self.assertIn("critical_boundary_dual_prompt_disagreement:b001", review)


if __name__ == "__main__":
    unittest.main()
