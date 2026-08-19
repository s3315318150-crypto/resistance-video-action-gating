from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import qwen_hierarchical_v1_contract as contract  # noqa: E402
from qwen_hierarchical_v1_prompts import build_map_prompt, build_map_retry_prompt  # noqa: E402


class HierarchicalContractTests(unittest.TestCase):
    def test_map_prompt_does_not_expose_source_identity(self) -> None:
        source_identity = "student-name-private-video.mp4"
        prompt = build_map_prompt(
            source_identity,
            {"window_id": "window_001", "window_seconds": [0.0, 2.0]},
            [{"image_id": "frame_000001"}],
        )
        self.assertNotIn(source_identity, prompt)
        self.assertIn("匿名", prompt)

    def test_schema_is_independent_and_has_no_battery_stage(self) -> None:
        schema = contract.load_stage_schema(ROOT / "configs" / "action_schemas" / "resistance_7stage_no_battery_v1.json")
        self.assertEqual(contract.STAGE_SCHEMA_ID, schema["stage_schema_id"])
        self.assertEqual(
            [item["id"] for item in schema["stages"]],
            list(contract.STAGES),
        )
        self.assertNotIn("battery_change", [item["id"] for item in schema["stages"]])

    def test_schema_rejects_non_object_extra_entries(self) -> None:
        source = json.loads((ROOT / "configs" / "action_schemas" / "resistance_7stage_no_battery_v1.json").read_text(encoding="utf-8"))
        source["stages"].append("unexpected")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "schema.json"
            path.write_text(json.dumps(source, ensure_ascii=False), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "stages_shape_invalid"):
                contract.load_stage_schema(path)

    def test_source_contract_rejects_invalid_by_default_and_allows_only_explicit_override(self) -> None:
        summary = {
            "records": [
                {
                    "source_video_id": "valid.mp4",
                    "source_manifest": "valid.json",
                    "segment": {"start_seconds": 0.0, "end_seconds": 10.0, "segment_valid": True, "segment_errors": []},
                },
                {
                    "source_video_id": "invalid.mp4",
                    "source_manifest": "invalid.json",
                    "segment": {"start_seconds": 0.0, "end_seconds": 10.0, "segment_valid": False, "segment_errors": ["bad_source"]},
                },
            ]
        }
        accepted, rejected = contract.select_source_records(summary)
        self.assertEqual(["valid.mp4"], [item["source_video_id"] for item in accepted])
        self.assertEqual("source_contract_rejected", rejected[0]["reason"])
        accepted, rejected = contract.select_source_records(summary, allow_invalid_source_segments=True)
        self.assertEqual({"valid.mp4", "invalid.mp4"}, {item["source_video_id"] for item in accepted})
        invalid = next(item for item in accepted if item["source_video_id"] == "invalid.mp4")
        self.assertTrue(invalid["accepted_despite_invalid_source"])
        self.assertTrue(invalid["needs_review"])
        self.assertEqual([], rejected)

    def test_source_contract_rejects_negative_start_even_with_invalid_override(self) -> None:
        summary = {
            "records": [{
                "source_video_id": "bad.mp4",
                "source_manifest": "bad.json",
                "segment": {"start_seconds": -1.0, "end_seconds": 10.0, "segment_valid": False, "segment_errors": ["bad"]},
            }]
        }
        accepted, rejected = contract.select_source_records(summary, allow_invalid_source_segments=True)
        self.assertEqual([], accepted)
        self.assertEqual("source_contract_rejected", rejected[0]["reason"])

    def test_window_geometry_is_60_seconds_with_10_seconds_adjacent_overlap(self) -> None:
        windows = contract.build_overlapping_windows(0.0, 120.0, 60.0, 10.0)
        self.assertEqual([[0.0, 60.0], [50.0, 110.0], [100.0, 120.0]], [item["window_seconds"] for item in windows])
        for left, right in zip(windows, windows[1:]):
            self.assertAlmostEqual(left["window_seconds"][1] - right["window_seconds"][0], 10.0)
        self.assertEqual(0.0, windows[0]["window_seconds"][0])
        self.assertEqual(120.0, windows[-1]["window_seconds"][1])

    def test_fractional_tail_and_true_zero_start_are_preserved(self) -> None:
        windows = contract.build_overlapping_windows(0.0, 184.133, 60.0, 10.0)
        self.assertEqual(0.0, windows[0]["window_seconds"][0])
        self.assertEqual(184.133, windows[-1]["window_seconds"][1])
        timestamps = contract.sample_timestamps(180.0, 184.133, 2.0)
        self.assertEqual([180.0, 182.0, 184.0, 184.133], timestamps)

    def _frames(self) -> list[dict[str, object]]:
        return [
            {"image_id": "frame_00000000", "frame_number": 0, "timestamp_seconds": 0.0},
            {"image_id": "frame_00000060", "frame_number": 60, "timestamp_seconds": 2.0},
            {"image_id": "frame_00000120", "frame_number": 120, "timestamp_seconds": 4.0},
        ]

    def test_map_contract_accepts_only_visible_base_actions(self) -> None:
        value = {
            "window_id": "w000",
            "decision": "observed",
            "observations": [
                {
                    "action_type": "writing_action",
                    "first_frame_id": "frame_00000060",
                    "last_frame_id": "frame_00000120",
                    "representative_frame_id": "frame_00000120",
                    "evidence": "连续画面看到笔在纸上填写表格",
                    "confidence": 0.9,
                }
            ],
            "confidence": 0.9,
            "uncertainty": "",
        }
        self.assertEqual([], contract.validate_map_response(value, "w000", self._frames()))
        for forbidden in ("recording_1", "battery_change", "measurement_2"):
            value["observations"][0]["action_type"] = forbidden
            self.assertIn("observation_0_action_invalid", contract.validate_map_response(value, "w000", self._frames()))
        value["observations"][0]["action_type"] = "writing_action"

    def test_map_contract_rejects_foreign_or_reversed_frames_and_nonfinite_confidence(self) -> None:
        value = {
            "window_id": "w000",
            "decision": "observed",
            "observations": [
                {
                    "action_type": "wiring_action",
                    "first_frame_id": "frame_00000120",
                    "last_frame_id": "frame_00000060",
                    "representative_frame_id": "frame_00000000",
                    "evidence": "看到导线",
                    "confidence": float("nan"),
                }
            ],
            "confidence": True,
            "uncertainty": "",
        }
        errors = contract.validate_map_response(value, "w000", self._frames())
        self.assertIn("observation_0_frame_order_invalid", errors)
        self.assertIn("observation_0_representative_outside_interval", errors)
        self.assertIn("observation_0_confidence_invalid", errors)
        self.assertIn("confidence_invalid", errors)

    def test_map_retry_prompt_explicitly_repairs_missing_observation_confidence(self) -> None:
        prompt = build_map_retry_prompt(
            "BASE",
            ["observation_0_confidence_invalid", "observation_1_confidence_invalid"],
        )
        self.assertIn("每一个 observations 数组元素都必须单独包含 confidence 字段", prompt)
        self.assertIn("0.0 到 1.0 的 JSON 数字", prompt)
        self.assertIn("不能只返回顶层 confidence", prompt)

    def test_map_contract_rejects_observations_returned_out_of_time_order(self) -> None:
        value = {
            "window_id": "w000",
            "decision": "observed",
            "observations": [
                {
                    "action_type": "writing_action",
                    "first_frame_id": "frame_00000120",
                    "last_frame_id": "frame_00000120",
                    "representative_frame_id": "frame_00000120",
                    "evidence": "后段可见持续书写",
                    "confidence": 0.8,
                },
                {
                    "action_type": "wiring_action",
                    "first_frame_id": "frame_00000000",
                    "last_frame_id": "frame_00000060",
                    "representative_frame_id": "frame_00000060",
                    "evidence": "前段可见插接导线",
                    "confidence": 0.8,
                },
            ],
            "confidence": 0.8,
            "uncertainty": "",
        }
        self.assertIn("observation_1_observation_order_invalid", contract.validate_map_response(value, "w000", self._frames()))

    def test_contract_rejects_json_values_with_array_ids_without_throwing(self) -> None:
        map_value = {
            "window_id": "w000",
            "decision": [],
            "observations": [],
            "confidence": 0.5,
            "uncertainty": "",
        }
        self.assertIn("decision_invalid", contract.validate_map_response(map_value, "w000", self._frames()))
        reduce_value = {
            "accepted_event_ids": [[]],
            "rejected_events": [{"event_id": [], "reason": [], "explanation": "x"}],
            "conflicts": [{"event_ids": [[]], "resolution": "x"}],
            "terminal_cleanup_event_id": [],
            "confidence": 0.5,
            "uncertainty": "",
        }
        events = [{"event_id": "evt_0001", "action_type": "wiring_action"}]
        self.assertIn("accepted_event_ids_invalid", contract.validate_reduce_response(reduce_value, events))
        boundary_value = {
            "boundary_id": "b001",
            "decision": [],
            "last_from_frame_id": [],
            "first_to_frame_id": [],
            "evidence": "x",
            "confidence": 0.5,
            "uncertainty": "",
        }
        self.assertIn("decision_invalid", contract.validate_boundary_response(boundary_value, "b001", self._frames()))

    def test_reduce_contract_requires_exhaustive_known_event_ids(self) -> None:
        events = [
            {"event_id": "evt_0001", "action_type": "wiring_action"},
            {"event_id": "evt_0002", "action_type": "writing_action"},
        ]
        valid = {
            "accepted_event_ids": ["evt_0001"],
            "rejected_events": [{"event_id": "evt_0002", "reason": "duplicate", "explanation": "重叠窗口重复证据"}],
            "conflicts": [],
            "terminal_cleanup_event_id": None,
            "confidence": 0.8,
            "uncertainty": "",
        }
        self.assertEqual([], contract.validate_reduce_response(valid, events))
        invalid = {**valid, "accepted_event_ids": ["evt_new"]}
        self.assertIn("accepted_event_ids_invalid", contract.validate_reduce_response(invalid, events))
        invalid = {**valid, "rejected_events": []}
        self.assertIn("reduce_decision_not_exhaustive", contract.validate_reduce_response(invalid, events))

    def test_reduce_cannot_accept_events_after_terminal_cleanup(self) -> None:
        events = [
            {"event_id": "evt_0001", "action_type": "cleanup_action", "first_frame_number": 100, "last_frame_number": 110, "representative_frame_number": 100},
            {"event_id": "evt_0002", "action_type": "writing_action", "first_frame_number": 120, "last_frame_number": 130, "representative_frame_number": 120},
        ]
        value = {
            "accepted_event_ids": ["evt_0001", "evt_0002"],
            "rejected_events": [],
            "conflicts": [],
            "terminal_cleanup_event_id": "evt_0001",
            "confidence": 0.8,
            "uncertainty": "",
        }
        self.assertIn("accepted_event_after_terminal_cleanup", contract.validate_reduce_response(value, events))

    def test_reduce_cannot_accept_event_crossing_terminal_cleanup_start(self) -> None:
        events = [
            {"event_id": "evt_0001", "action_type": "writing_action", "first_frame_number": 95, "last_frame_number": 120, "representative_frame_number": 100},
            {"event_id": "evt_0002", "action_type": "cleanup_action", "first_frame_number": 105, "last_frame_number": 115, "representative_frame_number": 110},
        ]
        value = {
            "accepted_event_ids": ["evt_0001", "evt_0002"],
            "rejected_events": [],
            "conflicts": [],
            "terminal_cleanup_event_id": "evt_0002",
            "confidence": 0.8,
            "uncertainty": "",
        }
        self.assertIn("accepted_event_after_terminal_cleanup", contract.validate_reduce_response(value, events))

    def test_boundary_contract_requires_two_ordered_real_frames(self) -> None:
        frames = self._frames()
        valid = {
            "boundary_id": "b001",
            "decision": "observed",
            "last_from_frame_id": "frame_00000060",
            "first_to_frame_id": "frame_00000120",
            "evidence": "前帧仍在测量，后帧开始连续书写",
            "confidence": 0.8,
            "uncertainty": "",
        }
        self.assertEqual([], contract.validate_boundary_response(valid, "b001", frames))
        invalid = {**valid, "last_from_frame_id": "frame_00000120", "first_to_frame_id": "frame_00000060"}
        self.assertIn("boundary_frame_order_invalid", contract.validate_boundary_response(invalid, "b001", frames))

    def test_new_run_directory_refuses_existing_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = contract.create_run_directory(root, "run_a")
            (first / "legacy_sentinel.txt").write_text("preserve", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                contract.create_run_directory(root, "run_a")
            self.assertEqual("preserve", (first / "legacy_sentinel.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
