from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import qwen_experiment_action_hierarchical_v1 as engine  # noqa: E402
import qwen_experiment_action_hierarchical_v2_behavior_tolerant as behavior  # noqa: E402
import qwen_experiment_action_hierarchical_v2_behavior_tolerant_adaptive as adaptive_entry  # noqa: E402
import qwen_experiment_action_hierarchical_v2_behavior_tolerant_aux as aux_entry  # noqa: E402
import qwen_experiment_action_hierarchical_v2_behavior_tolerant_boundary as boundary_entry  # noqa: E402
import qwen_hierarchical_v1_contract as contract  # noqa: E402
from compare_v2_behavior_tolerant import compare_runs  # noqa: E402
from qwen_hierarchical_v2_behavior_tolerant_aux import (  # noqa: E402
    BASE_ACTIONS,
    build_map_prompt_auxiliary,
    deduplicate_map_events_auxiliary,
    select_events_auxiliary,
    validate_map_response_auxiliary,
)
from qwen_hierarchical_v2_behavior_tolerant_boundary import (  # noqa: E402
    discover_boundary_bridge_candidates,
    validate_boundary_bridge_response,
)
from qwen_hierarchical_v2_behavior_tolerant_reduce import (  # noqa: E402
    assign_seven_stages_behavior_tolerant,
)
from qwen_hierarchical_v2_behavior_tolerant_sampling import (  # noqa: E402
    scan_activity_compensated,
    select_supplemental_timestamps,
)


def event(
    event_id: str,
    action: str,
    start: float,
    end: float,
    evidence: str = "直接可见动作",
    subtype: str | None = None,
) -> dict[str, object]:
    first = int(start * 10)
    last = int(end * 10)
    value: dict[str, object] = {
        "event_id": event_id,
        "source_event_id": event_id + "_source",
        "window_id": "w001",
        "action_type": action,
        "first_frame_id": f"frame_{first:08d}",
        "last_frame_id": f"frame_{last:08d}",
        "representative_frame_id": f"frame_{(first + last) // 2:08d}",
        "first_frame_number": first,
        "last_frame_number": last,
        "representative_frame_number": (first + last) // 2,
        "first_seconds": start,
        "last_seconds": end,
        "representative_seconds": (start + end) / 2.0,
        "evidence": evidence,
        "confidence": 0.8,
    }
    if subtype is not None:
        value["auxiliary_subtype"] = subtype
    return value


class BehaviorTolerantDecoderTests(unittest.TestCase):
    def test_short_measurement_and_correction_cue_stay_in_first_cycle(self) -> None:
        events = [
            event("e1", "wiring_action", 0, 10),
            event("e2", "measurement_action", 11, 14),
            event("e3", "wiring_action", 15, 20, "发现接触不良后重新插紧导线"),
            event("e4", "measurement_action", 21, 30),
        ]
        result = assign_seven_stages_behavior_tolerant(events, None)
        stages = {item["event_id"]: item["stage"] for item in result["assigned_events"]}
        self.assertEqual("circuit_wiring", stages["e3"])
        self.assertEqual("measurement_1", stages["e4"])

    def test_formal_rewiring_enters_second_cycle(self) -> None:
        events = [
            event("e1", "wiring_action", 0, 10),
            event("e2", "measurement_action", 11, 22),
            event("e3", "wiring_action", 23, 35, "明确将导线换接到另一端"),
            event("e4", "measurement_action", 36, 45),
        ]
        result = assign_seven_stages_behavior_tolerant(events, None)
        stages = {item["event_id"]: item["stage"] for item in result["assigned_events"]}
        self.assertEqual("circuit_rewiring", stages["e3"])
        self.assertEqual("measurement_2", stages["e4"])

    def test_batched_writing_has_two_recording_aliases(self) -> None:
        events = [
            event("e1", "wiring_action", 0, 10),
            event("e2", "measurement_action", 11, 20),
            event("e3", "wiring_action", 21, 30, "明确换接导线形成第二组配置"),
            event("e4", "measurement_action", 31, 40),
            event("e5", "writing_action", 41, 55),
        ]
        result = assign_seven_stages_behavior_tolerant(events, None)
        writing = next(item for item in result["assigned_events"] if item["event_id"] == "e5")
        self.assertEqual("recording_2", writing["stage"])
        self.assertTrue(writing["batched_recording"])
        self.assertEqual(["recording_1", "recording_2"], writing["recording_search_aliases"])

    def test_legacy_rewiring_then_writing_is_marked_as_inferred(self) -> None:
        events = [
            event("e1", "wiring_action", 0, 10),
            event("e2", "writing_action", 11, 20),
            event("e3", "wiring_action", 21, 30),
            event("e4", "writing_action", 31, 40),
        ]
        result = assign_seven_stages_behavior_tolerant(events, None)
        writing = next(item for item in result["assigned_events"] if item["event_id"] == "e4")
        self.assertEqual("recording_2", writing["stage"])
        self.assertTrue(writing["inferred_stage"])
        self.assertFalse(writing["measurement_2_observed"])

    def test_recording_two_can_return_to_measurement_two_with_penalty(self) -> None:
        events = [
            event("e1", "wiring_action", 0, 10),
            event("e2", "writing_action", 11, 20),
            event("e3", "wiring_action", 21, 30),
            event("e4", "measurement_action", 31, 36),
            event("e5", "writing_action", 37, 42),
            event("e6", "measurement_action", 43, 48),
            event("e7", "writing_action", 49, 55),
        ]
        result = assign_seven_stages_behavior_tolerant(events, None)
        assigned = {item["event_id"]: item for item in result["assigned_events"]}
        self.assertEqual("measurement_2", assigned["e6"]["stage"])
        self.assertEqual(-0.25, assigned["e6"]["transition_penalty"])
        self.assertEqual("recording_2", assigned["e7"]["stage"])

    def test_terminal_cleanup_discards_later_state_events(self) -> None:
        events = [
            event("e1", "wiring_action", 0, 10),
            event("e2", "cleanup_action", 11, 20),
            event("e3", "writing_action", 21, 25),
        ]
        result = assign_seven_stages_behavior_tolerant(events, "e2")
        self.assertEqual(["e3"], result["analysis_termination"]["discarded_after_terminal_event_ids"])


class AuxiliaryActionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.base_actions = contract.BASE_ACTIONS
        contract.BASE_ACTIONS = BASE_ACTIONS

    def tearDown(self) -> None:
        contract.BASE_ACTIONS = self.base_actions

    def test_auxiliary_contract_requires_known_subtype(self) -> None:
        frames = [
            {"image_id": "frame_00000000", "frame_number": 0, "timestamp_seconds": 0.0},
            {"image_id": "frame_00000020", "frame_number": 20, "timestamp_seconds": 2.0},
        ]
        response = {
            "window_id": "w001",
            "decision": "observed",
            "observations": [
                {
                    "action_type": "auxiliary_action",
                    "auxiliary_subtype": "seat_change",
                    "first_frame_id": "frame_00000000",
                    "last_frame_id": "frame_00000020",
                    "representative_frame_id": "frame_00000020",
                    "evidence": "学生明确换座位",
                    "confidence": 0.8,
                }
            ],
            "confidence": 0.8,
            "uncertainty": "",
        }
        self.assertEqual([], validate_map_response_auxiliary(response, "w001", frames))
        response["observations"][0]["auxiliary_subtype"] = "unsupported"
        self.assertIn("observation_0_auxiliary_subtype_invalid", validate_map_response_auxiliary(response, "w001", frames))

    def test_prompt_keeps_seat_change_outside_cleanup(self) -> None:
        frames = [{"image_id": "frame_00000000", "frame_number": 0, "timestamp_seconds": 0.0}]
        prompt = build_map_prompt_auxiliary("sample", {"window_id": "w001", "window_seconds": [0.0, 1.0]}, frames)
        self.assertIn("battery_configuration_change", prompt)
        self.assertIn("换座位或闲聊不能单独输出 cleanup_action", prompt)

    def test_different_auxiliary_subtypes_are_not_merged(self) -> None:
        groups = deduplicate_map_events_auxiliary(
            [
                event("e1", "auxiliary_action", 0, 2, subtype="seat_change"),
                event("e2", "auxiliary_action", 0, 2, subtype="conversation"),
            ]
        )
        self.assertEqual({"seat_change", "conversation"}, {item["auxiliary_subtype"] for item in groups})

    def test_auxiliary_event_is_preserved_without_competing_with_main_event(self) -> None:
        canonical = deduplicate_map_events_auxiliary(
            [
                event("e1", "wiring_action", 0, 2),
                event("e2", "auxiliary_action", 0, 2, subtype="teacher_intervention"),
            ]
        )
        main = next(item for item in canonical if item["action_type"] == "wiring_action")
        aux = next(item for item in canonical if item["action_type"] == "auxiliary_action")
        reduce_result = {
            "accepted_event_ids": [main["event_id"]],
            "rejected_events": [{"event_id": aux["event_id"], "reason": "other", "explanation": "模型误删"}],
            "conflicts": [],
            "terminal_cleanup_event_id": None,
            "confidence": 0.8,
            "uncertainty": "",
        }
        selected, selection = select_events_auxiliary(canonical, reduce_result)
        self.assertEqual(2, len(selected))
        self.assertEqual([aux["event_id"]], selection["accepted_auxiliary_event_ids"])

    def test_auxiliary_event_after_terminal_is_not_reintroduced(self) -> None:
        canonical = deduplicate_map_events_auxiliary(
            [
                event("e1", "cleanup_action", 10, 12),
                event("e2", "auxiliary_action", 13, 15, subtype="conversation"),
            ]
        )
        cleanup = next(item for item in canonical if item["action_type"] == "cleanup_action")
        auxiliary = next(item for item in canonical if item["action_type"] == "auxiliary_action")
        reduce_result = {
            "accepted_event_ids": [cleanup["event_id"]],
            "rejected_events": [
                {"event_id": auxiliary["event_id"], "reason": "post_terminal_cleanup", "explanation": "终态后"}
            ],
            "conflicts": [],
            "terminal_cleanup_event_id": cleanup["event_id"],
            "confidence": 0.8,
            "uncertainty": "",
        }
        selected, selection = select_events_auxiliary(canonical, reduce_result)
        self.assertEqual([cleanup["event_id"]], [item["event_id"] for item in selected])
        self.assertEqual([auxiliary["event_id"]], selection["post_terminal_auxiliary_event_ids"])


class BoundaryReviewTests(unittest.TestCase):
    @staticmethod
    def prepared() -> dict[str, object]:
        return {
            "fixed_start": 0.0,
            "fixed_end": 110.0,
            "prepared_windows": [
                {"window_id": "w001", "window_seconds": [0.0, 60.0]},
                {"window_id": "w002", "window_seconds": [50.0, 110.0]},
            ],
        }

    def test_non_boundary_event_creates_no_candidate(self) -> None:
        results = [
            {"window_id": "w001", "valid": True, "normalized_events": [event("e1", "writing_action", 20, 30)]},
            {"window_id": "w002", "valid": True, "normalized_events": []},
        ]
        self.assertEqual([], discover_boundary_bridge_candidates(self.prepared(), results, 2.0))

    def test_edge_touching_same_action_creates_bounded_candidate(self) -> None:
        left = event("e1", "wiring_action", 54, 60)
        right = event("e2", "wiring_action", 58, 66)
        results = [
            {"window_id": "w001", "valid": True, "normalized_events": [left]},
            {"window_id": "w002", "valid": True, "normalized_events": [right]},
        ]
        candidates = discover_boundary_bridge_candidates(self.prepared(), results, 2.0)
        boundary_60 = next(item for item in candidates if item["boundary_seconds"] == 60.0)
        self.assertEqual([50.0, 70.0], boundary_60["review_window_seconds"])
        self.assertIn("same_action_reported_across_adjacent_windows", boundary_60["trigger_reasons"])

    def test_same_action_inside_overlap_without_edge_contact_needs_no_bridge(self) -> None:
        left = event("e1", "wiring_action", 54, 55)
        right = event("e2", "wiring_action", 54, 55)
        results = [
            {"window_id": "w001", "valid": True, "normalized_events": [left]},
            {"window_id": "w002", "valid": True, "normalized_events": [right]},
        ]
        self.assertEqual([], discover_boundary_bridge_candidates(self.prepared(), results, 2.0))

    def test_bridge_response_cannot_change_action_type(self) -> None:
        candidate = {"bridge_id": "boundary_bridge_001", "action_type": "wiring_action"}
        frames = [
            {"image_id": "frame_00000000"},
            {"image_id": "frame_00000010"},
        ]
        response = {
            "bridge_id": "boundary_bridge_001",
            "decision": "observed",
            "action_type": "writing_action",
            "first_frame_id": "frame_00000000",
            "last_frame_id": "frame_00000010",
            "representative_frame_id": "frame_00000010",
            "evidence": "直接可见",
            "confidence": 0.8,
        }
        self.assertIn("action_type_mismatch", validate_boundary_bridge_response(response, candidate, frames))


class SupplementalSamplingTests(unittest.TestCase):
    def test_budget_bucket_and_base_gap_contract(self) -> None:
        base = [float(value) for value in range(0, 62, 2)]
        activity = [
            {"timestamp_seconds": index * 0.5, "raw_activity_score": float(index % 11)}
            for index in range(121)
        ]
        selected, diagnostic = select_supplemental_timestamps(0.0, 60.0, base, activity)
        self.assertLessEqual(len(selected), int(len(base) * 0.25))
        self.assertEqual(len({int(value // 10.0) for value in selected}), len(selected))
        self.assertTrue(all(all(abs(value - anchor) >= 0.5 - 1e-9 for anchor in base) for value in selected))
        self.assertEqual(len(selected), diagnostic["extra_frame_count"])

    def test_activity_scan_handles_camera_translation_and_local_change(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            video = Path(temporary) / "motion.avi"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (96, 64))
            self.assertTrue(writer.isOpened())
            for index in range(30):
                frame = np.zeros((64, 96, 3), dtype=np.uint8)
                cv2.rectangle(frame, (10 + index // 10, 15), (30 + index // 10, 35), (255, 255, 255), -1)
                if index >= 20:
                    cv2.circle(frame, (70, 40), 8, (0, 255, 0), -1)
                writer.write(frame)
            writer.release()
            samples = scan_activity_compensated(video, 0.0, 2.9, 0.5)
        self.assertGreaterEqual(len(samples), 6)
        self.assertTrue(all("global_camera_motion" in item for item in samples))
        self.assertGreater(max(item["raw_activity_score"] for item in samples), 0.0)


class EntrypointAndGoldenTests(unittest.TestCase):
    def tearDown(self) -> None:
        adaptive_entry.restore_original_bindings()

    def test_each_entrypoint_has_an_independent_identity(self) -> None:
        variants = [
            (behavior.bind_behavior_tolerant, behavior.ALGORITHM_ID),
            (aux_entry.bind_behavior_tolerant_aux, aux_entry.ALGORITHM_ID),
            (boundary_entry.bind_behavior_tolerant_boundary, boundary_entry.ALGORITHM_ID),
            (adaptive_entry.bind_behavior_tolerant_adaptive, adaptive_entry.ALGORITHM_ID),
        ]
        for bind, algorithm_id in variants:
            bind()
            self.assertEqual(algorithm_id, engine.ALGORITHM_ID)
            adaptive_entry.restore_original_bindings()

    def test_golden_fixture_has_five_records_and_two_second_tolerance(self) -> None:
        value = json.loads((ROOT / "tests" / "fixtures" / "v2_behavior_tolerant_golden.json").read_text(encoding="utf-8"))
        self.assertEqual(5, len(value["records"]))
        self.assertEqual(2.0, value["boundary_tolerance_seconds"])

    def test_golden_comparator_rejects_stage_change_and_large_boundary_shift(self) -> None:
        expected = [["circuit_wiring", 0.0, 10.0]]
        passed, _ = compare_runs([{"stage": "circuit_wiring", "start_seconds": 1.0, "end_seconds": 12.0}], expected, 2.0)
        self.assertTrue(passed)
        passed, differences = compare_runs(
            [{"stage": "recording_1", "start_seconds": 0.0, "end_seconds": 13.0}], expected, 2.0
        )
        self.assertFalse(passed)
        self.assertEqual({"stage_mismatch", "boundary_outside_tolerance"}, {item["code"] for item in differences})
