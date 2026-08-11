from __future__ import annotations

import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import qwen_experiment_action_hierarchical_v3 as v3  # noqa: E402
import qwen_hierarchical_v1_reduce as base_reduce  # noqa: E402
from qwen_hierarchical_v3_reduce import assign_seven_stages_v3  # noqa: E402


def event(event_id: str, action: str, start: float, end: float, evidence: str = "直接可见动作") -> dict[str, object]:
    first = int(start * 10)
    last = int(end * 10)
    return {
        "event_id": event_id,
        "source_event_id": event_id + "_source",
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
        "evidence": evidence,
        "confidence": 0.8,
    }


class HierarchicalV3ReduceTests(unittest.TestCase):
    def test_short_measurement_then_correction_wiring_stays_in_first_cycle(self) -> None:
        events = [
            event("e1", "wiring_action", 0, 10),
            event("e2", "measurement_action", 11, 14),
            event("e3", "wiring_action", 15, 20, "发现接触不良后插紧导线"),
            event("e4", "measurement_action", 21, 30),
        ]
        result = assign_seven_stages_v3(events, None)
        stages = {item["event_id"]: item["stage"] for item in result["assigned_events"]}
        self.assertEqual("circuit_wiring", stages["e3"])
        self.assertEqual("measurement_1", stages["e4"])

    def test_formal_rewiring_enters_second_cycle(self) -> None:
        events = [
            event("e1", "wiring_action", 0, 10),
            event("e2", "measurement_action", 11, 22),
            event("e3", "wiring_action", 23, 35, "将导线换接到电池盒另一端，形成新接法"),
            event("e4", "measurement_action", 36, 45),
        ]
        result = assign_seven_stages_v3(events, None)
        stages = {item["event_id"]: item["stage"] for item in result["assigned_events"]}
        self.assertEqual("circuit_rewiring", stages["e3"])
        self.assertEqual("measurement_2", stages["e4"])

    def test_batched_writing_is_kept_for_both_record_searches(self) -> None:
        events = [
            event("e1", "wiring_action", 0, 10),
            event("e2", "measurement_action", 11, 22),
            event("e3", "wiring_action", 23, 35, "明确换接导线到另一端"),
            event("e4", "measurement_action", 36, 45),
            event("e5", "writing_action", 46, 60),
        ]
        result = assign_seven_stages_v3(events, None)
        writing = next(item for item in result["assigned_events"] if item["event_id"] == "e5")
        self.assertEqual("recording_2", writing["stage"])
        self.assertTrue(writing["batched_recording"])
        self.assertEqual(["recording_1", "recording_2"], writing["recording_search_aliases"])

    def test_auxiliary_and_out_of_order_events_are_preserved_as_anomalies(self) -> None:
        auxiliary = event("e1", "auxiliary_action", 0, 2, "学生换座位")
        auxiliary["auxiliary_subtype"] = "seat_change"
        result = assign_seven_stages_v3([auxiliary], None)
        self.assertEqual(1, len(result["anomalous_events"]))
        self.assertEqual("seat_change", result["anomalous_events"][0]["auxiliary_subtype"])

    def test_unconfirmed_cleanup_restores_post_cleanup_event(self) -> None:
        raw_events = [
            event("source_cleanup", "cleanup_action", 10, 12),
            event("source_wiring", "wiring_action", 13, 16),
            event("source_duplicate", "measurement_action", 17, 18),
        ]
        canonical = base_reduce.deduplicate_map_events(raw_events)
        terminal, later, duplicate = canonical
        fake_result = {
            "selection": {"terminal_cleanup_event_id": terminal["event_id"], "needs_review": False},
            "effective_parsed_result": {
                "accepted_event_ids": [terminal["event_id"]],
                "rejected_events": [
                    {"event_id": later["event_id"], "reason": "post_terminal_cleanup", "explanation": "候选终态之后"},
                    {"event_id": duplicate["event_id"], "reason": "duplicate", "explanation": "重复弱证据"},
                ],
                "conflicts": [],
                "terminal_cleanup_event_id": terminal["event_id"],
                "confidence": 0.8,
                "uncertainty": "",
            },
            "recovery": {"applied": True, "repairs": [], "ignored_noise_events": [later]},
            "accepted_events": [terminal],
            "ignored_noise_events": [later],
        }
        fake_base = ([terminal], fake_result, [])
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
            original = v3._ORIGINALS["run_reduce"]
            v3._ORIGINALS["run_reduce"] = lambda *_args: fake_base
            try:
                response = {
                    "parsed_result": {
                        "event_id": terminal["event_id"],
                        "completed_cleanup": "no",
                        "multiple_wires_disconnected": "no",
                        "instrument_returned_upper_left": "no",
                        "seat_change_or_person_change": "no",
                        "experiment_activity_continues_afterward": "yes",
                        "evidence_frame_ids": [next(iter(prepared["frame_registry"].values()))["image_id"]],
                        "evidence": "学生之后继续接线。",
                        "confidence": 0.8,
                    }
                }
                with patch.object(v3.engine, "_extract_source_frames"), patch.object(v3.engine, "_attempt_qwen", return_value=response):
                    selected, result, review = v3.run_reduce_v3(
                        prepared,
                        raw_events,
                        object(),
                        Namespace(max_model_edge=640, boundary_max_tokens=800, reduce_recovery_policy="local_partial"),
                    )
            finally:
                v3._ORIGINALS["run_reduce"] = original
        self.assertEqual({terminal["event_id"], later["event_id"]}, {item["event_id"] for item in selected})
        self.assertNotIn(duplicate["event_id"], {item["event_id"] for item in selected})
        self.assertIsNone(result["selection"]["terminal_cleanup_event_id"])
        self.assertEqual([], result["ignored_noise_events"])
        self.assertTrue(any(item.startswith("cleanup_confirmation_not_confirmed") for item in review))


if __name__ == "__main__":
    unittest.main()
