from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import qwen_experiment_action_hierarchical_v1 as engine  # noqa: E402
import qwen_experiment_action_hierarchical_v2_temporal_guard as entrypoint  # noqa: E402
from qwen_hierarchical_v1_contract import validate_reduce_response  # noqa: E402
from qwen_hierarchical_v1_reduce import assign_seven_stages  # noqa: E402
from qwen_hierarchical_v2_temporal_guard_reduce import (  # noqa: E402
    salvage_reduce_response_with_temporal_guard,
    select_events_with_temporal_guard,
)


def event(event_id: str, action: str, first: int, last: int, confidence: float = 0.9) -> dict[str, object]:
    return {
        "event_id": event_id,
        "source_event_id": event_id,
        "window_id": "w000",
        "action_type": action,
        "first_frame_id": f"frame_{first:06d}",
        "last_frame_id": f"frame_{last:06d}",
        "representative_frame_id": f"frame_{first:06d}",
        "first_frame_number": first,
        "last_frame_number": last,
        "representative_frame_number": first,
        "first_seconds": first / 30.0,
        "last_seconds": last / 30.0,
        "representative_seconds": first / 30.0,
        "evidence": "画面直接观察到动作。",
        "confidence": confidence,
    }


def reduce_result(
    accepted: list[str],
    rejected: list[tuple[str, str]],
    terminal: str | None,
) -> dict[str, object]:
    return {
        "accepted_event_ids": accepted,
        "rejected_events": [
            {"event_id": event_id, "reason": reason, "explanation": "被后续更强事件覆盖。"}
            for event_id, reason in rejected
        ],
        "conflicts": [],
        "terminal_cleanup_event_id": terminal,
        "confidence": 0.9,
        "uncertainty": "",
    }


class TemporalGuardTests(unittest.TestCase):
    def tearDown(self) -> None:
        entrypoint.restore_original_bindings()

    def test_non_overlapping_conflict_rejection_is_restored(self) -> None:
        events = [
            event("e1", "wiring_action", 0, 100),
            event("e2", "writing_action", 120, 180),
            event("e3", "cleanup_action", 240, 300),
        ]
        model = reduce_result(["e1", "e3"], [("e2", "conflicts_with_stronger_evidence")], "e3")

        repaired, repairs = salvage_reduce_response_with_temporal_guard(events, model)

        self.assertIsNotNone(repaired)
        self.assertEqual([], validate_reduce_response(repaired, events))
        self.assertEqual(["e1", "e2", "e3"], repaired["accepted_event_ids"])
        self.assertEqual(["e2"], repaired["temporal_guard"]["restored_event_ids"])
        self.assertTrue(any(item["reason"] == "non_overlapping_rejection_restored" for item in repairs))

    def test_true_overlapping_conflict_rejection_remains_rejected(self) -> None:
        events = [
            event("e1", "wiring_action", 0, 100),
            event("e2", "writing_action", 50, 80),
            event("e3", "cleanup_action", 240, 300),
        ]
        model = reduce_result(["e1", "e3"], [("e2", "conflicts_with_stronger_evidence")], "e3")

        repaired, _repairs = salvage_reduce_response_with_temporal_guard(events, model)

        self.assertEqual(["e1", "e3"], repaired["accepted_event_ids"])
        self.assertEqual(["e2"], [item["event_id"] for item in repaired["rejected_events"]])
        self.assertEqual([], repaired["temporal_guard"]["restored_event_ids"])

    def test_post_terminal_event_is_never_restored(self) -> None:
        events = [
            event("e1", "wiring_action", 0, 100),
            event("e2", "cleanup_action", 120, 180),
            event("e3", "writing_action", 200, 240),
        ]
        model = reduce_result(["e1", "e2"], [("e3", "conflicts_with_stronger_evidence")], "e2")

        repaired, _repairs = salvage_reduce_response_with_temporal_guard(events, model)

        self.assertEqual(["e1", "e2"], repaired["accepted_event_ids"])
        rejected = next(item for item in repaired["rejected_events"] if item["event_id"] == "e3")
        self.assertEqual("post_terminal_cleanup", rejected["reason"])

    def test_generic_wiring_writing_rewiring_writing_sequence_is_recovered(self) -> None:
        events = [
            event("e1", "wiring_action", 0, 100),
            event("e2", "wiring_action", 120, 180),
            event("e3", "writing_action", 200, 230),
            event("e4", "wiring_action", 250, 280),
            event("e5", "writing_action", 300, 340),
            event("e6", "wiring_action", 360, 390),
            event("e7", "cleanup_action", 420, 480),
        ]
        model = reduce_result(
            ["e1", "e6", "e7"],
            [
                ("e2", "conflicts_with_stronger_evidence"),
                ("e3", "conflicts_with_stronger_evidence"),
                ("e4", "conflicts_with_stronger_evidence"),
                ("e5", "conflicts_with_stronger_evidence"),
            ],
            "e7",
        )

        repaired, _repairs = salvage_reduce_response_with_temporal_guard(events, model)
        selected, selection = select_events_with_temporal_guard(events, repaired, preserve_equal_confidence=True)
        stages = assign_seven_stages(selected, "e7")

        self.assertEqual(4, repaired["temporal_guard"]["restored_event_count"])
        self.assertEqual("qwen_global_reduce_with_temporal_rejection_guard", selection["mode"])
        self.assertEqual(
            ["circuit_wiring", "circuit_wiring", "recording_1", "circuit_rewiring", "recording_2", "material_cleanup"],
            [item["stage"] for item in stages["observed_stage_intervals"]],
        )

    def test_entrypoint_is_archived_and_disabled(self) -> None:
        self.assertTrue(entrypoint.DEPRECATED)
        self.assertEqual("qwen_experiment_action_hierarchical_v2.py", entrypoint.REPLACEMENT_SCRIPT)
        self.assertEqual(2, entrypoint.main([]))
        self.assertNotEqual("qwen_experiment_action_hierarchical_v2_temporal_guard", engine.ALGORITHM_ID)


if __name__ == "__main__":
    unittest.main()
