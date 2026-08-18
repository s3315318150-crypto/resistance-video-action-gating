from __future__ import annotations

import sys
import unittest

from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import qwen_experiment_action_hierarchical_v2_screenshot_guard as entrypoint
from qwen_hierarchical_v2_screenshot_guard_reduce import (
    salvage_reduce_response_with_screenshot_guard,
)


def event(event_id: str, action: str, start: int, end: int) -> dict:
    return {
        "event_id": event_id,
        "action_type": action,
        "first_frame_number": start,
        "last_frame_number": end,
        "representative_frame_number": start,
        "first_frame_id": f"frame_{start:06d}",
        "last_frame_id": f"frame_{end:06d}",
        "representative_frame_id": f"frame_{start:06d}",
        "first_seconds": start / 10.0,
        "last_seconds": end / 10.0,
        "representative_seconds": start / 10.0,
        "confidence": 0.8,
        "evidence": action,
    }


class ScreenshotGuardTests(unittest.TestCase):
    def test_restores_non_overlapping_rejections_and_keeps_cleanup_terminal(self) -> None:
        events = [
            event("a", "wiring_action", 0, 10),
            event("b", "measurement_action", 20, 30),
            event("c", "wiring_action", 40, 50),
            event("d", "cleanup_action", 60, 70),
            event("e", "writing_action", 80, 90),
        ]
        reduced, repairs = salvage_reduce_response_with_screenshot_guard(
            events,
            {
                "accepted_event_ids": ["a", "d"],
                "rejected_events": [
                    {"event_id": "b", "reason": "conflicts_with_stronger_evidence"},
                    {"event_id": "c", "reason": "duplicate"},
                ],
                "terminal_cleanup_event_id": "d",
                "confidence": 0.9,
            },
        )
        self.assertIsNotNone(reduced)
        assert reduced is not None
        self.assertEqual(["a", "b", "c", "d"], reduced["accepted_event_ids"])
        self.assertEqual("d", reduced["terminal_cleanup_event_id"])
        self.assertEqual(["b", "c"], reduced["temporal_guard"]["restored_event_ids"])
        self.assertEqual(
            "post_terminal_cleanup",
            next(item["reason"] for item in reduced["rejected_events"] if item["event_id"] == "e"),
        )
        self.assertTrue(any(item["reason"] == "non_overlapping_rejection_restored" for item in repairs))

    def test_real_overlap_witness_is_not_restored(self) -> None:
        events = [
            event("a", "wiring_action", 0, 10),
            event("b", "measurement_action", 20, 30),
            event("c", "wiring_action", 20, 30),
            event("d", "cleanup_action", 60, 70),
        ]
        reduced, _ = salvage_reduce_response_with_screenshot_guard(
            events,
            {
                "accepted_event_ids": ["a", "c", "d"],
                "rejected_events": [{"event_id": "b", "reason": "conflicts_with_stronger_evidence"}],
                "terminal_cleanup_event_id": "d",
                "confidence": 0.9,
            },
        )
        assert reduced is not None
        self.assertNotIn("b", reduced["accepted_event_ids"])
        self.assertEqual("d", reduced["temporal_guard"]["terminal_cleanup_event_id"])

    def test_entrypoint_binds_seven_stage_identity(self) -> None:
        entrypoint.bind_screenshot_guard()
        import qwen_experiment_action_hierarchical_v1 as engine

        self.assertEqual(entrypoint.ALGORITHM_ID, engine.ALGORITHM_ID)
        self.assertEqual(entrypoint.STAGE_SCHEMA_ID, engine.STAGE_SCHEMA_ID)
        self.assertEqual(entrypoint.ALGORITHM_SCHEMA_VERSION, engine.ALGORITHM_SCHEMA_VERSION)


if __name__ == "__main__":
    unittest.main()
