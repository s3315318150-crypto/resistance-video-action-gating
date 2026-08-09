from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from qwen_hierarchical_v1_reduce import (  # noqa: E402
    assign_seven_stages,
    build_boundary_candidates,
    build_evidence_timeline,
    deduplicate_map_events,
    hard_stop_trigger_keywords,
    is_hard_stop_triggered,
    merge_observed_stage_runs,
    resolve_accepted_conflicts,
    salvage_reduce_response,
    select_events,
)


def event(event_id: str, action: str, first: int, last: int | None = None, confidence: float = 0.8) -> dict[str, object]:
    last = first if last is None else last
    return {
        "event_id": event_id,
        "source_event_id": event_id,
        "window_id": "w000",
        "action_type": action,
        "first_frame_id": f"frame_{first:08d}",
        "last_frame_id": f"frame_{last:08d}",
        "representative_frame_id": f"frame_{first:08d}",
        "first_frame_number": first,
        "last_frame_number": last,
        "representative_frame_number": first,
        "first_seconds": first / 30.0,
        "last_seconds": last / 30.0,
        "representative_seconds": first / 30.0,
        "evidence": action,
        "confidence": confidence,
    }


class HierarchicalReduceTests(unittest.TestCase):
    def test_overlapping_window_events_deduplicate_by_source_frame_range(self) -> None:
        raw = [
            event("w000_e01", "wiring_action", 100, 200, 0.7),
            {**event("w001_e01", "wiring_action", 160, 260, 0.9), "window_id": "w001"},
        ]
        canonical = deduplicate_map_events(raw)
        self.assertEqual(1, len(canonical))
        self.assertEqual({"w000_e01", "w001_e01"}, set(canonical[0]["source_event_ids"]))
        self.assertEqual(100, canonical[0]["first_frame_number"])
        self.assertEqual(260, canonical[0]["last_frame_number"])

    def test_canonical_sequence_maps_to_seven_stages(self) -> None:
        events = [
            event("e1", "wiring_action", 0, 30),
            event("e2", "measurement_action", 60, 90),
            event("e3", "writing_action", 120, 150),
            event("e4", "wiring_action", 180, 210),
            event("e5", "measurement_action", 240, 270),
            event("e6", "writing_action", 300, 330),
            event("e7", "cleanup_action", 360, 390),
            event("e8", "writing_action", 420, 450),
        ]
        result = assign_seven_stages(events, "e7")
        self.assertEqual(
            [
                "circuit_wiring",
                "measurement_1",
                "recording_1",
                "circuit_rewiring",
                "measurement_2",
                "recording_2",
                "material_cleanup",
            ],
            [item["stage"] for item in result["observed_stage_intervals"]],
        )
        self.assertEqual(["e8"], result["analysis_termination"]["discarded_after_terminal_event_ids"])

    def test_second_recording_can_be_inferred_after_rewiring_when_measurement_is_missing(self) -> None:
        events = [
            event("e1", "wiring_action", 0),
            event("e2", "writing_action", 30),
            event("e3", "wiring_action", 60),
            event("e4", "writing_action", 90),
        ]
        result = assign_seven_stages(events, None)
        self.assertEqual(
            ["circuit_wiring", "recording_1", "circuit_rewiring", "recording_2"],
            [item["stage"] for item in result["observed_stage_intervals"]],
        )
        self.assertIn("measurement_1", result["missing_stages"])
        self.assertIn("measurement_2", result["missing_stages"])

    def test_rewiring_after_observed_measurement_recovers_missing_first_recording(self) -> None:
        events = [
            event("e1", "wiring_action", 0),
            event("e2", "measurement_action", 30),
            event("e3", "wiring_action", 60),
            event("e4", "writing_action", 90),
        ]
        result = assign_seven_stages(events, None)
        self.assertEqual(
            ["circuit_wiring", "measurement_1", "circuit_rewiring", "recording_2"],
            [item["stage"] for item in result["observed_stage_intervals"]],
        )
        self.assertTrue(any(reason.startswith("recording_1_not_observed") for reason in result["review_reasons"]))

    def test_local_conflict_repair_keeps_stronger_event_and_marks_repair(self) -> None:
        weaker = event("e1", "writing_action", 100, 160, 0.5)
        stronger = event("e2", "measurement_action", 120, 180, 0.9)
        kept, repairs, unresolved = resolve_accepted_conflicts([weaker, stronger])
        self.assertEqual(["e2"], [item["event_id"] for item in kept])
        self.assertEqual([], unresolved)
        self.assertEqual("e1", repairs[0]["event_id"])

    def test_equal_confidence_conflicts_are_quarantined(self) -> None:
        left = event("e1", "writing_action", 100, 160, 0.8)
        right = event("e2", "measurement_action", 120, 180, 0.8)
        kept, repairs, unresolved = resolve_accepted_conflicts([left, right])
        self.assertEqual([], kept)
        self.assertEqual([], repairs)
        self.assertEqual(["e1", "e2"], unresolved[0]["event_ids"])

    def test_relaxed_equal_confidence_conflict_keeps_more_specific_event(self) -> None:
        broad = event("e1", "writing_action", 100, 180, 0.8)
        specific = event("e2", "measurement_action", 120, 150, 0.8)
        kept, repairs, unresolved = resolve_accepted_conflicts(
            [broad, specific],
            preserve_equal_confidence=True,
        )
        self.assertEqual(["e2"], [item["event_id"] for item in kept])
        self.assertEqual([], unresolved)
        self.assertEqual("equal_confidence_tie_resolved_by_temporal_specificity", repairs[0]["reason"])

    def test_uncertain_event_does_not_suppress_direct_visible_action(self) -> None:
        visible = event("e1", "writing_action", 100, 160, 0.8)
        uncertain = event("e2", "uncertain", 100, 160, 0.95)
        kept, repairs, unresolved = resolve_accepted_conflicts([visible, uncertain])
        self.assertEqual({"e1", "e2"}, {item["event_id"] for item in kept})
        self.assertEqual([], repairs)
        self.assertEqual([], unresolved)

    def test_invalid_reduce_fallback_quarantines_all_candidates(self) -> None:
        candidates = [event("e1", "writing_action", 0), event("e2", "measurement_action", 30)]
        selected, metadata = select_events(candidates, None)
        self.assertEqual([], selected)
        self.assertEqual("local_fallback_quarantine_all", metadata["mode"])
        self.assertEqual(2, len(metadata["rejected_events"]))

    def test_cleanup_barrier_locks_terminal_without_keywords_and_ignores_later_actions(self) -> None:
        candidates = [
            event("e1", "wiring_action", 0, 30),
            event("e2", "cleanup_action", 60, 90),
            event("e3", "writing_action", 120, 150),
        ]
        response = {
            "accepted_event_ids": ["e1", "e2", "e3"],
            "rejected_events": [],
            "conflicts": [],
            "terminal_cleanup_event_id": "e2",
            "confidence": 0.9,
            "uncertainty": "",
        }
        repaired, repairs = salvage_reduce_response(candidates, response)
        self.assertIsNotNone(repaired)
        assert repaired is not None
        self.assertEqual(["e1", "e2"], repaired["accepted_event_ids"])
        self.assertEqual("e2", repaired["terminal_cleanup_event_id"])
        self.assertEqual(["e3"], [item["event_id"] for item in repaired["ignored_noise_events"]])
        self.assertTrue(any(item["reason"] == "Hard stop triggered by [cleanup_action]" for item in repairs))

    def test_cleanup_barrier_promotes_accepted_cleanup_when_reduce_returns_null(self) -> None:
        candidates = [
            event("e1", "wiring_action", 0, 30),
            event("e2", "cleanup_action", 60, 90),
            event("e3", "writing_action", 120, 150),
        ]
        response = {
            "accepted_event_ids": ["e1", "e2", "e3"],
            "rejected_events": [],
            "conflicts": [],
            "terminal_cleanup_event_id": None,
            "confidence": 0.9,
            "uncertainty": "",
        }
        repaired, repairs = salvage_reduce_response(candidates, response)
        self.assertIsNotNone(repaired)
        assert repaired is not None
        self.assertEqual(["e1", "e2"], repaired["accepted_event_ids"])
        self.assertEqual("e2", repaired["terminal_cleanup_event_id"])
        self.assertEqual(["e3"], [item["event_id"] for item in repaired["ignored_noise_events"]])
        self.assertTrue(any(item["reason"] == "terminal_cleanup_promoted_by_cleanup_barrier" for item in repairs))

    def test_hard_stop_keyword_detection_uses_explicit_evidence_text(self) -> None:
        positive_evidence = (
            "学生换座位",
            "画面出现人脸",
            "学生抬头闲聊",
            "两人聊天",
            "实验后换人",
            "器材已经整理完",
            "所有线路拆完，橙红色仪器放回桌子左上角",
            "导线收完，橙红色仪器移到左上角",
            "桌面已经清空",
        )
        for evidence in positive_evidence:
            with self.subTest(evidence=evidence):
                self.assertTrue(is_hard_stop_triggered({"evidence": evidence}))
        self.assertFalse(is_hard_stop_triggered({"evidence": "学生继续连接导线"}))
        self.assertFalse(is_hard_stop_triggered({"evidence": None}))
        negated_evidence = (
            "没有看到人脸",
            "未发生换座位",
            "无闲聊",
            "不是换人",
            "未见抬头或聊天",
            "尚未整理完毕",
            "还没全部拆完",
            "橙红色仪器还没有放回左上角",
            "桌面未清空",
        )
        for evidence in negated_evidence:
            with self.subTest(negated_evidence=evidence):
                self.assertFalse(is_hard_stop_triggered({"evidence": evidence}))
        self.assertEqual(
            ["整理完毕"],
            hard_stop_trigger_keywords({"evidence": "器材已经整理完毕"}),
        )

    def test_hard_stop_locks_cleanup_and_moves_later_events_to_noise(self) -> None:
        candidates = [
            event("e1", "wiring_action", 0, 30),
            {**event("e2", "cleanup_action", 60, 90), "evidence": "整理后学生抬头闲聊"},
            event("e3", "writing_action", 120, 150),
            {**event("e4", "wiring_action", 180, 210), "evidence": "换座位后碰到桌面导线"},
        ]
        response = {
            "accepted_event_ids": ["e1", "e2", "e3", "e4"],
            "rejected_events": [],
            "conflicts": [],
            "terminal_cleanup_event_id": "e2",
            "confidence": 0.9,
            "uncertainty": "",
        }
        repaired, repairs = salvage_reduce_response(candidates, response)
        self.assertIsNotNone(repaired)
        assert repaired is not None
        self.assertEqual(["e1", "e2"], repaired["accepted_event_ids"])
        self.assertEqual("e2", repaired["terminal_cleanup_event_id"])
        self.assertEqual({"e3", "e4"}, {item["event_id"] for item in repaired["ignored_noise_events"]})
        self.assertTrue(all(item["label"] == "ignored_noise_post_experiment" for item in repaired["ignored_noise_events"]))
        self.assertTrue(all(item["ignored_label"] == "ignored_noise_post_experiment" for item in repaired["ignored_noise_events"]))
        hard_stop_repair = next(item for item in repairs if item["reason"].startswith("Hard stop triggered by ["))
        self.assertEqual(["抬头", "闲聊", "换座位"], hard_stop_repair["trigger_keywords"])
        self.assertEqual("Hard stop triggered by [抬头, 闲聊, 换座位]", hard_stop_repair["reason"])

    def test_hard_stop_records_noise_even_when_qwen_already_rejected_later_event(self) -> None:
        candidates = [
            event("e1", "wiring_action", 0, 30),
            {**event("e2", "cleanup_action", 60, 90), "evidence": "整理后学生换座位"},
            event("e3", "writing_action", 120, 150),
        ]
        response = {
            "accepted_event_ids": ["e1", "e2"],
            "rejected_events": [{"event_id": "e3", "reason": "post_terminal_cleanup", "explanation": "实验后噪声"}],
            "conflicts": [],
            "terminal_cleanup_event_id": "e2",
            "confidence": 0.9,
            "uncertainty": "",
        }
        repaired, repairs = salvage_reduce_response(candidates, response)
        self.assertIsNotNone(repaired)
        assert repaired is not None
        self.assertEqual(["e3"], [item["event_id"] for item in repaired["ignored_noise_events"]])
        hard_stop_repair = next(item for item in repairs if item["reason"].startswith("Hard stop triggered by ["))
        self.assertEqual("Hard stop triggered by [换座位]", hard_stop_repair["reason"])

    def test_completed_cleanup_locks_terminal_and_reports_matched_keywords(self) -> None:
        candidates = [
            event("e1", "wiring_action", 0, 30),
            {
                **event("e2", "cleanup_action", 60, 90),
                "evidence": "所有导线已经拆完，橙红色仪器放回桌子的左上角",
            },
            event("e3", "writing_action", 120, 150),
        ]
        response = {
            "accepted_event_ids": ["e1", "e2", "e3"],
            "rejected_events": [],
            "conflicts": [],
            "terminal_cleanup_event_id": "e2",
            "confidence": 0.9,
            "uncertainty": "",
        }
        repaired, repairs = salvage_reduce_response(candidates, response)
        self.assertIsNotNone(repaired)
        assert repaired is not None
        self.assertEqual(["e1", "e2"], repaired["accepted_event_ids"])
        self.assertEqual("e2", repaired["terminal_cleanup_event_id"])
        self.assertEqual(["e3"], [item["event_id"] for item in repaired["ignored_noise_events"]])
        self.assertEqual(
            ["拆完", "放回桌子的左上角"],
            repaired["ignored_noise_events"][0]["hard_stop_trigger_keywords"],
        )
        hard_stop_repair = next(item for item in repairs if item["reason"].startswith("Hard stop triggered by ["))
        self.assertEqual(
            "Hard stop triggered by [拆完, 放回桌子的左上角]",
            hard_stop_repair["reason"],
        )
        self.assertFalse(is_hard_stop_triggered({"evidence": "学生将导线收进器材盒并将仪器归位"}))

    def test_relaxed_recovery_quarantines_only_omitted_event(self) -> None:
        candidates = [event("e1", "wiring_action", 0), event("e2", "measurement_action", 30)]
        response = {
            "accepted_event_ids": ["e1"],
            "rejected_events": [],
            "conflicts": [],
            "terminal_cleanup_event_id": None,
            "confidence": 0.8,
            "uncertainty": "",
        }
        repaired, repairs = salvage_reduce_response(candidates, response)
        self.assertIsNotNone(repaired)
        assert repaired is not None
        self.assertEqual(["e1"], repaired["accepted_event_ids"])
        self.assertEqual("e2", repaired["rejected_events"][0]["event_id"])
        self.assertTrue(any(item["reason"] == "omitted_event_locally_quarantined" for item in repairs))

    def test_repeated_writing_before_rewiring_stays_first_recording(self) -> None:
        events = [event("e1", "wiring_action", 0), event("e2", "writing_action", 30), event("e3", "writing_action", 60)]
        result = assign_seven_stages(events, None)
        self.assertEqual(["circuit_wiring", "recording_1", "recording_1"], [item["stage"] for item in result["observed_stage_intervals"]])
        self.assertNotIn("recording_2", [item["stage"] for item in result["observed_stage_intervals"]])

    def test_unconfirmed_cleanup_does_not_become_terminal(self) -> None:
        events = [event("e1", "wiring_action", 0), event("e2", "cleanup_action", 30), event("e3", "measurement_action", 60)]
        result = assign_seven_stages(events, None)
        self.assertFalse(result["analysis_termination"]["terminal_cleanup_reached"])
        self.assertEqual(3, len(result["assigned_events"]))
        self.assertTrue(any(reason.startswith("nonterminal_cleanup") for reason in result["review_reasons"]))

    def test_evidence_timeline_marks_unobserved_gaps_instead_of_inventing_stages(self) -> None:
        intervals = [
            {"event_id": "e1", "stage": "circuit_wiring", "start_seconds": 10.0, "end_seconds": 20.0},
            {"event_id": "e2", "stage": "recording_1", "start_seconds": 30.0, "end_seconds": 40.0},
        ]
        timeline, review = build_evidence_timeline(0.0, 50.0, intervals)
        self.assertEqual([], review)
        self.assertEqual(
            [(None, 0.0, 10.0), ("circuit_wiring", 10.0, 20.0), (None, 20.0, 30.0), ("recording_1", 30.0, 40.0), (None, 40.0, 50.0)],
            [(item["stage"], item["start_seconds"], item["end_seconds"]) for item in timeline],
        )

    def test_boundary_candidates_follow_observed_stage_changes(self) -> None:
        intervals = [
            {"event_id": "e1", "stage": "circuit_wiring", "start_frame_number": 0, "end_frame_number": 30, "start_frame_id": "f0", "end_frame_id": "f30", "start_seconds": 0.0, "end_seconds": 1.0},
            {"event_id": "e2", "stage": "recording_1", "start_frame_number": 60, "end_frame_number": 90, "start_frame_id": "f60", "end_frame_id": "f90", "start_seconds": 2.0, "end_seconds": 3.0},
        ]
        candidates = build_boundary_candidates(intervals)
        self.assertEqual(1, len(candidates))
        self.assertEqual("circuit_wiring", candidates[0]["from_stage"])
        self.assertEqual("recording_1", candidates[0]["to_stage"])
        self.assertEqual(2.0, candidates[0]["coarse_selected_seconds"])

    def test_stage_runs_merge_sampling_gaps_but_keep_long_unclassified_gap(self) -> None:
        intervals = [
            {"event_id": "e1", "stage": "circuit_wiring", "start_frame_number": 0, "end_frame_number": 30, "start_frame_id": "f0", "end_frame_id": "f30", "start_seconds": 0.0, "end_seconds": 1.0, "evidence": "a", "confidence": 0.7},
            {"event_id": "e2", "stage": "circuit_wiring", "start_frame_number": 90, "end_frame_number": 120, "start_frame_id": "f90", "end_frame_id": "f120", "start_seconds": 3.0, "end_seconds": 4.0, "evidence": "b", "confidence": 0.8},
            {"event_id": "e3", "stage": "circuit_wiring", "start_frame_number": 600, "end_frame_number": 630, "start_frame_id": "f600", "end_frame_id": "f630", "start_seconds": 20.0, "end_seconds": 21.0, "evidence": "c", "confidence": 0.9},
        ]
        runs = merge_observed_stage_runs(intervals, max_gap_seconds=4.1)
        self.assertEqual(2, len(runs))
        self.assertEqual(0.0, runs[0]["start_seconds"])
        self.assertEqual(4.0, runs[0]["end_seconds"])
        self.assertEqual(["e1", "e2"], runs[0]["event_ids"])


if __name__ == "__main__":
    unittest.main()
