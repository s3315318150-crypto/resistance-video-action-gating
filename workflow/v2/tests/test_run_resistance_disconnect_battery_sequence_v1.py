from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import run_resistance_disconnect_battery_sequence_v1 as runner  # noqa: E402
from resistance_disconnect_battery_sequence_core import aggregate_episodes  # noqa: E402


def make_records() -> list[runner.FrameRecord]:
    """Create timestamped records without requiring image files on disk."""
    values = [
        ("before", 95.0),
        ("open", 99.0),
        ("start", 101.0),
        ("contact", 102.0),
        ("end", 103.0),
        ("after", 104.0),
        ("close", 105.0),
    ]
    return [
        runner.FrameRecord(
            frame_id=frame_id,
            frame_number=index,
            timestamp_seconds=timestamp,
            panorama_path="",
            battery_path="",
            sharpness=1.0,
            motion=0.0,
            battery_motion=0.0,
        )
        for index, (frame_id, timestamp) in enumerate(values)
    ]


def valid_summary(contact_ids: list[str] | None = None) -> dict[str, object]:
    return {
        "battery_object": "confirmed",
        "direct_contact_frame_ids": contact_ids if contact_ids is not None else ["contact"],
        "battery_before": {
            "frame_id": "before",
            "terminals": ["T0", "T2"],
            "stable": True,
        },
        "battery_after": {
            "frame_id": "after",
            "terminals": ["T0", "T1"],
            "stable": True,
        },
        "terminal_rewire": {
            "start_frame_id": "start",
            "end_frame_id": "end",
            "from_terminal": "T2",
            "to_terminal": "T1",
            "completed": True,
        },
        "switch": {
            "open_before_frame_id": "open",
            "closed_after_frame_id": "close",
            "closed_during_frame_ids": [],
        },
    }


def episode_metadata() -> dict[str, object]:
    return {"episode_id": "8_rewire_01", "core_interval_seconds": [100.0, 104.0]}


class StructuredSummaryValidationTests(unittest.TestCase):
    def test_valid_summary_satisfies_topology_order_and_contact_contract(self) -> None:
        errors = runner.validate_structured_summary(
            valid_summary(), make_records(), episode_metadata()
        )
        self.assertEqual([], errors)

    def test_invalid_summary_reports_semantic_contract_errors(self) -> None:
        summary = valid_summary(contact_ids=[])
        summary["battery_object"] = "rejected"
        summary["battery_after"] = {
            "frame_id": "after",
            "terminals": ["T0", "T2"],
            "stable": False,
        }
        summary["terminal_rewire"] = {
            "start_frame_id": "start",
            "end_frame_id": "end",
            "from_terminal": "T0",
            "to_terminal": "T1",
            "completed": False,
        }
        errors = runner.validate_structured_summary(
            summary, make_records(), episode_metadata()
        )
        self.assertIn("battery_object_not_confirmed", errors)
        self.assertIn("direct_battery_contact_missing", errors)
        self.assertIn("after_not_stable", errors)
        self.assertIn("rewire_not_completed", errors)
        self.assertIn("terminal_pair_not_two_to_one", errors)

    def test_contact_must_be_inside_the_cited_rewire_interval(self) -> None:
        errors = runner.validate_structured_summary(
            valid_summary(contact_ids=["open"]), make_records(), episode_metadata()
        )
        self.assertIn("no_direct_contact_during_rewire", errors)

    def test_completed_rewire_cannot_contradict_unchanged_terminal_pairs(self) -> None:
        summary = valid_summary()
        summary["battery_after"] = {
            "frame_id": "after",
            "terminals": ["T0", "T2"],
            "stable": True,
        }

        errors = runner.validate_structured_summary(
            summary, make_records(), episode_metadata()
        )

        self.assertIn("terminal_pair_not_two_to_one", errors)
        self.assertIn("rewire_claim_conflicts_with_terminal_pairs", errors)


class DirectContactGateTests(unittest.TestCase):
    def test_missing_direct_contact_cannot_create_terminal_observations(self) -> None:
        records = make_records()
        summary = valid_summary(contact_ids=[])
        observations, conversion_errors = runner.structured_summary_to_observations(
            summary, records
        )

        self.assertEqual([], conversion_errors)
        by_id = {item["frame_id"]: item for item in observations}
        self.assertFalse(by_id["start"]["direct_battery_contact"])
        self.assertIsNone(by_id["before"]["battery_terminals"])
        self.assertIsNone(by_id["after"]["battery_terminals"])
        self.assertFalse(by_id["before"]["terminal_state_stable"])

        reduced = aggregate_episodes(
            [{"episode_id": "8_rewire_01", "observations": observations}],
            video_id="8",
        )
        self.assertEqual("fail", reduced["decision"])
        self.assertEqual(0, reduced["predicted_score"])

    def test_rejected_object_is_also_blocked_even_when_contact_frame_is_cited(self) -> None:
        records = make_records()
        summary = valid_summary(contact_ids=["contact"])
        summary["battery_object"] = "rejected"
        observations, _ = runner.structured_summary_to_observations(summary, records)

        by_id = {item["frame_id"]: item for item in observations}
        self.assertIsNone(by_id["before"]["battery_terminals"])
        self.assertIsNone(by_id["after"]["battery_terminals"])
        self.assertEqual("rejected", by_id["contact"]["battery_object"])

    def test_contact_outside_rewire_interval_cannot_authorize_terminal_pairs(self) -> None:
        records = make_records()
        summary = valid_summary(contact_ids=["open"])
        observations, conversion_errors = runner.structured_summary_to_observations(
            summary, records
        )

        self.assertEqual([], conversion_errors)
        by_id = {item["frame_id"]: item for item in observations}
        self.assertIsNone(by_id["before"]["battery_terminals"])
        self.assertIsNone(by_id["after"]["battery_terminals"])
        self.assertFalse(by_id["open"]["direct_battery_contact"])

    def test_cached_summary_with_only_contact_errors_uses_independent_verifier(self) -> None:
        summary = valid_summary(contact_ids=[])
        with TemporaryDirectory() as directory:
            output_dir = Path(directory)
            runner.write_json(
                output_dir / "structured_summary.json",
                {"parsed": summary, "error": None, "raw": "cached"},
            )
            with patch.object(runner, "paired_image", return_value=output_dir / "unused.jpg"):
                parsed, error, raw = runner.run_structured_summary(
                    None,
                    "qwen",
                    make_records(),
                    episode_metadata(),
                    output_dir,
                    direct_contact_repair_available=True,
                )

        self.assertEqual(summary, parsed)
        self.assertIsNone(error)
        self.assertEqual("cached", raw)


class TerminalPairVerifierTests(unittest.TestCase):
    def test_selects_stable_context_on_both_sides_of_core(self) -> None:
        selected = runner.select_terminal_pair_records(make_records(), episode_metadata())

        self.assertEqual(["before", "open", "close"], [item.frame_id for item in selected])

    def test_focused_pair_response_confirms_two_to_one_transition(self) -> None:
        response = {
            "battery_object": "confirmed",
            "before": {"frame_id": "open", "terminals": ["T0", "T2"], "stable": True, "evidence": "outer pair"},
            "after": {"frame_id": "close", "terminals": ["T0", "T1"], "stable": True, "evidence": "middle pair"},
        }

        normalized, errors = runner.normalize_terminal_pair_verifier_response(
            response, make_records(), episode_metadata()
        )

        self.assertEqual([], errors)
        self.assertTrue(normalized["usable"])
        self.assertTrue(normalized["transition"]["completed"])

    def test_focused_pair_response_repairs_only_a_confirmed_transition(self) -> None:
        summary = valid_summary()
        summary["battery_after"] = {"frame_id": "after", "terminals": ["T0", "T2"], "stable": True}
        verifier = {
            "usable": True,
            "before": {"frame_id": "open", "terminals": ["T0", "T2"], "stable": True},
            "after": {"frame_id": "close", "terminals": ["T0", "T1"], "stable": True},
            "transition": runner.classify_relocation(["T0", "T2"], ["T0", "T1"]),
        }

        merged, applied = runner.merge_terminal_pair_summary(summary, verifier)

        self.assertTrue(applied)
        self.assertEqual(["T0", "T1"], merged["battery_after"]["terminals"])
        self.assertEqual("T2", merged["terminal_rewire"]["from_terminal"])
        self.assertEqual("T1", merged["terminal_rewire"]["to_terminal"])

    def test_structured_completion_frame_is_separate_from_later_stable_confirmation(self) -> None:
        observations, errors = runner.structured_summary_to_observations(
            valid_summary(), make_records()
        )

        self.assertEqual([], errors)
        by_id = {item["frame_id"]: item for item in observations}
        self.assertTrue(by_id["end"]["terminal_rewire_completed"])
        self.assertTrue(by_id["after"]["terminal_state_stable"])
        self.assertFalse(by_id["after"]["terminal_rewire_completed"])


class ProcessedEpisodeAggregationTests(unittest.TestCase):
    def test_aggregation_uses_top_level_decisions_without_reducing_nested_episode(self) -> None:
        passing_reducer = {
            "decision": "pass",
            "predicted_score": 1,
            "confidence": 0.81,
            "episodes": [{"decision": "fail", "episode_id": "nested"}],
        }
        failed_reducer = {
            "decision": "fail",
            "predicted_score": 0,
            "confidence": 0.92,
            "episodes": [{"decision": "pass", "episode_id": "should_not_leak"}],
        }
        result = runner.aggregate_processed_episodes(
            [
                {"episode_id": "ep_pass", "decision": "pass", "confidence": 0.81, "reducer": passing_reducer},
                {"episode_id": "ep_fail", "decision": "fail", "confidence": 0.92, "reducer": failed_reducer},
            ],
            video_id="8",
        )

        self.assertEqual("pass", result["decision"])
        self.assertEqual(1, result["predicted_score"])
        self.assertEqual(["ep_pass"], result["passing_episode_ids"])
        self.assertEqual("processed_episode_results_only", result["diagnostics"]["aggregation"])
        self.assertFalse(result["diagnostics"]["cross_episode_evidence_fusion"])
        self.assertEqual([passing_reducer, failed_reducer], result["episodes"])

    def test_no_top_level_pass_cannot_be_resurrected_by_nested_reducer(self) -> None:
        result = runner.aggregate_processed_episodes(
            [
                {
                    "episode_id": "ep_fail",
                    "decision": "fail",
                    "confidence": 0.7,
                    "reducer": {"decision": "pass", "episodes": [{"decision": "pass"}]},
                }
            ],
            video_id="38",
        )
        self.assertEqual("fail", result["decision"])
        self.assertEqual(0, result["predicted_score"])
        self.assertEqual([], result["passing_episode_ids"])


class EpisodeWindowTests(unittest.TestCase):
    def test_default_window_is_interval_plus_or_minus_ten_seconds(self) -> None:
        episode = runner.make_episode(
            "8",
            1,
            "video.mp4",
            "result.json",
            [100.0, 110.0],
            {"material_cleanup": []},
            duration=200.0,
        )
        self.assertEqual([90.0, 120.0], episode["expanded_interval_seconds"])
        self.assertEqual([100.0, 110.0], episode["core_interval_seconds"])

    def test_window_is_clipped_to_video_duration_and_earliest_cleanup(self) -> None:
        episode = runner.make_episode(
            "8",
            2,
            "video.mp4",
            "result.json",
            [2.0, 9.0],
            {"material_cleanup": [[12.0, 20.0], [15.0, 18.0]]},
            duration=14.0,
        )
        self.assertEqual([0.0, 12.0], episode["expanded_interval_seconds"])
        self.assertEqual(12.0, episode["cleanup_cutoff_seconds"])

    def test_cleanup_before_interval_end_does_not_make_window_end_before_start(self) -> None:
        episode = runner.make_episode(
            "8",
            3,
            "video.mp4",
            "result.json",
            [50.0, 80.0],
            {"material_cleanup": [[45.0, 47.0]]},
            duration=100.0,
        )
        self.assertEqual([40.0, 45.0], episode["expanded_interval_seconds"])


class RecoveryCandidateTests(unittest.TestCase):
    def test_repeated_first_recording_gap_becomes_independent_recovery_episode(self) -> None:
        candidates = runner.episode_candidates(
            {
                "recording_1": [[412.0, 422.0], [432.0, 452.0]],
                "circuit_rewiring": [[454.0, 476.0]],
            }
        )

        self.assertEqual(2, len(candidates))
        recovery = candidates[0]
        self.assertEqual([422.0, 432.0], recovery["interval"])
        self.assertEqual("recovery", recovery["episode_kind"])
        self.assertEqual("repeated_stage_gap_recovery", recovery["candidate_source"])
        self.assertEqual("recording_1", recovery["recovery_anchor"]["stage"])
        self.assertEqual([454.0, 476.0], candidates[1]["interval"])

    def test_recovery_gap_overlapping_a_supplied_rewire_is_not_duplicated(self) -> None:
        candidates = runner.episode_candidates(
            {
                "recording_1": [[100.0, 110.0], [118.0, 128.0]],
                "circuit_rewiring": [[111.0, 120.0]],
            }
        )

        self.assertEqual(1, len(candidates))
        self.assertEqual("stage_circuit_rewiring", candidates[0]["candidate_source"])

    def test_recovery_gap_has_a_bounded_duration(self) -> None:
        candidates = runner.episode_candidates(
            {"measurement_1": [[10.0, 20.0], [70.0, 80.0]]}
        )

        self.assertEqual([], candidates)

    def test_wiring_transition_mode_uses_initial_wiring_intervals(self) -> None:
        candidates = runner.episode_candidates(
            {
                "circuit_wiring": [[12.0, 18.0]],
                "circuit_rewiring": [[30.0, 36.0]],
            },
            time_mode="wiring_transition",
        )

        self.assertEqual(1, len(candidates))
        self.assertEqual([12.0, 18.0], candidates[0]["interval"])
        self.assertEqual("wiring", candidates[0]["episode_kind"])
        self.assertEqual("stage_circuit_wiring", candidates[0]["candidate_source"])

    def test_broad_transition_mode_merges_stage_sources_without_overlap_duplicates(self) -> None:
        candidates = runner.episode_candidates(
            {
                "circuit_wiring": [[10.0, 18.0]],
                "circuit_rewiring": [[30.0, 36.0]],
                "recording_1": [[40.0, 45.0], [52.0, 60.0]],
            },
            time_mode="broad_transition_search",
        )

        self.assertEqual(3, len(candidates))
        self.assertEqual(
            ["stage_circuit_wiring", "stage_circuit_rewiring", "repeated_stage_gap_recovery"],
            [item["candidate_source"] for item in candidates],
        )
        self.assertEqual([45.0, 52.0], candidates[2]["interval"])

    def test_unknown_time_mode_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            runner.episode_candidates({}, time_mode="video_8")


class DynamicBatteryRoiTests(unittest.TestCase):
    def test_qwen_timeout_becomes_an_unavailable_packet(self) -> None:
        class Completions:
            @staticmethod
            def create(**_: object) -> object:
                raise TimeoutError("fixture timeout")

        class Client:
            class Chat:
                completions = Completions()

            chat = Chat()

        parsed, error, raw = runner.qwen_call(Client(), "qwen", "prompt", [])

        self.assertIsNone(parsed)
        self.assertIn("TimeoutError", error or "")
        self.assertEqual("", raw)

    def test_live_detections_are_frame_bound_and_confidence_filtered(self) -> None:
        detections = runner.normalize_dynamic_battery_detections(
            [
                {"frame_id": "f1", "bbox_normalized": [0.1, 0.2, 0.5, 0.6], "confidence": 0.8},
                {"frame_id": "f1", "bbox_normalized": [0.2, 0.2, 0.4, 0.5], "confidence": 0.6},
                {"frame_id": "f2", "bbox_normalized": [0.1, 0.2, 0.5, 0.6], "confidence": 0.2},
                {"frame_id": "other", "bbox_normalized": [0.1, 0.2, 0.5, 0.6], "confidence": 0.9},
            ],
            {"f1", "f2"},
            minimum_confidence=0.45,
        )

        self.assertEqual(1, len(detections))
        self.assertEqual("f1", detections[0]["frame_id"])
        self.assertEqual([0.1, 0.2, 0.5, 0.6], detections[0]["bbox_normalized"])
        self.assertEqual("qwen_live_panorama", detections[0]["source"])

    def test_invalid_geometry_cannot_seed_tracking(self) -> None:
        detections = runner.normalize_dynamic_battery_detections(
            [
                {"frame_id": "f1", "bbox_normalized": [0.5, 0.2, 0.1, 0.6], "confidence": 0.9},
                {"frame_id": "f2", "bbox_normalized": [0.0, 0.0, 1.0, 1.0], "confidence": 0.9},
            ],
            {"f1", "f2"},
        )

        self.assertEqual([], detections)

    def test_nearest_live_detection_seeds_dense_range_without_video_lookup(self) -> None:
        records = [
            runner.FrameRecord("arbitrary_a", 1, 10.0, "", "", 1.0, 0.0, 0.0),
            runner.FrameRecord("arbitrary_b", 2, 20.0, "", "", 1.0, 0.0, 0.0),
        ]
        roi, source_frame_id = runner.nearest_dynamic_roi(
            records,
            [
                {"frame_id": "arbitrary_a", "bbox_normalized": [0.1, 0.2, 0.4, 0.6]},
                {"frame_id": "arbitrary_b", "bbox_normalized": [0.5, 0.3, 0.9, 0.8]},
            ],
            18.0,
        )

        self.assertEqual((0.5, 0.3, 0.9, 0.8), roi)
        self.assertEqual("arbitrary_b", source_frame_id)

    def test_video_identifier_is_not_limited_to_the_original_five(self) -> None:
        self.assertEqual("41", runner.video_id_from_name("41_new_resistance_video.mp4"))

    def test_anonymous_filename_is_valid_for_current_run_association(self) -> None:
        self.assertEqual(
            "anonymous-input",
            runner.video_id_from_name("anonymous-input.mp4"),
        )

    def test_source_discovery_uses_exact_filename_under_explicit_root(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            expected = root / "anonymous-input.mp4"
            expected.touch()

            self.assertEqual(
                expected.resolve(),
                runner.discover_video(expected.name, root),
            )

    def test_coarse_screening_packet_is_bounded_and_keeps_core_motion(self) -> None:
        records = [
            runner.FrameRecord(
                f"f{index}",
                index,
                float(index),
                "",
                "",
                1.0,
                100.0 if index == 50 else 0.0,
                0.0,
            )
            for index in range(60)
        ]

        selected = runner.select_screening_records(
            records,
            {"core_interval_seconds": [40.0, 55.0]},
            maximum=24,
        )

        self.assertEqual(24, len(selected))
        self.assertIn("f50", [record.frame_id for record in selected])
        self.assertEqual(sorted(record.timestamp_seconds for record in selected), [record.timestamp_seconds for record in selected])


if __name__ == "__main__":
    unittest.main()
