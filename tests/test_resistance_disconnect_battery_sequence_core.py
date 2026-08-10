from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import resistance_disconnect_battery_sequence_core as core  # noqa: E402


def observation(timestamp: float, **values: object) -> dict[str, object]:
    return {"timestamp_seconds": timestamp, **values}


def passing_observations(after: tuple[str, str] = ("T0", "T1")) -> list[dict[str, object]]:
    return [
        observation(1.0, frame_id="f001", battery_terminals=["T0", "T2"], confidence=0.9),
        observation(2.0, frame_id="f002", switch_state="open", confidence=0.95),
        observation(
            3.0,
            frame_id="f003",
            battery_terminals=["T0"],
            terminal_state_stable=False,
            terminal_action="remove",
            direct_battery_contact=True,
            confidence=0.85,
        ),
        observation(4.0, frame_id="f004", battery_terminals=list(after), confidence=0.92),
        observation(5.0, frame_id="f005", switch_state="closed", confidence=0.94),
    ]


class TerminalTopologyTests(unittest.TestCase):
    def test_effective_count_depends_on_terminal_pair_not_physical_cell_count(self) -> None:
        self.assertEqual(2, core.effective_series_cells(["T0", "T2"]))
        self.assertEqual(1, core.effective_series_cells(["T0", "T1"]))
        self.assertEqual(1, core.effective_series_cells(["T1", "T2"]))
        self.assertIsNone(core.effective_series_cells(["T0"]))

    def test_terminal_aliases_are_normalized(self) -> None:
        self.assertEqual(2, core.effective_series_cells(["left_outer", "right outer"]))
        self.assertEqual(1, core.effective_series_cells(["middle_tap", "right_outer"]))

    def test_only_outer_to_middle_relocation_is_completed(self) -> None:
        relocation = core.classify_relocation(["T0", "T2"], ["T0", "T1"])
        self.assertTrue(relocation["completed"])
        self.assertEqual(2, relocation["effective_cells_before"])
        self.assertEqual(1, relocation["effective_cells_after"])
        self.assertEqual(
            {"from": "T2", "to": "T1", "fixed_terminal": "T0"},
            relocation["moved_lead"],
        )

        self.assertFalse(core.classify_relocation(["T0", "T2"], ["T0", "T2"])["completed"])
        self.assertFalse(core.classify_relocation(["T0", "T1"], ["T0", "T2"])["completed"])


class EpisodeStateMachineTests(unittest.TestCase):
    def evaluate(self, values: list[dict[str, object]]) -> dict[str, object]:
        return core.evaluate_episode({"episode_id": "ep1", "observations": values})

    def test_passes_open_relocate_close_sequence(self) -> None:
        result = self.evaluate(passing_observations())
        self.assertEqual("pass", result["decision"])
        self.assertEqual(1, result["predicted_score"])
        self.assertEqual("ordered_sequence_confirmed", result["reason_code"])
        chain = result["ordered_chain"]
        assert isinstance(chain, dict)
        self.assertEqual("open", chain["switch_open"]["state"])
        self.assertEqual(["T0", "T1"], chain["terminal_relocation"]["after_connection"])
        self.assertEqual("closed", chain["switch_close"]["state"])

    def test_passes_when_other_outer_lead_moves_to_middle(self) -> None:
        result = self.evaluate(passing_observations(("T1", "T2")))
        self.assertEqual("pass", result["decision"])
        moved = result["ordered_chain"]["terminal_relocation"]["moved_lead"]
        self.assertEqual({"from": "T0", "to": "T1", "fixed_terminal": "T2"}, moved)

    def test_physical_cell_count_is_irrelevant(self) -> None:
        values = passing_observations()
        for item in values:
            item["visible_physical_cell_count"] = 2
        self.assertEqual("pass", self.evaluate(values)["decision"])

    def test_direct_stable_before_after_transition_can_complete(self) -> None:
        values = [
            observation(1.0, battery_terminals=["T0", "T2"]),
            observation(2.0, switch_state="open"),
            observation(3.0, battery_terminals=["T0", "T1"]),
            observation(4.0, switch_state="closed"),
        ]
        self.assertEqual("pass", self.evaluate(values)["decision"])

    def test_same_pair_disconnect_and_reinsert_is_not_a_change(self) -> None:
        values = [
            observation(1.0, battery_terminals=["T0", "T2"]),
            observation(2.0, switch_state="open"),
            observation(3.0, battery_terminals=["T0"], terminal_state_stable=False),
            observation(4.0, battery_terminals=["T0", "T2"]),
            observation(5.0, switch_state="closed"),
        ]
        result = self.evaluate(values)
        self.assertEqual("fail", result["decision"])
        self.assertEqual("no_completed_two_to_one_relocation", result["reason_code"])

    def test_unstable_after_state_does_not_complete(self) -> None:
        values = passing_observations()
        values[3]["terminal_state_stable"] = False
        result = self.evaluate(values)
        self.assertEqual("fail", result["decision"])
        self.assertEqual("no_completed_two_to_one_relocation", result["reason_code"])

    def test_opening_after_relocation_is_wrong_order(self) -> None:
        values = [
            observation(1.0, battery_terminals=["T0", "T2"]),
            observation(2.0, battery_terminals=["T0", "T1"]),
            observation(3.0, switch_state="open"),
            observation(4.0, switch_state="closed"),
        ]
        result = self.evaluate(values)
        self.assertEqual("fail", result["decision"])
        self.assertEqual("switch_not_open_before_relocation", result["reason_code"])

    def test_closed_switch_during_relocation_fails_even_if_reopened(self) -> None:
        values = [
            observation(1.0, battery_terminals=["T0", "T2"]),
            observation(2.0, switch_state="open"),
            observation(3.0, battery_terminals=["T0"], terminal_state_stable=False),
            observation(3.5, switch_state="closed"),
            observation(3.7, switch_state="open"),
            observation(4.0, battery_terminals=["T0", "T1"]),
            observation(5.0, switch_state="closed"),
        ]
        result = self.evaluate(values)
        self.assertEqual("fail", result["decision"])
        self.assertEqual("switch_closed_during_relocation", result["reason_code"])

    def test_missing_reclosure_fails(self) -> None:
        result = self.evaluate(passing_observations()[:-1])
        self.assertEqual("fail", result["decision"])
        self.assertEqual("switch_not_closed_after_relocation", result["reason_code"])

    def test_close_at_same_timestamp_as_completion_is_not_after(self) -> None:
        values = passing_observations()
        values[-1]["timestamp_seconds"] = 4.0
        result = self.evaluate(values)
        self.assertEqual("fail", result["decision"])
        self.assertEqual("switch_closed_during_relocation", result["reason_code"])

    def test_explicit_reconnect_can_precede_later_stable_confirmation(self) -> None:
        values = [
            observation(1.0, frame_id="before", battery_terminals=["T0", "T2"]),
            observation(2.0, frame_id="open", switch_state="open"),
            observation(
                3.0,
                frame_id="start",
                terminal_action="relocate",
                direct_battery_contact=True,
            ),
            observation(
                4.0,
                frame_id="end",
                terminal_action="reconnect",
                terminal_rewire_completed=True,
            ),
            observation(5.0, frame_id="close", switch_state="closed"),
            observation(6.0, frame_id="after", battery_terminals=["T0", "T1"]),
        ]

        result = self.evaluate(values)

        self.assertEqual("pass", result["decision"])
        relocation = result["ordered_chain"]["terminal_relocation"]
        self.assertEqual(4.0, relocation["change_end_seconds"])
        self.assertEqual("explicit_reconnect_completion", relocation["completion_evidence"]["source"])

    def test_explicit_reconnect_without_stable_one_cell_state_still_fails(self) -> None:
        values = [
            observation(1.0, battery_terminals=["T0", "T2"]),
            observation(2.0, switch_state="open"),
            observation(3.0, terminal_action="relocate", direct_battery_contact=True),
            observation(4.0, terminal_rewire_completed=True),
            observation(5.0, switch_state="closed"),
        ]

        result = self.evaluate(values)

        self.assertEqual("fail", result["decision"])
        self.assertEqual("no_completed_two_to_one_relocation", result["reason_code"])

    def test_rejected_battery_object_cannot_complete_relocation(self) -> None:
        values = passing_observations()
        values[3]["battery_object"] = "rejected"
        result = self.evaluate(values)
        self.assertEqual("fail", result["decision"])
        self.assertEqual("battery_object_rejected", result["diagnostics"]["ignored_observations"][0]["reason"])

    def test_non_monotonic_timeline_is_rejected_but_remains_binary(self) -> None:
        values = passing_observations()
        values[3]["timestamp_seconds"] = 1.5
        result = self.evaluate(values)
        self.assertEqual("fail", result["decision"])
        self.assertEqual(0, result["predicted_score"])
        self.assertEqual("invalid_observation_timeline", result["reason_code"])

    def test_conflicting_same_time_switch_states_are_rejected(self) -> None:
        values = [
            observation(1.0, switch_state="open"),
            observation(1.0, switch_state="closed"),
        ]
        result = self.evaluate(values)
        self.assertEqual("fail", result["decision"])
        self.assertEqual("invalid_observation_timeline", result["reason_code"])

    def test_invalid_inputs_always_return_binary_result(self) -> None:
        for value in (None, [], {}, {"observations": "bad"}, {"observations": [{}]}):
            with self.subTest(value=value):
                result = core.evaluate_episode(value)
                self.assertIn(result["decision"], {"pass", "fail"})
                self.assertIn(result["predicted_score"], {0, 1})


class EpisodeAggregationTests(unittest.TestCase):
    def test_events_from_different_episodes_are_never_fused(self) -> None:
        first = {
            "episode_id": "open_and_change",
            "observations": passing_observations()[:-1],
        }
        second = {
            "episode_id": "close_only",
            "observations": [observation(10.0, switch_state="closed")],
        }
        result = core.aggregate_episodes([first, second], video_id="8")
        self.assertEqual("fail", result["decision"])
        self.assertFalse(result["diagnostics"]["cross_episode_evidence_fusion"])
        self.assertEqual([], result["passing_episode_ids"])

    def test_any_independently_complete_episode_can_pass_video(self) -> None:
        failed = {"episode_id": "bad", "observations": []}
        passed = {"episode_id": "good", "observations": passing_observations()}
        result = core.aggregate_episodes([failed, passed], video_id="8")
        self.assertEqual("pass", result["decision"])
        self.assertEqual(1, result["predicted_score"])
        self.assertEqual(["good"], result["passing_episode_ids"])


if __name__ == "__main__":
    unittest.main()
