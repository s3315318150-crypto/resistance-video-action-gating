from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "generate_wiring_sequence_config.py"
SPEC = importlib.util.spec_from_file_location("generate_wiring_sequence_config", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class GenerateWiringSequenceConfigTests(unittest.TestCase):
    def test_arbitrary_filename_produces_stable_id(self) -> None:
        self.assertEqual("class_bench_07", MODULE.safe_id("class bench 07.mp4"))
        self.assertEqual("42", MODULE.safe_id("42_sample.mp4"))

    def test_each_wiring_stage_becomes_an_independent_episode(self) -> None:
        stages = [
            {"stage": "circuit_wiring", "start": 10.0, "end": 30.0},
            {"stage": "measurement_1", "start": 31.0, "end": 38.0},
            {"stage": "recording_1", "start": 38.0, "end": 44.0},
            {"stage": "circuit_rewiring", "start": 45.0, "end": 51.0},
            {"stage": "measurement_2", "start": 52.0, "end": 58.0},
        ]

        episodes = MODULE.episode_windows(stages, duration=60.0)

        self.assertEqual(2, len(episodes))
        self.assertEqual([31.0, 38.0], episodes[0]["recording_window_seconds"])
        self.assertEqual("measurement_1", episodes[0]["stable_window_source"])
        self.assertEqual([52.0, 58.0], episodes[1]["recording_window_seconds"])
        self.assertEqual("measurement_2", episodes[1]["stable_window_source"])

    def test_missing_following_stage_uses_bounded_fallback(self) -> None:
        episodes = MODULE.episode_windows(
            [{"stage": "circuit_wiring", "start": 4.0, "end": 12.0}],
            duration=22.0,
        )

        self.assertEqual([12.0, 22.0], episodes[0]["recording_window_seconds"])
        self.assertEqual("post_wiring_fallback", episodes[0]["stable_window_source"])

    def test_candidate_sampling_is_bounded_and_keeps_endpoints(self) -> None:
        values = MODULE.candidate_timestamps(10.0, 40.0, interval=1.0, limit=6)

        self.assertEqual(6, len(values))
        self.assertEqual(10.0, values[0])
        self.assertEqual(40.0, values[-1])


if __name__ == "__main__":
    unittest.main()
