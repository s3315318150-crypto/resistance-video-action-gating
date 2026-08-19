from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


merge = load_script("qwen_experiment_action_minute_merge.py")


class ActionMinuteMergeTests(unittest.TestCase):
    def test_invalid_minute_observations_are_ignored(self) -> None:
        record = {
            "source_video_id": "sample.mp4",
            "fixed_experiment_window_seconds": [0.0, 60.0],
            "minute_results": [
                {
                    "minute_index": 0,
                    "valid": False,
                    "input_frames": [{"image_id": "bad", "timestamp_seconds": 10.0}],
                    "observations": [
                        {
                            "stage": "measurement_1",
                            "evidence_frame_id": "bad",
                            "evidence": "invalid response residue",
                        }
                    ],
                }
            ],
        }
        result = merge.merge_record(record)
        self.assertTrue(result["evidence_insufficient"])
        self.assertEqual(result["segments"], [])

    def test_valid_events_are_merged_in_time_order(self) -> None:
        record = {
            "source_video_id": "sample.mp4",
            "fixed_experiment_window_seconds": [0.0, 60.0],
            "minute_results": [
                {
                    "minute_index": 0,
                    "valid": True,
                    "input_frames": [
                        {"image_id": "a", "timestamp_seconds": 5.0},
                        {"image_id": "b", "timestamp_seconds": 25.0},
                    ],
                    "observations": [
                        {"stage": "circuit_wiring", "evidence_frame_id": "a", "evidence": "wiring"},
                        {"stage": "measurement_1", "evidence_frame_id": "b", "evidence": "meter"},
                    ],
                }
            ],
        }
        result = merge.merge_record(record)
        self.assertFalse(result["evidence_insufficient"])
        self.assertEqual([item["stage"] for item in result["segments"]], ["circuit_wiring", "measurement_1"])
        self.assertEqual(result["segments"][0]["end_seconds"], 25.0)


if __name__ == "__main__":
    unittest.main()
