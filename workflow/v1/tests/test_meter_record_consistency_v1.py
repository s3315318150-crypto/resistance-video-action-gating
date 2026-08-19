from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
MODULE_PATH = ROOT / "scripts" / "run_meter_record_consistency_v1.py"
SPEC = importlib.util.spec_from_file_location("meter_record_consistency", MODULE_PATH)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


class MeterRecordConsistencyTests(unittest.TestCase):
    def test_normalize_decimal(self) -> None:
        self.assertEqual(module.normalize_decimal("约 2.50"), 2.5)
        self.assertEqual(module.normalize_decimal(0.12), 0.12)
        self.assertIsNone(module.normalize_decimal("看不清"))

    def test_range_aware_tolerance(self) -> None:
        self.assertEqual(module.one_division_tolerance("voltmeter", 3), 0.125)
        self.assertEqual(module.one_division_tolerance("ammeter", 0.6), 0.025)

    def test_compare_value_is_binary(self) -> None:
        matched = module.compare_value("2.5", 2.55, "voltmeter", 3)
        self.assertTrue(matched["matched"])
        missing = module.compare_value(None, 0.2, "ammeter", 0.6)
        self.assertFalse(missing["matched"])

    def test_parse_fenced_json_and_validate(self) -> None:
        payload = {
            "per_frame": [],
            "consensus": {
                "ammeter": {"selected_range": "0.6", "value": "0.20", "confidence": 0.8},
                "voltmeter": {"selected_range": "3", "value": "2.5", "confidence": 0.9},
            },
            "evidence": "stable",
        }
        parsed = module.parse_model_json(
            "```json\n" + json.dumps(payload, ensure_ascii=False) + "\n```"
        )
        validated = module.validate_observation(parsed)
        self.assertEqual(validated["consensus"]["ammeter"]["value"], 0.2)
        self.assertEqual(validated["consensus"]["voltmeter"]["selected_range"], 3.0)

    def test_load_anonymous_specs(self) -> None:
        specs = module.load_video_specs(module.DEFAULT_SPEC_CONFIG)
        self.assertEqual("sample_001", specs[0].video_id)
        self.assertEqual((72.0, 72.5, 73.0, 73.5), specs[0].timestamps)

    def test_reject_absolute_video_path(self) -> None:
        absolute_video = str(Path(ROOT.anchor) / "private" / "sample.mp4")
        config = {
            "videos": [
                {
                    "video_id": "sample",
                    "source_video": absolute_video,
                    "timestamps_seconds": [1.0],
                    "meter_roi_normalized_xyxy": [0.0, 0.0, 1.0, 1.0],
                    "precision_timestamp_seconds": 1.0,
                    "precision_rois": {
                        "ammeter": [0.0, 0.0, 0.5, 1.0],
                        "voltmeter": [0.5, 0.0, 1.0, 1.0]
                    }
                }
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "spec.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source_video_invalid"):
                module.load_video_specs(path)


if __name__ == "__main__":
    unittest.main()
