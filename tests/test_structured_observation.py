from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))


def load_script(name: str):
    path = SCRIPTS / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


observer = load_script("run_qwen_structured_observation.py")


class StructuredObservationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = {
            "rubric_id": "resistance.example",
            "observation_instruction": "Observe a visible relation.",
            "observation_enum": ["supported", "contradicted", "uncertain"],
        }

    def test_valid_response(self) -> None:
        value = {
            "rubric_id": "resistance.example",
            "observation": "supported",
            "cited_candidate_ids": ["candidate_001"],
            "confidence": 0.8,
            "evidence": "visible relation",
            "uncertainty": "",
        }
        self.assertEqual(observer.validate_response(value, self.config, ["candidate_001"]), [])

    def test_scoring_field_is_rejected(self) -> None:
        value = {
            "rubric_id": "resistance.example",
            "observation": "supported",
            "cited_candidate_ids": [],
            "confidence": 0.8,
            "evidence": "visible relation",
            "uncertainty": "",
            "predicted_score": 1,
        }
        self.assertIn("forbidden_scoring_field", observer.validate_response(value, self.config, []))


if __name__ == "__main__":
    unittest.main()
