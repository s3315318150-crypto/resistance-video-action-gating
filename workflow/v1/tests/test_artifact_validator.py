from __future__ import annotations

import importlib.util
import json
import tempfile
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


validator = load_script("validate_evidence_artifacts.py")


class ArtifactValidatorTests(unittest.TestCase):
    def validate(self, artifact: dict) -> dict:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "artifact.json"
            path.write_text(json.dumps(artifact), encoding="utf-8")
            report = validator.Report("evidence_package", path)
            validator.validate_artifact(artifact, "evidence_package", path, report)
            return report.as_dict()

    def test_abstained_package_has_null_score_and_reason(self) -> None:
        report = self.validate(
            {
                "rubric_id": "resistance.example",
                "source_video": "sample.mp4",
                "automated_decision": "abstained",
                "automated_outcome": "abstained",
                "abstention_reason": "evidence_insufficient",
                "predicted_score": None,
            }
        )
        self.assertTrue(report["valid"], report["errors"])

    def test_abstained_package_with_score_is_invalid(self) -> None:
        report = self.validate(
            {
                "rubric_id": "resistance.example",
                "source_video": "sample.mp4",
                "automated_decision": "abstained",
                "automated_outcome": "abstained",
                "abstention_reason": "evidence_insufficient",
                "predicted_score": 0,
            }
        )
        self.assertFalse(report["valid"])
        self.assertIn("ABSTAINED_WITH_SCORE", {item["code"] for item in report["errors"]})


if __name__ == "__main__":
    unittest.main()
