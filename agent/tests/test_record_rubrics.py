from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


AGENT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AGENT_ROOT / "resistance_agent"))

import record_rubrics  # noqa: E402
import toolkit  # noqa: E402


class RecordRubricGateTests(unittest.TestCase):
    @staticmethod
    def result(decision: str = "pass") -> dict:
        return {
            "decision": decision,
            "predicted_score": 1 if decision == "pass" else 0,
            "confidence": 0.9,
            "reason": "same_cycle_match" if decision == "pass" else "same_cycle_mismatch",
            "diagnostics": {},
        }

    def test_all_prerequisites_pass_keeps_record_result(self) -> None:
        prerequisites = {str(item): {"decision": "pass"} for item in (4, 5, 6)}
        result = record_rubrics.apply_meter_prerequisite_gate(self.result(), 1, prerequisites)
        self.assertEqual("pass", result["decision"])
        self.assertEqual([], result["diagnostics"]["meter_prerequisite_gate"]["failed_items"])

    def test_each_failed_prerequisite_forces_binary_fail(self) -> None:
        labels = {4: "polarity_terminals", 5: "normal_pointer_deflection", 6: "suitable_meter_range"}
        for failed_id, label in labels.items():
            with self.subTest(failed_id=failed_id):
                prerequisites = {
                    str(item): {"decision": "fail" if item == failed_id else "pass"}
                    for item in (4, 5, 6)
                }
                result = record_rubrics.apply_meter_prerequisite_gate(self.result(), 2, prerequisites)
                self.assertEqual("fail", result["decision"])
                self.assertEqual(0, result["predicted_score"])
                self.assertEqual(f"meter_prerequisite_failed:{label}", result["reason"])

    def test_record_mismatch_stays_fail_when_prerequisites_pass(self) -> None:
        prerequisites = {str(item): {"decision": "pass"} for item in (4, 5, 6)}
        result = record_rubrics.apply_meter_prerequisite_gate(self.result("fail"), 1, prerequisites)
        self.assertEqual("fail", result["decision"])

    def test_bundle_adds_r4_r5_r6_dependencies_for_r7_r9(self) -> None:
        plan = toolkit._rubric_bundle_plan([7, 9])
        self.assertEqual([4, 5, 6], plan["dependency_rubric_ids"])
        self.assertEqual(
            ["run_meter_rubrics", "run_polarity_rubric", "run_record_rubrics"],
            [item["tool"] for item in plan["producer_plan"]],
        )

    def test_live_stage_reader_rejects_replay_result(self) -> None:
        with tempfile.TemporaryDirectory(dir=AGENT_ROOT) as temporary:
            root = Path(temporary)
            summary = {
                "records": [
                    {"source_video_id": "current.mp4", "replay_result": str(root / "old.json")}
                ]
            }
            with self.assertRaisesRegex(ValueError, "replay_result is forbidden"):
                record_rubrics._source_record(summary, "current.mp4", "current", root)


if __name__ == "__main__":
    unittest.main()
