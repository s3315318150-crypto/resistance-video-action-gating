from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import qwen_experiment_action_hierarchical_v1 as engine  # noqa: E402
import qwen_experiment_action_hierarchical_v2_five_stage as five  # noqa: E402
import qwen_hierarchical_v1_contract as contract  # noqa: E402
import qwen_hierarchical_v1_reduce as reduce_engine  # noqa: E402


def event(index: int, action: str) -> dict[str, object]:
    first = index * 10
    return {
        "event_id": f"e{index}",
        "action_type": action,
        "representative_frame_number": first + 5,
        "first_frame_number": first,
        "last_frame_number": first + 8,
        "first_frame_id": f"frame_{first:08d}",
        "last_frame_id": f"frame_{first + 8:08d}",
        "first_seconds": float(first),
        "last_seconds": float(first + 8),
        "evidence": action,
        "confidence": 0.9,
    }


class HierarchicalFiveStageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.originals = {
            "contract_stage_schema_id": contract.STAGE_SCHEMA_ID,
            "contract_stages": contract.STAGES,
            "engine_stage_schema_id": engine.STAGE_SCHEMA_ID,
            "engine_algorithm_id": engine.ALGORITHM_ID,
            "engine_algorithm_schema_version": engine.ALGORITHM_SCHEMA_VERSION,
            "engine_default_schema": engine.DEFAULT_SCHEMA,
            "engine_default_output_root": engine.DEFAULT_OUTPUT_ROOT,
            "engine_assign": engine.assign_seven_stages,
            "engine_merge": engine.merge_observed_stage_runs,
        }

    def tearDown(self) -> None:
        contract.STAGE_SCHEMA_ID = self.originals["contract_stage_schema_id"]
        contract.STAGES = self.originals["contract_stages"]
        engine.STAGE_SCHEMA_ID = self.originals["engine_stage_schema_id"]
        engine.ALGORITHM_ID = self.originals["engine_algorithm_id"]
        engine.ALGORITHM_SCHEMA_VERSION = self.originals["engine_algorithm_schema_version"]
        engine.DEFAULT_SCHEMA = self.originals["engine_default_schema"]
        engine.DEFAULT_OUTPUT_ROOT = self.originals["engine_default_output_root"]
        engine.assign_seven_stages = self.originals["engine_assign"]
        engine.merge_observed_stage_runs = self.originals["engine_merge"]

    def test_schema_binds_five_stages_without_removing_v2_entrypoint(self) -> None:
        five.bind_five_stage_identity()
        schema = contract.load_stage_schema(five.DEFAULT_SCHEMA)
        self.assertEqual(five.STAGE_SCHEMA_ID, schema["stage_schema_id"])
        self.assertEqual(list(five.STAGES), [item["id"] for item in schema["stages"]])
        self.assertTrue((SCRIPTS / "qwen_experiment_action_hierarchical_v2.py").is_file())

    def test_measurement_and_writing_share_one_cycle_but_keep_subactions(self) -> None:
        actions = [
            "wiring_action",
            "measurement_action",
            "writing_action",
            "wiring_action",
            "measurement_action",
            "writing_action",
            "cleanup_action",
        ]
        state = reduce_engine.assign_five_stages(
            [event(index, action) for index, action in enumerate(actions)],
            "e6",
        )
        self.assertEqual([], state["missing_stages"])
        self.assertEqual(
            [
                "circuit_wiring",
                "recording_1",
                "recording_1",
                "circuit_rewiring",
                "recording_2",
                "recording_2",
                "material_cleanup",
            ],
            [item["stage"] for item in state["observed_stage_intervals"]],
        )
        runs = five._merge_five_stage_runs(state["observed_stage_intervals"])
        self.assertEqual(list(five.STAGES), [item["stage"] for item in runs])
        first_cycle = runs[1]
        self.assertEqual(
            ["measurement_action", "writing_action"],
            first_cycle["base_action_types"],
        )
        self.assertTrue(first_cycle["contains_measurement_evidence"])
        self.assertTrue(first_cycle["contains_writing_evidence"])
        self.assertEqual(2, len(first_cycle["observed_subintervals"]))
        self.assertEqual("measurement_and_recording_cycle", first_cycle["stage_semantics"])
        self.assertEqual(1, first_cycle["cycle_index"])
        self.assertEqual(1, len(first_cycle["measurement_subintervals"]))
        self.assertEqual(1, len(first_cycle["writing_subintervals"]))

    def test_only_merged_cycle_spans_long_evidence_gaps(self) -> None:
        intervals = []
        for index, (stage, action, start) in enumerate(
            [
                ("circuit_wiring", "wiring_action", 0),
                ("circuit_wiring", "wiring_action", 30),
                ("recording_1", "measurement_action", 40),
                ("recording_1", "writing_action", 100),
            ]
        ):
            item = event(index, action)
            item.update(
                {
                    "stage": stage,
                    "base_action_type": action,
                    "start_seconds": float(start),
                    "end_seconds": float(start + 2),
                    "start_frame_number": start,
                    "end_frame_number": start + 2,
                    "start_frame_id": f"frame_{start:08d}",
                    "end_frame_id": f"frame_{start + 2:08d}",
                }
            )
            intervals.append(item)
        runs = five._merge_five_stage_runs(intervals)
        self.assertEqual(
            ["circuit_wiring", "circuit_wiring", "recording_1"],
            [item["stage"] for item in runs],
        )
        self.assertEqual(2, len(runs[-1]["observed_subintervals"]))

    def test_five_stage_defaults_to_local_partial_reduce(self) -> None:
        self.assertEqual(
            ["--prepare-only", "--reduce-recovery-policy", "local_partial"],
            five.normalized_argv(["--prepare-only"]),
        )


if __name__ == "__main__":
    unittest.main()
