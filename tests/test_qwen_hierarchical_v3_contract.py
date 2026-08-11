from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import qwen_experiment_action_hierarchical_v3 as v3  # noqa: E402
import qwen_hierarchical_v1_contract as base_contract  # noqa: E402
import qwen_hierarchical_v3_contract as contract  # noqa: E402
from qwen_hierarchical_v3_prompts import (  # noqa: E402
    build_endpoint_cleanup_binary_prompt,
    build_map_prompt,
    build_measurement_binary_prompt,
    build_reduce_prompt,
)


class HierarchicalV3ContractTests(unittest.TestCase):
    def tearDown(self) -> None:
        v3.restore_v1_bindings()

    def test_schema_adds_auxiliary_action_but_not_a_battery_stage(self) -> None:
        v3.bind_v3_identity()
        schema = base_contract.load_stage_schema(v3.DEFAULT_SCHEMA)
        self.assertEqual(v3.STAGE_SCHEMA_ID, schema["stage_schema_id"])
        self.assertIn("auxiliary_action", [item["id"] for item in schema["base_actions"]])
        self.assertNotIn("battery_change", [item["id"] for item in schema["stages"]])

    def test_map_contract_requires_auxiliary_subtype(self) -> None:
        v3.bind_v3_identity()
        frames = [
            {"image_id": "frame_000001", "frame_number": 1, "timestamp_seconds": 1.0},
            {"image_id": "frame_000002", "frame_number": 2, "timestamp_seconds": 2.0},
        ]
        value = {
            "window_id": "w001",
            "decision": "observed",
            "observations": [
                {
                    "action_type": "auxiliary_action",
                    "auxiliary_subtype": "battery_configuration_change",
                    "first_frame_id": "frame_000001",
                    "last_frame_id": "frame_000002",
                    "representative_frame_id": "frame_000002",
                    "evidence": "导线从电池盒外端改接到中间端子。",
                    "confidence": 0.8,
                }
            ],
            "confidence": 0.8,
            "uncertainty": "",
        }
        self.assertEqual([], contract.validate_map_response(value, "w001", frames))
        value["observations"][0].pop("auxiliary_subtype")
        self.assertIn("observation_0_auxiliary_subtype_invalid", contract.validate_map_response(value, "w001", frames))

    def test_prompts_expose_auxiliary_json_and_cleanup_visual_confirmation(self) -> None:
        frames = [{"image_id": "frame_000001"}, {"image_id": "frame_000002"}]
        prompt = build_map_prompt("sample", {"window_id": "w001", "window_seconds": [0.0, 10.0]}, frames)
        self.assertIn('"auxiliary_subtype"', prompt)
        self.assertIn("battery_configuration_change", prompt)
        self.assertIn("本实验没有滑动变阻器", prompt)
        self.assertNotIn("本任务没有“换电池”基础动作或阶段", prompt)
        reduce_prompt = build_reduce_prompt("sample", [])
        self.assertIn("待复核的终态候选", reduce_prompt)
        self.assertIn("不能代替原图复核", reduce_prompt)
        measurement_prompt = build_measurement_binary_prompt(
            "sample", {"window_id": "w001"}, frames
        )
        self.assertIn('"measurement_observed": "yes" | "no"', measurement_prompt)
        self.assertIn('"decision_evidence_frame_ids"', measurement_prompt)
        self.assertIn("本实验没有滑动变阻器", measurement_prompt)
        cleanup_prompt = build_endpoint_cleanup_binary_prompt("sample", frames)
        self.assertIn('"cleanup_completed": "yes" | "no"', cleanup_prompt)
        self.assertIn("桌子左上角", cleanup_prompt)

    def test_bind_and_restore_leave_v1_identity_reproducible(self) -> None:
        v3.bind_v3_identity()
        self.assertEqual(v3.ALGORITHM_ID, v3.engine.ALGORITHM_ID)
        self.assertEqual(contract.BASE_ACTIONS, base_contract.BASE_ACTIONS)
        v3.restore_v1_bindings()
        self.assertEqual("qwen_experiment_action_hierarchical_v1", v3.engine.ALGORITHM_ID)
        self.assertEqual("resistance_7stage_no_battery_v1", base_contract.STAGE_SCHEMA_ID)
        self.assertNotIn("auxiliary_action", base_contract.BASE_ACTIONS)

    def test_schema_json_has_exact_seven_stages(self) -> None:
        schema = json.loads(v3.DEFAULT_SCHEMA.read_text(encoding="utf-8"))
        self.assertEqual(7, len(schema["stages"]))
        self.assertTrue(schema["stages"][-1]["absorbing_terminal"])


if __name__ == "__main__":
    unittest.main()
