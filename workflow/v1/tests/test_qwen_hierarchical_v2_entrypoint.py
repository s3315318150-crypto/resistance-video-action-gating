from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import qwen_experiment_action_hierarchical_v1 as v1  # noqa: E402
import qwen_experiment_action_hierarchical_v2 as v2  # noqa: E402
import qwen_hierarchical_v1_contract as contract  # noqa: E402


class HierarchicalV2EntrypointTests(unittest.TestCase):
    def tearDown(self) -> None:
        contract.STAGE_SCHEMA_ID = "resistance_7stage_no_battery_v1"
        v1.STAGE_SCHEMA_ID = "resistance_7stage_no_battery_v1"
        v1.ALGORITHM_ID = "qwen_experiment_action_hierarchical_v1"
        v1.ALGORITHM_SCHEMA_VERSION = "qwen_experiment_action_hierarchical_v1.v1"
        v1.DEFAULT_SCHEMA = ROOT / "configs" / "action_schemas" / "resistance_7stage_no_battery_v1.json"
        v1.DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "qwen_experiment_action_hierarchical_v1"

    def test_v2_schema_and_identity_bind_without_changing_stage_order(self) -> None:
        v2.bind_v2_identity()
        schema = contract.load_stage_schema(v2.DEFAULT_SCHEMA)
        self.assertEqual(v2.STAGE_SCHEMA_ID, schema["stage_schema_id"])
        self.assertEqual(v2.ALGORITHM_ID, v1.ALGORITHM_ID)
        self.assertEqual(list(contract.STAGES), [item["id"] for item in schema["stages"]])
        self.assertTrue(schema["stages"][-1]["absorbing_terminal"])
        self.assertNotIn("battery_change", json.dumps(schema, ensure_ascii=False))

    def test_v2_defaults_to_local_partial_without_overriding_explicit_policy(self) -> None:
        self.assertEqual(
            ["--prepare-only", "--reduce-recovery-policy", "local_partial"],
            v2.normalized_argv(["--prepare-only"]),
        )
        explicit = ["--reduce-recovery-policy", "strict"]
        self.assertEqual(explicit, v2.normalized_argv(explicit))


if __name__ == "__main__":
    unittest.main()
