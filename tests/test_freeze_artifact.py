from __future__ import annotations

import hashlib
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


freeze_artifact = load_script("freeze_artifact.py")


class FreezeArtifactTests(unittest.TestCase):
    def test_freeze_records_hash_without_modifying_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "prediction.json"
            source.write_text('{"predictions": []}\n', encoding="utf-8")
            before = source.read_bytes()
            target = root / "prediction.freeze.json"
            record = freeze_artifact.freeze(source, target)
            self.assertEqual(source.read_bytes(), before)
            self.assertEqual(record["sha256"], hashlib.sha256(before).hexdigest())
            self.assertFalse(record["ground_truth_accessed"])
            self.assertEqual(json.loads(target.read_text(encoding="utf-8"))["sha256"], record["sha256"])

    def test_freeze_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "prediction.json"
            source.write_text("{}", encoding="utf-8")
            target = root / "freeze.json"
            target.write_text("{}", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                freeze_artifact.freeze(source, target)


if __name__ == "__main__":
    unittest.main()
