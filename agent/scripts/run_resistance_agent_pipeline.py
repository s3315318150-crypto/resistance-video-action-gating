#!/usr/bin/env python3
"""Run the release pipeline with Agent-specific hierarchical-v2 fallbacks."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RELEASE_RUNNER = (
    PROJECT_ROOT
    / "workflow"
    / "v2"
    / "scripts"
    / "run_resistance_pipeline.py"
)


def load_release_runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location("resistance_release_pipeline", RELEASE_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load release pipeline: {RELEASE_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def install_agent_fallback(module: ModuleType) -> None:
    release_run_phase = module.run_phase

    def run_phase(
        name: str,
        command: list[str],
        run_root: Path,
        dry_run: bool,
    ) -> dict[str, Any]:
        effective_command = list(command)
        if name == "02_experiment_boundary":
            effective_command[1] = str(
                PROJECT_ROOT / "workflow" / "v2" / "scripts" / "qwen_experiment_segment_judge.py"
            )
        if name.startswith("03_action_") and "--allow-invalid-source-segments" not in effective_command:
            effective_command.append("--allow-invalid-source-segments")
        return release_run_phase(name, effective_command, run_root, dry_run)

    module.run_phase = run_phase


def main(argv: list[str] | None = None) -> int:
    module = load_release_runner()
    install_agent_fallback(module)
    return int(module.main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
