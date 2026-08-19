#!/usr/bin/env python3
"""Standalone entrypoint for the resistance-video scheduling agent."""

from __future__ import annotations

import sys
from pathlib import Path


AGENT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(AGENT_ROOT / "resistance_agent"))

from orchestrator import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
