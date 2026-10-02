#!/usr/bin/env python3
"""Refresh runs/wc96-loop/STATUS.json from the live loop files."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trainer.progress import build_progress  # noqa: E402


if __name__ == "__main__":
    print(json.dumps(build_progress(write_status=True), indent=2, ensure_ascii=False))
