#!/usr/bin/env python3
"""Run the native Swift generator for the multi-object grid dataset.

The Swift renderer uses AppKit so row labels are drawn as regular text glyphs.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parent
    swift_script = root / "generate_multiobject_grid_dataset.swift"
    if not swift_script.exists():
        print(f"Missing Swift generator: {swift_script}", file=sys.stderr)
        return 1

    env = dict(os.environ)
    env.setdefault("CLANG_MODULE_CACHE_PATH", "/tmp/clangmod")

    cmd = ["swift", str(swift_script), *sys.argv[1:]]
    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
