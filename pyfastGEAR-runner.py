#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parent
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

    from pyfastGEAR.__main__ import main as package_main

    return int(package_main())


if __name__ == "__main__":
    raise SystemExit(main())
