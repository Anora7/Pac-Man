"""
Root-level pytest configuration.

Adds the src/ directory to sys.path so that all source modules can be
imported directly in tests without a formal package installation.

    from config import CellType       # works in any test file
    from maze import Maze, Position   # works in any test file
"""

import sys
from pathlib import Path

# Insert src/ at the front of sys.path once, at collection time.
_src_path = str(Path(__file__).parent / "src")
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)
