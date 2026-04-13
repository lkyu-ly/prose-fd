from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = ROOT / "prose_fd"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
