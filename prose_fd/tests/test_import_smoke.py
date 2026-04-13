from __future__ import annotations

import importlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = ROOT / "prose_fd"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


MODULES = (
    "dataset",
    "trainer",
    "models.attention_utils",
    "models.transformer",
)


def test_package_imports_for_key_runtime_modules():
    for module_name in MODULES:
        importlib.import_module(module_name)
