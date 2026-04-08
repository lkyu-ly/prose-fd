from __future__ import annotations

import importlib


MODULES = (
    "prose_fd_paddle.dataset",
    "prose_fd_paddle.trainer",
    "prose_fd_paddle.models.attention_utils",
    "prose_fd_paddle.models.transformer",
    "prose_fd_paddle.utils.lr_scheduler",
    "prose_fd_paddle.utils.datapipe_compat",
)


def test_package_imports_for_key_runtime_modules():
    for module_name in MODULES:
        importlib.import_module(module_name)
