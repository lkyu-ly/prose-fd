from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import pytest

from prose_fd_paddle.models.build_model import build_model


MODULES = (
    "prose_fd_paddle.dataset",
    "prose_fd_paddle.trainer",
    "prose_fd_paddle.models.build_model",
    "prose_fd_paddle.models.attention_utils",
    "prose_fd_paddle.models.transformer",
    "prose_fd_paddle.utils.lr_scheduler",
    "prose_fd_paddle.utils.datapipe_compat",
)


def test_package_imports_for_key_runtime_modules():
    for module_name in MODULES:
        importlib.import_module(module_name)


def test_build_model_import_does_not_pull_in_baseline_modules():
    for module_name in (
        "prose_fd_paddle.models.baselines",
        "neuralop",
        "neuralop.models",
    ):
        sys.modules.pop(module_name, None)
    sys.modules.pop("prose_fd_paddle.models.build_model", None)

    importlib.import_module("prose_fd_paddle.models.build_model")

    assert "prose_fd_paddle.models.baselines" not in sys.modules
    assert "neuralop" not in sys.modules
    assert "neuralop.models" not in sys.modules


@pytest.mark.parametrize("model_name", ("fno", "vit", "unet", "deeponet"))
def test_build_model_rejects_unported_baselines(model_name):
    params = SimpleNamespace(input_len=1, reload_model=None, cpu=1)
    model_config = SimpleNamespace(name=model_name)
    data_config = SimpleNamespace()

    with pytest.raises(NotImplementedError, match=r"prose_1to1 / prose_2to1"):
        build_model(params, model_config, data_config, symbol_env=None)
