from __future__ import annotations

from pathlib import Path

import numpy as np
import paddle
import pytest
from omegaconf import OmegaConf

from prose_fd_paddle.symbol_utils.environment import SymbolicEnvironment
from prose_fd_paddle.models.transformer_wrappers import PROSE_2to1


ROOT = Path(__file__).resolve().parents[2]
TORCH_CKPT = ROOT / "models" / "prose_fd_torch" / "prose_fd.pth"


def load_cfgs():
    model_cfg = OmegaConf.load(ROOT / "prose_fd_paddle" / "configs" / "model" / "prose_2to1.yaml")
    data_cfg = OmegaConf.load(ROOT / "prose_fd_paddle" / "configs" / "data" / "fluids.yaml")
    symbol_cfg = OmegaConf.load(ROOT / "prose_fd_paddle" / "configs" / "symbol" / "symbol.yaml")
    OmegaConf.resolve(model_cfg)
    OmegaConf.resolve(data_cfg)
    OmegaConf.resolve(symbol_cfg)
    return model_cfg, data_cfg, symbol_cfg


def test_local_torch_checkpoint_exists_and_has_model_key():
    import torch

    payload = torch.load(TORCH_CKPT, map_location="cpu")
    assert "model" in payload
    assert any(name.startswith("module._orig_mod.") for name in payload["model"])


def test_paddle_prose_2to1_can_instantiate_on_cpu():
    model_cfg, data_cfg, symbol_cfg = load_cfgs()
    symbol_env = SymbolicEnvironment(symbol_cfg)
    model = PROSE_2to1(
        model_cfg,
        symbol_env,
        data_cfg.x_num,
        data_cfg.max_output_dimension,
        data_cfg.t_num - 10,
    )
    assert isinstance(model, paddle.nn.Layer)


def test_forward_input_contract_shapes_are_stable():
    bs = 1
    input_len = 10
    output_len = 1
    x_num = 128
    data_dim = 4
    symbol_len = 16
    rng = np.random.default_rng(1234)

    data_input = rng.standard_normal((bs, input_len, x_num, x_num, data_dim), dtype=np.float32)
    input_times = np.linspace(0.0, 9.0, input_len, dtype=np.float32).reshape(1, input_len, 1)
    output_times = np.array([[[1.0]]], dtype=np.float32)
    symbol_input = rng.integers(0, 32, size=(bs, symbol_len), dtype=np.int64)
    symbol_padding_mask = np.zeros((bs, symbol_len), dtype=bool)

    assert data_input.shape == (1, 10, 128, 128, 4)
    assert input_times.shape == (1, 10, 1)
    assert output_times.shape == (1, 1, 1)
    assert symbol_input.shape == (1, 16)
    assert symbol_padding_mask.shape == (1, 16)


def test_torch_forward_script_path_exists():
    script = ROOT / "prose_fd" / "tools" / "forward_pretrained_torch.py"
    assert script.is_file()
