from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from prose_fd.models.build_model import build_model
from prose_fd.symbol_utils.environment import SymbolicEnvironment


ROOT = Path(__file__).resolve().parents[2]
CKPT_PATH = ROOT / "models" / "prose_fd_torch" / "prose_fd.pth"


class Params:
    cpu = 1
    compile = 0
    input_len = 10
    reload_model = str(CKPT_PATH)


def build_inputs(symbol_env):
    rng = np.random.default_rng(1234)
    data_input = rng.standard_normal((1, 10, 128, 128, 4), dtype=np.float32)
    input_times = np.linspace(0.0, 9.0, 10, dtype=np.float32).reshape(1, 10, 1)
    output_times = np.array([[[1.0]]], dtype=np.float32)
    symbol_input = rng.integers(0, len(symbol_env.equation_word2id), size=(1, 16), dtype=np.int64)
    symbol_padding_mask = np.zeros((1, 16), dtype=bool)
    return {
        "data_input": torch.from_numpy(data_input),
        "input_times": torch.from_numpy(input_times),
        "output_times": torch.from_numpy(output_times),
        "symbol_input": torch.from_numpy(symbol_input),
        "symbol_padding_mask": torch.from_numpy(symbol_padding_mask),
    }


def main():
    model_cfg = OmegaConf.load(ROOT / "prose_fd" / "configs" / "model" / "prose_2to1.yaml")
    data_cfg = OmegaConf.load(ROOT / "prose_fd" / "configs" / "data" / "fluids.yaml")
    symbol_cfg = OmegaConf.load(ROOT / "prose_fd" / "configs" / "symbol" / "symbol.yaml")
    OmegaConf.resolve(model_cfg)
    OmegaConf.resolve(data_cfg)
    OmegaConf.resolve(symbol_cfg)
    symbol_env = SymbolicEnvironment(symbol_cfg)
    modules = build_model(Params(), model_cfg, data_cfg, symbol_env)
    model = modules["model"]
    model.eval()
    model_input = build_inputs(symbol_env)
    with torch.no_grad():
        output = model("fwd", **model_input)
    result = {
        "shape": list(output.shape),
        "values": output.cpu().numpy().tolist(),
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
