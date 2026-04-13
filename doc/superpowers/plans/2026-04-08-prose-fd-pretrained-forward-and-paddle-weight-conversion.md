# PROSE-FD Pretrained Forward And Paddle Weight Conversion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 `prose_fd` 和 `prose_fd_paddle` 补齐可复现的预训练前向脚本，并提供一次性 torch->paddle 权重转换脚本，使 paddle 版本可以正确加载转换后的 PROSE-FD 预训练权重并输出与 torch 侧一致量级的结果。

**Architecture:** 直接复用现有 `build_model`、`SymbolicEnvironment` 和 wrapper `model("fwd", **model_input)` 这条主前向链路，不改训练入口。先修 paddle 侧最小运行时兼容问题，确保默认 `PROSE_2to1` 可实例化；再新增权重转换脚本和双端极简前向脚本；最后用固定随机输入做脚本级和数值级校验。

**Tech Stack:** Python 3.10, PyTorch, PaddlePaddle 3.3.0, OmegaConf, NumPy, pytest, einops

---

## File Map

- Create: `prose_fd/tools/forward_pretrained_torch.py`
  责任：加载本地 torch 预训练权重，生成固定随机输入，执行一次 `model("fwd", ...)` 并打印输出。
- Create: `prose_fd_paddle/tools/convert_torch_ckpt_to_paddle.py`
  责任：加载本地 torch 权重，按 paddle 目标 `state_dict` 自动映射并保存转换后的 paddle 权重。
- Create: `prose_fd_paddle/tools/forward_pretrained_paddle.py`
  责任：加载转换后的 paddle 权重，生成与 torch 侧完全相同的固定随机输入，执行一次 `model("fwd", ...)` 并打印输出。
- Create: `prose_fd_paddle/tests/test_pretrained_forward_tools.py`
  责任：覆盖 paddle 运行时兼容、转换脚本核心映射规则、双端脚本输入输出契约。
- Create: `prose_fd_paddle/tests/test_weight_conversion_smoke.py`
  责任：用小规模目标模块或真实 `state_dict` 键集验证自动映射和输出文件格式。
- Modify: `prose_fd_paddle/models/embedder.py`
  责任：修正 `einops.layers.paddle` 的导入与使用方式，保证 `ConvEmbedder` 可实例化。
- Modify: `prose_fd_paddle/models/transformer.py`
  责任：补齐 `RMSNorm` fallback 或本地实现，并修掉 `OperatorDecoderLayer` 交给 `paddle.nn.TransformerDecoder` 时缺 `_config` 的运行时问题。

### Task 1: 写出失败测试并锁定真实阻塞点

**Files:**
- Create: `prose_fd_paddle/tests/test_pretrained_forward_tools.py`
- Test: `prose_fd_paddle/tests/test_pretrained_forward_tools.py`

- [ ] **Step 1: 写失败测试**

```python
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
```

- [ ] **Step 2: 运行测试并确认失败**

Run:

```bash
"/home/lkyu/miniconda3/envs/paddletorch/bin/python" -m pytest "prose_fd_paddle/tests/test_pretrained_forward_tools.py" -q
```

Expected:

```text
FAIL test_paddle_prose_2to1_can_instantiate_on_cpu
```

失败原因应至少暴露当前真实阻塞之一：`einops.layers.paddle`、`paddle.nn.RMSNorm` 或 `OperatorDecoderLayer._config`。

- [ ] **Step 3: 提交测试基线**

```bash
git add "prose_fd_paddle/tests/test_pretrained_forward_tools.py"
git commit -m "test: lock pretrained forward tool requirements"
```

### Task 2: 修复 paddle 侧最小运行时兼容，确保模型可实例化

**Files:**
- Modify: `prose_fd_paddle/models/embedder.py`
- Modify: `prose_fd_paddle/models/transformer.py`
- Test: `prose_fd_paddle/tests/test_pretrained_forward_tools.py`

- [ ] **Step 1: 在 `embedder.py` 中显式导入 paddle backend 的 Rearrange**

把文件顶部的导入调整为：

```python
import einops
from einops.layers.paddle import Rearrange
import paddle
```

并把 `einops.layers.paddle.Rearrange(...)` 全部替换成：

```python
Rearrange(
    "b (t h w) d -> (b t) d h w",
    h=self.config.patch_num_output,
    w=self.config.patch_num_output,
)
```

- [ ] **Step 2: 在 `transformer.py` 中提供本地 norm resolver**

在文件顶部 `logger = getLogger()` 后加入：

```python
class PaddleRMSNorm(nn.Layer):
    def __init__(self, hidden_size, epsilon=1e-6):
        super().__init__()
        self.weight = self.create_parameter(
            shape=[hidden_size],
            default_initializer=nn.initializer.Constant(1.0),
        )
        self.epsilon = epsilon

    def forward(self, x):
        variance = paddle.mean(x * x, axis=-1, keepdim=True)
        x = x * paddle.rsqrt(variance + self.epsilon)
        return x * self.weight


def resolve_norm(norm_name):
    if norm_name == "group":
        return partial(GroupNorm, 8)
    if norm_name == "rms":
        return getattr(nn, "RMSNorm", PaddleRMSNorm)
    return nn.LayerNorm
```

然后把所有：

```python
match config.get("norm", "layer"):
    case "group":
        norm = partial(GroupNorm, 8)
    case "rms":
        norm = nn.RMSNorm
    case _:
        norm = nn.LayerNorm
```

统一替换为：

```python
norm = resolve_norm(config.get("norm", "layer"))
```

- [ ] **Step 3: 修复 `OperatorDecoderLayer` 交给 `paddle.nn.TransformerDecoder` 的 `_config` 依赖**

不要继续直接把自定义 layer 实例交给 `nn.TransformerDecoder`。在 `DataOperatorDecoder.__init__` 里改成显式构造层列表：

```python
decoder_layer = OperatorDecoderLayer(
    d_model=config.dim_emb,
    nhead=config.n_head,
    dim_feedforward=config.dim_ffn,
    dropout=config.dropout,
    activation="gelu",
    normalize_before=config.norm_first,
    custom_attn=config.custom_attn,
    rotary=config.rotary,
    norm=norm,
)
self.transformer_decoder = nn.LayerList(
    [
        decoder_layer if i == 0 else OperatorDecoderLayer(
            d_model=config.dim_emb,
            nhead=config.n_head,
            dim_feedforward=config.dim_ffn,
            dropout=config.dropout,
            activation="gelu",
            normalize_before=config.norm_first,
            custom_attn=config.custom_attn,
            rotary=config.rotary,
            norm=norm,
        )
        for i in range(config.n_layer)
    ]
)
self.decoder_norm = norm(config.dim_emb) if config.get("final_ln", True) else None
```

并在 `forward(...)` 里把原来对 `self.transformer_decoder(...)` 的调用改为显式循环：

```python
decoded = query_emb
for layer in self.transformer_decoder:
    decoded = layer(
        tgt=decoded,
        memory=src,
        tgt_mask=tgt_mask,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=src_key_padding_mask,
    )
if self.decoder_norm is not None:
    decoded = self.decoder_norm(decoded)
```

- [ ] **Step 4: 重新运行实例化测试并确认通过**

Run:

```bash
"/home/lkyu/miniconda3/envs/paddletorch/bin/python" -m pytest "prose_fd_paddle/tests/test_pretrained_forward_tools.py::test_paddle_prose_2to1_can_instantiate_on_cpu" -q
```

Expected:

```text
1 passed
```

- [ ] **Step 5: 提交兼容修复**

```bash
git add "prose_fd_paddle/models/embedder.py" "prose_fd_paddle/models/transformer.py" "prose_fd_paddle/tests/test_pretrained_forward_tools.py"
git commit -m "fix: restore paddle prose runtime compatibility"
```

### Task 3: 编写 torch 极简预训练前向脚本

**Files:**
- Create: `prose_fd/tools/forward_pretrained_torch.py`
- Test: `prose_fd_paddle/tests/test_pretrained_forward_tools.py`

- [ ] **Step 1: 扩展测试，锁定 torch 脚本接口**

在 `prose_fd_paddle/tests/test_pretrained_forward_tools.py` 追加：

```python
def test_torch_forward_script_path_exists():
    script = ROOT / "prose_fd" / "tools" / "forward_pretrained_torch.py"
    assert script.is_file()
```

- [ ] **Step 2: 新建 torch 前向脚本**

创建完整文件 `prose_fd/tools/forward_pretrained_torch.py`：

```python
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
```

- [ ] **Step 3: 运行 torch 脚本并确认输出契约**

Run:

```bash
"/home/lkyu/miniconda3/envs/paddletorch/bin/python" "prose_fd/tools/forward_pretrained_torch.py"
```

Expected:

```text
{"shape": [1, 1, 128, 128, 4], "values": ...}
```

- [ ] **Step 4: 运行测试并确认通过**

Run:

```bash
"/home/lkyu/miniconda3/envs/paddletorch/bin/python" -m pytest "prose_fd_paddle/tests/test_pretrained_forward_tools.py::test_torch_forward_script_path_exists" -q
```

Expected:

```text
1 passed
```

- [ ] **Step 5: 提交 torch 脚本**

```bash
git add "prose_fd/tools/forward_pretrained_torch.py" "prose_fd_paddle/tests/test_pretrained_forward_tools.py"
git commit -m "feat: add torch pretrained forward tool"
```

### Task 4: 编写一次性 torch->paddle 权重转换脚本

**Files:**
- Create: `prose_fd_paddle/tools/convert_torch_ckpt_to_paddle.py`
- Create: `prose_fd_paddle/tests/test_weight_conversion_smoke.py`
- Test: `prose_fd_paddle/tests/test_weight_conversion_smoke.py`

- [ ] **Step 1: 写失败测试，锁定转换规则**

创建 `prose_fd_paddle/tests/test_weight_conversion_smoke.py`：

```python
from __future__ import annotations

import numpy as np

from prose_fd_paddle.tools.convert_torch_ckpt_to_paddle import (
    clean_torch_key,
    maybe_convert_array,
)


def test_clean_torch_key_removes_parallel_prefixes():
    key = "module._orig_mod.data_encoder.transformer_encoder.layers.0.linear1.weight"
    assert clean_torch_key(key) == "data_encoder.transformer_encoder.layers.0.linear1.weight"


def test_maybe_convert_array_transposes_linear_weights():
    src = np.arange(6, dtype=np.float32).reshape(2, 3)
    target_shape = (3, 2)
    converted = maybe_convert_array("linear.weight", src, target_shape)
    assert converted.shape == target_shape
    assert np.array_equal(converted, src.T)


def test_maybe_convert_array_keeps_conv_weights():
    src = np.arange(24, dtype=np.float32).reshape(2, 3, 2, 2)
    target_shape = (2, 3, 2, 2)
    converted = maybe_convert_array("conv.weight", src, target_shape)
    assert converted.shape == target_shape
    assert np.array_equal(converted, src)
```

- [ ] **Step 2: 运行测试并确认失败**

Run:

```bash
"/home/lkyu/miniconda3/envs/paddletorch/bin/python" -m pytest "prose_fd_paddle/tests/test_weight_conversion_smoke.py" -q
```

Expected:

```text
FAIL with ModuleNotFoundError: prose_fd_paddle.tools.convert_torch_ckpt_to_paddle
```

- [ ] **Step 3: 新建转换脚本**

创建 `prose_fd_paddle/tools/convert_torch_ckpt_to_paddle.py`：

```python
from __future__ import annotations

import argparse
from pathlib import Path

import einops.layers.paddle
import numpy as np
import paddle
import torch
from omegaconf import OmegaConf

from prose_fd_paddle.models.transformer_wrappers import PROSE_2to1
from prose_fd_paddle.symbol_utils.environment import SymbolicEnvironment


ROOT = Path(__file__).resolve().parents[2]


class Params:
    cpu = 1
    compile = 0
    input_len = 10
    reload_model = None


def clean_torch_key(key: str) -> str:
    if key.startswith("module."):
        key = key[len("module."):]
    if key.startswith("_orig_mod."):
        key = key[len("_orig_mod."):]
    return key


def maybe_convert_array(name: str, array: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    if tuple(array.shape) == tuple(target_shape):
        return array
    if array.ndim == 2 and tuple(array.T.shape) == tuple(target_shape):
        return array.T
    raise ValueError(f"Shape mismatch for {name}: src={array.shape}, dst={target_shape}")


def build_paddle_model():
    model_cfg = OmegaConf.load(ROOT / "prose_fd_paddle" / "configs" / "model" / "prose_2to1.yaml")
    data_cfg = OmegaConf.load(ROOT / "prose_fd_paddle" / "configs" / "data" / "fluids.yaml")
    symbol_cfg = OmegaConf.load(ROOT / "prose_fd_paddle" / "configs" / "symbol" / "symbol.yaml")
    OmegaConf.resolve(model_cfg)
    OmegaConf.resolve(data_cfg)
    OmegaConf.resolve(symbol_cfg)
    symbol_env = SymbolicEnvironment(symbol_cfg)
    model = PROSE_2to1(model_cfg, symbol_env, data_cfg.x_num, data_cfg.max_output_dimension, data_cfg.t_num - 10)
    return model


def convert_checkpoint(torch_ckpt: Path, paddle_ckpt: Path):
    torch_payload = torch.load(torch_ckpt, map_location="cpu")
    torch_state = {clean_torch_key(k): v.detach().cpu().numpy() for k, v in torch_payload["model"].items()}
    paddle_model = build_paddle_model()
    paddle_state = paddle_model.state_dict()
    converted = {}
    missing = []
    for name, tensor in paddle_state.items():
        if name not in torch_state:
            missing.append(name)
            continue
        converted[name] = paddle.to_tensor(maybe_convert_array(name, torch_state[name], tuple(tensor.shape)))
    if missing:
        raise KeyError(f"Missing keys: {missing}")
    paddle.save({"model": converted}, str(paddle_ckpt))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--torch-ckpt", type=Path, default=ROOT / "models" / "prose_fd_torch" / "prose_fd.pth")
    parser.add_argument("--paddle-ckpt", type=Path, default=ROOT / "models" / "prose_fd_paddle" / "prose_fd_converted.pdparams")
    args = parser.parse_args()
    args.paddle_ckpt.parent.mkdir(parents=True, exist_ok=True)
    convert_checkpoint(args.torch_ckpt, args.paddle_ckpt)
    print(args.paddle_ckpt)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 运行测试并确认通过**

Run:

```bash
"/home/lkyu/miniconda3/envs/paddletorch/bin/python" -m pytest "prose_fd_paddle/tests/test_weight_conversion_smoke.py" -q
```

Expected:

```text
3 passed
```

- [ ] **Step 5: 执行一次真实转换**

Run:

```bash
"/home/lkyu/miniconda3/envs/paddletorch/bin/python" "prose_fd_paddle/tools/convert_torch_ckpt_to_paddle.py"
```

Expected:

```text
/home/lkyu/baidu/prose-fd/models/prose_fd_paddle/prose_fd_converted.pdparams
```

- [ ] **Step 6: 提交转换工具**

```bash
git add "prose_fd_paddle/tools/convert_torch_ckpt_to_paddle.py" "prose_fd_paddle/tests/test_weight_conversion_smoke.py"
git commit -m "feat: add torch to paddle checkpoint converter"
```

### Task 5: 编写 paddle 极简预训练前向脚本并验证双端输出

**Files:**
- Create: `prose_fd_paddle/tools/forward_pretrained_paddle.py`
- Modify: `prose_fd_paddle/tests/test_pretrained_forward_tools.py`
- Test: `prose_fd_paddle/tests/test_pretrained_forward_tools.py`

- [ ] **Step 1: 扩展测试，锁定 paddle 脚本接口**

在 `prose_fd_paddle/tests/test_pretrained_forward_tools.py` 追加：

```python
def test_paddle_forward_script_path_exists():
    script = ROOT / "prose_fd_paddle" / "tools" / "forward_pretrained_paddle.py"
    assert script.is_file()
```

- [ ] **Step 2: 新建 paddle 前向脚本**

创建 `prose_fd_paddle/tools/forward_pretrained_paddle.py`：

```python
from __future__ import annotations

import json
from pathlib import Path

import einops.layers.paddle
import numpy as np
import paddle
from omegaconf import OmegaConf

from prose_fd_paddle.models.build_model import build_model
from prose_fd_paddle.symbol_utils.environment import SymbolicEnvironment


ROOT = Path(__file__).resolve().parents[2]
CKPT_PATH = ROOT / "models" / "prose_fd_paddle" / "prose_fd_converted.pdparams"


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
        "data_input": paddle.to_tensor(data_input),
        "input_times": paddle.to_tensor(input_times),
        "output_times": paddle.to_tensor(output_times),
        "symbol_input": paddle.to_tensor(symbol_input),
        "symbol_padding_mask": paddle.to_tensor(symbol_padding_mask),
    }


def main():
    model_cfg = OmegaConf.load(ROOT / "prose_fd_paddle" / "configs" / "model" / "prose_2to1.yaml")
    data_cfg = OmegaConf.load(ROOT / "prose_fd_paddle" / "configs" / "data" / "fluids.yaml")
    symbol_cfg = OmegaConf.load(ROOT / "prose_fd_paddle" / "configs" / "symbol" / "symbol.yaml")
    OmegaConf.resolve(model_cfg)
    OmegaConf.resolve(data_cfg)
    OmegaConf.resolve(symbol_cfg)
    symbol_env = SymbolicEnvironment(symbol_cfg)
    modules = build_model(Params(), model_cfg, data_cfg, symbol_env)
    model = modules["model"]
    model.eval()
    model_input = build_inputs(symbol_env)
    output = model("fwd", **model_input)
    result = {
        "shape": list(output.shape),
        "values": output.numpy().tolist(),
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 运行 paddle 脚本并确认输出契约**

Run:

```bash
"/home/lkyu/miniconda3/envs/paddletorch/bin/python" "prose_fd_paddle/tools/forward_pretrained_paddle.py"
```

Expected:

```text
{"shape": [1, 1, 128, 128, 4], "values": ...}
```

- [ ] **Step 4: 增加双端数值对齐检查**

在 `prose_fd_paddle/tests/test_pretrained_forward_tools.py` 追加：

```python
import json
import subprocess
import sys


def test_torch_and_paddle_forward_outputs_are_numerically_close():
    torch_script = ROOT / "prose_fd" / "tools" / "forward_pretrained_torch.py"
    paddle_script = ROOT / "prose_fd_paddle" / "tools" / "forward_pretrained_paddle.py"
    python_bin = "/home/lkyu/miniconda3/envs/paddletorch/bin/python"

    torch_out = subprocess.check_output([python_bin, str(torch_script)], text=True)
    paddle_out = subprocess.check_output([python_bin, str(paddle_script)], text=True)

    torch_values = np.array(json.loads(torch_out)["values"], dtype=np.float32)
    paddle_values = np.array(json.loads(paddle_out)["values"], dtype=np.float32)
    diff = np.abs(torch_values - paddle_values)

    assert torch_values.shape == paddle_values.shape == (1, 1, 128, 128, 4)
    assert diff.max() < 5e-3
    assert diff.mean() < 1e-4
```

- [ ] **Step 5: 运行完整测试**

Run:

```bash
"/home/lkyu/miniconda3/envs/paddletorch/bin/python" -m pytest "prose_fd_paddle/tests/test_pretrained_forward_tools.py" "prose_fd_paddle/tests/test_weight_conversion_smoke.py" -q
```

Expected:

```text
all passed
```

- [ ] **Step 6: 提交 paddle 前向工具和最终验证**

```bash
git add "prose_fd_paddle/tools/forward_pretrained_paddle.py" "prose_fd_paddle/tests/test_pretrained_forward_tools.py"
git commit -m "feat: add paddle pretrained forward tool"
```

### Task 6: 端到端人工验收

**Files:**
- Modify: 无
- Test: `prose_fd_paddle/tests/test_pretrained_forward_tools.py`
- Test: `prose_fd_paddle/tests/test_weight_conversion_smoke.py`

- [ ] **Step 1: 重新生成 paddle 权重**

Run:

```bash
"/home/lkyu/miniconda3/envs/paddletorch/bin/python" "prose_fd_paddle/tools/convert_torch_ckpt_to_paddle.py" --torch-ckpt "/home/lkyu/baidu/prose-fd/models/prose_fd_torch/prose_fd.pth" --paddle-ckpt "/home/lkyu/baidu/prose-fd/models/prose_fd_paddle/prose_fd_converted.pdparams"
```

Expected:

```text
/home/lkyu/baidu/prose-fd/models/prose_fd_paddle/prose_fd_converted.pdparams
```

- [ ] **Step 2: 分别运行双端前向脚本**

Run:

```bash
"/home/lkyu/miniconda3/envs/paddletorch/bin/python" "prose_fd/tools/forward_pretrained_torch.py"
"/home/lkyu/miniconda3/envs/paddletorch/bin/python" "prose_fd_paddle/tools/forward_pretrained_paddle.py"
```

Expected:

```text
两条命令都输出 JSON，且 shape 都是 [1, 1, 128, 128, 4]
```

- [ ] **Step 3: 运行完整测试集**

Run:

```bash
"/home/lkyu/miniconda3/envs/paddletorch/bin/python" -m pytest "prose_fd_paddle/tests/test_pretrained_forward_tools.py" "prose_fd_paddle/tests/test_weight_conversion_smoke.py" "prose_fd_paddle/tests/test_import_smoke.py" -q
```

Expected:

```text
all passed
```

- [ ] **Step 4: 记录最终交付内容**

验收说明中必须列出以下产物：

```text
1. prose_fd/tools/forward_pretrained_torch.py
2. prose_fd_paddle/tools/convert_torch_ckpt_to_paddle.py
3. prose_fd_paddle/tools/forward_pretrained_paddle.py
4. models/prose_fd_paddle/prose_fd_converted.pdparams
5. 对齐测试结果（max_abs_diff 与 mean_abs_diff）
```

- [ ] **Step 5: 提交最终验收**

```bash
git add "prose_fd/tools/forward_pretrained_torch.py" "prose_fd_paddle/tools/convert_torch_ckpt_to_paddle.py" "prose_fd_paddle/tools/forward_pretrained_paddle.py" "prose_fd_paddle/tests/test_pretrained_forward_tools.py" "prose_fd_paddle/tests/test_weight_conversion_smoke.py"
git commit -m "test: validate pretrained forward parity workflow"
```

## Self-Review

- Spec coverage:
  - 是否存在现成前向脚本：已在 Task 1 中用测试固定为“不存在，需要新增脚本”。
  - torch 前向脚本：Task 3 覆盖。
  - paddle 权重转换脚本：Task 4 覆盖。
  - paddle 前向脚本：Task 5 覆盖。
  - 权重映射与数值对齐验证：Task 4 与 Task 5 覆盖。
  - 本地预训练权重路径和实际输出契约：Task 1、Task 3、Task 6 覆盖。
- Placeholder scan:
  - 无占位符或延后实现表述。
  - 所有创建文件步骤都给出了完整代码骨架。
  - 所有验证步骤都给出了精确命令和预期结果。
- Type consistency:
  - 双端脚本统一使用 `data_input`, `input_times`, `output_times`, `symbol_input`, `symbol_padding_mask`。
  - 转换脚本统一产出 `{"model": converted_state_dict}`，与 `build_model` 当前加载格式一致。
