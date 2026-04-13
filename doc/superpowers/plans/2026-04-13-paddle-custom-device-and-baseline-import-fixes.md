# Paddle Custom Device And Baseline Import Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 以最小改动修复 `prose_fd_paddle` 在纯 Paddle 环境中的 baseline 导入阻塞和 custom device 设备名兼容问题。

**Architecture:** 第一处修复采用惰性导入，把 baseline 相关第三方依赖从模块导入期移到真正选择 baseline 模型时，避免 `prose_2to1` 训练路径间接依赖 `torch`。第二处修复保持现有 `device` / `runtime_device` 双层语义不变，只在 `resolve_runtime_device()` 中加入最小别名解析和回退逻辑，让 `device=gpu:0` 在 custom-device-only 环境中自动映射到单个可见 custom device。

**Tech Stack:** Python, PaddlePaddle, Hydra/OmegaConf, pytest-style smoke tests

---

### Task 1: 延迟导入 Paddle baseline，消除 `build_model` 的 Torch 间接依赖

**Files:**
- Modify: `prose_fd_paddle/models/build_model.py`
- Modify: `prose_fd_paddle/tests/test_import_smoke.py`
- Test: `prose_fd_paddle/tests/test_import_smoke.py`

- [ ] **Step 1: 写一个失败导入冒烟测试**

```python
def test_build_model_import_does_not_pull_baselines():
    import importlib

    module = importlib.import_module("prose_fd_paddle.models.build_model")
    assert hasattr(module, "build_model")
```

- [ ] **Step 2: 运行测试确认当前行为会在 baseline 依赖链上失败或存在导入风险**

Run: `python -m pytest "prose_fd_paddle/tests/test_import_smoke.py" -q`
Expected: 在无 Torch 环境下，`build_model` 导入会因为 `neuralop -> torch` 链路失败，或当前测试尚未覆盖该风险。

- [ ] **Step 3: 用最小代码把 baseline 导入移到分支内部**

```python
from .transformer_wrappers import PROSE_1to1, PROSE_2to1


def build_model(params, model_config, data_config, symbol_env):
    modules = {}
    name = model_config.name
    if name == "prose_1to1":
        ...
    elif name == "prose_2to1":
        ...
    elif name == "fno":
        from .baselines import FNO

        modules["model"] = FNO(model_config, data_config.max_output_dimension)
    elif name == "vit":
        from .baselines import ViT

        modules["model"] = ViT(
            model_config, data_config.x_num, data_config.max_output_dimension
        )
    elif name == "unet":
        from .baselines import UNet

        modules["model"] = UNet(model_config, data_config.max_output_dimension)
    elif name == "deeponet":
        from .baselines import DeepONet

        modules["model"] = DeepONet(model_config, data_config, params.input_len)
```

- [ ] **Step 4: 扩展导入测试，明确覆盖 `build_model`**

```python
MODULES = (
    "prose_fd_paddle.dataset",
    "prose_fd_paddle.trainer",
    "prose_fd_paddle.models.attention_utils",
    "prose_fd_paddle.models.transformer",
    "prose_fd_paddle.models.build_model",
    "prose_fd_paddle.utils.lr_scheduler",
    "prose_fd_paddle.utils.datapipe_compat",
)
```

- [ ] **Step 5: 运行测试确认修复生效**

Run: `python -m pytest "prose_fd_paddle/tests/test_import_smoke.py" -q`
Expected: PASS

### Task 2: 让 `device=gpu:0` 在 custom-device-only 环境下自动解析到可见 custom device

**Files:**
- Modify: `prose_fd_paddle/utils/mode.py`
- Create: `prose_fd_paddle/tests/test_mode_runtime_device.py`
- Test: `prose_fd_paddle/tests/test_mode_runtime_device.py`

- [ ] **Step 1: 写失败测试，覆盖 custom device 映射与正常 GPU 路径**

```python
def test_resolve_runtime_device_maps_requested_gpu_to_single_custom_device(monkeypatch):
    params = SimpleNamespace(cpu=0, device="gpu:0", multi_gpu=0, local_rank=0)
    monkeypatch.setattr(paddle.device, "get_all_custom_device_type", lambda: ["iluvatar_gpu"])
    monkeypatch.setattr(paddle.device, "get_all_device_type", lambda: ["cpu", "iluvatar_gpu"])
    calls = []

    def fake_set_device(name):
        calls.append(name)
        return name

    monkeypatch.setattr(paddle, "set_device", fake_set_device)
    monkeypatch.setattr(paddle.device, "get_device", lambda: calls[-1])

    assert resolve_runtime_device(params) == "iluvatar_gpu:0"
    assert calls == ["iluvatar_gpu:0"]
```

- [ ] **Step 2: 运行测试确认当前实现会把 `gpu:0` 直接传给 Paddle**

Run: `python -m pytest "prose_fd_paddle/tests/test_mode_runtime_device.py" -q`
Expected: FAIL，表现为没有映射到 custom device，或测试文件尚未存在。

- [ ] **Step 3: 在 `resolve_runtime_device()` 里加入最小别名解析**

```python
def _resolve_requested_device(requested: str, custom_types: list[str]) -> str:
    if requested and requested.startswith("gpu:") and custom_types:
        return f"{custom_types[0]}:{requested.split(':', 1)[1]}"
    return requested


def resolve_runtime_device(params) -> str:
    ...
    custom_types = paddle.device.get_all_custom_device_type()
    requested = getattr(params, "device", None)
    if requested:
        resolved = _resolve_requested_device(requested, custom_types)
        paddle.set_device(resolved)
        return paddle.device.get_device()
```

- [ ] **Step 4: 补充保护测试，保证原有路径不退化**

```python
def test_resolve_runtime_device_keeps_requested_gpu_when_native_gpu_available(...):
    ...
    assert resolve_runtime_device(params) == "gpu:0"


def test_resolve_runtime_device_prefers_custom_device_when_not_requested(...):
    ...
    assert resolve_runtime_device(params) == "iluvatar_gpu:0"
```

- [ ] **Step 5: 运行测试确认设备解析修复生效**

Run: `python -m pytest "prose_fd_paddle/tests/test_mode_runtime_device.py" -q`
Expected: PASS

### Task 3: 运行静态和针对性验证，确认两处修复没有破坏现有训练前置能力

**Files:**
- Verify only: `prose_fd_paddle/models/build_model.py`
- Verify only: `prose_fd_paddle/utils/mode.py`
- Verify only: `prose_fd_paddle/tests/test_import_smoke.py`
- Verify only: `prose_fd_paddle/tests/test_mode_runtime_device.py`

- [ ] **Step 1: 运行静态编译检查**

Run: `python -m py_compile "prose_fd_paddle/models/build_model.py" "prose_fd_paddle/utils/mode.py" "prose_fd_paddle/tests/test_import_smoke.py" "prose_fd_paddle/tests/test_mode_runtime_device.py"`
Expected: 无输出，退出码 0

- [ ] **Step 2: 运行针对性测试**

Run: `python -m pytest "prose_fd_paddle/tests/test_import_smoke.py" "prose_fd_paddle/tests/test_mode_runtime_device.py" -q`
Expected: PASS

- [ ] **Step 3: 做一次无 Torch 残留扫描**

Run: `rg -n "from \\.baselines|from baselines|import torch|from torch" "prose_fd_paddle/models/build_model.py" "prose_fd_paddle/utils/mode.py" "prose_fd_paddle/tests"`
Expected: `build_model.py` 不再有顶层 baseline 导入，也不引入任何新的 Torch 依赖。

