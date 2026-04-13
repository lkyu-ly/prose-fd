# PROSE-FD Paddle Single-GPU Training Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 `prose_fd_paddle` 能在单机单卡上用本地 `64x64` 缩小版 shallow-water 数据集跑通 README 风格的训练验证命令，并去掉训练链路中对 CUDA 的硬编码依赖。

**Architecture:** 不对数据做额外离线转换，直接新增一个只面向最小训练验证的 `shallow_water_minimal` 配置，令模型和数据管线原生支持 `64x64`。训练链路先用测试驱动修复两个真实阻塞点：Paddle `DataLoader` 与本地 datapipe 兼容层不匹配，以及运行时设备管理强绑定 `.cuda()` / `paddle.cuda.*`。最终用 CPU 单测覆盖数据和前向链路，再用一次安全的 GPU 单步前后向探针收敛单卡启动参数。

**Tech Stack:** Python 3.10, PaddlePaddle 3.3.0, Hydra / OmegaConf, h5py, pytest, local PDEBench HDF5 data

---

## Context And Findings

这些结论已经通过本地探索确认，执行时不要重新发散：

- 本地最小数据集位于 `"/home/lkyu/baidu/prose-fd/dataset/pdebench/2D/shallow-water/2D_rdb_NA_NA.h5"`。
- 数据结构与 PDEBench shallow-water 原格式一致，但网格缩小为 `64x64`，样本数为 `200`，每个样本 `data` 形状是 `(101, 64, 64, 1)`。
- `prose_fd/prose_fd/data_utils/README.md` 中“统一到 `128x128`”的要求是为多数据集混训设计的。对单独 `shallow_water` 训练验证，这不是硬性前提。
- 纯 CPU 冒烟已经证明 `data.x_num=64`、`data.shallow_water.x_num=64`、`model=prose_2to1` 时，`PROSE_2to1` 能成功前向，输出形状是 `(1, 10, 64, 64, 4)`。
- 当前训练链路的真实阻塞点不是模型，而是 `paddle.io.DataLoader` 把本地 datapipe 兼容层当成 map-style 数据集，触发 `TypeError: object of type 'Multiplexer' has no len()`。
- Paddle 侧还残留多个 `.cuda()` / `paddle.cuda.*` 分支，位于 `main.py`、`trainer.py`、`evaluate.py`、`models/build_model.py`、`utils/misc.py`、`utils/mode.py`。这会阻塞后续 custom device。
- `prose_fd_paddle/configs/main.yaml` 默认模型仍然写成了不存在的 `prose_2to1_S`，这会直接阻塞任何不显式传 `model=...` 的训练命令。
- 本机显卡可读到 `NVIDIA GeForce RTX 4060 Ti, 16380 MiB`。
- 受控 GPU 探针结果：`PROSE_2to1 + shallow_water 64x64 + batch=1` 单步前向反向约占 `2149 MB`，这个数字不包含完整优化器状态冗余，因此最终 dryrun 命令应从 `batch_size=2` 这样的保守值起步，而不是 README 脚本里的 `80`。

## File Map

**Create:**

- `prose_fd_paddle/configs/data/shallow_water_minimal.yaml`
  - 只描述本地 `64x64` shallow-water 最小训练验证数据源。
- `prose_fd_paddle/tests/test_training_data_smoke.py`
  - 覆盖训练 / 验证 `DataLoader`、`64x64` 前向、默认模型配置存在性。
- `prose_fd_paddle/tests/test_runtime_device_helpers.py`
  - 覆盖新的设备选择与张量搬运 helper，确保不再依赖 `.cuda()`。

**Modify:**

- `prose_fd_paddle/utils/datapipe_compat.py`
  - 让本地 datapipe 真正兼容 `paddle.io.DataLoader`。
- `prose_fd_paddle/utils/misc.py`
  - 新增运行时设备 helper，改造 `to_cuda` 为通用设备搬运实现。
- `prose_fd_paddle/utils/mode.py`
  - 统一运行时设备选择逻辑，兼容 CPU / GPU / custom device。
- `prose_fd_paddle/main.py`
  - 移除 `paddle.cuda.is_available()` 强依赖，接入运行时设备 helper，并修正显存日志逻辑。
- `prose_fd_paddle/models/build_model.py`
  - 用通用 `.to(device)` 替代 `.cuda()`。
- `prose_fd_paddle/trainer.py`
  - 保持现有训练语义不变，只改设备搬运与显存日志。
- `prose_fd_paddle/evaluate.py`
  - 保持现有评估语义不变，只改设备搬运与 `boundary_mask` 上卡逻辑。
- `prose_fd_paddle/configs/main.yaml`
  - 修复默认模型名，增加显式设备配置项。

**Optional Modify Only If Needed During Execution:**

- `prose_fd_paddle/README.md`
  - 仅在代码和命令稳定后补一个单卡最小验证命令；不先动它。

## Approach Options

### Option A: 先把 `64x64` 数据离线转成 `128x128`

- 优点：最大限度贴近原论文默认设置。
- 缺点：这是额外的数据工程，不解决当前真正阻塞训练的 dataloader / 设备问题；对“最小训练验证”明显过度设计。

### Option B: 直接为 `64x64` shallow-water 增加最小训练配置

- 优点：最小改动、最快闭环、已经被 CPU 前向冒烟证明可行。
- 缺点：得到的是“单数据集最小训练验证路径”，不是多数据集最终配置。

### Option C: 另做一套更小模型配置

- 优点：更容易塞进 16GB 显存。
- 缺点：改变了模型规模，不再是 README 所对应的主模型链路，优先级低于先跑通原模型的最小 batch。

**Recommendation:** 选择 Option B。只有在 `batch_size=1` 或 `2` 的单卡探针下仍然无法稳定运行，才回退到 Option C。

---

### Task 1: 用失败测试锁定当前训练阻塞

**Files:**
- Create: `prose_fd_paddle/tests/test_training_data_smoke.py`
- Create: `prose_fd_paddle/tests/test_runtime_device_helpers.py`
- Modify: `prose_fd_paddle/configs/main.yaml`

- [ ] **Step 1: 写训练数据链路失败测试**

在 `prose_fd_paddle/tests/test_training_data_smoke.py` 创建以下完整内容：

```python
from __future__ import annotations

from pathlib import Path

import paddle
from omegaconf import OmegaConf

from prose_fd_paddle.data_utils.collate import custom_collate
from prose_fd_paddle.dataset import get_dataset
from prose_fd_paddle.models.build_model import build_model
from prose_fd_paddle.symbol_utils.environment import SymbolicEnvironment


ROOT = Path(__file__).resolve().parents[2]
LOCAL_SW64 = ROOT / "dataset" / "pdebench" / "2D" / "shallow-water" / "2D_rdb_NA_NA.h5"


def build_sw64_cfg():
    main_cfg = OmegaConf.load(ROOT / "prose_fd_paddle" / "configs" / "main.yaml")
    data_cfg = OmegaConf.load(ROOT / "prose_fd_paddle" / "configs" / "data" / "fluids.yaml")
    model_cfg = OmegaConf.load(ROOT / "prose_fd_paddle" / "configs" / "model" / "prose_2to1.yaml")
    symbol_cfg = OmegaConf.load(ROOT / "prose_fd_paddle" / "configs" / "symbol" / "symbol.yaml")

    cfg = OmegaConf.create(OmegaConf.to_container(main_cfg, resolve=False))
    cfg.data = data_cfg
    cfg.model = model_cfg
    cfg.symbol = symbol_cfg
    cfg.cpu = 1
    cfg.compile = 0
    cfg.reload_model = None
    cfg.reload_checkpoint = None
    cfg.use_wandb = 0
    cfg.num_workers = 0
    cfg.num_workers_eval = 0
    cfg.batch_size = 2
    cfg.batch_size_eval = 2
    cfg.overfit_test = 0
    cfg.local_rank = 0
    cfg.global_rank = 0
    cfg.n_gpu_per_node = 1
    cfg.world_size = 1
    cfg.multi_gpu = 0
    cfg.multi_node = 0
    cfg.is_master = 1
    cfg.data.types = ["shallow_water"]
    cfg.data.x_num = 64
    cfg.data.shallow_water.x_num = 64
    cfg.data.shallow_water.data_path = str(LOCAL_SW64)
    OmegaConf.resolve(cfg)
    return cfg


def test_train_dataloader_can_yield_one_batch_from_local_sw64():
    cfg = build_sw64_cfg()
    symbol_env = SymbolicEnvironment(cfg.symbol)
    dataset = get_dataset(cfg, symbol_env, split="train")
    loader = paddle.io.DataLoader(
        dataset=dataset,
        batch_size=cfg.batch_size,
        num_workers=0,
        drop_last=True,
        collate_fn=custom_collate(
            cfg.data.max_output_dimension,
            symbol_env.pad_index,
            cfg.data.tie_fields,
            cfg.data.get("mixed_length", 0),
            cfg.input_len,
            cfg.symbol.pad_right,
        ),
    )
    batch = next(iter(loader))
    assert tuple(batch["data"].shape) == (2, 20, 64, 64, 4)
    assert tuple(batch["data_mask"].shape) == (2, 10, 1, 1, 4)
    assert batch["symbol_input"].shape[0] == 2


def test_eval_dataloader_can_yield_one_batch_from_local_sw64():
    cfg = build_sw64_cfg()
    symbol_env = SymbolicEnvironment(cfg.symbol)
    dataset = get_dataset(cfg, symbol_env, split="val")["shallow_water"]
    loader = paddle.io.DataLoader(
        dataset=dataset,
        batch_size=cfg.batch_size_eval,
        num_workers=0,
        collate_fn=custom_collate(
            cfg.data.max_output_dimension,
            symbol_env.pad_index,
            cfg.data.tie_fields,
            pad_right=cfg.symbol.pad_right,
        ),
    )
    batch = next(iter(loader))
    assert tuple(batch["data"].shape) == (2, 20, 64, 64, 4)


def test_prose_2to1_can_run_forward_on_local_sw64_batch():
    cfg = build_sw64_cfg()
    symbol_env = SymbolicEnvironment(cfg.symbol)
    modules = build_model(cfg, cfg.model, cfg.data, symbol_env)
    model = modules["model"]
    model.eval()

    sample = next(iter(get_dataset(cfg, symbol_env, split="train")))
    batch = custom_collate(
        cfg.data.max_output_dimension,
        symbol_env.pad_index,
        cfg.data.tie_fields,
        cfg.data.get("mixed_length", 0),
        cfg.input_len,
        cfg.symbol.pad_right,
    )([sample])

    input_len = cfg.input_len
    times = paddle.linspace(0, 10, cfg.data.t_num, dtype=paddle.float32)[None]
    model_input = {
        "data_input": batch["data"][:, :input_len],
        "input_times": times[:, :input_len, None],
        "output_times": times[:, input_len:, None] - times[:, input_len - 1 : input_len, None],
        "symbol_input": batch["symbol_input"],
        "symbol_padding_mask": batch["symbol_mask"],
    }
    with paddle.no_grad():
        output = model("fwd", **model_input)

    assert tuple(output.shape) == (1, 10, 64, 64, 4)


def test_default_model_reference_points_to_existing_config():
    main_cfg = OmegaConf.load(ROOT / "prose_fd_paddle" / "configs" / "main.yaml")
    assert main_cfg.defaults[1]["model"] == "prose_2to1"
```

- [ ] **Step 2: 写设备 helper 失败测试**

在 `prose_fd_paddle/tests/test_runtime_device_helpers.py` 创建以下完整内容：

```python
from __future__ import annotations

import paddle

from prose_fd_paddle.utils import misc


def test_to_cuda_uses_runtime_device_string_not_cuda_hardcode():
    misc.set_runtime_device("cpu")
    tensor = paddle.arange(4, dtype=paddle.float32)
    moved = misc.to_cuda(tensor)
    assert str(moved.place) == "Place(cpu)"


def test_to_cuda_preserves_none_and_multiple_args():
    misc.set_runtime_device("cpu")
    x = paddle.ones([2], dtype=paddle.float32)
    y = paddle.zeros([2], dtype=paddle.float32)
    moved_x, moved_none, moved_y = misc.to_cuda(x, None, y)
    assert str(moved_x.place) == "Place(cpu)"
    assert moved_none is None
    assert str(moved_y.place) == "Place(cpu)"


def test_max_memory_allocated_mb_is_none_on_cpu():
    misc.set_runtime_device("cpu")
    assert misc.max_memory_allocated_mb() is None
```

- [ ] **Step 3: 运行新测试，确认现在确实失败**

Run:

```bash
"/home/lkyu/miniconda3/envs/paddletorch/bin/python" -m pytest \
  "prose_fd_paddle/tests/test_training_data_smoke.py" \
  "prose_fd_paddle/tests/test_runtime_device_helpers.py" -q
```

Expected:

- `test_train_dataloader_can_yield_one_batch_from_local_sw64` 失败，报 `TypeError: object of type 'Multiplexer' has no len()`
- `test_eval_dataloader_can_yield_one_batch_from_local_sw64` 失败，报 `TypeError: object of type 'ShallowWater2D' has no len()`
- `test_default_model_reference_points_to_existing_config` 失败，因为 `main.yaml` 仍然是 `prose_2to1_S`
- 设备 helper 测试失败，因为 `utils.misc` 还没有 `set_runtime_device` / `max_memory_allocated_mb`

- [ ] **Step 4: 记录失败点，不做其他改动**

这一轮只确认失败，不修改实现。预期输出里必须能看到 dataloader 和默认模型名两个阻塞点。

---

### Task 2: 修复 datapipe 与 Paddle DataLoader 的训练兼容性

**Files:**
- Modify: `prose_fd_paddle/utils/datapipe_compat.py`
- Test: `prose_fd_paddle/tests/test_training_data_smoke.py`

- [ ] **Step 1: 让自定义 datapipe 成为真正的 Paddle IterableDataset**

把 `prose_fd_paddle/utils/datapipe_compat.py` 改成下面这份完整实现：

```python
from __future__ import annotations

import random
from collections.abc import Iterable

import paddle


class IterDataPipe(paddle.io.IterableDataset):
    def __iter__(self):
        raise NotImplementedError

    def shuffle(self, buffer_size: int, seed: int | None = None):
        return ShuffledIterDataPipe(self, buffer_size=buffer_size, seed=seed)

    def cycle(self):
        return CycledIterDataPipe(self)


class ShuffledIterDataPipe(IterDataPipe):
    def __init__(self, datapipe: Iterable, buffer_size: int, seed: int | None = None):
        super().__init__()
        self.datapipe = datapipe
        self.buffer_size = buffer_size
        self.seed = seed

    def __iter__(self):
        rng = random.Random(self.seed)
        iterator = iter(self.datapipe)
        buffer = []
        try:
            for _ in range(self.buffer_size):
                buffer.append(next(iterator))
        except StopIteration:
            pass
        while buffer:
            index = rng.randrange(len(buffer))
            yield buffer.pop(index)
            try:
                buffer.append(next(iterator))
            except StopIteration:
                continue


class CycledIterDataPipe(IterDataPipe):
    def __init__(self, datapipe: Iterable):
        super().__init__()
        self.datapipe = datapipe

    def __iter__(self):
        while True:
            yielded = False
            for item in self.datapipe:
                yielded = True
                yield item
            if not yielded:
                return


class Multiplexer(IterDataPipe):
    def __init__(self, *datapipes: Iterable):
        super().__init__()
        self.datapipes = datapipes

    def __iter__(self):
        iterators = [iter(datapipe) for datapipe in self.datapipes]
        while iterators:
            next_iterators = []
            for iterator in iterators:
                try:
                    yield next(iterator)
                    next_iterators.append(iterator)
                except StopIteration:
                    continue
            iterators = next_iterators


class SampleMultiplexer(IterDataPipe):
    def __init__(self, datapipes_to_weights: dict[Iterable, float], seed: int | None = None):
        super().__init__()
        self.datapipes_to_weights = datapipes_to_weights
        self.seed = seed

    def __iter__(self):
        rng = random.Random(self.seed)
        active = [
            [iter(datapipe), float(weight)]
            for datapipe, weight in self.datapipes_to_weights.items()
        ]
        while active:
            positive = [item for item in active if item[1] > 0]
            if not positive:
                return
            population = list(range(len(positive)))
            weights = [item[1] for item in positive]
            selected = positive[rng.choices(population, weights=weights, k=1)[0]][0]
            try:
                yield next(selected)
            except StopIteration:
                active = [item for item in active if item[0] is not selected]
```

- [ ] **Step 2: 单独运行 datapipe 兼容测试**

Run:

```bash
"/home/lkyu/miniconda3/envs/paddletorch/bin/python" -m pytest \
  "prose_fd_paddle/tests/test_datapipe_compat.py" -q
```

Expected:

```text
4 passed
```

- [ ] **Step 3: 回头运行训练数据链路测试**

Run:

```bash
"/home/lkyu/miniconda3/envs/paddletorch/bin/python" -m pytest \
  "prose_fd_paddle/tests/test_training_data_smoke.py::test_train_dataloader_can_yield_one_batch_from_local_sw64" \
  "prose_fd_paddle/tests/test_training_data_smoke.py::test_eval_dataloader_can_yield_one_batch_from_local_sw64" -q
```

Expected:

- 这两个测试现在都应通过。
- 如果仍然失败，不要绕过 `DataLoader`；继续围绕 iterable dataset 语义修。

---

### Task 3: 把运行时设备管理从 CUDA 硬编码改成通用设备逻辑

**Files:**
- Modify: `prose_fd_paddle/utils/misc.py`
- Modify: `prose_fd_paddle/utils/mode.py`
- Modify: `prose_fd_paddle/main.py`
- Modify: `prose_fd_paddle/models/build_model.py`
- Modify: `prose_fd_paddle/trainer.py`
- Modify: `prose_fd_paddle/evaluate.py`
- Modify: `prose_fd_paddle/configs/main.yaml`
- Test: `prose_fd_paddle/tests/test_runtime_device_helpers.py`

- [ ] **Step 1: 在主配置里显式加入设备配置，并修掉默认模型名**

把 `prose_fd_paddle/configs/main.yaml` 的顶部默认项改成：

```yaml
defaults:
  - data: fluids
  - model: prose_2to1
  - optim: adamw
  - symbol: symbol
  - _self_
  - override hydra/hydra_logging: none
  - override hydra/job_logging: none
```

并在 `cpu: 0` 前面新增：

```yaml
device: null
```

- [ ] **Step 2: 在 `utils/misc.py` 引入运行时设备 helper**

把 `prose_fd_paddle/utils/misc.py` 中与设备相关的部分改成下面这组实现：

```python
RUNTIME_DEVICE = "cpu"


def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    paddle.seed(seed_value)


def set_runtime_device(device: str):
    global RUNTIME_DEVICE
    RUNTIME_DEVICE = device


def get_runtime_device() -> str:
    return RUNTIME_DEVICE


def max_memory_allocated_mb():
    if RUNTIME_DEVICE.startswith("gpu"):
        return paddle.device.cuda.max_memory_allocated() / 1024**2
    return None


def to_device(*args, use_cpu=False, device: str | None = None):
    target = "cpu" if use_cpu else (device or RUNTIME_DEVICE)
    moved = [None if x is None else x.to(target) for x in args]
    if len(args) == 1:
        return moved[0]
    return moved


def to_cuda(*args, use_cpu=False):
    return to_device(*args, use_cpu=use_cpu)


def sync_tensor(t):
    if not paddle.distributed.is_initialized():
        return t
    source_place = t.place
    t_sync = t.to(RUNTIME_DEVICE)
    paddle.distributed.barrier()
    paddle.distributed.all_reduce(t_sync, op=paddle.distributed.ReduceOp.SUM)
    return t_sync.to(source_place)
```

删除这些旧逻辑：

- `CUDA = True`
- `paddle.cuda.manual_seed_all(seed_value)`
- 所有直接 `.cuda()` 的实现

- [ ] **Step 3: 在 `utils/mode.py` 统一运行时设备选择**

把 `prose_fd_paddle/utils/mode.py` 改成下面的核心结构：

```python
import os
import socket

import paddle


def resolve_runtime_device(params) -> str:
    if params.cpu:
        paddle.set_device("cpu")
        return paddle.device.get_device()

    requested = params.get("device", None)
    if requested:
        paddle.set_device(requested)
        return paddle.device.get_device()

    if params.multi_gpu:
        paddle.set_device(f"gpu:{params.local_rank}")
        return paddle.device.get_device()

    custom_types = paddle.device.get_all_custom_device_type()
    if custom_types:
        paddle.set_device(f"{custom_types[0]}:0")
        return paddle.device.get_device()

    device_types = paddle.device.get_all_device_type()
    if "gpu" in device_types:
        paddle.set_device("gpu:0")
        return paddle.device.get_device()

    paddle.set_device("cpu")
    return paddle.device.get_device()


def init_distributed_mode(params):
    params.world_size = int(paddle.distributed.get_world_size())
    if params.world_size > 1:
        params.global_rank = int(os.environ["RANK"])
        params.local_rank = int(os.environ["LOCAL_RANK"])
        params.n_gpu_per_node = int(os.environ.get("NGPU", params.world_size))
        params.n_nodes = params.world_size // params.n_gpu_per_node
        params.node_id = params.global_rank // params.n_gpu_per_node
    else:
        params.local_rank = 0
        params.n_nodes = 1
        params.node_id = 0
        params.global_rank = 0
        params.world_size = 1
        params.n_gpu_per_node = 1

    params.is_master = params.node_id == 0 and params.local_rank == 0
    params.multi_node = params.n_nodes > 1
    params.multi_gpu = params.world_size > 1
    params.runtime_device = resolve_runtime_device(params)

    if params.multi_gpu:
        print(f\"Initializing Paddle distributed on {params.runtime_device} ...\")
        paddle.distributed.init_parallel_env()
```

- [ ] **Step 4: 在 `main.py` 接入新的运行时设备 helper**

把 `main.py` 中设备相关逻辑改为：

```python
    init_distributed_mode(params)
    utils.misc.set_runtime_device(params.runtime_device)
```

删除这一段：

```python
    if not params.cpu:
        assert paddle.cuda.is_available()
    utils.misc.CUDA = not params.cpu
```

并把两处显存日志改为：

```python
        max_mem = utils.misc.max_memory_allocated_mb()
        if max_mem is not None:
            logger.info(" MEM: {:.2f} MB ".format(max_mem))
```

以及末尾：

```python
    max_mem = utils.misc.max_memory_allocated_mb()
    if max_mem is not None:
        logger.info(" MEM: {:.2f} MB ".format(max_mem))
```

- [ ] **Step 5: 替换模型和评估中的 `.cuda()`**

在 `prose_fd_paddle/models/build_model.py` 做这两个替换：

```python
try:
    from ..utils.misc import get_runtime_device
except ImportError:
    from utils.misc import get_runtime_device
```

和：

```python
    if not params.cpu:
        for v in modules.values():
            v.to(get_runtime_device())
```

在 `prose_fd_paddle/evaluate.py` 顶部沿用已有 `to_cuda` 导入，不额外新建 helper，并把：

```python
                self.boundary_mask = self.boundary_mask.cuda()
```

改成：

```python
                self.boundary_mask = to_cuda(self.boundary_mask)
```

- [ ] **Step 6: 把 `trainer.py` 的显存日志改成设备无关**

在 `trainer.py` 增加导入：

```python
    from .utils.misc import max_memory_allocated_mb, to_cuda
```

和 fallback：

```python
    from utils.misc import max_memory_allocated_mb, to_cuda
```

然后把：

```python
        max_mem = paddle.cuda.max_memory_allocated() / 1024**2
        s_mem = " MEM: {:.2f} MB - ".format(max_mem)
        logger.info(s_iter + s_mem + s_lr)
```

改成：

```python
        max_mem = max_memory_allocated_mb()
        s_mem = "" if max_mem is None else " MEM: {:.2f} MB - ".format(max_mem)
        logger.info(s_iter + s_mem + s_lr)
```

- [ ] **Step 7: 跑设备 helper 测试**

Run:

```bash
"/home/lkyu/miniconda3/envs/paddletorch/bin/python" -m pytest \
  "prose_fd_paddle/tests/test_runtime_device_helpers.py" -q
```

Expected:

```text
3 passed
```

- [ ] **Step 8: 扫描 Paddle 运行时代码，确保不再有活跃 CUDA 硬编码**

Run:

```bash
rg -n "paddle\\.cuda\\.|\\.cuda\\(|assert paddle\\.cuda\\.is_available|CUDA =" \
  "prose_fd_paddle"
```

Expected:

- 允许残留只出现在测试、探索脚本或注释里。
- `main.py`、`trainer.py`、`evaluate.py`、`models/build_model.py`、`utils/misc.py`、`utils/mode.py` 中不应再有活跃 `.cuda()`。

---

### Task 4: 新增最小训练数据配置，直接支持本地 `64x64` shallow-water

**Files:**
- Create: `prose_fd_paddle/configs/data/shallow_water_minimal.yaml`
- Test: `prose_fd_paddle/tests/test_training_data_smoke.py`

- [ ] **Step 1: 新增本地最小训练数据配置**

创建 `prose_fd_paddle/configs/data/shallow_water_minimal.yaml`，内容如下：

```yaml
types: [shallow_water]

max_output_dimension: 4
train_val_test_ratio: [0.8, 0.1, 0.1]
t_num: 20
x_num: 64

mixed_length: ${.t_num}

random_start:
  train: true
  val: false
  test: false
  start_max: 40

tie_fields: 1

sampler:
  uniform: 1
  shallow_water: 1

shallow_water:
  data_path: /home/lkyu/baidu/prose-fd/dataset/pdebench/2D/shallow-water/2D_rdb_NA_NA.h5
  t_step: 1
  x_num: 64
  dim: 1
  c_mask: [0, 0, 0, 0, 0, 1]
```

- [ ] **Step 2: 加一个配置存在性测试**

在 `prose_fd_paddle/tests/test_training_data_smoke.py` 末尾追加：

```python
def test_shallow_water_minimal_config_matches_local_dataset():
    data_cfg = OmegaConf.load(ROOT / "prose_fd_paddle" / "configs" / "data" / "shallow_water_minimal.yaml")
    assert data_cfg.types == ["shallow_water"]
    assert data_cfg.x_num == 64
    assert data_cfg.shallow_water.x_num == 64
    assert Path(data_cfg.shallow_water.data_path).is_file()
```

- [ ] **Step 3: 运行整组数据 smoke 测试**

Run:

```bash
"/home/lkyu/miniconda3/envs/paddletorch/bin/python" -m pytest \
  "prose_fd_paddle/tests/test_training_data_smoke.py" -q
```

Expected:

```text
5 passed
```

---

### Task 5: 用 CPU 级 smoke 覆盖 Trainer / Evaluator 的最小链路

**Files:**
- Modify: `prose_fd_paddle/tests/test_training_data_smoke.py`
- Modify: `prose_fd_paddle/trainer.py`
- Modify: `prose_fd_paddle/evaluate.py`

- [ ] **Step 1: 增加 Trainer 构造级 smoke**

在 `prose_fd_paddle/tests/test_training_data_smoke.py` 末尾追加：

```python
from prose_fd_paddle.trainer import Trainer
from prose_fd_paddle.evaluate import Evaluator


def test_trainer_can_build_train_dataloader_on_local_sw64():
    cfg = build_sw64_cfg()
    symbol_env = SymbolicEnvironment(cfg.symbol)
    modules = build_model(cfg, cfg.model, cfg.data, symbol_env)
    trainer = Trainer(modules, cfg, symbol_env)
    batch = trainer.get_batch()
    assert tuple(batch["data"].shape) == (2, 20, 64, 64, 4)


def test_evaluator_can_build_validation_dataloader_on_local_sw64():
    cfg = build_sw64_cfg()
    symbol_env = SymbolicEnvironment(cfg.symbol)
    modules = build_model(cfg, cfg.model, cfg.data, symbol_env)
    trainer = Trainer(modules, cfg, symbol_env)
    evaluator = Evaluator(trainer, symbol_env)
    batch = next(iter(evaluator.dataloaders["shallow_water"]))
    assert tuple(batch["data"].shape) == (2, 20, 64, 64, 4)
```

- [ ] **Step 2: 让 AMP 上下文不再写死 `"cuda"`**

把 `trainer.py` 和 `evaluate.py` 中的 AMP 上下文：

```python
with paddle.amp.autocast(
    "cpu" if params.cpu else "cuda",
    enabled=bool(params.amp),
    dtype=paddle.bfloat16,
):
```

统一改成：

```python
with paddle.amp.autocast(
    enabled=bool(params.amp),
    dtype=paddle.bfloat16,
):
```

原因：Paddle 3.3.0 的 `device_type` 参数实际上不参与设备判断，保留它只会继续给人错误的 CUDA 心智模型。

- [ ] **Step 3: 跑 Trainer / Evaluator smoke**

Run:

```bash
"/home/lkyu/miniconda3/envs/paddletorch/bin/python" -m pytest \
  "prose_fd_paddle/tests/test_training_data_smoke.py::test_trainer_can_build_train_dataloader_on_local_sw64" \
  "prose_fd_paddle/tests/test_training_data_smoke.py::test_evaluator_can_build_validation_dataloader_on_local_sw64" -q
```

Expected:

```text
2 passed
```

---

### Task 6: 做语法与仓库级静态自检

**Files:**
- Modify: none if everything above already通过

- [ ] **Step 1: 对 Paddle 包做语法检查**

Run:

```bash
"/home/lkyu/miniconda3/envs/paddletorch/bin/python" -m py_compile \
  "prose_fd_paddle/main.py" \
  "prose_fd_paddle/trainer.py" \
  "prose_fd_paddle/evaluate.py" \
  "prose_fd_paddle/models/build_model.py" \
  "prose_fd_paddle/utils/misc.py" \
  "prose_fd_paddle/utils/mode.py" \
  "prose_fd_paddle/utils/datapipe_compat.py"
```

Expected:

- 无输出，退出码为 `0`。

- [ ] **Step 2: 跑本次相关测试全集**

Run:

```bash
"/home/lkyu/miniconda3/envs/paddletorch/bin/python" -m pytest \
  "prose_fd_paddle/tests/test_datapipe_compat.py" \
  "prose_fd_paddle/tests/test_runtime_device_helpers.py" \
  "prose_fd_paddle/tests/test_training_data_smoke.py" -q
```

Expected:

- 全部通过。

- [ ] **Step 3: 再扫一遍训练链路中的活跃 CUDA 写死点**

Run:

```bash
rg -n "paddle\\.cuda\\.|\\.cuda\\(|assert paddle\\.cuda\\.is_available|CUDA =" \
  "prose_fd_paddle/main.py" \
  "prose_fd_paddle/trainer.py" \
  "prose_fd_paddle/evaluate.py" \
  "prose_fd_paddle/models/build_model.py" \
  "prose_fd_paddle/utils/misc.py" \
  "prose_fd_paddle/utils/mode.py"
```

Expected:

- 无匹配。

---

### Task 7: 用安全 GPU 探针收敛单卡 dryrun 参数，并给出最终训练命令

**Files:**
- Modify: none unless必须为探针增加临时脚本

- [ ] **Step 1: 先用单步前向反向探针验证最小 batch**

从仓库根目录执行下面这条一次性探针命令，不保存 checkpoint，不跑 epoch 循环：

```bash
"/home/lkyu/miniconda3/envs/paddletorch/bin/python" - <<'PY'
from pathlib import Path
from omegaconf import OmegaConf
import paddle
from prose_fd_paddle.data_utils.collate import custom_collate
from prose_fd_paddle.dataset import get_dataset
from prose_fd_paddle.models.build_model import build_model
from prose_fd_paddle.symbol_utils.environment import SymbolicEnvironment

root = Path("/home/lkyu/baidu/prose-fd")
paddle.set_device("gpu:0")

main_cfg = OmegaConf.load(root / "prose_fd_paddle/configs/main.yaml")
data_cfg = OmegaConf.load(root / "prose_fd_paddle/configs/data/shallow_water_minimal.yaml")
model_cfg = OmegaConf.load(root / "prose_fd_paddle/configs/model/prose_2to1.yaml")
symbol_cfg = OmegaConf.load(root / "prose_fd_paddle/configs/symbol/symbol.yaml")

cfg = OmegaConf.create(OmegaConf.to_container(main_cfg, resolve=False))
cfg.data = data_cfg
cfg.model = model_cfg
cfg.symbol = symbol_cfg
cfg.cpu = 0
cfg.device = "gpu:0"
cfg.reload_model = None
cfg.reload_checkpoint = None
cfg.use_wandb = 0
cfg.compile = 0
cfg.batch_size = 1
cfg.batch_size_eval = 1
cfg.num_workers = 0
cfg.num_workers_eval = 0
cfg.local_rank = 0
cfg.global_rank = 0
cfg.world_size = 1
cfg.n_gpu_per_node = 1
cfg.multi_gpu = 0
cfg.is_master = 1
OmegaConf.resolve(cfg)

symbol_env = SymbolicEnvironment(cfg.symbol)
modules = build_model(cfg, cfg.model, cfg.data, symbol_env)
model = modules["model"]
model.train()

sample = next(iter(get_dataset(cfg, symbol_env, split="train")))
batch = custom_collate(
    cfg.data.max_output_dimension,
    symbol_env.pad_index,
    cfg.data.tie_fields,
    cfg.data.get("mixed_length", 0),
    cfg.input_len,
    cfg.symbol.pad_right,
)([sample])

times = paddle.linspace(0, 10, cfg.data.t_num, dtype=paddle.float32)[None].to("gpu:0")
input_len = cfg.input_len
out = model(
    "fwd",
    data_input=batch["data"][:, :input_len].to("gpu:0"),
    input_times=times[:, :input_len, None],
    output_times=times[:, input_len:, None] - times[:, input_len - 1 : input_len, None],
    symbol_input=batch["symbol_input"].to("gpu:0"),
    symbol_padding_mask=batch["symbol_mask"].to("gpu:0"),
)
loss = out.mean()
loss.backward()
print("output_shape", tuple(out.shape))
print("max_memory_mb", paddle.device.cuda.max_memory_allocated() / 1024**2)
PY
```

Expected:

- `output_shape (1, 10, 64, 64, 4)`
- 显存峰值约 `2.1 GB` 量级

- [ ] **Step 2: 基于探针结果确定 dryrun 参数**

使用以下保守规则：

- 如果 `batch_size=1` 探针在 `3 GB` 以下，则 dryrun 训练命令用 `batch_size=2`
- 如果 `batch_size=1` 探针超过 `3 GB`，则 dryrun 命令退回 `batch_size=1`
- `batch_size_eval` 固定设为 `1`
- `num_workers` 和 `num_workers_eval` 先固定 `0`，避免 HDF5 多进程变量
- `use_wandb=0`
- `log_eval_plots=-1`

- [ ] **Step 3: 给出最终单卡训练验证命令**

执行者最终应把以下命令作为验收命令交付给用户：

```bash
cd "/home/lkyu/baidu/prose-fd/prose_fd_paddle"
CUDA_VISIBLE_DEVICES=0 "/home/lkyu/miniconda3/envs/paddletorch/bin/python" "main.py" \
  dryrun=1 \
  use_wandb=0 \
  data=shallow_water_minimal \
  model=prose_2to1 \
  optim=wsd \
  device=gpu:0 \
  batch_size=2 \
  batch_size_eval=1 \
  num_workers=0 \
  num_workers_eval=0 \
  log_eval_plots=-1 \
  exp_name=sw64_single_gpu \
  exp_id=prose_fd_sw64_dryrun
```

如果第 1 步显存探针不满足 `batch_size=2`，把这里的 `batch_size=2` 改成 `1`，其余参数不变。

- [ ] **Step 4: 不要在本任务里真正启动上面的训练命令**

这条命令只作为最终交付物写入结论，由用户手动执行验收。执行者本轮只允许做探针，不允许启动真实训练循环。

---

## Self-Review

### Spec coverage

- 数据是否必须先转 `128x128`：Task 4 明确采用“直接支持 `64x64` 单数据集”的推荐方案，并给出 `shallow_water_minimal.yaml`。
- 如果不能直接训练，最小方案是什么：Task 2 到 Task 5 先修 dataloader 与设备抽象，再用 Task 7 给出可执行 dryrun 命令。
- 探索训练链路、找明显阻塞点并写单测：Task 1、Task 2、Task 5 覆盖 dataloader、默认模型名、Trainer / Evaluator 构造与前向。
- 去掉 `.cuda()` 这类 N 卡专属逻辑：Task 3 完成运行时设备抽象。
- 结合本机 16GB 显卡给出训练参数建议：Task 7 用 GPU 探针和明确规则收敛出 `batch_size=2` 的保守命令。
- 不直接开始训练：Task 7 最后一步明确禁止启动真实训练。

### Placeholder scan

已检查本计划，不包含写计划时常见的占位符、延后实现标记或“参照上一任务”式省略写法。

### Type consistency

- 新增 helper 名称在全文保持一致：`set_runtime_device`、`get_runtime_device`、`max_memory_allocated_mb`、`to_device`、`to_cuda`
- 新增数据配置名全文统一为 `shallow_water_minimal`
- 最终训练命令与测试里使用的模型名统一为 `prose_2to1`
