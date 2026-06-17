# PROSE-FD Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 `/home/lkyu/baidu/prose-fd/prose_fd_paddle` 里的 Paddle 版 `prose_fd` 模型按 PaddleCFD 现有模型组织规范并入当前仓库，确保可通过 `from import` 导入核心模型，并可通过 `examples` 入口完成单卡 dryrun 训练。

**Architecture:** 采用“两层拆分”方案：将网络定义、符号编码运行时和少量模型运行时兼容代码放入 `ppcfd/models/prose_fd/`；将训练入口、数据管线、Hydra 配置、训练/评估工具、转换脚本和 README 放入 `examples/prose_fd/`。优先保持 `prose_fd_paddle` 原始目录语义，避免大规模重构；只做导入路径、包路径、绝对路径和设备初始化相关的必要修改。

**Tech Stack:** Python 3.10+, PaddlePaddle, Hydra, OmegaConf, h5py, einops, scipy, matplotlib, tabulate, tqdm, editable install (`pip install -e .`)

---

## 先决判断

1. Python 包目录不能使用连字符，因此：
   - `ppcfd/models/` 下使用 `prose_fd`
   - `examples/` 下也建议使用 `prose_fd`
   - README 文案中继续对外写模型名 `prose_fd`
2. 当前 `PaddleCFD` 分支是 `feat/mpp`。实现前需要新建并切换到 `feat/prose_fd`。
3. `prose_fd_paddle` 当前存在以下必须处理的问题：
   - 多处硬编码绝对路径：`sys.path.append("/home/lkyu/baidu/prose-fd/prose_fd_paddle")`
   - 配置中的绝对数据路径：`configs/data/shallow_water_minimal.yaml`
   - 以源码根目录为前提的裸导入和 fallback 导入
   - 工具脚本直接导入 `prose_fd_paddle.*`
4. `PaddleCFD` 的常见模式不是把整个原仓直接塞进 `ppcfd/models`，而是：
   - `ppcfd/models/*` 提供可复用模型实现和必要运行时
   - `examples/*` 放训练编排、配置、数据准备和 README
5. 本计划不包含 `git commit` / `git push`，因为用户明确要求代码完成后先审阅。

## 目标目录与职责划分

### 进入 `ppcfd/models/prose_fd/` 的文件

这些文件组成“可导入的模型运行时最小闭包”，满足 `from ppcfd.models.prose_fd import ...`：

- 来自 `prose_fd_paddle/models/`
  - `attention_utils.py`
  - `build_model.py`
  - `embedder.py`
  - `transformer.py`
  - `transformer_wrappers.py`
  - `__init__.py`
- 来自 `prose_fd_paddle/symbol_utils/`
  - `__init__.py`
  - `encoders.py`
  - `environment.py`
  - `generators.py`
  - `node_utils.py`
- 来自 `prose_fd_paddle/paddle_utils.py`
  - `paddle_utils.py`
- 来自 `prose_fd_paddle/utils/`
  - `rotary_embedding_paddle.py`
  - 从 `misc.py` 中抽出模型运行时需要的最小设备辅助逻辑，建议新建 `runtime.py`

### 进入 `examples/prose_fd/` 的文件

这些文件保留原始训练与数据工作流：

- `main.py`
- `trainer.py`
- `evaluate.py`
- `dataset.py`
- `configs/`
- `data_utils/`
- `scripts/`
- `README.md`
- `tools/convert_torch_ckpt_to_paddle.py`
- `utils/`
  - `logger.py`
  - `mode.py`
  - `metrics.py`
  - `plot.py`
  - `lr_scheduler.py`
  - `dadapt_adan_paddle.py`
  - `custom_optimizer_base.py`
  - `datapipe_compat.py`
  - `misc.py`，但改成 example 侧训练/实验工具，运行时设备辅助应转调 `ppcfd.models.prose_fd.runtime`

### 不进入 PaddleCFD 的文件

- `tools/forward_pretrained_paddle.py`
- `tests/`
- `__pycache__/`
- `.pytest_cache/`
- `checkpoint/`
- 任何本地训练产物

## 依赖结论

对照 `/home/lkyu/baidu/prose-fd/environment.txt` 与当前 `PaddleCFD/requirements.txt`：

- 已被 PaddleCFD 直接覆盖或可接受为现有子集：
  - `hydra-core`
  - `h5py`
  - `einops`
  - `matplotlib`
  - `numpy`
  - `scipy`
  - `tabulate`
  - `tqdm`
- 不应为了 `prose_fd` 这次并入而新增到 PaddleCFD 全局依赖：
  - `torch`
    - 仅 `convert_torch_ckpt_to_paddle.py` 需要
  - `neuraloperator`
    - Paddle 侧 `build_model.py` 已显式拒绝 `fno` / `vit` / `unet` / `deeponet`
- 需要写入 `examples/prose_fd/README.md` 的附加依赖说明：
  - `wandb`
    - 训练入口直接 `import wandb`，但不是当前 `PaddleCFD/requirements.txt` 的一部分
  - `torch`
    - 仅用于权重转换脚本

## 文件结构总表

### 新建目录

- `ppcfd/models/prose_fd/`
- `ppcfd/models/prose_fd/symbol_utils/`
- `examples/prose_fd/`
- `examples/prose_fd/__init__.py`
- `examples/prose_fd/configs/`
- `examples/prose_fd/configs/data/`
- `examples/prose_fd/configs/model/`
- `examples/prose_fd/configs/optim/`
- `examples/prose_fd/configs/symbol/`
- `examples/prose_fd/data_utils/`
- `examples/prose_fd/data_utils/cfdbench/`
- `examples/prose_fd/scripts/`
- `examples/prose_fd/tools/`
- `examples/prose_fd/utils/`

### 需要修改的 PaddleCFD 既有文件

- `ppcfd/models/__init__.py`
- `requirements.txt`
- 如需要补充包导出，再修改：
  - `ppcfd/models/prose_fd/__init__.py`

### 需要新增的 README 位置

- `examples/prose_fd/README.md`

---

### Task 1: 建立开发分支并记录基线

**Files:**
- Modify: 无代码文件修改

- [ ] **Step 1: 记录当前工作区状态**

Run:
```bash
git -C "/home/lkyu/baidu/PaddleCFD" status --short
git -C "/home/lkyu/baidu/PaddleCFD" branch --show-current
```

Expected:
- 能看到当前分支名
- 了解是否存在未提交改动，避免误覆盖

- [ ] **Step 2: 新建并切换到目标分支**

Run:
```bash
git -C "/home/lkyu/baidu/PaddleCFD" checkout -b "feat/prose_fd"
```

Expected:
- 输出 `Switched to a new branch 'feat/prose_fd'`

- [ ] **Step 3: 再次确认当前分支**

Run:
```bash
git -C "/home/lkyu/baidu/PaddleCFD" branch --show-current
```

Expected:
- 输出 `feat/prose_fd`

### Task 2: 建立目标目录骨架并复制源文件

**Files:**
- Create: `ppcfd/models/prose_fd/**`
- Create: `examples/prose_fd/**`

- [ ] **Step 1: 建立目标目录**

Run:
```bash
mkdir -p "/home/lkyu/baidu/PaddleCFD/ppcfd/models/prose_fd/symbol_utils"
mkdir -p "/home/lkyu/baidu/PaddleCFD/examples/prose_fd/configs/data"
mkdir -p "/home/lkyu/baidu/PaddleCFD/examples/prose_fd/configs/model"
mkdir -p "/home/lkyu/baidu/PaddleCFD/examples/prose_fd/configs/optim"
mkdir -p "/home/lkyu/baidu/PaddleCFD/examples/prose_fd/configs/symbol"
mkdir -p "/home/lkyu/baidu/PaddleCFD/examples/prose_fd/data_utils/cfdbench"
mkdir -p "/home/lkyu/baidu/PaddleCFD/examples/prose_fd/scripts"
mkdir -p "/home/lkyu/baidu/PaddleCFD/examples/prose_fd/tools"
mkdir -p "/home/lkyu/baidu/PaddleCFD/examples/prose_fd/utils"
touch "/home/lkyu/baidu/PaddleCFD/examples/prose_fd/__init__.py"
```

Expected:
- 所有目录创建成功，无报错

- [ ] **Step 2: 复制核心模型文件到 `ppcfd/models/prose_fd/`**

Run:
```bash
cp "/home/lkyu/baidu/prose-fd/prose_fd_paddle/models/attention_utils.py" "/home/lkyu/baidu/PaddleCFD/ppcfd/models/prose_fd/attention_utils.py"
cp "/home/lkyu/baidu/prose-fd/prose_fd_paddle/models/build_model.py" "/home/lkyu/baidu/PaddleCFD/ppcfd/models/prose_fd/build_model.py"
cp "/home/lkyu/baidu/prose-fd/prose_fd_paddle/models/embedder.py" "/home/lkyu/baidu/PaddleCFD/ppcfd/models/prose_fd/embedder.py"
cp "/home/lkyu/baidu/prose-fd/prose_fd_paddle/models/transformer.py" "/home/lkyu/baidu/PaddleCFD/ppcfd/models/prose_fd/transformer.py"
cp "/home/lkyu/baidu/prose-fd/prose_fd_paddle/models/transformer_wrappers.py" "/home/lkyu/baidu/PaddleCFD/ppcfd/models/prose_fd/transformer_wrappers.py"
cp "/home/lkyu/baidu/prose-fd/prose_fd_paddle/paddle_utils.py" "/home/lkyu/baidu/PaddleCFD/ppcfd/models/prose_fd/paddle_utils.py"
cp "/home/lkyu/baidu/prose-fd/prose_fd_paddle/utils/rotary_embedding_paddle.py" "/home/lkyu/baidu/PaddleCFD/ppcfd/models/prose_fd/rotary_embedding_paddle.py"
cp "/home/lkyu/baidu/prose-fd/prose_fd_paddle/symbol_utils/"*.py "/home/lkyu/baidu/PaddleCFD/ppcfd/models/prose_fd/symbol_utils/"
```

Expected:
- 目标目录中出现核心模型与符号处理文件

- [ ] **Step 3: 复制 example 训练与工具文件**

Run:
```bash
cp "/home/lkyu/baidu/prose-fd/prose_fd_paddle/main.py" "/home/lkyu/baidu/PaddleCFD/examples/prose_fd/main.py"
cp "/home/lkyu/baidu/prose-fd/prose_fd_paddle/trainer.py" "/home/lkyu/baidu/PaddleCFD/examples/prose_fd/trainer.py"
cp "/home/lkyu/baidu/prose-fd/prose_fd_paddle/evaluate.py" "/home/lkyu/baidu/PaddleCFD/examples/prose_fd/evaluate.py"
cp "/home/lkyu/baidu/prose-fd/prose_fd_paddle/dataset.py" "/home/lkyu/baidu/PaddleCFD/examples/prose_fd/dataset.py"
cp -r "/home/lkyu/baidu/prose-fd/prose_fd_paddle/configs/." "/home/lkyu/baidu/PaddleCFD/examples/prose_fd/configs/"
cp -r "/home/lkyu/baidu/prose-fd/prose_fd_paddle/data_utils/." "/home/lkyu/baidu/PaddleCFD/examples/prose_fd/data_utils/"
cp -r "/home/lkyu/baidu/prose-fd/prose_fd_paddle/scripts/." "/home/lkyu/baidu/PaddleCFD/examples/prose_fd/scripts/"
cp -r "/home/lkyu/baidu/prose-fd/prose_fd_paddle/utils/." "/home/lkyu/baidu/PaddleCFD/examples/prose_fd/utils/"
cp "/home/lkyu/baidu/prose-fd/prose_fd_paddle/tools/convert_torch_ckpt_to_paddle.py" "/home/lkyu/baidu/PaddleCFD/examples/prose_fd/tools/convert_torch_ckpt_to_paddle.py"
cp "/home/lkyu/baidu/prose-fd/prose_fd_paddle/README.md" "/home/lkyu/baidu/PaddleCFD/examples/prose_fd/README.md"
```

Expected:
- example 目录具备原始训练所需脚手架

### Task 3: 在 `ppcfd/models/prose_fd` 建立可导入的库级 API

**Files:**
- Create: `ppcfd/models/prose_fd/__init__.py`
- Create: `ppcfd/models/prose_fd/runtime.py`
- Modify: `ppcfd/models/prose_fd/build_model.py`
- Modify: `ppcfd/models/prose_fd/attention_utils.py`
- Modify: `ppcfd/models/prose_fd/embedder.py`
- Modify: `ppcfd/models/prose_fd/symbol_utils/*.py`

- [ ] **Step 1: 编写 `__init__.py`，导出核心类与工厂**

写入：
```python
from ppcfd.models.prose_fd.build_model import build_model
from ppcfd.models.prose_fd.transformer_wrappers import PROSE_1to1
from ppcfd.models.prose_fd.transformer_wrappers import PROSE_2to1
from ppcfd.models.prose_fd.symbol_utils.environment import SymbolicEnvironment

__all__ = [
    "build_model",
    "PROSE_1to1",
    "PROSE_2to1",
    "SymbolicEnvironment",
]
```

- [ ] **Step 2: 从 `utils/misc.py` 抽出最小运行时设备辅助**

新建 `ppcfd/models/prose_fd/runtime.py`，只保留模型运行时需要的部分：
```python
import paddle


RUNTIME_DEVICE = "cpu"


def set_runtime_device(device: str):
    global RUNTIME_DEVICE
    RUNTIME_DEVICE = device


def get_runtime_device() -> str:
    return RUNTIME_DEVICE


def to_device(*args, use_cpu=False, device: str | None = None):
    target = "cpu" if use_cpu else (device or RUNTIME_DEVICE)
    moved = [None if x is None else x.to(target) for x in args]
    if len(args) == 1:
        return moved[0]
    return moved


def to_cuda(*args, use_cpu=False):
    return to_device(*args, use_cpu=use_cpu)


def get_amp_device_type() -> str:
    return RUNTIME_DEVICE.split(":")[0]


def max_memory_allocated_mb():
    if RUNTIME_DEVICE.startswith("gpu"):
        return paddle.device.cuda.max_memory_allocated() / 1024**2
    return None


def sync_tensor(t):
    if not paddle.distributed.is_initialized():
        return t
    source_place = t.place
    t_sync = t.to(RUNTIME_DEVICE)
    paddle.distributed.barrier()
    paddle.distributed.all_reduce(t_sync, op=paddle.distributed.ReduceOp.SUM)
    return t_sync.to(source_place)
```

- [ ] **Step 3: 修改 `build_model.py` 只依赖 models 包内部路径**

将：
```python
from ..utils.misc import get_runtime_device
from ..paddle_utils import *
from .transformer_wrappers import PROSE_1to1, PROSE_2to1
```

改成：
```python
from ppcfd.models.prose_fd.paddle_utils import *
from ppcfd.models.prose_fd.runtime import get_runtime_device
from ppcfd.models.prose_fd.transformer_wrappers import PROSE_1to1
from ppcfd.models.prose_fd.transformer_wrappers import PROSE_2to1
```

并保留已有的 `_UNPORTED_BASELINE_MODELS` 逻辑，不要重新引入 `neuraloperator`。

- [ ] **Step 4: 修改注意力与嵌入模块的导入**

将 `attention_utils.py` 中的：
```python
from ..utils.rotary_embedding_paddle import RotaryEmbedding
```

改成：
```python
from ppcfd.models.prose_fd.rotary_embedding_paddle import RotaryEmbedding
```

将 `embedder.py`、`symbol_utils/*.py` 内所有“裸导入 fallback”改成稳定的包内相对导入或 `ppcfd.models.prose_fd.*` 绝对导入，删除：
```python
try:
    ...
except ImportError:
    ...
```

Expected:
- `ppcfd.models.prose_fd` 成为可直接导入的稳定包

### Task 4: 清理 example 侧绝对路径与脚本式硬编码

**Files:**
- Modify: `examples/prose_fd/main.py`
- Modify: `examples/prose_fd/trainer.py`
- Modify: `examples/prose_fd/evaluate.py`
- Modify: `examples/prose_fd/dataset.py`
- Modify: `examples/prose_fd/utils/misc.py`
- Modify: `examples/prose_fd/data_utils/collate.py`
- Modify: `examples/prose_fd/data_utils/all_datasets.py`
- Modify: `examples/prose_fd/data_utils/convert_cfdbench.py`
- Modify: `examples/prose_fd/data_utils/cfdbench/save_data.py`
- Modify: `examples/prose_fd/tools/convert_torch_ckpt_to_paddle.py`

- [ ] **Step 1: 删除所有 `sys.path.append("/home/lkyu/...")`**

需要删除的已知位置：
```text
examples/prose_fd/evaluate.py
examples/prose_fd/data_utils/collate.py
examples/prose_fd/utils/misc.py
```

删除后，改为显式包导入。

- [ ] **Step 2: 让 example 入口统一从 `ppcfd.models.prose_fd` 获取核心能力**

`main.py` 目标导入形式：
```python
import hydra
import numpy as np
import paddle
import wandb
from omegaconf import DictConfig, OmegaConf

from ppcfd.models.prose_fd import SymbolicEnvironment
from ppcfd.models.prose_fd import build_model
from ppcfd.models.prose_fd import runtime as prose_runtime

from examples.prose_fd.evaluate import Evaluator, metric_to_header
from examples.prose_fd.trainer import Trainer
from examples.prose_fd.utils.misc import initialize_exp, set_seed
from examples.prose_fd.utils.mode import init_distributed_mode
```

如果直接使用 `examples.prose_fd.*` 导致脚本运行方式不兼容，则采用“双模式入口”：
```python
try:
    from examples.prose_fd.evaluate import Evaluator, metric_to_header
    ...
except ImportError:
    from evaluate import Evaluator, metric_to_header
    ...
```

但核心模型相关导入不要再 fallback 到源码根目录。

- [ ] **Step 3: 让 `trainer.py` / `evaluate.py` 也按同一原则改导入**

`trainer.py` 目标导入形式：
```python
from ppcfd.models.prose_fd.runtime import get_amp_device_type
from ppcfd.models.prose_fd.runtime import max_memory_allocated_mb
from ppcfd.models.prose_fd.runtime import to_cuda

from examples.prose_fd.data_utils.collate import custom_collate
from examples.prose_fd.dataset import get_dataset
from examples.prose_fd.utils.dadapt_adan_paddle import DAdaptAdan
from examples.prose_fd.utils.lr_scheduler import build_lr_scheduler
```

`evaluate.py` 目标导入形式：
```python
from ppcfd.models.prose_fd.paddle_utils import *
from ppcfd.models.prose_fd.runtime import get_amp_device_type
from ppcfd.models.prose_fd.runtime import sync_tensor
from ppcfd.models.prose_fd.runtime import to_cuda

from examples.prose_fd.data_utils.collate import custom_collate
from examples.prose_fd.dataset import get_dataset
from examples.prose_fd.utils.metrics import compute_metrics
from examples.prose_fd.utils.plot import plot_2d_pde, plot_2d_pde_formal
```

- [ ] **Step 4: 精简 `utils/misc.py`，把设备逻辑改为转调模型运行时**

删除这里的绝对路径和 `from paddle_utils import *`。

改成：
```python
import json
import logging
import os
import random
import re
import subprocess
import sys

import numpy as np
import paddle
from omegaconf import OmegaConf

from ppcfd.models.prose_fd.runtime import get_amp_device_type
from ppcfd.models.prose_fd.runtime import get_runtime_device
from ppcfd.models.prose_fd.runtime import max_memory_allocated_mb
from ppcfd.models.prose_fd.runtime import set_runtime_device
from ppcfd.models.prose_fd.runtime import sync_tensor
from ppcfd.models.prose_fd.runtime import to_cuda

from examples.prose_fd.utils.logger import create_logger
```

保留：
- `initialize_exp`
- `get_dump_path`
- `load_json`
- `zip_dic`
- `set_seed`

- [ ] **Step 5: 修复 `data_utils` 中的裸导入**

重点改动：
```python
# data_utils/collate.py
from ppcfd.models.prose_fd.paddle_utils import *

# data_utils/all_datasets.py
from examples.prose_fd.utils.datapipe_compat import IterDataPipe

# data_utils/convert_cfdbench.py
from examples.prose_fd.data_utils.cfdbench import get_auto_dataset

# data_utils/cfdbench/save_data.py
from examples.prose_fd.data_utils.cfdbench import get_auto_dataset
```

`if __name__ == "__main__"` 下的调试导入可以保留，但不要再依赖旧仓根目录。

- [ ] **Step 6: 修复转换脚本的导入与默认路径**

`examples/prose_fd/tools/convert_torch_ckpt_to_paddle.py` 目标导入：
```python
from ppcfd.models.prose_fd import PROSE_2to1
from ppcfd.models.prose_fd import SymbolicEnvironment
```

并把 `ROOT / "prose_fd_paddle" / ...` 改成相对于 `examples/prose_fd/` 的路径，例如：
```python
EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
model_cfg = OmegaConf.load(EXAMPLE_ROOT / "configs" / "model" / "prose_2to1.yaml")
data_cfg = OmegaConf.load(EXAMPLE_ROOT / "configs" / "data" / "fluids.yaml")
symbol_cfg = OmegaConf.load(EXAMPLE_ROOT / "configs" / "symbol" / "symbol.yaml")
```

### Task 5: 接入 PaddleCFD 模型注册并补齐包导出

**Files:**
- Modify: `ppcfd/models/__init__.py`

- [ ] **Step 1: 按现有 graceful fallback 模式注册 `prose_fd`**

在 `ppcfd/models/__init__.py` 末尾追加：
```python
# PROSE-FD
try:
    from ppcfd.models import prose_fd

    __all__.append("prose_fd")
except ImportError:
    pass  # Optional dependency
```

- [ ] **Step 2: 验证顶层导入接口**

目标接口：
```python
from ppcfd.models import prose_fd
from ppcfd.models.prose_fd import PROSE_1to1
from ppcfd.models.prose_fd import PROSE_2to1
from ppcfd.models.prose_fd import SymbolicEnvironment
from ppcfd.models.prose_fd import build_model
```

### Task 6: 调整最小训练配置，移除仓外绝对路径依赖

**Files:**
- Modify: `examples/prose_fd/configs/data/shallow_water_minimal.yaml`
- Modify: `examples/prose_fd/configs/main.yaml`
- Modify: `examples/prose_fd/README.md`

- [ ] **Step 1: 把 `shallow_water_minimal.yaml` 的绝对数据路径改成待用户覆盖的占位值**

将：
```yaml
data_path: /home/lkyu/baidu/prose-fd/dataset/pdebench/2D/shallow-water/2D_rdb_NA_NA.h5
```

改成：
```yaml
data_path: ./data/shallow_water/2D_rdb_NA_NA.h5
```

说明：
- 不要求仓库内实际携带数据
- 训练时允许用户通过命令行覆盖
- 如果本机验证需要直接复用旧仓数据，可在执行命令时临时传入绝对路径

- [ ] **Step 2: 保持 `main.yaml` 的 Hydra 输出行为，但确保 GPU dryrun 参数可覆盖**

这里不建议大改默认配置，只确认这些字段不会阻碍命令行覆盖：
```yaml
hydra:
  output_subdir: null
  run:
    dir: .
```

### Task 7: README 重写为 PaddleCFD 使用方式

**Files:**
- Modify: `examples/prose_fd/README.md`

- [ ] **Step 1: 改写“如何导入模型网络”部分**

README 中加入：
```python
from ppcfd.models.prose_fd import PROSE_2to1
from ppcfd.models.prose_fd import SymbolicEnvironment
from ppcfd.models.prose_fd import build_model
```

并给出最小示例：
```python
from omegaconf import OmegaConf

from ppcfd.models.prose_fd import PROSE_2to1
from ppcfd.models.prose_fd import SymbolicEnvironment

model_cfg = OmegaConf.load("examples/prose_fd/configs/model/prose_2to1.yaml")
data_cfg = OmegaConf.load("examples/prose_fd/configs/data/shallow_water_minimal.yaml")
symbol_cfg = OmegaConf.load("examples/prose_fd/configs/symbol/symbol.yaml")

symbol_env = SymbolicEnvironment(symbol_cfg)
model = PROSE_2to1(
    model_cfg,
    symbol_env,
    data_cfg.x_num,
    data_cfg.max_output_dimension,
    data_cfg.t_num - 10,
)
```

- [ ] **Step 2: 改写“如何启动训练”部分**

README 中加入基于 examples 的训练说明：
```bash
cd "/home/lkyu/baidu/PaddleCFD/examples/prose_fd"
python "main.py" dryrun=1 use_wandb=0 data=shallow_water_minimal model=prose_2to1 optim=wsd device=gpu:0 batch_size=2 batch_size_eval=1 num_workers=0 num_workers_eval=0 log_eval_plots=-1 exp_name=sw64_single_gpu exp_id=prose_fd_sw64_dryrun data.shallow_water.data_path="/home/lkyu/baidu/prose-fd/dataset/pdebench/2D/shallow-water/2D_rdb_NA_NA.h5"
```

- [ ] **Step 3: 删除 README 中的分布式训练部分**

删除原文中类似下面这段：
```markdown
## Distributed training
Distributed training is available via ...
```

原因：
- 用户明确要求删除，因尚未测试

- [ ] **Step 4: 加入权重转换脚本说明**

README 中加入：
```bash
cd "/home/lkyu/baidu/PaddleCFD/examples/prose_fd"
python "tools/convert_torch_ckpt_to_paddle.py" --torch-ckpt "/path/to/model.pth" --paddle-ckpt "/path/to/model.pdparams"
```

并明确说明：
- 该脚本依赖 `torch`
- `tools/forward_pretrained_paddle.py` 不会并入 PaddleCFD

- [ ] **Step 5: 写明超出 PaddleCFD 全局依赖范围的附加依赖**

README 中单列：
```markdown
## Extra dependencies for PROSE-FD

- `wandb`: required only if `use_wandb=1`
- `torch`: required only for `tools/convert_torch_ckpt_to_paddle.py`
```

### Task 8: 更新安装与导入验证

**Files:**
- Modify: `requirements.txt` 仅在确实必须时修改

- [ ] **Step 1: 判断是否需要修改 `requirements.txt`**

执行前先做事实核对：
```bash
python - <<'PY'
import pkgutil
mods = ["hydra", "omegaconf", "h5py", "einops", "matplotlib", "scipy", "tabulate", "tqdm"]
for name in mods:
    print(name, bool(pkgutil.find_loader(name)))
PY
```

处理原则：
- 如果只缺 `wandb`，不要改 `requirements.txt`，写入 README 即可
- 不要为了未支持 baseline 引入 `neuraloperator`
- 不要把 `torch` 加入全局依赖

- [ ] **Step 2: 重新安装本地开发版本 PaddleCFD**

在用于验证的环境中执行：
```bash
python -m pip install -e "/home/lkyu/baidu/PaddleCFD"
```

Expected:
- 安装成功，无 `ImportError`

- [ ] **Step 3: 做导入冒烟测试**

Run:
```bash
python - <<'PY'
import ppcfd
from ppcfd.models import prose_fd
from ppcfd.models.prose_fd import PROSE_1to1, PROSE_2to1, SymbolicEnvironment, build_model
print("IMPORT_OK")
print(prose_fd.__all__ if hasattr(prose_fd, "__all__") else "NO_MODULE_ALL")
print(PROSE_1to1.__name__, PROSE_2to1.__name__, SymbolicEnvironment.__name__, build_model.__name__)
PY
```

Expected:
- 输出 `IMPORT_OK`
- 无绝对路径相关报错
- 无 `ModuleNotFoundError`

### Task 9: 启动最小 dryrun 训练验证

**Files:**
- Modify: 无新增代码修改

- [ ] **Step 1: 在 examples 目录运行用户指定命令**

Run:
```bash
cd "/home/lkyu/baidu/PaddleCFD/examples/prose_fd"
python "main.py" dryrun=1 use_wandb=0 data=shallow_water_minimal model=prose_2to1 optim=wsd device=gpu:0 batch_size=2 batch_size_eval=1 num_workers=0 num_workers_eval=0 log_eval_plots=-1 exp_name=sw64_single_gpu exp_id=prose_fd_sw64_dryrun data.shallow_water.data_path="/home/lkyu/baidu/prose-fd/dataset/pdebench/2D/shallow-water/2D_rdb_NA_NA.h5"
```

Expected:
- 训练可以成功初始化
- 模型、数据集、优化器和评估器构建成功
- 看到至少一次迭代推进或至少一个 `Epoch` / `step` 日志

- [ ] **Step 2: 训练验证通过后记录输出目录，但不要立即删除**

Run:
```bash
find "/home/lkyu/baidu/PaddleCFD/examples/prose_fd" -maxdepth 3 \( -type d -o -type f \) | rg "prose_fd_sw64_dryrun|sw64_single_gpu|checkpoint|evals_all|train.log"
```

Expected:
- 明确列出本次 dryrun 生成的目录与文件

- [ ] **Step 3: 停在清理前，向用户请求确认**

必须使用以下确认格式：
```text
⚠️ Dangerous Operation Detected
Operation Type: 删除 dryrun 训练生成文件
Impact Scope: /home/lkyu/baidu/PaddleCFD/examples/prose_fd 下本次 prose_fd dryrun 生成的 checkpoint、日志和评估输出
Risk Assessment: 删除后将丢失本次运行产物，若路径判断错误可能误删其他实验结果

Please confirm to continue? [requires explicit "yes", "confirm", "continue"]
```

只有在用户明确确认后，才执行删除命令。

### Task 10: 代码卫生检查与交付前核对

**Files:**
- Modify: 所有本次新增/修改文件

- [ ] **Step 1: 搜索是否还残留旧仓硬编码**

Run:
```bash
rg -n "/home/lkyu/baidu/prose-fd|sys.path.append|from prose_fd_paddle|import utils$|from utils\\.|from data_utils\\.|from symbol_utils\\.|from paddle_utils import|from collate import|from cfdbench import" "/home/lkyu/baidu/PaddleCFD/ppcfd/models/prose_fd" "/home/lkyu/baidu/PaddleCFD/examples/prose_fd"
```

Expected:
- 不应再有 `/home/lkyu/baidu/prose-fd`
- 不应再有 `sys.path.append`
- 不应再有 `from prose_fd_paddle`

- [ ] **Step 2: 搜索是否意外带入 `forward_pretrained_paddle.py` 和测试目录**

Run:
```bash
find "/home/lkyu/baidu/PaddleCFD/examples/prose_fd" -path "*/tests/*" -o -name "forward_pretrained_paddle.py"
find "/home/lkyu/baidu/PaddleCFD/ppcfd/models/prose_fd" -path "*/tests/*" -o -name "forward_pretrained_paddle.py"
```

Expected:
- 无输出

- [ ] **Step 3: 搜索是否带入缓存目录**

Run:
```bash
find "/home/lkyu/baidu/PaddleCFD/ppcfd/models/prose_fd" "/home/lkyu/baidu/PaddleCFD/examples/prose_fd" -type d \( -name "__pycache__" -o -name ".pytest_cache" \)
```

Expected:
- 无输出

- [ ] **Step 4: 记录最终交付范围，不做提交**

Run:
```bash
git -C "/home/lkyu/baidu/PaddleCFD" status --short
```

Expected:
- 只显示本次 `prose_fd` 相关新增/修改
- 不执行 `git commit`

## 实施重点说明

### 为什么 `symbol_utils/` 要进入 `ppcfd/models/prose_fd/`

`PROSE_1to1` / `PROSE_2to1` 的构建并不是“纯网络层无外部上下文”，其构造流程依赖 `SymbolicEnvironment`、symbol vocabulary 和 mask/padding 规则。把 `symbol_utils/` 放在 examples 会让 `from ppcfd.models.prose_fd import PROSE_2to1` 仍然缺少核心运行时，破坏用户要求的“从已安装的 PaddleCFD 中通过 from import 导入模型网络”。

### 为什么 `data_utils/`、`trainer.py`、`evaluate.py` 留在 `examples/`

它们服务于训练编排、数据组织、评估和工具脚本，属于 Example 层而不是库层。PaddleCFD 当前多个模型也是这种模式：example 训练脚本通过 `ppcfd.models.*` 调核心模型，但训练过程本身保留在 `examples/`。

### 为什么不引入 baseline 模型

`prose_fd_paddle/models/build_model.py` 已明确：
```python
_UNPORTED_BASELINE_MODELS = frozenset({"fno", "vit", "unet", "deeponet"})
```

Paddle 版本次只承诺 `prose_1to1` / `prose_2to1`。重新打开 baseline 会把 `neuraloperator` 等额外依赖带入 PaddleCFD，违背最小变更和 YAGNI。

## 最终验收标准

满足以下全部条件才算完成：

1. 已在 `feat/prose_fd` 分支上完成修改。
2. `ppcfd/models/prose_fd/` 只包含模型运行时和必要兼容代码，不包含训练脚手架。
3. `examples/prose_fd/` 能独立承载训练入口、配置、数据工具和 README。
4. 代码中不残留旧仓绝对路径、`sys.path.append`、或 `from prose_fd_paddle` 导入。
5. 已完成本地 `pip install -e .`。
6. 已通过 `from ppcfd.models.prose_fd import ...` 导入测试。
7. 已通过用户指定的 dryrun 单卡训练命令，并看到正常迭代。
8. README 已更新：
   - 包含 from-import 用法
   - 包含 example 训练命令
   - 删除分布式训练部分
   - 包含转换脚本说明
   - 包含额外依赖说明
9. 训练输出清理前已等待用户确认。
10. 未进行 `git commit` 或 `git push`。

## 自检结论

- 需求覆盖：已覆盖分支创建、源码探索、模型/外围拆分、必要导入修改、安装导入验证、训练 dryrun、README 改写、依赖审查、清理确认、禁止提交。
- 占位符扫描：无 `TODO` / `TBD` / “稍后实现”等占位描述。
- 一致性检查：
  - 统一使用目标目录名 `prose_fd`
  - 统一把核心导入落到 `ppcfd.models.prose_fd`
  - 统一把训练与数据入口落到 `examples/prose_fd`
