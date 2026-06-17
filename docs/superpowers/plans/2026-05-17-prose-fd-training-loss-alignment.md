# Prose-FD Training Loss Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立 Torch/Paddle 训练 loss 对齐的可复现实验链路，定位并消除当前训练曲线差异中的非模型实现因素。

**Architecture:** 先用只读诊断确认不受控变量，再加入默认关闭的 debug 工具和少量测试。默认训练逻辑保持不变；严格对齐训练只通过显式参数加载同一份 64x64 初始权重，并固定 `base_seed`、batch 顺序、AMP、梯度裁剪和优化器变量。

**Tech Stack:** Python, PyTorch, PaddlePaddle, Hydra/OmegaConf, pytest, h5py, NumPy.

---

## 0. 已确认事实与根因假设

### 当前证据

- 用户训练命令没有传 `base_seed`。两侧 `configs/main.yaml` 默认 `base_seed: -1`，会在 `initialize_exp()` 中各自调用 `np.random.randint(...)` 生成训练种子，然后才 `set_seed(params.base_seed)`。
- 用户训练命令没有传 `reload_model`。因此 5 epoch 日志不是基于已验证过的 `models/prose_fd_torch/prose_fd.pth` 和 `models/prose_fd_paddle/prose_fd_converted.pdparams` 继续训练，而是两侧各自随机初始化的 64x64 模型。
- 已做只读诊断：在 `base_seed=123`、`batch_size=4`、`num_workers=0`、`shallow_water_minimal` 条件下，Torch/Paddle 前 45 个训练 batch 的 `data` SHA256 前缀完全一致，覆盖一次 160 样本训练集循环后继续迭代。
- 已做只读诊断：在同一 `base_seed=123` 下新建 64x64 模型，两侧 fresh state hash 不同。Torch: `208cdf7062f4feea`；Paddle: `0a3b1b22259d3e3c`。这证明相同 seed 不能保证跨框架随机初始化一致。
- 已做只读诊断：现有 Paddle 预训练权重是 128x128 配置转换结果，直接加载到 64x64 `shallow_water_minimal` 模型会出现 shape mismatch：`embedder.conv_proj.0.weight` 期望 `[1024, 4, 8, 8]`，权重为 `[1024, 4, 16, 16]`；`embedder.post_proj.1.weight` 期望 `[1024, 32, 4, 4]`，权重为 `[1024, 32, 8, 8]`。
- 已做只读诊断：同一 batch 的 `meanvar` 归一化在两侧不是 bitwise 一致。差异很小，但需要在共享初始权重后继续量化，不应先作为主因修复。

### 当前优先级判断

第一优先级根因是训练未共享初始权重，第二优先级是不固定 `base_seed`。数据 batch 在固定 seed 下已证明可对齐。后续所有修复必须先验证共享初始权重和固定 batch 后的 loss，再逐步打开 AMP、clip grad 和完整 epoch。

## 1. 文件结构

### 新增文件

- `tools/debug_loss_alignment/prose_fd_config.py`: 两侧 debug 脚本共享的配置构造与 hash 工具，不导入 torch/paddle。
- `tools/debug_loss_alignment/dump_training_batches.py`: 按 backend 导出训练 batch hash，验证数据顺序和内容。
- `tools/debug_loss_alignment/export_torch_initial_state.py`: 用 Torch 构造 64x64 模型初始态，保存为可被转换工具读取的 checkpoint。
- `tools/debug_loss_alignment/run_training_probe.py`: 在单一 backend 内运行少量训练 step，输出 JSONL loss、lr 和可选参数 hash。
- `tools/debug_loss_alignment/compare_jsonl.py`: 比较 Torch/Paddle probe 输出并给出阈值判断。
- `tools/debug_loss_alignment/README.md`: 最终中文工作报告，执行完真实训练验证后填写。

### 修改文件

- `prose_fd_paddle/tools/convert_torch_ckpt_to_paddle.py`: 增加 `--data-config`、`--model-config`、`--symbol-config`、`--input-len` 参数，支持把 64x64 Torch 初始态转换为 Paddle 初始态。
- `prose_fd/tests/test_training_alignment_debug.py`: Torch 侧 debug 数据与初始态导出单测。
- `prose_fd_paddle/tests/test_training_alignment_debug.py`: Paddle 侧 debug 数据、转换工具和 64x64 权重加载单测。

### 不修改文件

- 默认训练主逻辑先不改。`reload_model`、`base_seed`、`amp=0/1`、`clip_grad_norm` 已能通过参数控制。
- 如果共享初始态后仍无法对齐，再按 Task 6 的顺序决定是否增加默认关闭的 debug 参数。

## 2. Task 1: 添加共享配置与 batch dump 工具

**Files:**
- Create: `tools/debug_loss_alignment/prose_fd_config.py`
- Create: `tools/debug_loss_alignment/dump_training_batches.py`
- Test: `prose_fd/tests/test_training_alignment_debug.py`
- Test: `prose_fd_paddle/tests/test_training_alignment_debug.py`

- [ ] **Step 1: 创建共享配置工具**

写入 `tools/debug_loss_alignment/prose_fd_config.py`：

```python
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[2]


def tensor_hash(array: Any) -> str:
    if hasattr(array, "detach"):
        array = array.detach().cpu().numpy()
    elif hasattr(array, "numpy"):
        array = array.numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def load_backend_cfg(backend: str, *, base_seed: int, batch_size: int, max_epoch: int, n_steps_per_epoch: int):
    if backend not in {"torch", "paddle"}:
        raise ValueError(f"Unknown backend: {backend}")

    code_dir = "prose_fd" if backend == "torch" else "prose_fd_paddle"
    cfg = OmegaConf.create(OmegaConf.to_container(OmegaConf.load(ROOT / code_dir / "configs" / "main.yaml"), resolve=False))
    cfg.data = OmegaConf.load(ROOT / code_dir / "configs" / "data" / "shallow_water_minimal.yaml")
    cfg.model = OmegaConf.load(ROOT / code_dir / "configs" / "model" / "prose_2to1.yaml")
    cfg.symbol = OmegaConf.load(ROOT / code_dir / "configs" / "symbol" / "symbol.yaml")
    cfg.optim = OmegaConf.load(ROOT / code_dir / "configs" / "optim" / "wsd.yaml")

    cfg.cpu = 1
    cfg.device = "cpu"
    cfg.runtime_device = "cpu"
    cfg.compile = 0
    cfg.amp = 0
    cfg.reload_model = None
    cfg.reload_checkpoint = None
    cfg.use_wandb = 0
    cfg.num_workers = 0
    cfg.num_workers_eval = 0
    cfg.batch_size = batch_size
    cfg.batch_size_eval = max(1, batch_size)
    cfg.max_epoch = max_epoch
    cfg.n_steps_per_epoch = n_steps_per_epoch
    cfg.overfit_test = 0
    cfg.eval_only = 0
    cfg.rollout = 0
    cfg.local_rank = 0
    cfg.global_rank = 0
    cfg.n_gpu_per_node = 1
    cfg.n_nodes = 1
    cfg.node_id = 0
    cfg.world_size = 1
    cfg.multi_gpu = 0
    cfg.multi_node = 0
    cfg.is_master = 1
    cfg.dump_path = "/tmp/prose_fd_alignment_debug"
    cfg.eval_dump_path = "/tmp/prose_fd_alignment_debug/evals_all"
    cfg.base_seed = base_seed
    cfg.test_seed = 42
    cfg.data.types = ["shallow_water"]
    cfg.data.x_num = 64
    cfg.data.shallow_water.x_num = 64
    cfg.data.shallow_water.data_path = str(ROOT / "dataset" / "pdebench" / "2D" / "shallow-water" / "2D_rdb_NA_NA.h5")
    cfg.accumulate_gradients = cfg.get("accumulate_gradients", 1)
    cfg.optim.max_iters = cfg.max_epoch * cfg.n_steps_per_epoch // cfg.accumulate_gradients
    if cfg.optim.warmup is not None and cfg.optim.warmup < 1:
        cfg.optim.warmup = max(1, int(cfg.optim.warmup * cfg.max_epoch * cfg.n_steps_per_epoch // cfg.accumulate_gradients))
    OmegaConf.resolve(cfg)
    return cfg


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
```

- [ ] **Step 2: 创建 batch dump 脚本**

写入 `tools/debug_loss_alignment/dump_training_batches.py`，要求只在 backend 分支内导入对应框架：

```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from prose_fd_config import ROOT, load_backend_cfg, tensor_hash, write_jsonl


def dump_torch(steps: int, base_seed: int, batch_size: int) -> list[dict]:
    sys.path.insert(0, str(ROOT / "prose_fd"))
    import torch
    from data_utils.collate import custom_collate
    from dataset import get_dataset
    from symbol_utils.environment import SymbolicEnvironment
    from utils.misc import set_seed

    cfg = load_backend_cfg("torch", base_seed=base_seed, batch_size=batch_size, max_epoch=1, n_steps_per_epoch=steps)
    set_seed(cfg.base_seed)
    symbol_env = SymbolicEnvironment(cfg.symbol)
    loader = torch.utils.data.DataLoader(
        get_dataset(cfg, symbol_env, split="train"),
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
    rows = []
    for step, batch in zip(range(steps), loader):
        rows.append(
            {
                "backend": "torch",
                "step": step,
                "data_hash": tensor_hash(batch["data"]),
                "symbol_hash": tensor_hash(batch["symbol_input"]),
                "data_shape": list(batch["data"].shape),
                "data_sum": float(batch["data"].sum().item()),
            }
        )
    return rows


def dump_paddle(steps: int, base_seed: int, batch_size: int) -> list[dict]:
    sys.path.insert(0, str(ROOT))
    import paddle
    from prose_fd_paddle.data_utils.collate import custom_collate
    from prose_fd_paddle.dataset import get_dataset
    from prose_fd_paddle.symbol_utils.environment import SymbolicEnvironment
    from prose_fd_paddle.utils.misc import set_runtime_device, set_seed

    cfg = load_backend_cfg("paddle", base_seed=base_seed, batch_size=batch_size, max_epoch=1, n_steps_per_epoch=steps)
    set_runtime_device("cpu")
    set_seed(cfg.base_seed)
    symbol_env = SymbolicEnvironment(cfg.symbol)
    loader = paddle.io.DataLoader(
        dataset=get_dataset(cfg, symbol_env, split="train"),
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
    rows = []
    for step, batch in zip(range(steps), loader):
        rows.append(
            {
                "backend": "paddle",
                "step": step,
                "data_hash": tensor_hash(batch["data"]),
                "symbol_hash": tensor_hash(batch["symbol_input"]),
                "data_shape": list(batch["data"].shape),
                "data_sum": float(batch["data"].sum().item()),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["torch", "paddle"], required=True)
    parser.add_argument("--steps", type=int, default=45)
    parser.add_argument("--base-seed", type=int, default=123)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = dump_torch(args.steps, args.base_seed, args.batch_size) if args.backend == "torch" else dump_paddle(args.steps, args.base_seed, args.batch_size)
    write_jsonl(args.out, rows)
    print(args.out)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 运行 batch dump 验证**

Run:

```bash
cd "/home/lkyu/baidu/prose-fd"
"/home/lkyu/miniconda3/envs/py312_torch291_cu128/bin/python" "tools/debug_loss_alignment/dump_training_batches.py" --backend torch --steps 45 --base-seed 123 --batch-size 4 --out "/tmp/prose_fd_alignment/torch_batches.jsonl"
"/home/lkyu/miniconda3/envs/paddletorch/bin/python" "tools/debug_loss_alignment/dump_training_batches.py" --backend paddle --steps 45 --base-seed 123 --batch-size 4 --out "/tmp/prose_fd_alignment/paddle_batches.jsonl"
```

Expected: 两个 JSONL 除 `backend` 字段外，`step/data_hash/symbol_hash/data_shape/data_sum` 全部一致。已知 step 0 `data_hash` 应以 `dea2d31259a77832` 开头。

## 3. Task 2: 生成 64x64 共享初始权重

**Files:**
- Create: `tools/debug_loss_alignment/export_torch_initial_state.py`
- Modify: `prose_fd_paddle/tools/convert_torch_ckpt_to_paddle.py`
- Test: `prose_fd/tests/test_training_alignment_debug.py`
- Test: `prose_fd_paddle/tests/test_training_alignment_debug.py`

- [ ] **Step 1: 创建 Torch 初始态导出脚本**

写入 `tools/debug_loss_alignment/export_torch_initial_state.py`：

```python
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

from prose_fd_config import ROOT, load_backend_cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base-seed", type=int, default=123)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT / "prose_fd"))
    import torch
    from models.build_model import build_model
    from symbol_utils.environment import SymbolicEnvironment
    from utils.misc import set_seed

    cfg = load_backend_cfg("torch", base_seed=args.base_seed, batch_size=args.batch_size, max_epoch=1, n_steps_per_epoch=1)
    cfg.reload_model = None
    set_seed(cfg.base_seed)
    symbol_env = SymbolicEnvironment(cfg.symbol)
    modules = build_model(cfg, cfg.model, cfg.data, symbol_env)
    state = modules["model"].state_dict()

    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu()
        digest.update(name.encode("utf-8"))
        digest.update(tensor.numpy().tobytes())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": state, "meta": {"base_seed": args.base_seed, "state_hash": digest.hexdigest()}}, args.out)
    print({"path": str(args.out), "state_hash": digest.hexdigest(), "keys": len(state)})


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 泛化 Paddle 转换工具的模型配置**

修改 `prose_fd_paddle/tools/convert_torch_ckpt_to_paddle.py`：

```python
def build_paddle_model(
    *,
    data_config: Path | None = None,
    model_config: Path | None = None,
    symbol_config: Path | None = None,
    input_len: int = 10,
):
    model_cfg = OmegaConf.load(model_config or ROOT / "prose_fd_paddle" / "configs" / "model" / "prose_2to1.yaml")
    data_cfg = OmegaConf.load(data_config or ROOT / "prose_fd_paddle" / "configs" / "data" / "fluids.yaml")
    symbol_cfg = OmegaConf.load(symbol_config or ROOT / "prose_fd_paddle" / "configs" / "symbol" / "symbol.yaml")
    OmegaConf.resolve(model_cfg)
    OmegaConf.resolve(data_cfg)
    OmegaConf.resolve(symbol_cfg)
    symbol_env = SymbolicEnvironment(symbol_cfg)
    model = PROSE_2to1(
        model_cfg,
        symbol_env,
        data_cfg.x_num,
        data_cfg.max_output_dimension,
        data_cfg.t_num - input_len,
    )
    return model
```

同时修改 `convert_checkpoint()` 签名：

```python
def convert_checkpoint(
    torch_ckpt: Path,
    paddle_ckpt: Path,
    *,
    data_config: Path | None = None,
    model_config: Path | None = None,
    symbol_config: Path | None = None,
    input_len: int = 10,
):
    torch_payload = torch.load(torch_ckpt, map_location="cpu")
    torch_state = {clean_torch_key(k): v.detach().cpu().numpy() for k, v in torch_payload["model"].items()}
    paddle_model = build_paddle_model(
        data_config=data_config,
        model_config=model_config,
        symbol_config=symbol_config,
        input_len=input_len,
    )
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
```

最后给 CLI 增加参数并传入：

```python
parser.add_argument("--data-config", type=Path, default=None)
parser.add_argument("--model-config", type=Path, default=None)
parser.add_argument("--symbol-config", type=Path, default=None)
parser.add_argument("--input-len", type=int, default=10)
...
convert_checkpoint(
    args.torch_ckpt,
    args.paddle_ckpt,
    data_config=args.data_config,
    model_config=args.model_config,
    symbol_config=args.symbol_config,
    input_len=args.input_len,
)
```

- [ ] **Step 3: 生成两侧可加载的 64x64 初始态**

Run:

```bash
cd "/home/lkyu/baidu/prose-fd"
"/home/lkyu/miniconda3/envs/py312_torch291_cu128/bin/python" "tools/debug_loss_alignment/export_torch_initial_state.py" --base-seed 123 --batch-size 4 --out "/tmp/prose_fd_alignment/initial_state/torch_initial_sw64.pth"
"/home/lkyu/miniconda3/envs/paddletorch/bin/python" "prose_fd_paddle/tools/convert_torch_ckpt_to_paddle.py" --torch-ckpt "/tmp/prose_fd_alignment/initial_state/torch_initial_sw64.pth" --paddle-ckpt "/tmp/prose_fd_alignment/initial_state/paddle_initial_sw64.pdparams" --data-config "prose_fd_paddle/configs/data/shallow_water_minimal.yaml" --model-config "prose_fd_paddle/configs/model/prose_2to1.yaml" --symbol-config "prose_fd_paddle/configs/symbol/symbol.yaml" --input-len 10
```

Expected: 不出现 shape mismatch；Paddle `build_model(... reload_model=...)` 能加载 `/tmp/prose_fd_alignment/initial_state/paddle_initial_sw64.pdparams`。

## 4. Task 3: 添加训练 probe 脚本

**Files:**
- Create: `tools/debug_loss_alignment/run_training_probe.py`
- Create: `tools/debug_loss_alignment/compare_jsonl.py`

- [ ] **Step 1: 创建训练 probe 脚本**

写入 `tools/debug_loss_alignment/run_training_probe.py`。脚本职责：构建指定 backend、加载共享初始态、运行少量 `trainer.iter()`，每 step 输出 loss 和 lr，不调用评估、不保存 checkpoint。

关键实现要求：

```python
parser.add_argument("--backend", choices=["torch", "paddle"], required=True)
parser.add_argument("--reload-model", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--steps", type=int, default=40)
parser.add_argument("--base-seed", type=int, default=123)
parser.add_argument("--batch-size", type=int, default=4)
parser.add_argument("--amp", type=int, default=0)
parser.add_argument("--clip-grad-norm", type=float, default=0.0)
parser.add_argument("--device", default="cpu")
```

运行逻辑必须满足：

```python
cfg = load_backend_cfg(args.backend, base_seed=args.base_seed, batch_size=args.batch_size, max_epoch=1, n_steps_per_epoch=args.steps)
cfg.reload_model = str(args.reload_model)
cfg.amp = args.amp
cfg.clip_grad_norm = args.clip_grad_norm
cfg.cpu = int(args.device == "cpu")
cfg.device = args.device
cfg.runtime_device = args.device
cfg.dump_path = f"/tmp/prose_fd_alignment/probe_{args.backend}"
```

每一步输出行格式：

```json
{"backend": "torch", "step": 0, "loss": 0.123456789, "lr": 0.0001}
```

Torch 分支导入：

```python
sys.path.insert(0, str(ROOT / "prose_fd"))
from models.build_model import build_model
from symbol_utils.environment import SymbolicEnvironment
from trainer import Trainer
from utils.misc import set_seed
```

Paddle 分支导入：

```python
sys.path.insert(0, str(ROOT))
from prose_fd_paddle.models.build_model import build_model
from prose_fd_paddle.symbol_utils.environment import SymbolicEnvironment
from prose_fd_paddle.trainer import Trainer
from prose_fd_paddle.utils.misc import set_runtime_device, set_seed
```

loss 读取方式：

```python
before = trainer.data_loss
trainer.iter()
loss = trainer.data_loss - before
trainer.data_loss = 0.0
```

- [ ] **Step 2: 创建 JSONL 比较脚本**

写入 `tools/debug_loss_alignment/compare_jsonl.py`：

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--torch", type=Path, required=True)
    parser.add_argument("--paddle", type=Path, required=True)
    parser.add_argument("--metric", default="loss")
    parser.add_argument("--threshold", type=float, default=1e-4)
    args = parser.parse_args()

    torch_rows = read_jsonl(args.torch)
    paddle_rows = read_jsonl(args.paddle)
    if len(torch_rows) != len(paddle_rows):
        raise SystemExit(f"row count mismatch: torch={len(torch_rows)} paddle={len(paddle_rows)}")

    max_diff = 0.0
    for t_row, p_row in zip(torch_rows, paddle_rows):
        if t_row["step"] != p_row["step"]:
            raise SystemExit(f"step mismatch: {t_row['step']} != {p_row['step']}")
        diff = abs(float(t_row[args.metric]) - float(p_row[args.metric]))
        max_diff = max(max_diff, diff)
        print({"step": t_row["step"], "torch": t_row[args.metric], "paddle": p_row[args.metric], "abs_diff": diff})

    print({"metric": args.metric, "max_abs_diff": max_diff, "threshold": args.threshold})
    if max_diff > args.threshold:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
```

## 5. Task 4: 以 TDD 补充两侧单测

**Files:**
- Modify/Create: `prose_fd/tests/test_training_alignment_debug.py`
- Modify/Create: `prose_fd_paddle/tests/test_training_alignment_debug.py`

- [ ] **Step 1: Torch 侧单测**

写入 `prose_fd/tests/test_training_alignment_debug.py`：

```python
from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PYTHON = "/home/lkyu/miniconda3/envs/py312_torch291_cu128/bin/python"


def test_torch_batch_dump_is_deterministic(tmp_path):
    out = tmp_path / "torch_batches.jsonl"
    subprocess.run(
        [
            PYTHON,
            str(ROOT / "tools" / "debug_loss_alignment" / "dump_training_batches.py"),
            "--backend",
            "torch",
            "--steps",
            "3",
            "--base-seed",
            "123",
            "--batch-size",
            "4",
            "--out",
            str(out),
        ],
        cwd=ROOT,
        check=True,
    )
    text = out.read_text(encoding="utf-8")
    assert "dea2d31259a77832" in text
    assert "b621926695ad17eb" in text


def test_torch_initial_state_export_creates_checkpoint(tmp_path):
    out = tmp_path / "torch_initial_sw64.pth"
    subprocess.run(
        [
            PYTHON,
            str(ROOT / "tools" / "debug_loss_alignment" / "export_torch_initial_state.py"),
            "--base-seed",
            "123",
            "--batch-size",
            "4",
            "--out",
            str(out),
        ],
        cwd=ROOT,
        check=True,
    )
    assert out.is_file()
```

- [ ] **Step 2: Paddle 侧单测**

写入 `prose_fd_paddle/tests/test_training_alignment_debug.py`：

```python
from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PYTHON = "/home/lkyu/miniconda3/envs/paddletorch/bin/python"
TORCH_PYTHON = "/home/lkyu/miniconda3/envs/py312_torch291_cu128/bin/python"


def test_paddle_batch_dump_matches_known_torch_hash(tmp_path):
    out = tmp_path / "paddle_batches.jsonl"
    subprocess.run(
        [
            PYTHON,
            str(ROOT / "tools" / "debug_loss_alignment" / "dump_training_batches.py"),
            "--backend",
            "paddle",
            "--steps",
            "3",
            "--base-seed",
            "123",
            "--batch-size",
            "4",
            "--out",
            str(out),
        ],
        cwd=ROOT,
        check=True,
    )
    text = out.read_text(encoding="utf-8")
    assert "dea2d31259a77832" in text
    assert "b621926695ad17eb" in text


def test_convert_sw64_torch_initial_state_to_paddle(tmp_path):
    torch_ckpt = tmp_path / "torch_initial_sw64.pth"
    paddle_ckpt = tmp_path / "paddle_initial_sw64.pdparams"
    subprocess.run(
        [
            TORCH_PYTHON,
            str(ROOT / "tools" / "debug_loss_alignment" / "export_torch_initial_state.py"),
            "--base-seed",
            "123",
            "--batch-size",
            "4",
            "--out",
            str(torch_ckpt),
        ],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        [
            PYTHON,
            str(ROOT / "prose_fd_paddle" / "tools" / "convert_torch_ckpt_to_paddle.py"),
            "--torch-ckpt",
            str(torch_ckpt),
            "--paddle-ckpt",
            str(paddle_ckpt),
            "--data-config",
            str(ROOT / "prose_fd_paddle" / "configs" / "data" / "shallow_water_minimal.yaml"),
            "--model-config",
            str(ROOT / "prose_fd_paddle" / "configs" / "model" / "prose_2to1.yaml"),
            "--symbol-config",
            str(ROOT / "prose_fd_paddle" / "configs" / "symbol" / "symbol.yaml"),
            "--input-len",
            "10",
        ],
        cwd=ROOT,
        check=True,
    )
    assert paddle_ckpt.is_file()
```

- [ ] **Step 3: 运行单测**

Run:

```bash
cd "/home/lkyu/baidu/prose-fd"
"/home/lkyu/miniconda3/envs/py312_torch291_cu128/bin/python" -m pytest "prose_fd/tests/test_training_alignment_debug.py" -v
"/home/lkyu/miniconda3/envs/paddletorch/bin/python" -m pytest "prose_fd_paddle/tests/test_training_alignment_debug.py" -v
```

Expected: 两侧新增测试全部通过。

## 6. Task 5: 控制变量训练验证

**Files:**
- No production code changes unless this task finds a confirmed root cause.

- [ ] **Step 1: FP32、无裁剪、40 step probe**

Run:

```bash
cd "/home/lkyu/baidu/prose-fd"
"/home/lkyu/miniconda3/envs/py312_torch291_cu128/bin/python" "tools/debug_loss_alignment/run_training_probe.py" --backend torch --reload-model "/tmp/prose_fd_alignment/initial_state/torch_initial_sw64.pth" --out "/tmp/prose_fd_alignment/probe_torch_fp32.jsonl" --steps 40 --base-seed 123 --batch-size 4 --amp 0 --clip-grad-norm 0 --device "cpu"
"/home/lkyu/miniconda3/envs/paddletorch/bin/python" "tools/debug_loss_alignment/run_training_probe.py" --backend paddle --reload-model "/tmp/prose_fd_alignment/initial_state/paddle_initial_sw64.pdparams" --out "/tmp/prose_fd_alignment/probe_paddle_fp32.jsonl" --steps 40 --base-seed 123 --batch-size 4 --amp 0 --clip-grad-norm 0 --device "cpu"
"/home/lkyu/miniconda3/envs/paddletorch/bin/python" "tools/debug_loss_alignment/compare_jsonl.py" --torch "/tmp/prose_fd_alignment/probe_torch_fp32.jsonl" --paddle "/tmp/prose_fd_alignment/probe_paddle_fp32.jsonl" --metric loss --threshold 1e-3
```

Expected: step 0 loss 应接近；40 step 最大差异应足够小。若差异大于 `1e-3`，不要继续启用 AMP 或 clip，进入 Task 6。

- [ ] **Step 2: FP32、启用梯度裁剪**

Run:

```bash
cd "/home/lkyu/baidu/prose-fd"
"/home/lkyu/miniconda3/envs/py312_torch291_cu128/bin/python" "tools/debug_loss_alignment/run_training_probe.py" --backend torch --reload-model "/tmp/prose_fd_alignment/initial_state/torch_initial_sw64.pth" --out "/tmp/prose_fd_alignment/probe_torch_clip.jsonl" --steps 40 --base-seed 123 --batch-size 4 --amp 0 --clip-grad-norm 1.0 --device "cpu"
"/home/lkyu/miniconda3/envs/paddletorch/bin/python" "tools/debug_loss_alignment/run_training_probe.py" --backend paddle --reload-model "/tmp/prose_fd_alignment/initial_state/paddle_initial_sw64.pdparams" --out "/tmp/prose_fd_alignment/probe_paddle_clip.jsonl" --steps 40 --base-seed 123 --batch-size 4 --amp 0 --clip-grad-norm 1.0 --device "cpu"
"/home/lkyu/miniconda3/envs/paddletorch/bin/python" "tools/debug_loss_alignment/compare_jsonl.py" --torch "/tmp/prose_fd_alignment/probe_torch_clip.jsonl" --paddle "/tmp/prose_fd_alignment/probe_paddle_clip.jsonl" --metric loss --threshold 2e-3
```

Expected: 如果 Step 1 通过但 Step 2 失败，根因在 `clip_grad_norm_` 语义或梯度范数差异，补充记录每步 clip 前 global norm。

- [ ] **Step 3: GPU BF16 AMP probe**

Run only when GPU is available:

```bash
cd "/home/lkyu/baidu/prose-fd"
"/home/lkyu/miniconda3/envs/py312_torch291_cu128/bin/python" "tools/debug_loss_alignment/run_training_probe.py" --backend torch --reload-model "/tmp/prose_fd_alignment/initial_state/torch_initial_sw64.pth" --out "/tmp/prose_fd_alignment/probe_torch_amp.jsonl" --steps 40 --base-seed 123 --batch-size 4 --amp 1 --clip-grad-norm 1.0 --device "gpu:0"
"/home/lkyu/miniconda3/envs/paddletorch/bin/python" "tools/debug_loss_alignment/run_training_probe.py" --backend paddle --reload-model "/tmp/prose_fd_alignment/initial_state/paddle_initial_sw64.pdparams" --out "/tmp/prose_fd_alignment/probe_paddle_amp.jsonl" --steps 40 --base-seed 123 --batch-size 4 --amp 1 --clip-grad-norm 1.0 --device "gpu:0"
"/home/lkyu/miniconda3/envs/paddletorch/bin/python" "tools/debug_loss_alignment/compare_jsonl.py" --torch "/tmp/prose_fd_alignment/probe_torch_amp.jsonl" --paddle "/tmp/prose_fd_alignment/probe_paddle_amp.jsonl" --metric loss --threshold 5e-3
```

Expected: 若 FP32 对齐而 AMP 不对齐，默认训练逻辑不立刻改；先把最终对齐命令固定为 `amp=0`，并在报告中说明 BF16 AMP 是剩余差异来源。若用户要求 AMP 对齐，再单独开计划排查 autocast allowlist 和 GradScaler。

## 7. Task 6: 若 FP32 probe 仍失败，按顺序定位

**Files:**
- Modify only after each hypothesis is confirmed by probe.

- [ ] **Step 1: 验证 prepare_data/normalize_data**

扩展 `run_training_probe.py` 增加 `--dump-prepared-batch`，输出 `data_input/data_label/data_mask/input_times/output_times/symbol_input/symbol_padding_mask/mean/std` 的 hash、mean、max、min。先比较 step 0，不训练。

Expected: raw batch 已知一致；如果 prepared batch 差异超过 `1e-5`，优先比较 `paddle.std` 与 `torch.std` 的 correction、dtype promotion 和 reduction 行为。只允许在 Paddle 侧用默认关闭参数修复，例如 `debug_normalize_with_numpy=1`；默认不改变生产训练。

- [ ] **Step 2: 验证 forward/loss**

在共享初始态、同一 prepared batch 上，只跑 forward 和 `data_loss_fn`，不 backward。输出 `data_output_hash`、`data_loss`。

Expected: 若 loss 差异大，复用已存在的 forward 对齐方法定位层级；不要改优化器或数据。

- [ ] **Step 3: 验证 backward 梯度**

选择这些参数输出 grad hash 和 grad norm：

```text
embedder.conv_proj.0.weight
data_encoder.transformer_encoder.layers.0.self_attn.linear_q.weight
fusion.transformer_encoder.layers.0.linear1.weight
data_decoder.transformer_decoder.0.multihead_attn.linear_q.weight
```

Expected: 若 forward/loss 对齐但梯度不对齐，定位对应 Paddle autograd API 或自定义 attention 实现。

- [ ] **Step 4: 验证 AdamW 单步更新**

在 `amp=0`、`clip_grad_norm=0` 条件下比较同一参数的 `param_after - param_before`。若差异只在 `1e-6` 到 `1e-4` 级别，记录为框架优化器差异；若达到 `1e-3` 以上，单独写最小 AdamW primitive test。

## 8. Task 7: 正式 5 epoch 对齐命令

**Files:**
- No code change.

前置条件：Task 5 的 probe 已通过，且 `/tmp/prose_fd_alignment/initial_state/*` 已生成。

- [ ] **Step 1: Torch 正式对齐训练**

Run:

```bash
cd "/home/lkyu/baidu/prose-fd/prose_fd"
"/home/lkyu/miniconda3/envs/py312_torch291_cu128/bin/python" "main.py" \
  use_wandb=0 \
  data=shallow_water_minimal \
  model=prose_2to1 \
  optim=wsd \
  device=gpu:0 \
  base_seed=123 \
  reload_model="/tmp/prose_fd_alignment/initial_state/torch_initial_sw64.pth" \
  max_epoch=5 \
  n_steps_per_epoch=800 \
  batch_size=4 \
  batch_size_eval=6 \
  num_workers=0 \
  num_workers_eval=0 \
  log_eval_plots=-1 \
  exp_name=sw64_alignment \
  exp_id=prose_fd_sw64_ep5_torch_seed123_init123
```

- [ ] **Step 2: Paddle 正式对齐训练**

Run:

```bash
cd "/home/lkyu/baidu/prose-fd/prose_fd_paddle"
"/home/lkyu/miniconda3/envs/paddletorch/bin/python" "main.py" \
  use_wandb=0 \
  data=shallow_water_minimal \
  model=prose_2to1 \
  optim=wsd \
  device=gpu:0 \
  base_seed=123 \
  reload_model="/tmp/prose_fd_alignment/initial_state/paddle_initial_sw64.pdparams" \
  max_epoch=5 \
  n_steps_per_epoch=800 \
  batch_size=4 \
  batch_size_eval=6 \
  num_workers=0 \
  num_workers_eval=0 \
  log_eval_plots=-1 \
  exp_name=sw64_alignment \
  exp_id=prose_fd_sw64_ep5_paddle_seed123_init123
```

如果 AMP probe 失败但 FP32 probe 通过，则正式命令追加：

```bash
amp=0
```

如果某个 `exp_id` 已经存在 checkpoint，必须换新的 `exp_id`；不要复用会触发自动续训的目录。

## 9. Task 8: 写完整中文工作报告

**Files:**
- Create/Modify: `tools/debug_loss_alignment/README.md`

- [ ] **Step 1: 记录最终报告**

报告必须包含：

```markdown
# Prose-FD Torch/Paddle Training Loss Alignment Report

## 结论

说明默认随机初始化训练不可逐 step 对齐；严格对齐需要固定 `base_seed` 并加载同一份 64x64 初始权重。

## 根因

1. 用户命令未设置 `base_seed`，两侧运行时各自随机生成训练种子。
2. 用户命令未设置 `reload_model`，训练未使用已前向对齐的预训练权重。
3. 现有预训练权重是 128x128 配置，不能直接加载 64x64 minimal 数据训练模型。

## 证据

写入 batch hash 对齐结果、fresh state hash 差异、shape mismatch 信息、probe loss 表格。

## 修复方案

写入新增 debug 工具、64x64 初始态生成方式、正式训练命令。

## 验证结果

用表格列出 Torch/Paddle 每个 epoch 的 train loss、eval loss、rel l2 和差异。

## 剩余差异

如 AMP 或优化器仍有残余差异，记录可接受阈值和原因。
```

- [ ] **Step 2: 最终验证命令**

Run:

```bash
cd "/home/lkyu/baidu/prose-fd"
"/home/lkyu/miniconda3/envs/py312_torch291_cu128/bin/python" -m pytest "prose_fd/tests/test_training_alignment_debug.py" -v
"/home/lkyu/miniconda3/envs/paddletorch/bin/python" -m pytest "prose_fd_paddle/tests/test_training_alignment_debug.py" -v
```

Expected: 全部通过。正式训练日志中，若使用 `amp=0`，5 epoch train/eval loss 应与 Torch 在同一量级并满足报告中定义阈值；若使用 `amp=1`，只在 Task 5 AMP probe 通过后采用。

## 10. 交付标准

- batch dump 证明固定 `base_seed=123` 后 Torch/Paddle 前 45 个 batch 完全一致。
- 64x64 Torch 初始态可导出，并可转换为 Paddle `.pdparams`，Paddle 加载不报 shape mismatch。
- FP32 probe 在共享初始态、同 batch、无 clip 条件下 loss 对齐；再逐步验证 clip 和 AMP。
- 正式训练命令必须包含 `base_seed=123` 和对应 backend 的 `reload_model=...`。
- 默认训练行为不改变；所有严格对齐行为由显式 debug 文件或启动参数控制。
