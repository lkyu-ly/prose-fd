# Paddle Empty GradScaler Checkpoint Resume Implementation Plan

**Goal:** 修复 `prose_fd_paddle` 在 `dryrun` 成功保存检查点后，下一次启动因空 `GradScaler` 状态恢复失败而无法续训的问题。

**Architecture:** 保持现有训练链路和 AMP 开关语义不变，只修补检查点保存与加载的兼容逻辑。保存时不再写入空 `scaler` 状态；加载时对历史坏检查点中的空 `scaler` 状态执行跳过恢复，从而同时修复新旧检查点。

**Tech Stack:** Python, PaddlePaddle, pytest, OmegaConf

---

## Context And Root Cause

已经确认以下事实：

- 复现命令是：

```bash
cd "/home/lkyu/baidu/prose-fd/prose_fd_paddle"
python "main.py" dryrun=1 use_wandb=0 data=shallow_water_minimal model=prose_2to1 optim=wsd device=gpu:0 batch_size=2 batch_size_eval=1 num_workers=0 num_workers_eval=0 log_eval_plots=-1 exp_name=sw64_single_gpu exp_id=prose_fd_sw64_dryrun
```

- 首次运行会成功保存 `checkpoint/debug/prose_fd_sw64_dryrun/checkpoint.pth`。
- 第二次运行会在 [trainer.py](/home/lkyu/baidu/prose-fd/prose_fd_paddle/trainer.py) 的 `reload_checkpoint()` 内调用 `self.scaler.load_state_dict(data["scaler"])` 时崩溃。
- 实际检查点内容已经验证为：

```python
obj = paddle.load("checkpoint/debug/prose_fd_sw64_dryrun/checkpoint.pth")
assert obj["scaler"] == {}
```

- Paddle 3.3.0 的 `AmpScaler.scale()` 在 AMP dtype 不是 `float16` 时会把 scaler 自动 disable；当前训练代码在 [trainer.py](/home/lkyu/baidu/prose-fd/prose_fd_paddle/trainer.py) 中使用的是：

```python
with paddle.amp.autocast(
    get_amp_device_type(),
    enabled=bool(params.amp),
    dtype=paddle.bfloat16,
):
```

- 因此本次 dryrun 的实际过程是：
  - 创建 `GradScaler(enable=True)`
  - 进入 `bfloat16` autocast
  - `scale()` 时被 Paddle 自动 disable
  - `state_dict()` 变成空字典
  - 保存进检查点
  - 下次启动又新建了 enabled scaler
  - 对空字典执行 `load_state_dict({})`
  - Paddle 按设计抛出 `RuntimeError`

本次修复只处理“空 scaler 状态导致无法续训”的问题，不顺带改 AMP 策略。是否彻底去掉 `bfloat16` 路径上的 `GradScaler`，留作单独问题处理。

## File Map

**Modify:**
- `prose_fd_paddle/trainer.py`

**Create:**
- `prose_fd_paddle/tests/test_checkpoint_resume.py`

**Reference Only:**
- `prose_fd_paddle/tests/test_training_data_smoke.py`
- `prose_fd_paddle/configs/main.yaml`
- `"/home/lkyu/miniconda3/envs/paddletorch/lib/python3.10/site-packages/paddle/amp/grad_scaler.py"`

### Task 1: Add Regression Tests For Empty Scaler Checkpoints

**Files:**
- Create: `prose_fd_paddle/tests/test_checkpoint_resume.py`
- Reference: `prose_fd_paddle/tests/test_training_data_smoke.py`
- Reference: `prose_fd_paddle/trainer.py`

- [ ] **Step 1: Write the failing test file**

Create `prose_fd_paddle/tests/test_checkpoint_resume.py` with this content:

```python
from __future__ import annotations

from pathlib import Path

import paddle

from prose_fd_paddle.models.build_model import build_model
from prose_fd_paddle.symbol_utils.environment import SymbolicEnvironment
from prose_fd_paddle.trainer import Trainer
from prose_fd_paddle.tests.test_training_data_smoke import build_sw64_cfg


class FakeDisabledScaler:
    def __init__(self, enable: bool = True):
        self.enable = enable

    def state_dict(self):
        return {}

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        return None

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        return None


class FakeEnabledScaler:
    def __init__(self, enable: bool = True):
        self.enable = enable
        self.loaded = None

    def state_dict(self):
        return {"scale": [65536.0]}

    def load_state_dict(self, state):
        if not state:
            raise RuntimeError("empty scaler state must not be loaded")
        self.loaded = state

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        return None

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        return None


def build_cfg(tmp_path: Path):
    cfg = build_sw64_cfg()
    cfg.amp = 1
    cfg.dump_path = str(tmp_path / "exp")
    cfg.eval_dump_path = str(tmp_path / "exp" / "evals_all")
    return cfg


def build_modules(cfg):
    symbol_env = SymbolicEnvironment(cfg.symbol)
    modules = build_model(cfg, cfg.model, cfg.data, symbol_env)
    return modules, symbol_env


def test_save_checkpoint_omits_empty_scaler_state(monkeypatch, tmp_path):
    monkeypatch.setattr(paddle.amp, "GradScaler", FakeDisabledScaler)
    cfg = build_cfg(tmp_path)
    modules, symbol_env = build_modules(cfg)
    trainer = Trainer(modules, cfg, symbol_env)

    trainer.save_checkpoint("checkpoint")

    checkpoint = paddle.load(str(Path(cfg.dump_path) / "checkpoint.pth"))
    assert "scaler" not in checkpoint


def test_reload_checkpoint_skips_legacy_empty_scaler_state(monkeypatch, tmp_path):
    monkeypatch.setattr(paddle.amp, "GradScaler", FakeEnabledScaler)
    cfg = build_cfg(tmp_path)
    modules, symbol_env = build_modules(cfg)
    trainer = Trainer(modules, cfg, symbol_env)

    checkpoint_path = Path(cfg.dump_path) / "checkpoint.pth"
    checkpoint = {
        "epoch": 0,
        "n_total_iter": 5,
        "dataloader_count": 0,
        "best_metrics": trainer.best_metrics,
        "model": modules["model"].state_dict(),
        "optimizer": trainer.optimizer.state_dict(),
        "scheduler": trainer.scheduler.state_dict() if trainer.scheduler is not None else None,
        "scaler": {},
    }
    if checkpoint["scheduler"] is None:
        checkpoint.pop("scheduler")
    paddle.save(checkpoint, str(checkpoint_path))

    modules_2, symbol_env_2 = build_modules(cfg)
    trainer_2 = Trainer(modules_2, cfg, symbol_env_2)

    assert trainer_2.epoch == 1
    assert trainer_2.n_total_iter == 5
    assert trainer_2.scaler.loaded is None
```

- [ ] **Step 2: Run the new tests to verify the bug is reproduced**

Run:

```bash
cd "/home/lkyu/baidu/prose-fd"
"/home/lkyu/miniconda3/envs/paddletorch/bin/python" -m pytest "prose_fd_paddle/tests/test_checkpoint_resume.py" -q
```

Expected before the fix:
- `test_save_checkpoint_omits_empty_scaler_state` fails because checkpoint 里仍然有 `"scaler": {}`
- 或 `test_reload_checkpoint_skips_legacy_empty_scaler_state` fails because `Trainer(...)` 初始化时触发空 scaler 恢复

- [ ] **Step 3: Confirm the tests model the real root cause**

Before touching implementation, manually verify以下两点：

```python
from paddle.amp.grad_scaler import AmpScaler

assert AmpScaler.state_dict.__name__ == "state_dict"
assert "returns an empty dict" in AmpScaler.state_dict.__doc__
assert "saved from a disabled instance of GradScaler" in AmpScaler.load_state_dict.__doc__ or True
```

以及确认训练实现确实使用 `bfloat16`：

```bash
rg -n "dtype=paddle.bfloat16|GradScaler\\(enable=True\\)|load_state_dict\\(data\\[\"scaler\"\\]\\)" "prose_fd_paddle/trainer.py"
```

- [ ] **Step 4: Re-run the failing tests one more time and capture the exact failure line**

Run:

```bash
cd "/home/lkyu/baidu/prose-fd"
"/home/lkyu/miniconda3/envs/paddletorch/bin/python" -m pytest "prose_fd_paddle/tests/test_checkpoint_resume.py" -vv
```

Expected before the fix:
- 失败行落在 [trainer.py](/home/lkyu/baidu/prose-fd/prose_fd_paddle/trainer.py) 的保存或恢复 scaler 分支上

### Task 2: Apply The Minimal Fix In Trainer Checkpoint Logic

**Files:**
- Modify: `prose_fd_paddle/trainer.py`
- Test: `prose_fd_paddle/tests/test_checkpoint_resume.py`

- [ ] **Step 1: Change checkpoint saving so empty scaler state is not written**

In `save_checkpoint()`, replace:

```python
if self.scaler is not None:
    data["scaler"] = self.scaler.state_dict()
```

with:

```python
if self.scaler is not None:
    scaler_state = self.scaler.state_dict()
    if scaler_state:
        data["scaler"] = scaler_state
    else:
        logger.info("Skipping empty gradient scaler state when saving checkpoint.")
```

This is the smallest forward fix:
- 新生成的检查点不再带坏数据
- 不改 AMP 开关
- 不改优化器行为

- [ ] **Step 2: Change checkpoint loading so legacy empty scaler state is skipped**

In `reload_checkpoint()`, replace:

```python
if "scaler" in data and self.scaler is not None:
    logger.warning("Reloading gradient scaler ...")
    self.scaler.load_state_dict(data["scaler"])
```

with:

```python
scaler_state = data.get("scaler")
if self.scaler is not None and scaler_state:
    logger.warning("Reloading gradient scaler ...")
    self.scaler.load_state_dict(scaler_state)
elif self.scaler is not None and "scaler" in data:
    logger.warning("Skipping empty gradient scaler state from checkpoint.")
```

This is the smallest backward-compatible fix:
- 老检查点里已有 `"scaler": {}` 也能恢复
- 不需要删除已有检查点
- 不会吞掉真正非空 scaler state 的恢复逻辑

- [ ] **Step 3: Do not change any other AMP behavior in this patch**

Explicitly keep these lines unchanged in this patch:

```python
if params.amp:
    self.scaler = paddle.amp.GradScaler(enable=True)
```

and:

```python
with paddle.amp.autocast(
    get_amp_device_type(),
    enabled=bool(params.amp),
    dtype=paddle.bfloat16,
):
```

Reason:
- 这是单独的训练策略问题，不是这次续训崩溃的最小修复范围
- 本次只解决“检查点无法恢复”
- 避免把回归修复扩散成 AMP 语义改造

### Task 3: Verify New And Legacy Checkpoint Compatibility

**Files:**
- Test: `prose_fd_paddle/tests/test_checkpoint_resume.py`
- Reference: `prose_fd_paddle/tests/test_training_data_smoke.py`

- [ ] **Step 1: Run the targeted regression tests**

Run:

```bash
cd "/home/lkyu/baidu/prose-fd"
"/home/lkyu/miniconda3/envs/paddletorch/bin/python" -m pytest "prose_fd_paddle/tests/test_checkpoint_resume.py" -q
```

Expected after the fix:

```text
2 passed
```

- [ ] **Step 2: Run the existing training smoke tests to ensure no regression**

Run:

```bash
cd "/home/lkyu/baidu/prose-fd"
"/home/lkyu/miniconda3/envs/paddletorch/bin/python" -m pytest "prose_fd_paddle/tests/test_training_data_smoke.py" -q
```

Expected after the fix:
- 全部通过
- 不应出现新的 checkpoint / optimizer / scheduler 回归

- [ ] **Step 3: Run a focused syntax and static scan**

Run:

```bash
cd "/home/lkyu/baidu/prose-fd"
"/home/lkyu/miniconda3/envs/paddletorch/bin/python" -m py_compile "prose_fd_paddle/trainer.py" "prose_fd_paddle/tests/test_checkpoint_resume.py"
rg -n "load_state_dict\\(data\\[\"scaler\"\\]\\)|data\\[\"scaler\"\\] = self\\.scaler\\.state_dict\\(\\)" "prose_fd_paddle/trainer.py"
```

Expected after the fix:
- `py_compile` 无输出并成功退出
- `rg` 无匹配

- [ ] **Step 4: Manual runtime validation for the exact user scenario**

Do not delete the old checkpoint. Re-run the exact user command:

```bash
cd "/home/lkyu/baidu/prose-fd/prose_fd_paddle"
python "main.py" dryrun=1 use_wandb=0 data=shallow_water_minimal model=prose_2to1 optim=wsd device=gpu:0 batch_size=2 batch_size_eval=1 num_workers=0 num_workers_eval=0 log_eval_plots=-1 exp_name=sw64_single_gpu exp_id=prose_fd_sw64_dryrun
```

Expected after the fix:
- 不再因为空 `scaler` 状态崩溃
- 会正常打印 `Reloading checkpoint ...`
- 如果命中旧坏检查点，会打印 `Skipping empty gradient scaler state from checkpoint.`
- 训练可以继续进入 epoch / eval 流程

## Notes For The Implementer

- 不要为这次修复引入新配置项。
- 不要修改 optimizer、scheduler、model 参数加载逻辑。
- 不要把这次修复扩展到“AMP 用 `float16` 还是 `bfloat16`”的策略争论。
- 如果你在实现时发现 `scheduler` 也存在同类空状态问题，再单独提出，不要顺手混进这个补丁。

## Self-Review

### Spec Coverage

- “探索和检查这个问题的原因”：已覆盖，计划开头给出了完整因果链。
- “给出合理的解决方案”：已覆盖，方案限定为保存端和加载端的最小兼容修补。
- “代价尽量小”：已覆盖，不改训练策略、不改 AMP 配置、不改其他模块。
- “需要能处理再次运行相同命令时报错”：已覆盖，Task 3 Step 4 直接要求复跑用户命令，且不删除旧检查点。

### Placeholder Scan

已检查本计划，没有未落地的占位语句、延后实现标记、跨任务引用替代说明，测试和代码步骤都给出了可直接执行的内容。

### Type Consistency

- 统一使用 `scaler_state` 作为保存/加载分支变量名。
- 测试文件中统一通过 `build_sw64_cfg()` 构建配置，避免测试和主链路配置分叉。
- 所有运行命令统一使用 `"/home/lkyu/miniconda3/envs/paddletorch/bin/python"` 进行测试验证。

## Recommended Solution Summary

推荐直接采用本计划中的双端兼容修复：

1. 保存检查点时不再写入空 `scaler` 状态
2. 加载检查点时跳过历史空 `scaler` 状态

这是当前成本最低、影响面最小、同时兼容新旧检查点的方案。

不推荐在这次补丁里直接改成“`bfloat16` 不创建 `GradScaler`”，因为那会改变训练控制流，属于另一个问题域。

---
