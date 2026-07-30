# CINN 动转静训练优化 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. **项目约定：禁止主动 git commit/分支操作，每个 task 以「验证通过」收尾，不写 commit step。**

**Goal:** 为 `prose_fd_paddle` 接入 `paddle.jit.to_static`，使 CINN 真正编译训练图并获得稳态训练加速；单一环境变量控制；本地单卡验证两个数据集跑通并给出确定性加速比。

**Architecture:** 单一环境变量 `PROSE_TO_STATIC` 在 `import paddle` 之前设三个 FLAGS，并在 `build_model.py` 权重加载后 / DataParallel 前对 model 施加 `paddle.jit.to_static(full_graph=True)`（lazy trace）。审计并修复 forward 链路的 torch 风格 API（`.view→.reshape` 等）以满足 trace。计时注入 `trainer.iter` 测稳态纯训练 + 端到端加速比。

**Tech Stack:** PaddlePaddle 3.3.0（CINN 已编译）、Hydra、einops、h5py。

## Global Constraints
- 唯一修改目标：`prose_fd_paddle/`（不动 `prose_fd/` 参考实现）。
- **禁止主动 git commit/分支操作**（项目 engineer-professional 约定），除非用户明确要求。
- paddle 3.3.0；CINN 生效唯一判据：日志出现 `Compiling subgraph with CINN backend`。
- FLAGS 必须在 `main.py:6` `import paddle` 之前设置。
- to_static 顺序：权重加载（`build_model.py:52-65`）后、DataParallel（`trainer.py:33-36`）前。
- to_static 插入点：`build_model.py` 函数体**顶层缩进**（与 `return modules` 同级），**不可**沿用 `:75` 的 `if not params.cpu:` 块缩进。
- 数据集：小型 `/home/lkyu/baidu/PDEBench/data/2D_rdb_NA_NA/2D_rdb_NA_NA.h5`（64×64）；原版 `/home/lkyu/baidu/PDEBench/pdebench/data_download/data/2D/shallow-water/2D_rdb_NA_NA.h5`（128×128，**必须** `data.shallow_water.x_num=128`）。
- 预训练权重：`/home/lkyu/baidu/prose-fd/models/prose_fd_paddle/prose_fd_converted.pdparams`。
- 工作目录：`prose_fd_paddle/`。
- 「跑通」判据：≥3 步 forward+backward+optimize 完成、loss 有限、无异常。

## File Structure
| 文件 | 职责 | 改动 |
|------|------|------|
| `prose_fd_paddle/main.py` | 入口 | 顶部 `import paddle` 前 FLAGS 双向注入 |
| `prose_fd_paddle/models/build_model.py` | 模型装配 | `return` 前 to_static 包裹（顶层缩进） |
| `prose_fd_paddle/models/{transformer_wrappers,transformer,embedder,attention_utils}.py` | forward 链路 | trace 链路 `.view→.reshape`、`.size→tuple(.shape)`、`.repeat→paddle.tile`（审计后定位） |
| `prose_fd_paddle/trainer.py` | 训练循环 | benchmark 计时注入（`PROSE_BENCHMARK=1` 守卫） |
| `docs/cinn-to-static-benchmark.md` | 新建 | 加速比报告 |

---

### Task 1: 注入 PROSE_TO_STATIC 开关与 to_static 包裹

**Files:**
- Modify: `prose_fd_paddle/main.py:1-8`（`import paddle` 前插入）
- Modify: `prose_fd_paddle/models/build_model.py:73-76`（`return modules` 前插入，顶层缩进）

**Interfaces:**
- Produces: 环境变量 `PROSE_TO_STATIC`（`"1"` 开 / 其他关）；开启时 model 被 `paddle.jit.to_static` 包裹，日志打印 `Wrapping model with paddle.jit.to_static (CINN)`。

- [ ] **Step 1: 改 `main.py`——在 `import paddle`（第 6 行）前插入 FLAGS 双向注入**

在 `main.py` 现有 `import os`（第 1 行）之后、`import paddle`（第 6 行）之前插入：
```python
# --- CINN 动转静开关：必须在 import paddle 之前设置 FLAGS ---
if os.environ.get("PROSE_TO_STATIC", "0") == "1":
    os.environ["FLAGS_prim_enable_dynamic"] = "true"
    os.environ["FLAGS_prim_all"] = "true"
    os.environ["FLAGS_use_cinn"] = "true"
else:
    os.environ["FLAGS_prim_enable_dynamic"] = "false"
    os.environ["FLAGS_prim_all"] = "false"
    os.environ["FLAGS_use_cinn"] = "false"
```

- [ ] **Step 2: 改 `build_model.py`——`return modules` 前插入 to_static 包裹（顶层缩进）**

在 `build_model.py` 的 `return modules`（第 76 行）之前、`.to(device)` 块（第 73-75 行）之后，以**函数体顶层缩进**插入：
```python
    if os.environ.get("PROSE_TO_STATIC", "0") == "1":
        from paddle.jit import to_static
        logger.info("Wrapping model with paddle.jit.to_static (CINN) ...")
        for k in list(modules.keys()):
            modules[k] = to_static(modules[k], full_graph=True)
```
（注意：`build_model.py` 顶部已 `import paddle`，需补 `import os` 若无。）

- [ ] **Step 3: 验证 import smoke + 开关行为**

Run:
```bash
cd prose_fd_paddle
python -c "import main" 2>&1 | tail -5    # 无异常
PROSE_TO_STATIC=1 python -c "
import os, logging; logging.basicConfig(level=logging.INFO)
import main  # 仅 import，不跑训练
" 2>&1 | grep -i "Wrapping model with paddle.jit.to_static" || echo "（import 阶段不构建模型，无此日志属正常）"
```
Expected：`import main` 无报错（语法/import 正确）。

- [ ] **Step 4: 验证开关=0 时行为不变（动态图）**

Run（动态图 forward smoke，几步）:
```bash
cd prose_fd_paddle
python main.py model=prose_2to1 data=shallow_water_minimal optim=adamw dryrun=1 use_wandb=0 \
  reload_model=/home/lkyu/baidu/prose-fd/models/prose_fd_paddle/prose_fd_converted.pdparams \
  data.shallow_water.data_path=/home/lkyu/baidu/PDEBench/data/2D_rdb_NA_NA/2D_rdb_NA_NA.h5 \
  exp_name=t1_smoke 2>&1 | tail -30
```
Expected：跑完 5 步，无 `Wrapping model with paddle.jit.to_static` 日志（开关未开），无异常。（若此时 forward 链路有 torch API 在动态图下也会报错——那是 Task 2 要修的，先记录。）

---

### Task 2: 审计并修复 forward 链路 torch 风格 API

**Files:**
- Audit: `prose_fd_paddle/models/transformer_wrappers.py`, `transformer.py`, `embedder.py`, `attention_utils.py`
- Modify: 审计命中处

**Interfaces:**
- Produces: forward 链路不含 `.view()`/`.size()`/`.repeat()` torch 风格调用（assert 内同样修），动态图 forward 数值不变。

- [ ] **Step 1: 审计 forward 链路 torch 风格调用**

Run:
```bash
cd prose_fd_paddle
grep -rnE "\.view\(|\.size\(|\.repeat\(" models/ | grep -v "_test\|baseline"
```
记录每一处命中（文件:行号 + 上下文），判定是否在 forward trace 链路。

- [ ] **Step 2: 逐处修复（按下表规则）**

| 命中 | 改为 | 备注 |
|------|------|------|
| `x.view(...)` | `x.reshape(...)` | 冗余恒等 view 直接删 |
| `x.size()` | `x.shape` | |
| `x.size(N)` | `x.shape[N]` | |
| `assert x.size() == (a,b)` | `assert tuple(x.shape) == (a,b)` | **必加 `tuple()`**：paddle `.shape` 是 list，漏 `tuple()` → list==tuple 恒 False |
| `x.repeat(...)` | `paddle.tile(x, ...)` | 按手册语义对齐参数 |

逐处 Edit。每改一处核对语义等价。

- [ ] **Step 3: 验证动态图 forward 仍正确（不破坏数值）**

Run（动态图 dryrun）:
```bash
cd prose_fd_paddle
python main.py model=prose_2to1 data=shallow_water_minimal optim=adamw dryrun=1 use_wandb=0 \
  reload_model=/home/lkyu/baidu/prose-fd/models/prose_fd_paddle/prose_fd_converted.pdparams \
  data.shallow_water.data_path=/home/lkyu/baidu/PDEBench/data/2D_rdb_NA_NA/2D_rdb_NA_NA.h5 \
  exp_name=t2_smoke 2>&1 | tail -30
```
Expected：5 步训练通过，loss 有限。若报 `.view`/`.size`/`.repeat` 相关错，回 Step 1 补漏。

---

### Task 3: 基线跑通（两组数据集，PROSE_TO_STATIC 未设）

**Files:** 无改动（纯验证；若发现 bug 回 Task 1/2 修）

- [ ] **Step 1: 基线-小型数据集跑通**

Run:
```bash
cd prose_fd_paddle
python main.py model=prose_2to1 data=shallow_water_minimal optim=adamw dryrun=1 use_wandb=0 \
  reload_model=/home/lkyu/baidu/prose-fd/models/prose_fd_paddle/prose_fd_converted.pdparams \
  data.shallow_water.data_path=/home/lkyu/baidu/PDEBench/data/2D_rdb_NA_NA/2D_rdb_NA_NA.h5 \
  exp_name=t3_base_small 2>&1 | tail -30
```
Expected：≥3 步训练，loss 有限，无异常。

- [ ] **Step 2: 基线-原版数据集跑通（须 x_num=128）**

Run:
```bash
cd prose_fd_paddle
python main.py model=prose_2to1 data=shallow_water_minimal optim=adamw dryrun=1 use_wandb=0 \
  reload_model=/home/lkyu/baidu/prose-fd/models/prose_fd_paddle/prose_fd_converted.pdparams \
  data.shallow_water.data_path=/home/lkyu/baidu/PDEBench/pdebench/data_download/data/2D/shallow-water/2D_rdb_NA_NA.h5 \
  data.shallow_water.x_num=128 \
  exp_name=t3_base_full 2>&1 | tail -30
```
Expected：≥3 步训练，loss 有限，输入为 64×64（下采样生效），无维度错误。

---

### Task 4: CINN 跑通（PROSE_TO_STATIC=1，两组数据集）

**Files:** 无改动（纯验证；若 trace/反向报错，按 spec §7 应对）

- [ ] **Step 1: CINN-小型数据集跑通**

Run:
```bash
cd prose_fd_paddle
PROSE_TO_STATIC=1 python main.py model=prose_2to1 data=shallow_water_minimal optim=adamw dryrun=1 use_wandb=0 \
  reload_model=/home/lkyu/baidu/prose-fd/models/prose_fd_paddle/prose_fd_converted.pdparams \
  data.shallow_water.data_path=/home/lkyu/baidu/PDEBench/data/2D_rdb_NA_NA/2D_rdb_NA_NA.h5 \
  exp_name=t4_cinn_small 2>&1 | tee /tmp/t4_cinn_small.log | tail -40
```
Expected：日志出现 `Wrapping model with paddle.jit.to_static (CINN)` 与 `Compiling subgraph with CINN backend`；≥3 步训练，loss 有限，无 trace/反向错误。首步编译数十秒属正常。
- 若遇 `view_shape_grad` 错：回 Task 2 查漏 `.view()`；勿试 pass flag/升级 paddle（手册验证无效）。

- [ ] **Step 2: CINN-原版数据集跑通**

Run:
```bash
cd prose_fd_paddle
PROSE_TO_STATIC=1 python main.py model=prose_2to1 data=shallow_water_minimal optim=adamw dryrun=1 use_wandb=0 \
  reload_model=/home/lkyu/baidu/prose-fd/models/prose_fd_paddle/prose_fd_converted.pdparams \
  data.shallow_water.data_path=/home/lkyu/baidu/PDEBench/pdebench/data_download/data/2D/shallow-water/2D_rdb_NA_NA.h5 \
  data.shallow_water.x_num=128 \
  exp_name=t4_cinn_full 2>&1 | tee /tmp/t4_cinn_full.log | tail -40
```
Expected：同 Step 1（CINN 生效 + 跑通）。

---

### Task 5: 加速比测量与报告

**Files:**
- Modify: `prose_fd_paddle/trainer.py`（`iter` 方法内注入计时，`PROSE_BENCHMARK=1` 守卫）
- Create: `docs/cinn-to-static-benchmark.md`

**Interfaces:**
- Produces: `docs/cinn-to-static-benchmark.md`，含命令、原始计时、精度口径、稳态 step/s 加速% + 端到端加速%。

- [ ] **Step 1: trainer.iter 注入 benchmark 计时**

在 `trainer.py` 顶部 `import time`（若无）。在 `iter` 方法训练循环内，`model("fwd", ...)`（:465）前与 `self.optimize(data_loss)`（:471）后插入（用 `PROSE_BENCHMARK` 守卫）：
```python
_bench = os.environ.get("PROSE_BENCHMARK", "0") == "1"
# ... 循环内：
if _bench: paddle.device.cuda.synchronize()
_t0 = time.perf_counter()
data_output = model("fwd", **model_input)        # :465
# ... denormalize + loss（:466-:469）
self.optimize(data_loss)                         # :471
if _bench:
    paddle.device.cuda.synchronize()
    self._pure_train_sec += time.perf_counter() - _t0
```
并在 epoch 结束处打印 `稳态纯训练 step/s = n_steps / _pure_train_sec`、`端到端 step/s = n_steps / wall_clock`。（具体打印位置在 `train_epoch` 末尾，实现时定位。）

- [ ] **Step 2: 基线组测量（N≥1500 步，小型数据集，amp=1）**

Run（后台，长耗时）:
```bash
cd prose_fd_paddle
PROSE_BENCHMARK=1 python main.py model=prose_2to1 data=shallow_water_minimal optim=adamw use_wandb=0 \
  reload_model=/home/lkyu/baidu/prose-fd/models/prose_fd_paddle/prose_fd_converted.pdparams \
  data.shallow_water.data_path=/home/lkyu/baidu/PDEBench/data/2D_rdb_NA_NA/2D_rdb_NA_NA.h5 \
  max_epoch=1 n_steps_per_epoch=1600 eval_only=0 log_eval_plots=-1 \
  exp_name=t5_base 2>&1 | tee /tmp/t5_base.log
```
Expected：打印稳态 step/s 与端到端 step/s。记录数值。

- [ ] **Step 3: CINN 组测量（同参数 + PROSE_TO_STATIC=1）**

Run:
```bash
cd prose_fd_paddle
PROSE_TO_STATIC=1 PROSE_BENCHMARK=1 python main.py model=prose_2to1 data=shallow_water_minimal optim=adamw use_wandb=0 \
  reload_model=/home/lkyu/baidu/prose-fd/models/prose_fd_paddle/prose_fd_converted.pdparams \
  data.shallow_water.data_path=/home/lkyu/baidu/PDEBench/data/2D_rdb_NA_NA/2D_rdb_NA_NA.h5 \
  max_epoch=1 n_steps_per_epoch=1600 eval_only=0 log_eval_plots=-1 \
  exp_name=t5_cinn 2>&1 | tee /tmp/t5_cinn.log
```
Expected：日志 `Compiling subgraph with CINN backend`；打印稳态 step/s 与端到端 step/s（丢弃前 K 步 warmup）。

- [ ] **Step 4: 写报告 `docs/cinn-to-static-benchmark.md`**

包含：硬件（GPU 型号）、paddle 版本、数据集、amp 精度口径、N、warmup 丢弃 K、四组验证结论（Task 3/4）、稳态 step/s（基线 vs CINN）+ 加速%、端到端 step/s + 加速%、原始命令（可复现）、结论与注意事项（CINN warmup、稳态口径）。

- [ ] **Step 5: 验证报告完整**

Run: `cat docs/cinn-to-static-benchmark.md`
Expected：含上述全部字段，加速% 有具体数值。

---

## Self-Review（计划完成后填写）
- Spec coverage：§3.1 开关→Task1；§3.2 to_static→Task1；§3.3 trace 审计→Task2；§4 验证→Task3/4；§5 加速比→Task5；§6/§7 不做/风险→Global Constraints + 各 task Expected。✓
- Placeholder：无 TBD；命令与代码均具体。
- 类型一致：`PROSE_TO_STATIC` / `PROSE_BENCHMARK` 全程一致。
