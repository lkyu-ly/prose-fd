# PROSE-FD Paddle CINN 动转静训练加速比测量

- 测量日期：2026-07-29
- 范围：`prose_fd_paddle/`，CINN 动转静（`PROSE_TO_STATIC=1`）vs 动态图基线

## 1. 环境

| 项 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Ti（16GB），WSL2 |
| PaddlePaddle | 3.3.0（`is_compiled_with_cinn()=True`） |
| CUDA 驱动 | 13.1（libcuda.so 在 `/usr/lib/wsl/lib`） |
| 数据集 | shallow_water 小型 64×64（`/home/lkyu/baidu/PDEBench/data/2D_rdb_NA_NA/2D_rdb_NA_NA.h5`） |

## 2. 测量参数

两组完全同参数，仅 `PROSE_TO_STATIC` 不同（精度口径一致：`amp=0` fp32）：

```
model=prose_2to1 data=shallow_water_minimal optim=adamw
amp=0 compile=0 use_wandb=0
num_workers=0 num_workers_eval=0
batch_size=2 batch_size_eval=2
print_freq=200 max_epoch=1 n_steps_per_epoch=1600
log_eval_plots=-1
data.shallow_water.data_path=<小型 64×64>
```

- `num_workers=0`、`batch_size=2`：WSL2 单卡稳定参数（避 DataLoader worker 崩 / OOM）。
- `N=1600` 步、`print_freq=200`：每 200 步一个纯训练计时窗口，共 8 窗口。
- 纯训练计时（`PROSE_BENCHMARK=1` 守卫）：`forward 前 paddle.device.cuda.synchronize()+t0` → `optimize 后 synchronize()+t1`，**不含数据加载**。

## 3. 原始窗口数据（纯训练 step/s，越大越快）

| 窗口 (step) | 基线 step/s | CINN step/s |
|---|---|---|
| 1 (200) | 1.9193 | 1.0252（含 ~108s CINN 编译 warmup） |
| 2 (400) | 2.0152 | 2.1583 |
| 3 (600) | 1.9522 | 2.1425 |
| 4 (800) | 1.9546 | 2.0862 |
| 5 (1000) | 2.0024 | 2.0851 |
| 6 (1200) | 1.9376 | 2.1110 |
| 7 (1400) | 1.9911 | 2.0941 |
| 8 (1600) | 2.0235 | 2.0666 |

CINN `Compiling subgraph with CINN backend` 出现 **1 次**，编译耗时约 108s（08:08:20→08:09:54 Executor Running）。

## 4. 加速比结论

**稳态纯训练（核心指标，丢弃首窗口 warmup，window 2-8 同口径）：**
- 基线稳态 step/s = (2.0152+1.9522+1.9546+2.0024+1.9376+1.9911+2.0235)/7 = **1.9824**
- CINN 稳态 step/s = (2.1583+2.1425+2.0862+2.0851+2.1110+2.0941+2.0666)/7 = **2.1063**
- **稳态纯训练加速 = 2.1063 / 1.9824 − 1 = +6.25%**

**端到端（训练段 wall-clock，含数据加载，不含 eval）：**
- 基线：07:53:46 → 08:07:28 = 822s → 1.946 step/s
- CINN：08:08:15 → 08:22:45 = 870s（含 108s 编译）→ 1.839 step/s
- **端到端 −5.5%**（CINN 因首次编译 warmup 在 1600 步内未被完全摊销，端到端略慢；符合踩坑手册"短跑端到端 CINN 更慢、稳态才有加速"的预期）。

## 5. 精度一致性

两组 loss 轨迹一致（训练有效，无 CINN 破坏）：
- 基线：step 200 loss 0.248 → step 1600 loss 0.0031
- CINN：step 200 loss 0.241 → step 1600 loss 0.0058

精度口径：两组均 `amp=0`（fp32），无 AMP 引入的差异。

## 6. 结论与注意

- **CINN 在本场景（bs=2、fp32、小型 64×64）稳态纯训练加速约 +6.3%**，与 Poseidon 项目 CINN 记录（~+8%）同量级。
- **端到端需长训练摊销编译成本**：1600 步内 CINN 端到端仍略慢（108s 编译 warmup）；稳态口径才能反映 CINN 收益。生产长训练（数万步以上）净收益为正。
- **生效验证**：CINN 组日志出现 `Compiling subgraph with CINN backend`（唯一判据）。
- **避坑**：本机必须 `num_workers=0` + `batch_size=2` 跑（WSL2 GPU worker 不稳 + 显存溢出致关机）；更大 batch 的加速比待多卡/大显存环境另测。

## 7. 复现命令

```
cd prose_fd_paddle
# 基线
PROSE_BENCHMARK=1 python main.py model=prose_2to1 data=shallow_water_minimal optim=adamw \
  amp=0 compile=0 use_wandb=0 num_workers=0 num_workers_eval=0 batch_size=2 batch_size_eval=2 \
  print_freq=200 max_epoch=1 n_steps_per_epoch=1600 log_eval_plots=-1 \
  data.shallow_water.data_path=/home/lkyu/baidu/PDEBench/data/2D_rdb_NA_NA/2D_rdb_NA_NA.h5 \
  exp_name=t5_base
# CINN（同参数 + PROSE_TO_STATIC=1）
PROSE_TO_STATIC=1 PROSE_BENCHMARK=1 python main.py model=prose_2to1 data=shallow_water_minimal optim=adamw \
  amp=0 compile=0 use_wandb=0 num_workers=0 num_workers_eval=0 batch_size=2 batch_size_eval=2 \
  print_freq=200 max_epoch=1 n_steps_per_epoch=1600 log_eval_plots=-1 \
  data.shallow_water.data_path=/home/lkyu/baidu/PDEBench/data/2D_rdb_NA_NA/2D_rdb_NA_NA.h5 \
  exp_name=t5_cinn
```

日志中 `BENCH pure_train step/s (last 200): X.XXXX` 即每个窗口的纯训练 step/s。
