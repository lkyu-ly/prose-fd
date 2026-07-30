# CINN 动转静训练优化设计

- 日期：2026-07-29（已按三视角自审修正）
- 范围：`prose_fd_paddle/`（PaddlePaddle 实现，唯一修改目标）
- 状态：已批准，待实现

## 1. 背景与目标

PaddlePaddle 3.3.0 的原生编译器 CINN 可加速训练，但**仅在动转静（`paddle.jit.to_static`）后生效**。当前 `prose_fd_paddle` 虽能在设了 CINN FLAGS 的环境下运行，但未做动转静，CINN 实际未参与训练图编译，无实质性加速。

**目标**：
1. 接入 `paddle.jit.to_static`，使 CINN 真正编译训练图并获得稳态训练加速。
2. 用单一环境变量开关控制，不破坏现有动态图运行能力（向后兼容）。
3. 本地单卡验证：两个 shallow-water 数据集（小型 64×64 / 原版 128×128）各跑 ≥3 步训练通过（forward+backward+optimize 完成、loss 有限、无异常）。
4. 给出确定性的加速比数据（稳态纯训练 + 端到端），写入仓库文件。

**非目标**：多卡/分布式 CINN 验证；评估侧 CINN；推理部署导出（`paddle.jit.save`）。

## 2. 上下文发现（决策依据）

### 2.1 环境（已核对，全部就绪）
- paddle 3.3.0，`is_compiled_with_cinn() = True`，GPU ×1。
- 数据集：小型 `/home/lkyu/baidu/PDEBench/data/2D_rdb_NA_NA/2D_rdb_NA_NA.h5`（64×64，200 样本，317MB）；原版 `/home/lkyu/baidu/PDEBench/pdebench/data_download/data/2D/shallow-water/2D_rdb_NA_NA.h5`（128×128，1000 样本，6.2GB）。
- 预训练权重：`/home/lkyu/baidu/prose-fd/models/prose_fd_paddle/prose_fd_converted.pdparams`（635MB）。
- `shallow_water_minimal.yaml` 默认 `data_path` 不存在，须命令行覆盖。

### 2.2 CINN 生效约束（官方文档 + 踩坑手册，同属 3.3.0）
- 仅设 `FLAGS_use_cinn=true` **不生效**；必须 `paddle.jit.to_static` 包裹模型，CINN 才编译训练图。
- 生效唯一判据：日志出现 `Compiling subgraph with CINN backend`。
- **3.3.0 下 `to_static` 一旦包裹，CINN 必然接管，`FLAGS_use_cinn=false` 被覆盖** → to_static 与 CINN 是同一件事，不分设两个开关。
- **FLAGS 必须在 `import paddle` 之前设置**（进程内首次 import paddle 时读取）。
- to_static 包裹顺序：权重加载**之后**、`DataParallel` **之前**。
- 首次编译 ~100s warmup；短跑总耗时不准，**加速比只看稳态吞吐**。
- **三个 FLAGS 各自的必要性**：`FLAGS_use_cinn` 是 CINN 总开关；`FLAGS_prim_enable_dynamic` + `FLAGS_prim_all` 是前端组合算子降级（把非基础算子拆为基础算子组合），CINN 后端依赖它——文档明确"现阶段需手动设"，单独 `use_cinn` 不够。故开关打开时**三个都设**。

### 2.3 代码接入点（已核对行号，源码一致性已确认）
- `main.py:6` 的 `import paddle` 在 Hydra 解析配置**之前** → FLAGS 无法用 Hydra 配置控制，只能环境变量。
- 训练 forward 调用：`trainer.py:465` `model("fwd", **model_input)`；字符串 `mode` 作为位置参数，trace 时当常量内联，`"fwd"`/`"generate"` 各编译一图，无需改 `forward(mode, **kwargs)` 签名（`transformer_wrappers.py:55-65`）。
- 训练循环：`trainer.py:460-464` autocast（bf16）→ `:465` forward → `:466-468` denormalize → `:469` loss → `:470` `self.data_loss += data_loss.item()` → `:471` `self.optimize(data_loss)`（内部 `:181` 非 AMP backward / `:193` AMP scaled backward + step）。
- DataParallel：`trainer.py:33-36`，包裹 `modules["model"]`。
- 权重加载：`build_model.py:52-65`；`.to(device)`：`:73-75`；`return`：`:76`。

## 3. 方案设计（已选：单一环境变量开关）

### 3.1 开关
- 新增环境变量 `PROSE_TO_STATIC`（默认 `"0"` 关）。
- `main.py` 顶部、`import paddle`（第 6 行）**之前**插入。开关**双向显式**（幂等，避免 shell 残留 FLAGS 污染）：
  ```python
  import os
  if os.environ.get("PROSE_TO_STATIC", "0") == "1":
      os.environ["FLAGS_prim_enable_dynamic"] = "true"
      os.environ["FLAGS_prim_all"] = "true"
      os.environ["FLAGS_use_cinn"] = "true"
  else:
      os.environ["FLAGS_prim_enable_dynamic"] = "false"
      os.environ["FLAGS_prim_all"] = "false"
      os.environ["FLAGS_use_cinn"] = "false"
  ```
- **不加 Hydra 配置项**，避免与 env 形成双开关（死分支）。
- 关闭分支行为与原版纯动态一致（FLAGS 显式置 false）。

### 3.2 to_static 包裹
- 位置：`build_model.py`，紧贴 `return modules`（`:76`）**之前**，且在 `reload_model`（`:52-65`）与 `.to(device)`（`:73-75`）**之后**。
- **关键落地细节**：插入点必须在**函数体顶层缩进**（与 `return` 同级），**不要**沿用第 75 行的缩进——第 75 行位于 `if not params.cpu:` 条件块内，照搬缩进会把 to_static 错误包进 CPU 分支，CPU 模式下不包裹。
- 写法：
  ```python
  if os.environ.get("PROSE_TO_STATIC", "0") == "1":
      from paddle.jit import to_static
      logger.info("Wrapping model with paddle.jit.to_static (CINN) ...")
      for k in list(modules.keys()):
          modules[k] = to_static(modules[k], full_graph=True)
  return modules
  ```
- **lazy trace**：不传 `input_spec`，首次 forward 按实际输入 trace。
- 下游 `Trainer.__init__` 的 `DataParallel`（`trainer.py:33-36`）自动位于其后，顺序正确（DP 在外、to_static 在内）。

### 3.3 trace 路径审计（动转静能跑的前提）
审计 forward 链路（`transformer_wrappers.py` → `embedder.py` → `transformer.py` → `attention_utils.py`）中所有 torch 风格调用，规则：
- `.view(...)` → `.reshape(...)`（手册坑二：3.3 `pd_op.view_shape` 反向 `view_shape_grad` 输出定义注册为空，必崩）。
- 冗余恒等 view（如 `.view(-1, d)` 而原形状已是 `[-1, d]`）→ 直接删（KISS）。
- `.size()` / `.size(N)` → `.shape` / `.shape[N]`；`.repeat(...)` → `paddle.tile(...)`（手册坑一：SOT/AST 翻译器无法解析兼容层方法，两个翻译器都中招，`full_graph` 绕不开）。
- **`assert` 内的 `.size()` 同样要改**（手册推荐彻底解法）：forward 链路上的 assert 在 trace 时仍被追踪器看到，必须改 paddle 原生。**陷阱**：paddle `.shape` 返回 **list**（非 torch 的 tuple），故 `assert x.size() == (a,b)` 必须写成 `assert tuple(x.shape) == (a,b)`——漏 `tuple()` 会变 `list == tuple` 恒 False。
- 「断言内不改」**仅指不在 forward 链路上的断言**（如纯推理路径 `generate_beam` 等）。

## 4. 验证方案（本地单卡）

每组「≥3 步 forward+backward+optimize 完成、loss 有限、无异常」即算通过。用 `dryrun=1`（自动 `n_steps_per_epoch=5`）或显式小 `n_steps_per_epoch`。

### 4.1 `x_num` 双层语义（关键，避免维度失配）
`x_num` 有两层（`all_datasets.py:249`）：
- 顶层 `data.x_num`（=64）：模型目标空间分辨率。
- 子配置 `data.shallow_water.x_num`：数据**原始**空间分辨率，用于算下采样步长 `x_step = shallow_water.x_num // data.x_num`，切片 `data[::x_step, ::x_step]`。

| 数据集 | 原始分辨率 | shallow_water.x_num | x_step | 模型输入 | 命令行覆盖 |
|--------|-----------|---------------------|--------|---------|-----------|
| 小型 | 64×64 | 64（默认） | 1 | 64×64 | 仅覆盖 `data_path` |
| 原版 | 128×128 | **128（必须覆盖）** | 2 | 64×64 | `data_path` + `data.shallow_water.x_num=128` |
| 原版不改 | 128×128 | 64（默认） | 1 | **128×128** | ❌ 与预训练权重（64）失配 |

### 4.2 四组验证矩阵
| 组别 | PROSE_TO_STATIC | 数据集 | 关键覆盖 |
|------|-----------------|--------|---------|
| 基线-小型 | 不设 | 小型 64×64 | `data=shallow_water_minimal` + `data.shallow_water.data_path=<小型>` |
| 基线-原版 | 不设 | 原版 128×128 | 同上 + `data.shallow_water.x_num=128` |
| CINN-小型 | `=1` | 小型 64×64 | 同基线-小型 |
| CINN-原版 | `=1` | 原版 128×128 | 同基线-原版 |

- 四组均加载预训练权重：`reload_model=/home/lkyu/baidu/prose-fd/models/prose_fd_paddle/prose_fd_converted.pdparams`。
- CINN 组须确认日志出现 `Compiling subgraph with CINN backend`。
- 工作目录 `prose_fd_paddle/`。

## 5. 加速比测量

- 数据集：小型 64×64（快）。两组：基线（env 不设）vs CINN（`PROSE_TO_STATIC=1`），其余参数完全一致。
- **精度对齐**：基线与 CINN 组**必须在同一 `amp` 设置**下测量（默认 `amp=1` bf16），并在产出文件记录实际精度；若 CINN+bf16 不兼容只能 fp32 跑通，则两组均改 fp32 对比并在报告注明口径。
- **稳态纯训练时间**（排除数据加载 + GPU 异步）——计时边界伪代码（注入 `trainer.iter`，仅 benchmark 模式开启）：
  ```python
  # data_input 已在循环顶部取数（不计入）
  if benchmark: paddle.device.cuda.synchronize()
  t0 = time.perf_counter()
  data_output = model("fwd", **model_input)   # trainer.py:465
  ...                                         # denormalize + loss（:466-469）
  self.optimize(data_loss)                    # :471（含 backward + step）
  if benchmark: paddle.device.cuda.synchronize()
  t1 = time.perf_counter()
  pure_train_sec += (t1 - t0)                 # 累计 N 步，再 /N 求均值
  ```
  - `paddle.device.cuda.synchronize()` 必须有，否则 GPU 异步执行致计时失真。
  - 数据加载位于循环顶部取数，不在 `[t0, t1]` 区间内。
- **端到端时间**：同 N 步 wall-clock / N（含数据加载与日志）。
- **步数 N**：默认 **N ≥ 1500**（手册：<1500 步总耗时反比基线长；长训练才净收益）。实现时先跑确认 warmup 已过。
- **稳态判据**：丢弃 CINN 组前 K 步（K = 日志中 `Compiling subgraph with CINN backend` 出现到不再有新编译为止的步数），对剩余步求单步耗时变异系数 `CV < 5%` 视为稳态。
- **指标定义**：`eq/s` 中 **eq = step**（避免 PDE equation 歧义），即 step/s。报告稳态 step/s 加速% + 端到端加速%。
- **预期**：端到端短跑口径下 CINN 因 warmup 占比大可能更慢，属正常；以稳态 step/s 为准。

## 6. 不做清单（YAGNI / 敏捷）
- 不预改 `symbol_utils`：**已核实** `prose_fd_paddle/symbol_utils/environment.py` 是离线 batch 编码（`batch_equations` 等），**无**手册坑三的 `gen_expr`/`X_to_fit` 在线逐样本生成路径 → **坑三不适用于本项目**。
- 不改 `evaluate.py:387` 的 `"cuda"` 硬编码（不阻塞训练侧）。
- 不传 `input_spec`；遇 retrace 风暴再给关键维设 `-1` 或分桶。
- 不加 Hydra `to_static` 配置项。
- 不做 `paddle.jit.save` 推理导出。

## 7. 风险与应对
| 风险 | 概率 | 应对 |
|------|------|------|
| forward 链路含 `.view()` 触发反向 bug | 中 | §3.3 审计替换为 `.reshape()`；排查时复现脚本须保持图规模 > SOT 的 `MIN_GRAPH_SIZE=10`，否则静默回退动态图误判 |
| 遇 `view_shape_grad` 错误 | — | **只查 `.view()` 替换**；勿试 pass flag / 方法级 to_static / 升级 paddle（手册已验证均无效，`FLAGS_pir_apply_inplace_pass=false` 亦无用） |
| 动态 shape（batch / symbol 长度）触发 retrace | 中 | lazy trace + 看日志；风暴再分桶/设动态维 |
| AMP(bf16)+to_static 不兼容 | 低 | 实测；必要时 `amp=0` 二分排查；确认不兼容则两组均 fp32 对比并注明 |
| 进程「不退出但无进展」 | 低 | 通用排查预案：`ps` 看 `Sl` 状态 + 日志 `Exception in thread`（数据线程异常不致主进程退出） |
| lazy trace 首步编译卡顿误判挂死 | 低 | 日志提示「首步编译需数十秒」 |

## 8. 实现步骤（高层，由 writing-plans 细化）
1. trace 路径审计：grep + 人工确认 forward 链路全部 `.view()`/`.size()`/`.repeat()`（含 assert 内）。
2. 改 `main.py`（顶部 FLAGS 双向注入）、`build_model.py`（to_static 包裹，顶层缩进）、trace 路径文件（view→reshape、size→tuple(shape)、repeat→tile）。
3. 基线两组跑通（确认不破坏原功能）。
4. CINN 两组跑通（确认日志 `Compiling subgraph with CINN backend` + 无 trace/反向错误）。
5. 注入计时 → 加速比测量 → 写 `docs/cinn-to-static-benchmark.md`（含命令、原始计时、精度口径、稳态 step/s 与端到端加速%）。
