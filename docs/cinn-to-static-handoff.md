# CINN 动转静训练优化 — 交接报告（Handoff）

> 生成时间：2026-07-29。用途：主线任务交接给其他 agent 深入调查 einops 阻塞。
> 工作目录：`/home/lkyu/baidu/prose-fd/prose_fd_paddle/`（所有命令在此目录执行）。

---

## 1. 主线任务目标

为 `prose_fd_paddle` 接入 `paddle.jit.to_static`，使 CINN 编译器真正参与训练图编译，获得稳态训练加速。

三个交付要求（用户原话）：
1. **接入动转静**：用一个开关控制 to_static+CINN（Paddle 3.3.0 下 to_static 必开 CINN，二者绑定）。
2. **本地单卡跑通两个数据集**：小型 64×64 / 原版 128×128，"跑通"= 几步训练能正常进行。
3. **确定性加速比数据**：稳态纯训练 + 端到端，写入仓库文件（若单 epoch 可接受用单 epoch，否则固定步数消除随机误差）。

约束：敏捷开发，仅解决提出的问题，禁止过度设计与假想问题；**禁止主动 git commit/分支**（项目 engineer-professional 约定）；**禁止改 `prose_fd/`（torch 参考实现）**。

## 2. 设计文档与计划（已批准）

- **Spec**：`docs/superpowers/specs/2026-07-29-cinn-to-static-design.md`
- **Plan**：`docs/superpowers/plans/2026-07-29-cinn-to-static.md`
- **方案 A（已选）**：单一环境变量 `PROSE_TO_STATIC`（默认 `"0"` 关），在 `main.py` 顶部 `import paddle` 之前设三个 FLAGS（`FLAGS_prim_enable_dynamic`/`FLAGS_prim_all`/`FLAGS_use_cinn`，双向显式），并在 `build_model.py` return 前对 model 施加 `paddle.jit.to_static(model, full_graph=True)`（lazy trace，不传 input_spec）。

## 3. 关键运行参数（重要，避免 OOM/worker 崩）

**WSL2 单卡 RTX 4060 Ti 16GB，paddle 3.3.0（CINN 已编译），libcuda.so 在 `/usr/lib/wsl/lib`。**

稳定跑法（经 `tests/test_training_data_smoke.py` 与 Poseidon 记录确认）：
```
dryrun=1 amp=0 compile=0 use_wandb=0 num_workers=0 num_workers_eval=0 batch_size=2 batch_size_eval=2
```
- **`num_workers=0` 必需**：paddle GPU DataLoader 多进程 worker（fork+CUDA）不稳定（`CUDA initialization error`），Poseidon 记录原文 "Using single-process data loading on gpu:0 to avoid Paddle worker instability"。
- **`batch_size=2` 必需**：`dryrun` 默认 `batch_size=128` 会让显存溢出到主机内存 → WSL 关机。
- 权重路径：`/home/lkyu/baidu/prose-fd/models/prose_fd_paddle/prose_fd_converted.pdparams`。
- **权重分辨率**：CKPT `conv_proj.0.weight=(1024,4,16,16)` → 训练时 `x_num=128`（patch_resolution=16）。**只与原版(128)兼容，与小型(64, kernel=8)不兼容**。
  - 小型跑通 = from scratch（不加载权重）。
  - 原版加载权重 = `data.x_num=128 data.shallow_water.x_num=128`（不下采样，输入 128×128 匹配权重）。
- 数据集：小型 `/home/lkyu/baidu/PDEBench/data/2D_rdb_NA_NA/2D_rdb_NA_NA.h5`；原版 `/home/lkyu/baidu/PDEBench/pdebench/data_download/data/2D/shallow-water/2D_rdb_NA_NA.h5`。

## 4. 已完成

- **T1 开关注入 ✅**：`main.py` 顶部 FLAGS 双向注入；`build_model.py` return 前 to_static 包裹（顶层缩进）。import smoke 通过。开关=0 时动态图行为等价原版。
- **T3 基线两组跑通 ✅**（开关=0，from scratch / 加载权重）：
  - 基线-小型 from scratch：5 步 loss 0.92→0.85，eval rel_l2=0.124。
  - 基线-原版加载权重：loss 0.00027（预训练近收敛），eval rel_l2=0.0348（权重生效）。无 OOM。
- **T2 fwd 链路 torch 风格 API 修复 ✅**（动态图等价）：`.size(N)→.shape[N]`、`.size()→.shape`、`.view(→.reshape(`，在 `transformer_wrappers.py`/`embedder.py`/`transformer.py`/`attention_utils.py`。详见第 6 节。

## 5. CINN trace 迭代修复记录（T4，已逐个解除 4 个 trace 错）

`PROSE_TO_STATIC=1` 后 to_static trace 顺序暴露并修复：
1. **omegaconf DictConfig.get**（transformer_wrappers.py:88 `self.config.get("carry_last_frame",0)`）→ SOT trace 触发 `isinstance(..., DictKeyType.__args__)` 崩。**修**：`__init__` 提取 `self.carry_last_frame = config.get(...)`，fwd 用 `self.carry_last_frame`。**DictConfig 属性访问（self.config.X）未崩**，无需转 SimpleNamespace。
2. **`.size(N)` AST 错**（embedder.py:314 `data.size(0)`）→ 手册坑一典型 `full(): argument (position 2) must be double, but got str`。**修**：`.size→.shape`。
3. **list==tuple assert**（attention_utils.py:77 `assert key_padding_mask.shape == (bs,k_len)`）→ list `[2,40]` == tuple `(2,40)` 恒 False。**修**：`tuple(key_padding_mask.shape) == (...)`。
4. **当前阻塞：einops**（见第 7 节）。

## 6. 代码改动清单（接手前仓库实际状态）

| 文件 | 改动 |
|------|------|
| `prose_fd_paddle/main.py` | 顶部 `import paddle` 前插入 `PROSE_TO_STATIC` → 三 FLAGS 双向注入 |
| `prose_fd_paddle/models/build_model.py` | 加 `import os`；`return modules` 前 to_static 包裹（顶层缩进，读同 env） |
| `prose_fd_paddle/models/transformer_wrappers.py` | PROSE_2to1 `__init__` 加 `self.carry_last_frame`；fwd 的 `.get(...)` → `self.carry_last_frame`；`:90/:116/:173/:189` `.size(0)→.shape[0]` |
| `prose_fd_paddle/models/embedder.py` | `.size(0)→.shape[0]`、`.size(1)→.shape[1]`、`.size()→.shape`、`.view(→.reshape(`（replace_all，含 `__main__` print，无害） |
| `prose_fd_paddle/models/transformer.py` | 同上四模式 replace_all（含非 fwd 的 SymbolDecoder:642/644，无害） |
| `prose_fd_paddle/models/attention_utils.py` | 同四模式 + `.size(-2)→.shape[-2]`；`:77` `assert ...shape==(…)→tuple(...shape)==(…)` |

注：`.view(→.reshape(` 保持原参数（paddle reshape 支持 *args 和 list 两种形式，等价）。

## 7. 当前阻塞：einops 0.8.2 在 to_static 下不兼容

**现象**（`PROSE_TO_STATIC=1`，trace 穿过整个 fwd 到 `embedder.decode` 的 `post_proj`）：
```
File "models/transformer_wrappers.py", line 122, in fwd
    data_output = self.embedder.decode(data_output)
File "models/embedder.py", line 336, in decode
    data_output = self.post_proj(data_output)
... paddle.nn.Sequential → Rearrange layer
File "einops/layers/paddle.py", line 13, in forward
    return self._apply_recipe(input)
File "einops/layers/__init__.py", line 43, in _apply_recipe
    return _apply_recipe(backend=backend, recipe=self._multirecipe[len(x.shape)], ...)
TypeError: _apply_recipe() got an unexpected keyword argument 'backend'
```

**已确认事实**：
- einops 版本 `0.8.2`（`/home/lkyu/miniconda3/envs/paddletorch/lib/python3.10/site-packages/einops`）。
- 动态图（开关=0）下 einops 正常（基线两组跑通）。
- `einops/layers/__init__.py:43` 的 `_apply_recipe(self, x)` 方法**调用**全局 `_apply_recipe(backend=backend, recipe=..., tensor=x, reduction_type="rearrange", axes_lengths=...)`，但被调用的全局 `_apply_recipe` 报不接受 `backend` kwarg。
- embedder 的 einops 用法：
  - `einops.layers.paddle.Rearrange` 层：`:241`（deep 分支，默认不用）、`:276`（**post_proj 非 deep 分支，fwd 路径**）。
  - `einops.rearrange` 函数：`:315`、`:317`（encode）、`:337`（decode）。函数式 rearrange 在 trace 中是否也会崩待验证（当前崩在层式 Rearrange :276 via post_proj :336）。
- prose_2to1 默认 `embedder.deep=0`，故 post_proj 走 `:275-303` 非 deep 分支（含 Rearrange :276）。

**疑似根因**：einops 0.8.2 的 `einops.layers.__init__._apply_recipe` 与其调用的全局 `_apply_recipe` 签名不一致（`backend` kwarg）。动态图未触发可能是因为 trace 静态化 `len(x.shape)` / 改变了全局 `_apply_recipe` 的解析（SOT 转换），暴露签名不匹配。需接手 agent 确认。

**待探索的解法方向**（按优先级，未验证）：
1. **einops 版本**：升/降级 einops（如 0.7.x 或最新）看是否修复 paddle+to_static。需评估对动态图的影响。
2. **替换 einops 为 paddle 原生**：把 `Rearrange` 层（:276）与 `einops.rearrange`（:315/317/337）等价改写为 `reshape+transpose`。工作量大但根治。需逐处推导 einops pattern 的 reshape/transpose 等价。
3. **让 einops 层不被静态化**：to_static 部分图？或 `paddle.no_grad()`？但 post_proj 在 fwd 训练路径，需参与反向，不能简单绕过。
4. **einops 是否有 to_static 兼容开关/补丁**：查 einops changelog / issue。

**接手验证命令**（确认阻塞复现）：
```
cd /home/lkyu/baidu/prose-fd/prose_fd_paddle
PROSE_TO_STATIC=1 timeout 290 python main.py model=prose_2to1 data=shallow_water_minimal optim=adamw dryrun=1 amp=0 compile=0 use_wandb=0 num_workers=0 num_workers_eval=0 batch_size=2 batch_size_eval=2 data.shallow_water.data_path=/home/lkyu/baidu/PDEBench/data/2D_rdb_NA_NA/2D_rdb_NA_NA.h5 exp_name=t4_cinn_einops 2>&1 | tail -45
```

## 8. 未完成

- **T4 CINN 跑通**（阻塞在 einops，第 7 节）：解除阻塞后需验证：
  - CINN-小型 from scratch 跑通（日志 `Compiling subgraph with CINN backend` + 几步训练 loss 正常）。
  - CINN-原版加载权重跑通。
- **T5 加速比测量**：`trainer.iter` 注入 `PROSE_BENCHMARK=1` 计时（forward 前 sync+t0，optimize 后 sync+t1，累计 N 步）；N≥1500；基线 vs CINN 同 amp 口径（建议 amp=0 fp32，避开 AMP+CINN 未知交互）；写 `docs/cinn-to-static-benchmark.md`。
  - 注意 CINN 首次编译 ~100s warmup，看稳态 step/s 不看短跑总耗时（踩坑手册经验）。

## 9. 参考资料

- **CINN 官方文档**：https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/paddle_v3_features/cinn_cn.html （要点：`paddle.jit.to_static(full_graph=True)` + 三 FLAGS；FLAGS 须 import paddle 前设）
- **踩坑手册**：`/home/lkyu/baidu/工作总结/CINN动转静训练踩坑手册.md` （3.3.0；坑一 .size/.view trace、坑二 view_shape_grad 反向、坑四 to_static 与 CINN 绑定；生效判据 `Compiling subgraph with CINN backend`；稳态测量）
- **Poseidon 迁移记录**：`/home/lkyu/baidu/工作总结/Poseidon迁移实验记录.md` （single-process data loading 经验；NaN backward 修复；N卡/天数 CINN 训练日志）
- **Spec**：`docs/superpowers/specs/2026-07-29-cinn-to-static-design.md`
- **Plan**：`docs/superpowers/plans/2026-07-29-cinn-to-static.md`
- **稳定跑法来源**：`prose_fd_paddle/tests/test_training_data_smoke.py`（`build_sw64_cfg`：bs=2, num_workers=0, amp=0, compile=0）
- **权重转换工具**：`prose_fd_paddle/tools/convert_torch_ckpt_to_paddle.py`；对齐工具 `prose_fd_paddle/tools/forward_pretrained_paddle.py`

## 10. 接手建议

1. 先复现第 7 节阻塞命令，确认 einops `_apply_recipe backend` 错误。
2. 调查 einops 0.8.2 全局 `_apply_recipe` 签名（`einops/einops.py`）vs `layers/__init__.py:43` 调用，定位 `backend` kwarg 不匹配根因；判断动态图为何不触发。
3. 优先尝试 einops 版本调整（最低风险）；若不可行，评估替换 einops.Rearrange/rearrange 为 paddle 原生 reshape+transpose（embedder 5 处）。
4. 解除 einops 阻塞后，继续 T4（CINN 跑通两数据集）→ T5（加速比）。
5. 全程不主动 git commit；改 `prose_fd_paddle/` 不动 `prose_fd/`；运行用第 3 节稳定参数（num_workers=0, bs=2）防 OOM。
