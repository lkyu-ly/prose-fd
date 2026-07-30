# Paddle full-graph residual Tensor kwargs 运行时输入缺失根因与修复报告

- 日期：2026-07-29
- 范围：`prose_fd_paddle/` 的 `paddle.jit.to_static(full_graph=True) + CINN` 训练/评估链
- 环境：PaddlePaddle 3.3.0、WSL2、RTX 4060 Ti 16GB、CUDA 13.x
- 最终状态：64×64 from scratch 与 128×128 预训练权重两组均完成 5 步训练和评估

## 1. 最终结论

根因不是 CINN 重复编译、`EventsWaiter` 死锁、Embedding、`index_select` 或
`Conv2DTranspose` 本身，而是 Paddle 3.3.0 的 full-graph AST/PIR 路径没有把顶层
`forward(self, mode, **kwargs)` 中残余的 Tensor kwargs 正确传到缓存图运行时输入：

- `FunctionSpec.args_to_input_spec()` 会为 residual kwargs 创建 InputSpec；
- `ConcreteProgram.pir_from_func_spec()` 转换 kwargs 时丢弃对应的 `program_inputs`；
- `ASTStaticFunction._perform_call()` 执行缓存图时只调用 `partial_program_layer(args)`。

因此“错误绑定/内部缓冲复用”是这条输入传递缺口在执行期的结果；非法值具体来自哪个
allocator block 仍需 Paddle 维护者确认，不能把它写成已证实的底层实现机制。

项目原调用链为：

```python
data_output = model("fwd", **model_input)
```

其中 `model_input` 同时包含 float32 数据、时间 Tensor、int64 symbol ids 和 bool mask。
错误运行时输入最初在 `EmbeddingKernel` 中表现为长期不返回；绕过 Embedding 后，同一损坏
状态又分别在 `cinn_jit` 和 `pd_op.conv2d_transpose` 中表现为 illegal memory access
或 CUDNN internal error。这些算子是检测点，不是源头。

最终修复是把顶层模型接口改为显式 Tensor 参数，并让训练、评估、rollout 全部使用
位置参数调用。恢复原生 symbol/type `nn.Embedding` 后，两组完整 CINN dryrun 仍通过，
证明不需要逐算子规避。

## 2. 与 Paddle 官方 issue 的对应关系

| Issue | 结论 | 与本问题关系 |
|---|---|---|
| [#79559](https://github.com/PaddlePaddle/Paddle/issues/79559) | `view_shape_grad` kernel output metadata mismatch | 前序 `.view → .reshape` 问题，已修；不是本次运行时输入问题 |
| [#79579](https://github.com/PaddlePaddle/Paddle/issues/79579) | `full_graph=True` 下方法与全局函数同名时错误绑定 | 对应前序 einops 问题；不是本次执行期根因 |
| [#79580](https://github.com/PaddlePaddle/Paddle/issues/79580) | residual Tensor kwargs 未正确进入缓存图运行时输入，位置参数对照正常 | 与本问题同款，且本地 A/B、源码数据流与项目完整修复均吻合 |

#79580 的最小复现使用 float32 `data` 和 int64 `token_ids`：kwargs 版本第一次正常、
第二次 Embedding 读到不存在的 ID；显式签名并使用位置参数后连续三次正常。仓库环境中
已有对应复现与对照：

- `/tmp/paddle-cinn-kwargs-embedding-no-mode.py`
- `/tmp/paddle-cinn-kwargs-embedding-no-mode.log`
- `/tmp/paddle-cinn-positional-embedding-control.py`
- `/tmp/paddle-cinn-positional-embedding-control.log`

本项目图更大，因此输入缺口在第一次完整执行就暴露，但“kwargs 失败、位置参数正常”
的 A/B 关系完全一致。

源码证据：[`FunctionSpec.args_to_input_spec()`](https://github.com/PaddlePaddle/Paddle/blob/v3.3.0/python/paddle/jit/dy2static/function_spec.py#L112-L166)、[`ConcreteProgram.pir_from_func_spec()`](https://github.com/PaddlePaddle/Paddle/blob/v3.3.0/python/paddle/jit/dy2static/program_translator.py#L1271-L1324) 与 [`ASTStaticFunction._perform_call()`](https://github.com/PaddlePaddle/Paddle/blob/v3.3.0/python/paddle/jit/dy2static/program_translator.py#L834-L850)。截至 2026-07-29，#79580 仍无维护者评论或关联修复 PR；调查时 `develop` HEAD `0c4f2753507a8634aff45637ac77826d25ded5cd` 仍保留相同输入传递逻辑。

## 3. 调查证据链

### 3.1 原始等待栈只是表象

原始日志 `/tmp/t4_cinn_symbol2.log`：

- CINN 仅编译一次，约 108 秒。
- Executor 启动后 8 分钟无 step loss。
- timeout 栈停在 `MultiThreadRunInstructionList → EventsWaiter::WaitEvent`。

这只能证明主线程在等待，不能定位工作线程中的实际阻塞。

### 3.2 serial-run 暴露 EmbeddingKernel

只增加 `FLAGS_new_executor_serial_run=true` 后，原配置仍超时，但栈变为：

```text
RunInstructionBaseAsync
→ RunInstructionBase
→ PhiKernelInstruction::Run
→ phi::EmbeddingKernel<float, phi::GPUContext>
```

日志：`/tmp/t4_cinn_serial.log`。

该实验排除了 `EventsWaiter` 自身死锁，但仍只定位到第一个检测错误缓冲的算子。

### 3.3 逐算子规避只会移动失败位置

阶段性实验结果：

- 固定 type embedding 改为 Parameter 行切片：不再等待 Embedding，但随后
  `pd_op.conv2d_transpose` 报 `CUDNN_STATUS_INTERNAL_ERROR`。
- type embedding 改为 `index_select`：失败提前为两个 `cinn_jit` illegal memory access。
- 使用完全一致输入形状 `[2, 2560, 1024]` 单独运行真实
  `ConvEmbedder.post_proj`：CINN 正常输出 `[20, 4, 64, 64]`，约 10.9 秒。

因此 ConvTranspose 本身正常；不同 lookup 写法只是改变缓冲布局，使损坏在不同位置暴露。

相关日志：

- `/tmp/t4_cinn_type_embedding_fix.log`
- `/tmp/t4_cinn_type_index_select.log`

### 3.4 根接口修复使全部症状同时消失

改为显式签名和位置参数后：

- `EmbeddingKernel` 不再停滞。
- `cinn_jit` 不再 illegal memory access。
- `pd_op.conv2d_transpose` 不再报 CUDNN internal error。
- 恢复两个原生 Embedding 后仍通过。

这确认三个执行期现象共享同一个 residual kwargs 运行时输入传递根因。

## 4. 最终代码修改

### 4.1 顶层模型使用显式参数

文件：`prose_fd_paddle/models/transformer_wrappers.py`

`PROSE_2to1.forward` 从：

```python
def forward(self, mode, **kwargs):
    return self.fwd(**kwargs)
```

改为显式参数：

```python
def forward(
    self,
    mode,
    data_input,
    input_times,
    output_times,
    symbol_input,
    symbol_padding_mask=None,
):
    ...
```

`PROSE_1to1.forward` 同步使用显式参数；两个 `fwd` 均移除未使用的 `**kwargs`。

### 4.2 所有顶层调用使用位置参数

修改文件：

- `prose_fd_paddle/trainer.py`
- `prose_fd_paddle/evaluate.py`
- `prose_fd_paddle/tests/test_training_data_smoke.py`

训练、常规评估和 rollout 均按固定顺序传入：

```text
mode, data_input, input_times, output_times, symbol_input, symbol_padding_mask
```

### 4.3 移除症状规避

最终实现恢复：

- `TransformerSymbolEncoder.word_embeddings(x)`
- `TransformerFusion.type_embeddings(type_ids)`

没有保留 Parameter slice、one-hot、`index_select` 或 ConvTranspose 替换。

## 5. 为什么这是最小且优雅的修复

- KISS：修复一个输入边界，不为多个下游算子增加特殊分支。
- YAGNI：不更换 Paddle/CUDNN，不增加 FLAGS，不改模型结构。
- DRY：训练与评估遵循同一固定参数顺序，不复制 Embedding 语义。
- 兼容性：参数名、shape、checkpoint schema 和模型数学行为不变。
- 性能：保留框架原生 Embedding/ConvTranspose，不引入额外 gather 或 one-hot。

## 6. 最终验证

### 6.1 64×64 from scratch

日志：`/tmp/t4_cinn_positional_native_embedding.log`

结果：

```text
Compiling subgraph with CINN backend     1 次
New Executor is Running                  1 次
step 1 loss = 0.92370588
step 2 loss = 0.79512453
step 3 loss = 1.02005661
step 4 loss = 0.85602361
step 5 loss = 0.85464537
Eval data loss = 0.663602
Eval rel_l2 = 0.124147
```

无 traceback、FatalError、illegal memory access 或 CUDNN 错误。

### 6.2 128×128 预训练权重

日志：`/tmp/t4_cinn_positional_native_embedding_128.log`

结果：

```text
Compiling subgraph with CINN backend     1 次
New Executor is Running                  1 次
step 1 loss = 0.00027129
step 2 loss = 0.00013354
step 3 loss = 0.05661615
step 4 loss = 0.09739588
step 5 loss = 0.09828979
Eval data loss = 0.069546
Eval rel_l2 = 0.034804
```

无 traceback、FatalError、illegal memory access 或 CUDNN 错误。

### 6.3 测试与静态检查

```text
python -m pytest -q \
  prose_fd_paddle/tests/test_import_smoke.py \
  prose_fd_paddle/tests/test_transformer_parity.py \
  prose_fd_paddle/tests/test_attention_and_rotary.py

12 passed
```

`git diff --check` 通过。

`test_repository_hygiene.py` 另有一个与本次无关的既有失败：
`prose_fd_paddle/tools/convert_torch_ckpt_to_paddle.py` 被规则判为含 torch residual；
该文件未被本次修改。

## 7. 状态

- T4 CINN 小型与原版两组训练链已跑通。
- 未修改 `prose_fd/`。
- 未提交、未切分支、未修改依赖或系统配置。
- T5 稳态加速比测量不属于本次执行期问题调查，尚未执行。
