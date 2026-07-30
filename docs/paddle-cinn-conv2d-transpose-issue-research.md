# Paddle 3.3.0 CINN → `conv2d_transpose` 执行错误一手资料调查

> 调查日期：2026-07-29  
> 范围：仅使用 PaddlePaddle 官方源码 / issue / PR / 文档，以及 NVIDIA CUDA / cuDNN 官方文档；未修改项目源码，未安装或升级依赖。

## 1. 结论摘要

最终本地 A/B 已确认：**根因是 Paddle 3.3.0 dy2static + CINN 对顶层 Tensor `forward(**kwargs)` 的缓存图输入绑定/复用错误，与官方 issue [#79580](https://github.com/PaddlePaddle/Paddle/issues/79580) 同款；`pd_op.conv2d_transpose` 只是 index-select 临时掩盖 Embedding 越界后，下游第一个暴露损坏 CUDA 状态的算子。**

决定性 A/B 是：把顶层 `forward(**kwargs)` 及训练/评估调用改成显式位置参数后，恢复原生 symbol/type Embedding，64 和 128 两组完整 CINN 训练链均通过；保留 Tensor `**kwargs` 时，原生 Embedding 读到不存在的 ID，改用 `index_select` 仅把错误延后到 decoder 的 `conv2d_transpose`。因此修复点应是 **移除顶层 Tensor `**kwargs` 边界**，不是替换 Conv2DTranspose、切 cuDNN 算法或禁用 CINN。

Paddle 3.3.0 的 CINN 源码中没有 `conv2d_transpose` 的 CINN lowering/strategy 注册；该算子停留在 PHI kernel，最终在 [`ConvTransposeCudnnKernelImplV7`](https://github.com/PaddlePaddle/Paddle/blob/v3.3.0/paddle/phi/kernels/gpudnn/conv_transpose_kernel.cu#L51-L119) 中把转置卷积前向实现为 cuDNN 的 `cudnnConvolutionBackwardData()`，现场报错行 [`conv_cudnn_v7.h:727`](https://github.com/PaddlePaddle/Paddle/blob/v3.3.0/paddle/phi/kernels/gpudnn/conv_cudnn_v7.h#L704-L744) 正是这次调用。CUDA 官方明确说明后续 API 可以返回先前异步 launch 的错误，[`cudaGetLastError()` 文档](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-runtime-api/group__CUDART__ERROR.html#group__CUDART__ERROR_1g3529f94cb530a83a76613616782bd233) 写明其可能返回 “previous, asynchronous launches” 的错误；这解释了为什么输入绑定错误最终可在 cuDNN 调用处显形。

在 A/B 根因确认前，最有判别力、无需改代码的诊断是：

```bash
FLAGS_check_cuda_error=1 PROSE_TO_STATIC=1 ...
```

Paddle 3.3.0 的 [`CinnJitInstruction::Run()`](https://github.com/PaddlePaddle/Paddle/blob/v3.3.0/paddle/fluid/framework/new_executor/instruction/cinn_jit_instruction.cc#L341-L382) 会在每个 CINN instruction 前后执行 `cudaDeviceSynchronize()` + `cudaGetLastError()`；定义见 [`CUDAErrorCheck`](https://github.com/PaddlePaddle/Paddle/blob/v3.3.0/paddle/fluid/framework/new_executor/instruction/instruction_util.h#L72-L78)。结果解释：

- 若在 `CinnJitInstruction finish` 就报 illegal access：根因在前一个 CINN kernel / 其 buffer 映射，不是 cuDNN 算法。
- 若所有 CINN instruction 检查通过、仍在 `conv2d_transpose` 报错：再用 `FLAGS_cudnn_deterministic=1` 隔离 cuDNN heuristic 算法。
- 若只要逐 instruction 同步就能跑通：优先判断为 CINN JIT 与多线程 Executor 的事件/生命周期竞态；再用 trace/serial/sequential FLAGS 确定最小稳定规避。

最终 A/B 已提供比上述 FLAGS 更直接的因果证据，故这些 FLAGS 只保留为框架 issue 的定位资料，不是项目最终修复。

## 2. 现场调用链的官方源码含义

### 2.1 `conv2d_transpose` 前向实际调用 cuDNN backward-data

Paddle 3.3.0 在 [`ConvTransposeCudnnKernelImplV7`](https://github.com/PaddlePaddle/Paddle/blob/v3.3.0/paddle/phi/kernels/gpudnn/conv_transpose_kernel.cu#L51-L119) 中交换输入/输出 descriptor，选择 `ConvKind::kBackwardData` 算法，再调用 `ConvRunner<T, kBackwardData>`。后者在现场报错的 [`conv_cudnn_v7.h:727`](https://github.com/PaddlePaddle/Paddle/blob/v3.3.0/paddle/phi/kernels/gpudnn/conv_cudnn_v7.h#L723-L741) 调用：

```cpp
cudnnConvolutionBackwardData(...)
```

NVIDIA 对该函数的官方说明见 [`cudnnConvolutionBackwardData()`](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.13.0/api/cudnn-cnn-library.html#cudnnconvolutionbackwarddata)。因此堆栈中的 “BackwardData” 并不表示模型已进入反向传播；它是转置卷积前向的实现方式。

PROSE-FD 当前配置可静态推出该层参数：输入约为 `[20, 1024, 16, 16]`，weight `[1024, 32, 4, 4]`，`stride=4, padding=0, dilation=1, groups=1, NCHW, float32`，输出 `[20, 32, 64, 64]`。这是一组普通且有效的配置；它与 Paddle issue [#71179](https://github.com/PaddlePaddle/Paddle/issues/71179) 的非法 padding，以及 [#71180](https://github.com/PaddlePaddle/Paddle/issues/71180) 的 depthwise/groups 特殊路径不相同。

### 2.2 该 wheel 确实走 legacy v7 路径

Paddle 源码只有在编译时开启 `PADDLE_WITH_CUDNN_FRONTEND` 且运行时启用 frontend 时，才走 V8/frontend；否则走 V7 路径，分支见 [`conv_transpose_kernel.cu:392-431`](https://github.com/PaddlePaddle/Paddle/blob/v3.3.0/paddle/phi/kernels/gpudnn/conv_transpose_kernel.cu#L392-L431)。官方 pybind 的 [`is_compiled_with_cudnn_frontend()`](https://github.com/PaddlePaddle/Paddle/blob/v3.3.0/paddle/fluid/pybind/pybind.cc#L304-L309) 直接反映该编译宏。

本机只读探测结果为：

```text
paddle 3.3.0, build commit cbf3469
build CUDA 13.0, build cuDNN 9.13.0
runtime cuDNN 9.19.0
is_compiled_with_cudnn_frontend() == False
```

所以 `FLAGS_enable_cudnn_frontend=1` 对当前 wheel **不能构成规避**；二进制没有 frontend 路径。

编译时 cuDNN 9.13、运行时 9.19 也不是当前首要嫌疑。NVIDIA 的 [cuDNN API Compatibility](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.13.0/developer/forward-compatibility.html#cudnn-api-compatibility) 明确说明：同一 major 下，较新的 minor runtime 对较早 minor 构建的应用二进制向后兼容；且 `cudnnConvolutionBackwardData()` 被列为可 forward-compatible 的 legacy API。cuDNN 9.13 的 [Support Matrix](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.13.0/reference/support-matrix.html#gpu-cuda-toolkit-and-cuda-driver-requirements) 还把 CUDA 13.0 + Ada（compute capability 8.9）列为支持组合，并称 CUDA 13.0 是推荐、用于 heuristic tuning 的配置。

### 2.3 `CUDNN_STATUS_INTERNAL_ERROR` 不能单独证明 cuDNN 自身有 bug

NVIDIA 文档在相关 API 中把 `CUDNN_STATUS_INTERNAL_ERROR` 描述为内部 CUDA 函数返回错误状态（例如 [`cudnnSetFilterNdDescriptor()` 的状态说明](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.13.0/api/cudnn-ops-library.html#cudnnsetfilternddescriptor)）。而 `cudnnConvolutionBackwardData()` 自身的 documented returns 主要是 `BAD_PARAM`、`NOT_SUPPORTED`、`MAPPING_ERROR` 和 `EXECUTION_FAILED`；现场得到 `INTERNAL_ERROR` 且紧接着 CUDA context 报 illegal access，更符合“调用 cuDNN 时已遇到一个 CUDA 错误状态”这一解释，不能据此把首因锁死在该卷积算法。

## 3. 为什么 `conv2d_transpose` 是下游症状而非根因

本地因果链现已闭合：

```text
顶层 Tensor **kwargs 输入映射错误（#79580 同款）
  → 原生 Embedding 首先读到非法 ID
  → 用 index_select 临时绕开 Embedding 后，错误绑定/错误 buffer 仍存在
  → 后续 CINN/PHI 计算继续消费错误数据或损坏设备状态
  → 外部 pd_op.conv2d_transpose 的 cuDNN 调用首先同步并报 INTERNAL_ERROR
  → 异常清理时 cuModuleUnload 再报 illegal memory access
```

显式位置参数修复后，原生 Embedding 和 `conv2d_transpose` 同时恢复，且 64/128 两组完整 CINN 均通过。单一上游改动同时消除两个相隔很远的错误点，是“共同上游输入绑定错误”的强因果证据；若 Conv2DTranspose、其 weight layout 或 cuDNN algorithm 本身有独立缺陷，位置参数改动不应使其消失。

以下源码分析解释该错误为什么可能延迟暴露，但不再作为项目根因假设。

### 3.1 CINN JIT 只传 data pointer，不传 DenseTensor stride

[`CinnJitInstruction::FnPtrImpl::Run()`](https://github.com/PaddlePaddle/Paddle/blob/v3.3.0/paddle/fluid/framework/new_executor/instruction/cinn_jit_instruction.cc#L101-L116) 对每个 tensor argument 只执行：

```cpp
cinn_buffer->memory = reinterpret_cast<uint8_t*>(dense_tensor->data());
```

compiled kernel 的参数中没有传入 `DenseTensor::strides()`。因此：

- 简单 slice 的 byte offset 并不会天然丢失，因为 [`DenseTensor::data()`](https://github.com/PaddlePaddle/Paddle/blob/v3.3.0/paddle/phi/core/dense_tensor.cc#L178-L195) 已把 `meta_.offset` 加到 holder pointer 上。
- 但若 CINN group 的外部输入/输出是一个 **非连续 strided view**，JIT 边界没有运行时 stride 信息可供 kernel 使用；它必须在分组/lowering 时被正确物化为连续 buffer，否则存在按连续布局解释的风险。

这使“Parameter slice/view alias”需要拆成两类判断：普通 contiguous slice/offset 不是首要嫌疑；`transpose`、不可 reshape 的 view 或其他非连续 stride 才是相关方向。

PROSE-FD 在 `conv2d_transpose` 前正好有 `PatchTokensToGrid` 的 `reshape → transpose → reshape`。这不证明输出一定非连续——CINN 可能已生成物化 kernel——但它是应优先用 IR/同步检查验证的边界。

### 3.2 CINN JIT 的 Event/GC 逻辑在 3.3.0 源码中仍标 TODO

[`CinnJitInstruction` 头文件](https://github.com/PaddlePaddle/Paddle/blob/v3.3.0/paddle/fluid/framework/new_executor/instruction/cinn_jit_instruction.h#L29-L63) 明确写有：

```cpp
// TODO: Only implement core interface and need implement GC and Event logic.
```

同一实现会异步 launch CINN kernel，随后释放 argument wrapper，并清空 instruction 私有 temp-space tensor，见 [`cinn_jit_instruction.cc:356-374`](https://github.com/PaddlePaddle/Paddle/blob/v3.3.0/paddle/fluid/framework/new_executor/instruction/cinn_jit_instruction.cc#L356-L374)。Paddle allocator 可能通过 stream-aware 机制延迟真正回收，因此不能仅凭 `clear()` 断言 use-after-free；但在“多线程 Executor + CINN 外部 consumer”场景，Event/GC 尚未完整实现是 buffer lifetime / 跨 instruction 同步仍需优先隔离的直接源码证据。

### 3.3 Executor 默认启用 inplace pass，且当前确实使用多线程调度

`FLAGS_pir_apply_inplace_pass` 在 Paddle 3.3.0 默认 `true`，定义见 [`paddle/common/flags.cc:1812-1823`](https://github.com/PaddlePaddle/Paddle/blob/v3.3.0/paddle/common/flags.cc#L1812-L1823)；Executor 在 kernel lowering 后应用该 pass，见 [`executor_cache.cc:194-205`](https://github.com/PaddlePaddle/Paddle/blob/v3.3.0/paddle/fluid/framework/executor_cache.cc#L194-L205)。如果某个 CINN group output 的最后使用者、view alias 或外部 consumer 依赖分析有缺陷，inplace buffer 复用可能把问题放大。

PIR Executor 的 trace/multi-thread 选择及实际执行分支见 [`pir_interpreter.cc:119-126`](https://github.com/PaddlePaddle/Paddle/blob/v3.3.0/paddle/fluid/framework/new_executor/pir_interpreter.cc#L119-L126) 和 [`pir_interpreter.cc:1625-1642`](https://github.com/PaddlePaddle/Paddle/blob/v3.3.0/paddle/fluid/framework/new_executor/pir_interpreter.cc#L1625-L1642)。现场日志明确为 multi-thread mode。

## 4. 指定 issue/PR 的逐一对照

### 4.1 #79559：不是同款 buffer lifetime / illegal access

Issue [#79559](https://github.com/PaddlePaddle/Paddle/issues/79559) 的失败点是 `pd_op.view_shape_grad` 在 PIR lowering 时“PIR 有 1 个 result、kernel metadata 有 0 个 output definition”。它发生在 `loss.backward()`，用 `reshape` 替换 `view` 即绕过；没有 CUDA illegal access、cuDNN 或 `conv2d_transpose`。

判断：**不同根因**。它证明 Paddle 3.3.0 的 `Tensor.view()`/`view_shape_grad` 路径仍有 PIR metadata 缺口，但不能作为当前 parameter alias 或 buffer lifetime 的证据。当前错误发生在首个 forward。

### 4.2 #79579：纯 dy2static AST 名字绑定问题，与本问题无关

Issue [#79579](https://github.com/PaddlePaddle/Paddle/issues/79579) 在 CPU 即可复现，条件是 Layer 方法与模块级函数同名，`full_graph=True` 恢复 closure 时错误绑定，表现为转换期 `TypeError`。

判断：**完全不同根因**。它对应已修复的 einops 同名全局函数问题，不涉及 Parameter slice/view、CINN buffer、CUDA 或 cuDNN。

### 4.3 #79580：本问题的同款根因

Issue [#79580](https://github.com/PaddlePaddle/Paddle/issues/79580) 是 `forward(**kwargs)` 中 float32/int64 Tensor 在第二次缓存图执行时错误绑定/复用，Embedding 读到调用者输入中不存在的 ID，device assert 后同样在 `cuModuleUnload` 报 launch failure。

官方 issue 的最小复现给出了四个关键对照：

- 调用者侧 `token_ids` 始终合法且未被原地修改。
- 第一次缓存图执行正常，第二次 Embedding 读到输入中不存在的 ID。
- 计算与 Tensor 内容不变，只把 `forward(**kwargs)` 改为位置参数即可连续成功。
- device assert 后，进程清理同样在 `cuModuleUnload` 报 launch failure。

PROSE-FD 的完整图 A/B 与上述必要变量一致：

- 保留顶层 Tensor `**kwargs`：原生 Embedding 越界；改 `index_select` 后错误延后到 `conv2d_transpose`。
- 改为显式位置参数：可恢复原生 symbol/type Embedding；64/128 两组完整 CINN 均通过。
- 动态图一直正常，限定在 dy2static + CINN。

判断：**已通过项目级 A/B 确认为 #79580 同款。** issue 截至调查时仍是 open/new-issue、无 maintainer 评论或关联修复 PR，所以“Paddle 内部具体是哪一层错误复用 buffer”尚未由上游确认；但项目侧最小充分规避已经确定为显式位置参数。

### 4.4 三个指定 issue 的同款判定汇总

| Issue | 官方复现失败阶段 | 关键触发条件 | 与当前问题关系 |
|---|---|---|---|
| [#79559](https://github.com/PaddlePaddle/Paddle/issues/79559) | backward 的 `view_shape_grad` kernel metadata lowering | `Tensor.view()` + backward | 不同：PIR output definition 缺失，无 kwargs/Embedding/CUDA 越界 |
| [#79579](https://github.com/PaddlePaddle/Paddle/issues/79579) | full-graph AST 转换期 | Layer 方法与全局函数同名 | 不同：CPU 可复现的 closure/name binding；对应已解决 einops 问题 |
| [#79580](https://github.com/PaddlePaddle/Paddle/issues/79580) | CINN 缓存图执行期 | Tensor `forward(**kwargs)`；位置参数正常 | **同款：完整图位置参数 A/B 同时消除 Embedding 与 conv 报错** |

### 4.5 已知 transpose/local-buffer 修复 #69608 已包含在 3.3.0

官方 PR [#69608](https://github.com/PaddlePaddle/Paddle/pull/69608) 曾修复 CINN 在含 transpose kernel 时，local buffer index 的 unit variable 未在 resize buffer 前消除的问题。Paddle 3.3.0 源码已包含该修复后的 `ReplaceUnitVarToZero → EliminateCommonFactorOfLocalIndex → ResizeBufferToMaxVarRange` 顺序。

判断：它说明 **transpose + local buffer indexing 确实是 CINN 历史高风险区**，但该已知补丁不是当前版本缺失项；没有找到与当前形状和堆栈完全相同、尚未合入 3.3.0 的官方 issue/PR。

## 5. 已确认修复与备用诊断矩阵

### 5.1 项目最终修复

最小且优雅的修复是：顶层模型 forward 使用显式参数签名，训练和评估调用同步改为显式位置参数；移除为掩盖症状而引入的 symbol `index_select` 替代，恢复原生 Embedding。该修复：

- 不改变模型数学语义、参数布局或 checkpoint。
- 不修改 `Conv2DTranspose`，不依赖 cuDNN 算法 FLAGS。
- 不关闭 CINN，也不牺牲整个图的编译优化。
- 64 和 128 两组完整 CINN 已通过，直接覆盖原 agent 后续必经链路。

### 5.2 上游框架诊断备用项

所有 FLAGS 都必须在 `import paddle` 前设置，并应每次用新进程测试，避免 illegal access 后污染的 CUDA context 影响下一组结果。

| 优先级 | 单变量设置 | 官方源码作用 | 结果解释 / 是否适合长期规避 |
|---|---|---|---|
| 1 | `FLAGS_check_cuda_error=1` | 每个 instruction 前后 `cudaDeviceSynchronize()` + `cudaGetLastError()`；[定义](https://github.com/PaddlePaddle/Paddle/blob/v3.3.0/paddle/common/flags.cc#L2299-L2308)、[CINN 使用点](https://github.com/PaddlePaddle/Paddle/blob/v3.3.0/paddle/fluid/framework/new_executor/instruction/cinn_jit_instruction.cc#L341-L382) | 最强定位手段；很慢，只建议诊断一次。若因此跑通，说明异步同步/生命周期竞态高度可疑。 |
| 2 | `FLAGS_enable_pir_in_executor_trace_run=1` | 强制 PIR Executor 走 trace mode；[选择逻辑](https://github.com/PaddlePaddle/Paddle/blob/v3.3.0/paddle/fluid/framework/new_executor/pir_interpreter.cc#L119-L126)、[逐 instruction host 循环](https://github.com/PaddlePaddle/Paddle/blob/v3.3.0/paddle/fluid/framework/new_executor/pir_interpreter.cc#L1711-L1754) | 最直接隔离 multi-thread 调度。若稳定，可作为无代码规避候选，再测实际吞吐。 |
| 3 | `FLAGS_new_executor_serial_run=1` | 把 Executor thread pool 收敛为单 device thread；[定义](https://github.com/PaddlePaddle/Paddle/blob/v3.3.0/paddle/fluid/framework/new_executor/interpretercore.cc#L22-L25)、[线程数逻辑](https://github.com/PaddlePaddle/Paddle/blob/v3.3.0/paddle/fluid/framework/new_executor/interpreter/execution_config.cc#L113-L125) | 若 trace 成功，用它区分是否只是并发 host 调度竞态。 |
| 4 | `FLAGS_new_executor_sequential_run=1` | 按 Program 顺序给 GPU op 加依赖；[定义和应用](https://github.com/PaddlePaddle/Paddle/blob/v3.3.0/paddle/fluid/framework/new_executor/interpreter/dependency_builder.cc#L34-L42) | 若只有它成功，指向 dependency/event 缺失；单卡下是较优雅的稳定规避候选。 |
| 5 | `FLAGS_pir_apply_inplace_pass=0` | 关闭 kernel lowering 后的 PIR inplace pass | 若成功，指向 alias/lifetime/reuse；代价主要是峰值显存增加，必须继续使用稳定小 batch 防 OOM。 |
| 6 | `FLAGS_cudnn_deterministic=1` | 对 backward-data 强制 `CUDNN_CONVOLUTION_BWD_DATA_ALGO_1`；[算法选择](https://github.com/PaddlePaddle/Paddle/blob/v3.3.0/paddle/phi/kernels/gpudnn/conv_cudnn_v7.h#L271-L307) | 只有在 CINN instruction 同步检查通过后再测。若成功，是最小的 cuDNN 算法规避，可能变慢。 |
| 7 | `FLAGS_deny_cinn_ops=transpose` | 官方 deny-list 将指定 op 排除出 CINN；[flag 语义](https://github.com/PaddlePaddle/Paddle/blob/v3.3.0/paddle/common/flags.cc#L1161-L1171)、[匹配逻辑](https://github.com/PaddlePaddle/Paddle/blob/v3.3.0/paddle/cinn/hlir/framework/pir/utils.cc#L108-L125) | 若同步检查锁定 `PatchTokensToGrid` 所在 group，先只 deny `transpose`；仍失败再试 `transpose;reshape`。比全局关闭 CINN 更小。 |

不建议作为首选：

- `FLAGS_cudnn_exhaustive_search=1`：Paddle 源码明确注明 `conv2d_transpose` 禁用自动 autotune，是因为 exhaustive search 选出的某些算法会产生错误结果；见 [`conv_cudnn_v7.h:598-607`](https://github.com/PaddlePaddle/Paddle/blob/v3.3.0/paddle/phi/kernels/gpudnn/conv_cudnn_v7.h#L598-L607)。显式 exhaustive 仍可能触发该风险。
- `FLAGS_enable_cudnn_frontend=1`：当前 wheel 未编译 frontend，无效。
- `FLAGS_cinn_sync_run=1`：Paddle 3.3.0 在 [`paddle/cinn/runtime/flags.cc`](https://github.com/PaddlePaddle/Paddle/blob/v3.3.0/paddle/cinn/runtime/flags.cc#L179-L183) 定义了它，但在本次检出的 3.3.0 相关执行源码中未找到读取点，不能依赖它产生同步效果。
- 单纯调大/调小 `FLAGS_conv_workspace_size_limit`：该 flag 只约束算法 workspace，官方定义见 [`paddle/phi/core/flags.cc:19-33`](https://github.com/PaddlePaddle/Paddle/blob/v3.3.0/paddle/phi/core/flags.cc#L19-L33)。现场显存正常且错误不是 `ALLOC_FAILED`，优先级低于 deterministic 和边界同步检查。

## 6. 若向 Paddle 上游补充 issue 的最短证据链

项目根因已经通过 A/B 关闭，不再需要为项目修复继续枚举 FLAGS。若要补强 #79580 的上游报告，建议只保留以下最短证据：

1. 同一完整图、同一输入和参数，Tensor `**kwargs` 失败，显式位置参数成功。
2. 位置参数版本恢复原生 Embedding，并同时跑通 64/128 两组完整 CINN。
3. `index_select` 只改变错误暴露点：Embedding → `conv2d_transpose`，不能真正修复。
4. 可选增加 `FLAGS_check_cuda_error=1`，标出首次 device error 所在 instruction；每个 illegal access 后必须重启进程。

## 7. `cuModuleUnload` 的定性

Paddle 3.3.0 只在 [`CUDAModule` 析构函数](https://github.com/PaddlePaddle/Paddle/blob/v3.3.0/paddle/cinn/runtime/cuda/cuda_module.cc#L171-L177) 中调用 `cuModuleUnload()`。现场 C++ 栈又是 `CompilationCache::Clear → Compiler::~Compiler → CUDAModule::~CUDAModule`，因此它发生在异常后的编译缓存清理阶段。

结论：`cuModuleUnload ... illegal memory access` 是 CUDA context 已进入错误状态后的 **次生清理错误**，不能作为首个非法访问发生在 module unload 的证据。它与 #79580 的末尾症状相同，但不用于判断两个问题是否同根。

## 8. 最终判断

- **已确认根因**：Paddle 3.3.0 dy2static + CINN 对顶层 Tensor `forward(**kwargs)` 的缓存图输入绑定/复用错误，属于 #79580 同款。
- **已确认症状迁移**：原生 Embedding 最早暴露非法 ID；`index_select` 掩盖后，错误延迟到 `conv2d_transpose`/cuDNN；`cuModuleUnload` 是异常清理次生错误。
- **已排除为独立根因**：Conv2DTranspose 参数/layout、cuDNN heuristic、cuDNN 9.13/9.19 minor 兼容性、Parameter `view_shape_grad`（#79559）、AST 同名绑定（#79579）、transpose/local-buffer 历史修复 #69608。
- **最优雅修复**：用显式位置参数跨越顶层 to_static/CINN 边界，并恢复原生 Embedding；无需改 Conv2DTranspose、无需 FLAGS 降级、无需关闭 CINN。
