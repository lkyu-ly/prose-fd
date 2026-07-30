# [Dy2St][CINN] Tensor kwargs 在第二次执行时导致 Embedding 读取错误 ID

## bug描述 Describe the Bug

`paddle.jit.to_static(..., full_graph=True)` 下，模型通过 `forward(**kwargs)` 接收 float32 与 int64 Tensor 时，第一次执行正常，第二次执行中 Embedding 会读取到输入中不存在的 ID，随后触发 CUDA device assert。

下面的复现中 `token_ids` 始终为 `0..31`，词表大小为 128；第二次执行时 kernel 稳定读取到非法 ID `128`。预期三次执行均成功并返回有限值。

## 最小复现

```python
import os

os.environ["FLAGS_prim_enable_dynamic"] = "true"
os.environ["FLAGS_prim_all"] = "true"
os.environ["FLAGS_use_cinn"] = "true"

import paddle


class Model(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.embedding = paddle.nn.Embedding(128, 16, padding_idx=0)

    def forward(self, **kwargs):
        tokens = self.embedding(kwargs["token_ids"])
        return tokens + kwargs["data"] * 2


paddle.set_device("gpu:0")
model = paddle.jit.to_static(Model(), full_graph=True)
token_ids = (paddle.arange(32, dtype="int64") % 128).reshape([2, 16])
data = paddle.ones([2, 16, 16], dtype="float32")

for step in range(3):
    output = model(data=data, token_ids=token_ids)
    paddle.device.synchronize()
    print(f"step={step + 1}", output.shape, bool(paddle.isfinite(output).all()))
```

## 最小运行环境

```text
Python 3.10
paddlepaddle-gpu 3.3.0（CINN enabled）
NVIDIA GPU / CUDA 13.0
Linux x86_64
```

## 报错输出

<details>
<summary>完整原始输出</summary>

```text
all prim enabled:  True
W0729 21:26:07.604071 581781 gpu_resources.cc:116] Please NOTE: device: 0, GPU Compute Capability: 8.9, Driver API Version: 13.1, Runtime API Version: 13.3
W0729 21:26:07.761314 581781 eager_utils.cc:3584] Paddle static graph(PIR) not support input out tensor for now!!!!!
/home/lkyu/miniconda3/envs/paddletorch/lib/python3.10/site-packages/paddle/pir/math_op_patch.py:241: UserWarning: Tensor do not have 'place' interface for pir graph mode, try not to use it. None will be returned.
  warnings.warn(
I0729 21:26:07.851070 581781 add_cinn_pass.cc:334] Compiling subgraph with CINN backend ...
I0729 21:26:09.301119 581781 pir_interpreter.cc:1606] New Executor is Running ...
I0729 21:26:09.301272 581781 pir_interpreter.cc:1627] pir interpreter is running by trace mode ...
step=1 paddle.Size([2, 16, 16]) True
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Traceback (most recent call last):
  File "/tmp/paddle-cinn-kwargs-embedding-no-mode.py", line 27, in <module>
    paddle.device.synchronize()
  File "/home/lkyu/miniconda3/envs/paddletorch/lib/python3.10/site-packages/paddle/device/__init__.py", line 1827, in synchronize
    core._device_synchronize(place.get_device_id())
OSError: (External) CUDA error(719), unspecified launch failure. 
  [Hint: 'cudaErrorLaunchFailure'. An exception occurred on the device while executing a kernel. Common causes include dereferencing an invalid device pointerand accessing out of bounds shared memory. Less common cases can be system specific - more information about these cases canbe found in the system specific user guide. This leaves the process in an inconsistent state and any further CUDA work willreturn the same error. To continue using CUDA, the process must be terminated and relaunched.] (at /paddle/paddle/fluid/pybind/cuda_streams_py.cc:165)

terminate called after throwing an instance of 'common::enforce::EnforceNotMet'
  what():  (Fatal) CUDA Driver Error: cuModuleUnload(module) failed with error: unspecified launch failure (at /paddle/paddle/cinn/runtime/cuda/cuda_module.cc:176)



--------------------------------------
C++ Traceback (most recent call last):
--------------------------------------
0   cinn::hlir::framework::CompilationCache::Clear()
1   std::_Sp_counted_ptr_inplace<cinn::hlir::framework::pir::CompilationResult, std::allocator<cinn::hlir::framework::pir::CompilationResult>, (__gnu_cxx::_Lock_policy)2>::_M_dispose()
2   std::_Sp_counted_ptr_inplace<cinn::hlir::framework::pir::BackendResource, std::allocator<cinn::hlir::framework::pir::BackendResource>, (__gnu_cxx::_Lock_policy)2>::_M_dispose()
3   std::_Sp_counted_deleter<cinn::backends::Compiler*, std::default_delete<cinn::backends::Compiler>, std::allocator<void>, (__gnu_cxx::_Lock_policy)2>::_M_dispose()
4   cinn::backends::Compiler::~Compiler()

----------------------
Error Message Summary:
----------------------
FatalError: `Process abort signal` is detected by the operating system.
  [TimeInfo: *** Aborted at 1785331569 (unix time) try "date -d @1785331569" if you are using GNU date ***]
  [SignalInfo: *** SIGABRT (@0x3e80008e095) received by PID 581781 (TID 0x7074ea2f6780) from PID 581781 ***]

Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
Error: /paddle/paddle/phi/kernels/gpu/embedding_kernel.cu:43 Assertion `id < N` failed. Id should smaller than 128 but received an id value: 128.
```

</details>

## 问题猜测

可能是 to_static+CINN 缓存图在第二次执行时，对 `**kwargs` 中的 Tensor 输入发生了错误绑定或内部缓冲复用。Embedding 可能只是第一个通过范围检查暴露该问题的算子。

依据：

- 最小复现使用 3 个独立进程运行，均为第一次成功、第二次读取 ID `128` 并退出。
- 第一次执行后，从调用者侧检查 `token_ids`，其范围仍为 `0..31`，没有被原地修改。
- 保持计算和输入不变，将 `forward(**kwargs)` 改为位置参数后，连续执行三次均正常。
- 去掉原项目中的字符串 mode、模型结构、数据集、反向和优化器后仍可复现。

以上只是根据对照实验作出的判断，具体问题位置还请维护者确认。

## 相似 issue

- #43225：同样出现动转静后 Embedding 的 Ids 异常，但该问题是 Paddle 2.3 导出阶段的 `Ids is null`，最终定位为 `if/else + dict`；本问题发生在 Paddle 3.3 CINN 图第二次执行期。
- #79220：同样涉及 to_static 对 keyword arguments 的处理，但该问题是 `masked_fill_` wrapper 首次转换时访问空 `args[0]`；本问题是 Tensor kwargs 在缓存图第二次执行时触发 CUDA 错误。
- #79579：同样是 `full_graph=True` 下的调用绑定异常，但触发条件是方法与全局函数同名，表现为转换期 `TypeError`；本问题不涉及同名函数。

