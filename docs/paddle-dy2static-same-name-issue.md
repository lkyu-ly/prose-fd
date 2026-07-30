# [Dy2St] `full_graph=True` 下方法名与全局函数同名时调用被错误绑定

## bug描述 Describe the Bug

当 `paddle.nn.Layer` 的方法与其调用的模块级函数同名时，`paddle.jit.to_static(..., full_graph=True)` 会因参数签名不匹配而报错。从转换结果看，该调用可能被错误绑定到了当前方法。

动态图和默认 `to_static` 均正常。预期 `full_graph=True` 也输出 `[2.0]`。

## 最小复现

```python
import paddle


def transform(*, tensor):
    return tensor + 1


class Repro(paddle.nn.Layer):
    def transform(self, x):
        return transform(tensor=x)

    def forward(self, x):
        return self.transform(x)


paddle.set_device("cpu")
x = paddle.to_tensor([1.0])

print("dynamic:", Repro()(x).numpy().tolist())
print(
    "full_graph:",
    paddle.jit.to_static(Repro(), full_graph=True)(x).numpy().tolist(),
)
```

运行环境：

```text
Python 3.10.19
paddlepaddle-gpu 3.3.0（使用 CPU 复现，不依赖 CUDA/CINN）
Linux x86_64
```

报错输出：

```text
dynamic: [2.0]
Traceback (most recent call last):
  File "/tmp/paddle_dy2static_same_name_repro.py", line 22, in <module>
    paddle.jit.to_static(Repro(), full_graph=True)(x).numpy().tolist(),
  File "/home/lkyu/miniconda3/envs/paddletorch/lib/python3.10/site-packages/paddle/nn/layer/layers.py", line 1833, in __call__
    return self.forward(*inputs, **kwargs)
  File "/home/lkyu/miniconda3/envs/paddletorch/lib/python3.10/site-packages/paddle/jit/dy2static/program_translator.py", line 548, in __call__
    return self._perform_call(*args, **kwargs)
  File "/home/lkyu/miniconda3/envs/paddletorch/lib/python3.10/site-packages/paddle/jit/dy2static/program_translator.py", line 859, in _perform_call
    error_data.raise_new_exception()
  File "/home/lkyu/miniconda3/envs/paddletorch/lib/python3.10/site-packages/paddle/jit/dy2static/error.py", line 454, in raise_new_exception
    raise new_exception from None
TypeError: In transformed code:

    File "/tmp/paddle_dy2static_same_name_repro.py", line 13, in forward
	return self.transform(x)
    File "/tmp/paddle_dy2static_same_name_repro.py", line 10, in transform
        class Repro(paddle.nn.Layer):
            def transform(self, x):
                return transform(tensor=x)
                ~~~~~~~~~~~~~~~~~~~~~~~~~~ <--- HERE

            def forward(self, x):


    TypeError: transform() got an unexpected keyword argument 'tensor'
```

## 问题猜测

可能是 AST dy2static 恢复闭包时把同名全局函数误判成了递归调用：

- 仅把全局函数重命名为 `transform_impl`，其他代码不变，`full_graph=True` 即可正常输出 `[2.0]`。
- `convert_to_static(Repro.transform)` 生成的函数包含 `co_freevars == ("transform",)`；对应 cell 是生成的 `create_transform.<locals>.transform`，不是模块级 `transform`。
- Paddle 3.3.0 的 [`get_new_closure`](https://github.com/PaddlePaddle/Paddle/blob/v3.3.0/python/paddle/jit/dy2static/utils.py#L602-L616) 在原函数的 `nonlocals` 中找不到该名字时，会保留生成函数的 free variable。这里原始引用属于 `globals`，可能因此保留了生成方法自身。

以上只是根据转换结果作出的判断，具体修复位置还请维护者确认。

## 相似 issue

- #79325：同样涉及 `to_static` 下的方法调用，但该问题是默认 SOT 中两个分别编译的方法通过 `self.xxx()` 调用时丢失 `self`，报 `KeyError: 'self'`。本问题只在 AST `full_graph=True` 下触发，条件是方法与模块级函数同名，报 `TypeError`。
