# Paddle dy2static 同名全局函数误绑定调查

> 调查时间：2026-07-29。范围仅限 PaddlePaddle 与 einops 官方 GitHub 的 issue、PR、commit 和源码；未发布 issue。

## 1. 结论

1. [Paddle #79325](https://github.com/PaddlePaddle/Paddle/issues/79325) **不是同一 bug**。两者都属于 `to_static` 下 Python 名字/方法绑定语义被破坏，但触发条件、转换器、异常和修复位置均不同。
2. 在 PaddlePaddle/Paddle 官方 issue、PR 中，未检索到与本问题完全相同的报告。可以新建 issue，且应注明“与 #79325 相关但非重复”。
3. 当前问题可由不依赖 einops、CINN 或 GPU 的纯 Paddle 最小用例稳定复现：动态图和默认 `to_static`（SOT）通过，`full_graph=True`（AST）失败；只要让模块全局函数与实例方法不再同名，AST 也通过。
4. 根因位于经典 dy2static 的 AST 函数重建。更具体地说，[PR #67021](https://github.com/PaddlePaddle/Paddle/pull/67021) 引入 `create_xxx` wrapper 以恢复 closure，[PR #67482](https://github.com/PaddlePaddle/Paddle/pull/67482) 为支持全局递归而保留“生成函数中新出现的同名 freevar”；该逻辑无法区分真正递归和“方法调用同名模块全局 helper”，导致后者被误绑定为转换后方法自身。
5. Paddle [v3.3.0](https://github.com/PaddlePaddle/Paddle/blob/v3.3.0/python/paddle/jit/dy2static/utils.py#L602-L616) 与当前 [develop](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/jit/dy2static/utils.py#L602-L616) 都仍含该逻辑。因此这不是仅存在于旧发行版、已在主干修复的问题。

## 2. 为什么 #79325 不是同一问题

| 对比项 | #79325 | 当前问题 |
|---|---|---|
| 标题 | `[SOT] 被 to_static 编译后的方法通过 self.xxx() 调用另一个编译方法时 KeyError: 'self'` | AST 转写后，同名模块全局函数被误绑定为实例方法自身 |
| 状态 | Open；2026-06-16 创建；当前指派给 `xuxinyi389` | 尚未报告 |
| 环境 | Paddle `3.5.0.dev20260614` | Paddle `3.3.0`，且 develop 源码仍存在风险逻辑 |
| 必要结构 | 同一对象的两个方法都分别包装为 `to_static`，外层通过 `self.forward()` 调另一个已编译方法 | 一个实例方法在函数体内调用与自己同名的模块全局函数；只需顶层 model 使用 `full_graph=True` |
| 转换路径 | SOT opcode translator / inline executor | 经典 dy2static AST：`convert_to_static -> ast_to_func -> wrap_as_closure/get_new_closure` |
| 异常 | `KeyError: 'self'` / `InnerError` | helper 被解析为方法自身，最终出现参数签名错，如 `unexpected keyword argument` |
| 已知修复位置 | SOT 的 `StaticFunction -> MethodVariable`、`LOAD_METHOD/CALL_METHOD` self 绑定 | dy2static `utils.py` 的 wrapper/freevar/global 绑定恢复 |

#79325 的正文明确说，只有 caller 和 callee 两个方法都单独编译才触发；其栈进入 `paddle/jit/sot/opcode_translator`。[关联 PR #79326](https://github.com/PaddlePaddle/Paddle/pull/79326) 也把根因定位为 bound `StaticFunction` 被错误表示成 `UserDefinedFunctionVariable`，使 `CALL_METHOD` 丢失 `self`。该 PR 当前仍为 Open，提交了两个 commit：[实现修复](https://github.com/PaddlePaddle/Paddle/commit/eda2540202e71e687ca37aee23e3b103afe64864)与[回归测试](https://github.com/PaddlePaddle/Paddle/commit/314b87e7634e4e0780a5b664a903efc62b292c01)，尚未合入 Paddle 主干。

因此，#79325 可作为同类上下文链接，但不能把当前问题作为它的重复项提交。

## 3. 纯 Paddle 最小复现

以下用例去掉了 einops、CINN、CUDA 和项目代码，只保留唯一必要变量：实例方法与模块全局函数同名。

```python
import paddle


def same_name(*, value):
    return value + 1


class Repro(paddle.nn.Layer):
    def same_name(self, x):
        return same_name(value=x)

    def forward(self, x):
        return self.same_name(x)


def run(label, layer):
    try:
        out = layer(paddle.to_tensor([1.0]))
        print(label, out.numpy())
    except Exception as exc:
        print(label, type(exc).__name__, exc)


run("dynamic", Repro())
run("default", paddle.jit.to_static(Repro()))
run("full_graph", paddle.jit.to_static(Repro(), full_graph=True))
```

Paddle 3.3.0、CPU 执行的实际结果：

```text
dynamic [2.]
default [2.]
full_graph TypeError: same_name() got an unexpected keyword argument 'value'
```

控制变量结果：将实例方法改名，例如 `call_same_name(self, x)`，而其函数体仍调用全局 `same_name(value=x)`，`full_graph=True` 即可通过。这证明关键条件是名称冲突，不是函数本身、Tensor 操作或第三方库。

本机环境：

```text
PaddlePaddle: 3.3.0
Paddle commit: cbf3469113cd76b7d5f4cba7b8d7d5f55d9e9911
Python: 3.10.19
OS: Linux / WSL2 x86_64
Device used by MRE: CPU
```

模式差异同样很重要：默认 `to_static` 使用 SOT 并通过，`full_graph=True` 走 AST 并失败，所以 issue 应标记为 Dy2St/AST，不应标记成 SOT 或 CINN bug。

## 4. einops 为什么触发它

einops 0.8.2 的 [RearrangeMixin._apply_recipe](https://github.com/arogozhnikov/einops/blob/v0.8.2/einops/layers/__init__.py#L41-L49) 是一个实例方法，同时其模块从 `einops.einops` 导入了也叫 `_apply_recipe` 的全局 helper。方法体调用的是后者：

```python
def _apply_recipe(self, x):
    backend = get_backend(x)
    return _apply_recipe(
        backend=backend,
        recipe=self._multirecipe[len(x.shape)],
        tensor=x,
        reduction_type="rearrange",
        axes_lengths=self._axes_lengths,
    )
```

真正 helper 的[官方签名](https://github.com/arogozhnikov/einops/blob/v0.8.2/einops/einops.py#L230-L232)接收 `backend, recipe, tensor, reduction_type, axes_lengths`，所以动态图调用完全合法。

本机直接检查 `paddle.jit.dy2static.program_translator.convert_to_static(RearrangeMixin._apply_recipe)` 得到：

```text
original helper signature: (backend, recipe, tensor, reduction_type, axes_lengths)
converted method signature: (self, x)
converted freevars: ('_apply_recipe',)
closure cell _apply_recipe signature: (self, x)
```

也就是说，转换后的 `_apply_recipe` 闭包 cell 确实指向 `(self, x)` 的实例方法自身，而不是模块 globals 中接收 `backend=` 的 helper。报错文本只是这个错误绑定的后果，并不是 einops 两个发行文件的 API 签名不匹配。

## 5. Paddle 源码级根因

### 5.1 `create_xxx` wrapper 改变了 Python 名字解析

[PR #67021](https://github.com/PaddlePaddle/Paddle/pull/67021) 为支持无参 `super()` 所需的 `__class__` 自由变量，引入了将转换函数包进 `create_xxx` wrapper、再向新函数注入 closure 的方案。该 PR 于 2024-08-08 合并，merge commit 为 [`77a6200a`](https://github.com/PaddlePaddle/Paddle/commit/77a6200a49c780df599e7129402c91179f187268)，官方 review [approved](https://github.com/PaddlePaddle/Paddle/pull/67021#pullrequestreview-2226777968)。

当前 `wrap_as_closure` 生成的结构近似：

```python
def create__apply_recipe():
    def _apply_recipe(self, x):
        return _apply_recipe(backend=...)
    return _apply_recipe

_apply_recipe = create__apply_recipe()
```

原方法内的 `_apply_recipe` 是 global lookup；包入 wrapper 后，Python 编译器会把它视为 wrapper 中 `_apply_recipe` 绑定的 freevar，也就是生成方法自身。语义已由“调用模块 helper”变为“递归调用自己”。

### 5.2 #67482 的递归兼容逻辑保留了错误 freevar

[PR #67482](https://github.com/PaddlePaddle/Paddle/pull/67482) 正是为修复 #67021 对全局递归函数的破坏而提交。其说明指出：原始全局递归函数从 globals 获取自身，生成函数在 `create_xxx` 内则从 closure 获取自身，因此生成函数可能拥有原函数没有的 closure。修复使用：

```python
original_closure_vars.get(freevar_name, freevar)
```

即原函数 nonlocals 中找不到同名变量时，保留生成函数的 freevar。该 PR 于 2024-08-16 合并，merge commit 为 [`06c0c78d`](https://github.com/PaddlePaddle/Paddle/commit/06c0c78d104df0a294ba524fc3e7801ac29e4321)，官方 review [approved](https://github.com/PaddlePaddle/Paddle/pull/67482#pullrequestreview-2241624212)。

这能支持真正的全局递归，却没有区分另一种情况：原方法从 globals 调用“碰巧与自己同名的另一个函数”。本问题正落在这个歧义上。Paddle [develop 的 `get_new_closure`](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/jit/dy2static/utils.py#L602-L616)、[`wrap_as_closure` 调用处](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/jit/dy2static/utils.py#L630-L631)及[`types.FunctionType` 重建处](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/jit/dy2static/utils.py#L674-L680)仍是这条路径。

官方现有 [`test_capture_free_vars.py`](https://github.com/PaddlePaddle/Paddle/blob/develop/test/dygraph_to_static/test_capture_free_vars.py#L25-L135)覆盖普通 closure、无参 `super()` 与全局递归，但未覆盖“成员方法和模块全局函数同名且前者调用后者”。

## 6. 重复项检索结果

检索范围为 PaddlePaddle/Paddle 官方 issues/PRs，包含开放和关闭条目；关键词包括错误原文、`_apply_recipe`、`einops to_static`、`dy2static closure`、`same name`、`shadowing`、`global function`、`free variable`、`recursive function` 等。

- `is:issue "_apply_recipe"`：0 条。
- `is:issue einops to_static`：0 条。
- `is:pr "_apply_recipe"`、`is:pr einops to_static`：均为 0 条。
- `repo:PaddlePaddle/Paddle einops` 的宽泛检索会命中历史条目，但逐项没有当前 AST/closure/name-binding 组合。
- [#36898](https://github.com/PaddlePaddle/Paddle/issues/36898) 是 2021 年用户自行实现 `Rearrange` 类时的普通动态图问题，维护者也指出报错看起来不来自 rearrange；没有 `to_static`、closure 或 `backend=` 签名错。
- [#45627](https://github.com/PaddlePaddle/Paddle/issues/45627) 是 einops 当时尚未支持 Paddle backend 的旧 feature request，随 einops 上游增加 Paddle backend 而关闭。
- [#43713](https://github.com/PaddlePaddle/Paddle/pull/43713) 是控制流 closure analysis；[#23900](https://github.com/PaddlePaddle/Paddle/pull/23900) 是早期递归调用支持；均不是同名 global/method 的 AST 绑定恢复问题。
- [#67021](https://github.com/PaddlePaddle/Paddle/pull/67021) 与 [#67482](https://github.com/PaddlePaddle/Paddle/pull/67482) 不是重复报告，但它们是解释根因和选择修复点最重要的历史记录。

基于上述检索，建议新发 issue。

## 7. 修复建议

对 Paddle 的修复应满足两个既有语义并增加第三个语义：

1. 保留原函数真实 nonlocal closure，继续支持无参 `super()`。
2. 保留真正的全局递归函数语义，不能回退 #67482。
3. 当生成函数因 wrapper 新产生一个与当前函数同名的 freevar，但原函数该名字实际来自 `inspect.getclosurevars(original_fn).globals` 且对象不是原函数自身时，应恢复原 global 对象，而不是保留生成函数自身。

最小回归测试应采用第 3 节的纯 Paddle 用例，并至少验证：动态图、SOT、AST 三种模式结果一致；真正的全局递归现有用例继续通过。具体补丁设计应由 Paddle 维护者结合 `get_new_closure`/`get_new_globals` 的职责边界决定，本报告不建议在业务仓库 monkey-patch `site-packages`。

## 8. 待发布 issue 草稿

以下内容可直接作为 Paddle issue 基础；发布前建议在 Paddle 最新 nightly/develop wheel 上再跑一次纯 Paddle MRE。如果 nightly 仍失败，环境矩阵同时列 3.3.0 和 nightly，证据最强。

### 标题

```text
[Dy2St] full_graph AST misbinds a same-named global helper as the transformed method itself
```

### 正文

````markdown
### bug描述 Describe the Bug

`paddle.jit.to_static(..., full_graph=True)` changes Python name-binding semantics when an instance method calls a module-level global function with the same name as the method.

The original method resolves the called name from `__globals__`. After dy2static AST conversion, `wrap_as_closure` places the method inside a `create_xxx` wrapper. The same name then becomes a closure free variable bound to the transformed method itself, so the call becomes an unintended recursive self-call.

Dynamic execution and default `to_static` (SOT) work. Only `full_graph=True` (AST) fails. Renaming either the instance method or the global helper also makes AST mode pass.

This was first observed through `einops.layers.paddle.Rearrange`, whose `RearrangeMixin._apply_recipe(self, x)` calls an imported global helper also named `_apply_recipe`. However, the following reproduction uses only Paddle.

### Minimal reproduction

```python
import paddle


def same_name(*, value):
    return value + 1


class Repro(paddle.nn.Layer):
    def same_name(self, x):
        return same_name(value=x)

    def forward(self, x):
        return self.same_name(x)


def run(label, layer):
    try:
        out = layer(paddle.to_tensor([1.0]))
        print(label, out.numpy())
    except Exception as exc:
        print(label, type(exc).__name__, exc)


run("dynamic", Repro())
run("default", paddle.jit.to_static(Repro()))
run("full_graph", paddle.jit.to_static(Repro(), full_graph=True))
```

Actual output:

```text
dynamic [2.]
default [2.]
full_graph TypeError: same_name() got an unexpected keyword argument 'value'
```

Expected output:

```text
dynamic [2.]
default [2.]
full_graph [2.]
```

### Root-cause evidence

For the original function, `same_name` is present in `inspect.getclosurevars(Repro.same_name).globals`, not in `nonlocals`.

After `paddle.jit.dy2static.program_translator.convert_to_static(Repro.same_name)`, the generated function has `('same_name',)` in `co_freevars`, and that closure cell contains the generated `(self, x)` method itself rather than the module-level `(*, value)` helper.

The relevant implementation appears to be:

- `wrap_as_closure`: https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/jit/dy2static/utils.py#L479-L567
- `get_new_closure`: https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/jit/dy2static/utils.py#L602-L616
- closure reconstruction: https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/jit/dy2static/utils.py#L674-L680

PR #67482 intentionally keeps a generated free variable when the original function has no matching nonlocal, in order to support a truly recursive global function. This seems to conflate true recursion with the distinct case where a method calls another global object that happens to have the same name:

- #67021: https://github.com/PaddlePaddle/Paddle/pull/67021
- #67482: https://github.com/PaddlePaddle/Paddle/pull/67482

### Relation to #79325

This is related to #79325 only in the broad sense that both bugs break Python binding semantics under `to_static`; it is not the same bug:

- #79325 is in SOT, requires two separately compiled methods and loses `self` during `self.xxx()` dispatch (`KeyError: 'self'`).
- This issue is in the classic dy2static AST path, needs only one top-level `full_graph=True` conversion, and misbinds a global helper as a closure self-reference.

### Environment

```text
PaddlePaddle: 3.3.0
Paddle commit: cbf3469113cd76b7d5f4cba7b8d7d5f55d9e9911
Python: 3.10.19
OS: Linux / WSL2 x86_64
Device used by reproduction: CPU
```

The same `get_new_closure` logic is still present on `develop` as of 2026-07-29. I have not yet tested a freshly built develop/nightly wheel.

### 其他补充信息 Additional Supplementary Information

Renaming the layer method to `call_same_name` while keeping the global helper as `same_name` makes all three modes pass. This isolates the bug to the same-name global/method collision.
````

## 9. 一手来源索引

- Paddle issue #79325：https://github.com/PaddlePaddle/Paddle/issues/79325
- #79325 关联 PR #79326：https://github.com/PaddlePaddle/Paddle/pull/79326
- closure wrapper PR #67021：https://github.com/PaddlePaddle/Paddle/pull/67021
- #67021 merge commit：https://github.com/PaddlePaddle/Paddle/commit/77a6200a49c780df599e7129402c91179f187268
- recursive-call follow-up PR #67482：https://github.com/PaddlePaddle/Paddle/pull/67482
- #67482 merge commit：https://github.com/PaddlePaddle/Paddle/commit/06c0c78d104df0a294ba524fc3e7801ac29e4321
- Paddle v3.3.0 dy2static utils：https://github.com/PaddlePaddle/Paddle/blob/v3.3.0/python/paddle/jit/dy2static/utils.py#L602-L616
- Paddle develop dy2static utils：https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/jit/dy2static/utils.py#L602-L616
- Paddle closure 测试：https://github.com/PaddlePaddle/Paddle/blob/develop/test/dygraph_to_static/test_capture_free_vars.py#L25-L135
- einops 0.8.2 layer method：https://github.com/arogozhnikov/einops/blob/v0.8.2/einops/layers/__init__.py#L41-L49
- einops 0.8.2 global helper：https://github.com/arogozhnikov/einops/blob/v0.8.2/einops/einops.py#L230-L232
