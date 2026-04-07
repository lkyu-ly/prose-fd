# prose_fd模型转换到paddle

现在已经使用paconvert机器语法转换工具将prose-fd/prose_fd中的torch版代码转换成paddle代码。

转换结果见prose-fd/convert.log，其中，由于转换器特殊情况导致prose-fd/prose_fd_paddle/models/transformer.py无法被转换，完全保留torch的版本。

现在需要你参照转换结果和torch原版代码，在prose-fd/prose_fd_paddle的paddle版代码中做如下事情

1. 手动转换paconvert没能成功转换的所有代码和api，这类api在paddle版本的转换完代码中使用“>>>>>>”标记，并且在convert.log中也有记录。
2. dadaptation 是个依赖torch的第三方库，prose-fd/prose_fd/trainer.py导入了这个库，paconvert无法检测这类对torch的引用，我已经从这个库唯一要导入的函数DAdaptAdan的实现抽取出来，放到了paddle版本的prose-fd/prose_fd_paddle/utils/dadapt_adan_paddle.py和prose-fd/prose_fd_paddle/utils/custom_optimizer_base.py，你需要在使用这个库的地方改成本地文件导入。
3. 完成上述操作之后paddle版本文件夹下应该是一个纯paddle无torch依赖的库，并且不应该有任何python语法错误。

注意和限制：

1. 修改代码仅限于明确标为转换失败的部分和因为机器转换导致有明显逻辑错误的代码，对于正常转换成paddle代码的部分，如果其不是最佳实践也没有关系，只要没有明显逻辑错误和导致不可跑通的错误都可以容许。
2. 如果有任何需要修改的地方涉及到比较大规模的修改（不是简单几行修复或修补就能完成的），比如需要整个重新实现或者暂时不能实现需要另寻其他方法的部分，必须在探索完代码后与我商讨解决方案。
3. 对于任何允许进行大规模重写的代码或者函数，必须以重写完的版本和torch原有实现逻辑一致为优先。如果代码较为复杂不能直接判断逻辑，必须在修改完采用单测调用覆盖的方式进行评判，确认和原有实现无异。单测放在各自版本代码目录的tests文件夹下。
4. 不得自动运行训练或推理主逻辑，可以用其他任何方式检查代码，比如单测，静态检查，手动查看逻辑等。
5. 记住本次工作核心就两件事：完全转换为不报错的paddle版本代码，和保证与原torch所有实现在代码逻辑上对齐。

现在开始工作。
