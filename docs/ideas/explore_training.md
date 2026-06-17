# paddle版本模型训练探索

在先前的代码中已经修复所有影响模型加载和运行的问题，完成了模型前向精度和torch原版代码对齐的功能。现在需要进行模型训练验证，目标是能在单机单卡上跑通readme中的训练案例。

本次目标是探索代码和修改下面所述的问题，撰写和保存一份详细的执行方案，我将手动交由claude code进行执行。

## 数据部分

流体力学数据过大，不能使用readme中的全部数据集。现使用pdebench构建了一个缩小版的SWE数据集，在prose-fd/dataset/pdebench/2D/shallow-water/2D_rdb_NA_NA.h5。构建这个数据集使用的是/home/lkyu/baidu/PDEBench项目，使用脚本/home/lkyu/baidu/PDEBench/pdebench/data_gen/configs/radial_dam_break.yaml和/home/lkyu/baidu/PDEBench/pdebench/data_gen/gen_radial_dam_break.py，维度64x64，步数200.

这个数据集比原版数据集减少了维度和步数。根据prose-fd/prose_fd/data_utils/README.md，维度不同的数据集需要转换。

因此，你的第一个任务是探索我只用上述这个缩小数据集能不能完成模型的基础训练验证，包括是否符合项目对数据的要求以及数据集是否符合代码中的一些设定。

如果不能的话能够启动支持最小训练的方案是什么（允许修改paddle侧代码实现）

## 代码部分

探索代码训练链路，有没有明显阻塞训练的点位，可以撰写单测验证，不要直接启动训练，如果有的话应当修复。

我观察到很多数据移动到显卡用的是.cuda()方法。我们后期的目标之一是需要在paddle custom device上跑通（非本机环境），所以不能用仅在N卡上生效的 cuda，需要使用更通用的搬运到device方法，请结合paddle的device逻辑探明修改方案。

## 训练命令

结合readme给出的示例，在完成上述所有代码修复工作之后以本机16G nvidia显卡为目标，探索训练参数是否需要修改，最终给出启动训练的命令，完成任务并交由我手动运行训练命令验收。探索和修改过程中不能直接开始训练，但是可以调用GPU做安全的小测试（针对codex：如果环境中没有GPU可以申请提权）。