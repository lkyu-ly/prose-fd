# Repository Guidelines

## Project Structure & Module Organization

仓库根目录包含总览文档 `README.md` 和环境定义 `env.yml`。当前主要可维护代码位于 `prose_fd/`：`main.py` 是训练入口，`trainer.py` 与 `evaluate.py` 分别负责训练和评估流程，`configs/` 存放 Hydra 配置，`models/` 放模型实现，`data_utils/` 放数据转换与数据集组装，`symbol_utils/` 处理符号表示，`utils/` 提供日志、指标和杂项工具。复现实验命令集中在 `prose_fd/scripts/`。

## Build, Test, and Development Commands

先创建环境：

```bash
conda env create --name prose --file env.yml
conda activate prose
```

常用开发命令建议在 `prose_fd/` 目录执行：

```bash
cd prose_fd
python main.py dryrun=1 use_wandb=0
torchrun --standalone --nnodes 1 --nproc_per_node 2 main.py data=fluids_sample
bash scripts/prose-fd.sh
python data_utils/convert_cfdbench.py
```

`dryrun=1` 用于快速检查训练链路；`torchrun` 用于多卡训练；脚本文件用于复现实验；数据转换脚本用于预处理外部数据集。

## Coding Style & Naming Conventions

项目使用 Python，沿用现有风格：4 空格缩进，函数和变量使用 `snake_case`，类使用 `PascalCase`，配置键保持 Hydra 风格的短横线外避免混用大小写。优先复用现有工具函数，避免在 `main.py` 堆积逻辑。仓库未配置统一格式化工具；提交前至少保证导入有序、注释简洁、无未使用变量。

## Testing Guidelines

当前仓库没有独立的 `tests/` 或 `pytest` 套件，贡献时请提供最小可复现验证。模型改动至少跑一次 `python main.py dryrun=1 use_wandb=0`；评估相关改动再补一条 `eval_only=1` 命令；数据处理改动请记录输入路径、输出路径和样本数量。若新增自动化测试，建议放在根目录 `tests/` 下，并使用 `test_<module>.py` 命名。

## Commit & Pull Request Guidelines

现有提交信息以简短祈使句为主，例如 `Update README to redirect to prose_v1`。继续使用单一主题提交，标题聚焦模块和动作，例如 `Refine fluids evaluator logging`。PR 需说明修改范围、涉及的 Hydra 覆盖项、数据依赖、验证命令，以及对指标或产物的影响；若改动训练/评估行为，请附关键日志、表格或截图。

## Configuration & Data Notes

不要提交数据集、检查点或密钥。数据路径应写入 `configs/data/*` 或通过命令行覆盖传入；`wandb` 相关设置保持可关闭，默认优先保证 `use_wandb=0` 时也能正常运行。
