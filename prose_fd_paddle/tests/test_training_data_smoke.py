from __future__ import annotations

from pathlib import Path

import paddle
from omegaconf import OmegaConf

from prose_fd_paddle.data_utils.collate import custom_collate
from prose_fd_paddle.dataset import get_dataset
from prose_fd_paddle.models.build_model import build_model
from prose_fd_paddle.symbol_utils.environment import SymbolicEnvironment


ROOT = Path(__file__).resolve().parents[2]
LOCAL_SW64 = ROOT / "dataset" / "pdebench" / "2D" / "shallow-water" / "2D_rdb_NA_NA.h5"


def build_sw64_cfg():
    main_cfg = OmegaConf.load(ROOT / "prose_fd_paddle" / "configs" / "main.yaml")
    data_cfg = OmegaConf.load(ROOT / "prose_fd_paddle" / "configs" / "data" / "fluids.yaml")
    model_cfg = OmegaConf.load(ROOT / "prose_fd_paddle" / "configs" / "model" / "prose_2to1.yaml")
    symbol_cfg = OmegaConf.load(ROOT / "prose_fd_paddle" / "configs" / "symbol" / "symbol.yaml")
    optim_cfg = OmegaConf.load(ROOT / "prose_fd_paddle" / "configs" / "optim" / "adamw.yaml")

    cfg = OmegaConf.create(OmegaConf.to_container(main_cfg, resolve=False))
    cfg.data = data_cfg
    cfg.model = model_cfg
    cfg.symbol = symbol_cfg
    cfg.optim = optim_cfg
    cfg.cpu = 1
    cfg.compile = 0
    cfg.amp = 0
    cfg.reload_model = None
    cfg.reload_checkpoint = None
    cfg.use_wandb = 0
    cfg.num_workers = 0
    cfg.num_workers_eval = 0
    cfg.batch_size = 2
    cfg.batch_size_eval = 2
    cfg.overfit_test = 0
    cfg.local_rank = 0
    cfg.global_rank = 0
    cfg.n_gpu_per_node = 1
    cfg.world_size = 1
    cfg.multi_gpu = 0
    cfg.multi_node = 0
    cfg.is_master = 1
    cfg.dump_path = "/tmp/prose_fd_test"
    cfg.eval_dump_path = "/tmp/prose_fd_test/evals_all"
    cfg.runtime_device = "cpu"
    cfg.data.types = ["shallow_water"]
    cfg.data.x_num = 64
    cfg.data.shallow_water.x_num = 64
    cfg.data.shallow_water.data_path = str(LOCAL_SW64)
    # Compute derived values normally set by main.py
    cfg.accumulation_gradients = cfg.get("accumulate_gradients", 1)
    cfg.optim.max_iters = cfg.max_epoch * cfg.n_steps_per_epoch // cfg.accumulate_gradients
    if cfg.optim.warmup is not None and cfg.optim.warmup < 1:
        cfg.optim.warmup = max(1, int(cfg.optim.warmup * cfg.optim.max_iters))
    OmegaConf.resolve(cfg)
    return cfg


def test_train_dataloader_can_yield_one_batch_from_local_sw64():
    cfg = build_sw64_cfg()
    symbol_env = SymbolicEnvironment(cfg.symbol)
    dataset = get_dataset(cfg, symbol_env, split="train")
    loader = paddle.io.DataLoader(
        dataset=dataset,
        batch_size=cfg.batch_size,
        num_workers=0,
        drop_last=True,
        collate_fn=custom_collate(
            cfg.data.max_output_dimension,
            symbol_env.pad_index,
            cfg.data.tie_fields,
            cfg.data.get("mixed_length", 0),
            cfg.input_len,
            cfg.symbol.pad_right,
        ),
    )
    batch = next(iter(loader))
    assert tuple(batch["data"].shape) == (2, 20, 64, 64, 4)
    assert tuple(batch["data_mask"].shape) == (2, 10, 1, 1, 4)
    assert batch["symbol_input"].shape[0] == 2


def test_eval_dataloader_can_yield_one_batch_from_local_sw64():
    cfg = build_sw64_cfg()
    symbol_env = SymbolicEnvironment(cfg.symbol)
    dataset = get_dataset(cfg, symbol_env, split="val")["shallow_water"]
    loader = paddle.io.DataLoader(
        dataset=dataset,
        batch_size=cfg.batch_size_eval,
        num_workers=0,
        collate_fn=custom_collate(
            cfg.data.max_output_dimension,
            symbol_env.pad_index,
            cfg.data.tie_fields,
            pad_right=cfg.symbol.pad_right,
        ),
    )
    batch = next(iter(loader))
    assert tuple(batch["data"].shape) == (2, 20, 64, 64, 4)


def test_prose_2to1_can_run_forward_on_local_sw64_batch():
    cfg = build_sw64_cfg()
    symbol_env = SymbolicEnvironment(cfg.symbol)
    modules = build_model(cfg, cfg.model, cfg.data, symbol_env)
    model = modules["model"]
    model.eval()

    sample = next(iter(get_dataset(cfg, symbol_env, split="train")))
    batch = custom_collate(
        cfg.data.max_output_dimension,
        symbol_env.pad_index,
        cfg.data.tie_fields,
        cfg.data.get("mixed_length", 0),
        cfg.input_len,
        cfg.symbol.pad_right,
    )([sample])

    input_len = cfg.input_len
    times = paddle.linspace(0, 10, cfg.data.t_num, dtype=paddle.float32)[None]
    model_input = {
        "data_input": batch["data"][:, :input_len],
        "input_times": times[:, :input_len, None],
        "output_times": times[:, input_len:, None] - times[:, input_len - 1 : input_len, None],
        "symbol_input": batch["symbol_input"],
        "symbol_padding_mask": batch["symbol_mask"],
    }
    with paddle.no_grad():
        output = model("fwd", **model_input)

    assert tuple(output.shape) == (1, 10, 64, 64, 4)


def test_default_model_reference_points_to_existing_config():
    main_cfg = OmegaConf.load(ROOT / "prose_fd_paddle" / "configs" / "main.yaml")
    assert main_cfg.defaults[1]["model"] == "prose_2to1"


def test_shallow_water_minimal_config_matches_local_dataset():
    data_cfg = OmegaConf.load(ROOT / "prose_fd_paddle" / "configs" / "data" / "shallow_water_minimal.yaml")
    assert data_cfg.types == ["shallow_water"]
    assert data_cfg.x_num == 64
    assert data_cfg.shallow_water.x_num == 64
    assert Path(data_cfg.shallow_water.data_path).is_file()


from prose_fd_paddle.trainer import Trainer
from prose_fd_paddle.evaluate import Evaluator


def test_trainer_can_build_train_dataloader_on_local_sw64():
    cfg = build_sw64_cfg()
    symbol_env = SymbolicEnvironment(cfg.symbol)
    modules = build_model(cfg, cfg.model, cfg.data, symbol_env)
    trainer = Trainer(modules, cfg, symbol_env)
    batch = trainer.get_batch()
    assert tuple(batch["data"].shape) == (2, 20, 64, 64, 4)


def test_evaluator_can_build_validation_dataloader_on_local_sw64():
    cfg = build_sw64_cfg()
    symbol_env = SymbolicEnvironment(cfg.symbol)
    modules = build_model(cfg, cfg.model, cfg.data, symbol_env)
    trainer = Trainer(modules, cfg, symbol_env)
    evaluator = Evaluator(trainer, symbol_env)
    batch = next(iter(evaluator.dataloaders["shallow_water"]))
    assert tuple(batch["data"].shape) == (2, 20, 64, 64, 4)
