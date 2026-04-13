from __future__ import annotations

import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = ROOT / "prose_fd"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from data_utils.collate import custom_collate
from dataset import get_dataset
from evaluate import Evaluator
from models.build_model import build_model
from symbol_utils.environment import SymbolicEnvironment
from trainer import Trainer


LOCAL_SW64 = ROOT / "dataset" / "pdebench" / "2D" / "shallow-water" / "2D_rdb_NA_NA.h5"


def build_sw64_cfg():
    main_cfg = OmegaConf.load(ROOT / "prose_fd" / "configs" / "main.yaml")
    data_cfg = OmegaConf.load(ROOT / "prose_fd" / "configs" / "data" / "shallow_water_minimal.yaml")
    model_cfg = OmegaConf.load(ROOT / "prose_fd" / "configs" / "model" / "prose_2to1.yaml")
    symbol_cfg = OmegaConf.load(ROOT / "prose_fd" / "configs" / "symbol" / "symbol.yaml")
    optim_cfg = OmegaConf.load(ROOT / "prose_fd" / "configs" / "optim" / "adamw.yaml")

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
    cfg.n_nodes = 1
    cfg.node_id = 0
    cfg.world_size = 1
    cfg.multi_gpu = 0
    cfg.multi_node = 0
    cfg.is_master = 1
    cfg.dump_path = "/tmp/prose_fd_test"
    cfg.eval_dump_path = "/tmp/prose_fd_test/evals_all"
    cfg.runtime_device = "cpu"
    cfg.base_seed = 42
    cfg.test_seed = 42
    cfg.eval_only = 0
    cfg.rollout = 0
    cfg.use_raw_time = 0
    cfg.noise = 0
    cfg.noise_type = "additive"
    cfg.flip = 0
    cfg.rotate = 0

    cfg.data.types = ["shallow_water"]
    cfg.data.x_num = 64
    cfg.data.shallow_water.x_num = 64
    cfg.data.shallow_water.data_path = str(LOCAL_SW64)

    cfg.model.name = "prose_2to1"

    cfg.accumulate_gradients = cfg.get("accumulate_gradients", 1)
    cfg.optim.max_iters = cfg.max_epoch * cfg.n_steps_per_epoch // cfg.accumulate_gradients
    if cfg.optim.warmup is not None and cfg.optim.warmup < 1:
        cfg.optim.warmup = max(1, int(cfg.optim.warmup * cfg.optim.max_iters))
    OmegaConf.resolve(cfg)
    return cfg


def _build_runtime_components(cfg):
    symbol_env = SymbolicEnvironment(cfg.symbol)
    modules = build_model(cfg, cfg.model, cfg.data, symbol_env)
    return symbol_env, modules


def test_shallow_water_minimal_config_matches_local_dataset():
    data_cfg = OmegaConf.load(ROOT / "prose_fd" / "configs" / "data" / "shallow_water_minimal.yaml")
    assert data_cfg.types == ["shallow_water"]
    assert data_cfg.x_num == 64
    assert data_cfg.shallow_water.x_num == 64
    assert Path(data_cfg.shallow_water.data_path).is_file()


def test_train_and_val_dataloaders_yield_single_batches():
    cfg = build_sw64_cfg()
    symbol_env, _ = _build_runtime_components(cfg)

    train_dataset = get_dataset(cfg, symbol_env, split="train")
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
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
    train_batch = next(iter(train_loader))
    assert tuple(train_batch["data"].shape) == (2, 20, 64, 64, 4)
    assert tuple(train_batch["data_mask"].shape) == (2, 10, 1, 1, 4)
    assert train_batch["symbol_input"].shape[0] == 2

    val_dataset = get_dataset(cfg, symbol_env, split="val")["shallow_water"]
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.batch_size_eval,
        num_workers=0,
        collate_fn=custom_collate(
            cfg.data.max_output_dimension,
            symbol_env.pad_index,
            cfg.data.tie_fields,
            pad_right=cfg.symbol.pad_right,
        ),
    )
    val_batch = next(iter(val_loader))
    assert tuple(val_batch["data"].shape) == (2, 20, 64, 64, 4)


def test_prose_2to1_can_run_forward_on_local_sw64_batch():
    cfg = build_sw64_cfg()
    symbol_env, modules = _build_runtime_components(cfg)
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

    times = torch.linspace(0, 10, cfg.data.t_num, dtype=torch.float32)[None]
    model_input = {
        "data_input": batch["data"][:, : cfg.input_len],
        "input_times": times[:, : cfg.input_len, None],
        "output_times": times[:, cfg.input_len :, None] - times[:, cfg.input_len - 1 : cfg.input_len, None],
        "symbol_input": batch["symbol_input"],
        "symbol_padding_mask": batch["symbol_mask"],
    }

    with torch.no_grad():
        output = model("fwd", **model_input)

    assert tuple(output.shape) == (1, 10, 64, 64, 4)


def test_trainer_and_evaluator_can_build_dataloaders():
    cfg = build_sw64_cfg()
    symbol_env, modules = _build_runtime_components(cfg)

    trainer = Trainer(modules, cfg, symbol_env)
    train_batch = trainer.get_batch()
    assert tuple(train_batch["data"].shape) == (2, 20, 64, 64, 4)

    evaluator = Evaluator(trainer, symbol_env)
    val_batch = next(iter(evaluator.dataloaders["shallow_water"]))
    assert tuple(val_batch["data"].shape) == (2, 20, 64, 64, 4)
