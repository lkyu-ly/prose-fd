from __future__ import annotations

from pathlib import Path

import paddle

from prose_fd_paddle.models.build_model import build_model
from prose_fd_paddle.symbol_utils.environment import SymbolicEnvironment
from prose_fd_paddle.trainer import Trainer
from prose_fd_paddle.tests.test_training_data_smoke import build_sw64_cfg


class FakeDisabledScaler:
    def __init__(self, enable: bool = True):
        self.enable = enable

    def state_dict(self):
        return {}

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        return None

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        return None


class FakeEnabledScaler:
    def __init__(self, enable: bool = True):
        self.enable = enable
        self.loaded = None

    def state_dict(self):
        return {"scale": [65536.0]}

    def load_state_dict(self, state):
        if not state:
            raise RuntimeError("empty scaler state must not be loaded")
        self.loaded = state

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        return None

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        return None


def build_cfg(tmp_path: Path):
    cfg = build_sw64_cfg()
    cfg.amp = 1
    cfg.dump_path = str(tmp_path / "exp")
    cfg.eval_dump_path = str(tmp_path / "exp" / "evals_all")
    return cfg


def build_modules(cfg):
    symbol_env = SymbolicEnvironment(cfg.symbol)
    modules = build_model(cfg, cfg.model, cfg.data, symbol_env)
    return modules, symbol_env


def test_save_checkpoint_omits_empty_scaler_state(monkeypatch, tmp_path):
    monkeypatch.setattr(paddle.amp, "GradScaler", FakeDisabledScaler)
    cfg = build_cfg(tmp_path)
    modules, symbol_env = build_modules(cfg)
    trainer = Trainer(modules, cfg, symbol_env)

    trainer.save_checkpoint("checkpoint")

    checkpoint = paddle.load(str(Path(cfg.dump_path) / "checkpoint.pth"))
    assert "scaler" not in checkpoint


def test_reload_checkpoint_skips_legacy_empty_scaler_state(monkeypatch, tmp_path):
    monkeypatch.setattr(paddle.amp, "GradScaler", FakeEnabledScaler)
    cfg = build_cfg(tmp_path)
    modules, symbol_env = build_modules(cfg)
    trainer = Trainer(modules, cfg, symbol_env)

    checkpoint_path = Path(cfg.dump_path) / "checkpoint.pth"
    checkpoint = {
        "epoch": 0,
        "n_total_iter": 5,
        "dataloader_count": 0,
        "best_metrics": trainer.best_metrics,
        "model": modules["model"].state_dict(),
        "optimizer": trainer.optimizer.state_dict(),
        "scaler": {},
    }
    paddle.save(checkpoint, str(checkpoint_path))

    modules_2, symbol_env_2 = build_modules(cfg)
    trainer_2 = Trainer(modules_2, cfg, symbol_env_2)

    assert trainer_2.epoch == 1
    assert trainer_2.n_total_iter == 5
    assert trainer_2.scaler.loaded is None
