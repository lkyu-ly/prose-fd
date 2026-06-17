from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PYTHON = "/home/lkyu/miniconda3/envs/paddletorch/bin/python"
TORCH_PYTHON = "/home/lkyu/miniconda3/envs/py312_torch291_cu128/bin/python"
ENV = {**os.environ, "PYTHONPATH": str(ROOT)}


def test_paddle_batch_dump_matches_known_torch_hash(tmp_path):
    out = tmp_path / "paddle_batches.jsonl"
    subprocess.run(
        [
            PYTHON,
            str(ROOT / "tools" / "debug_loss_alignment" / "dump_training_batches.py"),
            "--backend",
            "paddle",
            "--steps",
            "3",
            "--base-seed",
            "123",
            "--batch-size",
            "4",
            "--out",
            str(out),
        ],
        cwd=ROOT,
        check=True,
    )
    text = out.read_text(encoding="utf-8")
    assert "dea2d31259a77832" in text
    assert "b621926695ad17eb" in text


def test_convert_sw64_torch_initial_state_to_paddle(tmp_path):
    torch_ckpt = tmp_path / "torch_initial_sw64.pth"
    paddle_ckpt = tmp_path / "paddle_initial_sw64.pdparams"
    subprocess.run(
        [
            TORCH_PYTHON,
            str(ROOT / "tools" / "debug_loss_alignment" / "export_torch_initial_state.py"),
            "--base-seed",
            "123",
            "--batch-size",
            "4",
            "--out",
            str(torch_ckpt),
        ],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        [
            PYTHON,
            str(ROOT / "prose_fd_paddle" / "tools" / "convert_torch_ckpt_to_paddle.py"),
            "--torch-ckpt",
            str(torch_ckpt),
            "--paddle-ckpt",
            str(paddle_ckpt),
            "--data-config",
            str(ROOT / "prose_fd_paddle" / "configs" / "data" / "shallow_water_minimal.yaml"),
            "--model-config",
            str(ROOT / "prose_fd_paddle" / "configs" / "model" / "prose_2to1.yaml"),
            "--symbol-config",
            str(ROOT / "prose_fd_paddle" / "configs" / "symbol" / "symbol.yaml"),
            "--input-len",
            "10",
        ],
        cwd=ROOT,
        env=ENV,
        check=True,
    )
    assert paddle_ckpt.is_file()
