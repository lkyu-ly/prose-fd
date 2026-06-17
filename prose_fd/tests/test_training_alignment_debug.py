from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PYTHON = "/home/lkyu/miniconda3/envs/py312_torch291_cu128/bin/python"


def test_torch_batch_dump_is_deterministic(tmp_path):
    out = tmp_path / "torch_batches.jsonl"
    subprocess.run(
        [
            PYTHON,
            str(ROOT / "tools" / "debug_loss_alignment" / "dump_training_batches.py"),
            "--backend",
            "torch",
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


def test_torch_initial_state_export_creates_checkpoint(tmp_path):
    out = tmp_path / "torch_initial_sw64.pth"
    subprocess.run(
        [
            PYTHON,
            str(ROOT / "tools" / "debug_loss_alignment" / "export_torch_initial_state.py"),
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
    assert out.is_file()
