from __future__ import annotations

import math

import pytest

from prose_fd_paddle.utils.lr_scheduler import (
    cosine_schedule_lambda,
    inverse_sqrt_schedule_lambda,
    warmup_stable_decay_lambda,
)


def test_cosine_schedule_matches_expected_warmup_and_decay():
    values = [
        cosine_schedule_lambda(
            step,
            num_warmup_steps=2,
            num_training_steps=6,
            num_cycles=0.5,
            min_lr_rate=0.0,
        )
        for step in range(7)
    ]
    assert values[0] == pytest.approx(0.0)
    assert values[1] == pytest.approx(0.5)
    assert values[2] == pytest.approx(1.0)
    assert values[4] == pytest.approx(0.5)
    assert values[6] == pytest.approx(0.0)


def test_inverse_sqrt_schedule_matches_reference_formula():
    value = inverse_sqrt_schedule_lambda(
        20, num_warmup_steps=10, timescale=10
    )
    assert value == pytest.approx(1.0 / math.sqrt(2.0))


def test_warmup_stable_decay_has_three_stages():
    values = [
        warmup_stable_decay_lambda(
            step,
            num_warmup_steps=2,
            num_stable_steps=2,
            num_decay_steps=4,
            min_lr_ratio=0.1,
            num_cycles=0.5,
        )
        for step in range(10)
    ]
    assert values[0] == pytest.approx(0.0)
    assert values[1] == pytest.approx(0.5)
    assert values[2] == pytest.approx(1.0)
    assert values[3] == pytest.approx(1.0)
    assert values[8] == pytest.approx(0.1)
    assert values[9] == pytest.approx(0.1)
