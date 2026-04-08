from __future__ import annotations

import numpy as np

from prose_fd_paddle.tools.convert_torch_ckpt_to_paddle import (
    clean_torch_key,
    maybe_convert_array,
)


def test_clean_torch_key_removes_parallel_prefixes():
    key = "module._orig_mod.data_encoder.transformer_encoder.layers.0.linear1.weight"
    assert clean_torch_key(key) == "data_encoder.transformer_encoder.layers.0.linear1.weight"


def test_maybe_convert_array_transposes_linear_weights():
    src = np.arange(6, dtype=np.float32).reshape(2, 3)
    target_shape = (3, 2)
    converted = maybe_convert_array("linear.weight", src, target_shape)
    assert converted.shape == target_shape
    assert np.array_equal(converted, src.T)


def test_maybe_convert_array_keeps_conv_weights():
    src = np.arange(24, dtype=np.float32).reshape(2, 3, 2, 2)
    target_shape = (2, 3, 2, 2)
    converted = maybe_convert_array("conv.weight", src, target_shape)
    assert converted.shape == target_shape
    assert np.array_equal(converted, src)
