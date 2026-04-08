from __future__ import annotations

from types import SimpleNamespace

import paddle

from prose_fd_paddle.models.attention_utils import (
    CustomTransformerEncoder,
    CustomTransformerEncoderLayer,
    _generate_square_subsequent_mask,
)


class Config(SimpleNamespace):
    def get(self, key, default=None):
        return getattr(self, key, default)


def test_generate_square_subsequent_mask_matches_causal_layout():
    mask = _generate_square_subsequent_mask(4, dtype=paddle.float32)

    assert list(mask.shape) == [4, 4]
    assert mask[0, 0].item() == 0.0
    assert mask[0, 1].item() == float("-inf")
    assert mask[1, 0].item() == 0.0
    assert mask[3, 3].item() == 0.0


def test_custom_transformer_encoder_with_rotary_keeps_shape():
    paddle.seed(2026)
    config = Config(dim_emb=16, n_head=2, rotary=True)
    layer = CustomTransformerEncoderLayer(
        d_model=16,
        nhead=2,
        dim_feedforward=32,
        dropout=0.0,
        activation="gelu",
        batch_first=True,
        norm_first=True,
        rotary=True,
        custom_attn=True,
    )
    encoder = CustomTransformerEncoder(
        encoder_layer=layer,
        num_layers=1,
        norm=None,
        config=config,
    )
    x = paddle.randn([2, 4, 16], dtype="float32")
    padding_mask = paddle.to_tensor(
        [[False, False, False, True], [False, False, True, True]],
        dtype="bool",
    )

    output = encoder(x, src_key_padding_mask=padding_mask)

    assert list(output.shape) == [2, 4, 16]
    assert paddle.isfinite(output).all().item()
