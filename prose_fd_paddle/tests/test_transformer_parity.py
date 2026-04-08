from __future__ import annotations

from types import SimpleNamespace

import paddle

from prose_fd_paddle.models.transformer import (
    TransformerDataDecoder,
    TransformerDataEncoder,
    TransformerFusion,
)


class Config(SimpleNamespace):
    def get(self, key, default=None):
        return getattr(self, key, default)


def make_transformer_config(**overrides):
    config = Config(
        dim_emb=8,
        n_layer=1,
        n_head=2,
        dim_ffn=16,
        dropout=0.0,
        norm_first=True,
        custom_encoder=True,
        rotary=False,
        custom_attn=True,
        positional_embedding="sinusoidal",
        kv_cache=False,
        type_embeddings=True,
        norm="layer",
        query_dim=1,
        patch_num_output=1,
        final_ln=False,
        self_attn=0,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def test_transformer_data_encoder_forward_shape():
    paddle.seed(2026)
    model = TransformerDataEncoder(make_transformer_config())
    x = paddle.randn([2, 5, 8], dtype="float32")
    padding_mask = paddle.to_tensor(
        [[False, False, False, False, True], [False, False, False, True, True]],
        dtype="bool",
    )

    output = model(x, src_key_padding_mask=padding_mask)

    assert list(output.shape) == [2, 5, 8]
    assert paddle.isfinite(output).all().item()


def test_transformer_data_decoder_fwd_shape():
    paddle.seed(2026)
    model = TransformerDataDecoder(make_transformer_config(), output_dim=3)
    tgt = paddle.randn([2, 4, 8], dtype="float32")
    memory = paddle.randn([2, 5, 8], dtype="float32")
    memory_padding_mask = paddle.to_tensor(
        [[False, False, False, False, True], [False, False, False, True, True]],
        dtype="bool",
    )

    output = model.fwd(tgt=tgt, memory=memory, memory_key_padding_mask=memory_padding_mask)

    assert list(output.shape) == [2, 4, 3]
    assert paddle.isfinite(output).all().item()


def test_transformer_fusion_forward_concatenates_mask():
    paddle.seed(2026)
    model = TransformerFusion(make_transformer_config())
    x0 = paddle.randn([2, 3, 8], dtype="float32")
    x1 = paddle.randn([2, 2, 8], dtype="float32")
    mask0 = paddle.to_tensor([[False, False, True], [False, True, True]], dtype="bool")
    mask1 = paddle.to_tensor([[False, True], [False, False]], dtype="bool")

    fused, fused_mask = model(x0, x1, key_padding_mask0=mask0, key_padding_mask1=mask1)

    assert list(fused.shape) == [2, 5, 8]
    assert fused_mask.tolist() == [
        [False, False, True, False, True],
        [False, True, True, False, False],
    ]
