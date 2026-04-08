from __future__ import annotations

from prose_fd_paddle.utils.datapipe_compat import (
    IterDataPipe,
    Multiplexer,
    SampleMultiplexer,
)


class ToyPipe(IterDataPipe):
    def __init__(self, values):
        self.values = list(values)

    def __iter__(self):
        yield from self.values


def test_shuffle_preserves_values():
    pipe = ToyPipe([1, 2, 3, 4]).shuffle(buffer_size=4, seed=7)
    values = list(pipe)
    assert sorted(values) == [1, 2, 3, 4]
    assert values != [1, 2, 3, 4]


def test_cycle_repeats_stream():
    values = []
    for idx, item in enumerate(ToyPipe([1, 2]).cycle()):
        values.append(item)
        if idx == 4:
            break
    assert values == [1, 2, 1, 2, 1]


def test_multiplexer_round_robins_inputs():
    merged = list(Multiplexer(ToyPipe([1, 2]), ToyPipe([10, 20])))
    assert merged == [1, 10, 2, 20]


def test_sample_multiplexer_respects_weights_with_seed():
    merged = list(
        SampleMultiplexer(
            {ToyPipe([1, 2, 3]): 0.0, ToyPipe([10, 20, 30]): 1.0},
            seed=3,
        )
    )
    assert merged == [10, 20, 30]
