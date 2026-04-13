from __future__ import annotations

import paddle

from prose_fd_paddle.utils import misc


def test_to_cuda_uses_runtime_device_string_not_cuda_hardcode():
    misc.set_runtime_device("cpu")
    tensor = paddle.arange(4, dtype=paddle.float32)
    moved = misc.to_cuda(tensor)
    assert str(moved.place) == "Place(cpu)"


def test_to_cuda_preserves_none_and_multiple_args():
    misc.set_runtime_device("cpu")
    x = paddle.ones([2], dtype=paddle.float32)
    y = paddle.zeros([2], dtype=paddle.float32)
    moved_x, moved_none, moved_y = misc.to_cuda(x, None, y)
    assert str(moved_x.place) == "Place(cpu)"
    assert moved_none is None
    assert str(moved_y.place) == "Place(cpu)"


def test_max_memory_allocated_mb_is_none_on_cpu():
    misc.set_runtime_device("cpu")
    assert misc.max_memory_allocated_mb() is None
