from __future__ import annotations

from types import SimpleNamespace

import paddle

from prose_fd_paddle.utils.mode import resolve_runtime_device


def _make_params(**overrides):
    params = SimpleNamespace(cpu=0, device=None, multi_gpu=False, local_rank=0)
    for key, value in overrides.items():
        setattr(params, key, value)
    return params


def test_requested_gpu_maps_to_custom_device_when_native_gpu_is_unavailable(monkeypatch):
    recorded = []

    monkeypatch.setattr(paddle.device, "get_all_device_type", lambda: ["cpu"])
    monkeypatch.setattr(paddle.device, "get_all_custom_device_type", lambda: ["iluvatar_gpu"])
    monkeypatch.setattr(paddle, "set_device", lambda device: recorded.append(device))
    monkeypatch.setattr(paddle.device, "get_device", lambda: recorded[-1])

    runtime_device = resolve_runtime_device(_make_params(device="gpu:0"))

    assert runtime_device == "iluvatar_gpu:0"
    assert recorded == ["iluvatar_gpu:0"]


def test_requested_gpu_keeps_native_gpu_when_available(monkeypatch):
    recorded = []

    monkeypatch.setattr(paddle.device, "get_all_device_type", lambda: ["cpu", "gpu"])
    monkeypatch.setattr(paddle.device, "get_all_custom_device_type", lambda: ["iluvatar_gpu"])
    monkeypatch.setattr(paddle, "set_device", lambda device: recorded.append(device))
    monkeypatch.setattr(paddle.device, "get_device", lambda: recorded[-1])

    runtime_device = resolve_runtime_device(_make_params(device="gpu:0"))

    assert runtime_device == "gpu:0"
    assert recorded == ["gpu:0"]


def test_cpu_flag_still_forces_cpu(monkeypatch):
    recorded = []

    monkeypatch.setattr(paddle.device, "get_all_device_type", lambda: ["gpu"])
    monkeypatch.setattr(paddle.device, "get_all_custom_device_type", lambda: ["iluvatar_gpu"])
    monkeypatch.setattr(paddle, "set_device", lambda device: recorded.append(device))
    monkeypatch.setattr(paddle.device, "get_device", lambda: recorded[-1])

    runtime_device = resolve_runtime_device(_make_params(cpu=1, device="gpu:0"))

    assert runtime_device == "cpu"
    assert recorded == ["cpu"]


def test_default_custom_device_fallback_is_unchanged(monkeypatch):
    recorded = []

    monkeypatch.setattr(paddle.device, "get_all_device_type", lambda: ["cpu"])
    monkeypatch.setattr(paddle.device, "get_all_custom_device_type", lambda: ["iluvatar_gpu"])
    monkeypatch.setattr(paddle, "set_device", lambda device: recorded.append(device))
    monkeypatch.setattr(paddle.device, "get_device", lambda: recorded[-1])

    runtime_device = resolve_runtime_device(_make_params())

    assert runtime_device == "iluvatar_gpu:0"
    assert recorded == ["iluvatar_gpu:0"]
