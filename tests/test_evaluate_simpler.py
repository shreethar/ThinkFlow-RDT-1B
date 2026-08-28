from __future__ import annotations

import os

from scripts.evaluate_simpler_b0_oxe import (
    AF_UNIX_SAFE_PATH_BYTES,
    worker_socket_path,
)


def test_worker_socket_path_stays_below_linux_limit(monkeypatch, tmp_path):
    deliberately_long = tmp_path / ("long-output-component-" * 8)
    monkeypatch.setenv("THINKFLOW_SIMPLER_SOCKET_DIR", str(deliberately_long))

    socket_path = worker_socket_path()

    assert len(os.fsencode(socket_path)) <= AF_UNIX_SAFE_PATH_BYTES
    assert socket_path.suffix == ".sock"
    assert socket_path.name.startswith("tfse_")
    assert socket_path.parent != deliberately_long
