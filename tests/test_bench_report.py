import hashlib
import json
import os

import numpy as np
import pytest

from bench import run as bench_run

from bench.run import (
    SNAP_CACHE_SCHEMA_VERSION,
    build_igraph_graph,
    build_rustworkx_graph,
    build_suite,
    json_safe,
    load_snap_cache,
    render_markdown,
    validate_dataset,
)


def test_json_safe_produces_strict_json():
    report = {
        "finite": 1.5,
        "values": [
            float("nan"),
            float("inf"),
            float("-inf"),
            np.float32("inf"),
        ],
        "integer": np.int64(7),
    }
    encoded = json.dumps(json_safe(report), allow_nan=False)
    assert json.loads(encoded) == {
        "finite": 1.5,
        "values": [None, None, None, None],
        "integer": 7,
    }


def test_validate_dataset_checks_size_and_digest(tmp_path):
    payload = b"metal-graph benchmark fixture"
    path = tmp_path / "dataset.gz"
    path.write_bytes(payload)
    spec = {
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    validate_dataset(path, spec)

    path.write_bytes(payload + b"!")
    with pytest.raises(RuntimeError, match="expected .* bytes"):
        validate_dataset(path, spec)


def test_snap_datasets_only_run_in_full_suite():
    assert [name for name, _ in build_suite("smoke")] == [
        "rmat18",
        "kg-hipporag",
    ]
    assert [name for name, _ in build_suite("v01")] == [
        "rmat18",
        "kg-hipporag",
        "rmat22",
        "rmat24",
        "soc-LiveJournal1",
        "com-orkut",
    ]


def test_snap_cache_requires_schema_and_structure(tmp_path):
    path = tmp_path / "dataset.npz"
    spec = {
        "sha256": "abc123",
        "edges": 2,
        "vertices": 3,
        "directed": True,
    }
    np.savez_compressed(
        path,
        cache_schema_version=SNAP_CACHE_SCHEMA_VERSION,
        source_sha256=spec["sha256"],
        src=np.array([0, 1], dtype=np.uint32),
        dst=np.array([1, 2], dtype=np.uint32),
        v=3,
        directed=True,
    )
    loaded = load_snap_cache(path, spec)
    assert loaded is not None
    np.testing.assert_array_equal(loaded[0], [0, 1])
    np.testing.assert_array_equal(loaded[1], [1, 2])

    np.savez_compressed(
        path,
        source_sha256=spec["sha256"],
        src=np.array([0, 1], dtype=np.uint32),
        dst=np.array([1, 2], dtype=np.uint32),
        v=3,
        directed=True,
    )
    assert load_snap_cache(path, spec) is None


def test_renderer_accepts_legacy_canonical_artifact():
    meta = {
        "suite": "smoke",
        "timestamp_utc": "2026-07-28T00:00:00+00:00",
        "chip": "Apple test",
        "macos": "14.0",
        "python": "3.10",
        "versions": {"metal_graph": "0.1.0"},
    }
    rows = [{
        "dataset": "tiny",
        "algo": "pagerank",
        "item": "warm",
        "median_ms": 1.0,
        "p95_ms": None,
        "peak_rss_delta_mb": None,
    }]
    rendered = render_markdown(meta, rows)
    assert "git `unknown` (unknown)" in rendered


def test_energy_capture_stops_on_exception(monkeypatch, tmp_path):
    process = object()
    stopped = []
    monkeypatch.setattr(
        bench_run, "start_energy_capture", lambda path: process
    )
    monkeypatch.setattr(
        bench_run, "stop_energy_capture", lambda proc: stopped.append(proc)
    )

    with pytest.raises(RuntimeError, match="benchmark failed"):
        with bench_run.energy_capture(True, tmp_path / "power.txt"):
            raise RuntimeError("benchmark failed")
    assert stopped == [process]


def test_energy_stop_uses_pre_authorized_control_channel(
    monkeypatch, tmp_path
):
    class FakeProcess:
        waited = False

        @staticmethod
        def poll():
            return None

        def wait(self, timeout):
            assert timeout == 30
            self.waited = True

    monkeypatch.setattr(
        bench_run.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail(
            "shutdown must not require a fresh sudo credential"
        ),
    )
    control_dir = tmp_path / "energy-control"
    control_dir.mkdir()
    control_fifo = control_dir / "stop"
    os.mkfifo(control_fifo, mode=0o600)
    control_fd = os.open(control_fifo, os.O_RDWR | os.O_NONBLOCK)
    process = FakeProcess()
    handle = bench_run.EnergyCaptureHandle(
        process=process,
        control_fd=control_fd,
        control_fifo=control_fifo,
        control_dir=control_dir,
    )
    bench_run.stop_energy_capture(handle)
    assert process.waited
    assert not control_dir.exists()


def weighted_fixture():
    src = np.array([0, 0, 1, 2], dtype=np.uint32)
    dst = np.array([1, 2, 0, 0], dtype=np.uint32)
    left = np.array([9.0, 1.0, 1.0, 1.0], dtype=np.float32)
    right = np.array([1.0, 9.0, 1.0, 1.0], dtype=np.float32)
    return src, dst, left, right


def test_rustworkx_gate_baseline_respects_weights():
    rx = pytest.importorskip("rustworkx")
    src, dst, left, right = weighted_fixture()
    graph_left, fn_left = build_rustworkx_graph(
        rx, src, dst, left, 3, True
    )
    graph_right, fn_right = build_rustworkx_graph(
        rx, src, dst, right, 3, True
    )
    rank_left = rx.pagerank(graph_left, weight_fn=fn_left)
    rank_right = rx.pagerank(graph_right, weight_fn=fn_right)
    assert not np.allclose(
        [rank_left[i] for i in range(3)],
        [rank_right[i] for i in range(3)],
    )


def test_igraph_gate_baseline_respects_weights():
    ig = pytest.importorskip("igraph")
    src, dst, left, right = weighted_fixture()
    graph_left, weight_left = build_igraph_graph(
        ig, src, dst, left, 3, True
    )
    graph_right, weight_right = build_igraph_graph(
        ig, src, dst, right, 3, True
    )
    rank_left = graph_left.pagerank(weights=weight_left)
    rank_right = graph_right.pagerank(weights=weight_right)
    assert not np.allclose(rank_left, rank_right)
