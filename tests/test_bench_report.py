import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest

from bench import run as bench_run

from bench.run import (
    SNAP_CACHE_SCHEMA_VERSION,
    build_igraph_graph,
    build_rustworkx_graph,
    build_suite,
    download_dataset,
    json_safe,
    load_snap_cache,
    make_igraph_dense_bfs,
    make_rustworkx_dense_bfs,
    normalize_snap_ids,
    render_markdown,
    stored_out_degrees,
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


def test_download_is_bounded_and_stream_verified(monkeypatch, tmp_path):
    payload = b"pinned benchmark payload"
    spec = {
        "url": "https://example.test/dataset.gz",
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    monkeypatch.setattr(
        bench_run.urllib.request,
        "urlopen",
        lambda request, timeout: io.BytesIO(payload),
    )
    destination = tmp_path / "dataset.part"
    download_dataset(spec, destination)
    assert destination.read_bytes() == payload

    oversized = dict(spec, bytes=len(payload) - 1)
    with pytest.raises(RuntimeError, match="exceeds pinned size"):
        download_dataset(oversized, tmp_path / "oversized.part")


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


def test_snap_ids_are_validated_and_densely_renumbered():
    dense_src = np.array([0, 1, 2], dtype=np.uint32)
    dense_dst = np.array([1, 2, 0], dtype=np.uint32)
    src, dst, vertices = normalize_snap_ids(
        dense_src, dense_dst, 3, "dense fixture"
    )
    assert src is dense_src
    assert dst is dense_dst
    assert vertices == 3

    sparse_src = np.array([10, 40, 10], dtype=np.uint32)
    sparse_dst = np.array([40, 70, 70], dtype=np.uint32)
    src, dst, vertices = normalize_snap_ids(
        sparse_src, sparse_dst, 3, "sparse fixture"
    )
    np.testing.assert_array_equal(src, [0, 1, 0])
    np.testing.assert_array_equal(dst, [1, 2, 2])
    assert vertices == 3

    with pytest.raises(RuntimeError, match="expected 4 unique vertices"):
        normalize_snap_ids(sparse_src, sparse_dst, 4, "bad fixture")


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


def test_checked_in_canonical_artifact_is_strict_and_reproducible():
    results = Path(__file__).resolve().parents[1] / "bench/results"
    json_path = results / "bench-20260729T000207Z.json"
    data = json.loads(
        json_path.read_text(),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )
    assert data["schema_version"] == 1
    assert data["meta"]["git_sha"] == (
        "be8763c8b87de72d1b92e36739809a02651cc116"
    )
    assert data["meta"]["git_dirty"] is False
    assert len(data["meta"]["native_module_sha256"]) == 64
    assert set(data["meta"]["snap_datasets"]) == {
        "soc-LiveJournal1",
        "com-orkut",
    }
    expected = (
        results / "bench-20260729T000207Z.md"
    ).read_text()
    assert render_markdown(data["meta"], data["rows"]) == expected


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


def test_energy_stop_does_not_need_a_second_sudo(monkeypatch):
    class FakeProcess:
        waited = False
        terminated = False

        @staticmethod
        def poll():
            return None

        def terminate(self):
            self.terminated = True

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
    process = FakeProcess()
    output_stream = io.BytesIO()
    handle = bench_run.EnergyCaptureHandle(
        process=process,
        output_stream=output_stream,
    )
    bench_run.stop_energy_capture(handle)
    assert process.terminated
    assert process.waited
    assert output_stream.closed


def test_energy_start_elevates_only_fixed_powermetrics(
    monkeypatch, tmp_path
):
    command = []

    class Result:
        returncode = 0

    class FakeProcess:
        @staticmethod
        def poll():
            return 0

    monkeypatch.setattr(bench_run.os, "geteuid", lambda: 501)
    monkeypatch.setattr(
        bench_run.subprocess, "run", lambda *args, **kwargs: Result()
    )

    def fake_popen(args, **kwargs):
        command.extend(args)
        assert kwargs["stdout"].name == str(tmp_path / "power.txt")
        return FakeProcess()

    monkeypatch.setattr(bench_run.subprocess, "Popen", fake_popen)
    handle = bench_run.start_energy_capture(tmp_path / "power.txt")
    assert command == [
        "sudo",
        "-n",
        "/usr/bin/powermetrics",
        "-i",
        "100",
        "--samplers",
        "cpu_power,gpu_power",
    ]
    handle.output_stream.write(b"sample")
    handle.output_stream.flush()
    bench_run.stop_energy_capture(handle)


def test_energy_stop_timeout_is_a_hard_error():
    class FakeProcess:
        @staticmethod
        def poll():
            return None

        @staticmethod
        def terminate():
            return None

        @staticmethod
        def wait(timeout):
            raise bench_run.subprocess.TimeoutExpired("sudo", timeout)

    output_stream = io.BytesIO()
    handle = bench_run.EnergyCaptureHandle(
        process=FakeProcess(),
        output_stream=output_stream,
    )
    with pytest.raises(RuntimeError, match="did not stop"):
        bench_run.stop_energy_capture(handle)
    assert output_stream.closed


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


def test_rustworkx_bfs_gate_returns_dense_dist_and_parent():
    rx = pytest.importorskip("rustworkx")
    src = np.array([0, 0, 1, 3], dtype=np.uint32)
    dst = np.array([1, 2, 3, 4], dtype=np.uint32)
    graph, _ = build_rustworkx_graph(
        rx, src, dst, None, 6, True
    )
    dist, parent = make_rustworkx_dense_bfs(rx, graph, 6, 0)()
    np.testing.assert_array_equal(dist, [0, 1, 1, 2, 3, -1])
    assert parent[0] == -1
    assert parent[1] == 0
    assert parent[2] == 0
    assert parent[3] == 1
    assert parent[4] == 3
    assert parent[5] == -1


def test_stored_out_degrees_matches_directed_and_undirected_storage():
    src = np.array([0, 0, 1, 2], dtype=np.uint32)
    dst = np.array([0, 1, 2, 1], dtype=np.uint32)
    np.testing.assert_array_equal(
        stored_out_degrees(src, dst, 3, True), [2, 1, 1]
    )
    np.testing.assert_array_equal(
        stored_out_degrees(src, dst, 3, False), [2, 3, 2]
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


def test_igraph_bfs_gate_returns_dense_dist_and_parent():
    ig = pytest.importorskip("igraph")
    src = np.array([0, 0, 1, 3, 5], dtype=np.uint32)
    dst = np.array([1, 2, 3, 4, 0], dtype=np.uint32)
    graph, _ = build_igraph_graph(
        ig, src, dst, None, 6, True
    )

    dist, parent = make_igraph_dense_bfs(graph, 6, 0)()

    assert dist.shape == (6,)
    assert parent.shape == (6,)
    assert dist.dtype == np.int32
    assert parent.dtype == np.int32
    assert dist.flags.c_contiguous
    assert parent.flags.c_contiguous
    np.testing.assert_array_equal(dist, [0, 1, 1, 2, 3, -1])
    np.testing.assert_array_equal(parent, [-1, 0, 0, 1, 3, -1])


def test_igraph_bfs_gate_handles_singleton_graph():
    ig = pytest.importorskip("igraph")
    graph = ig.Graph(n=1, directed=True)

    dist, parent = make_igraph_dense_bfs(graph, 1, 0)()

    np.testing.assert_array_equal(dist, [0])
    np.testing.assert_array_equal(parent, [-1])


def test_tiny_bfs_slo_row_semantics():
    from bench.run import TINY_BFS_SLO_MS, tiny_bfs_slo

    assert TINY_BFS_SLO_MS == 0.050
    row = tiny_bfs_slo(0.012)
    assert row["slo_pass"] is True and row["slo_ms"] == 0.050
    assert tiny_bfs_slo(0.050)["slo_pass"] is True  # boundary inclusive
    assert tiny_bfs_slo(0.051)["slo_pass"] is False
    assert tiny_bfs_slo(0.012, slo_ms=0.01)["slo_pass"] is False
