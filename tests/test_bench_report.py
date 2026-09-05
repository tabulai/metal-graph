import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest

from bench import run as bench_run

from bench.run import (
    HITS_ABS_L1_TARGET,
    SNAP_CACHE_SCHEMA_VERSION,
    build_structural_adjacency,
    build_igraph_graph,
    build_rustworkx_graph,
    build_suite,
    download_dataset,
    hits_agreement,
    json_safe,
    load_snap_cache,
    make_igraph_dense_bfs,
    make_rustworkx_dense_bfs,
    make_scipy_hits_runner,
    normalize_snap_ids,
    normalized_hits_arrays,
    render_markdown,
    stored_out_degrees,
    validate_dataset,
    validate_hits_l1_output,
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


def test_hits_benchmark_rows_are_self_describing(monkeypatch):
    calls = []

    class FakeMetalGraph:
        mode = "auto"

        @classmethod
        def set_execution(cls, mode):
            cls.mode = mode

        @staticmethod
        def has_gpu():
            return True

        @staticmethod
        def hits(graph, **options):
            calls.append((graph, options))
            values = np.full(4, 0.25, np.float32)
            return values, values.copy()

        @classmethod
        def last_run_info(cls):
            return {
                "op": "hits",
                "path": cls.mode,
                "iterations": 5,
                "ms": 8.25 if cls.mode == "cpu" else 1.75,
            }

    class Graph:
        num_vertices = 4

    graph = Graph()
    monkeypatch.setattr(
        bench_run,
        "timed",
        lambda fn: (
            12.5 if FakeMetalGraph.mode == "cpu" else 3.0,
            fn(),
        ),
    )

    def fake_stats(fn, runs):
        assert runs == 20
        fn()
        median = 10.0 if FakeMetalGraph.mode == "cpu" else 2.0
        return {
            "median_ms": median,
            "p95_ms": median * 1.2,
            "min_ms": median * 0.9,
            "runs": runs,
        }

    monkeypatch.setattr(bench_run, "stats_ms", fake_stats)
    rows = []

    def add(algo, item, stats, **extra):
        rows.append({"dataset": "tiny", "algo": algo, "item": item,
                     **stats, **extra})

    bench_run.bench_hits(FakeMetalGraph, graph, 20, add)

    assert FakeMetalGraph.mode == "auto"
    assert len(calls) == 4
    assert all(call_graph is graph for call_graph, _ in calls)
    assert all(options == {"tol": HITS_ABS_L1_TARGET / 4,
                           "max_iter": 100}
               for _, options in calls)
    assert [row["item"] for row in rows] == [
        "baseline_metal_cpu", "warm_full_run", "per_iteration"
    ]

    cpu, warm, per_iteration = rows
    assert cpu["cold_ms"] == 12.5
    assert cpu["path"] == "cpu"
    assert warm["cold_ms"] == 3.0
    assert warm["engine_ms"] == 1.75
    assert warm["iterations"] == 5
    assert warm["path"] == "gpu"
    assert warm["tol"] == HITS_ABS_L1_TARGET / 4
    assert warm["absolute_l1_target"] == HITS_ABS_L1_TARGET
    assert warm["max_iter"] == 100
    assert warm["normalization"] == "l1"
    assert warm["edge_weights"] == "ignored"
    assert warm["matches_reference"] is True
    assert warm["speedup_vs_metal_cpu"] == 5.0
    assert "A^T*h" in warm["iteration_unit"]
    assert per_iteration["median_ms"] == 0.4
    assert per_iteration["p95_ms"] == 0.48
    assert per_iteration["min_ms"] == 0.36
    assert per_iteration["iterations"] == 5

    meta = {
        "suite": "smoke",
        "timestamp_utc": "2026-09-04T00:00:00+00:00",
        "chip": "Apple test",
        "macos": "26.0",
        "python": "3.13",
        "versions": {"metal_graph": "0.1.0"},
    }
    rendered = render_markdown(meta, rows)
    assert "| tiny | hits | warm_full_run | 2.000 | 2.400" in rendered
    assert "cold_ms=3" in rendered
    assert "engine_ms=1.75" in rendered
    assert "edge_weights=ignored" in rendered
    assert "speedup_vs_metal_cpu=5" in rendered
    assert "iteration_unit=one authority update" in rendered


def test_hits_result_adapter_handles_mappings_and_eigenvector_sign():
    hubs = {0: -1.0, 1: -3.0}
    authorities = {0: -2.0, 1: -2.0}
    got_h, got_a = normalized_hits_arrays((hubs, authorities), 2)
    np.testing.assert_allclose(got_h, [0.25, 0.75])
    np.testing.assert_allclose(got_a, [0.5, 0.5])
    agreement = hits_agreement(
        (np.array([0.25, 0.75]), np.array([0.5, 0.5])),
        (hubs, authorities),
        2,
    )
    assert agreement["matches_reference"] is True
    assert agreement["hub_l1_error"] == 0.0


def test_native_hits_output_validator_rejects_scale_regressions():
    valid = (np.array([0.25, 0.75]), np.array([0.5, 0.5]))
    contract = validate_hits_l1_output(valid, 2)
    assert contract == {
        "raw_hubs_l1_norm": 1.0,
        "raw_authorities_l1_norm": 1.0,
    }

    with pytest.raises(ValueError, match="not L1-normalized"):
        validate_hits_l1_output((valid[0] * 1000.0, valid[1]), 2)


def test_structural_scipy_hits_preserves_multiplicity_and_recurrence():
    scipy_sparse = pytest.importorskip("scipy.sparse")
    src = np.array([0, 0, 0, 1, 2], dtype=np.uint32)
    dst = np.array([0, 1, 1, 2, 0], dtype=np.uint32)
    adjacency = build_structural_adjacency(
        scipy_sparse, src, dst, 3, True
    )
    np.testing.assert_array_equal(
        adjacency.toarray(),
        [[1.0, 2.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
    )

    run, state = make_scipy_hits_runner(
        adjacency, tol=1e-30, max_iter=3, audit_interval=5
    )
    hubs, authorities = run()
    expected_hubs = np.full(3, 1.0 / 3.0)
    for _ in range(3):
        expected_authorities = adjacency.T @ expected_hubs
        expected_authorities /= expected_authorities.sum()
        expected_hubs = adjacency @ expected_authorities
        expected_hubs /= expected_hubs.sum()
    assert state["iterations"] == 3
    np.testing.assert_allclose(hubs, expected_hubs)
    np.testing.assert_allclose(authorities, expected_authorities)


@pytest.mark.parametrize(
    ("stem", "expected_sha", "expected_dirty"),
    [
        (
            "bench-20260729T000207Z",
            "be8763c8b87de72d1b92e36739809a02651cc116",
            False,
        ),
        (
            "bench-20260729T131101Z-orkut-bounded",
            "a4a8bc325785b4b73f92ff896ec4e7e44ed5b4a6",
            False,
        ),
        (
            "bench-20260729T201007Z",
            "7afb4b2ff3dc2a520ecb1388e4b22e4619ab1989",
            False,
        ),
        (
            "bench-20260904T235705Z",
            "0fd0dc590313395a49979498caaa54ed3701b29d",
            True,
        ),
    ],
)
def test_checked_in_artifact_pairs_are_strict_and_reproducible(
    stem, expected_sha, expected_dirty
):
    results = Path(__file__).resolve().parents[1] / "bench/results"
    json_path = results / f"{stem}.json"
    data = json.loads(
        json_path.read_text(),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )
    assert data["schema_version"] == 1
    assert data["meta"]["git_sha"] == expected_sha
    assert data["meta"]["git_dirty"] is expected_dirty
    assert len(data["meta"]["native_module_sha256"]) == 64
    assert set(data["meta"]["snap_datasets"]) == {
        "soc-LiveJournal1",
        "com-orkut",
    }
    expected = (results / f"{stem}.md").read_text()
    assert render_markdown(data["meta"], data["rows"]) == expected


def test_checked_in_hits_artifact_has_verified_speedup_rows():
    results = Path(__file__).resolve().parents[1] / "bench/results"
    data = json.loads(
        (results / "bench-20260904T235705Z.json").read_text()
    )
    assert data["meta"]["argv"][2:4] == ["--algorithm", "hits"]
    harness = Path(__file__).resolve().parents[1] / "bench/run.py"
    assert data["meta"]["harness_sha256"] == hashlib.sha256(
        harness.read_bytes()
    ).hexdigest()
    for dataset in ("rmat18", "kg-hipporag"):
        rows = {
            row["item"]: row for row in data["rows"]
            if row["dataset"] == dataset and row["algo"] == "hits"
        }
        gpu = rows["warm_full_run"]
        assert gpu["path"] == "gpu"
        assert gpu["matches_reference"] is True
        assert gpu["runs"] >= 20
        assert gpu["speedup_vs_metal_cpu"] > 1.0
        for item in ("baseline_metal_cpu", "warm_full_run"):
            assert np.isclose(rows[item]["raw_hubs_l1_norm"], 1.0,
                              rtol=0.0, atol=5e-5)
            assert np.isclose(rows[item]["raw_authorities_l1_norm"], 1.0,
                              rtol=0.0, atol=5e-5)
        for item in ("baseline_scipy", "baseline_rustworkx",
                     "baseline_igraph"):
            assert rows[item]["matches_reference"] is True
            assert rows[item]["speedup_vs_gpu"] > 1.0
        for item in ("baseline_metal_cpu", "warm_full_run",
                     "baseline_scipy", "baseline_rustworkx",
                     "baseline_igraph"):
            row = rows[item]
            samples = row["samples_ms"]
            assert len(samples) == row["runs"] == 20
            assert samples == sorted(samples)
            assert row["median_ms"] == (samples[9] + samples[10]) / 2
            assert row["p95_ms"] == samples[18]
        for item in ("baseline_rustworkx", "baseline_igraph"):
            assert "absolute_l1_target" not in rows[item]
        if dataset == "kg-hipporag":
            assert "absolute_l1_target" not in rows["baseline_networkx"]


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


def test_tiny_bfs_slo_metadata_semantics():
    from bench.run import (
        TINY_BFS_SLO_GATE,
        TINY_BFS_SLO_MS,
        tiny_bfs_slo,
    )

    assert TINY_BFS_SLO_MS == 0.050
    metadata = tiny_bfs_slo(0.012)
    assert metadata == {
        "slo_ms": 0.050,
        "slo_pass": True,
        "gate": TINY_BFS_SLO_GATE,
    }
    # This helper contributes annotations to the real measured row; it must
    # not manufacture a second timing row with invented sample provenance.
    assert {"median_ms", "p95_ms", "runs"}.isdisjoint(metadata)
    assert tiny_bfs_slo(0.050)["slo_pass"] is True  # boundary inclusive
    assert tiny_bfs_slo(0.051)["slo_pass"] is False
    assert tiny_bfs_slo(0.012, slo_ms=0.01)["slo_pass"] is False


def test_build_suite_dataset_filter():
    from bench.run import build_suite

    names = [n for n, _ in build_suite("smoke", only="kg-hipporag")]
    assert names == ["kg-hipporag"]
    with pytest.raises(SystemExit, match="rmat18, kg-hipporag"):
        build_suite("smoke", only="soc-LiveJournal1")  # v01-only dataset
