"""Installed-wheel smoke test run outside the source tree by cibuildwheel."""

from __future__ import annotations

import argparse
from importlib.metadata import version
import os
from pathlib import Path
import platform


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expect", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--require-macos-major")
    args = parser.parse_args()

    assert "PYTHONPATH" not in os.environ, "PYTHONPATH must be genuinely unset"
    assert "MG_METALLIB" not in os.environ, "must test the embedded Metal library"
    assert platform.system() == "Darwin", platform.platform()
    assert platform.machine() == "arm64", platform.machine()
    if args.require_macos_major:
        actual_major = platform.mac_ver()[0].split(".", 1)[0]
        assert actual_major == args.require_macos_major, (
            actual_major,
            args.require_macos_major,
        )

    project_root = args.project_root.resolve()
    working_directory = Path.cwd().resolve()
    assert not working_directory.is_relative_to(project_root), (
        working_directory,
        project_root,
    )

    import numpy as np
    import metal_graph as mg

    installed_package = Path(mg.__file__).resolve()
    assert not installed_package.is_relative_to(project_root), (
        installed_package,
        project_root,
    )
    assert version("metal-graph") == mg.__version__
    assert mg.__version__ == mg._core.__version__ == mg._core.version()
    initial = mg.last_run_info()
    assert initial["converged"] is None
    assert initial["stop_reason"] == "not_applicable"
    assert initial["final_delta_l1"] is None

    if args.expect == "cpu":
        assert os.environ.get("MG_FORCE_CPU") == "1"
        assert "MG_REQUIRE_GPU" not in os.environ
        assert mg.has_gpu() is False
    else:
        assert "MG_FORCE_CPU" not in os.environ
        assert os.environ.get("MG_REQUIRE_GPU") == "1"
        assert mg.has_gpu() is True
    mg.set_execution(args.expect)

    graph = mg.Graph.from_edges(
        np.array([0, 1, 2], dtype=np.uint32),
        np.array([1, 2, 0], dtype=np.uint32),
        directed=True,
        num_vertices=3,
    )
    rank = np.asarray(mg.pagerank(graph, tol=1e-8, max_iter=100))
    np.testing.assert_allclose(
        rank,
        np.full(3, 1.0 / 3.0),
        atol=1e-6,
        rtol=1e-6,
    )
    assert mg.last_run_info()["path"] == args.expect

    hubs, authorities = mg.hits(graph, tol=1e-8, max_iter=100)
    np.testing.assert_allclose(hubs, np.full(3, 1.0 / 3.0), atol=1e-6)
    np.testing.assert_allclose(
        authorities,
        np.full(3, 1.0 / 3.0),
        atol=1e-6,
    )
    info = mg.last_run_info()
    assert info["op"] == "hits"
    assert info["path"] == args.expect
    assert info["converged"] is True
    assert info["stop_reason"] == "converged"
    assert 0.0 <= info["final_delta_l1"] < 3e-8

    edge = mg.Graph.from_edges(
        np.array([0], dtype=np.uint32),
        np.array([1], dtype=np.uint32),
        directed=True,
        num_vertices=2,
    )
    for budget, converged, reason, delta in (
            (1, False, "max_iter", 1.0),
            (2, True, "converged", 0.0)):
        hubs, authorities = mg.hits(edge, tol=1e-6, max_iter=budget)
        np.testing.assert_array_equal(hubs, [1.0, 0.0])
        np.testing.assert_array_equal(authorities, [0.0, 1.0])
        info = mg.last_run_info()
        assert info["path"] == args.expect
        assert info["iterations"] == budget
        assert info["converged"] is converged
        assert info["stop_reason"] == reason
        assert info["final_delta_l1"] == delta

    for vertices in (0, 2):
        empty = mg.Graph.from_edges(
            np.array([], dtype=np.uint32),
            np.array([], dtype=np.uint32),
            directed=True,
            num_vertices=vertices,
        )
        hubs, authorities = mg.hits(empty)
        np.testing.assert_array_equal(hubs, np.zeros(vertices))
        np.testing.assert_array_equal(authorities, np.zeros(vertices))
        info = mg.last_run_info()
        assert info["path"] == (args.expect if vertices else "cpu")
        assert info["iterations"] == 0
        assert info["converged"] is True
        assert info["stop_reason"] == "empty_graph"
        assert info["final_delta_l1"] == 0.0

    distance, parent = mg.bfs(graph, sources=[0], direction="out")
    np.testing.assert_array_equal(distance, np.array([0, 1, 2]))
    np.testing.assert_array_equal(parent, np.array([-1, 0, 1]))
    assert mg.last_run_info()["path"] == args.expect
    assert mg.last_run_info()["converged"] is None
    assert mg.last_run_info()["stop_reason"] == "not_applicable"
    assert mg.last_run_info()["final_delta_l1"] is None

    # A three-cycle with alpha=1/2 has exact personalized scores 4/7, 2/7,
    # 1/7 starting at the seed. Test batching, ranking, and padding together.
    ids, scores = mg.ppr_topk(
        graph,
        seeds=[0, 1],
        seed_weights=[1.0, 1.0],
        seed_offsets=[0, 1, 2],
        k=4,
        alpha=0.5,
        tol=1e-8,
        max_iter=100,
    )
    np.testing.assert_array_equal(ids, [[0, 1, 2, -1], [1, 2, 0, -1]])
    np.testing.assert_allclose(
        scores,
        [[4.0 / 7.0, 2.0 / 7.0, 1.0 / 7.0, 0.0]] * 2,
        atol=1e-6,
        rtol=1e-6,
    )
    assert mg.last_run_info()["op"] == "ppr_topk"
    assert mg.last_run_info()["path"] == args.expect

    # Uncapped, outward traversal honors the requested execution path.
    # The nonconsecutive reached ids also exercise as_graph's dense remap.
    vertices, edges = mg.k_hop(graph, seeds=[2], k=1, direction="out")
    np.testing.assert_array_equal(vertices, [0, 2])
    np.testing.assert_array_equal(edges, [2])
    assert mg.last_run_info()["op"] == "k_hop"
    assert mg.last_run_info()["path"] == args.expect
    subgraph = mg.k_hop(
        graph, seeds=[2], k=1, direction="out", as_graph=True
    )
    assert mg.last_run_info()["path"] == args.expect
    assert subgraph.num_vertices == 2
    assert subgraph.num_edges == 1
    np.testing.assert_array_equal(subgraph.external_ids, [0, 2])
    distance, parent = mg.bfs(subgraph, sources=[1], direction="out")
    np.testing.assert_array_equal(distance, [1, 0])
    np.testing.assert_array_equal(parent, [1, -1])
    assert mg.last_run_info()["path"] == args.expect

    # Include disconnected components and isolated vertices; exercise both
    # directed weak connectivity and the undirected GPU filtering path.
    for directed in (True, False):
        disconnected = mg.Graph.from_edges(
            np.array([2, 1, 5], dtype=np.uint32),
            np.array([0, 2, 3], dtype=np.uint32),
            directed=directed,
            num_vertices=7,
        )
        components = mg.experimental.wcc(disconnected)
        np.testing.assert_array_equal(components, [0, 0, 0, 1, 2, 1, 3])
        info = mg.last_run_info()
        assert info["op"] == "wcc"
        assert info["path"] == args.expect
        if args.expect == "gpu":
            assert info["iterations"] == 1
        assert mg.experimental.wcc_count(disconnected) == 4
        assert mg.last_run_info()["path"] == args.expect

    # Cross the parallel-numbering threshold and force threaded CPU union
    # find, including its atomic fallback in wheels built on macOS 14.
    os.environ["MG_THREADS"] = "4"
    os.environ["MG_WCC_CPU_PAR_MIN_EDGES"] = "1"
    vertex_count = 131_080
    chain_sources = np.arange(vertex_count - 3, dtype=np.uint32)
    chain = mg.Graph.from_edges(
        chain_sources,
        chain_sources + 1,
        directed=True,
        num_vertices=vertex_count,
    )
    expected_components = np.zeros(vertex_count, dtype=np.int32)
    expected_components[-2:] = [1, 2]
    np.testing.assert_array_equal(
        mg.experimental.wcc(chain), expected_components
    )
    info = mg.last_run_info()
    assert info["op"] == "wcc"
    assert info["path"] == args.expect
    if args.expect == "gpu":
        assert info["iterations"] == 1
    assert mg.experimental.wcc_count(chain) == 3
    assert mg.last_run_info()["path"] == args.expect

    print(
        f"installed wheel smoke passed: version={mg.__version__} "
        f"path={args.expect} package={installed_package}"
    )


if __name__ == "__main__":
    main()
