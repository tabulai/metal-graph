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

    print(
        f"installed wheel smoke passed: version={mg.__version__} "
        f"path={args.expect} package={installed_package}"
    )


if __name__ == "__main__":
    main()
