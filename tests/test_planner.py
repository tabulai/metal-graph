# test_planner.py — execution planner: modes, telemetry, env knobs
# (subprocess-isolated: env is not guaranteed to be re-read per call),
# and GPU/CPU agreement across the fixture matrix.
import numpy as np
import pytest

mg = pytest.importorskip("metal_graph")

from conftest import (ALL_CASES, build_mg, has_gpu, khop_ref, make_weights,
                      run_python)

TINY_GRAPH_CODE = """
import numpy as np
import metal_graph as mg
g = mg.Graph.from_edges(np.array([0, 1, 2], dtype=np.uint32),
                        np.array([1, 2, 0], dtype=np.uint32),
                        directed=True, num_vertices=3)
"""


# ---------------------------------------------------------------------------
# modes + telemetry (in-process)
# ---------------------------------------------------------------------------

def test_set_execution_accepts_documented_modes():
    mg.set_execution("auto")
    mg.set_execution("cpu")
    if has_gpu():
        mg.set_execution("gpu")
    mg.set_execution("auto")


def test_set_execution_rejects_unknown_mode():
    with pytest.raises(ValueError):
        mg.set_execution("tpu")
    mg.set_execution("auto")


def test_last_run_info_pagerank_telemetry():
    case = ALL_CASES["gnp_dir"]
    g = build_mg(case)
    mg.set_execution("cpu")
    try:
        mg.pagerank(g, alpha=0.85, tol=1e-8, max_iter=100)
        info = mg.last_run_info()
        assert info["path"] == "cpu"
        assert "pagerank" in info["op"]
        assert info["iterations"] >= 1
        assert info["ms"] >= 0.0
    finally:
        mg.set_execution("auto")


def test_last_run_info_op_per_algorithm():
    case = ALL_CASES["gnp_dir"]
    g = build_mg(case)
    mg.set_execution("cpu")
    try:
        mg.bfs(g, np.asarray([0], np.uint32), direction="out")
        assert "bfs" in mg.last_run_info()["op"]
        mg.experimental.wcc(g)
        assert "wcc" in mg.last_run_info()["op"]
        mg.ppr_topk(g, np.array([0], np.uint32), np.array([1.0], np.float32),
                    np.array([0, 1], np.uint64), k=4, alpha=0.85, tol=1e-6,
                    max_iter=20)
        assert "ppr" in mg.last_run_info()["op"]
    finally:
        mg.set_execution("auto")


def test_auto_mode_picks_cpu_for_tiny_graph_inprocess():
    # Default MG_E_GPU_MIN is 1M stored edges; every fixture is far below.
    case = ALL_CASES["gnp_dir"]
    g = build_mg(case)
    mg.set_execution("auto")
    mg.pagerank(g, alpha=0.85, tol=1e-8, max_iter=50)
    assert mg.last_run_info()["path"] == "cpu"


# ---------------------------------------------------------------------------
# env knobs (subprocess — env may be read once at Runtime construction)
# ---------------------------------------------------------------------------

def test_gpu_mode_without_device_raises_subprocess():
    code = TINY_GRAPH_CODE + """
import sys
try:
    mg.set_execution("gpu")
    mg.pagerank(g, alpha=0.85, tol=1e-6, max_iter=10)
except Exception as e:
    print("RAISED", type(e).__name__)
    sys.exit(0)
print("NO_RAISE")
sys.exit(1)
"""
    r = run_python(code, {"MG_FORCE_CPU": "1", "MG_REQUIRE_GPU": "0"})
    assert r.returncode == 0 and "RAISED" in r.stdout, (
        f"mode='gpu' without a device must raise\n{r.stdout}\n{r.stderr}")


def test_require_gpu_with_force_cpu_hard_fails_subprocess():
    code = """
import sys
try:
    import numpy as np
    import metal_graph as mg
    g = mg.Graph.from_edges(np.array([0], dtype=np.uint32),
                            np.array([1], dtype=np.uint32),
                            directed=True, num_vertices=2)
    mg.pagerank(g, alpha=0.85, tol=1e-6, max_iter=10)
except Exception as e:
    print("HARD_FAIL", type(e).__name__)
    sys.exit(0)
print("NO_FAIL")
sys.exit(1)
"""
    r = run_python(code, {"MG_REQUIRE_GPU": "1", "MG_FORCE_CPU": "1"})
    # Import itself hard-failing (nonzero rc) also satisfies the contract.
    hard_failed = ("HARD_FAIL" in r.stdout) or (
        r.returncode != 0 and "NO_FAIL" not in r.stdout)
    assert hard_failed, (
        f"MG_REQUIRE_GPU=1 + MG_FORCE_CPU=1 must hard-fail\n"
        f"rc={r.returncode}\n{r.stdout}\n{r.stderr}")


def test_auto_mode_cpu_below_threshold_subprocess():
    code = TINY_GRAPH_CODE + """
mg.set_execution("auto")
mg.pagerank(g, alpha=0.85, tol=1e-6, max_iter=10)
print("PATH", mg.last_run_info()["path"])
"""
    r = run_python(code, {"MG_E_GPU_MIN": "1000000000",
                          "MG_REQUIRE_GPU": "0", "MG_FORCE_CPU": "0"})
    assert r.returncode == 0, r.stderr
    assert "PATH cpu" in r.stdout


@pytest.mark.gpu
def test_auto_mode_gpu_when_threshold_lowered_subprocess():
    if not has_gpu():
        pytest.skip("no Metal device")
    code = TINY_GRAPH_CODE + """
mg.set_execution("auto")
mg.pagerank(g, alpha=0.85, tol=1e-6, max_iter=10)
print("PATH", mg.last_run_info()["path"])
"""
    r = run_python(code, {"MG_E_GPU_MIN": "1", "MG_REQUIRE_GPU": "0",
                          "MG_FORCE_CPU": "0"})
    assert r.returncode == 0, r.stderr
    assert "PATH gpu" in r.stdout


# ---------------------------------------------------------------------------
# GPU/CPU agreement across the matrix
# ---------------------------------------------------------------------------

def _both_paths(fn, gpu_expected="gpu"):
    mg.set_execution("cpu")
    try:
        cpu = fn()
        assert mg.last_run_info()["path"] == "cpu"
        mg.set_execution("gpu")
        gpu = fn()
        assert mg.last_run_info()["path"] == gpu_expected
    finally:
        mg.set_execution("auto")
    return cpu, gpu


@pytest.mark.gpu
def test_agreement_pagerank(graph_case):
    if not has_gpu():
        pytest.skip("no Metal device")
    case = graph_case
    if case.num_vertices == 0:
        pytest.skip("V=0")
    g = build_mg(case, weights=make_weights(case))
    cpu, gpu = _both_paths(
        lambda: np.asarray(mg.pagerank(g, alpha=0.85, tol=1e-8,
                                       max_iter=200)))
    np.testing.assert_allclose(gpu, cpu, atol=5e-5, rtol=1e-3)


@pytest.mark.gpu
def test_agreement_bfs_dist(graph_case):
    if not has_gpu():
        pytest.skip("no Metal device")
    case = graph_case
    if case.num_vertices == 0:
        pytest.skip("V=0")
    g = build_mg(case)
    src = np.asarray([0], np.uint32)
    for direction in ("out", "in", "both"):
        expected = ("cpu" if (case.directed and direction == "both")
                    else "gpu")
        cpu, gpu = _both_paths(
            lambda d=direction: np.asarray(mg.bfs(g, src, direction=d)[0]),
            gpu_expected=expected)
        np.testing.assert_array_equal(gpu, cpu)


@pytest.mark.gpu
def test_agreement_wcc_partitions(graph_case):
    if not has_gpu():
        pytest.skip("no Metal device")
    case = graph_case
    if case.num_vertices == 0:
        pytest.skip("V=0")
    g = build_mg(case)
    cpu, gpu = _both_paths(lambda: np.asarray(mg.experimental.wcc(g)))
    # canonical numbering makes the label arrays identical, not just the
    # partitions
    np.testing.assert_array_equal(gpu, cpu)


@pytest.mark.gpu
def test_agreement_khop(graph_case):
    if not has_gpu():
        pytest.skip("no Metal device")
    case = graph_case
    if case.num_vertices == 0:
        pytest.skip("V=0")
    g = build_mg(case)
    seeds = np.asarray([0], np.uint32)

    def run():
        vs, es = mg.k_hop(g, seeds, k=2, direction="both")
        return np.asarray(vs), np.asarray(es)

    # both+directed traversal is CPU-routed in v0.1 and reported honestly.
    expected = "cpu" if case.directed else "gpu"
    (vs_c, es_c), (vs_g, es_g) = _both_paths(run, gpu_expected=expected)
    np.testing.assert_array_equal(vs_g, vs_c)
    np.testing.assert_array_equal(es_g, es_c)
    ref_vs, ref_es = khop_ref(case, [0], 2, "both")
    np.testing.assert_array_equal(vs_c, ref_vs)
    np.testing.assert_array_equal(es_c, ref_es)
