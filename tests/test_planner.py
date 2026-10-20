# test_planner.py — execution planner: modes, telemetry, env knobs
# (subprocess-isolated: env is not guaranteed to be re-read per call),
# and GPU/CPU agreement across the fixture matrix.
import json

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
    # The lowest per-op GPU threshold is 200k stored edges (pagerank/hits);
    # every fixture is far below.
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


@pytest.mark.gpu
def test_auto_bfs_sparse_component_overrides_graph_size_subprocess():
    if not has_gpu():
        pytest.skip("no Metal device")
    code = TINY_GRAPH_CODE + """
mg.set_execution("auto")
mg.bfs(g, [0], direction="out")
info = mg.last_run_info()
print("PATH", info["path"])
print("OP", info["op"])
"""
    r = run_python(code, {
        "MG_E_GPU_MIN": "1",
        "MG_BFS_SPARSE_MAX_VERTICES": "16",
        "MG_BFS_SPARSE_MAX_EDGES": "16",
        "MG_REQUIRE_GPU": "0",
        "MG_FORCE_CPU": "0",
    })
    assert r.returncode == 0, r.stderr
    assert "PATH cpu" in r.stdout
    assert "OP bfs_sparse" in r.stdout


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


# ---------------------------------------------------------------------------
# per-op planner thresholds (subprocess)
# ---------------------------------------------------------------------------

PLAN_OPS_CODE = TINY_GRAPH_CODE + """
import json
mg.set_execution("auto")
paths = {}
mg.pagerank(g, alpha=0.85, tol=1e-6, max_iter=10)
paths["pagerank"] = mg.last_run_info()["path"]
mg.hits(g)
paths["hits"] = mg.last_run_info()["path"]
mg.ppr_topk(g, np.array([0], np.uint32), np.array([1.0], np.float32),
            np.array([0, 1], np.uint64), k=2)
paths["ppr"] = mg.last_run_info()["path"]
mg.bfs(g, [0])
paths["bfs"] = mg.last_run_info()["path"]
mg.experimental.wcc(g)
paths["wcc"] = mg.last_run_info()["path"]
mg.k_hop(g, [0], k=1, direction="out")
paths["khop"] = mg.last_run_info()["path"]
print("PATHS", json.dumps(paths))
"""


# Empty values count as unset: shields these tests from the caller's shell.
_NO_THRESHOLD_ENV = {f"MG_E_GPU_MIN{sfx}": "" for sfx in (
    "", "_PAGERANK", "_HITS", "_PPR", "_BFS", "_WCC", "_KHOP")}


def _plan_paths(env):
    base = dict(_NO_THRESHOLD_ENV)
    base.update({"MG_REQUIRE_GPU": "0", "MG_FORCE_CPU": "0",
            "MG_BFS_SPARSE_MAX_VERTICES": "0", "MG_BFS_SPARSE_MAX_EDGES": "0",
            "MG_PPR_SPARSE_MAX_VERTICES": "0", "MG_PPR_SPARSE_MAX_EDGES": "0"})
    base.update(env)
    r = run_python(PLAN_OPS_CODE, base)
    assert r.returncode == 0, r.stderr
    line = next(x for x in r.stdout.splitlines() if x.startswith("PATHS "))
    return json.loads(line[len("PATHS "):])


@pytest.mark.gpu
def test_per_op_threshold_precedence_subprocess():
    """MG_E_GPU_MIN_<OP> beats the global MG_E_GPU_MIN, which beats the
    per-op default; the global override applies to every op, k_hop too."""
    if not has_gpu():
        pytest.skip("no Metal device")
    ops = ("pagerank", "hits", "ppr", "bfs", "wcc", "khop")
    assert _plan_paths({}) == {op: "cpu" for op in ops}
    assert _plan_paths({"MG_E_GPU_MIN": "0"}) == {op: "gpu" for op in ops}
    paths = _plan_paths({"MG_E_GPU_MIN": "0",
                         "MG_E_GPU_MIN_KHOP": "1000000000",
                         "MG_E_GPU_MIN_HITS": "1000000000"})
    assert paths == {"pagerank": "gpu", "hits": "cpu", "ppr": "gpu",
                     "bfs": "gpu", "wcc": "gpu", "khop": "cpu"}
    paths = _plan_paths({"MG_E_GPU_MIN": "1000000000",
                         "MG_E_GPU_MIN_PAGERANK": "0",
                         "MG_E_GPU_MIN_WCC": "0"})
    assert paths == {"pagerank": "gpu", "hits": "cpu", "ppr": "cpu",
                     "bfs": "cpu", "wcc": "gpu", "khop": "cpu"}


@pytest.mark.gpu
def test_default_per_op_thresholds_subprocess():
    """Defaults: pagerank/hits switch to the GPU at 200k stored edges,
    ppr/bfs/wcc stay on the CPU below 1M, and auto k_hop never uses the
    GPU, even at 1M+ stored edges."""
    if not has_gpu():
        pytest.skip("no Metal device")
    code = """
import json
import numpy as np
import metal_graph as mg
rng = np.random.default_rng(1)
def graph(e):
    return mg.Graph.from_edges(rng.integers(0, 50_000, e).astype(np.uint32),
                               rng.integers(0, 50_000, e).astype(np.uint32),
                               directed=True, num_vertices=50_000)
mg.set_execution("auto")
out = {}
g = graph(250_000)
mg.pagerank(g, max_iter=5); out["pagerank_250k"] = mg.last_run_info()["path"]
mg.hits(g, max_iter=5); out["hits_250k"] = mg.last_run_info()["path"]
mg.experimental.wcc(g); out["wcc_250k"] = mg.last_run_info()["path"]
g = graph(1_000_000)
mg.k_hop(g, [0], k=2, direction="out"); out["khop_1m"] = mg.last_run_info()["path"]
mg.experimental.wcc(g); out["wcc_1m"] = mg.last_run_info()["path"]
print("PATHS", json.dumps(out))
"""
    r = run_python(code, dict(_NO_THRESHOLD_ENV, MG_REQUIRE_GPU="0",
                              MG_FORCE_CPU="0"))
    assert r.returncode == 0, r.stderr
    line = next(x for x in r.stdout.splitlines() if x.startswith("PATHS "))
    assert json.loads(line[len("PATHS "):]) == {
        "pagerank_250k": "gpu", "hits_250k": "gpu", "wcc_250k": "cpu",
        "khop_1m": "cpu", "wcc_1m": "gpu"}
