# test_scale.py — medium-scale GPU-vs-CPU agreement (RMAT-shape, ~2M edges).
#
# The small-fixture matrix cannot reach several GPU code paths: the
# bottom-up BFS switch, multi-block reduce/finalize, scan recursion depth
# > 1, and >65k-thread dispatches. This module runs one seeded power-law
# graph big enough to hit them and cross-checks the paths against each
# other (CPU implementations are the oracles).
import numpy as np
import pytest

mg = pytest.importorskip("metal_graph")

from metal_graph import _core

pytestmark = pytest.mark.gpu

V = 200_000
E = 2_000_000
SEED = 20260728


@pytest.fixture(scope="module")
def big_graph():
    if not _core.has_gpu():
        pytest.skip("no Metal device")
    rng = np.random.default_rng(SEED)
    # Power-law-ish endpoints: zipf exponents keep a heavy head so all three
    # degree bins (>=1024, >=32, >=1) are populated in both orientations.
    src = (rng.zipf(1.3, E).astype(np.int64) - 1) % V
    dst = (rng.zipf(1.3, E).astype(np.int64) - 1) % V
    w = rng.uniform(0.1, 2.0, E).astype(np.float32)
    g = mg.Graph.from_edges(src.astype(np.uint32), dst.astype(np.uint32),
                            weights=w, directed=True, num_vertices=V)
    return g


def _both(fn):
    mg.set_execution("cpu")
    try:
        cpu = fn()
        assert mg.last_run_info()["path"] == "cpu"
        mg.set_execution("gpu")
        gpu = fn()
        assert mg.last_run_info()["path"] == "gpu"
    finally:
        mg.set_execution("auto")
    return cpu, gpu


def test_scale_bins_populated(big_graph):
    # The premise of this module: high/mid/low bins all non-empty.
    # (bin sizes aren't exposed; the degree distribution is the proxy)
    counts = np.bincount(
        np.asarray(big_graph._src_ext, dtype=np.int64), minlength=V)
    assert (counts >= 1024).sum() > 0, "no high-bin vertex — fixture too weak"
    assert ((counts >= 32) & (counts < 1024)).sum() > 0
    assert ((counts >= 1) & (counts < 32)).sum() > 0


def test_scale_pagerank_agreement(big_graph):
    cpu, gpu = _both(lambda: np.asarray(
        mg.pagerank(big_graph, alpha=0.85, tol=1e-9, max_iter=64)))
    assert np.isfinite(gpu).all()
    np.testing.assert_allclose(gpu.sum(), 1.0, atol=1e-3)
    np.testing.assert_allclose(gpu, cpu, atol=1e-6, rtol=5e-3)


def test_scale_bfs_agreement(big_graph):
    src = np.asarray([0, 1, 12345], np.uint32)
    for direction in ("out", "in"):
        cpu, gpu = _both(lambda d=direction: np.asarray(
            mg.bfs(big_graph, src, direction=d)[0]))
        np.testing.assert_array_equal(gpu, cpu)


def test_scale_bfs_bottomup_matches_topdown(big_graph):
    # Force both settings through the GPU path; dist must be identical.
    import os
    src = np.asarray([0], np.uint32)
    mg.set_execution("gpu")
    try:
        os.environ["MG_BFS_BOTTOMUP"] = "0"
        d_td = np.asarray(mg.bfs(big_graph, src, direction="out")[0])
        os.environ["MG_BFS_BOTTOMUP"] = "1"
        d_du = np.asarray(mg.bfs(big_graph, src, direction="out")[0])
    finally:
        os.environ.pop("MG_BFS_BOTTOMUP", None)
        mg.set_execution("auto")
    np.testing.assert_array_equal(d_td, d_du)


def test_scale_wcc_agreement(big_graph):
    cpu, gpu = _both(lambda: np.asarray(mg.experimental.wcc(big_graph)))
    np.testing.assert_array_equal(gpu, cpu)


def test_scale_ppr_topk_agreement(big_graph):
    rng = np.random.default_rng(SEED + 1)
    B, k = 16, 64
    seeds = rng.integers(0, V, B * 3).astype(np.uint32)
    weights = np.ones(B * 3, np.float32)
    offsets = np.arange(0, 3 * B + 1, 3, dtype=np.uint64)

    def run():
        ids, scores = mg.ppr_topk(big_graph, seeds, weights, offsets,
                                  k=k, alpha=0.85, tol=1e-8, max_iter=50)
        return np.asarray(ids), np.asarray(scores)

    (ids_c, sc_c), (ids_g, sc_g) = _both(run)
    assert ids_g.shape == (B, k)
    # fp32 vs fp64 paths: scores close; ids may swap only within score ties.
    np.testing.assert_allclose(sc_g, sc_c, atol=1e-6, rtol=5e-3)
    agree = (ids_g == ids_c).mean()
    assert agree > 0.95, f"top-k id agreement too low: {agree:.3f}"


def test_scale_khop_agreement(big_graph):
    seeds = np.asarray([0, 777], np.uint32)

    def run():
        vs, es = mg.k_hop(big_graph, seeds, k=2, direction="out")
        return np.asarray(vs), np.asarray(es)

    (vs_c, es_c), (vs_g, es_g) = _both(run)
    np.testing.assert_array_equal(vs_g, vs_c)
    np.testing.assert_array_equal(es_g, es_c)
