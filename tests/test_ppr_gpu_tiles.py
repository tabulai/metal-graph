# test_ppr_gpu_tiles.py — GPU ppr_topk tile grouping and radix select.
#
# A query's GPU result must not depend on which tile (or lane) it runs in:
# mixed sparse/GPU batches regroup the GPU rows, and every row must still be
# bit-identical to the same query run alone. The radix select must match the
# CPU selection oracle bit for bit, including when the k-th largest score is
# an exact zero.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from conftest import has_gpu

mg = pytest.importorskip("metal_graph")

pytestmark = pytest.mark.gpu

PPR_KW = dict(alpha=0.85, tol=1e-9, max_iter=200)


@pytest.fixture
def gpu_env(monkeypatch):
    if not has_gpu():
        pytest.skip("no Metal device")
    for name in ("MG_E_GPU_MIN", "MG_E_GPU_MIN_PPR", "MG_GPU_TOPK",
                 "MG_PPR_SPARSE_MAX_VERTICES", "MG_PPR_SPARSE_MAX_EDGES"):
        monkeypatch.delenv(name, raising=False)
    yield monkeypatch
    mg.set_execution("auto")


def _pack(queries):
    seeds = np.concatenate([np.asarray(s, np.uint32) for s, _ in queries])
    w = np.concatenate([np.asarray(x, np.float32) for _, x in queries])
    offs = np.zeros(len(queries) + 1, np.uint64)
    np.cumsum([len(s) for s, _ in queries], out=offs[1:])
    return seeds, w, offs


def _run(g, queries, k):
    seeds, w, offs = _pack(queries)
    ids, scores = mg.ppr_topk(g, seeds, w, offs, k=k, **PPR_KW)
    info = mg.last_run_info()
    return np.asarray(ids).copy(), np.asarray(scores).copy(), info


def _core_and_chains(n_vertices, seed=11):
    """Random core (large reach) plus short dangling chains (small reach),
    padded with isolated vertices to n_vertices."""
    rng = np.random.default_rng(seed)
    core = 300
    src = [rng.integers(0, core, 4000)]
    dst = [rng.integers(0, core, 4000)]
    nxt = core
    heads = []
    for length in (1, 2, 3, 5, 8) * 4:
        chain = np.arange(nxt, nxt + length + 1)
        src.append(chain[:-1])
        dst.append(chain[1:])
        heads.append(nxt)
        nxt += length + 1
    g = mg.Graph.from_edges(np.concatenate(src).astype(np.uint32),
                            np.concatenate(dst).astype(np.uint32),
                            directed=True, num_vertices=n_vertices)
    return g, core, heads


# (batch size, period, GPU rows per period): 4 + 4 + 3 GPU rows pack whole
# into two GPU tiles; 5 + 5 + 5 would need three, so they are compacted into
# two full tiles instead, splitting the middle all-GPU tile.
@pytest.mark.parametrize("layout", [(21, 2, 1), (24, 8, 5)],
                         ids=["whole_tiles", "compacted"])
@pytest.mark.parametrize("n_vertices", [1000, 6000])  # oracle / GPU select
def test_mixed_batch_rows_match_single_queries(gpu_env, n_vertices, layout):
    g, core, heads = _core_and_chains(n_vertices)
    rng = np.random.default_rng(3)
    n_queries, period, n_dense = layout
    queries = []
    for q in range(n_queries):
        if q % period >= n_dense:
            s = [heads[q % len(heads)]]
        else:
            s = rng.integers(0, core, 2)
        queries.append((s, rng.uniform(0.5, 1.5, len(s))))
    k = 16
    mg.set_execution("auto")
    gpu_env.setenv("MG_E_GPU_MIN", "0")
    gpu_env.setenv("MG_PPR_SPARSE_MAX_VERTICES", "16")
    gpu_env.setenv("MG_PPR_SPARSE_MAX_EDGES", "16")
    ids, scores, info = _run(g, queries, k)
    assert info["path"] == "gpu"
    paths, iters = [], []
    for q, query in enumerate(queries):
        ids_q, scores_q, info_q = _run(g, [query], k)
        np.testing.assert_array_equal(ids_q[0], ids[q])
        assert scores_q[0].tobytes() == scores[q].tobytes()
        paths.append(info_q["path"])
        iters.append(info_q["iterations"])
    assert "cpu" in paths and "gpu" in paths  # the batch really was mixed
    assert info["iterations"] == max(iters)


@pytest.mark.parametrize("k", [2100, 3000])
def test_gpu_select_threshold_at_exact_zero(gpu_env, k):
    # Every query reaches a handful of vertices, so the k-th largest score
    # is an exact 0.0 and, with cap = k + MG_TOPK_TIE_SLACK >= V, the GPU
    # select (not the overflow fallback) must produce the oracle's rows:
    # the nonzero scores, then zero-score vertices by ascending user id.
    g, _, heads = _core_and_chains(4096)
    seeds = np.asarray(heads[:5], np.uint32)
    weights = np.ones(5, np.float32)
    offsets = np.arange(6, dtype=np.uint64)
    mg.set_execution("gpu")
    out = {}
    for flag in ("1", "0"):
        gpu_env.setenv("MG_GPU_TOPK", flag)
        out[flag] = mg.ppr_topk(g, seeds, weights, offsets, k=k, **PPR_KW)
    np.testing.assert_array_equal(out["1"][0], out["0"][0])
    scores = [np.asarray(out[flag][1]).tobytes() for flag in ("1", "0")]
    assert scores[0] == scores[1]
    assert np.count_nonzero(np.asarray(out["1"][1])[0]) < 16
