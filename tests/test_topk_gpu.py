# test_topk_gpu.py — GPU radix-select vs the CPU selection oracle.
#
# The GPU select must be BIT-IDENTICAL to the CPU selection over the same
# tile state: both read the same fp32 scores and apply the same
# (score desc, user index asc) tie rule, so ids AND scores must match
# exactly (np.array_equal, not allclose). MG_GPU_TOPK=0 forces the oracle
# on the same gpu execution path.
# SPDX-License-Identifier: Apache-2.0

import os

import numpy as np
import pytest

mg = pytest.importorskip("metal_graph")

from conftest import has_gpu

pytestmark = pytest.mark.gpu

SEED = 20260728


@pytest.fixture
def gpu_exec():
    if not has_gpu():
        pytest.skip("no Metal device")
    mg.set_execution("gpu")
    yield
    mg.set_execution("auto")


def _run_both(g, seeds, weights, offsets, k):
    saved = os.environ.get("MG_GPU_TOPK")
    try:
        os.environ["MG_GPU_TOPK"] = "1"
        ids_g, sc_g = mg.ppr_topk(g, seeds, weights, offsets, k=k,
                                  alpha=0.85, tol=1e-8, max_iter=50)
        assert mg.last_run_info()["path"] == "gpu"
        os.environ["MG_GPU_TOPK"] = "0"
        ids_c, sc_c = mg.ppr_topk(g, seeds, weights, offsets, k=k,
                                  alpha=0.85, tol=1e-8, max_iter=50)
        assert mg.last_run_info()["path"] == "gpu"
    finally:
        if saved is None:
            os.environ.pop("MG_GPU_TOPK", None)
        else:
            os.environ["MG_GPU_TOPK"] = saved
    return (np.asarray(ids_g), np.asarray(sc_g),
            np.asarray(ids_c), np.asarray(sc_c))


@pytest.mark.parametrize("k", [1, 64, 1024])
def test_gpu_select_bit_identical_random(gpu_exec, k):
    rng = np.random.default_rng(SEED)
    v, e = 20_000, 120_000
    src = rng.integers(0, v, e).astype(np.uint32)
    dst = rng.integers(0, v, e).astype(np.uint32)
    w = rng.uniform(0.1, 2.0, e).astype(np.float32)
    g = mg.Graph.from_edges(src, dst, weights=w, directed=True,
                            num_vertices=v)
    B = 20  # multi-tile (3 tiles, last partial)
    seeds = rng.integers(0, v, B * 2).astype(np.uint32)
    weights = rng.uniform(0.5, 2.0, B * 2).astype(np.float32)
    offsets = np.arange(0, 2 * B + 1, 2, dtype=np.uint64)

    ids_g, sc_g, ids_c, sc_c = _run_both(g, seeds, weights, offsets, k)
    np.testing.assert_array_equal(ids_g, ids_c)
    np.testing.assert_array_equal(sc_g, sc_c)


def test_gpu_select_tie_flood_falls_back_correctly(gpu_exec):
    # Bidirectional megastar: every non-seed spoke ties exactly => the
    # >=-threshold class exceeds cap and the lane must take the in-place
    # CPU-oracle fallback — output still bit-identical.
    n = 40_000
    spokes = np.arange(1, n + 1, dtype=np.uint32)
    hub = np.zeros(n, np.uint32)
    g = mg.Graph.from_edges(np.concatenate([spokes, hub]),
                            np.concatenate([hub, spokes]),
                            directed=True, num_vertices=n + 1)
    seeds = np.asarray([123, 77, 9001], np.uint32)
    weights = np.ones(3, np.float32)
    offsets = np.asarray([0, 1, 2, 3], np.uint64)

    ids_g, sc_g, ids_c, sc_c = _run_both(g, seeds, weights, offsets, k=64)
    np.testing.assert_array_equal(ids_g, ids_c)
    np.testing.assert_array_equal(sc_g, sc_c)
    # tie rule visible: after the seed and hub, spokes in ascending order
    row = ids_g[0]
    assert row[0] in (123, 0) and row[1] in (123, 0)
    tail = row[2:]
    assert np.all(np.diff(tail) > 0), "tie class must be ascending user ids"


def test_gpu_select_ineligible_shapes_use_oracle(gpu_exec):
    # V < 4096 and k >= V shapes must run the oracle (still gpu path) and
    # stay correct: compare against the cpu execution path end-to-end.
    rng = np.random.default_rng(SEED + 1)
    v, e = 500, 4_000
    src = rng.integers(0, v, e).astype(np.uint32)
    dst = rng.integers(0, v, e).astype(np.uint32)
    g = mg.Graph.from_edges(src, dst, directed=True, num_vertices=v)
    seeds = np.asarray([1, 2, 3], np.uint32)
    weights = np.ones(3, np.float32)
    offsets = np.asarray([0, 1, 2, 3], np.uint64)
    for k in (64, v, v + 10):  # k >= V exercises -1/0.0 padding too
        ids_g, sc_g, ids_c, sc_c = _run_both(g, seeds, weights, offsets, k)
        np.testing.assert_array_equal(ids_g, ids_c)
        np.testing.assert_array_equal(sc_g, sc_c)


def test_gpu_select_zero_threshold_tie_flood_and_isolated_seeds(gpu_exec):
    # Mix normal queries, small-reach zero-threshold queries (tie flood at 0.0
    # with V >> cap), and isolated-vertex seeds in the same 8-lane tile so
    # early-stop lanes, strict-greater tie-flood lanes, and isolated-seed
    # copy-through/dangling paths all execute together and stay bit-identical
    # to the CPU oracle.
    rng = np.random.default_rng(SEED + 2)
    v, core = 12_000, 1_500
    src = rng.integers(0, core, 24_000, dtype=np.uint32)
    dst = rng.integers(0, core, 24_000, dtype=np.uint32)
    # Small disconnected component of 5 vertices at [core, core+5)
    c_src = np.arange(core, core + 4, dtype=np.uint32)
    c_dst = np.arange(core + 1, core + 5, dtype=np.uint32)
    g = mg.Graph.from_edges(np.concatenate([src, c_src]),
                            np.concatenate([dst, c_dst]),
                            directed=True, num_vertices=v)
    # Lanes: 0,1 core (normal); 2,3 tiny component (0.0 tie flood);
    # 4,5 pure isolated seeds (in_deg==0 && out_deg==0);
    # 6 mixed core + isolated seed; 7 core (normal)
    queries = [
        ([0, 1], [1.0, 2.0]),
        ([5], [1.0]),
        ([core], [1.0]),
        ([core + 1, core + 2], [0.5, 1.5]),
        ([v - 1], [1.0]),
        ([v - 10, v - 20], [1.0, 3.0]),
        ([10, v - 5], [2.0, 1.0]),
        ([25], [1.0]),
    ]
    seeds = np.concatenate([np.asarray(s, np.uint32) for s, _ in queries])
    weights = np.concatenate([np.asarray(w, np.float32) for _, w in queries])
    offsets = np.zeros(len(queries) + 1, np.uint64)
    np.cumsum([len(s) for s, _ in queries], out=offsets[1:])

    ids_g, sc_g, ids_c, sc_c = _run_both(g, seeds, weights, offsets, k=64)
    np.testing.assert_array_equal(ids_g, ids_c)
    np.testing.assert_array_equal(sc_g, sc_c)
    # Pure isolated seed lane must put all mass (1.0) on its seed vertex.
    assert ids_g[4, 0] == v - 1
    assert sc_g[4, 0] == pytest.approx(1.0, abs=1e-5)
    assert sc_g[4, 1] == 0.0


@pytest.mark.parametrize("scale", [1e-20, 1e20], ids=["tiny_weights", "huge_weights"])
def test_gpu_ppr_extreme_weights_fallback(gpu_exec, scale):
    # Exercises !ows_safe_rescale in batched GPU PPR + radix select.
    rng = np.random.default_rng(SEED + 3)
    v, e = 5_000, 30_000
    src = rng.integers(0, v, e, dtype=np.uint32)
    dst = rng.integers(0, v, e, dtype=np.uint32)
    w = (rng.uniform(0.5, 2.0, e) * scale).astype(np.float32)
    g = mg.Graph.from_edges(src, dst, weights=w, directed=True, num_vertices=v)
    seeds = np.arange(0, 10, dtype=np.uint32)
    weights = np.ones(10, np.float32)
    offsets = np.arange(0, 11, dtype=np.uint64)
    ids_g, sc_g, ids_c, sc_c = _run_both(g, seeds, weights, offsets, k=32)
    np.testing.assert_array_equal(ids_g, ids_c)
    np.testing.assert_array_equal(sc_g, sc_c)
    assert np.isfinite(sc_g).all()
