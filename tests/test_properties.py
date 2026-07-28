# test_properties.py — plan §9 property tests.
#
# 1. Worklist invariants: for every fixture graph and BOTH orientations, the
#    per-orientation degree bins (high/mid/low/zero) exposed by the private
#    _core._debug_orientation accessor form a partition of {0..V-1} in
#    canonical id space, membership matches that orientation's degree against
#    the mg_params.h thresholds (1024 / 32 / 1) recomputed INDEPENDENTLY from
#    the input edge arrays, each list is ascending, and the canonical<->user
#    permutations are inverse bijections.
# 2. Repeated-run determinism (gpu): BFS dist and WCC labels must be
#    bit-identical across 25 runs on a ~50k-edge random directed graph —
#    hammers the atomic claim (visited bitmap) and hook (atomic_min label)
#    paths for lost-update bugs. Min-label convergence and BFS levels are
#    order-independent, so ANY run-to-run difference is a real atomics bug.
# SPDX-License-Identifier: Apache-2.0

import os

import numpy as np
import pytest

from conftest import ALL_CASES, CASE_NAMES, build_mg, has_gpu, mg_mod

# Degree-bin thresholds — MUST mirror MG_BIN_HUGE_MIN / MG_BIN_HIGH_MIN /
# MG_BIN_MID_MIN / MG_HUGE_TILE_EDGES in src/kernels/mg_params.h (hardcoded
# on purpose: the test is the contract).
HUGE_MIN = 16384
HIGH_MIN = 1024
MID_MIN = 32
TILE_EDGES = 16384

DEBUG_KEYS = ("wl_high", "wl_mid", "wl_low", "zero_list", "wl_huge",
              "tile_owner", "tile_first_edge", "huge_tile_off", "degrees",
              "canon_of_user", "user_of_canon")


def _core():
    return pytest.importorskip("metal_graph._core")


def _ref_degrees_user(case, direction):
    """Independent per-orientation degree recomputation in USER index space
    from the case's input arrays. direction: 0=out, 1=in.

    Mirrors the documented input policy (src/graph/graph.hpp): duplicate
    edges and self-loops kept; undirected graphs symmetrized at build (each
    u != v edge stored both ways, self-loops stored once), so for undirected
    graphs both orientations carry the symmetrized degree."""
    deg = np.zeros(case.num_vertices, dtype=np.int64)
    s = case.src.astype(np.int64)
    d = case.dst.astype(np.int64)
    if case.directed:
        np.add.at(deg, s if direction == 0 else d, 1)
    else:
        np.add.at(deg, s, 1)
        not_loop = s != d
        np.add.at(deg, d[not_loop], 1)
    return deg


# ---------------------------------------------------------------------------
# Worklist / permutation invariants (full fixture matrix x both orientations;
# the matrix includes the adversarial high-in/high-out case, which crosses
# both bin thresholds in each orientation)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("direction", [0, 1], ids=["out", "in"])
@pytest.mark.parametrize("name", CASE_NAMES)
def test_worklist_partition_invariants(name, direction):
    case = ALL_CASES[name]
    core = _core()
    G = build_mg(case)
    info = core._debug_orientation(G._g, direction)
    v = case.num_vertices

    assert set(info.keys()) == set(DEBUG_KEYS)
    arrs = {k: np.asarray(info[k]) for k in DEBUG_KEYS}
    for k, a in arrs.items():
        assert a.dtype == np.uint32, f"{k}: expected uint32, got {a.dtype}"
        assert a.ndim == 1

    wl_high, wl_mid, wl_low, zero_list = (
        arrs["wl_high"], arrs["wl_mid"], arrs["wl_low"], arrs["zero_list"])
    wl_huge = arrs["wl_huge"]
    canon_of_user = arrs["canon_of_user"]
    user_of_canon = arrs["user_of_canon"]
    degrees = arrs["degrees"].astype(np.int64)  # canonical order
    assert degrees.size == v

    # Each list strictly ascending (=> internally duplicate-free).
    for k in ("wl_high", "wl_mid", "wl_low", "zero_list", "wl_huge"):
        a = arrs[k].astype(np.int64)
        assert np.all(np.diff(a) > 0), f"{k}: not strictly ascending"

    # Disjoint union == {0..V-1} in canonical space.
    union = np.concatenate([wl_huge, wl_high, wl_mid, wl_low, zero_list])
    assert union.size == v, "bins must partition the vertex set (sizes)"
    assert np.array_equal(np.sort(union.astype(np.int64)), np.arange(v)), (
        "bins must be disjoint and cover exactly {0..V-1}")

    # Permutations are inverse bijections.
    assert canon_of_user.size == v and user_of_canon.size == v
    idx = np.arange(v, dtype=np.int64)
    assert np.array_equal(np.sort(canon_of_user.astype(np.int64)), idx)
    assert np.array_equal(np.sort(user_of_canon.astype(np.int64)), idx)
    assert np.array_equal(canon_of_user[user_of_canon].astype(np.int64), idx)
    assert np.array_equal(user_of_canon[canon_of_user].astype(np.int64), idx)

    # Accessor degrees cross-check against the independent recomputation.
    deg_user_ref = _ref_degrees_user(case, direction)
    assert np.array_equal(degrees[canon_of_user], deg_user_ref), (
        "accessor degrees (canonical order) disagree with independent "
        "recomputation from the input edge arrays")

    # Bin membership matches this orientation's INDEPENDENT degrees against
    # the thresholds. Expected lists come from np.where => ascending, so this
    # also re-verifies the ordering rule end-to-end.
    deg_canon = deg_user_ref[user_of_canon.astype(np.int64)]
    assert np.array_equal(wl_huge.astype(np.int64),
                          np.where(deg_canon >= HUGE_MIN)[0])
    assert np.array_equal(wl_high.astype(np.int64),
                          np.where((deg_canon >= HIGH_MIN)
                                   & (deg_canon < HUGE_MIN))[0])
    assert np.array_equal(wl_mid.astype(np.int64),
                          np.where((deg_canon >= MID_MIN)
                                   & (deg_canon < HIGH_MIN))[0])
    assert np.array_equal(wl_low.astype(np.int64),
                          np.where((deg_canon >= 1)
                                   & (deg_canon < MID_MIN))[0])
    assert np.array_equal(zero_list.astype(np.int64),
                          np.where(deg_canon == 0)[0])

    # Huge-bin tile decomposition (mg_params.h tiling section): per huge
    # vertex i, ceil(deg/TILE) contiguous tiles owned by i, tile starts
    # spaced exactly TILE_EDGES apart within the vertex's CSR range.
    t_owner = arrs["tile_owner"].astype(np.int64)
    t_first = arrs["tile_first_edge"].astype(np.int64)
    t_off = arrs["huge_tile_off"].astype(np.int64)
    n_huge = wl_huge.size
    assert t_off.size == n_huge + 1
    assert t_off[0] == 0 and t_off[-1] == t_owner.size == t_first.size
    assert np.all(np.diff(t_off) > 0) if n_huge else True
    for i in range(n_huge):
        lo, hi = t_off[i], t_off[i + 1]
        d = deg_canon[wl_huge[i]]
        assert hi - lo == -(-d // TILE_EDGES), f"huge[{i}]: tile count"
        assert np.all(t_owner[lo:hi] == i)
        assert np.all(np.diff(t_first[lo:hi]) == TILE_EDGES)


def test_adversarial_case_populates_every_bin():
    """Guard the guard: the adversarial fixture must actually exercise all
    four bins in BOTH orientations, or the invariant test above is toothless
    at the thresholds."""
    core = _core()
    G = build_mg(ALL_CASES["adversarial"])
    for direction in (0, 1):
        info = core._debug_orientation(G._g, direction)
        for k in ("wl_high", "wl_mid", "wl_low", "zero_list"):
            assert np.asarray(info[k]).size > 0, (
                f"adversarial case: empty {k} for direction={direction}")


def test_debug_orientation_returns_fresh_copies():
    """Two calls must return distinct arrays (copies, not aliases of the
    shared MTLBuffers); mutating one must not corrupt the next read."""
    core = _core()
    G = build_mg(ALL_CASES["gnp_dir"])
    a = core._debug_orientation(G._g, 0)
    b = core._debug_orientation(G._g, 0)
    for k in DEBUG_KEYS:
        aa, bb = np.asarray(a[k]), np.asarray(b[k])
        if aa.size == 0:
            continue
        assert aa.__array_interface__["data"][0] != \
            bb.__array_interface__["data"][0], f"{k}: aliased storage"
        orig = bb.copy()
        aa += np.uint32(1)  # vandalize copy A
        assert np.array_equal(bb, orig)
    c = core._debug_orientation(G._g, 0)
    for k in DEBUG_KEYS:
        assert np.array_equal(np.asarray(b[k]), np.asarray(c[k]))


def _megastar():
    """Bidirectional star: hub 0 <-> 40k spokes. Hub degree 40 000 >= HUGE_MIN
    in BOTH orientations => the huge bin and its 3-tile decomposition are
    exercised by PageRank (IN) and WCC hook (OUT)."""
    mg = mg_mod()
    n = 40_000
    spokes = np.arange(1, n + 1, dtype=np.uint32)
    hub = np.zeros(n, np.uint32)
    src = np.concatenate([spokes, hub])
    dst = np.concatenate([hub, spokes])
    return mg.Graph.from_edges(src, dst, directed=True, num_vertices=n + 1)


def test_megastar_populates_huge_bin():
    core = _core()
    G = _megastar()
    for direction in (0, 1):
        info = core._debug_orientation(G._g, direction)
        wl_huge = np.asarray(info["wl_huge"])
        assert wl_huge.size == 1, f"direction={direction}: hub not in huge bin"
        n_tiles = np.asarray(info["tile_owner"]).size
        assert n_tiles == -(-40_000 // TILE_EDGES)  # ceil = 3


@pytest.mark.gpu
def test_megastar_huge_bin_gpu_matches_cpu():
    """Numeric agreement through the tiled two-pass gather / tiled hook."""
    mg = mg_mod()
    if not has_gpu():
        pytest.skip("no Metal device")
    G = _megastar()

    def run_all():
        pr = np.asarray(mg.pagerank(G, alpha=0.85, tol=1e-10, max_iter=100))
        wcc = np.asarray(mg.experimental.wcc(G))
        seeds = np.asarray([123], np.uint32)
        w = np.asarray([1.0], np.float32)
        off = np.asarray([0, 1], np.uint64)
        ids, scores = mg.ppr_topk(G, seeds, w, off, k=8, alpha=0.85,
                                  tol=1e-10, max_iter=50)
        return pr, wcc, np.asarray(ids), np.asarray(scores)

    try:
        mg.set_execution("cpu")
        pr_c, wcc_c, ids_c, sc_c = run_all()
        mg.set_execution("gpu")
        pr_g, wcc_g, ids_g, sc_g = run_all()
        assert mg.last_run_info()["path"] == "gpu"
    finally:
        mg.set_execution("auto")

    np.testing.assert_allclose(pr_g, pr_c, atol=1e-7, rtol=1e-3)
    np.testing.assert_allclose(pr_g.sum(), 1.0, atol=1e-4)
    np.testing.assert_array_equal(wcc_g, wcc_c)
    # The seed spoke and the hub dominate every other (symmetric) spoke.
    assert set(ids_g[0, :2].tolist()) == {123, 0}
    assert set(ids_c[0, :2].tolist()) == {123, 0}
    np.testing.assert_allclose(sc_g, sc_c, atol=1e-7, rtol=1e-3)


def test_debug_orientation_rejects_bad_direction():
    core = _core()
    G = build_mg(ALL_CASES["ring"])
    for bad in (-1, 2, 3):
        with pytest.raises(ValueError):
            core._debug_orientation(G._g, bad)


# ---------------------------------------------------------------------------
# Repeated-run determinism / atomics sanity (gpu path)
# ---------------------------------------------------------------------------

SEED = 20260728
RAND_V = 20_000
RAND_E = 50_000
N_RUNS = 25


@pytest.fixture(scope="module")
def rand_graph():
    """Seeded ~50k-edge random directed graph (duplicates/self-loops kept
    by policy — fine, they only add contention on the atomic paths)."""
    mg = mg_mod()
    rng = np.random.default_rng(SEED)
    src = rng.integers(0, RAND_V, RAND_E).astype(np.uint32)
    dst = rng.integers(0, RAND_V, RAND_E).astype(np.uint32)
    return mg.Graph.from_edges(src, dst, directed=True,
                               num_vertices=RAND_V)


@pytest.fixture
def gpu_exec():
    mg = mg_mod()
    if not has_gpu():
        pytest.skip("no Metal device")
    mg.set_execution("gpu")
    yield mg
    mg.set_execution("auto")


@pytest.mark.gpu
@pytest.mark.parametrize("bottomup", ["0", "1"])
def test_bfs_repeated_runs_bit_identical(rand_graph, gpu_exec, bottomup):
    """BFS dist is a pure function of the graph — any run-to-run wobble on
    the GPU path is a lost update in the visited-bitmap claim / frontier
    append / level protocol. Exercised with the direction-optimizing
    bottom-up switch both disabled and enabled."""
    mg = gpu_exec
    sources = np.asarray([0, 7, 12345], np.uint32)
    saved = os.environ.get("MG_BFS_BOTTOMUP")
    os.environ["MG_BFS_BOTTOMUP"] = bottomup
    try:
        # Independent oracle: the CPU path on the same graph.
        mg.set_execution("cpu")
        dist_cpu = np.asarray(mg.bfs(rand_graph, sources,
                                     direction="out")[0])
        mg.set_execution("gpu")

        base = np.asarray(mg.bfs(rand_graph, sources, direction="out")[0])
        assert mg.last_run_info()["path"] == "gpu"
        np.testing.assert_array_equal(base, dist_cpu)
        for run in range(1, N_RUNS):
            dist = np.asarray(mg.bfs(rand_graph, sources,
                                     direction="out")[0])
            assert np.array_equal(dist, base), (
                f"bfs dist diverged on run {run} "
                f"(MG_BFS_BOTTOMUP={bottomup}): "
                f"{int(np.count_nonzero(dist != base))} mismatches")
    finally:
        if saved is None:
            os.environ.pop("MG_BFS_BOTTOMUP", None)
        else:
            os.environ["MG_BFS_BOTTOMUP"] = saved


@pytest.mark.gpu
def test_wcc_repeated_runs_identical(rand_graph, gpu_exec):
    """WCC min-label convergence is order-independent, so the converged
    labels are unique regardless of hook/jump scheduling — 25 runs hammer
    the atomic_min hook paths; any difference is a lost-update bug."""
    mg = gpu_exec
    mg.set_execution("cpu")
    labels_cpu = np.asarray(mg.experimental.wcc(rand_graph))
    mg.set_execution("gpu")

    base = np.asarray(mg.experimental.wcc(rand_graph))
    assert mg.last_run_info()["path"] == "gpu"
    np.testing.assert_array_equal(base, labels_cpu)
    for run in range(1, N_RUNS):
        labels = np.asarray(mg.experimental.wcc(rand_graph))
        assert np.array_equal(labels, base), (
            f"wcc labels diverged on run {run}: "
            f"{int(np.count_nonzero(labels != base))} mismatches")
