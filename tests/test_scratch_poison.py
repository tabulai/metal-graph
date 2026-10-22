# test_scratch_poison.py — GPU scratch buffers are written before any read.
#
# Runtime::alloc_scratch skips the zero-fill, so a kernel that read an
# element nothing wrote would see whatever the fresh pages hold (almost
# always zeros, which would hide the bug). MG_DEBUG_POISON_SCRATCH=1 fills
# those buffers with 0xFF bytes (NaN as floats); every GPU PageRank, HITS,
# ppr_topk and WCC output and its run info must be byte-identical either way.
# Subprocess-isolated: the flag is read once at Runtime construction.
# SPDX-License-Identifier: Apache-2.0
import pytest

from conftest import has_gpu, run_python

pytest.importorskip("metal_graph")

pytestmark = pytest.mark.gpu

CODE = r"""
import hashlib
import numpy as np
import metal_graph as mg

h = hashlib.sha1()

def put(*arrays):
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    info = mg.last_run_info()
    h.update(repr(sorted((k, str(v)) for k, v in info.items()
                         if k != "ms")).encode())

mg.set_execution("gpu")
rng = np.random.default_rng(5)
for n, m, directed in ((57, 300, True), (5007, 40000, False),
                       (70007, 300000, True)):
    src = rng.integers(0, n // 2, m).astype(np.uint32)  # upper half dangles
    dst = rng.integers(0, n, m).astype(np.uint32)
    w = rng.uniform(0.0, 2.0, m).astype(np.float32)
    w[::7] = 0.0
    for weights in (None, w):
        g = mg.Graph.from_edges(src, dst, directed=directed, num_vertices=n,
                                weights=weights)
        put(mg.experimental.wcc(g))  # giant set plus isolated upper ids
        for max_iter in (1, 2, 100):
            put(mg.pagerank(g, alpha=0.85, tol=1e-9, max_iter=max_iter))
            put(*mg.hits(g, tol=1e-9, max_iter=max_iter))
        pers = rng.uniform(0.0, 1.0, n)
        pers[rng.integers(0, n, n // 3)] = 0.0
        put(mg.pagerank(g, alpha=0.85, tol=1e-9, max_iter=100,
                        personalization=pers))
        for B in (1, 11):
            sizes = rng.integers(1, 4, B)
            offs = np.zeros(B + 1, np.uint64)
            np.cumsum(sizes, out=offs[1:])
            seeds = rng.integers(0, n, int(offs[-1])).astype(np.uint32)
            sw = rng.uniform(0.5, 1.5, int(offs[-1])).astype(np.float32)
            for k in (8, 64):
                put(*mg.ppr_topk(g, seeds, sw, offs, k=k, alpha=0.85,
                                 tol=1e-9, max_iter=200))
print("DIGEST", h.hexdigest())
"""


def _digest(poison):
    r = run_python(CODE, {"MG_DEBUG_POISON_SCRATCH": poison,
                          "MG_REQUIRE_GPU": "1"}, timeout=600)
    assert r.returncode == 0, r.stderr
    line = [s for s in r.stdout.splitlines() if s.startswith("DIGEST")]
    assert line, r.stdout
    return line[0]


def test_poisoned_scratch_gives_identical_outputs():
    if not has_gpu():
        pytest.skip("no Metal device")
    assert _digest("1") == _digest("0")
