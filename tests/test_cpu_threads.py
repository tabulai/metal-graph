# test_cpu_threads.py — CPU loops run on a persistent worker pool: results
# must not depend on MG_THREADS, and callers on several Python threads at
# once (the bindings release the GIL; all but one fall back to fresh
# threads) or in a forked child (the workers do not survive fork) must get
# the same answers as sequential calls.
import os
import threading

import numpy as np
import pytest

from conftest import run_python

mg = pytest.importorskip("metal_graph")


@pytest.fixture
def cpu_mode():
    mg.set_execution("cpu")
    yield
    mg.set_execution("auto")


def _graph(directed, weighted, scale=13, seed=4):
    # Skewed sources (hubs) and enough edges that every CPU loop threads:
    # many gather items, multi-chunk frontiers, many union-find pieces.
    rng = np.random.default_rng(seed)
    v = 1 << scale
    e = 16 * v
    src = (v * rng.random(e) ** 3).astype(np.uint32)
    dst = rng.integers(0, v, e).astype(np.uint32)
    w = rng.uniform(0.5, 2.0, e).astype(np.float32) if weighted else None
    g = mg.Graph.from_edges(src, dst, weights=w, directed=directed,
                            num_vertices=v)
    return g, int(np.argmax(np.bincount(src, minlength=v)))


def _outputs(g, hub):
    seeds = np.array([hub, 1, 2, 3, 4, 5, 6, 7], np.uint32)
    offsets = np.array([0, 1, 3, 5, 8], np.uint64)
    out = {"pagerank": mg.pagerank(g), "hits": np.concatenate(mg.hits(g))}
    ids, scores = mg.ppr_topk(g, seeds, np.ones(seeds.size, np.float32),
                              offsets, k=32)
    out["ppr_ids"], out["ppr_scores"] = ids, scores
    out["bfs"] = mg.bfs(g, [hub])[0]
    out["bfs_both"] = mg.bfs(g, [hub], direction="both")[0]
    vs, es = mg.k_hop(g, [hub], k=2, direction="both")
    out["khop_v"], out["khop_e"] = vs, es
    out["wcc"] = mg.experimental.wcc(g)
    return {k: np.asarray(x).copy() for k, x in out.items()}


def _assert_same(a, b):
    assert a.keys() == b.keys()
    for k in a:
        np.testing.assert_array_equal(a[k], b[k], err_msg=k)


@pytest.mark.parametrize("directed,weighted",
                         [(True, False), (True, True), (False, False)])
def test_results_independent_of_thread_count(cpu_mode, monkeypatch,
                                             directed, weighted):
    monkeypatch.setenv("MG_THREADS", "1")
    g, hub = _graph(directed, weighted)
    ref = _outputs(g, hub)
    for threads in ("2", "3", "16"):
        monkeypatch.setenv("MG_THREADS", threads)
        g2, _ = _graph(directed, weighted)
        _assert_same(_outputs(g2, hub), ref)
        _assert_same(_outputs(g, hub), ref)


def test_concurrent_callers(cpu_mode):
    g, hub = _graph(True, False)
    ref = _outputs(g, hub)
    errors = []

    def worker():
        try:
            for _ in range(3):
                _assert_same(_outputs(g, hub), ref)
        except BaseException as exc:  # reported on the main thread
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors[0]


FORK_CODE = """
import os, signal, sys, time, warnings
sys.path.insert(0, os.path.abspath("tests"))  # run_python's cwd: repo root
import metal_graph as mg
import test_cpu_threads as t

mg.set_execution("cpu")
g, hub = t._graph(True, False)
ref = t._outputs(g, hub)  # the parent's loops start the pool workers
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)  # multi-threaded fork
    pid = os.fork()
if pid == 0:
    code = 3
    try:
        g2, _ = t._graph(True, False)
        t._assert_same(t._outputs(g, hub), ref)
        t._assert_same(t._outputs(g2, hub), ref)
        code = 0
    except BaseException:
        import traceback
        traceback.print_exc()
        sys.stderr.flush()
    finally:
        os._exit(code)
deadline = time.monotonic() + 120
while True:
    done, status = os.waitpid(pid, os.WNOHANG)
    if done:
        break
    if time.monotonic() > deadline:
        os.kill(pid, signal.SIGKILL)
        os.waitpid(pid, 0)
        print("CHILD_HUNG")
        sys.exit(1)
    time.sleep(0.01)
t._assert_same(t._outputs(g, hub), ref)  # the parent's pool still works
print("CHILD_EXIT", os.waitstatus_to_exitcode(status))
"""


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
def test_forked_child():
    # MG_FORCE_CPU=1 keeps Metal out of the process, so the fork is safe.
    r = run_python(FORK_CODE, {"MG_FORCE_CPU": "1", "MG_REQUIRE_GPU": "0"})
    assert r.returncode == 0 and "CHILD_EXIT 0" in r.stdout, (
        f"rc={r.returncode}\n{r.stdout}\n{r.stderr}")
