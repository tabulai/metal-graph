# test_wcc.py — weakly connected components (experimental tier).
import networkx as nx
import numpy as np
import pytest

mg = pytest.importorskip("metal_graph")

from conftest import ALL_CASES, build_mg, build_nx_multi


def _partition_from_labels(comp):
    parts = {}
    for u, c in enumerate(np.asarray(comp).tolist()):
        parts.setdefault(c, set()).add(u)
    return frozenset(frozenset(s) for s in parts.values())


def _nx_partition(case):
    g = build_nx_multi(case)
    if case.directed:
        comps = nx.weakly_connected_components(g)
    else:
        comps = nx.connected_components(g)
    return frozenset(frozenset(c) for c in comps)


def test_wcc_partition_matches_networkx(graph_case, exec_path, assert_path):
    case = graph_case
    g = build_mg(case)
    comp = np.asarray(mg.experimental.wcc(g))
    assert comp.shape == (case.num_vertices,)
    if case.num_vertices == 0:
        return
    assert_path(exec_path)
    assert _partition_from_labels(comp) == _nx_partition(case)


def test_wcc_canonical_numbering(graph_case, exec_path, assert_path):
    # Component ids are numbered by first occurrence in USER index order:
    # walking user indices 0..V-1, each new id must be the next integer.
    case = graph_case
    g = build_mg(case)
    comp = np.asarray(mg.experimental.wcc(g))
    if case.num_vertices == 0:
        return
    assert_path(exec_path)
    seen = []
    for u in range(case.num_vertices):
        c = int(comp[u])
        if c not in seen:
            seen.append(c)
    assert seen == list(range(len(seen))), (
        f"component ids not canonical (first-occurrence order): {seen[:10]}")


def test_wcc_component_count(exec_path, assert_path):
    # forest2: two trees; isolated: one 40-cycle component + 20 singletons.
    for name, expected in [("forest2", 2), ("isolated", 21),
                           ("singleton", 1)]:
        case = ALL_CASES[name]
        g = build_mg(case)
        comp = np.asarray(mg.experimental.wcc(g))
        assert_path(exec_path)
        assert len(np.unique(comp)) == expected, name


def test_wcc_deterministic(exec_path, assert_path):
    case = ALL_CASES["gnp_dir"]
    g = build_mg(case)
    a = np.asarray(mg.experimental.wcc(g))
    assert_path(exec_path)
    b = np.asarray(mg.experimental.wcc(g))
    np.testing.assert_array_equal(a, b)
