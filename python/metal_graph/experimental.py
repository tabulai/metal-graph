# metal_graph.experimental — experimental-tier algorithms (v0.1: WCC, the
# atomics canary).
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from . import _core


def _core_graph(G):
    core = getattr(G, "_g", None)
    if core is None:
        raise TypeError("G: expected a metal_graph.Graph")
    return core


def wcc(G):
    """Weakly connected components over the stored edges.

    Returns an int32 array of length num_vertices: comp[user_idx] =
    component id, numbered by first occurrence in user-index order
    (canonical-partition semantics). Directed edges are treated as
    undirected.
    """
    comp, _n = _core.wcc(_core_graph(G))
    return comp


def wcc_count(G):
    """Number of weakly connected components."""
    _comp, n = _core.wcc(_core_graph(G))
    return int(n)
