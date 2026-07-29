# metal_graph — Metal-native graph analytics for Apple Silicon (v0.1).
#
# Public API per README: Graph.from_edges, pagerank, ppr_topk, bfs, k_hop,
# experimental.wcc, set_execution, last_run_info. All algorithm inputs and
# outputs use dense USER indices (0..V-1); external-ID mapping lives here.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

try:
    from . import _core
except ImportError as _exc:
    raise ImportError(
        "metal_graph._core extension module is missing; build it with cmake "
        "(cmake -S . -B build && cmake --build build -j) — see the README "
        "quick start."
    ) from _exc

from ._ids import factorize, index_in_sorted

__version__ = "0.1.0"

__all__ = [
    "Graph",
    "pagerank",
    "ppr_topk",
    "bfs",
    "k_hop",
    "set_execution",
    "last_run_info",
    "has_gpu",
    "experimental",
    "__version__",
]

_DIRECTIONS = {"out": 0, "in": 1, "both": 2}
_EXEC_MODES = {"auto": 0, "gpu": 1, "cpu": 2}
_MAX_V = 2**31 - 1  # int32 outputs (ids, parents, comp) cap V in v0.1


def _reserved(name, value):
    if value is not None:
        raise NotImplementedError(
            f"{name}= requires the edge-property/mask subsystem planned for "
            "v0.2"
        )


def _direction_code(direction):
    if isinstance(direction, str):
        code = _DIRECTIONS.get(direction)
        if code is None:
            raise ValueError("direction: must be 'out', 'in' or 'both'")
        return code
    if direction in (0, 1, 2):
        return int(direction)
    raise ValueError("direction: must be 'out', 'in' or 'both'")


def _as_index_array(a, n_vertices, what):
    """Scalar or sequence of user indices -> validated contiguous uint32."""
    arr = np.atleast_1d(np.asarray(a))
    if arr.ndim != 1:
        raise ValueError(f"{what}: must be a scalar or 1-D array")
    if arr.size:
        if not np.issubdtype(arr.dtype, np.integer):
            raise ValueError(f"{what}: integer user indices required")
        if int(arr.min()) < 0 or int(arr.max()) >= n_vertices:
            raise ValueError(
                f"{what}: user index out of range [0, {n_vertices})"
            )
    return np.ascontiguousarray(arr, dtype=np.uint32)


def _core_graph(G):
    if not isinstance(G, Graph):
        raise TypeError("G: expected a metal_graph.Graph")
    return G._g


class Graph:
    """Immutable graph snapshot with external-ID mapping.

    Construct via Graph.from_edges. External IDs (any ints or strings) map
    to dense user indices 0..V-1 in np.unique order; when num_vertices= is
    given the IDs must already be dense integers (identity mapping) and
    isolated tail vertices are allowed.
    """

    def __init__(self, *args, **kwargs):
        raise TypeError("use Graph.from_edges(...) to construct a Graph")

    @classmethod
    def _wrap(cls, core, external_ids, identity, src_ext, dst_ext, weights):
        self = object.__new__(cls)
        self._g = core
        self._external_ids = external_ids
        self._identity = identity
        # Kept for k_hop(as_graph=True): input edge id -> external endpoints.
        self._src_ext = src_ext
        self._dst_ext = dst_ext
        self._weights = weights
        return self

    @classmethod
    def from_edges(cls, src, dst, weights=None, directed=True,
                   num_vertices=None, ids="auto"):
        if ids != "auto":
            raise ValueError("ids: only 'auto' is supported in v0.1")
        src = np.asarray(src)
        dst = np.asarray(dst)
        if src.ndim != 1 or dst.ndim != 1:
            raise ValueError("src/dst: must be 1-D arrays")
        if src.shape[0] != dst.shape[0]:
            raise ValueError("src/dst: length mismatch")
        w = None
        if weights is not None:
            w = np.ascontiguousarray(weights, dtype=np.float32)
            if w.ndim != 1 or w.shape[0] != src.shape[0]:
                raise ValueError(
                    "weights: must be 1-D of the same length as src/dst"
                )
        if num_vertices is not None:
            num_vertices = int(num_vertices)
            if num_vertices < 0 or num_vertices > _MAX_V:
                raise ValueError(
                    "num_vertices: must be in [0, 2**31) (int32 id limit)")
            for name, a in (("src", src), ("dst", dst)):
                if a.size and not np.issubdtype(a.dtype, np.integer):
                    raise ValueError(
                        f"{name}: integer ids required when num_vertices is "
                        "given"
                    )
            if src.size:
                lo = min(int(src.min()), int(dst.min()))
                hi = max(int(src.max()), int(dst.max()))
                if lo < 0 or hi >= num_vertices:
                    raise ValueError(
                        "src/dst: vertex ids must be in [0, num_vertices)"
                    )
            src_idx = np.ascontiguousarray(src, dtype=np.uint32)
            dst_idx = np.ascontiguousarray(dst, dtype=np.uint32)
            external_ids = np.arange(num_vertices, dtype=np.int64)
            identity = True
        else:
            external_ids, src_idx, dst_idx = factorize(src, dst)
            identity = False
        core = _core.build_graph(
            np.ascontiguousarray(src_idx, dtype=np.uint32),
            np.ascontiguousarray(dst_idx, dtype=np.uint32),
            w, len(external_ids), bool(directed),
        )
        # Defensive copies: k_hop(as_graph=True) indexes these later, and a
        # caller mutating their arrays must not corrupt the immutable
        # snapshot's view of its own inputs.
        return cls._wrap(core, external_ids, identity, src.copy(), dst.copy(),
                         None if w is None else w.copy())

    @classmethod
    def _from_universe(cls, src_ext, dst_ext, weights, universe_ext, directed):
        """Build a graph over an explicit vertex universe (external IDs).

        Used by k_hop(as_graph=True): the universe carries reached vertices
        (so isolated reached vertices appear) plus the induced-edge
        endpoints. User indices follow np.unique order of the universe.
        """
        src_ext = np.asarray(src_ext)
        dst_ext = np.asarray(dst_ext)
        uniques = np.unique(np.asarray(universe_ext))
        src_idx = index_in_sorted(uniques, src_ext, "subgraph src")
        dst_idx = index_in_sorted(uniques, dst_ext, "subgraph dst")
        w = None
        if weights is not None:
            w = np.ascontiguousarray(weights, dtype=np.float32)
        core = _core.build_graph(src_idx, dst_idx, w, len(uniques),
                                 bool(directed))
        return cls._wrap(core, uniques, False, src_ext, dst_ext, w)

    @property
    def num_vertices(self):
        return self._g.num_vertices

    @property
    def num_edges(self):
        return self._g.num_edges

    @property
    def directed(self):
        return self._g.directed

    @property
    def weighted(self):
        return self._g.weighted

    @property
    def external_ids(self):
        return self._external_ids

    def index_of(self, ids):
        """Map external IDs back to user indices (scalar or array)."""
        scalar = np.ndim(ids) == 0
        arr = np.atleast_1d(np.asarray(ids))
        if self._identity:
            if arr.size and not np.issubdtype(arr.dtype, np.integer):
                raise ValueError("index_of: integer ids required")
            if arr.size and (int(arr.min()) < 0
                             or int(arr.max()) >= self.num_vertices):
                raise ValueError("index_of: id out of range")
            out = arr.astype(np.int64)
        else:
            out = index_in_sorted(self._external_ids, arr,
                                  "index_of").astype(np.int64)
        return int(out[0]) if scalar else out

    def __repr__(self):
        return (
            f"Graph(num_vertices={self.num_vertices}, "
            f"num_edges={self.num_edges}, directed={self.directed}, "
            f"weighted={self.weighted})"
        )


def pagerank(G, alpha=0.85, tol=1e-6, max_iter=100, personalization=None):
    """Full-vector PageRank (NetworkX-compatible iteration); float32 [V]."""
    core = _core_graph(G)
    p = None
    if personalization is not None:
        V = G.num_vertices
        if isinstance(personalization, dict):
            p = np.zeros(V, dtype=np.float32)
            for idx, wgt in personalization.items():
                i = int(idx)
                if i < 0 or i >= V:
                    raise ValueError(
                        "personalization: user index out of range"
                    )
                p[i] = wgt
        else:
            p = np.ascontiguousarray(personalization, dtype=np.float32)
            if p.ndim != 1 or p.shape[0] != V:
                raise ValueError(
                    "personalization: must be 1-D of length num_vertices"
                )
    return _core.pagerank(core, float(alpha), float(tol), int(max_iter), p)


def ppr_topk(G, seeds, seed_weights, seed_offsets, k=64, alpha=0.85, tol=1e-6,
             max_iter=50, edge_types=None, time_range=None):
    """Batched personalized PageRank with top-k.

    Query q owns seeds[seed_offsets[q]:seed_offsets[q+1]]. Returns
    (ids int32 (B, k) with -1 padding, scores float32 (B, k)).
    """
    core = _core_graph(G)
    _reserved("edge_types", edge_types)
    _reserved("time_range", time_range)
    s = _as_index_array(seeds, G.num_vertices, "seeds")
    sw = np.ascontiguousarray(seed_weights, dtype=np.float32)
    if sw.ndim != 1 or sw.shape[0] != s.shape[0]:
        raise ValueError("seed_weights: must be 1-D of the same length as "
                         "seeds")
    off = np.asarray(seed_offsets)
    if off.ndim != 1 or off.size < 1:
        raise ValueError("seed_offsets: must be 1-D with length B + 1")
    if not np.issubdtype(off.dtype, np.integer):
        raise ValueError("seed_offsets: integer array required")
    if off.size and int(off.min()) < 0:
        raise ValueError("seed_offsets: entries must be >= 0")
    if int(off[0]) != 0:
        raise ValueError("seed_offsets: must start at 0")
    if np.any(off[1:] < off[:-1]):
        raise ValueError("seed_offsets: must be non-decreasing")
    if int(off[-1]) != s.shape[0]:
        raise ValueError("seed_offsets: last entry must equal len(seeds)")
    off = np.ascontiguousarray(off, dtype=np.uint64)
    k = int(k)
    if k < 0:
        raise ValueError("k: must be >= 0")
    return _core.ppr_topk(core, s, sw, off, k, float(alpha), float(tol),
                          int(max_iter))


def bfs(G, sources, direction="out", output="dense"):
    """Multi-source BFS.

    output="dense" (default): returns (dist, parent), both int32 [V],
    -1 = unreachable / no parent — unchanged contract.

    output="sparse": returns (vertices, dist, parent) aligned arrays of
    length |reached| (uint32/int32/int32), vertices ascending, sources at
    depth 0 with parent -1. Same reachability as dense; O(|reached|)
    output instead of O(V). When the bounded collector finishes within the
    configured caps, it also avoids dense result initialization. Cap
    overflow or forced-GPU execution compacts from dense state.
    """
    core = _core_graph(G)
    srcs = _as_index_array(sources, G.num_vertices, "sources")
    if output == "dense":
        return _core.bfs(core, srcs, _direction_code(direction))
    if output == "sparse":
        return _core.bfs_sparse(core, srcs, _direction_code(direction))
    raise ValueError('output: must be "dense" or "sparse"')


def k_hop(G, seeds, k=2, direction="both", max_vertices=None, max_edges=None,
          as_graph=False, edge_types=None, time_range=None):
    """Bounded k-hop neighborhood.

    Returns (vs, es): reached user indices and induced original input edge
    ids, both uint32 ascending. as_graph=True returns a new Graph over the
    induced edges (isolated reached vertices included).
    """
    core = _core_graph(G)
    _reserved("edge_types", edge_types)
    _reserved("time_range", time_range)
    s = _as_index_array(seeds, G.num_vertices, "seeds")
    k = int(k)
    if k < 0:
        raise ValueError("k: must be >= 0")
    mv = 0 if max_vertices is None else int(max_vertices)
    me = 0 if max_edges is None else int(max_edges)
    if mv < 0:
        raise ValueError("max_vertices: must be >= 0 (None = unlimited)")
    if me < 0:
        raise ValueError("max_edges: must be >= 0 (None = unlimited)")
    vs, es = _core.k_hop(core, s, k, _direction_code(direction), mv, me)
    if not as_graph:
        return vs, es
    sub_src = G._src_ext[es]
    sub_dst = G._dst_ext[es]
    sub_w = None if G._weights is None else G._weights[es]
    reached_ext = np.asarray(G.external_ids)[vs]
    universe = np.concatenate([reached_ext, sub_src, sub_dst])
    return Graph._from_universe(sub_src, sub_dst, sub_w, universe,
                                G.directed)


def set_execution(mode="auto"):
    """Per-operation planner mode: 'auto' | 'gpu' | 'cpu'."""
    code = _EXEC_MODES.get(mode)
    if code is None:
        raise ValueError("mode: must be 'auto', 'gpu' or 'cpu'")
    _core.set_execution(code)


def last_run_info():
    """Telemetry for the last operation: {'op', 'path', 'iterations', 'ms'}."""
    return _core.last_run_info()


def has_gpu():
    """True when a Metal device is available (and not hidden by
    MG_FORCE_CPU=1)."""
    return _core.has_gpu()


from . import experimental  # noqa: E402  (needs _core imported above)
