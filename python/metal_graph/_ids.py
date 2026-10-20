# _ids.py — external-ID factorization helpers shared by Graph.from_edges
# and Graph.index_of.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np


def factorize(src, dst):
    """Map two external-ID edge arrays to dense user indices.

    Returns (uniques, src_idx, dst_idx). User-index order is np.unique
    order (sorted uniques); works for integer AND str/object arrays.
    src_idx/dst_idx are contiguous uint32.
    """
    src = np.asarray(src)
    dst = np.asarray(dst)
    if (src.dtype.kind in "iu" and dst.dtype.kind in "iu"
            and np.result_type(src.dtype, dst.dtype).kind not in "iu"):
        # e.g. int64 + uint64 promotes to float64, silently corrupting ids
        # above 2**53 and colliding distinct vertices.
        raise ValueError(
            "src/dst: mixing signed and unsigned 64-bit integer ids would "
            "promote to float64 and corrupt large ids; cast both arrays to "
            "a common integer dtype (int64 or uint64) first")
    dense = _factorize_dense(src, dst)
    if dense is not None:
        return dense
    combined = np.concatenate([src, dst])
    uniques, inverse = np.unique(combined, return_inverse=True)
    inverse = np.ascontiguousarray(inverse, dtype=np.uint32)
    return uniques, inverse[: src.size], inverse[src.size:]


def _factorize_dense(src, dst):
    """Dense-range fast path for integer ids; None when it does not apply.

    When the ids span a range not much wider than the edge-endpoint count,
    marking seen ids in a flag array and ranking them with a cumulative sum
    yields exactly np.unique's sorted uniques and inverse in O(N + span)
    instead of np.unique's O(N log N) argsort. Sparse ids (span above
    2 * (2E), i.e. proportional to the edge count with no fixed slack, so a
    small graph with large ids never allocates a huge flag array) and
    non-integer ids fall back to np.unique.
    """
    if (src.dtype.kind not in "iu" or dst.dtype.kind not in "iu"
            or src.size == 0 or dst.size == 0):
        return None
    out_dtype = np.concatenate((src[:0], dst[:0])).dtype  # np.unique's dtype
    if out_dtype.kind not in "iu":
        return None
    n = src.size + dst.size
    lo = min(int(src.min()), int(dst.min()))
    hi = max(int(src.max()), int(dst.max()))
    span = hi - lo + 1
    if span > 2 * n:
        return None
    # Offsets into [0, span) computed in 64 bits: native-width `src - lo`
    # would wrap for narrow dtypes (int8/uint8). uint64 ids >= 2**63 shift
    # in uint64; every offset is < span < 2**63, so the int64 view is exact.
    wide = np.uint64 if hi > np.iinfo(np.int64).max else np.int64
    base = wide(lo)
    s_off = src.astype(wide)
    d_off = dst.astype(wide)
    if lo != 0:
        s_off -= base
        d_off -= base
    s_off = s_off.view(np.intp)
    d_off = d_off.view(np.intp)
    seen = np.zeros(span, dtype=np.bool_)
    seen[s_off] = True
    seen[d_off] = True
    # seen[0] is set (lo is an id), so rank >= 0 everywhere.
    rank = np.cumsum(seen, dtype=np.uint32)
    rank -= np.uint32(1)
    present = np.flatnonzero(seen).astype(wide, copy=False)
    present += base
    return present.astype(out_dtype, copy=False), rank[s_off], rank[d_off]


def index_in_sorted(uniques, values, what="ids"):
    """Map external IDs to user indices via searchsorted on sorted uniques.

    Raises ValueError when any value is not present in `uniques`.
    Returns contiguous uint32.
    """
    values = np.asarray(values)
    if values.size == 0:
        return np.zeros(0, dtype=np.uint32)
    if uniques.size == 0:
        raise ValueError(f"{what}: unknown external id (graph has no vertices)")
    pos = np.searchsorted(uniques, values)
    pos = np.minimum(pos, uniques.size - 1)
    if not np.all(uniques[pos] == values):
        raise ValueError(f"{what}: unknown external id(s)")
    return np.ascontiguousarray(pos, dtype=np.uint32)
