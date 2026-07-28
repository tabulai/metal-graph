# _ids.py — external-ID factorization helpers shared by Graph.from_edges
# and Graph._from_universe.
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
    combined = np.concatenate([src, dst])
    uniques, inverse = np.unique(combined, return_inverse=True)
    inverse = np.ascontiguousarray(inverse, dtype=np.uint32)
    return uniques, inverse[: src.size], inverse[src.size:]


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
