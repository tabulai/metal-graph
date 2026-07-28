// scan.metal — two-pass exclusive scan over uint32.
// MG_SCAN_BLOCK == MG_TG_SIZE == 256: one element per thread per block.
// Host recurses on `sums` until one block, then applies back up.
// Binding tables: src/kernels/mg_params.h (authoritative).
// SPDX-License-Identifier: Apache-2.0

#include <metal_stdlib>
using namespace metal;

#include "mg_params.h"
#include "mg_common_msl.h"

// Block-local exclusive scan: simd prefix sums + cross-simdgroup offsets
// via threadgroup memory. out = exclusive scan within each block;
// sums[block] = block total. Elements beyond count contribute 0.
kernel void mg_exscan_partial(
    constant MGScanParams& p [[buffer(0)]],
    device const uint* src   [[buffer(1)]],
    device uint* dst         [[buffer(2)]],
    device uint* sums        [[buffer(3)]],
    uint gid        [[thread_position_in_grid]],
    uint tg_id      [[threadgroup_position_in_grid]],
    uint tid        [[thread_index_in_threadgroup]],
    ushort sg_in_tg [[simdgroup_index_in_threadgroup]],
    ushort lane     [[thread_index_in_simdgroup]],
    ushort sgs_per_tg [[simdgroups_per_threadgroup]]) {
  const uint x = (gid < p.count) ? src[gid] : 0u;
  const uint pre = simd_prefix_exclusive_sum(x);
  const uint sg_total = simd_sum(x);
  threadgroup uint sg_totals[MG_TG_SCRATCH];
  if (lane == 0u) sg_totals[sg_in_tg] = sg_total;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint sg_offset = 0u;
  for (ushort s = 0; s < sg_in_tg; ++s) sg_offset += sg_totals[s];
  if (gid < p.count) dst[gid] = sg_offset + pre;
  if (tid == 0u) {
    uint total = 0u;
    for (ushort s = 0; s < sgs_per_tg; ++s) total += sg_totals[s];
    sums[tg_id] = total;
  }
}

// out[i] += scanned_sums[block(i)]; block(i) == tg_id (1 element/thread).
kernel void mg_exscan_apply(
    constant MGScanParams& p        [[buffer(0)]],
    device uint* out                [[buffer(1)]],
    device const uint* scanned_sums [[buffer(2)]],
    uint gid   [[thread_position_in_grid]],
    uint tg_id [[threadgroup_position_in_grid]]) {
  if (gid >= p.count) return;
  out[gid] += scanned_sums[tg_id];
}
