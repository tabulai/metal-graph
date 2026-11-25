// spmv.metal — PageRank / HITS / batched-PPR fused single-dispatch gather
// kernels (sub-SIMD + 2048-edge tiling) and initialization helpers.
// Binding tables and write contracts: src/kernels/mg_params.h.
// SPDX-License-Identifier: Apache-2.0

#include <metal_stdlib>
using namespace metal;

#include "mg_params.h"
#include "mg_common_msl.h"

constant bool mg_weighted  [[function_constant(MG_FC_WEIGHTED)]];
constant bool mg_uniform_p [[function_constant(MG_FC_UNIFORM_P)]];

// ---------------------------------------------------------------------------
// Single-query PageRank / HITS
// ---------------------------------------------------------------------------

// Write next iterate (either contrib = r/ows for ows>0, raw rank r, or raw
// gather acc for HITS) to out_buf[v].
static inline void mg_pr_store_fused(device float* out_buf,
                                     device const float* out_weight_sum,
                                     uint v, float acc,
                                     constant MGFusedPrParams& p,
                                     device const float* iter_scalars,
                                     device const float* pvec) {
  if ((p.flags & MG_PR_FLAG_RAW_GATHER) != 0u) {
    out_buf[v] = acc;
    return;
  }
  const float pv = mg_uniform_p ? p.p_uniform : pvec[v];
  const float r = p.one_minus_alpha * pv +
                  p.alpha * (acc + iter_scalars[MG_SCALAR_DANGLING] * pv);
  if ((p.flags & MG_PR_FLAG_RESCALE_OWS) != 0u) {
    const float w = out_weight_sum[v];
    out_buf[v] = (w > 0.0f) ? (r / w) : r;
  } else {
    out_buf[v] = r;
  }
}

// Legacy prepare (used only when !ows_safe_rescale requires a separate contrib
// buffer): contrib[v] = ows[v] > 0 ? rank[v] / ows[v] : 0.
kernel void mg_pr_prepare(
    constant MGPrParams& p             [[buffer(0)]],
    device const float* rank_cur       [[buffer(1)]],
    device const float* out_weight_sum [[buffer(2)]],
    device float* contrib              [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= p.count) return;
  const float w = out_weight_sum[gid];
  contrib[gid] = (w > 0.0f) ? rank_cur[gid] / w : 0.0f;
}

// Uniform PageRank iterate-0 init (once at done == 0):
//   pad0 != 0 (rescale): dst[v] = ows[v] > 0 ? p_uniform / ows[v] : p_uniform
//   pad0 == 0 (raw):     dst[v] = p_uniform
kernel void mg_pr_init_uniform_contrib(
    constant MGPrParams& p             [[buffer(0)]],
    device const float* out_weight_sum [[buffer(1)]],
    device float* dst                  [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= p.count) return;
  float val = p.p_uniform;
  if (p.pad0 != 0u) {
    const float w = out_weight_sum[gid];
    if (w > 0.0f) val /= w;
  }
  dst[gid] = val;
}

// Personalized PageRank iterate-0 init (once at done == 0):
//   dst[v] = ows[v] > 0 ? pvec[v] / ows[v] : pvec[v]
kernel void mg_pr_init_pvec_contrib(
    constant MGPrParams& p             [[buffer(0)]],
    device const float* pvec           [[buffer(1)]],
    device const float* out_weight_sum [[buffer(2)]],
    device float* dst                  [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= p.count) return;
  const float pv = pvec[gid];
  const float w = out_weight_sum[gid];
  dst[gid] = (w > 0.0f) ? (pv / w) : pv;
}

// Fused single-dispatch gather covering ALL degree bins with no inter-bin
// GPU drain. Threadgroup ranges in order:
//   [0, n_tiles)  : 2048-edge tiles of huge rows -> tile_partials[tg]
//   [+n_tg)       : 1 threadgroup / vertex (deg 1024..4095)
//   [+sg32_tgs)   : 32 lanes / vertex (deg 64..1023)
//   [+sg16_tgs)   : 16 lanes / vertex (deg 16..63)
//   [+sg8_tgs)    : 8 lanes / vertex  (deg 8..15)
//   [+sg4_tgs)    : 4 lanes / vertex  (deg 4..7)
//   [+th_tgs)     : 1 thread / vertex (deg 1..3)
//   [+zero_tgs)   : 1 thread / vertex (zero-in-degree)
kernel void mg_pr_gather_fused(
    constant MGFusedPrParams& p        [[buffer(0)]],
    device const uint* vertices        [[buffer(1)]],
    device const uint* tile_owner      [[buffer(2)]],
    device const uint* tile_first      [[buffer(3)]],
    device const uint* row_off         [[buffer(4)]],
    device const uint* col_idx         [[buffer(5)]],
    device const float* contrib        [[buffer(6)]],
    device float* out_buf              [[buffer(7)]],
    device const float* out_weight_sum [[buffer(8)]],
    device const float* iter_scalars   [[buffer(9)]],
    device const float* pvec           [[buffer(10)]],
    device const float* weights        [[buffer(11)]],
    device float* tile_partials        [[buffer(12)]],
    uint tg_id        [[threadgroup_position_in_grid]],
    uint tid          [[thread_index_in_threadgroup]],
    uint tg_size      [[threads_per_threadgroup]],
    ushort sg_in_tg   [[simdgroup_index_in_threadgroup]],
    ushort sgs_per_tg [[simdgroups_per_threadgroup]],
    ushort lane       [[thread_index_in_simdgroup]],
    ushort simd_sz    [[threads_per_simdgroup]]) {
  threadgroup float scratch[MG_TG_SCRATCH];
  uint t = tg_id;
  if (t < p.n_tiles) {
    const uint v = vertices[tile_owner[t]];
    const uint begin = tile_first[t];
    const uint end = min(row_off[v + 1u], begin + MG_PR_TILE_EDGES);
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    uint e = begin + tid;
    const uint s2 = 2u * tg_size, s3 = 3u * tg_size, s4 = 4u * tg_size;
    if (mg_weighted) {
      for (; e + s3 < end; e += s4) {
        a0 += contrib[col_idx[e]] * weights[e];
        a1 += contrib[col_idx[e + tg_size]] * weights[e + tg_size];
        a2 += contrib[col_idx[e + s2]] * weights[e + s2];
        a3 += contrib[col_idx[e + s3]] * weights[e + s3];
      }
      for (; e < end; e += tg_size) a0 += contrib[col_idx[e]] * weights[e];
    } else {
      for (; e + s3 < end; e += s4) {
        a0 += contrib[col_idx[e]];
        a1 += contrib[col_idx[e + tg_size]];
        a2 += contrib[col_idx[e + s2]];
        a3 += contrib[col_idx[e + s3]];
      }
      for (; e < end; e += tg_size) a0 += contrib[col_idx[e]];
    }
    const float tot = mg_tg_reduce_sum_tid0((a0 + a1) + (a2 + a3), scratch,
                                            sg_in_tg, lane, sgs_per_tg);
    if (tid == 0u) tile_partials[t] = tot;
    return;
  }
  t -= p.n_tiles;
  if (t < p.n_tg) {
    const uint v = vertices[p.off_tg + t];
    const uint begin = row_off[v];
    const uint end = row_off[v + 1u];
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    uint e = begin + tid;
    const uint s2 = 2u * tg_size, s3 = 3u * tg_size, s4 = 4u * tg_size;
    if (mg_weighted) {
      for (; e + s3 < end; e += s4) {
        a0 += contrib[col_idx[e]] * weights[e];
        a1 += contrib[col_idx[e + tg_size]] * weights[e + tg_size];
        a2 += contrib[col_idx[e + s2]] * weights[e + s2];
        a3 += contrib[col_idx[e + s3]] * weights[e + s3];
      }
      for (; e < end; e += tg_size) a0 += contrib[col_idx[e]] * weights[e];
    } else {
      for (; e + s3 < end; e += s4) {
        a0 += contrib[col_idx[e]];
        a1 += contrib[col_idx[e + tg_size]];
        a2 += contrib[col_idx[e + s2]];
        a3 += contrib[col_idx[e + s3]];
      }
      for (; e < end; e += tg_size) a0 += contrib[col_idx[e]];
    }
    const float tot = mg_tg_reduce_sum_tid0((a0 + a1) + (a2 + a3), scratch,
                                            sg_in_tg, lane, sgs_per_tg);
    if (tid == 0u)
      mg_pr_store_fused(out_buf, out_weight_sum, v, tot, p, iter_scalars, pvec);
    return;
  }
  t -= p.n_tg;
  if (t < p.sg32_tgs) {
    const uint wi = t * sgs_per_tg + sg_in_tg;
    if (wi >= p.n_sg32) return;
    const uint v = vertices[p.off_sg32 + wi];
    const uint end = row_off[v + 1u];
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    uint e = row_off[v] + lane;
    const uint s1 = simd_sz, s2 = 2u * simd_sz, s3 = 3u * simd_sz,
               s4 = 4u * simd_sz;
    if (mg_weighted) {
      for (; e + s3 < end; e += s4) {
        a0 += contrib[col_idx[e]] * weights[e];
        a1 += contrib[col_idx[e + s1]] * weights[e + s1];
        a2 += contrib[col_idx[e + s2]] * weights[e + s2];
        a3 += contrib[col_idx[e + s3]] * weights[e + s3];
      }
      for (; e < end; e += s1) a0 += contrib[col_idx[e]] * weights[e];
    } else {
      for (; e + s3 < end; e += s4) {
        a0 += contrib[col_idx[e]];
        a1 += contrib[col_idx[e + s1]];
        a2 += contrib[col_idx[e + s2]];
        a3 += contrib[col_idx[e + s3]];
      }
      for (; e < end; e += s1) a0 += contrib[col_idx[e]];
    }
    const float tot = simd_sum((a0 + a1) + (a2 + a3));
    if (lane == 0u)
      mg_pr_store_fused(out_buf, out_weight_sum, v, tot, p, iter_scalars, pvec);
    return;
  }
  t -= p.sg32_tgs;
  if (t < p.sg16_tgs) {
    const uint sub_id = tid >> 4u;
    const uint sub_lane = tid & 15u;
    const uint wi = t * (tg_size >> 4u) + sub_id;
    const bool valid = wi < p.n_sg16;
    uint v = 0u;
    float a0 = 0.0f, a1 = 0.0f;
    if (valid) {
      v = vertices[p.off_sg16 + wi];
      const uint end = row_off[v + 1u];
      uint e = row_off[v] + sub_lane;
      if (mg_weighted) {
        for (; e + 16u < end; e += 32u) {
          a0 += contrib[col_idx[e]] * weights[e];
          a1 += contrib[col_idx[e + 16u]] * weights[e + 16u];
        }
        if (e < end) a0 += contrib[col_idx[e]] * weights[e];
      } else {
        for (; e + 16u < end; e += 32u) {
          a0 += contrib[col_idx[e]];
          a1 += contrib[col_idx[e + 16u]];
        }
        if (e < end) a0 += contrib[col_idx[e]];
      }
    }
    float acc = a0 + a1;
    acc += simd_shuffle_down(acc, 8u);
    acc += simd_shuffle_down(acc, 4u);
    acc += simd_shuffle_down(acc, 2u);
    acc += simd_shuffle_down(acc, 1u);
    if (valid && sub_lane == 0u)
      mg_pr_store_fused(out_buf, out_weight_sum, v, acc, p, iter_scalars, pvec);
    return;
  }
  t -= p.sg16_tgs;
  if (t < p.sg8_tgs) {
    const uint sub_id = tid >> 3u;
    const uint sub_lane = tid & 7u;
    const uint wi = t * (tg_size >> 3u) + sub_id;
    const bool valid = wi < p.n_sg8;
    uint v = 0u;
    float acc = 0.0f;
    if (valid) {
      v = vertices[p.off_sg8 + wi];
      const uint end = row_off[v + 1u];
      if (mg_weighted) {
        for (uint e = row_off[v] + sub_lane; e < end; e += 8u)
          acc += contrib[col_idx[e]] * weights[e];
      } else {
        for (uint e = row_off[v] + sub_lane; e < end; e += 8u)
          acc += contrib[col_idx[e]];
      }
    }
    acc += simd_shuffle_down(acc, 4u);
    acc += simd_shuffle_down(acc, 2u);
    acc += simd_shuffle_down(acc, 1u);
    if (valid && sub_lane == 0u)
      mg_pr_store_fused(out_buf, out_weight_sum, v, acc, p, iter_scalars, pvec);
    return;
  }
  t -= p.sg8_tgs;
  if (t < p.sg4_tgs) {
    const uint sub_id = tid >> 2u;
    const uint sub_lane = tid & 3u;
    const uint wi = t * (tg_size >> 2u) + sub_id;
    const bool valid = wi < p.n_sg4;
    uint v = 0u;
    float acc = 0.0f;
    if (valid) {
      v = vertices[p.off_sg4 + wi];
      const uint end = row_off[v + 1u];
      if (mg_weighted) {
        for (uint e = row_off[v] + sub_lane; e < end; e += 4u)
          acc += contrib[col_idx[e]] * weights[e];
      } else {
        for (uint e = row_off[v] + sub_lane; e < end; e += 4u)
          acc += contrib[col_idx[e]];
      }
    }
    acc += simd_shuffle_down(acc, 2u);
    acc += simd_shuffle_down(acc, 1u);
    if (valid && sub_lane == 0u)
      mg_pr_store_fused(out_buf, out_weight_sum, v, acc, p, iter_scalars, pvec);
    return;
  }
  t -= p.sg4_tgs;
  if (t < p.th_tgs) {
    const uint i = t * tg_size + tid;
    if (i >= p.n_th) return;
    const uint v = vertices[p.off_th + i];
    const uint end = row_off[v + 1u];
    float acc = 0.0f;
    if (mg_weighted) {
      for (uint e = row_off[v]; e < end; ++e)
        acc += contrib[col_idx[e]] * weights[e];
    } else {
      for (uint e = row_off[v]; e < end; ++e) acc += contrib[col_idx[e]];
    }
    mg_pr_store_fused(out_buf, out_weight_sum, v, acc, p, iter_scalars, pvec);
    return;
  }
  t -= p.th_tgs;
  const uint i = t * tg_size + tid;
  if (i >= p.n_zero) return;
  mg_pr_store_fused(out_buf, out_weight_sum, vertices[p.off_zero + i], 0.0f, p,
                    iter_scalars, pvec);
}

// One simdgroup per tiled vertex: sums tile partials in fixed stride order.
kernel void mg_pr_tiled_finalize(
    constant MGFusedPrParams& p        [[buffer(0)]],
    device const uint* vertices        [[buffer(1)]],
    device const uint* tiled_tile_off  [[buffer(2)]],
    device const float* tile_partials  [[buffer(3)]],
    device float* out_buf              [[buffer(4)]],
    device const float* out_weight_sum [[buffer(5)]],
    device const float* iter_scalars   [[buffer(6)]],
    device const float* pvec           [[buffer(7)]],
    uint tg_id        [[threadgroup_position_in_grid]],
    ushort sg_in_tg   [[simdgroup_index_in_threadgroup]],
    ushort sgs_per_tg [[simdgroups_per_threadgroup]],
    ushort lane       [[thread_index_in_simdgroup]],
    ushort simd_sz    [[threads_per_simdgroup]]) {
  const uint gi = tg_id * sgs_per_tg + sg_in_tg;
  if (gi >= p.n_tiled) return;
  const uint end = tiled_tile_off[gi + 1u];
  float acc = 0.0f;
  for (uint tl = tiled_tile_off[gi] + lane; tl < end; tl += simd_sz)
    acc += tile_partials[tl];
  acc = simd_sum(acc);
  if (lane == 0u)
    mg_pr_store_fused(out_buf, out_weight_sum, vertices[gi], acc, p,
                      iter_scalars, pvec);
}

// ---------------------------------------------------------------------------
// Batched PPR (tile of MG_PPR_TILE == 8 lanes, vertex-major interleaved as
// two float4s per vertex).
// ---------------------------------------------------------------------------
static_assert(MG_PPR_TILE == 8u, "a vertex's lanes are two float4s");

static inline void mg_pr_store_b_f4(device float4* out4,
                                    device const float4* cur4,
                                    device const float* out_weight_sum, uint v,
                                    float4 lo, float4 hi,
                                    constant MGFusedPprParams& p) {
  float scale = p.alpha;
  if ((p.flags & MG_PR_FLAG_RESCALE_OWS) != 0u) {
    const float w = out_weight_sum[v];
    if (w > 0.0f) scale /= w;
  }
  const ulong b2 = ulong(v) * 2ul;
  if (p.active_mask == 0xFFu) {
    out4[b2] = lo * scale;
    out4[b2 + 1ul] = hi * scale;
  } else {
    const uint m = p.active_mask;
    if ((m & 0x0Fu) == 0x0Fu) {
      out4[b2] = lo * scale;
    } else if ((m & 0x0Fu) == 0u) {
      out4[b2] = cur4[b2];
    } else {
      const float4 c0 = cur4[b2];
      out4[b2] = select(c0, lo * scale,
                        bool4((m & 1u) != 0u, (m & 2u) != 0u, (m & 4u) != 0u,
                              (m & 8u) != 0u));
    }
    if ((m & 0xF0u) == 0xF0u) {
      out4[b2 + 1ul] = hi * scale;
    } else if ((m & 0xF0u) == 0u) {
      out4[b2 + 1ul] = cur4[b2 + 1ul];
    } else {
      const float4 c1 = cur4[b2 + 1ul];
      out4[b2 + 1ul] = select(c1, hi * scale,
                              bool4((m & 16u) != 0u, (m & 32u) != 0u,
                                    (m & 64u) != 0u, (m & 128u) != 0u));
    }
  }
}

// Fallback prepare_b when !ows_safe_rescale (vectorized with float4).
kernel void mg_pr_prepare_b(
    constant MGPprParams& p            [[buffer(0)]],
    device const float4* rank4_cur_b   [[buffer(1)]],
    device const float* out_weight_sum [[buffer(2)]],
    device float4* contrib4_b          [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= p.count) return;
  const float w = out_weight_sum[gid];
  const float inv = (w > 0.0f) ? (1.0f / w) : 0.0f;
  const ulong b2 = ulong(gid) * 2ul;
  contrib4_b[b2] = rank4_cur_b[b2] * inv;
  contrib4_b[b2 + 1ul] = rank4_cur_b[b2 + 1ul] * inv;
}

// Tile init step 1: zero all of buf0 and isolated slots of buf1 (so non-seed
// isolated vertices stay 0 in both ping-pong buffers without per-iter writes).
kernel void mg_pr_tile_zero_b(
    constant MGPprParams& p     [[buffer(0)]],
    device float4* buf0_4       [[buffer(1)]],
    device float4* buf1_4       [[buffer(2)]],
    device const uint* vertices [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid < p.count) {
    buf0_4[2ul * gid] = float4(0.0f);
    buf0_4[2ul * gid + 1ul] = float4(0.0f);
  }
  if (gid < p.pad0) {  // pad0 = n_iso, pad1 = off_iso
    const ulong v2 = ulong(vertices[p.pad1 + gid]) * 2ul;
    buf1_4[v2] = float4(0.0f);
    buf1_4[v2 + 1ul] = float4(0.0f);
  }
}

// Tile init step 2: scatter deduplicated seeds into buf0 (rescaled by 1/ows
// when MG_PR_FLAG_RESCALE_OWS is set in p.pad0).
kernel void mg_pr_seed_init_b(
    constant MGPprParams& p            [[buffer(0)]],
    device const uint* seed_vertex     [[buffer(1)]],
    device const uint* seed_lane       [[buffer(2)]],
    device const float* seed_weight    [[buffer(3)]],
    device const float* out_weight_sum [[buffer(4)]],
    device float* buf0                 [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= p.count) return;
  const uint v = seed_vertex[gid];
  const uint t = seed_lane[gid] & 0xFFu;
  float w = seed_weight[gid];
  if ((p.pad0 & MG_PR_FLAG_RESCALE_OWS) != 0u) {
    const float ow = out_weight_sum[v];
    if (ow > 0.0f) w /= ow;
  }
  buf0[ulong(v) * MG_PPR_TILE + t] = w;
}

// Seed scatter: add teleport + dangling mass on active lanes; for isolated
// seeds (skipped by gather), assign on active lanes and copy-through on
// inactive lanes.
kernel void mg_pr_seed_scatter_b(
    constant MGPprParams& p            [[buffer(0)]],
    device const uint* seed_vertex     [[buffer(1)]],
    device const uint* seed_lane       [[buffer(2)]],
    device const float* seed_weight    [[buffer(3)]],
    device const float* iter_scalars_b [[buffer(4)]],
    device const float* out_weight_sum [[buffer(5)]],
    device float* out_b                [[buffer(6)]],
    device const float* cur_b          [[buffer(7)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= p.count) return;
  const uint raw = seed_lane[gid];
  const uint t = raw & 0xFFu;
  const bool is_iso = (raw & MG_PPR_SEED_ISO_BIT) != 0u;
  const uint v = seed_vertex[gid];
  const ulong slot = ulong(v) * MG_PPR_TILE + t;
  if (((p.active_mask >> t) & 1u) == 0u) {
    if (is_iso) out_b[slot] = cur_b[slot];
    return;
  }
  const float w = seed_weight[gid];
  float add = p.one_minus_alpha * w + p.alpha * iter_scalars_b[t] * w;
  if (is_iso) {
    out_b[slot] = add;
  } else {
    if ((p.pad0 & MG_PR_FLAG_RESCALE_OWS) != 0u) {
      const float ow = out_weight_sum[v];
      if (ow > 0.0f) add /= ow;
    }
    out_b[slot] += add;
  }
}

// 2x-unrolled strided batched gather loop. All three mask specializations
// (both halves, lo only, hi only) use the IDENTICAL pair-addition order so
// every active query lane produces bit-identical floats regardless of
// tile grouping or active_mask.
static inline void mg_gather_b_strided2(device const uint* col_idx,
                                        device const float4* contrib4_b,
                                        device const float* weights, uint start,
                                        uint end, uint step, bool both,
                                        bool active_lo, thread float4& lo,
                                        thread float4& hi) {
  uint e = start;
  const uint s2 = 2u * step;
  if (both) {
    if (mg_weighted) {
      for (; e + step < end; e += s2) {
        const ulong u0 = ulong(col_idx[e]) * 2ul;
        const ulong u1 = ulong(col_idx[e + step]) * 2ul;
        const float w0 = weights[e];
        const float w1 = weights[e + step];
        lo += contrib4_b[u0] * w0 + contrib4_b[u1] * w1;
        hi += contrib4_b[u0 + 1ul] * w0 + contrib4_b[u1 + 1ul] * w1;
      }
      if (e < end) {
        const ulong u0 = ulong(col_idx[e]) * 2ul;
        const float w0 = weights[e];
        lo += contrib4_b[u0] * w0;
        hi += contrib4_b[u0 + 1ul] * w0;
      }
    } else {
      for (; e + step < end; e += s2) {
        const ulong u0 = ulong(col_idx[e]) * 2ul;
        const ulong u1 = ulong(col_idx[e + step]) * 2ul;
        lo += contrib4_b[u0] + contrib4_b[u1];
        hi += contrib4_b[u0 + 1ul] + contrib4_b[u1 + 1ul];
      }
      if (e < end) {
        const ulong u0 = ulong(col_idx[e]) * 2ul;
        lo += contrib4_b[u0];
        hi += contrib4_b[u0 + 1ul];
      }
    }
  } else if (active_lo) {
    if (mg_weighted) {
      for (; e + step < end; e += s2) {
        const ulong u0 = ulong(col_idx[e]) * 2ul;
        const ulong u1 = ulong(col_idx[e + step]) * 2ul;
        lo += contrib4_b[u0] * weights[e] +
              contrib4_b[u1] * weights[e + step];
      }
      if (e < end) lo += contrib4_b[ulong(col_idx[e]) * 2ul] * weights[e];
    } else {
      for (; e + step < end; e += s2) {
        lo += contrib4_b[ulong(col_idx[e]) * 2ul] +
              contrib4_b[ulong(col_idx[e + step]) * 2ul];
      }
      if (e < end) lo += contrib4_b[ulong(col_idx[e]) * 2ul];
    }
  } else {
    if (mg_weighted) {
      for (; e + step < end; e += s2) {
        const ulong u0 = ulong(col_idx[e]) * 2ul + 1ul;
        const ulong u1 = ulong(col_idx[e + step]) * 2ul + 1ul;
        hi += contrib4_b[u0] * weights[e] +
              contrib4_b[u1] * weights[e + step];
      }
      if (e < end)
        hi += contrib4_b[ulong(col_idx[e]) * 2ul + 1ul] * weights[e];
    } else {
      for (; e + step < end; e += s2) {
        hi += contrib4_b[ulong(col_idx[e]) * 2ul + 1ul] +
              contrib4_b[ulong(col_idx[e + step]) * 2ul + 1ul];
      }
      if (e < end) hi += contrib4_b[ulong(col_idx[e]) * 2ul + 1ul];
    }
  }
}

// 1x strided batched gather loop for tiny-degree bins (sg8/sg4/th) with
// hoisted active-half branches and identical association across masks.
static inline void mg_gather_b_strided1(device const uint* col_idx,
                                        device const float4* contrib4_b,
                                        device const float* weights, uint start,
                                        uint end, uint step, bool both,
                                        bool active_lo, thread float4& lo,
                                        thread float4& hi) {
  if (both) {
    if (mg_weighted) {
      for (uint e = start; e < end; e += step) {
        const ulong u2 = ulong(col_idx[e]) * 2ul;
        const float w = weights[e];
        lo += contrib4_b[u2] * w;
        hi += contrib4_b[u2 + 1ul] * w;
      }
    } else {
      for (uint e = start; e < end; e += step) {
        const ulong u2 = ulong(col_idx[e]) * 2ul;
        lo += contrib4_b[u2];
        hi += contrib4_b[u2 + 1ul];
      }
    }
  } else if (active_lo) {
    if (mg_weighted) {
      for (uint e = start; e < end; e += step)
        lo += contrib4_b[ulong(col_idx[e]) * 2ul] * weights[e];
    } else {
      for (uint e = start; e < end; e += step)
        lo += contrib4_b[ulong(col_idx[e]) * 2ul];
    }
  } else {
    if (mg_weighted) {
      for (uint e = start; e < end; e += step)
        hi += contrib4_b[ulong(col_idx[e]) * 2ul + 1ul] * weights[e];
    } else {
      for (uint e = start; e < end; e += step)
        hi += contrib4_b[ulong(col_idx[e]) * 2ul + 1ul];
    }
  }
}

// Fused single-dispatch batched PPR gather (float4 + 1-barrier 8-lane reduce).
// A single hoisted scratch array keeps threadgroup memory at 512 B (vs 2 KB).
kernel void mg_pr_gather_fused_b(
    constant MGFusedPprParams& p       [[buffer(0)]],
    device const uint* vertices        [[buffer(1)]],
    device const uint* tile_owner      [[buffer(2)]],
    device const uint* tile_first      [[buffer(3)]],
    device const uint* row_off         [[buffer(4)]],
    device const uint* col_idx         [[buffer(5)]],
    device const float4* contrib4_b    [[buffer(6)]],
    device float4* out4_b              [[buffer(7)]],
    device const float4* cur4_b        [[buffer(8)]],
    device const float* out_weight_sum [[buffer(9)]],
    device const float* weights        [[buffer(10)]],
    device float* tile_partials_b      [[buffer(11)]],
    uint tg_id        [[threadgroup_position_in_grid]],
    uint tid          [[thread_index_in_threadgroup]],
    uint tg_size      [[threads_per_threadgroup]],
    ushort sg_in_tg   [[simdgroup_index_in_threadgroup]],
    ushort sgs_per_tg [[simdgroups_per_threadgroup]],
    ushort lane       [[thread_index_in_simdgroup]],
    ushort simd_sz    [[threads_per_simdgroup]]) {
  threadgroup float scratch[MG_PPR_TILE * MG_TG_SCRATCH];
  const bool active_lo = (p.active_mask & 0x0Fu) != 0u;
  const bool active_hi = (p.active_mask & 0xF0u) != 0u;
  const bool both = active_lo && active_hi;
  uint t = tg_id;
  if (t < p.n_tiles) {
    const uint v = vertices[tile_owner[t]];
    const uint begin = tile_first[t];
    const uint end = min(row_off[v + 1u], begin + MG_PR_TILE_EDGES);
    float4 lo = float4(0.0f), hi = float4(0.0f);
    mg_gather_b_strided2(col_idx, contrib4_b, weights, begin + tid, end,
                         tg_size, both, active_lo, lo, hi);
    const float s = mg_tg_reduce_sum_b_lane(lo, hi, scratch, sg_in_tg, lane,
                                            sgs_per_tg);
    if (sg_in_tg == 0u && lane < MG_PPR_TILE)
      tile_partials_b[ulong(t) * MG_PPR_TILE + lane] = s;
    return;
  }
  t -= p.n_tiles;
  if (t < p.n_tg) {
    const uint v = vertices[p.off_tg + t];
    const uint end = row_off[v + 1u];
    float4 lo = float4(0.0f), hi = float4(0.0f);
    mg_gather_b_strided2(col_idx, contrib4_b, weights, row_off[v] + tid, end,
                         tg_size, both, active_lo, lo, hi);
    const float s = mg_tg_reduce_sum_b_lane(lo, hi, scratch, sg_in_tg, lane,
                                            sgs_per_tg);
    if (sg_in_tg == 0u && lane < MG_PPR_TILE) {
      const ulong slot = ulong(v) * MG_PPR_TILE + lane;
      if (((p.active_mask >> lane) & 1u) != 0u) {
        float scale = p.alpha;
        if ((p.flags & MG_PR_FLAG_RESCALE_OWS) != 0u) {
          const float ow = out_weight_sum[v];
          if (ow > 0.0f) scale /= ow;
        }
        ((device float*)out4_b)[slot] = s * scale;
      } else {
        ((device float*)out4_b)[slot] = ((device const float*)cur4_b)[slot];
      }
    }
    return;
  }
  t -= p.n_tg;
  if (t < p.sg32_tgs) {
    const uint wi = t * sgs_per_tg + sg_in_tg;
    if (wi >= p.n_sg32) return;
    const uint v = vertices[p.off_sg32 + wi];
    const uint end = row_off[v + 1u];
    float4 lo = float4(0.0f), hi = float4(0.0f);
    mg_gather_b_strided2(col_idx, contrib4_b, weights, row_off[v] + lane, end,
                         simd_sz, both, active_lo, lo, hi);
    if (active_lo) lo = simd_sum(lo);
    if (active_hi) hi = simd_sum(hi);
    if (lane == 0u)
      mg_pr_store_b_f4(out4_b, cur4_b, out_weight_sum, v, lo, hi, p);
    return;
  }
  t -= p.sg32_tgs;
  if (t < p.sg16_tgs) {
    const uint sub_id = tid >> 4u;
    const uint sub_lane = tid & 15u;
    const uint wi = t * (tg_size >> 4u) + sub_id;
    const bool valid = wi < p.n_sg16;
    uint v = 0u;
    float4 lo = float4(0.0f), hi = float4(0.0f);
    if (valid) {
      v = vertices[p.off_sg16 + wi];
      mg_gather_b_strided2(col_idx, contrib4_b, weights, row_off[v] + sub_lane,
                           row_off[v + 1u], 16u, both, active_lo, lo, hi);
    }
    if (active_lo) {
      lo += simd_shuffle_down(lo, 8u);
      lo += simd_shuffle_down(lo, 4u);
      lo += simd_shuffle_down(lo, 2u);
      lo += simd_shuffle_down(lo, 1u);
    }
    if (active_hi) {
      hi += simd_shuffle_down(hi, 8u);
      hi += simd_shuffle_down(hi, 4u);
      hi += simd_shuffle_down(hi, 2u);
      hi += simd_shuffle_down(hi, 1u);
    }
    if (valid && sub_lane == 0u)
      mg_pr_store_b_f4(out4_b, cur4_b, out_weight_sum, v, lo, hi, p);
    return;
  }
  t -= p.sg16_tgs;
  if (t < p.sg8_tgs) {
    const uint sub_id = tid >> 3u;
    const uint sub_lane = tid & 7u;
    const uint wi = t * (tg_size >> 3u) + sub_id;
    const bool valid = wi < p.n_sg8;
    uint v = 0u;
    float4 lo = float4(0.0f), hi = float4(0.0f);
    if (valid) {
      v = vertices[p.off_sg8 + wi];
      mg_gather_b_strided1(col_idx, contrib4_b, weights, row_off[v] + sub_lane,
                           row_off[v + 1u], 8u, both, active_lo, lo, hi);
    }
    if (active_lo) {
      lo += simd_shuffle_down(lo, 4u);
      lo += simd_shuffle_down(lo, 2u);
      lo += simd_shuffle_down(lo, 1u);
    }
    if (active_hi) {
      hi += simd_shuffle_down(hi, 4u);
      hi += simd_shuffle_down(hi, 2u);
      hi += simd_shuffle_down(hi, 1u);
    }
    if (valid && sub_lane == 0u)
      mg_pr_store_b_f4(out4_b, cur4_b, out_weight_sum, v, lo, hi, p);
    return;
  }
  t -= p.sg8_tgs;
  if (t < p.sg4_tgs) {
    const uint sub_id = tid >> 2u;
    const uint sub_lane = tid & 3u;
    const uint wi = t * (tg_size >> 2u) + sub_id;
    const bool valid = wi < p.n_sg4;
    uint v = 0u;
    float4 lo = float4(0.0f), hi = float4(0.0f);
    if (valid) {
      v = vertices[p.off_sg4 + wi];
      mg_gather_b_strided1(col_idx, contrib4_b, weights, row_off[v] + sub_lane,
                           row_off[v + 1u], 4u, both, active_lo, lo, hi);
    }
    if (active_lo) {
      lo += simd_shuffle_down(lo, 2u);
      lo += simd_shuffle_down(lo, 1u);
    }
    if (active_hi) {
      hi += simd_shuffle_down(hi, 2u);
      hi += simd_shuffle_down(hi, 1u);
    }
    if (valid && sub_lane == 0u)
      mg_pr_store_b_f4(out4_b, cur4_b, out_weight_sum, v, lo, hi, p);
    return;
  }
  t -= p.sg4_tgs;
  if (t < p.th_tgs) {
    const uint i = t * tg_size + tid;
    if (i >= p.n_th) return;
    const uint v = vertices[p.off_th + i];
    float4 lo = float4(0.0f), hi = float4(0.0f);
    mg_gather_b_strided1(col_idx, contrib4_b, weights, row_off[v],
                         row_off[v + 1u], 1u, both, active_lo, lo, hi);
    mg_pr_store_b_f4(out4_b, cur4_b, out_weight_sum, v, lo, hi, p);
    return;
  }
  t -= p.th_tgs;
  const uint i = t * tg_size + tid;
  if (i >= p.n_zero) return;
  const uint v = vertices[p.off_zero + i];
  mg_pr_store_b_f4(out4_b, cur4_b, out_weight_sum, v, float4(0.0f),
                   float4(0.0f), p);
}

// One simdgroup per tiled vertex, batched (float4).
kernel void mg_pr_tiled_finalize_b(
    constant MGFusedPprParams& p       [[buffer(0)]],
    device const uint* vertices        [[buffer(1)]],
    device const uint* tiled_tile_off  [[buffer(2)]],
    device const float4* partials4_b   [[buffer(3)]],
    device float4* out4_b              [[buffer(4)]],
    device const float4* cur4_b        [[buffer(5)]],
    device const float* out_weight_sum [[buffer(6)]],
    uint tg_id        [[threadgroup_position_in_grid]],
    ushort sg_in_tg   [[simdgroup_index_in_threadgroup]],
    ushort sgs_per_tg [[simdgroups_per_threadgroup]],
    ushort lane       [[thread_index_in_simdgroup]],
    ushort simd_sz    [[threads_per_simdgroup]]) {
  const uint gi = tg_id * sgs_per_tg + sg_in_tg;
  if (gi >= p.n_tiled) return;
  const bool active_lo = (p.active_mask & 0x0Fu) != 0u;
  const bool active_hi = (p.active_mask & 0xF0u) != 0u;
  const uint end = tiled_tile_off[gi + 1u];
  float4 lo = float4(0.0f), hi = float4(0.0f);
  for (uint tl = tiled_tile_off[gi] + lane; tl < end; tl += simd_sz) {
    const ulong t2 = ulong(tl) * 2ul;
    if (active_lo) lo += partials4_b[t2];
    if (active_hi) hi += partials4_b[t2 + 1ul];
  }
  if (active_lo) lo = simd_sum(lo);
  if (active_hi) hi = simd_sum(hi);
  if (lane == 0u)
    mg_pr_store_b_f4(out4_b, cur4_b, out_weight_sum, vertices[gi], lo, hi, p);
}
