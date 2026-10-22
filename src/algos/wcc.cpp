// wcc.cpp — mg::wcc entry point and mg::gpu::wcc (Afforest-style sample
// link pass and snapshot, one filtered concurrent union-find link pass over
// the OUT-orientation worklists, then one compress pass; labels are the
// minimum canonical id per component).
// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>

#include "../kernels/mg_params.h"
#include "algos.hpp"

namespace mg {

namespace gpu {

BufferPtr wcc(Graph& g) {
  Runtime& rt = Runtime::instance();
  if (!rt.has_gpu())
    throw Error(ErrorCode::no_gpu, "gpu::wcc requires a Metal device");
  const uint32_t V = g.V;
  if (V == 0) throw Error(ErrorCode::internal, "gpu::wcc requires V > 0");

  const Orientation& out_o = g.orientation(Dir::out);
  MTL::ComputePipelineState* pso_iota = rt.pipeline("mg_iota_u32");
  MTL::ComputePipelineState* pso_sample = rt.pipeline("mg_wcc_sample");
  MTL::ComputePipelineState* pso_snapshot = rt.pipeline("mg_wcc_snapshot");
  MTL::ComputePipelineState* pso_link = rt.pipeline("mg_wcc_link");
  MTL::ComputePipelineState* pso_compress = rt.pipeline("mg_wcc_compress");
  // mg_wcc_snapshot builds each 32-bit giant word from one simd_ballot.
  if (rt.exec_width(pso_snapshot) % 32u != 0u)
    throw Error(ErrorCode::internal,
                "gpu::wcc requires a SIMD width that is a multiple of 32");

  // Every element of these is written on the GPU before anything reads it
  // (labels by iota, giant words by snapshot, roots by snapshot for
  // vertices outside the giant set and by compress for all), so they skip
  // the zero fill.
  const std::size_t bytes = std::size_t(V) * sizeof(uint32_t);
  BufferPtr labels = rt.alloc_scratch(bytes, "wcc.labels");
  BufferPtr roots = rt.alloc_scratch(bytes, "wcc.roots");
  BufferPtr giant = rt.alloc_scratch(
      std::size_t(ceil_div_u32(V, 32u)) * sizeof(uint32_t), "wcc.giant");

  const uint32_t sgs_per_tg =
      std::max(1u, MG_TG_SIZE / std::max(1u, rt.exec_width(pso_link)));
  // Undirected graphs store every u != v edge both ways, which makes the
  // Afforest giant-set skip and twin skipping exact.
  const uint32_t flags =
      g.directed ? 0u : (MG_WCC_SKIP_TWINS | MG_WCC_SKIP_GIANT);
  // Canonical ids are out-degree descending: [0, V - n_zero) have edges.
  const uint32_t n_sample = V - out_o.n_zero;
  const MGWccParams p_sample{V, n_sample, 0u, 0u};
  const MGWccParams p_all{V, V, 0u, 0u};
  const uint32_t mid_tgs = ceil_div_u32(out_o.n_mid, sgs_per_tg);
  const MGWccLinkParams p_link{flags,         out_o.n_tiles, out_o.n_high,
                               out_o.n_mid,   mid_tgs,       out_o.n_low,
                               0u,            0u};
  const uint32_t link_tgs = out_o.n_tiles + out_o.n_high + mid_tgs +
                            ceil_div_u32(out_o.n_low, MG_TG_SIZE);

  // Single batch, serial encoder (each dispatch sees the previous one's
  // writes): iota -> sample link pass -> snapshot -> one filtered
  // union-find link pass over every stored edge (all bins in one
  // dispatch) -> final compress. Concurrent union-find needs no
  // convergence rounds: after the link pass every edge's endpoints share a
  // tree rooted at the component's minimum canonical id, and compress
  // flattens each vertex to that root. (MG_WCC_ROUNDS_PER_BATCH no longer
  // applies; it is ignored.)
  CommandBatch cb(rt);
  const MGFillParams ip{0.0f, 0u, V, 0u};
  cb.dispatch(pso_iota, &ip, sizeof(ip), {labels.get()},
              ceil_div_u32(V, MG_TG_SIZE), MG_TG_SIZE);
  cb.dispatch(pso_sample, &p_sample, sizeof(p_sample),
              {out_o.row_offsets.get(), out_o.col_indices.get(),
               labels.get()},
              ceil_div_u32(n_sample, MG_TG_SIZE), MG_TG_SIZE);
  cb.dispatch(pso_snapshot, &p_all, sizeof(p_all),
              {labels.get(), roots.get(), giant.get()},
              ceil_div_u32(V, MG_TG_SIZE), MG_TG_SIZE);
  cb.dispatch(pso_link, &p_link, sizeof(p_link),
              {out_o.tile_owner.get(), out_o.tile_first_edge.get(),
               out_o.wl_huge.get(), out_o.wl_high.get(), out_o.wl_mid.get(),
               out_o.wl_low.get(), out_o.row_offsets.get(),
               out_o.col_indices.get(), roots.get(), giant.get(),
               labels.get()},
              link_tgs, MG_TG_SIZE);
  cb.dispatch(pso_compress, &p_all, sizeof(p_all),
              {labels.get(), giant.get(), roots.get()},
              ceil_div_u32(V, MG_TG_SIZE), MG_TG_SIZE);
  cb.commit_and_wait();
  return roots;
}

}  // namespace gpu

namespace {

// Canonical-partition ids: components numbered by first occurrence
// scanning USER index order (algos.hpp). labels[c] is the root (minimum
// canonical id) of canonical vertex c's component. Returns the count.
int number_serial(const Graph& g, const uint32_t* labels, int32_t* out_comp) {
  const uint32_t V = g.V;
  // id_of_root[r] = 1 + r's component id (0 = not seen yet). calloc hands
  // out lazily zeroed pages for large V, so only roots' slots are touched.
  std::unique_ptr<int32_t, decltype(&std::free)> id_mem(
      static_cast<int32_t*>(std::calloc(V, sizeof(int32_t))), &std::free);
  if (!id_mem) throw Error(ErrorCode::out_of_memory, "wcc: numbering");
  int32_t* id_of_root = id_mem.get();
  int n_components = 0;
  for (uint32_t u = 0; u < V; ++u) {
    const uint32_t root = labels[g.canon_of_user[u]];
    if (root >= V) throw Error(ErrorCode::internal, "wcc: label out of range");
    int32_t& slot = id_of_root[root];
    if (slot == 0) slot = ++n_components;
    out_comp[u] = slot - 1;
  }
  return n_components;
}

// Vertices per numbering chunk (a multiple of 32), and the V from which the
// threaded numbering runs (below it the serial scan wins).
constexpr uint32_t kNumberChunk = 1u << 14;
constexpr uint32_t kNumberParMinV = 1u << 17;

// Same numbering, threaded. Each phase is one parallel_items over fixed
// chunks (independent of the thread count), so the end of a dispatch is the
// phase barrier; every write to shared state is either atomic or to a range
// only one chunk owns.
// Phase 1 (canonical chunks, sequential reads): first[r] = 1 + the
//   smallest user index in root r's component (atomic min; each chunk
//   flushes one running minimum per run of equal roots).
// Phase 2 (user chunks): out_comp[u] temporarily holds root(u)'s bits;
//   bit u of `bits` marks a first occurrence, with per-word counts of the
//   earlier first occurrences in the same chunk. A serial prefix over the
//   chunk totals then gives each chunk its first component id.
// Phase 3 (user chunks): out_comp[u] = number of first occurrences before
//   first[root(u)] - 1, read off the bitmap and prefixes.
int number_parallel(const Graph& g, const uint32_t* labels,
                    int32_t* out_comp) {
  const uint32_t V = g.V;
  const uint32_t n_chunks = ceil_div_u32(V, kNumberChunk);
  const uint32_t n_words = ceil_div_u32(V, 32u);
  // calloc: lazily zeroed pages, so only the slots of roots are touched.
  std::unique_ptr<uint32_t, decltype(&std::free)> first_mem(
      static_cast<uint32_t*>(std::calloc(V, sizeof(uint32_t))), &std::free);
  if (!first_mem) throw Error(ErrorCode::out_of_memory, "wcc: numbering");
  uint32_t* first = first_mem.get();
  // bits[2w]: first-occurrence bits of users 32w..32w+31; bits[2w + 1]:
  // first occurrences in the same chunk before user 32w.
  std::unique_ptr<uint32_t[]> bits(new uint32_t[2 * std::size_t(n_words)]);
  uint32_t* roots_user = reinterpret_cast<uint32_t*>(out_comp);
  std::vector<uint32_t> first_id(n_chunks + 1, 0);
  std::atomic<bool> bad{false};
  auto chunk_end = [V](uint32_t c) {
    return static_cast<uint32_t>(
        std::min<uint64_t>(V, uint64_t(c + 1) * kNumberChunk));
  };

  parallel_items(n_chunks, [&](std::size_t ci) {
    const uint32_t c = static_cast<uint32_t>(ci);
    const uint32_t x1 = chunk_end(c);
    uint32_t run_root = V, run_min = 0;
    auto flush = [&] {
      if (run_root == V) return;
      atomic_ref<uint32_t> f(first[run_root]);
      uint32_t cur = f.load(std::memory_order_relaxed);
      while ((cur == 0 || run_min < cur) &&
             !f.compare_exchange_weak(cur, run_min,
                                      std::memory_order_relaxed)) {
      }
    };
    for (uint32_t x = c * kNumberChunk; x < x1; ++x) {
      const uint32_t r = labels[x];
      if (r >= V) {
        bad.store(true, std::memory_order_relaxed);
        return;
      }
      const uint32_t u1 = g.user_of_canon[x] + 1u;
      if (r != run_root) {
        flush();
        run_root = r;
        run_min = u1;
      } else {
        run_min = std::min(run_min, u1);
      }
    }
    flush();
  });
  if (bad.load(std::memory_order_relaxed))
    throw Error(ErrorCode::internal, "wcc: label out of range");

  parallel_items(n_chunks, [&](std::size_t ci) {
    const uint32_t c = static_cast<uint32_t>(ci);
    const uint32_t u1 = chunk_end(c);
    uint32_t n = 0;
    for (uint32_t u0 = c * kNumberChunk; u0 < u1; u0 += 32u) {
      uint32_t word = 0;
      const uint32_t ue = u1 - u0 > 32u ? u0 + 32u : u1;
      for (uint32_t u = u0; u < ue; ++u) {
        const uint32_t r = labels[g.canon_of_user[u]];
        roots_user[u] = r;
        word |= uint32_t(first[r] == u + 1u) << (u & 31u);
      }
      bits[2 * std::size_t(u0 >> 5)] = word;
      bits[2 * std::size_t(u0 >> 5) + 1] = n;
      n += static_cast<uint32_t>(__builtin_popcount(word));
    }
    first_id[c + 1] = n;
  });
  for (uint32_t c = 0; c < n_chunks; ++c) first_id[c + 1] += first_id[c];

  parallel_items(n_chunks, [&](std::size_t ci) {
    const uint32_t c = static_cast<uint32_t>(ci);
    const uint32_t u1 = chunk_end(c);
    for (uint32_t u = c * kNumberChunk; u < u1; ++u) {
      const uint32_t f = first[roots_user[u]] - 1u;
      const std::size_t w = f >> 5;
      out_comp[u] = static_cast<int32_t>(
          first_id[f / kNumberChunk] + bits[2 * w + 1] +
          __builtin_popcount(bits[2 * w] & ((1u << (f & 31u)) - 1u)));
    }
  });
  return static_cast<int>(first_id[n_chunks]);
}

int number_components(const Graph& g, const uint32_t* labels,
                      int32_t* out_comp) {
  if (num_threads() > 1 && g.V >= kNumberParMinV)
    return number_parallel(g, labels, out_comp);
  return number_serial(g, labels, out_comp);
}

}  // namespace

int wcc(Graph& g, int32_t* out_comp) {
  const auto t0 = std::chrono::steady_clock::now();
  Runtime& rt = Runtime::instance();
  const uint32_t V = g.V;
  if (V > 0 && !out_comp) throw_invalid("wcc: output array is null");

  int rounds = 0;
  int n_components = 0;
  ExecPath path = ExecPath::cpu;
  if (V > 0) {
    path = rt.plan(g.stored_edges(), PlanOp::wcc);
    if (path == ExecPath::gpu) {
      const BufferPtr labels = gpu::wcc(g);
      rounds = 1;  // one fixed pass sequence, no convergence rounds
      n_components = number_components(g, labels->as<uint32_t>(), out_comp);
    } else {
      std::unique_ptr<uint32_t[]> labels(new uint32_t[V]);
      rounds = cpu::wcc(g, labels.get());
      n_components = number_components(g, labels.get(), out_comp);
    }
  }

  const double ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0)
                        .count();
  RunInfo info;
  info.op = "wcc";
  info.path = path;
  info.iterations = rounds;
  info.elapsed_ms = ms;
  rt.record_run(info);
  return n_components;
}

}  // namespace mg
