// khop.cpp — mg::k_hop entry point. Cap-bounded runs force the
// deterministic CPU path (v0.1 semantics). Only vertex discovery depends on
// the path (gpu::bfs or cpu::khop_vertices); induced-edge extraction is a
// host pass shared by both: edges with both endpoints reached are marked in
// a bitmap over ORIGINAL input edge ids and enumerated ascending, so the
// list is sorted and deduplicated (undirected twins share one id) by
// construction and max_edges keeps the smallest ids without a sort (small
// neighborhoods sort their few ids directly instead; same output).
// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include "algos.hpp"
#include "boundary.hpp"

namespace mg {

namespace {

inline uint32_t bit_test(const uint32_t* bm, uint32_t i) {  // 0 or 1
  return (bm[i >> 5] >> (i & 31u)) & 1u;
}
inline void bit_set(uint32_t* bm, uint32_t i) { bm[i >> 5] |= 1u << (i & 31u); }

// Parallel granularity, in units of ~one bitmap word scanned (an emitted id
// counts as 4): a chunk below this runs inline, since handing it to another
// thread costs more than scanning a few hundred KB.
constexpr uint64_t k_par_grain = 1u << 17;
// Popcount block for enumeration (prefix granularity).
constexpr std::size_t k_block_words = 2048;
// Edge ids per marking chunk below which the full scan runs inline.
constexpr uint64_t k_mark_grain_edges = 65536;
// Full edge-list scan vs a serial walk of the reached CSR rows: the walk
// costs ~deg_sum = sum(out_deg(reached)); the scan streams every stored edge
// on up to num_threads() threads (memory-bound, so little gain past 8
// threads) after a fixed parallel start-up. Scan when
//   deg_sum >= k_fullscan_fixed + stored * (0.05 + 0.2 / min(threads, 8)),
// fitted on 1/4/8/16 threads (rmat16-rmat20, rmat18u, kg).
constexpr double k_fullscan_fixed = 65536.0;
// Vertex list: when the reached vertices number at most vwords / this, sort
// them directly instead of touching the V-bit user-space bitmap.
constexpr uint64_t k_sparse_sort_div = 64;
// Serial walk: sort the n collected edge ids directly while
// n * log2(n) <= words * ratio; otherwise dedupe through the E_input-bit
// bitmap (zero + mark + popcount enumerate, ~words work). The enumerate
// runs inline up to k_par_grain words and fans out above it, which makes
// the bitmap relatively cheaper there. Ratios measured on 16 cores
// (rmat16-rmat20, rmat18u, kg; k=1/2, out/in/both): sorting wins up to
// ~1.4x words inline and ~0.5x words threaded.
constexpr double k_sort_ratio_inline = 1.4;
constexpr double k_sort_ratio_threaded = 0.5;

bool prefer_full_scan(uint64_t deg_sum, uint64_t stored) {
  const double t = std::min(num_threads(), 8u);
  return double(deg_sum) >=
         k_fullscan_fixed + double(stored) * (0.05 + 0.2 / t);
}

bool prefer_sort(uint64_t n, std::size_t words) {
  if (n < 2) return true;
  const bool threaded = words > k_par_grain && num_threads() > 1;
  const double ratio =
      threaded ? k_sort_ratio_threaded : k_sort_ratio_inline;
  return double(n) * std::log2(double(n)) <= ratio * double(words);
}

// parallel_for grain (in items) giving ~k_par_grain work per chunk.
std::size_t grain_for(std::size_t items, uint64_t work) {
  if (work <= k_par_grain) return items;  // inline
  const uint64_t g = uint64_t(items) * k_par_grain / work;
  return static_cast<std::size_t>(std::max<uint64_t>(1, g));
}

// Writes the set bits of bm[0..words) to `out` in ascending order, keeping
// only the first `cap` when cap != 0. Per-block popcount prefix => the fill
// is parallel and deterministic.
void enumerate_bits(const uint32_t* bm, std::size_t words, uint64_t cap,
                    std::vector<uint32_t>& out) {
  out.clear();
  if (words == 0) return;
  const std::size_t nb = (words + k_block_words - 1) / k_block_words;
  std::vector<uint64_t> off(nb + 1, 0);
  parallel_for(
      nb,
      [&](std::size_t b, std::size_t e) {
        for (std::size_t k = b; k < e; ++k) {
          const std::size_t lo = k * k_block_words;
          const std::size_t hi = std::min(words, lo + k_block_words);
          uint64_t s = 0;
          for (std::size_t w = lo; w < hi; ++w) s += __builtin_popcount(bm[w]);
          off[k + 1] = s;
        }
      },
      grain_for(nb, words));
  for (std::size_t k = 0; k < nb; ++k) off[k + 1] += off[k];
  uint64_t total = off[nb];
  if (cap != 0 && total > cap) total = cap;
  out.resize(static_cast<std::size_t>(total));
  if (total == 0) return;
  uint32_t* o = out.data();
  parallel_for(
      nb,
      [&](std::size_t b, std::size_t e) {
        uint64_t pos = off[b];
        const uint64_t end = std::min<uint64_t>(off[e], total);
        const std::size_t hi = std::min(words, e * k_block_words);
        for (std::size_t w = b * k_block_words; w < hi && pos < end; ++w) {
          for (uint32_t x = bm[w]; x != 0u && pos < end; x &= x - 1u)
            o[pos++] = static_cast<uint32_t>(w * 32u + __builtin_ctz(x));
        }
      },
      grain_for(nb, words + 4 * total));
}

// Full scan: fills ebm (ceil(E_input/32) words, uninitialized on entry)
// with one bit per ORIGINAL input edge id whose stored edge has both
// endpoints in rbm. Edge-parallel over the stored canonical edge list; each
// chunk owns a 32-aligned range of original ids, hence a disjoint word
// range — no atomics. Relies on two Graph::build invariants (graph.hpp):
// directed graphs store edge i at position i (orig id == position);
// undirected graphs store orig ids nondecreasing (twins adjacent).
void mark_full_scan(Graph& g, const uint32_t* rbm, uint32_t* ebm) {
  const uint64_t E = g.E_input;
  const std::size_t words = static_cast<std::size_t>((E + 31u) / 32u);
  const uint64_t stored = g.stored_edges();
  const uint32_t* src = g.edges_src_canon().data();
  const uint32_t* dst = g.edges_dst_canon().data();
  const uint32_t* oid = g.edges_orig_id().data();
  const std::size_t chunks = std::max<std::size_t>(
      1, std::min<std::size_t>(
             k_par_balance * num_threads(),
             static_cast<std::size_t>((E + k_mark_grain_edges - 1) /
                                      k_mark_grain_edges)));
  const std::size_t per = (words + chunks - 1) / chunks;  // words per chunk
  const bool directed = g.directed;
  parallel_for_balanced(  // each word range claimed on its own
      chunks,
      [&](std::size_t b, std::size_t e) {
        for (std::size_t c = b; c < e; ++c) {
          const std::size_t w0 = c * per, w1 = std::min(words, w0 + per);
          if (w0 >= w1) continue;
          if (directed) {  // orig id == stored position; every word written
            for (std::size_t w = w0; w < w1; ++w) {
              const uint64_t i0 = uint64_t(w) * 32u;
              const uint64_t i1 = std::min<uint64_t>(E, i0 + 32u);
              uint32_t m = 0u;
              for (uint64_t i = i0; i < i1; ++i)
                m |= (bit_test(rbm, src[i]) & bit_test(rbm, dst[i]))
                     << (i - i0);
              ebm[w] = m;
            }
          } else {  // orig ids nondecreasing in stored order
            std::fill(ebm + w0, ebm + w1, 0u);
            const uint64_t lo = uint64_t(w0) * 32u;
            const uint64_t hi = std::min<uint64_t>(E, uint64_t(w1) * 32u);
            const std::size_t s0 = static_cast<std::size_t>(
                std::lower_bound(oid, oid + stored, lo) - oid);
            const std::size_t s1 = static_cast<std::size_t>(
                std::lower_bound(oid, oid + stored, hi) - oid);
            for (std::size_t s = s0; s < s1; ++s)
              if (bit_test(rbm, src[s]) && bit_test(rbm, dst[s]))
                bit_set(ebm, oid[s]);
          }
        }
      },
      1);
}

// Induced edges among `reached` (unique canonical ids), as ascending unique
// original input edge ids, truncated to the smallest max_edges when != 0.
// Large neighborhoods: full edge-list scan into a bitmap over original ids.
// Small ones (or no retained edge list): serial walk of the reached rows of
// the OUT CSR via edge_orig; the few ids found are sorted directly, more
// are deduplicated through the bitmap.
std::vector<uint32_t> induced_edges(Graph& g,
                                    const std::vector<uint32_t>& reached,
                                    uint64_t max_edges) {
  std::vector<uint32_t> edges;
  const uint64_t stored = g.stored_edges();
  if (reached.empty() || g.V == 0 || g.E_input == 0 || stored == 0)
    return edges;
  const Orientation& o = g.orientation(Dir::out);
  const uint32_t* row = o.row_offsets->as<uint32_t>();
  std::vector<uint32_t> rbm((std::size_t(g.V) + 31u) / 32u, 0u);
  uint64_t deg_sum = 0;
  for (uint32_t c : reached) {
    bit_set(rbm.data(), c);
    deg_sum += row[c + 1] - row[c];
  }
  if (deg_sum == 0) return edges;
  const std::size_t words = static_cast<std::size_t>((g.E_input + 31u) / 32u);
  const bool have_list = g.edges_src_canon().size() == stored &&
                         g.edges_dst_canon().size() == stored &&
                         g.edges_orig_id().size() == stored;

  if (have_list && prefer_full_scan(deg_sum, stored)) {
    std::unique_ptr<uint32_t[]> ebm(new uint32_t[words]);
    mark_full_scan(g, rbm.data(), ebm.get());
    enumerate_bits(ebm.get(), words, max_edges, edges);
    return edges;
  }

  const uint32_t* col = o.col_indices->as<uint32_t>();
  const uint32_t* eorig = o.edge_orig->as<uint32_t>();
  for (uint32_t v : reached)
    for (uint32_t s = row[v], end = row[v + 1]; s < end; ++s)
      if (bit_test(rbm.data(), col[s])) edges.push_back(eorig[s]);
  if (prefer_sort(edges.size(), words)) {
    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
    if (max_edges != 0 && edges.size() > max_edges)
      edges.resize(static_cast<std::size_t>(max_edges));
    return edges;
  }
  std::unique_ptr<uint32_t[]> ebm(new uint32_t[words]());  // zeroed
  for (uint32_t id : edges) bit_set(ebm.get(), id);
  enumerate_bits(ebm.get(), words, max_edges, edges);
  return edges;
}

// cpu::khop_vertices reports no level count; use the depth budget.
int depth_budget(uint32_t k, uint32_t v) {
  const uint64_t b = std::min<uint64_t>(k, v);
  return static_cast<int>(std::min<uint64_t>(b, 0x7FFFFFFFull));
}

}  // namespace

KhopResult k_hop(Graph& g, const uint32_t* seeds, uint32_t n_seeds,
                 const KhopOpts& o) {
  if (n_seeds > 0 && !seeds) throw_invalid("k_hop: seeds is null");
  const auto t0 = std::chrono::steady_clock::now();
  Runtime& rt = Runtime::instance();
  std::vector<uint32_t> seeds_canon =
      boundary::to_canon(g, seeds, n_seeds, "k_hop seeds");

  ExecPath path = rt.plan(g.stored_edges(), PlanOp::khop);
  const bool caps = o.max_vertices != 0 || o.max_edges != 0;
  if (caps) path = ExecPath::cpu;  // deterministic cap semantics (v0.1)
  if (o.dir == BfsDir::both && g.directed && o.k > 0)
    path = ExecPath::cpu;  // documented v0.1 routing (see gpu::bfs) —
                           // recorded honestly instead of a gpu/cpu hybrid

  KhopResult res;
  std::vector<uint32_t> reached;  // unique canonical ids
  int iterations = 0;

  if (path == ExecPath::cpu) {
    reached = cpu::khop_vertices(g, seeds_canon.data(), n_seeds, o.k, o.dir,
                                 o.max_vertices);
    iterations = depth_budget(o.k, g.V);
  } else {
    if (o.k == 0) {
      reached = seeds_canon;  // depth 0: the seeds themselves
      std::sort(reached.begin(), reached.end());
      reached.erase(std::unique(reached.begin(), reached.end()),
                    reached.end());
    } else {
      std::vector<int32_t> dist(g.V), parent(g.V);
      iterations = gpu::bfs(g, seeds_canon.data(), n_seeds, o.dir,
                            dist.data(), parent.data(), o.k);
      reached.reserve(seeds_canon.size());
      for (uint32_t c = 0; c < g.V; ++c)
        if (dist[c] != -1) reached.push_back(c);
    }
  }

  // Shared output contract: edges = unique original ids ascending, cap
  // keeps the smallest; vertices = user indices ascending (few are sorted
  // directly, more go through a bitmap over V in user space).
  res.edges = induced_edges(g, reached, o.max_edges);
  const std::size_t vwords = (std::size_t(g.V) + 31u) / 32u;
  if (uint64_t(reached.size()) * k_sparse_sort_div <= vwords) {
    res.vertices.resize(reached.size());
    for (std::size_t i = 0; i < reached.size(); ++i)
      res.vertices[i] = g.user_of_canon[reached[i]];
    std::sort(res.vertices.begin(), res.vertices.end());
  } else {
    std::vector<uint32_t> ubm(vwords, 0u);
    for (uint32_t c : reached) bit_set(ubm.data(), g.user_of_canon[c]);
    enumerate_bits(ubm.data(), vwords, 0, res.vertices);
  }

  const double ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0)
                        .count();
  RunInfo info;
  info.op = "k_hop";
  info.path = path;
  info.iterations = iterations;
  info.elapsed_ms = ms;
  rt.record_run(info);
  return res;
}

}  // namespace mg
