// pagerank.cpp — mg::pagerank / mg::ppr_topk entry points plus the GPU
// drivers mg::gpu::pagerank and mg::gpu::ppr (batched tiles of MG_PPR_TILE).
//
// Iteration-count semantics (both paths, tests may rely on this):
//   * Convergence is evaluated ONLY at audit boundaries: after every
//     audit_interval iterations (MG_PR_AUDIT_INTERVAL, default 5) and at
//     max_iter. The returned count is the first boundary where
//     L1(r_n - r_{n-1}) < V * tol (fp64 audit), else max_iter.
//   * ppr: per-lane counts freeze at the boundary where that lane
//     converged; the function returns the MAX over all queries.
//   * V == 0 returns 0 iterations.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "../engines/reduce_engine.hpp"
#include "../engines/spmv_engine.hpp"
#include "../kernels/mg_params.h"
#include "algos.hpp"
#include "boundary.hpp"

namespace mg {
namespace {

// Fixed audit chunk => chunk partials independent of MG_THREADS; the
// sequential sum over chunk results makes the fp64 audit deterministic.
constexpr std::size_t kAuditChunk = std::size_t{1} << 16;

int audit_interval_env() {
  // Default 5: measured winner of the interleaved cadence sweep on both
  // canonical shapes (M4 Max, 2026-07-28) — RMAT-18 queries converge by
  // iteration ~5 and the KG shape's slowest lanes by ~20, so boundaries at
  // multiples of 5 waste the fewest post-convergence iterations (8 ran
  // 24/8 iterations where 5 runs 20/5) at a modest sync-count cost.
  long v = env_long("MG_PR_AUDIT_INTERVAL", 5);
  return v < 1 ? 1 : static_cast<int>(v);
}

double elapsed_ms(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - t0)
      .count();
}

// Deterministic fp64 L1 distance of two float vectors.
double l1_diff(const float* a, const float* b, uint32_t n) {
  const std::size_t chunks = (n + kAuditChunk - 1) / kAuditChunk;
  std::vector<double> part(chunks, 0.0);
  parallel_for(
      chunks,
      [&](std::size_t cb, std::size_t ce) {
        for (std::size_t c = cb; c < ce; ++c) {
          const std::size_t lo = c * kAuditChunk;
          const std::size_t hi = std::min<std::size_t>(n, lo + kAuditChunk);
          double s = 0.0;
          for (std::size_t i = lo; i < hi; ++i)
            s += std::fabs(static_cast<double>(a[i]) -
                           static_cast<double>(b[i]));
          part[c] = s;
        }
      },
      1);
  double total = 0.0;
  for (double v : part) total += v;
  return total;
}

// Per-lane deterministic fp64 L1 over vertex-major interleaved tile state
// (state[v * MG_PPR_TILE + lane]); only lanes in `mask` are computed.
void l1_diff_lanes(const float* a, const float* b, uint32_t v_count,
                   uint32_t mask, double err[MG_PPR_TILE]) {
  const std::size_t chunks = (v_count + kAuditChunk - 1) / kAuditChunk;
  std::vector<std::array<double, MG_PPR_TILE>> part(
      chunks, std::array<double, MG_PPR_TILE>{});
  parallel_for(
      chunks,
      [&](std::size_t cb, std::size_t ce) {
        for (std::size_t c = cb; c < ce; ++c) {
          const std::size_t lo = c * kAuditChunk;
          const std::size_t hi =
              std::min<std::size_t>(v_count, lo + kAuditChunk);
          std::array<double, MG_PPR_TILE> s{};
          for (std::size_t v = lo; v < hi; ++v) {
            const std::size_t base = v * MG_PPR_TILE;
            for (uint32_t l = 0; l < MG_PPR_TILE; ++l) {
              if ((mask >> l) & 1u)
                s[l] += std::fabs(static_cast<double>(a[base + l]) -
                                  static_cast<double>(b[base + l]));
            }
          }
          part[c] = s;
        }
      },
      1);
  for (uint32_t l = 0; l < MG_PPR_TILE; ++l) err[l] = 0.0;
  for (const auto& s : part)
    for (uint32_t l = 0; l < MG_PPR_TILE; ++l) err[l] += s[l];
}

// GPU single-query PageRank. uniform_p comes from the caller (the entry
// point knows personalization == nullptr); when true the pvec buffer slot
// gets the dummy and kernels use MGPrParams.p_uniform via MG_FC_UNIFORM_P.
int gpu_pagerank_impl(Graph& g, const double* pvec_canon, double alpha,
                      double tol, int max_iter, int audit_interval,
                      double* rank_canon, bool uniform_p) {
  Runtime& rt = Runtime::instance();
  const uint32_t V = g.V;
  if (V == 0) return 0;
  if (audit_interval < 1) audit_interval = 1;
  const Orientation& in_o = g.orientation(Dir::in);

  SpmvEngine spmv(rt, g.weighted, uniform_p, /*batched=*/false);
  ReduceEngine reduce(rt, /*batched=*/false);

  BufferPtr dummy = rt.dummy();
  BufferPtr rank[2] = {rt.alloc(std::size_t{V} * 4, "pr.rank0"),
                       rt.alloc(std::size_t{V} * 4, "pr.rank1")};
  BufferPtr contrib = rt.alloc(std::size_t{V} * 4, "pr.contrib");
  BufferPtr scalars = rt.alloc(MG_SCALARS_LEN * 4, "pr.iter_scalars");
  BufferPtr pvec_buf = dummy;
  if (!uniform_p) {
    pvec_buf = rt.alloc(std::size_t{V} * 4, "pr.pvec");
    float* pf = pvec_buf->as<float>();
    parallel_for(V, [&](std::size_t b, std::size_t e) {
      for (std::size_t i = b; i < e; ++i)
        pf[i] = static_cast<float>(pvec_canon[i]);
    });
  }
  BufferPtr partials = dummy;
  if (g.n_dangling)
    partials =
        rt.alloc(std::size_t{ReduceEngine::n_blocks(g.n_dangling)} * 4,
                 "pr.dangling_partials");

  // Host init: rank0 = pvec (fp32); iter_scalars[MG_SCALAR_DANGLING] =
  // dangling mass of rank0, summed on the host in fp64. When n_dangling==0
  // the slot stays 0 (alloc zero-fills) and dangling kernels never run.
  float* r0 = rank[0]->as<float>();
  parallel_for(V, [&](std::size_t b, std::size_t e) {
    for (std::size_t i = b; i < e; ++i)
      r0[i] = static_cast<float>(pvec_canon[i]);
  });
  if (g.n_dangling) {
    const uint32_t* dl = g.dangling_list->as<uint32_t>();
    double d = 0.0;
    for (uint32_t i = 0; i < g.n_dangling; ++i) d += pvec_canon[dl[i]];
    scalars->as<float>()[MG_SCALAR_DANGLING] = static_cast<float>(d);
  }

  MGPrParams base{};
  base.alpha = static_cast<float>(alpha);
  base.one_minus_alpha = static_cast<float>(1.0 - alpha);
  base.p_uniform = static_cast<float>(1.0 / static_cast<double>(V));
  base.v_count = V;

  Buffer* weights =
      (g.weighted && in_o.weights) ? in_o.weights.get() : dummy.get();

  int cur = 0;
  int done = 0;
  while (done < max_iter) {
    const int k = boundary::next_audit_batch(done, max_iter, audit_interval);
    CommandBatch cb(rt);
    for (int i = 0; i < k; ++i) {
      spmv.encode_iteration(cb, in_o, base, rank[cur].get(), contrib.get(),
                            rank[cur ^ 1].get(), scalars.get(),
                            pvec_buf.get(), g.out_weight_sum.get(), weights);
      reduce.encode_list_sum(cb, base, g.n_dangling, g.dangling_list.get(),
                             rank[cur ^ 1].get(), partials.get(),
                             scalars.get(), MG_SCALAR_DANGLING);
      cur ^= 1;
    }
    cb.commit_and_wait();
    done += k;
    const double err =
        l1_diff(rank[cur]->as<float>(), rank[cur ^ 1]->as<float>(), V);
    if (err < static_cast<double>(V) * tol) break;
  }

  const float* rc = rank[cur]->as<float>();
  parallel_for(V, [&](std::size_t b, std::size_t e) {
    for (std::size_t i = b; i < e; ++i)
      rank_canon[i] = static_cast<double>(rc[i]);
  });
  return done;
}

}  // namespace

// ---------------------------------------------------------------------------
// GPU drivers (canonical space)
// ---------------------------------------------------------------------------
namespace gpu {

int pagerank(Graph& g, const double* pvec_canon, double alpha, double tol,
             int max_iter, int audit_interval, double* rank_canon) {
  // The entry point passes uniformity explicitly (personalization ==
  // nullptr). External callers of gpu::pagerank get exact-value detection:
  // boundary::pvec_to_canon fills the uniform vector with bit-identical
  // 1/V entries, and a personalization that equals uniform exactly is
  // semantically identical under MG_FC_UNIFORM_P anyway.
  bool uniform = true;
  if (g.V) {
    const double u = 1.0 / static_cast<double>(g.V);
    for (uint32_t i = 0; i < g.V; ++i) {
      if (pvec_canon[i] != u) {
        uniform = false;
        break;
      }
    }
  }
  return gpu_pagerank_impl(g, pvec_canon, alpha, tol, max_iter,
                           audit_interval, rank_canon, uniform);
}

// Shared tile loop: iterate every tile to convergence/budget and hand the
// FINAL interleaved tile state to on_tile(q0, lanes, rank_final). Returns
// the max iteration count over all queries. Callers layer their own
// extraction (sink deinterleave, or GPU top-k select) on top.
static int ppr_tiles(Graph& g, const uint32_t* seeds_canon,
                     const float* seed_w, const uint64_t* offsets,
                     uint32_t n_queries, double alpha, double tol,
                     int max_iter, int audit_interval,
                     const std::function<void(uint32_t, uint32_t, Buffer*)>&
                         on_tile) {
  Runtime& rt = Runtime::instance();
  const uint32_t V = g.V;
  if (n_queries == 0 || V == 0) return 0;
  if (audit_interval < 1) audit_interval = 1;
  const Orientation& in_o = g.orientation(Dir::in);

  // Working-set guard: tile state is 3 * V * MG_PPR_TILE fp32 (rank x2 +
  // contrib) on top of the resident IN-orientation CSR.
  if (rt.has_gpu()) {
    const uint64_t budget = rt.device()->recommendedMaxWorkingSetSize();
    const uint64_t need =
        3ull * V * MG_PPR_TILE * sizeof(float) +
        (uint64_t{V} + 1) * sizeof(uint32_t) +
        in_o.e_stored * sizeof(uint32_t) +
        (g.weighted ? in_o.e_stored * sizeof(float) : 0ull) +
        uint64_t{V} * sizeof(float);
    if (budget && need > budget)
      throw Error(
          ErrorCode::invalid_argument,
          "ppr_topk: graph working set (" + std::to_string(need) +
              " bytes: CSR + fixed 8-lane tile state, ~100 B/vertex) exceeds "
              "the GPU budget (recommendedMaxWorkingSetSize = " +
              std::to_string(budget) +
              "). Batch size does not affect this; use "
              "set_execution(\"cpu\") or MG_FORCE_CPU=1.");
  }

  SpmvEngine spmv(rt, g.weighted, /*uniform_p=*/false, /*batched=*/true);
  ReduceEngine reduce(rt, /*batched=*/true);

  const std::size_t state = std::size_t{V} * MG_PPR_TILE;
  BufferPtr dummy = rt.dummy();
  BufferPtr rank_b[2] = {rt.alloc(state * 4, "ppr.rank0"),
                         rt.alloc(state * 4, "ppr.rank1")};
  BufferPtr contrib_b = rt.alloc(state * 4, "ppr.contrib");
  BufferPtr scalars_b = rt.alloc(MG_PPR_TILE * 4, "ppr.iter_scalars");
  BufferPtr partials_b = dummy;
  if (g.n_dangling)
    partials_b = rt.alloc(std::size_t{ReduceEngine::n_blocks(g.n_dangling)} *
                              MG_PPR_TILE * 4,
                          "ppr.dangling_partials");

  const uint32_t n_tiles = ceil_div_u32(n_queries, MG_PPR_TILE);
  uint64_t max_tile_seeds = 1;
  for (uint32_t t = 0; t < n_tiles; ++t) {
    const uint32_t q0 = t * MG_PPR_TILE;
    const uint32_t q1 = std::min(n_queries, q0 + MG_PPR_TILE);
    max_tile_seeds = std::max(max_tile_seeds, offsets[q1] - offsets[q0]);
  }
  BufferPtr seed_vertex = rt.alloc(max_tile_seeds * 4, "ppr.seed_vertex");
  BufferPtr seed_lane = rt.alloc(max_tile_seeds * 4, "ppr.seed_lane");
  BufferPtr seed_weight = rt.alloc(max_tile_seeds * 4, "ppr.seed_weight");

  Buffer* weights =
      (g.weighted && in_o.weights) ? in_o.weights.get() : dummy.get();
  const float* ows = g.out_weight_sum->as<float>();

  int max_iters = 0;
  for (uint32_t t = 0; t < n_tiles; ++t) {
    const uint32_t q0 = t * MG_PPR_TILE;
    const uint32_t lanes = std::min<uint32_t>(MG_PPR_TILE, n_queries - q0);
    const uint64_t s0 = offsets[q0];
    const uint32_t seed_total =
        static_cast<uint32_t>(offsets[q0 + lanes] - s0);

    // Host init: rank0 = per-lane seed scatter (seeds are deduped per
    // (vertex, lane) => plain assignment); per-lane dangling seed mass in
    // fp64. Unused lanes stay zero and are masked out.
    float* r0 = rank_b[0]->as<float>();
    std::memset(r0, 0, state * 4);
    double dmass[MG_PPR_TILE] = {0.0};
    uint32_t* sv = seed_vertex->as<uint32_t>();
    uint32_t* sl = seed_lane->as<uint32_t>();
    float* sw = seed_weight->as<float>();
    for (uint32_t q = q0; q < q0 + lanes; ++q) {
      const uint32_t lane = q - q0;
      for (uint64_t i = offsets[q]; i < offsets[q + 1]; ++i) {
        const uint32_t v = seeds_canon[i];
        const float w = seed_w[i];
        const uint64_t j = i - s0;
        sv[j] = v;
        sl[j] = lane;
        sw[j] = w;
        r0[std::size_t{v} * MG_PPR_TILE + lane] = w;
        if (ows[v] == 0.0f) dmass[lane] += static_cast<double>(w);
      }
    }
    float* sc = scalars_b->as<float>();
    for (uint32_t l = 0; l < MG_PPR_TILE; ++l)
      sc[l] = static_cast<float>(dmass[l]);

    MGPprParams base{};
    base.alpha = static_cast<float>(alpha);
    base.one_minus_alpha = static_cast<float>(1.0 - alpha);
    base.v_count = V;
    base.seed_total = seed_total;

    uint32_t active = (1u << lanes) - 1u;
    int iters[MG_PPR_TILE] = {0};
    int done = 0;
    int cur = 0;
    while (active && done < max_iter) {
      const int k =
          boundary::next_audit_batch(done, max_iter, audit_interval);
      base.active_mask = active;
      CommandBatch cb(rt);
      for (int i = 0; i < k; ++i) {
        spmv.encode_iteration_b(cb, in_o, base, rank_b[cur].get(),
                                contrib_b.get(), rank_b[cur ^ 1].get(),
                                scalars_b.get(), g.out_weight_sum.get(),
                                weights, seed_vertex.get(), seed_lane.get(),
                                seed_weight.get());
        reduce.encode_list_sum_b(cb, base, g.n_dangling,
                                 g.dangling_list.get(), rank_b[cur ^ 1].get(),
                                 partials_b.get(), scalars_b.get());
        cur ^= 1;
      }
      cb.commit_and_wait();
      done += k;
      double err[MG_PPR_TILE];
      l1_diff_lanes(rank_b[cur]->as<float>(), rank_b[cur ^ 1]->as<float>(),
                    V, active, err);
      for (uint32_t l = 0; l < lanes; ++l) {
        if (((active >> l) & 1u) && err[l] < static_cast<double>(V) * tol) {
          active &= ~(1u << l);
          iters[l] = done;
        }
      }
    }
    for (uint32_t l = 0; l < lanes; ++l) {
      if ((active >> l) & 1u) iters[l] = done;
      max_iters = std::max(max_iters, iters[l]);
    }

    // Frozen lanes survived the ping-pong via copy-through, so rank_b[cur]
    // holds every lane's value at its freeze boundary.
    on_tile(q0, lanes, rank_b[cur].get());
  }
  return max_iters;
}

int ppr(Graph& g, const uint32_t* seeds_canon, const float* seed_w,
        const uint64_t* offsets, uint32_t n_queries, double alpha, double tol,
        int max_iter, int audit_interval,
        const std::function<void(uint32_t, const float*)>& sink) {
  // One V-length scratch reused for every lane copy-out: each query's final
  // vector is deinterleaved into it and handed to the sink immediately, so
  // dense state never exceeds the tile buffers + this single vector.
  std::vector<float> lane_scratch(g.V);
  return ppr_tiles(
      g, seeds_canon, seed_w, offsets, n_queries, alpha, tol, max_iter,
      audit_interval,
      [&](uint32_t q0, uint32_t lanes, Buffer* rank_final) {
        const float* rc = rank_final->as<float>();
        for (uint32_t l = 0; l < lanes; ++l) {
          float* dst = lane_scratch.data();
          parallel_for(g.V, [&](std::size_t b, std::size_t e) {
            for (std::size_t v = b; v < e; ++v)
              dst[v] = rc[v * MG_PPR_TILE + l];
          });
          sink(q0 + l, dst);  // sequential, per the gpu::ppr contract
        }
      });
}

// GPU radix-select path (plan M3): the CPU oracle stays behind the same
// API; eligibility keeps degenerate shapes (tiny V, k ~ V, huge k) on it.
bool topk_select_eligible(uint32_t V, uint32_t k) {
  return env_flag("MG_GPU_TOPK", true) && V >= 4096 && k <= 4096 && k < V;
}

int ppr_topk_select(Graph& g, const uint32_t* seeds_canon, const float* seed_w,
                    const uint64_t* offsets, uint32_t n_queries, double alpha,
                    double tol, int max_iter, int audit_interval, uint32_t k,
                    int32_t* out_ids, float* out_scores) {
  Runtime& rt = Runtime::instance();
  const uint32_t V = g.V;
  const uint32_t cap = k + MG_TOPK_TIE_SLACK;
  MTL::ComputePipelineState* pso_thresh = rt.pipeline("mg_topk_threshold_b");
  MTL::ComputePipelineState* pso_compact = rt.pipeline("mg_topk_compact_b");
  BufferPtr thresholds = rt.alloc(MG_PPR_TILE * 4, "topk.thresholds");
  BufferPtr counters = rt.alloc(MG_PPR_TILE * 4, "topk.counters");
  BufferPtr cands =
      rt.alloc(std::size_t{MG_PPR_TILE} * cap * 4, "topk.candidates");

  return ppr_tiles(
      g, seeds_canon, seed_w, offsets, n_queries, alpha, tol, max_iter,
      audit_interval,
      [&](uint32_t q0, uint32_t lanes, Buffer* rank_final) {
        MGTopkParams tp{V, k, cap, (1u << lanes) - 1u};
        std::memset(counters->data(), 0, MG_PPR_TILE * 4);
        CommandBatch cb(rt);
        cb.dispatch(pso_thresh, &tp, sizeof(tp),
                    {rank_final, thresholds.get()}, MG_PPR_TILE, MG_TG_SIZE);
        cb.dispatch(pso_compact, &tp, sizeof(tp),
                    {rank_final, thresholds.get(), counters.get(),
                     cands.get()},
                    ceil_div_u32(V, MG_TG_SIZE), MG_TG_SIZE);
        cb.commit_and_wait();

        const float* rc = rank_final->as<float>();
        const uint32_t* cnt = counters->as<uint32_t>();
        const uint32_t* cd = cands->as<uint32_t>();
        parallel_for(
            lanes,
            [&](std::size_t lb, std::size_t le) {
              for (std::size_t l = lb; l < le; ++l) {
                int32_t* row_ids = out_ids + std::size_t{q0 + l} * k;
                float* row_sc = out_scores + std::size_t{q0 + l} * k;
                const uint32_t n = cnt[l];
                if (n > cap) {
                  // Degenerate tie flood (e.g. all-equal scores): the CPU
                  // oracle finishes this lane, reading the tile state
                  // strided in place.
                  boundary::topk_user(g, rc, MG_PPR_TILE,
                                      static_cast<uint32_t>(l), k, row_ids,
                                      row_sc);
                  continue;
                }
                // n >= k by construction (count of scores >= k-th largest).
                std::vector<std::pair<float, uint32_t>> items(n);
                const uint32_t* lane_cd = cd + std::size_t{l} * cap;
                for (uint32_t i = 0; i < n; ++i) {
                  const uint32_t v = lane_cd[i];
                  items[i] = {rc[std::size_t{v} * MG_PPR_TILE + l],
                              g.user_of_canon[v]};
                }
                auto cmp = [](const std::pair<float, uint32_t>& a,
                              const std::pair<float, uint32_t>& b) {
                  if (a.first != b.first) return a.first > b.first;
                  return a.second < b.second;
                };
                std::partial_sort(items.begin(), items.begin() + k,
                                  items.end(), cmp);
                for (uint32_t i = 0; i < k; ++i) {
                  row_ids[i] = static_cast<int32_t>(items[i].second);
                  row_sc[i] = items[i].first;
                }
              }
            },
            /*grain=*/1);
      });
}

}  // namespace gpu

// ---------------------------------------------------------------------------
// Entry points (USER space)
// ---------------------------------------------------------------------------

int pagerank(Graph& g, const PagerankOpts& o, float* out_rank) {
  if (!(o.alpha > 0.0 && o.alpha < 1.0))
    throw_invalid("pagerank: alpha must be in (0, 1)");
  if (!(o.tol > 0.0)) throw_invalid("pagerank: tol must be > 0");
  if (o.max_iter < 1) throw_invalid("pagerank: max_iter must be >= 1");

  const auto t0 = std::chrono::steady_clock::now();
  Runtime& rt = Runtime::instance();

  RunInfo ri;
  ri.op = "pagerank";
  if (g.V == 0) {  // nothing to rank; skip planner + pvec validation
    ri.path = ExecPath::cpu;
    ri.iterations = 0;
    ri.elapsed_ms = elapsed_ms(t0);
    rt.record_run(ri);
    return 0;
  }
  if (!out_rank) throw_invalid("pagerank: out_rank is null");

  const std::vector<double> pvec = boundary::pvec_to_canon(g, o.personalization);
  const ExecPath path = rt.plan(g.stored_edges());
  const int aud = audit_interval_env();

  std::vector<double> rank_canon(g.V, 0.0);
  int iters = 0;
  if (path == ExecPath::gpu) {
    iters = gpu_pagerank_impl(g, pvec.data(), o.alpha, o.tol, o.max_iter, aud,
                              rank_canon.data(),
                              o.personalization == nullptr);
  } else {
    iters = cpu::pagerank(g, pvec.data(), o.alpha, o.tol, o.max_iter, aud,
                          rank_canon.data());
  }
  boundary::canon_to_user(g, rank_canon.data(), out_rank);

  ri.path = path;
  ri.iterations = iters;
  ri.elapsed_ms = elapsed_ms(t0);
  rt.record_run(ri);
  return iters;
}

int ppr_topk(Graph& g, const uint32_t* seeds, const float* seed_weights,
             const uint64_t* offsets, uint32_t n_queries,
             const PprTopkOpts& o, int32_t* out_ids, float* out_scores) {
  if (!(o.alpha > 0.0 && o.alpha < 1.0))
    throw_invalid("ppr_topk: alpha must be in (0, 1)");
  if (!(o.tol > 0.0)) throw_invalid("ppr_topk: tol must be > 0");
  if (o.max_iter < 1) throw_invalid("ppr_topk: max_iter must be >= 1");
  if (o.k < 1) throw_invalid("ppr_topk: k must be >= 1");

  const auto t0 = std::chrono::steady_clock::now();
  Runtime& rt = Runtime::instance();
  RunInfo ri;
  ri.op = "ppr_topk";

  if (n_queries == 0) {
    ri.path = ExecPath::cpu;
    ri.iterations = 0;
    ri.elapsed_ms = elapsed_ms(t0);
    rt.record_run(ri);
    return 0;
  }
  if (!seeds || !offsets)
    throw_invalid("ppr_topk: seeds/offsets must not be null");
  if (!out_ids || !out_scores)
    throw_invalid("ppr_topk: output buffers must not be null");

  if (g.V == 0) {  // every row is padding
    const std::size_t total = std::size_t{n_queries} * o.k;
    std::fill(out_ids, out_ids + total, int32_t{-1});
    std::fill(out_scores, out_scores + total, 0.0f);
    ri.path = ExecPath::cpu;
    ri.iterations = 0;
    ri.elapsed_ms = elapsed_ms(t0);
    rt.record_run(ri);
    return 0;
  }

  // Canonicalize once for both paths: validate, map to canonical ids,
  // dedupe per query (weights summed), normalize per query. Deduped seeds
  // are emitted in ascending canonical id (deterministic).
  std::vector<uint32_t> s_canon;
  std::vector<float> s_w;
  std::vector<uint64_t> offs(std::size_t{n_queries} + 1, 0);
  {
    std::vector<std::pair<uint32_t, double>> items;
    for (uint32_t q = 0; q < n_queries; ++q) {
      const uint64_t b = offsets[q], e = offsets[q + 1];
      if (e < b) throw_invalid("ppr_topk: offsets must be non-decreasing");
      if (e == b)
        throw_invalid("ppr_topk: each query needs at least one seed");
      items.clear();
      items.reserve(static_cast<std::size_t>(e - b));
      double sum = 0.0;
      for (uint64_t i = b; i < e; ++i) {
        if (seeds[i] >= g.V)
          throw_invalid("ppr_topk: seed vertex index out of range");
        const double w =
            seed_weights ? static_cast<double>(seed_weights[i]) : 1.0;
        if (!(w >= 0.0) || !std::isfinite(w))
          throw_invalid("ppr_topk: seed weights must be finite and >= 0");
        sum += w;
        items.emplace_back(g.canon_of_user[seeds[i]], w);
      }
      if (!(sum > 0.0) || !std::isfinite(sum))
        throw_invalid("ppr_topk: seed weights must sum to > 0 per query");
      std::sort(items.begin(), items.end(),
                [](const std::pair<uint32_t, double>& a,
                   const std::pair<uint32_t, double>& b2) {
                  return a.first < b2.first;
                });
      for (std::size_t i = 0; i < items.size();) {
        const uint32_t v = items[i].first;
        double w = 0.0;
        while (i < items.size() && items[i].first == v) {
          w += items[i].second;
          ++i;
        }
        s_canon.push_back(v);
        s_w.push_back(static_cast<float>(w / sum));
      }
      offs[q + 1] = s_canon.size();
    }
  }

  const ExecPath path = rt.plan(g.stored_edges());
  const int aud = audit_interval_env();

  // Streaming top-k selection (deterministic ties): the per-query sink runs
  // boundary::topk_user on each FINAL dense vector as it becomes available,
  // so peak dense state is one GPU tile (plus one V-length scratch) or one
  // in-flight fp64 vector per CPU worker — never B x V.
  // Concurrency: gpu::ppr calls the sink sequentially from this thread;
  // cpu::ppr workers call it concurrently for DISTINCT q (chosen over a
  // mutex: topk_user is pure and each call writes only its own disjoint
  // output rows, so no serialization is needed).
  int iters = 0;
  if (path == ExecPath::gpu && gpu::topk_select_eligible(g.V, o.k)) {
    // GPU radix-select (plan M3): O(V) threshold + compaction on the GPU,
    // exact tie-rule finish on a ~k-sized candidate set. Bit-identical to
    // the CPU selection oracle (tests assert it); MG_GPU_TOPK=0 forces the
    // oracle.
    iters = gpu::ppr_topk_select(g, s_canon.data(), s_w.data(), offs.data(),
                                 n_queries, o.alpha, o.tol, o.max_iter, aud,
                                 o.k, out_ids, out_scores);
  } else if (path == ExecPath::gpu) {
    iters = gpu::ppr(g, s_canon.data(), s_w.data(), offs.data(), n_queries,
                     o.alpha, o.tol, o.max_iter, aud,
                     [&](uint32_t q, const float* rank) {
                       boundary::topk_user(g, rank, 1u, 0u, o.k,
                                           out_ids + std::size_t{q} * o.k,
                                           out_scores + std::size_t{q} * o.k);
                     });
  } else {
    iters = cpu::ppr(g, s_canon.data(), s_w.data(), offs.data(), n_queries,
                     o.alpha, o.tol, o.max_iter, aud,
                     [&](uint32_t q, const double* rank) {
                       boundary::topk_user(g, rank, 1u, 0u, o.k,
                                           out_ids + std::size_t{q} * o.k,
                                           out_scores + std::size_t{q} * o.k);
                     });
  }

  ri.path = path;
  ri.iterations = iters;
  ri.elapsed_ms = elapsed_ms(t0);
  rt.record_run(ri);
  return iters;
}

}  // namespace mg
