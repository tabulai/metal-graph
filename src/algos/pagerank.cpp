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
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
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

// Deterministic fp64 sum of p over a vertex list `list[0..n)`.
double p_list_sum(const Graph& g, const uint32_t* list, uint32_t n,
                  const double* p_canon, const float* p_user, double p_sum) {
  if (n == 0) return 0.0;
  const uint32_t* user_of = g.user_of_canon.data();
  const std::size_t chunks = (std::size_t{n} + kAuditChunk - 1) / kAuditChunk;
  if (chunks <= 1) {
    double d = 0.0;
    if (p_canon) {
      for (uint32_t i = 0; i < n; ++i) d += p_canon[list[i]];
    } else {
      for (uint32_t i = 0; i < n; ++i)
        d += static_cast<double>(p_user[user_of[list[i]]]) / p_sum;
    }
    return d;
  }
  std::vector<double> part(chunks, 0.0);
  parallel_for(
      chunks,
      [&](std::size_t cb, std::size_t ce) {
        for (std::size_t c = cb; c < ce; ++c) {
          const std::size_t lo = c * kAuditChunk;
          const std::size_t hi = std::min<std::size_t>(n, lo + kAuditChunk);
          double s = 0.0;
          if (p_canon) {
            for (std::size_t i = lo; i < hi; ++i) s += p_canon[list[i]];
          } else {
            for (std::size_t i = lo; i < hi; ++i)
              s += static_cast<double>(p_user[user_of[list[i]]]) / p_sum;
          }
          part[c] = s;
        }
      },
      1);
  double total = 0.0;
  for (double x : part) total += x;
  return total;
}

// Teleport vector p (also iterate 0) of one GPU PageRank solve, in exactly
// one of three forms. The fp32 values the GPU sees are identical in all of
// them: static_cast<float>(p_canon[c]) with p_canon as built by
// boundary::pvec_to_canon (uniform: 1/V; USER form: p_user[u] / p_user_sum).
struct PrTeleport {
  bool uniform = false;           // p = 1/V (kernels use MG_FC_UNIFORM_P)
  const double* canon = nullptr;  // canonical fp64 p, or
  const float* user = nullptr;    // validated USER-order weights, with
  double user_sum = 0.0;          // boundary::personalization_sum(user)
};

// GPU single-query PageRank. emit(buf, rescaled) receives the FINAL fp32
// iterate (canonical order, length V) while device buffers are still alive;
// when rescaled == true, non-dangling entries hold rank / out_weight_sum.
int gpu_pagerank_impl(Graph& g, const PrTeleport& p, double alpha, double tol,
                      int max_iter, int audit_interval,
                      const std::function<void(const float*, bool)>& emit) {
  Runtime& rt = Runtime::instance();
  const uint32_t V = g.V;
  if (V == 0) return 0;
  if (audit_interval < 1) audit_interval = 1;
  const Orientation& in_o = g.orientation(Dir::in);
  const GpuGatherPlan& plan = g.gpu_gather_plan(Dir::in);
  const bool rescale = plan.ows_safe_rescale;

  SpmvEngine spmv(rt, g.weighted, p.uniform, /*batched=*/false);
  ReduceEngine reduce(rt, /*batched=*/false);
  MTL::ComputePipelineState* pso_init_u =
      p.uniform ? rt.pipeline("mg_pr_init_uniform_contrib") : nullptr;
  MTL::ComputePipelineState* pso_init_p =
      (!p.uniform && rescale) ? rt.pipeline("mg_pr_init_pvec_contrib")
                              : nullptr;

  BufferPtr dummy = rt.dummy();
  BufferPtr buf[2] = {rt.alloc_scratch(std::size_t{V} * 4, "pr.buf0"),
                      rt.alloc_scratch(std::size_t{V} * 4, "pr.buf1")};
  BufferPtr contrib =
      rescale ? dummy : rt.alloc_scratch(std::size_t{V} * 4, "pr.contrib");
  BufferPtr scalars = rt.alloc_scratch(MG_SCALARS_LEN * 4, "pr.iter_scalars");
  BufferPtr pvec_buf = dummy;
  BufferPtr dang_partials =
      (plan.n_active_dangling > ReduceEngine::kSingleTgMax)
          ? rt.alloc_scratch(
                std::size_t{ReduceEngine::n_blocks(plan.n_active_dangling)} * 4,
                "pr.dangling_partials")
          : dummy;
  const uint32_t n_active_v = plan.off_iso;
  BufferPtr res_partials =
      n_active_v
          ? rt.alloc_scratch(
                std::size_t{ReduceEngine::n_blocks(n_active_v)} * 4,
                "pr.residual_partials")
          : dummy;

  const float* p_user = p.user;
  const double p_sum = p.user_sum;
  const uint32_t* user_of = g.user_of_canon.data();
  Buffer* cur = buf[0].get();
  if (!p.uniform) {
    pvec_buf = rt.alloc_scratch(std::size_t{V} * 4, "pr.pvec");
    float* pf = pvec_buf->as<float>();
    const double* pc = p.canon;
    parallel_for(V, [&](std::size_t b, std::size_t e) {
      if (pc) {
        for (std::size_t i = b; i < e; ++i) pf[i] = static_cast<float>(pc[i]);
      } else {
        for (std::size_t i = b; i < e; ++i)
          pf[i] = static_cast<float>(static_cast<double>(p_user[user_of[i]]) /
                                     p_sum);
      }
    });
    if (!rescale) cur = pvec_buf.get();
  }

  double d0 = 0.0;
  double iso_p_sum = 0.0;
  if (g.n_dangling) {
    if (p.uniform) {
      d0 = static_cast<double>(g.n_dangling) / static_cast<double>(V);
      iso_p_sum = static_cast<double>(plan.n_iso) / static_cast<double>(V);
    } else {
      d0 = p_list_sum(g, g.dangling_list->as<uint32_t>(), g.n_dangling, p.canon,
                      p_user, p_sum);
      if (plan.n_iso) {
        iso_p_sum =
            p_list_sum(g, plan.vertices->as<uint32_t>() + plan.off_iso,
                       plan.n_iso, p.canon, p_user, p_sum);
      }
    }
  }
  float* sc = scalars->as<float>();
  sc[MG_SCALAR_DANGLING] = static_cast<float>(d0);
  sc[MG_SCALAR_ISO_SCALE] = 1.0f;
  sc[MG_SCALAR_ISO_SCALE_PREV] = 1.0f;
  sc[3] = 0.0f;

  MGPrParams base{};
  base.alpha = static_cast<float>(alpha);
  base.one_minus_alpha = static_cast<float>(1.0 - alpha);
  base.p_uniform = static_cast<float>(1.0 / static_cast<double>(V));
  base.v_count = V;

  Buffer* weights =
      (g.weighted && in_o.weights) ? in_o.weights.get() : dummy.get();

  Buffer* prev = nullptr;
  int next = (cur == buf[0].get()) ? 1 : 0;
  int done = 0;
  while (done < max_iter) {
    const int k = boundary::next_audit_batch(done, max_iter, audit_interval);
    CommandBatch cb(rt);
    if (done == 0) {
      if (p.uniform) {
        MGPrParams ip = base;
        ip.count = V;
        ip.pad0 = rescale ? 1u : 0u;
        cb.dispatch(pso_init_u, &ip, sizeof(ip),
                    {g.out_weight_sum.get(), cur}, ceil_div_u32(V, MG_TG_SIZE),
                    MG_TG_SIZE);
      } else if (rescale) {
        MGPrParams ip = base;
        ip.count = V;
        cb.dispatch(pso_init_p, &ip, sizeof(ip),
                    {pvec_buf.get(), g.out_weight_sum.get(), cur},
                    ceil_div_u32(V, MG_TG_SIZE), MG_TG_SIZE);
      }
    }
    for (int i = 0; i < k; ++i) {
      Buffer* out = buf[next].get();
      spmv.encode_iteration(cb, in_o, plan, base, cur, contrib.get(), out,
                            scalars.get(), pvec_buf.get(),
                            g.out_weight_sum.get(), weights, rescale,
                            /*include_iso=*/(i == k - 1));
      reduce.encode_pr_dangling(cb, base.alpha, base.one_minus_alpha,
                                static_cast<float>(iso_p_sum),
                                plan.n_active_dangling, plan.n_iso,
                                plan.active_dangling.get(), out,
                                dang_partials.get(), scalars.get());
      prev = cur;
      cur = out;
      next ^= 1;
    }
    const uint32_t res_blocks = reduce.encode_residual(
        cb, n_active_v, rescale, plan.vertices.get(), cur, prev,
        g.out_weight_sum.get(), res_partials.get());
    cb.commit_and_wait();
    done += k;
    double err = 0.0;
    if (res_blocks > 0) {
      const float* rp = res_partials->as<float>();
      for (uint32_t b = 0; b < res_blocks; ++b)
        err += static_cast<double>(rp[b]);
    }
    if (plan.n_iso > 0) {
      const float* s = scalars->as<float>();
      err += std::fabs(static_cast<double>(s[MG_SCALAR_ISO_SCALE]) -
                       static_cast<double>(s[MG_SCALAR_ISO_SCALE_PREV])) *
             iso_p_sum;
    }
    if (err < static_cast<double>(V) * tol) break;
  }
  if (done == 0) {  // max_iter < 1: GPU init never ran
    if (p.uniform) {
      std::fill_n(cur->as<float>(), V, base.p_uniform);
    } else {
      cur = pvec_buf.get();
    }
  }

  emit(cur->as<float>(), done > 0 && rescale);
  return done;
}

}  // namespace

// ---------------------------------------------------------------------------
// GPU drivers (canonical space)
// ---------------------------------------------------------------------------
namespace gpu {

int pagerank(Graph& g, const double* pvec_canon, double alpha, double tol,
             int max_iter, int audit_interval, double* rank_canon) {
  PrTeleport p;
  p.uniform = true;
  if (g.V) {
    const double u = 1.0 / static_cast<double>(g.V);
    for (uint32_t i = 0; i < g.V; ++i) {
      if (pvec_canon[i] != u) {
        p.uniform = false;
        p.canon = pvec_canon;
        break;
      }
    }
  }
  return gpu_pagerank_impl(
      g, p, alpha, tol, max_iter, audit_interval,
      [&](const float* rc, bool rescaled) {
        const float* ows = g.out_weight_sum->as<float>();
        parallel_for(g.V, [&](std::size_t b, std::size_t e) {
          for (std::size_t i = b; i < e; ++i) {
            float v = rc[i];
            if (rescaled && ows[i] > 0.0f) v *= ows[i];
            rank_canon[i] = static_cast<double>(v);
          }
        });
      });
}

// Shared tile loop: iterate every tile to convergence/budget and hand the
// FINAL interleaved tile state to on_tile(q0, lanes, rank_final, needs_unscale).
static int ppr_tiles(
    Graph& g, const uint32_t* seeds_canon, const float* seed_w,
    const uint64_t* offsets, uint32_t n_queries, double alpha, double tol,
    int max_iter, int audit_interval, const std::vector<uint32_t>& tile_q,
    const std::function<void(uint32_t, uint32_t, Buffer*, bool)>& on_tile) {
  Runtime& rt = Runtime::instance();
  const uint32_t V = g.V;
  if (n_queries == 0 || V == 0) return 0;
  if (audit_interval < 1) audit_interval = 1;
  const Orientation& in_o = g.orientation(Dir::in);
  const Orientation& out_o = g.orientation(Dir::out);
  const GpuGatherPlan& plan = g.gpu_gather_plan(Dir::in);
  const bool rescale = plan.ows_safe_rescale;

  // Working-set guard: tile state is 2 * V * MG_PPR_TILE fp32 (3 in the rare
  // extreme-weight fallback) on top of the resident IN-orientation CSR.
  if (rt.has_gpu()) {
    const uint64_t budget = rt.device()->recommendedMaxWorkingSetSize();
    const uint64_t n_tile_bufs = rescale ? 2ull : 3ull;
    const uint64_t need =
        n_tile_bufs * V * MG_PPR_TILE * sizeof(float) +
        (uint64_t{V} + 1) * sizeof(uint32_t) +
        in_o.e_stored * sizeof(uint32_t) +
        (g.weighted ? in_o.e_stored * sizeof(float) : 0ull) +
        uint64_t{V} * sizeof(float);
    if (budget && need > budget)
      throw Error(
          ErrorCode::invalid_argument,
          "ppr_topk: graph working set (" + std::to_string(need) +
              " bytes: CSR + fixed 8-lane tile state) exceeds "
              "the GPU budget (recommendedMaxWorkingSetSize = " +
              std::to_string(budget) +
              "). Batch size does not affect this; use "
              "set_execution(\"cpu\") or MG_FORCE_CPU=1.");
  }

  SpmvEngine spmv(rt, g.weighted, /*uniform_p=*/false, /*batched=*/true);
  ReduceEngine reduce(rt, /*batched=*/true);
  MTL::ComputePipelineState* pso_tile_zero = rt.pipeline("mg_pr_tile_zero_b");
  MTL::ComputePipelineState* pso_seed_init = rt.pipeline("mg_pr_seed_init_b");

  const std::size_t state = std::size_t{V} * MG_PPR_TILE;
  BufferPtr dummy = rt.dummy();
  BufferPtr rank_b[2] = {rt.alloc_scratch(state * 4, "ppr.rank0"),
                         rt.alloc_scratch(state * 4, "ppr.rank1")};
  BufferPtr contrib_b =
      rescale ? dummy : rt.alloc_scratch(state * 4, "ppr.contrib");
  BufferPtr scalars_b = rt.alloc_scratch(MG_PPR_TILE * 4, "ppr.iter_scalars");
  BufferPtr dang_partials_b =
      (plan.n_active_dangling > ReduceEngine::kSingleTgMax)
          ? rt.alloc_scratch(
                std::size_t{ReduceEngine::n_blocks(plan.n_active_dangling)} *
                    MG_PPR_TILE * 4,
                "ppr.dangling_partials")
          : dummy;
  const uint32_t n_active_v = plan.off_iso;
  BufferPtr res_partials_b =
      n_active_v
          ? rt.alloc_scratch(
                std::size_t{ReduceEngine::n_blocks(n_active_v)} * MG_PPR_TILE *
                    4,
                "ppr.residual_partials")
          : dummy;

  std::vector<uint32_t> bounds = tile_q;
  if (bounds.empty()) {
    for (uint32_t q = 0; q < n_queries; q += MG_PPR_TILE) bounds.push_back(q);
    bounds.push_back(n_queries);
  }
  const uint32_t n_tiles = static_cast<uint32_t>(bounds.size() - 1);
  uint64_t max_tile_seeds = 1;
  for (uint32_t t = 0; t < n_tiles; ++t)
    max_tile_seeds = std::max(max_tile_seeds,
                              offsets[bounds[t + 1]] - offsets[bounds[t]]);
  BufferPtr seed_vertex =
      rt.alloc_scratch(max_tile_seeds * 4, "ppr.seed_vertex");
  BufferPtr seed_lane = rt.alloc_scratch(max_tile_seeds * 4, "ppr.seed_lane");
  BufferPtr seed_weight =
      rt.alloc_scratch(max_tile_seeds * 4, "ppr.seed_weight");

  Buffer* weights =
      (g.weighted && in_o.weights) ? in_o.weights.get() : dummy.get();
  const float* ows = g.out_weight_sum->as<float>();
  const uint32_t* in_row = in_o.row_offsets->as<uint32_t>();
  const uint32_t* out_row = out_o.row_offsets->as<uint32_t>();

  std::vector<std::pair<uint32_t, uint32_t>> iso_seeds;
  int max_iters = 0;
  for (uint32_t t = 0; t < n_tiles; ++t) {
    const uint32_t q0 = bounds[t];
    const uint32_t lanes = bounds[t + 1] - q0;
    const uint64_t s0 = offsets[q0];
    const uint32_t seed_total =
        static_cast<uint32_t>(offsets[q0 + lanes] - s0);

    double dmass[MG_PPR_TILE] = {0.0};
    uint32_t* sv = seed_vertex->as<uint32_t>();
    uint32_t* sl = seed_lane->as<uint32_t>();
    float* sw = seed_weight->as<float>();
    iso_seeds.clear();
    for (uint32_t q = q0; q < q0 + lanes; ++q) {
      const uint32_t lane = q - q0;
      for (uint64_t i = offsets[q]; i < offsets[q + 1]; ++i) {
        const uint32_t v = seeds_canon[i];
        const float w = seed_w[i];
        const uint64_t j = i - s0;
        const bool is_iso =
            (in_row[v] == in_row[v + 1]) && (out_row[v] == out_row[v + 1]);
        sv[j] = v;
        sl[j] = lane | (is_iso ? MG_PPR_SEED_ISO_BIT : 0u);
        sw[j] = w;
        if (is_iso) iso_seeds.emplace_back(v, lane);
        if (ows[v] == 0.0f) dmass[lane] += static_cast<double>(w);
      }
    }
    const bool has_iso_seed = !iso_seeds.empty();
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
      if (done == 0) {
        MGPprParams init = base;
        init.count = V;
        init.pad0 = plan.n_iso;
        init.pad1 = plan.off_iso;
        cb.dispatch(pso_tile_zero, &init, sizeof(init),
                    {rank_b[0].get(), rank_b[1].get(), plan.vertices.get()},
                    ceil_div_u32(V, MG_TG_SIZE), MG_TG_SIZE);
        init.count = seed_total;
        init.pad0 = rescale ? MG_PR_FLAG_RESCALE_OWS : 0u;
        init.pad1 = 0u;
        cb.dispatch(pso_seed_init, &init, sizeof(init),
                    {seed_vertex.get(), seed_lane.get(), seed_weight.get(),
                     g.out_weight_sum.get(), rank_b[0].get()},
                    ceil_div_u32(seed_total, MG_TG_SIZE), MG_TG_SIZE);
      }
      for (int i = 0; i < k; ++i) {
        spmv.encode_iteration_b(
            cb, in_o, plan, base, rank_b[cur].get(), contrib_b.get(),
            rank_b[cur ^ 1].get(), scalars_b.get(), g.out_weight_sum.get(),
            weights, seed_vertex.get(), seed_lane.get(), seed_weight.get(),
            rescale);
        reduce.encode_pr_dangling_b(
            cb, base, plan.n_active_dangling, has_iso_seed,
            plan.active_dangling.get(), rank_b[cur ^ 1].get(),
            dang_partials_b.get(), seed_vertex.get(), seed_lane.get(),
            scalars_b.get());
        cur ^= 1;
      }
      const uint32_t res_blocks = reduce.encode_residual_b(
          cb, n_active_v, active, rescale, plan.vertices.get(),
          rank_b[cur].get(), rank_b[cur ^ 1].get(), g.out_weight_sum.get(),
          res_partials_b.get());
      cb.commit_and_wait();
      done += k;

      double err[MG_PPR_TILE] = {0.0};
      if (res_blocks > 0) {
        const float* rp = res_partials_b->as<float>();
        for (uint32_t b = 0; b < res_blocks; ++b) {
          const float* rowp = rp + std::size_t{b} * MG_PPR_TILE;
          for (uint32_t l = 0; l < lanes; ++l)
            if ((active >> l) & 1u) err[l] += static_cast<double>(rowp[l]);
        }
      }
      if (has_iso_seed) {
        const float* rcur = rank_b[cur]->as<float>();
        const float* rprv = rank_b[cur ^ 1]->as<float>();
        for (const auto& [v, l] : iso_seeds) {
          if ((active >> l) & 1u) {
            const std::size_t slot = std::size_t{v} * MG_PPR_TILE + l;
            err[l] += std::fabs(static_cast<double>(rcur[slot]) -
                                static_cast<double>(rprv[slot]));
          }
        }
      }
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
    if (done == 0) {  // max_iter < 1: the GPU init never ran
      float* r0 = rank_b[0]->as<float>();
      std::memset(r0, 0, state * 4);
      for (uint32_t j = 0; j < seed_total; ++j)
        r0[std::size_t{sv[j]} * MG_PPR_TILE + (sl[j] & 0xFFu)] = sw[j];
    }

    on_tile(q0, lanes, rank_b[cur].get(), done > 0 && rescale);
  }
  return max_iters;
}

// gpu::ppr with explicit tile grouping (see ppr_tiles).
static int ppr_sink_tiles(
    Graph& g, const uint32_t* seeds_canon, const float* seed_w,
    const uint64_t* offsets, uint32_t n_queries, double alpha, double tol,
    int max_iter, int audit_interval, const std::vector<uint32_t>& tile_q,
    const std::function<void(uint32_t, const float*)>& sink) {
  std::vector<float> lane_scratch(g.V);
  const float* ows = g.out_weight_sum->as<float>();
  return ppr_tiles(
      g, seeds_canon, seed_w, offsets, n_queries, alpha, tol, max_iter,
      audit_interval, tile_q,
      [&](uint32_t q0, uint32_t lanes, Buffer* rank_final, bool needs_unscale) {
        const float* rc = rank_final->as<float>();
        for (uint32_t l = 0; l < lanes; ++l) {
          float* dst = lane_scratch.data();
          parallel_for(g.V, [&](std::size_t b, std::size_t e) {
            if (needs_unscale) {
              for (std::size_t v = b; v < e; ++v) {
                float x = rc[v * MG_PPR_TILE + l];
                const float w = ows[v];
                if (w > 0.0f) x *= w;
                dst[v] = x;
              }
            } else {
              for (std::size_t v = b; v < e; ++v)
                dst[v] = rc[v * MG_PPR_TILE + l];
            }
          });
          sink(q0 + l, dst);
        }
      });
}

int ppr(Graph& g, const uint32_t* seeds_canon, const float* seed_w,
        const uint64_t* offsets, uint32_t n_queries, double alpha, double tol,
        int max_iter, int audit_interval,
        const std::function<void(uint32_t, const float*)>& sink) {
  return ppr_sink_tiles(g, seeds_canon, seed_w, offsets, n_queries, alpha,
                        tol, max_iter, audit_interval, {}, sink);
}

// GPU radix-select path (plan M3): the CPU oracle stays behind the same
// API; eligibility keeps degenerate shapes (tiny V, k ~ V, huge k) on it.
bool topk_select_eligible(uint32_t V, uint32_t k) {
  return env_flag("MG_GPU_TOPK", true) && V >= 4096 && k <= 4096 && k < V;
}

int ppr_topk_select(Graph& g, const uint32_t* seeds_canon, const float* seed_w,
                    const uint64_t* offsets, uint32_t n_queries, double alpha,
                    double tol, int max_iter, int audit_interval, uint32_t k,
                    const std::vector<uint32_t>& tile_q, int32_t* out_ids,
                    float* out_scores) {
  Runtime& rt = Runtime::instance();
  const uint32_t V = g.V;
  const uint32_t cap = k + MG_TOPK_TIE_SLACK;
  MTL::ComputePipelineState* pso_hist = rt.pipeline("mg_topk_hist_b");
  MTL::ComputePipelineState* pso_digit = rt.pipeline("mg_topk_digit_b");
  MTL::ComputePipelineState* pso_compact = rt.pipeline("mg_topk_compact_b");
  BufferPtr thresholds = rt.alloc_scratch(MG_PPR_TILE * 4, "topk.thresholds");
  BufferPtr hist = rt.alloc_scratch(MG_PPR_TILE * 256 * 4, "topk.hist");
  BufferPtr select_state =
      rt.alloc_scratch(MG_TOPK_STATE_LEN * 4, "topk.select_state");
  BufferPtr counters = rt.alloc_scratch(MG_PPR_TILE * 4, "topk.counters");
  BufferPtr cands =
      rt.alloc_scratch(std::size_t{MG_PPR_TILE} * cap * 4, "topk.candidates");
  const uint32_t hist_tgs = std::min(1024u, ceil_div_u32(V, MG_TG_SIZE * 2u));

  return ppr_tiles(
      g, seeds_canon, seed_w, offsets, n_queries, alpha, tol, max_iter,
      audit_interval, tile_q,
      [&](uint32_t q0, uint32_t lanes, Buffer* rank_final, bool needs_unscale) {
        const uint32_t active_mask = (1u << lanes) - 1u;
        MGTopkParams tp{V,
                        k,
                        cap,
                        active_mask,
                        0u,
                        needs_unscale ? MG_TOPK_FLAG_UNSCALE_OWS : 0u,
                        0u,
                        0u};
        std::memset(counters->data(), 0, MG_PPR_TILE * 4);
        std::memset(thresholds->data(), 0, MG_PPR_TILE * 4);
        std::memset(hist->data(), 0, MG_PPR_TILE * 256 * 4);
        uint32_t* st = select_state->as<uint32_t>();
        for (uint32_t l = 0; l < MG_PPR_TILE; ++l) {
          st[l] = k;                // remaining rank
          st[MG_PPR_TILE + l] = V;  // pass-0 class: every score
        }
        st[MG_TOPK_STATE_PENDING] = active_mask;
        CommandBatch cb(rt);
        for (uint32_t pass = 0; pass < 4; ++pass) {
          tp.pass = pass;
          cb.dispatch(pso_hist, &tp, sizeof(tp),
                      {rank_final, g.out_weight_sum.get(), thresholds.get(),
                       select_state.get(), hist.get()},
                      hist_tgs, MG_TG_SIZE);
          cb.dispatch(pso_digit, &tp, sizeof(tp),
                      {hist.get(), thresholds.get(), select_state.get()}, 1,
                      MG_TG_SIZE);
        }
        cb.dispatch(pso_compact, &tp, sizeof(tp),
                    {rank_final, thresholds.get(), counters.get(), cands.get()},
                    ceil_div_u32(V, MG_TG_SIZE), MG_TG_SIZE);
        cb.commit_and_wait();

        const float* rc = rank_final->as<float>();
        const uint32_t* cnt = counters->as<uint32_t>();
        const uint32_t* thr = thresholds->as<uint32_t>();
        const uint32_t* cd = cands->as<uint32_t>();
        const uint32_t* canon_of = g.canon_of_user.data();
        uint64_t lane_work = 0;
        for (uint32_t l = 0; l < lanes; ++l)
          lane_work += cnt[l] > cap ? V : cnt[l];
        lane_work *= std::bit_width(k);
        parallel_for(
            lanes,
            [&](std::size_t lb, std::size_t le) {
              for (std::size_t l = lb; l < le; ++l) {
                int32_t* row_ids = out_ids + std::size_t{q0 + l} * k;
                float* row_sc = out_scores + std::size_t{q0 + l} * k;
                const uint32_t raw_thr = thr[l];
                const bool strict_gt = (raw_thr & MG_TOPK_STRICT_GT_BIT) != 0u;
                const uint32_t n = cnt[l];
                if (n > cap || (!strict_gt && n < k) || (strict_gt && n > k)) {
                  boundary::topk_user(g, rc, MG_PPR_TILE,
                                      static_cast<uint32_t>(l), k, row_ids,
                                      row_sc);
                  continue;
                }
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
                if (!strict_gt) {
                  std::partial_sort(items.begin(), items.begin() + k,
                                    items.end(), cmp);
                  for (uint32_t i = 0; i < k; ++i) {
                    row_ids[i] = static_cast<int32_t>(items[i].second);
                    row_sc[i] = items[i].first;
                  }
                } else {
                  std::sort(items.begin(), items.end(), cmp);
                  uint32_t out = 0;
                  for (; out < n; ++out) {
                    row_ids[out] = static_cast<int32_t>(items[out].second);
                    row_sc[out] = items[out].first;
                  }
                  const uint32_t thr_bits = raw_thr & 0x7FFFFFFFu;
                  for (uint32_t u = 0; u < V && out < k; ++u) {
                    const float s =
                        rc[std::size_t{canon_of[u]} * MG_PPR_TILE + l];
                    const bool eq =
                        (thr_bits == 0u)
                            ? (s == 0.0f)
                            : (std::bit_cast<uint32_t>(s) == thr_bits);
                    if (eq) {
                      row_ids[out] = static_cast<int32_t>(u);
                      row_sc[out] = s;
                      ++out;
                    }
                  }
                  if (out < k) {
                    boundary::topk_user(g, rc, MG_PPR_TILE,
                                        static_cast<uint32_t>(l), k, row_ids,
                                        row_sc);
                  }
                }
              }
            },
            work_grain(lanes, lane_work));
      });
}

}  // namespace gpu

// ---------------------------------------------------------------------------
// Entry points (USER space)
// ---------------------------------------------------------------------------

int pagerank(Graph& g, const PagerankOpts& o, float* out_rank) {
  if (!(o.alpha > 0.0 && o.alpha < 1.0))
    throw_invalid("pagerank: alpha must be in (0, 1)");
  if (!(o.tol > 0.0) || !std::isfinite(o.tol))
    throw_invalid("pagerank: tol must be finite and > 0");
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

  const double p_sum =
      o.personalization ? boundary::personalization_sum(g, o.personalization)
                        : 0.0;
  const ExecPath path = rt.plan(g.stored_edges(), PlanOp::pagerank);
  const int aud = audit_interval_env();

  int iters = 0;
  if (path == ExecPath::gpu) {
    // Straight from the USER-order input to the fp32 device vector and
    // back (fp32 -> fp64 -> fp32 is the identity, so no fp64 copies).
    PrTeleport p;
    p.uniform = o.personalization == nullptr;
    p.user = o.personalization;
    p.user_sum = p_sum;
    iters = gpu_pagerank_impl(
        g, p, o.alpha, o.tol, o.max_iter, aud,
        [&](const float* rc, bool rescaled) {
          if (!rescaled) {
            boundary::canon_to_user(g, rc, out_rank);
          } else {
            const float* ows = g.out_weight_sum->as<float>();
            const uint32_t* canon_of = g.canon_of_user.data();
            parallel_for(g.V, [&](std::size_t b, std::size_t e) {
              for (std::size_t u = b; u < e; ++u) {
                const uint32_t c = canon_of[u];
                float v = rc[c];
                const float w = ows[c];
                if (w > 0.0f) v *= w;
                out_rank[u] = v;
              }
            });
          }
        });
  } else {
    const std::vector<double> pvec =
        boundary::pvec_to_canon(g, o.personalization, p_sum);
    std::vector<double> rank_canon(g.V, 0.0);
    iters = cpu::pagerank(g, pvec.data(), o.alpha, o.tol, o.max_iter, aud,
                          rank_canon.data());
    boundary::canon_to_user(g, rank_canon.data(), out_rank);
  }

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
  if (!(o.tol > 0.0) || !std::isfinite(o.tol))
    throw_invalid("ppr_topk: tol must be finite and > 0");
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

  ExecPath path = rt.plan(g.stored_edges(), PlanOp::ppr);
  const int aud = audit_interval_env();

  // Bounded-reach latency path (auto/CPU modes; forced GPU bypasses it).
  // A query whose seeds' OUT-reach fits the caps has its whole support
  // inside that set, so cpu::ppr_topk_sparse answers it exactly (fp64,
  // bit-identical to cpu::ppr) without touching the other V - |R|
  // vertices. Routing is per query, so batched == sequential still holds.
  const long sp_vertices = env_long("MG_PPR_SPARSE_MAX_VERTICES", 1024);
  const long sp_edges = env_long("MG_PPR_SPARSE_MAX_EDGES", 8192);
  const bool sparse_on =
      rt.mode() != ExecMode::gpu && sp_vertices > 0 && sp_edges > 0;
  std::function<int(uint32_t)> try_sparse;
  if (sparse_on) {
    // One-time per graph, on this thread before queries fan out to workers.
    g.cpu_in_tile_keys();
    const uint32_t vcap = static_cast<uint32_t>(std::min<long>(
        sp_vertices, static_cast<long>(std::numeric_limits<uint32_t>::max())));
    const uint64_t ecap = static_cast<uint64_t>(sp_edges);
    try_sparse = [&, vcap, ecap](uint32_t q) -> int {
      const uint64_t s0 = offs[q];
      return cpu::ppr_topk_sparse(
          g, s_canon.data() + s0, s_w.data() + s0,
          static_cast<uint32_t>(offs[q + 1] - s0), o.alpha, o.tol,
          o.max_iter, aud, vcap, ecap, o.k, out_ids + std::size_t{q} * o.k,
          out_scores + std::size_t{q} * o.k);
    };
  }

  // Streaming top-k selection (deterministic ties): the per-query sink runs
  // boundary::topk_user on each FINAL dense vector as it becomes available,
  // so peak dense state is one GPU tile (plus one V-length scratch) or one
  // in-flight fp64 vector per CPU worker — never B x V.
  // Concurrency: gpu::ppr calls the sink sequentially from this thread;
  // cpu::ppr workers call it concurrently for DISTINCT q (chosen over a
  // mutex: topk_user is pure and each call writes only its own disjoint
  // output rows, so no serialization is needed).
  int iters = 0;
  if (path == ExecPath::cpu) {
    // The sparse attempt runs inside cpu::ppr's per-query worker: one
    // fan-out per batch, and an overflowing query falls through to its
    // dense solve on the same thread.
    iters = cpu::ppr(g, s_canon.data(), s_w.data(), offs.data(), n_queries,
                     o.alpha, o.tol, o.max_iter, aud,
                     [&](uint32_t q, const double* rank) {
                       boundary::topk_user(g, rank, 1u, 0u, o.k,
                                           out_ids + std::size_t{q} * o.k,
                                           out_scores + std::size_t{q} * o.k);
                     },
                     try_sparse);
  } else {
    auto run_gpu = [&](const uint32_t* sc, const float* sw,
                       const uint64_t* of, uint32_t nq,
                       const std::vector<uint32_t>& tile_q, int32_t* ids,
                       float* scores) -> int {
      if (gpu::topk_select_eligible(g.V, o.k)) {
        // GPU radix-select (plan M3): O(V) threshold + compaction on the
        // GPU, exact tie-rule finish on a ~k-sized candidate set.
        // Bit-identical to the CPU selection oracle (tests assert it);
        // MG_GPU_TOPK=0 forces the oracle.
        return gpu::ppr_topk_select(g, sc, sw, of, nq, o.alpha, o.tol,
                                    o.max_iter, aud, o.k, tile_q, ids,
                                    scores);
      }
      return gpu::ppr_sink_tiles(
          g, sc, sw, of, nq, o.alpha, o.tol, o.max_iter, aud, tile_q,
          [&](uint32_t q, const float* rank) {
            boundary::topk_user(g, rank, 1u, 0u, o.k,
                                ids + std::size_t{q} * o.k,
                                scores + std::size_t{q} * o.k);
          });
    };
    // Answer what fits on the CPU first (queries in parallel), then send
    // only the overflowing rows through the GPU tiles. Each attempt is
    // capped (~tens of microseconds even when it overflows), so 2 queries
    // per chunk (B=16 -> 8 threads) keeps the hand-offs from dominating
    // (kg, every row sparse: 0.090 ms at 4 per chunk, 0.061 at 2).
    std::vector<uint32_t> dense_rows;
    if (sparse_on) {
      std::vector<int> sp_it(n_queries, -1);
      parallel_for(
          n_queries,
          [&](std::size_t b, std::size_t e) {
            for (std::size_t q = b; q < e; ++q)
              sp_it[q] = try_sparse(static_cast<uint32_t>(q));
          },
          /*grain=*/2);
      for (uint32_t q = 0; q < n_queries; ++q) {
        if (sp_it[q] < 0)
          dense_rows.push_back(q);
        else
          iters = std::max(iters, sp_it[q]);
      }
    } else {
      dense_rows.resize(n_queries);
      for (uint32_t q = 0; q < n_queries; ++q) dense_rows[q] = q;
    }
    if (dense_rows.empty()) {
      path = ExecPath::cpu;  // every row was answered by the sparse path
    } else if (dense_rows.size() == n_queries) {
      iters = run_gpu(s_canon.data(), s_w.data(), offs.data(), n_queries, {},
                      out_ids, out_scores);
    } else {
      // Mixed batch: compact the overflowing rows, run them on the GPU,
      // then scatter their output rows back into place. A tile iterates
      // until its slowest lane freezes. Packing the rows of each all-GPU
      // tile (queries [t, t + 1) * MG_PPR_TILE) whole costs at most the
      // tile iterations of running every row on the GPU (a merged tile
      // takes the max of its parts). When that packing needs more than
      // ceil(nd / MG_PPR_TILE) tiles, the rows are compacted into
      // consecutive full tiles instead: fewer tiles, but no such bound.
      // Rows are tile-independent, so both give the same results.
      const uint32_t nd = static_cast<uint32_t>(dense_rows.size());
      std::vector<uint32_t> tile_q{0};
      uint32_t open_lanes = 0;
      for (uint32_t i = 0; i < nd;) {
        uint32_t j = i + 1;
        while (j < nd &&
               dense_rows[j] / MG_PPR_TILE == dense_rows[i] / MG_PPR_TILE)
          ++j;
        if (open_lanes + (j - i) > MG_PPR_TILE) {
          tile_q.push_back(i);
          open_lanes = 0;
        }
        open_lanes += j - i;
        i = j;
      }
      tile_q.push_back(nd);
      if (tile_q.size() - 1 > ceil_div_u32(nd, MG_PPR_TILE)) tile_q.clear();
      std::vector<uint32_t> d_canon;
      std::vector<float> d_w;
      std::vector<uint64_t> d_offs(std::size_t{nd} + 1, 0);
      for (uint32_t i = 0; i < nd; ++i) {
        const uint32_t q = dense_rows[i];
        d_canon.insert(d_canon.end(), s_canon.begin() + offs[q],
                       s_canon.begin() + offs[q + 1]);
        d_w.insert(d_w.end(), s_w.begin() + offs[q],
                   s_w.begin() + offs[q + 1]);
        d_offs[i + 1] = d_canon.size();
      }
      std::vector<int32_t> d_ids(std::size_t{nd} * o.k);
      std::vector<float> d_scores(std::size_t{nd} * o.k);
      iters = std::max(iters, run_gpu(d_canon.data(), d_w.data(),
                                      d_offs.data(), nd, tile_q, d_ids.data(),
                                      d_scores.data()));
      for (uint32_t i = 0; i < nd; ++i) {
        const std::size_t src = std::size_t{i} * o.k;
        const std::size_t dst = std::size_t{dense_rows[i]} * o.k;
        std::copy_n(d_ids.data() + src, o.k, out_ids + dst);
        std::copy_n(d_scores.data() + src, o.k, out_scores + dst);
      }
    }
  }

  ri.path = path;
  ri.iterations = iters;
  ri.elapsed_ms = elapsed_ms(t0);
  rt.record_run(ri);
  return iters;
}

}  // namespace mg
