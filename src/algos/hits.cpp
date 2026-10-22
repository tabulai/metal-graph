// hits.cpp — planner-aware HITS entry point and Metal driver.
//
// The GPU path deliberately reuses the production degree-binned PageRank
// gather as a raw sparse-matrix/vector multiply, skipping PageRank's prepare
// pass. Separate engines are required for IN and OUT because each owns
// orientation-sized huge-row scratch.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <vector>

#include "../engines/reduce_engine.hpp"
#include "../engines/spmv_engine.hpp"
#include "../kernels/mg_params.h"
#include "algos.hpp"
#include "boundary.hpp"

namespace mg {
namespace {

constexpr std::size_t kAuditChunk = std::size_t{1} << 16;

int audit_interval_env() {
  const long value = env_long("MG_HITS_AUDIT_INTERVAL", 5);
  return value < 1 ? 1 : static_cast<int>(value);
}

double elapsed_ms(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - start)
      .count();
}

double l1_diff(const float* a, const float* b, uint32_t count) {
  const std::size_t chunks =
      (static_cast<std::size_t>(count) + kAuditChunk - 1) / kAuditChunk;
  std::vector<double> partials(chunks, 0.0);
  parallel_for(
      chunks,
      [&](std::size_t begin, std::size_t end) {
        for (std::size_t chunk = begin; chunk < end; ++chunk) {
          const std::size_t lo = chunk * kAuditChunk;
          const std::size_t hi =
              std::min<std::size_t>(count, lo + kAuditChunk);
          double sum = 0.0;
          for (std::size_t i = lo; i < hi; ++i)
            sum += std::fabs(static_cast<double>(a[i]) -
                             static_cast<double>(b[i]));
          partials[chunk] = sum;
        }
      },
      1);
  double total = 0.0;
  for (double value : partials) total += value;
  return total;
}

// GPU HITS solve. emit(hubs, authorities) receives the FINAL fp32 vectors
// (canonical order, length V) while the device buffers are still alive.
HitsResult gpu_hits_impl(
    Graph& g, double tol, int max_iter, int audit_interval,
    const std::function<void(const float*, const float*)>& emit) {
  Runtime& rt = Runtime::instance();
  if (!rt.has_gpu())
    throw Error(ErrorCode::no_gpu, "gpu::hits requires a Metal device");
  const uint32_t V = g.V;
  if (V == 0) return {};
  if (g.stored_edges() == 0) {
    const std::vector<float> zeros(V, 0.0f);
    emit(zeros.data(), zeros.data());
    return {};
  }
  audit_interval = std::max(audit_interval, 1);

  const Orientation& in_o = g.orientation(Dir::in);
  const Orientation& out_o = g.orientation(Dir::out);
  SpmvEngine spmv_in(rt, /*weighted=*/false, /*uniform_p=*/true,
                     /*batched=*/false);
  SpmvEngine spmv_out(rt, /*weighted=*/false, /*uniform_p=*/true,
                      /*batched=*/false);
  ReduceEngine reduce(rt, /*batched=*/false);
  MTL::ComputePipelineState* scale = rt.pipeline("mg_scale_f32_by_scalar");

  // V-sized buffers skip the zero-fill: hubs[0] is filled on the GPU at the
  // head of the first batch, and the gather bins plus the zero list
  // partition V, so every gather writes its whole output.
  BufferPtr dummy = rt.dummy();
  BufferPtr hubs[2] = {
      rt.alloc_scratch(std::size_t{V} * sizeof(float), "hits.hubs0"),
      rt.alloc_scratch(std::size_t{V} * sizeof(float), "hits.hubs1")};
  BufferPtr authorities =
      rt.alloc_scratch(std::size_t{V} * sizeof(float), "hits.authorities");
  BufferPtr norm = rt.alloc(sizeof(float), "hits.norm");
  BufferPtr partials =
      rt.alloc(std::size_t{ReduceEngine::n_blocks(V)} * sizeof(float),
               "hits.norm_partials");

  const MGFillParams init{1.0f / static_cast<float>(V), 0u, V, 0u};

  MGPrParams raw{};
  raw.alpha = 1.0f;
  raw.one_minus_alpha = 0.0f;
  raw.p_uniform = 0.0f;
  raw.v_count = V;
  MGFillParams scale_params{0.0f, 0u, V, 0u};

  auto encode_raw = [&](SpmvEngine& engine, CommandBatch& cb,
                        const Orientation& orientation, Buffer* input,
                        Buffer* output) {
    engine.encode_prepared_gather(cb, orientation, raw, input, output,
                                  dummy.get(), dummy.get(), dummy.get());
  };
  auto encode_normalize = [&](CommandBatch& cb, Buffer* values) {
    reduce.encode_sum(cb, raw, V, values, partials.get(), norm.get(), 0u);
    cb.dispatch(scale, &scale_params, sizeof(scale_params),
                {values, norm.get()}, ceil_div_u32(V, MG_TG_SIZE), MG_TG_SIZE);
  };

  int cur = 0;
  int done = 0;
  HitsResult result;
  result.stop_reason = StopReason::max_iter;
  while (done < max_iter) {
    const int batch =
        boundary::next_audit_batch(done, max_iter, audit_interval);
    CommandBatch cb(rt);
    if (done == 0) {
      cb.dispatch(rt.pipeline("mg_fill_f32"), &init, sizeof(init),
                  {hubs[0].get()}, ceil_div_u32(V, MG_TG_SIZE), MG_TG_SIZE);
    }
    for (int i = 0; i < batch; ++i) {
      encode_raw(spmv_in, cb, in_o, hubs[cur].get(), authorities.get());
      encode_normalize(cb, authorities.get());
      encode_raw(spmv_out, cb, out_o, authorities.get(), hubs[cur ^ 1].get());
      encode_normalize(cb, hubs[cur ^ 1].get());
      cur ^= 1;
    }
    cb.commit_and_wait();
    done += batch;

    const float latest_norm = norm->as<float>()[0];
    if (!(latest_norm > 0.0f) || !std::isfinite(latest_norm))
      throw Error(ErrorCode::internal,
                  "hits: encountered a degenerate normalization");
    const double err =
        l1_diff(hubs[cur]->as<float>(), hubs[cur ^ 1]->as<float>(), V);
    result.final_delta_l1 = err;
    if (err < static_cast<double>(V) * tol) {
      result.stop_reason = StopReason::converged;
      break;
    }
  }

  if (done == 0) {  // max_iter < 1: the GPU init never ran
    std::fill_n(hubs[0]->as<float>(), V, init.f_value);
    std::fill_n(authorities->as<float>(), V, 0.0f);
  }

  emit(hubs[cur]->as<float>(), authorities->as<float>());
  result.iterations = done;
  return result;
}

}  // namespace

namespace gpu {

HitsResult hits(Graph& g, double tol, int max_iter, int audit_interval,
                double* hubs_canon, double* authorities_canon) {
  return gpu_hits_impl(
      g, tol, max_iter, audit_interval,
      [&](const float* final_hubs, const float* final_authorities) {
        parallel_for(g.V, [&](std::size_t begin, std::size_t end) {
          for (std::size_t i = begin; i < end; ++i) {
            hubs_canon[i] = static_cast<double>(final_hubs[i]);
            authorities_canon[i] = static_cast<double>(final_authorities[i]);
          }
        });
      });
}

}  // namespace gpu

int hits(Graph& g, const HitsOpts& o, float* out_hubs,
         float* out_authorities) {
  if (!(o.tol > 0.0) || !std::isfinite(o.tol))
    throw_invalid("hits: tol must be finite and > 0");
  if (o.max_iter < 1) throw_invalid("hits: max_iter must be >= 1");

  const auto start = std::chrono::steady_clock::now();
  Runtime& rt = Runtime::instance();
  RunInfo info;
  info.op = "hits";
  if (g.V == 0) {
    info.path = ExecPath::cpu;
    info.iterations = 0;
    info.stop_reason = StopReason::empty_graph;
    info.final_delta_l1 = 0.0;
    info.elapsed_ms = elapsed_ms(start);
    rt.record_run(info);
    return 0;
  }
  if (!out_hubs || !out_authorities)
    throw_invalid("hits: output buffers must not be null");

  const ExecPath path = rt.plan(g.stored_edges(), PlanOp::hits);
  const int audit_interval = audit_interval_env();
  HitsResult result;
  if (path == ExecPath::gpu) {
    // fp32 device vectors straight to USER order in one pass (fp32 -> fp64
    // -> fp32 is the identity, so no fp64 copies).
    result = gpu_hits_impl(
        g, o.tol, o.max_iter, audit_interval,
        [&](const float* hubs, const float* authorities) {
          const uint32_t* canon_of = g.canon_of_user.data();
          for (uint32_t u = 0; u < g.V; ++u) {
            const uint32_t c = canon_of[u];
            out_hubs[u] = hubs[c];
            out_authorities[u] = authorities[c];
          }
        });
  } else {
    std::vector<double> hubs_canon(g.V, 0.0);
    std::vector<double> authorities_canon(g.V, 0.0);
    result = cpu::hits(g, o.tol, o.max_iter, audit_interval,
                       hubs_canon.data(), authorities_canon.data());
    boundary::canon_to_user(g, hubs_canon.data(), out_hubs);
    boundary::canon_to_user(g, authorities_canon.data(), out_authorities);
  }

  info.path = path;
  info.iterations = result.iterations;
  info.stop_reason = result.stop_reason;
  info.final_delta_l1 = result.final_delta_l1;
  info.elapsed_ms = elapsed_ms(start);
  rt.record_run(info);
  return result.iterations;
}

}  // namespace mg
