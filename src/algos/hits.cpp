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

}  // namespace

namespace gpu {

int hits(Graph& g, double tol, int max_iter, int audit_interval,
         double* hubs_canon, double* authorities_canon) {
  Runtime& rt = Runtime::instance();
  if (!rt.has_gpu())
    throw Error(ErrorCode::no_gpu, "gpu::hits requires a Metal device");
  const uint32_t V = g.V;
  if (V == 0) return 0;
  if (g.stored_edges() == 0) {
    std::fill_n(hubs_canon, V, 0.0);
    std::fill_n(authorities_canon, V, 0.0);
    return 0;
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

  BufferPtr dummy = rt.dummy();
  BufferPtr hubs[2] = {rt.alloc(std::size_t{V} * sizeof(float), "hits.hubs0"),
                       rt.alloc(std::size_t{V} * sizeof(float), "hits.hubs1")};
  BufferPtr authorities =
      rt.alloc(std::size_t{V} * sizeof(float), "hits.authorities");
  BufferPtr vertices =
      rt.alloc(std::size_t{V} * sizeof(uint32_t), "hits.vertices");
  BufferPtr norm = rt.alloc(sizeof(float), "hits.norm");
  BufferPtr partials =
      rt.alloc(std::size_t{ReduceEngine::n_blocks(V)} * sizeof(float),
               "hits.norm_partials");

  const float initial = 1.0f / static_cast<float>(V);
  float* h0 = hubs[0]->as<float>();
  uint32_t* ids = vertices->as<uint32_t>();
  parallel_for(V, [&](std::size_t begin, std::size_t end) {
    for (std::size_t i = begin; i < end; ++i) {
      h0[i] = initial;
      ids[i] = static_cast<uint32_t>(i);
    }
  });

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
    reduce.encode_list_sum(cb, raw, V, vertices.get(), values, partials.get(),
                           norm.get(), 0u);
    cb.dispatch(scale, &scale_params, sizeof(scale_params),
                {values, norm.get()}, ceil_div_u32(V, MG_TG_SIZE), MG_TG_SIZE);
  };

  int cur = 0;
  int done = 0;
  while (done < max_iter) {
    const int batch =
        boundary::next_audit_batch(done, max_iter, audit_interval);
    CommandBatch cb(rt);
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
    if (err < static_cast<double>(V) * tol) break;
  }

  const float* final_hubs = hubs[cur]->as<float>();
  const float* final_authorities = authorities->as<float>();
  parallel_for(V, [&](std::size_t begin, std::size_t end) {
    for (std::size_t i = begin; i < end; ++i) {
      hubs_canon[i] = static_cast<double>(final_hubs[i]);
      authorities_canon[i] = static_cast<double>(final_authorities[i]);
    }
  });
  return done;
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
    info.elapsed_ms = elapsed_ms(start);
    rt.record_run(info);
    return 0;
  }
  if (!out_hubs || !out_authorities)
    throw_invalid("hits: output buffers must not be null");

  const ExecPath path = rt.plan(g.stored_edges());
  const int audit_interval = audit_interval_env();
  std::vector<double> hubs_canon(g.V, 0.0);
  std::vector<double> authorities_canon(g.V, 0.0);
  int iterations = 0;
  if (path == ExecPath::gpu) {
    iterations = gpu::hits(g, o.tol, o.max_iter, audit_interval,
                           hubs_canon.data(), authorities_canon.data());
  } else {
    iterations = cpu::hits(g, o.tol, o.max_iter, audit_interval,
                           hubs_canon.data(), authorities_canon.data());
  }
  boundary::canon_to_user(g, hubs_canon.data(), out_hubs);
  boundary::canon_to_user(g, authorities_canon.data(), out_authorities);

  info.path = path;
  info.iterations = iterations;
  info.elapsed_ms = elapsed_ms(start);
  rt.record_run(info);
  return iterations;
}

}  // namespace mg
