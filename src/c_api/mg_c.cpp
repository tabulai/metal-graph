// mg_c.cpp — stable C ABI (include/mg.h) over the C++ core.
//
// Every entry point wraps its body in try/catch: mg::Error maps to its
// mg_status code, std::bad_alloc to MG_OUT_OF_MEMORY, anything else to
// MG_INTERNAL_ERROR. The failure message lands in a thread-local buffer
// served by mg_last_error_message().
//
// SPDX-License-Identifier: Apache-2.0
#include "mg.h"

#include <cstring>
#include <exception>
#include <new>
#include <string>

#include "../algos/algos.hpp"
#include "../common/mg_common.hpp"
#include "../graph/graph.hpp"
#include "../runtime/runtime.hpp"

struct mg_graph {
  mg::GraphPtr impl;
};

namespace {

thread_local std::string t_last_error;

mg_status status_of(mg::ErrorCode c) {
  switch (c) {
    case mg::ErrorCode::ok: return MG_OK;
    case mg::ErrorCode::invalid_argument: return MG_INVALID_ARGUMENT;
    case mg::ErrorCode::out_of_memory: return MG_OUT_OF_MEMORY;
    case mg::ErrorCode::no_gpu: return MG_NO_GPU;
    case mg::ErrorCode::internal: return MG_INTERNAL_ERROR;
    case mg::ErrorCode::not_implemented: return MG_NOT_IMPLEMENTED;
  }
  return MG_INTERNAL_ERROR;
}

template <typename Fn>
mg_status wrap(Fn&& fn) {
  try {
    fn();
    return MG_OK;
  } catch (const mg::Error& e) {
    t_last_error = e.what();
    return status_of(e.code());
  } catch (const std::bad_alloc&) {
    t_last_error = "out of memory";
    return MG_OUT_OF_MEMORY;
  } catch (const std::exception& e) {
    t_last_error = e.what();
    return MG_INTERNAL_ERROR;
  } catch (...) {
    t_last_error = "unknown internal error";
    return MG_INTERNAL_ERROR;
  }
}

void require(bool ok, const char* msg) {
  if (!ok) mg::throw_invalid(msg);
}

mg::Graph& deref(mg_graph* g) {
  require(g != nullptr && g->impl != nullptr, "graph: must not be NULL");
  return *g->impl;
}

const mg::Graph& deref(const mg_graph* g) {
  require(g != nullptr && g->impl != nullptr, "graph: must not be NULL");
  return *g->impl;
}

mg::BfsDir bfs_dir(mg_direction d) {
  switch (d) {
    case MG_DIR_OUT: return mg::BfsDir::out;
    case MG_DIR_IN: return mg::BfsDir::in;
    case MG_DIR_BOTH: return mg::BfsDir::both;
  }
  mg::throw_invalid("direction: must be MG_DIR_OUT, MG_DIR_IN or MG_DIR_BOTH");
}

}  // namespace

extern "C" {

#define MG_STR2(x) #x
#define MG_STR(x) MG_STR2(x)

const char* mg_version(void) {
  return MG_STR(MG_VERSION_MAJOR) "." MG_STR(MG_VERSION_MINOR) "." MG_STR(
      MG_VERSION_PATCH);
}

const char* mg_last_error_message(void) { return t_last_error.c_str(); }

mg_status mg_set_execution(mg_exec_mode mode) {
  return wrap([&] {
    switch (mode) {
      case MG_EXEC_AUTO:
        mg::Runtime::instance().set_mode(mg::ExecMode::autom);
        return;
      case MG_EXEC_GPU:
        mg::Runtime::instance().set_mode(mg::ExecMode::gpu);
        return;
      case MG_EXEC_CPU:
        mg::Runtime::instance().set_mode(mg::ExecMode::cpu);
        return;
    }
    mg::throw_invalid("mode: must be MG_EXEC_AUTO, MG_EXEC_GPU or MG_EXEC_CPU");
  });
}

mg_status mg_last_run(int* out_path, int* out_iterations, double* out_ms) {
  return wrap([&] {
    mg::RunInfo info = mg::Runtime::instance().last_run();
    if (out_path) *out_path = static_cast<int>(info.path);  // 0=gpu 1=cpu
    if (out_iterations) *out_iterations = info.iterations;
    if (out_ms) *out_ms = info.elapsed_ms;
  });
}

mg_status mg_graph_create_from_coo(const uint32_t* src, const uint32_t* dst,
                                   const float* weights, uint64_t num_edges,
                                   uint32_t num_vertices, int directed,
                                   mg_graph** out_graph) {
  return wrap([&] {
    require(out_graph != nullptr, "out_graph: must not be NULL");
    *out_graph = nullptr;
    require(num_edges == 0 || (src != nullptr && dst != nullptr),
            "src/dst: must not be NULL when num_edges > 0");
    mg::GraphPtr g = mg::Graph::build(src, dst, weights, num_edges,
                                      num_vertices, directed != 0);
    *out_graph = new mg_graph{std::move(g)};
  });
}

void mg_graph_destroy(mg_graph* g) { delete g; }

mg_status mg_graph_num_vertices(const mg_graph* g, uint32_t* out) {
  return wrap([&] {
    const mg::Graph& gr = deref(g);
    require(out != nullptr, "out: must not be NULL");
    *out = gr.V;
  });
}

mg_status mg_graph_num_edges(const mg_graph* g, uint64_t* out) {
  return wrap([&] {
    const mg::Graph& gr = deref(g);
    require(out != nullptr, "out: must not be NULL");
    *out = gr.E_input;
  });
}

mg_status mg_pagerank(mg_graph* g, double alpha, double tol, int max_iter,
                      const float* personalization, float* out_rank,
                      int* out_iterations) {
  return wrap([&] {
    mg::Graph& gr = deref(g);
    require(out_rank != nullptr || gr.V == 0, "out_rank: must not be NULL");
    mg::PagerankOpts o;
    o.alpha = alpha;
    o.tol = tol;
    o.max_iter = max_iter;
    o.personalization = personalization;
    int iters = mg::pagerank(gr, o, out_rank);
    if (out_iterations) *out_iterations = iters;
  });
}

mg_status mg_ppr_topk(mg_graph* g, const uint32_t* seeds,
                      const float* seed_weights, const uint64_t* offsets,
                      uint32_t num_queries, uint32_t k, double alpha,
                      double tol, int max_iter, int32_t* out_ids,
                      float* out_scores) {
  return wrap([&] {
    mg::Graph& gr = deref(g);
    if (num_queries > 0) {
      require(offsets != nullptr, "offsets: must not be NULL");
      require(offsets[num_queries] == 0 ||
                  (seeds != nullptr && seed_weights != nullptr),
              "seeds/seed_weights: must not be NULL when offsets span seeds");
      require(k == 0 || (out_ids != nullptr && out_scores != nullptr),
              "out_ids/out_scores: must not be NULL when k > 0");
    }
    mg::PprTopkOpts o;
    o.alpha = alpha;
    o.tol = tol;
    o.max_iter = max_iter;
    o.k = k;
    mg::ppr_topk(gr, seeds, seed_weights, offsets, num_queries, o, out_ids,
                 out_scores);
  });
}

mg_status mg_bfs(mg_graph* g, const uint32_t* sources, uint32_t num_sources,
                 mg_direction direction, int32_t* out_dist,
                 int32_t* out_parent) {
  return wrap([&] {
    mg::Graph& gr = deref(g);
    require(num_sources == 0 || sources != nullptr,
            "sources: must not be NULL when num_sources > 0");
    require((out_dist != nullptr && out_parent != nullptr) || gr.V == 0,
            "out_dist/out_parent: must not be NULL");
    mg::bfs(gr, sources, num_sources, bfs_dir(direction), out_dist,
            out_parent);
  });
}

mg_status mg_khop(mg_graph* g, const uint32_t* seeds, uint32_t num_seeds,
                  uint32_t k, mg_direction direction, uint32_t max_vertices,
                  uint64_t max_edges, uint32_t* out_vertices,
                  uint32_t* inout_num_vertices, uint32_t* out_edges,
                  uint64_t* inout_num_edges) {
  return wrap([&] {
    mg::Graph& gr = deref(g);
    require(inout_num_vertices != nullptr,
            "inout_num_vertices: must not be NULL");
    require(inout_num_edges != nullptr, "inout_num_edges: must not be NULL");
    require(num_seeds == 0 || seeds != nullptr,
            "seeds: must not be NULL when num_seeds > 0");
    mg::KhopOpts o;
    o.k = k;
    o.dir = bfs_dir(direction);
    o.max_vertices = max_vertices;
    o.max_edges = max_edges;
    mg::KhopResult r = mg::k_hop(gr, seeds, num_seeds, o);
    const uint32_t nv = static_cast<uint32_t>(r.vertices.size());
    const uint64_t ne = static_cast<uint64_t>(r.edges.size());
    // Capture capacities, then publish the needed counts (also on the
    // capacity-error path, so callers learn the required sizes).
    const uint32_t cap_v =
        out_vertices != nullptr ? *inout_num_vertices : 0;
    const uint64_t cap_e = out_edges != nullptr ? *inout_num_edges : 0;
    *inout_num_vertices = nv;
    *inout_num_edges = ne;
    if (out_vertices != nullptr) {
      require(cap_v >= nv,
              "out_vertices: capacity insufficient for k-hop result "
              "(required count stored in *inout_num_vertices)");
      if (nv > 0)
        std::memcpy(out_vertices, r.vertices.data(), nv * sizeof(uint32_t));
    }
    if (out_edges != nullptr) {
      require(cap_e >= ne,
              "out_edges: capacity insufficient for k-hop result "
              "(required count stored in *inout_num_edges)");
      if (ne > 0)
        std::memcpy(out_edges, r.edges.data(),
                    static_cast<size_t>(ne) * sizeof(uint32_t));
    }
  });
}

mg_status mg_wcc(mg_graph* g, int32_t* out_comp, int* out_num_components) {
  return wrap([&] {
    mg::Graph& gr = deref(g);
    require(out_comp != nullptr || gr.V == 0, "out_comp: must not be NULL");
    int n = mg::wcc(gr, out_comp);
    if (out_num_components) *out_num_components = n;
  });
}

}  // extern "C"
