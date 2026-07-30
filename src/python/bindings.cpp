// bindings.cpp — nanobind module `_core`: thin, validated bridge between
// numpy arrays and the C++ core. External-ID mapping lives in the Python
// package; everything here speaks dense USER indices. Core calls release
// the GIL; numpy allocation/wrapping happens with the GIL held.
// SPDX-License-Identifier: Apache-2.0
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

#include <algorithm>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <optional>
#include <utility>

#include "../algos/algos.hpp"
#include "../common/mg_common.hpp"
#include "../engines/bfs_batch_schedule.hpp"
#include "../graph/graph.hpp"
#include "../runtime/runtime.hpp"
#include "mg_version.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

struct GraphHandle {
  mg::GraphPtr g;
};

using ArrU32 =
    nb::ndarray<const uint32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using ArrU64 =
    nb::ndarray<const uint64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using ArrF32 =
    nb::ndarray<const float, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

// Transfer a new[]-allocated buffer to numpy; a capsule delete[]s it.
template <typename T>
nb::ndarray<nb::numpy, T> take_1d(std::unique_ptr<T[]>& p, size_t n) {
  T* raw = p.release();
  nb::capsule owner(raw, [](void* q) noexcept { delete[] static_cast<T*>(q); });
  return nb::ndarray<nb::numpy, T>(raw, {n}, owner);
}

template <typename T>
nb::ndarray<nb::numpy, T> take_2d(std::unique_ptr<T[]>& p, size_t rows,
                                  size_t cols) {
  T* raw = p.release();
  nb::capsule owner(raw, [](void* q) noexcept { delete[] static_cast<T*>(q); });
  return nb::ndarray<nb::numpy, T>(raw, {rows, cols}, owner);
}

mg::BfsDir bfs_dir(int d) {
  if (d < 0 || d > 2)
    mg::throw_invalid("direction: must be 0 (out), 1 (in) or 2 (both)");
  return static_cast<mg::BfsDir>(d);
}

GraphHandle build_graph(ArrU32 src, ArrU32 dst, std::optional<ArrF32> weights,
                        uint32_t num_vertices, bool directed) {
  if (src.size() != dst.size()) mg::throw_invalid("src/dst: length mismatch");
  const float* w = nullptr;
  if (weights) {
    if (weights->size() != src.size())
      mg::throw_invalid("weights: length must match src/dst");
    w = weights->data();
  }
  GraphHandle h;
  h.g = mg::Graph::build(src.data(), dst.data(), w, src.size(), num_vertices,
                         directed);
  return h;
}

nb::ndarray<nb::numpy, float> pagerank(GraphHandle& h, double alpha, double tol,
                                       int max_iter,
                                       std::optional<ArrF32> personalization) {
  mg::Graph& g = *h.g;
  const float* p = nullptr;
  if (personalization) {
    if (personalization->size() != g.V)
      mg::throw_invalid("personalization: length must equal num_vertices");
    p = personalization->data();
  }
  std::unique_ptr<float[]> rank(new float[g.V]);
  {
    nb::gil_scoped_release rel;
    mg::PagerankOpts o;
    o.alpha = alpha;
    o.tol = tol;
    o.max_iter = max_iter;
    o.personalization = p;
    mg::pagerank(g, o, rank.get());
  }
  return take_1d(rank, g.V);
}

nb::tuple ppr_topk(GraphHandle& h, ArrU32 seeds, ArrF32 seed_weights,
                   ArrU64 offsets, uint32_t k, double alpha, double tol,
                   int max_iter) {
  mg::Graph& g = *h.g;
  if (offsets.size() < 1)
    mg::throw_invalid("offsets: length must be B + 1 (>= 1)");
  const size_t B = offsets.size() - 1;
  if (B > std::numeric_limits<uint32_t>::max())
    mg::throw_invalid("offsets: too many queries");
  if (seed_weights.size() != seeds.size())
    mg::throw_invalid("seed_weights: length must match seeds");
  const uint64_t* off = offsets.data();
  if (off[0] != 0) mg::throw_invalid("offsets: must start at 0");
  for (size_t q = 0; q < B; ++q)
    if (off[q] > off[q + 1]) mg::throw_invalid("offsets: must be non-decreasing");
  if (off[B] != seeds.size())
    mg::throw_invalid("offsets: last entry must equal len(seeds)");
  const size_t total = B * static_cast<size_t>(k);
  std::unique_ptr<int32_t[]> ids(new int32_t[total]);
  std::unique_ptr<float[]> scores(new float[total]);
  {
    nb::gil_scoped_release rel;
    mg::PprTopkOpts o;
    o.alpha = alpha;
    o.tol = tol;
    o.max_iter = max_iter;
    o.k = k;
    mg::ppr_topk(g, seeds.data(), seed_weights.data(), off,
                 static_cast<uint32_t>(B), o, ids.get(), scores.get());
  }
  auto ids_arr = take_2d(ids, B, k);
  auto scores_arr = take_2d(scores, B, k);
  return nb::make_tuple(ids_arr, scores_arr);
}

nb::tuple bfs(GraphHandle& h, ArrU32 sources, int direction) {
  mg::Graph& g = *h.g;
  mg::BfsDir d = bfs_dir(direction);
  std::unique_ptr<int32_t[]> dist(new int32_t[g.V]);
  std::unique_ptr<int32_t[]> parent(new int32_t[g.V]);
  {
    nb::gil_scoped_release rel;
    mg::bfs(g, sources.data(), static_cast<uint32_t>(sources.size()), d,
            dist.get(), parent.get());
  }
  auto dist_arr = take_1d(dist, g.V);
  auto parent_arr = take_1d(parent, g.V);
  return nb::make_tuple(dist_arr, parent_arr);
}

nb::tuple bfs_sparse(GraphHandle& h, ArrU32 sources, int direction) {
  mg::Graph& g = *h.g;
  mg::BfsDir d = bfs_dir(direction);
  mg::SparseBfsResult r;
  {
    nb::gil_scoped_release rel;
    mg::bfs_sparse(g, sources.data(), static_cast<uint32_t>(sources.size()),
                   d, r);
  }
  const std::size_t n = r.vertices.size();
  std::unique_ptr<uint32_t[]> vs(new uint32_t[n]);
  std::unique_ptr<int32_t[]> dist(new int32_t[n]);
  std::unique_ptr<int32_t[]> parent(new int32_t[n]);
  std::copy(r.vertices.begin(), r.vertices.end(), vs.get());
  std::copy(r.dist.begin(), r.dist.end(), dist.get());
  std::copy(r.parent.begin(), r.parent.end(), parent.get());
  return nb::make_tuple(take_1d(vs, n), take_1d(dist, n),
                        take_1d(parent, n));
}

nb::tuple k_hop(GraphHandle& h, ArrU32 seeds, uint32_t k, int direction,
                uint32_t max_vertices, uint64_t max_edges) {
  mg::Graph& g = *h.g;
  mg::KhopOpts o;
  o.k = k;
  o.dir = bfs_dir(direction);
  o.max_vertices = max_vertices;
  o.max_edges = max_edges;
  mg::KhopResult r;
  {
    nb::gil_scoped_release rel;
    r = mg::k_hop(g, seeds.data(), static_cast<uint32_t>(seeds.size()), o);
  }
  const size_t nv = r.vertices.size();
  const size_t ne = r.edges.size();
  std::unique_ptr<uint32_t[]> vs(new uint32_t[nv]);
  std::unique_ptr<uint32_t[]> es(new uint32_t[ne]);
  std::copy(r.vertices.begin(), r.vertices.end(), vs.get());
  std::copy(r.edges.begin(), r.edges.end(), es.get());
  auto vs_arr = take_1d(vs, nv);
  auto es_arr = take_1d(es, ne);
  return nb::make_tuple(vs_arr, es_arr);
}

nb::tuple wcc(GraphHandle& h) {
  mg::Graph& g = *h.g;
  std::unique_ptr<int32_t[]> comp(new int32_t[g.V]);
  int n = 0;
  {
    nb::gil_scoped_release rel;
    n = mg::wcc(g, comp.get());
  }
  auto comp_arr = take_1d(comp, g.V);
  return nb::make_tuple(comp_arr, n);
}

// PRIVATE debug accessor (test-only, undocumented): worklist/permutation
// snapshot for one orientation. Everything is COPIED into fresh numpy
// arrays (capsule-owned heap memory) — never aliases the shared MTLBuffers.
nb::dict debug_orientation(GraphHandle& h, int direction) {
  if (direction != 0 && direction != 1)
    mg::throw_invalid("direction: must be 0 (out) or 1 (in)");
  mg::Graph& g = *h.g;
  const size_t V = g.V;
  std::unique_ptr<uint32_t[]> high, mid, low, zero, degrees, c_of_u, u_of_c;
  std::unique_ptr<uint32_t[]> huge, t_owner, t_first, t_off;
  size_t nh = 0, nm = 0, nl = 0, nz = 0, ng = 0, nt = 0;
  {
    nb::gil_scoped_release rel;  // may lazily build the IN orientation
    const mg::Orientation& o = g.orientation(static_cast<mg::Dir>(direction));
    nh = o.n_high;
    nm = o.n_mid;
    nl = o.n_low;
    nz = o.n_zero;
    ng = o.n_huge;
    nt = o.n_tiles;
    high.reset(new uint32_t[nh]);
    mid.reset(new uint32_t[nm]);
    low.reset(new uint32_t[nl]);
    zero.reset(new uint32_t[nz]);
    huge.reset(new uint32_t[ng]);
    t_owner.reset(new uint32_t[nt]);
    t_first.reset(new uint32_t[nt]);
    t_off.reset(new uint32_t[ng + 1]);
    if (nh) std::copy_n(o.wl_high->as<uint32_t>(), nh, high.get());
    if (nm) std::copy_n(o.wl_mid->as<uint32_t>(), nm, mid.get());
    if (nl) std::copy_n(o.wl_low->as<uint32_t>(), nl, low.get());
    if (nz) std::copy_n(o.zero_list->as<uint32_t>(), nz, zero.get());
    if (ng) {
      std::copy_n(o.wl_huge->as<uint32_t>(), ng, huge.get());
      std::copy_n(o.tile_owner->as<uint32_t>(), nt, t_owner.get());
      std::copy_n(o.tile_first_edge->as<uint32_t>(), nt, t_first.get());
      std::copy_n(o.huge_tile_off->as<uint32_t>(), ng + 1, t_off.get());
    } else {
      t_off[0] = 0;
    }
    degrees.reset(new uint32_t[V]);
    if (V) {
      const uint32_t* row = o.row_offsets->as<uint32_t>();
      for (size_t c = 0; c < V; ++c) degrees[c] = row[c + 1] - row[c];
    }
    c_of_u.reset(new uint32_t[V]);
    u_of_c.reset(new uint32_t[V]);
    std::copy(g.canon_of_user.begin(), g.canon_of_user.end(), c_of_u.get());
    std::copy(g.user_of_canon.begin(), g.user_of_canon.end(), u_of_c.get());
  }
  nb::dict d;
  d["wl_high"] = take_1d(high, nh);
  d["wl_mid"] = take_1d(mid, nm);
  d["wl_low"] = take_1d(low, nl);
  d["zero_list"] = take_1d(zero, nz);
  d["wl_huge"] = take_1d(huge, ng);
  d["tile_owner"] = take_1d(t_owner, nt);
  d["tile_first_edge"] = take_1d(t_first, nt);
  d["huge_tile_off"] = take_1d(t_off, ng + 1);
  d["degrees"] = take_1d(degrees, V);
  d["canon_of_user"] = take_1d(c_of_u, V);
  d["user_of_canon"] = take_1d(u_of_c, V);
  return d;
}

void set_execution(int mode) {
  if (mode < 0 || mode > 2)
    mg::throw_invalid("mode: must be 0 (auto), 1 (gpu) or 2 (cpu)");
  mg::Runtime::instance().set_mode(static_cast<mg::ExecMode>(mode));
}

nb::dict last_run_info() {
  mg::RunInfo info = mg::Runtime::instance().last_run();
  nb::dict d;
  d["op"] = info.op;
  d["path"] = (info.path == mg::ExecPath::gpu) ? "gpu" : "cpu";
  d["iterations"] = info.iterations;
  d["ms"] = info.elapsed_ms;
  return d;
}

// Private test hook for the exact parser/scheduler used by FrontierEngine.
// Simulates a device that never sets `done`, making both bounded traversal
// and the defensive termination guard observable without requiring a GPU.
nb::dict debug_bfs_batch_schedule(uint32_t vertex_count, uint32_t max_levels,
                                  uint64_t encoded, uint32_t max_batches) {
  mg::detail::BfsBatchSchedule schedule =
      mg::detail::BfsBatchSchedule::from_environment();
  nb::list batches;
  const uint32_t configured_levels = schedule.levels();
  const bool fixed = schedule.fixed();
  const uint64_t prepare_limit =
      mg::detail::bfs_prepare_limit(vertex_count);
  const char* stopped = "batch_limit";

  for (uint32_t i = 0; i < max_batches; ++i) {
    if (mg::detail::bfs_prepare_limit_exceeded(encoded, vertex_count)) {
      stopped = "safety";
      break;
    }
    uint32_t batch = schedule.next_batch(encoded, max_levels);
    batch = mg::detail::clamp_batch_to_prepare_limit(
        encoded, vertex_count, batch);
    if (batch == 0) {
      stopped = "level_bound";
      break;
    }
    batches.append(batch);
    encoded += batch;
    if (mg::detail::bfs_prepare_limit_exceeded(encoded, vertex_count)) {
      stopped = "safety";
      break;
    }
    schedule.advance();
  }

  nb::dict d;
  d["fixed"] = fixed;
  d["configured_levels"] = configured_levels;
  d["command_buffer_level_cap"] =
      mg::detail::k_bfs_command_buffer_level_cap;
  d["prepare_limit"] = prepare_limit;
  d["batches"] = batches;
  d["encoded"] = encoded;
  d["stopped"] = stopped;
  return d;
}

}  // namespace

NB_MODULE(_core, m) {
  m.doc() = "metal-graph core bindings (dense user-index space)";
  m.attr("__version__") = MG_VERSION_STRING;

  nb::register_exception_translator([](const std::exception_ptr& p, void*) {
    try {
      std::rethrow_exception(p);
    } catch (const mg::Error& e) {
      switch (e.code()) {
        case mg::ErrorCode::invalid_argument:
          PyErr_SetString(PyExc_ValueError, e.what());
          break;
        case mg::ErrorCode::not_implemented:
          PyErr_SetString(PyExc_NotImplementedError, e.what());
          break;
        default:
          PyErr_SetString(PyExc_RuntimeError, e.what());
          break;
      }
    }
    // Non-mg::Error exceptions rethrow past this handler and fall through
    // to nanobind's default translators.
  });

  nb::class_<GraphHandle>(m, "CoreGraph")
      .def_prop_ro("num_vertices", [](const GraphHandle& h) { return h.g->V; })
      .def_prop_ro("num_edges",
                   [](const GraphHandle& h) { return h.g->E_input; })
      .def_prop_ro("directed",
                   [](const GraphHandle& h) { return h.g->directed; })
      .def_prop_ro("weighted",
                   [](const GraphHandle& h) { return h.g->weighted; })
      .def_prop_ro("stored_edges",
                   [](const GraphHandle& h) { return h.g->stored_edges(); });

  m.def("build_graph", &build_graph, "src"_a, "dst"_a, "weights"_a.none(),
        "num_vertices"_a, "directed"_a,
        nb::call_guard<nb::gil_scoped_release>());
  m.def("pagerank", &pagerank, "graph"_a, "alpha"_a, "tol"_a, "max_iter"_a,
        "personalization"_a.none());
  m.def("ppr_topk", &ppr_topk, "graph"_a, "seeds"_a, "seed_weights"_a,
        "offsets"_a, "k"_a, "alpha"_a, "tol"_a, "max_iter"_a);
  m.def("bfs", &bfs, "graph"_a, "sources"_a, "direction"_a);
  m.def("bfs_sparse", &bfs_sparse, "graph"_a, "sources"_a, "direction"_a);
  m.def("k_hop", &k_hop, "graph"_a, "seeds"_a, "k"_a, "direction"_a,
        "max_vertices"_a, "max_edges"_a);
  m.def("wcc", &wcc, "graph"_a);
  m.def("set_execution", &set_execution, "mode"_a,
        nb::call_guard<nb::gil_scoped_release>());
  m.def("last_run_info", &last_run_info);
  // Private, test-only; no stability guarantee. direction: 0=out, 1=in.
  m.def("_debug_orientation", &debug_orientation, "graph"_a, "direction"_a);
  m.def("_debug_bfs_batch_schedule", &debug_bfs_batch_schedule,
        "vertex_count"_a, "max_levels"_a, "encoded"_a = 0,
        "max_batches"_a = 8);
  m.def("has_gpu", [] { return mg::Runtime::instance().has_gpu(); },
        nb::call_guard<nb::gil_scoped_release>());
  m.def("version", [] { return MG_VERSION_STRING; });
}
