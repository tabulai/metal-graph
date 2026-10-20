// cpu_wcc.cpp — CPU union-find WCC (CANONICAL space).
// Each component's representative is its MINIMUM canonical id, so labels
// are deterministic and identical across both implementations below.
// Directed edges are treated as undirected by uniting both endpoints of
// every stored OUT edge (matches the GPU link-both-endpoints semantics).
//
//  * wcc_serial: the sequential oracle (path halving + union by smaller
//    root id). Used when MG_THREADS=1, when the graph is small (stored
//    edges < MG_WCC_CPU_PAR_MIN_EDGES; thread start-up would dominate), or
//    if worker threads cannot be started.
//  * wcc_parallel: lock-free union-find over one set of worker threads with
//    phase barriers. link() CASes the LARGER root's parent slot from itself
//    to the smaller root, so every parent pointer is strictly smaller than
//    its owner and each tree's root is its component minimum under any
//    interleaving; path-halving stores only ever write an ancestor into a
//    non-root slot (never a CAS target), so trees never split. The final
//    labeling pass walks read-only and stores only each thread's own
//    vertices (path halving there could clobber another thread's stored
//    root with a stale ancestor). Edges are
//    split evenly by count (canonical order is degree-descending, so equal
//    vertex ranges would be badly imbalanced).
//    On SYMMETRIZED (undirected) storage it uses the Afforest skip: link the
//    first kAfforestK edges of every vertex, sample the most frequent root
//    (the giant component), then link the remaining edges only of vertices
//    outside it. Exact because every stored u!=v edge also exists reversed:
//    the copy at the non-giant endpoint is linked. Never used on directed
//    storage, where an out-edge from the giant component may be the only
//    stored copy.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <numeric>
#include <thread>
#include <vector>

#include "../algos.hpp"

namespace mg::cpu {

namespace {

int wcc_serial(uint32_t V, const uint32_t* row, const uint32_t* col,
               uint32_t* labels_canon) {
  std::vector<uint32_t> parent(V);
  std::iota(parent.begin(), parent.end(), 0u);
  auto find = [&](uint32_t x) {
    while (parent[x] != x) {
      parent[x] = parent[parent[x]];  // path halving
      x = parent[x];
    }
    return x;
  };

  for (uint32_t v = 0; v < V; ++v) {
    for (uint32_t e = row[v]; e < row[v + 1]; ++e) {
      const uint32_t ru = find(v);
      const uint32_t rv = find(col[e]);
      if (ru == rv) continue;
      if (ru < rv)
        parent[rv] = ru;  // smaller root wins -> min-id representative
      else
        parent[ru] = rv;
    }
  }

  for (uint32_t v = 0; v < V; ++v) labels_canon[v] = find(v);
  int n_components = 0;
  for (uint32_t v = 0; v < V; ++v)
    if (labels_canon[v] == v) ++n_components;
  return n_components;
}

using ARef = std::atomic_ref<uint32_t>;

// Root of x's tree (path halving). May return a stale root while other
// threads are linking; link() validates with CAS.
inline uint32_t uf_find(uint32_t* parent, uint32_t x) {
  for (;;) {
    const uint32_t p = ARef(parent[x]).load(std::memory_order_relaxed);
    if (p == x) return x;
    const uint32_t gp = ARef(parent[p]).load(std::memory_order_relaxed);
    if (gp == p) return p;
    ARef(parent[x]).store(gp, std::memory_order_relaxed);
    x = gp;
  }
}

// Root of x's tree without path compression: never writes. Safe while other
// threads store an ancestor (or the root) into their own slots, since every
// value read is still an ancestor of x.
inline uint32_t uf_root(uint32_t* parent, uint32_t x) {
  for (;;) {
    const uint32_t p = ARef(parent[x]).load(std::memory_order_relaxed);
    if (p == x) return x;
    x = p;
  }
}

inline void uf_link(uint32_t* parent, uint32_t a, uint32_t b) {
  for (;;) {
    uint32_t ra = uf_find(parent, a);
    uint32_t rb = uf_find(parent, b);
    if (ra == rb) return;
    if (ra < rb) std::swap(ra, rb);  // ra = larger root -> child of rb
    uint32_t expect = ra;
    if (ARef(parent[ra]).compare_exchange_strong(expect, rb,
                                                 std::memory_order_relaxed))
      return;
    a = ra;  // ra gained a parent concurrently: retry from the old roots
    b = rb;
  }
}

// Sense-free generation barrier for a fixed set of spinning threads.
class SpinBarrier {
 public:
  void set_count(unsigned n) { n_ = n; }
  void wait() {
    const unsigned g = gen_.load(std::memory_order_acquire);
    if (count_.fetch_add(1, std::memory_order_acq_rel) + 1 == n_) {
      count_.store(0, std::memory_order_relaxed);
      gen_.store(g + 1, std::memory_order_release);
      return;
    }
    unsigned spins = 0;
    while (gen_.load(std::memory_order_acquire) == g)
      if (++spins > 256) std::this_thread::yield();
  }

 private:
  unsigned n_ = 1;
  std::atomic<unsigned> count_{0};
  std::atomic<unsigned> gen_{0};
};

constexpr uint32_t kAfforestK = 2;         // edges per vertex linked first
constexpr int kAfforestSamples = 1024;     // root samples for the giant comp

// Link every edge in the CSR slot range [eb, ee) (whole graph split evenly
// by edge count). skip_root != UINT32_MAX enables the Afforest skip: the
// first kAfforestK edges of each vertex are excluded (already linked) and
// vertices whose root is skip_root are skipped entirely.
inline void link_edge_range(uint32_t* parent, const uint32_t* row,
                            const uint32_t* col, uint32_t V, uint64_t eb,
                            uint64_t ee, uint32_t skip_root) {
  if (eb >= ee) return;
  // First vertex whose row contains slot eb.
  uint32_t v = static_cast<uint32_t>(
      std::upper_bound(row, row + V + 1, static_cast<uint32_t>(eb)) - row -
      1);
  for (; v < V && row[v] < ee; ++v) {
    uint64_t b = std::max<uint64_t>(row[v], eb);
    const uint64_t e = std::min<uint64_t>(row[v + 1], ee);
    if (skip_root != UINT32_MAX)
      b = std::max<uint64_t>(b, uint64_t(row[v]) + kAfforestK);
    if (b >= e) continue;
    // rv: a node of v's set (its root when last looked). ru == rv proves
    // both endpoints share a set even if rv has since gone stale.
    uint32_t rv = uf_find(parent, v);
    if (rv == skip_root) continue;  // Afforest: v is in the giant component
    for (uint64_t k = b; k < e; ++k) {
      const uint32_t ru = uf_find(parent, col[k]);
      if (ru == rv) continue;
      uf_link(parent, rv, ru);
      rv = uf_find(parent, rv);
    }
  }
}

// Most frequent root among a fixed pseudo-random vertex sample (ties ->
// smallest root). Deterministic: the partition is fixed when it runs.
uint32_t sample_giant_root(uint32_t* parent, uint32_t V) {
  std::vector<uint32_t> roots(kAfforestSamples);
  uint64_t s = 88172645463325252ull;  // xorshift64, fixed seed
  for (auto& r : roots) {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    r = uf_find(parent, static_cast<uint32_t>(s % V));
  }
  std::sort(roots.begin(), roots.end());
  uint32_t best_root = roots[0], best = 0, run = 0;
  for (std::size_t i = 0; i < roots.size(); ++i) {
    run = (i != 0 && roots[i] == roots[i - 1]) ? run + 1 : 1;
    if (run > best) {
      best = run;
      best_root = roots[i];
    }
  }
  return best_root;
}

// Returns the number of components, or -1 if worker threads could not be
// started (caller falls back to the serial path).
int wcc_parallel(uint32_t V, const uint32_t* row, const uint32_t* col,
                 bool afforest, unsigned T, uint32_t* parent) {
  const uint64_t E = row[V];
  SpinBarrier barrier;
  std::atomic<int> go{0};  // 0 wait, 1 run, -1 abort
  std::vector<uint32_t> n_roots(T, 0);
  uint32_t giant = UINT32_MAX;

  auto vrange = [&](unsigned t, uint32_t& b, uint32_t& e) {
    b = static_cast<uint32_t>(uint64_t(V) * t / T);
    e = static_cast<uint32_t>(uint64_t(V) * (t + 1) / T);
  };
  auto worker = [&](unsigned t) {
    int state;
    while ((state = go.load(std::memory_order_acquire)) == 0)
      std::this_thread::yield();
    if (state < 0) return;
    uint32_t vb, ve;
    vrange(t, vb, ve);
    for (uint32_t v = vb; v < ve; ++v)
      ARef(parent[v]).store(v, std::memory_order_relaxed);
    barrier.wait();
    if (afforest) {
      for (uint32_t v = vb; v < ve; ++v) {
        const uint32_t end = std::min(row[v] + kAfforestK, row[v + 1]);
        for (uint32_t k = row[v]; k < end; ++k) uf_link(parent, v, col[k]);
      }
      barrier.wait();
      if (t == 0) giant = sample_giant_root(parent, V);
      barrier.wait();
    }
    link_edge_range(parent, row, col, V, E * t / T, E * (t + 1) / T,
                    afforest ? giant : UINT32_MAX);
    barrier.wait();
    // Final labeling: the forest is frozen, but other threads are storing
    // their own vertices' roots. uf_find's path halving would write a
    // (now stale) grandparent into slots owned by other threads and could
    // overwrite a root stored there, so walk read-only and store only v.
    uint32_t roots = 0;
    for (uint32_t v = vb; v < ve; ++v) {
      const uint32_t r = uf_root(parent, v);
      roots += (r == v);
      ARef(parent[v]).store(r, std::memory_order_relaxed);
    }
    n_roots[t] = roots;
  };

  std::vector<std::thread> threads;
  threads.reserve(T - 1);
  try {
    for (unsigned t = 1; t < T; ++t) threads.emplace_back(worker, t);
  } catch (...) {
    go.store(-1, std::memory_order_release);
    for (auto& th : threads) th.join();
    return -1;
  }
  barrier.set_count(T);
  go.store(1, std::memory_order_release);
  worker(0);
  for (auto& th : threads) th.join();
  uint64_t total = 0;
  for (uint32_t r : n_roots) total += r;
  return static_cast<int>(total);
}

}  // namespace

int wcc(Graph& g, uint32_t* labels_canon) {
  const uint32_t V = g.V;
  if (V == 0) return 0;
  const Orientation& o = g.orientation(Dir::out);
  const uint32_t* row = o.row_offsets->as<uint32_t>();
  const uint32_t* col = o.col_indices->as<uint32_t>();
  const uint64_t E = row[V];

  const unsigned nt = num_threads();
  // Below ~128K edges the serial pass beats thread start-up (~12 us per
  // thread); above it, ~64K edges per thread (at least 4 threads).
  const long min_edges = env_long("MG_WCC_CPU_PAR_MIN_EDGES", 131072);
  if (nt > 1 && E > 0 && E >= static_cast<uint64_t>(std::max(0L, min_edges))) {
    const uint64_t want = std::max<uint64_t>(4, E >> 16);
    const unsigned T = static_cast<unsigned>(
        std::min<uint64_t>(std::min<uint64_t>(nt, want), V));
    // labels_canon doubles as the parent array; it ends fully compressed.
    const int n = wcc_parallel(V, row, col, !g.directed, T, labels_canon);
    if (n >= 0) return n;
  }
  return wcc_serial(V, row, col, labels_canon);
}

}  // namespace mg::cpu
