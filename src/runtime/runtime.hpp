// runtime.hpp — Metal device/runtime layer: device, queue, pipeline cache,
// tracked-buffer allocator, command-batch helper, planner + telemetry.
//
// v0.1 synchronization protocol (plan §6): individually tracked MTLBuffers
// (default hazard tracking), ONE command queue, serial compute encoders.
// The CPU reads results only after commit_and_wait() at operation
// boundaries. Encoder breaks are REQUIRED before any dispatch that consumes
// GPU-written indirect arguments (CommandBatch::break_encoder()).
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <limits>
#include <memory>
#include <string>

#include "../common/mg_common.hpp"

// metal-cpp forward-includable header (namespaces MTL/NS). Implementation
// TU (src/runtime/metalcpp_impl.cpp) defines the *_PRIVATE_IMPLEMENTATION
// macros exactly once.
#include <Metal/Metal.hpp>

namespace mg {

// ---------------------------------------------------------------------------
// Buffer: shared-storage MTLBuffer when a GPU exists, plain heap otherwise.
// data() is always a valid CPU pointer of byte_size() bytes.
// ---------------------------------------------------------------------------
class Buffer {
 public:
  virtual ~Buffer() = default;
  virtual void* data() = 0;
  virtual std::size_t byte_size() const = 0;
  virtual MTL::Buffer* mtl() = 0;  // nullptr in CPU-only mode

  template <typename T>
  T* as() {
    return static_cast<T*>(data());
  }
};
using BufferPtr = std::shared_ptr<Buffer>;

enum class ExecMode : int { autom = 0, gpu = 1, cpu = 2 };
enum class ExecPath : int { gpu = 0, cpu = 1 };

// Operation families with their own auto-planner GPU threshold (stored
// edges). Defaults (Runtime::e_gpu_min): 1'000'000 for numeric and traversal
// operations, preserving the original CPU fp64 range for pagerank/hits;
// khop never (CPU vertex discovery beats a GPU BFS at every measured size;
// induced-edge extraction is a host pass on both paths). Env precedence:
// MG_E_GPU_MIN_<OP> > MG_E_GPU_MIN (global override for every op) > per-op
// default.
enum class PlanOp : int { generic = 0, pagerank, hits, ppr, bfs, wcc, khop };

enum class StopReason { not_applicable, converged, max_iter, empty_graph };

// Telemetry for the last executed operation (planner assertion in tests).
struct RunInfo {
  std::string op;
  ExecPath path = ExecPath::cpu;
  int iterations = 0;   // pagerank/HITS iterations; WCC passes; BFS levels
  double elapsed_ms = 0.0;
  // Only HITS reports convergence so far. Each operation starts with these
  // defaults so an unrelated run cannot retain a previous HITS outcome.
  StopReason stop_reason = StopReason::not_applicable;
  double final_delta_l1 = std::numeric_limits<double>::quiet_NaN();
};

// ---------------------------------------------------------------------------
// Runtime singleton
// ---------------------------------------------------------------------------
class CommandBatch;

class Runtime {
 public:
  // Creates the Metal device + queue + library on first use.
  // MG_FORCE_CPU=1 disables the GPU entirely (buffers become heap memory).
  // MG_REQUIRE_GPU=1 turns "no Metal device" into a hard Error(no_gpu).
  static Runtime& instance();

  bool has_gpu() const;
  MTL::Device* device() const;        // nullptr in CPU-only mode
  MTL::CommandQueue* queue() const;   // nullptr in CPU-only mode

  // Cached compute pipeline. `weighted` / `uniform_p` map to the function
  // constants MG_FC_WEIGHTED / MG_FC_UNIFORM_P; kernels that do not declare
  // a constant ignore the flag (cache key includes both flags regardless).
  MTL::ComputePipelineState* pipeline(const char* kernel_name,
                                      bool weighted = false,
                                      bool uniform_p = false);
  // threadExecutionWidth of a pipeline (for sgs_per_tg host computations).
  uint32_t exec_width(MTL::ComputePipelineState* p) const;

  // Shared-mode tracked buffer (zero-initialized: MG guarantees this).
  BufferPtr alloc(std::size_t bytes, const char* label);
  // Same, but contents are UNSPECIFIED: for scratch that the host or the GPU
  // fully writes before any read. Skipping the zero-fill skips its host
  // first-touch page faults, the dominant cost of a fresh V-sized buffer.
  // MG_DEBUG_POISON_SCRATCH=1 fills these buffers with 0xFF bytes instead.
  BufferPtr alloc_scratch(std::size_t bytes, const char* label);
  // 16-byte dummy buffer for never-dereferenced bindings.
  BufferPtr dummy() const;

  // ---- planner ----
  void set_mode(ExecMode m);
  ExecMode mode() const;
  // Base decision per operation: GPU iff mode==gpu, or mode==auto &&
  // has_gpu() && stored_edges >= e_gpu_min(op). Algorithms may apply a
  // documented bounded preflight (BFS's tiny-component CPU latency path)
  // before using this route. mode==gpu bypasses such preflights and errors
  // without a device.
  ExecPath plan(uint64_t stored_edges, PlanOp op = PlanOp::generic) const;
  // Per-op threshold (see PlanOp); UINT64_MAX = never GPU under auto.
  uint64_t e_gpu_min(PlanOp op = PlanOp::generic) const;

  // ---- telemetry ----
  void record_run(const RunInfo& info);
  RunInfo last_run() const;

  ~Runtime();
  Runtime(const Runtime&) = delete;
  Runtime& operator=(const Runtime&) = delete;

 private:
  Runtime();
  BufferPtr make_buffer(std::size_t bytes, const char* label, bool zero);
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// ---------------------------------------------------------------------------
// CommandBatch: one MTLCommandBuffer with serial compute encoders.
// Dispatches within one encoder are serial (memory effects ordered).
// Call break_encoder() before a dispatch that consumes indirect args
// written by an earlier dispatch in this batch.
// ---------------------------------------------------------------------------
class CommandBatch {
 public:
  explicit CommandBatch(Runtime& rt);
  ~CommandBatch();

  // Bind-and-dispatch helpers. Params are copied via setBytes at buffer
  // index 0; `buffers` are bound at indices 1..n in order (matching the
  // mg_params.h binding tables). Pass Runtime::dummy() (never nullptr) for
  // unused slots.
  void dispatch(MTL::ComputePipelineState* pso, const void* params,
                std::size_t params_size, std::initializer_list<Buffer*> buffers,
                uint32_t threadgroups, uint32_t threads_per_tg);

  void dispatch_indirect(MTL::ComputePipelineState* pso, const void* params,
                         std::size_t params_size,
                         std::initializer_list<Buffer*> buffers,
                         Buffer* args, std::size_t args_byte_offset,
                         uint32_t threads_per_tg);

  void break_encoder();  // end current encoder; next dispatch opens a new one

  // Commit, wait, and surface MTLCommandBuffer errors as mg::Error.
  void commit_and_wait();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace mg
