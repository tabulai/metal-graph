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

// Telemetry for the last executed operation (planner assertion in tests).
struct RunInfo {
  std::string op;
  ExecPath path = ExecPath::cpu;
  int iterations = 0;   // pagerank/wcc rounds; bfs levels
  double elapsed_ms = 0.0;
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
  // 16-byte dummy buffer for never-dereferenced bindings.
  BufferPtr dummy() const;

  // ---- planner ----
  void set_mode(ExecMode m);
  ExecMode mode() const;
  // Base decision per operation: GPU iff mode==gpu, or mode==auto &&
  // has_gpu() && stored_edges >= e_gpu_min(). Algorithms may apply a
  // documented bounded preflight (BFS's tiny-component CPU latency path)
  // before using this route. mode==gpu bypasses such preflights and errors
  // without a device.
  ExecPath plan(uint64_t stored_edges) const;
  uint64_t e_gpu_min() const;  // MG_E_GPU_MIN env, default 1'000'000

  // ---- telemetry ----
  void record_run(const RunInfo& info);
  RunInfo last_run() const;

  ~Runtime();
  Runtime(const Runtime&) = delete;
  Runtime& operator=(const Runtime&) = delete;

 private:
  Runtime();
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
