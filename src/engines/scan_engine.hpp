// scan_engine.hpp — device exclusive scan over uint32 (host driver for
// scan.metal). v0.1 is synchronous: commit_and_wait between recursion
// levels; totals are read from shared memory after the batch completes.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

#include "../runtime/runtime.hpp"

namespace mg {

class ScanEngine {
 public:
  explicit ScanEngine(Runtime& rt);

  // Exclusive scan of in[0..n) into out[0..n). Both are uint32 device
  // buffers of at least n elements; they must not alias (`in` is not
  // modified). Returns the total sum (out[n-1] + in[n-1]).
  uint64_t exclusive_scan(Buffer& in, Buffer& out, uint32_t n);

 private:
  Runtime& rt_;
  MTL::ComputePipelineState* partial_ = nullptr;
  MTL::ComputePipelineState* apply_ = nullptr;
};

}  // namespace mg
