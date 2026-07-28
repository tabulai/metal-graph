// scan_engine.cpp — recursive two-pass exclusive scan (scan.metal):
// mg_exscan_partial per block, recurse on the per-block sums while more
// than one block remains, then mg_exscan_apply adds block offsets back.
// SPDX-License-Identifier: Apache-2.0
#include "scan_engine.hpp"

#include "../kernels/mg_params.h"

namespace mg {

ScanEngine::ScanEngine(Runtime& rt) : rt_(rt) {
  if (!rt_.has_gpu())
    throw Error(ErrorCode::no_gpu, "scan engine requires a Metal device");
  partial_ = rt_.pipeline("mg_exscan_partial");
  apply_ = rt_.pipeline("mg_exscan_apply");
}

uint64_t ScanEngine::exclusive_scan(Buffer& in, Buffer& out, uint32_t n) {
  if (n == 0) return 0;
  const uint32_t blocks = ceil_div_u32(n, MG_SCAN_BLOCK);
  BufferPtr sums =
      rt_.alloc(std::size_t(blocks) * sizeof(uint32_t), "scan.sums");
  MGScanParams p{n, 0u, 0u, 0u};
  {
    CommandBatch cb(rt_);
    cb.dispatch(partial_, &p, sizeof(p), {&in, &out, sums.get()}, blocks,
                MG_TG_SIZE);
    cb.commit_and_wait();
  }
  if (blocks > 1) {
    BufferPtr sums_scan =
        rt_.alloc(std::size_t(blocks) * sizeof(uint32_t), "scan.sums_scan");
    exclusive_scan(*sums, *sums_scan, blocks);
    CommandBatch cb(rt_);
    cb.dispatch(apply_, &p, sizeof(p), {&out, sums_scan.get()}, blocks,
                MG_TG_SIZE);
    cb.commit_and_wait();
  }
  const uint32_t last_off = out.as<uint32_t>()[n - 1];
  const uint32_t last_val = in.as<uint32_t>()[n - 1];
  return uint64_t(last_off) + uint64_t(last_val);
}

}  // namespace mg
