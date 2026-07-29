// bfs_batch_schedule.hpp — bounded, overflow-safe command-buffer scheduling
// for the GPU BFS level protocol. Internal to the host implementation and
// its private test hook; this is not part of the public API.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdlib>

namespace mg::detail {

inline constexpr uint32_t k_bfs_initial_batch = 8;
inline constexpr uint32_t k_bfs_adaptive_cap = 64;
// One level encodes three compute encoders and up to seven dispatches. Keep
// even explicit tuning overrides below a practical per-command-buffer bound.
inline constexpr uint32_t k_bfs_command_buffer_level_cap = 256;
inline constexpr uint64_t k_bfs_termination_slack = 2;

class BfsBatchSchedule {
 public:
  static BfsBatchSchedule from_environment() {
    const char* value = std::getenv("MG_BFS_LEVELS_PER_BATCH");
    if (value == nullptr || *value == '\0')
      return BfsBatchSchedule(k_bfs_initial_batch, false);

    char* end = nullptr;
    errno = 0;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || end == nullptr || *end != '\0' || errno == ERANGE)
      return BfsBatchSchedule(k_bfs_initial_batch, false);

    uint32_t levels = 1;
    if (parsed > 0) {
      levels = static_cast<uint32_t>(
          std::min<uint64_t>(static_cast<uint64_t>(parsed),
                             k_bfs_command_buffer_level_cap));
    }
    return BfsBatchSchedule(levels, true);
  }

  uint32_t next_batch(uint64_t encoded, uint32_t max_levels) const {
    if (max_levels == 0) return levels_;
    const uint64_t bound = max_levels;
    if (encoded >= bound) return 0;
    return static_cast<uint32_t>(
        std::min<uint64_t>(levels_, bound - encoded));
  }

  void advance() {
    if (fixed_) return;
    ++adaptive_batches_issued_;
    // Commit at cumulative levels 8, 16, 32, 64, then every 64 levels.
    // Compared with 8,16,32,... batch sizes, the repeated initial 8 avoids
    // overshooting the latency-sensitive power-of-two transition depths.
    if (adaptive_batches_issued_ >= 2 && levels_ < k_bfs_adaptive_cap)
      levels_ = std::min(levels_ * 2u, k_bfs_adaptive_cap);
  }

  uint32_t levels() const { return levels_; }
  bool fixed() const { return fixed_; }

 private:
  BfsBatchSchedule(uint32_t levels, bool fixed)
      : levels_(levels), fixed_(fixed) {}

  uint32_t levels_;
  uint32_t adaptive_batches_issued_ = 0;
  bool fixed_;
};

inline uint64_t bfs_prepare_limit(uint32_t vertex_count) {
  return uint64_t{vertex_count} + k_bfs_termination_slack +
         k_bfs_command_buffer_level_cap;
}

inline bool bfs_prepare_limit_exceeded(uint64_t encoded,
                                       uint32_t vertex_count) {
  return encoded > bfs_prepare_limit(vertex_count);
}

// If the device protocol fails to set `done`, make the final command buffer
// land exactly one prepare past the guard rather than overshooting it by a
// whole batch. A valid traversal sets done by V+1 prepares, well before this
// defensive V+2+cap boundary.
inline uint32_t clamp_batch_to_prepare_limit(uint64_t encoded,
                                             uint32_t vertex_count,
                                             uint32_t requested) {
  const uint64_t limit = bfs_prepare_limit(vertex_count);
  if (encoded > limit) return 0;
  return static_cast<uint32_t>(
      std::min<uint64_t>(requested, (limit + 1u) - encoded));
}

}  // namespace mg::detail
