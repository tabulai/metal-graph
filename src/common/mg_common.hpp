// mg_common.hpp — shared host-side basics: errors, env helpers, threading.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <cstdlib>
#include <functional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace mg {

// Error codes mirror mg_status in include/mg.h (keep in sync).
enum class ErrorCode : int {
  ok = 0,
  invalid_argument = 1,
  out_of_memory = 2,
  no_gpu = 3,
  internal = 4,
  not_implemented = 5,
};

class Error : public std::runtime_error {
 public:
  Error(ErrorCode code, const std::string& msg)
      : std::runtime_error(msg), code_(code) {}
  ErrorCode code() const { return code_; }

 private:
  ErrorCode code_;
};

[[noreturn]] inline void throw_invalid(const std::string& msg) {
  throw Error(ErrorCode::invalid_argument, msg);
}

inline const char* env_str(const char* name, const char* fallback) {
  const char* v = std::getenv(name);
  return (v && *v) ? v : fallback;
}

inline long env_long(const char* name, long fallback) {
  const char* v = std::getenv(name);
  if (!v || !*v) return fallback;
  char* end = nullptr;
  long r = std::strtol(v, &end, 10);
  return (end && *end == '\0') ? r : fallback;
}

inline bool env_flag(const char* name, bool fallback = false) {
  const char* v = std::getenv(name);
  if (!v || !*v) return fallback;
  return !(v[0] == '0' && v[1] == '\0');
}

inline unsigned num_threads() {
  long t = env_long("MG_THREADS", 0);
  if (t > 0) return static_cast<unsigned>(t);
  unsigned hc = std::thread::hardware_concurrency();
  return hc ? hc : 4;
}

// Chunked parallel for over [0, n): fn(begin, end) per chunk.
// Runs inline when n is small or one thread is available.
inline void parallel_for(std::size_t n,
                         const std::function<void(std::size_t, std::size_t)>& fn,
                         std::size_t grain = 4096) {
  unsigned nt = num_threads();
  if (n == 0) return;
  if (nt <= 1 || n <= grain) {
    fn(0, n);
    return;
  }
  std::size_t chunks = std::min<std::size_t>(nt, (n + grain - 1) / grain);
  std::size_t per = (n + chunks - 1) / chunks;
  std::vector<std::thread> ts;
  ts.reserve(chunks);
  for (std::size_t c = 0; c < chunks; ++c) {
    std::size_t b = c * per, e = std::min(n, b + per);
    if (b >= e) break;
    ts.emplace_back([&fn, b, e] { fn(b, e); });
  }
  for (auto& t : ts) t.join();
}

inline uint32_t ceil_div_u32(uint64_t a, uint64_t b) {
  return static_cast<uint32_t>((a + b - 1) / b);
}

}  // namespace mg
