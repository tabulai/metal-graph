# Changelog

All notable changes to metal-graph are documented here.

## 0.1.0 — 2026-07-28

- Added Metal-native PageRank, batched personalized PageRank with top-k,
  direction-optimizing BFS, bounded k-hop extraction, and experimental WCC.
- Added CPU fallbacks, execution-path telemetry, stable external-ID mapping,
  a C ABI, and GPU-required golden tests against NetworkX.
- Added deterministic benchmark reporting for physical Apple Silicon.
- Rejected non-finite graph, personalization, and seed weights, including
  overflowing fp32 outgoing-weight sums.
- Added installable CPython wheels, CMake installation for the C ABI,
  explicit ABI symbol visibility, and macOS 14 deployment targeting.
- Added a bounded, output-direct CPU BFS latency path for tiny reachable
  components while preserving forced-GPU execution and dense results.
- Reduced the default GPU BFS command batch from 16 to 8 levels and guarded
  bottom-up traversal against sparse high-degree hub frontiers.
- Made the rustworkx BFS gate return equivalent dense distance and parent
  arrays, with no-output and sparse-layer calls reported separately.
