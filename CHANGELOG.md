# Changelog

All notable changes to metal-graph are documented here.

## Unreleased

- Pinned the wheel platform tag to the binary's real macOS 14.0 floor
  (`cmake.define.CMAKE_OSX_DEPLOYMENT_TARGET` in `pyproject.toml`); wheels
  built on newer hosts were tagged with the host OS version and could not
  install on supported macOS 14/15.
- Stopped pip wheel builds from copying `_core` into the source tree (the
  POST_BUILD copy is now dev-loop only, guarded by `if(NOT SKBUILD)`).
- Raised the Python upper bound to `<3.15` and added the 3.14 classifier
  and CI lane.
- Made the GPU BFS command-batch size adaptive (8 doubling to 64 levels;
  `MG_BFS_LEVELS_PER_BATCH` pins a fixed size): keeps shallow-graph latency
  while restoring deep-diameter sync counts that the fixed 8-level default
  had doubled versus the original 16. Added a 4096-level chain regression
  test.
- Restored golden-matrix coverage of the threaded CPU BFS oracle: the BFS
  matrix now also runs with the sparse preflight disabled.
- Annotated the retroactive edits to the dated v0.1 plan record (rev C
  changelog) and flagged that the canonical benchmark table's rustworkx BFS
  baseline predates the equivalent-work harness semantics.

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
