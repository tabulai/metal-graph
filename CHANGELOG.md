# Changelog

All notable changes to metal-graph are documented here.

## Unreleased

- Added opt-in sparse-output BFS: `mg.bfs(..., output="sparse")` returns
  `(vertices, dist, parent)` of length |reached| (ascending user order on
  every execution path, structural-parent semantics unchanged). Tiny
  neighborhoods on huge graphs avoid the O(V) dense-output cost entirely
  (calloc-backed visited scratch touches O(reached) pages).
- Reframed the tiny-component BFS gate as an absolute-latency SLO (≤ 50 µs,
  `tiny_component_slo` bench row): microsecond-scale ratio gates compare
  output contracts, not engines. `rustworkx bfs_layers` is now explicitly
  labeled as excluded from gates (different output contract); a
  `warm_single_source_sparse_output` context row measures the new API.

- Pinned the wheel platform tag to the binary's real macOS 14.0 floor
  (`cmake.define.CMAKE_OSX_DEPLOYMENT_TARGET` in `pyproject.toml`); wheels
  built on newer hosts were tagged with the host OS version and could not
  install on supported macOS 14/15.
- Stopped pip wheel builds from copying `_core` into the source tree (the
  POST_BUILD copy is now dev-loop only, guarded by `if(NOT SKBUILD)`).
- Raised the Python upper bound to `<3.15` and added the 3.14 classifier
  and CI lane.
- Made the GPU BFS command-batch size adaptive (8, 8, 16, 32, then 64
  levels) to reduce command-buffer completions on deep-diameter traversals
  while preserving cumulative checks at depths 8, 16, 32, and 64.
  `MG_BFS_LEVELS_PER_BATCH` pins a fixed size clamped to 1–256; malformed
  values retain the adaptive default. The scheduler and its termination
  guard now use overflow-safe accounting; direct schedule, bounded k-hop,
  and 4096-level chain regressions cover the behavior.
- Restored golden-matrix coverage of the threaded CPU BFS oracle: the BFS
  matrix now also runs with the sparse preflight disabled.
- Annotated the retroactive edits to the dated v0.1 plan record (rev C
  changelog) and flagged that the canonical benchmark table's rustworkx BFS
  baseline predates the equivalent-work harness semantics. Removed
  unpublished candidate timings and clarified that documentation performance
  claims require a matched, checked-in JSON and rendered report.

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
