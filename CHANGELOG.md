# Changelog

All notable changes to metal-graph are documented here.

## Unreleased

## 0.1.0 — 2026-07-29

- Added tag-only Trusted Publishing through isolated GitHub environments.
  Release tags now build and validate once, publish the exact artifacts to
  TestPyPI, install and exercise them on macOS 14 Apple Silicon, then publish
  to PyPI and repeat the installation, CPU, and forced-GPU smoke tests.
- Added production cibuildwheel builds for interpreter-specific CPython
  3.10–3.14 wheels on macOS 14/arm64 only. Wheel builds now omit the standalone
  C library and native test executables, use a pinned build toolchain, verify
  binary deployment metadata, and retain Xcode, SDK, CMake, nanobind, and
  scikit-build-core provenance with the artifacts. abi3, non-arm64, and
  non-macOS variants remain deferred.
- Strengthened release artifacts with per-wheel metadata, RECORD, license,
  Apple-only linkage, code-signature, and embedded-Metal validation. Every
  wheel is now installed outside the repository with `PYTHONPATH` unset,
  checked by pip and strict Twine rendering, and exercised through forced CPU
  operations on actual macOS 14 Apple Silicon plus forced GPU operations for
  every interpreter on macOS 15 Apple Silicon. Complete notices for the
  statically linked nanobind runtime and its bundled robin-map dependency are
  included.
- Made `pyproject.toml` the single source of truth for the package, Python
  extension, and C API version; CMake now generates the public version header.
- Enforced native Apple Silicon/arm64 source builds at host, requested-target,
  and compiler-target levels.
- Made `Graph.external_ids` a detached, read-only NumPy snapshot so callers
  cannot reach and invalidate the immutable graph's internal ID mapping.
- Enabled GitHub private vulnerability reporting and corrected the security
  policy to use the private advisory form.
- Added `bench/run.py --dataset` for isolated single-dataset runs with
  unchanged artifact provenance, and checked in the isolated HippoRAG KG
  re-measurement (`bench-20260729T201007Z`): PPR B=16 batch 10.707 ms /
  0.669 ms per query (amortized ≤0.7 ms target met; ≤10 ms batch target
  missed by 7%), superseding the interrupted full-suite run's anomalous
  15.116 ms reading for gate purposes. The same artifact records KG
  high-degree BFS beating the corrected equivalent-output igraph adapter
  (1.857 ms vs 2.321 ms).

- Added opt-in sparse-output BFS: `mg.bfs(..., output="sparse")` returns
  `(vertices, dist, parent)` of length |reached| (ascending user order on
  every execution path, structural-parent semantics unchanged). Tiny
  neighborhoods that finish within the configured caps avoid O(V) dense
  result initialization; the calloc-backed V-byte visited map touches
  O(reached) pages on that path.
- Reframed the tiny-component BFS comparison as an absolute-latency SLO
  assessment (≤ 50 µs, annotated on the measured `warm_single_source` row):
  microsecond-scale ratios can compare output contracts rather than engines.
  `rustworkx bfs_layers` is explicitly excluded from equivalent-work
  comparisons; a `warm_single_source_sparse_output` context row measures the
  new API.

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
