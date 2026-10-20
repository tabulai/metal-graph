# Changelog

All notable changes to metal-graph are documented here.

## Unreleased

- Exposed HITS convergence status and final hub L1 change through Python's
  `last_run_info()` and the additive C accessor `mg_last_run_convergence()`.
  Callers can distinguish convergence (including on the final allowed
  iteration), an exhausted iteration budget, and empty/edgeless graphs;
  existing score returns and C function signatures are unchanged.
- Performance: `k_hop` extracts induced edges through a bitmap over input
  edge ids instead of sorting them, and splits hub rows across CPU threads;
  `k_hop(as_graph=True)` remaps the subgraph through a dense index. GPU WCC
  is a single union-find link + compress pass, and CPU WCC uses a threaded
  union-find above `MG_WCC_CPU_PAR_MIN_EDGES` stored edges. Graph
  construction and the lazy IN transpose use per-chunk degree histograms,
  and `Graph.from_edges` factorizes dense integer ids without `np.unique`.
  CPU PageRank, personalized PageRank and HITS gather over edge-balanced,
  dynamically scheduled work items. GPU BFS splits high-degree frontier rows
  across threadgroups (`MG_BFS_HIGH_SPLIT_MAX`), and sparse-output BFS
  compacts its dense fallback in parallel.
- `ppr_topk` answers each query whose seeds' out-reach fits
  `MG_PPR_SPARSE_MAX_VERTICES` / `MG_PPR_SPARSE_MAX_EDGES` with an exact fp64
  solve on that reach, bit-identical to the CPU path. In a batch planned for
  the GPU only the remaining rows run there, so one batch can mix fp64 and
  fp32 rows; `last_run_info()` reports `path="cpu"` only when every row took
  the latency path.
- The auto planner uses per-operation GPU thresholds (`MG_E_GPU_MIN_<OP>`;
  `MG_E_GPU_MIN` still overrides every operation): PageRank and HITS move to
  the GPU from 200,000 stored edges, so graphs with 200,000 to 1,000,000
  edges now get float32 GPU results within the documented CPU/GPU tolerance,
  and `k_hop` no longer uses the GPU in auto mode.
- BFS's serial preflight now defaults to 8,192 vertices / 524,288 scanned
  edges when an overflowing query would run on the GPU (1,024 / 8,192 when it
  would run on the threaded CPU) and rejects a reach that cannot fit as soon
  as it is discovered.
- GPU WCC reports `iterations == 1` in `last_run_info()`, and
  `MG_WCC_ROUNDS_PER_BATCH` is ignored. CPU HITS on graphs with rows of
  16,384 or more edges can differ from 0.1.1 in the last float32 bit
  (tiled fp64 summation order); results remain deterministic for every
  thread count.

## 0.1.1 — 2026-09-04

- Added Metal-native HITS hub/authority centrality with a threaded fp64 CPU
  fallback, L1 normalization on-device, Python and C APIs, execution
  telemetry, and forced CPU/GPU correctness coverage. Edge weights are
  intentionally ignored; parallel edges retain their structural weight.
- Added a focused, accuracy-gated HITS benchmark against the matching CPU
  path, SciPy, NetworkX, rustworkx, and python-igraph, with graph setup and
  warm algorithm timings reported separately.

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
