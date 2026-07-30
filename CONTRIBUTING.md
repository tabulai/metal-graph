# Contributing

metal-graph targets Apple Silicon Macs running macOS 14 or newer. Native
development requires Xcode with the Metal compiler, CMake 3.24+, and
CPython 3.10–3.14.

## Development setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install ".[test]"

cmake -S . -B build/dev \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DPython_EXECUTABLE="$(python -c 'import sys; print(sys.executable)')"
cmake --build build/dev -j
```

The CMake developer build copies the native extension into
`python/metal_graph/`. Run the complete correctness suite with both CPU and
GPU paths enabled:

```bash
MG_REQUIRE_GPU=1 PYTHONPATH=python python -m pytest tests -q -ra
MG_REQUIRE_GPU=1 ctest --test-dir build/dev --output-on-failure
```

Build and smoke-test a single local wheel:

```bash
env -u MACOSX_DEPLOYMENT_TARGET python -m build --wheel
python -m pip install --force-reinstall dist/*.whl
cd /tmp
python -c "import metal_graph as mg; print(mg.__version__, mg.has_gpu())"
```

Unsetting the ambient variable makes this exercise the deployment target
declared by the project rather than silently substituting a shell setting.

## Production wheels

The `production wheels` workflow builds five interpreter-specific wheels:
CPython 3.10–3.14, each targeting only Apple Silicon and macOS 14 or newer.
Stable-ABI (`abi3`), Intel, universal2, Linux, Windows, PyPy, and free-threaded
Python wheels are intentionally outside the current release scope.

Production builds use the exact package versions in
`.github/wheel-build-requirements.txt`. Each workflow artifact includes the
wheels, SHA-256 hashes, the cibuildwheel log, and a per-wheel environment
record containing the Xcode, macOS SDK, CMake, nanobind, and
scikit-build-core versions that actually built it.

Every wheel is inspected as a standalone artifact. The release gate verifies
its metadata and RECORD hashes, bundled licenses, embedded README, arm64
Mach-O and macOS 14 minimum, Apple-only dynamic dependencies, code signature,
and the complete embedded Metal kernel library. cibuildwheel then installs
each wheel outside the checkout with `PYTHONPATH` unset, runs strict Twine and
`pip check`, and exercises both forced-CPU and forced-GPU operations. The
workflow deliberately requires an arm64 machine actually running macOS 14;
losing access to that runner blocks a release rather than weakening the gate.

## Releasing

Releases use GitHub Trusted Publishing; there are no long-lived PyPI tokens.
The `testpypi` and `pypi` GitHub environments accept only tags matching `v*`.
The tag must exactly match the version in `pyproject.toml`, which is the
project's single source of truth.

After the release changes have passed pull-request checks and landed on
`main`, push that exact tag (for example, `v0.1.0`). The production-wheel
workflow then:

1. builds and validates all five wheels on macOS 14 Apple Silicon;
2. publishes those same artifacts to TestPyPI;
3. installs the TestPyPI wheel on macOS 14 and exercises CPU and forced-GPU
   operations;
4. publishes to PyPI only after that smoke test succeeds; and
5. repeats the installation and runtime checks from PyPI.

Do not rerun a partially published release with altered artifacts or use
`skip-existing`. If a release fails after any upload, diagnose the existing
immutable release and publish a new version.

## C ABI

Configure without Python and install into an isolated prefix:

```bash
cmake -S . -B build/c-api \
  -DCMAKE_BUILD_TYPE=Release \
  -DMG_BUILD_PYTHON=OFF
cmake --build build/c-api -j
cmake --install build/c-api --prefix build/prefix
```

This installs `include/mg.h`, `lib/libmetalgraph.dylib`, and license notices.
The C smoke test and exported-symbol allowlist run through CTest.

## Benchmarks

Performance claims must come from `bench/run.py` on physical Apple Silicon.
Do not publish hosted-runner timing. For every numeric timing, speedup, or
comparative performance conclusion cited in project documentation, check in
and link a matched raw JSON and rendered Markdown report generated from the
same harness run, including the hardware and software metadata produced by
the harness. Exploratory or candidate results must not appear in project
documentation until that artifact pair exists. Baseline comparisons must
also use equivalent output semantics; label non-equivalent diagnostic rows
as context rather than gate evidence.

Keep changes focused, add regression tests for behavior changes, and run the
build, CTest, Python tests, and wheel smoke test before merging.
