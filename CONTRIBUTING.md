# Contributing

metal-graph targets Apple Silicon Macs running macOS 14 or newer. Native
development requires Xcode with the Metal compiler, CMake 3.24+, and
CPython 3.10–3.13.

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

Build and smoke-test a distributable wheel:

```bash
MACOSX_DEPLOYMENT_TARGET=14.0 python -m build --wheel
python -m pip install --force-reinstall dist/*.whl
cd /tmp
python -c "import metal_graph as mg; print(mg.__version__, mg.has_gpu())"
```

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
Do not publish hosted-runner timing. Include the raw JSON and rendered
Markdown for any result cited in project documentation, together with the
hardware and software metadata produced by the harness.

Keep changes focused, add regression tests for behavior changes, and run the
build, CTest, Python tests, and wheel smoke test before merging.
