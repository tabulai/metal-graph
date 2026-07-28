# Curated benchmark results

This directory retains only benchmark artifacts cited by project
documentation. Other local runs remain ignored.

- `bench-20260728T205737Z.{json,md}` is the canonical v0.1 smoke run used by
  the README table: Apple M4 Max, macOS 26.2, Python 3.13.3, 20 warm runs.

The JSON is the machine-readable source of truth; the Markdown file is the
rendered view produced by `bench/run.py`. Hosted CI timing must never be
added here.
