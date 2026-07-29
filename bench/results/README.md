# Curated benchmark results

This directory retains only benchmark artifacts cited by project
documentation. Other local runs remain ignored.

- `bench-20260729T000207Z.{json,md}` is the canonical v0.1 smoke run used by
  the README table: Apple M4 Max, macOS 26.2, Python 3.13.3, 20 warm runs,
  clean source commit `be8763c8b87d`.

The JSON is the machine-readable source of truth; the Markdown file is the
rendered view produced by `bench/run.py`. It includes a strict schema,
source and native-binary provenance, and the pinned SNAP manifest. Hosted CI
timing must never be added here.
