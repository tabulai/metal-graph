# Curated benchmark results

This directory retains only benchmark artifacts cited by project
documentation. Other local runs remain ignored.

- `bench-20260729T000207Z.{json,md}` is the canonical v0.1 smoke run used by
  the README table: Apple M4 Max, macOS 26.2, Python 3.13.3, 20 warm runs,
  clean source commit `be8763c8b87d`.
- `bench-20260729-full-partial.log` preserves the completed rows from the
  2026-07-28–29 full-suite rerun. The monolithic process was stopped after
  LiveJournal's contextual igraph PPR comparison exceeded six hours, before
  it could serialize a final JSON artifact.
- `bench-20260729T131101Z-orkut-bounded.{json,md}` and
  `bench-20260729-orkut-bounded.log` complete the Orkut core and practical
  baseline matrix while explicitly skipping the unbounded igraph PPR row.
  These files support `docs/benchmark-report-2026-07-29.md`; they do not
  replace the canonical smoke artifact.

The JSON is the machine-readable source of truth; the Markdown file is the
rendered view produced by `bench/run.py`. It includes a strict schema,
source and native-binary provenance, and the pinned SNAP manifest. Hosted CI
timing must never be added here.
