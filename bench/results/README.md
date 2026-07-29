# Curated benchmark results

This directory retains only benchmark artifacts cited by project
documentation. Other local runs remain ignored.

- `bench-20260729T000207Z.{json,md}` is the canonical earlier v0.1 smoke
  run: Apple M4 Max, macOS 26.2, Python 3.13.3, 20 warm runs, clean source
  commit `be8763c8b87d`. It is retained for historical comparison; the
  current README table uses the later full-suite values.
- `bench-20260729-full-partial.log` preserves the completed rows from the
  2026-07-28–29 full-suite rerun. The monolithic process was stopped after
  LiveJournal's contextual igraph PPR comparison exceeded six hours, before
  it could serialize a final JSON artifact. The current README's first five
  dataset rows are provisional reconstructions from this log.
- `bench-20260729T131101Z-orkut-bounded.{json,md}` and
  `bench-20260729-orkut-bounded.log` complete the Orkut core and practical
  baseline matrix while explicitly skipping the unbounded igraph PPR row.
  These files support the current README's Orkut row and
  `docs/benchmark-report-2026-07-29.md`; they do not replace the earlier
  smoke artifact for historical comparisons.
- `bench-20260729T201007Z.{json,md}` is the isolated clean-process
  re-measurement of the HippoRAG KG dataset (28 rows, 20 warm runs, clean
  source commit `7afb4b2ff3dc`, `--dataset kg-hipporag`), produced after
  the full-suite run's KG PPR row showed anomalous per-row accounting. It
  is the artifact behind the README's HippoRAG PPR gate numbers
  (10.707 ms per batch, 0.669 ms/query) and the first artifact measuring
  KG high-degree BFS against the corrected equivalent-output igraph
  adapter.

Where a completed pair exists, JSON is the machine-readable source of truth
and Markdown is the rendered view produced by `bench/run.py`. The
full-suite log is the disclosed exception and should be superseded by a
checkpointed rerun. Completed JSON includes a strict schema, source and
native-binary provenance, and the pinned SNAP manifest. Hosted CI timing
must never be added here.
