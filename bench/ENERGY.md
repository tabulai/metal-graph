# Energy measurement methodology (bench/run.py --energy)

Energy numbers are published only from physical hardware runs following
this protocol (plan section 10). They are context for the "worth the
dependency" argument, never a ship gate.

## Tooling

`powermetrics` at 100 ms sampling, CPU + GPU power samplers. Refresh the
sudo credential first, then run the Python harness as the normal user:

```bash
sudo -v
PYTHONPATH=python python3 bench/run.py --suite smoke --energy
```

Only the fixed `/usr/bin/powermetrics` command is elevated; the benchmark,
Python environment, repository code, and output-file creation remain under
the normal user. Cleanup signals the original `sudo` process, which relays
the signal to `powermetrics`, so it does not depend on the sudo timestamp
still being valid after a long run. The raw sample stream is written to
`bench/results/powermetrics-<stamp>.txt` and its path recorded in the result
JSON.

## What run.py records

With `--energy`, `run.py`:

1. starts `powermetrics` (100 ms interval, `cpu_power,gpu_power`), then
2. **sleeps for `--idle-seconds` (default 60, `0` disables) before any
   workload**, so the idle baseline is captured inside the same
   powermetrics stream, on the same machine state, and
3. records in the result JSON:
   - `meta.energy_idle_window`: `{t_start_utc, t_end_utc, seconds}` for
     the idle sleep (ISO-8601 UTC),
   - `meta.powermetrics_file`: path of the raw sample stream,
   - on **every** line-item row (with or without `--energy`):
     `t_start_utc` / `t_end_utc` ISO-8601 UTC timestamps bracketing that
     item's work — including its warmup and all timed runs — so each row
     maps to a window of powermetrics samples.

To compute a published figure:

1. Parse the powermetrics stream; take the samples whose timestamps fall
   inside `meta.energy_idle_window` → mean idle package power `P_idle`.
2. For a line item, take the samples inside that row's
   `[t_start_utc, t_end_utc)` → mean package power `P_item`.
3. Report `P_item − P_idle` (W above idle) and
   `(P_item − P_idle) × median wall time` (J/op).

Note: `powermetrics` timestamps are local time; convert to UTC before
matching against the row windows.

## Protocol

1. **Idle baseline first.** `run.py --energy` captures it for you:
   >= 60 s of idle samples (`--idle-seconds`, do not touch the machine
   during the sleep) on the same machine state (same power source,
   screen, background load) immediately before the workload. The mean
   idle package power is subtracted from every reported figure, as
   described above. Numbers without an idle baseline are not
   publishable; do not pass `--idle-seconds 0` for publishable runs.
2. **Plugged in, no Low Power Mode**, lid open, display on. Record the
   power source in the run notes; battery runs are a separate table.
3. **Warm graphs only.** Energy windows cover the warm kernel loop of a
   line item (>= 20 runs), aligned via the row's recorded
   `t_start_utc`/`t_end_utc` against the powermetrics stream.
   Build/compile energy is reported as its own window (the
   `build/from_edges` and `pipeline/first_op_vs_second` rows carry their
   own timestamps), never folded into per-iteration figures.
4. **Reported figures** per line item:
   - mean package power above idle (W) during the window,
   - energy per operation (J) = power x median wall time,
   - CPU-path vs GPU-path energy for the same operation where both paths
     run the same semantics (the efficiency argument for the planner).
5. **Sampling error.** 100 ms sampling across a >= 10 s window gives
   >= 100 samples; report the sample count. Windows shorter than 2 s are
   not publishable — lengthen by raising `--runs`.
6. Chip model, macOS version, and `powermetrics` schema version are
   recorded; Apple changes the samplers between macOS releases, so raw
   streams are archived next to the JSON.

## Caveats

- `powermetrics` package power on Apple Silicon includes DRAM in the
  package domain; unified-memory traffic shows up here — that is the
  point, do not subtract it.
- Do not compare absolute watts across chip generations; compare J/op on
  the same machine, CPU path vs GPU path, at identical semantics.
- Thermal state matters: report ambient conditions for runs longer than
  5 minutes, and discard windows where `powermetrics` reports thermal
  pressure.
