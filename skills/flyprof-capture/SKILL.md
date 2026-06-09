---
name: flyprof-capture
description: >
  Capture a rocprofv3 ATT (Advanced Thread Trace) for a FlyDSL kernel with flyprof,
  including kernel discovery, tile sizing, and handling empty/out-of-range dispatches.
  Use when you need to record a kernel's per-instruction trace before analysis, or
  when `flyprof capture` returned EMPTY_ATT / DISPATCH_OUT_OF_RANGE / NO_KERNEL_DISCOVERED.
allowed-tools: Bash(flyprof:*), Read
---

# flyprof-capture — record an ATT trace for one kernel

The recon→size→capture loop. Each step's output is one JSON envelope; check it
before the next step.

## Runbook

- [ ] `flyprof doctor -f json` → confirm `data.verdict == "ready"`. If `blockers`
      is non-empty, STOP and see `flyprof-triage` (don't try to capture).
- [ ] `flyprof list -f json` → confirm the kernel exists (or `recipe_source:
      "synthesizable"`, which is fine — a recipe is derived on the fly).
- [ ] `flyprof tile <k> --shape M,N[,K] --dtype D -f json` → size *before* capturing.
      Note `recommended.regime` (memory/compute-bound) and `grid.tail_verdict`
      (is the shape big enough to fill the GPU? a tiny shape gives an empty ATT).
- [ ] `flyprof capture <k> --tag both --with-pmc --gpu 0 -f json`. Confirm:
      `data.tags.big.kept_dir` is set, `data.tags.big.mapped_pct > 90`,
      `data.match_confidence` is `exact`/`fuzzy` (if `heuristic`, double-check
      `data.candidates` is the kernel you meant).

## Downgrade table (error.code → action)

| code | meaning | action |
|---|---|---|
| `NO_KERNEL_DISCOVERED` | `--stats` found no FlyDSL kernel | check `error.candidates`; the test may not launch a kernel, or all were noise-filtered. Try a non-trivial `--shape`. |
| `EMPTY_ATT` | dispatches were empty shells | the grid was too small to hit `att_target_cu`. Enlarge `--shape`, or set `--iter-range` to a launch that actually runs. |
| `DISPATCH_OUT_OF_RANGE` | iteration range missed | use a value from `error.available`. |
| `CAPTURE_TIMEOUT` (75) | rocprofv3 ran too long | retry **once** with a larger `--timeout` or smaller `--shape`. Budget one retry. |
| `BUILD_TREE_MISSING` (78) | no built FlyDSL | STOP; fix `--worktree`. |

## Notes that bite

- **Never reuse a `dispatch_<N>` number across runs** — rocprofv3 renumbers dispatches
  every run. Always read `kept_dir` from the fresh `capture.json`; never hard-code it.
- A `KERNEL_COMPILE_FAIL` is a **finding**, not a loop trigger — report it and stop
  (don't retry with tweaks hoping it compiles).
- `--with-pmc` runs a *separate* PMC pass (PMC inside an ATT job is unreliable); it
  re-runs the workload several times and is the slow part — budget minutes, not seconds.

See `references/rocprofv3-quirks.md` for dispatch numbering / iteration-range fence-posts.
