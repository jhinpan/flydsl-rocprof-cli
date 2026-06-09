# rocprofv3 ATT/PMC quirks (gfx950, rocprofv3 1.1.0)

Loaded only when an `flyprof capture`/`counters` step hits one of these.

## Dispatch numbering
- `ui_output_agent_<PID>_dispatch_<N>` — `<N>` is **renumbered every run**. Never
  hard-code a dispatch number across runs; always read `kept_dir` from the fresh
  `capture.json`. `flyprof capture` already picks the best non-empty dispatch for you.

## Empty-shell folders
- rocprofv3 often emits several `ui_output_*` dirs per run; only the ones whose grid
  actually landed on the ATT-targeted CU have a populated `code.json` + wave files.
  The others are empty shells. `flyprof capture` deletes the empties and keeps the
  best (most waves × instructions). If **all** are empty → `EMPTY_ATT`: the grid was
  too small to hit `att_target_cu`. Enlarge `--shape`.

## kernel_iteration_range fence-posts
- The YAML `kernel_iteration_range: "[skip, [m-n]]"` is **0-indexed and inclusive**;
  it selects which *dispatches* of the matched kernel get traced, counting from 0
  after `skip`. `flyprof capture` auto-derives a safe range from the kernel's call
  count; override with `--iter-range` only if you know the launch you want.
- It does **not** reduce how many times the workload runs — the Python process still
  executes fully. For PMC, every counter *pass* re-runs the whole workload, so the
  cost is `#passes × workload`. Keep the captured shape single (`ROCDSL_*_SHAPES`).

## ATT vs PMC
- PMC collected *inside* an ATT job is unreliable — `flyprof` runs a **separate** PMC
  pass (`--with-pmc` / `flyprof counters`). The ATT job is ATT-only.
- ATT capture runs the workload **once** (fast, ~seconds). The PMC pass multiplexes
  counters across passes and re-runs the workload each pass (slow, minutes). Budget
  accordingly; this is why `counters` has a generous timeout.

## Source mapping
- Needs `FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1` (flyprof sets it) **and** a cold JIT cache
  (flyprof uses a fresh `FLYDSL_RUNTIME_CACHE_DIR` per capture) so DWARF line tables
  are emitted. A warm cache can yield `code.json` with empty `source` columns.

## Counter caveats (absolute bandwidth)
- `TCC_EA0_*` counts one memory channel; `GRBM_GUI_ACTIVE` is summed across XCDs.
  Deriving an *absolute* HBM GB/s needs per-XCD/channel normalization we don't fully
  reproduce, so `counters.memory.calibrated == false` and the GB/s is a lower bound.
  Trust the **ratios** (L2 hit, 32B fraction, MFMA fraction) and the bubble taxonomy.
