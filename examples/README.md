# Examples

Real `flyprof` output bundles on FlyDSL kernels — the artifact an agent/user gets back.
Each `<kernel>/` holds the canonical layout: `REPORT.md` (+ `report.json`,
`headlines.json`), the full `att_viewer/<tag>/ui_output_*` trace (loadable in
rocprof-compute-viewer, or by `flyprof bubbles/map --bundle <dir>`), `compute_viewer/`
(counters + rocprofv3 support files), and `source/` (the exact profiled `.py`).

## `flash_attn_fwd/` — the hard one

FlyDSL's dual-wave software-pipelined flash attention on gfx950/MI350X. Profiled with:

```bash
flyprof run flash_attn_fwd --gpu 0 --examples-dir examples
```

Recipe-driven: the recipe pins a representative `(1, 2048, 16, 128)` causal MHA via the
test's own CLI args (flash-attn takes its shape from `--batch/--seq_len/...`, not
`ROCDSL_*_SHAPES`), and discovery is an **exact match** on the gfx950 kernel. (The hub's
recipe was stale — pointing at a renamed test file — and was repaired in
[flydsl-kernel-profiling#8](https://github.com/jhinpan/flydsl-kernel-profiling/pull/8);
before that, this was captured via a `--invocation` override.)

**Discovered kernel:** `flash_attn_dualwave_swp_gfx950_kernel` — 2670 ISA instructions,
100% source-mapped, **arch_vgpr 249 → only 4 waves/CU** (12.5% of peak).

**Verdict (`REPORT.md`):** memory-bound, barrier-dominated (~20% of ~61% total stall).
Ranked recommendations:

| # | bubble | knob | evidence |
|---|---|---|---|
| 1 | occupancy | **raise occupancy by cutting register footprint** (async-copy / shrink tile; *not* `maxnreg`) | 4 waves/CU @ arch_vgpr 249 |
| 2–3 | barrier / vmcnt (near-tied) | relax redundant dual-wave sync · software-prefetch the global loads | `flash_attn_gfx950.py:192` |

The root cause is the classic attention pathology: the kernel is register-pressure
capped to 4 waves/CU, so it can't hide the memory and barrier latency that then show up
as the dominant stalls. Every recommendation is traceable back to the ISA + the `.py` line.

Open it: `flyprof bubbles --bundle examples/flash_attn_fwd -f json` ·
`flyprof map --bundle examples/flash_attn_fwd --source-line flash_attn_gfx950.py:192`.
