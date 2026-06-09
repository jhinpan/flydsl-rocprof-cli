# Examples

Real `flyprof` output bundles on FlyDSL kernels — the artifact an agent/user gets back.
Each `<kernel>/` holds the canonical layout: `REPORT.md` (+ `report.json`,
`headlines.json`), the full `att_viewer/<tag>/ui_output_*` trace (loadable in
rocprof-compute-viewer, or by `flyprof bubbles/map --bundle <dir>`), `compute_viewer/`
(counters + rocprofv3 support files), and `source/` (the exact profiled `.py`).

## `flash_attn_fwd/` — the hard one

FlyDSL's dual-wave software-pipelined flash attention on gfx950/MI350X. Profiled with:

```bash
flyprof run flash_attn_fwd --gpu 0 \
  --invocation "python tests/kernels/test_flash_attn_fwd.py \
                --batch 1 --seq_len 2048 --num_heads 16 --head_dim 128 --causal" \
  --examples-dir examples
```

(`--invocation` pins a single shape via the test's own args — flash-attn takes its
shape from CLI flags, not `ROCDSL_*_SHAPES`. The registry recipe was stale, so the
kernel was discovered live and the recipe synthesized.)

**Discovered kernel:** `flash_attn_dualwave_swp_gfx950_kernel` — 2670 ISA instructions,
100% source-mapped, **arch_vgpr 249 → only 4 waves/CU** (12.5% of peak).

**Verdict (`REPORT.md`):** memory-bound, barrier-dominated (20% of 64.9% total stall).
Ranked recommendations:

| # | bubble | knob | evidence |
|---|---|---|---|
| 1 | occupancy | **raise occupancy by cutting register footprint** (async-copy / shrink tile; *not* `maxnreg`) | 4 waves/CU @ arch_vgpr 249 |
| 2 | vmcnt | software-prefetch / double-buffer the global loads | `flash_attn_gfx950.py:192` |
| 3 | barrier | relax / remove redundant dual-wave sync | `flash_attn_gfx950.py:192` |

The root cause is the classic attention pathology: the kernel is register-pressure
capped to 4 waves/CU, so it can't hide the memory and barrier latency that then show up
as the dominant stalls. Every recommendation is traceable back to the ISA + the `.py` line.

Open it: `flyprof bubbles --bundle examples/flash_attn_fwd -f json` ·
`flyprof map --bundle examples/flash_attn_fwd --source-line flash_attn_gfx950.py:192`.
