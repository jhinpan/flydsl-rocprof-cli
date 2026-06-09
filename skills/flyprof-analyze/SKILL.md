---
name: flyprof-analyze
description: >
  Analyze a captured FlyDSL kernel bundle with flyprof: read the roofline/occupancy
  (counters), tile advice, stall/bubble taxonomy (bubbles), and the ISA<->Python
  mapping (map). Use after `flyprof capture`, when you need to understand where a
  kernel spends its stall cycles and how the ISA lines up with the .py source.
allowed-tools: Bash(flyprof:*), Read
---

# flyprof-analyze — read the diagnosis off a captured bundle

All of these are pure data transforms over the bundle (no GPU). Run them against
the `--bundle DIR` produced by `flyprof capture`.

## Order & what each tells you

- [ ] `flyprof counters --bundle DIR -f json` (if a PMC pass was captured) →
      `data.roofline.bound` (memory/compute/latency), `data.memory.l2_hit_rate`,
      `data.compute.mfma_inst_frac`, `data.lds.bank_conflict_pct`.
- [ ] `flyprof tile <k> --shape … --dtype … -f json` → occupancy/LDS/regime if you
      want the pre-capture estimate (and `min_bk_to_cross_ridge`).
- [ ] `flyprof bubbles --bundle DIR --tag big -f json` → `data.stall_taxonomy.by_class`
      (ranked), `data.hotspots` (top stalls + `waitcnt_sources`), `data.inst_mix`.
- [ ] `flyprof map --bundle DIR -f json` → `data.mappings` (ISA → `.py:line` + snippet),
      `data.src_mapped_pct`, `data.source_flamegraph`.

## Reading the numbers honestly

- **`counters.memory.calibrated == false`**: the absolute HBM GB/s is an EA0-channel
  *lower bound*. Trust the **ratios** (L2 hit, 32B-partial fraction, MFMA fraction)
  and the **bubble taxonomy** for the bound decision — not the absolute bandwidth.
- **The bubble taxonomy is the reliable evidence.** A high `vmcnt` % *is* a memory
  wait; a high `lgkmcnt` % *is* an LDS/SMEM wait; `is_bubble:true` marks the classes
  that mean the wave is idle-waiting, not doing work.
- **`waitcnt_sources`** on a hotspot is the bubble attribution: the `s_waitcnt` is
  waiting on those source lines' memory ops. That edge is what a fix must break.
- **Cost guide:** `bubbles` summary is cheap — call once. For one hotspot's full
  per-instruction record use `flyprof bubbles --detail <code_line>`; for one source
  line use `flyprof map --code-line N` / `--source-line FILE:LINE`. Never dump raw ATT.

## Confidence

Each mapping carries `confidence` (`exact` when the snapshot source matched,
`heuristic` when only a suffix matched). Before publishing a perf claim about a
specific `.py` line, cross-check the hotspot's `waitcnt_sources` actually point at
the op you think is the cause.
