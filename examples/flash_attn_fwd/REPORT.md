# flash_attn_dualwave_swp_gfx950_kernel_0 — gfx950/MI350X Profiling Report
_Provenance: FlyDSL · rocprofv3 · MI350X SPX · 2026-06-09 · avg 49392.319149 ns/call_

## 1. Headline
flash_attn_dualwave_swp_gfx950_kernel_0: memory-bound, barrier-dominated (20.84% of 60.6% total stall) -> Raise occupancy by cutting register footprint

## 2. Workload & Tile
- recommended tile: BM=128 BN=128 BK=64 buffers=2; regime=memory-bound, limiter=LDS, occupancy≈25.0%

## 3. Roofline & Occupancy
- bound: **memory**; dtype bf16, peak MFMA 2306867.2 GFLOP/s, ridge AI 288.36 FLOP/B
- occupancy ≈ 4 waves/CU (arch_vgpr 249, accum_vgpr 0)

## 4. Bubble / Stall Taxonomy
- total stall: 60.6% of 781596 latency cycles

| class | stall % | rank | bubble? |
|---|---|---|---|
| barrier | 20.84 | 1 | yes |
| valu | 17.28 | 2 |  |
| lds | 15.07 | 3 |  |
| vmem_store | 14.92 | 4 |  |
| vmcnt | 11.34 | 5 | yes |
| vmem_load | 10.65 | 6 |  |
| mfma | 4.68 | 7 |  |
| salu | 3.47 | 8 |  |
| lgkmcnt | 1.72 | 9 | yes |
| branch | 0.03 | 10 |  |

## 5. ISA ↔ Python Evidence
- `buffer_store_dwordx2 v[32:33], v46, s[0:3], 0 offen` @0x5828 (code_line 2630, stall 20376) → /sgl-workspace/FlyDSL-lab/kernels/flash_attn_gfx950.py:300
- `s_waitcnt vmcnt(2)` @0x1abc (code_line 120, stall 20168) → /sgl-workspace/FlyDSL-lab/kernels/flash_attn_gfx950.py:192
    waits on `buffer_load_dwordx4 v[6:9], v1, s[4:7], 0 offen` → /sgl-workspace/FlyDSL-lab/kernels/flash_attn_gfx950.py:192
    waits on `buffer_load_dwordx4 v[2:5], v1, s[4:7], 0 offen offset:32` → /sgl-workspace/FlyDSL-lab/kernels/flash_attn_gfx950.py:192
- `s_barrier` @0x2be0 (code_line 849, stall 15704) → /sgl-workspace/FlyDSL-lab/kernels/flash_attn_gfx950.py:192
- `buffer_load_dwordx4 v[6:9], v1, s[4:7], 0 offen` @0x1984 (code_line 75, stall 14944) → /sgl-workspace/FlyDSL-lab/kernels/flash_attn_gfx950.py:192
- `s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)` @0x1960 (code_line 69, stall 10860) → /sgl-workspace/FlyDSL-lab/kernels/flash_attn_gfx950.py:192
    waits on `buffer_load_dwordx4 v211, s[12:15], 0 offen lds` → /sgl-workspace/FlyDSL-lab/kernels/flash_attn_gfx950.py:295
    waits on `buffer_load_dwordx4 v213, s[12:15], 0 offen lds` → /sgl-workspace/FlyDSL-lab/kernels/flash_attn_gfx950.py:295

## 6. Ranked Optimization Opportunities
### #1 — occupancy (evidence: occupancy_waves_per_cu=4 @ None)
- **Knob:** Raise occupancy by cutting register footprint
- **API:** use_async_copy=True -> rocdl.buffer_load_to_lds(...) (bypasses ~32 arch_vgpr of the A-tile); shrink tile_m/tile_k; @autotune Config(waves_per_eu=2). Do NOT force maxnreg to push accum_vgpr=0 (spills via v_accvgpr_read, ~4.5x regression).
- **Change:** move the A/B tile load through LDS via async copy to free the staging VGPRs, or reduce the tile
- **Expected:** more resident waves to hide the exposed memory/LDS latency — often the root cause when stalls are exposed at low occupancy  ·  confidence: high

### #2 — barrier (evidence: barrier.pct=20.84 @ /sgl-workspace/FlyDSL-lab/kernels/flash_attn_gfx950.py:192)
- **Knob:** Relax / remove redundant synchronization
- **API:** drop a redundant gpu.barrier(); in ping-pong, barrier only the swapped LDS region; use rocdl.s_wait_dscnt(0) (gfx950) for intra-wave order; replace an LDS cross-wave reduce with gpu.ShuffleOp(v, off, 64, mode='xor') over [32,16,8,4,2,1]
- **Change:** remove barriers that don't guard a real cross-wave LDS dependency; prefer DPP/shuffle reduce
- **Expected:** cuts the s_barrier stall where waves wait on each other unnecessarily  ·  confidence: medium

### #3 — vmcnt (evidence: vmcnt.pct=11.34 @ /sgl-workspace/FlyDSL-lab/kernels/flash_attn_gfx950.py:192)
- **Knob:** Software-prefetch / double-buffer the global loads (loop-carried SSA)
- **API:** for-loop with fx.Index bounds + init=[...]; issue next-iter buffer_ops.buffer_load at loop top, consume the prior iter's data in the MMA/compute body
- **Change:** convert `for i in range(N)` -> `for iv, st in range(fx.Index(0), fx.Index(N-1), fx.Index(1), init=[...])`; prefetch load[i+1] before using load[i]. Bounds MUST be fx.Index or the rewriter unrolls and drops init=.
- **Expected:** overlaps VMEM latency with compute; drains the s_waitcnt vmcnt stall  ·  confidence: high
- waits on: /sgl-workspace/FlyDSL-lab/kernels/flash_attn_gfx950.py:192, /sgl-workspace/FlyDSL-lab/kernels/flash_attn_gfx950.py:192, /sgl-workspace/FlyDSL-lab/kernels/flash_attn_gfx950.py:192

## 7. Verification Plan
- `FLYDSL_DUMP_IR=1` → inspect final_isa.s; `grep -c v_mfma|s_barrier|buffer_load|ds_read`
- re-run `flyprof run <kernel>` and diff `report.json` stall % + roofline before/after
- target: the rank-1 bubble's stall % drops; bound moves toward the roof

## 8. Counters (raw)
- see report.json.roofline / counters.json

## 9. Dispatch / Occupancy
- arch_vgpr 249, accum_vgpr 0, occ≈4 waves/CU

## 10. Repro
- `flyprof run flash_attn_fwd --gpu 0`