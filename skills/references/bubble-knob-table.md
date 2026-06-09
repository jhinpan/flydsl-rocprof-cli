# Bubble → FlyDSL thread-level knob table

The full catalog behind `flyprof report` / `flyprof.knobs`. Each row: the stall/bubble
class, the diagnostic signal that fires it, the FlyDSL knob, and the concrete code
change. Rules are gated by the roofline `bound` so an irrelevant knob never fires
(e.g. no tile-growth advice on a compute-bound kernel, no MMA-K change on a
memory-bound one). Recommendations are ranked by the firing bubble's stall %.

Signals come from `bubbles.json` (`stall_taxonomy.by_class[*].pct`, `inst_mix`,
`hotspots[*].waitcnt_sources`) and `counters.json` (`memory.l2_hit_rate`,
`memory.partial_32b_frac`, `compute.mfma_inst_frac`, `lds.bank_conflict_pct`) and
`capture.json` (`tags.*.arch_vgpr`, `occupancy_waves_per_cu`).

| # | Bubble class | Signal (field ≷ threshold) | Bound gate | FlyDSL knob | Concrete change | Conf |
|---|---|---|---|---|---|---|
| 1 | **VMEM latency** | `vmcnt.pct > 10` (HBM not at roof) | memory, latency | SW-prefetch / double-buffer (loop-carried SSA) | `for iv,st in range(fx.Index(0),fx.Index(N-1),fx.Index(1),init=[...])`; issue next-iter `buffer_ops.buffer_load` at loop top, consume prior iter in the body. **Bounds MUST be `fx.Index`** or the rewriter unrolls and drops `init=`. | high |
| 2 | **VMEM queue depth** | `vmcnt.pct > 10` & `inst_mix.vmem_load > 0.20` | memory, latency | tune VMEM grouping / vmcnt-only fence | `rocdl.sched_vmem(N)` to group loads; replace a full barrier with `rocdl.s_wait_loadcnt(0)` (gfx950) / inline `s_waitcnt vmcnt(N)`. | med |
| 3 | **LDS/SMEM-wait (lgkmcnt)** | `lgkmcnt.pct > 15` | any | grow LDS write→read distance / prefetch LDS read | insert independent `buffer_load`/SALU between `lds_ptr.store(...)` and the dependent `gpu.barrier()`/load; A0-prefetch `lds_load_pack_k32(...)` right after `gpu.barrier()`. | high |
| 4 | **LDS bank conflict** | `lds_conflict_pct > 5` or `lds.pct > 8` | any | XOR swizzle (preferred) / pad / transpose-load | `swizzle_xor16(row,col,k_blocks16)` identically on store+load; or `+PADDING` in `make_layout` stride; or `rocdl.cdna4.LDSReadTrans8_64b()` (gfx950). | high |
| 5 | **MFMA RAW** | `mfma.pct > 5` & `mfma_frac > 0.30` | compute, latency | fill MFMA pipe / break accumulator RAW | interleave `rocdl.sched_mfma(N)` with `sched_dsrd`/`sched_vmem`; tune `dsrd_preload`/`dvmem_preload`; `rocdl.s_setprio(...)`. | med |
| 6 | **Barrier** | `barrier.pct > 5` | any | relax / remove redundant sync | drop a redundant `gpu.barrier()`; barrier only the swapped LDS region in ping-pong; `rocdl.s_wait_dscnt(0)` (gfx950); replace LDS reduce with `gpu.ShuffleOp(v,off,64,mode='xor')` over `[32,16,8,4,2,1]`. | med |
| 7 | **Low occupancy / VGPR pressure** | `occupancy_waves_per_cu < 8` (or spills) | memory, latency | cut register footprint | `use_async_copy=True` → `rocdl.buffer_load_to_lds(...)` (frees ~32 arch_vgpr of the A-tile); shrink `tile_m`/`tile_k`; `@autotune Config(waves_per_eu=2)`. **Do NOT force `maxnreg` to push accum_vgpr=0** — spills via `v_accvgpr_read`, ~4.5× regression. | high |
| 8 | **Poor vectorization** | `partial_32b_frac > 0.15` | memory, latency | widen copy/load width | `buffer_ops.buffer_load(..., vec_width=4)`; 128-bit copy atom `fx.UniversalCopy(128)` / `rocdl.BufferCopy128b()`; f32 rule `vec_width = 128 // elem_bits`. Verify `vec_width*sizeof(elem) ≤ atom_bits`. | high |
| 9 | **MMA selection** | `bound==compute` & `mfma_frac > 0.30` | compute | widest-K MMA atom | `fx.make_mma_atom(fx.rocdl.MFMA(m,n,k,dtype))` at max K: fp8 `mfma_f32_16x16x32_fp8_fp8` (K32), bf16 K16, fp4 `rocdl.cdna4.MFMA_Scale(16,16,128,...)` (K128). | med |
| 10 | **Layout / L2 thrash** | `l2_hit_rate < 70` & `vmcnt.pct > 10` | memory, latency | layout/basis + tiled-MMA + XCD remap | `fx.make_layout`/`zipped_divide`/`slice` for contiguous per-lane addresses; `fx.make_tiled_mma(atom, make_layout((M_rep,N_rep,K_rep),...))`; `xcd_remap_bx_by(..., xcd_swizzle=N)`. | med |
| — | **Already optimal** | HBM ≥ 85% roof & single load dominates | — | none | report well-tuned; recommend no change. | high |

Source of truth for the FlyDSL APIs: `FlyDSL-lab/python/flydsl/expr/rocdl/*`, `buffer_ops.py`,
`gpu.py`, `autotune.py`, `kernels/pipeline_utils.py`; technique background in ROCmKernelWiki.
Keep this table in sync with `flyprof/knobs.py`.
