# PLAN — Autonomous "Humanize Loop" for FlyDSL Flash-Attn on gfx950/MI350X

**Goal:** ~1.5× speedup on our already-benchmarked flash-attn configs, via a self-improving optimizer that fuses **flyprof** (diagnose), **ROCmKernelWiki** (retrieve technique), and the **humanize/KDA** agentic loop (propose→apply→verify→learn). This is a plan for approval — nothing here is executed.

---

## 1. THESIS

We already have the three legs of a closed loop and they currently live apart: `flyprof report` tells us *what's wrong* (memory-bound, barrier 20.84%, vmcnt 11.34%, occupancy 4 waves/CU at arch_vgpr=249), the ROCmKernelWiki tells us *what fix humans have shipped for exactly that symptom* (the FlyDSL dual-wave PR-629 lineage: direct-to-LDS, ds_read_tr, permlane16, lazy-rescale), and the humanize/KDA methodology tells us *how to iterate without fooling ourselves* (one hypothesis at a time, correctness gate before any speed claim, promote only on evidence, log every attempt). Wiring them into one loop turns each into the next's input — a diagnosed bubble becomes a wiki query becomes a concrete `flash_attn_gfx950.py` edit becomes a re-profile that either shrank the targeted bubble or didn't — and the attempt log makes the loop *compound* by never re-trying a dead end. **Occupancy is the lead hypothesis** because every other bubble is downstream of it: at 4 waves/CU (1 wave/SIMD, capped by 249 live ArchVGPRs) there is no second wave to hide the barrier and VMEM-load latency behind, so the 20.84% barrier and 11.34% vmcnt stalls are *symptoms of the occupancy floor*, not independent problems — which is exactly why flyprof's rank-1 recommendation is `occupancy_vgpr` even though `barrier` is the rank-1 stall *class*.

---

## 2. THE HUMANIZE LOOP (the core)

One optimization round = one hypothesis. The loop is **RLCR** (Propose→Measure→Critique→Revise→Revalidate) grounded in our tools. Steps, with explicit gates:

**Step 0 — Baseline lock (once, outside the loop).** Freeze the config set (§3) and the correctness oracle. Capture the baseline bundle. This is the KDA "fair benchmark stays outside the loop" rule — only the bundle path and the shape ledger are immutable; everything else iterates.

**Step 1 — DIAGNOSE (flyprof).**
```
flyprof report --bundle <current_bundle> -f json
```
Read from `report.json`: `bound_type`, `totals.stall_pct`, `stall_taxonomy.by_class` (cycles + pct + rank + is_bubble), `occupancy.{from_capture_waves_per_cu, arch_vgpr, accum_vgpr}`, `roofline`, and `recommendations[]` (already ranked: stall% × priority, with `occupancy_vgpr` weighted by the occupancy deficit `(8 − waves/CU)/8 × 50`). The rank-1 recommendation is the round's target. Record its `evidence.source` (the `flash_attn_gfx950.py:line` the claim traces to).

**Step 2 — HYPOTHESIZE (ROCmKernelWiki retrieval).** Map the rank-1 bubble_class → wiki query:
```
python3 /sgl-workspace/ROCmKernelWiki/scripts/query.py \
  "<bubble_class> <symptom> flash attention gfx950" \
  --architecture gfx950 --limit 5
python3 /sgl-workspace/ROCmKernelWiki/scripts/get_page.py kernel-flydsl-flash-attention --follow-sources
```
Retrieve the technique page + its `implemented_by` PR list (the prior-art "humanize" check: has a human already done this?). Output a one-line hypothesis: *"bubble X is caused by Y; wiki technique Z (PR-NNN) addresses it; expected flyprof delta = ..."*.

**Step 3 — PROPOSE a concrete edit.** Translate the wiki technique into a single, named `flash_attn_gfx950.py` change (the levers are real — §4 lists the first three). One knob per round. Write the proposed diff and the predicted flyprof delta into the attempt log *before* applying (KDA: state expected value before implementation, so a miss flags a design-understanding gap).

**Step 4 — APPLY + COMPILE.**
```
FLYDSL_DUMP_IR=1 FLYDSL_RUNTIME_ENABLE_CACHE=0 python -c "import kernel; build_..."
hipcc-style resource check via -Rpass-analysis=kernel-resource-usage  (or the JIT final_isa.s)
```
**COMPILE GATE:** kernel must build, and `.private_segment_fixed_size == 0` (scratch == 0). *Any scratch spill = automatic REJECT* (the maxnreg→accum-spill trap; an HBM round-trip is 50–200× a register and will silently erase the win).

**Step 5 — CORRECTNESS GATE (hard, never relaxed).**
```
python -m pytest tests/kernels/test_flash_attn_fwd.py -v --tb=short   # ALL configs
```
Pass iff for **every** config: `max_err < 1e-2` AND `min_cosine > 0.99` AND no NaN/Inf, vs the PyTorch fp32 SDPA oracle. **First failure → REVERT, do not benchmark.** This is the inviolable line: we never trade correctness for a number.

**Step 6 — RE-PROFILE (flyprof, the targeted-delta check).** Fresh cache, re-capture, re-report on the *primary* config (B=1, S=2048, H=32, D=128, bf16, causal). Did the **targeted bubble** shrink? Did `arch_vgpr` drop? Did `waves_per_cu` rise? This is the mechanistic check — distinct from the speed check — and catches "got faster by accident / for the wrong reason," which doesn't generalize.

**Step 7 — BENCHMARK (TFLOPS vs baseline, full config set).**
```
python tests/kernels/test_flash_attn_fwd.py --batch B --seq_len S ... --warmup 10 --iters 100
```
Kernel-only median µs → TFLOPS, on **all** configs in §3 (not just the profiled one).

**Step 8 — ACCEPT / REJECT / REVISE.**
- **ACCEPT** iff: correctness passes AND scratch==0 AND **geometric-mean TFLOPS across all configs ≥ baseline** (strictly: no config regresses by >2%, and gmean improves by ≥1% — the humanize ">1% or revert" gate). Occupancy/bubble movement is *supporting evidence, not the acceptance criterion* (§6).
- **REVISE** iff: correctness passes but the bubble moved less than predicted, or one shape regressed — adjust the knob magnitude (e.g., BLOCK_M 256→192 instead of →128) and re-enter at Step 4.
- **REJECT** iff: correctness fails, scratch>0, or gmean TFLOPS regresses. Revert the file (`git checkout`).

**Step 9 — RECORD (the humanize PR-history analogue).** Append to the attempt log (`attempts.jsonl`, §5) regardless of outcome: hypothesis, wiki technique + PR cited, the diff, predicted vs actual flyprof delta, per-config TFLOPS before/after, verdict, reason. **A rejection is a recorded finding, not a deleted dead end** — the next round queries this log first so the loop never re-proposes a known failure. New accepted state becomes the baseline bundle for Step 1 of the next round.

---

## 3. TARGET & BASELINE TABLE

**Primary profiled config (measured, from `report.json`):** B=1, S=2048, H=32, D=128, bf16, causal. Memory-bound, 60.6% total stall, 4 waves/CU, arch_vgpr=249.

| # | B | S | H | KV-H | D | dtype | causal | FlyDSL now (µs / TFLOPS) | Baseline AITER-CK | 1.5× goal (TFLOPS) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 (primary) | 1 | 2048 | 32 | 32 | 128 | bf16 | yes | 92.4 / 371.7 (measured) | ~404 (0.92× ratio, FINDINGS.md) | **557** |
| 2 | 1 | 4096 | 32 | 32 | 128 | bf16 | yes | TBD at lock | AITER-CK | 1.5× of own base |
| 3 | 4 | 2048 | 32 | 8 (GQA) | 128 | bf16 | yes | TBD | AITER-CK | 1.5× |
| 4 | 8 | 1024 | 32 | 32 | 128 | bf16 | yes | TBD | AITER-CK | 1.5× |
| 5 | 1 | 8192 | 32 | 32 | 128 | bf16 | yes | TBD | AITER-CK | 1.5× |
| 6 | 1 | 2048 | 32 | 32 | 128 | f16 | yes | TBD | AITER-CK | 1.5× |

*(Configs 2-6 µs/TFLOPS filled at Step-0 baseline lock; all are already in `test_flash_attn_fwd.py:54-82` and the `flydsl-kernel-profiling/benchmarks/providers/flash_attn.py` provider.)*

**Avoiding single-shape overfit (KDA multi-shape discipline):**
- **Headline metric = geometric mean across all 6 configs**, equal-weight. The 1.5× claim is the gmean, not the best shape.
- Per-config we report median/mean/std/min so a single lucky run can't carry the verdict.
- **Acceptance requires no individual config regresses >2%** — this is what stops a knob that helps S=2048 but wrecks S=8192 (a real risk for BLOCK_M / LDS-buffer changes).
- The dual-wave fast path only fires on `seq_len % 256 == 0 && seq_len ≥ 384 && D==128 && dtype∈{bf16,f16}`; all 6 configs satisfy this, so we're tuning the path that's actually hit. If a knob needs different settings per seqlen bucket, we add a **dispatch table** (KDA pattern) rather than picking one compromise.

---

## 4. FIRST 2-3 ITERATIONS (concrete hypotheses)

### Iteration 1 — Raise occupancy by cutting ArchVGPR (the lead hypothesis)
- **Hypothesis:** 249 live ArchVGPRs → `floor(256/round_up(249,8)=256)=1` wave/SIMD = 4 waves/CU. Crossing below the 248 threshold unlocks 2 waves/SIMD = 8 waves/CU, giving a second wave to hide the barrier (20.84%) and vmcnt (11.34%) stalls behind. Wiki Technique 1.1 (direct-to-LDS) + 1.2 (tile shrink); `implemented_by` PR-629.
- **The edit (one of, escalating):**
  - **(a) First try — verify the existing direct-to-LDS path isn't leaving staging VGPRs live.** The kernel already uses `_buffer_load_lds_128` (line 285, the `buffer_load ... lds` path) — confirm K/V never round-trip through ArchVGPRs in the prologue (lines 942-971). If staging VGPRs persist, eliminate them. *Expected: frees 20-50 ArchVGPRs, 249→~220, no algorithm change.*
  - **(b) If (a) insufficient — `waves_per_eu=2` is already the default (line 82); try the launch-bound contract harder** by ensuring the compiler honors it without spill, or
  - **(c) BLOCK_M 256→192** (line 105). Halves the O-accumulator and operand-fragment footprint per wave. *Risk: halves query-tile K-loop reuse — measure throughput; this is the REVISE candidate, not first choice.*
- **Predicted flyprof delta:** `occupancy.from_capture_waves_per_cu` 4→6+; `arch_vgpr` 249→<248; barrier pct 20.84→~12-15%; vmcnt 11.34→~7-9%. Bound stays memory but stall_pct drops from 60.6%.
- **Risk / guardrail:** the **maxnreg→accum-spill trap** — forcing occupancy can push the compiler to spill the O accumulator to scratch. COMPILE GATE (`scratch==0`) catches it. Lever (c) can regress long seqlens (config 5) — the no-config-regresses-2% rule guards it.

### Iteration 2 — Reduce the dual-wave barrier stall (rank-1 stall class)
- **Hypothesis:** barrier is 20.84% (98,732 cycles, `is_bubble: true`). flyprof rule `barrier_relax` (priority 0.8): some `gpu.barrier()` calls (lines ~929/942/946 prologue, loop body) guard LDS regions that don't have a true cross-wave dependency, or could be a vmcnt-only fence. Wiki Technique 2.1 (phase-shifted dual-wave barrier) + 5.1 (replace `ds_bpermute` softmax reduce with `v_permlane16_swap`, removing an LGKMCNT dependency).
- **The edit:** (i) audit each `rocdl.s_waitcnt(_LGKMCNT_0_ONLY)` + `gpu.barrier()` pair in the mainloop (lines 997-1104); replace barriers that only guard the *swapped* LDS buffer with a region-scoped fence; (ii) if softmax row-reduce uses LDS/bpermute, switch to permlane16 (gfx950 ALU-only). The `dualwave_swp_enable_stagger` flag (line 87) already phase-shifts the two waves — confirm it's doing real work post-Iteration-1 (only meaningful once we have 2 waves/SIMD).
- **Predicted flyprof delta:** barrier pct 20.84→~14%; lgkmcnt/lds may drop slightly; TFLOPS +3-5%.
- **Risk / guardrail:** removing a barrier that *does* guard a real dependency → wrong results. CORRECTNESS GATE is the catch (softmax reduce precision is tight; bf16 max_err<1e-2 will flag it). Revert on any fail.

### Iteration 3 — vmcnt prefetch depth (rank-3 recommendation, high confidence)
- **Hypothesis:** vmcnt 11.34% (53,708 cycles, `is_bubble: true`); flyprof rule `vmem_prefetch` (priority 1.2, confidence high). HBM→LDS load (~384-400 cyc) isn't fully hidden behind the 2-tile lookahead. Wiki Technique 3.1 (deeper lookahead + relaxed `s_waitcnt vmcnt(N)`).
- **The edit:** `NUM_PREFETCH_K = 2 → 3` (line 147) — adds a third K/V LDS buffer for 3-tile lookahead; and/or relax the prologue `rocdl.s_waitcnt(0)` (line 929) / loop `_LGKMCNT_0_ONLY` fences to `vmcnt(N>0)` so the k+2 load stays in flight during k's MFMA. gfx950 LDS is 160 kB so a third buffer fits (current LDS_KV_TOTAL_SIZE=68,096 B → ~102 kB at 3 buffers).
- **Predicted flyprof delta:** vmcnt pct 11.34→~6-8%; TFLOPS +4-6%. **Interaction risk:** more buffers = more live state = +ArchVGPRs, which can *undo Iteration 1's occupancy win*. This is why order matters and why each round re-profiles occupancy.
- **Risk / guardrail:** LDS over-allocation (compile fail — caught at COMPILE GATE) and the occupancy regression above. If `arch_vgpr` climbs back ≥248, REVISE down to NUM_PREFETCH_K=2 with relaxed fences only.

**Compounding note:** Iterations 2 and 3 only pay off *after* Iteration 1 creates a second wave — a single resident wave has nothing to overlap the hidden latency with. The loop discovers this ordering automatically because flyprof re-ranks after each accepted round.

---

## 5. HARNESS TO BUILD

Reuse what exists (`flyprof` has `doctor/capture/report/run`; the wiki has `query.py/get_page.py/grep_wiki.py`; tests + benchmark provider exist). **Net-new** is glue, not a profiler:

1. **`flyprof diff` subcommand** — `flyprof/diff.py`, registered in `flyprof/cli.py` (alongside the existing `report`/`run` entries at lines 109/114). Takes `--before <bundle> --after <bundle>`, emits the before/after table: `bound_type`, `totals.stall_pct`, per-class `stall_taxonomy` deltas, `occupancy.{waves_per_cu, arch_vgpr}` delta, and the rank-1-bubble-shrank verdict. This is the Step-6/Step-8 mechanical check, made first-class. *(Reuses the `report.json` schema we just confirmed.)*

2. **`flyprof optimize` driver** — `flyprof/optimize.py` (+ cli entry), the loop orchestrator that runs §2 Steps 1-9 for one round and is callable in a sweep. It does NOT contain optimization logic; it calls `report` → wiki `query.py` → presents the proposed edit → invokes the correctness/compile/bench gates → calls `diff` → writes the attempt log. The actual *edit* is the agent's job (humanize: the agent proposes, the harness measures and gates).

3. **ROCmKernelWiki retrieval step** — a thin `flyprof/wiki.py` that maps `bubble_class → wiki query string` and shells `query.py`/`get_page.py --follow-sources`, returning `{technique_id, implemented_by_prs, code_lever}`. Seed the bubble→technique map from `skills/references/bubble-knob-table.md` (already exists) cross-referenced with the wiki technique IDs.

4. **Attempt log (the humanize PR-history analogue)** — `runs/flash_attn_humanize/attempts.jsonl`, one line per round: `{round, hypothesis, wiki_technique, cited_prs, diff, predicted_delta, flyprof_before/after, tflops_per_config_before/after, verdict, reason}`. Plus `candidates.jsonl` with parent-links (KDA DAG) so we can ablate which change drove which gain. Queried at Step-2 to skip known dead ends.

5. **KDA-style per-kernel task spec** — `runs/flash_attn_humanize/` containing:
   - `prompt.md` — operation, the 6 shapes, baseline (AITER-CK), target (1.5× gmean), constraints (gfx950, dual-wave path), completion bar.
   - `interface.md` — the `build_flash_attn_dualwave_swp_module(...)` signature + tunable knobs (waves_per_eu, dualwave_swp_*, BLOCK_M, NUM_PREFETCH_K, MFMA atom), the PyTorch-SDPA tolerance methodology, source lineage (PRs #225/#334/#346/#462/#629/#661).
   - `dispatch.md` — when/if to route seqlen buckets to different knob configs.
   - `config.toml` + `workloads.json` — frozen shape ledger (locked at Step 0).

**Files touched/created:** `flyprof/diff.py`, `flyprof/optimize.py`, `flyprof/wiki.py`, `flyprof/cli.py` (3 new subcommand registrations), `runs/flash_attn_humanize/{prompt,interface,dispatch}.md`, `runs/flash_attn_humanize/{config.toml,workloads.json,attempts.jsonl,candidates.jsonl}`. The kernel under optimization stays `/sgl-workspace/FlyDSL-lab/kernels/flash_attn_gfx950.py`.

---

## 6. GUARDRAILS & HONESTY

- **Correctness is never relaxed.** `max_err<1e-2` AND `cosine>0.99` AND no NaN/Inf vs PyTorch fp32 SDPA, on every config, before any speed number is even computed. A failing gate ends the round — we do not "tune the tolerance."
- **Empty/regression is a finding, not a fix.** A rejected attempt is logged with its reason and fed forward; we report "tried X, regressed Y%, here's the flyprof evidence" rather than burying it. The loop's value is partly the map of what *doesn't* work.
- **Occupancy is a means, not the goal.** We REJECT any change that raises `waves_per_cu` but regresses gmean TFLOPS. The whole point of §2 Step 7 being separate from Step 6: a higher occupancy that costs locality/reuse (the classic BLOCK_M-shrink failure) is a loss. The acceptance criterion is TFLOPS; occupancy/bubble deltas are only *explanatory evidence*.
- **The maxnreg→accum-spill trap.** Forcing occupancy via launch-bounds/maxnreg can spill the O accumulator to scratch (HBM, 50-200× slower). The COMPILE GATE asserts `.private_segment_fixed_size == 0` / `ScratchSize: 0`. Spill = automatic reject regardless of what occupancy says.
- **Uncalibrated-BW caveat.** The roofline reports `peak_hbm_gbs: 8000.0` with `bound_basis: counters(reliable ratios)` and `roofline.bound: "unknown"` — the absolute HBM ceiling is not calibrated on this node. We therefore drive decisions off **stall-class ratios and measured TFLOPS deltas** (which are reliable), **not** off "% of HBM peak" claims. Any "we're at N% of bandwidth" statement is flagged as uncalibrated.
- **Confidence honesty.** flyprof tags each recommendation `high/medium`; we carry that tag into the attempt log so a `medium`-confidence barrier_relax that worked isn't over-claimed.

---

## 7. SUCCESS CRITERIA & STOP CONDITION

- **"1.5× achieved" =** geometric-mean kernel-only TFLOPS across the 6 frozen configs ≥ 1.5× the **Step-0 FlyDSL baseline gmean**, with correctness passing on all configs and no individual config regressed >2% from its own baseline. Primary config concrete target: **371.7 → ≥557 TFLOPS** (92.4µs → ≤61.6µs).
- **Secondary success (partial win worth shipping):** ≥1.15× gmean *and* FlyDSL ≥ AITER-CK gmean (closing the 0.92× gap to parity-plus). The decomposed budget (barrier −5%, vmcnt −5%, occupancy lift) realistically lands ~1.5× *if* memory-latency is the true bottleneck; we ship whatever clears 1.15× with a PR + flyprof diff bundle.
- **Iteration / token budget:** target ≤8 accepted-or-rejected rounds to reach 1.5×; hard cap **15 rounds** or the agent's token budget, whichever first. Each round is bounded: 1 diagnose + 1 wiki query + 1 edit + 1 correctness run + 1 re-profile + 1 bench.
- **Stop / escalate to human (Jinn) when:** (a) 1.5× reached — stop, write PR; (b) 3 consecutive rounds with no accepted improvement — stop, report the attempt log and the flyprof "is_optimal" state; (c) `flyprof` reports `optimal: True` (the `is_optimal` signal: bound + low recoverable bubbles) before 1.5× — we've hit the algorithmic ceiling of this kernel structure, escalate (the fix is now algorithm/tile-redesign, a human call); (d) any change that would require touching the *generic* path or the MMA atom shape in a way that risks other kernels — escalate before applying.

---

## 8. RISKS & OPEN QUESTIONS (flag before launch)

1. **Baseline provenance.** The 404 TFLOPS AITER-CK baseline is *inferred* from the FINDINGS.md 0.92× ratio, not freshly measured. **Before launch we should run the AITER-CK provider once to get a real baseline per config** — otherwise the 1.5× denominator is soft. (The 371.7 FlyDSL number IS measured.)
2. **Is the bottleneck truly latency-hideable, or algorithmic?** flyprof says memory-bound with 60.6% stall, but if the kernel is fundamentally bandwidth-starved at S=8192 (no calibrated roofline), occupancy won't buy 1.5× and the win caps lower. Open question until we have calibrated BW or see the first re-profile.
3. **Occupancy headroom may be smaller than hoped.** ROCmKernelWiki's own war-story says FlyDSL dual-wave sits at 0.92× *because* pushing to 4 waves/SIMD trades occupancy for worse locality — the documented blueprint targets **2** waves/SIMD, not more. So "4→6+ waves/CU" may not be the right target; "4→8 (2/SIMD) cleanly, scratch-free" might be the realistic ceiling, capping the occupancy lever's contribution.
4. **Knob interactions.** Iteration 3's deeper prefetch raises VGPRs and can undo Iteration 1's occupancy. The loop handles this by re-ranking, but it means rounds aren't independent — we may need a small autotune over (BLOCK_M × NUM_PREFETCH_K × waves_per_eu) rather than pure greedy.
5. **Harness reuses an un-built tree.** Per our env notes, FlyDSL-lab runs from a non-built tree via PYTHONPATH; the `rocprofv3`/JIT-cache/`FLYDSL_DUMP_IR` recipe must be validated on the target node before the loop runs unattended (this is a `flyprof doctor` preflight).
6. **Scope of edits.** `BLOCK_M`, `NUM_PREFETCH_K`, the MFMA atom are *hard-coded constants* (not parameters) in `flash_attn_gfx950.py` — the loop edits source lines, not just call-args. Confirm Jinn is OK with the agent editing kernel internals (vs only flipping the boolean knobs) and which file/branch it commits to.

**Question for Jinn before I build this:** do you want the loop to (a) edit kernel source freely within the dual-wave path, or (b) only toggle the existing boolean/int knobs (`waves_per_eu`, `dualwave_swp_*`, and the few int constants) for a safer first run? And should I run the AITER-CK baseline measurement first to harden the §3 denominators?

---

**Key files this plan touches:** `/sgl-workspace/FlyDSL-lab/kernels/flash_attn_gfx950.py` (kernel under optimization), `/sgl-workspace/FlyDSL-lab/tests/kernels/test_flash_attn_fwd.py` (correctness oracle), `/sgl-workspace/flydsl-rocprof-cli/flyprof/{cli.py,knobs.py,report.py}` (+ new `diff.py`, `optimize.py`, `wiki.py`), `/sgl-workspace/flydsl-rocprof-cli/examples/flash_attn_fwd/source/report.json` (verified baseline numbers), `/sgl-workspace/flydsl-rocprof-cli/skills/references/bubble-knob-table.md` (bubble→technique seed map), `/sgl-workspace/ROCmKernelWiki/scripts/{query.py,get_page.py}` (technique retrieval), and a new `/sgl-workspace/flydsl-rocprof-cli/runs/flash_attn_humanize/` task-spec + attempt-log dir.

---

## DECISIONS LOCKED (2026-06-10)

- **Edit scope = FREE SOURCE EDITS within the gfx950 dual-wave path.** The loop may edit
  `flash_attn_gfx950.py` internals (barrier placement, LDS staging, prefetch structure,
  the int constants), not just toggle booleans — every edit gated by the COMPILE gate
  (scratch==0) and the CORRECTNESS gate (vs torch SDPA, all configs). Work on a branch.
- **Baseline = MEASURE AITER-CK FIRST.** Before launching the loop, run the multishape
  benchmark (AITER-CK / AITER-Triton / torch providers) on all 6 configs to replace the
  *inferred* 404 TFLOPS with real per-config numbers, hardening the 1.5x denominator.

### Execution order from here
1. `flyprof doctor` preflight on the target node (rocprofv3 / JIT cache / FLYDSL_DUMP_IR).
2. Measure AITER-CK baseline (use the `flydsl-kernel-multishape-benchmark` skill) → fill §3.
3. Lock the Step-0 FlyDSL baseline bundle (`flyprof run flash_attn_fwd` on the 6 configs).
4. Build the harness glue: `flyprof/{diff,optimize,wiki}.py` + `runs/flash_attn_humanize/`.
5. Run the RLCR loop (free source edits), Iteration 1 = occupancy/ArchVGPR.
