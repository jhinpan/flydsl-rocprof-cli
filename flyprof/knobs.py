"""The core IP: map a stall/bubble to the FlyDSL thread-level knob that fixes it.

A rule fires on a *diagnostic signal* (a field from ``bubbles.json`` / ``counters.json``
crossing a threshold), is *gated by the roofline bound* (so we never tell a
compute-bound kernel to grow its tile, or a memory-bound one to change its MMA
K-depth), and emits a recommendation carrying its **evidence** — the field+value
that fired it plus the ``code_line``/``source:line`` the bubble maps to — so every
claim is auditable back to the ISA. Recommendations are ranked by the stall % of
the firing bubble (times a small priority weight).

This module is pure (no GPU, no I/O); it consumes the JSON artifacts other
commands produced. The full human-readable table lives in
``skills/references/bubble-knob-table.md`` — keep the two in sync.
"""
from __future__ import annotations

import dataclasses
from typing import Callable, Optional


@dataclasses.dataclass
class Signals:
    by_class: dict          # bubbles.stall_taxonomy.by_class
    inst_mix: dict          # bubbles.inst_mix
    hotspots: list          # bubbles.hotspots
    bound: Optional[str]    # counters.roofline.bound  (memory|compute|latency|None)
    pct_of_hbm_roof: Optional[float]
    partial_32b_frac: Optional[float]
    l2_hit_rate: Optional[float]
    lds_conflict_pct: Optional[float]
    mfma_frac: Optional[float]
    occ_waves_per_cu: Optional[int]
    arch_vgpr: Optional[int]
    accum_vgpr: Optional[int]

    def cls_pct(self, cls: str) -> float:
        return float(self.by_class.get(cls, {}).get("pct", 0.0))

    def mix(self, cls: str) -> float:
        return float(self.inst_mix.get(cls, 0.0))

    def evidence(self, cls: str, fired_field: str, fired_value) -> dict:
        """Top hotspot of a class -> the source line + waitcnt attribution."""
        hs = next((h for h in self.hotspots if h.get("class") == cls), None)
        ev = {"signal": fired_field, "value": fired_value, "stall_pct_of_class": self.cls_pct(cls)}
        if hs:
            ev["code_line"] = hs.get("code_line")
            ev["source"] = hs.get("source")
            ev["inst"] = hs.get("inst")
            if hs.get("waitcnt_sources"):
                ev["waits_on"] = [{"source": w.get("source"), "inst": w.get("inst")}
                                  for w in hs["waitcnt_sources"][:3]]
        return ev


@dataclasses.dataclass
class Rule:
    id: str
    bubble_class: str
    when: Callable[[Signals], bool]
    fired_field: str
    fired_value: Callable[[Signals], object]
    knob: str
    api: str
    code_change: str
    expected: str
    confidence: str
    gate: Optional[set] = None   # bounds this rule is relevant for (None = any)
    priority: float = 1.0


def _bound_ok(rule: Rule, sig: Signals) -> bool:
    if rule.gate is None or sig.bound is None:
        return True
    return sig.bound in rule.gate


RULES: list[Rule] = [
    Rule(
        id="vmem_prefetch", bubble_class="vmcnt",
        when=lambda s: s.cls_pct("vmcnt") > 10 and (s.pct_of_hbm_roof is None or s.pct_of_hbm_roof < 70),
        fired_field="vmcnt.pct", fired_value=lambda s: s.cls_pct("vmcnt"),
        knob="Software-prefetch / double-buffer the global loads (loop-carried SSA)",
        api="for-loop with fx.Index bounds + init=[...]; issue next-iter buffer_ops.buffer_load at loop top, "
            "consume the prior iter's data in the MMA/compute body",
        code_change="convert `for i in range(N)` -> `for iv, st in range(fx.Index(0), fx.Index(N-1), fx.Index(1), "
                    "init=[...])`; prefetch load[i+1] before using load[i]. Bounds MUST be fx.Index or the rewriter "
                    "unrolls and drops init=.",
        expected="overlaps VMEM latency with compute; drains the s_waitcnt vmcnt stall",
        confidence="high", gate={"memory", "latency"}, priority=1.2,
    ),
    Rule(
        id="vmem_queue_depth", bubble_class="vmcnt",
        when=lambda s: s.cls_pct("vmcnt") > 10 and s.mix("vmem_load") > 0.20,
        fired_field="inst_mix.vmem_load", fired_value=lambda s: s.mix("vmem_load"),
        knob="Tune VMEM issue grouping / use a vmcnt-only fence",
        api="rocdl.sched_vmem(N) to group buffer_loads between compute; replace a full barrier with "
            "rocdl.s_wait_loadcnt(0) (gfx950) / inline s_waitcnt vmcnt(N)",
        code_change="group N independent buffer_load ops, then a single vmcnt(0) fence instead of per-load waits",
        expected="raises memory-level parallelism; fewer, later vmcnt drains",
        confidence="medium", gate={"memory", "latency"}, priority=0.8,
    ),
    Rule(
        id="lds_write_read_distance", bubble_class="lgkmcnt",
        when=lambda s: s.cls_pct("lgkmcnt") > 15,
        fired_field="lgkmcnt.pct", fired_value=lambda s: s.cls_pct("lgkmcnt"),
        knob="Increase LDS write->read distance / prefetch the LDS read",
        api="insert independent buffer_load or SALU work between lds_ptr.store(...) and the dependent "
            "gpu.barrier()/load; A0-prefetch lds_load_pack_k32(...) right after gpu.barrier()",
        code_change="reorder so the consuming ds_read is far from the producing ds_write; hoist the next tile's "
                    "LDS read across the barrier",
        expected="hides the LDS write->read latency that the s_waitcnt lgkmcnt(0) is exposing",
        confidence="high", gate=None, priority=1.1,
    ),
    Rule(
        id="lds_bank_conflict", bubble_class="lds",
        when=lambda s: (s.lds_conflict_pct is not None and s.lds_conflict_pct > 5)
                       or (s.cls_pct("lds") > 8),
        fired_field="lds_conflict_pct",
        fired_value=lambda s: s.lds_conflict_pct if s.lds_conflict_pct is not None else s.cls_pct("lds"),
        knob="Remove LDS bank conflicts (XOR swizzle preferred)",
        api="swizzle_xor16(row, col, k_blocks16) applied identically on store+load; or +PADDING in make_layout "
            "stride; or rocdl.cdna4.LDSReadTrans8_64b() (gfx950, 256B conflict-free transpose load)",
        code_change="apply the same XOR-swizzle to the LDS store and load index maps so 32 banks are hit evenly",
        expected="eliminates serialized bank-conflict replays on ds_read/ds_write",
        confidence="high", gate=None, priority=1.0,
    ),
    Rule(
        id="mfma_raw", bubble_class="mfma",
        when=lambda s: s.cls_pct("mfma") > 5 and (s.mfma_frac is None or s.mfma_frac > 0.30 or s.mix("mfma") > 0.30),
        fired_field="mfma.pct", fired_value=lambda s: s.cls_pct("mfma"),
        knob="Fill the MFMA pipeline / break the accumulator RAW chain",
        api="interleave rocdl.sched_mfma(N) with sched_dsrd/sched_vmem; tune dsrd_preload/dvmem_preload; "
            "rocdl.s_setprio(...) to raise the compute wave's priority",
        code_change="schedule independent MFMA groups so the next MFMA issues before the prior accumulator is read",
        expected="reduces idle/s_nop between dependent v_mfma instructions",
        confidence="medium", gate={"compute", "latency"}, priority=0.9,
    ),
    Rule(
        id="barrier_relax", bubble_class="barrier",
        when=lambda s: s.cls_pct("barrier") > 5,
        fired_field="barrier.pct", fired_value=lambda s: s.cls_pct("barrier"),
        knob="Relax / remove redundant synchronization",
        api="drop a redundant gpu.barrier(); in ping-pong, barrier only the swapped LDS region; use "
            "rocdl.s_wait_dscnt(0) (gfx950) for intra-wave order; replace an LDS cross-wave reduce with "
            "gpu.ShuffleOp(v, off, 64, mode='xor') over [32,16,8,4,2,1]",
        code_change="remove barriers that don't guard a real cross-wave LDS dependency; prefer DPP/shuffle reduce",
        expected="cuts the s_barrier stall where waves wait on each other unnecessarily",
        confidence="medium", gate=None, priority=0.8,
    ),
    Rule(
        id="occupancy_vgpr", bubble_class="occupancy",
        when=lambda s: (s.occ_waves_per_cu is not None and s.occ_waves_per_cu < 8),
        fired_field="occupancy_waves_per_cu", fired_value=lambda s: s.occ_waves_per_cu,
        knob="Raise occupancy by cutting register footprint",
        api="use_async_copy=True -> rocdl.buffer_load_to_lds(...) (bypasses ~32 arch_vgpr of the A-tile); shrink "
            "tile_m/tile_k; @autotune Config(waves_per_eu=2). Do NOT force maxnreg to push accum_vgpr=0 "
            "(spills via v_accvgpr_read, ~4.5x regression).",
        code_change="move the A/B tile load through LDS via async copy to free the staging VGPRs, or reduce the tile",
        expected="more resident waves to hide the exposed memory/LDS latency",
        confidence="high", gate={"memory", "latency"}, priority=1.0,
    ),
    Rule(
        id="vectorize_copy", bubble_class="vmem_load",
        when=lambda s: (s.partial_32b_frac is not None and s.partial_32b_frac > 0.15),
        fired_field="partial_32b_frac", fired_value=lambda s: s.partial_32b_frac,
        knob="Widen the copy / load width to full cache lines",
        api="buffer_ops.buffer_load(..., vec_width=4); choose a 128-bit copy atom fx.UniversalCopy(128) / "
            "rocdl.BufferCopy128b(); f32 rule vec_width = 128 // elem_bits. Verify VEC_WIDTH*sizeof(elem) <= atom_bits.",
        code_change="coalesce loads to 128-bit so each HBM request fills a full 64B line (no 32B partials)",
        expected="removes the 32B-partial bandwidth waste; fewer VMEM requests for the same bytes",
        confidence="high", gate={"memory", "latency"}, priority=1.0,
    ),
    Rule(
        id="mma_select", bubble_class="mfma",
        when=lambda s: s.bound == "compute" and (s.mfma_frac or 0) > 0.30,
        fired_field="bound==compute & mfma_frac", fired_value=lambda s: s.mfma_frac,
        knob="Pick the widest-K MMA atom for the dtype",
        api="fx.make_mma_atom(fx.rocdl.MFMA(m,n,k,dtype)) at max K: fp8 mfma_f32_16x16x32_fp8_fp8 (K32), bf16 K16, "
            "fp4 rocdl.cdna4.MFMA_Scale(16,16,128,...) (K128)",
        code_change="replace a small-K MMA atom with the largest-K variant so each MFMA does more work per issue",
        expected="higher MFMA throughput toward the compute roof",
        confidence="medium", gate={"compute"}, priority=0.9,
    ),
    Rule(
        id="layout_l2", bubble_class="vmcnt",
        when=lambda s: (s.l2_hit_rate is not None and s.l2_hit_rate < 70 and s.cls_pct("vmcnt") > 10),
        fired_field="l2_hit_rate", fired_value=lambda s: s.l2_hit_rate,
        knob="Improve coalescing / L2 reuse via layout + XCD remap",
        api="fx.make_layout / zipped_divide / slice so lanes load contiguous addresses; "
            "fx.make_tiled_mma(atom, make_layout((M_rep,N_rep,K_rep),...)); xcd_remap_bx_by(..., xcd_swizzle=N)",
        code_change="re-map the thread->data layout for contiguous per-lane addresses and cross-CTA L2 reuse",
        expected="raises L2 hit rate / coalescing; fewer HBM round-trips",
        confidence="medium", gate={"memory", "latency"}, priority=0.7,
    ),
]


def build_signals(bubbles: dict, counters: Optional[dict], capture: Optional[dict],
                  tag: str = "big") -> Signals:
    by_class = (bubbles.get("stall_taxonomy") or {}).get("by_class", {})
    mem = (counters or {}).get("memory", {})
    comp = (counters or {}).get("compute", {})
    rfl = (counters or {}).get("roofline", {})
    occ = None
    avgpr = accvgpr = None
    if capture:
        tdata = (capture.get("tags") or {}).get(tag) or {}
        occ = tdata.get("occupancy_waves_per_cu")
        avgpr = tdata.get("arch_vgpr")
        accvgpr = tdata.get("accum_vgpr")
    return Signals(
        by_class=by_class,
        inst_mix=bubbles.get("inst_mix", {}),
        hotspots=bubbles.get("hotspots", []),
        bound=rfl.get("bound"),
        pct_of_hbm_roof=mem.get("pct_of_hbm_roof"),
        partial_32b_frac=mem.get("partial_32b_frac"),
        l2_hit_rate=mem.get("l2_hit_rate"),
        lds_conflict_pct=((counters or {}).get("lds") or {}).get("bank_conflict_pct"),
        mfma_frac=comp.get("mfma_inst_frac"),
        occ_waves_per_cu=occ, arch_vgpr=avgpr, accum_vgpr=accvgpr,
    )


def is_optimal(sig: Signals) -> bool:
    """Near the HBM roof and dominated by a single load -> recommend no change."""
    return (sig.pct_of_hbm_roof is not None and sig.pct_of_hbm_roof > 85
            and sig.cls_pct("vmcnt") + sig.cls_pct("vmem_load") > 40)


def recommend(sig: Signals) -> list[dict]:
    if is_optimal(sig):
        return []  # caller emits the "already optimal" verdict
    recs = []
    for rule in RULES:
        try:
            if rule.when(sig) and _bound_ok(rule, sig):
                fv = rule.fired_value(sig)
                recs.append({
                    "rule": rule.id,
                    "bubble_class": rule.bubble_class,
                    "evidence": sig.evidence(rule.bubble_class, rule.fired_field, fv),
                    "knob": rule.knob,
                    "api": rule.api,
                    "code_change": rule.code_change,
                    "expected": rule.expected,
                    "confidence": rule.confidence,
                    "_weight": sig.cls_pct(rule.bubble_class) * rule.priority,
                })
        except Exception:
            continue
    recs.sort(key=lambda r: r["_weight"], reverse=True)
    for rank, r in enumerate(recs, 1):
        r["rank"] = rank
        r.pop("_weight", None)
    return recs
