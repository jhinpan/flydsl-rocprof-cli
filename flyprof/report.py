"""Synthesize the optimization report.

Consumes the bundle's artifacts (``capture`` + ``bubbles`` [+ ``counters`` +
``tile`` + ``map``]), runs the knob rules engine, and emits three isomorphic
outputs: a human ``REPORT.md`` (10-section), a machine ``report.json``, and the
auto-populated ``headlines.json`` (the fields a human used to hand-author). Every
recommendation carries ISA<->Python evidence.

Hard rule: if no bubble crosses threshold / the kernel sits at the HBM roof, emit
"already optimal" with the supporting evidence — never manufacture an opportunity.
"""
from __future__ import annotations

import time
from typing import Optional

from . import knobs
from .envelope import FlyprofError, Result
from .rcv import analyze, resolve_dispatch


def _load_optional(cfg, name: str) -> Optional[dict]:
    try:
        return cfg.read_artifact(name)
    except FlyprofError:
        return None


def _infer_bound(bubbles: dict, counters: Optional[dict]) -> str:
    """Bubble taxonomy is the primary, reliable evidence (a vmcnt stall *is* a
    memory wait); counters' MFMA fraction confirms the compute case."""
    bc = (bubbles.get("stall_taxonomy") or {}).get("by_class", {})
    pct = lambda k: bc.get(k, {}).get("pct", 0)  # noqa: E731
    mem_stall = pct("vmcnt") + pct("vmem_load") + pct("vmem_store")
    mfma_stall = pct("mfma")
    mfma_mix = bubbles.get("inst_mix", {}).get("mfma", 0)
    cmfma = ((counters or {}).get("compute") or {}).get("mfma_inst_frac")
    # compute-bound: MFMA dominates the work
    if (cmfma is not None and cmfma >= 0.5) or mfma_mix > 0.4 or mfma_stall > 25:
        return "compute"
    # memory-bound: VMEM waits dominate the stalls (direct evidence)
    if mem_stall > 35:
        return "memory"
    # counters corroborate memory (low L2 hit, no MFMA)
    if (counters or {}).get("roofline", {}).get("bound") == "memory":
        return "memory"
    return "latency"


def _headline(kernel: str, bound: str, bubbles: dict, recs: list, optimal: bool) -> str:
    bc = (bubbles.get("stall_taxonomy") or {}).get("by_class", {})
    top = next(iter(bc.items()), None)
    sp = bubbles.get("totals", {}).get("stall_pct", 0)
    if optimal:
        return f"{kernel}: {bound}-bound and near the HBM roofline — no change recommended."
    if top:
        cls, info = top
        lead = recs[0]["knob"] if recs else "see recommendations"
        return f"{kernel}: {bound}-bound, {cls}-dominated ({info.get('pct', 0)}% of {sp}% total stall) -> {lead}"
    return f"{kernel}: {bound}-bound; {sp}% stalled."


def build_report(cfg, tag: str) -> dict:
    capture = _load_optional(cfg, "capture.json")
    if not capture:
        raise FlyprofError("MISSING_PREREQ", "no capture.json in the bundle",
                           help="run `flyprof capture <kernel>` first")
    bubbles = _load_optional(cfg, "bubbles.json")
    if not bubbles:
        bubbles = analyze(resolve_dispatch(cfg, tag))
    counters = _load_optional(cfg, "counters.json")
    tile = _load_optional(cfg, "tile.json")

    sig = knobs.build_signals(bubbles, counters, capture, tag)
    optimal = knobs.is_optimal(sig)
    recs = knobs.recommend(sig)
    bound = _infer_bound(bubbles, counters)
    kernel = capture.get("kernel", "?")
    headline = _headline(kernel, bound, bubbles, recs, optimal)

    report = {
        "kernel": kernel,
        "recipe_kernel": capture.get("recipe_kernel"),
        "bound_type": bound,
        "headline": headline,
        "optimal": optimal,
        "totals": bubbles.get("totals", {}),
        "stall_taxonomy": bubbles.get("stall_taxonomy", {}),
        "roofline": (counters or {}).get("roofline"),
        "occupancy": {
            "from_capture_waves_per_cu": ((capture.get("tags") or {}).get(tag) or {}).get("occupancy_waves_per_cu"),
            "arch_vgpr": ((capture.get("tags") or {}).get(tag) or {}).get("arch_vgpr"),
            "accum_vgpr": ((capture.get("tags") or {}).get(tag) or {}).get("accum_vgpr"),
        },
        "tile": (tile or {}).get("recommended") or (tile or {}).get("evaluated"),
        "recommendations": recs,
        "evidence_hotspots": bubbles.get("hotspots", [])[:5],
        "provenance": {
            "arch": cfg.arch(),
            "rocprofv3": (capture.get("provenance") or {}).get("rocprofv3"),
            "date": time.strftime("%Y-%m-%d"),
            "avg_ns": capture.get("avg_ns"),
        },
    }
    return report


def render_md(r: dict) -> str:
    L = []
    p = r["provenance"]
    L.append(f"# {r['kernel']} — {p['arch']}/MI350X Profiling Report")
    L.append(f"_Provenance: FlyDSL · rocprofv3 · MI350X SPX · {p['date']} · "
             f"avg {r.get('provenance',{}).get('avg_ns')} ns/call_\n")

    L.append("## 1. Headline")
    L.append(r["headline"] + "\n")

    L.append("## 2. Workload & Tile")
    t = r.get("tile") or {}
    if t:
        tl = t.get("tile", {})
        L.append(f"- recommended tile: BM={tl.get('BM')} BN={tl.get('BN')} BK={tl.get('BK')} "
                 f"buffers={tl.get('buffers')}; regime={t.get('regime')}, limiter={t.get('binding_limiter')}, "
                 f"occupancy≈{t.get('occupancy_pct')}%")
    else:
        L.append("- (no tile.json; run `flyprof tile`)")
    L.append("")

    L.append("## 3. Roofline & Occupancy")
    rfl = r.get("roofline")
    if rfl:
        L.append(f"- bound: **{r['bound_type']}**; dtype {rfl.get('dtype')}, peak MFMA {rfl.get('peak_mfma_gflops')} "
                 f"GFLOP/s, ridge AI {rfl.get('ridge_ai_hbm')} FLOP/B")
    else:
        L.append(f"- bound (from ATT): **{r['bound_type']}** (no counters.json; run `flyprof counters`)")
    occ = r["occupancy"]
    L.append(f"- occupancy ≈ {occ.get('from_capture_waves_per_cu')} waves/CU "
             f"(arch_vgpr {occ.get('arch_vgpr')}, accum_vgpr {occ.get('accum_vgpr')})\n")

    L.append("## 4. Bubble / Stall Taxonomy")
    L.append(f"- total stall: {r['totals'].get('stall_pct')}% of {r['totals'].get('total_latency')} latency cycles")
    L.append("")
    L.append("| class | stall % | rank | bubble? |")
    L.append("|---|---|---|---|")
    for cls, info in (r["stall_taxonomy"].get("by_class") or {}).items():
        L.append(f"| {cls} | {info.get('pct')} | {info.get('rank')} | {'yes' if info.get('is_bubble') else ''} |")
    L.append("")

    L.append("## 5. ISA ↔ Python Evidence")
    for h in r["evidence_hotspots"]:
        line = f"- `{h.get('inst')}` @{h.get('addr')} (code_line {h.get('code_line')}, stall {h.get('stall')}) → {h.get('source')}"
        L.append(line)
        for w in h.get("waitcnt_sources", [])[:2]:
            L.append(f"    waits on `{w.get('inst')}` → {w.get('source')}")
    L.append("")

    L.append("## 6. Ranked Optimization Opportunities")
    if r["optimal"]:
        L.append("**Already optimal** — the kernel is near the HBM roofline and dominated by a single load. "
                 "No FlyDSL change recommended.")
    elif not r["recommendations"]:
        L.append("_No bubble crossed threshold; nothing to recommend._")
    else:
        for rec in r["recommendations"]:
            ev = rec["evidence"]
            L.append(f"### #{rec['rank']} — {rec['bubble_class']} "
                     f"(evidence: {ev.get('signal')}={ev.get('value')} @ {ev.get('source')})")
            L.append(f"- **Knob:** {rec['knob']}")
            L.append(f"- **API:** {rec['api']}")
            L.append(f"- **Change:** {rec['code_change']}")
            L.append(f"- **Expected:** {rec['expected']}  ·  confidence: {rec['confidence']}")
            if ev.get("waits_on"):
                L.append(f"- waits on: {', '.join(w.get('source','') for w in ev['waits_on'])}")
            L.append("")

    L.append("## 7. Verification Plan")
    L.append("- `FLYDSL_DUMP_IR=1` → inspect final_isa.s; `grep -c v_mfma|s_barrier|buffer_load|ds_read`")
    L.append("- re-run `flyprof run <kernel>` and diff `report.json` stall % + roofline before/after")
    L.append("- target: the rank-1 bubble's stall % drops; bound moves toward the roof\n")

    L.append("## 8. Counters (raw)")
    L.append("- see report.json.roofline / counters.json" if rfl else "- (none)")
    L.append("\n## 9. Dispatch / Occupancy")
    L.append(f"- arch_vgpr {occ.get('arch_vgpr')}, accum_vgpr {occ.get('accum_vgpr')}, "
             f"occ≈{occ.get('from_capture_waves_per_cu')} waves/CU")
    L.append("\n## 10. Repro")
    L.append(f"- `flyprof run {r.get('recipe_kernel') or r['kernel']} --gpu 0`")
    return "\n".join(L)


def cmd_report(args, cfg) -> Result:
    cfg.require_bundle()
    tag = "big"
    report = build_report(cfg, tag)
    # write the three isomorphic outputs
    (cfg.bundle / "REPORT.md").write_text(render_md(report))
    cfg.write_artifact("report.json", report)
    headlines = {"kernel": report["kernel"], "headline": report["headline"],
                 "bound_type": report["bound_type"], "optimal": report["optimal"],
                 "top_recommendation": report["recommendations"][0] if report["recommendations"] else None}
    cfg.write_artifact("headlines.json", headlines)

    data = {
        "kernel": report["kernel"], "bound_type": report["bound_type"],
        "headline": report["headline"], "optimal": report["optimal"],
        "recommendations": report["recommendations"],
        "report_md": str(cfg.bundle / "REPORT.md"),
        "report_json": str(cfg.bundle / "report.json"),
        "headlines_json": str(cfg.bundle / "headlines.json"),
    }
    warnings = []
    if not report["recommendations"] and not report["optimal"]:
        warnings.append("no recommendation fired; bubbles below thresholds")
    return Result(data=data, bundle=str(cfg.bundle), warnings=warnings)
