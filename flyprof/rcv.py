"""Headless replication of rocprof-compute-viewer's data extraction.

RCV is a Qt GUI; an agent can't drive it. But its inputs — the
``ui_output_agent_<PID>_dispatch_<N>`` directory rocprofv3 emits — are plain JSON,
and the analyses RCV runs over them are pure data transforms. This module
reproduces the two an optimizer needs, with no GPU, no Qt, no rocprofv3:

  * ``bubbles`` — the stall/bubble taxonomy (which classes of stall dominate,
    where in the ISA, attributed to the memory op that caused each waitcnt).
  * ``map``     — the ISA <-> Python-source bridge.

Column semantics are taken verbatim from RCV's own parser
(``src/code/codeload.cpp`` / ``.hpp`` and ``src/data/wavemanager.cpp``):

  code.json "code"[i] = [asm, _, pc_index(c2), source(c3), codeobj_id(c4),
                         addr(c5), hitcount(c6), latency_sum(c7),
                         stall_sum(c8), idle_sum(c9)]
  wave.instructions[j] = [clock, type, stall, cycles, code_line]
  wave.waitcnt        = [[waitcnt_code_line, [[source_code_line, count], ...]], ...]

(The legacy hotspot_analyzer.py mislabels c9 as "issue"; RCV treats c9 as *idle*
and issue == latency_sum - stall_sum. We follow RCV.)
"""
from __future__ import annotations

import glob
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

from .envelope import FlyprofError, Result

HOTSPOT_CAP = 25  # max hotspots emitted by default; full record via `map --code-line`


# --- stall / instruction classification (mirrors RCV ApplyCustomType + ISA) --
def classify(asm: str) -> str:
    a = asm.lower()
    if "s_waitcnt" in a or "s_wait_" in a:
        if "vmcnt" in a or "loadcnt" in a or "storecnt" in a:
            return "vmcnt"
        if "lgkmcnt" in a or "dscnt" in a or "kmcnt" in a:
            return "lgkmcnt"
        if "expcnt" in a:
            return "expcnt"
        return "waitcnt"
    if "s_barrier" in a or "s_wait_idle" in a or "barrier" in a:
        return "barrier"
    if "v_mfma" in a or "v_smfma" in a:
        return "mfma"
    if "buffer_load" in a or "global_load" in a or "flat_load" in a or "_load_to_lds" in a:
        return "vmem_load"
    if "buffer_store" in a or "global_store" in a or "flat_store" in a:
        return "vmem_store"
    if "ds_read" in a or "ds_load" in a or "ds_write" in a or "ds_store" in a or a.startswith("ds_"):
        return "lds"
    if "s_load" in a or "s_store" in a or "s_buffer_load" in a:
        return "smem"
    if a.startswith("v_"):
        return "valu"
    if a.startswith("s_") and ("branch" in a or "cbranch" in a or a.startswith("s_branch")):
        return "branch"
    if a.startswith("s_"):
        return "salu"
    return "other"


# "bubble" classes = stalls that mean the wave is *waiting*, not doing useful work
BUBBLE_CLASSES = {"lgkmcnt", "vmcnt", "expcnt", "waitcnt", "barrier"}


@dataclass
class Instr:
    code_line: int       # pc_index (c2) — also the index used by wave.waitcnt
    asm: str
    source: str
    addr: int
    hit: int
    latency: int
    stall: int
    idle: int

    @property
    def cls(self) -> str:
        return classify(self.asm)

    @property
    def issue(self) -> int:
        return max(0, self.latency - self.stall)


# --- dispatch-dir resolution ------------------------------------------------
def resolve_dispatch(cfg, tag: str) -> str:
    """Find the ui_output_*_dispatch_* dir for a tag, or accept a bundle that *is* one."""
    bundle = cfg.require_bundle()
    # 1. bundle points directly at a dispatch dir (fixtures / ad-hoc)
    if (bundle / "code.json").exists():
        return str(bundle)
    # 2. canonical capture layout: <bundle>/att/<tag>/ui_output_*_dispatch_*
    for base in (bundle / "att" / tag, bundle / "att"):
        hits = sorted(glob.glob(str(base / "ui_output_agent_*_dispatch_*")))
        hits = [h for h in hits if os.path.exists(os.path.join(h, "code.json"))]
        if hits:
            return hits[-1]
    # 3. anywhere under the bundle
    hits = [os.path.dirname(p) for p in glob.glob(str(bundle / "**" / "code.json"), recursive=True)]
    if hits:
        return sorted(hits)[-1]
    raise FlyprofError("NO_ARTIFACT", f"no ATT dispatch (code.json) under {bundle}",
                       help="run `flyprof capture` first, or point --bundle at a ui_output dir")


def load_code(dispatch_dir: str) -> list[Instr]:
    with open(os.path.join(dispatch_dir, "code.json")) as f:
        rows = json.load(f).get("code", [])
    out = []
    for c in rows:
        if len(c) < 10 or not isinstance(c[2], int) or c[2] == 0:
            continue  # header / comment lines have pc_index 0
        out.append(Instr(
            code_line=int(c[2]), asm=str(c[0]),
            source=str(c[3]) if c[3] else "", addr=int(c[5]),
            hit=int(c[6]), latency=int(c[7]), stall=int(c[8]), idle=int(c[9]),
        ))
    return out


def load_waitcnt(dispatch_dir: str) -> dict[int, set[int]]:
    """Aggregate every wave's waitcnt dependency edges: {waitcnt_code_line -> {source_code_lines}}."""
    edges: dict[int, set[int]] = defaultdict(set)
    for wf in glob.glob(os.path.join(dispatch_dir, "se*_sm*_*.json")):
        try:
            wave = json.load(open(wf)).get("wave", {})
        except Exception:
            continue
        for entry in wave.get("waitcnt", []) or []:
            if not entry:
                continue
            wc_line = int(entry[0])
            for pair in (entry[1] if len(entry) > 1 else []):
                edges[wc_line].add(int(pair[0]))
    return edges


def occupancy_summary(dispatch_dir: str) -> dict:
    """Coarse occupancy from wstates (#active-waves timeline) and occupancy.json presence."""
    out: dict = {"available": False}
    wpath = os.path.join(dispatch_dir, "wstates1.json")
    if os.path.exists(wpath):
        try:
            st = json.load(open(wpath)).get("state", [])
            if st:
                out.update(available=True, peak_waves=max(st),
                           mean_waves=round(sum(st) / len(st), 2), samples=len(st))
        except Exception:
            pass
    opath = os.path.join(dispatch_dir, "occupancy.json")
    if os.path.exists(opath):
        out["occupancy_json"] = True
    return out


# --- bubble taxonomy --------------------------------------------------------
def analyze(dispatch_dir: str) -> dict:
    instrs = load_code(dispatch_dir)
    if not instrs:
        raise FlyprofError("NO_ARTIFACT", f"code.json in {dispatch_dir} has no mapped instructions")
    edges = load_waitcnt(dispatch_dir)
    by_line = {i.code_line: i for i in instrs}

    total_latency = sum(i.latency for i in instrs)
    total_stall = sum(i.stall for i in instrs)
    total_idle = sum(i.idle for i in instrs)
    total_hit = sum(i.hit for i in instrs)

    # stall by class
    by_class_cyc: dict[str, int] = defaultdict(int)
    for i in instrs:
        if i.stall > 0:
            by_class_cyc[i.cls] += i.stall
    ranked = sorted(by_class_cyc.items(), key=lambda kv: kv[1], reverse=True)
    by_class = {}
    for rank, (cls, cyc) in enumerate(ranked, 1):
        by_class[cls] = {
            "cycles": cyc,
            "pct": round(100.0 * cyc / total_stall, 2) if total_stall else 0.0,
            "rank": rank,
            "is_bubble": cls in BUBBLE_CLASSES,
        }

    # instruction mix (by count)
    mix_cnt: dict[str, int] = defaultdict(int)
    for i in instrs:
        mix_cnt[i.cls] += 1
    n = len(instrs)
    inst_mix = {k: round(v / n, 3) for k, v in sorted(mix_cnt.items(), key=lambda kv: kv[1], reverse=True)}

    # by source line
    src_cyc: dict[str, int] = defaultdict(int)
    src_dom: dict[str, dict] = defaultdict(lambda: defaultdict(int))
    for i in instrs:
        if i.source and i.stall > 0:
            src_cyc[i.source] += i.stall
            src_dom[i.source][i.cls] += i.stall
    by_source = []
    for src, cyc in sorted(src_cyc.items(), key=lambda kv: kv[1], reverse=True)[:HOTSPOT_CAP]:
        dom = max(src_dom[src], key=src_dom[src].get)
        by_source.append({"source": src, "stall": cyc,
                          "pct": round(100.0 * cyc / total_stall, 2) if total_stall else 0.0,
                          "dom_class": dom})

    # hotspots: top instructions by stall, enriched with waitcnt attribution
    hot = sorted([i for i in instrs if i.stall > 0], key=lambda i: i.stall, reverse=True)
    hotspots = []
    for i in hot[:HOTSPOT_CAP]:
        rec = {
            "code_line": i.code_line, "addr": hex(i.addr), "inst": i.asm,
            "class": i.cls, "source": i.source,
            "hit": i.hit, "latency": i.latency, "stall": i.stall, "idle": i.idle,
            "stall_pct": round(100.0 * i.stall / i.latency, 1) if i.latency else 0.0,
        }
        if i.code_line in edges:
            srcs = []
            for sl in sorted(edges[i.code_line]):
                si = by_line.get(sl)
                srcs.append({"code_line": sl,
                             "inst": si.asm if si else "?",
                             "source": si.source if si else ""})
            rec["waitcnt_sources"] = srcs
        hotspots.append(rec)

    return {
        "dispatch_dir": os.path.basename(dispatch_dir),
        "totals": {
            "instructions": n,
            "total_latency": total_latency,
            "total_stall": total_stall,
            "total_idle": total_idle,
            "hitcount": total_hit,
            "stall_pct": round(100.0 * total_stall / total_latency, 1) if total_latency else 0.0,
        },
        "stall_taxonomy": {"by_class": by_class},
        "hotspots": hotspots,
        "by_source": by_source,
        "inst_mix": inst_mix,
        "occupancy": occupancy_summary(dispatch_dir),
        "stall_reason_available": False,  # SQTT trace carries no PC-sampling stall reasons
    }


# --- source mapping ---------------------------------------------------------
def load_source_map(dispatch_dir: str) -> dict[str, list]:
    """snapshots.json nested tree -> {virtual_source_path: [file lines]}."""
    sp = os.path.join(dispatch_dir, "snapshots.json")
    if not os.path.exists(sp):
        return {}
    tree = json.load(open(sp))
    path_map: dict[str, str] = {}

    def walk(node, prefix):
        for key, val in node.items():
            seg = "" if key == "/" else key
            path = (prefix.rstrip("/") + "/" + seg) if seg else prefix
            if isinstance(val, dict):
                walk(val, path)
            else:
                path_map[path] = val

    walk(tree, "")
    cache: dict[str, list] = {}
    for vpath, local in path_map.items():
        lp = os.path.join(dispatch_dir, local)
        if os.path.exists(lp):
            cache[vpath] = open(lp).read().splitlines()
    return cache


def snippet(cache: dict[str, list], source_loc: str, context: int = 2) -> Optional[dict]:
    if ":" not in source_loc:
        return None
    path, _, lineno_s = source_loc.rpartition(":")
    try:
        lineno = int(lineno_s)
    except ValueError:
        return None
    # match by suffix (snapshots vpaths and code.json paths can differ in prefix)
    lines = cache.get(path)
    if lines is None:
        for vpath, ls in cache.items():
            if vpath.endswith(path) or path.endswith(vpath.lstrip("/")):
                lines = ls
                break
    if not lines:
        return None
    lo = max(0, lineno - context - 1)
    hi = min(len(lines), lineno + context)
    return {"file": path, "line": lineno,
            "lines": [{"n": j + 1, "text": lines[j], "hot": j + 1 == lineno} for j in range(lo, hi)]}


def cmd_bubbles(args, cfg) -> Result:
    d = resolve_dispatch(cfg, args.tag)
    data = analyze(d)
    if args.detail is not None:
        instrs = {i.code_line: i for i in load_code(d)}
        i = instrs.get(args.detail)
        if not i:
            raise FlyprofError("NO_ARTIFACT", f"no instruction at code_line {args.detail}",
                               available=sorted(list(instrs))[:40])
        data["detail"] = {"code_line": i.code_line, "inst": i.asm, "class": i.cls,
                          "source": i.source, "addr": hex(i.addr), "hit": i.hit,
                          "latency": i.latency, "stall": i.stall, "idle": i.idle,
                          "issue": i.issue}
    n_total = len(load_code(d))
    truncated = {"hotspots_capped_at": HOTSPOT_CAP, "instructions_total": n_total} if n_total > HOTSPOT_CAP else None
    if cfg.bundle is not None and (cfg.bundle / "code.json").exists() is False:
        cfg.write_artifact("bubbles.json", data)
    return Result(data=data, bundle=str(cfg.bundle), truncated=truncated)


def cmd_map(args, cfg) -> Result:
    d = resolve_dispatch(cfg, args.tag)
    instrs = load_code(d)
    cache = load_source_map(d)

    sel = instrs
    if args.code_line is not None:
        sel = [i for i in instrs if i.code_line == args.code_line]
    elif args.source_line:
        sel = [i for i in instrs if i.source.endswith(args.source_line) or args.source_line in i.source]

    # default to the top stalling lines if no explicit selection
    explicit = args.code_line is not None or bool(args.source_line)
    if not explicit:
        sel = sorted([i for i in instrs if i.stall > 0], key=lambda i: i.stall, reverse=True)[:HOTSPOT_CAP]

    mappings = []
    unmapped = 0
    for i in sel:
        snip = snippet(cache, i.source) if i.source else None
        if not i.source or snip is None:
            unmapped += 1
        mappings.append({
            "code_line": i.code_line, "isa": i.asm, "addr": hex(i.addr),
            "source": i.source or "<unmapped>",
            "stall_attributed": i.stall, "latency": i.latency,
            "source_snippet": snip,
            "confidence": "exact" if snip else ("heuristic" if i.source else "none"),
        })

    # source flamegraph: latency/stall rolled up per file
    files: dict[str, dict] = defaultdict(lambda: {"latency": 0, "stall": 0})
    total_mapped = total_all = 0
    for i in instrs:
        total_all += i.latency
        if i.source and ":" in i.source:
            f = i.source.rsplit(":", 1)[0]
            files[f]["latency"] += i.latency
            files[f]["stall"] += i.stall
            total_mapped += i.latency
    flame = sorted(({"file": f, **v} for f, v in files.items()), key=lambda x: x["latency"], reverse=True)

    data = {
        "dispatch_dir": os.path.basename(d),
        "mappings": mappings,
        "unmapped_pct": round(100.0 * unmapped / max(len(sel), 1), 1),
        "src_mapped_pct": round(100.0 * total_mapped / total_all, 1) if total_all else 0.0,
        "source_flamegraph": {"files": flame},
    }
    if cfg.bundle is not None and (cfg.bundle / "code.json").exists() is False:
        cfg.write_artifact("map.json", data)
    return Result(data=data, bundle=str(cfg.bundle))
