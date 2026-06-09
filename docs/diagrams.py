#!/usr/bin/env python3
"""Generate the architecture SVGs for flydsl-rocprof-cli.

Run:  python docs/diagrams.py     # writes docs/*.svg

Diagrams are generated (not hand-placed) so they stay aligned and regenerate
when the command surface changes. One shared design system below.
"""
from __future__ import annotations

import html
from pathlib import Path

OUT = Path(__file__).resolve().parent

# ---- design system ---------------------------------------------------------
INK = "#1f2933"
SUB = "#5b6b7b"
FONT = "-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif"
MONO = "ui-monospace,SFMono-Regular,Menlo,Consolas,monospace"

# semantic palettes: (fill, stroke, text)
C = {
    "free":     ("#e6f4f1", "#0f766e", "#0b5750"),   # GPU-free pure transform
    "gpu":      ("#fde7d6", "#c2670a", "#8f4e07"),   # needs the GPU / rocprofv3
    "pre":      ("#eef1f4", "#64748b", "#44505e"),   # preflight / discovery
    "bundle":   ("#e9f1fc", "#2563eb", "#1d4ed8"),   # bundle artifacts (state)
    "ip":       ("#f3eafc", "#7c3aed", "#6d28d9"),   # core IP (knobs)
    "ext":      ("#f1f5f9", "#94a3b8", "#5b6b7b"),   # external / runtime
    "out":      ("#fef9c3", "#ca8a04", "#854d0e"),   # output artifact
    "plain":    ("#ffffff", "#cbd5e1", "#334155"),
}


def esc(s: str) -> str:
    return html.escape(str(s), quote=True)


def header(w, h, title=None, subtitle=None):
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
         f'viewBox="0 0 {w} {h}" font-family="{FONT}">']
    s.append('<defs>')
    for name, col in [("ar", SUB), ("arb", "#2563eb"), ("arp", "#7c3aed"), ("arg", "#c2670a")]:
        s.append(f'<marker id="{name}" markerWidth="9" markerHeight="9" refX="7.5" refY="3" '
                 f'orient="auto" markerUnits="userSpaceOnUse">'
                 f'<path d="M0,0 L8,3 L0,6 Z" fill="{col}"/></marker>')
    s.append('</defs>')
    s.append(f'<rect x="0" y="0" width="{w}" height="{h}" fill="#ffffff"/>')
    if title:
        s.append(txt(w / 2, 38, title, 23, INK, weight="700", anchor="middle"))
    if subtitle:
        s.append(txt(w / 2, 62, subtitle, 13.5, SUB, anchor="middle"))
    return "\n".join(s)


def footer():
    return "</svg>"


def txt(x, y, s, size=13, fill=INK, weight="400", anchor="start", mono=False, ls="0"):
    fam = MONO if mono else FONT
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
            f'font-weight="{weight}" text-anchor="{anchor}" font-family="{fam}" '
            f'letter-spacing="{ls}">{esc(s)}</text>')


def box(x, y, w, h, kind="plain", rx=9, sw=1.6, dash=None):
    fill, stroke, _ = C[kind]
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')


def node(x, y, w, h, label, kind="plain", sub=None, mono=True, rx=9, size=14):
    _, _, tcol = C[kind]
    s = [box(x, y, w, h, kind, rx=rx)]
    cy = y + h / 2 + (size * 0.34 if not sub else -2)
    s.append(txt(x + w / 2, cy, label, size, tcol, weight="600", anchor="middle", mono=mono))
    if sub:
        s.append(txt(x + w / 2, cy + 15, sub, 10.5, tcol, anchor="middle"))
    return "\n".join(s)


def arrow(x1, y1, x2, y2, color=SUB, sw=1.7, dash=None, marker="ar"):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{color}" stroke-width="{sw}"{d} marker-end="url(#{marker})"/>')


def cpath(d, color=SUB, sw=1.7, dash=None, marker="ar"):
    da = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}"{da} '
            f'marker-end="url(#{marker})"/>')


def legend(x, y, items):
    """items: list of (kind|None, label, swatch_color_override)."""
    s = []
    cx = x
    for kind, label, *over in items:
        if kind == "dash":
            s.append(f'<line x1="{cx}" y1="{y+7}" x2="{cx+26}" y2="{y+7}" stroke="{SUB}" '
                     f'stroke-width="1.7" stroke-dasharray="4 3"/>')
            sw_w = 26
        elif kind == "solid":
            s.append(f'<line x1="{cx}" y1="{y+7}" x2="{cx+26}" y2="{y+7}" stroke="{SUB}" '
                     f'stroke-width="1.7" marker-end="url(#ar)"/>')
            sw_w = 26
        else:
            fill, stroke, _ = C[kind]
            s.append(f'<rect x="{cx}" y="{y}" width="16" height="14" rx="3" fill="{fill}" '
                     f'stroke="{stroke}" stroke-width="1.4"/>')
            sw_w = 16
        s.append(txt(cx + sw_w + 7, y + 11, label, 11.5, SUB))
        cx += sw_w + 7 + len(label) * 6.6 + 20
    return "\n".join(s)


# ---- diagram 1: pipeline + bundle dataflow ---------------------------------
def build_pipeline():
    W, H = 1280, 565
    s = [header(W, H, "flyprof — the fixed workflow",
                "one JSON envelope per command · state flows through the bundle dir · each command writes one artifact, later commands read it")]

    cmds = [
        ("list", "pre", "kernels"),
        ("doctor", "pre", "preflight"),
        ("tile", "free", "GPU-free"),
        ("capture", "gpu", "rocprofv3 ATT"),
        ("counters", "gpu", "rocprofv3 PMC"),
        ("bubbles", "free", "GPU-free"),
        ("map", "free", "GPU-free"),
        ("report", "free", "GPU-free"),
        ("bundle", "free", "GPU-free"),
    ]
    arts = {
        "doctor": ["doctor.json"],
        "tile": ["tile.json"],
        "capture": ["capture.json", "att/<tag>/"],
        "counters": ["counters.json", "pmc/"],
        "bubbles": ["bubbles.json"],
        "map": ["map.json"],
        "report": ["REPORT.md", "report.json"],
        "bundle": ["examples/<k>/"],
    }

    n = len(cmds)
    m = 40
    gap = 14
    bw = (W - 2 * m - (n - 1) * gap) / n
    bh = 50
    yc = 100
    centers = []
    # command rail
    for i, (name, kind, sub) in enumerate(cmds):
        x = m + i * (bw + gap)
        centers.append(x + bw / 2)
        s.append(node(x, yc, bw, bh, name, kind, sub=sub, size=15))
        if i < n - 1:
            ax = x + bw
            s.append(arrow(ax + 1, yc + bh / 2, ax + gap - 1, yc + bh / 2))

    # bundle container
    by, bhgt = 250, 150
    s.append(f'<rect x="{m}" y="{by}" width="{W-2*m}" height="{bhgt}" rx="14" '
             f'fill="#f7faff" stroke="#2563eb" stroke-width="1.4" stroke-dasharray="2 4"/>')
    s.append(txt(m + 16, by + 26, "bundle/", 16, "#1d4ed8", weight="700", mono=True))
    s.append(txt(m + 100, by + 26, "the cross-command state handle", 12, SUB))

    # artifact chips under their producing command + write arrows
    chip_y = by + 52
    for i, (name, kind, sub) in enumerate(cmds):
        if name not in arts:
            continue
        cx = centers[i]
        # write arrow command -> chips
        s.append(arrow(cx, yc + bh + 2, cx, chip_y - 8, color="#2563eb", marker="arb"))
        yy = chip_y
        for a in arts[name]:
            cw = max(bw, len(a) * 7.4 + 14)
            s.append(box(cx - cw / 2, yy, cw, 26, "bundle", rx=7, sw=1.3))
            s.append(txt(cx, yy + 17, a, 11.5, "#1d4ed8", weight="600", anchor="middle", mono=True))
            yy += 34

    # representative "read" arrows: report reads the analysis artifacts
    rep_cx = centers[7]
    for src in (2, 3, 4, 5, 6):   # tile, capture, counters, bubbles, map
        sx = centers[src]
        s.append(cpath(f"M {sx:.1f} {by+bhgt-6} C {sx:.1f} {by+bhgt+30}, "
                       f"{rep_cx:.1f} {by+bhgt+30}, {rep_cx:.1f} {by+bhgt+8}",
                       color="#7c3aed", sw=1.3, dash="4 3", marker="arp"))
    s.append(txt(rep_cx, by + bhgt + 52, "report reads bubbles + counters + tile + map + capture",
                 11.5, "#6d28d9", anchor="middle", weight="600"))

    # input callout (single line, clear of the title)
    s.append(txt(m, yc - 14, "INPUT", 11, SUB, weight="700"))
    s.append(txt(m + 48, yc - 14, "flyprof run <kernel> --shape M,N,dtype --gpu 0", 12.5, INK, mono=True))

    # output callout (bottom-right, 3 lines)
    ow, ox, oy = 430, W - m - 430, 470
    s.append(box(ox, oy, ow, 66, "out", rx=10))
    s.append(txt(ox + 16, oy + 21, "OUTPUT", 11, "#854d0e", weight="700"))
    s.append(txt(ox + 16, oy + 40, "REPORT.md  +  report.json", 13, "#854d0e", weight="700", mono=True))
    s.append(txt(ox + 16, oy + 57, "ranked bubble→knob recommendations, traceable to .py:line", 11.5, "#854d0e"))
    s.append(arrow(rep_cx, by + bhgt + 60, ox + ow / 2, oy - 3, color="#ca8a04", marker="arg", sw=1.4))

    s.append(legend(m, H - 22, [
        ("pre", "preflight/discovery"), ("free", "GPU-free transform"),
        ("gpu", "needs GPU (rocprofv3)"), ("bundle", "bundle artifact"),
        ("solid", "writes"), ("dash", "reads"),
    ]))
    s.append(footer())
    (OUT / "pipeline.svg").write_text("\n".join(s))


def chips_row(labels, x, y, total_w, h, kind="plain", gap=12, kinds=None, subs=None, size=13.5, mono=True):
    n = len(labels)
    cw = (total_w - (n - 1) * gap) / n
    s, centers = [], []
    for i, lab in enumerate(labels):
        cx = x + i * (cw + gap)
        k = kinds[i] if kinds else kind
        sub = subs[i] if subs else None
        s.append(node(cx, y, cw, h, lab, k, sub=sub, size=size, mono=mono))
        centers.append(cx + cw / 2)
    return "\n".join(s), centers, cw


# ---- diagram 2: the four-repo fusion ---------------------------------------
def build_repos():
    W, H = 1180, 480
    s = [header(W, H, "What flyprof fuses",
                "four ROCm/FlyDSL tools + one agent-CLI pattern, wrapped behind one workflow")]
    contributors = [
        ("rocprofv3", "rocm-systems", "ATT trace + PMC counters — the raw capture engine", "gpu"),
        ("rocprof-compute", "rocm-systems", "gfx950 SoC spec → roofline & tile math (soc.py)", "pre"),
        ("rocprof-compute-viewer", "ROCm", "trace data model (code.json/wave/waitcnt) — replicated headlessly", "free"),
        ("flydsl-kernel-profiling", "the hub", "ATT capture harness + bundle layout — vendored", "bundle"),
        ("opencli", "pattern only", "deterministic CLI + companion-skill philosophy (not its code)", "ip"),
    ]
    lx, lw, bh, gap, top = 36, 470, 56, 16, 100
    centers = []
    for i, (name, org, contrib, kind) in enumerate(contributors):
        y = top + i * (bh + gap)
        s.append(box(lx, y, lw, bh, kind))
        _, _, tc = C[kind]
        s.append(txt(lx + 14, y + 23, name, 14.5, tc, weight="700", mono=True))
        s.append(txt(lx + 14, y + 41, org, 10.5, tc, weight="600"))
        s.append(txt(lx + 150, y + 33, contrib, 11.8, INK))
        centers.append((lx + lw, y + bh / 2))

    # center: flyprof
    fx, fw, fy, fh = 640, 230, 210, 130
    s.append(box(fx, fy, fw, fh, "plain", rx=14, sw=2.2))
    s.append(f'<rect x="{fx}" y="{fy}" width="{fw}" height="6" rx="3" fill="#7c3aed"/>')
    s.append(txt(fx + fw / 2, fy + 40, "flyprof", 26, INK, weight="800", anchor="middle", mono=True))
    s.append(txt(fx + fw / 2, fy + 64, "one deterministic CLI", 12.5, SUB, anchor="middle"))
    s.append(txt(fx + fw / 2, fy + 84, "+ companion skills", 12.5, SUB, anchor="middle"))
    s.append(txt(fx + fw / 2, fy + 110, "capture → analyze → report", 11.5, "#6d28d9", anchor="middle", weight="600"))
    for (ex, ey) in centers:
        s.append(cpath(f"M {ex+4} {ey} C {(ex+fx)/2} {ey}, {(ex+fx)/2} {fy+fh/2}, {fx-4} {fy+fh/2}",
                       color=SUB, sw=1.5))

    # runtime inputs (top) + outputs (right)
    s.append(box(fx - 10, 86, fw + 20, 30, "ext", rx=8, dash="4 3"))
    s.append(txt(fx + fw / 2, 105, "runtime: gfx950 / MI350X  ·  FlyDSL worktree (kernels/*.py + build tree)",
                 11, "#5b6b7b", anchor="middle"))
    s.append(arrow(fx + fw / 2, 116, fx + fw / 2, fy - 3, color=SUB, sw=1.4))

    oy = 150
    for i, (lab, sub, kind) in enumerate([
        ("REPORT.md", "ranked bubble→knob recs", "out"),
        ("report.json", "machine-readable, same data", "out"),
        ("examples/<k>/", "canonical bundle for the hub", "bundle"),
    ]):
        yy = oy + i * 66
        s.append(node(W - 36 - 250, yy, 250, 50, lab, kind, sub=sub, size=14))
        s.append(arrow(fx + fw + 3, fy + fh / 2, W - 36 - 250 - 3, yy + 25 if i == 1 else (yy + 25),
                       color="#ca8a04" if kind == "out" else "#2563eb",
                       marker="arg" if kind == "out" else "arb", sw=1.4))
    s.append(footer())
    (OUT / "repos.svg").write_text("\n".join(s))


# ---- diagram 3: layered module architecture --------------------------------
def build_architecture():
    W, H = 1180, 680
    s = [header(W, H, "Internal architecture",
                "generic core (any rocprofv3 ATT trace) + a FlyDSL-specific layer; GPU-free except capture/counters")]
    m = 36
    bw = W - 2 * m

    def band(y, h, name, note):
        s.append(f'<rect x="{m}" y="{y}" width="{bw}" height="{h}" rx="12" fill="#fbfcfe" '
                 f'stroke="#e2e8f0" stroke-width="1.4"/>')
        s.append(txt(m + 14, y + 20, name, 12.5, SUB, weight="700"))
        if note:
            s.append(txt(m + 14, y + 36, note, 10.5, "#94a3b8"))

    inner_x, inner_w = m + 150, bw - 165
    # Band A: agent surface
    band(82, 64, "AGENT SURFACE", "drive it, don't read source")
    a_svg, ac, _ = chips_row(
        ["skills/ — usage·capture·analyze·report·triage", "flyprof CLI (console script)"],
        inner_x, 94, inner_w, 40, kinds=["ip", "plain"], mono=False, size=12.5)
    s.append(a_svg)
    # Band B: contract
    band(166, 64, "CONTRACT", "envelope.py")
    b_svg, bc, _ = chips_row(
        ["cli.py — argparse dispatch", "envelope.py — ok/error + exit codes", "config.py — Config + bundle handle"],
        inner_x, 178, inner_w, 40, kind="plain", size=12.5)
    s.append(b_svg)
    # Band C: commands
    band(250, 64, "COMMANDS", "10 handlers")
    c_svg, cc, _ = chips_row(
        ["list", "doctor", "tile", "capture", "counters", "bubbles", "map", "report", "bundle", "run"],
        inner_x, 262, inner_w, 40,
        kinds=["pre", "pre", "free", "gpu", "gpu", "free", "free", "free", "free", "plain"], size=12)
    s.append(c_svg)
    # Band D: engines
    band(334, 110, "ENGINES", "the logic")
    d1, d1c, _ = chips_row(
        ["rcv — bubbles + map", "tile — roofline", "knobs ★ bubble→knob", "soc — gfx950", "report — synth"],
        inner_x, 348, inner_w, 38, kinds=["free", "free", "ip", "free", "free"], mono=False, size=12)
    s.append(d1)
    d2, d2c, _ = chips_row(
        ["capture — rocprofv3 ATT (vendored)", "counters — rocprofv3 PMC", "manifest — recipes"],
        inner_x, 396, inner_w, 38, kinds=["gpu", "gpu", "free"], mono=False, size=12)
    s.append(d2)
    # Band E: vendored + external
    band(456, 96, "VENDORED + EXTERNAL", "")
    e_svg, ec, _ = chips_row(
        ["_vendor/hotspot — VGPR/occ", "rocprofv3 (rocm)", "FlyDSL worktree", "gfx950 / MI350X"],
        inner_x, 472, inner_w, 40, kinds=["free", "ext", "ext", "ext"], mono=False, size=12)
    s.append(e_svg)

    # vertical flow arrows between band centers
    for y0, y1 in [(146, 166), (230, 250), (314, 334), (444, 456)]:
        s.append(arrow(W / 2, y0, W / 2, y1 - 2, color=SUB, sw=1.5))

    # generic vs FlyDSL-specific side rails
    s.append(f'<rect x="{m}" y="600" width="{bw}" height="56" rx="10" fill="#ffffff" stroke="#e2e8f0"/>')
    s.append(txt(m + 16, 622, "GENERIC CORE", 11.5, "#0b5750", weight="700"))
    s.append(txt(m + 16, 640, "envelope · config · rcv · tile · soc · hotspot — works on any rocprofv3 ATT trace",
                 11, INK))
    s.append(txt(W / 2 + 70, 622, "FlyDSL-SPECIFIC LAYER", 11.5, "#6d28d9", weight="700"))
    s.append(txt(W / 2 + 70, 640, "knobs (knob vocabulary) · manifest (recipes) · capture (DWARF source map)",
                 11, INK))
    s.append(footer())
    (OUT / "architecture.svg").write_text("\n".join(s))


# ---- diagram 4: bubble -> knob decision flow -------------------------------
def build_bubble_knob():
    W, H = 1200, 470
    s = [header(W, H, "How a stall becomes a recommendation",
                "the core IP (knobs.py): reliable signals → bound-gated rules → ranked recs, each traceable to a .py line")]
    col_w = 250
    # column 1: inputs
    x1 = 36
    s.append(txt(x1, 96, "EVIDENCE (from the bundle)", 11.5, SUB, weight="700"))
    inputs = [
        ("code.json", "per-line stall cycles (c8) by ISA class — the taxonomy", "free"),
        ("wave.waitcnt", "dep edges: each s_waitcnt → the load it waits on", "free"),
        ("PMC counters", "ratios: L2 hit · 32B · MFMA-frac (abs BW uncalib.)", "gpu"),
    ]
    iy = 116
    icent = []
    for i, (name, sub, kind) in enumerate(inputs):
        yy = iy + i * 78
        s.append(box(x1, yy, col_w, 60, kind))
        _, _, tc = C[kind]
        s.append(txt(x1 + 14, yy + 24, name, 14, tc, weight="700", mono=True))
        s.append(txt(x1 + 14, yy + 44, sub, 10.3, INK))
        icent.append((x1 + col_w, yy + 30))

    # column 2: signals
    x2 = x1 + col_w + 70
    sy, sh = 150, 200
    s.append(box(x2, sy, 180, sh, "plain", rx=12, sw=1.8))
    s.append(txt(x2 + 90, sy + 34, "build_signals", 14, INK, weight="700", anchor="middle", mono=True))
    s.append(txt(x2 + 90, sy + 56, "normalize +", 11, SUB, anchor="middle"))
    s.append(txt(x2 + 90, sy + 71, "attach source", 11, SUB, anchor="middle"))
    s.append(txt(x2 + 90, sy + 100, "bound:", 11.5, "#6d28d9", anchor="middle", weight="700"))
    s.append(txt(x2 + 90, sy + 118, "memory /", 11, "#6d28d9", anchor="middle"))
    s.append(txt(x2 + 90, sy + 133, "compute /", 11, "#6d28d9", anchor="middle"))
    s.append(txt(x2 + 90, sy + 148, "latency", 11, "#6d28d9", anchor="middle"))
    for (ex, ey) in icent:
        s.append(cpath(f"M {ex+3} {ey} C {(ex+x2)/2} {ey}, {(ex+x2)/2} {sy+sh/2}, {x2-3} {sy+sh/2}",
                       color=SUB, sw=1.5))

    # column 3: rules engine
    x3 = x2 + 180 + 64
    s.append(box(x3, sy, 210, sh, "ip", rx=12, sw=2.2))
    s.append(txt(x3 + 105, sy + 30, "knobs.RULES", 15, "#6d28d9", weight="800", anchor="middle", mono=True))
    s.append(txt(x3 + 105, sy + 50, "bound-gated rules engine", 10.5, "#6d28d9", anchor="middle"))
    rules = ["vmcnt → prefetch / double-buffer", "lgkmcnt → LDS write→read dist",
             "lds → XOR swizzle", "mfma → sched_mfma / wider-K",
             "occupancy → async-copy", "32B → widen copy width"]
    for i, r in enumerate(rules):
        s.append(txt(x3 + 14, sy + 76 + i * 19, "• " + r, 10.3, "#4c1d95"))
    s.append(arrow(x2 + 180 + 3, sy + sh / 2, x3 - 3, sy + sh / 2, color="#7c3aed", marker="arp", sw=1.7))

    # column 4: ranked recs + report
    x4 = x3 + 210 + 64
    s.append(txt(x4, 96, "RANKED RECOMMENDATIONS", 11.5, SUB, weight="700"))
    recs = [("#1 vmcnt", "prefetch the global loads", "high"),
            ("#2 vmcnt", "layout / L2 reuse", "med"),
            ("#3 barrier", "relax sync", "med")]
    ry = 116
    for i, (rk, txt_, conf) in enumerate(recs):
        yy = ry + i * 66
        s.append(box(x4, yy, col_w + 10, 54, "out", rx=9))
        s.append(txt(x4 + 12, yy + 22, rk, 13, "#854d0e", weight="700", mono=True))
        s.append(txt(x4 + 100, yy + 22, txt_, 11.3, INK))
        s.append(txt(x4 + 12, yy + 41, "evidence → softmax_kernel.py:43", 10.2, "#6d28d9", weight="600", mono=True))
    s.append(arrow(x3 + 210 + 3, sy + sh / 2, x4 - 3, sy + sh / 2, color="#ca8a04", marker="arg", sw=1.7))

    # traceability ribbon
    s.append(f'<rect x="{x1}" y="{H-66}" width="{W-2*x1}" height="40" rx="9" fill="#f3eafc" '
             f'stroke="#7c3aed" stroke-width="1.4"/>')
    s.append(txt((W) / 2, H - 41, "every recommendation carries its firing signal + the .py:line the bubble maps to "
                 "— so the claim is auditable back to the ISA", 12, "#6d28d9", anchor="middle", weight="600"))
    s.append(footer())
    (OUT / "bubble-to-knob.svg").write_text("\n".join(s))


if __name__ == "__main__":
    build_pipeline()
    build_repos()
    build_architecture()
    build_bubble_knob()
    for f in ("pipeline", "repos", "architecture", "bubble-to-knob"):
        print("wrote", OUT / f"{f}.svg")
