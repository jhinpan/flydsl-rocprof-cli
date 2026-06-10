"""Before/after report comparison — the Step-6 mechanical check, made first-class.

Compares two bundles' ``report.json`` (the analysis artifact) and answers the only
question that matters after an optimization round: did the targeted bubble actually
shrink? Emits the before/after table (``bound_type``, ``totals.stall_pct``, per-class
``stall_taxonomy`` deltas with rank movement, occupancy deltas) plus the
``rank1_bubble_shrank`` verdict the optimize loop branches on — *shrank* means the
BEFORE report's top-ranked bubble class lost stall share, the mechanistic check that
is distinct from "got faster".

Pure: no GPU, no subprocess — just two report.json files.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

from .envelope import FlyprofError, Result

REPORT_NAME = "report.json"


def _load_report(bundle_dir: Union[str, Path]) -> dict:
    p = Path(bundle_dir) / REPORT_NAME
    if not p.exists():
        raise FlyprofError(
            "MISSING_PREREQ",
            f"{REPORT_NAME} not found in bundle {bundle_dir}",
            help="run `flyprof report --bundle <dir>` first",
        )
    return json.loads(p.read_text())


def _delta(b, a, ndigits: Optional[int] = None):
    """after − before; None unless both sides are numbers (a class can vanish)."""
    if isinstance(b, (int, float)) and isinstance(a, (int, float)):
        return round(a - b, ndigits) if ndigits else round(a - b)
    return None


def _by_class(report: dict) -> dict:
    return (report.get("stall_taxonomy") or {}).get("by_class") or {}


def _rank1_bubble(by_class: dict) -> Optional[str]:
    """The BEFORE-side optimization target: best-ranked class with is_bubble."""
    bubbles = [(info.get("rank", 99), cls) for cls, info in by_class.items() if info.get("is_bubble")]
    return min(bubbles)[1] if bubbles else None


def diff_reports(before: dict, after: dict) -> dict:
    """Pure diff of two report.json dicts -> the envelope ``data``."""
    bc_b, bc_a = _by_class(before), _by_class(after)

    taxonomy = {}
    classes = sorted(set(bc_b) | set(bc_a), key=lambda c: ((bc_b.get(c) or bc_a.get(c) or {}).get("rank", 99), c))
    for cls in classes:
        b, a = bc_b.get(cls), bc_a.get(cls)
        taxonomy[cls] = {
            "before": b,
            "after": a,
            "delta": {
                "cycles": _delta((b or {}).get("cycles"), (a or {}).get("cycles")),
                "pct": _delta((b or {}).get("pct"), (a or {}).get("pct"), 2),
                "rank": _delta((b or {}).get("rank"), (a or {}).get("rank")),
            },
            "is_bubble": bool((b or a or {}).get("is_bubble")),
        }

    occ_b, occ_a = before.get("occupancy") or {}, after.get("occupancy") or {}
    occupancy = {}
    for key, src in (("waves_per_cu", "from_capture_waves_per_cu"),
                     ("arch_vgpr", "arch_vgpr"),
                     ("accum_vgpr", "accum_vgpr")):
        occupancy[key] = {"before": occ_b.get(src), "after": occ_a.get(src),
                          "delta": _delta(occ_b.get(src), occ_a.get(src))}

    sp_b = (before.get("totals") or {}).get("stall_pct")
    sp_a = (after.get("totals") or {}).get("stall_pct")

    r1 = _rank1_bubble(bc_b)
    verdict = {"rank1_bubble_class": r1, "rank1_bubble_shrank": None,
               "before_pct": None, "after_pct": None, "delta_pct": None}
    if r1 is not None:
        bp = (bc_b.get(r1) or {}).get("pct") or 0.0
        ap = (bc_a.get(r1) or {}).get("pct") or 0.0  # class gone in AFTER == fully drained
        verdict.update(before_pct=bp, after_pct=ap, delta_pct=round(ap - bp, 2),
                       rank1_bubble_shrank=ap < bp)

    headline = (
        f"rank-1 bubble {r1 or 'none'}: {verdict['before_pct']}% -> {verdict['after_pct']}% "
        f"({'shrank' if verdict['rank1_bubble_shrank'] else 'did NOT shrink'}); "
        f"total stall {sp_b}% -> {sp_a}%; "
        f"waves/CU {occupancy['waves_per_cu']['before']} -> {occupancy['waves_per_cu']['after']}; "
        f"arch_vgpr {occupancy['arch_vgpr']['before']} -> {occupancy['arch_vgpr']['after']}"
    )

    return {
        "kernel": {"before": before.get("kernel"), "after": after.get("kernel")},
        "headline": headline,
        "bound_type": {"before": before.get("bound_type"), "after": after.get("bound_type"),
                       "changed": before.get("bound_type") != after.get("bound_type")},
        "totals": {"stall_pct": {"before": sp_b, "after": sp_a, "delta": _delta(sp_b, sp_a, 2)}},
        "stall_taxonomy": taxonomy,
        "occupancy": occupancy,
        "verdict": verdict,
    }


def run_diff(before_dir: Union[str, Path], after_dir: Union[str, Path]) -> dict:
    """Load <dir>/report.json from both bundles and diff them. Returns envelope data."""
    return diff_reports(_load_report(before_dir), _load_report(after_dir))


def cmd_diff(args, cfg) -> Result:
    before, after = getattr(args, "before", None), getattr(args, "after", None)
    if not before or not after:
        raise FlyprofError("BAD_ARGS", "diff needs --before <bundle> --after <bundle>")
    data = run_diff(before, after)
    if not data["stall_taxonomy"]:
        raise FlyprofError("NO_ARTIFACT", "neither report.json has a stall taxonomy to compare")
    return Result(data=data)
