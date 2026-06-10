"""One humanize round, Steps 1-2 + the ledger append — the loop orchestrator.

Thin by design (plan §5.2): contains NO optimization logic. It reads the
bundle's ``report.json`` (Step 1, DIAGNOSE), picks the rank-1 recommendation,
asks the ROCmKernelWiki for prior art (Step 2, HYPOTHESIZE), checks the attempt
ledger for known dead ends, and appends a round *skeleton* to ``attempts.jsonl``.
The agent proposes the actual edit; the compile/correctness/bench gates measure
it; ``flyprof diff`` is the Step-6 mechanical check. The skeleton's empty fields
(hypothesis, diff, predicted_delta, verdict, …) are the agent's contract: filled
in that order, prediction before edit, verdict after the gates.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Optional, Union

from .config import RUNS_DIR
from .envelope import FlyprofError, Result
from .wiki import run_wiki

DEFAULT_ATTEMPTS = RUNS_DIR / "flash_attn_humanize" / "attempts.jsonl"

# What the agent does next with the skeleton (plan §2 steps 3-9, one knob per round).
NEXT_STEPS = [
    "fill hypothesis/diff/predicted_delta in the round record BEFORE editing (predict first)",
    "apply ONE edit to the kernel; COMPILE gate: scratch == 0",
    "CORRECTNESS gate: all configs vs torch fp32 SDPA (max_err<1e-2, cosine>0.99) — fail => revert",
    "re-profile, then `flyprof diff --before <baseline> --after <new>` (rank-1 bubble must shrink)",
    "benchmark all configs; append verdict ACCEPT/REVISE/REJECT + reason to attempts.jsonl",
]


def _load_attempts(path: Path) -> list:
    """Tolerant JSONL read: a malformed line is skipped, never fatal."""
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _rank1(report: dict) -> Optional[dict]:
    recs = report.get("recommendations") or []
    return min(recs, key=lambda r: r.get("rank", 99)) if recs else None


def run_optimize(report: dict, bundle: str, attempts_path: Union[str, Path],
                 wiki_fn: Optional[Callable] = None, arch: str = "gfx950",
                 limit: int = 5, wiki_root: Optional[str] = None,
                 timeout: int = 60) -> tuple:
    """One round of Steps 1-2 + ledger append. Returns ``(data, warnings)``."""
    kernel = report.get("kernel", "?")
    if report.get("optimal"):
        raise FlyprofError("NO_RECOMMENDATION",
                           f"{kernel} is already optimal per report.json — stop the loop, escalate")
    rec = _rank1(report)
    if rec is None:
        raise FlyprofError("NO_RECOMMENDATION",
                           f"{kernel}: report.json has no recommendations; nothing to target this round")

    attempts_path = Path(attempts_path)
    prior = _load_attempts(attempts_path)
    round_no = max((int(a.get("round") or 0) for a in prior), default=0) + 1
    same_rule = [a for a in prior if (a.get("target") or {}).get("rule") == rec.get("rule")]

    warnings = []
    for a in same_rule:
        if str(a.get("verdict") or "").upper() == "REJECT":
            warnings.append(f"rule {rec.get('rule')} was REJECTED in round {a.get('round')} "
                            f"({a.get('reason')}) — do not re-propose the same edit")

    kernel_hint = str(report.get("recipe_kernel") or kernel).replace("_", " ")
    wiki_out = (wiki_fn or run_wiki)(rec.get("bubble_class"), kernel=kernel_hint, arch=arch,
                                     limit=limit, wiki_root=wiki_root, timeout=timeout)
    if wiki_out.get("ok"):
        warnings.extend(wiki_out.get("warnings") or [])
        wiki_summary = {
            "query": wiki_out.get("query"),
            "technique": wiki_out.get("technique_hint"),
            "implemented_by_prs": wiki_out.get("implemented_by_prs") or [],
            "pages": [{"page_id": r.get("page_id"), "title": r.get("title")}
                      for r in wiki_out.get("results") or []],
        }
    else:
        err = wiki_out.get("error") or {}
        warnings.append(f"wiki unavailable ({err.get('code')}): falling back to knob-table priors")
        wiki_summary = {"error": err}

    evidence = rec.get("evidence") or {}
    occ = report.get("occupancy") or {}
    attempt = {
        "round": round_no,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "status": "proposed",
        "bundle": bundle,
        "kernel": kernel,
        "bound_type": report.get("bound_type"),
        "target": {
            "rule": rec.get("rule"),
            "bubble_class": rec.get("bubble_class"),
            "knob": rec.get("knob"),
            "confidence": rec.get("confidence"),
            "evidence_source": evidence.get("source"),
            "stall_pct": evidence.get("stall_pct_of_class"),
        },
        "wiki": wiki_summary,
        # the agent's half of the contract — filled before/after the gates:
        "hypothesis": None,
        "diff": None,
        "predicted_delta": None,
        "flyprof_before": {
            "stall_pct": (report.get("totals") or {}).get("stall_pct"),
            "waves_per_cu": occ.get("from_capture_waves_per_cu"),
            "arch_vgpr": occ.get("arch_vgpr"),
        },
        "flyprof_after": None,
        "tflops_before": None,
        "tflops_after": None,
        "verdict": None,
        "reason": None,
    }
    attempts_path.parent.mkdir(parents=True, exist_ok=True)
    with attempts_path.open("a") as f:
        f.write(json.dumps(attempt) + "\n")

    data = {
        "round": round_no,
        "attempts_path": str(attempts_path),
        "target": attempt["target"],
        "wiki": wiki_summary,
        "prior_attempts_on_rule": [{"round": a.get("round"), "verdict": a.get("verdict"),
                                    "reason": a.get("reason")} for a in same_rule],
        "next_steps": NEXT_STEPS,
    }
    return data, warnings


def cmd_optimize(args, cfg) -> Result:
    cfg.require_bundle()
    report = cfg.read_artifact("report.json")
    attempts = Path(getattr(args, "attempts", None) or DEFAULT_ATTEMPTS)
    data, warnings = run_optimize(
        report, bundle=str(cfg.bundle), attempts_path=attempts, arch=cfg.arch(),
        limit=getattr(args, "limit", 5), wiki_root=getattr(args, "wiki_root", None),
        timeout=getattr(args, "timeout", 60))
    return Result(data=data, bundle=str(cfg.bundle), warnings=warnings)
