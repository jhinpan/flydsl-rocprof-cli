"""ROCmKernelWiki retrieval: bubble_class -> prior-art technique + PR lineage.

Step 2 of the humanize loop: the rank-1 ``bubble_class`` from ``report.json``
becomes a wiki query, the wiki answers with technique pages and the
``implemented_by`` PR lineage — the "has a human already done this?" check.

The bubble -> technique seed map mirrors ``skills/references/bubble-knob-table.md``
(and ``flyprof.knobs``), cross-referenced with the wiki technique page ids. We
shell ``scripts/query.py``/``get_page.py`` instead of importing them so a missing
or broken wiki checkout degrades to an ``ok:false`` payload, never an exception.
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Optional

from .envelope import EXIT_INTERNAL, EXIT_TIMEOUT, EXIT_UNAVAILABLE, FlyprofError, Result

DEFAULT_WIKI_ROOT = "/sgl-workspace/ROCmKernelWiki"

# bubble_class -> (query keywords, technique page id). Page ids must exist in the
# wiki (wiki/techniques|patterns); keep in sync with bubble-knob-table.md rows.
BUBBLE_TECHNIQUES: dict[str, dict] = {
    "occupancy":  {"keywords": "occupancy vgpr pressure register budgeting waves per cu",
                   "technique": "technique-vgpr-budgeting"},
    "barrier":    {"keywords": "barrier s_barrier redundant sync wave reduce permlane",
                   "technique": "technique-wave-reduce"},
    "vmcnt":      {"keywords": "vmcnt prefetch double buffer direct-to-lds latency hiding",
                   "technique": "technique-lds-double-buffering"},
    "lgkmcnt":    {"keywords": "lgkmcnt lds write read distance software pipelining",
                   "technique": "technique-mfma-pipelining"},
    "lds":        {"keywords": "lds bank conflict swizzle padding transpose",
                   "technique": "technique-lds-swizzling"},
    "mfma":       {"keywords": "mfma pipeline raw accumulator interleave schedule",
                   "technique": "technique-mfma-pipelining"},
    "vmem_load":  {"keywords": "vectorized 128-bit loads coalescing partial cache line",
                   "technique": "technique-vectorized-loads"},
    "vmem_store": {"keywords": "vectorized 128-bit stores coalescing write combine",
                   "technique": "technique-vectorized-loads"},
}

# compact line: "  [wiki-kernel] kernel-flydsl-flash-attention: Title  (wiki/kernels/x.md)"
_COMPACT_RE = re.compile(r"^\s+\[[\w-]+\]\s+(?P<pid>\S+):\s+(?P<title>.*?)\s+\((?P<path>\S+)\)\s*$")


def _err(code: str, exit_code: int, message: str, help: str) -> dict:
    """An ok:false payload shaped like the envelope's error object (not raised:
    a missing wiki degrades the loop to knob-table priors, it doesn't stop it)."""
    return {"ok": False, "error": {"code": code, "exitCode": exit_code, "message": message, "help": help}}


def _run(cmd: list, timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _parse_compact(stdout: str, limit: int) -> list:
    """Compact query.py lines -> [{page_id,title,score}]. query.py doesn't print
    its numeric score, so score is rank-derived: 1.0 for rank 1, linear to 1/limit."""
    results = []
    for line in stdout.splitlines():
        m = _COMPACT_RE.match(line)
        if m:
            results.append({"page_id": m["pid"], "title": m["title"], "path": m["path"]})
    span = max(limit, len(results), 1)
    for i, r in enumerate(results):
        r["score"] = round(1.0 - i / span, 3)
    return results


def _implemented_by(root: Path, page_id: str, python: str, timeout: int) -> list:
    """``get_page.py <id> --frontmatter-only`` -> implemented_by PR ids ([] on any failure)."""
    try:
        p = _run([python, str(root / "scripts" / "get_page.py"), page_id, "--frontmatter-only"], timeout)
        if p.returncode != 0:
            return []
        import yaml
        fm = yaml.safe_load(p.stdout) or {}
        impl = fm.get("implemented_by") or []
        return [str(x) for x in impl] if isinstance(impl, list) else []
    except Exception:
        return []


def run_wiki(bubble_class: str, kernel: str = "flash attention", arch: str = "gfx950",
             limit: int = 5, wiki_root: Optional[str] = None, timeout: int = 60) -> dict:
    """Map a flyprof bubble_class to a ROCmKernelWiki query and run it.

    Returns ``{ok, query, results:[{page_id,title,path,score}], technique_hint,
    implemented_by_prs, warnings}`` on success, or an ``ok:false`` envelope-style
    payload (WIKI_MISSING/69, WIKI_TIMEOUT/75, WIKI_QUERY_FAILED/70) on failure.
    """
    root = Path(wiki_root or os.environ.get("FLYPROF_WIKI_ROOT", DEFAULT_WIKI_ROOT))
    query_py = root / "scripts" / "query.py"
    if not query_py.is_file():
        return _err("WIKI_MISSING", EXIT_UNAVAILABLE, f"ROCmKernelWiki not found at {root}",
                    "clone github.com/jhinpan/ROCmKernelWiki or set FLYPROF_WIKI_ROOT")

    seed = BUBBLE_TECHNIQUES.get(bubble_class)
    keywords = seed["keywords"] if seed else f"{bubble_class} stall"
    technique = seed["technique"] if seed else None
    query = f"{keywords} {kernel}"
    python = os.environ.get("FLYPROF_PYTHON", "python3")

    try:
        p = _run([python, str(query_py), query, "--architecture", arch, "--limit", str(limit), "--compact"], timeout)
    except subprocess.TimeoutExpired:
        return _err("WIKI_TIMEOUT", EXIT_TIMEOUT, f"query.py exceeded {timeout}s", "retry once with a larger timeout")
    if p.returncode != 0:
        return _err("WIKI_QUERY_FAILED", EXIT_INTERNAL, (p.stderr or p.stdout).strip()[:500],
                    "run query.py by hand against the wiki checkout")

    results = _parse_compact(p.stdout, limit)
    warnings = [] if results else [f"no wiki page matched '{query}'"]
    if technique is None:
        warnings.append(f"bubble_class '{bubble_class}' has no seeded technique; fell back to a literal query")
    return {
        "ok": True,
        "bubble_class": bubble_class,
        "query": query,
        "results": results,
        "technique_hint": technique,
        "implemented_by_prs": _implemented_by(root, technique, python, timeout) if technique else [],
        "warnings": warnings,
    }


def cmd_wiki(args, cfg) -> Result:
    """CLI wrapper: an ok:false payload becomes the envelope error it mirrors."""
    out = run_wiki(args.bubble_class, kernel=args.kernel, arch=cfg.arch(), limit=args.limit,
                   wiki_root=args.wiki_root, timeout=args.timeout)
    if not out.get("ok"):
        e = out["error"]
        raise FlyprofError(e["code"], e["message"], help=e.get("help"))
    warnings = out.pop("warnings", [])
    out.pop("ok", None)
    return Result(data=out, warnings=warnings)
