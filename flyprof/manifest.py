"""Kernel discovery + recipe synthesis.

A *recipe* tells ``capture`` how to run a kernel under rocprofv3: which test
script, the invocation (with inline ``ROCDSL_*`` shape env), the kernel-function
name hints, and the op category. Two sources:

  * a curated **registry** (the hub's ``recipes.json``, if present) — high-confidence
    invocations with verified shapes;
  * **synthesis** — for any kernel not in the registry, derive a runnable recipe
    from the worktree (``tests/kernels/test_<k>.py`` + ``kernels/<k>.py``), so the
    CLI works on *any* kernel name without a pre-written entry.

``list`` is the discoverability entrypoint: an agent calls it first instead of
memorizing kernel names.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from .envelope import FlyprofError, Result

DEFAULT_RECIPES = os.environ.get(
    "FLYPROF_RECIPES", "/sgl-workspace/flydsl-kernel-profiling/tools/data/recipes.json"
)

_CATEGORY_RULES = [
    (r"moe", "moe"),
    (r"gemm|mma|matmul", "gemm"),
    (r"mla|pa_|paged|flash|attn|attention", "attention"),
    (r"softmax|layernorm|rmsnorm|norm|reduce|topk", "reduction"),
    (r"rope|quant|cast|elementwise|vec_add|copy", "elementwise"),
]


def _short_name(test_file: str) -> str:
    stem = Path(test_file).stem
    return stem[5:] if stem.startswith("test_") else stem


def op_category(name: str) -> str:
    low = name.lower()
    for pat, cat in _CATEGORY_RULES:
        if re.search(pat, low):
            return cat
    return "compute"


def load_registry(cfg) -> dict[str, dict]:
    """Map short kernel name -> recipe, from the curated registry if available."""
    path = Path(getattr(cfg, "recipes_path", None) or DEFAULT_RECIPES)
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    try:
        recipes = json.loads(path.read_text())
    except Exception:
        return out
    for rec in recipes:
        name = _short_name(rec.get("test_file", ""))
        if name:
            rec.setdefault("recipe_source", "registry")
            out[name] = rec
    return out


def _infer_shape_env(test_file: Path) -> list[str]:
    if not test_file.exists():
        return []
    try:
        text = test_file.read_text()
    except Exception:
        return []
    return sorted(set(re.findall(r"ROCDSL_[A-Z0-9_]*SHAPES?", text)))


def synthesize_recipe(kernel: str, cfg) -> dict:
    """Build a runnable recipe for a kernel with no registry entry."""
    wt = cfg.worktree
    test_candidates = [
        wt / "tests" / "kernels" / f"test_{kernel}.py",
        wt / "tests" / "kernels" / f"{kernel}.py",
    ]
    test_file = next((p for p in test_candidates if p.exists()), None)
    if test_file is None:
        raise FlyprofError(
            "KERNEL_NOT_FOUND",
            f"no test harness for kernel {kernel!r} under {wt}/tests/kernels",
            help="run `flyprof list` to see profilable kernels",
        )
    rel = test_file.relative_to(wt)
    shape_env = _infer_shape_env(test_file)
    return {
        "test_file": str(test_file),
        "kernels": [f"{kernel}_kernel", kernel],   # hints; --stats discovery confirms
        "kernel_name_hint": kernel,
        "arch": cfg.arch(),
        "op_category": op_category(kernel),
        "invocation": f"python {rel}",
        "shape_env": {k: "" for k in shape_env},
        "recipe_source": "synthesized",
        "confidence": "heuristic",
    }


def get_recipe(kernel: str, cfg) -> dict:
    """Registry hit (by short name or kernel-fn) else synthesize."""
    reg = load_registry(cfg)
    if kernel in reg:
        return reg[kernel]
    # match a kernel-function name (e.g. "softmax_kernel" -> softmax recipe)
    for name, rec in reg.items():
        if kernel in (rec.get("kernels") or []) or kernel == rec.get("kernel_name_hint"):
            return rec
    return synthesize_recipe(kernel, cfg)


def _scan_worktree(cfg) -> dict[str, dict]:
    """Kernels discoverable from the worktree that may lack a curated recipe."""
    wt = cfg.worktree
    found: dict[str, dict] = {}
    tdir = wt / "tests" / "kernels"
    if tdir.exists():
        for p in sorted(tdir.glob("test_*.py")):
            name = _short_name(str(p))
            found[name] = {"test_file": str(p), "kernel_src": None}
    kdir = wt / "kernels"
    if kdir.exists():
        for p in sorted(kdir.glob("*.py")):
            stem = p.stem
            name = stem[:-7] if stem.endswith("_kernel") else stem
            if name in found:
                found[name]["kernel_src"] = str(p)
    return found


def cmd_list(args, cfg) -> Result:
    reg = load_registry(cfg)
    scanned = _scan_worktree(cfg)
    names = sorted(set(reg) | set(scanned))
    kernels = []
    for name in names:
        rec = reg.get(name)
        if rec:
            kernels.append({
                "kernel": name,
                "test_file": rec.get("test_file"),
                "kernel_fns": rec.get("kernels", []),
                "op_category": rec.get("op_category", op_category(name)),
                "shape_env": list((rec.get("shape_env") or {}).keys()),
                "has_recipe": True,
                "recipe_source": "registry",
                "confidence": rec.get("confidence", "high"),
                "needs_multi_gpu": rec.get("needs_multi_gpu", False),
                "requires": ["rocprofv3", "gpu", "build-tree"],
                "args": [{"name": "--shape", "type": "str", "required": False,
                          "help": "M,N[,K],dtype; pins the kernel's ROCDSL_*_SHAPES"}],
            })
        else:
            sc = scanned[name]
            kernels.append({
                "kernel": name,
                "test_file": sc.get("test_file"),
                "kernel_fns": [],
                "op_category": op_category(name),
                "shape_env": _infer_shape_env(Path(sc["test_file"])) if sc.get("test_file") else [],
                "has_recipe": False,
                "recipe_source": "synthesizable",
                "confidence": "heuristic",
                "needs_multi_gpu": False,
                "requires": ["rocprofv3", "gpu", "build-tree"],
                "args": [{"name": "--shape", "type": "str", "required": False, "help": "M,N[,K],dtype"}],
            })
    data = {
        "worktree": str(cfg.worktree),
        "registry": str(Path(DEFAULT_RECIPES)) if Path(DEFAULT_RECIPES).exists() else None,
        "count": len(kernels),
        "kernels": kernels,
    }
    return Result(data=data)
