"""Preflight: is this box able to capture? A pure status report.

``doctor`` never fails the process — it returns ``ok:true`` with a ``verdict``
("ready" / "not-ready") and per-check detail, so an agent reads structure rather
than catching an exception. The ``capture``/``counters`` commands enforce the
hard gates (and return UNAVAILABLE/CONFIG errors); ``doctor`` just tells you in
advance what would block.
"""
from __future__ import annotations

import re
import shutil
import subprocess

from .envelope import Result
from .soc import get_spec


def _rocprofv3(cfg) -> dict:
    path = shutil.which(cfg.rocprofv3)
    if not path:
        return {"present": False}
    ver = None
    try:
        out = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=20).stdout
        m = re.search(r"version:\s*([\d.]+)", out)
        ver = m.group(1) if m else None
    except Exception:
        pass
    return {"present": True, "path": path, "version": ver}


def _gpu(cfg) -> dict:
    exe = shutil.which("rocminfo")
    if not exe:
        return {"present": False}
    try:
        out = subprocess.run([exe], capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return {"present": False}
    arch = None
    name = None
    m = re.search(r"\b(gfx\d+[a-z]*)\b", out)
    if m:
        arch = m.group(1)
    # rocminfo lists the CPU agent first; pick the GPU's Marketing Name, not the CPU's.
    names = re.findall(r"Marketing Name:\s*(.+)", out)
    names = [n.strip() for n in names]
    gpu_names = [n for n in names if re.search(r"instinct|radeon|\bMI\d", n, re.I)]
    if gpu_names:
        name = gpu_names[0]
    elif names:
        name = names[-1]
    return {"present": bool(arch), "arch": arch, "name": name}


def _build_tree(cfg) -> dict:
    bt = cfg.build_tree
    flydsl_pkg = bt / "flydsl"
    return {"path": str(bt), "exists": bt.exists(), "flydsl_pkg": flydsl_pkg.exists()}


def cmd_doctor(args, cfg) -> Result:
    rp = _rocprofv3(cfg)
    gpu = _gpu(cfg)
    bt = _build_tree(cfg)
    arch = gpu.get("arch") or cfg.arch()
    spec = get_spec(arch)

    blockers = []
    if not rp["present"]:
        blockers.append("ROCPROFV3_MISSING")
    if not gpu["present"]:
        blockers.append("NO_GPU")
    if not bt["exists"]:
        blockers.append("BUILD_TREE_MISSING")

    data = {
        "rocprofv3": rp,
        "gpu": gpu,
        "build_tree": bt,
        "soc_spec": {"loaded": True, "arch": spec.arch, "name": spec.name,
                     "cu": spec.cu, "lds_bytes_per_cu": spec.lds_bytes_per_cu,
                     "clock_mhz": spec.clock_mhz, "hbm_gbs": spec.hbm_gbs,
                     "waves_per_cu": spec.waves_per_cu},
        "worktree": str(cfg.worktree),
        "python": cfg.python,
        "gpu_index": cfg.gpu,
        "verdict": "ready" if not blockers else "not-ready",
        "blockers": blockers,
    }
    warnings = []
    if gpu.get("arch") and gpu["arch"] != spec.arch:
        warnings.append(f"detected {gpu['arch']} but using {spec.arch} spec table")
    if not bt.get("flydsl_pkg"):
        warnings.append("build tree present but flydsl package not found under it")
    return Result(data=data, warnings=warnings)
