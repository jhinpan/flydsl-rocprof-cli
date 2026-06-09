"""rocprofv3 ATT capture for one kernel — vendored & de-hardcoded.

Ported from the harness's ``att_capture.py``. The hard-won bits are kept verbatim
in spirit: a fresh (cold) debug cache so DWARF line tables are emitted,
``--stats`` kernel discovery with a noise filter + hint match, automatic
``kernel_iteration_range`` derivation, and empty-shell dispatch cleanup. The
module-level ``ROOT``/``PROF``/``PY`` constants are gone — everything comes from
the injected :class:`~flyprof.config.Config`, so capture works on any worktree and
any kernel name (recipe synthesized on a registry miss).

The ATT job is ATT-only: PMC inside an ATT job is unreliable, so counters are a
separate pass (see ``flyprof counters``).
"""
from __future__ import annotations

import csv
import glob
import json
import os
import re
import shlex
import shutil
import subprocess
import time

from . import doctor, manifest
from ._vendor.hotspot import reg_pressure
from .envelope import FlyprofError, Result
from .rcv import load_code

NOISE = re.compile(
    r"at::native|rocclr|rocprim|Cijk|hipblas|rocblas|elementwise|reduce_kernel|vectorized|"
    r"__amd|memset|memcpy|fill_|cast_|to_copy|triton_|sort|scan|cub::|::detail",
    re.I,
)

ATT_YAML = """GlobalParameters:
  KeepBuildTmp: True
  AsmDebug: True
jobs:
    -
        kernel_include_regex: "{regex}"
        kernel_iteration_range: "{iters}"
        output_file: out
        output_directory: {odir}
        output_format: [json, csv]
        truncate_kernels: true
        sys_trace: false
        advanced_thread_trace: true
        att_target_cu: {tcu}
        att_shader_engine_mask: "0xf"
        att_simd_select: "0xf"
        att_buffer_size: "{buf}"
"""


def split_inline_env(invocation: str, cfg) -> tuple[dict, list]:
    toks = shlex.split(invocation)
    env: dict[str, str] = {}
    i = 0
    while i < len(toks) and re.match(r"^[A-Z_][A-Z0-9_]*=", toks[i]):
        k, v = toks[i].split("=", 1)
        env[k] = v
        i += 1
    cmd = toks[i:]
    if cmd and cmd[0] not in ("python", "python3") and cmd[0].endswith(".py"):
        cmd = [cfg.python] + cmd
    elif cmd and cmd[0] in ("python", "python3"):
        cmd[0] = cfg.python
    # resolve a bare relative .py against the worktree
    for j, tok in enumerate(cmd):
        if tok.endswith(".py") and not tok.startswith("-"):
            if not os.path.exists(os.path.join(cfg.worktree, tok)):
                cand = os.path.join("tests", "kernels", os.path.basename(tok))
                if os.path.exists(os.path.join(cfg.worktree, cand)):
                    cmd[j] = cand
            break
    return env, cmd


def _shape_key(recipe: dict) -> str | None:
    for k in (recipe.get("shape_env") or {}):
        if k.endswith("SHAPES"):
            return k
    return None


def build_env(recipe: dict, cfg, cache_dir: str, shape: str | None) -> tuple[dict, list]:
    inline, cmd = split_inline_env(recipe["invocation"], cfg)
    env = cfg.base_env()
    env.update(inline)
    for k, v in (recipe.get("shape_env") or {}).items():
        if v and re.match(r"^[A-Za-z0-9_,.;:\- ]+$", str(v)):
            env.setdefault(k, str(v))
    if shape:
        key = _shape_key(recipe)
        if key:
            env[key] = shape
    env["ROCDSL_COMPARE_AITER"] = "0"             # FlyDSL kernel only -> clean capture
    env["FLYDSL_RUNTIME_CACHE_DIR"] = cache_dir   # fresh cold cache -> DWARF line tables
    env["FLYDSL_RUNTIME_ENABLE_CACHE"] = "1"
    return env, cmd


def discover_kernel(recipe: dict, env: dict, cmd: list, outdir: str, cfg) -> tuple[dict | None, list, int]:
    disc = os.path.join(outdir, "discover")
    p = subprocess.run(
        [cfg.rocprofv3, "--stats", "--kernel-trace", "-f", "csv", "-o", disc, "--", *cmd],
        cwd=str(cfg.worktree), env=env, capture_output=True, text=True, timeout=900,
    )
    stats = f"{disc}_kernel_stats.csv"
    cands: list[dict] = []
    if os.path.exists(stats):
        with open(stats) as f:
            for row in csv.DictReader(f):
                name = row.get("Name") or row.get("KernelName") or ""
                if not name or NOISE.search(name):
                    continue

                def num(*keys):
                    for k in keys:
                        if row.get(k) not in (None, ""):
                            try:
                                return float(row[k])
                            except ValueError:
                                pass
                    return 0.0

                cands.append({
                    "name": name,
                    "calls": int(num("Calls", "TotalCalls") or 0),
                    "total_ns": num("TotalDurationNs", "TotalDuration", "DurationNs"),
                    "avg_ns": num("AverageNs", "AvgNs", "Average"),
                })
    chosen = None
    confidence = "heuristic"
    hint = (recipe.get("kernel_name_hint") or "").strip()
    hint_tok = re.split(r"[ (]", hint)[0] if hint else ""
    if hint_tok:
        for c in cands:
            if hint_tok in c["name"]:
                chosen, confidence = c, "exact"
                break
    if not chosen:
        for k in recipe.get("kernels", []):
            for c in cands:
                if k.replace("_kernel", "") in c["name"]:
                    chosen, confidence = c, "fuzzy"
                    break
            if chosen:
                break
    if not chosen and cands:
        chosen = sorted(cands, key=lambda c: (c["total_ns"], c["calls"]), reverse=True)[0]
        confidence = "heuristic"
    if chosen:
        chosen["confidence"] = confidence
    return chosen, cands, p.returncode


def auto_iter_range(calls: int) -> str:
    c = max(calls, 1)
    if c >= 14:
        skip, m = 6, 8
    elif c >= 6:
        skip = max(2, c // 2 - 1)
        m = skip + 1
    elif c >= 3:
        skip, m = 1, 2
    else:
        skip, m = 0, max(1, c - 1)
    return f"[{skip}, [{m}-{m}]]"


def inspect_dispatch(d: str) -> tuple[int, int, int]:
    waves = len(glob.glob(os.path.join(d, "se*_sm*_*.json")))
    ins = mapped = 0
    cj = os.path.join(d, "code.json")
    if os.path.exists(cj):
        try:
            code = json.load(open(cj)).get("code") or []
            ins = len(code)
            mapped = sum(1 for r in code if len(r) > 3 and r[3])
        except Exception:
            pass
    return waves, ins, mapped


def _shrink_shape(shape: str | None) -> str | None:
    if not shape:
        return None
    parts = shape.split(",")
    out = []
    shrunk = False
    for p in parts:
        try:
            n = int(p)
            out.append(str(max(256, n // 8)) if not shrunk else str(n))
            shrunk = True
        except ValueError:
            out.append(p)
    return ",".join(out)


def capture_one(recipe, cfg, kernel_regex, tag, shape, iters, timeout) -> dict:
    bundle = cfg.ensure_bundle()
    attdir = bundle / "att" / tag
    shutil.rmtree(attdir, ignore_errors=True)
    attdir.mkdir(parents=True, exist_ok=True)
    cache = str(bundle / f".cache_{tag}")
    shutil.rmtree(cache, ignore_errors=True)
    env, cmd = build_env(recipe, cfg, cache, shape)

    yml = bundle / f"input_trace_{tag}.yaml"
    yml.write_text(ATT_YAML.format(regex=kernel_regex, iters=iters, odir=str(attdir),
                                   tcu=1, buf="0x6000000"))
    t0 = time.time()
    try:
        cap = subprocess.run([cfg.rocprofv3, "-i", str(yml), "--", *cmd],
                             cwd=str(cfg.worktree), env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise FlyprofError("CAPTURE_TIMEOUT", f"ATT capture for tag={tag} exceeded {timeout}s")
    (bundle / f"capture_{tag}.log").write_text((cap.stdout or "") + "\n--STDERR--\n" + (cap.stderr or ""))

    # inspect + delete empty shells
    disps = sorted(glob.glob(str(attdir / "ui_output_agent_*")))
    kept, best, inv = None, (-1, -1, -1), []
    for d in disps:
        w, i, m = inspect_dispatch(d)
        inv.append({"dir": os.path.basename(d), "waves": w, "ins": i, "mapped": m})
        if (w, i, m) > best and w > 0 and i > 0:
            best, kept = (w, i, m), d
    for d in disps:
        if d != kept:
            w, i, _ = inspect_dispatch(d)
            if w == 0 or i == 0:
                shutil.rmtree(d, ignore_errors=True)

    res = {"tag": tag, "shape": shape, "iter_range": iters, "capture_rc": cap.returncode,
           "capture_s": round(time.time() - t0, 1), "dispatch_inventory": inv}
    if not kept:
        res["error"] = "empty_att"
        return res
    w, i, m = best
    instrs = load_code(kept)
    rp = reg_pressure([x.asm for x in instrs])
    stall_by_cls: dict[str, int] = {}
    for x in instrs:
        if x.stall > 0:
            stall_by_cls[x.cls] = stall_by_cls.get(x.cls, 0) + x.stall
    top_stall = max(stall_by_cls, key=stall_by_cls.get) if stall_by_cls else None
    res.update({
        "kept_dir": os.path.relpath(kept, bundle),
        "waves": w, "ins": i, "mapped": m, "mapped_pct": round(100 * m / max(i, 1), 1),
        "arch_vgpr": rp["arch_vgpr"], "accum_vgpr": rp["accum_vgpr"],
        "occupancy_waves_per_cu": rp["occupancy_waves_per_cu"],
        "top_stall_type": top_stall,
    })
    # copy compute-viewer support files
    for f in ("out_results.json", "out_agent_info.csv"):
        src = attdir / f
        if src.exists():
            shutil.copy(src, bundle / f"{tag}_{f}")
    return res


def cmd_capture(args, cfg) -> Result:
    cfg.check_rocprofv3()
    cfg.check_build_tree()
    recipe = manifest.get_recipe(args.kernel, cfg)
    cfg.ensure_bundle(args.kernel)

    tags = ["small", "big"] if args.tag == "both" else [args.tag]
    # discovery once (using the big/default env)
    disc_dir = str(cfg.bundle)
    env, cmd = build_env(recipe, cfg, str(cfg.bundle / ".cache_discover"), args.shape)
    chosen, cands, drc = discover_kernel(recipe, env, cmd, disc_dir, cfg)
    if not chosen:
        raise FlyprofError("NO_KERNEL_DISCOVERED",
                           f"rocprofv3 --stats found no FlyDSL kernel for {args.kernel}",
                           discover_rc=drc, candidates=[c["name"] for c in cands[:10]])
    name = chosen["name"]
    regex = re.escape(name) if not re.search(r"[.*+?\[]", name) else name
    iters = args.iter_range or auto_iter_range(chosen["calls"])

    per_tag = {}
    warnings = []
    for tag in tags:
        shape = _shrink_shape(args.shape) if tag == "small" else args.shape
        r = capture_one(recipe, cfg, regex, tag, shape, iters, args.timeout)
        per_tag[tag] = r
        if r.get("error") == "empty_att":
            warnings.append(f"tag={tag}: all ATT dispatches were empty shells")

    if all(r.get("error") for r in per_tag.values()):
        raise FlyprofError("EMPTY_ATT", "no valid ATT dispatch on any tag",
                           dispatch_inventory={t: r["dispatch_inventory"] for t, r in per_tag.items()})

    data = {
        "kernel": name,                       # discovered JIT kernel name (for PMC/ATT filtering)
        "recipe_kernel": args.kernel,          # user-facing name (for recipe resolution / re-runs)
        "match_confidence": chosen.get("confidence"),
        "recipe_source": recipe.get("recipe_source"),
        "calls": chosen["calls"], "avg_ns": chosen.get("avg_ns"), "total_ns": chosen.get("total_ns"),
        "iter_range": iters,
        "tags": {t: {k: v for k, v in r.items() if k != "dispatch_inventory"} for t, r in per_tag.items()},
        "dispatch_inventory": {t: r["dispatch_inventory"] for t, r in per_tag.items()},
        "candidates": [c["name"] for c in cands[:8]],
        "provenance": {"rocprofv3": doctor._rocprofv3(cfg).get("version"), "arch": cfg.arch()},
    }
    cfg.write_artifact("capture.json", data)

    if getattr(args, "with_pmc", False):
        from . import counters
        try:
            pmc = counters.capture_pmc(recipe, cfg, name, args.shape, getattr(args, "timeout", 900))
            data["pmc_csv"] = pmc
        except FlyprofError as e:
            warnings.append(f"pmc pass failed: [{e.code}] {e.message}")

    return Result(data=data, bundle=str(cfg.bundle), warnings=warnings)
