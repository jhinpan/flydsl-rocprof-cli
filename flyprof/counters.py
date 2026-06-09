"""PMC counter pass -> roofline / occupancy / memory efficiency.

Closes the gap that an ATT job's PMC is unreliable: this runs a *separate*
rocprofv3 counter-collection pass and parses ``*_counter_collection.csv``
directly (rows: Kernel_Name, Counter_Name, Counter_Value, Dispatch_Id), then
derives metrics against the gfx950 theoretical peaks in ``soc.py``.

We report only what the counters actually support. Exact FLOP/s needs the full
per-dtype rocprof-compute counter set; here we compute the *reliable* memory-side
picture (achieved HBM BW vs roof, L2 hit, 32B-partial fraction, LDS bank-conflict
rate, achieved occupancy, instruction mix) and infer a **tri-state bound**:
``memory`` (HBM saturated), ``compute`` (MFMA/VALU saturated), or ``latency``
(neither — occupancy/stall limited, the case the bubble taxonomy explains). The
L2/HBM interpretation follows the harness's ``pmc_l2_analyzer.py``.
"""
from __future__ import annotations

import csv
import glob
import os
import re
import shutil
import subprocess
from collections import defaultdict

from . import manifest
from .envelope import FlyprofError, Result
from .soc import canon_dtype, get_spec

# Curated minimal counter set. rocprofv3 re-runs the whole workload once per pass,
# so every extra block-collision adds a full run — keep this tight. These cover the
# tri-state bound (HBM bytes + mfma mix), L2 behaviour, and LDS conflicts in ~2 passes.
PMC_COUNTERS = [
    "GRBM_GUI_ACTIVE",          # GRBM block: active cycles (timing fallback)
    "SQ_WAVES", "SQ_BUSY_CYCLES", "SQ_INSTS_VALU", "SQ_INSTS_MFMA", "SQ_LDS_BANK_CONFLICT",  # SQ block (<=8/pass)
    "TCC_HIT_sum", "TCC_MISS_sum", "TCC_EA0_RDREQ_sum", "TCC_EA0_RDREQ_32B_sum",  # TCC block (HBM/L2)
]

PMC_YAML = """jobs:
    -
        kernel_include_regex: "{regex}"
        kernel_iteration_range: "{iters}"
        output_file: pmc
        output_directory: {odir}
        output_format: [csv]
        pmc:
{counter_lines}
"""


def capture_pmc(recipe, cfg, kernel_name, shape, timeout) -> str:
    """Run a separate PMC pass; return the directory holding the counter CSV(s)."""
    from .capture import build_env  # reuse env builder (inline env, shape pin, cold cache)

    cfg.check_rocprofv3()
    bundle = cfg.ensure_bundle()
    pmcdir = bundle / "pmc"
    shutil.rmtree(pmcdir, ignore_errors=True)
    pmcdir.mkdir(parents=True, exist_ok=True)
    env, cmd = build_env(recipe, cfg, str(bundle / ".cache_pmc"), shape)

    regex = re.escape(kernel_name) if not re.search(r"[.*+?\[]", kernel_name) else kernel_name
    counter_lines = "\n".join(f"          - {c}" for c in PMC_COUNTERS)
    yml = bundle / "input_pmc.yaml"
    yml.write_text(PMC_YAML.format(regex=regex, iters="[0, [2-4]]", odir=str(pmcdir), counter_lines=counter_lines))
    try:
        p = subprocess.run([cfg.rocprofv3, "-i", str(yml), "--", *cmd],
                           cwd=str(cfg.worktree), env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise FlyprofError("CAPTURE_TIMEOUT", f"PMC pass exceeded {timeout}s")
    (bundle / "pmc.log").write_text((p.stdout or "") + "\n--STDERR--\n" + (p.stderr or ""))
    csvs = glob.glob(str(pmcdir / "**" / "*counter_collection.csv"), recursive=True)
    if not csvs:
        # some rocprofv3 builds name it differently
        csvs = glob.glob(str(pmcdir / "**" / "*.csv"), recursive=True)
    if not csvs:
        raise FlyprofError("NO_ARTIFACT", "PMC pass produced no counter CSV",
                           help="check pmc.log; the counter names may not be supported on this arch")
    return str(pmcdir)


def aggregate(csv_dir: str, kernel: str | None) -> tuple[dict, int]:
    agg: dict[str, float] = defaultdict(float)
    dispatches: set[str] = set()
    for path in glob.glob(os.path.join(csv_dir, "**", "*.csv"), recursive=True):
        try:
            with open(path) as f:
                for row in csv.DictReader(f):
                    kn = row.get("Kernel_Name", "") or row.get("KernelName", "")
                    if kernel and kernel not in kn:
                        continue
                    name = row.get("Counter_Name")
                    val = row.get("Counter_Value")
                    if not name or val in (None, ""):
                        continue
                    try:
                        agg[name] += float(val)
                    except ValueError:
                        continue
                    dispatches.add(row.get("Dispatch_Id", ""))
        except Exception:
            continue
    return dict(agg), len(dispatches)


def roofline(agg: dict, ndisp: int, time_s: float | None, dtype: str, spec) -> dict:
    """Derive metrics from raw PMC.

    RELIABLE (ratios, no normalization needed): L2 hit rate, 32B-partial fraction,
    MFMA instruction fraction, LDS bank-conflict rate. These drive the bound decision.

    ROUGH (needs per-XCD/channel normalization we don't fully reverse-engineer):
    absolute achieved HBM GB/s. TCC_EA0_* is one channel and GRBM_GUI_ACTIVE is
    summed across XCDs, so the absolute number is a *lower bound only* — flagged
    ``calibrated: false`` and never used to decide the bound.
    """
    g = lambda k: agg.get(k, 0.0)  # noqa: E731
    ndisp = max(ndisp, 1)

    # ---- reliable ratios -------------------------------------------------
    ea = g("TCC_EA0_RDREQ_sum")
    ea32 = g("TCC_EA0_RDREQ_32B_sum")
    partial_32b_frac = round(ea32 / ea, 3) if ea else None
    hit, miss = g("TCC_HIT_sum"), g("TCC_MISS_sum")
    l2_hit_rate = round(100 * hit / (hit + miss), 1) if (hit + miss) else None
    bc, idx = g("SQ_LDS_BANK_CONFLICT"), g("SQ_LDS_IDX_ACTIVE")
    lds_conflict_pct = round(100 * bc / idx, 2) if idx else (0.0 if "SQ_LDS_BANK_CONFLICT" in agg and bc == 0 else None)
    valu, mfma = g("SQ_INSTS_VALU"), g("SQ_INSTS_MFMA")
    inst_total = valu + mfma
    mfma_frac = round(mfma / inst_total, 3) if inst_total else None

    # ---- rough absolute BW (lower bound; EA0-only) -----------------------
    hbm_read_bytes_ea0 = (ea - ea32) * 64 + ea32 * 32
    achieved_hbm_gbs_rough = None
    if time_s and time_s > 0:
        per_call_bytes = hbm_read_bytes_ea0 / ndisp
        achieved_hbm_gbs_rough = round(per_call_bytes / time_s / 1e9, 1)

    # ---- bound from RELIABLE signals (refined against bubbles in `report`) -
    if mfma_frac is not None and mfma_frac >= 0.50:
        bound = "compute"
    elif l2_hit_rate is not None and l2_hit_rate < 50:
        bound = "memory"            # high miss rate -> streaming / HBM pressure
    elif mfma_frac == 0.0:
        bound = "latency"           # no MFMA, L2 reusing -> occupancy/stall limited
    else:
        bound = "unknown"

    peaks = spec.peaks(dtype)
    return {
        "dtype": canon_dtype(dtype),
        "dispatches": ndisp,
        "time_s_per_call": round(time_s, 9) if time_s else None,
        "memory": {
            "l2_hit_rate": l2_hit_rate, "partial_32b_frac": partial_32b_frac,
            "ea0_read_bytes": round(hbm_read_bytes_ea0),
            "achieved_hbm_gbs_rough": achieved_hbm_gbs_rough, "peak_hbm_gbs": spec.hbm_gbs,
            "calibrated": False,
            "note": "absolute BW is an EA0-channel lower bound; trust the ratios + bubble taxonomy",
        },
        "lds": {"bank_conflict_pct": lds_conflict_pct},
        "compute": {"mfma_inst_frac": mfma_frac,
                    "inst_counts": {"valu": valu, "mfma": mfma}},
        "roofline": {**peaks, "bound": bound, "bound_basis": "counters(reliable ratios)"},
        "occupancy": {"waves_dispatched": g("SQ_WAVES"),
                      "note": "VGPR-based occupancy is in capture.json (exact); see report"},
        "counters_raw": {k: agg.get(k) for k in PMC_COUNTERS if k in agg},
    }


def cmd_counters(args, cfg) -> Result:
    spec = get_spec(args.arch or cfg.arch())
    time_s = None
    dtype = "bf16"

    # learn kernel / shape / time from a prior capture if present
    cap = None
    try:
        cap = cfg.read_artifact("capture.json")
    except FlyprofError:
        pass
    # filter_name: the discovered JIT name used to filter PMC rows.
    # recipe_name: the user-facing name used to resolve a runnable recipe.
    filter_name = args.kernel or (cap.get("kernel") if cap else None)
    recipe_name = args.kernel or (cap.get("recipe_kernel") if cap else None) or filter_name
    if cap and cap.get("avg_ns"):
        time_s = cap["avg_ns"] * 1e-9
    if not filter_name:
        raise FlyprofError("BAD_ARGS", "no kernel given and no capture.json in the bundle",
                           help="run `flyprof capture <kernel>` first, or pass a kernel name")

    # dtype from shape (last token) or tile.json
    shape = args.shape
    if shape and "," in shape:
        last = shape.split(",")[-1]
        if not last.isdigit():
            dtype = last
    else:
        try:
            t = cfg.read_artifact("tile.json")
            dtype = t.get("shape", {}).get("dtype", dtype)
        except FlyprofError:
            pass

    # find or run the PMC pass
    if cfg.bundle and (cfg.bundle / "pmc").exists():
        pmc_dir = str(cfg.bundle / "pmc")
    else:
        recipe = manifest.get_recipe(recipe_name, cfg)
        pmc_dir = capture_pmc(recipe, cfg, filter_name, shape, args.timeout)

    agg, ndisp = aggregate(pmc_dir, filter_name)
    warnings = []
    if not agg:
        # empty != broken
        raise FlyprofError("NO_ARTIFACT", f"no counter rows matched kernel {filter_name!r}",
                           help="check pmc.log / the Kernel_Name filter")
    zero = [c for c in PMC_COUNTERS if agg.get(c, 0) == 0]
    if zero:
        warnings.append(f"{len(zero)} counters read 0 (may be unsupported on this arch): {zero[:6]}")

    data = {"kernel": filter_name, **roofline(agg, ndisp, time_s, dtype, spec)}
    cfg.write_artifact("counters.json", data)
    return Result(data=data, bundle=str(cfg.bundle), warnings=warnings)
