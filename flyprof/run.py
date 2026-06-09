"""The full-pipeline sequencer.

`run` is a thin convenience over the primitives — it calls them in order, sharing
one bundle directory, and stops on the first hard error (propagating its code so
the exit status reflects the failure class). The intelligence lives in the
primitives + the companion skill, not here.

    doctor -> tile -> capture(--with-pmc) -> counters -> bubbles -> map -> report -> bundle
"""
from __future__ import annotations

from types import SimpleNamespace as NS

from . import bundle as bundle_mod
from . import capture as capture_mod
from . import counters as counters_mod
from . import doctor as doctor_mod
from . import rcv
from . import report as report_mod
from . import tile as tile_mod
from .envelope import FlyprofError, Result


def _dtype_of(shape: str | None) -> str:
    if shape and "," in shape:
        last = shape.split(",")[-1]
        if not last.isdigit():
            return last
    return "bf16"


def cmd_run(args, cfg) -> Result:
    cfg.ensure_bundle(args.kernel)
    steps: dict[str, str] = {}
    warnings: list[str] = []
    tag = args.tag

    # 1. doctor (hard gate)
    dr = doctor_mod.cmd_doctor(NS(arch=args.arch), cfg)
    steps["doctor"] = dr.data["verdict"]
    if dr.data["blockers"]:
        code = dr.data["blockers"][0]
        raise FlyprofError(code, f"doctor not ready: {dr.data['blockers']}")

    def record_partial():
        cfg.write_artifact("run.json", {"kernel": args.kernel, "steps": steps, "warnings": warnings})

    try:
        # 2. tile (GPU-free, also pins shape/dtype into the bundle)
        tr = tile_mod.cmd_tile(NS(kernel=args.kernel, shape=args.shape, dtype=_dtype_of(args.shape),
                                  out_dtype=None, bm=None, bn=None, bk=None, buffers=2, arch=args.arch), cfg)
        steps["tile"] = tr.data.get("recommended", {}).get("regime", "ok")

        # 3. capture (ATT, both tags by default, + PMC pass)
        cap = capture_mod.cmd_capture(NS(kernel=args.kernel, shape=args.shape, tag=tag,
                                         with_pmc=True, iter_range=None, timeout=1200,
                                         invocation=getattr(args, "invocation", None)), cfg)
        steps["capture"] = cap.data["kernel"]
        warnings += cap.warnings

        # 4. counters (parses the PMC pass capture just ran)
        try:
            co = counters_mod.cmd_counters(NS(kernel=None, shape=args.shape, timeout=900, arch=args.arch), cfg)
            steps["counters"] = co.data.get("roofline", {}).get("bound", "ok")
            warnings += co.warnings
        except FlyprofError as e:
            steps["counters"] = f"skipped ({e.code})"
            warnings.append(f"counters: [{e.code}] {e.message}")

        # 5. bubbles + 6. map (headless)
        atag = "big" if tag in ("big", "both") else "small"
        bu = rcv.cmd_bubbles(NS(tag=atag, detail=None), cfg)
        steps["bubbles"] = f"{bu.data['totals']['stall_pct']}% stall"
        mp = rcv.cmd_map(NS(tag=atag, code_line=None, source_line=None), cfg)
        steps["map"] = f"{mp.data['src_mapped_pct']}% mapped"

        # 7. report
        rp = report_mod.cmd_report(NS(), cfg)
        steps["report"] = rp.data["headline"]
        warnings += rp.warnings

        # 8. bundle (optional)
        examples = None
        if not getattr(args, "skip_bundle", False):
            try:
                bd = bundle_mod.cmd_bundle(NS(no_clobber=False), cfg)
                steps["bundle"] = bd.data["examples_dir"]
                examples = bd.data["examples_dir"]
            except FlyprofError as e:
                steps["bundle"] = f"skipped ({e.code})"
                warnings.append(f"bundle: [{e.code}] {e.message}")
    except FlyprofError:
        record_partial()
        raise

    record_partial()
    rep = report_mod.build_report(cfg, atag)
    data = {
        "kernel": cap.data["kernel"],
        "bound_type": rep["bound_type"],
        "headline": rep["headline"],
        "optimal": rep["optimal"],
        "recommendations": rep["recommendations"],
        "steps": steps,
        "artifacts": {
            "report_md": str(cfg.bundle / "REPORT.md"),
            "report_json": str(cfg.bundle / "report.json"),
            "examples_dir": examples,
        },
    }
    return Result(data=data, bundle=str(cfg.bundle), warnings=warnings)
