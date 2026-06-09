"""Argparse dispatch + the single point where Results/errors become envelopes.

A command handler is ``def handle(args, cfg) -> Result`` and either returns a
:class:`~flyprof.envelope.Result` or raises :class:`~flyprof.envelope.FlyprofError`.
``main`` wraps the return value into the ok/error envelope, prints it, and exits
with the bound code. Handlers are imported lazily so GPU-free commands never pull
in capture machinery.
"""
from __future__ import annotations

import argparse
import importlib
import sys

from . import __version__
from .config import Config
from .envelope import (
    EXIT_INTERRUPT,
    FlyprofError,
    Result,
    emit,
    error_envelope,
    ok_envelope,
)

# command -> "module:function" (lazy import). Modules are added as they are built.
DISPATCH = {
    "list": "flyprof.manifest:cmd_list",
    "doctor": "flyprof.doctor:cmd_doctor",
    "tile": "flyprof.tile:cmd_tile",
    "capture": "flyprof.capture:cmd_capture",
    "counters": "flyprof.counters:cmd_counters",
    "bubbles": "flyprof.rcv:cmd_bubbles",
    "map": "flyprof.rcv:cmd_map",
    "report": "flyprof.report:cmd_report",
    "bundle": "flyprof.bundle:cmd_bundle",
    "run": "flyprof.run:cmd_run",
}


def _add_universal(p: argparse.ArgumentParser) -> None:
    p.add_argument("--format", "-f", choices=["json", "table"], default=None,
                   help="output format (default: json when piped, table on a TTY)")
    p.add_argument("--bundle", default=None, help="bundle directory: the cross-command state handle")
    p.add_argument("--worktree", default=None, help="FlyDSL checkout (default: $FLYPROF_WORKTREE or /sgl-workspace/FlyDSL-lab)")
    p.add_argument("--python", default=None, help="python executable used to run kernels under rocprofv3")
    p.add_argument("--gpu", type=int, default=0, help="GPU/device index (HIP_VISIBLE_DEVICES)")
    p.add_argument("--arch", default=None, help="override GPU arch (default: detect via rocminfo)")
    p.add_argument("--examples-dir", default=None, help="artifact-hub examples/ dir for `bundle`")
    p.add_argument("--quiet", action="store_true", help="suppress sub-tool stderr chatter")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="flyprof",
        description="Agent-drivable rocprofv3 kernel profiling for FlyDSL on AMD CDNA. "
                    "Workflow: list -> doctor -> tile -> capture -> counters -> bubbles -> map -> report -> bundle. "
                    "Every command prints one JSON envelope; state flows through --bundle.",
    )
    ap.add_argument("--version", action="version", version=f"flyprof {__version__}")
    sub = ap.add_subparsers(dest="command", metavar="<command>")

    def add(name, help):
        sp = sub.add_parser(name, help=help)
        _add_universal(sp)
        return sp

    add("list", "enumerate profilable kernels (registry ∪ synthesizable)").add_argument(
        "--refresh", action="store_true", help="re-scan the worktree, ignore the cache")

    add("doctor", "preflight: rocprofv3 / GPU / build-tree / soc spec")

    sp = add("tile", "GPU-free tile-size advisor (LDS/VGPR/occupancy/roofline)")
    sp.add_argument("kernel", nargs="?", help="kernel name (for op category / defaults)")
    sp.add_argument("--shape", help="problem shape, e.g. M,N,K or M,N")
    sp.add_argument("--dtype", default="bf16", help="element dtype (bf16/fp16/fp8/fp4/fp32)")
    sp.add_argument("--out-dtype", default=None, help="output/store dtype (default: same as --dtype)")
    sp.add_argument("--bm", type=int, default=None)
    sp.add_argument("--bn", type=int, default=None)
    sp.add_argument("--bk", type=int, default=None)
    sp.add_argument("--buffers", type=int, default=2, help="LDS pipeline buffers (double-buffer=2)")

    sp = add("capture", "rocprofv3 ATT (+optional PMC) capture for one kernel")
    sp.add_argument("kernel", help="kernel name (resolved via `list`/synthesize_recipe)")
    sp.add_argument("--shape", help="override shape; pins ROCDSL_*_SHAPES where known")
    sp.add_argument("--invocation", default=None,
                    help="override the recipe's run command (e.g. to pin a shape via the test's own "
                         "args: 'python tests/kernels/test_flash_attn_fwd.py --batch 1 --seq_len 2048 ...')")
    sp.add_argument("--tag", choices=["small", "big", "both"], default="big")
    sp.add_argument("--with-pmc", action="store_true", help="also run a separate PMC counter pass")
    sp.add_argument("--iter-range", default=None, help="rocprofv3 kernel_iteration_range (auto if omitted)")
    sp.add_argument("--timeout", type=int, default=1200)

    sp = add("counters", "PMC pass -> roofline + occupancy + L2/HBM")
    sp.add_argument("kernel", nargs="?", help="kernel (omit to reuse the bundle's capture.json)")
    sp.add_argument("--shape", default=None)
    sp.add_argument("--timeout", type=int, default=900)

    sp = add("bubbles", "headless stall/bubble taxonomy from an ATT bundle")
    sp.add_argument("--tag", choices=["small", "big"], default="big")
    sp.add_argument("--detail", type=int, default=None, metavar="CODE_LINE",
                    help="full per-instruction record for one hotspot's code line")

    sp = add("map", "ISA <-> Python source mapping for an ATT bundle")
    sp.add_argument("--tag", choices=["small", "big"], default="big")
    sp.add_argument("--code-line", type=int, default=None)
    sp.add_argument("--source-line", default=None, metavar="FILE:LINE")

    add("report", "synthesize the optimization report (bubbles -> FlyDSL knobs)")

    sp = add("bundle", "assemble the canonical examples/<kernel>/ artifact")
    sp.add_argument("--no-clobber", action="store_true")

    sp = add("run", "full pipeline on one kernel (doctor->...->report->bundle)")
    sp.add_argument("kernel", help="kernel name")
    sp.add_argument("--shape", default=None)
    sp.add_argument("--invocation", default=None, help="override the recipe's run command (see `capture --invocation`)")
    sp.add_argument("--tag", choices=["small", "big", "both"], default="both")
    sp.add_argument("--skip-bundle", action="store_true", help="stop after report; don't write to examples/")
    return ap


def _resolve(command: str):
    mod_name, func_name = DISPATCH[command].split(":")
    try:
        mod = importlib.import_module(mod_name)
        return getattr(mod, func_name)
    except (ImportError, AttributeError) as e:
        raise FlyprofError("NOT_IMPLEMENTED", f"{command} is not available: {e}")


def main(argv=None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    if not args.command:
        ap.print_help()
        return 0

    cfg = Config.from_args(args)
    fmt = cfg.fmt
    try:
        handler = _resolve(args.command)
        result = handler(args, cfg)
        if not isinstance(result, Result):  # tolerate a bare dict
            result = Result(data=result)
        if result.bundle is None and cfg.bundle is not None:
            result.bundle = str(cfg.bundle)
        envelope = ok_envelope(args.command, result)
    except FlyprofError as e:
        envelope = error_envelope(args.command, e)
    except KeyboardInterrupt:
        return EXIT_INTERRUPT
    except Exception as e:  # never leak a traceback as the contract
        import traceback
        envelope = error_envelope(
            args.command,
            FlyprofError("INTERNAL", f"{type(e).__name__}: {e}", trace=traceback.format_exc().splitlines()[-6:]),
        )
    return emit(envelope, fmt)


if __name__ == "__main__":
    sys.exit(main())
