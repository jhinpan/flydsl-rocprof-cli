"""The output contract.

Every ``flyprof`` subcommand prints exactly one envelope on stdout and exits
with a code that is a *function of the envelope*, so an agent never has to parse
prose. Two shapes:

    {"ok": true,  "command": "...", "bundle": "...", "data": {...},
     "warnings": [...], "truncated": {...}?}
    {"ok": false, "command": "...",
     "error": {"code": "...", "exitCode": N, "message": "...", "help": "...", ...}}

Design rules (mirrored from opencli's agent-CLI discipline):
  * An *empty result is a result, not a crash* — it is ``ok:false`` with an
    EMPTY-class code and exit 66, never a stack trace. This guards the whole
    "false regression" / "dead-code looks broken" failure class.
  * Each error ``code`` hard-binds one ``exitCode`` (sysexits.h conventions) so
    the triage skill can branch on the number without reading the message.
  * ``help`` always tells the caller what to do next.
"""
from __future__ import annotations

import dataclasses
import json
import sys
from typing import Any, Optional

# --- exit codes (subset of sysexits.h) --------------------------------------
EXIT_OK = 0
EXIT_EMPTY = 66        # EX_NOINPUT: nothing matched / empty result — a finding, not a bug
EXIT_UNAVAILABLE = 69  # EX_UNAVAILABLE: required tool/GPU absent — STOP, do not retry
EXIT_INTERNAL = 70     # EX_SOFTWARE: an unexpected internal error
EXIT_TIMEOUT = 75      # EX_TEMPFAIL: transient (capture timed out) — retry is sane
EXIT_CONFIG = 78       # EX_CONFIG: bad config / missing build tree / missing prerequisite
EXIT_INTERRUPT = 130   # SIGINT

# error code -> (exitCode, default help). The single source of truth.
ERROR_CODES: dict[str, tuple[int, str]] = {
    # EMPTY class — a valid answer that happens to be "nothing"
    "KERNEL_NOT_FOUND":      (EXIT_EMPTY, "run `flyprof list` to see profilable kernels"),
    "NO_KERNEL_DISCOVERED":  (EXIT_EMPTY, "rocprofv3 --stats found no FlyDSL kernel; check the test runs at all"),
    "DISPATCH_OUT_OF_RANGE": (EXIT_EMPTY, "use one of the dispatch ids in error.available"),
    "EMPTY_ATT":             (EXIT_EMPTY, "all ATT dispatches were empty shells; enlarge the shape or adjust --iter-range"),
    "NO_ARTIFACT":           (EXIT_EMPTY, "no matching data in the bundle"),
    "NO_RECOMMENDATION":     (EXIT_EMPTY, "no bubble crossed threshold — the kernel may already be at the roofline"),
    # UNAVAILABLE class — STOP, do not retry
    "ROCPROFV3_MISSING":     (EXIT_UNAVAILABLE, "install ROCm / put rocprofv3 on PATH; do NOT retry"),
    "NO_GPU":                (EXIT_UNAVAILABLE, "no GPU visible (rocminfo/amd-smi); do NOT retry"),
    # CONFIG class — fix inputs / run a prerequisite
    "BUILD_TREE_MISSING":    (EXIT_CONFIG, "point --worktree at a built FlyDSL checkout (needs build-fly/python_packages)"),
    "BAD_ARGS":              (EXIT_CONFIG, "check the argument format (e.g. --shape M,N,dtype)"),
    "MISSING_PREREQ":        (EXIT_CONFIG, "a prerequisite command has not been run for this bundle"),
    # TEMPFAIL — retry is sane
    "CAPTURE_TIMEOUT":       (EXIT_TIMEOUT, "retry once with a larger --timeout or a smaller shape"),
    # INTERNAL
    "NOT_IMPLEMENTED":       (EXIT_INTERNAL, "this subcommand is not built yet"),
    "INTERNAL":              (EXIT_INTERNAL, "unexpected internal error; see error.message"),
}


class FlyprofError(Exception):
    """Raised by a command handler to produce an ``ok:false`` envelope.

    ``code`` must be a key of :data:`ERROR_CODES`. ``extra`` keys are merged into
    the ``error`` object (e.g. ``available=[0,1,2]`` for DISPATCH_OUT_OF_RANGE).
    """

    def __init__(self, code: str, message: str, help: Optional[str] = None, **extra: Any) -> None:
        super().__init__(message)
        if code not in ERROR_CODES:
            # never silently mis-bind an exit code
            extra["unmapped_code"] = code
            code = "INTERNAL"
        self.code = code
        self.message = message
        self.help = help if help is not None else ERROR_CODES[code][1]
        self.extra = extra

    @property
    def exit_code(self) -> int:
        return ERROR_CODES[self.code][0]


@dataclasses.dataclass
class Result:
    """What a command handler returns on success. ``cli`` wraps it into an envelope."""

    data: dict
    bundle: Optional[str] = None
    warnings: list = dataclasses.field(default_factory=list)
    truncated: Optional[dict] = None


def ok_envelope(command: str, result: Result) -> dict:
    env: dict[str, Any] = {"ok": True, "command": command}
    if result.bundle is not None:
        env["bundle"] = result.bundle
    env["data"] = result.data
    env["warnings"] = result.warnings
    if result.truncated:
        env["truncated"] = result.truncated
    return env


def error_envelope(command: str, err: FlyprofError) -> dict:
    error: dict[str, Any] = {
        "code": err.code,
        "exitCode": err.exit_code,
        "message": err.message,
        "help": err.help,
    }
    error.update(err.extra)
    return {"ok": False, "command": command, "error": error}


def exit_code_of(envelope: dict) -> int:
    if envelope.get("ok"):
        return EXIT_OK
    return int(envelope.get("error", {}).get("exitCode", EXIT_INTERNAL))


# --- rendering --------------------------------------------------------------
def render(envelope: dict, fmt: str) -> str:
    if fmt == "json":
        return json.dumps(envelope, indent=2, default=str)
    return _render_table(envelope)


def _render_table(env: dict) -> str:
    cmd = env.get("command", "?")
    if not env.get("ok"):
        e = env.get("error", {})
        lines = [f"✗ {cmd}: [{e.get('code')}] {e.get('message')}  (exit {e.get('exitCode')})"]
        if e.get("help"):
            lines.append(f"  → {e['help']}")
        for k, v in e.items():
            if k not in ("code", "exitCode", "message", "help"):
                lines.append(f"  {k}: {v}")
        return "\n".join(lines)
    lines = [f"✓ {cmd}"]
    if env.get("bundle"):
        lines.append(f"  bundle: {env['bundle']}")
    lines.extend(_flatten(env.get("data", {}), indent="  "))
    for w in env.get("warnings") or []:
        lines.append(f"  ⚠ {w}")
    if env.get("truncated"):
        lines.append(f"  truncated: {env['truncated']}")
    return "\n".join(lines)


def _flatten(obj: Any, indent: str, depth: int = 0) -> list[str]:
    out: list[str] = []
    if depth > 3:
        out.append(f"{indent}{obj}")
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (dict, list)) and v:
                out.append(f"{indent}{k}:")
                out.extend(_flatten(v, indent + "  ", depth + 1))
            else:
                out.append(f"{indent}{k}: {v}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:25]):
            if isinstance(v, dict):
                out.append(f"{indent}- ")
                out.extend(_flatten(v, indent + "  ", depth + 1))
            else:
                out.append(f"{indent}- {v}")
        if len(obj) > 25:
            out.append(f"{indent}… (+{len(obj) - 25} more)")
    else:
        out.append(f"{indent}{obj}")
    return out


def emit(envelope: dict, fmt: str) -> int:
    """Print the envelope and return the process exit code."""
    sys.stdout.write(render(envelope, fmt) + "\n")
    sys.stdout.flush()
    return exit_code_of(envelope)
