"""Runtime configuration — the de-hardcoded replacement for the harness's
module-level ``ROOT``/``PROF``/``PY`` constants.

A single :class:`Config` is built once from CLI flags + environment, then threaded
through every command and into the vendored capture/bundle logic. The *bundle
directory* it points at is the cross-command state handle: each command writes one
JSON artifact into it, later commands read those artifacts back.
"""
from __future__ import annotations

import dataclasses
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from .envelope import FlyprofError

# Sensible defaults for this environment; all overridable via flags or env.
DEFAULT_WORKTREE = os.environ.get("FLYPROF_WORKTREE", "/sgl-workspace/FlyDSL-lab")
DEFAULT_PYTHON = os.environ.get("FLYPROF_PYTHON", sys.executable)
REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "runs"
# `bundle` deposits the canonical examples/<k>/ here by default (gitignored, repo-local).
# To publish into the flydsl-kernel-profiling hub, pass --examples-dir <hub>/examples explicitly.
DEFAULT_EXAMPLES_DIR = os.environ.get("FLYPROF_EXAMPLES_DIR", str(RUNS_DIR / "examples"))


@dataclasses.dataclass
class Config:
    worktree: Path
    python: str = DEFAULT_PYTHON
    rocprofv3: str = os.environ.get("FLYPROF_ROCPROFV3", "rocprofv3")
    gpu: int = 0
    fmt: str = "json"
    quiet: bool = False
    examples_dir: Path = dataclasses.field(default_factory=lambda: Path(DEFAULT_EXAMPLES_DIR))
    _arch: Optional[str] = None
    bundle: Optional[Path] = None

    # ---- construction ------------------------------------------------------
    @classmethod
    def from_args(cls, args) -> "Config":
        worktree = Path(getattr(args, "worktree", None) or DEFAULT_WORKTREE).resolve()
        cfg = cls(
            worktree=worktree,
            python=getattr(args, "python", None) or DEFAULT_PYTHON,
            gpu=getattr(args, "gpu", 0) or 0,
            fmt=getattr(args, "format", None) or default_format(),
            quiet=bool(getattr(args, "quiet", False)),
        )
        if getattr(args, "examples_dir", None):
            cfg.examples_dir = Path(args.examples_dir).resolve()
        if getattr(args, "arch", None):
            cfg._arch = args.arch
        b = getattr(args, "bundle", None)
        if b:
            cfg.bundle = Path(b).resolve()
        return cfg

    # ---- derived paths -----------------------------------------------------
    @property
    def build_tree(self) -> Path:
        return self.worktree / "build-fly" / "python_packages"

    @property
    def pythonpath(self) -> str:
        return f"{self.build_tree}:{self.worktree}"

    def base_env(self) -> dict:
        """Environment that makes a FlyDSL kernel importable + emit DWARF line tables."""
        env = dict(os.environ)
        env["PYTHONPATH"] = self.pythonpath + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        env["HIP_VISIBLE_DEVICES"] = str(self.gpu)
        env["FLYDSL_DEBUG_ENABLE_DEBUG_INFO"] = "1"  # source mapping
        return env

    # ---- arch detection ----------------------------------------------------
    def arch(self) -> str:
        if self._arch:
            return self._arch
        self._arch = detect_arch() or "gfx950"
        return self._arch

    # ---- bundle as state handle -------------------------------------------
    def ensure_bundle(self, kernel: Optional[str] = None) -> Path:
        if self.bundle is None:
            stamp = time.strftime("%Y%m%d-%H%M%S")
            name = f"{kernel or 'kernel'}-{stamp}"
            self.bundle = (RUNS_DIR / name).resolve()
        self.bundle.mkdir(parents=True, exist_ok=True)
        return self.bundle

    def require_bundle(self) -> Path:
        if self.bundle is None or not self.bundle.exists():
            raise FlyprofError("BAD_ARGS", "this command needs --bundle <dir> from a prior capture")
        return self.bundle

    def artifact_path(self, name: str) -> Path:
        return self.require_bundle() / name

    def read_artifact(self, name: str) -> dict:
        p = self.require_bundle() / name
        if not p.exists():
            raise FlyprofError(
                "MISSING_PREREQ",
                f"{name} not found in bundle {self.bundle}",
                help=f"run the command that produces {name} first",
            )
        return json.loads(p.read_text())

    def write_artifact(self, name: str, obj: dict) -> Path:
        self.ensure_bundle()
        p = self.bundle / name
        p.write_text(json.dumps(obj, indent=2, default=str))
        return p

    # ---- preflight helpers -------------------------------------------------
    def check_build_tree(self) -> None:
        if not self.build_tree.exists():
            raise FlyprofError(
                "BUILD_TREE_MISSING",
                f"no build tree at {self.build_tree}",
            )

    def check_rocprofv3(self) -> str:
        path = shutil.which(self.rocprofv3)
        if not path:
            raise FlyprofError("ROCPROFV3_MISSING", f"{self.rocprofv3} not found on PATH")
        return path


def default_format() -> str:
    """JSON for agents/pipes; a friendly table only for an interactive terminal."""
    try:
        return "table" if sys.stdout.isatty() else "json"
    except Exception:
        return "json"


def detect_arch() -> Optional[str]:
    """Best-effort gfx arch from rocminfo (cached by caller)."""
    exe = shutil.which("rocminfo")
    if not exe:
        return None
    try:
        out = subprocess.run([exe], capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return None
    m = re.search(r"\b(gfx\d+[a-z]*)\b", out)
    return m.group(1) if m else None
