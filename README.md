# flydsl-rocprof-cli (`flyprof`)

**The front half of FlyDSL kernel optimization on AMD CDNA (gfx950/MI350X).** Give
it a kernel name; it runs `rocprofv3`, decodes the trace, and returns a complete,
instruction-level diagnosis — every stall *bubble*, where it lives in the `.py`, and
which FlyDSL thread-level knob addresses it — as a stable bundle of JSON + a
`REPORT.md`. It does **not** optimize the kernel; it produces the evidence to.

It is built to be **driven by an agent**: deterministic subcommands, one JSON
envelope per command, a clear input→output at every step, paired with companion
skills under [`skills/`](skills/). (CLI+skill philosophy borrowed from
[opencli](https://github.com/jackwener/opencli); not its code.)

It wraps four things into one workflow: **rocprofv3** (capture), **rocprof-compute**
SoC spec (roofline/tile math), **rocprof-compute-viewer**'s data model (replicated
headlessly — no Qt), and the **flydsl-kernel-profiling** bundle layout (the artifact hub).

---

## The I/O contract

**Input**

```
flyprof <command> [<kernel>] [--shape M,N[,K],dtype] [--gpu N]
                  [--worktree PATH] [--bundle DIR] [--format json|table]
```

- `<kernel>` is resolved against the FlyDSL worktree (a curated recipe if one exists,
  otherwise synthesized on the fly — *any* kernel name works). Arch auto-detected.

**Output** — exactly **one envelope** on stdout, every command, every time:

```json
{ "ok": true,  "command": "capture", "bundle": "/abs/dir",
  "data": { …command-specific… }, "warnings": [], "truncated": {…}? }

{ "ok": false, "command": "capture",
  "error": { "code": "EMPTY_ATT", "exitCode": 66, "message": "…", "help": "…", "available": […] } }
```

- **State flows through the bundle directory.** The first command makes one (under
  `runs/<kernel>-<ts>/`, or pass `--bundle DIR`); each later command reads the prior
  artifacts (`capture.json`, `bubbles.json`, `counters.json`, …) from it.
- **Exit codes are sysexits-style and bound to `error.code`** — branch on the number,
  not the prose: `0` ok · `66` empty/no-match (*a result, not a bug*) · `69`
  rocprofv3/GPU missing (STOP) · `75` timeout (retry) · `78` bad config/missing build
  tree · `70` internal.

```
list → doctor → tile → capture → counters → bubbles → map → report → bundle
        (each command writes one JSON artifact into the bundle; later commands read it)
```

---

## Quickstart

```bash
pip install -e .                      # console script: flyprof  (dep: pyyaml)
flyprof doctor                        # rocprofv3 / GPU / build-tree / soc -> verdict: ready
flyprof list                          # which kernels can I profile?
flyprof run softmax --shape 32768,8192,bf16 --gpu 0     # the whole pipeline
```

`flyprof run` is the convenience path; step through the primitives when you want to
inspect or re-run one stage. Point at your FlyDSL checkout with
`--worktree` (default `$FLYPROF_WORKTREE` or `/sgl-workspace/FlyDSL-lab`).

## Commands

| command | input → output | GPU? |
|---|---|---|
| `list` | worktree → profilable kernels (registry ∪ synthesizable) | no |
| `doctor` | → rocprofv3/GPU/build-tree/soc verdict | no |
| `tile` | shape+dtype+soc → recommended BM/BN/BK + LDS/VGPR/occupancy/roofline | **no** |
| `capture` | kernel → ATT bundle(s) + `capture.json` (dispatch, vgpr, occ, mapped%) | yes |
| `counters` | kernel → roofline / occupancy / L2 / LDS / tri-state bound | yes |
| `bubbles` | bundle → stall taxonomy by class + hotspots + waitcnt attribution | **no** |
| `map` | bundle → ISA ↔ `.py:line` + source snippets + flamegraph | **no** |
| `report` | bundle → ranked bubble→knob recommendations + `REPORT.md`/`report.json` | no |
| `bundle` | bundle → canonical `examples/<kernel>/` artifact | no |
| `run` | full pipeline on one kernel | yes |

## How it decides what to recommend

`report` runs a rules engine ([`flyprof/knobs.py`](flyprof/knobs.py), catalog in
[`skills/references/bubble-knob-table.md`](skills/references/bubble-knob-table.md)):
each stall/bubble class → the diagnostic signal that fires it → the FlyDSL knob → the
concrete code change, **gated by the roofline bound** and carrying **evidence** (the
field that fired + the `.py:line` the bubble maps to) so every claim is auditable
back to the ISA.

**On honesty:** the bubble taxonomy and counter *ratios* (L2 hit, 32B-partial
fraction, MFMA fraction) are reliable and drive the decisions. Absolute HBM bandwidth
from raw PMC needs per-XCD/channel normalization we don't fully reproduce, so it is
reported as a lower bound flagged `calibrated: false` — never used to decide the bound.

## Companion skills (for agents)

[`skills/`](skills/) holds the runbooks an agent follows: `flyprof-usage` (router),
`flyprof-capture`, `flyprof-analyze`, `flyprof-report`, `flyprof-triage`, plus
references. Copy/symlink into `.claude/skills/` to register them.

## Notes

- **Standalone.** The harness logic (`att_capture`, hotspot reg-pressure, etc.) is
  vendored + de-hardcoded here; no runtime dependency on `flydsl-kernel-profiling`.
  To publish a bundle into that hub, pass `--examples-dir <hub>/examples` explicitly
  (the default is repo-local `runs/examples/`).
- **Validated** on MI350X/gfx950 (rocprofv3 1.1.0) on `softmax`: `flyprof run` →
  memory-bound, vmcnt-dominated (49% of 77% stall) → "software-prefetch / double-buffer
  the global loads", traceable to `softmax_kernel.py`. Offline tests (`pytest tests/`)
  exercise the tile math, headless bubble taxonomy + ISA↔source map (committed
  fixture), knob rules, and the envelope contract — no GPU needed.
