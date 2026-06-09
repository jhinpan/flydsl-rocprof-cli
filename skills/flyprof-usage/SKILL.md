---
name: flyprof-usage
description: >
  Start here to profile a FlyDSL kernel on AMD CDNA (gfx950/MI350X). Router for the
  flyprof CLI: how to run the fixed capture->analyze->report workflow, read its JSON
  envelopes, and pick the right sub-skill. Use whenever asked to profile/analyze a
  kernel, find why a kernel is slow, build an optimization report, or capture an ATT
  trace for a FlyDSL kernel.
allowed-tools: Bash(flyprof:*), Read, Write, Edit
---

# flyprof — profile any FlyDSL kernel, get an instruction-level optimization report

`flyprof` is the *front half* of FlyDSL kernel optimization: given a kernel name it
produces a complete, agent-readable diagnosis — every stall bubble, where it lives
in the `.py`, and which FlyDSL thread-level knob addresses it. It does **not**
optimize the kernel; it gives you the evidence to. Drive it the same way every time.

## The contract (read this once)

- **Every command prints ONE JSON envelope.** Pass `-f json` (it's the default when
  piped). Shape: `{"ok": true, "command": ..., "bundle": ..., "data": {...}, "warnings": [...]}`
  or `{"ok": false, "command": ..., "error": {"code", "exitCode", "message", "help", ...}}`.
  **Read `data`/`error` — never scrape prose.**
- **State flows through a bundle directory.** The first command makes one (under
  `runs/<kernel>-<ts>` by default, or pass `--bundle DIR`); every later command takes
  `--bundle DIR` and reads the prior artifacts (`capture.json`, `bubbles.json`, …).
- **Exit codes are the truth, not the message** (sysexits): `0` ok · `66` empty/
  no-match (*a result, not a bug*) · `69` rocprofv3/GPU missing (STOP) · `75` timeout
  (retry) · `78` bad config/missing build tree · `70` internal. See `flyprof-triage`.
- **Don't memorize kernel names — query them.** `flyprof list -f json` is the source
  of truth and changes as the worktree moves.

## The fixed workflow

```
flyprof list                       # what can I profile?
flyprof doctor                     # can this box capture? (verdict: ready)
flyprof run <kernel> --gpu 0       # the whole pipeline, or step through it:
  flyprof tile <k> --shape … --dtype …      (GPU-free: tile/occupancy/roofline advice)
  flyprof capture <k> --tag both --with-pmc (rocprofv3 ATT + PMC -> bundle)
  flyprof counters <k>                       (roofline / occupancy / L2 / bound)
  flyprof bubbles --bundle <dir>             (stall taxonomy)
  flyprof map --bundle <dir>                 (ISA <-> .py:line)
  flyprof report --bundle <dir>              (ranked bubble->knob recommendations)
  flyprof bundle --bundle <dir>              (canonical examples/<k>/ for the hub)
```

`flyprof run` is the convenience path; the per-command path is for when you want to
inspect or re-run one stage.

## Where to go next (intent -> sub-skill)

| You want to… | Sub-skill |
|---|---|
| capture an ATT trace / it returned empty or out-of-range | `flyprof-capture` |
| read counters / tile / bubbles / map on a captured bundle | `flyprof-analyze` |
| produce the optimization report and act on a recommendation | `flyprof-report` |
| a command failed — what does the error code mean | `flyprof-triage` |

**Don't guess. Lean on the envelopes.** If a number looks wrong, re-run the command
that produced it and read its `confidence` field before making a claim.
