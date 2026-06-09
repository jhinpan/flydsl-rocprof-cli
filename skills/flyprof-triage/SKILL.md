---
name: flyprof-triage
description: >
  Diagnose a failing flyprof command by its error code. Use when any flyprof command
  returns ok:false / a non-zero exit, to decide whether to stop, fix config, retry, or
  treat the result as a finding. Maps each error.code to a hard action so you don't loop.
allowed-tools: Bash(flyprof:*), Read
---

# flyprof-triage — branch on error.code, don't loop

Every failure is one envelope: `{"ok": false, "error": {"code", "exitCode", "message",
"help", ...}}`. **Read the code, take the bound action. Do not retry the same command
hoping for a different result.**

## Hard-stop table (error.code → exitCode → action)

| code | exit | action |
|---|---|---|
| `ROCPROFV3_MISSING` | 69 | **STOP.** rocprofv3 isn't on PATH. Tell the user to install ROCm / fix PATH. Do not retry. |
| `NO_GPU` | 69 | **STOP.** No GPU visible. Do not retry. |
| `BUILD_TREE_MISSING` | 78 | **STOP.** No built FlyDSL. Fix `--worktree` to a checkout with `build-fly/python_packages`. |
| `BAD_ARGS` | 78 | Fix the argument the message names (e.g. `--shape M,N,dtype`), then re-run. |
| `MISSING_PREREQ` | 78 | Run the prerequisite command (`error.help` says which) into the **same** `--bundle`, then re-run. |
| `KERNEL_NOT_FOUND` | 66 | The kernel isn't in the worktree. `flyprof list` to find the right name. Not a bug. |
| `NO_KERNEL_DISCOVERED` | 66 | `--stats` saw no FlyDSL kernel. The test may not launch one; check `error.candidates`. Try a real `--shape`. |
| `EMPTY_ATT` | 66 | All dispatches were empty shells — the grid was too small. Enlarge `--shape`. A *result*, not a crash. |
| `DISPATCH_OUT_OF_RANGE` | 66 | Use a dispatch id from `error.available`. |
| `NO_ARTIFACT` / `NO_RECOMMENDATION` | 66 | Empty ≠ broken. Likely a valid "nothing here" / "already optimal". Investigate, don't "fix". |
| `CAPTURE_TIMEOUT` | 75 | Transient. Retry **once** with a bigger `--timeout` or a smaller `--shape`. Budget one retry. |
| `INTERNAL` | 70 | A real bug in flyprof. Capture `error.trace`, report it. Don't loop. |

## A compile failure is a finding, not a loop

If `capture` reports the kernel failed to compile (e.g. a missing fast-dispatch slot),
that is a **real result to report**, not a signal to retry with tweaks. Surface it and stop.

## Budget

At most **3 triage rounds** for a single goal. If you've fixed config (worktree, shape,
prerequisite) twice and still fail, stop and report the last envelope verbatim — don't
keep mutating arguments.
