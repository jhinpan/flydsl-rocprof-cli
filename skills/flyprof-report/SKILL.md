---
name: flyprof-report
description: >
  Produce and act on a FlyDSL kernel optimization report with flyprof: synthesize the
  ranked bubble->FlyDSL-knob recommendations, read the ISA<->Python evidence, optionally
  apply one knob, and verify the change re-profiles better. Use when asked what to change
  to make a kernel faster, or to turn a captured bundle into an optimization plan.
allowed-tools: Bash(flyprof:*), Read, Edit, Write
---

# flyprof-report — bubbles -> ranked FlyDSL changes, with proof

`flyprof report` is the synthesis step: it runs the knob rules engine over the
bundle's `bubbles`/`counters`/`tile`/`capture` artifacts and emits `REPORT.md` +
`report.json` + `headlines.json`. This is *prep* — it tells you what to change and
why; applying the change and re-measuring is a separate, deliberate act.

## Runbook

- [ ] Ensure the bundle has at least `capture.json` + an ATT dispatch (run
      `flyprof-capture`). `counters.json` is optional but improves the bound call.
- [ ] `flyprof report --bundle DIR -f json`. Read:
      - `data.bound_type` (memory/compute/latency) and `data.headline`,
      - `data.optimal` — if `true`, the kernel is near the roofline; **do not invent
        an optimization**, report that it's well-tuned,
      - `data.recommendations[]` — ranked; each has `evidence` (the firing signal +
        the `source:line` it maps to), `knob`, `api`, `code_change`, `confidence`.
- [ ] For the rank-1 recommendation, open the cited `evidence.source` line and confirm
      the `evidence.waits_on` op is really the dependency you'd break. The full
      bubble→knob catalog is `references/bubble-knob-table.md`.

## If you apply a knob (optional, deliberate)

- [ ] Apply exactly ONE recommendation's `code_change` to the kernel `.py`.
- [ ] Re-run `flyprof run <kernel> --gpu 0` into a NEW bundle.
- [ ] Diff `report.json`: did the rank-1 bubble's stall % drop? did `bound_type`
      move toward the roof? Compare `totals.stall_pct` before/after.

## Fixture gate (do not cheat the measurement)

When you have a known-good profile, write a fixture (expected top bubble class,
non-zero `mapped_pct`, dispatch present). On a later run, compare against it.
**Never relax the fixture to make a run pass** — a "regression" that's really a
measurement artifact (wrong shape, parser grabbing the wrong line, empty dispatch)
must be diagnosed, not silenced. Empty/odd results are findings; investigate the
capture before trusting a perf delta.
