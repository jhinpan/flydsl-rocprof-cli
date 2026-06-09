"""flyprof — agent-drivable rocprofv3 kernel-profiling CLI for FlyDSL on AMD CDNA.

The package is a thin, deterministic CLI over a fixed workflow:

    list -> doctor -> tile -> capture -> counters -> bubbles -> map -> report -> bundle

Every subcommand emits a single JSON *envelope* (see ``flyprof.envelope``) and
threads state through a *bundle directory*. The intelligence — mapping a stall
bubble to the FlyDSL thread-level knob that fixes it — lives in ``flyprof.knobs``.
"""

__version__ = "0.1.0"
