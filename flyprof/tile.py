"""GPU-free tile-size advisor.

Given a problem shape + dtype and the SoC constants, estimate a tile's LDS
footprint, register/occupancy ceiling, and where it sits on the roofline
(memory- vs compute-bound), then recommend a tile. Every number is an *estimate*
(``confidence: heuristic``) until ``capture`` returns the compiler's real
``arch_vgpr`` — at which point ``tile`` can be re-run with the measured VGPR for an
``exact`` occupancy verdict.

Tile convention: a workgroup computes a ``BM x BN`` output tile, streaming the K
dimension in ``BK`` chunks through LDS with ``buffers`` pipeline stages. For a
reduction kernel (softmax/norm) read ``BM`` = rows/block, ``BN`` = cols, ``BK`` =
reduction chunk; the LDS/occupancy math still applies, the roofline is informational.
"""
from __future__ import annotations

import math
from typing import Optional

from .envelope import FlyprofError, Result
from .soc import DTYPE_BYTES, SocSpec, canon_dtype, get_spec

# assumed workgroup size when the kernel's own value is unknown (stated in output)
DEFAULT_WG_SIZE = 256


def _bytes(dtype: str) -> float:
    dt = canon_dtype(dtype)
    if dt not in DTYPE_BYTES:
        raise FlyprofError("BAD_ARGS", f"unknown dtype {dtype!r}", help="use bf16/fp16/fp8/fp4/fp32")
    return DTYPE_BYTES[dt]


def _ceil_to(g: int, x: float) -> int:
    return int(math.ceil(x / g) * g)


def evaluate_tile(spec: SocSpec, BM: int, BN: int, BK: int, dtype: str,
                  out_dtype: Optional[str] = None, buffers: int = 2,
                  wg_size: int = DEFAULT_WG_SIZE) -> dict:
    dtype = canon_dtype(dtype)
    out_dtype = canon_dtype(out_dtype or dtype)
    b = _bytes(dtype)
    b_out = _bytes(out_dtype)
    waves_per_wg = max(1, wg_size // spec.wave_size)

    # 1. LDS footprint (A tile BM*BK + B tile BK*BN, staged `buffers` deep)
    lds_bytes = _ceil_to(512, (BM * BK + BK * BN) * b * buffers)
    lds_ok = lds_bytes <= spec.lds_bytes_per_cu
    lds_wgs_per_cu = (spec.lds_bytes_per_cu // lds_bytes) if lds_bytes > 0 else spec.waves_per_cu

    # 2. VGPR / occupancy.  FP32 accumulate => 1 VGPR per output element per thread.
    acc_regs = math.ceil(BM * BN / wg_size)
    operand_regs = math.ceil((BM * BK + BK * BN) / wg_size * b / 4.0)
    overhead_regs = 16
    vgpr_est = acc_regs + operand_regs + overhead_regs
    vgpr_alloc = _ceil_to(spec.vgpr_granularity, vgpr_est)
    waves_per_simd_vgpr = min(spec.waves_per_simd, spec.vgpr_per_simd // max(vgpr_alloc, 1))
    occ_by_vgpr = spec.simd_per_cu * waves_per_simd_vgpr
    occ_by_lds = lds_wgs_per_cu * waves_per_wg
    waves_per_cu = min(spec.waves_per_cu, occ_by_vgpr, occ_by_lds) if lds_ok else 0

    limiters = {"hardware": spec.waves_per_cu, "VGPR": occ_by_vgpr, "LDS": occ_by_lds}
    binding_limiter = "LDS (overflow)" if not lds_ok else min(limiters, key=limiters.get)
    occupancy_pct = round(100.0 * waves_per_cu / spec.waves_per_cu, 1)

    # 3. arithmetic intensity vs the roofline ridge
    flops_tile = 2 * BM * BN * BK
    hbm_tile = (BM * BK + BK * BN) * b + BM * BN * b_out
    ai_tile = flops_tile / hbm_tile if hbm_tile else 0.0
    ridge_ai = spec.ridge_ai_hbm(dtype)
    regime = "memory-bound" if ai_tile < ridge_ai else "compute-bound"

    # 4. minimal BK to cross the ridge (hold BM, BN): solve ai(BK) = ridge_ai
    denom = 2 * BM * BN - ridge_ai * (BM + BN) * b
    min_bk = ridge_ai * BM * BN * b_out / denom if denom > 0 else None

    return {
        "tile": {"BM": BM, "BN": BN, "BK": BK, "buffers": buffers},
        "assumed_wg_size": wg_size,
        "lds_bytes": lds_bytes,
        "lds_limit": spec.lds_bytes_per_cu,
        "lds_ok": lds_ok,
        "lds_wgs_per_cu": lds_wgs_per_cu,
        "vgpr_per_thread_est": vgpr_est,
        "vgpr_alloc": vgpr_alloc,
        "waves_per_simd_by_vgpr": waves_per_simd_vgpr,
        "waves_per_cu": waves_per_cu,
        "occupancy_pct": occupancy_pct,
        "binding_limiter": binding_limiter,
        "ai_tile": round(ai_tile, 2),
        "ridge_ai": round(ridge_ai, 2),
        "regime": regime,
        "min_bk_to_cross_ridge": (math.ceil(min_bk) if min_bk and min_bk > 0 else None),
    }


def recommend(spec: SocSpec, dtype: str, out_dtype: Optional[str], buffers: int = 2) -> dict:
    """Grid-search a feasible tile that balances occupancy and arithmetic intensity."""
    best = None
    candidates = []
    for BM in (64, 128, 256):
        for BN in (64, 128, 256):
            for BK in (16, 32, 64, 128, 256):
                ev = evaluate_tile(spec, BM, BN, BK, dtype, out_dtype, buffers)
                if not ev["lds_ok"] or ev["waves_per_cu"] < 4:
                    continue
                # score: reward crossing the ridge and holding >=8 waves/CU, then AI
                score = (
                    (1000 if ev["regime"] == "compute-bound" else 0)
                    + min(ev["waves_per_cu"], 8) * 50
                    + min(ev["ai_tile"], ev["ridge_ai"])
                )
                ev["_score"] = round(score, 2)
                candidates.append(ev)
    if not candidates:
        # fall back to the smallest tile that fits LDS
        rec = evaluate_tile(spec, 64, 64, 16, dtype, out_dtype, buffers)
        return {"recommended": rec, "candidates": []}
    candidates.sort(key=lambda e: e["_score"], reverse=True)
    best = candidates[0]
    return {"recommended": best, "candidates": candidates[:5]}


def grid_occupancy(spec: SocSpec, M: Optional[int], N: Optional[int], BM: int, BN: int) -> Optional[dict]:
    """Generic tile-grid saturation: are there enough output tiles to fill the GPU?"""
    if not M or not N:
        return None
    tiles = math.ceil(M / BM) * math.ceil(N / BN)
    cu = spec.cu
    full_waves = tiles // cu
    tail = tiles % cu
    if tiles <= cu:
        verdict = f"underfilled-{round(100 * (1 - tiles / cu))}pct"
    elif tail == 0:
        verdict = "saturated"
    else:
        # the last partial wave wastes (cu - tail)/cu of one wave
        waste = (cu - tail) / cu / (full_waves + 1)
        verdict = "saturated" if waste < 0.02 else f"tail-underfill-{round(100 * waste)}pct"
    return {"total_tiles": tiles, "target_ctas": cu, "full_waves": full_waves,
            "tail_tiles": tail, "tail_verdict": verdict}


def parse_shape(shape: Optional[str]) -> dict:
    if not shape:
        return {}
    parts = [p for p in shape.replace("x", ",").split(",") if p.strip()]
    dims = []
    for p in parts:
        try:
            dims.append(int(p))
        except ValueError:
            pass
    out = {}
    if len(dims) >= 1:
        out["M"] = dims[0]
    if len(dims) >= 2:
        out["N"] = dims[1]
    if len(dims) >= 3:
        out["K"] = dims[2]
    return out


def cmd_tile(args, cfg) -> Result:
    spec = get_spec(args.arch or cfg.arch())
    dtype = canon_dtype(args.dtype)
    out_dtype = canon_dtype(args.out_dtype or dtype)
    shape = parse_shape(args.shape)
    buffers = args.buffers

    data = {
        "kernel": args.kernel,
        "arch": spec.arch,
        "shape": {**shape, "dtype": dtype, "out_dtype": out_dtype},
        "soc": spec.peaks(dtype),
        "confidence": "heuristic",
    }

    if args.bm and args.bn and args.bk:
        ev = evaluate_tile(spec, args.bm, args.bn, args.bk, dtype, out_dtype, buffers)
        data["evaluated"] = ev
        BM, BN = args.bm, args.bn
    else:
        rec = recommend(spec, dtype, out_dtype, buffers)
        data["recommended"] = rec["recommended"]
        data["candidates"] = rec["candidates"]
        BM = rec["recommended"]["tile"]["BM"]
        BN = rec["recommended"]["tile"]["BN"]

    grid = grid_occupancy(spec, shape.get("M"), shape.get("N"), BM, BN)
    if grid:
        data["grid"] = grid

    # persist into the bundle if one is active, so capture/report can read shape+tile
    if cfg.bundle is not None:
        cfg.write_artifact("tile.json", data)
    return Result(data=data, bundle=str(cfg.bundle) if cfg.bundle else None)
