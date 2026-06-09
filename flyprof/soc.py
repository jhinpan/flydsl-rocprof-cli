"""AMD CDNA SoC spec + theoretical roofline peaks.

The numbers below are documented constants, not guesses. Sources:
  * gfx950/MI350X (CDNA4): AMD CDNA4 ISA ref + FlyDSL CLAUDE.md ("gfx950 ... 160KB
    LDS", combined 512-VGPR pool). 256 CU across 8 XCD in SPX (single-partition).
  * Peaks via the calibrated form  gflops = clock_MHz * CU * MULT / 1000, where
    MULT is FLOP/cycle/CU. The MULTs reproduce MI350X published dense peaks:
    bf16 MFMA 2200*256*4096/1000 = 2.31 PFLOP/s; fp8 4.61; fp4 9.23; FP32 VALU 72 TFLOP/s.
  * HBM3E ~8 TB/s.

``rocprof-compute``'s installed ``soc_gfx950.py`` only carries perfmon/partition
config (l2_banks=16, lds_banks_per_cu=32, pipes=4) — the FLOP/BW peaks are not in
its YAML (it measures them empirically), so we keep the theoretical table here and
expose every field in the CLI output for transparency/override.
"""
from __future__ import annotations

import dataclasses
from typing import Optional

# bytes per element
DTYPE_BYTES = {
    "fp4": 0.5, "e2m1": 0.5,
    "fp6": 0.75, "e3m2": 0.75, "e2m3": 0.75,
    "fp8": 1.0, "e4m3": 1.0, "e5m2": 1.0, "e4m3fn": 1.0, "int8": 1.0, "i8": 1.0,
    "fp16": 2.0, "f16": 2.0, "bf16": 2.0,
    "fp32": 4.0, "f32": 4.0, "tf32": 4.0,
    "fp64": 8.0, "f64": 8.0,
}

# MFMA FLOP/cycle/CU by dtype family (the "MULT" in the peak formula)
_MFMA_MULT = {
    "fp4": 16384, "fp6": 16384,
    "fp8": 8192, "int8": 8192,
    "fp16": 4096, "bf16": 4096,
    "fp32": 256, "tf32": 1024,
    "fp64": 128,
}
_VALU_MULT_F32 = 128  # FP32 packed-FMA VALU peak (=> 72 TFLOP/s on gfx950)


def canon_dtype(dt: str) -> str:
    dt = (dt or "bf16").lower()
    alias = {
        "f16": "fp16", "f32": "fp32", "f64": "fp64",
        "e4m3": "fp8", "e5m2": "fp8", "e4m3fn": "fp8", "i8": "int8",
        "e2m1": "fp4", "e3m2": "fp6", "e2m3": "fp6",
    }
    return alias.get(dt, dt)


@dataclasses.dataclass
class SocSpec:
    arch: str
    name: str
    cu: int
    simd_per_cu: int
    waves_per_simd: int          # max wavefronts a SIMD can hold (occupancy ceiling)
    wave_size: int
    vgpr_per_simd: int           # register-file depth per SIMD (combined pool on CDNA4)
    vgpr_granularity: int
    lds_bytes_per_cu: int        # usable LDS per CU == per-workgroup allocation ceiling
    clock_mhz: float
    hbm_gbs: float
    num_xcd: int
    unified_vgpr_pool: bool      # CDNA4: arch+accum share one 512 pool; CDNA3: two 256 pools

    @property
    def waves_per_cu(self) -> int:
        return self.simd_per_cu * self.waves_per_simd

    # ---- theoretical peaks -------------------------------------------------
    def peak_mfma_gflops(self, dtype: str) -> float:
        mult = _MFMA_MULT.get(canon_dtype(dtype), _MFMA_MULT["bf16"])
        return self.clock_mhz * self.cu * mult / 1000.0

    def peak_valu_gflops(self) -> float:
        return self.clock_mhz * self.cu * _VALU_MULT_F32 / 1000.0

    def ridge_ai_hbm(self, dtype: str) -> float:
        """Arithmetic-intensity ridge point (FLOP/byte) against HBM for this dtype."""
        return self.peak_mfma_gflops(dtype) / self.hbm_gbs

    def peaks(self, dtype: str) -> dict:
        dt = canon_dtype(dtype)
        return {
            "dtype": dt,
            "peak_mfma_gflops": round(self.peak_mfma_gflops(dt), 1),
            "peak_valu_gflops": round(self.peak_valu_gflops(), 1),
            "peak_hbm_gbs": self.hbm_gbs,
            "ridge_ai_hbm": round(self.ridge_ai_hbm(dt), 2),
        }

    def as_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["waves_per_cu"] = self.waves_per_cu
        return d


# ---- the table -------------------------------------------------------------
_SPECS = {
    "gfx950": SocSpec(
        arch="gfx950", name="AMD Instinct MI350X (CDNA4, SPX)",
        cu=256, simd_per_cu=4, waves_per_simd=8, wave_size=64,
        vgpr_per_simd=512, vgpr_granularity=8,
        lds_bytes_per_cu=163840,  # 160 KB (CLAUDE.md); == per-workgroup ceiling on CDNA4
        clock_mhz=2200.0, hbm_gbs=8000.0, num_xcd=8, unified_vgpr_pool=True,
    ),
    # CDNA3 baseline kept for completeness; numbers are MI300X dense peaks.
    "gfx942": SocSpec(
        arch="gfx942", name="AMD Instinct MI300X (CDNA3, SPX)",
        cu=304, simd_per_cu=4, waves_per_simd=8, wave_size=64,
        vgpr_per_simd=512, vgpr_granularity=8,
        lds_bytes_per_cu=65536,
        clock_mhz=2100.0, hbm_gbs=5300.0, num_xcd=8, unified_vgpr_pool=False,
    ),
}


def get_spec(arch: Optional[str]) -> SocSpec:
    """Return the SoC spec for an arch, falling back to gfx950 with a gfx95* prefix match."""
    if not arch:
        return _SPECS["gfx950"]
    arch = arch.lower()
    if arch in _SPECS:
        return _SPECS[arch]
    if arch.startswith("gfx95"):
        return _SPECS["gfx950"]
    if arch.startswith("gfx94"):
        return _SPECS["gfx942"]
    return _SPECS["gfx950"]
