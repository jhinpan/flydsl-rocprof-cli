"""Register-pressure / occupancy estimation from a decoded ISA listing.

Ported from the flydsl-kernel-profiling harness's ``hotspot_analyzer.py``
(``detect_arch_and_reg_pressure``), given a dict API so ``capture`` can record
the compiler's *real* VGPR usage + the occupancy it implies. This is the one
piece of occupancy that is exact (it reads the allocated register indices out of
the ISA), as opposed to ``tile.py``'s pre-capture estimate.

Architecture notes (CDNA ISA reference):
  * CDNA3 (gfx942): 256 arch_vgpr + 256 accum_vgpr in two SEPARATE pools.
    occupancy = 256 / max(arch_alloc, accum_alloc).
  * CDNA4 (gfx950): one COMBINED 512-VGPR pool, flexibly split.
    occupancy = 512 / (arch_alloc + accum_alloc).
  VGPRs allocate in groups of 8; max 8 waves/SIMD.
"""
from __future__ import annotations

import re

MAX_WAVES_PER_SIMD = 8


def reg_pressure(asms: list[str]) -> dict:
    is_gfx950 = any(
        "v_mfma_scale" in a or "mfma_f32_16x16x128" in a or "mfma_f32_32x32x64" in a or "v_mfma_ld_scale" in a
        for a in asms
    )
    arch = "gfx950" if is_gfx950 else "gfx942"

    max_vgpr = max_agpr = 0
    for a in asms:
        for m in re.finditer(r"\bv(\d+)\b", a):
            max_vgpr = max(max_vgpr, int(m.group(1)))
        for m in re.finditer(r"\bv\[(\d+)", a):
            max_vgpr = max(max_vgpr, int(m.group(1)))
        for m in re.finditer(r"\ba(\d+)\b", a):
            max_agpr = max(max_agpr, int(m.group(1)))
        for m in re.finditer(r"\ba\[(\d+)", a):
            max_agpr = max(max_agpr, int(m.group(1)))

    arch_vgpr = max_vgpr + 1 if max_vgpr else 0
    accum_vgpr = max_agpr + 1 if max_agpr else 0
    arch_alloc = ((arch_vgpr + 7) // 8) * 8
    accum_alloc = ((accum_vgpr + 7) // 8) * 8 if accum_vgpr else 0

    if is_gfx950:
        total = arch_alloc + accum_alloc
        occ = min(MAX_WAVES_PER_SIMD, 512 // total) if total else MAX_WAVES_PER_SIMD
        nxt = occ + 1
        target = 512 // nxt if nxt <= MAX_WAVES_PER_SIMD else None
    else:
        lim = max(arch_alloc, accum_alloc)
        occ = min(MAX_WAVES_PER_SIMD, 256 // lim) if lim else MAX_WAVES_PER_SIMD
        nxt = occ + 1
        target = 256 // nxt if nxt <= MAX_WAVES_PER_SIMD else None

    return {
        "arch_detected": arch,
        "arch_vgpr": arch_vgpr,
        "arch_vgpr_alloc": arch_alloc,
        "accum_vgpr": accum_vgpr,
        "accum_vgpr_alloc": accum_alloc,
        "occupancy_waves_per_simd": occ,
        "occupancy_waves_per_cu": occ * 4,
        "next_occupancy": nxt if nxt <= MAX_WAVES_PER_SIMD else None,
        "vgpr_target_for_next_occ": target,
        "mfma_count": sum(1 for a in asms if "v_mfma" in a or "v_smfma" in a),
        "buffer_load": sum(1 for a in asms if "buffer_load" in a or "global_load" in a),
        "ds_read": sum(1 for a in asms if "ds_read" in a or "ds_load" in a),
        "ds_write": sum(1 for a in asms if "ds_write" in a or "ds_store" in a),
    }
