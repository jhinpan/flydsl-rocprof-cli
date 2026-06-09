"""SoC peaks + the GPU-free tile advisor."""
from flyprof.soc import get_spec
from flyprof.tile import evaluate_tile, grid_occupancy, recommend


def test_bf16_mfma_peak():
    s = get_spec("gfx950")
    assert abs(s.peak_mfma_gflops("bf16") - 2306867.2) < 1.0   # 2.31 PFLOP/s
    assert abs(s.peak_mfma_gflops("fp8") - 2 * 2306867.2) < 1.0
    assert abs(s.ridge_ai_hbm("bf16") - 288.36) < 0.5


def test_gfx950_spec_constants():
    s = get_spec("gfx950")
    assert s.cu == 256 and s.waves_per_cu == 32 and s.lds_bytes_per_cu == 163840
    assert s.unified_vgpr_pool is True


def test_tile_eval_lds_and_regime():
    s = get_spec("gfx950")
    ev = evaluate_tile(s, 128, 128, 64, "bf16")
    assert ev["lds_bytes"] == (128 * 64 + 64 * 128) * 2 * 2          # 65536
    assert ev["lds_ok"] is True
    assert ev["regime"] == "memory-bound"                            # bf16 ridge is very high
    assert ev["waves_per_cu"] >= 4
    assert ev["binding_limiter"] in ("hardware", "VGPR", "LDS", "LDS (overflow)")


def test_tile_lds_overflow_rejected():
    s = get_spec("gfx950")
    ev = evaluate_tile(s, 256, 256, 256, "fp32")   # huge -> overflow LDS
    assert ev["lds_ok"] is False
    assert ev["waves_per_cu"] == 0


def test_recommend_returns_feasible():
    s = get_spec("gfx950")
    rec = recommend(s, "bf16", None)
    r = rec["recommended"]
    assert r["lds_ok"] and r["waves_per_cu"] >= 4


def test_grid_saturation():
    s = get_spec("gfx950")
    g = grid_occupancy(s, 32768, 8192, 128, 128)
    assert g["total_tiles"] == 16384 and g["tail_verdict"] == "saturated"
    g2 = grid_occupancy(s, 128, 128, 128, 128)          # one tile -> underfilled
    assert "underfilled" in g2["tail_verdict"]
