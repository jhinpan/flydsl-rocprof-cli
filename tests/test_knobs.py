"""The bubble -> FlyDSL-knob rules engine."""
from flyprof import knobs


def _memory_bound_bubbles():
    return {
        "stall_taxonomy": {"by_class": {
            "vmcnt": {"pct": 49.0, "rank": 1, "is_bubble": True},
            "vmem_load": {"pct": 17.0, "rank": 2, "is_bubble": False},
            "barrier": {"pct": 11.0, "rank": 3, "is_bubble": True},
        }},
        "inst_mix": {"vmem_load": 0.30},
        "hotspots": [
            {"class": "vmcnt", "code_line": 45, "inst": "s_waitcnt vmcnt(1)",
             "source": "softmax_kernel.py:43",
             "waitcnt_sources": [{"source": "softmax_kernel.py:122", "inst": "buffer_load_dwordx4"}]},
            {"class": "barrier", "code_line": 110, "inst": "s_barrier", "source": "softmax_kernel.py:80"},
        ],
    }


def _memory_counters():
    return {"memory": {"l2_hit_rate": 33.0, "partial_32b_frac": 0.0},
            "compute": {"mfma_inst_frac": 0.0}, "lds": {"bank_conflict_pct": 0.0},
            "roofline": {"bound": "memory"}}


def test_memory_bound_ranks_vmcnt_prefetch_first():
    sig = knobs.build_signals(_memory_bound_bubbles(), _memory_counters(), capture=None)
    recs = knobs.recommend(sig)
    assert recs, "expected recommendations for a vmcnt-dominated kernel"
    assert recs[0]["bubble_class"] == "vmcnt"
    assert "prefetch" in recs[0]["knob"].lower() or "double-buffer" in recs[0]["knob"].lower()
    # evidence is auditable back to the source line + the op it waits on
    ev = recs[0]["evidence"]
    assert ev["source"] == "softmax_kernel.py:43"
    assert ev["waits_on"][0]["source"] == "softmax_kernel.py:122"
    assert all("rank" in r for r in recs)


def test_compute_gate_blocks_tile_growth_rules():
    # a compute-bound kernel should not get the memory-only vmem prefetch rule
    bubbles = {"stall_taxonomy": {"by_class": {"mfma": {"pct": 40, "rank": 1, "is_bubble": False},
                                               "vmcnt": {"pct": 12, "rank": 2, "is_bubble": True}}},
               "inst_mix": {"mfma": 0.5}, "hotspots": [{"class": "mfma", "source": "g.py:9"}]}
    counters = {"compute": {"mfma_inst_frac": 0.6}, "memory": {}, "roofline": {"bound": "compute"}}
    sig = knobs.build_signals(bubbles, counters, None)
    recs = knobs.recommend(sig)
    assert all(r["rule"] != "vmem_prefetch" for r in recs)     # gated out on compute-bound


def test_unknown_bound_surfaces_occupancy_and_skips_zero_lds():
    # Regression (flash-attn): counters bound == "unknown" must NOT gate out relevant
    # rules; a severe occupancy deficit should rank first; a 0-reading LDS-conflict
    # counter must NOT produce an (unsourced) swizzle recommendation.
    bubbles = {"stall_taxonomy": {"by_class": {
        "barrier": {"pct": 20.0, "rank": 1, "is_bubble": True},
        "vmcnt": {"pct": 14.0, "rank": 2, "is_bubble": True},
        "lds": {"pct": 12.0, "rank": 3, "is_bubble": False}}},  # high lds *stall* but no conflict
        "inst_mix": {}, "hotspots": [{"class": "barrier", "source": "fa.py:192"},
                                     {"class": "vmcnt", "source": "fa.py:192"}]}  # no lds hotspot
    counters = {"roofline": {"bound": "unknown"}, "memory": {},
                "compute": {"mfma_inst_frac": 0.2}, "lds": {"bank_conflict_pct": 0.0}}
    capture = {"tags": {"big": {"occupancy_waves_per_cu": 4, "arch_vgpr": 245}}}
    sig = knobs.build_signals(bubbles, counters, capture, "big")
    recs = knobs.recommend(sig)
    rules = [r["rule"] for r in recs]
    assert "occupancy_vgpr" in rules                  # unknown bound did not gate it out
    assert recs[0]["rule"] == "occupancy_vgpr"        # occ=4/8 is severe -> ranks first
    assert "lds_bank_conflict" not in rules           # 0 conflict counter -> no false swizzle rec


def test_already_optimal_emits_no_rec():
    bubbles = _memory_bound_bubbles()
    sig = knobs.build_signals(bubbles, _memory_counters(), None)
    sig.pct_of_hbm_roof = 92.0          # near the HBM roof
    assert knobs.is_optimal(sig) is True
    assert knobs.recommend(sig) == []   # never manufacture an opportunity
