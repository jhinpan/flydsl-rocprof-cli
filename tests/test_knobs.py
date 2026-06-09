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


def test_already_optimal_emits_no_rec():
    bubbles = _memory_bound_bubbles()
    sig = knobs.build_signals(bubbles, _memory_counters(), None)
    sig.pct_of_hbm_roof = 92.0          # near the HBM roof
    assert knobs.is_optimal(sig) is True
    assert knobs.recommend(sig) == []   # never manufacture an opportunity
