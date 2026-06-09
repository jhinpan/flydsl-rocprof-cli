"""Headless RCV replication against a committed softmax ATT fixture (no GPU)."""
import os

from flyprof import rcv

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "softmax_dispatch")


def test_classify():
    assert rcv.classify("s_waitcnt vmcnt(1)") == "vmcnt"
    assert rcv.classify("s_waitcnt lgkmcnt(0)") == "lgkmcnt"
    assert rcv.classify("buffer_load_dwordx4 v[4:7], v2, s[4:7], 0 offen") == "vmem_load"
    assert rcv.classify("ds_read_b128 v[0:3], v8") == "lds"
    assert rcv.classify("v_mfma_f32_16x16x16_bf16 a[0:3], v0, v1, a[0:3]") == "mfma"
    assert rcv.classify("s_barrier") == "barrier"


def test_load_code_columns():
    instrs = rcv.load_code(FIX)
    assert len(instrs) > 100
    assert any(i.source for i in instrs)
    hot = max(instrs, key=lambda i: i.stall)
    assert hot.stall > 0 and hot.latency >= hot.stall          # stall is a subset of latency
    assert hot.issue == max(0, hot.latency - hot.stall)


def test_taxonomy_softmax_is_vmcnt_dominated():
    d = rcv.analyze(FIX)
    bc = d["stall_taxonomy"]["by_class"]
    top = min(bc.items(), key=lambda kv: kv[1]["rank"])[0]
    assert top == "vmcnt"                                       # softmax is HBM-streaming
    assert bc["vmcnt"]["is_bubble"] is True
    assert d["totals"]["stall_pct"] > 50


def test_waitcnt_attribution_present():
    d = rcv.analyze(FIX)
    attributed = [h for h in d["hotspots"] if h.get("waitcnt_sources")]
    assert attributed, "expected at least one hotspot with a waitcnt->source edge"
    assert attributed[0]["waitcnt_sources"][0]["source"]       # the op it waits on has a source


def test_source_map_and_snippet():
    cache = rcv.load_source_map(FIX)
    assert cache, "snapshots.json should yield a source map"
    instrs = rcv.load_code(FIX)
    mapped = next(i for i in instrs if i.source and ":" in i.source)
    snip = rcv.snippet(cache, mapped.source)
    assert snip is not None and any(ln["hot"] for ln in snip["lines"])
