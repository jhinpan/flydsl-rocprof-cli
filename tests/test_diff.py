"""Offline before/after diff of two bundles' report.json (no GPU)."""
import copy
import json
from pathlib import Path

import pytest

from flyprof.diff import diff_reports, run_diff
from flyprof.envelope import FlyprofError

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "flash_attn_fwd" / "source" / "report.json"


def _baseline() -> dict:
    return json.loads(EXAMPLE.read_text())


def _improved() -> dict:
    """The plan's Iteration-1 outcome: barrier shrinks, occupancy rises, VGPRs drop."""
    after = copy.deepcopy(_baseline())
    bc = after["stall_taxonomy"]["by_class"]
    bc["barrier"].update(cycles=56000, pct=12.0, rank=2)   # 20.84% -> 12.0%, drops behind valu
    bc["valu"]["rank"] = 1
    after["totals"]["stall_pct"] = 52.0
    after["occupancy"].update(from_capture_waves_per_cu=8, arch_vgpr=230)
    return after


def _write_bundle(tmp_path: Path, name: str, report: dict) -> Path:
    d = tmp_path / name
    d.mkdir()
    (d / "report.json").write_text(json.dumps(report))
    return d


def test_rank1_bubble_shrank_with_deltas(tmp_path):
    before = _write_bundle(tmp_path, "before", _baseline())
    after = _write_bundle(tmp_path, "after", _improved())
    data = run_diff(before, after)

    v = data["verdict"]
    assert v["rank1_bubble_class"] == "barrier"            # BEFORE rank-1 bubble, not AFTER's
    assert v["rank1_bubble_shrank"] is True
    assert v["before_pct"] == 20.84 and v["after_pct"] == 12.0 and v["delta_pct"] == -8.84

    bar = data["stall_taxonomy"]["barrier"]
    assert bar["is_bubble"] is True
    assert bar["delta"] == {"cycles": 56000 - 98732, "pct": -8.84, "rank": 1}

    assert data["totals"]["stall_pct"]["delta"] == pytest.approx(-8.6)
    assert data["occupancy"]["waves_per_cu"] == {"before": 4, "after": 8, "delta": 4}
    assert data["occupancy"]["arch_vgpr"] == {"before": 249, "after": 230, "delta": -19}
    assert data["occupancy"]["accum_vgpr"]["delta"] == 0
    assert data["bound_type"] == {"before": "memory", "after": "memory", "changed": False}
    assert "shrank" in data["headline"]


def test_rank1_bubble_grew_is_a_negative_verdict():
    after = _baseline()
    after["stall_taxonomy"]["by_class"]["barrier"]["pct"] = 25.0
    data = diff_reports(_baseline(), after)
    assert data["verdict"]["rank1_bubble_shrank"] is False
    assert data["verdict"]["delta_pct"] == pytest.approx(4.16)


def test_class_missing_in_after_counts_as_fully_drained():
    after = _baseline()
    del after["stall_taxonomy"]["by_class"]["barrier"]
    data = diff_reports(_baseline(), after)
    assert data["verdict"]["rank1_bubble_shrank"] is True and data["verdict"]["after_pct"] == 0.0
    bar = data["stall_taxonomy"]["barrier"]                 # union of classes keeps it visible
    assert bar["after"] is None and bar["delta"]["pct"] is None


def test_missing_report_is_missing_prereq(tmp_path):
    before = _write_bundle(tmp_path, "before", _baseline())
    with pytest.raises(FlyprofError) as ei:
        run_diff(before, tmp_path / "after")
    assert ei.value.code == "MISSING_PREREQ"
