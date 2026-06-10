"""One humanize round: rank-1 target -> wiki -> attempt skeleton (wiki mocked, no GPU)."""
import json
from pathlib import Path

import pytest

from flyprof.envelope import FlyprofError
from flyprof.optimize import NEXT_STEPS, run_optimize

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "flash_attn_fwd" / "source" / "report.json"

WIKI_OK = {
    "ok": True, "bubble_class": "occupancy",
    "query": "occupancy vgpr pressure flash attn fwd",
    "results": [{"page_id": "technique-vgpr-budgeting", "title": "VGPR Budgeting",
                 "path": "wiki/techniques/vgpr-budgeting.md", "score": 1.0}],
    "technique_hint": "technique-vgpr-budgeting",
    "implemented_by_prs": ["pr-flydsl-629"], "warnings": [],
}


def _report() -> dict:
    return json.loads(EXAMPLE.read_text())


def _ledger(path: Path) -> list:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_round_skeleton_targets_rank1_and_appends(tmp_path):
    attempts = tmp_path / "attempts.jsonl"
    calls = []

    def fake_wiki(bubble_class, **kw):
        calls.append((bubble_class, kw))
        return dict(WIKI_OK)

    data, warnings = run_optimize(_report(), bundle="/b/step0", attempts_path=attempts,
                                  wiki_fn=fake_wiki, arch="gfx950")
    # rank-1 recommendation, not the rank-1 stall class
    assert data["round"] == 1
    assert data["target"]["rule"] == "occupancy_vgpr" and data["target"]["bubble_class"] == "occupancy"
    assert calls[0][0] == "occupancy" and calls[0][1]["kernel"] == "flash attn fwd"
    assert data["wiki"]["technique"] == "technique-vgpr-budgeting"
    assert data["wiki"]["implemented_by_prs"] == ["pr-flydsl-629"]
    assert data["next_steps"] == NEXT_STEPS and not warnings

    (rec,) = _ledger(attempts)
    assert rec["round"] == 1 and rec["status"] == "proposed" and rec["bundle"] == "/b/step0"
    assert rec["flyprof_before"] == {"stall_pct": 60.6, "waves_per_cu": 4, "arch_vgpr": 249}
    # the agent's half of the contract is left empty
    assert all(rec[k] is None for k in
               ("hypothesis", "diff", "predicted_delta", "flyprof_after", "verdict", "reason"))


def test_round_number_increments_and_prior_reject_warns(tmp_path):
    attempts = tmp_path / "attempts.jsonl"
    prior = {"round": 3, "target": {"rule": "occupancy_vgpr"}, "verdict": "REJECT",
             "reason": "scratch>0 (accum spill)"}
    attempts.write_text(json.dumps(prior) + "\nnot json\n")

    data, warnings = run_optimize(_report(), bundle="/b", attempts_path=attempts,
                                  wiki_fn=lambda *a, **k: dict(WIKI_OK))
    assert data["round"] == 4
    assert data["prior_attempts_on_rule"] == [{"round": 3, "verdict": "REJECT",
                                               "reason": "scratch>0 (accum spill)"}]
    assert any("REJECTED in round 3" in w for w in warnings)


def test_wiki_failure_degrades_to_warning_not_error(tmp_path):
    attempts = tmp_path / "attempts.jsonl"
    fail = {"ok": False, "error": {"code": "WIKI_MISSING", "exitCode": 69, "message": "no wiki", "help": "clone"}}
    data, warnings = run_optimize(_report(), bundle="/b", attempts_path=attempts,
                                  wiki_fn=lambda *a, **k: fail)
    assert data["wiki"]["error"]["code"] == "WIKI_MISSING"
    assert any("knob-table priors" in w for w in warnings)
    assert _ledger(attempts)[0]["wiki"] == {"error": fail["error"]}  # round still recorded


def test_optimal_report_stops_the_loop(tmp_path):
    report = _report()
    report["optimal"] = True
    with pytest.raises(FlyprofError) as ei:
        run_optimize(report, bundle="/b", attempts_path=tmp_path / "a.jsonl",
                     wiki_fn=lambda *a, **k: dict(WIKI_OK))
    assert ei.value.code == "NO_RECOMMENDATION" and ei.value.exit_code == 66
    assert not (tmp_path / "a.jsonl").exists()


def test_no_recommendations_is_empty_not_crash(tmp_path):
    report = _report()
    report["recommendations"] = []
    with pytest.raises(FlyprofError) as ei:
        run_optimize(report, bundle="/b", attempts_path=tmp_path / "a.jsonl",
                     wiki_fn=lambda *a, **k: dict(WIKI_OK))
    assert ei.value.code == "NO_RECOMMENDATION"
