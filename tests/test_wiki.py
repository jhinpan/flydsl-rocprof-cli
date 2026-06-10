"""bubble_class -> wiki query mapping + graceful degradation (subprocess mocked)."""
import subprocess

from flyprof import wiki

COMPACT_OUT = """# 3 result(s)

  [wiki-kernel] kernel-flydsl-flash-attention: FlyDSL Flash Attention — gfx950 dual-wave  (wiki/kernels/flydsl-flash-attention.md)
        ↳ …snippet text that must be ignored…
  [wiki-technique] technique-vgpr-budgeting: VGPR Budgeting  (wiki/techniques/vgpr-budgeting.md)
  [wiki-pattern] pattern-vgpr-pressure: VGPR Pressure and Occupancy Collapse  (wiki/patterns/vgpr-pressure.md)
"""

FRONTMATTER_OUT = """id: technique-vgpr-budgeting
title: VGPR Budgeting
implemented_by:
- pr-flydsl-629
- pr-flydsl-462
"""


def _fake_wiki(tmp_path):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "query.py").write_text("")
    (tmp_path / "scripts" / "get_page.py").write_text("")
    return str(tmp_path)


def _completed(cmd, stdout, rc=0):
    return subprocess.CompletedProcess(cmd, rc, stdout=stdout, stderr="")


def test_occupancy_maps_to_vgpr_budgeting_and_parses_results(tmp_path, monkeypatch):
    calls = []

    def fake_run(cmd, timeout):
        calls.append(cmd)
        out = FRONTMATTER_OUT if "get_page.py" in cmd[1] else COMPACT_OUT
        return _completed(cmd, out)

    monkeypatch.setattr(wiki, "_run", fake_run)
    out = wiki.run_wiki("occupancy", wiki_root=_fake_wiki(tmp_path))
    assert out["ok"] is True
    assert "vgpr" in out["query"] and "flash attention" in out["query"]
    assert out["technique_hint"] == "technique-vgpr-budgeting"
    assert out["implemented_by_prs"] == ["pr-flydsl-629", "pr-flydsl-462"]
    assert [r["page_id"] for r in out["results"]] == [
        "kernel-flydsl-flash-attention", "technique-vgpr-budgeting", "pattern-vgpr-pressure"]
    assert out["results"][0]["title"].startswith("FlyDSL Flash Attention")
    scores = [r["score"] for r in out["results"]]
    assert scores == sorted(scores, reverse=True) and scores[0] == 1.0
    # query.py is called with the arch filter and compact output
    assert "--architecture" in calls[0] and "--compact" in calls[0]


def test_every_seeded_bubble_has_real_query_and_technique():
    for cls, seed in wiki.BUBBLE_TECHNIQUES.items():
        assert seed["keywords"] and seed["technique"].startswith("technique-"), cls


def test_unknown_bubble_falls_back_to_literal_query(tmp_path, monkeypatch):
    monkeypatch.setattr(wiki, "_run", lambda cmd, timeout: _completed(cmd, "No matching pages.\n"))
    out = wiki.run_wiki("flat_scratch", wiki_root=_fake_wiki(tmp_path))
    assert out["ok"] is True
    assert out["query"].startswith("flat_scratch stall")
    assert out["technique_hint"] is None and out["implemented_by_prs"] == []
    assert out["results"] == [] and out["warnings"]


def test_missing_wiki_degrades_to_unavailable(tmp_path):
    out = wiki.run_wiki("barrier", wiki_root=str(tmp_path / "nope"))
    assert out["ok"] is False
    assert out["error"]["code"] == "WIKI_MISSING"
    assert out["error"]["exitCode"] == 69
    assert "FLYPROF_WIKI_ROOT" in out["error"]["help"]


def test_timeout_degrades_to_tempfail(tmp_path, monkeypatch):
    def boom(cmd, timeout):
        raise subprocess.TimeoutExpired(cmd, timeout)

    monkeypatch.setattr(wiki, "_run", boom)
    out = wiki.run_wiki("vmcnt", wiki_root=_fake_wiki(tmp_path), timeout=1)
    assert out["ok"] is False and out["error"]["exitCode"] == 75


def test_query_failure_is_internal_not_raise(tmp_path, monkeypatch):
    monkeypatch.setattr(wiki, "_run", lambda cmd, timeout: _completed(cmd, "", rc=2))
    out = wiki.run_wiki("lds", wiki_root=_fake_wiki(tmp_path))
    assert out["ok"] is False and out["error"]["code"] == "WIKI_QUERY_FAILED"
