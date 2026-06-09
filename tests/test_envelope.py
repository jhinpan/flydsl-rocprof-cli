"""The output contract: envelope shapes, exit-code binding, error mapping."""
from flyprof.envelope import (
    EXIT_EMPTY,
    EXIT_INTERNAL,
    EXIT_UNAVAILABLE,
    FlyprofError,
    Result,
    error_envelope,
    exit_code_of,
    ok_envelope,
)


def test_ok_envelope_shape():
    env = ok_envelope("tile", Result(data={"a": 1}, bundle="/b", warnings=["w"]))
    assert env["ok"] is True
    assert env["command"] == "tile"
    assert env["bundle"] == "/b"
    assert env["data"] == {"a": 1}
    assert env["warnings"] == ["w"]
    assert exit_code_of(env) == 0


def test_error_envelope_binds_exit_code():
    env = error_envelope("capture", FlyprofError("EMPTY_ATT", "all empty", available=[0, 1]))
    assert env["ok"] is False
    assert env["error"]["code"] == "EMPTY_ATT"
    assert env["error"]["exitCode"] == EXIT_EMPTY
    assert env["error"]["available"] == [0, 1]
    assert exit_code_of(env) == EXIT_EMPTY


def test_unavailable_is_stop_class():
    env = error_envelope("doctor", FlyprofError("ROCPROFV3_MISSING", "no rocprofv3"))
    assert env["error"]["exitCode"] == EXIT_UNAVAILABLE


def test_unknown_code_falls_back_to_internal():
    err = FlyprofError("NOT_A_REAL_CODE", "boom")
    assert err.code == "INTERNAL"
    assert err.exit_code == EXIT_INTERNAL
    assert err.extra.get("unmapped_code") == "NOT_A_REAL_CODE"


def test_help_defaults_from_registry():
    err = FlyprofError("BUILD_TREE_MISSING", "x")
    assert "worktree" in err.help.lower()
