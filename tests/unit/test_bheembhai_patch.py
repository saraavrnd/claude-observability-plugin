"""Pins the BheemBhai patch: the four surfaces upstream has no injection point
for — trace name, trace version, lineage metadata, and secret masking.

Masking is the load-bearing one: upstream has none, and its truncation funnel
only inspects STRING tool inputs, so a dict-shaped payload (Write file content,
Edit strings, env dumps) reaches the trace raw. The contract here is that
masking runs BEFORE any length or digest is computed (a digest of a raw secret
is itself a leak), and that the dict path is deep-masked."""

import contextlib
import hashlib
from typing import Any

import pytest


# ----------------- masking: token shapes -----------------

def test_each_recognized_token_shape_is_masked(hook_module):
    cases = [
        ("sk-ant-api03-AAAAAAAAAAAAAAAAAAAAAAAA", "anthropic-key"),
        ("sk-lf-00000000-1111-2222-3333-444444444444", "langfuse-key"),
        ("pk-lf-00000000-1111-2222-3333-444444444444", "langfuse-key"),
        ("ghp_" + "A" * 36, "github-token"),
        ("github_pat_" + "A" * 22, "github-pat"),
        ("ATATT" + "A" * 20, "atlassian-token"),
        ("AKIAIOSFODNN7EXAMPLE", "aws-access-key-id"),
        ("sk-" + "a" * 24, "api-key"),
    ]
    for token, kind in cases:
        masked = hook_module.mask_secrets(f"token is {token} ok")
        assert token not in masked, token
        assert f"[REDACTED:{kind}]" in masked, token


def test_a_pem_private_key_block_is_masked_across_lines(hook_module):
    pem = "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA\n-----END RSA PRIVATE KEY-----"
    masked = hook_module.mask_secrets(f"found:\n{pem}\ndone")
    assert "PRIVATE KEY" not in masked
    assert masked.endswith("done")


def test_presigned_url_credentials_are_masked_but_the_url_survives(hook_module):
    url = (
        "https://bucket.s3.amazonaws.com/key"
        "?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Signature=deadbeef0123&X-Amz-Expires=3600"
    )
    masked = hook_module.mask_secrets(url)
    assert "deadbeef0123" not in masked
    assert "X-Amz-Expires=3600" in masked
    assert "https://bucket.s3.amazonaws.com/key" in masked


def test_env_assignments_keep_the_name_and_drop_the_value(hook_module):
    masked = hook_module.mask_secrets('ANTHROPIC_API_KEY="sk-ant-secretvalue123"\nJIRA_API_TOKEN=abcdefgh12345')
    assert 'ANTHROPIC_API_KEY="[REDACTED:env-assignment]"' in masked
    assert "JIRA_API_TOKEN=[REDACTED:env-assignment]" in masked
    assert "secretvalue123" not in masked
    assert "abcdefgh12345" not in masked


def test_masking_is_idempotent(hook_module):
    once = hook_module.mask_secrets("key sk-ant-abcdefghijkl end")
    assert hook_module.mask_secrets(once) == once


def test_ordinary_text_is_untouched(hook_module):
    text = "ran pytest, 128 passed; changed src/api.py for LNPRTL-42"
    assert hook_module.mask_secrets(text) == text


def test_mask_secrets_in_value_walks_dicts_and_lists(hook_module):
    value = {
        "content": "sk-ant-aaaaaaaaaaaaaaaa",
        "nested": ["ok", {"cmd": "export GH_TOKEN=" + "a" * 40}],
        "count": 5,
        "flag": True,
    }
    masked = hook_module.mask_secrets_in_value(value)
    assert "[REDACTED:anthropic-key]" == masked["content"]
    # The env-assignment rule runs before the bare-token shapes, keeping the
    # name and eating the value.
    assert masked["nested"][1]["cmd"] == "export GH_TOKEN=[REDACTED:env-assignment]"
    assert masked["nested"][0] == "ok"
    assert masked["count"] == 5
    assert masked["flag"] is True
    # The caller's value is not mutated in place.
    assert value["content"] == "sk-ant-aaaaaaaaaaaaaaaa"


# ----------------- masking: the truncation funnel -----------------

def test_truncate_text_masks_the_payload(hook_module):
    token = "sk-ant-" + "a" * 40
    text, meta = hook_module.truncate_text(f"before {token} after")
    assert token not in text
    assert "[REDACTED:anthropic-key]" in text
    assert meta["truncated"] is False


def test_truncation_measures_the_masked_payload(hook_module):
    token = "ghp_" + "A" * 40
    raw = "leading text " + token
    text, meta = hook_module.truncate_text(raw, max_chars=5)
    masked = "leading text [REDACTED:github-token]"
    assert text == "leadi"
    assert meta["orig_len"] == len(masked)
    assert meta["kept_len"] == 5
    # The digest is of the MASKED payload: a digest of the raw secret would be
    # a leak vector of its own.
    assert meta["sha256"] == hashlib.sha256(masked.encode("utf-8")).hexdigest()


def test_a_secret_straddling_the_cut_is_never_partially_leaked(hook_module):
    token = "sk-ant-" + "a" * 40
    text, _ = hook_module.truncate_text(token, max_chars=10)
    assert text == "[REDACTED:"


# ----------------- masking: the dict-shaped tool input -----------------

def test_a_dict_tool_input_is_deep_masked(hook_module):
    tool_use = {"name": "Write", "input": {"file_path": ".env", "content": "ANTHROPIC_API_KEY=secretvalue123"}}
    value, meta = hook_module.get_tool_input_for_observation(tool_use)
    assert meta is None
    assert value["file_path"] == ".env"
    assert "secretvalue123" not in value["content"]
    assert "ANTHROPIC_API_KEY" in value["content"]


def test_a_plain_dict_tool_input_passes_through_unchanged(hook_module):
    tool_use = {"name": "Read", "input": {"file_path": "src/api.py", "offset": 10}}
    value, meta = hook_module.get_tool_input_for_observation(tool_use)
    assert value == {"file_path": "src/api.py", "offset": 10}
    assert meta is None


def test_a_string_tool_input_is_masked_through_the_funnel(hook_module):
    tool_use = {"name": "Bash", "input": "curl -H 'Authorization: Bearer sk-ant-aaaaaaaaaaaaaaaaaaaa'"}
    value, _ = hook_module.get_tool_input_for_observation(tool_use)
    assert "sk-ant-" not in value


# ----------------- masking: status messages -----------------

def test_status_messages_are_masked(hook_module):
    message = hook_module.build_status_message("API Error: 401 invalid key sk-ant-aaaaaaaaaaaaaaaaaaaa", "fallback")
    assert "sk-ant-" not in message
    assert message.startswith("API Error")


# ----------------- trace metadata / name / version -----------------

def test_trace_metadata_parses_a_json_object(hook_module):
    assert hook_module.parse_trace_metadata('{"bb_project_id": "p1", "visit_no": "2"}') == {
        "bb_project_id": "p1",
        "visit_no": "2",
    }


def test_trace_metadata_tolerates_bad_input(hook_module):
    assert hook_module.parse_trace_metadata("") is None
    assert hook_module.parse_trace_metadata("   ") is None
    assert hook_module.parse_trace_metadata("not json") is None
    assert hook_module.parse_trace_metadata('["a"]') is None
    assert hook_module.parse_trace_metadata('"str"') is None


def test_trace_attributes_default_to_upstream(hook_module, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(hook_module, "TRACE_NAME_OVERRIDE", "")
    monkeypatch.setattr(hook_module, "TRACE_VERSION", "")
    monkeypatch.setattr(hook_module, "TRACE_METADATA", None)
    assert hook_module.trace_attribute_kwargs() == {"trace_name": "Claude Code Turn"}


def test_trace_attributes_carry_the_engine_surfaces(hook_module, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(hook_module, "TRACE_NAME_OVERRIDE", "implement")
    monkeypatch.setattr(hook_module, "TRACE_VERSION", "3")
    monkeypatch.setattr(hook_module, "TRACE_METADATA", {"exec_id": "r:s:1:1"})
    assert hook_module.trace_attribute_kwargs() == {
        "trace_name": "implement",
        "version": "3",
        "metadata": {"exec_id": "r:s:1:1"},
    }


# ----------------- the propagation seam -----------------

def test_emit_turn_propagates_the_trace_surfaces(
    hook_module: Any,
    fake_langfuse: Any,
    fixture_transcript_path: Any,
    read_fixture_jsonl: Any,
    monkeypatch: pytest.MonkeyPatch,
):
    rows = read_fixture_jsonl(fixture_transcript_path("simple_turn"))
    turn = hook_module.build_turns(rows)[0]
    captured: dict = {}

    @contextlib.contextmanager
    def recorder(**kwargs: Any):
        captured.update(kwargs)
        yield

    monkeypatch.setattr(hook_module, "propagate_attributes", recorder)
    monkeypatch.setattr(hook_module, "TRACE_NAME_OVERRIDE", "implement")
    monkeypatch.setattr(hook_module, "TRACE_VERSION", "3")
    monkeypatch.setattr(hook_module, "TRACE_METADATA", {"exec_id": "r:s:1:1"})

    hook_module.emit_turn(
        fake_langfuse,
        "session-1",
        1,
        turn,
        fixture_transcript_path("simple_turn"),
        trace_seed="seed-1",
    )

    assert captured["trace_name"] == "implement"
    assert captured["version"] == "3"
    assert captured["metadata"] == {"exec_id": "r:s:1:1"}
    assert captured["session_id"] == "session-1"
