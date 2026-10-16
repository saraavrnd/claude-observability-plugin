"""Pins the BheemBhai pricing evaluator (ADR-029) to the platform's vectors.

The platform owns vendor pricing as DATA: an admin keeps each vendor's rule in a
registry, the engine freezes the run's row at init and hands it to this hook in
``CC_LANGFUSE_PRICING_RULES``. This file is the fork's mirror of the platform's
``shared/bheembhai/pricing_rules.py`` reference — every vector here appears in
``tests/unit/shared/test_pricing_rules.py`` in the BheemBhai repository, and the
two must agree: a divergence means the editor's test box promises a label the
container would not produce.

The DeepSeek rules are a COPY of ``config/pricing/deepseek.json`` in that repo
(the digest below is asserted there too, so a change to the seed file is caught
on the platform side, where such a change is actually made).

Three seams beyond the vectors, because the vectors alone would not notice if
nothing ever called the evaluator:
  * the label lands in the generation metadata the SDK receives;
  * it is derived from THAT message's own model + timestamp;
  * the report file is written only when rules were evaluated — its ABSENCE is
    the engine's signal that nobody looked, so a silent step must never look
    like a cleared problem.
"""

import hashlib
import io
import json
from typing import Any, Dict, List

import pytest

#: Verbatim from BheemBhai's ``config/pricing/deepseek.json`` (the seeded
#: registry row). Keep in step with that file — the digest is asserted there.
DEEPSEEK_RULES: List[Dict[str, Any]] = [
    {
        "kind": "time_window",
        "key": "price_window",
        "tz": "UTC",
        "default": "peak",
        "date_overrides": {
            "2026-10-01": "off_peak",
            "2026-10-02": "off_peak",
        },
        "windows": [
            {"label": "off_peak", "days": [0, 1, 2, 3, 4], "from": "00:00", "to": "01:00"},
            {"label": "off_peak", "days": [0, 1, 2, 3, 4], "from": "04:00", "to": "06:00"},
            {"label": "off_peak", "days": [0, 1, 2, 3, 4], "from": "10:00", "to": "24:00"},
            {"label": "off_peak", "days": [5, 6], "from": "00:00", "to": "24:00"},
        ],
    }
]

#: sha256 of the canonical form above; the platform suite asserts the seed file
#: hashes to the same value.
DEEPSEEK_RULES_SHA256 = "928496109658cf3823eb835a7c8ed6d1965f4b672a3a0c46d2d8aac0484504ec"


def canonical(rules: Any) -> str:
    return json.dumps(rules, sort_keys=True, separators=(",", ":"))


@pytest.fixture(autouse=True)
def pricing(hook_module: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    """A clean pricing state per test: no rules, no report file, no errors."""
    monkeypatch.setattr(hook_module, "PRICING_RULES", [])
    monkeypatch.setattr(hook_module, "PRICING_REPORT_FILE", "")
    monkeypatch.setattr(hook_module, "PRICING_PROVIDER", "")
    monkeypatch.setattr(hook_module, "PRICING_RULES_VERSION", "")
    monkeypatch.setattr(hook_module, "_PRICING_REPORT", {"evaluated": 0, "errors": []})
    return hook_module


def _ctx(timestamp: Any) -> Dict[str, Any]:
    return {
        "model": "deepseek-flash",
        "timestamp": timestamp,
        "provider": "deepseek",
        "env": {},
        "metadata": {},
    }


def _deepseek_labels(hook_module: Any, timestamp: Any) -> Dict[str, str]:
    labels = hook_module.evaluate_pricing_rules(DEEPSEEK_RULES, _ctx(timestamp))
    assert hook_module._PRICING_REPORT["errors"] == []
    return labels


# ----------------- the copied seed -----------------

def test_the_seed_copy_is_this_fork_s_copy_of_the_registry_row(pricing: Any):
    assert pricing.evaluate_pricing_rules(DEEPSEEK_RULES, _ctx(None)) == {"price_window": "peak"}
    assert hashlib.sha256(canonical(DEEPSEEK_RULES).encode("utf-8")).hexdigest() == \
        DEEPSEEK_RULES_SHA256


# ----------------- DeepSeek boundary vectors (2026-10-16 is a Friday) -----------------

class TestDeepSeekVectors:
    def label(self, pricing: Any, timestamp: Any) -> str:
        return _deepseek_labels(pricing, timestamp)["price_window"]

    def test_weekday_off_peak_windows(self, pricing: Any):
        assert self.label(pricing, "2026-10-16T00:30:00.000Z") == "off_peak"  # [00:00,01:00)
        assert self.label(pricing, "2026-10-16T04:00:00Z") == "off_peak"      # start inclusive
        assert self.label(pricing, "2026-10-16T12:00:00Z") == "off_peak"      # [10:00,24:00)
        assert self.label(pricing, "2026-10-16T23:59:59Z") == "off_peak"      # "24:00" = end of day

    def test_weekday_peak_gaps(self, pricing: Any):
        assert self.label(pricing, "2026-10-16T01:00:00Z") == "peak"          # end exclusive
        assert self.label(pricing, "2026-10-16T02:00:00Z") == "peak"
        assert self.label(pricing, "2026-10-16T06:00:00Z") == "peak"
        assert self.label(pricing, "2026-10-16T09:59:59Z") == "peak"

    def test_straddling_the_peak_boundary(self, pricing: Any):
        assert self.label(pricing, "2026-10-16T09:55:00Z") == "peak"
        assert self.label(pricing, "2026-10-16T10:05:00Z") == "off_peak"

    def test_weekend_is_all_day_off_peak(self, pricing: Any):
        assert self.label(pricing, "2026-10-17T02:00:00Z") == "off_peak"      # Saturday
        assert self.label(pricing, "2026-10-18T23:00:00Z") == "off_peak"      # Sunday

    def test_date_override_wins_the_whole_day(self, pricing: Any):
        assert self.label(pricing, "2026-09-24T02:00:00Z") == "peak"          # Thursday
        assert self.label(pricing, "2026-10-01T02:00:00Z") == "off_peak"      # holiday override
        assert self.label(pricing, "2026-10-02T23:00:00Z") == "off_peak"

    def test_no_usable_timestamp_falls_to_default_and_is_not_an_error(self, pricing: Any):
        for timestamp in (None, "", "   ", "not-a-time", 20261016):
            labels = pricing.evaluate_pricing_rules(DEEPSEEK_RULES, _ctx(timestamp))
            assert labels["price_window"] == "peak", timestamp
            assert pricing._PRICING_REPORT["errors"] == [], timestamp

    def test_naive_timestamp_is_read_as_utc(self, pricing: Any):
        assert self.label(pricing, "2026-10-16T12:00:00") == "off_peak"


# ----------------- vendor-neutral vectors -----------------

class TestGenericVectors:
    def test_three_label_rule_in_a_non_utc_timezone(self, pricing: Any):
        rule = {
            "kind": "time_window", "key": "band", "tz": "Asia/Shanghai", "default": "high",
            "windows": [
                {"label": "low", "days": [0, 1, 2, 3, 4, 5, 6], "from": "00:00", "to": "08:00"},
                {"label": "mid", "days": [0, 1, 2, 3, 4, 5, 6], "from": "08:00", "to": "24:00"},
            ],
        }
        # 18:00 UTC = 02:00 next day in Shanghai.
        labels = pricing.evaluate_pricing_rules([rule], _ctx("2026-10-15T18:00:00Z"))
        assert pricing._PRICING_REPORT["errors"] == [] and labels["band"] == "low"
        # The Shanghai day boundary: 15:59:59Z = 23:59:59 CST -> mid.
        labels = pricing.evaluate_pricing_rules([rule], _ctx("2026-10-16T15:59:59Z"))
        assert labels["band"] == "mid"

    def test_first_matching_window_wins(self, pricing: Any):
        rule = {
            "kind": "time_window", "key": "band", "default": "high",
            "windows": [
                {"label": "first", "days": [4], "from": "00:00", "to": "24:00"},
                {"label": "second", "days": [4], "from": "00:00", "to": "24:00"},
            ],
        }
        labels = pricing.evaluate_pricing_rules([rule], _ctx("2026-10-16T12:00:00Z"))
        assert pricing._PRICING_REPORT["errors"] == [] and labels["band"] == "first"

    def test_day_filter_selects_the_window(self, pricing: Any):
        rule = {
            "kind": "time_window", "key": "band", "default": "high",
            "windows": [{"label": "only_monday", "days": [0], "from": "00:00", "to": "24:00"}],
        }
        assert pricing.evaluate_pricing_rules([rule], _ctx("2026-10-19T12:00:00Z")) == {
            "band": "only_monday"}                                            # Monday
        assert pricing.evaluate_pricing_rules([rule], _ctx("2026-10-16T12:00:00Z")) == {
            "band": "high"}                                                   # Friday

    def test_two_rules_write_two_keys(self, pricing: Any):
        rules = [
            {"kind": "time_window", "key": "a", "default": "expensive",
             "windows": [{"label": "weekend", "days": [5, 6], "from": "00:00", "to": "24:00"}]},
            {"kind": "time_window", "key": "b", "default": "cheap",
             "windows": [{"label": "cheap", "days": [4], "from": "00:00", "to": "24:00"}]},
        ]
        labels = pricing.evaluate_pricing_rules(rules, _ctx("2026-10-16T12:00:00Z"))
        assert pricing._PRICING_REPORT["errors"] == []
        assert labels == {"a": "expensive", "b": "cheap"}


# ----------------- the error channel -----------------

class TestEvaluationErrors:
    def test_no_rules_yields_no_labels(self, pricing: Any):
        assert pricing.evaluate_pricing_rules(None, _ctx(None)) == {}
        assert pricing.evaluate_pricing_rules([], _ctx(None)) == {}

    def test_unknown_kind_is_skipped_and_recorded(self, pricing: Any):
        # A rule newer than the deployed image: the key stays absent, so
        # Langfuse applies the model's default (most expensive) tier.
        rule = {"kind": "value_map", "key": "region", "default": "high"}
        labels = pricing.evaluate_pricing_rules([rule], _ctx("2026-10-16T12:00:00Z"))
        assert labels == {}
        errors = pricing._PRICING_REPORT["errors"]
        assert len(errors) == 1 and "value_map" in errors[0]

    def test_other_rules_still_evaluate_after_an_error(self, pricing: Any):
        rules = [
            {"kind": "value_map", "key": "region", "default": "high"},
            {"kind": "time_window", "key": "price_window", "tz": "UTC", "default": "peak",
             "windows": [{"label": "off_peak", "days": [4], "from": "10:00", "to": "24:00"}]},
        ]
        labels = pricing.evaluate_pricing_rules(rules, _ctx("2026-10-16T12:00:00Z"))
        assert labels == {"price_window": "off_peak"}
        assert len(pricing._PRICING_REPORT["errors"]) == 1

    def test_bad_timezone_is_an_error_not_a_label(self, pricing: Any):
        rule = {"kind": "time_window", "key": "price_window", "tz": "Nowhere/Land",
                "default": "peak", "windows": [{"label": "x", "days": [4], "from": "10:00", "to": "23:00"}]}
        labels = pricing.evaluate_pricing_rules([rule], _ctx("2026-10-16T12:00:00Z"))
        assert labels == {}
        errors = pricing._PRICING_REPORT["errors"]
        assert len(errors) == 1 and "timezone" in errors[0]

    def test_evaluate_rule_raises_on_unknown_kind(self, pricing: Any):
        with pytest.raises(ValueError):
            pricing.evaluate_pricing_rule({"kind": "nope", "key": "k", "default": "x"}, _ctx(None))


# ----------------- the env contract -----------------

class TestRuleParsing:
    def test_valid_json_array_parses(self, pricing: Any):
        assert pricing.parse_pricing_rules(canonical(DEEPSEEK_RULES)) == DEEPSEEK_RULES

    def test_absent_or_blank_is_no_rules(self, pricing: Any):
        assert pricing.parse_pricing_rules("") == []
        assert pricing.parse_pricing_rules("   ") == []
        assert pricing.parse_pricing_rules(None) == []

    def test_unusable_input_is_dropped_not_fatal(self, pricing: Any):
        assert pricing.parse_pricing_rules("not json") == []
        assert pricing.parse_pricing_rules('{"kind": "time_window"}') == []   # object, not array
        assert pricing.parse_pricing_rules('[1, {"kind": "x", "key": "k"}]') == [
            {"kind": "x", "key": "k"}]


# ----------------- the seams -----------------

class TestStamping:
    def test_no_rules_stamp_nothing_at_all(self, pricing: Any):
        metadata: Dict[str, Any] = {"assistant_index": 0}
        pricing.stamp_pricing_metadata({"model": "m", "timestamp": "2026-10-16T12:00:00Z"}, metadata)
        assert metadata == {"assistant_index": 0}
        assert pricing._PRICING_REPORT["evaluated"] == 0

    def test_the_label_lands_in_the_generation_metadata(self, pricing: Any,
                                                        monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(pricing, "PRICING_RULES", DEEPSEEK_RULES)
        monkeypatch.setattr(pricing, "PRICING_PROVIDER", "deepseek")
        metadata: Dict[str, Any] = {"assistant_index": 3, "tool_count": 1}
        pricing.stamp_pricing_metadata(
            {"model": "deepseek-flash", "timestamp": "2026-10-16T12:00:00Z"}, metadata)
        assert metadata["price_window"] == "off_peak"
        assert metadata["assistant_index"] == 3          # nothing else touched
        assert pricing._PRICING_REPORT["evaluated"] == 1

    def test_every_rule_error_reaches_the_report(self, pricing: Any,
                                                 monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(pricing, "PRICING_RULES", [{"kind": "value_map", "key": "region"}])
        metadata: Dict[str, Any] = {}
        pricing.stamp_pricing_metadata({"model": "m", "timestamp": None}, metadata)
        assert metadata == {}                            # key absent -> default tier
        assert len(pricing._PRICING_REPORT["errors"]) == 1


class TestTheGenerationSeam:
    def test_a_walked_generation_carries_the_label_of_its_own_timestamp(
        self, pricing: Any, fixture_transcript_path: Any, read_fixture_jsonl: Any,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """The vectors above prove the evaluator; this proves a generation the
        hook is about to emit actually gets the label — derived from THAT
        message's model and timestamp, not the session's start."""
        rows = read_fixture_jsonl(fixture_transcript_path("simple_turn"))
        turn = pricing.build_turns(rows)[0]
        # A transcript row: timestamp at the top level, model nested in
        # `message` — the evaluator's facts come from where the hook itself
        # reads them, so a copy must be deep.
        message = json.loads(json.dumps(turn.assistant_msgs[0]))
        message["timestamp"] = "2026-10-16T12:00:00.000Z"     # Friday, off-peak
        message["message"]["model"] = "deepseek-flash"
        monkeypatch.setattr(pricing, "PRICING_RULES", DEEPSEEK_RULES)
        monkeypatch.setattr(pricing, "PRICING_PROVIDER", "deepseek")

        generation_kwargs, _ = pricing.build_generation_kwargs(0, message, "do it", [], [])

        assert generation_kwargs["model"] == "deepseek-flash"
        assert generation_kwargs["metadata"]["price_window"] == "off_peak"

    def test_a_peak_hour_message_is_labelled_peak(
        self, pricing: Any, fixture_transcript_path: Any, read_fixture_jsonl: Any,
        monkeypatch: pytest.MonkeyPatch,
    ):
        rows = read_fixture_jsonl(fixture_transcript_path("simple_turn"))
        turn = pricing.build_turns(rows)[0]
        message = json.loads(json.dumps(turn.assistant_msgs[0]))
        message["timestamp"] = "2026-10-16T02:00:00.000Z"     # Friday 02:00 = peak gap
        monkeypatch.setattr(pricing, "PRICING_RULES", DEEPSEEK_RULES)

        generation_kwargs, _ = pricing.build_generation_kwargs(0, message, "do it", [], [])

        assert generation_kwargs["metadata"]["price_window"] == "peak"


class TestTheMainSeam:
    def test_main_writes_the_report_on_its_way_out(
        self, pricing: Any, fake_langfuse: Any, isolated_hook_state: Any,
        fixture_transcript_path: Any, monkeypatch: pytest.MonkeyPatch,
    ):
        """The end of the contract: main() runs, generations get labelled, and
        the record exists when the process exits. Every other test here would
        stay green if main() never called the writer — and the engine would see
        a step that looked like nobody ever priced it."""
        report = isolated_hook_state / "pricing_report.jsonl"
        monkeypatch.setattr(pricing, "PRICING_RULES", DEEPSEEK_RULES)
        monkeypatch.setattr(pricing, "PRICING_PROVIDER", "deepseek")
        monkeypatch.setattr(pricing, "PRICING_REPORT_FILE", str(report))
        monkeypatch.setattr(pricing, "get_langfuse_config", lambda: pricing.LangfuseConfig(
            "public", "secret", "https://example.test", "user-1"))
        monkeypatch.setattr(pricing, "create_langfuse_client", lambda _config: fake_langfuse)
        monkeypatch.setattr(pricing, "flush_and_shutdown_langfuse_client", lambda _client: None)
        payload = json.dumps({
            "session_id": "pricing",
            "transcript_path": str(fixture_transcript_path("simple_turn")),
            "hook_event_name": "Stop",
        })
        monkeypatch.setattr("sys.stdin", io.StringIO(payload))

        assert pricing.main() == 0

        records = [json.loads(line) for line in report.read_text().splitlines()]
        assert len(records) == 1
        assert records[0]["evaluated"] >= 1        # it really walked generations
        assert records[0]["errors"] == []


class TestTheReportFile:
    def test_one_json_line_per_invocation(self, pricing: Any, tmp_path: Any,
                                          monkeypatch: pytest.MonkeyPatch):
        report = tmp_path / "pricing_report.jsonl"
        monkeypatch.setattr(pricing, "PRICING_RULES", DEEPSEEK_RULES)
        monkeypatch.setattr(pricing, "PRICING_PROVIDER", "deepseek")
        monkeypatch.setattr(pricing, "PRICING_RULES_VERSION", "7")
        monkeypatch.setattr(pricing, "PRICING_REPORT_FILE", str(report))

        for _ in range(2):
            pricing.stamp_pricing_metadata(
                {"model": "deepseek-flash", "timestamp": "2026-10-16T12:00:00Z"}, {})

        pricing.write_pricing_report()

        lines = report.read_text().splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0]) == {
            "provider": "deepseek", "rules_version": "7", "evaluated": 2, "errors": [],
        }

    def test_a_second_invocation_appends_never_erases(self, pricing: Any, tmp_path: Any,
                                                      monkeypatch: pytest.MonkeyPatch):
        """The hook runs many times per session: one invocation's clean report
        must not wipe an earlier one's error, which is the whole reason the
        engine can trust a step whose last turn happened to be fine."""
        report = tmp_path / "pricing_report.jsonl"
        monkeypatch.setattr(pricing, "PRICING_REPORT_FILE", str(report))

        monkeypatch.setattr(pricing, "PRICING_RULES", [{"kind": "value_map", "key": "region"}])
        pricing.stamp_pricing_metadata({"model": "m", "timestamp": None}, {})
        pricing.write_pricing_report()

        monkeypatch.setattr(pricing, "_PRICING_REPORT", {"evaluated": 0, "errors": []})
        monkeypatch.setattr(pricing, "PRICING_RULES", DEEPSEEK_RULES)
        pricing.stamp_pricing_metadata(
            {"model": "deepseek-flash", "timestamp": "2026-10-16T12:00:00Z"}, {})
        pricing.write_pricing_report()

        records = [json.loads(line) for line in report.read_text().splitlines()]
        assert len(records) == 2
        assert len(records[0]["errors"]) == 1        # still there
        assert records[1]["errors"] == []

    def test_without_rules_there_is_no_file(self, pricing: Any, tmp_path: Any,
                                            monkeypatch: pytest.MonkeyPatch):
        """ABSENCE is the engine's signal that nobody looked — a run without
        rules must not leave an empty report behind that reads as 'all clean'."""
        report = tmp_path / "pricing_report.jsonl"
        monkeypatch.setattr(pricing, "PRICING_REPORT_FILE", str(report))
        pricing.write_pricing_report()
        assert not report.exists()

    def test_without_a_report_path_nothing_is_written(self, pricing: Any,
                                                      monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(pricing, "PRICING_RULES", DEEPSEEK_RULES)
        pricing.write_pricing_report()   # no path configured: must not raise

    def test_rules_that_were_never_used_write_no_file(self, pricing: Any, tmp_path: Any,
                                                      monkeypatch: pytest.MonkeyPatch):
        """Rules present, but this invocation walked no assistant message (a
        PreToolUse-only hook run): nothing was evaluated, so there is nothing to
        report."""
        report = tmp_path / "pricing_report.jsonl"
        monkeypatch.setattr(pricing, "PRICING_RULES", DEEPSEEK_RULES)
        monkeypatch.setattr(pricing, "PRICING_REPORT_FILE", str(report))
        pricing.write_pricing_report()
        assert not report.exists()

    def test_an_unwritable_path_is_never_fatal(self, pricing: Any,
                                               monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(pricing, "PRICING_RULES", DEEPSEEK_RULES)
        monkeypatch.setattr(pricing, "PRICING_REPORT_FILE", "/nonexistent-dir/report.jsonl")
        pricing.stamp_pricing_metadata(
            {"model": "deepseek-flash", "timestamp": "2026-10-16T12:00:00Z"}, {})
        pricing.write_pricing_report()   # must not raise
