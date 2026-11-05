"""Check that --outline and --trace reach the model, using fake clients."""

import json
from types import SimpleNamespace

import pytest

import pipeline
import repair
from planner import ROOT, load_instance
from safeguards import SafeguardConfig
from schemas import Schedule
from simulator import simulate

EASY = ROOT / "data" / "easy.json"
VALID = json.loads((ROOT / "tests" / "fixtures" / "easy_valid_schedule.json").read_text())


class FakeClient:
    """Returns the queued replies in order and records every request."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def with_options(self, **kwargs):
        return self

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.replies.pop(0)), finish_reason="stop")],
            usage=None,
        )


def reply(outline, schedule):
    return f"OUTLINE:\n{outline}\nSCHEDULE:\n{json.dumps(schedule)}"


def user_input(request):
    return json.loads(request["messages"][1]["content"])


def test_config_rejects_unknown_outline_mode():
    with pytest.raises(ValueError):
        SafeguardConfig(outline="sometimes")


def test_safeguard_flag_only_sets_existing_safeguards():
    config = SafeguardConfig.from_flags(safeguard=True)
    assert config.grounded_feedback and config.output_recovery and config.delivery_preservation
    assert config.outline == "off" and not config.trace


def test_repair_sends_outline_and_trace_and_returns_new_outline(monkeypatch):
    instance = load_instance(EASY)
    previous = Schedule.model_validate({"voyages": []})
    report = simulate(instance, previous, with_trace=True)
    fake = FakeClient([reply("CHANGED: C1 via F2", VALID)])
    monkeypatch.setattr(repair, "get_client", lambda provider: fake)

    schedule, meta = repair.repair(
        instance, previous, report.violations, model="m", include_metadata=True,
        outline=True, previous_outline="old plan", trace=report.trace,
    )

    request = fake.requests[0]
    system = request["messages"][0]["content"]
    assert "SIMULATION TRACE" in system and "OUTPUT FORMAT OVERRIDE" in system
    sent = user_input(request)
    assert sent["previous_outline"] == "old plan"
    assert sent["simulation_trace"] == report.trace
    assert meta["outline"] == "CHANGED: C1 via F2"
    assert schedule.model_dump() == VALID


def test_repair_without_flags_is_unchanged(monkeypatch):
    instance = load_instance(EASY)
    previous = Schedule.model_validate({"voyages": []})
    fake = FakeClient([json.dumps(VALID)])
    monkeypatch.setattr(repair, "get_client", lambda provider: fake)

    repair.repair(instance, previous, [], model="m", previous_outline="ignored")

    request = fake.requests[0]
    assert "OUTPUT FORMAT OVERRIDE" not in request["messages"][0]["content"]
    assert "SIMULATION TRACE" not in request["messages"][0]["content"]
    assert set(user_input(request)) == {"problem_instance", "previous_schedule", "violations"}


@pytest.mark.parametrize("mode, expected_previous", [("rolling", "plan 0"), ("fresh", None)])
def test_pipeline_outline_modes(monkeypatch, mode, expected_previous):
    planner_client = FakeClient([reply("plan 0", {"voyages": []})])
    repair_client = FakeClient([reply("plan 1", VALID)])
    monkeypatch.setattr(pipeline.planner, "get_client", lambda provider: planner_client)
    monkeypatch.setattr(repair, "get_client", lambda provider: repair_client)

    result = pipeline.run_pipeline(EASY, provider="openai", model="m", outline=mode, trace=True)

    assert result["success"]
    assert result["planner_metadata"]["outline"] == "plan 0"
    assert result["planner_metadata"]["safeguards"]["outline"] == mode
    assert [a["outline"] for a in result["attempts"]] == ["plan 0", "plan 1"]
    sent = user_input(repair_client.requests[0])
    assert sent.get("previous_outline") == expected_previous
    assert "vessels" in sent["simulation_trace"]


def test_pipeline_rolling_keeps_outline_when_repair_is_rejected(monkeypatch):
    # The first repair loses C1's delivered units and is still invalid, so
    # delivery preservation rejects it; the outline must stay with the kept schedule.
    instance = load_instance(EASY)
    partial = {"voyages": [v for v in VALID["voyages"] if any(c["shipment_id"] == "C1" for c in v["cargo"])
                           or v["depart_hour"] == 0]}
    assert not simulate(instance, Schedule.model_validate(partial)).valid
    planner_client = FakeClient([reply("plan 0", partial)])
    repair_client = FakeClient([reply("bad plan", {"voyages": []}), reply("plan 2", VALID)])
    monkeypatch.setattr(pipeline.planner, "get_client", lambda provider: planner_client)
    monkeypatch.setattr(repair, "get_client", lambda provider: repair_client)

    result = pipeline.run_pipeline(EASY, provider="openai", model="m", safeguard=True, outline="rolling")

    assert result["success"]
    assert result["attempts"][0]["delivery_preservation"]["accepted"] is False
    assert user_input(repair_client.requests[1])["previous_outline"] == "plan 0"


def test_repair_asks_for_outline_in_user_message_only_when_on(monkeypatch):
    instance = load_instance(EASY)
    previous = Schedule.model_validate({"voyages": []})
    fake = FakeClient([reply("plan", VALID), json.dumps(VALID)])
    monkeypatch.setattr(repair, "get_client", lambda provider: fake)

    _, meta = repair.repair(instance, previous, [], model="m", include_metadata=True, outline=True)
    repair.repair(instance, previous, [], model="m")

    assert "output_format" in user_input(fake.requests[0])
    assert "output_format" not in user_input(fake.requests[1])
    assert meta["schedule_marker_found"] is True and "reply_start" not in meta


def test_format_retry_drops_output_format_and_records_missing_marker(monkeypatch):
    instance = load_instance(EASY)
    previous = Schedule.model_validate({"voyages": []})
    fake = FakeClient(["not a schedule", json.dumps(VALID)])
    monkeypatch.setattr(repair, "get_client", lambda provider: fake)
    monkeypatch.setattr(repair, "_save_failed_response", lambda content: "unused")

    _, meta = repair.repair(instance, previous, [], model="m", include_metadata=True,
                            outline=True, format_recovery=True)

    assert "output_format" in user_input(fake.requests[0])
    assert "output_format" not in user_input(fake.requests[1])
    assert meta["format_retries"] == 1
    assert meta["schedule_marker_found"] is False
    assert meta["reply_start"] == "not a schedule"


def test_planner_records_schedule_marker(monkeypatch):
    planner_client = FakeClient([reply("plan 0", VALID)])
    monkeypatch.setattr(pipeline.planner, "get_client", lambda provider: planner_client)

    result = pipeline.planner.generate_candidate(planner_client, load_instance(EASY), model="m", outline=True)

    assert result["schedule_marker_found"] is True


def test_repair_failed_result_has_planner_metadata(monkeypatch):
    planner_client = FakeClient([reply("plan 0", {"voyages": []})])
    repair_client = FakeClient(["not a schedule"])
    monkeypatch.setattr(pipeline.planner, "get_client", lambda provider: planner_client)
    monkeypatch.setattr(repair, "get_client", lambda provider: repair_client)
    monkeypatch.setattr(repair, "_save_failed_response", lambda content: "unused")

    result = pipeline.run_pipeline(EASY, provider="openai", model="m", outline="rolling")

    assert result["status"] == "repair_failed"
    assert result["planner_metadata"]["safeguards"]["outline"] == "rolling"
    assert result["planner_metadata"]["outline"] == "plan 0"
