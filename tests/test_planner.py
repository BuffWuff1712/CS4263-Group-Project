import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from planner import ROOT, generate_candidate, load_instance, parse_schedule


class FakeClient:
    def __init__(self, text):
        self.requests = []
        self.text = text
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.text), finish_reason="stop")],
            usage=None,
        )


def test_easy_instance_has_expected_inputs():
    instance = load_instance(ROOT / "data" / "easy.json")
    assert len(instance["ports"]) == 3
    assert len(instance["vessels"]) == 2
    assert sum(item["quantity"] for item in instance["shipments"]) == 12


def test_parse_schedule_json():
    value = {"voyages": [{"vessel_id": "F1", "origin": "Shanghai", "destination": "Taicang", "depart_hour": 0, "cargo": []}]}
    schedule = parse_schedule(json.dumps(value))
    assert schedule.voyages[0].vessel_id == "F1"


def test_parse_schedule_rejects_malformed():
    with pytest.raises(ValidationError):
        parse_schedule('{"voyages": [}')


def test_generation_keeps_user_endpoint_parameters():
    fake = FakeClient('{"voyages": []}')
    instance = load_instance(ROOT / "data" / "easy.json")
    result = generate_candidate(fake, instance, model="qwen3.5:9b", system_prompt="Return JSON")
    request = fake.requests[0]
    assert request["model"] == "qwen3.5:9b"
    # Some models reject reasoning_effort, so it is only sent when requested.
    assert "reasoning_effort" not in request
    assert request["messages"][0] == {"role": "system", "content": "Return JSON"}
    assert result["schedule"] == {"voyages": []}
    # The structure parser is not a constraint validator: empty schedules will fail later.


def test_generation_sends_requested_reasoning_effort():
    fake = FakeClient('{"voyages": []}')
    instance = load_instance(ROOT / "data" / "easy.json")
    generate_candidate(fake, instance, model="qwen3.5:9b", reasoning="medium", system_prompt="Return JSON")
    assert fake.requests[0]["reasoning_effort"] == "medium"
