import json

from planner import ROOT, load_instance
from schemas import Schedule
from simulator import simulate


def easy():
    return load_instance(ROOT / "data" / "easy.json")


def schedule(voyages):
    return Schedule.model_validate({"voyages": voyages})


def test_trace_is_off_by_default():
    report = simulate(easy(), schedule([]))
    assert report.trace is None


def test_trace_lists_vessel_legs_in_time_order():
    sch = schedule([
        {"vessel_id": "F2", "origin": "Taicang", "destination": "Nantong", "depart_hour": 5, "cargo": [{"shipment_id": "C1", "quantity": 4}]},
        {"vessel_id": "F2", "origin": "Nantong", "destination": "Taicang", "depart_hour": 0, "cargo": []},
    ])
    trace = simulate(easy(), sch, with_trace=True).trace
    f2 = trace["vessels"]["F2"]
    assert f2["voyages_used"] == "2/3"
    assert f2["legs"] == [
        "voyage 1: Nantong h0 -> Taicang h5, empty",
        "voyage 0: Taicang h5 -> Nantong h10, carrying C1 x4",
    ]
    assert trace["vessels"]["F1"]["legs"] == []


def test_trace_groups_units_and_reports_status():
    sch = schedule([
        {"vessel_id": "F2", "origin": "Nantong", "destination": "Taicang", "depart_hour": 0, "cargo": [{"shipment_id": "C2", "quantity": 4}]},
    ])
    trace = simulate(easy(), sch, with_trace=True).trace
    c2 = trace["shipments"]["C2"]
    assert c2["delivered"] == 0
    moved, stayed = c2["groups"]
    assert moved["units"] == 4
    assert moved["legs"] == ["F2 (voyage 0): Nantong h0 -> Taicang h5"]
    assert moved["status"] == "NOT DELIVERED: waiting at Taicang since hour 5"
    assert stayed["units"] == 1
    assert stayed["legs"] == []
    assert stayed["status"] == "NOT DELIVERED: waiting at Nantong since hour 0"


def test_trace_marks_late_delivery():
    instance = easy()
    c1 = next(s for s in instance["shipments"] if s["id"] == "C1")
    c1["deadline_hour"] = 8
    sch = schedule([
        {"vessel_id": "F2", "origin": "Nantong", "destination": "Taicang", "depart_hour": 0, "cargo": []},
        {"vessel_id": "F2", "origin": "Taicang", "destination": "Nantong", "depart_hour": 5, "cargo": [{"shipment_id": "C1", "quantity": 4}]},
    ])
    trace = simulate(instance, sch, with_trace=True).trace
    assert trace["shipments"]["C1"]["groups"][0]["status"] == "LATE: delivered at hour 10, 2h after deadline 8"


def test_trace_matches_valid_csp_schedule():
    instance = easy()
    sch = Schedule.model_validate(json.loads((ROOT / "tests" / "fixtures" / "easy_valid_schedule.json").read_text()))
    report = simulate(instance, sch, with_trace=True)
    assert report.valid
    for shipment in report.trace["shipments"].values():
        assert all(g["status"].startswith("on time") for g in shipment["groups"])
