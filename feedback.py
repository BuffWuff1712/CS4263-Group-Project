"""Format simulator violations for the repair model."""

from collections import defaultdict
import re

from schemas import Schedule, Violation


# Structural and cargo-movement errors can cause downstream delivery errors.
PRIORITY = {
    "C1": 0, "C2": 0, "C3": 0, "C4": 0, "C5": 0, "C6": 0,
    "C7": 1, "C8": 1, "C10": 1, "C12": 1, "C13": 1,
    "C11": 2,
}


def _verified_facts(violation: Violation, instance: dict, schedule: Schedule) -> dict:
    """Add only facts calculated from the instance, schedule, or verifier message."""
    facts = {}
    index = violation.voyage_index
    voyage = schedule.voyages[index] if index is not None and 0 <= index < len(schedule.voyages) else None
    vessels = {v["id"]: v for v in instance["vessels"]}

    if voyage and voyage.vessel_id in vessels:
        vessel = vessels[voyage.vessel_id]
        if violation.constraint in {"C1", "C3", "C4", "C5"}:
            facts["vessel_start_port"] = vessel["start_port"]
            facts["allowed_routes_from_departure_port"] = [
                {"destination": r["destination"], "duration_hours": r["duration_hours"]}
                for r in instance["routes"]
                if r["origin"] == voyage.origin and vessel["type"] in r["allowed_types"]
            ]
        if violation.constraint == "C6":
            loaded = sum(load.quantity for load in voyage.cargo)
            facts.update(capacity=vessel["capacity"], loaded_units=loaded,
                         excess_units=max(0, loaded - vessel["capacity"]))
        if violation.constraint == "C4":
            route = next((r for r in instance["routes"] if
                          r["origin"] == voyage.origin and r["destination"] == voyage.destination), None)
            if route:
                arrival = voyage.depart_hour + route["duration_hours"]
                horizon = instance["planning_horizon_hours"]
                facts.update(arrival_hour=arrival, horizon_hour=horizon,
                             hours_beyond_horizon=max(0, arrival - horizon))

    if violation.constraint == "C7":
        match = re.fullmatch(r"Only (\d+) units available at (.*); requested (\d+)", violation.message)
        if match:
            available, requested = int(match[1]), int(match[3])
            facts.update(available_units=available, requested_units=requested,
                         shortage_units=requested - available)
    elif violation.constraint == "C5":
        match = re.fullmatch(r"Vessel unavailable until hour (\d+), but departs at hour (\d+)", violation.message)
        if match:
            available_at, departure = int(match[1]), int(match[2])
            facts.update(earliest_departure_hour=available_at,
                         hours_early=available_at - departure)
    elif violation.constraint == "C11":
        shipment = next((s for s in instance["shipments"] if s["id"] == violation.shipment_id), None)
        if shipment:
            facts.update(
                shipment_origin=shipment["origin"],
                shipment_destination=shipment["destination"],
                deadline_hour=shipment["deadline_hour"],
            )
            routes = {(r["origin"], r["destination"]): r for r in instance["routes"]}
            scheduled_legs = [
                (i, v, routes.get((v.origin, v.destination)))
                for i, v in enumerate(schedule.voyages)
                if any(load.shipment_id == violation.shipment_id for load in v.cargo)
            ]
            scheduled_legs = [(i, v, r) for i, v, r in scheduled_legs if r]
            if scheduled_legs:
                last_index, last_voyage, last_route = max(
                    scheduled_legs, key=lambda item: item[1].depart_hour + item[2]["duration_hours"]
                )
                facts["last_scheduled_cargo_leg"] = {
                    "voyage_index": last_index,
                    "vessel_id": last_voyage.vessel_id,
                    "origin": last_voyage.origin,
                    "destination": last_voyage.destination,
                    "arrival_hour": last_voyage.depart_hour + last_route["duration_hours"],
                }
                facts["routes_from_last_scheduled_port"] = [
                    {"destination": r["destination"], "duration_hours": r["duration_hours"]}
                    for r in instance["routes"] if r["origin"] == last_voyage.destination
                ]
        match = re.fullmatch(r"Incomplete delivery: (\d+)/(\d+) units delivered", violation.message)
        if match:
            delivered, required = int(match[1]), int(match[2])
            facts.update(delivered_units=delivered, required_units=required,
                         missing_units=required - delivered)
        late = re.fullmatch(
            r"Cargo unit \d+ delivered at hour (\d+), exceeding deadline (\d+)",
            violation.message,
        )
        if late:
            arrival, deadline = int(late[1]), int(late[2])
            facts.update(delivered_at_hour=arrival, hours_late=arrival - deadline)

    return facts


def format_feedback(
    violations: list[Violation], *,
    instance: dict, schedule: Schedule,
):
    by_constraint = defaultdict(list)
    for violation in violations:
        by_constraint[violation.constraint].append(violation)

    groups = []
    for constraint in sorted(by_constraint, key=lambda code: (PRIORITY.get(code, 1), code)):
        items = by_constraint[constraint]
        # Keep every actionable route, vessel, cargo, and chokepoint error.
        # For C11, keep every shipment-level delivery shortfall and a few
        # representative unit-level deadline/location messages.
        if constraint == "C11":
            shortfalls = [v for v in items if v.message.startswith("Incomplete delivery:")]
            other_examples = [v for v in items if not v.message.startswith("Incomplete delivery:")][:3]
            details = shortfalls + other_examples
        else:
            details = items
        entries = []
        for violation in details:
            entry = violation.model_dump(exclude_none=True)
            facts = _verified_facts(violation, instance, schedule)
            if facts:
                entry["verified_facts"] = facts
            entries.append(entry)
        groups.append({
            "constraint": constraint,
            "count": len(items),
            "details": entries,
            "omitted_unit_details": len(items) - len(details),
        })

    result = {
        "total_violations": len(violations),
        "instruction": (
            "Address structural and cargo-movement groups before delivery symptoms, "
            "then check every affected vessel, shipment, and voyage index in the details. "
            "Counts cover the full simulator report; repeated unit-level "
            "delivery details may be abbreviated, but every shipment shortfall is listed."
        ),
        "groups_in_repair_order": groups,
    }
    if set(by_constraint) == {"C11"}:
        result["instruction"] = (
            "Focus on the remaining late or incomplete shipments. Preserve the current "
            "deliveries of all other shipments while checking vessel, cargo, and timing rules."
        )
        result["focus_shipments"] = [
            {
                "shipment_id": shipment_id,
                "deadline_violation_count": sum(
                    "exceeding deadline" in v.message for v in violations
                    if v.shipment_id == shipment_id
                ),
            }
            for shipment_id in sorted({v.shipment_id for v in violations if v.shipment_id})
        ]
    return result
