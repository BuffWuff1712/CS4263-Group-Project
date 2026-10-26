from collections import defaultdict

from schemas import (
    Schedule,
    Voyage,
    Violation,
    ValidationReport
)



# schedule = Schedule(
#     voyages=[
#         Voyage(
#             vessel_id="F1",
#             origin="Shanghai",
#             destination="Taicang",
#             depart_hour=0,
#             cargo=[]
#         ),
#         Voyage(
#             vessel_id="F1",
#             origin="Taicang",
#             destination="Shanghai",
#             depart_hour=8,
#             cargo=[
#                 CargoLoad(
#                     shipment_id="C1",
#                     quantity=4
#                 )
#             ]
#         ),
#         Voyage(
#             vessel_id="L1",
#             origin="Shanghai",
#             destination="Singapore",
#             depart_hour=16,
#             cargo=[
#                 CargoLoad(shipment_id="C1", quantity=4),
#                 CargoLoad(shipment_id="C2", quantity=2)
#             ]
#         ),
#         Voyage(
#             vessel_id="F2",
#             origin="Singapore",
#             destination="Batam",
#             depart_hour=151,
#             cargo=[
#                 CargoLoad(shipment_id="C1", quantity=4)
#             ]
#         )
#     ]
# )

# voyages = schedule.voyages


# ==========================================
# 1. ROUTE VALIDATION
# ==========================================

def validate_routes(instance, voyages, violations):
    """
    Check:
    - Vessel IDs exist
    - Ports and routes exist
    - Routes are permitted for the vessel type
    - Feeder vessels remain within their region
    """

    routes = instance["routes"]

    for i, voyage in enumerate(voyages):

        route_exists = any(
            route["origin"] == voyage.origin
            and route["destination"] == voyage.destination
            for route in routes
        )

        if not route_exists:
            violations.append(
                Violation(
                    constraint="C1",
                    vessel_id=voyage.vessel_id,
                    voyage_index=i,
                    message=(
                        f"Invalid route: {voyage.origin} "
                        f"-> {voyage.destination}"
                    )
                )
            )
        


# ==========================================
# 2. VESSEL SCHEDULING
# ==========================================

def validate_vessel_schedule(instance, voyages, violations):
    """
    Check:
    - Vessel starts at its assigned initial port
    - Consecutive voyages have valid route continuity
    - No overlapping voyages
    - Voyages finish within the planning horizon
    - Maximum voyage count is respected
    """

    voyages_by_vessel = {}

    # Group voyages by vessel
    # Preserve original indices for violation reports
    for i, voyage in enumerate(voyages):
        vessel_id = voyage.vessel_id

        if vessel_id not in voyages_by_vessel:
            voyages_by_vessel[vessel_id] = []

        voyages_by_vessel[vessel_id].append((i, voyage))

    # Get vessel information
    vessels = {
        vessel["id"]: vessel
        for vessel in instance["vessels"]
    }

    # Get route durations
    routes = {
        (route["origin"], route["destination"]): route["duration_hours"]
        for route in instance["routes"]
    }

    H = instance["planning_horizon_hours"]

    # Validate each vessel's schedule
    for vessel_id, vessel_voyages in voyages_by_vessel.items():

        # Check that vessel exists
        if vessel_id not in vessels:
            violations.append(
                Violation(
                    constraint="C2",
                    vessel_id=vessel_id,
                    message="Unknown vessel"
                )
            )
            continue

        vessel = vessels[vessel_id]

        # Check maximum voyage count
        if len(vessel_voyages) > vessel["max_voyages"]:
            violations.append(
                Violation(
                    constraint="C5",
                    vessel_id=vessel_id,
                    message=(
                        f"Maximum voyages exceeded: "
                        f"{len(vessel_voyages)} > "
                        f"{vessel['max_voyages']}"
                    )
                )
            )

        # Sort voyages by departure time
        vessel_voyages.sort(
            key=lambda item: item[1].depart_hour
        )

        # Initialize vessel state
        current_port = vessel["start_port"]
        available_at = 0

        for i, voyage in vessel_voyages:

            # Check route continuity
            if voyage.origin != current_port:
                violations.append(
                    Violation(
                        constraint="C3",
                        vessel_id=vessel_id,
                        voyage_index=i,
                        message=(
                            f"Vessel expected at {current_port}, "
                            f"but departs from {voyage.origin}"
                        )
                    )
                )

            # Check overlapping voyages
            if voyage.depart_hour < available_at:
                violations.append(
                    Violation(
                        constraint="C5",
                        vessel_id=vessel_id,
                        voyage_index=i,
                        message=(
                            f"Vessel unavailable until hour "
                            f"{available_at}, but departs at "
                            f"hour {voyage.depart_hour}"
                        )
                    )
                )

            # Find route duration
            route_key = (
                voyage.origin,
                voyage.destination
            )

            if route_key not in routes:
                # Invalid route is also handled by validate_routes()
                violations.append(
                    Violation(
                        constraint="C1",
                        vessel_id=vessel_id,
                        voyage_index=i,
                        message="Cannot calculate arrival: invalid route"
                    )
                )
                continue

            # Calculate arrival time
            duration = routes[route_key]

            arrival_time = voyage.depart_hour + duration

            # Check planning horizon
            if arrival_time > H:
                violations.append(
                    Violation(
                        constraint="C4",
                        vessel_id=vessel_id,
                        voyage_index=i,
                        message=(
                            f"Voyage arrives at hour {arrival_time}, "
                            f"exceeding planning horizon {H}"
                        )
                    )
                )

            # Update vessel state
            current_port = voyage.destination
            available_at = arrival_time



# ==========================================
# 3. VESSEL CAPACITY
# ==========================================

def validate_capacity(
    instance: dict,
    voyages: list[Voyage],
    violations: list[Violation]
):
    # Retrieve vessel information
    vessels = {
        v["id"]: v
        for v in instance["vessels"]
    }

    # Check each voyage
    for i, voyage in enumerate(voyages):

        vessel_id = voyage.vessel_id

        # Unknown vessels are handled by validate_routes()
        if vessel_id not in vessels:
            continue

        # Retrieve the vessel's capacity
        capacity = vessels[vessel_id]["capacity"]

        # Calculate total cargo onboard
        total_cargo = sum(
            load.quantity
            for load in voyage.cargo
        )

        # Check capacity constraint
        if total_cargo > capacity:

            violations.append(
                Violation(
                    constraint="C6",
                    vessel_id=vessel_id,
                    voyage_index=i,
                    message=(
                        f"Capacity exceeded: "
                        f"{total_cargo} > {capacity}"
                    )
                )
            )


# ==========================================
# 4. CARGO MOVEMENT SIMULATION
# ==========================================

def simulate_cargo_movements(instance, voyages, violations):
    """
    Replay vessel voyages chronologically.

    Checks:
    - Cargo availability at departure ports
    - Cargo conservation and valid transfers
    - Shipment splitting
    - Mandatory intermediate stops
    - Maximum transport legs per cargo unit

    Returns:
    - Delivered quantities
    - Delivery times
    - Remaining port inventory
    - Individual cargo-unit histories
    """

    # ======================================
    # 1. LOAD PROBLEM INFORMATION
    # ======================================

    H = instance["planning_horizon_hours"]

    max_legs = instance["rules"][
        "max_transport_legs_per_container"
    ]

    shipments = {
        s["id"]: s
        for s in instance["shipments"]
    }

    vessels = {
        v["id"]: v
        for v in instance["vessels"]
    }

    routes = {
        (r["origin"], r["destination"]): r
        for r in instance["routes"]
    }

    # ======================================
    # 2. INITIALIZE CARGO INVENTORY
    # ======================================

    # inventory[port][shipment_id] = list of cargo units
    inventory = defaultdict(lambda: defaultdict(list))

    units = {}
    delivered = defaultdict(int)
    delivery_times = defaultdict(list)

    for shipment in shipments.values():

        for n in range(1, shipment["quantity"] + 1):

            unit = {
                "shipment_id": shipment["id"],
                "unit_id": n,
                "legs": 0,
                "visited_stop": (
                    shipment.get("mandatory_stop") is None
                ),
                "delivered_at": None,
                "history": [(0, shipment["origin"])]
            }

            units[(shipment["id"], n)] = unit

            # Handle cargo already at its destination
            if (
                shipment["origin"] == shipment["destination"]
                and unit["visited_stop"]
            ):
                unit["delivered_at"] = 0
                delivered[shipment["id"]] += 1
                delivery_times[shipment["id"]].append(0)
            else:
                inventory[shipment["origin"]][
                    shipment["id"]
                ].append(unit)

    # ======================================
    # 3. HELPER TO RECORD VIOLATIONS
    # ======================================

    def flag(code, message, i=None,
             voyage=None, shipment_id=None):

        violations.append(
            Violation(
                constraint=code,
                message=message,
                voyage_index=i,
                vessel_id=(
                    voyage.vessel_id
                    if voyage is not None else None
                ),
                shipment_id=shipment_id
            )
        )

    # ======================================
    # 4. GENERATE VOYAGE EVENTS
    # ======================================

    events = []

    for i, voyage in enumerate(voyages):

        route = routes.get(
            (voyage.origin, voyage.destination)
        )

        vessel = vessels.get(voyage.vessel_id)

        if route is None:
            flag(
                "C1",
                "Unknown route; cargo cannot be simulated",
                i, voyage
            )
            continue

        if vessel is None:
            flag(
                "C2",
                "Unknown vessel",
                i, voyage
            )
            continue

        if vessel["type"] not in route["allowed_types"]:
            flag(
                "C2",
                "Vessel type not allowed on route",
                i, voyage
            )
            continue

        arrival = (
            voyage.depart_hour + route["duration_hours"]
        )

        if arrival > H:
            flag(
                "C4",
                f"Arrival at hour {arrival} exceeds horizon {H}",
                i, voyage
            )

        # Priority 0 = arrival
        # Priority 1 = departure
        events.append(
            (voyage.depart_hour, 1, i, voyage)
        )

        events.append(
            (arrival, 0, i, voyage)
        )

    # Sort chronologically.
    # Arrivals are processed before departures
    # when events occur in the same hour.
    events.sort(
        key=lambda e: (e[0], e[1], e[2])
    )

    # Stores cargo currently assigned to each voyage
    onboard = {}

    # ======================================
    # 5. REPLAY EVENTS
    # ======================================

    for hour, priority, i, voyage in events:

        # ----------------------------------
        # ARRIVAL EVENT
        # ----------------------------------

        if priority == 0:

            for unit in onboard.pop(i, []):

                shipment = shipments[
                    unit["shipment_id"]
                ]

                stop = shipment.get("mandatory_stop")

                unit["history"].append(
                    (hour, voyage.destination)
                )

                # Mark mandatory stop as visited
                if voyage.destination == stop:
                    unit["visited_stop"] = True

                # Check if final destination reached
                if (
                    voyage.destination
                    == shipment["destination"]
                ):

                    if not unit["visited_stop"]:
                        flag(
                            "C10",
                            f"Mandatory stop {stop} not visited",
                            i,
                            voyage,
                            shipment["id"]
                        )

                        inventory[voyage.destination][
                            shipment["id"]
                        ].append(unit)

                    else:
                        unit["delivered_at"] = hour

                        delivered[shipment["id"]] += 1

                        delivery_times[
                            shipment["id"]
                        ].append(hour)

                else:
                    # Store cargo at arrival port
                    inventory[voyage.destination][
                        shipment["id"]
                    ].append(unit)

        # ----------------------------------
        # DEPARTURE EVENT
        # ----------------------------------

        else:

            transported = []

            for load in voyage.cargo:

                shipment_id = load.shipment_id

                if shipment_id not in shipments:
                    flag(
                        "C8",
                        "Unknown shipment ID",
                        i, voyage, shipment_id
                    )
                    continue

                available = inventory[
                    voyage.origin
                ][shipment_id]

                # Check cargo availability
                if len(available) < load.quantity:

                    flag(
                        "C7",
                        (
                            f"Only {len(available)} units "
                            f"available at {voyage.origin}; "
                            f"requested {load.quantity}"
                        ),
                        i, voyage, shipment_id
                    )
                    continue

                # Deterministically select units.
                # Prefer units that have visited
                # mandatory stops and used fewer legs.
                available.sort(
                    key=lambda u: (
                        0 if u["visited_stop"] else 1,
                        u["legs"],
                        u["unit_id"]
                    )
                )

                selected = available[:load.quantity]

                # Check maximum transport legs
                if any(
                    u["legs"] >= max_legs
                    for u in selected
                ):
                    flag(
                        "C8",
                        f"Maximum {max_legs} legs exceeded",
                        i, voyage, shipment_id
                    )
                    continue

                # Remove cargo from origin inventory
                del available[:load.quantity]

                for unit in selected:
                    unit["legs"] += 1
                    transported.append(unit)

            # Store the actual cargo transported
            onboard[i] = transported

    # ======================================
    # 6. RETURN SIMULATION RESULTS
    # ======================================

    return {
        "delivered": {
            sid: delivered[sid]
            for sid in shipments
        },

        "delivery_times": {
            sid: delivery_times[sid]
            for sid in shipments
        },

        "inventory": {
            port: {
                sid: len(cargo_units)
                for sid, cargo_units in stock.items()
            }
            for port, stock in inventory.items()
        },

        "unit_histories": {
            f"{sid}:{n}": unit
            for (sid, n), unit in units.items()
        }
    }


# ==========================================
# 5. DELIVERY VALIDATION
# ==========================================

def validate_deliveries(instance, cargo_state, violations):
    """
    Check:
    - All cargo units reach final destinations
    - Delivery deadlines are satisfied
    - Mandatory intermediate stops are respected
    - Maximum cargo transport legs are respected
    """

    shipments = {
        s["id"]: s
        for s in instance["shipments"]
    }

    delivered = cargo_state["delivered"]
    unit_histories = cargo_state["unit_histories"]

    max_legs = instance["rules"][
        "max_transport_legs_per_container"
    ]

    # ======================================
    # 1. CHECK COMPLETE CARGO DELIVERY
    # ======================================

    for shipment_id, shipment in shipments.items():

        required = shipment["quantity"]
        actual = delivered.get(shipment_id, 0)

        if actual != required:
            violations.append(
                Violation(
                    constraint="C11",
                    shipment_id=shipment_id,
                    message=(
                        f"Incomplete delivery: "
                        f"{actual}/{required} units delivered"
                    )
                )
            )

    # ======================================
    # 2. CHECK INDIVIDUAL CARGO UNITS
    # ======================================

    for unit in unit_histories.values():

        shipment_id = unit["shipment_id"]
        unit_id = unit["unit_id"]

        shipment = shipments[shipment_id]

        deadline = shipment["deadline_hour"]
        mandatory_stop = shipment.get("mandatory_stop")
        destination = shipment["destination"]

        delivered_at = unit["delivered_at"]
        legs = unit["legs"]
        history = unit["history"]

        # ----------------------------------
        # CHECK DELIVERY DEADLINE
        # ----------------------------------

        if delivered_at is not None:

            if delivered_at > deadline:
                violations.append(
                    Violation(
                        constraint="C11",
                        shipment_id=shipment_id,
                        message=(
                            f"Cargo unit {unit_id} delivered "
                            f"at hour {delivered_at}, "
                            f"exceeding deadline {deadline}"
                        )
                    )
                )

            # Verify delivery location
            if not history or history[-1][1] != destination:
                violations.append(
                    Violation(
                        constraint="C11",
                        shipment_id=shipment_id,
                        message=(
                            f"Cargo unit {unit_id} was not "
                            f"delivered to {destination}"
                        )
                    )
                )

        # ----------------------------------
        # CHECK MANDATORY INTERMEDIATE STOP
        # ----------------------------------

        if mandatory_stop is not None and delivered_at is not None:

            # All visited ports before final delivery
            visited_ports = [
                port
                for hour, port in history[:-1]
            ]

            if mandatory_stop not in visited_ports:
                violations.append(
                    Violation(
                        constraint="C10",
                        shipment_id=shipment_id,
                        message=(
                            f"Cargo unit {unit_id} did not "
                            f"visit mandatory stop "
                            f"{mandatory_stop} before delivery"
                        )
                    )
                )

        # ----------------------------------
        # CHECK MAXIMUM TRANSPORT LEGS
        # ----------------------------------

        if legs > max_legs:
            violations.append(
                Violation(
                    constraint="C8",
                    shipment_id=shipment_id,
                    message=(
                        f"Cargo unit {unit_id} used "
                        f"{legs} transport legs, "
                        f"exceeding maximum {max_legs}"
                    )
                )
            )



# ==========================================
# 6. MARITIME CHOKEPOINT VALIDATION
# ==========================================

def validate_chokepoints(instance, voyages, violations):
    """
    Check:
    - Suez: even hours = southbound, odd = northbound
    - Suez: maximum 1 admission per hour
    - Malacca: maximum 2 admissions per hour
    """

    chokepoints = instance["rules"].get("chokepoints", [])

    # Easy test set has no chokepoints
    if not chokepoints:
        return

    # ======================================
    # 1. PREPARE CHOKEPOINT INFORMATION
    # ======================================

    route_durations = {
        (r["origin"], r["destination"]): r["duration_hours"]
        for r in instance["routes"]
    }

    # Map shipping routes to chokepoint information
    passage_lookup = defaultdict(list)
    capacities = {}

    for cp in chokepoints:

        cp_id = cp["id"].lower()
        capacity = cp["capacity_per_hour"]

        if cp_id not in ("suez", "malacca"):
            raise ValueError(f"Unknown chokepoint: {cp_id}")

        if not isinstance(capacity, int) or capacity < 1:
            raise ValueError(f"Invalid capacity for {cp_id}")

        if cp_id in capacities:
            raise ValueError(f"Duplicate chokepoint: {cp_id}")

        capacities[cp_id] = capacity

        for passage in cp["routes"]:

            route_key = (
                passage["origin"],
                passage["destination"]
            )

            offset = passage["offset_hours"]
            duration = route_durations.get(route_key)

            # Validate the problem configuration
            if duration is None or not 0 <= offset < duration:
                raise ValueError(
                    f"Invalid chokepoint offset for {route_key}"
                )

            direction = passage.get("direction")

            if cp_id == "suez" and direction not in (
                "northbound", "southbound"
            ):
                raise ValueError(
                    f"Invalid Suez direction for {route_key}"
                )

            passage_lookup[route_key].append({
                "id": cp_id,
                "offset": offset,
                "direction": direction
            })

    # ======================================
    # 2. CALCULATE CHOKEPOINT ENTRY TIMES
    # ======================================

    # admissions[(chokepoint, hour)] = list of vessels
    admissions = defaultdict(list)

    for i, voyage in enumerate(voyages):

        route_key = (
            voyage.origin,
            voyage.destination
        )

        for passage in passage_lookup.get(route_key, []):

            cp_id = passage["id"]

            entry_hour = (
                voyage.depart_hour + passage["offset"]
            )

            admissions[(cp_id, entry_hour)].append(
                voyage.vessel_id
            )

            # ==================================
            # 3. CHECK SUEZ DIRECTION
            # ==================================

            if cp_id == "suez":

                allowed_direction = (
                    "southbound"
                    if entry_hour % 2 == 0
                    else "northbound"
                )

                if passage["direction"] != allowed_direction:

                    violations.append(
                        Violation(
                            constraint="C12",
                            vessel_id=voyage.vessel_id,
                            voyage_index=i,
                            message=(
                                f"Suez entry at hour {entry_hour}: "
                                f"{passage['direction']} not allowed. "
                                f"Expected {allowed_direction}."
                            )
                        )
                    )

    # ======================================
    # 4. CHECK CHOKEPOINT CAPACITY
    # ======================================

    for (cp_id, hour), vessels in admissions.items():

        capacity = capacities[cp_id]

        if len(vessels) > capacity:

            violations.append(
                Violation(
                    constraint="C13",
                    message=(
                        f"{cp_id.capitalize()} capacity exceeded "
                        f"at hour {hour}: "
                        f"{len(vessels)} admissions, "
                        f"maximum {capacity}. "
                        f"Vessels: {', '.join(vessels)}"
                    )
                )
            )

# ==========================================
# 7. CARGO SPLITTING VALIDATION
# ==========================================

def validate_cargo_splitting(instance, voyages, violations):
    """
    If cargo splitting is disabled, every voyage
    transporting a shipment must carry its full quantity.
    """

    # If splitting is allowed, no restrictions needed
    if instance["rules"].get("cargo_splitting_allowed", True):
        return

    shipments = {
        s["id"]: s
        for s in instance["shipments"]
    }

    for i, voyage in enumerate(voyages):

        seen_shipments = set()

        for load in voyage.cargo:

            shipment_id = load.shipment_id

            # Unknown shipment IDs are checked elsewhere
            if shipment_id not in shipments:
                continue

            # Prevent multiple cargo entries for the same
            # shipment within one voyage
            if shipment_id in seen_shipments:
                violations.append(
                    Violation(
                        constraint="C8",
                        vessel_id=voyage.vessel_id,
                        shipment_id=shipment_id,
                        voyage_index=i,
                        message=(
                            f"Duplicate cargo entries for "
                            f"indivisible shipment {shipment_id}"
                        )
                    )
                )
                continue

            seen_shipments.add(shipment_id)

            # Get original shipment quantity
            required_quantity = shipments[
                shipment_id
            ]["quantity"]

            # Check whether full shipment is transported
            if load.quantity != required_quantity:

                violations.append(
                    Violation(
                        constraint="C8",
                        vessel_id=voyage.vessel_id,
                        shipment_id=shipment_id,
                        voyage_index=i,
                        message=(
                            f"Cargo splitting prohibited: "
                            f"{shipment_id} requires "
                            f"{required_quantity} units, "
                            f"but voyage carries "
                            f"{load.quantity} units"
                        )
                    )
                )



# ==========================================
# 7. MAIN SIMULATOR
# ==========================================

def simulate(
    instance: dict,
    schedule: Schedule
) -> ValidationReport:
    """
    Validate an LLM-generated shipping schedule.

    Returns a ValidationReport containing:
    - valid: True/False
    - violations: List of constraint violations
    - delivered: Delivered quantity per shipment
    """

    violations = []

    voyages = schedule.voyages

    # 1. Route checks
    validate_routes(instance, voyages, violations)

    # 2. Vessel scheduling checks
    validate_vessel_schedule(
        instance, voyages, violations
    )

    # 3. Cargo capacity checks
    validate_capacity(instance, voyages, violations)

    # 4. Replay cargo movements
    cargo_state = simulate_cargo_movements(
        instance, voyages, violations
    )

    # 5. Verify deliveries
    validate_deliveries(
        instance, cargo_state, violations
    )

    # 6. Verify maritime chokepoints
    validate_chokepoints(
        instance, voyages, violations
    )

    # 7. Verify cargo splitting rules
    validate_cargo_splitting(
        instance, voyages, violations
    )

    # Extract delivery totals from simulation
    delivered = cargo_state["delivered"]

    return ValidationReport(
        valid=(len(violations) == 0),
        violations=violations,
        delivered=delivered
    )
