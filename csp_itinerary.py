from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

from schemas import CargoLoad, Schedule, Voyage
from simulator import simulate

sys.setrecursionlimit(10000000)
TIME_LIMIT = 0
RESTARTS = 0

# Precompute simple (cycle-free) port itineraries for each shipment.
PATH_PREPROCESSING = True
PATH_ENUMERATION_LIMIT = 20_000
# Independent switches for controlled comparisons.
CARGO_AWARE_ORDER = True
CRITICAL_CARGO_BRANCHING = True
TRANSFER_PRUNING = False
EMPTY_CYCLE_PRUNING = False
EMPTY_CYCLE_POSTPROCESS = True
# Group compatible Q choices without adding exponential itinerary variables.
# "auto" groups only scarce cross-vessel two-leg connections.
ITINERARY_MODE = "auto"         # auto, full, pair, off
ITINERARY_CANDIDATE_LIMIT = 128

class Timeout(Exception):
    pass


class Restart(Exception):
    pass


def luby(i: int) -> int:
    """i-th term (1-based) of the Luby sequence 1,1,2,1,1,2,4,1,1,2,..."""
    k = 1
    while (1 << k) - 1 < i:
        k += 1
    while i != (1 << k) - 1:
        i -= (1 << (k - 1)) - 1
        k = 1
        while (1 << k) - 1 < i:
            k += 1
    return 1 << (k - 1)


def bits(mask: int):
    while mask:
        low = mask & -mask
        yield low.bit_length() - 1
        mask ^= low


# ---------------------------------------------------------------------------
# Generic CSP engine: bitmask domains, trail, propagator queue
# ---------------------------------------------------------------------------
class CSP:
    def __init__(self):
        self.dom: list[int] = []
        self.watch: list[list[int]] = []
        self.props = []
        self.trail: list[tuple[int, int]] = []
        self.changed: list[int] = []
        self.version = 0
        self.nodes = 0
        self.backtracks = 0
        self.propagations = 0
        self.weight: list[float] = []

    def new_var(self, mask: int) -> int:
        self.dom.append(mask)
        self.watch.append([])
        return len(self.dom) - 1

    def add_prop(self, prop, scope):
        idx = len(self.props)
        self.props.append(prop)
        self.weight.append(1.0)
        prop.scope = list(scope)
        for v in prop.scope:
            self.watch[v].append(idx)

    def set_dom(self, v: int, mask: int) -> bool:
        if mask == self.dom[v]:
            return True
        if mask == 0:
            return False
        self.trail.append((v, self.dom[v]))
        self.dom[v] = mask
        self.changed.append(v)
        self.version += 1
        return True

    def undo(self, mark: int):
        while len(self.trail) > mark:
            v, old = self.trail.pop()
            self.dom[v] = old
        self.version += 1

    def propagate(self, start_vars=None, learn=True) -> bool:
        """AC-3 style fixpoint: re-run every propagator watching a changed var.
        learn=False (trial propagation) leaves the dom/wdeg weights alone."""
        queue, queued = deque(), set()
        if start_vars is None:
            todo = range(len(self.props))
        else:
            todo = {p for v in start_vars for p in self.watch[v]}
        for p in todo:
            queue.append(p)
            queued.add(p)
        while queue:
            p = queue.popleft()
            queued.discard(p)
            self.propagations += 1
            self.changed = []
            if not self.props[p].run(self):
                if learn:
                    self.weight[p] += 1.0     # dom/wdeg: remember hard constraints
                return False
            for v in set(self.changed):
                for q in self.watch[v]:
                    if q not in queued:
                        queued.add(q)
                        queue.append(q)
        return True


class BinaryTable:
    """Binary constraint given as support masks; revising both directions is
    one AC-3 arc pair."""

    def __init__(self, a, b, sup_a, sup_b):
        self.a, self.b, self.sup_a, self.sup_b = a, b, sup_a, sup_b

    def run(self, csp: CSP) -> bool:
        da, db = csp.dom[self.a], csp.dom[self.b]
        na = 0
        for x in bits(da):
            if self.sup_a[x] & db:
                na |= 1 << x
        if not csp.set_dom(self.a, na):
            return False
        nb = 0
        for y in bits(db):
            if self.sup_b[y] & na:
                nb |= 1 << y
        return csp.set_dom(self.b, nb)


# ---------------------------------------------------------------------------
# MTRSP model
# ---------------------------------------------------------------------------
class Model:
    """Original CSP + reversible complete-itinerary/pair search decisions."""
    def __init__(self, instance: dict, *, path_preprocessing: bool = PATH_PREPROCESSING,
                 path_enumeration_limit: int = PATH_ENUMERATION_LIMIT,
                 cargo_aware_order: bool = CARGO_AWARE_ORDER,
                 critical_cargo_branching: bool = CRITICAL_CARGO_BRANCHING,
                 transfer_pruning: bool = TRANSFER_PRUNING,
                 empty_cycle_pruning: bool = EMPTY_CYCLE_PRUNING,
                 empty_cycle_postprocess: bool = EMPTY_CYCLE_POSTPROCESS,
                 itinerary_mode: str = ITINERARY_MODE,
                 itinerary_candidate_limit: int = ITINERARY_CANDIDATE_LIMIT):
        self.H = instance["planning_horizon_hours"]
        rules = instance["rules"]
        self.max_legs = rules["max_transport_legs_per_container"]
        self.split = rules.get("cargo_splitting_allowed", True)
        if rules.get("chokepoints"):
            raise NotImplementedError("chokepoints are not modelled")
        if any(s.get("mandatory_stop") for s in instance["shipments"]):
            raise NotImplementedError("mandatory stops are not modelled")

        self.ports = [p["name"] for p in instance["ports"]]
        self.pid = {p: i for i, p in enumerate(self.ports)}
        n = self.n = len(self.ports)
        self.NONE_P = n
        self.pmask = (1 << n) - 1

        # duration[type][(o,d)]; out_mask[type][o]
        types = {v["type"] for v in instance["vessels"]}
        self.dur = {t: {} for t in types}
        self.out_mask = {t: [0] * n for t in types}
        inf = 10**9
        self.sp = [[inf] * n for _ in range(n)]
        for i in range(n):
            self.sp[i][i] = 0
        for r in instance["routes"]:
            o, d = self.pid[r["origin"]], self.pid[r["destination"]]
            self.sp[o][d] = min(self.sp[o][d], r["duration_hours"])
            for t in r["allowed_types"]:
                if t in self.dur:
                    self.dur[t][(o, d)] = r["duration_hours"]
                    self.out_mask[t][o] |= 1 << d
        for k in range(n):  # Floyd-Warshall lower bound on travel time
            for i in range(n):
                for j in range(n):
                    if self.sp[i][k] + self.sp[k][j] < self.sp[i][j]:
                        self.sp[i][j] = self.sp[i][k] + self.sp[k][j]

        # voyage slots
        self.slots = []          # (vessel_index, k)
        self.slot_id = {}
        self.vessels = instance["vessels"]
        for vi, v in enumerate(self.vessels):
            for k in range(v["max_voyages"]):
                self.slot_id[(vi, k)] = len(self.slots)
                self.slots.append((vi, k))
        self.S = len(self.slots)
        self.NONE_Q = self.S

        # cargo items
        self.items = []          # dict(shipment, origin, dest, qty, deadline)
        for s in instance["shipments"]:
            o, d = self.pid[s["origin"]], self.pid[s["destination"]]
            if o == d:
                continue
            if self.split:
                for _ in range(s["quantity"]):
                    self.items.append(dict(sid=s["id"], o=o, d=d, qty=1,
                                           dl=s["deadline_hour"]))
            else:
                self.items.append(dict(sid=s["id"], o=o, d=d,
                                       qty=s["quantity"], dl=s["deadline_hour"]))
        self.m = min(self.max_legs, n - 1)   # simple paths need <= n-1 legs

        self.slot_type = [self.vessels[vi]["type"] for vi, _ in self.slots]
        self.slot_capacity = [self.vessels[vi]["capacity"] for vi, _ in self.slots]
        self.cargo_aware_order = cargo_aware_order
        self.critical_cargo_branching = critical_cargo_branching
        self.transfer_pruning = transfer_pruning
        self.empty_cycle_pruning = empty_cycle_pruning
        self.empty_cycle_postprocess = empty_cycle_postprocess
        if itinerary_mode not in ("auto", "full", "pair", "off"):
            raise ValueError(f"invalid itinerary_mode: {itinerary_mode}")
        if itinerary_candidate_limit < 1:
            raise ValueError("itinerary_candidate_limit must be positive")
        self.itinerary_mode = itinerary_mode
        self.itinerary_candidate_limit = itinerary_candidate_limit
        self.itinerary_group_decisions = 0
        self.itinerary_full_decisions = 0
        self.itinerary_pair_decisions = 0
        self.itinerary_candidates_generated = 0
        self.itinerary_candidates_tried = 0
        self.itinerary_candidate_failures = 0
        self.itinerary_enumeration_overflows = 0
        self.itinerary_generation_seconds = 0.0
        self.itinerary_fallbacks = 0
        self.itinerary_decisions_by_shipment = {}
        self.empty_cycles_removed_from_schedule = 0
        self.empty_voyages_removed_from_schedule = 0
        self.transfer_calls = 0
        self.transfer_q_values_removed = 0
        self.transfer_conflicts = 0
        self.empty_cycle_values_removed = 0
        self.empty_cycle_conflicts = 0
        self.cargo_decisions = 0
        self.route_decisions = 0

        # All-pairs distances for an individual vessel type (repositioning).
        INF = 10**9
        self.type_sp = {}
        for typ in self.dur:
            mat = [[0 if i == j else INF for j in range(n)] for i in range(n)]
            for (o, d), duration in self.dur[typ].items():
                mat[o][d] = min(mat[o][d], duration)
            for z in range(n):
                for i in range(n):
                    for j in range(n):
                        if mat[i][z] + mat[z][j] < mat[i][j]:
                            mat[i][j] = mat[i][z] + mat[z][j]
            self.type_sp[typ] = mat

        # Static path preprocessing. A template is a tuple of consecutive
        # (origin_port, destination_port, eligible_slot_bitmask) legs.
        # NONE means path enumeration was skipped because the cap was reached.
        self.path_preprocessing = path_preprocessing
        self.path_enumeration_limit = path_enumeration_limit
        self.path_support_calls = 0
        self.path_q_values_removed = 0
        self.path_d_values_removed = 0
        self.path_fallback_items = 0
        self.path_templates = []
        self.path_static_masks = []
        if path_preprocessing:
            memo = {}
            for it in self.items:
                key = (it["o"], it["d"], it["qty"], it["dl"])
                if key not in memo:
                    memo[key] = self._enumerate_simple_paths(it)
                paths = memo[key]
                self.path_templates.append(paths)
                if paths is None:
                    self.path_fallback_items += 1
                    self.path_static_masks.append(None)
                else:
                    masks = [0] * self.m
                    for path in paths:
                        for j in range(self.m):
                            masks[j] |= (path[j][2] if j < len(path)
                                         else 1 << self.NONE_Q)
                    self.path_static_masks.append(masks)
        else:
            self.path_templates = [None] * len(self.items)
            self.path_static_masks = [None] * len(self.items)

        # Long, tightly constrained itineraries deserve earlier carrier-slot
        # decisions, but this is a soft priority rather than strict cargo-first.
        self.cargo_criticality = []
        for ii, it in enumerate(self.items):
            paths = self.path_templates[ii]
            if paths:
                minimum = min(len(path) for path in paths)
                scarcity = 1.0 / len(paths)
            else:
                minimum = max(1, min(self.m, (self.sp[it["o"]][it["d"]] + 29) // 30))
                scarcity = 0.0
            self.cargo_criticality.append(
                1.0 + 0.48 * (minimum - 1) + 0.18 * scarcity
            )

        self._bounds_version, self._bounds = -1, None
        self.deadline, self.node_limit, self.restarts = None, float("inf"), 0
        self.csp = CSP()
        self._build()

    def _enumerate_simple_paths(self, it):
        """Enumerate *all* cycle-free cargo port paths of at most m legs.

        A route is eligible only if some actual vessel slot has a route type
        allowed on that arc and enough capacity for the indivisible item.
        Travel times are optimistic: we ignore repositioning and waiting, so
        this preprocessing may leave impossible paths but never excludes one
        on the basis of an overly pessimistic time estimate.

        Cyclic cargo travel is dominated under this instance's assumptions:
        unlimited port storage, no mandatory stops, nonnegative durations,
        and waiting at ports permitted. Removing a cargo cycle preserves
        deliveries and cannot increase vessel loads or arrival time.

        For a dense, large graph, enumeration can be exponential. In that
        case we *disable* path pruning for the item rather than mistakenly
        treat an incomplete path list as exhaustive.
        """
        qty = it["qty"]
        deadline = min(it["dl"], self.H)
        arcs = {}  # (o,d) -> [optimistic_min_duration, eligible_slot_mask]
        for s, t in enumerate(self.slot_type):
            if self.slot_capacity[s] < qty:
                continue
            for (o, d), travel in self.dur[t].items():
                edge = (o, d)
                if edge not in arcs:
                    arcs[edge] = [travel, 0]
                elif travel < arcs[edge][0]:
                    arcs[edge][0] = travel
                arcs[edge][1] |= 1 << s

        outgoing = [[] for _ in range(self.n)]
        for (o, d), (duration, slot_mask) in arcs.items():
            outgoing[o].append((d, duration, slot_mask))

        paths = []
        overflow = False

        def dfs(port, visited_mask, legs, elapsed):
            nonlocal overflow
            if overflow:
                return
            if port == it["d"]:
                if len(paths) >= self.path_enumeration_limit:
                    overflow = True
                else:
                    paths.append(tuple(legs))
                return
            if len(legs) >= self.m:
                return
            for dest, duration, slot_mask in outgoing[port]:
                if visited_mask & (1 << dest):
                    continue
                new_elapsed = elapsed + duration
                # Unrestricted shortest path is a lower bound even when
                # capacities/types and time-dependent connections matter.
                if new_elapsed + self.sp[dest][it["d"]] > deadline:
                    continue
                legs.append((port, dest, slot_mask))
                dfs(dest, visited_mask | (1 << dest), legs, new_elapsed)
                legs.pop()
                if overflow:
                    return

        dfs(it["o"], 1 << it["o"], [], 0)
        return None if overflow else paths

    def _propagate_cargo_paths(self, ii: int) -> bool:
        """Propagate support from complete simple port paths into Q and D.

        A path survives if each cargo leg has at least one slot whose vessel
        could traverse its corresponding edge, under current domains.
        Surviving paths support Q choices. If a Q is fixed to a particular
        slot, we also constrain that slot's origin/destination to ports
        supported by surviving paths. This is intentionally a relaxation of
        full time/capacity/slot-all-different feasibility; those constraints
        remain the responsibility of the existing propagators.
        """
        paths = self.path_templates[ii]
        if paths is None:
            return True                # no precomputation or size-cap fallback
        self.path_support_calls += 1
        if not paths:
            return False

        csp = self.csp
        Q = self.Q[ii]
        qdom = [csp.dom[q] for q in Q]
        none_bit = 1 << self.NONE_Q
        supports = [0] * self.m
        supported_paths = []  # (path, per-leg supporting slot masks)
        origins = {}
        destinations = {}

        def can_carry_arc(slot: int, o: int, d: int) -> bool:
            if slot not in origins:
                origins[slot] = self.org_set(slot)
                destinations[slot] = self.dest_set(slot)
            return (bool(origins[slot] & (1 << o))
                    and bool(destinations[slot] & (1 << d)))

        for path in paths:
            edge_support = []
            good = True
            for j in range(self.m):
                if j >= len(path):
                    if not qdom[j] & none_bit:
                        good = False
                        break
                    edge_support.append(none_bit)
                    continue
                o, d, static_slots = path[j]
                candidates = qdom[j] & static_slots
                supported = 0
                for s in bits(candidates):
                    if can_carry_arc(s, o, d):
                        supported |= 1 << s
                if not supported:
                    good = False
                    break
                edge_support.append(supported)
            if good:
                supported_paths.append((path, edge_support))
                for j, mask in enumerate(edge_support):
                    supports[j] |= mask

        if not supported_paths:
            return False

        for j, q in enumerate(Q):
            new_dom = qdom[j] & supports[j]
            if not new_dom:
                return False
            if new_dom != qdom[j]:
                self.path_q_values_removed += qdom[j].bit_count() - new_dom.bit_count()
                if not csp.set_dom(q, new_dom):
                    return False

        # Conditional arc propagation: only constrain a vessel's D variables
        # when this cargo's carrier slot is *fixed*. Never constrain a slot
        # solely because it is a potential carrier.
        for j, q in enumerate(Q):
            m = csp.dom[q]
            if m & (m - 1):
                continue
            s = m.bit_length() - 1
            if s == self.NONE_Q:
                continue
            bit = 1 << s
            possible_origins = 0
            possible_dests = 0
            for path, edge_support in supported_paths:
                if j < len(path) and edge_support[j] & bit:
                    o, d, _ = path[j]
                    possible_origins |= 1 << o
                    possible_dests |= 1 << d
            if not possible_dests:
                return False
            dv = self.D[s]
            old = csp.dom[dv]
            new = old & possible_dests
            if not new:
                return False
            if old != new:
                self.path_d_values_removed += old.bit_count() - new.bit_count()
                if not csp.set_dom(dv, new):
                    return False

            vi, k = self.slots[s]
            if k > 0:
                pv = self.D[self.slot_id[(vi, k - 1)]]
                old = csp.dom[pv]
                new = old & possible_origins
                if not new:
                    return False
                if old != new:
                    self.path_d_values_removed += old.bit_count() - new.bit_count()
                    if not csp.set_dom(pv, new):
                        return False
            elif not (possible_origins &
                      (1 << self.pid[self.vessels[vi]["start_port"]])):
                return False
        return True

    def _propagate_transfer_compatibility(self, ii: int) -> bool:
        """Relaxed temporal support for consecutive cargo-leg carrier slots.

        For each precomputed port path, do a forward dynamic program over its
        *actual* candidate voyage slots. Each state stores the earliest possible
        arrival for that slot, respecting existing lower bounds, travel time,
        deadline, and vessel voyage ordering. Backward support removes Q values
        that cannot appear in any complete supported chain.

        Waiting at ports and delaying sailings are allowed: we NEVER reject a
        pair merely because the second slot's earliest start is too early.
        Instead its departure may be delayed to the cargo's arrival time.

        This DP deliberately ignores some cross-cargo and non-adjacent temporal
        interactions, so it over-approximates feasible schedules. That makes
        domain pruning safe, while leaving harder conflicts to Timing/bounds.
        """
        if not self.transfer_pruning:
            return True
        paths = self.path_templates[ii]
        if paths is None:
            return True  # Enumeration overflow/disabled: preserve baseline.
        self.transfer_calls += 1
        if not paths:
            self.transfer_conflicts += 1
            return False
        csp = self.csp
        q = self.Q[ii]
        qdom = [csp.dom[v] for v in q]
        none_bit = 1 << self.NONE_Q
        bounds = self.bounds()
        if bounds is None:
            self.transfer_conflicts += 1
            return False
        earliest_depart = bounds[0]
        deadline = min(self.items[ii]["dl"], self.H)
        supports = [0] * self.m
        origin_cache = {}
        dest_cache = {}

        for path in paths:
            L = len(path)
            if any(not (qdom[j] & none_bit) for j in range(L, self.m)):
                continue

            # A candidate must permit this exact arc under the current D domains.
            candidates = []
            tail = [0] * (L + 1)
            valid = True
            for j, (o, d, static_mask) in enumerate(path):
                choices = []
                for slot in bits(qdom[j] & static_mask):
                    if slot not in origin_cache:
                        origin_cache[slot] = self.org_set(slot)
                        dest_cache[slot] = self.dest_set(slot)
                    if not (origin_cache[slot] & (1 << o)) or not (dest_cache[slot] & (1 << d)):
                        continue
                    if earliest_depart[slot] >= 10**9:
                        continue
                    duration = self.dur[self.slot_type[slot]].get((o, d))
                    if duration is not None:
                        choices.append((slot, duration))
                if not choices:
                    valid = False
                    break
                candidates.append(choices)
            if not valid:
                continue
            # Optimistic remaining travel time along THIS port path.
            for j in range(L - 1, -1, -1):
                tail[j] = tail[j + 1] + min(dur for _, dur in candidates[j])

            arrivals = []
            parents = []
            for j, choices in enumerate(candidates):
                best = {}
                links = {}
                for slot, duration in choices:
                    if j == 0:
                        a = earliest_depart[slot] + duration
                        if a + tail[j + 1] <= deadline:
                            best[slot] = a
                            links[slot] = ()
                        continue
                    predecessor_links = []
                    earliest = 10**9
                    for prev, prev_arr in arrivals[-1].items():
                        # A vessel cannot perform a later cargo leg using an
                        # earlier voyage of that same vessel.
                        v_prev, k_prev = self.slots[prev]
                        v_now, k_now = self.slots[slot]
                        if prev == slot or (v_prev == v_now and k_prev >= k_now):
                            continue
                        a = max(prev_arr, earliest_depart[slot]) + duration
                        if a + tail[j + 1] > deadline:
                            continue
                        predecessor_links.append(prev)
                        if a < earliest:
                            earliest = a
                    if predecessor_links:
                        best[slot] = earliest
                        links[slot] = predecessor_links
                if not best:
                    valid = False
                    break
                arrivals.append(best)
                parents.append(links)
            if not valid:
                continue

            # Only values that can reach a complete path survive.
            reachable = set(arrivals[-1])
            for j in range(L - 1, -1, -1):
                for slot in reachable:
                    supports[j] |= 1 << slot
                if j:
                    reachable = {prev for s in reachable for prev in parents[j][s]}
            for j in range(L, self.m):
                supports[j] |= none_bit

        for j, var in enumerate(q):
            revised = csp.dom[var] & supports[j]
            if not revised:
                self.transfer_conflicts += 1
                return False
            if revised != csp.dom[var]:
                self.transfer_q_values_removed += (csp.dom[var] & ~revised).bit_count()
                if not csp.set_dom(var, revised):
                    self.transfer_conflicts += 1
                    return False
        return True

    def _route_priority(self, slot: int, dest: int):
        """Rank a candidate D value by supported cargo arcs and future pickups.

        This is ordering ONLY; every valid value remains searchable.
        Prefer serving cargo immediately, then shortest feasible repositioning
        toward a future cargo pickup. A terminal NONE is chosen when there is
        no potential cargo work left for this vessel.
        """
        vessel_idx, k = self.slots[slot]
        qdomains = self.csp.dom
        origin_mask = self.org_set(slot)
        if dest == self.NONE_P:
            future = any(
                (qdomains[q] & (1 << s))
                for row in self.Q for q in row
                for s in range(slot, self.S)
                if self.slots[s][0] == vessel_idx
            )
            return (0 if not future else 3, 0, 0, dest)

        typ = self.slot_type[slot]
        active_edge = 0.0
        pickup_ports = set()
        for ii, row in enumerate(self.Q):
            paths = self.path_templates[ii]
            for j, var in enumerate(row):
                dom = qdomains[var]
                if not dom:
                    continue
                if dom & (1 << slot):
                    # Prefer arcs required by high-criticality cargo.
                    if paths is not None:
                        for path in paths:
                            if j < len(path):
                                o, d, mask = path[j]
                                if d == dest and (mask & (1 << slot)) and (origin_mask & (1 << o)):
                                    active_edge += self.cargo_criticality[ii] * (3 if dom.bit_count() == 1 else 1)
                                    break
                    else:
                        # Conservative hint when path enumeration hit its cap.
                        if dest == self.items[ii]["d"]:
                            active_edge += 0.5
                if paths is None:
                    continue
                has_future_slot = False
                for s in bits(dom & ((1 << self.S) - 1)):
                    vi, pos = self.slots[s]
                    if vi == vessel_idx and pos > k:
                        has_future_slot = True
                        break
                if has_future_slot:
                    for path in paths:
                        if j < len(path):
                            o, d, mask = path[j]
                            if any(
                                mask & (1 << s)
                                for s in bits(dom & ((1 << self.S) - 1))
                                if self.slots[s][0] == vessel_idx and self.slots[s][1] > k
                            ):
                                pickup_ports.add(o)

        closest = min((self.type_sp[typ][dest][p] for p in pickup_ports), default=10**9)
        # Ranks: direct cargo-compatible arcs first, then useful repositioning.
        return (0 if active_edge else 1, -active_edge, closest, dest)

    # -- build variables and constraints ------------------------------------
    def _build(self):
        csp, n = self.csp, self.n
        NONE_P = self.NONE_P
        self.D = []
        for sid, (vi, k) in enumerate(self.slots):
            v = self.vessels[vi]
            t, start = v["type"], self.pid[v["start_port"]]
            if k == 0:
                mask = self.out_mask[t][start] | (1 << NONE_P)
            else:
                mask = (1 << NONE_P)
                for o in range(n):
                    mask |= self.out_mask[t][o]
            self.D.append(csp.new_var(mask))

        for sid, (vi, k) in enumerate(self.slots):
            if k == 0:
                continue
            t = self.vessels[vi]["type"]
            prev = self.slot_id[(vi, k - 1)]
            sup_a = [0] * (n + 1)      # prev value -> allowed current values
            sup_b = [0] * (n + 1)      # current value -> allowed prev values
            for a in range(n + 1):
                if a == NONE_P:
                    sup_a[a] = 1 << NONE_P
                else:
                    sup_a[a] = self.out_mask[t][a] | (1 << NONE_P)
            for a in range(n + 1):
                for b in bits(sup_a[a]):
                    sup_b[b] |= 1 << a
            csp.add_prop(BinaryTable(self.D[prev], self.D[sid], sup_a, sup_b),
                         [self.D[prev], self.D[sid]])

        self.q_lo = len(csp.dom)          # Q variables are numbered after all D
        self.Q = []
        for ii, it in enumerate(self.items):
            row = []
            static_masks = self.path_static_masks[ii]
            for j in range(self.m):
                full = (1 << (self.S + 1)) - 1
                if j == 0:
                    full &= ~(1 << self.NONE_Q)   # origin != destination
                # This mask is the union of every cycle-free path's carrier
                # candidates at this leg; no extra variables are introduced.
                if static_masks is not None and static_masks[j]:
                    full &= static_masks[j]
                row.append(csp.new_var(full))
            self.Q.append(row)

        for ii in range(len(self.items)):
            csp.add_prop(ItemFlow(self, ii), self.Q[ii] + self.D)
        for ii in range(1, len(self.items)):          # identical units: order leg 0
            a, b = self.items[ii - 1], self.items[ii]
            if self.split and a["sid"] == b["sid"]:
                csp.add_prop(LexLeg0(self.Q[ii - 1][0], self.Q[ii][0]),
                             [self.Q[ii - 1][0], self.Q[ii][0]])
        csp.add_prop(Purpose(self), self.D + [q for row in self.Q for q in row])
        csp.add_prop(Capacity(self), [q for row in self.Q for q in row])
        csp.add_prop(Timing(self), self.D + [q for row in self.Q for q in row])

        # static degree: number of propagators touching each variable
        self.degree = [len(w) for w in csp.watch]

    # -- helpers used by propagators -----------------------------------------
    def org_set(self, sid: int) -> int:
        vi, k = self.slots[sid]
        if k == 0:
            return 1 << self.pid[self.vessels[vi]["start_port"]]
        return self.csp.dom[self.D[self.slot_id[(vi, k - 1)]]] & self.pmask

    def dest_set(self, sid: int) -> int:
        return self.csp.dom[self.D[sid]] & self.pmask

    def bounds(self):
        """Earliest-start lower bounds for every voyage slot, from the current
        domains.  Edges: a vessel's slots in order, and cargo transfers between
        consecutive legs whose carriers are already fixed.  Slot duration lower
        bound = cheapest arc still possible.  Returns (dep, arr, mind) or None
        when the horizon is exceeded or the precedence graph has a cycle."""
        csp = self.csp
        if self._bounds_version == csp.version:
            return self._bounds
        INF = 10**9
        S, H = self.S, self.H
        mind = [INF] * S
        for sid, (vi, k) in enumerate(self.slots):
            t = self.vessels[vi]["type"]
            dsts = self.dest_set(sid)
            best = INF
            for o in bits(self.org_set(sid)):
                for d in bits(self.out_mask[t][o] & dsts):
                    dd = self.dur[t][(o, d)]
                    if dd < best:
                        best = dd
            mind[sid] = best
        edges = []
        for sid, (vi, k) in enumerate(self.slots):
            if k > 0:
                edges.append((self.slot_id[(vi, k - 1)], sid))
        for row in self.Q:
            seq = []
            for q in row:
                m = csp.dom[q]
                s = m.bit_length() - 1 if m & (m - 1) == 0 else -1
                seq.append(s if s != self.NONE_Q else -1)
            for x, y in zip(seq, seq[1:]):
                if x >= 0 and y >= 0:
                    edges.append((x, y))
        dep = [0] * S
        arr = [mind[s] for s in range(S)]
        result = (dep, arr, mind)
        for _ in range(S + 2):
            changed = False
            for x, y in edges:
                if arr[x] >= INF:
                    if dep[y] < INF:
                        dep[y] = INF
                        arr[y] = INF
                        changed = True
                elif arr[x] > dep[y]:
                    dep[y] = arr[x]
                    arr[y] = INF if mind[y] >= INF else dep[y] + mind[y]
                    changed = True
                    if arr[y] < INF and arr[y] > H and self._used(y):
                        result = None
                        changed = False
                        break
            if result is None or not changed:
                break
        else:
            result = None                      # precedence cycle
        self._bounds_version, self._bounds = csp.version, result
        return result

    def _used(self, sid: int) -> bool:
        """Slot certainly used (its destination cannot be NONE)."""
        return not (self.csp.dom[self.D[sid]] >> self.NONE_P) & 1

    def fixed_port(self, var: int):
        m = self.csp.dom[var]
        if m & (m - 1) == 0:
            p = m.bit_length() - 1
            return None if p == self.NONE_P else p
        return None

    # -- itinerary-level search -----------------------------------------------
    def _scarce_transfer_groups(self):
        """Conservative AUTO itinerary branching: scarce, multi-vessel transfers.

        Grouping every shipment adds a potentially huge number of grouped
        alternatives before the route has been settled. We therefore apply
        early itinerary branching only when an item has a two-leg geographic
        path connecting *different* vessel types and at least one leg would
        fill a feeder vessel completely. This is a general resource bottleneck
        (not a shipment-ID special case). Other cases retain weighted-MRV.

        These groups only affect branching order: no candidates are excluded.
        """
        csp = self.csp
        result = []
        for ii, row in enumerate(self.Q):
            paths = self.path_templates[ii]
            if not paths or self.m < 2:
                continue
            if (csp.dom[row[0]].bit_count() <= 1
                    or csp.dom[row[1]].bit_count() <= 1
                    or csp.dom[row[0]].bit_count() > 4):
                # A wide first-leg domain creates large grouped branches.
                # Preserve the old search instead of grouping speculatively.
                continue
            item = self.items[ii]
            qty = item["qty"]
            eligible_paths = []
            for path in paths:
                if len(path) != 2:
                    continue
                leg0 = path[0][2] & csp.dom[row[0]]
                leg1 = path[1][2] & csp.dom[row[1]]
                if not leg0 or not leg1:
                    continue
                types0 = {self.slot_type[s] for s in bits(leg0)}
                types1 = {self.slot_type[s] for s in bits(leg1)}
                if types0 == types1:  # no scarce inter-type connection
                    continue
                # A full feeder is highly constraining for other shipments.
                # Only consider genuinely compatible per-leg slot candidates.
                full_feeder = any(
                    self.vessels[self.slots[s][0]]["type"] == "feeder"
                    and self.slot_capacity[s] == qty
                    for s in bits(leg0 | leg1)
                )
                if full_feeder:
                    eligible_paths.append((leg0.bit_count() * leg1.bit_count(),
                                           path))
            if not eligible_paths:
                continue
            # Prefer shipments with tighter compatible slot options and fewer
            # alternative two-leg paths; deadlines break ties only.
            estimate = min(p[0] for p in eligible_paths)
            if estimate > min(32, self.itinerary_candidate_limit):
                continue
            score = (estimate, len(paths), item["dl"])
            result.append((score, ii, (0, 1), "pair"))
        result.sort()
        return result

    def _itinerary_groups(self):
        """Choose promising shipments with >=2 undecided cargo legs.

        Each option is (ii, selected Q-leg positions, candidate kind, rank).
        The itinerary heuristic is *only* about choosing an existing branch:
        every allowed Q combination stays in the search tree. A modest
        preference for constrained inter-vessel connections avoids rigid
        cargo-first branching, which regressed on earlier HARD subsets.
        """
        if self.itinerary_mode == "auto":
            return self._scarce_transfer_groups()
        csp = self.csp
        choices = []
        for ii, row in enumerate(self.Q):
            paths = self.path_templates[ii]
            if not paths:  # None means enumeration overflow/disabled
                continue
            unsettled = [j for j, q in enumerate(row)
                         if csp.dom[q].bit_count() > 1]
            if len(unsettled) < 2:
                continue

            # Estimate a full tuple enumeration before trying it; an estimate
            # that is too high merely selects a smaller group, not a prune.
            full_cost = 0
            if self.itinerary_mode in ("auto", "full"):
                for path in paths:
                    product = 1
                    for j in range(self.m):
                        mask = ((csp.dom[row[j]] & path[j][2]) if j < len(path)
                                else (csp.dom[row[j]] & (1 << self.NONE_Q)))
                        product *= mask.bit_count()
                        if product > self.itinerary_candidate_limit * 2:
                            break
                    full_cost += product
                    if full_cost > self.itinerary_candidate_limit * 2:
                        break

            if (self.itinerary_mode in ("auto", "full")
                    and full_cost <= self.itinerary_candidate_limit):
                indices = tuple(range(self.m))
                kind = "full"
                estimate = max(1, full_cost)
            elif self.itinerary_mode in ("auto", "pair"):
                # Prefer two adjacent unresolved Q legs, so we can check the
                # order of their actual carrier voyages together.
                pairs = [(j, j + 1) for j in range(self.m - 1)
                         if j in unsettled and (j + 1) in unsettled]
                if not pairs:
                    continue
                indices = min(pairs, key=lambda pair:
                              csp.dom[row[pair[0]]].bit_count()
                              * csp.dom[row[pair[1]]].bit_count())
                kind = "pair"
                estimate = (csp.dom[row[indices[0]]].bit_count()
                            * csp.dom[row[indices[1]]].bit_count())
                if estimate > self.itinerary_candidate_limit * 3:
                    continue  # leave this shipment to normal Q branching
            else:
                continue

            # With lots of candidate assignments, a large grouped choice can
            # cost more than it saves. Criticality is a soft ranking only.
            it = self.items[ii]
            needed = min(len(path) for path in paths)
            short_deadline_bonus = 150.0 / max(30, it["dl"])
            score = (estimate ** 0.55
                     / (1.0 + 0.30 * max(0, needed - 1)
                        + 0.35 * short_deadline_bonus))
            # Full tuples tend to constrain the most, but only favor them if
            # their enumeration is already small.
            if kind == "full":
                score *= 0.85
            choices.append((score, ii, tuple(indices), kind))
        choices.sort()
        return choices

    def _generate_itinerary_candidates(self, ii, positions, limit):
        """Enumerate ALL supported grouped assignments, or return None.

        None is an overflow signal, never a truncated candidate set. In that
        case the caller tries a smaller grouping or regular single-variable
        branching, so no valid solution can be lost from candidate limits.

        Lower-bound times are optimistic: a later sailing may wait for cargo.
        Arrival feasibility is checked as max(earliest sailing, preceding
        arrival) + travel time, not by assuming fixed departure times.
        """
        t0 = time.perf_counter()
        csp = self.csp
        row = self.Q[ii]
        paths = self.path_templates[ii]
        earliest = self.bounds()
        if earliest is None:
            return []  # old propagation will also report the contradiction
        dep = earliest[0]
        deadline = min(self.items[ii]["dl"], self.H)
        origin_cache = {}
        dest_cache = {}
        options = []
        seen = set()
        overflow = False
        # Position-specific data is gathered across *every* static port path.
        # Path differences with the same Q tuple are deliberately deduplicated;
        # propagation later determines which route can actually be used.
        for path in paths:
            if overflow:
                break
            path_positions = []
            valid_path = True
            for j in positions:
                if j >= len(path):
                    mask = csp.dom[row[j]] & (1 << self.NONE_Q)
                    candidates = [(self.NONE_Q, 0)] if mask else []
                else:
                    o, d, eligible = path[j]
                    candidates = []
                    for slot in bits(csp.dom[row[j]] & eligible):
                        if slot not in origin_cache:
                            origin_cache[slot] = self.org_set(slot)
                            dest_cache[slot] = self.dest_set(slot)
                        if not (origin_cache[slot] & (1 << o)):
                            continue
                        if not (dest_cache[slot] & (1 << d)):
                            continue
                        if dep[slot] >= 10**9:
                            continue
                        travel = self.dur[self.slot_type[slot]].get((o, d))
                        if travel is not None:
                            candidates.append((slot, travel))
                if not candidates:
                    valid_path = False
                    break
                path_positions.append(candidates)
            if not valid_path:
                continue

            # Other Q legs must have at least basic support from the same port
            # path. This is a necessary condition, not a timing assumption.
            for j in range(self.m):
                if j in positions:
                    continue
                mask = ((csp.dom[row[j]] & path[j][2]) if j < len(path)
                        else (csp.dom[row[j]] & (1 << self.NONE_Q)))
                if not mask:
                    valid_path = False
                    break
            if not valid_path:
                continue

            vals = [0] * len(positions)
            def rec(k, visited, prior_slot, prior_arr):
                nonlocal overflow
                if overflow:
                    return
                if k == len(positions):
                    key = tuple(vals)
                    if key not in seen:
                        if len(seen) >= limit:
                            overflow = True
                            return
                        seen.add(key)
                        # Earlier slots usually mean less repositioning and
                        # more slack; use this solely as VALUE ORDERING.
                        total = sum(dep[s] for s in key if s != self.NONE_Q)
                        options.append((prior_arr, total, key))
                    return
                j = positions[k]
                for slot, duration in path_positions[k]:
                    if overflow:
                        return
                    if slot != self.NONE_Q and slot in visited:
                        continue
                    if slot == self.NONE_Q:
                        # NONE must occur only after the path reaches its goal,
                        # which the corresponding path template guarantees.
                        vals[k] = slot
                        rec(k + 1, visited, slot, prior_arr)
                        continue
                    if prior_slot != -1 and prior_slot != self.NONE_Q:
                        v1, k1 = self.slots[prior_slot]
                        v2, k2 = self.slots[slot]
                        if v1 == v2 and k2 <= k1:
                            continue
                    # For adjacent selected legs this also checks transfer
                    # feasibility; full tuples propagate arrival across all
                    # legs. Non-adjacent selected legs are never grouped.
                    arrival = max(dep[slot], prior_arr) + duration
                    if arrival > deadline:
                        continue
                    vals[k] = slot
                    rec(k + 1, visited | {slot}, slot, arrival)

            rec(0, frozenset(), -1, 0)
        self.itinerary_generation_seconds += time.perf_counter() - t0
        if overflow:
            self.itinerary_enumeration_overflows += 1
            return None
        self.itinerary_candidates_generated += len(options)
        options.sort(key=lambda item: (item[0], item[1], item[2]))
        return [choice for _, _, choice in options]

    def _choose_itinerary_branch(self):
        """Return (item, positions, kind, ALL candidates), or None.

        At most a few candidate shipment groups are inspected. If none are
        small enough, the old weighted-MRV search remains a complete fallback.
        """
        if self.itinerary_mode == "off":
            return None
        groups = self._itinerary_groups()
        for _, ii, positions, kind in groups[:3]:
            candidates = self._generate_itinerary_candidates(
                ii, positions, self.itinerary_candidate_limit)
            if candidates is None:
                self.itinerary_fallbacks += 1
                if kind == "full" and self.itinerary_mode == "auto":
                    # A full domain can unexpectedly explode because several
                    # paths overlap. Reattempt a small, complete pair branch.
                    row = self.Q[ii]
                    pairs = [(j, j + 1) for j in range(self.m - 1)
                             if self.csp.dom[row[j]].bit_count() > 1
                             and self.csp.dom[row[j + 1]].bit_count() > 1]
                    for pair in sorted(pairs, key=lambda p:
                                       self.csp.dom[row[p[0]]].bit_count()
                                       * self.csp.dom[row[p[1]]].bit_count()):
                        candidates = self._generate_itinerary_candidates(
                            ii, pair, self.itinerary_candidate_limit)
                        if candidates is not None:
                            positions, kind = pair, "pair"
                            break
                if candidates is None:
                    continue
            if candidates:  # empty => defer contradiction check to old solver
                return ii, positions, kind, candidates
        if groups:
            self.itinerary_fallbacks += 1
        return None

    # -- search ----------------------------------------------------------------
    def select_var(self):
        """Weighted MRV with a *modest* critical-cargo bonus.

        Unlike rigid cargo-first, the bonus is bounded: a constrained vessel
        destination can still beat a cargo variable. D slots with a fixed
        cargo assignment receive a priority boost; speculative empty routes
        receive a small penalty. None of this removes valid values.
        """
        csp = self.csp
        dom, watch, weight = csp.dom, csp.watch, csp.weight
        committed = set()
        if self.critical_cargo_branching:
            for row in self.Q:
                for q in row:
                    m = dom[q]
                    if m.bit_count() == 1:
                        s = m.bit_length() - 1
                        if s != self.NONE_Q:
                            committed.add(s)
        best, best_key = None, None
        for v, m in enumerate(dom):
            if m.bit_count() == 1:
                continue
            wdeg = sum(weight[p] for p in watch[v])
            score = m.bit_count() / wdeg
            if self.critical_cargo_branching:
                if v >= self.q_lo:
                    ii = (v - self.q_lo) // self.m
                    score /= (1.0 + 0.10 * (self.cargo_criticality[ii] - 1.0))
                else:
                    sid = v
                    score *= 0.98 if sid in committed else 1.02
            key = (score, -self.degree[v])
            if best_key is None or key < best_key:
                best, best_key = v, key
        return best

    def order_values(self, var: int):
        """Cargo: earliest plausible carrier, or NONE if the leg is unused.

        Vessel route: prefer cargo-serving arcs and useful repositioning.
        --no-cargo-aware-order restores the original trial-LCV ordering.
        """
        csp = self.csp
        vals = list(bits(csp.dom[var]))
        if len(vals) == 1:
            return vals
        if var >= self.q_lo:
            bd = self.bounds()
            dep = bd[0] if bd else [0] * self.S
            return sorted(vals, key=lambda s: (0, 0) if s == self.NONE_Q
                          else (1, dep[s], s))
        if self.cargo_aware_order:
            ordered = sorted((v for v in vals if v != self.NONE_P),
                             key=lambda d: self._route_priority(var, d))
            vi, k = self.slots[var]
            if self.NONE_P in vals:
                # Early termination is especially useful when only one cargo
                # item may still require this vessel. With competing items,
                # stopping greedily tends to cause massive backtracking.
                users = 0
                vessel_mask = sum(1 << s for s, (v, _k) in enumerate(self.slots)
                                  if v == vi)
                for row in self.Q:
                    if any(self.csp.dom[q] & vessel_mask for q in row):
                        users += 1
                        if users > 1:
                            break
                if users <= 1:
                    mark = len(csp.trail)
                    csp.set_dom(var, 1 << self.NONE_P)
                    feasible_stop = csp.propagate([var], learn=False)
                    csp.undo(mark)
                    if feasible_stop:
                        return [self.NONE_P] + ordered
                # Delaying a stop is a heuristic preference, never permission
                # to DROP the NONE value: doing so breaks completeness.
                return ordered + [self.NONE_P]
            return ordered
        scored = []
        for val in vals:
            mark = len(csp.trail)
            csp.set_dom(var, 1 << val)
            ok = csp.propagate([var], learn=False)
            if ok:
                scored.append((-sum(m.bit_count() for m in csp.dom), val))
            csp.undo(mark)
        scored.sort()
        return [val for _, val in scored]

    def backtrack(self, depth=0):
        csp = self.csp
        csp.nodes += 1

        self.max_depth = max(self.max_depth, depth)

        if csp.nodes % 100000 == 0:
            fixed = sum(
                mask.bit_count() == 1
                for mask in csp.dom
            )

            elapsed = time.perf_counter() - self.search_start

            print(
                f"Nodes: {csp.nodes:,} | "
                f"Depth: {depth} | "
                f"Max depth: {self.max_depth} | "
                f"Fixed: {fixed}/{len(csp.dom)} | "
                f"Backtracks: {csp.backtracks:,} | "
                f"Cargo decisions: {self.cargo_decisions:,} | "
                f"Itinerary groups: {self.itinerary_group_decisions:,} | "
                f"Route decisions: {self.route_decisions:,} | "
                f"Transfer prunes: {self.transfer_q_values_removed:,} | "
                f"Empty-cycle prunes: {self.empty_cycle_values_removed:,} | "
                f"Elapsed: {elapsed:.1f}s",
                flush=True
            )

        if self.deadline is not None and time.perf_counter() > self.deadline:
            raise Timeout
        if csp.nodes > self.node_limit:
            raise Restart
        # A grouped itinerary branch sets two or more linked Q variables at
        # once, then performs the ordinary full propagation/rollback. Unlike
        # adding every itinerary as a CSP variable, this creates no new model
        # variables and consumes no persistent exponential domain.
        grouped = self._choose_itinerary_branch()
        if grouped is not None:
            ii, positions, kind, candidates = grouped
            self.itinerary_group_decisions += 1
            self.itinerary_full_decisions += kind == "full"
            self.itinerary_pair_decisions += kind == "pair"
            sid = self.items[ii]["sid"]
            self.itinerary_decisions_by_shipment[sid] = (
                self.itinerary_decisions_by_shipment.get(sid, 0) + 1)
            for candidate in candidates:
                if self.deadline is not None and time.perf_counter() > self.deadline:
                    raise Timeout
                self.itinerary_candidates_tried += 1
                self.cargo_decisions += 1
                mark = len(csp.trail)
                changed = []
                ok = True
                for j, slot in zip(positions, candidate):
                    q = self.Q[ii][j]
                    if not csp.set_dom(q, 1 << slot):
                        ok = False
                        break
                    changed.append(q)
                if ok and csp.propagate(changed) and self.backtrack(depth + 1):
                    return True
                self.itinerary_candidate_failures += 1
                csp.undo(mark)
            csp.backtracks += 1
            return False

        var = self.select_var()
        if var is None:
            return True
        if var >= self.q_lo:
            self.cargo_decisions += 1
        else:
            self.route_decisions += 1
        for val in self.order_values(var):
            mark = len(csp.trail)
            csp.set_dom(var, 1 << val)
            if csp.propagate([var]) and self.backtrack(depth + 1):
                return True
            csp.undo(mark)
        csp.backtracks += 1
        return False

    def solve(self, time_limit=None, restart_base=None):
        """Backtracking, optionally with Luby restarts: run i stops after
        restart_base * luby(i) nodes and restarts from the root, keeping the
        dom/wdeg weights learnt so far.  Run lengths are unbounded, so the
        search stays complete.  Off by default: on medium, restarts (base 100
        or 5000) turned a 32 s solve into a 600 s timeout.  Returns "timeout"
        once time_limit seconds pass."""
        csp = self.csp
        self.max_depth = 0
        self.search_start = time.perf_counter() 
        self.deadline = (None if time_limit is None
                         else time.perf_counter() + time_limit)
        if not csp.propagate():
            return "infeasible", None

        root = len(csp.trail)
        run = 0
        while True:
            run += 1
            self.restarts = run - 1
            self.node_limit = (float("inf") if restart_base is None
                               else csp.nodes + restart_base * luby(run))
            try:
                ok = self.backtrack()
            except Restart:
                csp.undo(root)
                continue
            except Timeout:
                csp.undo(root)
                return "timeout", None
            return ("feasible", self.extract()) if ok else ("infeasible", None)

    # -- schedule extraction -----------------------------------------------------
    def _compact_empty_cycles(self, voyages):
        """Safely remove empty closed walks in the *output* schedule.

        If a vessel makes an empty A->...->A loop, it can instead wait at A.
        Deleting those voyages keeps all loaded sailings at their original
        times, preserves the vessel's port at its next pickup, and cannot
        worsen deadlines or capacity. Compacting output is preferable to
        rejecting entire search branches, which was slower on HARD C9.
        """
        by_vessel = {v["id"]: [] for v in self.vessels}
        for voyage in voyages:
            by_vessel[voyage.vessel_id].append(voyage)
        kept = []
        for vessel_id, sequence in by_vessel.items():
            sequence.sort(key=lambda v: v.depart_hour)
            stack = []
            for voyage in sequence:
                stack.append(voyage)
                if voyage.cargo:
                    continue
                # Only a contiguous tail of certainly empty voyages can be
                # dropped; a loaded voyage would create a hard boundary.
                for i in range(len(stack) - 2, -1, -1):
                    if stack[i].cargo:
                        break
                    if stack[i].origin == voyage.destination:
                        count = len(stack) - i
                        del stack[i:]
                        self.empty_cycles_removed_from_schedule += 1
                        self.empty_voyages_removed_from_schedule += count
                        break
            kept.extend(stack)
        return sorted(kept, key=lambda v: (v.depart_hour, v.vessel_id))

    def extract(self) -> Schedule:
        csp = self.csp
        dep, arr, _ = self.bounds()
        carried = {}
        for ii, row in enumerate(self.Q):
            for q in row:
                s = csp.dom[q].bit_length() - 1
                if s != self.NONE_Q:
                    carried.setdefault(s, {}).setdefault(self.items[ii]["sid"], 0)
                    carried[s][self.items[ii]["sid"]] += self.items[ii]["qty"]
        voyages = []
        for sid, (vi, k) in enumerate(self.slots):
            d = self.fixed_port(self.D[sid])
            if d is None:
                continue
            o = (self.pid[self.vessels[vi]["start_port"]] if k == 0
                 else self.fixed_port(self.D[self.slot_id[(vi, k - 1)]]))
            cargo = [CargoLoad(shipment_id=s, quantity=q)
                     for s, q in sorted(carried.get(sid, {}).items())]
            voyages.append(Voyage(vessel_id=self.vessels[vi]["id"],
                                  origin=self.ports[o], destination=self.ports[d],
                                  depart_hour=dep[sid], cargo=cargo))
        voyages.sort(key=lambda v: (v.depart_hour, v.vessel_id))
        if self.empty_cycle_postprocess:
            voyages = self._compact_empty_cycles(voyages)
        return Schedule(voyages=voyages)


# ---------------------------------------------------------------------------
# Propagators
# ---------------------------------------------------------------------------
class ItemFlow:
    """Leg chaining for one cargo item (generalised arc consistency along the
    leg chain): forward reachability of ports, then backward pruning."""

    def __init__(self, model: Model, ii: int):
        self.M, self.ii = model, ii

    def run(self, csp: CSP) -> bool:
        M = self.M
        if not M._propagate_cargo_paths(self.ii):
            return False
        if not M._propagate_transfer_compatibility(self.ii):
            return False
        it = M.items[self.ii]
        Q = M.Q[self.ii]
        m, NQ = M.m, M.NONE_Q
        dbit = 1 << it["d"]
        out_mask = M.out_mask
        vtype = M.slot_type

        # fixed-slot bookkeeping: all-different, nothing after delivery
        for j in range(m):
            dj = csp.dom[Q[j]]
            if dj & (dj - 1) == 0:
                s = dj.bit_length() - 1
                if s == NQ:
                    for j2 in range(j + 1, m):
                        if not csp.set_dom(Q[j2], 1 << NQ):
                            return False
                else:
                    for j2 in range(m):
                        if j2 != j and not csp.set_dom(Q[j2], csp.dom[Q[j2]] & ~(1 << s)):
                            return False
        for j in range(m - 1):                 # a later active leg => this one active
            if not csp.dom[Q[j + 1]] & (1 << NQ):
                if not csp.set_dom(Q[j], csp.dom[Q[j]] & ~(1 << NQ)):
                    return False

        bd = M.bounds()
        if bd is None:
            return False
        lb = bd[0]
        H, dl, dst = M.H, it["dl"], it["d"]
        sp_to_dst = [M.sp[p][dst] for p in range(M.n)]
        tloc = {it["o"]: 0}                  # port -> earliest time item can be there
        pairs = []                           # per leg: slot -> [(o, d)]
        kept = []
        for j in range(m):
            allowed, nxt_t, leg_pairs = 0, {}, {}
            for s in bits(csp.dom[Q[j]]):
                if s == NQ:
                    if dst in tloc:
                        allowed |= 1 << NQ
                        if tloc[dst] < nxt_t.get(dst, 10**9):
                            nxt_t[dst] = tloc[dst]
                    continue
                if lb[s] >= 10**9:
                    continue
                t = vtype[s]
                dsts = M.dest_set(s)
                found = []
                for o in bits(M.org_set(s)):
                    if o == dst or o not in tloc:
                        continue
                    dep = lb[s] if lb[s] > tloc[o] else tloc[o]
                    for d in bits(out_mask[t][o] & dsts):
                        a = dep + M.dur[t][(o, d)]
                        if a <= H and a + sp_to_dst[d] <= dl:
                            found.append((o, d))
                            if a < nxt_t.get(d, 10**9):
                                nxt_t[d] = a
                if found:
                    allowed |= 1 << s
                    leg_pairs[s] = found
            if not allowed:
                return False
            kept.append(allowed)
            pairs.append(leg_pairs)
            tloc = nxt_t
        if dst not in tloc:
            return False

        need = dbit
        for j in range(m - 1, -1, -1):
            keep, need_new = 0, 0
            for s in bits(kept[j]):
                if s == NQ:
                    if need & dbit:
                        keep |= 1 << NQ
                        need_new |= dbit
                    continue
                good = 0
                for o, d in pairs[j][s]:
                    if need & (1 << d):
                        good |= 1 << o
                if good:
                    keep |= 1 << s
                    need_new |= good
            if not keep:
                return False
            if not csp.set_dom(Q[j], keep):
                return False
            need = need_new
        return bool(need & (1 << it["o"]))


class Purpose:
    """Dominance rule: a vessel's last used voyage must carry cargo (trailing
    empty voyages can be deleted without affecting any other event).  Empty
    voyages are only kept as repositioning before a later voyage."""

    def __init__(self, model: Model):
        self.M = model

    def run(self, csp: CSP) -> bool:
        M = self.M
        can_carry = [False] * M.S
        for row in M.Q:
            for q in row:
                for s in bits(csp.dom[q] & ~(1 << M.NONE_Q)):
                    can_carry[s] = True
        none_bit = 1 << M.NONE_P
        for vi, v in enumerate(M.vessels):
            nxt_possible = False                   # a later slot may still be useful
            for k in range(v["max_voyages"] - 1, -1, -1):
                s = M.slot_id[(vi, k)]
                useful = can_carry[s] or nxt_possible
                m = csp.dom[M.D[s]]
                if not useful:
                    if not m & none_bit:       # certainly used but serves no purpose
                        return False
                    if not csp.set_dom(M.D[s], none_bit):
                        return False
                    nxt_possible = False
                    continue
                nxt_possible = bool(m & ~none_bit)
        # certainly-used slots with no later use must still have a carrier
        for vi, v in enumerate(M.vessels):
            for k in range(v["max_voyages"]):
                s = M.slot_id[(vi, k)]
                if csp.dom[M.D[s]] & (1 << M.NONE_P):
                    continue                           # may be unused
                last = (k == v["max_voyages"] - 1) or \
                    csp.dom[M.D[M.slot_id[(vi, k + 1)]]] == (1 << M.NONE_P)
                if last and not can_carry[s]:
                    return False

        if M.empty_cycle_pruning:
            # A vessel that travels from P back to P with no cargo on any
            # intervening voyage can stay at P instead. Since every affected
            # voyage is *certainly* empty (absent from every Q domain), this
            # closed walk is dominated by deleting it and compacting the
            # remaining voyage slots; timing can only improve.
            #
            # We do not ban port revisits in general: loaded voyages and
            # cargo-eligible slots break the certainly-empty interval.
            for vi, vessel in enumerate(M.vessels):
                slots = [M.slot_id[(vi, k)] for k in range(vessel["max_voyages"])]
                for end_k, end_s in enumerate(slots):
                    end_var = M.D[end_s]
                    end_dom = csp.dom[end_var]
                    if can_carry[end_s] or not (end_dom & M.pmask):
                        continue
                    prohibited = 0
                    for start_k in range(end_k, -1, -1):
                        start_s = slots[start_k]
                        if can_carry[start_s]:
                            break
                        if end_k == start_k:  # no self-loops in legal routes
                            continue
                        if start_k == 0:
                            start_port = M.pid[vessel["start_port"]]
                        else:
                            before = csp.dom[M.D[slots[start_k - 1]]]
                            if before.bit_count() != 1:
                                continue
                            start_port = before.bit_length() - 1
                            if start_port == M.NONE_P:
                                continue
                        prohibited |= 1 << start_port
                    removed = end_dom & prohibited
                    if removed:
                        updated = end_dom & ~prohibited
                        M.empty_cycle_values_removed += removed.bit_count()
                        if not csp.set_dom(end_var, updated):
                            M.empty_cycle_conflicts += 1
                            return False
        return True


class LexLeg0:
    """Symmetry breaking: units of one shipment are interchangeable, so require
    their first-leg carriers to be non-decreasing (a sorted representative of
    every solution always exists)."""

    def __init__(self, a, b):
        self.a, self.b = a, b

    def run(self, csp: CSP) -> bool:
        da, db = csp.dom[self.a], csp.dom[self.b]
        lo_a = (da & -da).bit_length() - 1
        hi_b = db.bit_length() - 1
        if not csp.set_dom(self.a, da & ((1 << (hi_b + 1)) - 1)):
            return False
        return csp.set_dom(self.b, db & ~((1 << lo_a) - 1))


class Capacity:
    """Per-slot load <= capacity; forward-checks remaining candidates."""

    def __init__(self, model: Model):
        self.M = model

    def run(self, csp: CSP) -> bool:
        M = self.M
        load = [0] * M.S
        for ii, row in enumerate(M.Q):
            for q in row:
                m = csp.dom[q]
                if m & (m - 1) == 0:
                    s = m.bit_length() - 1
                    if s != M.NONE_Q:
                        load[s] += M.items[ii]["qty"]
        cap = M.slot_capacity
        for s in range(M.S):
            if load[s] > cap[s]:
                return False
        for ii, row in enumerate(M.Q):
            qty = M.items[ii]["qty"]
            for q in row:
                m = csp.dom[q]
                if m & (m - 1) == 0:
                    continue
                drop = 0
                for s in bits(m):
                    if s != M.NONE_Q and load[s] + qty > cap[s]:
                        drop |= 1 << s
                if drop and not csp.set_dom(q, m & ~drop):
                    return False
        return True


class Timing:
    """Deadline/horizon checks on the temporal lower bounds, plus forward
    checking of route destinations: a destination is removed when a cargo item
    already fixed to that voyage could no longer meet its deadline."""

    def __init__(self, model: Model):
        self.M = model

    def run(self, csp: CSP) -> bool:
        M = self.M
        bd = M.bounds()
        if bd is None:
            return False
        dep = bd[0]
        INF = 10**9
        on_slot = {}                         # slot -> items fixed onto it
        for ii, row in enumerate(M.Q):
            for q in row:
                m = csp.dom[q]
                if m & (m - 1) == 0:
                    s = m.bit_length() - 1
                    if s != M.NONE_Q:
                        on_slot.setdefault(s, []).append(ii)
        for sid, (vi, k) in enumerate(M.slots):
            if dep[sid] >= INF:
                if sid in on_slot:
                    return False
                continue
            m = csp.dom[M.D[sid]]
            items = on_slot.get(sid, ())
            t = M.vessels[vi]["type"]
            keep = m & (1 << M.NONE_P)
            if items:
                keep = 0                     # slot carries cargo => must be used
            for d in bits(m & M.pmask):
                ok = False
                for o in bits(M.org_set(sid)):
                    dd = M.dur[t].get((o, d))
                    if dd is None:
                        continue
                    a = dep[sid] + dd
                    if a > M.H:
                        continue
                    if all(a + M.sp[d][M.items[ii]["d"]] <= M.items[ii]["dl"]
                           for ii in items):
                        ok = True
                        break
                if ok:
                    keep |= 1 << d
            if keep != m and not csp.set_dom(M.D[sid], keep):
                return False
        return True


# ---------------------------------------------------------------------------
def solve_instance(instance: dict, time_limit=None, restart_base=None,
                   path_preprocessing=PATH_PREPROCESSING,
                   path_enumeration_limit=PATH_ENUMERATION_LIMIT,
                   show_paths=False, cargo_aware_order=CARGO_AWARE_ORDER,
                   critical_cargo_branching=CRITICAL_CARGO_BRANCHING,
                   transfer_pruning=TRANSFER_PRUNING,
                   empty_cycle_pruning=EMPTY_CYCLE_PRUNING,
                   empty_cycle_postprocess=EMPTY_CYCLE_POSTPROCESS,
                   itinerary_mode=ITINERARY_MODE,
                   itinerary_candidate_limit=ITINERARY_CANDIDATE_LIMIT):
    """solve_time_s covers model construction as well as search."""
    t0 = time.perf_counter()
    try:
        model = Model(instance, path_preprocessing=path_preprocessing,
                      path_enumeration_limit=path_enumeration_limit,
                      cargo_aware_order=cargo_aware_order,
                      critical_cargo_branching=critical_cargo_branching,
                      transfer_pruning=transfer_pruning,
                      empty_cycle_pruning=empty_cycle_pruning,
                      empty_cycle_postprocess=empty_cycle_postprocess,
                      itinerary_mode=itinerary_mode,
                      itinerary_candidate_limit=itinerary_candidate_limit)
    except NotImplementedError as exc:
        return None, {"status": "unsupported", "reason": str(exc)}
    if show_paths:
        for it, paths in zip(model.items, model.path_templates):
            if paths is None:
                print(f"[PATHS] {it['sid']}: fallback / disabled", flush=True)
                continue
            print(f"[PATHS] {it['sid']}: {len(paths)} simple port path(s)", flush=True)
            for path in paths[:15]:
                names = ([model.ports[path[0][0]]] +
                         [model.ports[d] for _, d, _ in path]) if path else []
                print("    " + " -> ".join(names), flush=True)
            if len(paths) > 15:
                print(f"    ... {len(paths)-15} more", flush=True)
    status, schedule = model.solve(time_limit=time_limit,
                                   restart_base=restart_base)
    elapsed = time.perf_counter() - t0
    metrics = {
        "status": status,
        "solve_time_s": round(elapsed, 3),
        "time_limit_s": time_limit,
        "nodes_expanded": model.csp.nodes,
        "backtracks": model.csp.backtracks,
        "restarts": model.restarts,
        "propagations": model.csp.propagations,
        "variables": len(model.csp.dom),
        "path_preprocessing": path_preprocessing,
        "static_paths_per_shipment": {
            it["sid"]: (None if paths is None else len(paths))
            for it, paths in zip(model.items, model.path_templates)
        },
        "path_fallback_items": model.path_fallback_items,
        "path_support_calls": model.path_support_calls,
        "path_q_values_removed": model.path_q_values_removed,
        "path_d_values_removed": model.path_d_values_removed,
        "cargo_aware_order": cargo_aware_order,
        "critical_cargo_branching": critical_cargo_branching,
        "transfer_pruning": transfer_pruning,
        "transfer_calls": model.transfer_calls,
        "transfer_q_values_removed": model.transfer_q_values_removed,
        "transfer_conflicts": model.transfer_conflicts,
        "empty_cycle_pruning": empty_cycle_pruning,
        "empty_cycle_values_removed": model.empty_cycle_values_removed,
        "empty_cycle_conflicts": model.empty_cycle_conflicts,
        "empty_cycle_postprocess": model.empty_cycle_postprocess,
        "empty_cycles_removed_from_schedule": model.empty_cycles_removed_from_schedule,
        "empty_voyages_removed_from_schedule": model.empty_voyages_removed_from_schedule,
        "itinerary_mode": model.itinerary_mode,
        "itinerary_candidate_limit": model.itinerary_candidate_limit,
        "itinerary_group_decisions": model.itinerary_group_decisions,
        "itinerary_full_decisions": model.itinerary_full_decisions,
        "itinerary_pair_decisions": model.itinerary_pair_decisions,
        "itinerary_candidates_generated": model.itinerary_candidates_generated,
        "itinerary_candidates_tried": model.itinerary_candidates_tried,
        "itinerary_candidate_failures": model.itinerary_candidate_failures,
        "itinerary_enumeration_overflows": model.itinerary_enumeration_overflows,
        "itinerary_fallbacks": model.itinerary_fallbacks,
        "itinerary_generation_seconds": round(model.itinerary_generation_seconds, 3),
        "itinerary_decisions_by_shipment": model.itinerary_decisions_by_shipment,
        "cargo_decisions": model.cargo_decisions,
        "route_decisions": model.route_decisions,
    }
    if schedule is not None:
        report = simulate(instance, schedule)
        metrics.update(valid=report.valid, voyages=len(schedule.voyages),
                       violations=[v.model_dump() for v in report.violations])
    return schedule, metrics


def main():
    parser = argparse.ArgumentParser(description="Classical CSP for MTRSP with selective itinerary-level branching")
    parser.add_argument("names", nargs="*", default=["easy", "medium", "hard"])
    parser.add_argument("--time-limit", type=float, default=TIME_LIMIT,
                        help="seconds per instance (0 = no limit)")
    parser.add_argument("--restart-base", type=int, default=RESTARTS,
                        help="nodes per Luby unit (0 = no restarts)")
    parser.add_argument("--show-paths", action="store_true",
                        help="print cycle-free port paths per shipment (up to 15)")
    parser.add_argument("--no-path-preprocessing", action="store_true",
                        help="disable cycle-free cargo-path preprocessing")
    parser.add_argument("--path-enumeration-limit", type=int,
                        default=PATH_ENUMERATION_LIMIT,
                        help="per-shipment cap; overflow disables path filtering for that shipment")
    parser.add_argument("--no-cargo-aware-order", action="store_true",
                        help="restore trial-LCV ordering for route destinations")
    parser.add_argument("--no-critical-cargo", action="store_true",
                        help="disable selective cargo-first branching")
    parser.add_argument("--no-transfer-pruning", action="store_true",
                        help="disable temporal compatibility checks (already off by default)")
    parser.add_argument("--enable-transfer-pruning", action="store_true",
                        help="EXPERIMENTAL: enable transfer-time propagation (slower on HARD 14/17)")
    parser.add_argument("--enable-empty-cycle-pruning", action="store_true",
                        help="EXPERIMENTAL: prune certain empty cycles during search (slower on C9 tests)")
    parser.add_argument("--no-empty-cycle-postprocess", action="store_true",
                        help="keep all empty closed walks in the output schedule")
    parser.add_argument("--itinerary-mode", choices=("auto", "full", "pair", "off"),
                        default=ITINERARY_MODE,
                        help="auto: scarce cross-vessel transfers; full/pair: broader experimental grouping; off: old search")
    parser.add_argument("--no-itinerary-branching", action="store_true",
                        help="use original single-variable CSP branching")
    parser.add_argument("--itinerary-candidate-limit", type=int,
                        default=ITINERARY_CANDIDATE_LIMIT,
                        help="maximum grouped candidates; fallback never drops solutions")
    parser.add_argument("--baseline", action="store_true",
                        help="disable all four new optimizations, retaining static path preprocessing")
    args = parser.parse_args()
    if args.path_enumeration_limit < 1:
        parser.error("--path-enumeration-limit must be at least 1")
    if args.itinerary_candidate_limit < 1:
        parser.error("--itinerary-candidate-limit must be at least 1")

    base = Path(__file__).resolve().parent
    out_dir = base / "results" / "classical_csp_itinerary"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in args.names:
        instance = json.loads((base / "data" / f"{name}.json").read_text())
        schedule, metrics = solve_instance(
            instance,
            time_limit=args.time_limit or None,
            restart_base=args.restart_base or None,
            path_preprocessing=not args.no_path_preprocessing,
            path_enumeration_limit=args.path_enumeration_limit,
            show_paths=args.show_paths,
            cargo_aware_order=not (args.baseline or args.no_cargo_aware_order),
            critical_cargo_branching=not (args.baseline or args.no_critical_cargo),
            transfer_pruning=(not args.baseline and args.enable_transfer_pruning
                              and not args.no_transfer_pruning),
            empty_cycle_pruning=(not args.baseline and args.enable_empty_cycle_pruning),
            empty_cycle_postprocess=not (args.baseline or args.no_empty_cycle_postprocess),
            itinerary_mode=("off" if args.baseline or args.no_itinerary_branching
                            else args.itinerary_mode),
            itinerary_candidate_limit=args.itinerary_candidate_limit,
        )
        print(f"{name.upper()}: {metrics}", flush=True)
        if schedule is not None:
            (out_dir / f"{name}_schedule.json").write_text(
                json.dumps(schedule.model_dump(), indent=2))
        (out_dir / f"{name}_metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
