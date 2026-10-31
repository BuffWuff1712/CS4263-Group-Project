from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

from schemas import CargoLoad, Schedule, Voyage
from simulator import simulate

sys.setrecursionlimit(20000)


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
    def __init__(self, instance: dict):
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

        self._bounds_version, self._bounds = -1, None
        self.deadline, self.node_limit, self.restarts = None, float("inf"), 0
        self.csp = CSP()
        self._build()

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
        for it in self.items:
            row = []
            for j in range(self.m):
                full = (1 << (self.S + 1)) - 1
                if j == 0:
                    full &= ~(1 << self.NONE_Q)   # origin != destination
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

    # -- search ----------------------------------------------------------------
    def select_var(self):
        """MRV refined by weighted degree: minimise |domain| / wdeg, where wdeg
        sums the weights of constraints watching the variable (weights grow
        each time a constraint causes a failure).  Static degree breaks ties.
        Branching order never affects completeness."""
        csp = self.csp
        dom, watch, weight = csp.dom, csp.watch, csp.weight
        best, best_key = None, None
        for v, m in enumerate(dom):
            if m & (m - 1) == 0:
                continue
            wdeg = 0.0
            for p in watch[v]:
                wdeg += weight[p]
            key = (bin(m).count("1") / wdeg, -self.degree[v])
            if best_key is None or key < best_key:
                best, best_key = v, key
        return best

    def order_values(self, var: int):
        """Value ordering.
        Route variables D: LCV by trial propagation (value that leaves the
        most remaining domain values across all variables first).
        Carrier variables Q: earliest-available slot first (smallest departure
        lower bound), finishing the item (NONE) first when possible - the
        usual 'least constraining' choice for deadline-driven scheduling."""
        csp = self.csp
        vals = list(bits(csp.dom[var]))
        if len(vals) == 1:
            return vals
        if var >= self.q_lo:
            bd = self.bounds()
            dep = bd[0] if bd else [0] * self.S
            return sorted(vals, key=lambda s: (0, 0) if s == self.NONE_Q
                          else (1, dep[s], s))
        scored = []
        for val in vals:                       # LCV by trial propagation
            mark = len(csp.trail)
            csp.set_dom(var, 1 << val)
            ok = csp.propagate([var], learn=False)
            if ok:
                scored.append((-sum(bin(m).count("1") for m in csp.dom), val))
            csp.undo(mark)
        scored.sort()
        return [val for _, val in scored]

    def backtrack(self):
        csp = self.csp
        csp.nodes += 1
        if self.deadline is not None and time.perf_counter() > self.deadline:
            raise Timeout
        if csp.nodes > self.node_limit:
            raise Restart
        var = self.select_var()
        if var is None:
            return True
        for val in self.order_values(var):
            mark = len(csp.trail)
            csp.set_dom(var, 1 << val)
            if csp.propagate([var]) and self.backtrack():
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
        cap = [M.vessels[M.slots[s][0]]["capacity"] for s in range(M.S)]
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
def solve_instance(instance: dict, time_limit=None, restart_base=None):
    """solve_time_s covers model construction as well as search."""
    t0 = time.perf_counter()
    try:
        model = Model(instance)
    except NotImplementedError as exc:
        return None, {"status": "unsupported", "reason": str(exc)}
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
    }
    if schedule is not None:
        report = simulate(instance, schedule)
        metrics.update(valid=report.valid, voyages=len(schedule.voyages),
                       violations=[v.model_dump() for v in report.violations])
    return schedule, metrics


def main():
    parser = argparse.ArgumentParser(description="Classical CSP baseline for MTRSP")
    parser.add_argument("names", nargs="*", default=["easy", "medium", "hard"])
    parser.add_argument("--time-limit", type=float, default=600,
                        help="seconds per instance (0 = no limit)")
    parser.add_argument("--restart-base", type=int, default=0,
                        help="nodes per Luby unit (0 = no restarts)")
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    out_dir = base / "results" / "classical_csp"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in args.names:
        instance = json.loads((base / "data" / f"{name}.json").read_text())
        schedule, metrics = solve_instance(
            instance,
            time_limit=args.time_limit or None,
            restart_base=args.restart_base or None,
        )
        print(f"{name.upper()}: {metrics}", flush=True)
        if schedule is not None:
            (out_dir / f"{name}_schedule.json").write_text(
                json.dumps(schedule.model_dump(), indent=2))
        (out_dir / f"{name}_metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
