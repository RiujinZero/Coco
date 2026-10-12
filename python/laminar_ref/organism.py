"""NumPy reference organism: the slower faithful kernel (runbook §15).

Mirrors cpp/src/organism.cpp operation-for-operation on the integer path. Only the readout softmax
is floating point, so traces agree with the native pipeline up to rare float rounding ties.
"""
import json
import math
import numpy as np
from . import config as C
from .config import Params
from .rng import Rng, mix64
from .int4 import unpack_unit, pack_unit, layer_forward
from .address import AddressTree, EvidencePool

M64 = (1 << 64) - 1
IN_SCALE = 4
SRC_SENSORY, SRC_ROUTED = 0, 1
R_L1, R_L23, R_L5, R_L6, R_L4 = 1, 2, 4, 8, 16


def level(v):
    return int(np.abs(v.astype(np.int32)).sum()) // len(v)


def scale_trunc(v, shift):
    """trunc(v / 2^shift) toward zero (an arithmetic shift would floor and bias updates one way)."""
    v = v.astype(np.int32)
    return np.where(v >= 0, v >> shift, -((-v) >> shift))


class Trace:
    __slots__ = ("unit", "reached", "source", "hop", "x2", "x3", "x5", "x6", "l1", "l2", "l3", "l5", "l6")

    def __init__(self, unit, source, hop):
        self.unit, self.reached, self.source, self.hop = unit, 0, source, hop


class Stats:
    def __init__(self):
        self.reset()

    def reset(self):
        for k in ("events", "correct", "l1_admissions", "l1_survive", "l23_survive", "l5_survive", "l6_survive", "l4_runs",
                  "neighbor_tests", "neighbor_accepts", "macs", "acc_updates", "level_changes", "lock_births", "lock_splits",
                  "leaf_splits", "novel_events", "candidate_locks", "addr_decisions", "no_candidate", "dedup_skips"):
            setattr(self, k, 0)
        self.loss_sum = 0.0

    def report(self, blk):
        e = max(1, self.events)
        return (f"blk {blk} | loss {self.loss_sum / e:.4f} acc {100 * self.correct / e:.2f}% | addr {self.addr_decisions / e:.1f} cand {self.candidate_locks / e:.2f}"
                f" | L1 {self.l1_admissions / e:.2f} surv {self.l1_survive / e:.2f} L23 {self.l23_survive / e:.2f} L5 {self.l5_survive / e:.2f} L6 {self.l6_survive / e:.2f} L4 {self.l4_runs / e:.3f}"
                f" | nbr {self.neighbor_tests / e:.2f}/{self.neighbor_accepts / e:.3f} | MAC {self.macs / e:.0f} | accupd {self.acc_updates / e:.1f} lvl {self.level_changes / e:.3f}"
                f" | births {self.lock_births} splits {self.lock_splits} leafsplit {self.leaf_splits} | novel {self.novel_events / e:.3f}")


class Organism:
    def __init__(self, p: Params):
        self.p = p
        self.rng = Rng(p.seed)
        U = p.units
        self.w = np.zeros((U, C.UNIT_BYTES), dtype=np.uint8)
        self.acc = np.zeros((U, C.UNIT_NIBBLES), dtype=np.int8) if p.train else None
        self.unit_lock = np.full(U, -1, dtype=np.int64)
        self.epoch = np.zeros(U, dtype=np.uint32)
        self.unit_k = np.zeros(U, dtype=np.int64)
        self.resist = np.full(U, p.resist_init, dtype=np.int64)
        self.alive = np.zeros(U, dtype=np.uint8)
        self.act_ema = np.zeros((U, 4), dtype=np.int64)
        self.assoc_value = [0] * U
        self.assoc_mask = [0] * U
        self.l4ret = np.zeros((U, C.R), dtype=np.int8)
        self.assoc_noconseq = [0] * U
        self.assoc_starve = [0] * U
        free = list(range(U))
        for i in range(U - 1, 0, -1):
            j = self.rng.below(i + 1)
            free[i], free[j] = free[j], free[i]
        free.reverse()
        self.free_units = free
        self.tree = AddressTree(p.leaf_max)
        self.evid = EvidencePool(p.evidence_slots, p.future_buckets)
        self.wout = None
        if p.vocab > C.F:   # [V][F], filled in the same sequential RNG order as the native code
            self.wout = np.array([[(self.rng.uniform() * 2.0 - 1.0) * 0.05 for _ in range(C.F)] for _ in range(p.vocab)], dtype=np.float32)
        self.cur_novel = False
        self.hist = [0] * 16
        self.ws_mem = np.zeros(C.F, dtype=np.int64)
        self.last_l4 = np.zeros(C.R, dtype=np.int8)
        self.event_idx = 0
        self.block_idx = 0
        self.cur_epoch = 1
        self.loss_ema = 0.0
        self.loss_ema_init = False
        self.wrong_ema = 65535
        self.st = Stats()
        self.trace_fp = None
        self.ws = np.zeros(C.F, dtype=np.int32)
        self.traces = []
        self.accepted = []
        self.executed = []
        self.shifts = p.shifts

    # ------------------------------------------------------------------ keys / inputs
    def make_key(self):
        h = self.hist
        p = self.p
        k = h[0] | (h[1] << 8) | (h[2] << 16) | (h[3] << 24)
        hh = 0x5A5A00000000
        for i in range(1, min(p.hist_bytes, 15) + 1):
            hh = ((hh * 0x100000001B3) & M64) ^ h[i]
        ctx = mix64(hh) & 0xFFFFFFFF00000000
        if p.l4_key:
            s = sum(1 << j for j in range(16) if self.last_l4[j] > 0)
            ctx = (ctx & 0xFFFF000000000000) | (s << 32)
        if p.ws_key:
            s = sum(1 << j for j in range(16) if self.ws_mem[j] > 0)
            ctx = (ctx & 0x0000FFFF00000000) | (s << 48)
        return k | ctx

    def make_input(self, key, sym):
        x = np.empty(C.IN, dtype=np.int8)
        for b in range(64):
            x[b] = IN_SCALE if (key >> b) & 1 else -IN_SCALE
        code = mix64(0xC0DE0000 + sym)
        for b in range(64):
            x[64 + b] = IN_SCALE if (code >> b) & 1 else -IN_SCALE
        if self.p.ws_input:
            x[96:128] = np.clip(self.ws_mem >> 3, -127, 127).astype(np.int8)
        return x

    def route_key(self, l2):
        k = 0
        for j in range(C.B):
            v = int(l2[j])
            if v > 0:
                k |= 1 << j
            if v >= 32 or v <= -32:
                k |= 1 << (24 + j)
        return k | (mix64(k) & 0xFFFF000000000000)

    def threshold(self, base, unit, novel, stage, lv):
        ema = int(self.act_ema[unit, stage])
        t = base
        if self.p.death_rel:
            rel = (ema * self.p.death_rel) >> 6
            if rel > t:
                t = rel
        t = (t * int(self.resist[unit])) // 64
        if novel:
            t = (t * self.p.novelty_num) // self.p.novelty_den
        self.act_ema[unit, stage] = ema + ((min(lv, 127) - ema) >> 4)
        return t

    def perturb(self, v, unit, layer, sign):
        if sign == 0:
            return v
        bits = mix64((self.event_idx * 0x9E3779B97F4A7C15) ^ (unit << 20) ^ (layer << 56))
        s = np.array([1 if (bits >> j) & 1 else -1 for j in range(len(v))], dtype=np.int32)
        return np.clip(v.astype(np.int32) + sign * s * self.p.perturb_delta, -127, 127).astype(np.int8)

    # ------------------------------------------------------------------ units
    def alloc_unit(self):
        if not self.free_units:
            return -1
        u = self.free_units.pop()
        p = self.p
        self.alive[u] = 1
        self.resist[u] = p.resist_init
        self.epoch[u] = 0
        self.act_ema[u] = 0
        self.l4ret[u] = 0
        motif = u // p.motif_size
        k = p.k_set[motif % len(p.k_set)]
        if k >= p.motif_size:
            k = p.motif_size - 1
        if k > 254:
            k = 254
        self.unit_k[u] = k
        r = Rng(mix64(((p.seed * 0x1234567) & M64) ^ (0xA55A0000 + u)))
        self.assoc_value[u] = r.next()
        mbits = p.assoc_extra_bits
        kk = k
        while kk > 1:
            mbits += 1
            kk >>= 1
        m = 0
        while mbits:
            b = 1 << r.below(64)
            if not (m & b):
                m |= b
                mbits -= 1
        self.assoc_mask[u] = m
        self.assoc_noconseq[u] = 0
        self.assoc_starve[u] = 0
        return u

    def birth_unit_weights(self, u):
        r = Rng(mix64(((self.p.seed * 0x9876543) & M64) ^ (0xB00B0000 + u)))
        w = np.array([r.below(5) - 2 for _ in range(C.UNIT_NIBBLES)], dtype=np.int8)
        self.w[u] = pack_unit(w)
        if self.acc is not None:
            self.acc[u] = 0

    def free_unit(self, u):
        self.alive[u] = 0
        self.unit_lock[u] = -1
        self.free_units.append(u)

    def neighbors(self, u, k):
        m = self.p.motif_size
        base = (u // m) * m
        off = u - base
        out = []
        for d in range(1, k // 2 + 1):
            out.append(base + (off + m - d) % m)
            out.append(base + (off + d) % m)
        return out

    # ------------------------------------------------------------------ body
    def run_body(self, ev, novel, sign, replay):
        p, st = self.p, self.st
        u, source, hop, src_trace = ev
        W = unpack_unit(self.w[u])
        tr = None
        if len(self.traces) < p.max_active:
            tr = Trace(u, source, hop)
            self.traces.append(tr)
        self.executed.append(u)
        ret = self.l4ret[u]
        sh = self.shifts

        l1 = layer_forward(W, C.NIB_L1, C.IN, C.A, sh[0], self.xin)
        st.macs += C.P1
        st.l1_admissions += 1
        l1 = self.perturb(l1, u, C.LI_1, sign)
        if tr:
            tr.reached |= R_L1
            tr.l1 = l1
        lv = level(l1)
        if lv < self.threshold(p.death_l1, u, novel, 0, lv):
            return
        st.l1_survive += 1

        x2 = np.concatenate([l1, ret[:C.R2]]).astype(np.int8)
        msg = self.traces[src_trace].l2 if (source == SRC_ROUTED and src_trace >= 0) else np.zeros(C.B, dtype=np.int8)
        x3 = np.concatenate([l1, msg]).astype(np.int8)
        l2 = self.perturb(layer_forward(W, C.NIB_L2, C.IN2, C.B, sh[1], x2), u, C.LI_2, sign)
        l3 = self.perturb(layer_forward(W, C.NIB_L3, C.IN3, C.C, sh[2], x3), u, C.LI_3, sign)
        st.macs += C.P2 + C.P3
        if tr:
            tr.reached |= R_L23
            tr.x2, tr.x3, tr.l2, tr.l3 = x2, x3, l2, l3

        if hop < p.max_hops and not replay:
            rk = self.route_key(l2)
            for v in self.neighbors(u, int(self.unit_k[u])):
                if not self.alive[v]:
                    continue
                st.neighbor_tests += 1
                if ((rk ^ self.assoc_value[v]) & self.assoc_mask[v]) == 0 and len(self.nxt) < p.max_active:
                    st.neighbor_accepts += 1
                    self.nxt.append((v, SRC_ROUTED, hop + 1, (len(self.traces) - 1) if tr else -1))
                    self.accepted.append((v, u))
        lv = (level(l2) * C.B + level(l3) * C.C) // (C.B + C.C)
        if lv < self.threshold(p.death_l23, u, novel, 1, lv):
            return
        st.l23_survive += 1

        x5 = np.concatenate([l2, l3]).astype(np.int8)
        l5 = self.perturb(layer_forward(W, C.NIB_L5, C.IN5, C.E, sh[3], x5), u, C.LI_5, sign)
        st.macs += C.P5
        if tr:
            tr.reached |= R_L5
            tr.x5, tr.l5 = x5, l5
        lv = level(l5)
        if lv < self.threshold(p.death_l5, u, novel, 2, lv):
            return
        st.l5_survive += 1

        x6 = np.concatenate([l5, ret[C.R2:]]).astype(np.int8)
        l6 = layer_forward(W, C.NIB_L6, C.IN6, C.F, sh[4], x6)
        st.macs += C.P6
        if tr:
            tr.reached |= R_L6
            tr.x6, tr.l6 = x6, l6
        lv = level(l6)
        if lv < self.threshold(p.death_l6, u, novel, 3, lv):
            return
        st.l6_survive += 1
        self.ws += l6.astype(np.int32)
        if lv < p.recirc_thresh:
            return
        l4 = layer_forward(W, C.NIB_L4, C.IN4, C.R, sh[5], l6)
        st.macs += C.P4
        st.l4_runs += 1
        self.l4ret[u] = l4
        self.last_l4 = l4
        if tr:
            tr.reached |= R_L4

    def run_pass(self, novel, sign, replay=None):
        self.ws[:] = 0
        self.traces = []
        self.cur_epoch += 1
        if replay is not None:
            for ev in replay:
                self.run_body(ev, novel, sign, True)
            return
        hop = 0
        while hop <= self.p.max_hops and self.cur:
            self.nxt = []
            for ev in self.cur:
                if self.epoch[ev[0]] == self.cur_epoch:
                    self.st.dedup_skips += 1
                    continue
                self.epoch[ev[0]] = self.cur_epoch
                self.run_body(ev, novel, sign, False)
            self.cur = self.nxt
            hop += 1

    def readout(self, w):
        V = self.p.vocab
        if V <= C.F:
            logits = w[:V].astype(np.float32) * np.float32(1.0 / 32.0)
        else:
            logits = (self.wout @ (w.astype(np.float32) * np.float32(1.0 / 128.0))).astype(np.float32)
        mx = logits.max()
        pr = np.exp(logits - mx)
        pr = pr / pr.sum()
        return logits, pr.astype(np.float32)

    # ------------------------------------------------------------------ event
    def process_event(self, sym, target):
        p, st = self.p, self.st
        key = self.make_key()
        self.xin = self.make_input(key, sym)
        leaf, depth = self.tree.walk(key)
        st.addr_decisions += depth
        cand = self.tree.match(leaf, key)
        st.candidate_locks += len(cand)
        if not cand:
            st.no_candidate += 1
            if p.structural:
                u = self.alloc_unit()
                if u >= 0:
                    self.birth_unit_weights(u)
                    lid = self.tree.new_lock(key, C.SENSORY_MASK, u, self.block_idx)
                    self.unit_lock[u] = lid
                    cand = [lid]
                    st.lock_births += 1
        self.cur, self.accepted, self.executed = [], [], []
        novel = False
        for lid in cand:
            lk = self.tree.locks[lid]
            if lk.hits < p.novel_hits or lk.err_ema > self.wrong_ema + p.surprise_thresh:
                novel = True
            lk.hits += 1
            lk.last_hit_block = self.block_idx
            self.cur.append((lk.unit, SRC_SENSORY, 0, -1))
        if novel:
            st.novel_events += 1
        self.cur_novel = novel

        perturb_mode = p.train and p.hidden == "perturb"
        self.run_pass(novel, 1 if perturb_mode else 0)
        lossA = lossB = 0.0
        trA = []
        if perturb_mode:
            wsA = self.ws.copy()
            _, pr = self.readout(wsA)
            lossA = -math.log(max(float(pr[target]), 1e-9))
            trA = self.traces
            sched = [[t.unit, t.source, t.hop, -1] for t in trA]
            for i, t in enumerate(trA):
                if t.source == SRC_ROUTED:
                    for (recv, sender) in self.accepted:
                        if recv == t.unit:
                            for j in range(i):
                                if trA[j].unit == sender:
                                    sched[i][3] = j
                                    break
                            break
            execA = self.executed
            self.executed = []
            self.run_pass(novel, -1, [tuple(s) for s in sched])
            _, pr = self.readout(self.ws)
            lossB = -math.log(max(float(pr[target]), 1e-9))
            self.ws = ((wsA + self.ws) / 2).astype(np.int32)   # C++ integer division truncates toward zero
            self.ws = np.where(wsA + self.ws * 0 < 0, self.ws, self.ws)
            self.executed = execA

        logits, pr = self.readout(self.ws)
        self.probs = pr
        pred = int(np.argmax(pr))   # first max, same as C++ strict '>' scan
        loss = -math.log(max(float(pr[target]), 1e-9))
        st.loss_sum += loss
        st.events += 1
        correct = pred == target
        st.correct += correct
        if not self.loss_ema_init:
            self.loss_ema, self.loss_ema_init = loss, True
        else:
            self.loss_ema += 0.01 * (loss - self.loss_ema)
        consequential = loss <= self.loss_ema

        if self.trace_fp:
            self.trace_fp.write(json.dumps({"t": self.event_idx, "key": key, "cand": len(cand), "novel": int(novel), "exec": [int(u) for u in self.executed],
                                            "reached": [t.reached for t in self.traces], "pred": pred, "target": int(target), "loss": round(loss, 5)}) + "\n")

        if p.train:
            if perturb_mode:
                dq = int(round((lossA - lossB) * p.perturb_gain))
                dq = max(-127, min(127, dq))
                self.learn_hidden(dq, trA)
            self.learn_l6(target)
            self.metabolic(consequential, novel)
            self.assoc_update(novel, consequential)
        wrong = 0 if correct else 65535
        self.wrong_ema += int((wrong - self.wrong_ema) / 256)   # C++ int division truncates toward zero
        for lid in cand:
            lk = self.tree.locks[lid]
            lk.err_ema += int((wrong - lk.err_ema) / 32)
            if p.structural and p.train:
                if lk.evidence < 0 and lk.err_ema > p.conflict_thresh:
                    lk.evidence = self.evid.acquire(lid)
                if lk.evidence >= 0:
                    self.evid.record(lk.evidence, key, target)

        if p.ws_decay:
            self.ws_mem = ((self.ws_mem * p.ws_decay) >> 8) + self.ws.astype(np.int64)
        self.hist = [target] + self.hist[:15]
        self.event_idx += 1

    # ------------------------------------------------------------------ learning
    def apply_acc_block(self, u, nib0, delta):
        """Vectorised equivalent of Organism::apply_acc over a contiguous block of synapses."""
        p, st = self.p, self.st
        n = delta.size
        nz = delta != 0
        if not nz.any():
            return
        st.acc_updates += int(nz.sum())
        W = unpack_unit(self.w[u]).astype(np.int32)
        wl = W[nib0:nib0 + n]
        v = self.acc[u, nib0:nib0 + n].astype(np.int32) + delta
        v = np.where(nz, v, self.acc[u, nib0:nib0 + n].astype(np.int32))
        up = nz & (v >= p.half_quantum)
        canup = up & (wl < 7)
        wl = np.where(canup, wl + 1, wl)
        v = np.where(canup, v - p.quantum, v)
        v = np.where(up & ~canup, p.half_quantum - 1, v)
        dn = nz & (v <= -p.half_quantum) & ~up
        candn = dn & (wl > -8)
        wl = np.where(candn, wl - 1, wl)
        v = np.where(candn, v + p.quantum, v)
        v = np.where(dn & ~candn, -(p.half_quantum - 1), v)
        st.level_changes += int(canup.sum() + candn.sum())
        W[nib0:nib0 + n] = wl
        self.w[u] = pack_unit(W.astype(np.int8))
        self.acc[u, nib0:nib0 + n] = np.clip(v, -127, 127).astype(np.int8)

    def learn_l6(self, target):
        p = self.p
        V = p.vocab
        pr = self.probs
        onehot = np.zeros(V, dtype=np.float32)
        onehot[target] = 1.0
        err = pr - onehot
        if V <= C.F:
            dl6 = np.zeros(C.F, dtype=np.float32)
            dl6[:V] = err
        else:
            S = np.abs(err) > np.float32(p.err_sparse)          # sparse error support, as in the native code
            dl6 = (err[S] @ self.wout[S]).astype(np.float32) if S.any() else np.zeros(C.F, dtype=np.float32)
            a = self.ws.astype(np.float32) * np.float32(1.0 / 128.0)
            self.wout[S] -= (np.float32(p.lr_out) * np.outer(err[S], a)).astype(np.float32)
        g = np.clip(dl6 * np.float32(p.err_gain), -1.0, 1.0)
        q6 = np.array([int(round(float(x) * 127.0)) for x in g], dtype=np.int32)   # lround: half away from zero
        shift = p.lr_shift + (0 if self.cur_novel else p.lr_anneal)
        for tr in self.traces:
            if not (tr.reached & R_L6):
                continue
            x6 = tr.x6.astype(np.int32)
            delta = scale_trunc(-(np.outer(q6, x6) * p.lr_num), shift)      # [F][IN6]
            self.apply_acc_block(tr.unit, C.NIB_L6, delta.reshape(-1))

    def learn_hidden(self, dq, trA):
        if dq == 0:
            return
        p = self.p
        shift = p.lr_shift + p.hidden_shift + (0 if self.cur_novel else p.lr_anneal)

        def upd(unit, layer, nib0, x, n_in, n_out):
            bits = mix64((self.event_idx * 0x9E3779B97F4A7C15) ^ (unit << 20) ^ (layer << 56))
            s = np.array([1 if (bits >> j) & 1 else -1 for j in range(n_out)], dtype=np.int32)
            delta = scale_trunc(-(dq * np.outer(s, x.astype(np.int32)) * p.lr_num), shift)
            self.apply_acc_block(unit, nib0, delta.reshape(-1))
        for tr in trA:
            if tr.reached & R_L1:
                upd(tr.unit, C.LI_1, C.NIB_L1, self.xin, C.IN, C.A)
            if tr.reached & R_L23:
                upd(tr.unit, C.LI_2, C.NIB_L2, tr.x2, C.IN2, C.B)
                upd(tr.unit, C.LI_3, C.NIB_L3, tr.x3, C.IN3, C.C)
            if tr.reached & R_L5:
                upd(tr.unit, C.LI_5, C.NIB_L5, tr.x5, C.IN5, C.E)

    def metabolic(self, consequential, novel):
        p = self.p
        for tr in self.traces:
            r = int(self.resist[tr.unit])
            if tr.reached & R_L6:
                r += -p.resist_down if consequential else p.resist_up
            else:
                r += -p.resist_down if novel else p.resist_up_dead
            self.resist[tr.unit] = max(p.resist_min, min(p.resist_max, r))

    def assoc_update(self, novel, consequential):
        p = self.p
        emitted = {tr.unit for tr in self.traces if tr.reached & R_L6}
        for (v, _sender) in self.accepted:
            if v in emitted and consequential:
                self.assoc_noconseq[v] = 0
                continue
            self.assoc_noconseq[v] += 1
            if self.assoc_noconseq[v] >= p.assoc_tighten_after:
                self.assoc_noconseq[v] = 0
                free_bits = (~self.assoc_mask[v]) & M64
                if not free_bits:
                    continue
                pick = 0
                for _ in range(8):
                    pick = 1 << self.rng.below(64)
                    if free_bits & pick:
                        break
                    pick = 0
                if pick:
                    self.assoc_mask[v] |= pick
                    self.assoc_value[v] ^= pick
        if not novel:
            return
        receivers = {v for (v, _s) in self.accepted}
        for u in self.executed:
            if u in receivers:
                self.assoc_starve[u] = 0
                continue
            self.assoc_starve[u] += 1
            if self.assoc_starve[u] >= p.assoc_loosen_after and self.assoc_mask[u]:
                self.assoc_starve[u] = 0
                b = self.rng.below(64)
                m = self.assoc_mask[u]
                while not (m >> b) & 1:
                    b = (b + 1) & 63
                self.assoc_mask[u] &= ~(1 << b)

    # ------------------------------------------------------------------ blocks / structure
    def run_block(self, s, n):
        if n < 2:
            return
        if self.event_idx == 0:
            self.hist[0] = s[0]
        for i in range(n - 1):
            self.process_event(s[i], s[i + 1])

    def commit_splits(self):
        p = self.p
        thr = p.split_g_per_df * (p.future_buckets - 1)
        for s in range(len(self.evid.lock)):
            lid = self.evid.lock[s]
            if lid is None:
                continue
            lk = self.tree.locks[lid]
            if not lk.alive:
                self.evid.release(s)
                continue
            b, g = self.evid.best_split(s, p.split_min_evidence)
            if b >= 0 and g > thr:
                u = self.alloc_unit()
                if u < 0:
                    lk.evidence = -1
                    self.evid.release(s)
                    continue
                bit = 1 << (b + C.SENSORY_BITS)
                self.w[u] = self.w[lk.unit]
                if self.acc is not None:
                    self.acc[u] = self.acc[lk.unit]
                self.resist[u] = self.resist[lk.unit]
                self.act_ema[u] = self.act_ema[lk.unit]
                lk.mask |= bit
                lk.value &= ~bit
                lk.err_ema = 0
                lk.evidence = -1
                nid = self.tree.new_lock(lk.value | bit, lk.mask, u, self.block_idx)
                self.unit_lock[u] = nid
                self.evid.release(s)
                self.st.lock_splits += 1
            elif lk.err_ema < p.conflict_thresh // 2:
                lk.evidence = -1
                self.evid.release(s)

    def end_block(self):
        p = self.p
        if p.structural and p.train:
            self.commit_splits()
        if p.structural:
            self.st.leaf_splits += self.tree.maintain()
        if p.structural and p.lock_death_blocks:
            for lid, lk in enumerate(self.tree.locks):
                if lk.alive and self.block_idx - lk.last_hit_block > p.lock_death_blocks:
                    if lk.evidence >= 0:
                        self.evid.release(lk.evidence)
                    self.free_unit(lk.unit)
                    self.tree.kill_lock(lid)
        self.block_idx += 1

    def set_mode(self, train, structural):
        self.p.train, self.p.structural = train, structural
