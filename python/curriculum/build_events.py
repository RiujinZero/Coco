"""Stage 5 (manifest S1-style): events, persistence, correction and uncertainty.

Rebuild 2026-09-22 (see docs/CURRICULUM_REBUILD_ISSUES.md #1 and #5): the previous version labelled
"unknown" by picking a person who had never held the object, in a fully-narrated, fully-observed
world -- in that world the correct answer for any non-holder is always "false" (the whole cast and
the current holder are both known), never "unknown". "Unknown" now comes from genuine, tracked
partial observability instead of an unearned heuristic:

  * an object's INITIAL holder may go unstated ("unseen initial holder" -- known[o] starts False);
  * an explicit OBSERVATION GAP sentence ("No one reported anything about the X for a while.") sets
    known[o] = False, and with some probability the hidden holder silently changes during the gap
    (an unobserved transfer): the learner-facing target reflects what the observation history can
    establish, never the hidden ground truth, which is tracked separately for audit only;
  * any narrated transfer of an object is itself an observation, so it always sets known[o] = True
    again afterward ("regained observation"), whether the transfer succeeded or failed.

When known[o] is True the world is closed (every person is named, the current holder is one of
them), so "does X have O" is answered exactly: true for the holder, false for everyone else --
never "unknown". When known[o] is False, every such question is "unknown", including for the
person who happens to still be the hidden holder, because the observation history alone cannot
establish that.

Corrections (issue #5's third bullet) are now genuine belief revisions, not confirmations: a
misreport is voiced as an explicitly attributed, hedged claim ("Someone said X has the Y."), which
does not change known/holder, and a later correction sentence retracts THAT SPECIFIC claim by name
("Correction: <true>, not <wrong>, has the Y.") -- there is always an identifiable prior report to
supersede.

Also fixes issue #5's first bullet (a distinct, matched single-transition bootstrap subset for the
"immediate" budget, instead of reusing the same 3-10-event generator) and adds paired
visibility-loss / regained-observation held-out cases for the acceptance criteria: same hidden
world, different observation history, different (known vs unknown) correct answer.

    python python/curriculum/build_events.py OUT_DIR [--n_episodes 60000] [--n_immediate 20000] [--seed 1]
"""
import argparse, json, os, random

PEOPLE = ["Mary", "John", "Sandra", "Daniel", "Fred", "Bill", "Julie", "Alice", "Kofi", "Priya",
          "Tomas", "Greta", "Omar", "Nia", "Ruth", "Ivan", "Wen", "Beth", "Sam", "Hana"]
OBJECTS = ["key", "ball", "book", "apple", "football", "map", "lantern", "cup", "hat", "phone",
           "ring", "coin", "letter", "box", "flag", "tool", "bottle", "bag", "watch", "chalk"]

UNSEEN_INITIAL_P = 0.15    # per object: initial holder goes unstated
GAP_P_PER_EVENT_SLOT = 0.10
GAP_SILENT_TRANSFER_P = 0.55   # while unobserved, the hidden holder actually changes
MISREPORT_P_PER_EVENT_SLOT = 0.08


def transfer_verb(success):
    return ("gave the {o} to", "handed the {o} to", "passed the {o} to") if success else \
           ("tried to give the {o} to", "tried to hand the {o} to", "offered the {o} to")


class World:
    def __init__(self, rng, n_people, n_objects):
        self.rng = rng
        self.people = rng.sample(PEOPLE, n_people)
        self.objects = rng.sample(OBJECTS, n_objects)
        self.hidden_holder = {o: rng.choice(self.people) for o in self.objects}   # ground truth, oracle-only
        self.holder = dict(self.hidden_holder)                                    # the observer's current belief
        self.known = {o: True for o in self.objects}                              # can the observer currently vouch for holder[o]?
        self.history = {o: [] for o in self.objects}                              # OBSERVED holders only, in order -- filled by start_fact/event_transfer, never seeded from hidden truth
        self.lines = []
        self.last_pronoun_subject = None
        self.pending_wrong_report = {}    # obj -> the person a misreport claimed, awaiting correction
        self.changed_known = set()        # objects whose known holder changed at least once (for changed/unchanged Qs)
        self.corrected = []                # [(obj, wrong_person)] every claim that has been explicitly retracted

    def start_fact(self):
        for o in self.objects:
            if self.rng.random() < UNSEEN_INITIAL_P:
                self.known[o] = False   # unseen initial holder: hidden_holder is set, but never narrated, and NOT recorded in history
            else:
                self.lines.append(f"{self.hidden_holder[o]} has the {o}.")
                self.history[o].append(self.hidden_holder[o])   # only a narrated fact enters the observed history

    def event_transfer(self, use_pronoun=False):
        o = self.rng.choice(self.objects)
        giver = self.hidden_holder[o]
        others = [p for p in self.people if p != giver]
        receiver = self.rng.choice(others)
        success = self.rng.random() < 0.75
        v = self.rng.choice(transfer_verb(success)).format(o=o)
        subj = "He" if use_pronoun and giver == self.last_pronoun_subject else giver
        self.lines.append(f"{subj} {v} {receiver}.")
        # This sentence names GIVER as the object's holder at this point -- whether this is the
        # object's first-ever mention (an unseen initial holder) or a holder the reader lost track of
        # in an earlier observation gap (hidden_holder may have silently moved since the last recorded
        # entry). Record it unless it is already the last thing on record, so "before"/"always had"
        # questions can never contradict a giver the story explicitly names.
        if not self.history[o] or self.history[o][-1] != giver:
            self.history[o].append(giver)
        if success:
            self.hidden_holder[o] = receiver
            self.holder[o] = receiver
            self.history[o].append(receiver)   # a real change: a new entry, distinct from giver by construction
        else:
            self.holder[o] = giver   # unchanged -- and no further history entry: nothing actually changed
        # a narrated transfer -- success or failure -- is itself an observation: knowledge is
        # (re)established regardless of whether it was previously unknown.
        if not self.known[o]:
            self.known[o] = True
        if len(self.history[o]) > 1 and self.history[o][-1] != self.history[o][-2]:
            self.changed_known.add(o)
        self.last_pronoun_subject = giver
        return o, success

    def event_unrelated(self):
        p = self.rng.choice(self.people)
        mood = self.rng.choice(["is tired", "is happy", "went to the market", "is reading", "is waiting outside"])
        self.lines.append(f"{p} {mood}.")

    def event_gap(self, force_obj=None):
        if force_obj is not None:
            o = force_obj
        else:
            known_objs = [o for o in self.objects if self.known[o]]
            o = self.rng.choice(known_objs or self.objects)
        self.lines.append(f"No one reported anything about the {o} for a while.")
        self.known[o] = False
        if self.rng.random() < GAP_SILENT_TRANSFER_P:
            others = [p for p in self.people if p != self.hidden_holder[o]]
            self.hidden_holder[o] = self.rng.choice(others)   # an unobserved transfer: hidden truth moves, belief does not
        return o

    def event_misreport(self):
        o = self.rng.choice(self.objects)
        actual = self.hidden_holder[o] if self.known[o] else None
        wrong_pool = [p for p in self.people if p != actual] if actual else self.people
        wrong = self.rng.choice(wrong_pool)
        self.lines.append(f"Someone said {wrong} has the {o}.")   # a hedged, attributed claim: does not change known/holder
        self.pending_wrong_report[o] = wrong
        return o, wrong

    def event_correction(self):
        # a genuine correction: retract a SPECIFIC earlier misreport by naming it, per issue #5.
        # A real event between the misreport and the correction can make the misreported person the
        # ACTUAL current holder by coincidence (e.g. a later transfer really does give it to them);
        # correcting "X, not X" would be self-contradictory, so such candidates are skipped, not
        # merely reworded -- the pending report is simply left unresolved by this call.
        candidates = [o for o in self.pending_wrong_report if self.known[o] and self.pending_wrong_report[o] != self.hidden_holder[o]]
        if not candidates:
            return None
        o = self.rng.choice(candidates)
        wrong = self.pending_wrong_report.pop(o)
        true = self.hidden_holder[o]
        self.lines.append(f"Correction: {true}, not {wrong}, has the {o}.")
        self.corrected.append((o, wrong))
        return o, wrong, true


def make_episode(rng, kind, n_events_range=(3, 10)):
    w = World(rng, rng.randint(3, 6), rng.randint(2, 5))
    w.start_fact()
    n_events = rng.randint(*n_events_range)
    for _ in range(n_events):
        r = rng.random()
        if r < 0.45:
            w.event_transfer(use_pronoun=rng.random() < 0.3)
        elif r < 0.60:
            w.event_unrelated()
        elif r < 0.60 + GAP_P_PER_EVENT_SLOT:
            w.event_gap()
        elif r < 0.60 + GAP_P_PER_EVENT_SLOT + MISREPORT_P_PER_EVENT_SLOT:
            w.event_misreport()
        else:
            w.event_correction()
    # give every still-pending misreport a chance to be corrected before the episode ends, so a
    # correction is never left dangling without its antecedent report having been resolved either way
    for _ in range(2):
        if w.pending_wrong_report and rng.random() < 0.6:
            w.event_correction()

    story = "\n".join(w.lines)
    o = None
    if kind == "retrieval":
        pool = [x for x in w.objects if w.known[x]] or w.objects
        o = rng.choice(pool)
        q = f"Who has the {o} now?"
        a = w.holder[o] if w.known[o] else "unknown"
    elif kind == "changed_unchanged":
        changed = [x for x in w.objects if x in w.changed_known and w.known[x]]
        unchanged = [x for x in w.objects if x not in w.changed_known and w.known[x]]
        pool = changed if changed and rng.random() < 0.5 else (unchanged or changed or w.objects)
        o = rng.choice(pool)
        q = f"Who has the {o} now?"
        a = w.holder[o] if w.known[o] else "unknown"
    elif kind == "tf":
        o = rng.choice(w.objects)
        if w.known[o]:
            roll = rng.random()
            if roll < 0.5:
                p, a = w.holder[o], "true"
            else:
                p, a = rng.choice([x for x in w.people if x != w.holder[o]]), "false"
        else:
            p, a = rng.choice(w.people), "unknown"
        q = f"Does {p} have the {o}?"
    elif kind == "temporal":
        # both branches must be backed by ACTUALLY NARRATED history entries (history[o] is only ever
        # filled by a narrated initial fact or an observed transfer -- never seeded from hidden truth,
        # per the fix below). "Before" needs >=2 real entries, which is always safe regardless of the
        # object's CURRENT known status (the two facts were each stated at the time). "Always had"
        # additionally requires known[o] now: a single narrated fact followed by an observation gap
        # no longer supports "always", since an unobserved transfer may have happened since.
        multi = [x for x in w.objects if len(w.history[x]) > 1]
        single_known = [x for x in w.objects if len(w.history[x]) == 1 and w.known[x]]
        if multi:
            o = rng.choice(multi)
            q, a = f"Who had the {o} before {w.history[o][-1]}?", w.history[o][-2]
        elif single_known:
            o = rng.choice(single_known)
            q, a = f"Who has always had the {o}?", w.history[o][0]
        else:
            # nothing in this episode has enough recorded history to support a temporal claim at
            # all: fall back to a retrieval-shaped question, which is always answerable (known or not).
            o = rng.choice(w.objects)
            q = f"Who has the {o} now?"
            a = w.holder[o] if w.known[o] else "unknown"
    elif kind == "retraction":
        # was a specific (now-corrected) claim true? tests belief revision, not mere negation.
        if not w.corrected:
            o = rng.choice(w.objects)
            p = rng.choice(w.people)
            q = f"Does {p} have the {o}?"
            a = "unknown" if not w.known[o] else ("true" if w.holder[o] == p else "false")
        else:
            # the answer is evaluated against the story's END state, not frozen at correction time:
            # a later, real transfer could legitimately move the object back to the once-misreported
            # person, or a later gap could make it unknown again -- both must be scored truthfully.
            o, wrong = rng.choice(w.corrected)
            q = f"Does {wrong} have the {o}?"
            a = "unknown" if not w.known[o] else ("true" if w.holder[o] == wrong else "false")
    else:
        raise ValueError(kind)
    meta = {"changed": (o in w.changed_known) if kind in ("retrieval", "changed_unchanged") else None,
            "known": w.known.get(o), "n_events": n_events, "kind": kind}
    return story, q, a, meta


def make_bootstrap_episode(rng):
    """A distinct single-transition subset (issue #5): exactly one transfer (success or failure),
    matched, plus persistence counterexamples, for the smallest possible learning signal."""
    w = World(rng, rng.randint(3, 4), rng.randint(1, 2))
    w.start_fact()
    o, success = w.event_transfer()
    if rng.random() < 0.6:
        w.event_unrelated()
    unaffected = [x for x in w.objects if x != o and w.known[x]]   # only ask persistence about objects actually observed
    story = "\n".join(w.lines)
    if unaffected and rng.random() < 0.5:
        other = rng.choice(unaffected)
        q, a = f"Who has the {other} now?", w.holder[other]
    else:
        q = f"Who has the {o} now?"
        a = w.holder[o] if w.known[o] else "unknown"
    meta = {"kind": "bootstrap_transfer", "success": success, "n_events": 1}
    return story, q, a, meta


def make_visibility_pair(rng):
    """A paired test case (acceptance criteria): the SAME hidden world, rendered twice with
    different observation histories -- once without a gap (known, answerable) and once with a gap
    over the same object (unknown) -- so the same underlying fact yields different correct answers
    depending only on what was actually observed."""
    seed_state = rng.getstate()
    w1 = World(rng, 4, 3)
    w1.start_fact()
    o, _ = w1.event_transfer()
    q = f"Who has the {o} now?"
    a_known = w1.holder[o]

    rng.setstate(seed_state)
    w2 = World(rng, 4, 3)
    w2.start_fact()
    o2, _ = w2.event_transfer()
    assert o2 == o
    w2.event_gap(force_obj=o)   # same hidden world, but this object's holder is now explicitly unobserved
    a_unknown = "unknown"
    return (w1.lines, q, a_known), (w2.lines, q, a_unknown), o


KINDS = ["retrieval", "changed_unchanged", "tf", "temporal", "retraction"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--n_episodes", type=int, default=60000)
    ap.add_argument("--n_immediate", type=int, default=20000)
    ap.add_argument("--n_test", type=int, default=1200)
    ap.add_argument("--n_pairs", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    oracle_dir = os.path.join(a.out_dir, "..", "_oracle")
    os.makedirs(oracle_dir, exist_ok=True)
    rng = random.Random(a.seed)

    train_path = os.path.join(a.out_dir, "stage5_events_train.txt")
    test_path = os.path.join(a.out_dir, "stage5_events_test.jsonl")
    pairs_path = os.path.join(a.out_dir, "stage5_events_visibility_pairs_test.jsonl")
    oracle_path = os.path.join(oracle_dir, "events_oracle.jsonl")
    n_train = 0
    test_rows = {k: [] for k in KINDS}
    n_boot_train = n_boot_test = 0

    with open(train_path, "w", encoding="utf-8", newline="\n") as tf_, \
         open(oracle_path, "w", encoding="utf-8", newline="\n") as of:
        for i in range(a.n_episodes):
            kind = KINDS[i % len(KINDS)]
            story, q, ans, meta = make_episode(rng, kind)
            want_test = len(test_rows[kind]) < a.n_test and rng.random() < 0.02
            if want_test:
                test_rows[kind].append([q, ans, "", story.split("\n")])
            else:
                tf_.write(story + "\nUSER: " + q + "\n\nASSISTANT: " + ans + "\n<|endofdialogue|>\n")
                n_train += 1
            meta["id"] = f"events_{i}"
            of.write(json.dumps(meta) + "\n")

        for i in range(a.n_immediate):
            story, q, ans, meta = make_bootstrap_episode(rng)
            want_test = n_boot_test < max(200, a.n_test // 4) and rng.random() < 0.02
            if want_test:
                test_rows.setdefault("bootstrap_transfer", []).append([q, ans, "", story.split("\n")])
                n_boot_test += 1
            else:
                tf_.write(story + "\nUSER: " + q + "\n\nASSISTANT: " + ans + "\n<|endofdialogue|>\n")
                n_train += 1
                n_boot_train += 1
            meta["id"] = f"events_boot_{i}"
            of.write(json.dumps(meta) + "\n")

    with open(test_path, "w", encoding="utf-8", newline="\n") as ef:
        for kind, rows in test_rows.items():
            for q, a2, sup, story in rows:
                ef.write(json.dumps({"task": f"events_{kind}", "qa": [[q, a2, sup, story]]}) + "\n")

    with open(pairs_path, "w", encoding="utf-8", newline="\n") as pf:
        for i in range(a.n_pairs):
            (lines1, q1, a1), (lines2, q2, a2), o = make_visibility_pair(rng)
            pf.write(json.dumps({"task": "events_visibility_pair_known", "qa": [[q1, a1, "", lines1]],
                                  "pair_id": f"vispair_{i}", "object": o}) + "\n")
            pf.write(json.dumps({"task": "events_visibility_pair_unknown", "qa": [[q2, a2, "", lines2]],
                                  "pair_id": f"vispair_{i}", "object": o}) + "\n")

    print(f"events: train {n_train} (bootstrap {n_boot_train}), test {sum(len(v) for v in test_rows.values())} "
          f"({ {k: len(v) for k, v in test_rows.items()} }), visibility pairs {a.n_pairs}")


if __name__ == "__main__":
    main()
