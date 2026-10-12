"""Replay live evidence learning independently of native code and teacher state.

Uses the preceding audit's physical-world specification, but implements the
evidence hypothesis space, branch correlations and question choice here. All
teaching inputs and native logs are read only; the audit writes its own report.
"""
import collections
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

from check_online_grounding import world, ports, menu, task, satisfied, execute, check_models
from check_observation_language import parse_knowledge

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "logs/live_evidence"
FIELDS = ("positive", "negative", "rows", "columns")


def freeze(k):
    return tuple(frozenset(k[f]) for f in FIELDS)


def public_knowledge(rows):
    return (frozenset((a, r, b) for (a, r), b in rows.items()),
            frozenset(), frozenset(rows), frozenset())


def truth(k, a, r, b):
    p, n, rows, columns = k
    return "yes" if (a, r, b) in p else "no" if (a, r, b) in n or (a, r) in rows or (r, b) in columns else "unknown"


def question_menu(names):
    out = []
    for a in names:
        for b in names:
            out.extend([(f"Does {b} have the {a}?", (a, 1, b)),
                        (f"Is {a} at {b}?", (a, 0, b))])
    for a in names:
        out.extend([(f"Who has the {a} now?", (a, 1, None)),
                    (f"Where is {a}?", (a, 0, None))])
    return out


def answer(k, query):
    a, r, b = query
    if b is not None:
        return truth(k, a, r, b)
    values = {o for s, rel, o in k[0] if s == a and rel == r}
    return next(iter(values)) if (a, r) in k[2] and len(values) == 1 else "unknown"


def hypotheses(arity):
    rules = {(0, -1, -1, 0)}
    for r in range(2):
        for a in range(arity):
            rules.update((kind, r, a, 1 << b) for kind in (1, 2, 7, 8, 9, 10) for b in range(arity))
            rules.update((kind, r, a, mask) for kind in (3, 4) for mask in range(1 << arity))
            rules.update((kind, r, a, 0) for kind in (5, 6))
    return rules


def apply(k, rule, args):
    kind, relation, focus, mask = rule
    if kind == 0:
        return k
    a = args[focus]
    values = {b for i, b in enumerate(args) if mask & (1 << i)}
    p, n, rows, columns = map(set, k)
    if kind in (1, 2):
        assert len(values) == 1
        b = next(iter(values))
        if truth(k, a, relation, b) not in ("unknown", "yes" if kind == 1 else "no"):
            return None
        (p if kind == 1 else n).add((a, relation, b))
    elif kind in (3, 4):
        atoms = {(a, relation, b) for b in values} if kind == 3 else {(b, relation, a) for b in values}
        scope = {f for f in p if f[1] == relation and f[0 if kind == 3 else 2] == a}
        if not scope <= atoms or any(truth(k, *f) == "no" for f in atoms):
            return None
        p |= atoms
        (rows if kind == 3 else columns).add((a, relation) if kind == 3 else (relation, a))
    elif kind in (5, 6):
        p, n = ({f for f in atoms if f[1] != relation or f[0 if kind == 5 else 2] != a} for atoms in (p, n))
        if kind == 5:
            rows.discard((a, relation))
            columns = {f for f in columns if f[0] != relation}
        else:
            columns.discard((relation, a))
            rows = {f for f in rows if f[1] != relation}
    else:
        assert kind in (7, 8, 9, 10) and len(values) == 1
        b = next(iter(values))
        atom = (a, relation, b)
        if kind == 8:
            p.discard(atom)
            n.add(atom)
        else:
            if kind in (9, 10):
                p = {f for f in p if f[1] != relation or f[0 if kind == 9 else 2] != (a if kind == 9 else b)}
                n = {f for f in n if f[1] != relation or f[0 if kind == 9 else 2] != (a if kind == 9 else b)}
            n.discard(atom)
            p.add(atom)
            if kind == 9:
                rows.add((a, relation))
            if kind == 10:
                columns.add((relation, b))
    return tuple(map(frozenset, (p, n, rows, columns)))


def parse_rules(text, seed):
    out = set()
    for raw in text.split(";"):
        if raw:
            kind, r, focus, mask = map(int, raw.split(","))
            out.add((kind, r ^ (seed % 2) if r >= 0 else r, focus, mask))
    return out


def records(seed, suffix):
    return list(csv.DictReader((OUT / f"{seed}.{suffix}.tsv").open(), delimiter="\t"))


def request_query(text):
    w = re.findall(r"[a-z0-9]+", text.lower())
    if w[:2] == ["please", "have"]:
        return ((w[2], 0, w[5]), True) if w[3] == "go" else ((w[5], 1, w[7]), True)
    if w[0] == "arrange":
        return (w[2], 1, w[7]), True
    assert w[0] == "ensure"
    return (w[1], 1, w[6]), False


def goal_truth(k, request):
    query, positive = request_query(request)
    value = answer(k, query)
    return -1 if value == "unknown" else int((value == "yes") == positive)


def audit(seed):
    score = collections.Counter()
    people, rooms, objects, owners, places = world("livepartial")
    names = [name for pair in zip(people, rooms) for name in pair] + objects
    options = question_menu(names)
    queries = dict(options)
    public = {(a, 0): b for a, b in places.items()} | {(o, 1): a for o, a in owners.items()}
    knowledge = public_knowledge(public)
    worlds = {(knowledge, ())}  # Full knowledge plus consistent frame choices.
    models, support = {}, collections.defaultdict(set)
    answers = collections.defaultdict(list)
    for row in records(seed, "answers"):
        answers[int(row["event"])].append(row)
    reports, actions = records(seed, "reports"), records(seed, "actions")
    assert [int(r["event"]) for r in reports] == list(range(len(reports)))
    event_states, expected_requests, attempts = {}, {}, collections.Counter()
    action_events = {int(r["report_event"]): r for r in actions}
    assert len(action_events) == len(actions)
    for event, report in enumerate(reports):
        request_id = int(report["request_id"])
        assert freeze(parse_knowledge(report["before"], seed)) == knowledge
        if report["type"] != "action":
            assert request_id == len(expected_requests)
            if request_id:
                assert satisfied(expected_requests[request_id - 1], owners, places)
            if request_id % 2 == 0:
                obj = objects[(request_id // 2) % 3]
                target = people[(request_id // 6 + seed) % 4]
                if target == owners[obj]:
                    target = people[(request_id // 6 + seed + 1) % 4]
                owners[obj] = target
                public.pop((obj, 1), None)
                expected_text = f"The holder of {obj} went unobserved."
                assert report["type"] == "hidden_owner"
            else:
                person = people[(request_id // 2) % 4]
                target = rooms[(request_id // 8 + seed) % 4]
                if target == places[person]:
                    target = rooms[(request_id // 8 + seed + 1) % 4]
                places[person] = target
                public.pop((person, 0), None)
                expected_text = f"The whereabouts of {person} were unobserved."
                assert report["type"] == "hidden_location"
            expected_requests[request_id] = task(request_id % 4, request_id, people, rooms, objects, owners, places)
            assert not satisfied(expected_requests[request_id], owners, places)
        else:
            row = action_events[event]
            request = expected_requests[request_id]
            version = int(request_id >= 80)
            assert int(row["request_id"]) == request_id and row["request"] == request
            assert int(row["attempt"]) == attempts[request_id] < 3
            attempts[request_id] += 1
            assert int(row["before_truth"]) == goal_truth(knowledge, request) != 1
            assert freeze(parse_knowledge(row["before"], seed)) == knowledge
            offered = [tuple(a.split(",")) for a in row["menu"].split(";") if a]
            assert len(set(offered)) == len(offered) and set(offered) == menu(people, rooms, owners, places, seed, version)
            # Verify the first universally sufficient choice in the actual menu.
            first = None
            for index, candidate in enumerate(offered):
                is_transfer = candidate[0] == ports(seed, version)[0]
                args = (candidate[2], candidate[3]) if is_transfer else candidate[1:]
                predicted = apply(knowledge, (9, int(is_transfer), 0, 2), args)
                if goal_truth(predicted, request) == 1:
                    first = index
                    break
            assert first is not None and int(row["index"]) == first
            action = tuple(row["action"].split(","))
            assert offered[first] == action
            executed = score["actions"] % 7 != 0
            assert bool(int(row["executed"])) == executed
            execute(action, executed, owners, places, seed, version)
            if action[0] == ports(seed, version)[0]:
                _, giver, obj, target = action
                public[obj, 1] = owners[obj]
                expected_text = f"The {obj} passed from {giver} into {target}'s keeping." if executed else f"{giver} offers the {obj} to {target}, but {target} does not take it."
            else:
                _, person, target = action
                public[person, 0] = places[person]
                expected_text = f"{person} relocated into {target}." if executed else f"{person} moved to the {places[person]}."
            score["actions"] += 1
            score["refusals"] += not executed
            score["unknown_goal_before"] += int(row["before_truth"]) == -1
            complete = int(satisfied(request, owners, places))
            assert int(row["satisfied"]) == int(row["after_truth"]) == complete
            score["completed"] += complete
        assert report["report"] == expected_text
        expected = public_knowledge(public)
        feedback = answers[event]
        assert int(report["questions"]) == len(feedback)
        if report["staged"] == "1":
            words = re.findall(r"[a-z0-9]+", expected_text.lower())
            args = [w for w in words if w in names]
            slots = iter(range(len(args)))
            frame = " ".join(f"${next(slots)}" if w in names else w for w in words) + " "
            assert frame == report["frame"]
            candidates = models.get(frame, hypotheses(len(args)))
            branches = []
            for k, choices in worlds:
                previous = dict(choices)
                for rule in candidates:
                    if frame in previous and previous[frame] != rule:
                        continue
                    revised = apply(k, rule, args)
                    if revised is not None:
                        choices_next = tuple(sorted((previous | {frame: rule}).items()))
                        branches.append((rule, revised, choices_next))
            assert branches
            assert feedback or len(support[frame]) >= 4
            for id, row in enumerate(feedback):
                assert int(row["answer_id"]) == id
                assert int(row["before_branches"]) == len(branches), (seed, event, id, "before branches")
                assert parse_rules(row["before_rules"], seed) == {r for r, _, _ in branches}
                expected_answer = answer(expected, queries[row["question"]])
                assert row["answer"] == expected_answer
                # Score every offered question, preserving multiplicity of
                # different hypotheses even when their knowledge is identical.
                counts = collections.Counter(k for _, k, _ in branches)
                best = (-1, None, None)
                for index, (q, query) in enumerate(options):
                    groups = collections.Counter()
                    for k, n in counts.items():
                        groups[answer(k, query)] += n
                    gain = len(branches) - max(groups.values())
                    if gain > best[0]:
                        best = gain, index, len(groups)
                assert (int(row["score"]), int(row["choice"]), int(row["partitions"])) == best, (seed, event, id, best)
                assert options[best[1]][0] == row["question"]
                branches = [b for b in branches if answer(b[1], queries[row["question"]]) == expected_answer]
                assert branches and int(row["after_branches"]) == len(branches)
                assert parse_rules(row["after_rules"], seed) == {r for r, _, _ in branches}
                score["verified_answers"] += 1
                score["verified_unknown"] += expected_answer == "unknown"
                score["hypothetical_evaluations"] += len(options) * sum(counts.values())
            models[frame] = {r for r, _, _ in branches}
            if feedback:
                support[frame].add(event)
            assert parse_rules(report["rules"], seed) == models[frame]
            assert int(report["support"]) == len(support[frame])
            worlds = {(k, choices) for _, k, choices in branches}
        else:
            # The only anchored report in this live stream is a failed move's
            # declaration of the actor's observed current location.
            m = re.fullmatch(r"(\w+) moved to the (\w+)\.", expected_text)
            assert m and not feedback and not report["frame"] and not report["rules"]
            worlds = {(apply(k, (9, 0, 0, 2), m.groups()), choices) for k, choices in worlds}
        assert int(report["worlds"]) == len(worlds), (seed, event, "world count")
        # Every retained interpretation agrees on these observed histories;
        # model ambiguity is retained rather than selecting a single meaning.
        assert {k for k, _ in worlds} == {expected}, (seed, event, "knowledge")
        knowledge = expected
        assert freeze(parse_knowledge(report["after"], seed)) == knowledge
        if event in action_events:
            assert freeze(parse_knowledge(action_events[event]["after"], seed)) == knowledge
        event_states[event] = knowledge
    assert set(expected_requests) == set(attempts) == set(range(160))
    assert satisfied(expected_requests[159], owners, places) and score["completed"] == 160
    assert set(answers) <= set(event_states)
    assert sum(len(v) for v in answers.values()) == score["verified_answers"]
    assert len(reports) == 160 + len(actions)
    expected_rules = {
        "the holder of $0 went unobserved ": {(5, 1, 0, 0)},
        "the whereabouts of $0 were unobserved ": {(5, 0, 0, 0)},
        "the $0 passed from $1 into $2 s keeping ": {(9, 1, 0, 4)},
        "$0 relocated into $1 ": {(3, 0, 0, 2), (9, 0, 0, 2)},
        "$0 offers the $1 to $2 but $3 does not take it ": {(3, 1, 1, 1), (9, 1, 1, 1)},
    }
    assert set(models) == set(expected_rules)
    for frame, rules in models.items():
        assert rules and rules <= expected_rules[frame] and len(support[frame]) >= 4
    exported = {}
    for line in (OUT / f"{seed}.models.txt").read_text().splitlines():
        frame, n, rules = line.split(":")
        assert frame not in exported and int(n) == len(support[frame])
        exported[frame] = parse_rules(rules, seed)
    assert exported == models
    retention = records(seed, "retention")
    assert [(int(r["event"]), r["question"]) for r in retention] == [(i, q) for i in event_states for q, _ in options]
    for row in retention:
        gold = answer(event_states[int(row["event"])], queries[row["question"]])
        assert row["reply"] == ("I do not know." if gold == "unknown" else gold)
        assert row["status"] in ({"missing_observation", "uncertain_answer"} if gold == "unknown" else {"answer", "grounded_answer"})
        score["retained_unknown" if gold == "unknown" else "retained_definite"] += 1
    held = records(seed, "holdout")
    assert [(int(r["case"]), int(r["phase"])) for r in held] == [(c, p) for c in range(48) for p in range(4)]
    for c in range(48):
        p, r, o, owners, places = world(f"unseen{c}")
        public = {(a, 0): b for a, b in places.items()} | {(obj, 1): a for obj, a in owners.items()}
        obj, actor, room, receiver = o[c % 3], p[c % 4], r[(c + 1) % 4], p[(c + 2) % 4]
        for phase in range(4):
            if phase == 0: public.pop((obj, 1))
            elif phase == 1: public.pop((actor, 0))
            elif phase == 2: public[obj, 1] = receiver
            else: public[actor, 0] = room
            row = held[c * 4 + phase]
            query = f"Who has the {obj} now?" if phase % 2 == 0 else f"Where is {actor}?"
            gold = "I do not know." if phase < 2 else receiver if phase == 2 else room
            assert row["question"] == query and row["reply"] == gold
            assert row["status"] in ({"missing_observation", "uncertain_answer"} if phase < 2 else {"answer", "grounded_answer"})
            assert freeze(parse_knowledge(row["knowledge"], seed)) == public_knowledge(public)
    check_models(ROOT / f"logs/online_grounding/{seed}.information.models.tsv", seed)
    score.update(reports=len(reports), retained_questions=len(retention), frozen_questions=len(held), learned_frames=len(models))
    print(f"seed={seed}: {dict(score)}", flush=True)
    return dict(score)


def main():
    seeds = list(map(int, sys.argv[1:])) or [0, 1, 2]
    results = {seed: audit(seed) for seed in seeds}
    paths = [Path(__file__), ROOT / "scripts/live_evidence.sh", ROOT / "cpp/apps/live_evidence_probe.cpp",
             ROOT / "cpp/apps/interaction_probe_support.h", ROOT / "cpp/apps/grounding_probe_support.h",
             ROOT / "cpp/src/organism.cpp", ROOT / "cpp/tests/test_live_evidence.cpp",
             ROOT / "build-wsl/live_evidence_probe", ROOT / "build-asan/test_live_evidence"]
    paths += [ROOT / f"cpp/include/laminar/{name}.h" for name in
              ("live_evidence_grounding", "observation_language", "observation_language_learner", "fact_knowledge",
               "reasoning_language", "request_grounding", "online_grounding", "bootstrap_grounding", "incremental_grounding",
               "event_grounding", "semantic_grounding", "procedure_memory", "reasoning_runtime", "organism", "sequence", "agency")]
    paths += [ROOT / f"logs/{name}/train.txt" for name in ("bootstrap_grounding", "incremental_grounding", "dialogue_grounding", "observation_language")]
    paths += [ROOT / f"logs/online_grounding/{s}.information.models.tsv" for s in seeds]
    paths += [ROOT / f"logs/reasoning_composition_verified/micro_{s}.txt" for s in seeds]
    paths += [p for s in seeds for p in OUT.glob(f"{s}.*") if p.suffix in (".tsv", ".txt")]
    paths += [ROOT / "python/check_online_grounding.py", ROOT / "python/check_observation_language.py"]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (OUT / "audit.json").write_text(json.dumps(dict(results=results, sha256=hashes), indent=2) + "\n")


if __name__ == "__main__":
    main()
