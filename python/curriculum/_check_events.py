"""Independent re-derivation of answers from story TEXT ALONE (not from the generator's internal
state), mirroring the kind of audit in docs/CURRICULUM_REBUILD_ISSUES.md #1 and its temporal-question
follow-up. Parses: "X has the Y.", "X (gave|handed|passed) the Y to Z.", "X (tried to give|tried to
hand|offered) the Y to Z." + "Z did not take it.", "No one reported anything about the Y for a
while.", "Correction: X, not Y, has the Z.", "Someone said X has the Y."

Also tracks each object's ORDERED history of textually-stated holders (append only on a narrated
initial fact or a narrated transfer -- never on a gap, never seeded from anywhere else) to check
"before"/"always had" temporal questions. If a temporal question is asked about an object with too
little recorded history to support it, that is reported as INSUFFICIENT (a distinct verdict from a
wrong value), matching the "not enough information to support an owner" class of problem.

Usage: python _check_events.py FILE.txt
"""
import re, sys

def check_story(lines, q, gold):
    holder = {}
    known = set()
    history = {}   # obj -> [stated holders in order], mirrors the generator's own history semantics
    last_named = None   # the most recently named human subject, for "He" pronoun resolution
    for ln in lines:
        m = re.match(r"^(\w+) has the (\w+)\.$", ln)
        if m:
            holder[m.group(2)] = m.group(1); known.add(m.group(2))
            history.setdefault(m.group(2), []).append(m.group(1)); continue
        m = re.match(r"^(\w+|He) (?:gave|handed|passed) the (\w+) to (\w+)\.$", ln)
        if m:
            subj = last_named if m.group(1) == "He" else m.group(1)   # the giver
            h = history.setdefault(m.group(2), [])
            if not h or h[-1] != subj:
                h.append(subj)   # the sentence names the giver as holder at this point: record it if new
            holder[m.group(2)] = m.group(3); known.add(m.group(2))
            h.append(m.group(3))   # a real change: a new, distinct entry
            last_named = subj; continue
        m = re.match(r"^(\w+|He) (?:tried to give|tried to hand|offered) the (\w+) to (\w+)\.$", ln)
        if m:
            # a failed transfer still reveals who currently holds the object (you can only try to
            # give away what you have) -- but nothing actually changes, so unlike a real transfer it
            # adds no new distinct history entry beyond (maybe) revealing the giver for the first time.
            subj = last_named if m.group(1) == "He" else m.group(1)
            h = history.setdefault(m.group(2), [])
            if not h or h[-1] != subj:
                h.append(subj)
            holder[m.group(2)] = subj; known.add(m.group(2))
            last_named = subj; continue
        m = re.match(r"^No one reported anything about the (\w+) for a while\.$", ln)
        if m:
            known.discard(m.group(1)); continue   # a gap does NOT add a history entry
        m = re.match(r"^Correction: (\w+), not (\w+), has the (\w+)\.$", ln)
        if m:
            holder[m.group(3)] = m.group(1); known.add(m.group(3)); continue
        # "Someone said ... has the ..." and plain unrelated-person sentences: no state change
    qm = re.match(r"^Does (\w+) have the (\w+)\?$", q)
    if qm:
        p, o = qm.group(1), qm.group(2)
        if o not in known:
            return "unknown"
        return "true" if holder.get(o) == p else "false"
    qm = re.match(r"^Who has the (\w+) now\?$", q)
    if qm:
        o = qm.group(1)
        if o not in known:
            return "unknown"
        return holder.get(o)
    qm = re.match(r"^Who had the (\w+) before (\w+)\?$", q)
    if qm:
        o, before_whom = qm.group(1), qm.group(2)
        h = history.get(o, [])
        if len(h) < 2 or h[-1] != before_whom:
            return "INSUFFICIENT"
        return h[-2]
    qm = re.match(r"^Who has always had the (\w+)\?$", q)
    if qm:
        o = qm.group(1)
        h = history.get(o, [])
        if len(h) != 1 or o not in known:
            return "INSUFFICIENT"
        return h[0]
    return None  # other question shapes: not checked here

def main():
    path = sys.argv[1]
    total = 0
    contradictions = []
    with open(path, encoding="utf-8") as f:
        block = []
        for line in f:
            line = line.rstrip("\n")
            if line == "<|endofdialogue|>":
                story = [l for l in block if not l.startswith("USER:") and not l.startswith("ASSISTANT:")]
                u = next((l[6:].strip() for l in block if l.startswith("USER: ")), None)
                a = next((l[len("ASSISTANT: "):].strip() for l in block if l.startswith("ASSISTANT: ")), None)
                if u and a is not None:
                    derived = check_story(story, u, a)
                    if derived is not None:
                        total += 1
                        if derived != a:
                            contradictions.append((story, u, a, derived))
                block = []
            else:
                block.append(line)
    print(f"checked {total} questions (tf/retrieval/temporal), {len(contradictions)} contradictions (includes INSUFFICIENT-support cases)")
    for story, q, gold, derived in contradictions[:8]:
        print("---")
        print("\n".join(story))
        print(f"Q: {q}  gold={gold!r} derived={derived!r}")

if __name__ == "__main__":
    main()
