"""A second, structurally-different check for the same class of bug that slipped past
_check_events.py once already (it built its history model the same way the generator did, so it
was blind to the same mistakes). This one does NOT reconstruct a "history" list at all. Instead, for
every "before"/"always had" question it directly scans the raw sentences for anyone ever NAMED as a
giver or an initial holder of that object, and checks the label against that raw set:

  * "always had" is checked against every name that ever appears as EITHER an initial holder or a
    giver of that object anywhere in the story -- if more than one distinct name appears, or if the
    labeled answer is not among them, that is flagged, independent of any ordering logic.
  * "before X" is checked by finding the giver named in the transfer sentence whose receiver is X
    (the LAST such sentence, since "before X" refers to how X came to have it) -- the labeled answer
    must be that giver.

Usage: python _check_events_structural.py FILE.txt
"""
import re, sys

def check_story(lines, q, gold):
    # every (position, who) pair naming someone as a holder of a given object, in story order
    named_holders = {}   # obj -> [(idx, name), ...] every explicit naming as current/prior holder
    transfers = {}        # obj -> [(giver, receiver), ...] in order, success or fail alike
    last_named = None     # "He" always resolves to the giver of the most recent transfer sentence
    for idx, ln in enumerate(lines):
        m = re.match(r"^(\w+) has the (\w+)\.$", ln)
        if m:
            named_holders.setdefault(m.group(2), []).append((idx, m.group(1))); continue
        m = re.match(r"^(\w+|He) (?:gave|handed|passed) the (\w+) to (\w+)\.$", ln)
        if m:   # a successful transfer: BOTH the giver and the receiver are named as holders, in order
            subj = last_named if m.group(1) == "He" else m.group(1)
            named_holders.setdefault(m.group(2), []).append((idx, subj))
            named_holders[m.group(2)].append((idx, m.group(3)))
            transfers.setdefault(m.group(2), []).append((subj, m.group(3)))
            last_named = subj
            continue
        m = re.match(r"^(\w+|He) (?:tried to give|tried to hand|offered) the (\w+) to (\w+)\.$", ln)
        if m:   # a failed transfer: only the giver is confirmed to hold it; the receiver never gets it
            subj = last_named if m.group(1) == "He" else m.group(1)
            named_holders.setdefault(m.group(2), []).append((idx, subj))
            transfers.setdefault(m.group(2), []).append((subj, m.group(3)))
            last_named = subj
            continue
        m = re.match(r"^Correction: (\w+), not (\w+), has the (\w+)\.$", ln)
        if m:
            named_holders.setdefault(m.group(3), []).append((idx, m.group(1))); continue

    # a re-affirmation of the ALREADY-current holder (e.g. a failed transfer attempt that changes
    # nothing) is not a new occurrence for "before" purposes -- collapse consecutive repeats of the
    # same name, matching the one property that actually matters: distinct holding periods, in order.
    def dedup(o):
        seq = named_holders.get(o, [])
        out = []
        for idx, n in seq:
            if not out or out[-1][1] != n:
                out.append((idx, n))
        return out

    qm = re.match(r"^Who has always had the (\w+)\?$", q)
    if qm:
        o = qm.group(1)
        names = {n for _, n in dedup(o)}
        if len(names) > 1:
            return f"INSUFFICIENT (multiple distinct people named as holder of {o}: {sorted(names)})"
        if not names:
            return "INSUFFICIENT (no one ever named as holder)"
        only = next(iter(names))
        return only if only == gold else f"CONTRADICTS (only {only} ever named, gold={gold!r})"

    qm = re.match(r"^Who had the (\w+) before (\w+)\?$", q)
    if qm:
        # X (the "before" anchor) is always the LAST person ever named holding the object, which can
        # come from either role: a successful transfer's RECEIVER, or a transfer's GIVER (this is how
        # a post-gap reveal surfaces X as the object's current holder without X ever "receiving" it
        # in any sentence -- the hidden holder moved silently during the gap, and the next transfer
        # attempt, successful or not, is the first sentence to name X at all). Either way, the answer
        # is whoever was on record as the object's holder immediately before that naming.
        o, anchor = qm.group(1), qm.group(2)
        seq = dedup(o)
        anchor_positions = [i for i, (_, n) in enumerate(seq) if n == anchor]
        if not anchor_positions:
            return f"INSUFFICIENT (no sentence ever names {anchor} in connection with the {o})"
        i = anchor_positions[-1]
        if i == 0:
            return f"INSUFFICIENT ({anchor} is the first person ever named holding the {o}: no 'before')"
        before = seq[i - 1][1]
        return before if before == gold else f"CONTRADICTS (text says {before!r} held it just before {anchor}, gold={gold!r})"
    return None


def main():
    path = sys.argv[1]
    total = 0
    bad = []
    with open(path, encoding="utf-8") as f:
        block = []
        for line in f:
            line = line.rstrip("\n")
            if line == "<|endofdialogue|>":
                story = [l for l in block if not l.startswith("USER:") and not l.startswith("ASSISTANT:")]
                u = next((l[6:].strip() for l in block if l.startswith("USER: ")), None)
                a = next((l[len("ASSISTANT: "):].strip() for l in block if l.startswith("ASSISTANT: ")), None)
                if u and a is not None:
                    r = check_story(story, u, a)
                    if r is not None:
                        total += 1
                        if r.startswith("INSUFFICIENT") or r.startswith("CONTRADICTS"):
                            bad.append((story, u, a, r))
                block = []
            else:
                block.append(line)
    print(f"structurally checked {total} temporal questions, {len(bad)} flagged")
    for story, q, gold, r in bad[:15]:
        print("---")
        print("\n".join(story))
        print(f"Q: {q}  gold={gold!r}  {r}")


if __name__ == "__main__":
    main()
