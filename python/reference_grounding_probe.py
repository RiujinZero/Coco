"""Reference phrases taught through event outcomes, tested in native dialogue."""
import argparse
import json
import random
from pathlib import Path
from semantic_grounding_probe import world, answer, term, letters
from reasoning_language_probe import FACTS
from event_grounding_probe import PHRASES, update, write_world, observed, channel

REFERENCES = (("it", 1, 1), ("that item", 1, 1), ("that owner", 1, 0), ("there", 0, 1), ("that thing", 1, 1))


def normalize(s):
    import re
    return " ".join(re.findall("[a-z0-9]+", s.lower()))


def sensory(out, seed, rng):
    for alias, (phrase, rel, role) in enumerate(REFERENCES):
        for i in range(12):
            scene = 1000 * alias + i
            names = ["train" + letters(seed+1) + "x" + letters(scene+1) + "e" + letters(j+1) for j in range(8)]
            a,b,c,d,x,token,old,spare = names
            context = [FACTS[rng.randrange(2)][1].format(a=a,b=b), FACTS[rng.randrange(2)][0].format(a=c,b=d)]
            rng.shuffle(context)
            before = {(a,1,b),(c,0,d),(old,1,token),(x,0,spare),(x,1,spare)}
            referent = (a if role == 0 else b) if rel == 1 else d
            if role == 0: text = f"{phrase} picked up {token}."; change = (1,3,referent,token)
            elif rel == 1: text = f"{x} picked up {phrase}."; change = (1,3,x,referent)
            else: text = f"{x} now connects to {phrase}."; change = (0,2,x,referent)
            after = update(before,*change)
            out.write(f"{'adapt' if alias == 4 else 'train'}\t{scene}\t{len(context)}\n")
            for sentence in context: out.write(sentence + "\n")
            out.write(text + "\n"); write_world(out,observed(before,seed)); write_world(out,observed(after,seed))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="logs/reference_grounding_corpus")
    ap.add_argument("--test-n", type=int, default=30)
    args = ap.parse_args(); root = Path(args.out); root.mkdir(parents=True,exist_ok=True)
    groups = ("object_active","object_passive","subject_event","location_event","object_question","subject_composition","location_composition","duplicate_mentions","recency","ambiguous_event","ambiguous_question","new_alias","episode_reset","same_turn_event","same_turn_question")
    for seed in range(3):
        rng = random.Random(830000+seed); records = []; serial = 0
        with (root / f"sensors_{seed}.txt").open("w") as f: sensory(f,seed,rng)
        for group in groups:
            for i in range(args.test_n):
                serial += 1
                a,b,facts = world(rng,seed,800000+serial,rng.choice((4,8,12,32)),rng.randint(4,24),"count_through")
                facts = set(facts); end = answer(term(4,0),a,b,facts)
                items = sorted(o for s,r,o in facts if s==a and r==1); item,other = items[:2]
                token = "token"+letters(seed+1)+"s"+letters(serial); old = "old"+letters(seed+1)+"s"+letters(serial)
                facts.add((old,1,token)); state = set(facts); turns = []
                query = f"What is item count of endpoint of {a}?"
                def add(stage,text,q,expected=None,phrase="",candidates=(),change=None,reset=False,self_text=""):
                    nonlocal state
                    if reset: state=set()
                    if change: state = update(state,*change)
                    if expected is None: expected=str(answer(term(7,term(4,0)),a,b,state))
                    turns.append(dict(stage=stage,text=text,question=normalize(q),gold=expected,phrase=phrase,candidates=sorted(set(candidates)),state=sorted(observed(state,seed)),change=None if change is None else [channel(change[0],seed),*change[1:]],reset=reset,self_text=self_text))
                initial = sorted(facts); rng.shuffle(initial)
                text = " ".join(FACTS[rng.randrange(2)][r].format(a=s,b=o) for s,r,o in initial)
                add("setup",text+" "+query,query)
                # Antecedents use active and passive wording. An unrelated link
                # is always present, defeating an untyped last-entity shortcut.
                focus = [FACTS[rng.randrange(2)][1].format(a=a,b=item), FACTS[rng.randrange(2)][0].format(a=a,b=end)]
                # The link assertion shortens a path but preserves its endpoint.
                state = update(state,0,2,a,end)
                if group in ("ambiguous_event","ambiguous_question"):
                    focus.insert(1,FACTS[rng.randrange(2)][1].format(a=a,b=other))
                if group == "duplicate_mentions": focus.insert(1,FACTS[1][1].format(a=a,b=item))
                rng.shuffle(focus)
                add("focus"," ".join(focus)+" "+query,query)
                phrase = rng.choice(("it","that item"))
                if group == "recency":
                    focus2 = FACTS[rng.randrange(2)][1].format(a=a,b=other)+" "+FACTS[rng.randrange(2)][0].format(a=a,b=end)
                    add("focus",focus2+" "+query,query,self_text=f"\nUSER: {old} holds {token}.\nASSISTANT:")
                    item = other
                if group in ("object_active","object_passive"):
                    event = f"{b} picked up {phrase}." if group=="object_active" else f"{phrase} was taken by {b}."
                    q=f"What is item count of {a}?"; change=(1,3,b,item)
                    expected=str(len({o for s,r,o in update(state,*change) if s==a and r==1}))
                    add("target",event+" "+q,q,expected,change=change)
                elif group in ("same_turn_event","same_turn_question"):
                    # A newer mention within this message must displace the
                    # previous turn's referent, without an assistant boundary.
                    introduction=FACTS[rng.randrange(2)][1].format(a=a,b=other)
                    if group=="same_turn_event":
                        q=f"What is item count of {a}?"; change=(1,3,b,other)
                        expected=str(len({o for s,r,o in update(state,*change) if s==a and r==1}))
                        add("target",introduction+f" {b} picked up {phrase}. "+q,q,expected,change=change)
                    else:
                        add("target",introduction+f" What is {phrase}?",f"What is {other}?",other)
                elif group == "subject_event":
                    q=f"What is item count of {a}?"; change=(1,3,a,token)
                    expected=str(len({o for s,r,o in update(state,*change) if s==a and r==1}))
                    add("target",f"that owner picked up {token}. "+q,q,expected,change=change)
                elif group == "location_event":
                    q=f"What is item count of endpoint of {b}?"; change=(0,2,b,end)
                    add("target",f"{b} now connects to there. "+q,q,str(answer(term(7,0),end,b,state)),change=change)
                elif group == "subject_composition":
                    add("target","What is item count of endpoint of that owner?",query)
                elif group == "location_composition":
                    add("target","What is item count of there?",f"What is item count of {end}?",str(answer(term(7,0),end,b,state)))
                elif group in ("ambiguous_event","ambiguous_question"):
                    q=f"What is item count of {a}?" if group=="ambiguous_event" else f"What is {phrase}?"
                    text=(f"{b} picked up {phrase}. " if group=="ambiguous_event" else "")+q
                    add("pending",text,"_","_",phrase,(item,other))
                    add("invalid",old+".","_","_",phrase,(item,other))
                    chosen=rng.choice((item,other))
                    change=(1,3,b,chosen) if group=="ambiguous_event" else None
                    expected=str(len({o for s,r,o in update(state,*change) if s==a and r==1})) if change else chosen
                    namedq=q if change else f"What is {chosen}?"
                    add("resolved",chosen+".",namedq,expected,change=change)
                elif group == "episode_reset":
                    # Only a link is introduced in a new episode: no prior owner
                    # or item may survive as an antecedent.
                    state=set(); state.add((a,0,end))
                    add("reset",FACTS[0][0].format(a=a,b=end)+f" What is {a}?",f"What is {a}?",a)
                    turns[-1]["reset"]=True
                    add("pending","What is it?","_","_","it",())
                else:
                    if group == "new_alias": phrase="that thing"
                    add("target",f"What is {phrase}?",f"What is {item}?",item)
                records.append(dict(group=group,turns=turns))
        (root/f"test_{seed}.json").write_text(json.dumps(records,indent=2)+"\n")
        with (root/f"test_{seed}.txt").open("w") as out:
            for record in records:
                out.write(f"{record['group']}\t{len(record['turns'])}\n")
                for t in record["turns"]:
                    out.write(f"{t['stage']}\t{int(t['reset'])}\t{t['gold']}\n{t['text']}\n{t['question']}\n{t['phrase']}\n{' '.join(t['candidates'])}\n")
                    out.write(t['self_text'].replace("\n","|")+"\n"); write_world(out,t["state"])
    print(f"Wrote reference curriculum and dialogue cases to {root}")


if __name__=="__main__": main()
