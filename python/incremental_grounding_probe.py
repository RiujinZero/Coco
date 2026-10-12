"""Mixed-question grounding probe plus a declared subset of curriculum v2.
All teacher states/gold stay in gold.json. Native training receives text and
final answers only; native evaluation receives text only. Dataset is read-only.
"""
import collections
import json
from pathlib import Path
import random
import re
from bootstrap_grounding_probe import write_episodes
from validate_curriculum_grounding import episodes

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"logs/incremental_grounding"
V2=Path("/home/user/curriculum_v2")
HAS=re.compile(r"(\w+) has the (\w+)\.")
GIVE=re.compile(r"(\w+) (?:gave|handed|passed) the (\w+) to (\w+)\.")


def visible_possession(story):
    owners={}
    for s in story:
        m=HAS.fullmatch(s)
        if m: owners[m[2].lower()]=m[1].lower();continue
        m=GIVE.fullmatch(s)
        if not m or m[1]=="He": return None
        giver,obj,receiver=(x.lower() for x in m.groups())
        # Exclude histories whose prior holder was not visible. This subset is
        # explicitly not an evaluation of partial observation or failed intent.
        if owners.get(obj)!=giver:return None
        owners[obj]=receiver
    return owners


def main():
    OUT.mkdir(parents=True,exist_ok=True);train=[];test=[];rng=random.Random(712)
    def record(story,q,gold,loc,own,group,training=True):
        row=dict(story=list(story),question=q,answer=gold,locations=dict(loc),owners=dict(own),group=group)
        (train if training else test).append(row)
    def make(i,phase,training):
        prefix="teach" if training else "fresh"
        people=[f"{prefix}person{j}" for j in range(4)] if training else [f"{prefix}{i}person{j}" for j in range(4)]
        rooms=[f"{prefix}room{j}" for j in range(3)] if training else [f"{prefix}{i}room{j}" for j in range(3)]
        things=[f"{prefix}item{j}" for j in range(4)] if training else [f"{prefix}{i}item{j}" for j in range(4)]
        rng.shuffle(people);rng.shuffle(rooms);rng.shuffle(things)
        a,b,c,d=people;x,y,z=rooms;loc={a:x,b:y,c:x if i%2 else z,d:z};own={}
        story=[f"{p} moved to the {r}." for p,r in loc.items()]
        def has(p,o):story.append(f"{p} has the {o}.");own[o]=p
        has(a,things[0]);has(b,things[1]);has(x,things[2]);has(x,things[3]);has(c,things[0])
        if phase:
            transfers=[(things[0],b if i%3==0 else a),
                       (things[1],b if i%3==1 else a),
                       (things[2],x if i%4==2 else y)]
            if i%4==3:transfers.append((things[3],b))
            for o,p in transfers:
                story.append(f"{own[o]} {phase} the {o} to {p}.");own[o]=p
        group=phase or "declaration"
        if training:
            for o in things:record(story,f"Who has the {o} now?",own[o],loc,own,group)
            for p in [a,b,x,y]:
                n=str(sum(owner==p for owner in own.values()))
                for q in [f"What is the number of items held by {p}?",f"Name the number of items held by {p}?"]:
                    record(story,q,n,loc,own,group)
            for p in [a,b]:
                record(story,f"Where is {p}?",loc[p],loc,own,group)
                for q in [f"What is the location of {p}?",f"Name the location of {p}?"]:
                    record(story,q,loc[p],loc,own,group)
            # Rooms distinguish identity from inverse location traversal. An
            # actor/object-only identity lesson cannot separate those meanings.
            for p in [a,x,y,things[0],things[1],things[2],things[3]]:
                for q in [f"What is {p}?",f"Name {p}?"]:record(story,q,p,loc,own,group)
            o=things[i%4];p=own[o] if i%2 else d
            record(story,f"Does {p} have the {o}?","yes" if own[o]==p else "no",loc,own,group)
            record(story,f"Is the {o} held by {p}?","yes" if own[o]==p else "no",loc,own,group)
            record(story,f"Is {a} equal to {a if i%2 else b}?","yes" if i%2 else "no",loc,own,group)
        else:
            for o in things:record(story,f"Who has the {o} now?",own[o],loc,own,"atomic_owner",False)
            for p in [a,b,x,y]:
                record(story,f"What is the number of items held by {p}?",str(sum(v==p for v in own.values())),loc,own,"atomic_count",False)
            for p in [a,b]:record(story,f"Where is {p}?",loc[p],loc,own,"retention_mixed",False)
            n=lambda p:sum(v==p for v in own.values())
            record(story,f"What is the number of items held by the location of {a}?",str(n(loc[a])),loc,own,"composition_count_at_location",False)
            record(story,f"Is the number of items held by {a} equal to the number of items held by {b}?","yes" if n(a)==n(b) else "no",loc,own,"composition_equal_counts",False)
            record(story,f"Is the location of {a} equal to the location of {c}?","yes" if loc[a]==loc[c] else "no",loc,own,"composition_equal_locations",False)
            record(story,f"Is the number of items held by the location of {a} equal to the number of items held by {b}?","yes" if n(loc[a])==n(b) else "no",loc,own,"composition_nested",False)
    for phase in (None,"gave","handed","passed"):
        for i in range(12):make(i,phase,True)
    for i in range(60):make(i,("gave","handed","passed")[i%3],False)
    counts=collections.Counter()
    # Add a bounded selection of genuine v2 training records without translating
    # their sentences or questions. Filtering is teacher-side curriculum selection.
    for ep in episodes(V2/"stage5/stage5_events_train.txt"):
        s,rest=ep.split("USER: ",1);q,a=rest.split("ASSISTANT: ",1);story=s.strip().splitlines();q=q.strip();a=a.strip()
        own=visible_possession(story)
        if own is None or not q.startswith("Who has the ") or a=="unknown":continue
        train.append(dict(story=story,question=q,answer=a,group="v2_training"));counts["v2_training_selected"]+=1
        if counts["v2_training_selected"]==160:break
    for line in (V2/"stage5/stage5_events_test.jsonl").open():
        r=json.loads(line)
        for q,a,_,story in r["qa"]:
            counts["v2_total_questions"]+=1;own=visible_possession(story)
            if own is None or a=="unknown" or not(q.startswith("Who has the ") or q.startswith("Does ")):continue
            mentioned=set(re.findall(r"\w+"," ".join(story)))
            m=re.fullmatch(r"Does (\w+) have the (\w+)\?",q)
            if m and (m[1] not in mentioned or m[2] not in mentioned):counts["v2_unbound_query_skipped"]+=1;continue
            answer={"true":"yes","false":"no"}.get(a,a.lower())
            record(story,q,answer,{},own,"v2_transfer",False);counts["v2_selected"]+=1
    # Frozen source benchmark from the preceding bootstrap experiment.
    old=json.loads((ROOT/"logs/bootstrap_grounding/gold.json").read_text())
    for row in old:
        if row["group"]=="babi":record(row["story"],row["question"],row["answer"],row["state"],{},"babi_retention",False)
    write_episodes(OUT/"train.txt",train,True);write_episodes(OUT/"test.txt",test,False)
    (OUT/"gold.json").write_text(json.dumps(test,indent=2)+"\n")
    (OUT/"teacher_train.json").write_text(json.dumps(train,indent=2)+"\n")
    meta=dict(training=len(train),test=len(test),counts=dict(counts),groups=dict(collections.Counter(r["group"] for r in test)),
              protocol="QA-only teaching; original v2 subset selected explicitly; true/false normalized to yes/no in scoring; all oracle states stay outside native inputs.")
    (OUT/"preparation.json").write_text(json.dumps(meta,indent=2)+"\n");print(json.dumps(meta,indent=2))


if __name__=="__main__":main()
