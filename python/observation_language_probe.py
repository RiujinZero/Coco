"""QA-only visibility lessons and read-only selection of original Stage 5 text."""
import collections
import hashlib
import json
from pathlib import Path
import random
import re
from bootstrap_grounding_probe import write_episodes

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "logs/observation_language"
SOURCE = Path("/home/user/curriculum_v2/stage5/stage5_events_test.jsonl")


class Knowledge:
    def __init__(self):
        self.positive, self.negative, self.rows, self.columns = set(), set(), set(), set()

    def update(self, a, r, b):
        self.positive = {x for x in self.positive if x[:2] != (a, r)} | {(a, r, b)}
        self.negative = {x for x in self.negative if x[:2] != (a, r)}
        self.rows.add((a, r))

    def forget(self, focus, column=False, relation=1):
        index = 2 if column else 0
        self.positive = {x for x in self.positive if not (x[1] == relation and x[index] == focus)}
        self.negative = {x for x in self.negative if not (x[1] == relation and x[index] == focus)}
        if column:
            self.columns.discard((relation, focus)); self.rows = {x for x in self.rows if x[1] != relation}
        else:
            self.rows.discard((focus, relation)); self.columns = {x for x in self.columns if x[0] != relation}

    def truth(self, a, r, b):
        if (a, r, b) in self.positive:
            return "yes"
        if (a, r, b) in self.negative or (a, r) in self.rows or (r, b) in self.columns:
            return "no"
        return "unknown"

    def unique(self, a, r):
        values = {b for s, rel, b in self.positive if s == a and rel == r}
        return next(iter(values)) if (a, r) in self.rows and len(values) == 1 else "unknown"

    def count(self, focus, r, column=False):
        if column:
            return str(len({a for a, rel, b in self.positive if rel == r and b == focus})) if (r, focus) in self.columns else "unknown"
        return str(len({b for a, rel, b in self.positive if rel == r and a == focus})) if (focus, r) in self.rows else "unknown"

    def export(self):
        return {key: sorted(getattr(self, key)) for key in ("positive", "negative", "rows", "columns")}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train, test = [], []
    rng = random.Random(921)
    kinds = ("known", "gap", "column_gap", "positive", "negative", "row_one", "row_empty", "column_one", "column_two", "column_empty", "location_gap", "claim")
    def scene(i, kind, training):
        prefix = "teach" if training else f"fresh{i}"
        people = [f"{prefix}person{j}" for j in range(4)]
        objects = [f"{prefix}item{j}" for j in range(4)]
        rooms = [f"{prefix}room{j}" for j in range(3)]
        rng.shuffle(people); rng.shuffle(objects)
        a,b,c,d = people; o,x,y,z = objects
        k = Knowledge(); story = []
        def move(subject, room):
            story.append(f"{subject} moved to the {room}."); k.update(subject, 0, room)
        def has(person, obj):
            story.append(f"{person} has the {obj}."); k.update(obj, 1, person)
        for j, name in enumerate(people+objects):
            move(name, rooms[(i+j)%3])
        has(c,y); has(d,z)
        # A second link distinguishes direct ownership from chain endpoints.
        has(c,a)
        if kind in ("known", "gap", "column_gap", "claim"):
            has(a,o); has(b,x)
        if kind == "gap":
            story.append(f"No one reported anything about the {o} for a while."); k.forget(o)
        elif kind == "column_gap":
            story.append(f"There was no observation of what {a} held for a while."); k.forget(a,True)
        elif kind == "positive":
            story.append(f"It is visible that {a} holds the {o}."); k.positive.add((o,1,a))
        elif kind == "negative":
            story.append(f"It is visible that {a} does not hold the {o}."); k.negative.add((o,1,a))
        elif kind == "row_one":
            story.append(f"The complete holder list for the {o} is {a}."); k.rows.add((o,1)); k.positive.add((o,1,a))
        elif kind == "row_empty":
            story.append(f"The complete holder list for the {o} is empty."); k.rows.add((o,1))
        elif kind in ("column_one", "column_two"):
            listed = [o] if kind == "column_one" else [o,x]
            text = f"The full inventory of {a} consists of the {o}" + (f" and the {x}" if len(listed)>1 else "") + "."
            story.append(text); k.columns.add((1,a)); k.positive.update((obj,1,a) for obj in listed)
        elif kind == "column_empty":
            story.append(f"{a} is visibly holding nothing."); k.columns.add((1,a))
        elif kind == "location_gap":
            story.append(f"No one saw the destinations of {a} for a while."); k.forget(a,relation=0)
        elif kind == "claim":
            has(d,b)  # Distinguish ignoring a claim from forgetting its subject.
            if i%2:
                for name in (b,o):
                    story.append(f"No one saw the destinations of {name} for a while."); k.forget(name,relation=0)
            story.append(f"Someone said {b} has the {o}.")
            if i%4>=2:
                story.append(f"There was no observation of what {d} held for a while."); k.forget(d,True)
        return story,k,people,objects,rooms

    def emit(story,k,people,objects,group,scene_id,training):
        cases=[]
        for p in people:
            for o in objects:
                cases.append((f"Does {p} have the {o}?",k.truth(o,1,p)))
        rooms=sorted({b for _,r,b in k.positive if r==0})
        if group.startswith("claim"):
            for p in people+objects:
                for o in people+objects:
                    if p not in people or o not in objects:
                        cases.append((f"Does {p} have the {o}?",k.truth(o,1,p)))
            rooms+=people+objects
        for name in people+objects:
            for room in rooms:
                cases.append((f"Is {name} at {room}?",k.truth(name,0,room)))
        for name in people+objects:
            cases.extend(((f"Who has the {name} now?",k.unique(name,1)),
                          (f"Where is {name}?",k.unique(name,0)),
                          (f"What is the number of items held by {name}?",k.count(name,1,True)),
                          (f"What is the number of destinations for {name}?",k.count(name,0))))
        for q,answer in cases:
            (train if training else test).append(dict(story=list(story),question=q,answer=answer,knowledge=k.export(),group=group,scene=scene_id))

    for training in (True,False):
        for index,kind in enumerate(kinds):
            for n in range(8 if training else 12):
                i=index*100+n; story,k,p,o,r=scene(i,kind,training)
                emit(story,k,p,o,kind,i,training)
                # Frozen evaluation interleaves learned meanings with known
                # events: a later report restores only the affected scope.
                if not training and kind in ("gap","column_gap","positive","negative","claim"):
                    story.append(f"{p[2]} has the {o[0]}.");k.update(o[0],1,p[2])
                    emit(story,k,p,o,kind+"_reveal",10000+i,False)
        if not training:
            # Held-out combinations: closing a scope, losing it, seeing a new
            # value and closing again. No multi-new-frame training story exists.
            for n in range(12):
                story,k,p,o,r=scene(3000+n,"column_two",False)
                story.append(f"There was no observation of what {p[0]} held for a while.");k.forget(p[0],True)
                story.append(f"It is visible that {p[0]} holds the {o[1]}.");k.positive.add((o[1],1,p[0]))
                emit(story,k,p,o,"composed_open",4000+n,False)
                story.append(f"The full inventory of {p[0]} consists of the {o[1]}.");k.columns.add((1,p[0]))
                emit(story,k,p,o,"composed_closed",5000+n,False)

    counts=collections.Counter()
    for record_index,line in enumerate(SOURCE.open()):
        for question_index,(q,answer,_,story) in enumerate(json.loads(line)["qa"]):
            counts["v2_total"]+=1
            if not(re.fullmatch(r"Does \w+ have the \w+\?",q) or re.fullmatch(r"Who has the \w+ now\?",q)):
                counts["v2_temporal_excluded"]+=1;continue
            k=Knowledge();names=set();supported=True;gap=False
            for s in story:
                m=re.fullmatch(r"(\w+) has the (\w+)\.",s)
                if m:
                    a,o=(v.lower() for v in m.groups());names.update((a,o));k.update(o,1,a);continue
                m=re.fullmatch(r"(\w+) (?:gave|handed|passed) the (\w+) to (\w+)\.",s)
                if m and m[1]!="He":
                    a,o,b=(v.lower() for v in m.groups());names.update((a,o,b));k.update(o,1,b);continue
                m=re.fullmatch(r"No one reported anything about the (\w+) for a while\.",s)
                if m:
                    o=m[1].lower();names.add(o);k.forget(o);gap=True;continue
                supported=False;break
            if not supported:
                counts["v2_other_language_excluded"]+=1;continue
            m=re.fullmatch(r"Does (\w+) have the (\w+)\?",q)
            arguments=[v.lower() for v in m.groups()] if m else [re.fullmatch(r"Who has the (\w+) now\?",q)[1].lower()]
            if not set(arguments)<=names:
                counts["v2_unbound_excluded"]+=1;continue
            expected=k.truth(arguments[1],1,arguments[0]) if m else k.unique(arguments[0],1)
            normalized={"true":"yes","false":"no"}.get(answer,answer.lower())
            assert expected==normalized,(story,q,answer,expected)
            test.append(dict(story=story,question=q,answer=normalized,knowledge=k.export(),group="v2_gap" if gap else "v2_visible",scene=20000+len(test),source_record=record_index,source_question=question_index))
            counts["v2_selected"]+=1;counts["v2_selected_"+normalized if normalized in ("yes","no","unknown") else "v2_selected_entity"]+=1
    # Shared scene IDs preserve independent support even with many questions.
    write_episodes(OUT/"train.txt",train,True)
    # write_episodes assigns consecutive IDs; replace IDs with physical scene IDs.
    data=(OUT/"train.txt").read_text().splitlines();cursor=0
    for row in train:
        fields=data[cursor].split("\t");fields[0]=str(row["scene"]);data[cursor]="\t".join(fields)
        cursor+=len(row["story"])+3
    assert cursor==len(data)
    (OUT/"train.txt").write_text("\n".join(data)+"\n")
    write_episodes(OUT/"test.txt",test,False)
    (OUT/"teacher_train.json").write_text(json.dumps(train,indent=2)+"\n")
    (OUT/"gold.json").write_text(json.dumps(test,indent=2)+"\n")
    meta=dict(training=len(train),testing=len(test),counts=dict(counts),groups=dict(collections.Counter(r["group"] for r in test)),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest())
    (OUT/"preparation.json").write_text(json.dumps(meta,indent=2)+"\n");print(json.dumps(meta,indent=2))


if __name__=="__main__":main()
