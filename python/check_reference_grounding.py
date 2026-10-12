"""Independent interpreter for reference fixtures plus native-result audit."""
import collections
import csv
import hashlib
import json
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PHRASES={"it":(1,1),"that item":(1,1),"that owner":(1,0),"there":(0,1),"that thing":(1,1)}
PATTERNS=((r"(\w+) leads to (\w+)",0,2,False),(r"(\w+) is reached from (\w+)",0,2,True),
          (r"(\w+) holds (\w+)",1,3,False),(r"(\w+) is held by (\w+)",1,3,True),
          (r"(\w+) now connects to (\w+)",0,2,False),(r"(\w+) is now reached from (\w+)",0,2,True),
          (r"(\w+) picked up (\w+)",1,3,False),(r"(\w+) was taken by (\w+)",1,3,True))


def norm(text): return " ".join(re.findall("[a-z0-9]+",text.lower()))
def channel(r,seed): return 1-r if seed%2 and r<2 else r


def event(text,seed):
    for pattern,r,effect,reverse in PATTERNS:
        match=re.fullmatch(pattern,norm(text))
        if match:
            a,b=match.groups()
            if reverse:a,b=b,a
            return a,channel(r,seed),b,effect
    raise AssertionError(f"Unrecognized oracle event: {text}")


def update(state,e):
    a,r,b,effect=e
    assert effect in (2,3)
    result={f for f in state if not(f[1]==r and (f[0]==a if effect==2 else f[2]==b))}
    result.add((a,r,b)); return result


def resolve(text,context,seed):
    text=norm(text)
    for phrase in sorted(PHRASES,key=len,reverse=True):
        if not re.search(r"\b"+phrase+r"\b",text):continue
        r,role=PHRASES[phrase]; r=channel(r,seed)
        names={e[2] if role else e[0] for e in context if e[1]==r}
        if len(names)!=1:return None,(text,phrase,names)
        text=re.sub(r"\b"+phrase+r"\b",next(iter(names)),text)
    return text,None


def answer(question,state,seed):
    q=norm(question)
    match=re.fullmatch(r"what is item count of (endpoint of )?(\w+)",q)
    if match:
        follow,x=match.groups(); seen=set()
        if follow:
            while True:
                values={b for a,r,b in state if a==x and r==channel(0,seed)}
                if not values:break
                assert len(values)==1 and x not in seen
                seen.add(x); x=next(iter(values))
        return str(len({b for a,r,b in state if a==x and r==channel(1,seed)}))
    match=re.fullmatch(r"what is (\w+)",q); assert match,q
    return match[1]


def read_world(lines):
    result=set()
    for _ in range(int(next(lines))):
        a,r,b=next(lines).split("\t"); result.add((a,int(r),b))
    return result


def verify_sensor(path,seed):
    lines=iter(path.read_text().splitlines()); counts=collections.Counter(); seen=set(); names=set()
    for head in lines:
        phase,scene,n=head.split("\t"); assert scene not in seen; seen.add(scene)
        context=[event(next(lines),seed) for _ in range(int(n))]
        text=next(lines); before,after=read_world(lines),read_world(lines)
        named,pending=resolve(text,context,seed); assert pending is None
        assert before!=after and update(before,event(named,seed))==after
        matches=[p for p in PHRASES if re.search(r"\b"+p+r"\b",text)]
        assert len(matches)==1; counts[(phase,matches[0])]+=1
        names|={x for state in (before,after) for a,_,b in state for x in (a,b)}
    assert counts=={("adapt" if p=="that thing" else "train",p):12 for p in PHRASES}
    return names


def verify_dialogues(records,seed):
    names=set(); expected=collections.Counter()
    for case in records:
        state=set(); context=[]; pending=None; question=""; current=[]
        for t in case["turns"]:
            if t["reset"]:state=set(); context=[]; pending=None; current=[]; question=""
            if pending is None:question=""
            for clause,delimiter in re.findall(r"([^.?]+)([.?])",t["text"]):
                clause=norm(clause)
                if delimiter=="?":question=clause;continue
                if pending:
                    original,phrase,candidates,is_question=pending
                    known={x for a,_,b in state for x in (a,b)}
                    if clause not in known or (candidates and clause not in candidates):continue
                    named=re.sub(r"\b"+phrase+r"\b",clause,original,count=1); pending=None
                    if is_question:question=named;continue
                else:
                    named,unresolved=resolve(clause,current or context,seed)
                    if unresolved:pending=(*unresolved,False);continue
                e=event(named,seed); state=update(state,e); current.append(e)
            if pending is None:
                named,unresolved=resolve(question,current or context,seed)
                if unresolved:pending=(*unresolved,True)
                else:question=named
            assert state==set(map(tuple,t["state"])),(case["group"],t["stage"],"state")
            if pending:
                assert t["gold"]=="_" and t["phrase"]==pending[1] and set(t["candidates"])==pending[2]
            else:
                assert t["question"]==question,(case["group"],t["stage"],question)
                assert t["gold"]==answer(question,state,seed)
                if current:context=current;current=[]
            names|={x for a,_,b in state for x in (a,b)}
            expected[(case["group"],t["stage"])]+=1
    return names,expected


def verify_serialization(records,path):
    lines=iter(path.read_text().splitlines())
    for c in records:
        assert next(lines)==f"{c['group']}\t{len(c['turns'])}"
        for t in c["turns"]:
            assert next(lines)==f"{t['stage']}\t{int(t['reset'])}\t{t['gold']}"
            for key in ("text","question","phrase"):assert next(lines)==t[key]
            assert next(lines)==" ".join(t["candidates"])
            assert next(lines)==t["self_text"].replace("\n","|")
            assert read_world(lines)==set(map(tuple,t["state"]))
    assert next(lines,None) is None


def main():
    totals=collections.defaultdict(collections.Counter); paths=[]
    for seed in range(3):
        root=ROOT/"logs/reference_grounding_corpus"
        sensors=root/f"sensors_{seed}.txt"; corpus=root/f"test_{seed}.json"; serialized=root/f"test_{seed}.txt"
        paths += [sensors,corpus,serialized]
        sensor_names=verify_sensor(sensors,seed); records=json.loads(corpus.read_text())
        names,expected=verify_dialogues(records,seed); assert not sensor_names & names
        verify_serialization(records,serialized)
        output=ROOT/f"logs/reference_grounding_native/{seed}.json"; trace=Path(str(output)+".trace.tsv");paths += [output,trace]
        results=json.loads(output.read_text()); assert results["seed"]==seed and results["initial_phrases"]==4 and results["adapted_phrases"]==5
        assert len(results["scores"])==4*len(expected)
        trace_totals=collections.defaultdict(collections.Counter)
        with trace.open() as f:
            for row in csv.DictReader(f,delimiter="\t"):
                key="/".join(row[x] for x in ("mode","group","stage")); sc=trace_totals[key];sc["n"]+=1
                for name in ("exact","state","binding","clarification_exact"):sc[name]+=int(row[name])
                sc["answered"]+=row["status"] in ("answer","grounded_answer");sc["clarification"]+=row["status"]=="clarification"
        for key,sc in results["scores"].items():
            mode,group,stage=key.split('/'); n=expected[(group,stage)]
            assert dict(trace_totals[key])==sc and sc["n"]==n
            if mode in ("learned","adapted"):
                assert sc["state"]==n,(seed,key,sc)
                if stage in ("pending","invalid"):
                    assert sc["clarification_exact"]==n and sc["answered"]==0,(seed,key,sc)
                elif mode=="learned" and group=="new_alias" and stage=="target":assert sc["answered"]==0
                else:assert sc["exact"]==sc["binding"]==sc["answered"]==n,(seed,key,sc)
            elif stage=="target":assert sc["exact"]==0,(seed,key,sc)
            totals[key].update(sc)
        for prefix in ("event_grounding_corpus/sensors","semantic_grounding_corpus/train","reasoning_composition_verified/micro"):
            paths.append(ROOT/f"logs/{prefix}_{seed}.txt")
    paths += [ROOT/p for p in ("cpp/include/laminar/reference_grounding.h","cpp/include/laminar/reasoning_language.h","cpp/include/laminar/event_grounding.h","cpp/include/laminar/semantic_grounding.h","cpp/include/laminar/procedure_memory.h","cpp/include/laminar/reasoning_runtime.h","cpp/apps/reference_grounding_probe.cpp","cpp/apps/grounding_probe_support.h","cpp/tests/test_reference_grounding.cpp","python/reference_grounding_probe.py","python/check_reference_grounding.py","scripts/reference_grounding.sh","build-wsl/reference_grounding_probe")]
    output=ROOT/"logs/reference_grounding_audit.json"
    output.write_text(json.dumps({"totals":totals,"sha256":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}},indent=2)+"\n")
    compact=collections.defaultdict(collections.Counter)
    for key,score in totals.items():
        mode,group,stage=key.split('/')
        bucket="clarification" if stage in ("pending","invalid") else "transfer" if stage in ("target","resolved") else "setup"
        compact[mode+'/'+bucket].update(score)
    print(json.dumps(compact,indent=2));print(f"Verified independent reference interpretation, state, bindings and native outputs: {output}")


if __name__=="__main__":main()
