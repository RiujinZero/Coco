"""Independent textual semantics and knowledge-state audit; no native imports."""
import collections
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"logs/observation_language"
SOURCE=Path("/home/user/curriculum_v2/stage5/stage5_events_test.jsonl")


def interpret(story):
    positive,negative,rows,columns=set(),set(),set(),set()
    def truth(a,r,b):
        if (a,r,b) in positive:return "yes"
        if (a,r,b) in negative or (a,r) in rows or (r,b) in columns:return "no"
        return "unknown"
    for raw in story:
        s=raw.lower()
        move=re.fullmatch(r"(\w+) moved to the (\w+)\.",s)
        has=re.fullmatch(r"(\w+) has the (\w+)\.",s)
        give=re.fullmatch(r"(\w+) (?:gave|handed|passed) the (\w+) to (\w+)\.",s)
        if move or has or give:
            if move:a,r,b=move[1],0,move[2]
            elif has:a,r,b=has[2],1,has[1]
            else:a,r,b=give[2],1,give[3]
            positive={f for f in positive if f[0]!=a or f[1]!=r}|{(a,r,b)}
            negative={f for f in negative if f[0]!=a or f[1]!=r};rows.add((a,r));continue
        gap=re.fullmatch(r"no one reported anything about the (\w+) for a while\.",s)
        location_gap=re.fullmatch(r"no one saw the destinations of (\w+) for a while\.",s)
        column_gap=re.fullmatch(r"there was no observation of what (\w+) held for a while\.",s)
        if gap or location_gap or column_gap:
            focus=(gap or location_gap or column_gap)[1];r=0 if location_gap else 1
            if column_gap:
                positive={f for f in positive if f[1]!=r or f[2]!=focus};negative={f for f in negative if f[1]!=r or f[2]!=focus}
                columns.discard((r,focus));rows={f for f in rows if f[1]!=r}
            else:
                positive={f for f in positive if f[0]!=focus or f[1]!=r};negative={f for f in negative if f[0]!=focus or f[1]!=r}
                rows.discard((focus,r));columns={f for f in columns if f[0]!=r}
            continue
        visible=re.fullmatch(r"it is visible that (\w+) (holds|does not hold) the (\w+)\.",s)
        if visible:
            a,b=visible[3],visible[1];present=visible[2]=="holds"
            assert truth(a,1,b) in ("unknown","yes" if present else "no")
            (positive if present else negative).add((a,1,b));continue
        row=re.fullmatch(r"the complete holder list for the (\w+) is (\w+)\.",s)
        inventory=re.fullmatch(r"the full inventory of (\w+) consists of the (\w+)(?: and the (\w+))?\.",s)
        empty=re.fullmatch(r"(\w+) is visibly holding nothing\.",s)
        if row:
            values=set() if row[2]=="empty" else {row[2]}
            assert {b for a,r,b in positive if a==row[1] and r==1}<=values
            assert all(truth(row[1],1,b)!="no" for b in values)
            positive|={(row[1],1,b) for b in values};rows.add((row[1],1));continue
        if inventory or empty:
            focus=(inventory or empty)[1];values={v for v in inventory.groups()[1:] if v} if inventory else set()
            assert {a for a,r,b in positive if r==1 and b==focus}<=values
            assert all(truth(a,1,focus)!="no" for a in values)
            positive|={(a,1,focus) for a in values};columns.add((1,focus));continue
        assert re.fullmatch(r"someone said \w+ has the \w+\.",s),s
    return dict(positive=positive,negative=negative,rows=rows,columns=columns)


def answer(question,k):
    q=question.lower();positive,negative,rows,columns=(k[x] for x in ("positive","negative","rows","columns"))
    owner=re.fullmatch(r"does (\w+) have the (\w+)\?",q)
    location=re.fullmatch(r"is (\w+) at (\w+)\?",q)
    if owner or location:
        a,r,b=(owner[2],1,owner[1]) if owner else (location[1],0,location[2])
        return "yes" if (a,r,b) in positive else "no" if (a,r,b) in negative or (a,r) in rows or (r,b) in columns else "unknown"
    owner=re.fullmatch(r"who has the (\w+) now\?",q)
    location=re.fullmatch(r"where is (\w+)\?",q)
    if owner or location:
        a=(owner or location)[1];relation=int(bool(owner));values={b for s,r,b in positive if s==a and r==relation}
        return next(iter(values)) if (a,relation) in rows and len(values)==1 else "unknown"
    items=re.fullmatch(r"what is the number of items held by (\w+)\?",q)
    destinations=re.fullmatch(r"what is the number of destinations for (\w+)\?",q)
    assert items or destinations,q
    if items:return str(sum(r==1 and b==items[1] for a,r,b in positive)) if (1,items[1]) in columns else "unknown"
    return str(sum(r==0 and a==destinations[1] for a,r,b in positive)) if (destinations[1],0) in rows else "unknown"


def parse_knowledge(text,seed):
    k={key:set() for key in ("positive","negative","rows","columns")}
    for atom in text.split(";"):
        if not atom:continue
        f=atom.split(",")
        if f[0] in ("p","n"):k["positive" if f[0]=="p" else "negative"].add((f[1],int(f[2])^(seed%2),f[3]))
        elif f[0]=="r":k["rows"].add((f[1],int(f[2])^(seed%2)))
        else:assert f[0]=="c";k["columns"].add((int(f[1])^(seed%2),f[2]))
    return k


def native_inputs(filename,records,training):
    lines=iter((OUT/filename).read_text().splitlines())
    for i,row in enumerate(records):
        assert next(lines)==f"{row['scene'] if training else i}\t{len(row['story'])}"
        assert [next(lines) for _ in row["story"]]==row["story"]
        assert next(lines)==row["question"]
        if training:assert next(lines)==row["answer"]
    assert next(lines,None) is None


def previous_lessons(path):
    lines=iter(path.read_text().splitlines())
    while (header:=next(lines,None)) is not None:
        _,n=header.split("\t")
        yield dict(story=[next(lines) for _ in range(int(n))],question=next(lines),answer=next(lines))


def qa_key(row):
    def normalize(text):return tuple(re.findall(r"[a-z0-9]+",text.lower()))
    return tuple(normalize(s) for s in row["story"]),normalize(row["question"])


def main():
    seeds=list(map(int,sys.argv[1:])) or [0,1,2]
    training=json.loads((OUT/"teacher_train.json").read_text());gold=json.loads((OUT/"gold.json").read_text())
    source=[json.loads(line) for line in SOURCE.open()]
    for records in (training,gold):
        for row in records:
            knowledge=interpret(row["story"])
            assert knowledge=={key:set(map(tuple,items)) for key,items in row["knowledge"].items()}
            assert answer(row["question"],knowledge)==row["answer"],row
            if row["group"].startswith("v2_"):
                q,a,_,story=source[row["source_record"]]["qa"][row["source_question"]]
                assert row["story"]==story and row["question"]==q and row["answer"]=={"true":"yes","false":"no"}.get(a,a.lower())
    native_inputs("train.txt",training,True);native_inputs("test.txt",gold,False)
    prior_paths=[ROOT/f"logs/{name}/train.txt" for name in ("bootstrap_grounding","incremental_grounding","dialogue_grounding")]
    all_training=list(training)
    for path in prior_paths:all_training.extend(previous_lessons(path))
    assert not {qa_key(r) for r in all_training}&{qa_key(r) for r in gold}
    scene_stories={}
    for row in training:
        assert row["scene"] not in scene_stories or scene_stories[row["scene"]]==row["story"]
        scene_stories[row["scene"]]=row["story"]
    expected_frames={
        "$0 is visibly holding nothing ":(4,1,0,0),
        "it is visible that $0 does not hold the $1 ":(2,1,1,1),
        "it is visible that $0 holds the $1 ":(1,1,1,1),
        "no one reported anything about the $0 for a while ":(5,1,0,0),
        "no one saw the destinations of $0 for a while ":(5,0,0,0),
        "someone said $0 has the $1 ":(0,-1,-1,0),
        "the complete holder list for the $0 is $1 ":(3,1,0,2),
        "the complete holder list for the $0 is empty ":(3,1,0,0),
        "the full inventory of $0 consists of the $1 ":(4,1,0,2),
        "the full inventory of $0 consists of the $1 and the $2 ":(4,1,0,6),
        "there was no observation of what $0 held for a while ":(6,1,0,0),
    }
    support=collections.defaultdict(set)
    for scene,story in scene_stories.items():
        names=set()
        for s in story:
            m=re.fullmatch(r"(\w+) (?:has the|moved to the) (\w+)\.",s.lower())
            if m:names.update(m.groups())
        for key in expected_frames:
            pattern=" ".join(r"(\w+)" if word.startswith("$") else re.escape(word) for word in key.split())
            for s in story:
                m=re.fullmatch(pattern," ".join(re.findall(r"[a-z0-9]+",s.lower())))
                if m and set(m.groups())<=names:support[key].add(scene)
    totals={}
    for seed in seeds:
        model_lines=(OUT/f"{seed}.model.txt").read_text().splitlines()
        assert model_lines[:5]==["status\tlearned",f"accepted\t{len(training)}","rejected\t0",f"checked\t{len(training)}","skipped\t0"]
        frames={};questions={}
        for line in model_lines[5:]:
            if line.startswith("question\t"):
                _,form,meanings=line.split("\t");questions[form]=meanings
            else:
                key,scenes,values=line.split(":");assert int(scenes)==len(support[key])>=4
                rules={tuple(map(int,r.split(","))) for r in values.split(";") if r}
                frames[key]={(kind,relation^(seed%2) if relation>=0 else -1,focus,mask) for kind,relation,focus,mask in rules}
        assert frames=={key:{rule} for key,rule in expected_frames.items()},frames
        expected_questions={
            "does $0 have the $1 ":f"{11^(seed%2)}($1,$0,);",
            "is $0 at $1 ":f"{10^(seed%2)}($0,$1,);",
            "where is $0 ":f"{16^(seed%2)}($0,);",
            "who has the $0 now ":f"{17^(seed%2)}($0,);",
            "what is the number of destinations for $0 ":f"{6^(seed%2)}($0,);",
            "what is the number of items held by $0 ":f"{15^(seed%2)}($0,);",
        }
        assert questions==expected_questions,questions
        scores=collections.defaultdict(collections.Counter);seen=set()
        with (OUT/f"{seed}.trace.tsv").open() as f:
            for row in csv.DictReader(f,delimiter="\t"):
                mode,index=row["mode"],int(row["case"]);key=(mode,index);assert key not in seen;seen.add(key)
                example=gold[index];counter=scores[mode+"/"+example["group"]];counter["total"]+=1
                expected="I do not know." if example["answer"]=="unknown" else example["answer"]
                correct=row["reply"]==expected;counter["correct"]+=correct
                counter["definite_targets"]+=example["answer"]!="unknown"
                counter["unknown_targets"]+=example["answer"]=="unknown"
                gap=any(s.startswith(("No one reported", "No one saw", "There was no observation")) for s in example["story"])
                if mode=="no_visibility_meanings" and gap:
                    assert row["complete"]=="0" and row["status"]=="unknown_event" and row["reply"]=="I do not know."
                    counter["blocked"]+=1
                else:
                    assert row["complete"]=="1" and row["worlds"]=="1",(seed,key,row)
                    assert parse_knowledge(row["knowledge"],seed)==interpret(example["story"]),(seed,key,row,example)
                    if mode!="chain_read" or not example["question"].startswith(("Who has", "Where is")):
                        assert correct,(seed,key,row,example)
                if mode=="grounded":assert correct,(seed,key,row,example)
        assert seen=={(mode,i) for mode in ("grounded","no_visibility_meanings","chain_read") for i in range(len(gold))}
        totals[str(seed)]={group:dict(score) for group,score in scores.items()}
    paths=list(OUT.glob("*.txt"))+list(OUT.glob("*.tsv"))
    paths+=[OUT/name for name in ("teacher_train.json","gold.json","preparation.json")]
    paths+=prior_paths+[ROOT/f"logs/reasoning_composition_verified/micro_{seed}.txt" for seed in seeds]
    paths+=[ROOT/name for name in ("cpp/include/laminar/observation_language.h","cpp/include/laminar/observation_language_learner.h","cpp/include/laminar/fact_knowledge.h","cpp/include/laminar/procedure_memory.h","cpp/include/laminar/reasoning_language.h","cpp/include/laminar/semantic_grounding.h","cpp/include/laminar/event_grounding.h","cpp/include/laminar/incremental_grounding.h","cpp/tests/test_observation_language.cpp","cpp/apps/interaction_probe_support.h","cpp/apps/grounding_probe_support.h","cpp/apps/observation_language_probe.cpp","python/observation_language_probe.py","python/check_observation_language.py","scripts/observation_language.sh","build-wsl/observation_language_probe")]
    result=dict(seeds=seeds,training_scenes=len(scene_stories),controlled_test_histories=len({tuple(r["story"]) for r in gold if not r["group"].startswith("v2_")}),prior_training_qa_overlap=0,totals=totals,failures=[],sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths+[SOURCE]})
    (OUT/("audit.json" if seeds==[0,1,2] else "partial_audit.json")).write_text(json.dumps(result,indent=2)+"\n")
    for seed,groups in totals.items():
        summary=collections.defaultdict(collections.Counter)
        for group,score in groups.items():summary[group.split("/")[0]].update(score)
        print(seed,json.dumps(summary,indent=2))
    print("Verified raw training/evaluation text, exact learned meanings, complete knowledge states, original v2 records and ablations.")


if __name__=="__main__":main()
