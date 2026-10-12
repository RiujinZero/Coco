"""Teach new event meanings with consequence answers, never after-state inputs."""
import json
import random
from pathlib import Path
from event_grounding_probe import update, observed, write_world
from reasoning_language_probe import FACTS

ROOT=Path("logs/answer_grounding_corpus")
PHRASES=(("acquire",1,3,"{a} acquired {t}."),
         ("passive",1,3,"{t} passed into the hands of {a}."),
         ("release",1,1,"{a} relinquished {t}."),
         ("redirect",0,2,"{a} redirected to {old}."),
         ("add",1,0,"{a} recorded {t}."))


def make(rng,seed,serial,kind,length,n):
    index=0
    def fresh():
        nonlocal index
        index+=1;return f"e{seed}x{serial}n{index}"
    a,old,b,c,t,other=(fresh() for _ in range(6)); facts=set()
    for source,dest in ((a,c),(old,b)):
        path=[source]+[fresh() for _ in range(length-1)]+[dest]
        facts.update((s,0,o) for s,o in zip(path,path[1:]))
    for actor,count in ((a,n),(old,n+1),(b,n+2),(c,n+3)):
        values=[fresh() for _ in range(count)]
        if actor==(a if kind=="release" else old):values[:2]=[t,other]
        facts.update((actor,1,item) for item in values)
    names=dict(a=a,old=old,b=b,c=c,t=t,other=other)
    shuffled=sorted(facts);rng.shuffle(shuffled)
    text=" ".join(FACTS[rng.randrange(2)][r].format(a=s,b=o) for s,r,o in shuffled)
    return names,facts,text


def count(facts,a):return str(len({o for s,r,o in facts if s==a and r==1}))
def endpoint(facts,a):
    seen=set()
    while True:
        values={o for s,r,o in facts if s==a and r==0}
        if not values:return a
        assert len(values)==1 and a not in seen;seen.add(a);a=next(iter(values))


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    for seed in range(3):
        rng=random.Random(950000+seed); train=[]; tests=[]
        for family,(kind,rel,effect,phrase) in enumerate(PHRASES):
            for i in range(12):
                scene=family*100+i
                names,before,text=make(rng,seed,scene,kind,rng.randint(1,3),rng.randint(3,5))
                a,old,t=names['a'],names['old'],names['t'];dest=old if kind=="redirect" else t
                after=update(before,rel,effect,a,dest)
                weak=f"What is {a}?" if kind=="redirect" else f"What is item count of {a}?"
                strong=f"What is endpoint of {a}?" if kind=="redirect" else f"What is item count of {old}?"
                weak_gold=a if kind=="redirect" else count(after,a)
                strong_gold=endpoint(after,a) if kind=="redirect" else count(after,old)
                followup=f"{old} picked up {t}." if kind in ("acquire","passive","add") else ""
                later=update(after,1,3,old,t) if followup else after
                finalq=f"What is item count of {t if kind=='release' else a}?"
                finalgold=count(later,t if kind=="release" else a)
                repeatgold=a if kind=="redirect" else count(later,a)
                records=(("weak",0,weak,weak_gold,"",after),("repeat",1,weak,weak_gold,"",after),("contrast",1,strong,strong_gold,"",after),
                         ("repeat",2,weak,repeatgold,followup,later),("contrast",2,finalq,finalgold,followup,later))
                for phase,qid,q,gold,later_text,outcome in records:
                    train.append(dict(phase=phase,scene=scene,qid=qid,event=phrase.format(**names),followup=later_text,before_text=text,question=q,gold=gold,before=sorted(observed(before,seed)),after=sorted(observed(outcome,seed)),effect=effect,relation=1-rel if seed%2 else rel,subject=a,object=dest))
        with (ROOT/f"train_{seed}.txt").open("w") as out:
            for x in train:
                out.write(f"{x['phase']}\t{x['scene']}\t{x['qid']}\n{x['before_text']}\n{x['event']}\n{x['followup']}\n{x['question']}\n{x['gold']}\n")
        (ROOT/f"train_{seed}.json").write_text(json.dumps(train,indent=2)+"\n")
        for kind in [x[0] for x in PHRASES]+["repeated"]:
            source_kind="acquire" if kind=="repeated" else kind
            _,rel,effect,phrase=next(x for x in PHRASES if x[0]==source_kind)
            for i in range(40):
                serial=100000+len(tests)
                names,before,text=make(rng,seed,serial,source_kind,rng.choice((4,8,12,32)),rng.randint(4,20))
                a,old,t=names['a'],names['old'],names['t'];dest=old if kind=="redirect" else t
                after=update(before,rel,effect,a,dest)
                event=phrase.format(**names)
                if kind=="repeated":
                    event+=" "+phrase.format(**{**names,"t":names['other']})
                    after=update(after,rel,effect,a,names['other'])
                query=f"What is endpoint of {a}?" if kind=="redirect" else f"What is item count of {a if kind=='release' else old}?"
                gold=endpoint(after,a) if kind=="redirect" else count(after,a if kind=="release" else old)
                invariant=f"What is {a}?" if kind=="redirect" else f"What is item count of {a}?"
                invariant_gold=a if kind=="redirect" else count(after,a)
                tests.append(dict(group=kind,before_text=text,event=event,before=sorted(observed(before,seed)),after=sorted(observed(after,seed)),turns=[dict(stage="before",text=f"{text} What is {a}?",gold=a,weak=a),dict(stage="diagnostic",text=event+" "+query,gold=gold,weak=gold if kind=="release" else "I do not know."),dict(stage="invariant",text=invariant,gold=invariant_gold,weak=invariant_gold)]))
        (ROOT/f"test_{seed}.json").write_text(json.dumps(tests,indent=2)+"\n")
        with (ROOT/f"test_{seed}.txt").open("w") as out:
            for x in tests:
                out.write(x['group']+"\n");write_world(out,x['before']);write_world(out,x['after'])
                for t in x['turns']:out.write(f"{t['stage']}\t{t['gold']}\t{t['weak']}\n{t['text']}\n")
    print(f"Wrote consequence-answer teaching and held-out event cases to {ROOT}")


if __name__=="__main__":main()
