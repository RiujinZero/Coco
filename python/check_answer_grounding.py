"""Independently enumerate event meanings and audit every retained world."""
import collections
import csv
import hashlib
import itertools
import json
import re
from pathlib import Path
from check_reference_grounding import event as known_event, norm, channel

ROOT=Path(__file__).resolve().parents[1]


def apply(world,r,effect,a,b):
    result=set()
    for s,rel,o in world:
        remove=rel==r and ((effect==1 and (s,o)==(a,b)) or (effect==2 and s==a) or (effect==3 and o==b))
        if not remove:result.add((s,rel,o))
    if effect!=1:result.add((a,r,b))
    return result


def observe_before(text,seed):
    world=set();ids={}
    for clause in text.split('.'):
        if not clause.strip():continue
        a,r,b,effect=known_event(clause,seed)
        for name in (a,b):
            if name not in ids:ids[name]=len(ids)
        world=apply(world,r,effect,a,b)
    return world,ids


def question(text,world,ids,seed):
    text=norm(text)
    match=re.fullmatch(r"what is item count of (\w+)",text)
    if match:return str(len({o for s,r,o in world if s==match[1] and r==channel(1,seed)}))
    match=re.fullmatch(r"what is endpoint of (\w+)",text)
    if match:
        x=match[1];seen=set()
        while True:
            options={o for s,r,o in world if s==x and r==channel(0,seed)}
            if not options:return x
            if x in seen:return None
            seen.add(x);x=min(options,key=ids.__getitem__)
    match=re.fullmatch(r"what is (\w+)",text);assert match,text
    return match[1]


def frame(text,ids):
    slots={};args=[];words=[]
    for word in norm(text).split():
        if word in ids:
            if word not in slots:slots[word]=len(args);args.append(word)
            words.append(f"${slots[word]}")
        else:words.append(word)
    return " ".join(words)+" ",args


def new_event(text,seed):
    patterns=((r"(\w+) acquired (\w+)",1,3,False),
              (r"(\w+) passed into the hands of (\w+)",1,3,True),
              (r"(\w+) relinquished (\w+)",1,1,False),
              (r"(\w+) redirected to (\w+)",0,2,False),
              (r"(\w+) recorded (\w+)",1,0,False))
    for pattern,r,effect,reverse in patterns:
        match=re.fullmatch(pattern,norm(text))
        if match:
            a,b=match.groups()
            if reverse:a,b=b,a
            return channel(r,seed),effect,a,b
    raise AssertionError(text)


def candidates(record,seed):
    world,ids=observe_before(record['before_text'],seed)
    assert world==set(map(tuple,record['before']))
    key,args=frame(record['event'],ids);assert len(args)==2
    true=new_event(record['event'],seed)
    assert true==(record['relation'],record['effect'],record['subject'],record['object'])
    actual=apply(world,*true)
    if record['followup']:
        a,r,b,effect=known_event(record['followup'],seed);actual=apply(actual,r,effect,a,b)
    assert actual==set(map(tuple,record['after']))
    assert question(record['question'],actual,ids,seed)==record['gold']
    choices=set()
    for r,s,o,effect in itertools.product((0,1),range(2),range(2),range(4)):
        result=apply(world,r,effect,args[s],args[o])
        if record['followup']:
            a,relation,b,later_effect=known_event(record['followup'],seed);result=apply(result,relation,later_effect,a,b)
        if question(record['question'],result,ids,seed)==record['gold']:choices.add((r,s,o,effect))
    truth=(true[0],args.index(true[2]),args.index(true[3]),true[1]);assert truth in choices
    return key,choices,truth,set(ids)


def training_audit(root,seed,result):
    records=json.loads((root/f"train_{seed}.json").read_text());lines=iter((root/f"train_{seed}.txt").read_text().splitlines())
    model={m:{} for m in ('weak','full')};scenes=collections.defaultdict(set);truths={};counts=collections.Counter();names=set()
    for record in records:
        assert next(lines)==f"{record['phase']}\t{record['scene']}\t{record['qid']}"
        for k in ('before_text','event','followup','question','gold'):assert next(lines)==record[k]
        key,choices,truth,entities=candidates(record,seed);names|=entities
        truths[key]=truth;scenes[key].add(record['scene']);counts[record['phase']]+=1
        for mode in model:
            if record['phase']=='weak' or record['phase']==('repeat' if mode=='weak' else 'contrast'):
                old=model[mode].get(key);model[mode][key]=choices if old is None else old&choices
    assert next(lines,None) is None
    assert counts=={'weak':60,'repeat':120,'contrast':120}
    for mode in model:
        reported=result[mode+'_model'];assert set(reported)==set(model[mode])
        for key,choices in model[mode].items():
            assert choices==set(map(tuple,reported[key]['candidates'])),(seed,mode,key,choices,reported[key])
            assert reported[key]['scenes']==len(scenes[key])==12
            if mode=='full':assert choices=={truths[key]},(seed,key,choices,truths[key])
    return model,names


def digest(worlds):
    encoded=[]
    for _,world in worlds:
        encoded.append(''.join(f"{a}\t{r}\t{b}\n" for a,r,b in sorted(world)))
    h=14695981039346656037
    for data in sorted(encoded):
        for byte in data.encode()+bytes([30]):h=((h^byte)*1099511628211)&((1<<64)-1)
    return str(h)


def advance(worlds,text,ids,rules):
    for clause in text.split('.'):
        if not clause.strip():continue
        key,args=frame(clause,ids);updated={}
        for bindings,world in worlds:
            for r,s,o,effect in rules[key]:
                rule=(r,s,o,effect)
                if key in bindings and bindings[key]!=rule:continue
                new_bindings={**bindings,key:rule};new=apply(world,r,effect,args[s],args[o])
                identity=(tuple(sorted(new_bindings.items())),tuple(sorted(new)))
                updated[identity]=(new_bindings,new)
        worlds=list(updated.values())
    return worlds


def read_world(lines):
    out=set()
    for _ in range(int(next(lines))):
        a,r,b=next(lines).split('\t');out.add((a,int(r),b))
    return out


def audit_tests(root,seed,models,training_names,result,trace):
    records=json.loads((root/f"test_{seed}.json").read_text());lines=iter((root/f"test_{seed}.txt").read_text().splitlines())
    by_turn={};all_names=set()
    for index,c in enumerate(records):
        assert next(lines)==c['group'];assert read_world(lines)==set(map(tuple,c['before']));assert read_world(lines)==set(map(tuple,c['after']))
        before,ids=observe_before(c['before_text'],seed);assert before==set(map(tuple,c['before']));all_names|=set(ids)
        actual=before
        for clause in c['event'].split('.'):
            if clause.strip():actual=apply(actual,*new_event(clause,seed))
        assert actual==set(map(tuple,c['after']))
        for t in c['turns']:
            assert next(lines)==f"{t['stage']}\t{t['gold']}\t{t['weak']}";assert next(lines)==t['text']
            q=t['text'].split('.')[-1];gold_world=before if t['stage']=='before' else actual
            assert question(q,gold_world,ids,seed)==t['gold']
            by_turn[(index,t['stage'])]=(c,t,before,ids,actual)
    assert next(lines,None) is None and not all_names&training_names
    totals=collections.defaultdict(collections.Counter)
    seen=set()
    for row in trace:
        mode,index,stage=row['mode'],int(row['case']),row['stage'];assert (mode,index,stage) not in seen;seen.add((mode,index,stage))
        c,t,before,ids,actual=by_turn[(index,stage)];assert c['group']==row['group']
        key=f"{mode}/{c['group']}/{stage}";score=totals[key];score['n']+=1
        conflicting=mode=='conflict' and c['group'] in ('acquire','repeated') and stage!='before'
        worlds=[({},before)]
        if stage!='before' and not conflicting:
            rules=models['weak' if mode in ('weak','first_candidate') else 'full']
            if mode=='first_candidate':rules={k:{min(v)} for k,v in rules.items()}
            worlds=advance(worlds,c['event'],ids,rules)
        assert int(row['worlds'])==len(worlds) and row['worlds_digest']==digest(worlds),(seed,key,row,len(worlds))
        if conflicting:
            expected_reply='The event evidence conflicts.';assert row['status']=='conflicting_event_evidence'
        else:
            values={question(t['text'].split('.')[-1],w,ids,seed) for _,w in worlds}
            if len(worlds)>1 and (len(values)!=1 or None in values):
                expected_reply='I do not know.';assert row['status']=='uncertain_answer'
            elif values=={None}:expected_reply='<no proposal>'
            else:expected_reply=next(iter(values));assert row['status'] in ('answer','grounded_answer')
        assert row['reply']==expected_reply,(seed,key,row,expected_reply)
        if mode=='weak':assert expected_reply==t['weak'],(seed,key,expected_reply,t['weak'])
        expected=before if stage=='before' else actual
        common=set.intersection(*(set(w) for _,w in worlds))
        exact_state=int(not conflicting and common==expected);contains=int(any(w==expected for _,w in worlds))
        assert int(row['state'])==exact_state and int(row['contains_truth'])==contains
        answered=row['status'] in ('answer','grounded_answer')
        score['exact']+=answered and row['reply']==t['gold'];score['uncertain']+=row['status']=='uncertain_answer'
        score['conflicted']+=row['status']=='conflicting_event_evidence';score['answered']+=answered
        score['state']+=exact_state;score['contains_truth']+=contains
        if mode in ('full','corrected'):assert answered and row['reply']==t['gold'] and exact_state
    assert len(seen)==5*len(by_turn) and dict(totals)==result['scores']
    return totals


def main():
    totals=collections.defaultdict(collections.Counter);paths=[]
    for seed in range(3):
        root=ROOT/'logs/answer_grounding_corpus';output=ROOT/f'logs/answer_grounding_native/{seed}.json';trace=Path(str(output)+'.trace.tsv')
        result=json.loads(output.read_text());assert result['seed']==seed
        models,names=training_audit(root,seed,result)
        with trace.open() as f:audited=audit_tests(root,seed,models,names,result,list(csv.DictReader(f,delimiter='\t')))
        for key,sc in audited.items():totals[key].update(sc)
        paths += [root/f'{stem}_{seed}.{suffix}' for stem in ('train','test') for suffix in ('txt','json')]+[output,trace]
        paths += [ROOT/f'logs/{prefix}_{seed}.txt' for prefix in ('event_grounding_corpus/sensors','semantic_grounding_corpus/train','reasoning_composition_verified/micro')]
    paths += [ROOT/p for p in ('cpp/include/laminar/answer_grounding.h','cpp/include/laminar/event_grounding.h','cpp/include/laminar/reasoning_language.h','cpp/include/laminar/procedure_memory.h','cpp/include/laminar/semantic_grounding.h','cpp/include/laminar/reasoning_runtime.h','cpp/apps/answer_grounding_probe.cpp','cpp/apps/grounding_probe_support.h','cpp/tests/test_answer_grounding.cpp','python/answer_grounding_probe.py','python/check_answer_grounding.py','python/check_reference_grounding.py','scripts/answer_grounding.sh','build-wsl/answer_grounding_probe')]
    output=ROOT/'logs/answer_grounding_audit.json'
    output.write_text(json.dumps(dict(totals=totals,sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}),indent=2)+'\n')
    compact=collections.defaultdict(collections.Counter)
    for key,score in totals.items():
        mode,group,stage=key.split('/');compact[mode+('/control' if stage=='before' else '/consequence')].update(score)
    print(json.dumps(compact,indent=2));print(f'Verified answer-only hypotheses, possible worlds, corrections and native outputs: {output}')


if __name__=='__main__':main()
