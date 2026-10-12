"""Freeze and run 48 paired control probes; no policy training or full replay."""
from collections import Counter, defaultdict
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def rows(path):
    with Path(path).open() as stream:
        for line in stream:
            yield json.loads(line)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


parser = argparse.ArgumentParser()
parser.add_argument("--out", type=Path, default=Path("logs/goal_targeted_isolation_v1"))
args = parser.parse_args()
root = args.out
root.mkdir(exist_ok=False)
pilot = Path("logs/goal_continuing_bootstrap_pilot_20260923_v1")
for folder in ("apps", "include", "tests"):
    shutil.copytree(Path("cpp")/folder, root/"source/cpp"/folder)
manifest = {str(p.relative_to(root/"source")): sha(p) for p in (root/"source").rglob("*") if p.is_file()}
(root/"source_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
checkpoint_labels = {"warm_m399": "warm", "autonomous_m599": "final"}
for row in rows(pilot/"pilot_weights.jsonl"):
    if row["label"] in checkpoint_labels:
        (root/(checkpoint_labels[row["label"]]+".weights")).write_text(" ".join(map(str,row["weights"]))+"\n")
support = {}
training = defaultdict(Counter)
evaluations = defaultdict(lambda: defaultdict(Counter))
for row in rows(pilot/"pilot_trace.jsonl"):
    if row["event"] != "decision":
        continue
    o = row["observation"]
    if row["mission"].startswith("warm_") and row["teacher"]:
        key = row["observation_bits"], tuple(row["menu"])
        group = 0 if o[29] else 1 if o[30] else 2 if o[5] and o[6] else 3
        if key in support:
            assert support[key][2] == row["operation"]
        support[key] = group, sum(1<<a for a in row["menu"]), row["operation"], o
    if row["mission"].startswith("eval_"):
        checkpoint, index = row["mission"][5:].rsplit("_m",1)
        if int(index)<32 and int(index)%8 in (1,2):
            c = evaluations[checkpoint][int(index)%8]
            c["decisions"] += 1
            c["agenda_decisions"] += bool(o[29])
            c["pending_arrival_decisions"] += not o[29] and bool(o[30])
            c["pending_arrival_switches"] += not o[29] and bool(o[30]) and row["operation"]==5
            c["derived_activations"] += bool(o[29]) and bool(o[13]) and row["operation"]==6
            c["root_finish"] += not o[29] and row["operation"]==4 and not row["active_before"].startswith("derived_")
            c["known_true_nonfinish"] += not o[29] and bool(o[5]) and bool(o[6]) and row["operation"]!=4
with (root/"support.txt").open("w") as stream:
    for group,mask,label,o in support.values():
        stream.write(" ".join(map(str,[group,mask,label,*o]))+"\n")
for m in rows(pilot/"pilot_missions.jsonl"):
    if m["arm"] in ("warm","autonomous") and m["index"]%8 in (1,2):
        stage = "fully_guided" if m["index"]<100 else "taper_with_fits" if m["index"]<250 else "taper_without_fits" if m["index"]<400 else "autonomous"
        c = training[(stage,m["index"]%8)]
        c["missions"]+=1;c["successes"]+=m["success"];c["hints"]+=m["hints"];c["decisions"]+=m["decisions"]
declaration = {"scope":"48 probes: two fresh libraries x two task families x four matched cases x reference/warm/final policy. Not restored final semantic archives or single-task retraining.",
    "reference":"unchanged common teacher; semantic learning on within each fresh case, zero policy updates",
    "neural":"saved warm/final weights; frozen learning; fresh clone each case; no hints",
    "training_phases":{f"{s}_family{f}":dict(c) for (s,f),c in training.items()},
    "actual_pilot_evaluation_behavior":evaluations,
    "unique_teacher_states":len(support),
    "support_groups":{"0":"agenda","1":"execution pending arrival","2":"execution cached true","3":"other execution"},
    "input_sha256":{name:sha(pilot/name) for name in ("pilot_config.json","pilot_trace.jsonl","pilot_weights.jsonl")}}
(root/"declaration.json").write_text(json.dumps(declaration,indent=2)+"\n")
binary = root/"probe"
with (root/"build.log").open("w") as log:
    subprocess.run(["g++","-std=c++20","-O2","-DNDEBUG","-I"+str(root/"source/cpp/include"),str(root/"source/cpp/apps/goal_targeted_isolation.cpp"),"-o",str(binary)],stdout=log,stderr=subprocess.STDOUT,check=True)
with (root/"run.log").open("w") as log:
    subprocess.run([str(binary),str(root/"probe"),str(root/"warm.weights"),str(root/"final.weights"),str(root/"support.txt")],stdout=log,stderr=subprocess.STDOUT,check=True)
groups=defaultdict(Counter)
missions=list(rows(root/"probe_missions.jsonl"))
assert len(missions)==48
for m in missions:
    assert m["updates_before"]==m["updates_after"]==0
    if "reference" not in m["id"]:assert m["hints"]==0 and not m["learning"]
    c=groups[m["id"].rsplit("_c",1)[0]];c["cases"]+=1;c["successes"]+=m["success"]
result={"declaration":declaration,"results":groups,"phase_checks":list(rows(root/"probe_phases.jsonl")),
        "binary_sha256":sha(binary),"source_manifest_sha256":sha(root/"source_manifest.json")}
(root/"results.json").write_text(json.dumps(result,indent=2)+"\n")
Path("docs/GOAL_TARGETED_ISOLATION_RESULTS.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(groups,indent=2))
