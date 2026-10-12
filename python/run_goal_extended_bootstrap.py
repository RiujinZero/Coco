"""Freeze, check, and supervise the declared 2,000-episode local training run.

The worker needs no model calls and writes durable status and final results.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save(path, value):
    path = Path(path)
    pending = path.with_suffix(path.suffix+".pending")
    pending.write_text(json.dumps(value, indent=2)+"\n")
    pending.replace(path)


def rows(path):
    with Path(path).open() as stream:
        for line in stream:
            yield json.loads(line)


def verify(root):
    manifest=load(root/"manifest.json")
    for path,digest in manifest["files"].items():
        assert sha(root/path)==digest, path
    return manifest


def prepare(root):
    root.mkdir(exist_ok=False)
    for folder in ("apps","include","tests"):
        shutil.copytree(Path("cpp")/folder,root/"source/cpp"/folder)
    (root/"source/python").mkdir()
    shutil.copyfile(__file__,root/"source/python/run_goal_extended_bootstrap.py")
    validation=load("docs/GOAL_BALANCED_DEMONSTRATIONS_VALIDATION.json")
    assert validation["passed"]
    for path,digest in validation["source_sha256"].items():
        assert sha(root/"source"/path)==digest, "validated source drift: "+path
    save(root/"balanced_fit_validation.json",validation)
    binary=root/"goal_extended_bootstrap"
    command=["g++","-std=c++20","-O2","-DNDEBUG","-I"+str(root/"source/cpp/include"),
             str(root/"source/cpp/apps/goal_extended_bootstrap.cpp"),"-o",str(binary)]
    with (root/"build.log").open("w") as log:
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    for profile in ("smoke","full"):
        subprocess.run([str(binary),"--declare",profile,str(root/profile)],check=True)
    files={str(p.relative_to(root)):sha(p) for p in (root/"source").rglob("*") if p.is_file()}
    for path in (binary,root/"smoke_config.json",root/"full_config.json",root/"balanced_fit_validation.json"):
        files[str(path.relative_to(root))]=sha(path)
    save(root/"manifest.json",{"schema":"goal_extended_frozen_v1","compile":command,"files":files})
    print("Prepared",root,flush=True)


def finish(root,profile):
    prefix=root/profile
    summary=load(str(prefix)+"_summary.json")
    missions=list(rows(str(prefix)+"_missions.jsonl"))
    evaluations=list(rows(str(prefix)+"_evaluations.jsonl"))
    phases=list(rows(str(prefix)+"_phases.jsonl"))
    training=[m for m in missions if m["learning"]]
    frozen=[m for m in missions if not m["learning"]]
    expected=48 if profile=="smoke" else 2000
    assert summary["complete"] and summary["episodes"]==summary["credits"]==len(training)==expected
    assert len(frozen)==sum(e["cases"] for e in evaluations)==(16 if profile=="smoke" else 276)
    assert all(m["updates_after"]-m["updates_before"]==512 for m in training)
    assert all(m["hints"]==0 and m["updates_after"]==m["updates_before"] and m["model_after"]==m["model_before"] for m in frozen)
    assert all(m["hints"]==0 for m in training if m["arm"]=="autonomous")
    fits=[p for p in phases if p["event"]=="balanced_fit"]
    assert sum(p["fits"] for p in fits)==summary["direct_fits"]<=800000
    assert summary["updates"]==expected*512+summary["direct_fits"]<=1824000
    if profile=="full":
        assert summary["family_counts"]==[300,600,600,100,100,100,100,100]
        assert [e["checkpoint"] for e in evaluations]==["before","after400","after800","after1200","after1600","after2000"]
        assert all(p["fits"]>0 for p in fits if p["after_episode"]<1600)
    for e in evaluations:
        group=[m for m in frozen if m["arm"]=="eval_"+e["checkpoint"]]
        assert len(group)==e["cases"] and sum(m["success"] for m in group)==e["successes"]
    verify(root)
    evidence={p.name:{"sha256":sha(p),"bytes":p.stat().st_size} for p in root.glob(profile+"_*") if p.is_file()}
    result={"scope":"Declared run and consistency checks, not independent causal attribution or complete-learner claim.",
            "configuration":load(str(prefix)+"_config.json"),"summary":summary,"evaluations":evaluations,
            "fits":fits,"mission_count":len(missions),"artifacts":evidence,"manifest_sha256":sha(root/"manifest.json")}
    save(root/(profile+"_completed.json"),result)
    if profile=="full":save("docs/GOAL_EXTENDED_BOOTSTRAP_RESULTS.json",result)
    return result


def run(root,profile):
    verify(root)
    if profile=="full":assert load(root/"smoke_completed.json")["summary"]["complete"]
    status=root/(profile+"_status.json")
    assert not status.exists(), "refusing to restart an existing worker"
    started=time.time()
    save(status,{"state":"starting","worker_pid":os.getpid(),"started_unix":started})
    try:
        with (root/(profile+"_run.log")).open("w") as log:
            process=subprocess.Popen([str(root/"goal_extended_bootstrap"),"--run",profile,str(root/profile)],stdout=log,stderr=subprocess.STDOUT)
            save(status,{"state":"running","worker_pid":os.getpid(),"trainer_pid":process.pid,"started_unix":started})
            code=process.wait()
        if code:raise RuntimeError("trainer exit code "+str(code))
        result=finish(root,profile)
        save(status,{"state":"complete","started_unix":started,"finished_unix":time.time(),"summary":result["summary"],"evaluations":result["evaluations"]})
        print(json.dumps(result["summary"]),flush=True)
    except BaseException as error:
        save(status,{"state":"failed","started_unix":started,"finished_unix":time.time(),"error":str(error)})
        raise


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=("prepare","smoke","worker"))
    parser.add_argument("--root",type=Path,default=Path("logs/goal_extended_bootstrap_20260923_v1"))
    args=parser.parse_args()
    if args.mode=="prepare":prepare(args.root)
    else:run(args.root,"smoke" if args.mode=="smoke" else "full")
