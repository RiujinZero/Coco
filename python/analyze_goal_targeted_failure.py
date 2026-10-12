"""Read-only explanation of the completed targeted isolation probes."""
from collections import Counter, defaultdict
import json
from pathlib import Path

from analyze_goal_policy_fit import scores
from audit_goal_continuing_smoke import public_demonstration

root = Path("logs/goal_targeted_isolation_v2")
pilot = Path("logs/goal_continuing_bootstrap_pilot_20260923_v1")


def rows(path):
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


policies = {label: {"hidden":1316,"actions":7,"weights":list(map(float,(root/(label+".weights")).read_text().split()))}
            for label in ("warm","final")}
groups = defaultdict(Counter)
first = {}
for row in rows(pilot/"pilot_trace.jsonl"):
    if row["event"] != "decision" or not row["mission"].startswith(("eval_warm_","eval_autonomous_")):
        continue
    phase,index = row["mission"][5:].rsplit("_m",1)
    if int(index)>=32 or int(index)%8 not in (1,2):
        continue
    o,menu = row["observation"],row["menu"]
    expected = public_demonstration(o,menu)
    key = phase+"_family"+str(int(index)%8)
    groups[key]["operation_"+str(row["operation"])]+=1
    groups[key]["teacher_operation_"+str(expected)]+=1
    groups[key]["teacher_agreement"]+=expected==row["operation"]
    if row["operation"]!=expected and row["mission"] not in first:
        values=scores(policies["warm" if phase=="warm" else "final"],o)
        choice=max(menu,key=lambda action:values[action])
        assert choice==row["operation"], "saved neural values disagree at first divergence"
        first[row["mission"]]={"clock":row["clock"],"neural":choice,"teacher":expected,
            "values":values,"observation":o,"inspected":row["inspected_before"],"status":row["status"]}

result = {"scope":"Read-only actual pilot trace/first-divergence diagnosis; no new training.",
          "actual_probe_action_counts":groups,"first_teacher_divergence":first}
(root/"failure_analysis.json").write_text(json.dumps(result,indent=2)+"\n")
Path("docs/GOAL_TARGETED_FAILURE_ANALYSIS.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(groups,indent=2))
print(json.dumps({k:{s:v[s] for s in ("clock","neural","teacher","inspected","status")} for k,v in first.items()},indent=2))
