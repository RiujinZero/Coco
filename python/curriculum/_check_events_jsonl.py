import sys, json
sys.path.insert(0, r"C:\Users\user\Documents\projects\Coco\python\curriculum")
from _check_events import check_story

total = 0
contradictions = []
for line in open(sys.argv[1], encoding="utf-8"):
    r = json.loads(line)
    q, a, sup, story = r["qa"][0]
    d = check_story(story, q, a)
    if d is not None:
        total += 1
        if d != a:
            contradictions.append((r["task"], story, q, a, d))
print(f"checked {total}, contradictions {len(contradictions)}")
for task, story, q, gold, derived in contradictions[:10]:
    print("---", task)
    print("\n".join(story))
    print(f"Q: {q} gold={gold!r} derived={derived!r}")
