"""Show the full text of a few rows of a {"text": ...} JSONL (to find where questions and answers sit).
    python3 python/peek_text_tail.py FILE [rows...]
"""
import sys, json
want = set(int(x) for x in sys.argv[2:]) or {0, 1}
with open(sys.argv[1], encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i in want:
            t = json.loads(line)['text']; print(f'===== row {i} ({len(t)} chars)'); print(t); print()
        if i > max(want): break
