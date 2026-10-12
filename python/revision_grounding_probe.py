"""Explicit report/evidence revision stream. No oracle states enter native input."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "logs/revision_grounding"
FIELDS = ["op", "report", "evidence", "revision", "source", "text", "question", "answer", "active"]


def curriculum():
    operations = []
    def add(op, report, evidence=0, revision=0, source="sensor", text="", question="", answer="", active=1):
        operations.append(dict(zip(FIELDS, (op, report, evidence, revision, source, text, question, answer, active))))
    def teach(report, obj, answer):
        add("feedback", report, source="teacher", question=f"Who has the revisionitem{obj} now?", answer=f"revisionperson{answer}")
    # A plausible wrong answer narrows the first report to the wrong meanings.
    add("report", 0, text="The revisionitem0 was logged beside revisionperson2.")
    teach(0, 0, 0)
    add("report", 1, text="revisionperson2 moved to the revisionroom0.")
    add("report", 2, text="The revisionitem1 was logged beside revisionperson3.")
    teach(2, 1, 3)  # Contradiction with the earlier wrong teaching answer.
    add("revise_feedback", 0, source="teacher", answer="revisionperson2")
    # Four independent report IDs support general admission. The original
    # corrected scene remains one scene, regardless of its revision count.
    add("report", 3, text="The revisionitem2 was logged beside revisionperson1.")
    teach(3, 2, 1)
    add("report", 4, text="The revisionitem0 was logged beside revisionperson3.")
    teach(4, 0, 3)
    # A later known event must remain later after a correction of earlier text.
    add("report", 5, text="revisionperson1 has the revisionitem0.")
    add("revise_report", 4, text="The revisionitem0 was logged beside revisionperson2.")
    # The old answer is now contradictory, and requires its own identified fix.
    add("revise_feedback", 4, source="teacher", answer="revisionperson2")
    add("revise_report", 1, text="revisionperson2 moved to the revisionroom1.")
    add("revise_report", 5, active=0)
    add("revise_report", 5, revision=1, text="revisionperson1 has the revisionitem0.")
    add("revise_feedback", 0, revision=1, source="teacher", active=0)
    add("revise_feedback", 0, revision=2, source="teacher", answer="revisionperson2")
    # Learn a second meaning under partial observation. Queries about other
    # people rule out operations that remove coverage for the whole relation.
    for i in range(4):
        add("report", 10 + 2*i, text=f"The location record for revisionperson{i} went dark.")
        for j in range(4):
            # Earlier gaps remain open. This is public evidence, not a hidden
            # true location masquerading as an observed target.
            place = 1 if j == 2 else j
            add("feedback", 10 + 2*i, evidence=j, source="teacher",
                question=f"Where is revisionperson{j}?",
                answer="unknown" if j <= i else f"revisionroom{place}")
        # A nonunique location also withholds a unique answer. This contrast
        # rejects spurious self-location additions that mimic an unknown read.
        add("feedback", 10 + 2*i, evidence=4, source="teacher",
            question=f"Is revisionperson{i} at revisionperson{i}?", answer="unknown")
    # Independent later teaching answers still constrain the earlier history;
    # withdrawing their premise can conflict until the report is restored.
    add("revise_report", 10, active=0)
    add("report", 20, text="revisionperson0 moved to the revisionroom3.")
    add("revise_report", 10, revision=1, text="The location record for revisionperson0 went dark.")
    # Adding a false answer makes the whole journal inconsistent. Withdrawal
    # removes that constraint and restores both language and current knowledge.
    add("feedback", 20, evidence=99, source="teacher", question="Where is revisionperson0?", answer="revisionroom1")
    add("revise_feedback", 20, evidence=99, source="teacher", active=0)
    # A second bad report is repaired while a distinct later event is retained.
    add("report", 21, text="The revisionitem2 was logged beside revisionperson2.")
    add("report", 22, text="revisionperson1 moved to the revisionroom0.")
    add("revise_report", 21, text="The revisionitem2 was logged beside revisionperson0.")
    return operations


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    operations = curriculum()
    with (OUT / "train.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(operations)
    (OUT / "protocol.json").write_text(json.dumps(dict(operations=len(operations),
        initial_world="revision", feedback_source="teacher", report_source="sensor",
        correction_target="explicit report/evidence ID and prior revision", supplied_states=False), indent=2) + "\n")
    print(f"Wrote {len(operations)} report/feedback operations; no structured state or rule labels.")


if __name__ == "__main__":
    main()
