# Rebuilt curriculum validation — 2026-09-22, latest Stage 5 revision

Read-only audit of `/home/user/curriculum_v2` in WSL Ubuntu. No dataset or builder
files were changed. Results and source hashes are in
`logs/curriculum_grounding/v2_audit.json`.

The independently checked current-possession labels have **zero contradictions**:
60,581 training and 1,171 held-out queries were scored. Another 7,775 training
and 163 test stories were skipped because their pronoun/wording interpretation
was outside this checker; temporal questions were counted separately. The checker
uses the builder's declared convention that `tried to ...` and `offered ...`
denote unsuccessful transfers. It does not infer that convention from ordinary
English or claim that the learner has acquired it.

The Stage 3 changes also pass: all 450 held-out worlds have three paraphrases and
consistent answers; none appears in Stage 2 or Stage 4 training by complete scene
hash. All 3,000 Stage 3 training worlds have multiple paraphrases. The canonical
grounded-dialogue test preserves 1,669 `ENVIRONMENT` turns separately from user,
assistant, and context turns. Native role ingestion is now tested separately in
[DIALOGUE_GROUNDING_EXPERIMENT.md](DIALOGUE_GROUNDING_EXPERIMENT.md).

## Latest temporal audit

The latest rebuild fixes both the earlier **71 unmentioned-object leaks** and
the subsequently recorded **1,980 training / 29 held-out temporal witnesses**.
The current files contain 78,457 training and 1,543 held-out questions. SHA-256
prefixes are `6b8b8811da98` (train) and `f89b8e87975a` (test). The independent
audit now reports **zero temporal witness errors**:

| Witness | Training | Held-out |
|---|---:|---:|
| Wrong predecessor among 7,090 / 139 scored predecessor questions | 0 | 0 |
| An `always had` answer contradicted by another observed holder | 0 | 0 |

It scores explicit-name predecessor questions only when the queried object has
no observation gap and the named holder is current. A successful transfer
identifies its giver immediately before its receiver; a failed attempt does not
create a new ownership tenure. It also detects direct counterexamples to
`always had` claims: another holder is explicitly observed. It skips 1,921
training and 46 test predecessor questions with relevant gaps, and does not
claim a complete temporal-language audit. Zero detected contradictions is not
proof that every temporal label is correct or sufficient.

The preceding revision contained this training error:

```text
Ivan has the box.
Ivan handed the box to Bill.
Bill tried to give the box to Greta.
Who had the box before Bill?
Target: Bill
```

The correct predecessor is Ivan. The old `event_transfer()` appended the holder
after failed attempts, so `history[-2]` could mistake a repeated observation for
an ownership change. The rebuilt history deduplicates consecutive holders.

The previous held-out `always` counterexample included `Alice passed the phone to Sandra.`
and asks `Who has always had the phone?`, with target Sandra. Other lines concern
other objects. Alice explicitly held the phone first. Because the initial holder
was hidden, the history stored only the receiver, and its single entry was
misread as evidence of “always”. The new builder records the visible giver too.

**Semantics to preserve:** separate observations from ownership changes; do not
append a new tenure for a failed attempt. Record both the visible giver and
receiver of a successful transfer, including when the initial holder was hidden.
Require visible initial ownership, uninterrupted relevant coverage and no
ownership change before producing an `always` target. Do not bridge observation
gaps by treating an older observed holder as the immediate predecessor.

The current (now empty) witness export is saved in
`logs/curriculum_grounding/v2_temporal_issues.jsonl`, including split, zero-based
question index, original story, target and witness.

## Matched visibility control remains unresolved

`make_visibility_pair()` also calls `event_gap()` on the second world, which may
silently change its hidden holder. The visible known/unknown labels remain valid,
but the helper does not guarantee the advertised *same hidden world* control.
Suppress hidden changes for that matched control, or label it as a different
control condition. No file-level hidden-state mismatch count is claimed because
the paired export does not contain those hidden states.

Run `python3 python/audit_curriculum_v2.py` under WSL to reproduce the checks.
The evaluator is not used to answer questions inside the native learner.

## Bounded learner validation

The incremental grounding experiment uses 160 selected original v2 training
examples alongside 1,680 controlled QA lessons and the preceding bAbI bootstrap.
The refreshed probe passes 105/105 selected Stage 5 held-out questions and
1,945/1,945 total evaluation cases per seed, with correct complete state, on
three seeds. The earlier 114/1,954 and 110/1,950 counts belong to preceding dataset
revisions. This covers explicit, fully observed possession
stories; it is not a full Stage 5 result. See
[INCREMENTAL_GROUNDING_EXPERIMENT.md](INCREMENTAL_GROUNDING_EXPERIMENT.md) for
selection, composition/retention tests, ablation and remaining learner limits.

The later [observation-language experiment](OBSERVATION_LANGUAGE_EXPERIMENT.md)
expands this to 169/169 original held-out questions per seed, including 42
histories with observation gaps and eight unknown targets. It learns visibility
and coverage meanings from a separate controlled QA curriculum and tests the
original v2 wording without teaching feedback. Its full evaluation contains
19,273 questions per seed, with complete knowledge-state audits. This still
excludes temporal questions, other unsupported story constructions and unmentioned
query referents; it is not a full Stage 5 result. Source files remain unchanged.
