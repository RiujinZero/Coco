# Incremental grounding from mixed questions — 2026-09-22

Coco's experimental native reader can extend its learned location model with
possession, transfers, counts and membership from raw stories and verified final
answers. It then answers withheld compositions using shared phrase meanings.
This is a bounded symbolic grounding experiment with pretrained primitive
policies, not a full curriculum training run or unrestricted language learner.

## Inputs and mechanism

Each seed first learns the original five movement frames and location question
from 900 bAbI training examples, as in the bootstrap experiment. The new learner
receives 1,840 additional QA lessons: 1,680 controlled examples and 160 unchanged
current-possession retrieval examples selected from the rebuilt WSL Stage 5
training split. It sees no annotated facts, event roles, relation labels or
question programs. Primitive micro-policies were already trained from action
demonstrations and remain frozen throughout language learning and evaluation.

`incremental_grounding.h` constrains novel sentence meanings and atomic question
procedures together, anchored by the prior learned model. Different questions
must agree with one interpretation of each sentence frame. Search enumerates
generic relation bindings and add/remove/replace effects. It retains supported
alternatives and refuses publication on unresolved question meanings, conflicting
evidence, incomplete search or a changed prior model. Projection discards some
correlations, so failure to resolve a meaning can be conservative.

The prior location channel is deliberately swapped on seed 1. New possession
meanings must align with the remaining channel without altering the prior
location procedures. Direction reversal on a previously unused relation is an
observational symmetry: reversing edges, replacement scopes and reads gives the
same answers. The learner chooses an explicit orientation convention on one
novel frame. This convention is not evidence that it inferred an inherently
correct direction for a relation.

The runtime now permits inverse follow/count operations by reversing the fact
view passed to the existing learned primitive policies. Partial-observation
reads require complete column coverage, just as forward reads require row
coverage. Learned event patterns bind argument positions, allowing repeated names
to share the same value without creating a separate reflexive sentence frame.

After publication, eleven question forms and nine event frames are available.
Ten question forms and four event frames are new. Shared phrase rules are derived
from the learned procedures. Actual answers are emitted through the native
Organism output distribution; the Python interpreter is used only for scoring.
Generated replies cannot change facts. Grammar, procedures, event meanings and
primitive weights remain frozen during evaluation.

## Results

All three seeds pass the following **1,945 distinct evaluation cases per seed**
with exact answers and complete world state:

| Evaluation | Cases per seed | Correct answer and state |
|---|---:|---:|
| Possession retrieval with new entity names | 240 | 240 |
| Possession counts with new entity names | 240 | 240 |
| Prior location questions in mixed worlds | 120 | 120 |
| Count possessions at a computed location | 60 | 60 |
| Compare two computed possession counts | 60 | 60 |
| Compare two computed locations | 60 | 60 |
| Compare a nested count with another count | 60 | 60 |
| Selected original v2 held-out transfer questions | 105 | 105 |
| Original bAbI location retention | 1,000 | 1,000 |

That is 5,835 successful executions of the same 1,945 cases across three seeds,
not 5,835 independent test problems. These refreshed counts use the latest Stage
5 revision (1,543 held-out questions). Earlier snapshots yielded 114/1,954 and
110/1,950 selected-transfer/total cases. Complete composed question forms are absent
from training. Each comparison family includes both yes and no targets; composed
counts include zero, one and two. The independent auditor replays every training
and evaluation story, verifies targets and complete states, checks the exact
native input exports, verifies the selected v2 records against their source
splits, and confirms no exact QA overlap between those selected training and
test records.

Disabling shared phrase meanings reduces composed-question coverage from
240/240 to **0/240 per seed**, while all atomic answers and complete states remain
correct. This separates composition from memorizing whole question forms.

Address/undefined-behavior sanitizer contracts pass for inverse reads, inverse
procedure discovery, column coverage, refusal on ambiguity and contradictions,
insufficient search, stale prior publication, ignored self-generated/missing
feedback and retention. The varied native probe also exercises repeated-name
transfer bindings. The existing bootstrap,
semantic, event, reference and answer-grounding suites and their independent
auditors pass. The separate partial-observation contracts also pass.

## Curriculum implication and boundaries

This supports teaching several questions about the same small, changing world,
with contrasts that distinguish candidate meanings. For example, identity
questions about actors alone could not distinguish identity from inverse
location traversal. Adding room referents supplied the missing contrast.
Teach atomic meanings and paraphrases in this setting, then reserve complete
combinations for evaluation. More repetitions of an indistinguishing example
would not provide the missing information.

The refreshed v2 result covers **105 of 1,543 Stage 5 held-out questions**. It selects
explicit-name, fully observed declarations/successful transfers with query
referents present in the story. One otherwise eligible question had an unbound
query name and was skipped. Boolean evaluation targets are explicitly
normalized from true/false to yes/no. This run does not evaluate unknown holders,
observation gaps, failed transfers, corrections, pronouns, temporal questions,
dialogue actions or TextWorld. The 160 v2 training records supplement a larger
controlled curriculum; this is not evidence that v2 alone teaches every tested
meaning.

The lexical heuristic uses single-token referents and question variation.
Inference supports two relation channels, at most four event argument positions
and two question arguments. Follow is the existing chain-following primitive,
not a general set-valued or uniqueness-aware relation query. No new online
correction ledger or automatic curriculum ingestion is provided here.
`ENVIRONMENT` role ingestion is now covered by the separate
[dialogue experiment](DIALOGUE_GROUNDING_EXPERIMENT.md). Learning partial-observation
language remains unfinished. See
[CURRICULUM_V2_VALIDATION.md](CURRICULUM_V2_VALIDATION.md) for remaining dataset
issues; no WSL dataset or builder was modified.

## Reproduction

Under WSL Ubuntu, from the repository root:

```bash
bash scripts/incremental_grounding.sh
python3 python/audit_curriculum_v2.py
```

The script expects primitive demonstrations in
`logs/reasoning_composition_verified/`, the preceding bootstrap corpus/results in
`logs/bootstrap_grounding/`, and v2 at `/home/user/curriculum_v2/`.
Native model/trace files, preparation counts and the independent audit with
source/artifact hashes are in `logs/incremental_grounding/`.
