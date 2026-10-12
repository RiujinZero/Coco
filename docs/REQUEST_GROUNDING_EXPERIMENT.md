# Grounded requests and executable actions — 2026-09-22

The optional native reader now learns a small request vocabulary from verified
satisfaction feedback, learns actuator effects from observed consequences, and
selects an executable action whose predicted effect satisfies the request. It
checks completion from a subsequent environment observation. Requests and
predictions never directly change factual memory.

This is a one-step symbolic controller using already grounded observations. It
adds a working request/action connection; it does not establish autonomous
long-horizon planning or a complete continually learning model.

## Learning signals and boundaries

`request_grounding.h` keeps alternatives for an atomic relational goal: relation,
two argument positions and positive/negative membership. The initial candidate
space and membership operation are supplied primitives. The learner receives a
raw request, its own grounded observed state, and an explicitly verified bit
indicating whether the request is satisfied. It receives no goal atom, semantic
role labels, correct action or plan. Entity segmentation initially uses names
already established by the reader; an acquired frame can bind new names.

The 496 teaching records comprise:

- 118 original request/outcome pairs from the first 32 Stage 6 training dialogues;
- 118 teacher-generated wrong-recipient outcomes for those same requests;
- 236 examples teaching a second request construction and its contrasts;
- 24 examples asking an actor to reach a location on the prior relation.

Each record supplies raw observation text and a satisfaction bit. Additional
known movement statements establish referents and constrain the unrelated
relation. These are supplementary diagnostic lessons, not changes to the user's
curriculum. Teacher-side targets are independently re-derived from the text.

After all 118 ordinary pairs, two meanings remain: the requested recipient holds
the object, or the original giver no longer holds it. A wrong-recipient outcome
distinguishes them. Candidate meanings remain separate until feedback resolves
them; an unsupported choice is not silently committed. Contradictory verified
evidence revokes a meaning, and unknown observations cannot count as negative
feedback. Generated text cannot supply teaching credit.

Two opaque actuator IDs expose typed argument lists through a caller-owned
interface. Their successful effects are inferred by trying generic relation
updates against the reader's before/after observations. Uniformly sampled local
exploration supplies 128 executed attempts per seed: 109 acknowledged executions
and 19 refused attempts. A refused attempt is not evidence for the successful
effect. The interface's execution acknowledgment is distinct from the goal's
satisfaction feedback. IDs are swapped between seeds, along with the preexisting
relation channel convention.

The environment supplies all currently executable actions, independently of the
request. That menu is an affordance interface, not learned preconditions. A
bounded enumerator predicts each candidate effect and accepts an action only
when every retained request/effect meaning agrees that it achieves the goal.
Predictions are separate snapshots; only observed reports update the real reader.
The harness retries a refused action, with a maximum of three attempts. The retry
rule, search strategy, source envelopes and execution interface are supplied.

## Native evaluation

All text passes through the actual Organism byte path and the attached reader.
Requests are captured only from user messages in the opt-in `grounded_requests`
mode. Actuation uses the reader's structured `choose_action()` interface, not
free-form generated action prose. The environment simulator is separate from
the controller and exposes no target, hidden state or correct action to it.

Per seed, the independent evaluator checks:

| Check | Correct / checked |
|---|---:|
| Request satisfaction before original environment reports | 1,669 / 1,669 |
| Request satisfaction after original environment reports | 1,669 / 1,669 |
| Original dialogue question answers, before target exposure | 674 / 674 |
| Completed requests in new interactive worlds | 576 / 576 |
| Forced failures recognized before retry | 116 / 116 |
| Earlier possession/location answers during interaction | 1,152 / 1,152 |
| Wrong-recipient semantic contrasts | 192 / 192 |

The same cases pass on three seeds. Repeating them with different local policy
initialization, channel conventions and menu order does not make them independent
test examples. Address/undefined-behavior sanitizer contracts and the refreshed
incremental/dialogue regressions pass as well.

The 576 requests run in 48 worlds with 12 successive requests each, interleaving
two possession request constructions and location requests. Test entities are
absent from request and actuator training. Worlds persist between requests;
available action order is shuffled. Failed attempts leave goals unsatisfied;
observed success completes them. Every state is checked against independent
physical replay, and all retained interpretation worlds must agree.

Removing request meanings prevents all 576 selections. Removing actuator effects
also prevents them. Predictions and assistant claims do not complete goals in
the absence of observations. The learner without wrong-recipient feedback remains
uncertain on all 192 counterfactual outcomes; the constrained learner correctly
rejects them. Native unit contracts additionally cover insufficient observation,
search-budget exhaustion, contradictory feedback, ignored generated feedback,
new user questions clearing pending requests, and episode reset.

Evaluation freezes request/effect memories, previous question/event meanings and
primitive weights. Request learning is incremental during teaching, and acting
is continuous within test worlds; this is not yet simultaneous open-ended
learning and acting. The prior observation learner still uses the earlier
bootstrap, mixed-question and dialogue lessons. Primitive policies were trained
from demonstrations. Current request/effect induction uses symbolic constraint
elimination and no end-to-end backpropagation.

The subsequent [online experiment](ONLINE_GROUNDING_EXPERIMENT.md) starts request
and actuator memories empty and updates them during interaction in one continuing
world. It retains the pretrained observation meanings and supplied exploration
policy; broader joint learning remains unfinished.

## Remaining requirements

The goal language represents one desired state predicate. In particular, the
transfer request's learned success condition does not enforce a history in which
the named giver personally performs a transfer. Goals requiring particular event
histories, conjunctions, temporal order, costs or permissions need richer meaning
primitives and distinguishing feedback.

Planning currently requires a complete, agreeing observed world and one-step
executable actions. Partial observations cause abstention; this probe does not
teach visibility language, information gathering, long-horizon credit assignment,
action preconditions or general chat. Generalized exploration and recovery from
incorrect feedback remain open. Integration into the standard online trainer
is also unfinished. Symbolic memory remains part of the execution architecture.

Reproduce under WSL with `bash scripts/request_grounding.sh`, after the bootstrap,
incremental and dialogue preparation runs. The script builds native and sanitizer
executables, evaluates three seeds, and runs the independent auditor. Detailed
feedback, canonical/request traces, actual actions and full states, controls,
model hypotheses and hashes are in `logs/request_grounding/`.
