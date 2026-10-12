# Learning request meanings during interaction — 2026-09-22

Coco's optional grounded reader now acquires request meanings and actuator effects
while acting in one continuing world. Request/effect memories start empty. New
wording and actuator IDs arrive during the run; the same learner keeps earlier
meanings and question answering. This closes the previous separation between
offline request teaching and frozen action evaluation for this controlled domain.

The observation vocabulary and primitive policies are still pretrained. The
exploration/controller code is supplied, not learned. This result does not
establish a complete learner, general chat, or autonomous long-horizon reasoning.

The subsequent [live observation experiment](LIVE_OBSERVATION_GROUNDING_EXPERIMENT.md)
also learns four unfamiliar report meanings during interaction, using verified
consequence answers. The protocol and results below describe the earlier run
with its observation vocabulary frozen.

## Live interface and feedback

`online_grounding.h` binds each issued action to its request, episode, previous
observation, relation channels and observation-model version. A unique feedback
token admits one subsequent trusted environment report. Duplicate, stale,
self-generated, unmatched or insufficiently observed transitions cannot update
the meanings. A valid transition publishes its request/effect updates together.
The caller must serialize the action and associated environment report; the
protocol does not infer causal provenance from arbitrary concurrent messages.

An outstanding action marks current state as awaiting observation. The reader
withholds current-state answers and further actions until an environment report
arrives. Generated claims cannot clear that condition. A replacement request does
not turn the unobserved outcome into known state. Frozen operation also tracks
issued actions and reports, while leaving learned meanings unchanged.

Each teaching request receives a verified satisfaction bit on arrival and after
each attempted action. An additional actuator acknowledgment reports whether the
action executed. These are explicit supervision, not inferred rewards. The learner
sees raw requests and observation text, its own grounded memory, and the full
executable action menu. It receives no correct action, goal atom, task-family
label, plan or hidden world state. Training ends a request when the external
feedback establishes satisfaction, or after 32 attempts; early completion is
therefore teacher-confirmed, not always independently recognized by the learner.

The environment retains state for all 160 requests. Four windows of 40 requests
introduce, in order, a transfer request, a location request, a differently ordered
ownership request, and a negative ownership request. Earlier forms remain in the
mix. Two new opaque actuator IDs replace the original two in the final window;
their effects must be acquired from interaction. The old IDs remain in memory and
are tested again afterward. Relation channel and actuator-name assignments vary
across three seeds.

The primitive policies, previously grounded event/question meanings and the
neural byte model stay frozen. Only request candidates and actuator-effect
candidates learn in this experiment. Their updates use symbolic constraint
elimination; this path uses no end-to-end backpropagation.

## Exploration and the ambiguity test

The supplied controller first tries to select an action that satisfies every
retained goal/effect interpretation. It explores unfamiliar actuator schemas to
acquire effects. To distinguish request meanings, it can choose an action whose
predicted outcomes divide the remaining goal hypotheses into agreeing/disagreeing
groups. It favors the most balanced guaranteed split over effect alternatives.
Tie choices and fallback exploration are sampled.

A first version explored only when it lacked an action that could satisfy all
meanings. It completed requests but retained an incorrect alternative on one
seed: “the original holder lost the object” alongside “the requested recipient
holds it.” Both recommend the same successful transfer, so ordinary exploitation
never supplies distinguishing feedback. The final controller can seek that
contrast even when a successful action is already available. A sanitizer contract
explicitly verifies selection of a wrong-recipient experiment and subsequent
elimination of the unsupported meaning.

Exploration is enabled for the controlled training environment. It can temporarily
move the world away from the requested outcome to learn its meaning. This is an
experimental teaching policy, not a general action policy for arbitrary external
systems. The action menu supplies valid preconditions; those preconditions are
not learned here.

## Results

The independent auditor replays every physical action and observation, verifies
every feedback bit and complete state, checks the full affordance menu, and
requires complete trace coverage. It also checks the exact learned request/effect
rules at every checkpoint.

| Policy | Seed 0 | Seed 1 | Seed 2 |
|---|---:|---:|---:|
| Information-seeking: requests completed | 160 / 160 | 160 / 160 | 160 / 160 |
| Information-seeking: attempted actions | 209 | 209 | 209 |
| Uniform fallback: requests completed | 159 / 160 | 159 / 160 | 158 / 160 |
| Uniform fallback: attempted actions | 278 | 269 | 265 |

Information-seeking runs each encounter 30 refused attempts. Uniform runs encounter
40, 39 and 38, respectively. The fixed refusal schedule rejects every seventh
attempt. The four uniform-run request failures exhausted the 32-attempt limit;
they remain in the results. Both policies eventually acquire the correct four
request meanings and four actuator IDs.

These are small descriptive trials, not evidence of a globally optimal exploration
policy or curriculum. Both arms use the same starting world and family schedule,
but exploration changes subsequent state and hence some later request bindings.
The individual tasks are not perfectly matched between arms. Three seeds also do
not turn repeated checkpoint templates into independent language problems.

Frozen copies are evaluated before interaction and after each window, using new
entity names, new world histories and shuffled action menus. Evaluation provides
no teaching feedback and never changes the live learner. Per seed and policy:

| Checkpoint | Completed requests using introduced wording |
|---|---:|
| Before interaction | 0 / 48; all wording is untaught |
| After transfer wording | 12 / 12 |
| After location wording | 24 / 24 |
| After ownership paraphrase | 36 / 36 |
| Final, both old and new actuator interfaces | 96 / 96 |

The 120 checkpoint requests using not-yet-taught wording receive no action.
At every checkpoint, the acquired meanings reduce to the intended single
predicate/effect; successful behavior alone is not the criterion. All **320
interleaved possession/location answers** per run remain correct, with the full
world state independently verified. No wrong definite satisfaction predictions
occur before feedback; early uncertainty is retained in the trace.

## Validation and remaining scope

Address/undefined-behavior sanitizer contracts cover feedback timing, ignored
generated input, duplicate tokens, new requests/episodes, multiple or wrong-source
observations, inconsistent failure acknowledgments, atomic update rejection,
pending-outcome uncertainty, frozen operation, and meaning-seeking action choice.
The preceding request and environment-dialogue suites were rebuilt and their
sanitizer tests, three-seed native evaluations and independent audits pass.

The language model remains a bounded frame learner over atomic state predicates.
The experiment does not learn observation vocabulary online, learn the exploration
policy, infer actuator preconditions, handle partially observed planning, recover
from noisy teaching, acquire event-history/conjunctive goals, or discover multi-step
plans. Integrating this interaction API with the standard byte trainer and the
broader curriculum remains unfinished. Symbolic memory remains part of the model.

Run `bash scripts/online_grounding.sh` under WSL after the bootstrap, incremental
and dialogue corpora have been prepared. The script builds the native probe and
sanitizer contracts, runs both policies on three seeds, and invokes the independent
auditor. Every request, action, feedback bit, checkpoint, model signature, state
and source hash is recorded in `logs/online_grounding/`.
