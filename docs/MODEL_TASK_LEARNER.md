# The model as the task learner — 23 September 2026

This integrates the whole system as the user's diagram describes: **the model
(bodies, routing, persistent workspace, sequence memory and agency kernel) is the
learner that performs the tasks; the existing subsystem supplies teaching, memory
and tools.** It follows [the continuing learner handoff](CONTINUING_LEARNER_HANDOFF.md).
The 2,000-episode run described in [GOAL_EXTENDED_BOOTSTRAP.md](GOAL_EXTENDED_BOOTSTRAP.md)
remains the *subsystem baseline*; it is not a substitute for this experiment.

```
requests, observations, action results ──► body population + persistent workspace ◄── agency kernel
          ▲                                          │                (attention, commitment)
          │                                          ▼                         ▲
   reader, memory and                     small decision readout               │
   computation tools ◄── retrieve / reason / ask / act / manage goals / finish │
          │                                                                    │
          └──────────► verified teaching and outcome credit ──► bodies ────────┘
```

## What is new, file by file

| Piece | File | Role |
|---|---|---|
| Task interface | `cpp/include/laminar/model_task.h` (`TaskModel`) | Separates **observing** (`observe`), **deciding** (`decide`), **acting** (`act`) and **receiving feedback** (`tool_outcome`, `mission_outcome`) on one persistent model |
| Decision readout | same | 7 × 97 linear map from body activity to the seven operations. Not a reasoning network: 679 weights |
| Body task credit | same + `organism.cpp` (`apply_l6_error`, `modulated_on`) | Demonstrations and delayed outcomes reach the bodies that produced each decision |
| Task-level agency | `organism.cpp` agency wiring, `agency.h` (`consequence(..., reported)`), `config.h` (`agency_task`) | Third goal channel with public task outcomes; eight task-state observation slots |
| External decision path | `relational_control.h` / `goal_control.h` (`choose(..., decided)`) | The model's choice executes; the controller's own policy is not evaluated; provenance in `ControlStep::decider` |
| Checkpoint | `cpp/include/laminar/state_io.h`, `Organism::state_io`, `TaskModel::state_io`, `state_io` on address tree, evidence, sequence memory and every agency structure | Complete learning state with a trailing hash; hash-only fingerprint for verification |
| Frozen-body control | `Organism::freeze_weights` | No body synapse changes (readout, routing utilities and agency still adapt) |
| Driver | `cpp/apps/model_task_learner.cpp` | Curriculum, frozen tools, teacher taper, probes, ablations, checkpoint/resume |
| Tests | `cpp/tests/test_model_task.cpp`, `scripts/model_task_resume_test.sh` | Integration contracts and the verified resume test |

### 1. The bodies get the actual task experience

Everything the task produces enters the one model as bytes, framed as
`0x1E <channel> payload \n`. Channels: request `R`, public observation `O`,
goal arrival `G` (request text, deadline, visible world), evidence `E` (read or
executed world), tool result `T` (status, reply, rejection), executed action `A`,
**demonstration `D`**, **generated proposal `P`**, outcome `V`, decision `?`.
Every frame is predicted byte by byte through the model's own local rules, so
prediction of observed events and tool results trains representations.

The distinctions the curriculum audit required are preserved: a teacher
demonstration arrives on its own channel and is never relabelled as the model's
own action; a controller-generated proposal (derived goal text) is processed with
learning and structure switched off, so generated content is never learned as
evidence; verified outcomes arrive only on the outcome channel.

### 2. Decisions depend on body computation

`decide(menu)` processes a decision frame (the legal operations) through the
bodies. The readout sees exactly three views of body output: the decision event's
workspace, the persistent workspace, and the workspace pooled over this step's
observation and decision frames. Each is normalised by a running magnitude,
learned only while training. It returns a distribution over the legal
operations; the controller executes the chosen one through the new external
decision path, and **its 1,316-unit (here 32-unit) policy is never evaluated**.
The seven operations in this controller take no arguments (activation applies to
the inspected goal, `next_goal` advances inspection), so none are needed yet.

In autonomous training the model samples from 0.9·π + 0.1·uniform. Probes use
the argmax with no teacher. Structured observations are still supplied: the
controller's 32 public floats, rendered as bytes. They are assistance. Whether
bodies can build that state themselves is a later step (see *Shortcuts* below).

The first smoke showed why pooling is needed: the persistent workspace halves
every byte (`ws_decay=128`), so by the decision event almost nothing of the
observation frame remained. Readout decisions then tracked the most recent
teacher label, one step behind. A non-normalised learning step made the same
failure overconfident (0.99). The fixes are the pooled view and an NLMS step
(`lr / (1 + |x|²)`).

### 3. The agency kernel gets task-level consequences

With `agency_task=1` the kernel has a third goal ("complete the public task")
and a fifth consequence channel. Its observation slots 56–63 carry public goal
state: agenda phase, active goal, uninspected arrival, deadline slack of the
active and inspected goals, the last tool rejection, the last tool/goal outcome
value, and the readout's own confidence. Tool outcomes (rejected 0, ordinary 0.5,
goal closed correct 1 / incorrect 0) and the verified mission result (1/0) are
posted as task consequences and delivered with the next event. Events without
task news report *nothing* on that channel: the kernel treats silence as its own
prediction, neither success nor failure. The trace lifetime is 32 events, so a
consequence reaches the focus choices made while the decision was formed.

**Division of labour.** The kernel never selects an operation. The readout never
moves attention. The kernel's commitments set the focus gain over the bodies'
input (episodic or generic stream, workspace, recent bytes) and how long that
focus persists. The bodies compute candidate behaviour, and the readout
translates it. Both are credited by the same verified task outcomes, and the
kernel observes the readout's confidence. That makes them one loop, not two
competing controllers.

### 4. The bodies themselves are trained

Hidden learning is **on** (`hidden=modulated`, `acc_bits=16`), with episodic
births (`episodic=1`, `hist=7`). The four declared signals:

| Signal | Implementation |
|---|---|
| Prediction of observed events and tool results | Ordinary byte prediction of every learned frame: L6 local error, modulated hidden layers, lock structure |
| Demonstrated decisions | Readout cross-entropy; the **exact workspace-space error** of that loss goes into the L6 layer of every body active in the pooled events and the decision event (`apply_l6_error`); a three-factor update of their hidden layers modulated by the demonstration's log-likelihood gain over its running mean (`modulated_on`) |
| Verified mission outcomes | For every decision the model itself executed: the same two updates weighted by the discounted advantage `γ^(T-1-t)(R - b)` (policy gradient); the running baseline covers only missions in which the model acted |
| Credit to participating bodies and routes | The same advantage lowers or raises the resistance of the emitting bodies, moves their lock utility, and moves the routed-emission utility of routed bodies |

No gradient is propagated between body layers. Backpropagation into bodies is
**not** provided; it would be a separately declared option.

### 5. Existing skills as an assistance scaffold

The reader, relational memory/computation library and goal controller are built
once with the continuing bootstrap's declared setup: raw fixture, reader, long
extension, initial and extension skills. The extension is applied upfront,
because the tools are frozen before the model trains. Then `set_learning(false)`,
and they are **copied fresh for every mission** (the reader cloned per mission).
So all learning that persists across missions is the model's. The teacher
(`continuing_curriculum_v1`, 32 public floats and the menu only) is consulted
with probability 1 for the guided phase, tapering linearly to 0. Autonomous
training and every probe have no teacher.

Successful tool use shows **learned coordination**, not that bodies learned
counting, planning or reading. Testing that stronger claim means removing
shortcuts one at a time, for example the 32-float summary (bodies must build
state from arrivals and tool results), and then particular tools. The
interfaces allow this; it has not been done.

### 6. Recoverable and interpretable long training

The checkpoint covers the bodies' INT4 weights, the INT16 plasticity residuals,
unit state (resistance, activity, association locks, L4 returns), routing and
recirculation utilities, the lock/address tree and evidence pool, the byte
readout, workspace and persistent workspace, stream history, sequence memory
(instance window and consolidated tier), the full agency kernel (expansion,
consequence routes, sensory predictor, signals, eligibility traces, DAN,
drive motif, competition pool and its RNG, commitment, focus), goal-credit
state, the model RNG, the task readout and its normalisers, the reward
baseline, the task RNG and the run counters. The episodic credit history spans
one mission and is empty at every checkpoint boundary. The checkpoint refuses to
save configurations whose extensions it does not cover (composer levels, chain,
relation or bubble memory, answer attention, attached reader). The file ends
with a hash; a corrupt or truncated file is refused. Writes go to a temporary
file and are renamed, keeping the previous checkpoint.

**Verified resume test** (`scripts/model_task_resume_test.sh`, resume profile:
12 missions, checkpoints every 3, probes at 6 and 12). Three runs: (A)
uninterrupted; (B) stopped after the mission-6 checkpoint, then resumed; (C)
killed with SIGKILL partway through, then resumed. All three end with the
identical model fingerprint `8a2f285316a4dd30` and byte-identical mission,
decision, evaluation and final logs. The first attempt found a real bug:
probe output after a checkpoint was truncated on resume. The checkpoint is now
rewritten after every evaluation. Resuming 12 missions took 36 s; at full scale
the cost is one checkpoint interval (20 missions) plus a load.

**Probes** start every mission from the checkpointed model, frozen (no
learning, no structural change) and without a teacher. Nothing a probe does
persists: the live model is reloaded afterwards. Every probe set is run twice:
normally, and with **body output ablated** (the readout receives zero
workspace features). Telemetry per mission covers live units, exact locks, task
L6 and modulated updates, credited bodies and routes, agency commits, route
updates and emissions per tendency, the reward baseline and bytes processed.

## Scenarios

| Scenario | Bodies | Task credit into bodies | Agency | Teacher | Pretraining |
|---|---|---|---|---|---|
| `learned` | learn | yes | learned, task channel | hand rule | none |
| `readout_only` | learn byte prediction only | no (readout only) | learned | hand rule | none |
| `frozen_bodies` | allocated, **no synapse changes** | no | learned | hand rule | none |
| `fixed_agency` | learn | yes | fixed focus (`agency=2`) | hand rule | none |
| `teacher_subsystem` | learn | yes | learned | trained subsystem policy (weights at mission 1,200) | none |
| `pretrained` | learn | yes | learned | hand rule | 32 MB curriculum corpus |
| `pretrained_frozen` | pretrained, then frozen | no | learned | hand rule | 32 MB curriculum corpus |

The hand rule is `continuing_curriculum_v1`. The subsystem policy teacher sees exactly the same inputs: the
32 public floats and the legal menu. The pretraining corpus (`make_pretrain.py`, manifest with SHA-256 per
slice) takes training splits only, in stage order and cut at episode boundaries: bAbI stages 1–2, CLEVR
stages 2 and 4, paraphrase, events, TextWorld and grounded dialogue. Pretraining is ordinary byte
prediction with every body learning rule and structure on, before any task. `pretrained_frozen` freezes the
bodies afterwards.

**The saved subsystem policy is a complete teacher over the frozen tools.**
`subsystem_teacher_probe` loads the exported controller weights at mission 1,200 of the 2,000-episode run
(`warm_m1199`) into a controller built on the model run's frozen tools, and runs the same 46 probes with
that policy deciding. Score: **45/46** (stopping 18/18, interrupted six-edge 4/4, long 3/4, every other
family 4/4), identical to its score in its own run. The relational knowledge that run accumulated during
training (model version 114 → 5,208, against 153 for the frozen tools) turned out not to matter to its
policy.

## Subsystem checkpoint and checkpointed rerun

The 2,000-episode subsystem run was killed at mission ~1,380 (see *Incidents*). It had no resumable
state. The subsystem can now be checkpointed completely: the goal controller (policy, target policy,
credits, empirical values, updates, agenda, mission history, computation receipts), its relational
knowledge (concept memory, language, transition and execution receipts, frames, primitive store, composed
learner, procedures, actuator, RNG) and the reader (grounded session, language reasoner, journal, archive).

`state_io.h` became a generic archive to make that tractable. One `io()` dispatch handles an explicit
`state_io` member, standard containers (string, vector, array, map, set, unordered, deque, list, optional,
pair, tuple, variant) and owned smart pointers (saved by value). It falls back to raw bytes for trivially
copyable types and to member-by-member reflection (structured bindings) for plain aggregates. It refuses
raw pointers and `std::function` at compile time. Only 16 classes needed an explicit `state_io`: those
with private members, constructors, base classes or C-array members. Four gained a default constructor
as a restore target (`ProcedureMemory`, `LanguageReasoner`, `GroundedArchive`, `GroundedSession`); every
field is overwritten on load.

Verification:

- `cpp/tests/test_subsystem_checkpoint.cpp` trains a controller through 6 real missions (stopping family,
  exploration, tapered hints, semantic learning, terminal credit). It saves it, restores it into a
  controller built with a *different seed*, and drives both through 4 more learning missions. Result:
  identical fingerprints, byte-identical decision logs, policy weights, update counts and knowledge
  versions. A mutation check that drops one member (the empirical value table) makes the test fail, so
  it genuinely detects gaps. It passes under MSVC and GCC.
- `cpp/apps/subsystem_run.cpp` is the declared extended run, unchanged (its schedule and declaration now
  live in `goal_extended_declaration.h`, shared with the original program), plus a checkpoint every 20
  missions and a final state file. `scripts/subsystem_resume_test.sh` compares an uninterrupted run, a
  stop/resume and a SIGKILL/resume on the smoke profile: all deterministic logs and the final state
  fingerprint are identical. It reproduces the original smoke record, 0/8 → 6/8.

## Incidents (23 September 2026)

1. **Host disk exhaustion crashed WSL twice.** The first full-scale launch ran four 200k-unit arms, each
   writing ~3 GB checkpoints (current, previous and temporary copies). WSL's virtual disk lives on C:,
   which reached 0 bytes free. The WSL VM failed (`Wsl/Service/E_UNEXPECTED`); later ext4 hit I/O errors.
   **The first crash killed the running 2,000-episode subsystem baseline at mission ~1,380.** Its logs up
   to that point are intact (frozen probes 0 → 39 → 44 → **45/46** at 0/400/800/1,200 missions); it had no
   resumable state. The rerun above replaces it.
2. **Fixes:** checkpoints store all-zero 4 KiB blocks as one tag byte. At 16k units that cut 362 MB to
   48 MB, and the saving grows with capacity. Checkpoints are fsynced before the rename. A damaged current
   checkpoint falls back to the previous one (tested by truncating it: byte-identical logs). An evaluation
   interrupted by a crash is redone on resume (tested by aborting inside one). The user freed the host disk
   and the virtual disk was compacted (624 → 376 GB).
3. **The first frozen-bodies arm was invalid:** freezing also switched off structure, leaving zero live
   bodies. The arm now keeps allocation (lock births, random or imprinted birth weights) and blocks only
   synaptic change.

## Smoke results (48 missions, 8 probes, 1 seed)

| Arm | Probes after 48 | Body output ablated | Guided / taper agreement (multi-option menus) | Taper successes |
|---|---|---|---|---|
| learned | **2/8** | 0/8 | 65% / 80% | 12/20 |
| readout_only | 0/8 | 0/8 | 58% / 70% | 9/20 |
| frozen_bodies (structure on) | 1/8 | 0/8 | 64% / 82% | 8/20 |
| fixed_agency | 0/8 | 0/8 | 64% / 83% | 9/20 |

Every probe success disappears when body output is ablated. The differences between arms are within what 8
probes and 48 missions can resolve, so they are not evidence for or against the hypothesis; the
full-scale run is the test.

## Results (full scale, 23–24 September 2026; one seed)

200k units, 2,000 missions per scenario: missions 0–800 fully guided, 800–1,600 guidance tapering linearly to
zero, 1,600–2,000 autonomous. 46 frozen probes (no learning, no teacher) at each checkpoint, repeated with body
output ablated. **Every ablated probe set at every checkpoint of every scenario scored 0/46**: all probe
successes depend on the bodies' workspace.

| Scenario | 400 | 800 | 1,200 | 1,600 | 2,000 |
|---|---|---|---|---|---|
| **learned** | 32 | **42** | 41 | 4 | 24 |
| frozen_bodies | 32 | 30 | 23 | 29 | 5 |
| readout_only | 25 | 4 | 30 | 16 | 0 |
| fixed_agency | 27 | 24 | 17 | 10 | 0 |
| teacher_subsystem | 36 | 35 | 42 | 27 | 23 |
| pretrained | 0 | 39 | 29 | 0 | 0 |
| pretrained_frozen | 26 | 29 | 29 | **34** | 25 |
| learned_critic | 32 | 42 | 35 | 12 | 0 |
| learned_critic_rehearsal | 38 | 35 | 33 | 28 | 0 |
| learned_rehearsal | 38 | 35 | 41 | 1 | 17 |
| learned_no_outcome | 32 | 42 | 31 | 18 | **17** |
| *subsystem (1,316-unit policy), rerun* | *39* | *44* | *45* | *34* | *32* |

The subsystem rerun matches the original, killed run exactly at every checkpoint it reached (39/44/45). Past the
original run's crash point it shows something new: **the subsystem also declines once its guidance tapers
(45 → 34 at 1,600, 32 at 2,000)**. The decline without a teacher is partly a property of this taper-and-outcome-credit
protocol, not only of the model's bodies, although the model declines more steeply.

**What the model learned as a unit (guided phase).** With demonstrations, the model reaches **42/46 at mission
800**, deciding every operation from body activity through a 679-weight readout. That includes all four
interrupted six-edge probes, which the earlier controller pilot never solved. The subsystem's 1,316-unit policy
was at 44/46 at the same point. The controls separate the contributions:

- **Training the bodies matters.** Learned vs frozen bodies: 42 vs 30 at 800. The cleaner measure is agreement
  with the teacher on multi-option decisions during missions 600–800, where every arm sees identical missions
  executed by the teacher: learned **0.93**, frozen **0.85**, readout-only (bodies learn byte prediction but get
  no task credit) **0.74**. The readout is identical across arms; only what the bodies learn differs. Bodies
  that learn only byte prediction do worse than frozen ones, consistent with the readout chasing features that
  drift for task-unrelated reasons.
- **Learned agency matters.** Learned vs fixed attention: 42 vs 24 at 800.
- **A better teacher did not raise the ceiling.** The trained subsystem policy as teacher reached 42 at 1,200,
  the same peak as the hand rule. Its reader family was 0/4 until mission 1,200.
- **Pretraining the bodies** on the 32 MB corpus (0.46 bits/byte) reached 39 at 800, including 4/4 long routes.
  It also had a transient failure at 400 (0/46, finishing goals before reasoning) and collapsed in the taper
  like the rest.

**What failed: keeping the behaviour once the teacher leaves.** Every scenario declines as guidance tapers, and
every scenario with outcome credit collapses by the autonomous phase. The ablations locate the cause:

- The decline starts **without outcome credit**: `learned_no_outcome` falls 42 → 31 → 18 through the taper.
  As the model acts on its own, what it learns from its own trajectories (byte prediction, demonstration credit
  on states it visits) moves the body features away from what the readout was fitted to.
- **Frozen features do not collapse in the taper.** `pretrained_frozen` (bodies pretrained on the corpus, then
  frozen) is the only scenario that *improves* through the taper (29 → 34 at 1,600); `frozen_bodies` also holds
  (23 → 29). With no body plasticity the readout keeps tracking a fixed feature space, which is what the drift
  diagnosis predicts. Task-trained bodies are better while the teacher is present (39 vs 29 at 800 with
  pretraining, 42 vs 30 without), but their features keep moving afterwards.
- **Outcome credit, as implemented, is destructive without a teacher.** With it, every scenario reaches 0–5 by
  2,000 (plain `learned` recovered to 24). Without it, the model holds at 17–18 through the whole autonomous
  phase. A terminal ±1 over 100+ decisions per mission is too sparse and noisy a signal at these rates.
- A **per-state critic did not help** (critic alone: 0 at 2,000), and **feature rehearsal only delayed** the
  collapse (critic + rehearsal 28 at 1,600, 0 at 2,000; rehearsal alone 41 at 1,200, then 1 at 1,600).
  Rehearsal replays the features stored when an example was demonstrated, so it cannot track drifting features.
- The inspection tool (`model_task_inspect`) shows body features stay diverse after the collapse: different
  observations still produce clearly different workspaces. The collapsed readout instead carries large biases,
  making its decision state-independent.

**Interpretation.** The whole model can learn the task as a unit from demonstrations: bodies, agency and a small
readout, with each part shown to contribute. The unsolved problem is continual learning without a teacher. It
needs a way to keep the readout aligned with features that keep learning (for example, rehearsal that
re-encodes stored situations through the current bodies rather than replaying old features), and an outcome
signal that is much better conditioned, or much weaker, than per-decision REINFORCE on a terminal reward.

### Subsystem rerun artifacts

The checkpointed rerun completed all 2,000 missions (1,227 guided and 229 autonomous training successes,
595,000 direct fits, 1,619,000 updates). Its complete final state (`~/subsystem_runs/full/full_final.state`,
693 MB) loads into a fresh controller and reader and reproduces the recorded fingerprint `11835a04afa882f6`
(`cpp/apps/subsystem_state_check.cpp`). The runner keeps only its latest two checkpoints, so the best state (45/46
at mission 1,200) survives only as exported policy weights (`warm_m1199`). Keeping milestone checkpoints is a
one-line change for future runs.


### Why the short continuing pilot did not collapse

The earlier pilot (`logs/goal_continuing_bootstrap_pilot_20260923_v1`: 400 guided/tapering missions, then 200
autonomous) scored 0 → 34 → 34/46 and never declined. `build-task/compare_runs.py` and `compare_runs2.py`
compare it with the extended subsystem run:

- **It had nothing fragile to lose.** Its 34/46 is `[14,0,0,4,4,4,4,4]`: stopping plus the five single-skill
  families, with interrupted six-edge and long routes at 0/4. The extended run falls back to *exactly* this
  profile at mission 1,600 (34/46, identical per family). Both runs keep the same stable core, and only the
  extended run ever held the hard skills (45/46 at 1,200).
- **That core is sustained by outcome credit alone.** In the pilot's autonomous phase, the five single-skill
  families succeed 125/125 and stopping 21/25, so the 512 reward updates per mission keep reinforcing them.
  Six-edge (2/25) and long (0/25) almost never succeed.
- **The hard skills lived on supervised fitting.** The pilot's demonstration fitting hit its cap and stopped
  at mission 250, before the hard skills formed (warm probes: six-edge and long loop to the 256-decision
  limit). The extended run kept fitting, tapering from 10,000 to 250 fits per 20 missions and reaching zero at
  1,600. Its hard probes pass greedily at 1,200 (in 65–192 decisions) and all loop to the 256 limit at 1,600
  and 2,000.
- **The per-decision taper hid the dependence.** The teacher is consulted at each decision with probability
  p. Hard missions take 60–150 decisions, so even at p ≈ 0.05 **every** hard training mission between 800
  and 1,600 received at least one hint (360/360). Their 70% training success during the taper was therefore
  never unassisted. The first unassisted hard missions come at 1,600, where success falls to 74/240 (with
  exploration; the greedy probes loop).

So the subsystem also cannot keep the hard skills on reward alone. Both runs converge on the same
reward-stable floor. The model differs in falling *below* that floor, because its input features drift while
the subsystem's 32 public floats stay fixed. Two implications: continued rehearsal of demonstrations (the
re-encoding experiment) targets the actual mechanism that kept the hard skills, and a taper that withdraws
the teacher for whole missions would expose hard-skill dependence while guidance is still available.

## Re-encoding rehearsal (24 September 2026)

Stored-feature rehearsal cannot track drifting bodies, so the next step stores demonstrated *situations*: the
last 128 bytes up to and including the decision frame, which of those bytes were observations, the menu and
the teacher's label. After each learning mission, 32 of them are replayed through the **current** model with
learning, structure and sequence-memory capture off (a new `Organism::replay` flag skips capture only). The
readout is then fitted to today's features; `--reencode-bodies` also sends the same L6 error into the replayed
bodies. Only multi-option teacher demonstrations are stored.

| Scenario (32 situations / mission) | 400 | 800 | 1,200 | 1,600 | 2,000 |
|---|---|---|---|---|---|
| reencode | 42 | 43 | 41 | 31 | 0 |
| reencode_bodies | 40 | 42 | 42 | 34 | 25 |
| reencode_no_outcome | 42 | 43 | 43 | 40 | 13 |
| **reencode_bodies_no_outcome** | 40 | 42 | 41 | **43** | 5 |
| *learned (no rehearsal)* | 32 | 42 | 41 | 4 | 24 |

- Re-encoding **speeds acquisition** (40–42 at mission 400, against 32 for every earlier variant) and
  **carries the hard skills through the whole taper** without outcome credit. `reencode_bodies_no_outcome` is at
  43/46 at mission 1,600 (six-edge 4/4, long 4/4), where the subsystem is at 34.
- With outcome credit on, the scenarios fall back to the stable floor by 1,600 (31–34).
- **Every scenario then decays in the teacherless phase** (5–25 at 2,000). Training success erodes gradually
  (about 60% → 30%) rather than all at once, with few decision-limit loops.

**Why re-encoding alone is not enough.** At mission 2,000 the readout still agrees with the teacher on
**91%** of re-encoded stored situations (`model_task_inspect`), yet live probes score 5/46. Two measurements
explain the gap:

- A 128-byte situation is **ambiguous**. For the exact observation and menu where the model now loops, the
  teacher chose `interpret` 182 times and `next_goal` 158 times. Earlier mission history decides which.
- Replay runs with **stale agency state**. Switching agency focus off during replay flips the readout's
  choice in about a quarter of those situations. A live mission builds its task-state slots and commitments
  step by step, while a snippet replay inherits whatever the previous mission left.

While live demonstrations continued (through mission 1,600) they kept the readout aligned with live features.
After that, it drifted toward the replay distribution.

**Next: whole-mission replay** (`--replay-missions K`). Every demonstrated learning mission is recorded as a
compact tape: bytes with pooled flags, plus marks for task-state updates, tool outcomes and decisions with
their menus and teacher labels. A uniform reservoir keeps 256 tapes over the whole run. After each learning
mission, K tapes are replayed start to end through the current model, restoring task state and tool outcomes
at their original positions. Every demonstrated decision is thus re-encoded in its full mission context, with
the agency kernel evolving as it would live.

## Whole-mission replay: the model keeps its skills without a teacher (24 September 2026)

`~/model_task_v5`, 200k units, 2,000 missions, one seed, the same curriculum and probes as every run above.
Body-output ablation scores 0/46 at every checkpoint of every scenario.

| Scenario | 400 | 800 | 1,200 | 1,600 | 2,000 | Autonomous training success, 1,600–2,000 |
|---|---|---|---|---|---|---|
| **replay 2 tapes, readout + bodies, no outcome credit** | 36 | 43 | 39 | 44 | **45** | 292/400 |
| **replay 1 tape, readout + bodies, no outcome credit** | 43 | **46** | 45 | 42 | **43** | 312/400 |
| replay 1 tape, readout only, no outcome credit | 35 | 43 | 43 | 43 | **42** | |
| replay 1 tape, readout + bodies, outcome credit | 43 | 46 | 40 | 38 | 15 | 149/400 |
| *snippet re-encoding, readout + bodies, no outcome* | 40 | 42 | 41 | 43 | 5 | 165/400 |
| *learned (no rehearsal)* | 32 | 42 | 41 | 4 | 24 | |
| *subsystem, 1,316-unit policy* | 39 | 44 | 45 | 34 | 32 | 229/400 |

- **The collapse is solved.** After 400 missions with no teacher, the model scores **43–45/46** in frozen
  probes, hard families included (six-edge 4/4; long 3–4/4). Live autonomous success holds at 74–80% per
  100 missions (with 10% exploration), where snippet re-encoding fell from 68% to 31%.
- **It learns faster and higher than the subsystem.** 46/46 at mission 800 with one tape per mission; the
  subsystem's best was 45/46 at 1,200. At mission 2,000 the model is at 45, the subsystem at 32.
- **Full context was the missing piece.** Snippets gave the readout 91% agreement on replay but not on live
  missions; tapes reproduce the mission history and the agency kernel's state, so what is rehearsed is what
  is used.
- **Outcome credit remains destructive** (15/46 at 2,000 with it on). The per-decision REINFORCE on a
  terminal ±1 should stay off, or be redesigned, before it is used again.
- Replay into the bodies as well as the readout gives the best early learning (43 vs 35 at mission 400);
  both hold equally at 2,000. Two tapes per mission learn more slowly at first but end highest.

**Caveats.** One seed per scenario, and these are the same 46 developmental probes used throughout (probe
entities are disjoint from training, but the families and topologies are the curriculum's). The rehearsal
material is the teacher's own demonstrations from the guided phases, kept in the model's memory; the
autonomous phase adds no new labels. Cost: one or two frozen mission replays per learning mission,
roughly +60–100% wall time at full scale.

## Held-out skills and success replay (24 September 2026)

`~/model_task_v6`, `~/model_task_v7`. One family is **withheld from the teacher for the whole run**: no
demonstrations and no labels in its missions, only the environment's verified outcome. Every scenario uses the
whole-mission replay configuration above (one teacher tape per mission, readout + bodies, no outcome credit).
*Success replay* (`--success-replay K`) keeps verified successes as tapes labelled with the model's own
choices and replays them like teacher tapes; `--success-budget B` limits each success tape to B replays.
Body-output ablation is 0/46 at all 36 checkpoints.

| Withheld family | Scenario | Withheld-family training success (per 200 missions) | Withheld probes 1,200 / 1,600 / 2,000 | Total probes at 2,000 |
|---|---|---|---|---|
| long routes | control | 7% → 27% → 38% (135/600 overall) | 1/4, 0/4, **3/4** | **45/46** |
| long routes | success replay | 2% → 25% → 15% (74/600) | 0/4, 2/4, 1/4 | 41/46 |
| long routes | success replay, budget 8 | 7% → 25% → 15% (86/600) | 3/4, **4/4**, 2/4 | 43/46 |
| union query | control | 90–100% throughout | 4/4, 4/4, 4/4 | 43/46 |
| union query | success replay | 80–100% | 4/4, 4/4, 4/4 | 42/46 |
| union query | success replay, budget 8 | 80–100% | 4/4, 4/4, 4/4 | 39/46 |

- **The model acquires withheld skills without any teacher in them.** Union query is solved from the first
  checkpoint by transfer from the taught families. Long routes, the hardest family, rise from 7% to 38%
  training success with no demonstrations, and the control ends at 45/46 overall. Withholding long routes
  from the teacher entirely cost nothing by mission 2,000. Nothing in the control trains on the withheld
  family's outcome, so this is transfer: the operations and subgoal handling taught on other families
  (six-edge shares the goal machinery) carry over.
- **Success replay did not help.** Unbudgeted, it slowed acquisition (2% for the first 600 missions) because a
  tiny store is rehearsed after every mission. The budget removed that slowdown, and its probes peaked
  earlier (4/4 long routes at 1,600), but withheld-family training success stayed below the control.
  With 4 probes per family and one seed, the probe differences are within noise; the training-success
  counts favour the control.
- **Likely reasons.** (1) The success store fills with successes from *taught* families during the taper
  (about 700 stored, only ~80 from long routes), so replay mostly repeats what teacher tapes already cover.
  (2) The model's own labels include exploratory detours, not only the moves that mattered.
  Candidate fixes: store only missions the model completed without any teacher action, weight replay
  toward families or situations with little teacher coverage, and keep only choices on the path to the
  outcome (for example, drop choices undone later in the mission).

### Replication and improved success replay (`~/model_task_v8`)

| Long routes withheld | Withheld-family training successes (of 600) | Long probes 1,600 / 2,000 | Total at 2,000 |
|---|---|---|---|
| control, seed 1 | 135 | 0/4, 3/4 | 45 |
| **control, seed 2** | **228** (12% → 53%) | 3/4, 3/4 | 42 |
| success replay: solo + loop pruning + coverage weighting, budget 8 | 120 | 2/4, 2/4 | 42 |
| success replay: solo + loop pruning, budget 8 | 107 | 3/4, 1/4 | 42 |

- **Transfer to the withheld family replicates** with a second seed and is stronger there.
- **The success-replay fixes did not help**; both variants stay below both controls. A plausible cause is that
  the model's own successful routes differ from the teacher's, and rehearsing both pulls the readout two ways.

## Grouped attempts (`--attempts K`)

For outcome credit without a teacher: every teacherless training mission is first attempted K times with the
model frozen (no learning, no structure, no sequence capture) and sampling on. Missions are exactly reproducible,
since the tools are copied fresh and the environment index is fixed. `TaskModel::finish_attempts`:
- stores each successful attempt as a success tape (with loop pruning);
- compares each failed attempt with a success. At the first decision where both were in the same state but chose
  differently, the failure's tape is labelled with the successful choice (a correction in the failure's own
  context) and every other label is cleared;
- stores nothing from groups with no success (nothing to contrast).

Nothing is learned directly; all credit flows through tape replay. Unit-tested in `test_model_task`: a success
is stored, both diverging failures are corrected, and all-fail or all-success groups behave as specified.
Running: `~/model_task_v10/withhold_long_attempts4`.

## Answer channel (`cpp/apps/model_answer_learner.cpp`, `answer_tasks.h`)

The model **writes** answers. `TaskModel::generate` produces bytes from the model's own next-byte distribution
(greedy, or sampled at a temperature) until a newline, frozen. Episodes are read in their curriculum_v2 text
form (story, `USER: question`, `ASSISTANT:`), so the format matches pretraining. An exact-match verifier checks
each answer.

- **Sources, in 4 waves of 10,000 episodes:** bAbI stages 1–2 · CLEVR, events, paraphrase counting · SCAN, COGS
  · ProPara. Probes: 12 per task from each source's **test** split, frozen and greedy (per-task scores logged).
- **Teacher:** the reference answer is read with learning (teacher forcing) with probability 1 in the first
  half of a source's introduction wave, tapered to 0 by its end, and never again. Each skill is taught once and
  must then be kept up without a teacher. `--withhold events` never teaches that source at all.
- **Credit without a teacher, all through re-reading with learning:**
  - `--replay R` re-reads stored teacher episodes (text rehearsal, the forgetting guard);
  - `--attempts K` samples K answers per teacherless episode and stores verified correct ones with the model's
    own answer as rehearsal (`--success-replay`, `--success-budget`);
  - the verdict also reaches the agency kernel as a task consequence.
- **Correction found while building it:** `Organism::replay` (no sequence-memory capture) must not be used
  when bytes are *predicted*. Sequence memory follows its own recorded stream, so without capture its
  predictions go stale; held-out text scored 8–21 bits per byte with it and 1.16–1.26 without. The goal
  learner's replays are frozen and use only body workspace (which never reads sequence memory), so they are
  unaffected. The answer channel captures everything it reads or writes.
- Verified: smoke run (first correct written answers on bAbI) and an exact kill/resume.

Running: `~/model_answer_v1`, three arms: text replay, text replay + grouped attempts, and no replay (forgetting
baseline). Events is withheld in all three.

## Results: forgetting, grouped attempts, answer channel (24 September 2026)

### Goal tasks
| Run | Probes 400 / 800 / 1,200 / 1,600 / 2,000 | Held-out text bits/byte, initial → 2,000 |
|---|---|---|
| best configuration, seed 1 | 43 / 46 / 45 / 42 / 43 | – |
| best configuration, seed 2 | 40 / 40 / 43 / 32 / 40 | 1.285 → 1.608 |
| + 8 KiB curriculum text per mission (`mix8k`) | 41 / 41 / 39 / 32 / **24** | 1.285 → **0.961** |
| pretrained (32 MB), no mixing | 37 / 42 / 41 / 29 / **21** | 1.165 → **1.282** (peak 1.317) |
| pretrained + mixing | 31 / 40 / 40 / 30 / 27 | 1.165 → 0.971 |
| long withheld, grouped attempts (4 per mission) | 37 / 42 / 42 / 42 / 31 | withheld successes **105/600** (controls 135, 208) |

- The best configuration **replicates** with a second seed (40/46 at 2,000; a dip at 1,600).
- **Forgetting runs both ways.**
  - Mixing text into task training erodes task skills in the teacherless phase (41 → 24), because one teacher
    tape per mission does not balance 8 KiB of text.
  - Task training erodes text: the pretrained model loses 0.12–0.15 bits/byte of its pretraining. Even from
    scratch, task-only training pushes held-out text from 1.29 to about 1.6.
  - Mixing improves held-out text substantially (1.29 → 0.96).
  - Pretraining without mixing also hurt task retention (21 at 2,000).
- **Grouped attempts did not help** (105 withheld successes against 135 and 208 in the controls; 946
  corrections and 3,524 attempts). Every form of self-generated rehearsal tried so far (success replay, its
  fixes, grouped attempts) has been neutral or harmful.

### Answer channel (`~/model_answer_v1`, 40,000 episodes, 300 probes)
| Arm | Best total | At 40,000 | Teacherless live accuracy, wave 3 (bAbI 1 / bAbI 2 / CLEVR) |
|---|---|---|---|
| text replay | 46 | 36 | 20% / 28% / 14% |
| text replay + grouped attempts | 43 | 30 | 12% / 30% / 9% |
| no replay | 40 | 14 | 5% / 14% / 3% |

- **Text replay prevents forgetting across waves** (36 vs 14 at the end; no-replay decays to 5–14).
- **Absolute accuracy is poor:** at best about 15%. bAbI yes/no tasks sit near chance, where-questions at 25%,
  events (never taught) 0/72, SCAN and COGS (long outputs) 0.
- **Written answers are not grounded in the story.** Greedy answers to test items: "kitchen" for most
  where-questions, "no" to a CLEVR material question and to "who has the box", "football" for a material. The
  format is right; the content comes from answer-word frequency, not from the story just read.
- **Attempts on answers added nothing:** 5,550 sampled correct answers were stored, but on yes/no and
  few-choice questions a correct sample is often a lucky guess, so self-imitation rehearses noise.

### Diagnosis
1. **No working memory of what was read.** The persistent workspace halves every byte (`ws_decay`), so the story
   has decayed by the time the question arrives. Only sequence memory and exact-key episodic bodies retain
   context, and neither binds "Where is Mary?" to "Mary moved to the bathroom." Goal tasks sidestep this
   because the public observation carries the state. Text does not. This is also why generated text lacks
   coherence beyond a sentence.
2. **Rehearsal budget is not proportional to what competes with it** (text volume, new sources).
3. **Outcome credit needs informative successes:** weight a success by how unlikely it was (1 − the group's
   success rate), require consistency across samples for few-choice answers, and keep self-labels from
   contradicting teacher tapes.

## Round 2 fixes (24 September 2026)

1. **Reading: the project's own learned within-episode reading (`seq_chain=2`).** `ARCHITECTURE.md` records a
   reading mechanism built earlier: relations between thoughts inside the current episode, credited from the
   stream's own `ASSISTANT:` turns. With nothing hand-written it reached bAbI qa1 100% and qa2 81%. It was off in
   every run here. It runs inside the organism's per-byte processing, so the answer channel uses it with no new
   code, with the settings the reading runs used: `seq_chain=2 seq_topic_gain=8 seq_max_depth=8`. It could not
   be checkpointed, so `SequenceMemory::extension_state_io` now covers the dialogue (bubble) tier, the chain and
   the learned reading. The chain's per-answer sentence pointers are saved as indices; stale ones from a finished
   episode are saved as none, since the chain rebuilds them before use. Only answer attention, attached
   reasoners and the relation store still refuse to checkpoint. Kill/resume with the chain on is byte-identical.
   **Smoke (16k units, 400 bAbI episodes): 8/16 with the chain vs 3/16 without.**
2. **Proportional rehearsal.** Answer channel: `--replay-scale` rehearses R × (active sources / 2) teacher
   episodes per episode. Goal tasks: `--text-replay-kb N` replays one extra teacher tape per N KiB of mixed-in text.
3. **Informative successes.**
   - Answer channel (`--informative`): a verified answer is stored only when the live greedy answer was wrong,
     with a replay budget of B × (1 − the fraction of attempts correct).
   - Goal attempts: the same weighting on success tapes; a group where every attempt succeeded stores nothing.
   - Correction to the earlier explanation: a verified answer is byte-identical to the teacher's, so storing
     "lucky guesses" cannot have been what made attempts useless. The bottleneck was reading.

Prior work that still applies and stays on: the provenance rule (self-generated bytes never become recallable
instances; generated answers are captured as `self`), bubbling (`seq_min_support=2`), and the frozen shared
readout (`readout_freeze`). The hard "bubble" commitment stays off (it caused degenerate loops).

Running: `~/model_answer_v2` (bAbI plain / chain / chain with `ws_decay=192`; full waves with all fixes, with
and without informative attempts) and `~/model_task_v11` (text mixing with proportional replay; informative
grouped attempts).

### Round 2 results

**Reading (focused bAbI, 10,000 episodes, 96 probes):** plain 27 → **learned reading 55** (qa1 12/12, qa5 11/12,
qa8 9/12, qa4 7/12). Adding `ws_decay=192` gives the same accuracy (53) and much better held-out text
(1.36 vs 1.67 bits/byte). Yes/no, counting and paths stay weak, as the reading mechanism's own record predicts.
**Full curriculum with all fixes:** 75/300 at 40,000 episodes (round 1: 36); bAbI stage 1 holds at 31/36 through
every later wave; CLEVR reaches 21/72; SCAN and COGS remain 0. **Informative attempts** lifted the never-taught
events source to 10–13/72 (0 in every other run), the first acquisition by outcome credit. It was unstable and
lowered bAbI stage 2 (12 vs 20).
**Proportional replay under text mixing:** probes 45/46 at missions 1,600 and 2,000 (unbalanced: 24), with
held-out text 0.913 bits/byte, the best so far. **Informative grouped attempts on goals:** 101 withheld successes;
still no gain.

## General circuits: recurrence-gated snapshots and promotion

About 88% of the text-mixing population was snapshot (exact-key) bodies born on single surprises; general bodies
grew slowly (12k → 22k). Two new organism options:
- `episodic_recur=N`: a snapshot is born only when the same exact context has surprised the model N times
  (class over instance, as in the sequence memory's bubbling). A 2^20-slot tag table holds pending surprises.
- `episodic_promote=N`: a snapshot reused N times after birth opens its general parent's split evidence and holds
  it open. The parent then records every context it admits (the contrast a split needs; a snapshot alone always
  fires on one exact key). When the parent splits, the evidence-driven split has produced a general child for the
  recurring family, and the snapshot is retired.
- Cumulative accounting (`Organism::episodic_totals`): births, deferred births, evictions, evictions never reused
  after birth, promotions, parent splits after promotion, retirements. All of it is checkpointed.

Splits narrow a body's coverage by design: the parent keeps one half of its contexts and a copy takes the other.
Breadth comes from a body's own computation over the contexts its key admits, and from composition (routing,
workspace), not from wider keys. Most key bits hash the recent context, so masking bits by hand would group
contexts at random; the evidence-driven split is what finds bits that separate outcomes.

Smoke (16k units, text-heavy): the current rule churns (12,935 births, 12,055 evictions, 4,018 never reused).
Gate plus promotion (N = 3, 4) roughly doubles general bodies (3,833 → 7,616), keeps snapshots at 7%, and
promotes 1,297 snapshots, whose parents split 242 times. Probes and text are within noise at that scale.
Running at full scale: `~/model_answer_v3`, `~/model_task_v12`.
