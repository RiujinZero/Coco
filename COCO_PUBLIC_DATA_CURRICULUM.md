# Coco: public-data curriculum

**Version:** 1.0  
**Prepared:** September 22, 2026  
**Status:** proposed training and evaluation plan; no Coco training runs performed for this document.

This is a companion to the progression supplied in the request, not a modification of the local Windows file `COCO_CURRICULUM.md`. Source pages were reviewed; dataset archives and training adapters have not been downloaded, executed, or validated here. Sample counts, mixtures, and gates below are proposed engineering defaults, not published benchmark thresholds or capability guarantees.

## 1. Recommendation

Use a small common world curriculum rather than concatenate unrelated benchmark files. Make Dyna-bAbI and text-rendered CLEVR the early data sources; use TextWorld for action/consequence experience; use COGS and SCAN as independent meaning/composition diagnostics. Introduce grounded multi-turn dialogue through the same worlds, then SGD. Only afterward add selected OASST1 and SmolTalk examples. [S1–S6, S10–S12]

The user's reported 2,340/2,340 unseen questions/compositions and subsequent 300/300 phrase-transfer answers should remain a frozen regression suite. They establish the reported controlled result, not independently verified evidence that event grounding or unrestricted chat is solved. Do not spend the next run simply relearning the same query compositions.

**Current priority:** learn a change in a world from observations, event language, and consequences without receiving the correct post-event state or a supplied semantic program as input.

A simulator can supply exact truth for its own rules. That is useful grounding relative to a controlled world, not a claim that text-only training has solved visual perception or physical-world grounding.

## 2. Data and teacher boundary

There are three distinct products in this plan:

| Product | Purpose | Where it is allowed |
|---|---|---|
| Learner stream | Natural observations, questions, the learner's actions, and feedback already received | Training and evaluation input |
| Teaching targets | Answers, optional state labels, or later observations used as learning signals | Training only, according to the declared supervision condition |
| Oracle metadata | Full simulator state, query programs, support sets, provenance and split identifiers | Dataset generation, validation and scoring only |

The oracle may compute an answer to score Coco. It must not answer for Coco during inference. Do not hide a full semantic parser, database executor, or planner in the input adapter. Surface rendering and output normalization should not perform the reasoning being measured.

For example, a scene verbalizer may say that the red cube is left of the blue sphere. It may not add the answer to a query about the number of objects left of the sphere. Query-specific filtering of the input world is also leakage unless explicitly tested as an assisted baseline.

For a text-first run, observations may be short, independently rendered descriptions of visible facts. This still contains human-designed observation semantics. A later perceptual branch should replace those descriptions with images or sensor observations to test whether the learned relations transfer.

## 3. Stage 1 — Entities, properties, relations and argument roles

**Public material.** Use Dyna-bAbI for configurable people/object/location stories and original bAbI definitions for additional coverage. Original tasks 1, 4 and 5 respectively address a simple supporting fact, subject/object relations, and giving roles. Use CLEVR's scene annotations for object attributes and spatial relations. Dyna-bAbI's published T7/T12 mixtures do not cover all 20 original tasks. [S1–S3]

**Proposed lessons.** Begin with two to six entities and one to four relevant facts. Include a small number of unambiguous wordings immediately, not after a canonical-only phase. Alternate who, what, where, property lookup, and yes/no probes. Teach one simple update early, but reserve long event chains and corrections for Stage 5.

Example learner stream:

```text
Lena holds the key. Sam holds the cup.
Who holds the key?
```

A teaching reply may be `Lena.` A matched counterexample swaps the holders while keeping the same words. Introduce `The key is held by Lena` alongside the active construction, but never call `Lena owns the key` equivalent without actually modeling ownership.

Randomize which names occur in each role, object order, irrelevant attributes, and answer position. A recurring name must not predict its role. When adding pronouns later, do not use gender stereotypes as the identity oracle.

**Pilot and gate.** Start with 5,000 independent world episodes. Expand toward 20,000–40,000 only when fresh development tests show a coverage need. Target at least 99% exact correctness separately for entity, attribute, location and role-reversal tests using held-out names and new instances. Inspect every systematic role error even when the aggregate passes.

## 4. Stage 2 — Individual operations

**Public material.** CLEVR includes scene graphs, answers, and functional programs for training and validation. Its no-image main archive is listed as 86 MB; the CoGenT no-image archive is 106 MB. Questions cover attributes, counting, comparisons, spatial relations, and logical operations. Downloading images is unnecessary for this proposed text adaptation. [S3]

Use the relevant original bAbI task definitions as complementary coverage: 6 yes/no, 7 counting, 8 lists/sets, 18 size reasoning and 19 path finding. Do not assume the Dyna-bAbI configuration implements each of these. [S1, S2]

**Proposed lessons.** Teach retrieval, one-hop following, filtering, counting, set enumeration, numerical comparison and returning a value separately before placing them in long chains. Keep the whole relevant world visible rather than preselecting the answer-bearing facts.

Generate atomic questions from the oracle when existing questions are too composite. This is new adapter work, not a claim that the public archive already supplies the exact curriculum. Include zero counts, ties, empty sets, absent matches, reversed comparisons, and multiple distractors. Define whether a question expects an entity, set, integer, Boolean or unknown answer.

Example sequence:

```text
What does Lena hold?
How many objects does Lena hold?
Does Lena hold more objects than Sam?
```

Train the first two separately before using the third as a composition probe. An equality test must include many unequal examples and several distinct equal counts.

**Gate.** At least 99% accuracy for each atomic operation on new worlds. Do not let common retrieval questions conceal a weak count or comparison operation. Report set exact match, not just per-item accuracy. Expansion budget: 40,000–80,000 mixed-source episodes if needed.

## 5. Stage 3 — Shared meanings across wording

**Public material.** COGS provides sentence-to-logical-form examples and distinct in-distribution and generalization files. SCAN maps commands to action sequences and supplies systematic composition splits. Neither is a complete replacement for Coco's world-grounded phrase-learning task. [S4, S5]

**Proposed lessons.** Reuse the same training worlds and teacher-side operations to make small verified phrase families:

```text
How many objects does Lena hold?
What is the number of objects Lena holds?
Count the objects held by Lena.
```

Assign these a shared teacher-side operation, but do not expose that identifier to Coco at runtime. Teach a newly introduced phrase on individual questions and test it inside previously withheld composites. This preserves the structure of the user's successful phrase-transfer experiment.

Use two different generalization tests: new constructions made from already known vocabulary, and truly new words taught through a few atomic examples. Do not demand arbitrary unknown synonym meanings with zero evidence and then label the failure compositional.

A candidate paraphrase must preserve meaning across many discriminating worlds, including empty sets, conflicting roles and ties. Identical answers in one world do not establish equivalence. Review new phrase templates manually or with a trustworthy semantic check before mass generation. LLM-proposed paraphrases, when used, are untrusted candidates rather than gold labels.

Use COGS as a separate role/structure diagnostic and keep SCAN's official held-out composition splits separate from this custom data. Semantic parsing success alone is not proof of grounded world knowledge.

**Gate.** At least 98% answer correctness across held-out phrase families, plus at least 98% paired paraphrase agreement, evaluated separately so consistent wrong answers cannot pass. Keep the original reported phrase-transfer suite unchanged. Expansion budget: 20,000–40,000 atomic phrase examples, not that many independent worlds.

## 6. Stage 4 — Unseen composition

**Public material.** Use Dyna-bAbI's configurable mixtures, CLEVR programs for operation-family splits, and independent SCAN/COGS generalization tests. CoGenT specifically tests changed attribute combinations; it is not automatically a test of deeper program structure. [S1, S3–S5]

**Proposed lessons.** Combine operations already taught individually. For the user's example, the teacher-side structure could be:

```text
resolve two destinations
→ find their people
→ find each person's held objects
→ count both sets
→ compare equality
```

Those arrows describe the test construction, not information passed to Coco. First verify that the primitive operations work in the same world domain. Then withhold whole composition families from training, including their paraphrases. Do not relabel a memorized full question as a held-out composition merely because names changed.

Use separate suites for a familiar-depth novel combination and for longer/deeper compositions. Increase train depth from two to three steps; reserve four-to-six-step questions as a separate extrapolation stress test. Report them separately rather than hiding length failures in a mean score.

A cross-domain combination, such as applying a count meaning learned in object scenes to people in rooms, is an additional transfer test. It does not follow automatically from success within CLEVR.

**Gate.** Proposed target: at least 97% in every preregistered familiar-depth held-out composition family and no more than a two-percentage-point drop relative to a matched seen-family control. Report the deeper stress suite regardless of pass/fail. Expansion budget: 50,000–100,000 world-query episodes. Stop generating redundant examples when errors are structural rather than coverage-related.

## 7. Stage 5 — Events, persistence, correction and uncertainty

**Public material.** Dyna-bAbI supplies controllable event stories. TextWorld supplies an interactive game generator and observation/action/feedback interface; it requires collecting trajectories rather than merely loading a ready chat corpus. ProPara supplies natural process descriptions with entity-state annotation for a later language-transfer check. [S1, S6, S9]

LLM-BabyBench is an optional ready-made offline diagnostic, with Predict, Plan and Decompose components and a public dataset card. Predict is useful for state-transition reasoning; this is not a substitute for learning from newly experienced consequences. Split by world/level/seed metadata and task family before adaptation. [S7]

**Proposed progression.** Begin with a single successful change; then contrast it with an unsuccessful attempt. Add persistence of unrelated facts, temporal order, repeated transfers, explicit negation, reference resolution, observation gaps, and corrections. Grow from two-to-five events to ten-to-thirty events before increasing language complexity at the same time.

Examples requiring different handling:

```text
Lena gave the key to Sam.             [world change]
Lena tried to give the key to Sam,
but he did not take it.               [failed attempt]
Correction: Sam, not Lena, had the key. [revision of an earlier report]
I do not know who has the key.        [uncertainty, not a negative fact]
```

These are proposed rendered lessons. Do not assume every line is natively generated by TextWorld or unmodified Dyna-bAbI. Use bAbI-style person/object transfers and TextWorld's native object interaction events in their appropriate adapters.

Distinguish false from unknown. In partially observed worlds, the evaluator must ask what follows from Coco's observation history, not demand an unobserved hidden-world fact. Define the environment's visibility and persistence rules explicitly. A successful intent is not a successful action until feedback establishes execution.

**Gate.** On controlled synthetic evaluation, target at least 98% accuracy separately on changed and unchanged facts, at least 95% wholly correct short trajectories, and at least 98% macro-F1 across true/false/unknown queries. Include impossible actions and ambiguous references. ProPara is an additional natural-language transfer report, not a justification for imposing the same synthetic thresholds on noisy annotation.

### 7.1 The immediate event-grounding experiment

Use the current Coco checkpoint and keep the user's existing tests frozen. Start with 20,000 fresh event episodes spanning movement, pickup, drop, transfer, failed actions and irrelevant-event persistence. Split examples by base world and trajectory before deriving questions or paraphrases.

Compare these three matched conditions:

| Arm | Learner experience | Purpose |
|---|---|---|
| A: annotated baseline | Observations plus explicit state/role targets during training | Establish whether the data and output task are learnable with assistance |
| B: observations and sparse questions | Observations, events/actions, consequences, and sparse answer feedback; no supplied state labels or query programs | Test the actual unresolved event-grounding step |
| C: annotation withdrawal | Same experience, with direct state-target coverage reduced through 100%, 25%, 5%, 0% | Measure dependence on the scaffold |

In all arms, gold labels are excluded from evaluation inputs. A label may be a training target without being an input; the implementation must keep those channels separate.

At zero direct state-label coverage, later observations and sparse answer or task-success feedback still teach the model. This is **not** a claim of zero supervision. Match observable streams, answer-feedback density and compute as closely as possible across arms; disclose differences introduced by extra state-target updates.

For each training transition, observe the prior world, present the event or action, predict the consequence, then reveal the next available observation. During evaluation, score the prediction before revealing that observation. Also ask delayed questions for which the relevant post-event fact was never explicitly restated. Otherwise the experiment can collapse into copying the most recent snapshot.

Run a cold-start or newly introduced relation-family condition as well. A warm-start model retaining an explicitly taught relation after label withdrawal does not show that it can acquire a new relation without those labels.

Useful matched interventions swap the recipient, reverse two event orders, remove an action prerequisite, or change only an irrelevant property. Changed facts should follow the intervention; unaffected facts should persist. A no-update/persistence baseline and a last-mention baseline help detect easy shortcuts.

If Arm A passes and Arm B fails, do not proceed to large chat. Locate whether the failure is argument assignment, transition learning, retention, uncertainty, or choosing the answer. More generic dialogue is not a targeted test of any of those defects.

## 8. Stage 6 — Grounded dialogue, then broader chat

### 8.1 Conversationalize worlds Coco already understands

Create two-to-six-turn dialogues over existing training worlds before importing unrelated conversations. Then extend to eight-to-twelve turns. Require actual world or task consequences.

```text
User: Bring me the key.
Coco: Which key—the red one or the blue one?
User: The red one. Put it on the table instead.
Coco: [acts through the environment]
Environment: [reports the actual result]
Coco: The red key is on the table.
```

The final assertion is correct only after successful execution. Include goal changes, underspecified requests, failed actions, changed availability, delayed replies, and irrelevant conversational turns. Train brief evidence-supported explanations, not mandatory long rationalizations.

Use executable task completion and world consistency as the primary metrics. Accepted paraphrases can vary, but the claimed object, destination, action status and timing must be right.

### 8.2 Add human-worded task dialogue

SGD contains more than 20,000 conversations across 20 service domains, with schema and dialogue annotations. Its evaluation includes unseen services/domains. SGD-X supplies five wording variants per schema, not wholesale paraphrases of every dialogue. [S10]

Start with a few simple service families and reconstruct turns, service calls and results. Preserve official splits. For interactive scoring, build a mock service backend; static dialogue annotations alone do not demonstrate that Coco can complete a new task. Reveal a service result only after the corresponding call. Keep gold dialogue state out of the learner input.

Train asking for missing required values, handling corrections, preserving constraints, and reporting only verified outcomes. Simulator goals and service definitions may be visible as the declared task interface; gold next actions and hidden state may not.

### 8.3 Broaden language without abandoning evidence

Use selected English, reviewed OASST1 conversation paths. Its public release is tree-structured and contains message-level review/rank metadata; group all branches by root to prevent leakage. Start SmolTalk with source-conditioned rewrite and summarize examples rather than the entire mixture. Its newly generated components and imported components do not all share one license. [S11, S12]

Treat conversation ratings as quality signals, not factual guarantees. Audit unsupported factual claims, personal information, malformed turns, contradictory trajectories, and fabricated tool use. Prefer concise paths that require maintaining prior-turn constraints.

Optionally add SQuAD 2.0 for passage-supported answers and unanswerable questions. TinyStories can be later fluency input, but it does not supply a verified evolving world. [S14, S15]

**Proposed gate.** At least 95% task completion on the controlled short-dialogue simulator, at least 95% correct clarification decisions on a balanced ambiguous/unambiguous set, and no greater than one percentage point regression on established deterministic skill gates. Report SGD state/task metrics separately rather than equating them to the synthetic gate.

Start broad-chat examples at 10% of the total learning budget, then 25%; move toward 50% only when grounding and retention gates remain intact. Count the budget by actual bytes/tokens/learning events, not by dialogue rows: a long chat must not silently outweigh dozens of world episodes.

## 9. Sampling, replay and evaluation

### 9.1 Default learning mix

Before broad chat, a reasonable starting mix is 60% current-stage material, 30% earlier skills, and 10% training/development-derived counterexamples. Balance replay by skill family, not raw source size. At Stage 1, the earlier-skill allocation becomes additional current-stage coverage. These ratios are experimental settings to log and ablate.

Counterexamples must come from training or development failures. Never turn a sealed test failure into a training example and continue advertising that same test as untouched.

Use a 5,000-episode pilot for each new capability before approaching the stage expansion budgets. The amounts are ceilings to investigate, not a requirement to consume the whole dataset. Track both unique world groups and derived QA/phrase counts; those are different measures of experience.

### 9.2 Separate tests for separate claims

| Evaluation axis | What to withhold | What may remain familiar |
|---|---|---|
| New names | Entity vocabulary and name-role pairings | Abstract world structures and known words |
| New phrasing | Surface construction families | World rules and atomic operation meanings |
| New lexical meaning | New word first taught only in atomic examples | Previously learned composition mechanisms |
| New composition | Canonical operation-tree families and their paraphrases | Constituent operations |
| New attributes | Attribute combinations, such as CoGenT A/B | Individual attributes |
| Longer dependence | Event distance, distractor count or operation depth | Vocabulary and primitive rules |
| New domain | Selected object/service/world families | Previously learned transferable operations |
| Partial observation | Visibility patterns and missing evidence | Declared observation/persistence rules |

Do not require every type of holdout simultaneously in every split. A pure name-transfer test deliberately reuses abstract structures; a structural-transfer test must exclude their canonical forms. Report combined holdouts as additional harder tests.

Keep all questions, paraphrases, prefixes and alternate renderings of a base episode together. For composition holdouts, canonicalize programs independently of names. Group adapted static data by scene, story, conversation root, or environment instance as appropriate. Near-duplicate templates and reused passages also need an audit.

Use a train split, development split for gate-driven iteration, and sealed final transfer suites. Where possible preserve official public splits alongside Coco-specific splits; report which protocol produced each score. Since official benchmark test files are public, keeping them out of broad chat ingestion is an explicit contamination task.

### 9.3 Reporting requirements

For the major synthetic families, aim for at least 1,000 independently generated evaluation world/trajectory groups and three training seeds where feasible. If the grammar cannot supply that many genuinely different groups, report the actual count instead of inflating it with paraphrases. Report confidence intervals clustered by world, not by question.

Report worst-family accuracy, complete-trajectory accuracy, answer correctness, paraphrase consistency and uncertainty behavior separately. Changing a phrasing must not pass simply because both variants produce the same wrong answer. Sample-level perfection is evidence about that test, not proof of universal generalization.

Track bytes or events processed, elapsed training time, inference latency, peak RAM, and performance as the number of stored or observed entities grows. A capability gain accompanied by a severe throughput collapse is a separate result to investigate, not hidden implementation overhead.

## 10. Adapter schema and data validation

The following illustrates the separation; it is not a required internal Coco representation.

```json
{
  "id": "episode_00001_q03",
  "learner_input": {
    "observations": [
      "Lena holds the key. Sam holds the cup.",
      "Lena gives the key to Sam."
    ],
    "question": "Who holds the key now?"
  },
  "training_target": {
    "answer": "Sam",
    "direct_state_target": null
  },
  "oracle_only": {
    "answer": "Sam",
    "full_state": {"holder_of_key": "Sam", "holder_of_cup": "Sam"},
    "query_program": ["holder", "key"],
    "group_id": "world_00001_trajectory_00001",
    "skill_family": "transfer_then_retrieve",
    "phrase_family": "who_holds_now",
    "source": "dyna_babi_adapter",
    "split": "train"
  }
}
```

Evaluation records omit `training_target` from the learner-facing interface. Prediction targets are revealed only after the prediction is scored when doing prequential/online evaluation. Generated descriptions of the world remain separate from Coco's own assertions; the latter are not automatically promoted to observations.

Before training, validate state-transition invariants, query-answer agreement, visibility-consistent uncertainty, prefix chronology, and zero cross-split group leakage. Independently spot-check each operation family. Dyna-bAbI's README explicitly cautions about limitations and an experimental inference engine; a public generator is not an excuse to skip oracle tests. [S1]

Minimal adapters to implement, in order: a Dyna-bAbI episodic stream reader; a CLEVR scene-to-text renderer with a teacher-only program executor; grouped split and replay tooling; a TextWorld rollout adapter; and a grounded dialogue/mock-service adapter. Formal semantic labels and complete-world snapshots are optional scaffolds, not prerequisites imposed on Coco's architecture.

## 11. Acquisition and licensing notes

The source register below is an acquisition map, not a bundle of downloaded data. Preserve source attribution, the exact revision or archive version, checksum, original split, transformation script version, and a copy of applicable terms for every ingested component. A repository's code license must not be silently substituted for a separate dataset's terms.

Keep CLUTRR outside the default pipeline pending permission review because its listed license is noncommercial. Do not infer that a mixed-source collection has one license from its top-level card. These are provenance cautions, not a legal clearance opinion. [S12, S13]

For TextWorld, the reviewed README supports Linux/macOS and gives Docker as a Windows workaround. Generate and export compact episodes in a compatible environment rather than making Coco's learner depend on a game engine process for every training example. On-policy interactive evaluation still needs the environment. [S6]

For offline trajectory acquisition, an oracle or scripted policy may provide demonstrations, but log it. At evaluation, measure Coco's selected actions; do not allow a demonstration policy, admissible-action filter, or expert subgoal executor to silently supply missing competence.

## 12. Immediate implementation order

Implement the world/teacher boundary and frozen regression tests first. Then run the 20,000-episode event-grounding comparison. Add CLEVR count/comparison and phrase replay to retain existing capabilities. Move to TextWorld interaction after the observation-only transition test is informative; use LLM-BabyBench Predict as an optional external check. Only then convert known worlds into dialogues and add SGD.

The decisive result is not that Coco can produce a plausible description of a transfer. It is that changing the actual recipient changes the inferred holder, a failed transfer leaves the holder unchanged, unrelated facts persist, and this remains true under unfamiliar names and wording without being handed the updated state.

## Sources

The identifiers used above refer to the primary project repositories, dataset cards or official project pages below. Dates are page-review dates, not claims that every source was first released or last updated on that date.

### [S1] Dyna-bAbI

Source: `https://github.com/ronentk/dyna-babi-kit`  
Type: `generator_and_pregenerated_data`  
Use: Stream story observations; control entity vocabulary and skill mixtures; keep solver/support annotations teacher-only.  
Terms noted: Repository lists CC0-1.0. Verify terms separately for inherited original bAbI downloads.

### [S2] Original bAbI task generator

Source: `https://github.com/facebookarchive/bAbI-tasks`  
Type: `legacy_generator_and_dataset_reference`  
Use: Use task definitions and public data for coverage not supplied by the selected Dyna-bAbI configuration.  
Terms noted: Check terms of the exact dataset archive separately from generator code.

### [S3] CLEVR and CoGenT

Source: `https://cs.stanford.edu/people/jcjohns/clevr/`  
Type: `static_dataset_with_scene_graphs_and_programs`  
Use: Use no-image archives; verbalize scene graphs; programs and answers are teacher-only. This is a text adaptation, not official visual CLEVR.  
Terms noted: Data: CC BY 4.0.

### [S4] COGS

Source: `https://github.com/najoungkim/COGS`  
Type: `semantic_parsing_dataset`  
Use: Use corrected data, and distinguish in-distribution test.tsv from generalization gen.tsv. Formal targets are an optional supervised diagnostic.  
Terms noted: Repository lists MIT. Record the exact data snapshot.

### [S5] SCAN

Source: `https://github.com/brendenlake/SCAN`  
Type: `command_to_action_dataset`  
Use: Keep official length and add-primitive splits separate; this tests command composition rather than world-state grounding.  
Terms noted: Repository LICENSE is BSD 3-clause.

### [S6] TextWorld

Source: `https://github.com/microsoft/TextWorld`  
Type: `interactive_generator_and_environment`  
Use: Generate worlds and collect rollouts; learner gets observations, own actions and feedback, not hidden facts, walkthroughs or oracle plans.  
Terms noted: TextWorld code: MIT. Dependencies have separate licenses; do not infer asset terms from code.

### [S7] LLM-BabyBench

Source: `https://huggingface.co/datasets/salem-mbzuai/LLM-BabyBench`  
Type: `offline_grounded_reasoning_dataset`  
Use: Start with Predict; create grouped splits using level and seed metadata, plus task-family holdouts. Do not assume the displayed train partition is a complete evaluation protocol.  
Terms noted: Dataset card lists MIT.

### [S8] BabyAI in MiniGrid

Source: `https://minigrid.farama.org/environments/babyai/`  
Type: `interactive_environment`  
Use: Preserve the selected observation regime when serializing to text; do not expose the full grid in a partial-observation evaluation.  
Terms noted: Review the pinned MiniGrid code and asset licenses.

### [S9] ProPara

Source: `https://github.com/allenai/propara`  
Type: `annotated_natural_process_text`  
Use: Use sentence-prefix state tracking; withhold future sentences and evaluation state annotations.  
Terms noted: Repository lists Apache 2.0; verify coverage of the exact data artifact before ingestion.

### [S10] Schema-Guided Dialogue and SGD-X

Source: `https://github.com/google-research-datasets/dstc8-schema-guided-dialogue`  
Type: `annotated_task_dialogue_and_schema_variants`  
Use: Reconstruct turns and service results; create a mock backend for interactive task verification. SGD-X varies schemas, not all dialogue utterances.  
Terms noted: Dataset: CC BY-SA 4.0.

### [S11] OpenAssistant OASST1

Source: `https://huggingface.co/datasets/OpenAssistant/oasst1`  
Type: `human_assistant_conversation_trees`  
Use: Select English, reviewed nondeleted messages and quality branches; split all paths from one message_tree_id together.  
Terms noted: Dataset card lists Apache 2.0.

### [S12] SmolTalk selected subsets

Source: `https://huggingface.co/datasets/HuggingFaceTB/smoltalk`  
Type: `synthetic_instruction_data`  
Use: Begin with smol-rewrite and smol-summarize; retain source text, audit outputs, and do not treat synthetic replies as a truth oracle.  
Terms noted: Newly produced subsets described in the card are Apache 2.0; imported components retain their own terms. Do not treat all as one license.

### [S13] CLUTRR

Source: `https://github.com/facebookresearch/clutrr`  
Type: `relational_reasoning_dataset`  
Use: Optional kinship-chain diagnostic only after permission review; no assumption that evaluation use is automatically exempt.  
Terms noted: CC BY-NC 4.0; noncommercial restriction.

### [S14] TinyStories

Source: `https://huggingface.co/datasets/roneneldan/TinyStories`  
Type: `synthetic_story_text`  
Use: Optional later fluency input, not an oracle for consistent world state.  
Terms noted: Dataset card lists CDLA-Sharing 1.0.

### [S15] SQuAD 2.0

Source: `https://rajpurkar.github.io/SQuAD-explorer/`  
Type: `passage_question_answering`  
Use: Use supplied passages and unanswerable questions before broad factual chat; do not confuse passage support with world truth.  
Terms noted: Dataset page lists CC BY-SA 4.0.

### Motivation supplied in the request

BabyAI: `https://arxiv.org/abs/1810.08272`  
SCAN paper: `https://arxiv.org/abs/1711.00350`

These motivate controlled instruction learning and explicit composition tests; they do not establish that the proposed Coco curriculum will succeed.
