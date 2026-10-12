# Grounding references to historical reports — 2026-09-23

The experimental reader learns bounded references such as “the earliest active
report about X” from verified answers about historical states. The learned
reference then selects a report for correction in a fresh history. Training does
not supply the target report ID or the intended selector. This extends the
[revision journal](REVISION_GROUNDING_EXPERIMENT.md), which already supported
replacing explicitly identified evidence and rebuilding its consequences.

Six definite reference frames acquire the intended unique selectors. A seventh,
deliberately underdetermined frame retains three meanings and asks for a report
choice when they disagree. All three native seeds pass the independent audit.
This is a symbolic, bounded reference learner; the correction operation and
replacement text are still separate, caller-supplied fields.

## Mechanism and supervision

`cpp/include/laminar/history_reference_grounding.h` enumerates generic selectors
over ordered journal records: argument-to-field bindings, active/inactive/any
filtering, and first/second/last/penultimate/all selection. The engine contains no
English mappings for “earliest,” “latest,” “about,” or “from.” The experiment
supplies fields for recognized entity mentions and source identity. Their numeric
identifiers are swapped in seed 1. Record order comes from the journal; randomized
record IDs do not encode chronology.

A teaching example supplies a reference phrase, an already understood ordinary
question, and a verified answer at the referenced historical state. For example,
the reference and “Who has the item now?” are separate interface fields. For each
candidate selector, the learner evaluates the question in the selected historical
views. It retains a selector only if every selected view agrees with the answer.
Evidence is intersected across examples; duplicate evidence cannot add support.
Publication requires at least four distinct scenes.

The learner binds known single-token names into whole reference frames. It does
not infer a compositional grammar of temporal adjectives, parse an unrestricted
correction utterance, or discover the supplied metadata fields. The report
language, question meanings, primitive policies and neural body are pretrained
and frozen throughout this experiment.

At correction time, unanimous selection of one report allows revision. Ambiguous
selection creates a pending choice with a token and snapshots of the episode,
observation, history, model and candidate revisions. An invalid choice preserves
the pending request; changed context invalidates it. A choice resolves this
occurrence without teaching a new reference meaning. The journal checks the
supplied source identity and expected record revision before applying the edit.
These are caller-provided identities, not learned source trust or authentication.

Withdrawing a report now retains its original text, and withdrawing a teaching
answer retains its original answer. Inactive records are excluded from evidence
replay but remain identifiable. Corrections rebuild current and historical
knowledge at the original record position, preserving subsequent valid events.
Historical views describe the replayed, currently corrected evidence history;
they are not snapshots of everything Coco believed at each past wall-clock time.

## Corpus and results

The generated teaching corpus contains 112 histories, sixteen per reference
frame, with eighteen reports per history. Four people, four rooms and three
objects provide location and possession changes. Seven ordinary consequence
questions per history yield 784 teaching answers. The six definite forms cover
earliest/latest/previous active reports about an entity, earliest/latest active
reports from a source, and latest active reports matching both source and entity.

“The active report about X” is taught only where one matching report is active.
Its first, last and all selectors remain indistinguishable. Fresh evaluation
histories contain several matching active records, so clarification is required.
Training and test entity names are disjoint. Some test source names coincide with
person names, preventing spelling alone from distinguishing the metadata fields.
These are controlled template transfers, not unseen grammatical constructions.

| Frozen evaluation, each of seeds 0, 1 and 2 | Result |
|---|---:|
| Definite references selecting and correcting the intended report | 72 / 72 |
| Ambiguous references corrected after a valid choice | 12 / 12 |
| Pending corrections rejected after a new environment report | 12 / 12 |
| Missing references withheld | 12 / 12 |
| Untaught reference wording withheld | 12 / 12 |
| Revisions rejected for a different source identity | 12 / 12 |
| Generated corrections rejected | 12 / 12 |
| Historical answers before correction | 588 / 588 |
| Current-state answers after each case | 1,008 / 1,008 |
| Full current/historical state snapshots independently checked | 5,484 |

There are 144 cases and 84 accepted corrections per seed. Accepted operations
include replacements and withdrawals. Definite test requests carry no target
ID. For ambiguous cases, the native probe reads the explicit choice only after
reference resolution and the initial clarification response. Every ambiguous
and stale case first tests an invalid choice. Stale cases include a legitimate
new environment report; rejection preserves that observation rather than
restoring the old world.

References and reports enter through the real Organism input path; historical
answers, current answers and clarification replies use its output path. The
correction operation, reference span, replacement payload, source identity and
clarification selection remain explicit interface fields. This is not a learned
end-to-end correction conversation.

## Curriculum confound found and fixed

The first experimental corpus always withdrew the first and last matching
reports. All expected edits succeeded, but the independent model audit failed:
“first active” was indistinguishable from “second regardless of activity,” and
“last active” from “penultimate regardless of activity.” Task accuracy concealed
incorrect surviving interpretations.

The revised generator varies the number of leading and trailing withdrawals
independently across zero, one and two. The same learner then identifies all six
definite selectors uniquely; the deliberately weak seventh frame retains exactly
the three intended alternatives. The failed pilot, its generator and raw results
are preserved under `logs/history_reference/pilot_uniform/`.

The curriculum lesson is to vary availability independently of chronological
position. Score the surviving interpretations as well as answers when a small
domain permits exact inspection. This change is confined to the local experiment;
the user-owned WSL curriculum and builders were not modified.

## Validation, limits and reproduction

`python/check_history_reference.py` independently interprets raw reports,
enumerates selector hypotheses, checks every teaching update, and reconstructs
all current/historical states before and after corrections. It verifies exact
model meanings, record revisions, candidate sets, replies, source restrictions,
stale choices, disjoint names and serialized input. Source, input, binary and
output hashes are saved in `logs/history_reference/audit.json`.

Address/undefined-behavior sanitizer contracts pass for historical-answer
grounding, shuffled record IDs, duplicate/conflicting feedback, generated inputs,
ambiguity, source restrictions, stale contexts, retained withdrawn text and search
budgets. The revision suite was rebuilt against the content-retention change;
its sanitizer contracts, all three native runs and independent audits pass,
including 1,536 frozen current/historical questions per seed.

This experiment supplies a finite rank/filter vocabulary, at most two reference
arguments, known entities, trusted metadata and already grounded ordinary
questions. Teaching occurs before frozen correction evaluation. It does not show
joint learning of reference language, question language, policies and action
effects; source-reliability learning; autonomous discovery of which report is
wrong; or unrestricted temporal language. Reference feedback currently has
duplicate/conflict guards but no general correction/replay mechanism of its own.
The underlying journal remains bounded and rebuilds from a snapshot.

Run `bash scripts/history_reference.sh` under WSL after generating the existing
primitive and bootstrap/incremental/dialogue/observation corpora. The script
generates the local corpus, builds native and sanitizer probes, runs three seeds
and invokes the auditor. Results are under `logs/history_reference/`.

Follow-up: [complete correction constructions](CORRECTION_LANGUAGE_EXPERIMENT.md)
learns the outer request meaning and reference/payload spans from post-correction
answers, removing those separate fields from the execution call. The report,
question and reference models remain pretrained and frozen in that experiment.
