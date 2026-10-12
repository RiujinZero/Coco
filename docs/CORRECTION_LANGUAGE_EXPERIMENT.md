# Learning complete correction constructions — 2026-09-23

Coco's experimental reader now learns the outer wording of a correction request
from verified answers about its consequences. It identifies a known report
reference and an optional known report sentence inside the request, then learns
whether the construction means preserve, withdraw or replace. The caller supplies
the whole utterance and source identity; it no longer supplies the operation,
reference span or replacement boundary.

This extends [historical-reference grounding](HISTORY_REFERENCE_GROUNDING_EXPERIMENT.md).
It remains a bounded symbolic learner over supplied operations and already learned
reference/report language. It does not establish general chat comprehension or
joint continual learning of every component.

## Mechanism

`cpp/include/laminar/correction_language_grounding.h` searches contiguous token
spans for previously learned report-reference frames and report sentences. It
replaces those spans with typed placeholders and retains the surrounding literal
words as a construction. All recognized names must be accounted for by the
selected spans. Overlapping reference and report spans are excluded. Alternative
parses are considered rather than choosing the first recognized span.

For each construction, the initial operation candidates are KEEP and WITHDRAW,
plus REPLACE when a report span exists. These operations are supplied primitives;
their association with the request's words is learned. The engine contains no
English correction-verb mappings, delimiter rules or annotated argument offsets.
For example, its eventual construction `use @report instead of @reference` is
induced from examples, not installed as a parser rule.

Teaching provides a whole request and ordinary verified questions/answers about
the state after the intended correction. The learner resolves the reference using
its prior reference model, tries each candidate operation on private copies of
the journal and reader, and evaluates those questions in the resulting worlds.
Only operations consistent with all answers survive. These hypothetical edits
never alter real evidence. No target report ID, correct operation, structured
after-state or execution trace is supplied to this teaching call.

Evidence is identified by scene and evidence IDs; repeated identical feedback is
idempotent and conflicting reuse is rejected. Context includes the original
journal state, current state, records, metadata fields, reference model, language
priors and primitive weights. A construction needs four distinct scenes before
use. Search has word, span and simulation budgets.

At execution time, a single supported interpretation invokes the existing
revision/clarification mechanism. Uncertain operation meanings withhold the edit.
Ambiguous report references can request an explicit report choice. Pending
choices now also check the correction-language model signature, so acquiring a
different meaning cannot silently change an outstanding correction. Source and
record-revision restrictions still come from the journal.

## Teaching and transfer

The experiment first reconstructs the previously tested reference learner from
its 784 historical-answer lessons. It then teaches six correction constructions
over 72 fresh histories, with seven post-correction answers each: 504 answers.
Question language, report language, reference meanings, primitive policies and
the neural body remain frozen during this second stage.

Five constructions acquire definite meanings:

- `replace @reference with @report` — replace;
- `use @report instead of @reference` — replace;
- `for @reference use @report instead` — replace;
- `withdraw @reference` — withdraw;
- `leave @reference unchanged` — preserve.

The sixth, `consider @reference`, is deliberately taught where preserving or
withdrawing the selected report has the same consequences. It retains both
operations and refuses to choose between them in distinguishing test histories.
This is an uncertainty control, not a claim about the unrestricted English
meaning of “consider.” Each construction receives twelve distinct scenes.

Correction teaching uses only “the latest active report about X” as its reference
and possession sentences as replacement text. Frozen evaluation combines the
learned constructions with six definite reference forms and movement sentences,
as well as ambiguous references requiring clarification. Entity names are disjoint
between teaching and evaluation. Record IDs are randomized independently of
chronology; relation and metadata field IDs swap across seeds.

## Results

All three seeds pass the independent audit, with the following results per seed:

| Request or check | Result |
|---|---:|
| Direct replacements | 72 / 72 |
| Direct withdrawals | 24 / 24 |
| Requests to preserve the report | 24 / 24 |
| Edits after report clarification | 24 / 24 |
| Stale clarification refusals | 12 / 12 |
| Missing report references | 12 / 12 |
| Different-source revision refusals | 12 / 12 |
| Generated correction rejections | 12 / 12 |
| Untaught command wording withheld | 12 / 12 |
| Underdetermined operation meanings withheld | 12 / 12 |
| Untaught replacement wording withheld | 12 / 12 |
| Reference model removed: requests withheld | 12 / 12 |
| Correction model removed: requests withheld | 12 / 12 |
| Current-state questions after requests | 1,764 / 1,764 |
| Current/historical state snapshots independently checked | 9,588 |

There are 252 request cases and 120 accepted edits. The 120 direct, supported
requests include 100 combinations with reference forms never used in correction
teaching. Thirty-six replacement cases use movement constructions never used as
correction payloads during teaching. These counts overlap and concern controlled
combinations, not arbitrary unseen English.

The auditor checks every historical view as well as the current state. A later
valid report can overwrite the corrected fact, so a correct current answer alone
would not prove that the right earlier report was edited. Invalid choices are
tested before every valid or stale clarification. Stale cases retain their newly
observed environment event while refusing the obsolete pending edit.

Whole requests and reports enter through the real Organism input path; replies
and current answers use its output path. The probe reads an explicit clarification
choice only after initial interpretation and clarification output. Definite test
inputs contain no selected report ID, operation field or replacement span.

## Validation and limits

`python/check_correction_language.py` independently interprets report text and
reference semantics, computes candidate correction consequences, checks every
teaching update and final operation set, and reconstructs all record versions and
current/historical knowledge. It checks serialization, names, exact replies,
source restrictions, revisions, clarification behavior and model-removal controls.
It imports only the existing native-knowledge decoder, not the generator or native
learner. Artifact hashes are recorded in `logs/correction_language/audit.json`.

Address/undefined-behavior sanitizer contracts cover raw request ingestion,
construction learning, span-order and reference transfer, unchanged training
worlds, duplicate/conflicting/generated feedback, unknown payloads, source
checks, retained ambiguity, stale language and bounded search. They also reject
reference metadata that names a report absent from the journal.

The learner is still given the operation vocabulary, typed-span composition
mechanism, known single-token entities, trusted source metadata and verified
answers. It accepts one report reference and at most one replacement sentence,
normalizing the existing alphanumeric token grammar. Clarification uses an
explicit ID protocol, and acknowledgement/clarification wording is supplied.
It does not learn source reliability, discover which report is wrong, learn new
report/question meanings concurrently, or revise its own correction-training
evidence through a general replay mechanism. It is invoked through an experimental
adapter, not integrated into the standard byte trainer. Longer plans, concept
growth and joint continual learning remain open.

This route uses symbolic candidate elimination over previously learned local
primitive policies, without end-to-end backpropagation. The neural body is frozen
in this experiment.

Run `bash scripts/correction_language.sh` under WSL with the prior primitive,
bootstrap/incremental/dialogue/observation and history-reference teaching corpora
present. The script generates the local experiment, builds the native and
sanitizer probes, runs three seeds and invokes the independent auditor. It does
not modify `/home/user/curriculum_v2` or its builders.

Follow-up: [the grounded session](GROUNDED_SESSION_EXPERIMENT.md) connects these
learned correction constructions, ordinary questions and report journaling to the
same byte stream and an optional standard-trainer path. Per-turn dispatch is no
longer supplied by the probe; symbolic meanings remain pretrained and frozen.
