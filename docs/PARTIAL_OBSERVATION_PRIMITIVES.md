# Partial observation primitives — contract validation

The optional grounded reader can now distinguish an unobserved relation from an
observed false relation. This is a memory and execution contract, not a claim that
Coco has learned the new curriculum or natural-language visibility descriptions.

The later [observation-language experiment](OBSERVATION_LANGUAGE_EXPERIMENT.md)
grounds eleven evidence/visibility frames from raw stories and verified answers.
It adds complete incoming observations and unique relation reads. The contracts
below describe the original primitive layer; its former language-learning gap is
now partially addressed by that separate, bounded experiment.

`fact_knowledge.h` stores positive facts, explicit negatives, and scopes known to
be completely observed. Scopes can be an outgoing relation row or an incoming
relation column. Missing membership is unknown unless a negative or complete
scope establishes its absence. Empty observations can therefore support zero;
an empty memory alone cannot.

`LanguageReasoner::partial_observation` is opt-in; previous closed-world probes
retain their behavior. The perception APIs are:

- `observe_fact(subject, relation, object, present)` for a positive or negative
  observation at the current state;
- `observe_complete_row(subject, relation, objects)` for a complete observed set;
- `lose_visibility(name, relation, column)` for an explicitly unobserved interval
  in which facts in that scope may have changed.

These APIs supply observation semantics. No current text rule infers their meaning
from words such as "unseen" or "all." `lose_visibility` invalidates facts and
negative evidence within its scope and invalidates intersecting completeness
claims. Ordinary observations constrain the same state; contradictions are
reported and withhold answers until reset. They are not automatically treated as
new events or corrections. Observation IDs, reliability learning, and retracting
an earlier observation during an ongoing episode remain open.

Known event effects update positive/negative evidence and completeness alongside
the fact store. Generic SET_SUBJECT and SET_OBJECT operations establish a complete
row or column, respectively. Unknown event language still makes the episode
incomplete. Generated text and self-generated perception calls cannot establish
evidence.

The procedure runtime adds generic membership operations 10/11 for its two
relation slots. Search over them is enabled by `lookup_enabled`. The contract
test induces a membership procedure from 16 final Boolean answers, with the
other relation and reversed bindings as distractors. Membership returns true,
false, or missing evidence; Boolean output remains `yes`/`no` and the reader
renders missing evidence as `I do not know.` No English question recipe is built
into the operation. This memory operation is supplied, rather than learned as a
new microcontroller policy.

Existing learned count calls require complete outgoing coverage. Follow calls
require completeness at each read, including the terminal node. This is
conservative: even one known outgoing edge is insufficient without closure.
The contract test checks that the terminal node's absence is not inferred from
an unobserved row.

Ambiguous event interpretations carry separate knowledge states. Answers are
evaluated in each world; unknown or disagreement withholds the answer. Positive
fact intersections do not imply that a projected set is complete: projected
closure is retained only when every branch has the same complete contents.
Trustworthy observations can eliminate inconsistent branches. A contradiction
with all branches is reported, without silently selecting a convenient world.

`cpp/tests/test_fact_knowledge.cpp`, run under AddressSanitizer and UBSan, passed
checks for explicit false versus unknown, learned membership, open counts versus
complete zero, complete nonempty counts, follow termination, visibility loss,
observation conflicts, self-generated input, episode reset, aligned observation
mode, branch agreement, and observation-driven branch filtering. Some language
forms and worlds in this contract test are fixtures; they do not count as learned
language generalization.

All earlier event, reference, shared-question and answer-grounding probes were
rebuilt and independently audited after this change. Their expected results and
ablations remain intact. The actual public curriculum readiness test is reported
separately in `CURRICULUM_REBUILD_ISSUES.md`: it exposed unsupported observation
language and dataset issues, not a successful curriculum training run.

Reproduce the new contract and curriculum audit with:

```sh
bash scripts/curriculum_grounding.sh
```

The primitive demonstration corpus in `logs/reasoning_composition_verified` is
required. On Windows run the script through WSL Ubuntu.
