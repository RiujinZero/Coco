# Two successive repairs in one continuing learner

The opt-in repair bank preserves an earlier supported primitive repair while
searching for another. A focused one-seed `ComposedLearner` contract and a real
`RelationalSession` continuation both repair two already available wrong skills
from whole-question outcomes, retain original direct outcomes and preserve the
independent repair through withdrawal and contradiction.

The session test starts with learned language that gives demonstrably wrong
answers for duplicate counts and longer follow chains. Later verified Boolean
language outcomes repair counting, then following. It passes **216 fresh count
language checks**, **12 fresh follow language checks**, **26 independent
primitive probes**, and all **11 original direct outcomes**. Count checks span
initial repair, retention after follow repair, and follow-support withdrawal.
Restoring the withdrawn language outcomes recovers the exact combined neural
weights. These are one-seed focused integration results, not a repeated-seed
mixed-controller campaign or simultaneous two-slot discovery.

## The defect and the new acceptance path

Previously, rebuilding first reset the active model to its direct-evidence
baseline. A cached repair could be reused only if it already explained every
new composed outcome. When a second skill needed repair, discovery restarted
from the original direct baseline and substituted one candidate slot. Neural
rehearsal also restored other slots from their original direct policies. Thus
the search could not combine a first accepted repair with a newly proposed one.

`enable_repair_bank()` is an explicit pre-learning option. The default and
existing `enable_repair()` paths keep their previous single-repair storage and
rebuild behavior. The candidate-selection correction described below also
applies to existing discovery callers. A bank entry stores one
accepted sealed policy and its outcome-verified rehearsal samples. Before any
entry is reused:

- Unchanged, still-active whole-question witnesses must replay successfully.
  Their execution must cover the entry's required state/action targets.
- Current direct outcomes contribute coverage but cannot be the only composed
  support for a derived repair. A newly contradicted composed query group cannot
  justify its old targets through an unchanged receipt with a different ID.
- Unsupported entries are removed. The remaining bank is rematerialized and
  replayed until support is stable. This avoids retaining a repair whose only
  proof relied on another now-invalid repair.
- The search runs against this supported model and still tries all candidate
  slots. The teacher supplies no faulty-slot identifier.

Rehearsal merges all retained entries and the newly discovered candidate into
one neural model, starting from the direct baseline. Conflicting sample targets
are rejected. A new repair is installed only if the final neural model fits
every active composed outcome, every active direct outcome, and the full
supported rehearsal bank. Discovery's `acquired` flag alone is not acceptance.

If contradictory direct evidence makes one original slot unavailable, bank
materialization keeps it unavailable and can retain independently supported
other repairs. It does not attempt to rehearse an unavailable direct policy.
New acquisition still fails its all-direct-outcome acceptance check while that
contradiction remains. The empty-bank version of this case now returns
unresolved instead of throwing during materialization.

The full accepted checkpoint stays cached during partial withdrawal. If its raw
support returns, it can restore the original neural weights rather than silently
replacing them with another merely equivalent fit.

## A second bug exposed by the contract

Candidate slots can have different numbers of direct prior outcomes. The old
result selector could discard a complete acquisition with 56/56 correct rows in
favor of a different slot's partial 57/60 fit, because it compared raw correct
counts first. Acquisition now takes precedence over every partial fit. The
focused contract exercises that exact unequal-denominator case. Ranking within
equal acquisition status is unchanged, preserving the earlier search behavior.

## Curriculum and evidence

The direct warm-up supplies ordinary final outcomes for empty and one-hop
following, zero/one/two distinct records without duplicates, and equality across
the three existing value types. Both follow and count are available afterward,
but independent counterexamples show that their learned meanings are wrong.
No explicit micro-policy, repaired instruction sequence or pretrained checkpoint
is imported.

Count repair receives only Boolean outcomes for comparisons between two counts
and comparisons with a typed numeric operand. The numeric operand grounds the
count's scale; count-to-count equality alone permits arbitrary injective numeric
encodings. Follow repair receives final Boolean comparisons with a chain's
endpoint. Longer training paths prevent a short learned micro-procedure plus
macro unrolling from masquerading as a repaired transitive-follow primitive.

The core contract uses 42 count outcomes, then 12 follow outcomes. Its follow
teaching reaches 48 hops. The real language session has a 32-entity scene limit,
so its follow teaching reaches 24 hops; it receives raw `LanguageReceipt`s and
rebuilds through the actual language materialization and shared primitive model.
It first learns the three language frames from 25 unique-record/short-path warm
examples, then receives 21 duplicate-count and 10 longer-follow counterexamples.
Words are opaque frame identifiers. Typed mentions, complete scenes and verified
final truth labels remain supplied priors.

Both tests' winning count candidate appears after six proposals and winning
follow candidate after 2,581 proposals. Other candidate slots also consume
proposals: these are winning-slot counts, not total search cost. The per-slot
budget is 5,000 with a 100-proposal interleaving quantum. Search scheduling is
supplied, and no learned repair selector is claimed.

The final primitive probes use fresh entity/relation IDs, chain lengths and
distinct-set sizes through 32, and duplicate multiplicities one through three.
The VM replay cap remains 256 instructions. An evaluator-only correct reference
classifies all 26 core probes as feasible (maximum 163 instructions). Reference
decisions never enter acquisition, rehearsal or warm starts. All 26 pass for the
learned core and language-session models. The session's primitive length-32 probe
is beyond its maximum taught path length; its language probes remain within the
declared 32-entity interface.

The bank-disabled control receives the same first evidence and learns exactly
the same initial neural weights. With later follow evidence it reproduces the
loss of the count repair under a bounded second search. This is a focused defect
control, not an equal-budget performance comparison. The support tests include
independent count retention, different-ID composed contradiction, direct-prior
contradiction, withdrawal, restoration, and an empty-bank contradiction.

## Verification and preserved failures

Optimized and ASAN/UBSAN core contracts pass. Existing `composed_discovery`,
`composed_support` and `composed_learner` regressions pass, including the original
redundant-support exact-weight behavior. The language-session sanitizer result
and artifact hashes are recorded in `SUCCESSIVE_REPAIR_BANK_VALIDATION.json`.
Evidence is under `logs/successive_repair_bank_v1/`.

Earlier focused builds are retained as `build-wsl/test_successive_repair_bank_v1`
through `v11`. They exposed and preserved the acquired-versus-partial selection
bug, insufficient outcome curricula, failed 5,000-proposal searches, and the
bounded-hop primitive that passed short composed tasks but failed broader
primitive probes. These failures do not count as successful repair. The final
core build is `v12`; the real session bridge passes as `v1` without further core
changes. `failure_history.json` records the observed failures and refinements.

The canonical profile is `controller50k_repair_bank_records`. Its independent
wiring smoke passes eight training records, eight frozen evaluations and 324/324
primitive checks. Its 6/8 endpoint score is a small smoke result, not a capability
threshold. The final combined trainer also passes canonical replay/correction/
rejection validation and reproduces all eight archived handoff artifacts exactly;
see `LEARNER_FOLLOWUP_BUILD_VALIDATION.json`. Those trainer results are separate
from the focused successive-repair contract.

## Remaining scope

There are still three supplied primitive slots. The bank preserves and replaces
their learned policies; it does not add a new operator or representation.
Discovery still varies one slot at a time. Two initially wrong slots that must
change together before any complete fit exists remain outside its candidate
class. A joint candidate mechanism and a prespecified multi-seed campaign are
still required. Noisy feedback, learned scheduling, broad long-credit behavior
and the full byte reader are not established by this change.

The latest accepted bank checkpoint is kept, not an unbounded history of every
past repair version. Restoration is verified across the tested withdrawal and
correction lifecycle; exact recovery after additional new repairs overwrite that
checkpoint during a withdrawal has not been demonstrated.

## Reproduce

From the repository root inside WSL/Linux:

```sh
g++ -std=c++20 -O2 -Icpp/include cpp/tests/test_successive_repair_bank.cpp -o build-wsl/test_successive_repair_bank_reproduce
./build-wsl/test_successive_repair_bank_reproduce
g++ -std=c++20 -O2 -Icpp/include cpp/tests/test_successive_repair_session.cpp -o build-wsl/test_successive_repair_session_reproduce
./build-wsl/test_successive_repair_session_reproduce
```

Use `-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer` for sanitizer
builds. Sources are `cpp/include/laminar/composed_learner.h` and
`cpp/include/laminar/composed_skill_discovery.h`; session/profile wiring lives in
the existing relational and continual-training interfaces. Bank behavior is
opt-in; the normal trainer executable has not been promoted by these tests.
