# Control transfer: learned preferences and search allowance

The baseline's six larger-goal failures are resolved by retaining its search
allowance across revisions that contain exactly the same observed scene. Across
three seeds, the frozen baseline controller now passes **384/384 standard cases,
89/89 recovery cases and 36/36 larger goals**, versus 30/36 larger goals before.
This is a scoped runtime correction, not completion of the four learner gates.

The complementary readout experiment is negative: fitting decision trees to the
same learned value preferences worsens performance. All nine new runs pass the
independent trace audit. Their training traces, neural and primitive weights,
and empirical value tables match their respective preceding runs byte for byte.

## Matched results

| Frozen evaluation configuration | Standard /384 | Recovery /89 | Larger /36 |
|---|---:|---:|---:|
| Baseline neural controller, original runtime | 384 | 89 | 30 |
| Baseline values → preference tree | 348 | 66 | 27 |
| Size-input neural controller, original runtime | 384 | 89 | 21 |
| Size-input values → preference tree | 326 | 48 | 0 |
| Baseline neural controller, retained allowance | **384** | **89** | **36** |

These reuse the preceding experiment's fixtures, including its recovery variants.
Larger goals contain 8 or 12 objects and three missing facts. The new comparisons
are paired on the same training and evaluation data; they are not a new unseen
benchmark selected before the preceding failures were inspected.

## The runtime correction

`accept_world` used to clear the answer, plan **and next search allowance** whenever
the incoming revision changed. Failed motors and physical no-ops advance the
revision even when the complete observed scene is identical. A controller that
chose an action after exhausting a four-prediction search therefore started its
next search at four again. This contributed to exhausting all 32 motor attempts.

The optional `ControlBody::preserve_search` behavior still invalidates cached
answers and plans on a revision change. It retains only `next_budget`, and only
when the canonical facts and completeness flag are unchanged. A changed scene
resets the allowance to four. Every subsequent search remains a chosen, charged
operation; no action, goal answer or correct next operation is supplied.

This retains a resource allowance, **not the search frontier**. Search still
recomputes from its current root. The experiment enables the option only on
frozen evaluation copies. Training and its random choices are unchanged; the
option remains off by default pending a continuing-learning comparison.

Larger-goal success is 12/12 in each seed. Total larger-goal decisions fall from
1,742 to 702, and motor attempts from 649 to 315. Standard and recovery cases also
use fewer motor attempts: 184 versus 259, and 414 versus 492 respectively.

The policy still never selects consecutive further reasoning while the previous
`search_budget` status persists in these evaluation groups. It explores, observes,
then reasons with the retained allowance. There are still 259 act decisions after
exhausted search on the larger goals. Thus the correction resolves the observed
failures without proving an efficient or broadly transferable computation policy.
It does not fix the separate size-input controller's retrieval loops after a
valid larger plan has been computed.

## Why a more faithful readout did not fix control

`PreferenceTree` is a generic CART classifier over the same 32 public inputs.
Its labels are the argmaxes of the learner's stored empirical values. It receives
no family IDs, hidden worlds, answers or correct-operation labels. Feature and
operation indices have no supplied semantics. Depth 12 and minimum leaf size 2
were fixed before the sweep. The tree is fitted independently at each frozen
checkpoint; it never chooses training operations or receives evaluation feedback.

For the size-input controller, neural predictions match the table's preferences
on 3,458/4,237 final stored states. The trees match 3,750/4,237, but all 36 larger
goals fail. More faithful copying of this table is not sufficient for transfer.

Inspection of actual training choices explains part of the concern. In 672 of
those 4,237 states, the highest stored value belongs to an operation never tried
in that exact public state. Untried values start at zero; they can win when the
observed alternatives have negative estimates. Those optimistic preferences are
then distilled as hard operation labels. This is measured support sparsity, not
proof that every such preference is wrong.

In all 474 experienced size-input states with an available plan, the neural
controller selects action. The value table prefers action in only 402 of them.
The neural approximation is therefore already smoothing some inconsistent
preferences, while its failure on larger plan states remains a generalization
problem. Both value-estimation support and readout extrapolation need attention;
this experiment does not isolate a single sufficient learning-rule fix.

## Verification and reproduction

The independent Python checker reconstructs public observations, charged costs,
physical outcomes, evidence inventories and selected operations. For trees it
also reconstructs the greedy split search, every partition and every leaf vote
from the saved public value table. The companion checker compares every training
trace outside evaluation and all saved training checkpoints against the originals.

Audits and paired comparisons:

- `logs/computational_readout/audit.json` and `paired_training.json`: six tree runs.
- `logs/computational_progress/audit.json` and `paired_training.json`: three retained-allowance runs.
- `logs/computational_readout/preference_support.json`: actual choice support and neural/table agreement.

Source snapshots preserve the preceding implementation, the tree build, and the
allowance-retention build. The final audits verify their manifests. ASAN/UBSAN
contracts pass for generic preference fitting, replacement after changed
preferences, action removal, unchanged-scene allowance retention, changed-scene
reset, cache invalidation, and existing controller/computational-session behavior.

```sh
bash scripts/computational_readout.sh
bash scripts/computational_progress.sh
python3 python/diagnose_control_preferences.py
```

## Remaining capability blockers

1. **Computation control:** one learner must transfer its choices to new demands,
   including further reasoning and acting on a completed plan, without depending
   on this particular exploration/reasoning cycle.
2. **Composed-outcome credit:** failure on a composed task must identify and teach
   or repair a missing primitive. The subsequent
   [composed-credit experiment](COMPOSED_CREDIT_EXPERIMENT.md) now acquires missing
   counting from composed final answers, with retained earlier neural skills.
   Repair of an available wrong skill, multiple missing components, efficient
   search and integration of this route with language/action control remain open.
3. **Representation growth:** new meanings are learned within supplied primitive
   operations and concept interfaces. The learner has not yet grown those
   interfaces or invented the missing operation structure from experience.
4. **Continuing learning:** corrections, primitive repair and representation growth
   must coexist with language, action and procedure acquisition while earlier
   skills remain usable. The current interleaved stream does not test all of them.

No full capability gate is marked complete by this follow-up.
