# Laminar 200k Organism — implementation map

This file maps every requirement in `Laminar_200k_Organism_Design_Doc.docx` and
`CPP_OPTIMIZATION_RUNBOOK.md` to the code that implements it, and records every place where the
implementation had to choose something the documents leave open.

## Layout

Current structured-learning entry point: **`laminar_train trainer=continual profile=controller50k`**.
This recommended profile uses a 50,013-parameter MLP, original-size Agkern and TD;
the original `coco_continual_v1` remains the historical CLI default. The
[canonical trainer](CANONICAL_TRAINER.md) consolidates the tested control schedule
and one continuing `ControlledRelationalSession`, with replay checkpoints and
versioned correction records. `cpp/include/laminar/continual_training.h` owns the
profile; `cpp/apps/continual_training_runner.h` is the shared training loop;
`relational_control_probe.cpp` is now a thin ablation entry point. The earlier
byte/grounded reader remains a separate model mode. The
[composed integration report](COMPOSED_SESSION_EXPERIMENT.md) records current
evidence and failures. The [tapered warm-start screen](WARM_START_CURRICULUM.md)
and [goal interruption tests](INTERRUPTED_GOALS_EXPERIMENT.md) now pass with the
50k TD controller. Episode-return credit passes a stronger propagation contract
but regresses on the matched task screen and is not promoted.
The [concept-isolation continuation](CONCEPT_ISOLATION_EXPERIMENT.md) removes a
global uncertainty gate: an unresolved new meaning no longer disables unrelated
known skills. The same continuing model retains old tasks while resolving the
new concept and learning actions/procedures that use it across three seeds.
The [complete-learner handoff](COMPLETE_LEARNER_HANDOFF.md) records the latest
retention and component-repair checks, remaining proof gaps, and suggested
mechanisms and validation order. All four broad capability gates remain open.

```
cpp/include/laminar/config.h     unit geometry (Dims), runtime parameters (Params), CLI parsing
cpp/include/laminar/rng.h        splitmix64 + murmur mixer (shared bit-for-bit with Python)
cpp/include/laminar/int4.h       packed INT4 store, scalar / AVX2 / expand kernels
cpp/include/laminar/address.h    locks, adaptive address tree (flat SoA), split evidence + G test
cpp/include/laminar/telemetry.h  per-block counters, stage timers (rdtsc), text + JSON reports
cpp/include/laminar/tasks.h      synthetic vocabulary generator, byte-file loader
cpp/include/laminar/organism.h   state (SoA), scheduler, traces
cpp/src/organism.cpp             the hot path, local learning, structural maintenance
cpp/apps/laminar_synth.cpp       milestones A/B/C: synthetic vocab, train then frozen eval
cpp/apps/laminar_bytes.cpp       milestone D: raw-byte language, adaptive / frozen / nomem modes
cpp/apps/laminar_bench.cpp       runbook §11 memory stress: familiar / moderate / storm / pathological
cpp/tests/test_kernels.cpp       kernel equivalence + address-tree invariants
python/laminar_ref/              NumPy reference organism (the slower faithful kernel)
python/run_synth.py              reference driver, dumps traces
python/tests/compare_traces.py   native-vs-reference parity check
```

## Unit geometry (design doc §2)

The doc's per-layer budgets (1800/700/700/1000/600/800 = 5,600) were rounded to tensor-friendly
shapes with the user's agreement. Every wiring in the doc's table is preserved exactly:

| Layer | receives (doc) | shape | params |
|---|---|---|---|
| L1 | accepted external information | 128 → 16 | 2048 |
| L2 | L1 + returned L4 | (16 + 16) → 24 | 768 |
| L3 | L1 + accepted L2 messages from neighbours | (16 + 24) → 16 | 640 |
| L5 | L2 + L3 | (24 + 16) → 16 | 640 |
| L6 | L5 + L4 | (16 + 8) → 32 | 768 |
| L4 | L6 → back to L2 and L6 | 32 → (16 + 8) | 768 |
| | | | **5632** |

5,632 nibbles = 2,816 bytes = exactly 44 cache lines per unit, no padding. At 200,000 units the
durable store is 563 MB (doc: ~560 MB). Layers are laid out in execution order (L1, L2, L3, L5, L6,
L4) so an early death touches only the leading lines of the block. `Dims` in `config.h`.

L6 has 32 cells, so for the 32-symbol task the first V cells are *designated output cells* and the
prediction error lands on them directly (doc §9). For 256-symbol byte streams a small shared FP32
readout (32×256, part of the workspace, not a unit synapse) maps L6 to logits; the L6 error is the
one-step local projection of the readout error. No traversal back through layers or time.

## Event flow (doc §3, runbook §1, §4)

`Organism::process_event` executes, in dependency order with an exit after every stage:

1. `make_key` → 64-bit key: bits 0–7 current symbol (sensory), 8–31 the last three symbols,
   32–63 a hash of the last four. Bits 8–63 are the 56 candidate tumblers.
2. `tree.walk` (≈ depth branch decisions) → leaf → `tree.match` (1–4 lock checks).
3. No candidate → birth a lock `(value=key, mask=sensory)` and a fresh route body (doc §5).
4. Admitted units go on the `cur` queue. `run_pass` drains `cur`/`nxt` for up to `max_hops`.
5. `run_body`: L1 → die? → L2 and L3 → cheap K-neighbour lock tests (accepted neighbours are pushed
   to `nxt`, they do **not** execute here) → die? → L5 → die? → L6 → emit to workspace → L4 only
   if `level(l6) >= recirc_thresh` (recirculation active); the L4 output is stored on the unit and
   consumed by that unit's L2/L6 on its next admission.
6. Readout → committed prediction → outcome (next symbol) → loss.
7. Local learning, metabolic pressure, associative-lock pressure, lock-local split evidence.

Death test: mean |activation| of the layer below `max(base, ema × death_rel/64) × resist/64 ×
(novel ? ½ : 1)`, where `ema` is the unit's own running activity at that stage (`act_ema`, per
unit per stage). `death_rel=0` (default) keeps the absolute floor only; see README for why.

"Novel" is true when any admitting lock is unfamiliar (`hits < novel_hits`) **or surprising**
(`err_ema > stream_wrong_rate + surprise_thresh`: the lock fails clearly more often than the
stream as a whole — relative, so that a uniformly hard stream such as raw bytes does not keep
every lock "surprised" forever, which would defeat the *temporary* in doc §7 and keep the
learning rate un-annealed). Both buy computation,
as doc §7 requires: thresholds halve, and a unit that dies during such an event has its resistance
*lowered* rather than raised. Without the surprise half of this rule a route body whose random L1
responded weakly to its own key died on every event, was charged for it, and could never recover
(observed as exactly 7/8 accuracy on a deterministic 8-cycle before the fix).

## Motifs and K (doc §4, runbook §6)

Units are grouped by id into motifs of 128; motif *m* uses `k_set[m % 6]` ∈ {2,4,8,16,32,64}. A
unit's K eligible neighbours are the K nearest ids inside its motif (ring). L2 does only the
`((route_key ^ assoc_value) & assoc_mask) == 0` test on each; the route key is the sign/magnitude
pattern of the L2 vector plus a hash. Each unit has its own associative lock, born with
`log2(K)+1` random mask bits so that expected acceptance is O(1) per event regardless of K; it
tightens (one more masked bit, set to reject the useless sender) after `assoc_tighten_after`
consequence-free accepts and loosens after `assoc_loosen_after` novel events with no messages.
Duplicates within an event are suppressed with a per-unit epoch counter.

## Addressing and structural learning (doc §5–6, runbook §3, §9)

`AddressTree` is the flat SoA tree from the runbook (`bit[]`, `left[]`, `right[]`,
`leaf_offset[]`, `leaf_count[]`, one lock-id pool). A leaf only splits on a bit that is masked in
every lock it holds, so a lock lands on exactly one side; a lock born later with a narrower mask is
inserted into every consistent leaf. Leaf maintenance runs between blocks.

Split evidence lives on the ambiguous lock (`Lock::evidence` → `EvidencePool`), never on the
incoming family (doc §14). A lock starts collecting when its wrong-rate EMA exceeds
`conflict_thresh`; for each candidate tumbler b it keeps H0(y)=P(y|lock,b=0), H1(y)=P(y|lock,b=1)
over the raw next symbol (bucketed). At block end the lock splits on the bit with the largest
G-statistic (2Σ n ln(n/e) — divergence measured against sampling variation) if both branches have
`split_min_evidence` samples and G exceeds `split_g_per_df × (buckets−1)`. Identical keys never
split (one branch never collects evidence). The child inherits a *copy* of the parent's route body.
The doc's continuous form (‖μ0−μ1‖/(σ0+σ1)) is not implemented; the discrete prototype form is.

## INT4 synapses and local accumulator (doc §8, runbook §2, §5)

`apply_acc`: `e ← e + Δ`; if `e ≥ half_quantum` and w < 7 then `w++`, `e −= quantum`; symmetric
downward; `e` is INT8 per synapse, allocated only when training. The accumulator never enters
inference. Kernels: `layer_scalar` (reference), `layer_avx2` (16 nibbles → int16 madd against a
de-interleaved activation copy), `layer_expand` (unpack the active unit's layer to INT8 scratch,
then INT8 dot). `test_kernels` proves all three produce identical INT32 sums.

## Learning rules (doc §9, runbook §8)

* L6: supervised local prediction-error update inside the admitted route bodies (the rule the 200k
  synthetic organism actually used). `hidden=none` (default).
* Hidden layers: `hidden=perturb` — node perturbation with a shared end-of-trial reward difference.
  Pass A runs with +δ on every hidden output and builds the schedule; pass B replays the *same*
  schedule with −δ (runbook §8: batch the ± passes on the same admitted units). Δw ∝ −(L₊−L₋)·σ·x
  only on traversed synapses. Committed prediction is the mean of the two workspaces, or an
  unperturbed third replay with `perturb_clean=1`.
* Scaled deltas are truncated toward zero (`scale_trunc`), never arithmetic-shifted: flooring
  turns every small positive product into −1 and every small negative one into 0, a one-way
  drift that made the hidden learner destructive and quietly biased L6 too.
* Prediction is committed before the outcome is observed (doc §14, "random futures can cheat").
* Consequence = loss ≤ running loss EMA. Consequential emission lowers a unit's resistance,
  consequence-free emission raises it, dying after consuming compute raises it slightly (doc §7).

L4 receives no direct error signal in either mode (its consequence is one event later); it is
trained only through inheritance on split. The design doc's own 200k result did not train hidden
layers at scale either; this is preserved as a stated limitation, not hidden.

## Scheduler (runbook §7)

`Event {unit, source, hop, src_trace}` is 8 bytes. `cur`/`nxt` are preallocated vectors; the
learning traces are a preallocated array of `max_active` records; no heap allocation on the hot
path (lock births allocate, they are rare and counted).

## Telemetry (doc §13, runbook §12)

`BlockStats` carries every listed counter; `report()` prints one line per block, `json()` writes a
JSONL log (`log=path`). Stage time uses `rdtsc` around each stage.

## Bench and milestones (runbook §11, §14)

`laminar_bench mode=familiar|moderate|storm|pathological` seeds a random population over the real
563 MB store and reports KB/s together with MACs/event, weight bytes/event and admissions/event.
`laminar_synth` = milestones A/B/C (`units=200000` for C). `laminar_bytes` = milestone D with the
three required comparisons (`mode=adaptive|frozen|nomem`).

## Working memory (added after the reconstruction)

The organism as specified has almost no state across events: the workspace is zeroed every event,
the key sees the last four symbols, and the only carried state is a 24-byte L4 return vector per
route body, consumed when that route next fires. Three mechanisms were added, all parameterised:

* **Persistent workspace** (`ws_decay`, `ws_input`, `ws_key`): `ws_mem ← ws_mem·decay/256 + ws`
  after every event; fed into L1 input cells 96..127 (`ws_input`, replaces half the symbol code)
  and/or into key bits 48..63 as a sign pattern (`ws_key`). `ws_decay=128 ws_input=1` is now the
  default: TinyStories BPB 2.90 → 2.76. `ws_key` is neutral.
* **L4 return into the next key** (`l4_key`): sign pattern of the last L4 output in key bits
  32..47 so a route's consequence can select the next route. Neutral to slightly negative.
* **Longer hashed history** (`hist=N`, up to 15): neutral on TinyStories — the limit is not the
  n-gram width.

Episodic (exact-key) bodies ignore `ws_input` and the L4 return so that their input is a pure
function of the key.

## Rolling context: episodic locks (the "2M-token window")

The lock/key fabric is a content-addressable store; what it lacked was one-shot entry. With
`episodic=1`:

1. **Birth on surprise.** When the committed prediction is wrong and its loss exceeds the
   stream's running loss by `episodic_margin`, and no exact lock already matches, a lock with
   `mask = all 64 bits` is born on the current key (the current symbol plus `hist` previous
   symbols), with a copy of the admitting route body.
2. **One-shot imprint.** The new body is run on its key-determined input; its L6 rows are written
   directly as INT4 `round(7·g_j/max|g|)·sign(x6_i)` where `g = −dl6` is the readout's matched
   filter for the outcome, refined perceptron-style (`imprint_iters`) until the body's L6 output
   decodes to the outcome through the shared readout. 98–99% decode at birth on TinyStories.
3. **Admission.** `exact_only=1`: an exact-key candidate silences generic ones (best recall);
   otherwise all candidates emit, weighted by `exact_gain` (better BPB, ~60% recall).
4. **Rolling eviction.** When free units fall below a reserve (≥ 2 blocks of births), the
   least-recently-hit exact locks are evicted between blocks. The unit pool *is* the window: its
   length in bytes ≈ units / episodic-birth rate.
5. `readout_freeze=N` stops the shared readout's learning after N events so imprints stay
   decodable across the whole window.

Acceptance test `laminar_needle`: `\n@KEY=VALUE\n` needles (6+6 random letters) are planted on a
4160-byte grid in a non-repeating TinyStories haystack and each queried once, `d ∈ {W/8 … 2W}`
later, on the same grid offset by 2080 bytes (no plant/query or query/query collisions for
distance differences below 65·W/8). Recall = accuracy on the VALUE bytes at query time.

Results (13.6 MB haystack, W = 2,097,152, 400 needles, `hist=7 exact_only=1 readout_freeze=512K`):

| pool | births | evictions | recall at 0.125–1.0 W | recall at 1.25–2 W |
|---|---|---|---|---|
| 800k units | 476k | 0 | 77–83% at every distance (2.1 MB → 4.2 MB) | 77–79% (not yet evicted) |
| 200k units (the doc's population) | 571k | 385k | 70–76% up to 1.0 MB, **0% from 1.5 MB** | 0% |
| **350k units** | 492k | 158k | **70–79% at every distance through 2.1 MB** | 74% at 2.6 MB, 44% at 3.1 MB, **0% at 4.2 MB** |

The 350k row is the target configuration: a rolling window of ≈2.5–3 MB with the LRU edge
visible beyond it. Window length scales linearly with the pool (≈ units / 0.1 on unseen text).

Chance is 3.8%. Recall is distance-independent inside the window and exactly zero outside it —
the signature of content-addressed memory with LRU eviction, which is what "rolling context"
means here. It is associative recall, not attention: the organism can complete a key it has
seen once, anywhere in the window; it cannot compare or aggregate across positions.

Costs: episodic bodies are ordinary units (2,816 B); admitting one costs a full body per event
(6–11k MACs/byte); throughput 180–270 KB/s during the test. With `exact_only=1` the generic
predictions are silenced whenever an exact lock fires, which costs BPB on ordinary text
(3.2–3.4 vs 2.8); the summing mode keeps BPB (2.6–3.0) but recalls only ~60%.

Not mirrored in the NumPy reference (native only); the default configuration (`episodic=0`)
still matches the reference bit-for-bit.

## Attention: the Agkern agency kernel, ported and wired in

`cpp/include/laminar/agency.h` is a plain-C++ port of `projects/Agkern/agkern/*.py`
(kernel, competition, prediction, plasticity, traces, signals, motifs, expansion, focus), keeping
the architecture's runtime loop (section 8.1) in the same order and each step on the same native
primitive: fixed sparse k-WTA expansion → per-tendency consequence routes → bilinear drive motif →
one shared inhibitory pool → commitment with the six release rules → emission → bounded
eligibility → DAN-gated three-factor credit. The invariants hold structurally: the expansion never
learns, the pool has no dopamine input, commitment never sees sensory surprise, one credit event
touches `k_active × consequence_dim` routes, and there is no backward pass.

**Wiring (`agency=1`).** Each byte is one event.
* *Observation* (64): the persistent workspace (32, tanh-scaled), current and previous symbol bits
  (16), and the organism's own signals (running loss, wrong rate, novelty, exact-lock matched,
  candidate count, last p[target], learning progress, sensory error).
* *Tendencies* are focus operations, the source's own idea of "a rule that moves the parameter":
  `hold`, `toward_error`, `sweep`, `to_recent` / `to_context` / `to_symbol` / `to_memory` (move
  the Gaussian focus centre over the 128-cell L1 input to that region), and `attend_episodic` /
  `attend_generic` (the discrete attended stream: whether an exact-key lock silences generic
  candidates). The `FocusField` gain (floor 0.25, width 24) multiplies the L1 input of generic
  bodies; episodic bodies keep their key-determined input.
* *Consequence* (4 channels): goal 0 = `1 − loss/ln V` (`agency_goal=1`, aligned with BPB; `0`
  gives p[target]), goal 1 = the kernel's own learning progress (epistemic channel, written by the
  kernel), effort = 1 (so `1 − predicted effort` is the novelty readout), and correctness.
* The kernel deliberates before the bodies run, so its committed focus shapes *this* prediction;
  the outcome arrives with the next byte and credits the eligible routes.

**Port decisions.** Baseline persistence cost `agency_cost` defaults to 0 here: the source's 0.10
is sized for 0/1 outcomes, while contrasts between focus settings live in hundredths of p[target],
so with the original cost every commitment was "invalidated" at exactly `patience` events (measured:
mean age 3.0). With cost 0 a focus persists unless it demonstrably hurts. The pool's `beta` is
scaled to keep the pooled inhibition at its K=4 level for larger K; the default tendency sets are
K=4, which is what the source tuned. `hold` is the default tendency, so its effectiveness is 0 by
construction (the no-op contrast, as in the source).

**What it does (measured).**

| setting | result |
|---|---|
| TinyStories 100 KB, agency off / control (fixed bump, `hold` only) / on (K=4 focus set) | BPB 2.90 / 2.92 / 2.95; on: `hold` 88%, mean commitment 39.5 events |
| Needle 256 KB, episodic memory, agency off (sum) / on (K=4 attend set) | recall 53–68% / **74–81%** at every distance; BPB 2.59 / 3.28–3.39; on: `attend_episodic` 21–23% vs `attend_generic` 9% |
| fixed `exact_only=1` for comparison | recall ~80%, BPB 3.6 |

So the kernel directs where directing changes outcomes: it learns to attend the episodic stream,
buying most of the recall of the fixed rule while giving back part of its BPB cost, and on calm
text it learns that moving focus does not pay and holds. Where nothing it can do changes the
outcome, drives are equal, the pool is noise-driven and it explores — the source's behaviour in the
same situation ("outcompeted" releases dominate, mean age ~3–6). Cost: ~3 µs/byte
(130–180 KB/s with the kernel vs 220–520 without); the focus gain and the 512×6 expansion are the
obvious targets, not yet optimised.

## Long streams: the readout-drift finding

On a 128 MB pass of TinyStories the baseline **got worse with more data**: window BPB 3.16 →
4.86, accuracy 52% → 19%. One-factor ablations on a 32 MB slice (`scripts/drift_ablation.sh`,
window BPB at 8 / 16 / 24 MB):

| variant | 8 MB | 16 MB | 24 MB |
|---|---|---|---|
| baseline | 3.17 | 3.69 | 3.89 |
| **readout frozen at 4 MB** | 3.05 | 3.05 | **3.03** (accuracy rising 54.6 → 57.7%) |
| readout frozen + no neighbour routing | 2.97 | 2.93 | **2.87** |
| no neighbour routing alone | 3.09 | 3.56 | 3.79 |
| workspace off | 3.24 | 3.71 | 3.92 |
| no annealing | 3.26 | 3.78 | 4.02 |
| L4 recirculation off | 3.76 | 3.97 | 4.03 |
| no splits | 4.14 | 4.48 | 4.69 |

The cause is the shared FP32 readout (32×256) that maps L6 to logits. It kept learning at a fixed
rate, so the INT4 bodies' L6 codes — fitted through one readout, then annealed — were scored
through a different one later: a moving target the bodies could not chase. With the readout
frozen after 4M events nothing drifts; `readout_freeze=4194304` is now the default. Splits and
L4 recirculation are load-bearing (removing either costs ~1 bit/byte); neighbour routing costs
~0.15 bit/byte on plain text at this scale (routed bodies add noise to the workspace) and is kept
because it is part of the design, not because it pays here.

Reference points on the same stream (online byte n-gram, Witten-Bell, `python/ngram_baseline.py`):
order 3 ≈ 1.84 BPB, order 5 ≈ 1.39, order 7 ≈ 1.20. The organism at ~3.0 sits between an order-2
and order-3 byte model: its contexts are the locks (8k after 24 MB, growing by evidence-gated
splits), whereas an order-5 model has millions. Capacity of *addresses*, not of bodies, is what
limits it on language.

## Motif packing and the K screen

Route bodies are now allocated motif by motif (`kpack=1`): each K class has a fill cursor, a split
child is placed in its parent's motif while there is room, so a motif's neighbourhood is a family of
related routes rather than random strangers scattered over the pool (before, a K=64 unit had ~4
live neighbours to test; now its motif is dense). Every unit in a motif has the motif's K.

`kscreen=1` chooses the K class of a new route by measured benefit for the kind of input it was
born on: a table of consequence EMAs per (current symbol, K class), updated from every emitting
body, argmax with 10% exploration. Consequence for a *routed* (neighbour-admitted) body is now
counterfactual — its emission is credited only if reading the workspace without it gives a higher
loss — and that per-body credit also drives the associative-lock tightening and the metabolic
resistance (design doc §7 "predictive consequence preserves routes", made literal).

Measured on 32 MB of TinyStories (`scripts/kpack.sh`):

| placement | K choice | accepts/byte | BPB | what the screen chose |
|---|---|---|---|---|
| random (old) | random | 0.14 | **2.97** | — |
| packed | screen | 0.64 | 3.34 | **K=2 for 98% of births** |
| packed | random | 0.98 | 3.63 | — |

(Counterfactual credit alone improved the unpacked baseline 3.05 → 2.97.) The screen works as
designed and its verdict is that neighbour routing, as it stands, is net-negative on plain text:
packing makes the eligibility radius real, routed bodies then emit ~1×/byte, and only ~60% of
those emissions help. The screen minimises K in response. Routing needs a harder gate than the
current tighten-after-16-consecutive-failures rule before dense motifs can pay; that is the next
thing to build, not a tuning knob.

## Where the organism stands on language (128 MB of TinyStories, one pass)

| configuration | window BPB at 16 / 64 / 112 MB | accuracy | KB/s |
|---|---|---|---|
| baseline, readout live (before the drift fix) | 3.64 / 4.42 / 4.76 | 47 → 20% | 330 |
| baseline, readout frozen | 3.07 / 3.06 / 3.08 (plateau from 32 MB) | 57–58% | 360 |
| + episodic memory (600k pool, sum mode) | 3.64 / 3.94 / 3.98 | 48 → 42% | 128 |
| online byte n-gram, order 3 / 5 / 7 | 1.84 / 1.39 / 1.20 | — | — |

What is missing, in order of leverage:
1. **Address capacity.** The organism plateaus at ~14k locks; splits fall from 6k to 0.6k per
   16 MB because each needs 64 samples per branch and a G-test. An order-5 byte model has
   millions of contexts. Evidence-gated splitting is the doc's rule and it is right about noise,
   but on language the rule needs to admit contexts faster (cheaper evidence for high-entropy
   locks, or splitting on several bits at once) or the fabric needs a second, cheaper address
   mechanism for frequent contexts.
2. **Routing that pays.** See above: dense motifs are ready, the routed bodies are not.
3. **Episodic memory is recall, not modelling.** It delivers the 2M-byte needle window but on
   unseen text it costs 0.9 bit/byte and 3× the compute; it should be gated by the agency
   kernel's attended stream, which is what the kernel learned to do on the needle task.
4. **Per-lock predictors are constants.** With one route body per byte and a near-constant
   target per lock, the six layers do the work of a lookup; the hidden layers matter only when a
   body must generalise across the contexts its lock admits.

## Generation: composition and serialization (the goal)

The organism as a next-byte predictor could not generate: fed its own output it produced letter-soup
(42% real words) and recalled nothing planted. Two things a transformer buys with attention were
missing — composing many partial matches, and serializing a recalled continuation in order.

**Sequence memory** (`cpp/include/laminar/sequence.h`), after the Atomic Thought model's sequences
(Brain Simulator Thought white paper §2.4–2.5, `UKS.Sequence.cs`) and ALM's route lifecycle:

* *Capture everything.* The byte stream is an explicit chain of steps (value, NXT) in a rolling
  window (`seq_window`, 32 MB default). Nothing is filtered at capture time.
* *Reverse links, bounded.* For each lag in the receptor ladder {2,3,4,6,8,12,16,24} a hash of the
  bytes preceding a step maps to the most recent steps with that context (6 per bucket). That is
  the paper's "locate all sequences containing this element by reverse value links", bounded the
  way ALM bounds its route cache.
* *Sequence search = prediction.* At each event, for the longest lag with a verified match, every
  remembered step whose preceding bytes equal the current ones proposes its NXT value, weighted by
  lag and by how far the real match extends (up to 64). Continuations compete; support saturates
  as `total/(κ+total)` (ALM's normalized competition: weak support → weak influence).
* *Composition.* The bodies' distribution and the sequence continuations are combined as a
  geometric blend weighted by that confidence (`seq_alpha × confidence`). No parameters are
  learned; capture → compete → consolidate (counts) → prune (window eviction).
* *Serialization* emerges: the longest match at step *t+1* is, with high probability, the
  continuation of the same remembered sequence, so emission chains along it byte by byte while the
  bodies keep the right to disagree. A hard "bubble" commitment (follow a ≥16-byte match for 32
  bytes as a unit) was built and measured: 6.6M commitments, BPB 4.76, degenerate `pplppl` runs —
  the same failure ALM's Phase 100 documents when recalled content overrides the loop. It is off.

Measured on 32 MB of TinyStories, 100k units (`scripts/generate.sh`, `laminar_generate`):

| configuration | BPB | free-run words valid | loops | planted recall (32 cues, 6-byte phrases) |
|---|---|---|---|---|
| bodies only, T=1.0 / 0.7 | 2.70 | 42% / 63% | 0 | 0/32 |
| + sequence memory, T=1.0 | **2.05** | 78% | 0 | 17/32 exact, 69% of bytes |
| + sequence memory, T=0.7 | 2.05 | **99%** | 0 | 17/32 |
| + sequence memory, greedy | 2.05 | 100% | 0 | 17/32 |

Samples at T=0.7: *"One day, the green octopus got sick. His friends were sad. The green octopus
began to suffer."* — *"He had nowhere to hide so he quickly ran back to his cottage. But when he
arrived, the door wa…"*. The harness also reports how much of the past is being copied verbatim
(mean match length behind each generated byte, and the fraction of bytes inside a ≥32-byte
verbatim match); greedy decoding is expected to copy, sampling to recombine — the number decides.

**The strict bar and the two rules that clear it.** The bar set for "success" was free generation of
at least 250 words with no loop, judged strictly: any span of ≥64 bytes recurring inside the text,
or more than 30% repeated 8-grams, is a loop; and the harness reports how much of the text is
verbatim copying of the window (mean match length behind each byte; fraction of bytes inside a
≥32-byte match). Greedy decoding fails it outright (8/8 loops, 96% copying) — retrieval, not
generation. Two rules from the source architectures make sampling clear it:

1. **Provenance** (Agkern: a self-generated event never grounds learning). The organism's own
   emissions are captured as *context* for later matching but flagged so they are never recalled
   as *instances*. Without this a generated span becomes an instance that matches itself and the
   loop is self-sustaining (2–3/8 loops, one prompt cycling 762 B); with it, 0/8.
2. **Bubbling** (`UKS.Bubble.cs`: attributes shared by enough children rise to the class). A lag
   level is used only when ≥`seq_min_support` remembered instances back it, so fluency comes from
   what several instances agree on rather than from one transcript.

| T=0.7, 32 MB, 8 prompts × 2,048 bytes | words | valid | loops | verbatim copying | planted recall |
|---|---|---|---|---|---|
| instance level (support 1) | 376 | 98% | 0/8 | 74% (mean match 48 B) | 17/32 |
| **consensus, support 2 (default)** | 380 | 98% | **0/8** | **11.7%** (mean match 19 B) | 18/32 |
| consensus, support 3 | 384 | 97% | 0/8 | 5.9% | 16/32 |
| greedy, any support | 376 | 100% | 8/8 | 96% | 17/32 |

At support 2 the organism writes ~380 words with no loop, 98% real words, a fifth of the words
unique-per-position as in the source text, ~12% of bytes inside verbatim spans, and still completes
a planted `@KEY=` cue with its 6-byte value 18/32 times. With a 128 MB window BPB reaches 1.92.

Where the agency kernel enters next: free-running has no verified outcome, so the kernel cannot be
credited by prediction there. Its natural role is the one the source built it for — commitment to
a source (memory vs bodies vs episodic) and to a temperature per condition, credited by the
organism's own evaluation of what it emitted (ALM: "generate and score").

### Bounded memory: the consolidated tier (`seq_consolidate`)

Recall from the instance window is flat with distance (85–91% at every distance up to the window
edge, `seq_bits=23`), so the window can be as large as memory allows — but a window is exposure-
bounded, not content-bounded. The lifecycle's *consolidate* step gives the sequence memory a second
tier whose size is fixed: per lag and context, a tagged entry of `CSLOT=4` continuations with
saturating counts (8 lags × 2^`seq_cbits` entries × 16 B; 128 MB at 20 bits). Observed outcomes
bubble up into the counts; self-generated ones never do (provenance). At prediction the class counts
join the instance competition at their lag level, weighted by the count share.

The first version had no tag and spoke noise: at lag 24 nearly every context is unique, so 32 MB of
stream puts ~32 unrelated contexts in each of 2^20 buckets, `csum` always clears the support bar,
and the longest level "decides" with garbage (BPB 5.78, 22% real words). Instances are verified
against the ring; class counts must be verified too. Each entry now carries a fingerprint of the
context that owns it, a different context ages the entry and may take it over only when its counts
have decayed to nothing (the same competition as within the entry), and a count is read only for
the context that wrote it.

| T=0.7, 32 MB TinyStories, 100k units | BPB | valid | loops | rep8 | copying | planted recall |
|---|---|---|---|---|---|---|
| 32 MB instance window (no tier) | 1.936 | 98% | 0/8 | 0.054 | 11.7% | 18/32 |
| 1 MB window, no tier | 2.182 | 93% | 0/8 | 0.043 | 5.7% | 0/32 |
| 256 KB window + tier (class gain 1) | 1.755 | 100% | 1/8 | 0.136 | 6.2% | 23/32 |
| 1 MB window + tier (class gain 1) | 1.778 | 100% | 0/8 | 0.116 | 8.1% | 24/32 |
| 256 KB window + tier, class gain 0.5 | 1.741 | 100% | 0/8 | 0.137 | 5.9% | 17/32 |
| 256 KB window + tier, gain 0.5, T 0.8 | 1.741 | 98% | 0/8 | 0.117 | 5.5% | 17/32 |

On wiki-en (`C:\path\to\wiki-en`, 32 MB, the same bounded configuration) BPB is 2.67 — encyclopedic
text is far harder than TinyStories — with 0/8 loops, 98% real words, 3.6% verbatim and planted
recall 10/32.

On wiki-en 512 MB with the same bounded configuration (1 MB window, 22-bit tier, 100k units) BPB
is 2.609, 99% real words, 320 words per prompt — but 2/8 prompts loop (337 B) and planted recall is
**0/32**: over 512 MB each tagged bucket is contested by ~128 contexts per lag and rare facts are
aged out by frequent ones. The tier is a *consensus* memory; a rare fact seen four times in 512 MB
survives only in the instance window or in episodic locks. Bounded memory answers "what the
language does", not "what was said once" — the two stores are complementary, not interchangeable.

At 128 MB of stream, a 1 MB window plus the tier at 22 bits (512 MB of counts) reaches BPB
**1.631** against 1.92 for a 128 MB instance window, with 0/8 loops, 100% real words, 11% verbatim
and planted recall 19/32.

A 256 KB instance window plus the tier beats the 32 MB instance window on every count but
repetition (BPB 1.75 against 1.94 and against the 1.84 n-gram reference; planted recall 23/32
against 18/32 — the planted phrases are recalled from *counts*, as classes, not from instances).
The price is more repetition (rep8 0.12–0.14 against 0.05): consensus from counts is stronger and
more deterministic than consensus from a handful of instances. Lower class gain trades recall for
diversity. Bounded memory is now the default (`seq_window` 4 MB, `seq_consolidate` on at 20 bits,
class gain 0.5) together with a third provenance rule, **don't repeat yourself**: when the current
context (≥ 8 bytes) already occurred in the organism's own output, the memory's semisaturation
rises with the number of such recurrences and the bodies get to vary. With it the strict test passes
on both seeds (0/8 loops, 390 words, 99% real words, ~9% verbatim, planted recall 13–18/32) where
the tier alone looped once in 8 (68 B). This is the mechanism that lets the organism keep a large *content* memory with a small
*text* memory — the window is for episodes (relations between thoughts), the tier for what the
language does.

## Goal-directed generation: relations between thoughts, commitment to an episode

The test: train on a chat stream, then for held-out dialogues feed `USER: <turn>\n\nASSISTANT:`
and free-run the reply (`laminar_chat`, `scripts/ontask.sh`). *On-topic* is the fraction of the
reference reply's content words (≥4 letters, stop words removed) that appear in the generated
reply, and the same against the user turn; *chance* is the same score for a reply generated for a
different, shuffled prompt. Without steering the organism is fluent and off-topic: 0.011 against
chance 0.005 — it copies a well-formed reply from some unrelated episode.

**Bubbles: the relation store.** Content words are thoughts. Each dialogue in the window keeps the
set of thoughts it contains (`BUBBLES=48` word hashes) — the relations between thoughts within an
episode — and a reverse index from each thought to the most recent episodes containing it
(`IDX_SLOT=8`), with a document frequency so that rarity is information (IDF). The current
dialogue's thoughts are the GOAL set.

Three designs were tried, and the first two failed in instructive ways:

1. *Weighting instances by goal overlap* (`topic_gain` multiplying every remembered instance whose
   episode shares goal words): 94.6% of instances qualified — most episodes share common words —
   so nothing was selected; on-topic stayed at chance and BPB worsened.
2. *Retrieval by relation, aligned at short lags*: the rarest goal thoughts select the related
   episodes; continuations at the positions where the current 4–8 byte context recurs inside them
   join the competition. Selection now worked (4× chance once) but alignment inside several
   episodes at once *splices* them — fragments of different replies, and the scripted opener of
   every episode ("Hello! How can I help you today?") as the most frequent alignment.
3. **Commitment** (Agkern): the goal is frozen at the user's words when the reply begins; the
   rarest of them select the single most related episode; within it the reply that *answered the
   shared thought* is located (the assistant turn after the first occurrence of a goal word — the
   first assistant turn is the opener); the organism then follows that reply as one instance,
   re-aligning on the current context when the bodies diverge, and drops the commitment only after
   sustained mismatch, when the next most related episode takes over.

| SmolTalk 1.7 MB, 40 held-out dialogues, T=0.7 | on-topic vs ref (chance) | vs prompt (chance) | valid | reply |
|---|---|---|---|---|
| no steering | 0.011 (0.005) | 0.003 (0.017) | 93% | 175 B |
| design 1, gain 8 | 0.014 (0.012) | — | — | — |
| design 2, gain 4 | 0.013 (0.020) | 0.037 (0.032) | 96% | 166 B |
| **commitment, gain 4** | **0.070 (0.003)** | **0.095 (0.038)** | 99% | 184 B |
| **commitment, gain 16** | **0.118 (0.016)** | **0.152 (0.031)** | 100% | 137 B |
| LMSYS 128 MB, 100k units, no steering | 0.009 (0.012) | 0.011 (0.009) | 95% | 356 B |
| **LMSYS 128 MB, commitment, gain 4** | **0.078 (0.014)** | **0.122 (0.023)** | 94% | 335 B |
| LMSYS 128 MB, commitment, gain 16 | 0.077 (0.011) | 0.103 (0.027) | 94% | 320 B |
| SmolTalk, **two turns** (first exchange fed as context, reply to the second user turn scored), no steering | 0.013 (0.022) | 0.014 (0.008) | 99% | 191 B |
| SmolTalk, two turns, commitment, gain 8 | **0.074 (0.018)** | **0.136 (0.021)** | 99% | 139 B |

"I'm looking for upcoming virtual concerts" → *"Yes, virtual concerts have gained popularity,
especially during the pandemic. Many artists have performed online, and it's a great way to enjoy
live music from anywhere."* That is what the mechanism is: the goal selects a remembered episode
by the relations between its thoughts, and the reply is grounded in what was said there, with the
bodies and the consensus tier filling in and the organism free to leave it. It is honest to say
this is retrieval-guided composition, not reasoning; it is also exactly the step from "coherent
text" to "text about the thing that was asked", and it uses the window as content rather than as
exposure. What it does not yet do: choose *which* episode to commit to by expected benefit (the
agency kernel's consequence routes and utility), or hold a goal across several turns.

Cost is bounded: selection scans ≤6 episodes × 6 KB once per commitment, following is one aligned
read per byte, and it runs only while generating (`topic_active`), never during training.

Two follow-ups on the selection itself:

* **Goal extension** (`seq_goal_extend`, a bubble-sequence step: when a commitment is dropped, the
  thoughts emitted so far extend the goal for the next selection) is inert at reply length — a
  400-byte reply never drops its commitment (0 drops in every run), so the rows are identical.
  Chaining thought to thought needs longer generation or a turn boundary; the two-turn result
  above is the form it takes today.
* **Selection by consequence** (`seq_util_rate`, the agency kernel's idea in the relation store):
  each thought that selected an episode is credited with how well the commitment held (followed
  bytes against re-alignments and misses, an EMA per thought), and selection weight becomes rarity
  × (0.5 + utility). SmolTalk, 42 dialogues, gain 8: on-topic 0.050 → **0.068** (chance 0.014 /
  0.016), re-alignments 83 → 74. LMSYS 128 MB, 343 held-out dialogues (≈740 commitments): on-topic
  0.026 → **0.034** against chance 0.021 / 0.016, re-alignments 2,047 → 1,729, drops 77 → 52 —
  the same modest, consistent gain: commitments chosen by consequence hold better.

## Reasoning over supporting facts: relations within the episode (bAbI)

After goal-directed generation the next phase is reasoning, and it is hard to instrument at small
scale because every metric is confounded by retrieval unless the answer is known and unreachable by
copying. bAbI (Weston et al.; the `Muennighoff/babi` mirror, since the original host and the HF
loader script are dead) gives that: short stories, a question, a one-word answer that needs one,
two or three supporting facts. `python/babi_stream.py` turns each training row into an episode
(the passage, the question as a `USER:` turn, the answer as an `ASSISTANT:` turn) and
`laminar_babi` scores exact matches on held-out stories (`scripts/babi.sh`).

**Baseline** (the organism as it stood, 50k units, tasks 1/2/3/15/16, 200 questions each): every
task at chance. BPB on the stream is 0.24 and only 171 locks form — the language is trivial — and
at `ASSISTANT:` the instance memory sees a generic context and the consensus emits the most
frequent location. Cross-episode commitment fires on every question (800 commitments) and does
not help: it selects *other* stories, while bAbI's supporting facts live in the *current* episode.

**The mechanism** (`seq_chain=1`): relations *within* the episode, as bubble sequences.

* Each sentence of the current episode is kept as a sequence of thoughts (its content words, with
  their ring positions) plus its *relation* word (the first non-thought word after the subject:
  "went", "dropped", "afraid"). Content words are those that are not function words or bare verbs
  of motion/transfer.
* At the reply boundary the question's most specific thought (the question word that occurs in
  the fewest earlier sentences: "Daniel", "football", "Bernhard") selects the most recent sentence
  relating it — the thought as *subject* wins over the thought as object, exact word before stem
  ("wolf" ~ "wolves"). If the thought is the subject, the candidate is the sentence's object; if
  it is the object, the candidate is the subject (the atomic-thought triple, read either way).
* The candidate becomes the thought of the next hop, up to three deep — a chain of hops through
  relations between thoughts: `football → John → kitchen`; `Bernhard → swan → Lily → white`.
* Which depth answers which question *form* (the question's non-thought words: "where", "what
  color", "what afraid") and which words are answer-like for that form are learned **by
  consequence**: at training time the observed answer credits the depth that produced it and the
  (form, word-stem) pair. Selection is answer-likelihood × the form's depth prior.
* Two learned preferences inside a hop: among a subject's sentences the hop prefers the latest
  whose object is answer-like for the form (for "where", the location relation rather than the
  latest object John touched); and the second hop may look *before* the first sentence rather
  than after it — the holder's location at a drop — with the direction learned per (form,
  relation): "dropped" → before (+0.99), "took" → latest (−0.99).
* Generation seeds the chosen candidate's first bytes with a weight relative to the rest of the
  competition and then lets the language finish the word ("wolves" → the answer form "wolf").

The chain is a legible chain of thought; the harness prints it per question:
`Where is the apple? → "office"  [thought=apple d1=Daniel d2=office d3=Mary (alt d2=kitchen dir 0.986) chosen=d2]`.

| bAbI (en-10k), 50k units, 200 held-out questions per task | baseline | chain | chance |
|---|---|---|---|
| qa1 single supporting fact | 15.0% | **100%** | 18.0% |
| qa2 two supporting facts | 9.5% | **100%** | 18.0% |
| qa3 three supporting facts ("before") | 17.0% | 41.0% | 22.5% |
| qa15 basic deduction | 38.5% | 18.0% | 38.5% |
| qa16 basic induction | 23.5% | **100%** | 25.5% |

All twenty tasks trained together (100 held-out questions each): qa2 100%, qa16 100%, qa20 70%,
qa1 74% (the "where" form is now shared with the coreference tasks 11–13), qa4/qa5/qa8 ~50%,
qa12 35%, and **0% on every task whose answer is not a thought in the story** — yes/no (qa6, 7,
9, 10, 17, 18), counting (qa7), path directions (qa19). That is the honest shape of the mechanism:
it walks relations between thoughts and emits one.

Each step from the baseline was driven by the trace: instances outweighing the candidate (the
boost is now relative), the question form including the name (forms are the non-thought words),
verbs taken as objects (the thought filter), the hop landing on an older sentence about the same
object (objects already in the chain are skipped), "Greg" matching "green" (exact before stem),
"Mice are afraid of wolves" read as being about wolves (subject before object), hops to the
latest object rather than the latest location (answer-like preference), drops (learned
direction). What remains is honest to state: qa15's chain is right (`emily → cat → wolves`
answers "wolf") but "mice" cannot become "mouse" through a stem — a surface-form gap, not a
reasoning one — and qa3 asks for a location *before* another, a temporal relation this chain does
not represent. Nothing here is a learned parametric model of the task: the organism reads the
relations the episode contains and learns, by consequence, which way to walk them.

### Is it bAbI-shaped? Three generalization tests

The user's question after the bAbI numbers, and the right one. Three tests, in order of cost.

**1. Template and lexical shift inside bAbI** (`python/babi_shift.py`): train on the original
stream, test on rewritten stories — every name, place, object, animal, colour and species replaced
by one never seen ("Kwame strolled over to the cellar", "Priya let go of the lantern", "Rafael
happens to be a tapir"), verbs and sentence templates rewritten. The first result was a collapse
(qa1 100 → 28%, qa2 100 → 18%, qa16 100 → 8.5%), and the trace showed *why*: the chains were
still right ("lantern → Priya → cellar") but the learned choices were lexical — answer-likelihood
keyed on stems of answers seen in training, so an unseen answer word had none and the depth fell
back to a prior that cannot tell "where is Kwame" from "where is the lantern"; and the bodies
vetoed bytes never seen after `ASSISTANT:` in the geometric blend ("Priya" → "briya"). The fix is
to key the learned choices on **structure**: the depth prior per (question form, *role of the
thought in its first sentence*: subject or object) — which is exactly what separates the two
"where" questions in any vocabulary; the candidate word is seeded whole and the memory decides
alone while it is being written; among a subject's sentences the hop prefers a relation different
from the one it arrived by. A learned thought filter (document frequency over episodes) was tried
in place of the stop list and is worse on bAbI (names and verbs occur in similar fractions of
episodes); the list stays, and is the remaining hand-written piece.

| tasks 1/2/16, trained on original bAbI | original test | shifted test, lexical keys | shifted test, structural keys |
|---|---|---|---|
| qa1 | 100% | 28% | **100%** |
| qa2 | 100% | 18% | **77%** |
| qa16 | 100% | 8.5% | **63.5%** |

The remaining shifted misses are legible in the sentence trace: qa2's "compass → Tobias → ribbon"
takes Tobias's latest sentence, a pickup written with an unseen verb, where a move was wanted —
with no history for "collected" or "headed", nothing distinguishes the two relation classes; that
distinction is a learned *type* of relation, the mechanism CLUTRR asks for below. qa15's chains
are right on the shifted stories too ("astrid → jackal → hares") and fail on the plural.

**2. CLUTRR** (`tasksource/clutrr`, `python/clutrr_stream.py`): kinship stories in ordinary prose,
a query pair, an answer that is a *composed* relation ("Ashley's daughter Lillian … Nicholas's
sister Lillian → Nicholas is Ashley's son"), and test stories longer than any in training — the
standard systematic-generalization test. The chain scores 0%: two separate gaps. (a) *Natural
sentences*: the thought filter and the "first thought is the subject, last is the object" reading
are simple-declarative assumptions; on prose they extract "pool", "friends", "pigeons" as objects.
The shift test showed the walk generalizes across vocabulary; CLUTRR shows the sentence reading
does not generalize across syntax. (b) *Composed answers*: the answer is never a story word; the
chain finds paths ("Donald → Victoria → Jeremy → brother") but nothing maps the relations along a
path to an answer. The chain-off memory beats chance on short stories (60% at two sentences
against 31%) by copying similar training episodes — the retrieval confound again — and falls to
chance by five sentences.

Both gaps have a form in this architecture. The relation between two thoughts is *the words
between adjacent thoughts in a sentence* — (Ashley, "'s daughter", Lillian), (Nicholas, "'s
sister", Lillian) — a bubble-sequence reading with no subject/object positions and no verb list.
And the answer of a path is a pairwise composition of its relations, (r1, r2) → r12, a table
credited by consequence from the training stream; if it is learned pairwise, chains longer than any
seen follow by associativity — which is what CLUTRR measures. That is the next mechanism; the
bAbI chain was its scaffold.

**Paths and composition, built** (`find_path`, `span_rel`, `unary`/`comp` in `sequence.h`). A
question about two thoughts that are *names* (capitalized mid-sentence — a byte-level surface
signal, English-specific and stated as such) is answered by the shortest path between them through
the episode's sentences, each hop between two names classed by the relation word nearest the pair
— a word the organism has observed as an answer, learned from the stream — with its possessive
marker and which name it attaches to ("Tony's sister June" and "June's sister Tony" are different
classes; every paraphrase of "sister" is one class). `unary[class]` gives a one-hop answer and
`comp[(answer so far, class)]` each further hop, both credited by consequence when the answer is
observed at training time; a longer chain is credited only once its prefix composes, so long chains
bootstrap from short ones. The answer is emitted from its last observed surface form.

| CLUTRR test by story length | chain only | paths + composition | chance |
|---|---|---|---|
| 2 sentences | 0% | 20.0% | 30.9% |
| 3 | 0% | 19.8% | 20.7% |
| 4 | 1% | **20.4%** | 13.0% |
| 5 | 3% | **19.5%** | 14.3% |
| 6 | 0% | 23.3% | 20.0% |

Paths are found for nearly every question (13,170) and 60% compose; the bAbI chain is unaffected
(qa1/qa2/qa16 still 100%). The result is above chance on the long stories, where copying cannot
help, and no better than that. Two limits are visible in the trace and are not template details:
CLUTRR's answers are *gendered* ("Clarence → Jennifer ⇒ grandson", wanted granddaughter) and the
gender of the endpoint lives in pronouns and names, not in any relation word on the path — a
composition over relation classes alone caps well below 100% by construction; and every step of
the way the intermediate steps were fixed by reading traces, which is the honest measure of how
far this is from a learned mechanism: what is learned by consequence are the tables (which words
are relations, which class answers which form, how classes compose); what is written is how a
sentence is read into thoughts and relations. That reading is the piece a real training run has
to make learnable, and it is where the effort should go next rather than into more CLUTRR structure.

**3. Planted facts in natural text** (two-hop facts written as ordinary sentences inside
TinyStories/wiki streams, with distractors): whether the chain is usable outside QA-shaped
episodes at all. Not yet run.

## The training run: what emerges from the general mechanisms

The user's framing: patches and hard-coding were only to test that the principles work; the organism
has to learn almost everything with what it has and generalize beyond it, so the next step is to
*train* and trace which skills emerge from bodies + sequence memory directed by the agency kernel.

`laminar_train` (`scripts/train_run.sh`) trains one organism on one mixed stream —
`python/mix_stream.py` interleaves wiki-en 512 MB, LMSYS 128 MB, SmolTalk, all twenty bAbI tasks
and CLUTRR by proportion in 64 KB pieces, so the small sources recur throughout — with only the
general mechanisms on (K-packed bodies with the lifecycle, sequence memory with the consolidated
tier, the relation store with commitment and selection by consequence, the agency kernel; the
bAbI chain and CLUTRR paths off) and runs the same probes at every 64 MB checkpoint: held-out
wiki-en BPB, free generation from held-out prose, stay-on-task on held-out LMSYS dialogues, and
exact-match QA on held-out bAbI (10 per task) and CLUTRR stories. First run: 200k units, 70 KB/s.

| checkpoint | wiki BPB | free-run (valid / loops) | stay-on-task (chance) | bAbI | CLUTRR |
|---|---|---|---|---|---|
| 64 MB | 2.926 | 96% / 0 of 2 | 0.009 (0.013) | 8.0% (≤ chance) | 0% |
| 128 MB | 2.845 | 95% / 1 of 2 (106 B) | 0.010 (0.015) | 3.5% | 9.0% (short stories only) |
| 192 MB | 2.839 | 95% / 0 of 2 | 0.020 (0.012) | 4.0% | 0% |
| 256 MB | 2.833 | 95% / 0 of 2 | 0.015 (0.015) | 4.0% | 1.5% |
| 320 MB | 2.821 | 95% / 0 of 2 | 0.018 (0.026) | 0.0% | 3.0% |
| 384 MB | 2.824 | 96% / 0 of 2 (17 B) | 0.008 (0.020) | 3.0% | 7.5% |
| 448 MB | 2.820 | 98% / 0 of 2 (0 B) | 0.002 (0.012) | 0.0% | 17.9% (≈ most-frequent answer) |
| 512 MB | 2.804 | 97% / 1 of 2 (188 B) | 0.003 (0.012) | 5.5% | 0% |
| 576 MB | 2.783 | 98% / 0 of 2 (51 B) | 0.004 (0.026) | 2.5% | 0% |
| 640 MB | 2.784 | 98% / 0 of 2 (16 B) | 0.023 (0.019) | 2.5% | 7.5% |
| **652 MB (end, 297 min)** | **2.730** | 97% / 0 of 2 (25 B) | 0.004 (0.011) | 2.5% | 0% |

**What emerged, and what did not.** Over 5 hours and 652 MB with only the general mechanisms:

* *Language.* Held-out wiki-en BPB fell from 2.93 to 2.73 and was still falling at the end; free
  generation stayed at 95–98% real words with loops in 2 of 20 probe prompts and none in the last
  four checkpoints. The mixed stream costs the prose model — 2.78 at 200k units against 2.61 for
  100k units on wiki alone — because five sources share one tier and one body population.
* *Register.* The organism learned each source's reply *format* from context: at `ASSISTANT:` a
  bAbI question gets the chat opener "Sure", a kinship question gets a relation word
  ("granddaughter", the most frequent). It tells the sources apart; it does not answer them.
* *Nothing above chance on content*: stay-on-task at chance at every checkpoint (0.002–0.023
  against 0.012–0.026), bAbI 0–8% against 20–50% per-task chance, CLUTRR at its most-frequent
  answer. The mechanisms that produced 0.07–0.12 on-topic and 100% on bAbI qa1/qa2/qa16 earlier in
  this document were the *scaffolds* (commitment to a related episode with gain 8; the within-
  episode chain), and the scaffolds were off — commitment was on but fired only during probes.
* *Agency.* The kernel's hold rate rose monotonically from 63% to 88% and its mean focus age from
  5.6 to 15.9 events while every redirect fell to ~4%: attention settled as the population
  saturated. On a stream where the bodies predict little, "hold" is the operation that costs
  least, and the kernel learned exactly that — which is the correct consequence of its credit rule
  and no evidence of direction.

The honest reading: bodies + sequence memory learn a language and a register; the relations
between thoughts that answered questions were built by hand in the scaffolds and did not appear on
their own in 650 MB. What the run isolates is the piece to make learnable next: the reading of a
sentence into thoughts and relations, credited by consequence, so that the relation store fills
itself — the scaffolds showed the store is enough once it is filled.

## Learned reading: the relation store fills itself (`seq_chain=2`)

The training run isolated the missing piece — the reading of a sentence into thoughts and
relations was hand-written. `seq_chain=2` replaces every hand-written part of the within-episode
chain with something credited from observed answers, and nothing else:

* **No stop list.** Every word of three or more letters is a candidate thought with a learned
  *utility* (initially 0.5). At an observed answer, the question thought, the link words along the
  chain that produced it and the answer word gain utility; the other words of the answering
  sentences lose a little; a chain that did not answer costs its words a little. A word is read as
  a thought while its utility stays above 0.15.
* **No subject/object rule.** A hop from a thought picks, among the other words of the sentences
  relating it (exact word first, stem as fallback, latest first), the one with the highest utility
  × a learned *position preference* per question form (before/after the link, first/last in the
  sentence) × answer-likelihood × recency. The position preferences are credited the same way.
* **No form list.** The question's form is its words that do not occur in the story.

On the original bAbI stories, with nothing hand-written: **qa1 100%, qa2 81%, qa16 98%**, qa3
34.5% (chance 18–25%). The trace shows what the filter converged to by consequence alone —
`{Daniel, bedroom} {Sandra, kitchen} {Mary, apple}`: names, objects and locations; "went",
"back", "the" and "picked" dropped out on their own. This is the relation store filling itself
from the stream, the thing the 650 MB run showed does not happen without a credited reading.

On the *shifted* stories the first version collapsed (qa1 0%): utility is per word, so an unseen
template word ("strolled") and an unseen place ("cellar") start at the same neutral utility. The
structural handle that separates them is position in the sentence — which the hand-written rule
had used silently — so first/last-in-sentence joined the learned position key.

| bAbI, nothing hand-written (`seq_chain=2`) | original | shifted (unseen names, places, objects, verbs, templates) |
|---|---|---|
| qa1 | **100%** | **84%** |
| qa2 | **81%** | **63%** |
| qa16 | **98%** | **63.5%** |
| qa3 | 34.5% | 3% |
| hand-written reading (`seq_chain=1`), for comparison | 100 / 100 / 100 | 100 / 77 / 63.5 |

The learned reading matches the hand-written one where it matters — qa16 on the shifted set is
identical, qa1 and qa2 are close — and it was written by nothing but consequence: which words are
thoughts, which position answers which form, which depth. qa3 (temporal "before") is out of reach
for both. This is the mechanism the training run asked for; the next run should carry it, with the
credit coming from the stream's own question–answer turns rather than a harness.

### Training run 2: the same stream, with the reading learned from it

Run 1 with only the general mechanisms; run 2 identical except `seq_chain=2`: the learned reading
is on, and its credit comes from the stream's own `ASSISTANT:` turns (bAbI and CLUTRR episodes
inside the mix), never from a harness. The question it answers is whether the relation store fills
itself *in the wild* — with prose and dialogue interleaved, one population and one tier for
everything — so that the bAbI probe leaves chance mid-run with nothing hand-written.

| checkpoint | wiki BPB (run 1) | bAbI (run 1) | CLUTRR (run 1) | stay-on-task (run 1) |
|---|---|---|---|---|
| 64 MB | 2.926 (2.926) | **24.0%** — qa1 10/10, qa2 8/10, qa12 10/10, qa4 7/10 (8.0%, ≤ chance) | 6.0% (0%) | 0.018 vs 0.006 (0.009 vs 0.013) |
| 128 MB | 2.843 (2.845) | 13.5% — qa1 6/10, qa2 0/10, qa4 7/10, qa14 5/10 (3.5%) | 6.0% (9.0%) | 0.016 vs 0.024 (0.010 vs 0.015) |

At 10% of the stream the reading had credited itself 7,598 times from the stream's own
`ASSISTANT:` turns and qa1 was 10/10 — run 1 never left chance in 652 MB. Then it drifted.

The drift from 24% to 13.5% has a legible cause in the credit rule: a chain that did not answer
cost its words a little (−0.1·u), and on this stream most `ASSISTANT:` turns are unanswerable by a
chain — fifteen of the twenty bAbI tasks (yes/no, counting, positions) and every chat turn — so
names and locations were penalized far more often than credited. The isolated five-task test had
hidden it. All twenty tasks trained together, 100 held-out questions each, nothing hand-written:

| learned reading, 20 tasks | qa1 | qa2 | qa4 | qa5 | qa8 | qa12 | qa14 | qa16 | qa20 |
|---|---|---|---|---|---|---|---|---|---|
| with the penalty (run 2's rule) | 15% | 1% | 0% | 35% | 0% | 16% | 14% | 100% | 65% |
| **without it** (`seq_chain_neg=0`, now default) | **74%** | **84%** | **66%** | **82%** | **51%** | **55%** | **53%** | **100%** | 70% |

Negative credit is right only for competitors inside a chain that *did* answer; a failed chain
says nothing about its words. Run 2 was stopped as the drifting baseline and run 3 started with
the corrected rule.

### Training run 3: learned reading with the corrected credit

Identical to run 2 except `seq_chain_neg=0`. Run 1 (general mechanisms only) in parentheses.

| checkpoint | wiki BPB | bAbI | CLUTRR | stay-on-task (chance) |
|---|---|---|---|---|
| 64 MB | 2.926 (2.926) | **22.0%** — qa1 10/10, qa2 8/10, qa12 10/10, qa14 5/10 (8.0%) | 4.5% (0%) | 0.014 (0.012) |
| 128 MB | 2.845 (2.845) | **17.5%** — qa1 10/10, qa2 8/10, qa12 10/10 (3.5%; run 2: 13.5%, qa2 0/10) | 4.5% (9.0%) | 0.028 (0.024) |
| 192 MB | 2.837 (2.839) | **24.5%** — qa1 10/10, qa2 8/10, qa12 10/10, qa4 7/10, qa14 6/10 (4.0%) | 4.5% (0%) | 0.010 (0.010) |
| 256 MB | 2.833 (2.833) | **27.0%** — qa1 10/10, qa2 8/10, qa12 10/10, qa4 6/10, qa14 6/10, qa5 5/10, qa8 5/10 (4.0%) | 0% (1.5%) | 0.026 (0.019) |
| 320 MB | 2.823 (2.821) | **28.0%** — qa1 10/10, qa2 8/10, qa12 10/10, qa4 7/10, qa14 6/10, qa5 5/10, qa8 5/10 (0.0%) | 6.0% (3.0%) | 0.027 (0.013) |
| 384 MB | 2.822 (2.824) | **28.0%** — same profile (3.0%) | **13.4%** (7.5%) — 9,835 path credits from the stream | 0.011 (0.012) |
| 448 MB | 2.818 (2.820) | 25.0% — qa1 7/10, qa12 7/10 (first dip since the fix), qa2 8/10 (0.0%) | 9.0% (17.9%) | **0.028 (0.007)** |
| 512 MB | 2.787 (2.804) | **29.0%** — **qa16 induction 0 → 10/10** (three hops; depth-3 credits 673 → 1,628), qa2 8/10, qa1 7/10, qa12 7/10 (5.5%) | 7.5% (0%) | 0.012 (0.012) |

| 576 MB | 2.789 (2.783) | **34.5%** — qa1 10/10, qa12 10/10, qa16 10/10, qa2 8/10, qa20 7/10 (2.5%) | 9.0% (0%) | 0.017 (0.010) |
| 640 MB | 2.786 (2.784) | 32.0% — qa16 10/10, qa2 8/10, qa20 8/10, qa1 7/10, qa12 7/10 (2.5%) | 13.4% (7.5%) | 0.013 (0.011) |
| **652 MB (end, 307 min)** | **2.730 (2.730)** | **32.5%** (run 1: 2.5%) | 7.5% (0%) | 0.009 (0.004) |

Run 3 against run 1, same stream and organism: the language is identical (wiki-en BPB 2.730 both,
same fluency) and the reasoning probe is not — bAbI 32.5% against 2.5%, nine tasks above their
most-frequent-answer chance, qa2/qa16/qa20 at 8–10/10, all from 78k credits the stream's own answer
turns gave the learned reading. Stay-on-task and CLUTRR stayed near chance in both.

qa16 is the first three-hop relation to appear from the stream on its own: `Bernhard → swan →
Lily → white`, learned from the mixed stream's answer turns after 500 MB with nothing written for
it — the depth prior for the "what colour" form had to accumulate enough depth-3 credits to beat
depth 1 before any answer came out right, and then all ten did.

## Relations as things, credited by the stream (`seq_rel_gain`)

The goal after run 3 named three gaps and one constraint: relations as *things* (not sentence
adjacency), learning where there is no observed answer, staying on topic in dialogue — with nothing
hard-coded. They share one root: the relation store was credited only at question–answer turns.

**The mechanism.** A triple (a, r, b) — thought, the words between, thought — is stored as each
sentence ends (both directions; `RSLOT=8` per source thought, a strong relation is not evicted by a
new one). Whenever a thought recurs, its stored triples *propose* the next thought at the following
word boundary; in a reply, the user turn's thoughts (the frozen goal) propose too. Proposals boost
the proposed word's bytes while the emitted prefix agrees. Credit comes from the stream itself: at
the end of the observed word, the triple that named it is credited and the ones that did not lose a
little — prose is self-supervised, dialogue is credited by observed replies, and both are the same
table. When the stream states (a, r3, c) directly while (a, r1, b) and (b, r2, c) are stored, that
closure credits the composition r1∘r2 → r3 in the same table the CLUTRR paths read.

**The first negative finding, and the rule it fixed.** Credited by *hits*, the store rewarded
predictability: "the" and "went" are predictable after anything, earned utility every sentence,
and the learned thought filter collapsed to `{Daniel, journeyed, the, bedroom}` — bAbI qa1 fell
from 100% to 45%, qa2 from 81% to 3%, and SmolTalk on-topic from 0.083 to 0.038. The rule the
organism already uses for routed bodies applies: credit **information**, not hits — a triple earns
what its proposal beat the organism's own base prediction of the word's first byte by (recorded in
`predict` at the word boundary, so proposals are now made before the first letter). "the" after
"went" gains nothing; "bathroom" after "Daniel" does.

With information credit the bAbI regression is clean again (qa1 100%, qa2 81%, qa16 98%; the
filter converges as before), but proposals still cost dialogue: SmolTalk on-topic 0.083 without
them, 0.044 with them at gain 0.25, and real words 0.99 → 0.94. The reason is symmetric with the
credit rule: proposals pushed even where the organism was already confident — mid-way through
following a committed reply, which is what produces on-topic content — and broke that flow. The
rule at generation time is therefore **propose where uncertain**: a proposal's push is scaled by
the organism's current uncertainty about the next byte (1 − the maximum probability), so a confident
continuation is never overridden and an open choice is informed.

Gated, proposals stop hurting (real words 0.95–0.98; on-topic 0.042 at gain 0.25 and 0.053 at
gain 1, against chance 0.005 / 0.016) but do not yet beat learned reading alone (0.083). Four runs,
one direction: on this dialogue set the triples add nothing the committed episode does not already
supply. Their case has to be made where the committed episode has nothing to say — prose, and
composition.

**Composition from closures, on CLUTRR:** starved twice over. Closures fired **2 times** in 4 MB —
a sentence that states (a, r3, c) with exactly two thoughts while (a, r1, b) and (b, r2, c) are
stored is rare in prose — and CLUTRR by length stayed where the path mechanism left it (18–24%
at 2–4 sentences vs chance 13–31%). The deeper reason is visible in the trace: on prose the learned
thought filter never converges — `{Tony, and, his, sister, June, went, …}` — because what drove
"and" and "went" out on bAbI was *answer* credit, and CLUTRR's composed answers never credit a
chain. The answer-free rule that fixes it is the same information principle one level down: a
word's utility as a thought tracks the **surprise it carries in context** (an EMA of 1 − the
organism's prediction of its first byte) — names, objects and places are surprising, "and", "his",
"the" are not — with answer credit adding on top where answers exist.

Surprise as the *only* utility was wrong in the other direction: on bAbI a verb is as unpredictable
as a name (four of each), so verbs came back as thoughts and fought the answer credit that had
removed them — bAbI original fell to qa1 61.5% / qa2 48%, shifted to 45% / 56.5%, CLUTRR did not
move, SmolTalk held (0.083 vs 0.017). The two signals are therefore two *gates*: a thought must be
surprising in context (answer-free; removes "and", "his", "the" on prose) **and** not discredited
by answers where answers exist (removes verbs on bAbI). Both come from the stream.

With the two gates the prose filter converges — CLUTRR sentences read `{Tony, sister, June, went,
sneak, preview, Avengers, movie, loved} {Clarence, has, one, grandson} {Grandsons, Tony}`, "and",
"his", "the" gone — while bAbI qa2 holds (81% original, 74.5% shifted) and SmolTalk holds (0.079
vs 0.018). One regression was a bookkeeping error, not a rule: with "where" no longer a thought
(predictable after `USER: `), the question *form* — built from the question's thoughts absent from
the story — collapsed to one key for every question, and the depth prior mixed "where is X" (depth
1) with "what colour is X" (depth 3): qa1 fell to 66%. The form now includes the question's
non-thought words as well.

| learned reading, two gates | original | shifted |
|---|---|---|
| qa1 | **100%** | **99%** (was 84%) |
| qa2 | 81% | 72.5% (was 63%) |
| qa16 | 98% | 63.5% |

The surprise gate generalizes *better* to unseen vocabulary than credited utility alone: an unseen
place is surprising by construction, and an unseen template verb is judged by the same measure on
the stream it appears in. CLUTRR by length is unchanged; SmolTalk holds (0.064 vs chance 0.015).

**Where the triples do earn their place: prose.** TinyStories 32 MB, learned reading on, relation
proposals off vs on (gain 0.25, information-credited, uncertainty-gated):

| | BPB | loops | real words | verbatim copying (match ≥ 32 B) | mean match | longest repeat | planted recall |
|---|---|---|---|---|---|---|---|
| relations off | 1.794 | 0/8 | 99% | 9.6% | 18.1 B | 61 B | 18/32 |
| **relations on** | 1.794 | 0/8 | 96% | **0.2%** | **10.8 B** | **19 B** | 17/32 |

The store learned 995k triples and was credited 2.6 million times from prose alone (mean gain
0.47) — no answer turn anywhere — and what it changes is *how the organism composes*: the same
loss, the same fluency, no loops, and a fiftieth of the verbatim copying. Generation is drawn
from learned relations between thoughts rather than from recalled spans. That is the "anything
without an observed answer" gap, answered on the thing that matters: free generation.

**Goal expansion by relations** (`seq_goal_rel`): the goal's thoughts bring the thoughts related to
them (utility × delivered information, at a discount) into episode selection — relations between
thoughts directing generation, as the original framing asked. SmolTalk at 40 dialogues is too
noisy to judge it (0.017 / 0.038 / 0.029 for off / 1 / 4, with chance swinging 0.000–0.046); the
LMSYS comparison at 343 dialogues decides. Its first leg (relations on, expansion off) gives
0.030 against chance 0.018 — the same level as learned reading alone at scale (0.033 / 0.016);
with expansion on, 0.032 (chance 0.018) against the reference and 0.080 (chance 0.039) against
the prompt: neutral relative to chance. The related thoughts the store offers are already inside
the episodes the rare goal thoughts select; expansion stays off (`seq_goal_rel=0`).

Note in passing what the "relations off" row says: learned reading plus commitment alone gives
0.083 on-topic against chance 0.013 on SmolTalk — the best stay-on-task result so far, with nothing
hand-written.

### Against the goal: three gaps, nothing hard-coded

*Anything without an observed answer.* Two mechanisms, both credited by the stream alone: the
surprise gate makes thought discovery answer-free and it generalizes better than credited utility
did (shifted bAbI qa1 84 → 99%); relation triples credited by information gain learn from prose
with no answer anywhere and change free generation for the better (verbatim copying 9.6 → 0.2%,
loops 0/8, BPB unchanged). Answered on the thing that matters.

*Staying on topic in dialogue.* Learned reading plus commitment, with nothing hand-written:
0.064–0.083 on-topic against chance 0.013–0.018 on SmolTalk (four runs, same direction); on
LMSYS 128 MB with 343 held-out dialogues, **0.033 against chance 0.016** (vs prompt 0.076 against
0.027) — twice chance at scale, the same level the hand-tuned commitment reached earlier
(0.034 / 0.016); inside the mixed training stream, above chance at the last two checkpoints
(0.026 / 0.027 vs 0.019 / 0.013 at 24 dialogues). Above chance everywhere, strong nowhere; the
triples add nothing here that the committed episode does not already supply.

*Relations as things.* Built end to end — stored, proposed, credited, composed, named by the
word the stream uses — and measurable in prose. Composition on CLUTRR went from 78 to **90** of
400 held-out questions through learned key resolution with evidence-weighted selection (twice
chance at four and five hops), with the key chosen by the organism per question form; inside the
mixed training stream it reached 13.4% at 384 MB from the stream's own answer turns. What remains
is starved rather than wrong-shaped: 170 wrong entries from one coarse hop class, and gender bound
to a word because attribute equivalence found only 9 merges in 15k answers. The earlier attempt
at composition from prose closures is the one that failed on principle: kinship is not a tree, so
prose closures credit compositions never asserted. Closure credit was made to fire (83k) on any
co-occurring pair, named by the stream's relation word, and finally aligned with the exact table
entry the path reader consults — and accuracy *fell* (9.9% at three sentences vs 19.8% from the
answer-credited paths alone). A sentence that mentions two names with a relation word near them
is not, in general, stating the composition of one particular stored path: kinship paths are not
unique (siblings share parents), so prose closures credit compositions the sentence never asserted,
and the noise corrupts the table that answers credit correctly. Closure credit is kept behind
`seq_rel_closure` (off). What composition needs is a closure signal that is *specific* to the path
— the same relation stated twice, or a question whose answer is observed — and the only such
signal the stream carries today is the answer turn. Relations as things exist; relations composed
from prose alone do not yet.

**What composition needs** (three things, all about the signal, none hard-coded):

1. *A closure signal specific to the path.* Kinship is not a tree, so a sentence stating a–c is the
   composition of only one of several stored paths a→?→c. Credit an operator only on an unambiguous
   example: exactly one class pair over all stored paths between the two names (`seq_rel_closure`,
   now with the ambiguity check).
2. *Attributes of the endpoint in the operator key.* "Clarence → June ⇒ grandson, wanted
   granddaughter": the answer depends on June's gender, which lives in the words the store attaches
   to June ("her", "sister", "daughter"), not on the path. The endpoint's most informative attached
   relation word joins the key, `comp[(prefix, class, attr)]`, with the plain key as backoff
   (`seq_attr`). Gender falls out unnamed; so would any attribute the stream attaches to a thought.
3. *Enough credited examples.* Every 1-hop answer builds `unary`, every 2-hop answer builds `comp`,
   and a longer chain is credited only once its prefix composes — the ladder needs enough short
   answers, and CLUTRR is 4 MB of a 650 MB stream. A data-proportion question for the next run.

Measured on CLUTRR (400 held-out questions, 50k units, relations credited):

| by story length | 2 | 3 | 4 | 5 | 6 | chance |
|---|---|---|---|---|---|---|
| paths only | 20.0% | 18.0% | 17.6% | 19.5% | 23.3% | 31 / 21 / 13 / 14 / 20 |
| + endpoint attributes (`seq_attr`) | **23.6%** | **20.7%** | **19.4%** | 16.9% | 16.7% | |
| + specific closures (`seq_rel_closure`, ambiguity check) | 14.5% | 18.0% | 20.4% | 16.9% | 16.7% | |
| + both | 16.4% | 14.4% | 15.7% | 15.6% | 16.7% | |

Attributes help modestly where the ladder is short (2–4 sentences) and are kept as an experiment.
Closures still hurt even after 26,424 ambiguous cases were skipped and 16,400 "unambiguous" ones
credited: unambiguous *in the store* is not unambiguous in the world — the store is bounded, other
paths exist unseen, and the stated relation word may belong to a different sense. Closure credit
stays off. Composition on CLUTRR sits where the answer-credited paths put it; the lever that
remains is the third one, the quantity of credited examples in the stream.

**Would more training help?** The trace answers per miss. Of 400 CLUTRR questions (attributes on):
78 correct; **100** where a path was found but the operator had *no entry* — more credited
examples fix these, and it is the count of question–answer turns that matters, not stream size;
**170** where the operator had an entry and it was *wrong* — "grandson for granddaughter" (17;
gender, a key that does not distinguish), "grandson for grandfather" / "grandson for father" /
"grandson for mother" (19; the direction of the relation collapsed in the key), "brother for son"
/ "son for brother" / "nephew for grandson" (29; two different compositions sharing one key) —
these are the key's resolution, and more training makes the majority entry more confident, not
more correct; and **61** where no path was found at all (a name not read as a name, or a link
missing). So training helps roughly a quarter of the misses; over half need a finer operator key
— both relation words of a hop rather than the nearest, the direction kept, the endpoint's
attributes — and a fifth need better path finding.

**Learned key resolution.** The user's objection to the attribute experiment was the right one:
"the difference between granddaughter and grandson should be learned, not coded." Nothing about
gender was coded — the attribute is the most informative relation word the store attached to a
thought — but the *shape* of the operator's key was mine. It no longer is: every hop keeps four
candidate keys (plain; with the destination's attribute; with the source's; with both), all four
are credited from each observed answer, and a per-form selector is credited by whether each key's
existing entry was right; at composition the best-accredited key that has an entry answers.
Which distinctions matter, for which question form, is then something the organism found.

Measured on the same 400 CLUTRR questions: 84 correct (78 with the coded attribute key, 78
with none); by length 23.6 / 18.0 / **25.9** / 15.6 / 26.7% at 2–6 sentences (chance 31 / 21 /
13 / 14 / 20) — the gain is at four hops, where a resolved key matters most. Empty entries fell
from 100 to 90, wrong entries did not fall (176): the four candidate keys are all built from the
*same* hop class (nearest relation word, possessive, attachment), and where that class already
conflates two relations no attribute separates them. The next candidate keys would carry both
relation words of a hop rather than the nearest — the same learned selection applies unchanged.

Added as two further candidates (a class over *every* relation word around the pair, plain and
destination-attributed): worse — 69 correct, wrong entries 185. The full-span classes are sparse,
so their entries exist on one or two credits, and a selector that scores a key only by whether an
existing entry was right trusts them on almost no evidence. The organism's own admission rule
applies — trust an entry in proportion to its evidence — so selection now scores accuracy × the
entry's confidence (its credit share), and the selector learns only from evidenced entries.

| CLUTRR, 400 held-out | correct | 2 | 3 | 4 | 5 | no entry | wrong entry | no path |
|---|---|---|---|---|---|---|---|---|
| paths only | 78 | 20.0% | 18.0% | 17.6% | 19.5% | 100 | 170 | 61 |
| coded attribute key | 78 | 23.6% | 20.7% | 19.4% | 16.9% | | | |
| learned key resolution, 4 keys | 84 | 23.6% | 18.0% | 25.9% | 15.6% | 90 | 176 | 61 |
| 6 keys, unweighted | 69 | 20.0% | 11.7% | 21.3% | 20.8% | 92 | 185 | 61 |
| **6 keys, evidence-weighted** | **90** | 25.5% | 21.6% | **25.9%** | **22.1%** | 87 | 170 | 61 |
| chance | | 31% | 21% | 13% | 14% | | | |

Twice chance at four and five hops, with the key chosen by the organism per question form and
trusted by its evidence. Wrong entries are the floor that remains (170): the same coarse class
conflating two relations, and gender bound to a *word* rather than to a class — the next
mechanism below.

**Attribute equivalence by consequence** (`seq_attr_eq`). The user's point: if "granddaughter is
female" and "female is girl" never occur in training, gender cannot bind consistently — an endpoint
attributed by "sister" and one attributed by "daughter" are different keys though they should act
alike. Without such statements the only route to the equivalence is consequence: each attribute
word keeps a signature of the answers it was credited with under (prefix, class) keys, and two
attributes that agree on two or more shared keys and disagree on none become one class, named by
its canonical member. "Female" is then a class the organism found, never a word it saw; and every
credited answer is a vote on which attributes co-behave, which is precisely what more
question–answer turns would buy.

First measurement: 83 correct against 90 without it (21.8 / 20.7 / 21.3 / 19.5% at 2–5
sentences), wrong entries 178. The report says why: **9 merges** in 15k question–answer turns.
Signatures of at most eight keys rarely agree on two without disagreeing on any, so the class is
almost never formed, and the few merges that do happen re-key entries mid-training. The mechanism
is the right shape and starved — exactly the case the user described: without either equivalence
statements or many more credited turns, gender cannot bind consistently. Off by default; the
lever is the quantity of credited examples, the same one composition needs.

*How general is this?* The pieces are domain-blind — thoughts are whatever is surprising in
context (identifiers in code, variables in math), relations whatever sits between them (`=`, `(`,
`return`, `+`, "'s sister"), paths run through shared thoughts, the operator is credited only by
consequence — so it applies wherever the relation algebra is **finite**: kinship, containment,
ordering, type systems, call graphs ("f calls g, g returns T ⇒ f returns T"). Where it does not
work as built is an **open** algebra such as arithmetic: `comp[(+2, =3)] → 5` is a lookup, so the
table would have to hold every number pair. Arithmetic could fit the same mechanism only by
decomposition into a finite algebra at a lower level (digits with carry) chained by the sequence
memory — plausible and untested. Composition here is the application of a learned finite operator.

## Before the long runs: accumulator width, fusion, cores

The user's direction after the reasoning phase: optimise before the long training runs — test
whether INT8 accumulators keep enough of the learning signal for the bodies against a wider
(BF16/INT16) residual, vectorise and fuse what can be, and put on separate cores whatever is truly
parallel.

**Accumulator width.** Each synapse carries an INT8 residual and a delta `(x · q · lr) >> 9`; with
|x|, |q| ≤ 127 most products are below 512 and truncate to *zero*, so "does INT8 keep enough" is
really "how much of the learning signal is thrown away before it reaches the residual". The
variant built for comparison (`acc_bits=16`) is a 16-bit fixed-point residual with the quantum
×64 and the delta shift −6: identical dynamics for large deltas, 64× finer for small ones, and
strictly more precise than BF16 in this range at the same two bytes (BF16's eight mantissa bits
give ~1/256 near the crossing threshold; INT16 gives 1/2048 everywhere). It costs one extra byte
per synapse — +1.1 GB at 200k units — and a 16-lane AVX2 block path replaces the 32-lane one.

| accumulator | synth 8ctx (4,096 units) | TinyStories 8 MB (50k units) |
|---|---|---|
| INT8 (default) | 83.47% frozen, loss 0.981, 2,751 locks | BPB 1.9279, acc 71.67%, **128.7 KB/s** |
| INT16 | 83.23%, loss 1.037, 3,869 locks | BPB 1.9216, acc 71.63%, 89.0 KB/s |

INT8 keeps enough. The 64× finer residual buys 0.3% of BPB on language and costs 31% of the
throughput and a third more memory; on the noisy synthetic task it is slightly worse and births
40% more locks — the truncation of small deltas is acting as a regulariser. INT8 stays the
default; `acc_bits=16` remains available.

**What is parallel.** The event loop is sequential by nature — byte t+1 depends on the workspace
after byte t, and at most three units are active per byte — so there is no wide per-event fan-out
to spread across cores. What is genuinely parallel is *between subsystems*: the bodies (admit →
forward → learn), the sequence memory (capture → consolidate → predict), the agency kernel and
the readout are independent per byte and meet only at the blend. That is a pipeline of three
cores, which pays off in proportion to how evenly the stages cost — the profile below decides.
Within a stage, the learn stage's synapse blocks and the tier's eight lag tables are independent
lanes (SIMD and fusion first, threads if the blocks are large enough).

**The profile, current defaults** (gprof, TinyStories 8 MB, 50k units, 112 KB/s, sequence memory
+ consolidated tier + lifecycle on):

| share | function | what it is |
|---|---|---|
| **28.9%** | `EvidencePool::nursery_candidates` | the lifecycle's admission scan |
| 17.1% | `Organism::process_event` | the event loop body (addressing, thresholds, routing) |
| 12.3% | `SequenceMemory::predict` | 8 lags × 6 instances, verified against the ring |
| 11.0% | `KernelCtx::run` | the INT4 bodies themselves |
| 7.2% | `Organism::readout` | 32 × 256 readout + softmax |
| 4.3% | `commit_splits` | structural learning |
| 3.8% | `apply_acc_block` | accumulator updates |
| 2.8% | `consolidate_step` | the tier |
| 2.5% | `learn_l6` | output-layer credit |

The bodies (kernel + learning + accumulators) are about a fifth of the time; the sequence memory
a sixth; the single largest cost is the nursery's candidate scan, which is bookkeeping, not
computation — the first target, and an algorithmic one.

**Nursery scan, incremental.** The scan rebuilt the parent distribution from all 256 × 64 nursery
cells and re-scored every row, for every evidence slot, every block. Now the parent distribution
is a running column sum kept by `record()`, only the symbol rows that received samples since the
last evaluation are scored (a 256-bit dirty mask per slot), and a slot with no new samples costs
nothing. Profiled time on the same 8 MB fell from 93.9 s to 66.2 s (−29%); BPB 1.9279 → 1.9266
and the lock count within 0.3% — the same organism, without the bookkeeping. The profile after:
event loop 23%, sequence-memory predict 18%, bodies' kernel 15%, readout 10%, splits 6%,
accumulators 5%, tier 4.7%, L6 credit 3.7%.

A correction that matters for everything that follows: the *release* build gained only 2% from
this (128.7 → 130.9 KB/s). The `-O2 -pg` profile had overstated the scan — at `-O3` the 16k-cell
sums auto-vectorise and cost a fraction of what the profile showed — so profile-build proportions
are not release proportions. The fix is still right (the work was bookkeeping), but the targets
for fusion and threading must come from a profile at `-O3`.

**Release ground truth.** The per-stage cycle counters (`timers=1`, printed on every block line)
at `-O3 -flto`, 4 MB, 50k units: **readout bucket 54%** (the readout matvec + softmax, the
sequence memory's predict, and the blend), learn 10%, address 8%, the bodies l1…l4 together ~22%,
struct 5%, evidence 1%. A `-O3` gprof (no LTO; 130 KB/s, i.e. release speed) divides it:
`process_event` self 19.8%, `SequenceMemory::predict` 19.4%, the bodies' kernel 17.5%, `capture`
10.0%, `learn_l6` 9.5%, `readout` 8.7%.

**Three fusions.** (1) The blend computed two `powf` per symbol per byte — 512 scalar
transcendentals after the readout had already done a softmax; it is now one vectorised pass,
`exp((1−a)·log body + a·log seq)`, with a `log8` beside the existing `exp8`. (2) `predict` cleared a
256-float scratch per lag (2 KB × 8 per byte) to touch at most ten entries, and verified matches a
byte at a time; it now uses a ≤16-entry sparse scratch and compares eight bytes at a time. (3) The
eight instance-table and eight tier lookups per byte are random accesses into hundreds of MB;
their buckets are known at `capture`, so they are hashed once there, prefetched, and reused by the
next `predict` instead of being hashed again.

| TinyStories 4 MB, 50k units, release | KB/s | BPB | locks | readout bucket |
|---|---|---|---|---|
| after the nursery fix | 139.5 | 2.0035 | 17,958 | 54% |
| **+ the three fusions** | **196.4** | 2.0030 | 17,987 | 33% |

+41% throughput, the same organism. The stage shares after: bodies l1…l4 ~32%, readout +
sequence memory 33%, learn 15%, address 11%, struct 6%.

The `-O3` profile after the fusions (184 KB/s on 8 MB): the bodies' kernel 20.8%, `process_event`
19.2%, `capture` 13.3%, `learn_l6` 11.7%, `readout` 11.1%, `predict` 6.5% (from 19.4%). A
line-level profile of `process_event` finds no single hot line: the trie walk 4%, the ring's `%`
on every `at()` 2.4%, lock indexing, and stall time attributed to the prefetches (~3% — the memory
latency they expose rather than hide). Two more targets fell out of it: the eight lag hashes were
computed independently from the oldest byte forward (75 byte-steps per set, twice per byte) and
now share one backward pass of 24; and the readout's 256 dot products with 256 horizontal
reductions became eight logits per 32 FMAs over a transposed weight copy kept in sync on the
sparse row updates — and the first form of it was *slower* (32 dependent FMAs into one register
per eight logits, latency-bound; the readout share rose to 37%); four independent accumulators
over 32 logits per pass brought it to 31%. With the ring's `%` replaced by a mask: **203 KB/s**
on the 4 MB run (from 196.4; the timing noise with other runs on the machine is about ±5%), BPB
1.9935 — the new hash changes bucket assignment and, as it happens, collides less. (Two lines in
every profile of a short run are not organism work: the file
load into a byte vector, 5–6%; the harness's clock excludes it.)

**Regression.** All harnesses rebuilt on the optimised code (`scripts/build_opt.sh`): bAbI with
learned reading is identical to the last digit — original 100 / 81 / 98 / 34.5%, shifted
99 / 72.5 / 63.5% — and the bAbI stream trains at **431 KB/s against 220 KB/s** before the pass:
2× where the sequence memory dominates, +46% on TinyStories where the bodies do.

**Cores: the verdict.** After the fusions the split is no longer even: readout + sequence memory
+ blend are about a quarter of the byte, the address walk, bodies, learning and structure the
rest — and those form a chain (the bodies need the address, learning needs the readout of the
same byte). A two-thread pipeline with one-byte-delayed credit would therefore cap at ~1.3×, for
a change to the core loop that alters learning timing. Within a byte there is nothing wider to
spread: at most three units are active, each ~1 µs of work, below the cost of handing it to
another core. So the honest multicore story for this organism is the one already in use: one
organism per core, and independent runs — sweeps, probes, comparisons — in parallel across the
32 threads. The single-core path is what to keep making faster.

## Compute per answer: four levers

The organism computes little per byte — about two bodies of six small INT4 layers — and keeps its
capacity in selection and storage. It does well at what reduces to lookup and walking stored
relations and fails where an answer needs computation. The user's direction: try the four levers
for more compute per answer without trashing throughput, noting that two bodies will struggle
with multi-step problems — which is what the motifs are for.

**The diagnostic** (`python/diag_stream.py`): containment chains whose length k is controlled
(k = 1…8, facts shuffled, with distractor chains), plus bAbI's yes/no (qa6) and counting (qa7) —
answers that need an operation rather than a walk. 8 MB of training episodes, 50 held-out
questions per k and per task, 50k units (`scripts/levers.sh`, `levers2.sh`, `levers3.sh`).

**The levers.**
1. *Depth per step* (`recirc_iters`): within the same byte, a body's L4 output re-enters L2/L3 and
   L5/L6 recompute, the new emission replacing the old in the workspace, N times.
2. *Breadth through the motifs* (`hops`, `kset`/`kscreen`, `route_gate`): more routed bodies per byte
   composing through their neighbours.
3. *Steps per answer* (`seq_max_depth`): the within-episode chain may walk up to eight hops instead
   of three, with a learned **END** slot — "the answer is at the end of the chain" — credited like
   the fixed depths, so the number of steps is chosen per question form by consequence.
4. *Thinking out loud* (`--rationale`): training answers spell out their steps ("the key is in the
   box. the box is in the drawer. answer: drawer"); the organism writes its own steps, reads them
   back, and is scored on the word after `answer:`.

| config | KB/s | MACs/B | k = 1–3 | k = 4–8 | yes/no | count |
|---|---|---|---|---|---|---|
| chain, 3 hops (before) | 282 | 8.9k | 50/50 each | 0 | 0 | 0 |
| **lever 3: 8 hops + END + abstention** | **310** | 8.9k | 50 | **50/50 at every k** | 10 | 19 |
| + lever 1, recirculation ×2 | 209 | 20.2k | 50 | 50 | 10 | 18 |
| + lever 2, hops 3, open routing | 308 | 8.9k | 50 | 50 | 10 | 20 |
| + lever 2, hops 4 | 296 | 9.4k | 50 | 50 | 10 | 19 |
| + lever 2, wide K (16/32/64) | 256 | 11.6k | 50 | 50 | 11 | 23 |
| lever 4, rationales | 325 | 8.4k | 0–1 | 0 | 11 | 21 |
| (recirculation ×4, chain 3 hops) | 146 | 32.9k | 50 | 0 | 0 | 0 |

*Lever 3 solves the multi-step chains outright, at no throughput cost.* Every k from one to eight
is 50/50: the END slot learned from credit alone that this question form's answer lies at the end
of the walk, however long. The first version also exposed a real defect — the chain answered *every*
question, including yes/no and counting (0/50) — so the chain now **abstains** where its credited
accuracy for the question form is below 0.3 and leaves the answer to the bodies and the memory
(it abstained on all 100 yes/no and counting questions and still learned from their credit).

*Lever 1 costs 30–50% of the throughput for nothing measurable; lever 2 is neutral to slightly
positive for the bodies (wide K: +1 yes/no, +4 counting, −17% throughput); lever 4 does not work at
this scale* — the bodies and memory can reproduce the shape of a rationale but not the right
steps, and the chain answers at the reply boundary, before any written step exists.

**Round 3 — does lever 3 generalize, does it regress?** On chains built only from objects and
containers never seen in training (`--shift`), depth 8 + END is **50/50 at every k**; the 3-hop
chain 1–5/50. On the 20 bAbI tasks the fixed abstention bar is too blunt — it hands yes/no and
counting back (qa6 0 → 16, qa7 0 → 44, qa17/18 0 → 47/53) but also silences the chain where it was
right (qa1 74 → 13, qa12 55 → 6): totals 739/2000 with the bar, 655 without. Arbitration by
consequence (`seq_arbitrate`: the chain speaks where its credited accuracy on the form beats the
memory's) gives 719 and does not fix qa1: the memory's accuracy measured on the *training* stream
is inflated by the stream's own redundancy — bAbI asks about the same story several times at
growing length, so the memory copies the previous row's answer — and arbitration trusts it. An
honest arbitration would score the memory only on instances from other episodes. The fixed bar
stays the default; arbitration is off.

**SynLogic (hard), learning curves.** 26 of the 35 SynLogic tasks store their answer in the
puzzle's game data (`python/synlogic_stream.py`: 23,508 puzzles as `USER:` → `ASSISTANT:`
episodes, 55.7 MB, half in Chinese; 20 held out per task). Four organisms (200k units, lever 3 on)
trained on a quarter, a half, all and two passes of the stream; each held-out reply scored for
exact match, partial credit (position-wise token agreement with its own answer), and
**specificity** — agreement with its own answer minus agreement with another puzzle's answer of
the same task, zero when a reply only reproduces a format or a memorized answer
(`python/synlogic_score.py`, `scripts/synlogic_curve.sh`).

| training | stream BPB | exact | own partial | specificity | tasks > +0.01 |
|---|---|---|---|---|---|
| ¼ (14 MB) | 0.943 | 1.2% | 0.208 | +0.019 | 11 / 26 |
| ½ (28 MB) | 0.904 | 1.3% | 0.211 | +0.016 | 12 |
| all (56 MB) | 0.878 | 1.2% | 0.187 | +0.005 | 7 |
| two passes | 0.843 | 1.2% | 0.193 | +0.007 | 7 |

Modelling improves at every step (the formats and the puzzle language). Puzzle-specific content
appears only where the answer can be taken from the prompt — wordscapes (+0.13–0.15, the grid is
the prompt's letters), goods_exchange (+0.06–0.09, pairing entities the prompt states),
web_of_lies (+0.06–0.10), object_properties (exact 5% → 10%) — and it appears early. Every task
that needs computation (sudoku, futoshiki, calcudoko, survo, the arithmetic, sorting, ciphers)
sits at specificity ≈ 0 at every training amount: the replies reproduce the answer's format with
no content from the puzzle. SynLogic's answers are computed — search, arithmetic, sorting — never
stated as a relation the chain could walk. What is missing is operators (compare, count, compute,
search), not data; the same gap bAbI's yes/no and counting tasks exposed.

**Learned use of the tools.** The user's correction to levers 1 and 2: compute should be a
*learned* decision — the organism must learn when to use the tools it has (fan out, kick back up
through L4), not be told to. Both tools were given the organism's own consequence rule, the one
routed bodies already live by:

* *Learned fan-out depth* (`hops_free`): beyond the base hops a routed body passes the message on
  only while its own route utility (the counterfactual bits its emission gained) is above a bar.
* *Learned recirculation* (`recirc_learn`), in three forms: (1) a per-unit counterfactual utility —
  the readout with the pre-recirculation emission against the post; (2) the same, but earned from
  zero by exploration; (3) a two-armed comparison of the unit's actual loss with and without the tool.

| 50k units | TinyStories 8 MB: BPB | KB/s | mixed stream 16 MB: BPB | KB/s |
|---|---|---|---|---|
| no extra compute (2 hops) | 1.9127 | 184 | 2.6849 | 156 |
| **learned fan-out, up to 4 hops** | **1.9097** | 179 | **2.6768** | 168 |
| learned fan-out, up to 6 hops | 1.9289 | 184 | | |
| recirculation, fixed ×2 | 1.9237 | 129 | | |
| recirculation, counterfactual utility | 1.9148 | 122 | 2.6856 | 120 |
| recirculation, earned from zero | 1.9105 | 132 | 2.6828 | 131 |
| recirculation, two-armed on actual loss | 1.9314 | 153 | 2.6798 | 147 |

*Fan-out is the tool the organism learns to use well*: −0.16% and −0.30% BPB at the same speed,
extra hops taken only by bodies whose routed messages have paid (1.6–2.8M extra hops). Six hops is
too deep and the gate does not yet stop it. *Recirculation does not pay in any form.* The
counterfactual credit is self-fulfilling — a unit that recirculates trains its weights on that
path, so its post-pass emission beats its untrained pre-pass by construction, and 60–95% of units
"earn" it; the unbiased two-armed comparison is worse still, because a unit alternating between
paths trains its weights on both and the interference costs more than the extra depth buys.
Recirculation stays available and off; the training-run script uses learned fan-out
(`hops=4 hops_free=2`), learned reading and chain depth 8. The `Params` defaults keep the 2-hop
behaviour so the NumPy parity configuration is unchanged.

Why the motifs do not move yes/no and counting here: those answers need a relation *across
sentences* — is this person's last location the one asked about, how many objects are held — and
a body sees a 128-cell window of recent bytes plus the workspace. Routing more bodies within a
byte composes more features of that window; it cannot compare two facts stated twenty bytes
apart. The composition that multi-step problems need happens where the relations are stored.

## Learned answer attention, warm-started by backprop, continued by forward gradients

**Why.** The frozen judgment sets (`python/partychat.py`: party head counts, yes/no and guest
counts after 0–60 real-chat tangent turns, names and phrasings disjoint between train and test;
`python/proofwriter_stream.py`, split by theory; `scripts/judgment.sh`) scored zero on everything
but ProofWriter's majority label. The reasons are structural: the chain sees 64 sentences, it can
only copy a word, and every learned choice is a table keyed by an exact hash of the question's
words, so a paraphrase starts from nothing.

**The block** (`cpp/include/laminar/attend.h`, `seq_attn`). It runs once per reply turn over every
sentence of the dialogue. A sentence's key is the mean of its words' learned embeddings, a
projection of the organism's workspace at the sentence's end, and a speaker vector.

* *Read*: bilinear relevance plus idf-weighted word overlap, plus a recency slope the question sets.
* *Aggregate*: a gated signed sum whose values may carry the numbers written in a sentence.
* *Answer*: a softmax mixture of three sources. The first is a classifier over the replies observed
  so far. The second is a pointer that copies a word from the attended sentences. The third is a
  proposal from the organism's other mechanisms, here the chain's pick.
* *Veto*: a trust gate. The block speaks only when it predicts it is right.

*Learning* is backprop during a warm start (a replay of its buffer at the switch), then
activity-perturbed forward gradients. One forward-mode pass along random tangents on every hidden
site gives the directional derivative. The output layers take their exact local error, the delta
rule the readout already uses. Backprop and forward-mode agree to 1e-6 (`cpp/apps/attn_probe.cpp
gradcheck=1`).

**The block alone** (`attn_probe`: bAbI's 20k training answers, three passes, in seconds; bAbI 1–10
held-out, the design set):

| learning | bAbI 1–10 |
|---|---|
| backprop throughout | 54.9% (qa1 94, qa7 counting 76) |
| backprop warm start (first pass), then frozen | 51.5% |
| warm start, then forward gradients (lr 0.003, exact output errors) | **52.7%**, last window back to 0.58 |
| warm start, then forward gradients (lr ≥ 0.01) | 29–48% |
| forward gradients from scratch | 17–30% |

The fruitfly result holds here too. A backprop head start survives the handoff to a local rule and
keeps improving slowly, provided the output layers use their exact local error and the rate is low.
From scratch, forward gradients do not get off the ground.

**In the organism, it does not yet pay.**

* *Bubble replay was overriding forced answers.* The sequence memory's "bubble" replay of a
  remembered continuation overrode every forced answer, the chain's included. A committed answer
  now cancels the bubble, which is a no-op for the baseline's scores.
* *Mixed stream, the block alone.* Trained on the judgment mix, the block scores 15–26% on bAbI
  1–10, against 54% on bAbI alone with the same bAbI data. About 170k other reply turns (chat
  openers, party acknowledgements, ProofWriter) share one classifier, one mixture and one gate. At
  a high rate the block tracks bursts of one task and its gate becomes confidently wrong: the
  training-stream inflation that sank arbitration, again.
* *In the organism, on bAbI alone.* Without the proposal the block overrode a chain that was right
  (qa1 100 → 71). With the chain's pick as a proposal it is neutral at lr 0.03 (538 vs 537 of
  1,000) and harmful at 0.1 (467). The chain's pick is right on 37% of training turns and the
  block gives it a flat 0.30 weight. It has not learned, from a mean of question-word embeddings,
  for which questions the chain is the one to trust.
* *Frozen sets* (first version, before the proposal): party chat 0 in every arm, ProofWriter
  unchanged, bits per byte identical (0.5170). The forward-gradient arm drifted after about 60k
  online answers on the long stream.

**Second form: the question from the organism's own state, trust judged on held-out turns.** Each
question word is taken with the workspace at that word, and a learned attention pools them, instead
of averaging word embeddings. One dialogue in five, keyed by its story's first sentence, trains only
trust (the gate and the source mixture). The other four train only the answering parts. So trust is
measured on material the answering parts never saw, and bAbI's growing-story repeats cannot inflate
it.

| | bAbI 1–10 |
|---|---|
| block alone, mixed judgment stream (was 15–26%) | **38.2%** |
| organism without the block, bAbI stream | 537/1000 |
| organism + block, backprop throughout | 514 |
| organism + block, warm start 10k then forward gradients | **598** (qa4 33 → 59, qa9 39 → 66, qa1 kept at 100) |

On the frozen judgment sets the block now changes nothing: every score equals the baseline. The
held-out gate is honest. Its own answers to unseen party phrasings are mostly chat acknowledgements
("Noted", "You"), occasionally right ("3", "no"), and it keeps silent (gate 0.01–0.3). It no longer
harms anything, but it does not recognise a new phrasing as a question it has learned. The
question read is better, but still not good enough to transfer.

**What this says.** The mechanism works where it was designed and learns by the route asked for.
It does not yet generalise across a mixed stream. That is the discipline's drop signal for this
form. The weak point is the question representation: one averaged embedding decides relevance,
copy-or-classify and whom to trust. The next form should build the question from the organism's
own state over the question, as the keys already partly are, rather than a bag of words. It should
also judge trust per source against held-out instances, not the training stream.

## Attention after the bodies: a hybrid block, and a "white matter" probe

**Hybrid block** (`cpp/include/laminar/hybrid.h`, `cpp/apps/laminar_hybrid.cpp`). A 0.93M-parameter
transformer block, with causal attention over 128 bytes and an MLP, sits after the bodies, as in
hybrid models that alternate recurrent and attention blocks. Its input is the byte embedding plus
the bodies' workspace. Its output is a logit correction on the organism's prediction. It trains by
backprop over 128-byte chunks and is scored before learning. Its gradient matches finite
differences.

| 8 MB TinyStories | bits per byte | KB/s |
|---|---|---|
| organism alone | 1.9097 | 110 |
| + block, sees the workspace | 1.8817 | 9.8 |
| + block, blind to it | 1.8726 | 9.9 |
| block alone (a byte transformer) | 3.3908 | 11.1 |

It gains 1.5–1.9% at eleven times the cost, and the bodies' state does not help it.

**White-matter probe** (`whitematter.h`, `laminar_wm.cpp`). This is attention *between motifs*,
kept cheap. Every emitting body (about 1.8 per byte) becomes a token: its emission plus a learned
embedding of its motif. The current byte attends over the tokens of the last 32 bytes. There are
16k parameters. Every arm has the same cost:

| arm | bits per byte | KB/s |
|---|---|---|
| organism alone | 1.9097 | 185 |
| motif tokens (the white-matter test) | 1.8795 | 76 |
| summed workspace tokens (no motif identity) | 1.8701 | 82 |
| byte embedding only (no organism state) | **1.8683** | 93 |
| motif tokens of the current byte only | 1.8705 | 84 |
| byte embedding only, attending 1 / 4 bytes | 1.8683 / 1.8683 | 98 |
| byte embedding only, width 8 (8k params) | 1.8691 | 111 |

Motif identity makes it slightly worse, and attention across bytes adds nothing. The whole gain is
a rank-16 correction keyed on the current byte, a next-byte table added to the organism's
prediction. It recovers about 2.2%, more than the 1M block did. As tested here, routing between
motifs carries nothing the readout misses. What is missing is a calibrated byte-conditional term at
the output, and that can be had for a table lookup.

## Can the bodies rewire toward the goal? (goal-credited plasticity)

Params `goal_gain`, `goal_elig`, `goal_decay` (`Organism::goal_track`, `goal_credit`). On the bytes of a reply's
first token, every existing signal is scaled by `goal_gain`: the L6 error, resistance, lock utility and the birth
margin. Bodies that emitted while the question was being read keep an eligibility trace. When the answer ends,
its loss relative to the running mean credits them through resistance and lock utility. The same code counts
per-motif emissions by phase, and the harness option `ablate=top:K|rand:K` silences motifs before the test. The
chain is off, so answers come from the bodies and the memory alone (`scripts/goal_test.sh`, shuffled bAbI,
held-out 1–10):

| | ¼ pass | 1 pass | 3 passes |
|---|---|---|---|
| byte-loss plasticity | 199 | 214 | 226 |
| goal-credited (gain 8, eligibility 1) | 153 | 205 | 223 |

No motif became specialised to questions and answers in either arm. The bAbI stream grows only about 500 bodies,
in 7 active motifs, so the ablations silenced every body. Both arms then score 221: the bodies carry about half a
percent of the answers, and the memory carries the rest. Crediting the goal cannot build an answer circuit out of
units that see one byte, a hashed context and a 32-number workspace. The facts never reach them in a usable form.
The missing piece is a composer: learned units above bytes (words, then concepts), with motifs that take only
their level's units as input.

## The composer: learned units above bytes, with bodies of their own

Params `levels`, `lvl_margin`, `lvl_motifs`, `lvl_max`, `lvl_feed` (`Organism::composer_step`, `close_unit`,
`level_event`). A level-1 unit closes where the organism's surprise about the next byte rises above its running mean
× `lvl_margin`. Level l+1 closes over level-l units by the same rule, using its own next-unit counts. A closing unit
is an event at its level:

* its identity and its predecessors form the key into the level's own address tree;
* the level's own bodies run on it, in reserved motifs, so they never see a byte;
* their output drives the byte prediction through the level's own readout weights until that level's next unit;
* the byte error over those bytes is credited back to them.

**The units are real.** Nothing states what a word is. On TinyStories, level 1 finds words and short phrases
("the water.", "They", "were", "big and gray"), level 2 clauses and level 3 roughly sentences. On bAbI, level 1
finds "Fred", "is", "in the", "school.", "went back to the", and level 2 finds one or two sentences. The boundary
needs *rising* surprise: without that, words split after their first letter.

**They do not carry answers yet.** On held-out bAbI 1–10 with the chain off (`scripts/composer_test.sh`, 3 passes),
the scores are: no composer 226; 2 levels 201; with goal credit 228; fed into the byte bodies 206. Silencing the
level-1 or the level-2 bodies moves the score by only a few points. The composer costs 2.5% bits per byte and halves
throughput on this stream, 305 to 156 KB/s. A level body sees the current unit, the previous one and a leaky 32-cell
summary. "Is Mary in the cinema?" needs the location bound to Mary several units back, and nothing at the unit
level can hold that binding and look it up again. The binding lives in the sequence memory, the relation store and
the chain, and those still run on bytes and hand-cut alphabetic words. Next: make the composer's units the thoughts
of the memory side, so binding and lookup happen at the level where questions are asked.

## The memory side on composed units (seq_units)

The composer's segmentation was replaced by a learned vocabulary: units start as single bytes, a pair that keeps
recurring becomes a unit of its own (in rounds, the most frequent pairs first, counts aged between rounds), and
merges apply greedily as the stream arrives. Identity is then stable — the same text always yields the same units —
which is what binding needs. `seq_units=1` makes level-1 units the chain's words, `seq_units=2` makes level-2 units
its sentences, replacing the alphabetic tokenizer (`SequenceMemory::unit_close`).

On bAbI the units are clean: `[Mary ] [went ] [to the ] [bedroom./] [Bill ] [is ] [either in the ]`. The chain still
falls from **537** to **223** of 1000 (`scripts/units_test2.sh`; earlier surprise-based segmentation: 229).

Why: the chain finds the question's entity in the story, and merges absorb the entity into a larger unit. The
question reads `[Bill in the ]` while the story has `[Bill ]`: different identities, no match. Stable identity is
necessary and not sufficient — the entity has to be a token of its own, and a frequency-driven vocabulary has no
reason to keep it one. Earlier criteria failed differently: rising surprise gives context-dependent units (the same
word segmented differently in different places), frequency-drop and successor-variety give sub-word units (2.2–2.3
bytes) because continuations get rarer and more varied everywhere inside words.

**The merge tree exposed.** Every unit keeps its two parts, their lengths and how often each stands on its own
(`Organism::emit_pieces`). A unit splits only when both parts stand alone at least `lvl_minc` times AND at least
twice as often as the whole, recursively — a rule on the tree and the counts alone, so the decomposition is the same
wherever the text occurs. Merges are applied by RANK inside a 24-unit window (the earliest-learned pair wins), not
in arrival order, and a piece spanning a sentence end is cut there. `lvl_warm` lets the vocabulary settle before the
memory side switches to units.

On the training stream the pieces come out right: `[Bill] [ travelled to the bedroom] [./Mary] [ is in the ]
[school]`. The chain still does not recover (3 passes, held-out 1–10): baseline **585**, tree 208, tree + settle 204,
stricter split + longer settle 186. Its traces show why: the baseline builds `{Daniel,bedroom} {Sandra,kitchen}` and
walks from the thought `Daniel` to `bathroom`, while units give `{ohn jour}`, `{e|>}` and often no selection at all.
Two causes: a vocabulary learned from 7–21 MB has ~1k merges, enough for frequent phrases but not to isolate every
entity and location; and the chain's selection (the rarest question thought, subject/object roles, the relation
between them) assumes one content word per thought, so phrase units crowd the entities out.

Conclusion: the composer stays off by default and is worth keeping for the body levels, where it is neutral. The
memory side keeps its hand-cut words — they are not what blocks reasoning, which is why every unit variant lands near
200 while the same chain on words gets 585. Revisit only with a vocabulary learned over hundreds of megabytes and
frozen before the memory side uses it.

## Reasoning primitives probe: execution is sufficient, learning remains open

An isolated read–act–write experiment (`python/reasoning_primitives_probe.py`,
`cpp/apps/reasoning_agency_probe.cpp`) tests structured facts before involving the
reader. Ten elementary instructions expose persistent focus/value registers, a
memory cursor, a scratch set, an accumulator, comparison and typed return. Their
use is selected by a learner; the answer-only arms receive no solution traces.

Trained on sizes at most three and tested on fresh entities at sizes 4/6/8/12,
a tabular TD controller reaches 88.2% relation following, 95.6% comparison and 0%
distinct counting (five seeds, 60k mixed episodes each). More training and a
policy-gradient control do not recover counting. The same policy representation
fit from 300 supplied short action demonstrations reaches 100% on all three;
removing the relevant primitives destroys those capabilities. This positive
control demonstrates execution sufficiency, not autonomous discovery.

The existing AgencyKernel connected to this VM scores 0% on the longer tasks
under both normal and extended eligibility (three seeds each). These are adapter
results, not whole-organism benchmarks. The primitives can support the tested
procedures, but their presence alone did not teach the current learner to use
them. Credit assignment, exploration and the supplied finite state abstraction
remain open issues. No production defaults changed.

Full protocol, limitations, ablations and reproduction commands:
[REASONING_PRIMITIVES_EXPERIMENT.md](REASONING_PRIMITIVES_EXPERIMENT.md).

## Composing learned skills: callable frames and verified procedure memory

The next investigation separates individual skill learning from composition.
`reasoning_runtime.h` makes skills callable with typed arguments and returned
values, keeps their scratch state local, and separates learned skill selection
from call/return control. It uses the existing agency sparse expansion, consequence
routes and local plasticity. With individual skills taught on short examples,
the native controller learns new compositions from whole-answer feedback when
given semantic goal graphs: 5,500/5,500 held-out cases and 3,900/3,900 harder cases
across five seeds. These graph inputs already specify a composition, so this
result alone does not establish procedure discovery.

`procedure_memory.h` removes that scaffold. It takes flat question cues,
arguments, relations and facts, searches sequences of callable learned skills,
and keeps a procedure only when it matches multiple verified training answers.
No composite instruction sequence or goal graph is supplied. With 12 examples
per question kind, trained on 1–3-link walks and counts 0–3, it scores 7,000/7,000
on fresh held-out questions (five seeds), with walks up to 32 and counts up to 24.
One-example learning often retains accidental shortcuts. Removing save/swap
preserves the individual skills but makes all three two-result composition
families fail. The procedures, including follow → count → swap → follow → count
→ compare → answer, are found by the search, not specified as task recipes.

Two other boundaries matter. Instruction competition must settle anew rather
than carry a previous instruction's winning activity indefinitely. And crediting
each decision against its own predicted return works substantially better than
broadcasting the last action's residual through earlier decisions. Both TD and
Monte Carlo return updates work in the graph experiment; backpropagation is not
required for these local controllers.

This is a tested native composition mechanism, not a reported improvement to
raw-text chat or bAbI. The input interface still requires reliable facts and a
question representation that supplies cues and arguments. Production defaults
remain unchanged. Full protocol, exclusions, limitations and reproduction:
[REASONING_COMPOSITION_EXPERIMENT.md](REASONING_COMPOSITION_EXPERIMENT.md).

## Native text bridge to learned procedures (experimental)

The composition runtime now has an optional raw-byte interface:
`reasoning_language.h`, installed with `Organism::attach_reasoner` and connected
to `SequenceMemory::capture` and answer competition. Grounded sentence examples
teach fact frames; normalized question forms acquire procedures from twelve
verified final answers on different worlds. Returned entities, booleans and
integers become bytes in the actual organism distribution. Generated bytes never
become verified feedback or new grounded facts.

Across three seeds, the bridge answers 2,520/2,520 held-out questions in learned
wording, including alphanumeric identities, paths up to 32 and counts up to 24.
Removing persistent result bindings again breaks the two-result compositions.
New question wording has zero procedure coverage despite correct fact extraction;
new fact wording prevents extraction. Teaching the corresponding language forms
on separate worlds restores 840/840 for each group. These results isolate language
grounding as the remaining boundary in this controlled domain.

The fact reader receives aligned subject/relation/object training examples and
uses an exact symbolic frame grammar. This is not raw-text-only learning or general
chat understanding. The interface is default-off, the experiment freezes body
training, and no bAbI or free-form chat improvement is claimed. Unsupported forms
leave the legacy generator available; module abstention is not global chat
abstention. See `docs/REASONING_LANGUAGE_EXPERIMENT.md` for supervision, ablations,
the native byte protocol, checks, limits and reproduction.

## Shared phrase meanings from verified procedures (experimental)

`semantic_grounding.h` extends the optional native language interface. A learned
individual procedure is converted into a typed meaning; phrases shared across
different question contexts acquire their common semantic subexpressions. A
bounded chart parser then composes these learned rules for a new question. English
predicate lists and composite question recipes are not coded into this layer.
The resulting expression calls the existing learned skill policies and returns
through the native byte-emission path.

The initial curriculum supplies only identity, follow, count and entity-equality
questions. Across three seeds, 2,340/2,340 withheld questions and compositions have
both the correct inferred expression and the correct native answer. All complete
test forms are absent from training, including 300 paired attachment interventions
whose correct answers are opposite. Removing learned phrase rules or using whole
question lookup gives zero coverage on these cases. Teaching two new phrase meanings
through individual answers adds 300/300 withheld composite successes. Unsupported
words remain ungrounded.

This is a symbolic memory-side language learner with supplied type/application
rules, a controlled grammar and supervised fact grounding. It does not establish
general chat, stateful belief updates or grounding facts from experience. See
`docs/SEMANTIC_GROUNDING_EXPERIMENT.md` for the experiment and
`docs/COCO_CURRICULUM.md` for the proposed progression. The broader grounding work
remains ongoing.

## Event meanings from observed transitions (experimental)

`event_grounding.h` learns sentence-role bindings, relation channels and update
meanings from utterances paired with before/after symbolic worlds. It selects
among four supplied generic operations: add, exact retraction, replacement for a
subject, and replacement for an object. It receives no per-sentence role or effect
annotation, but perception already identifies entities and relation channels.
Only a unique hypothesis supported by multiple observations may interpret text.

At equal exposure of 288 observations per seed, repeated simple scenes identify
2/12 sentence meanings; contrasting scenes identify 12/12. Through the optional
native language bridge, the latter gives correct current state and answers on
2,160/2,160 supported turns across three seeds. Transfers, retractions, corrections,
duplicates and unrelated changes are tested across dialogue turns, including
intermediate states. Swapping the link/possession channel numbers in one seed
preserves success without telling the learner the permutation. An append-only
ablation has 840 incorrect supported states even though some answers remain right.

This is a symbolic memory-side learner with literal event frames, reliable
observations and engineered update primitives. It does not establish raw
perception, pronouns, noisy evidence recovery or general chat. Unknown event text
withholds this module's proposal; other generator fallbacks remain available.
Self-generated text cannot change grounded facts or verified evidence. No defaults
changed. Protocol, matched-exposure controls, checks and reproduction are in
[EVENT_GROUNDING_EXPERIMENT.md](EVENT_GROUNDING_EXPERIMENT.md).

## Reference phrases and pending clarification (experimental)

`reference_grounding.h` learns which contextual entity-selection operation a
phrase denotes from verified event outcomes. Generic selectors choose subjects,
objects or either role, optionally restricted to a relation, from grounded events
in the current turn or, when none exist, the previous factual turn. The learner
contains no English reference-word list.
Observed state changes disambiguate candidate phrase spans and selectors; these
meanings transfer into event and question contexts that were never taught.

The optional native language bridge suspends an ambiguous utterance, emits a
clarification request, accepts a named candidate and resumes the pending change
or question. This control procedure and its output template are supplied. The
phrase-to-selector mapping is learned. Invalid choices, generated text and episode
boundaries cannot silently resolve a pending reference or contaminate grounding.

Across three seeds, 48 reference observations per seed support 1,170/1,170 held-out
answers and 450/450 exact clarification checks. Twelve more observations per seed
teach a new alias and add 90/90 answers, with earlier results retained. The scorer
checks exact state, resolved question, candidate sets and native output, excluding
ordinary setup answers from these transfer counts. Reference or role ablations
break direct transfer. This remains a small symbolic discourse model with reliable
perception and a supplied clarification policy, not general chat or a learned
attention mechanism. See [REFERENCE_GROUNDING_EXPERIMENT.md](REFERENCE_GROUNDING_EXPERIMENT.md).

## Event meanings from consequence answers and revisable evidence (experimental)

`answer_grounding.h` learns novel event frames from verified consequence answers.
Its prior facts and question meanings come from the already grounded reader. It
tries the generic update primitives with candidate relation/argument bindings,
executes the learned question skills, and retains meanings consistent with the
answer. Familiar later events can provide delayed constraints. No correct
post-event state, event-role label or oracle query program enters this learner.

Evidence has stable scene/observation IDs and can be explicitly corrected or
withdrawn. The optional `uncertain_events` mode retains possible worlds and the
shared frame meanings that produced them. It answers when learned computations
agree across those worlds and otherwise emits uncertainty. A model change during
a live episode marks that state stale until reset/replay. Uncertainty and conflict
responses are counted separately from answered questions.

At equal exposure of 180 answers per seed, questions distinguishing the old actor
and event object identify 5/5 new event frames; repeated questions about the same
actor leave all five ambiguous. Across three seeds, the former gives correct state
and output on 1,440/1,440 consequence questions. The weaker model answers 840
correctly and explicitly withholds 600 ambiguous outcomes, retaining the true
world throughout. Choosing its first hypothesis instead produces 408 wrong
assertions. Correcting deliberately bad evidence restores the full result on
replayed episodes. An independent interpreter audits every candidate meaning and
retained world, including repeated uses of one frame.

This removes direct after-state supervision for these new event meanings. It still
depends on already grounded descriptions/questions, fully described prior worlds,
engineered search/update primitives and externally verified feedback. It does not
establish arbitrary partial observation, autonomous noise recovery, or a learned
question-selection policy. See [ANSWER_GROUNDING_EXPERIMENT.md](ANSWER_GROUNDING_EXPERIMENT.md).

## Partial observation and public curriculum readiness

`fact_knowledge.h` adds explicit negatives and complete observed scopes to the
optional grounded reader. In `partial_observation` mode, absence alone is unknown;
counts and follow termination require complete coverage. Two generic membership
operations can be selected by learned procedures. Visibility loss invalidates
affected evidence, and ambiguous event worlds carry separate observation states.
Sanitizer contracts and all four existing grounding regressions pass. Visibility
semantics are still supplied through perception APIs, not learned from language;
see [PARTIAL_OBSERVATION_PRIMITIVES.md](PARTIAL_OBSERVATION_PRIMITIVES.md).

Initial public-curriculum validation found 5,423 contradictory training labels
and 116 held-out labels in explicit-name Stage 5 stories, missing environment role
markers, and Stage 3 world leakage and missing paired paraphrases. The current
warm-started reader abstained on all 1,952 sampled new QA prompts because their
fact language was unsupported; this was a readiness check, not a training result.
Dataset fixes and learner integration work are recorded separately in
[CURRICULUM_REBUILD_ISSUES.md](CURRICULUM_REBUILD_ISSUES.md).

## QA-only bootstrap of observation and question language

`bootstrap_grounding.h` jointly infers sentence update meanings and an atomic
question program from raw stories and verified final answers. Entity vocabulary
is proposed from question variation and answer referents; no annotated facts,
entity spans, relation roles or query programs are supplied. The existing native
primitive policies remain pretrained from action demonstrations. Joint symbolic
search retains supported alternatives and requires exhaustive evidence within
its declared budget; latent channel permutations are canonicalized.

On 900 bAbI location training records it learns five movement frames and the
location question. Each of three seeds passes 2,480 supported test cases with
exact answers and full state, and declines 30 untaught-verb cases. An all-opaque
vocabulary control also passes. Append-only updates yield just 1,062 correct
answers and 330 correct states per seed. This is a single-template, closed-world
bootstrap, not general curriculum training. Publication into an existing grounded
model is refused until channel alignment is resolved. See
[BOOTSTRAP_GROUNDING_EXPERIMENT.md](BOOTSTRAP_GROUNDING_EXPERIMENT.md).

## Incremental grounding with shared relations and mixed questions

`incremental_grounding.h` extends an already learned location model using raw
stories and verified answers to multiple question forms. It jointly constrains
new event meanings and atomic procedures, preserving established meanings and
requiring exhaustive, converged evidence before publication. Repeated names bind
argument positions without fragmenting the sentence frame. Inverse follow/count
adapters expose the opposite direction of a relation to existing learned
micro-policies; partial reads require complete column coverage.

Starting from the QA-only bootstrap, 1,840 additional lessons teach four event
frames and ten question forms. These include 160 selected original v2 training
records; the remainder are controlled lessons. All three seeds pass the same
1,945 evaluation cases on the refreshed Stage 5 data with exact output and complete
state: 240 withheld compositions, 105 selected v2 held-out transfer questions, 1,000 original location
questions and 600 atomic/mixed-world controls. Disabling shared phrase meanings
removes all composed answers while preserving atomic answers and state. A prior
channel permutation also passes. This is still a small, fully observed model;
v2 unknown/temporal/dialogue cases are outside this learner run. See
[INCREMENTAL_GROUNDING_EXPERIMENT.md](INCREMENTAL_GROUNDING_EXPERIMENT.md).

The rebuilt curriculum fixes the checked current-possession labels, Stage 3
world overlap/paired paraphrases and environment role export. The latest Stage 5
revision removes the direct unmentioned-owner leak and all temporal errors found
by the independent witness audit. Audit coverage and a remaining matched-visibility-control issue are recorded in
[CURRICULUM_V2_VALIDATION.md](CURRICULUM_V2_VALIDATION.md).

## Environment dialogue observation protocol

The optional `environment_dialogue` mode treats context and environment messages
as observations, user statements as requests/claims, and assistant text as
non-evidence. Environment lines can contain multiple sentences whose combined
meaning is learned as one frame. A pending question is answered once, before
reading its target assistant turn. Generated role/episode markers cannot update
state. This protocol is supplied source metadata, not learned source credibility.

From 2,919 QA lessons derived from 32 training dialogues, the incremental learner
adds two environment frames and a destination-count question. Preservation
questions constrain unintended changes to its prior relation. It retains up to
66 observationally equivalent candidate worlds under an explicit 128-world
budget. All 414 held-out canonical dialogues pass on three seeds: 674 original
answers, 10,074 before/after possession queries and 9,278 complete state checks per
seed. Altered assistant claims and user statements do not change the result;
omitting environment reports reduces actual-world answer accuracy. Every retained
world agrees with the independently reconstructed state on these cases.

Primitive weights stay frozen; language acquisition uses symbolic constraint
search without end-to-end backpropagation. This experiment establishes observation
ingestion, not an action policy, future-outcome prediction or general chat. See
[DIALOGUE_GROUNDING_EXPERIMENT.md](DIALOGUE_GROUNDING_EXPERIMENT.md).

## Request grounding and one-step actuation

`request_grounding.h` learns requested state predicates from raw requests,
grounded observations and explicit satisfaction feedback. Its candidate space
contains positive/negative relation membership with variable argument bindings.
Ordinary successful/failed transfers leave two supported meanings; wrong-recipient
feedback distinguishes achieving the requested ownership from merely removing
the original ownership. Verified contradictions revoke meanings, and generated
text cannot teach them.

The separate actuator memory learns effects for opaque command IDs from
acknowledged executions and the reader's observed before/after states. A bounded
controller enumerates the environment's executable action menu and accepts an
action only if all retained goal/effect meanings predict satisfaction. Predicted
states never replace observed memory. The opt-in reader captures requests from
user messages and exposes `choose_action()`; the caller executes the selected
structured action and returns a trusted observation.

From 496 request lessons and 128 exploratory attempts, each of three seeds
correctly evaluates all 1,669 original dialogue requests before and after their
outcomes, completes 576 requests in new interactive worlds, recognizes and
retries 116 forced failures, and retains 1,152 interleaved possession/location
answers. Actuator names and relation channels are swapped across seeds. Complete
physical state, source isolation, ablations and missing-evidence contracts are
checked independently. See [REQUEST_GROUNDING_EXPERIMENT.md](REQUEST_GROUNDING_EXPERIMENT.md).

This adds a supplied one-step search/retry controller with learned request and
effect meanings. It does not learn executable preconditions, exploration/search
policy, event-history goals, partial-observation language or long-horizon plans.
Meanings are learned incrementally during teaching, then frozen during acting;
simultaneous open-ended learning and standard-trainer integration remain open.
The grounding path remains symbolic and non-backpropagating, with previously
trained local primitive policies.

## Online request/effect learning in a continuing world

`online_grounding.h` connects selection, observation and feedback while request
and actuator meanings are still learning. Issued actions carry a feedback token,
request/episode revision and before-observation/model snapshot. One matching
trusted report can update both memories; duplicate, stale, generated or mismatched
feedback cannot. Current-state answers and further actions withhold while an
issued outcome is unobserved. A new request does not erase that uncertainty.

Starting with empty request/effect memories and the prior observation reader,
all three seeds complete 160/160 successive requests in a continuing world using
209 attempted actions each. Four successive curriculum windows introduce new
request constructions, a negative state goal, and renamed actuator interfaces.
Previously acquired meanings remain in the mix. A supplied information-seeking
controller can deliberately distinguish surviving meanings even when they agree
on a successful action. Uniform fallback completes 159, 159 and 158 requests in
278, 269 and 265 attempts, respectively, under the same 32-attempt request limit.

Frozen checkpoint copies transfer each acquired form to new names/worlds: 12,
24, 36 and finally 96 requests pass, with the last checkpoint testing both old and
new actuator IDs. Untaught wording is withheld. All 320 interleaved prior QA
answers per run and complete states pass independent replay. Exact goal/effect
meanings, rather than task completion alone, are checked at each checkpoint.
The native and sanitizer contracts pass; see
[ONLINE_GROUNDING_EXPERIMENT.md](ONLINE_GROUNDING_EXPERIMENT.md).

The feedback is explicit binary goal satisfaction plus an execution acknowledgment.
The body and observation/question language remain frozen; exploration and valid
action preconditions are supplied. This is online symbolic request/effect learning
over previously grounded text, not learned exploration, long-horizon planning or
a complete online byte trainer. Broader observation-language learning, partial
observations and general curriculum integration remain open.

## Learning new report meanings during interaction

`live_observation_grounding.h` extends the preceding online loop with observation
language learning. It stages an unfamiliar report against a complete historical
state, evaluates generic effect/binding hypotheses, and chooses known consequence
questions that distinguish them. Verified external answers prune the hypotheses;
predicted action effects and generated answers never provide teaching targets.
Only agreement on the complete resulting state allows the pending action to
receive feedback under the refined observation model. General frame admission
still requires four distinct supporting scenes.

Across three seeds, the same continuing interaction learns four report forms,
four request forms and two successive actuator interfaces. Each seed completes
160/160 requests, retains 320/320 earlier QA answers, and passes 96/96 frozen
transfer cases with fresh names and no teaching. Report grounding requires 25,
29 and 26 verified answers. Deleting the new report meanings makes every frozen
resulting state and completion judgment unknown despite physical action success.

Independent replay checks all transitions, hypotheses, question scores and
learned report meanings. Prediction traces expose eight occurrences where a
pretrained follow policy withholds on a hypothetical zero-edge state with a
defined symbolic answer. All definite predictions are correct; these gaps do
not change selected questions or retained meanings in the tested runs. They
remain a policy coverage limitation, rather than being counted as successes.
Native sanitizer contracts pass. Full protocol and reproduction:
[LIVE_OBSERVATION_GROUNDING_EXPERIMENT.md](LIVE_OBSERVATION_GROUNDING_EXPERIMENT.md).

Initial observation/question vocabulary, primitive policies and the neural body
remain frozen. This symbolic learning loop still requires known entities, two
relations, complete pre-states, supplied question/action menus and serialized
verified feedback. Partial-observation language, correction/history replay,
learned exploration, longer plans and standard-trainer integration remain open.

## Grounding partial-observation language

`observation_language.h` supplies generic evidence operations over positives,
negatives, complete rows/columns and lost visibility. The new
`observation_language_learner.h` searches their relation/argument bindings and
question procedures from raw stories with verified answers, including unknown.
It learns eleven evidence frames and six question meanings from 7,808 QA lessons
over 96 physical scenes. Repeated questions do not add independent support.

The runtime adds unique one-hop reads over complete scopes, distinct from learned
chain following. The operation is supplied; its selection and relation binding
are learned. Empty or multiple values withhold a unique answer, and open coverage
remains unknown. Complete incoming observations support inventory counts.
Learned literal words no longer become invented entities inside generic slots;
independently established names still preserve genuine lexical ambiguity. Event
interpretations now retain every recognized participant as a referent, including
a giver absent from the resulting ownership fact.

Across three seeds, frozen native evaluation passes 19,273/19,273 questions and
complete positive/negative/coverage states per seed, including 5,456 unknown
targets. The 169 selected original Stage 5 questions include 42 histories with
visibility gaps and eight unknown targets. This is a bounded selection from
1,543 source questions, supplemented by repeated controlled templates over 228
histories. No exact normalized story/question overlaps the new or prior teaching
sets. Removing visibility meanings blocks 9,402 cases; substituting chain reads
for unique reads reduces exact answers to 16,685. Independent audits, sanitizer
contracts and the live/incremental/reference regression suites pass.

This is batch induction with pretrained event language and primitive policies,
followed by stateful evaluation with frozen meanings. It does not yet acquire
visibility language inside the live action loop, which still requires complete
pre-states. Temporal/provenance history, corrections with dependent replay, broad
grammar and standard-trainer integration remain open. Full protocol and limits:
[OBSERVATION_LANGUAGE_EXPERIMENT.md](OBSERVATION_LANGUAGE_EXPERIMENT.md).

## Live report learning under partial observation

`live_evidence_grounding.h` stages an unfamiliar report against all retained
knowledge histories. It searches generic evidence/update operations and bindings,
then asks familiar questions that distinguish those hypotheses. Verified external
answers, including unknown, prune rule/history pairs. Publication preserves every
surviving history and its consistent frame choices. A meaning can remain ambiguous
even when all interpretations agree on the current observed state.

The shared actuator predictor now updates hypothetical positives, negatives and
coverage together. Previously learned actions can operate on partial knowledge
when every retained interpretation establishes the goal. Predictions do not
populate observations. Frozen online control accepts those observations and
verified report-model refinements; learning new action effects still requires
complete before/after states.

Three seeds each complete 160/160 requests over 347 reports and 187 attempted
actions, including 27 refusals and 93 choices made with unknown current goal truth.
The learner acquires five report frames from 35/35/36 verified answers. Exhaustive
retention checks pass 91,608/91,608 questions and full knowledge states per seed;
48 fresh-name worlds pass 192 frozen gap/recovery questions each seed. These are
repeated controlled tests, not independent general-language tasks. The movement
and refusal frames each retain two observationally indistinguishable meanings.
An old refusal interpretation that worked in closed worlds needed further
grounding once coverage could be missing.

Request/effect models, question language, primitive weights and the body remain
frozen in this experiment. Executable affordances and question menus are supplied.
Independent replay verifies physical/public states, candidate/history filtering,
question choices, model support and held-out answers. Broader joint learning,
correction with dependent history replay, learned search, new concepts, longer
plans and standard-trainer integration remain open. Full protocol and limits:
[LIVE_EVIDENCE_GROUNDING_EXPERIMENT.md](LIVE_EVIDENCE_GROUNDING_EXPERIMENT.md).

## Identified evidence corrections and dependent history replay

`revisable_grounding.h` retains a starting snapshot, ordered raw reports and
source-identified, versioned consequence-answer records. Explicit replacement or
withdrawal reconstructs report meanings and knowledge from the active evidence.
Previously eliminated hypotheses can return. Consistent frame choices preserve
dependencies across reports, and final surviving choices also refine historical
views. A correction applies at the original report's position; later valid
events remain in force. Replay failure blocks current answers, and a later
repair can recover within the same dialogue episode.

`history_revision` distinguishes a changed evidential history from a new physical
observation. The online and live language learners now reject stale pending
feedback across that revision, even if the model signature is unchanged.
Correcting a report does not count as observing a pending action's outcome.

Three seeds each process fifty report/feedback/revision operations without an
episode reset, recovering two new report meanings after deliberately wrong
teaching. Independent replay verifies every retained current/historical world,
model candidate, support count and provenance revision. Forty-eight fresh-name
worlds pass 1,536 frozen current/historical questions per seed, with full knowledge
states checked. An initial visibility curriculum admitted spurious extra-location
rules because both missing and nonunique locations withheld a unique answer;
contrasting membership feedback removes that ambiguity without an English rule.

This supplies bounded record-and-replay provenance, with externally identified
corrections. It does not learn natural-language temporal references, discover
which source to distrust, provide incremental persistent dependency indexes, or
jointly revise question language and neural policies. Initial question meanings,
primitive policies and the body remain frozen. Protocol and limitations:
[REVISION_GROUNDING_EXPERIMENT.md](REVISION_GROUNDING_EXPERIMENT.md).

## Learned references to historical reports

`history_reference_grounding.h` learns whole reference frames from ordinary
verified answers about historical states. It searches generic record-field
bindings, activity filters and chronological ranks, without English reference
keywords or target report IDs in teaching. Six definite frames acquire unique
selectors; a deliberately underdetermined seventh retains first/last/all
alternatives and requests clarification when they disagree. Record IDs are
shuffled independently of journal order, and field identifiers swap across seeds.

The learned reference selects a report for replacement or withdrawal through the
existing revision journal. Pending clarification is bound to the model, episode,
observation, history and record versions. Invalid choices preserve the request;
stale contexts, generated inputs and source mismatches cannot revise evidence.
Withdrawn reports retain their text so references can still identify them.
The correction operation, reference span, payload and source identity remain
separate supplied interface fields; only report-reference meaning is learned.

From 784 historical-answer lessons over 112 scenes, all three seeds pass 144
frozen correction cases: 72 direct edits, twelve clarified edits, and sixty
refusals for stale, missing, unknown, unauthorized or generated input. Independent
replay verifies 5,484 current/historical states and 1,596 answers per seed.
An initial corpus passed the edit cases but left activity and rank confounded;
varying withdrawal positions independently resolves that ambiguity. The revision
regression and new sanitizer contracts pass. Protocol and reproduction:
[HISTORY_REFERENCE_GROUNDING_EXPERIMENT.md](HISTORY_REFERENCE_GROUNDING_EXPERIMENT.md).

This remains bounded symbolic learning with fixed question meanings, known
entities, supplied metadata and frozen primitive policies/body. It does not
establish unrestricted temporal grammar, learned source trust, automatic error
discovery or joint continual learning across the complete model.

## Grounding complete correction constructions

`correction_language_grounding.h` identifies already learned reference/report
spans inside whole user requests and learns an outer construction over typed
placeholders. Verified answers about intended correction consequences select
among generic preserve, withdraw and replace operations. Hypotheses execute only
on private journal copies during teaching; no operation label, target report ID,
reference boundary or replacement boundary is supplied. The learned construction
then invokes the existing revision and clarification mechanism. Pending choices
also invalidate when correction-language meanings change.

From 504 post-correction answers over 72 histories, five definite constructions
acquire their intended operations. A deliberately weak sixth retains both
preserve and withdraw and withholds where those interpretations disagree. Each
of three seeds passes 252 frozen requests, including 96 direct edits, 24 edits
after clarification, 24 preserve requests and 108 refusals/ablation controls.
One hundred direct request cases use reference constructions not seen during
correction teaching; 36 replacements use a previously taught event construction
never used as a correction payload in training. Independent replay checks 9,588
current/historical states and 1,764 current answers per seed.

The caller now supplies a whole request and source identity, rather than separate
operation/reference/payload fields. Span types, three primitive operations,
trusted metadata, verified feedback and the clarification protocol remain
supplied. Reference/report/question meanings and primitive policies/body are
frozen. This does not establish simultaneous open-ended learning or standard
trainer integration. Protocol, sanitizer contracts and reproduction:
[CORRECTION_LANGUAGE_EXPERIMENT.md](CORRECTION_LANGUAGE_EXPERIMENT.md).

## A grounded session on the standard trainer's byte path

`grounded_session.h` owns the reader, journal and correction dispatcher. An
optional `SequenceMemory` observer delivers committed bytes exactly once;
`Organism::attach_grounded_observer` pairs it with the reader used for replies.
Role envelopes append environment reports automatically, and learned meanings
route user requests among ordinary questions and corrections. No per-turn task
category, correction target or operation is supplied. Clarification binds both
the historical context and the originating source; generated bytes cannot become
evidence. A new episode clears history and pending choices while retaining models.

`laminar_train grounded_bootstrap=PATHS` reconstructs the earlier learned models
from an explicit teaching manifest and installs this session. Bootstrap uses
runtime checks in release builds. Symbolic meanings stay frozen during ordinary
stream ingestion while the existing byte learner operates. Checkpoint probes
receive private copies of grounded state, including incomplete input segments.
The generation helper now preserves the first prompt byte by supplying the
leading carry expected by `run_block`; dropping that newline had corrupted probe
resets taken in the middle of messages. Grounded stream ingestion also supplies
its initial carry so a file's first role marker is complete.

Across three seeds and 1/7/113-byte chunks, each release run passes 408 replies
over 564 mixed turns, 60 corrections, 24 correct unknown answers, 2,976 complete
current/historical state checks and twelve checkpoint-isolation checks. Separate
actual-trainer runs compare final state with and without intermediate probes.
This initial session test keeps meanings frozen. The follow-up below adds
verified observation-language teaching. Physical actuation and persistence of
learned memories remain separate.

Integration also exposed a preexisting packed-allocation overflow in small
populations. Absent K classes now return unavailable, and uneven motif counts
wrap within the selected class. Sanitizer tests cover absent/partial/uneven
classes and exhaustion; this fixes placement semantics and can change older
uneven-population benchmark runs. Full protocol and reproduction:
[GROUNDED_SESSION_EXPERIMENT.md](GROUNDED_SESSION_EXPERIMENT.md).

## Verified observation teaching in the running session

`GroundedSession::teach` accepts host-verified question/answer evidence bound to
an episode, report revision, history revision and observation model. Unauthorized,
generated and stale feedback is rejected. Identified evidence can be revised or
withdrawn by its original teacher, allowing journal replay to recover from a
mistaken interpretation while retaining later events.

`laminar_train grounded_feedback=FILE` delivers these receipts at exact byte
offsets in the training file; they are separate from ordinary dialogue. The
trainer splits blocks at those offsets and isolates teaching state from probes.
The allowed identity defaults to `verified_teacher`, configurable through
`grounded_teacher=ID`. This identity is host metadata, not a learned credibility
judgment. Acquired frames retain globally distinct teaching-scene support across
episode resets, and their candidates can keep narrowing. Bootstrap frames remain
anchored; partially supported acquired frames are usable in teaching replay,
with four-scene admission retained for ordinary language inference. An unresolved
contradiction at reset rolls back that episode's observation-model changes.

Three seeds each acquire four report constructions from 288 verified answers over
eight episodes, then pass 600/600 replies in eight fresh-name episodes, including
24 corrections with newly learned report payloads. Independent replay checks
3,600 current/historical states and 4,060 frame/support snapshots per seed. A
no-feedback control scores 184/600. The actual byte trainer, with learning enabled,
finishes with identical models and states with or without intermediate probes.

This extends symbolic observation learning inside the trainer. Prior question,
reference, correction and primitive-skill models remain frozen. Completed-episode
candidate sets are consolidated without editable teaching provenance in the
default mode. The optional archive below addresses correction across resets.
Broader joint acquisition, physical actions, open concepts and learned exploration
remain open. See [GROUNDED_TEACHING_EXPERIMENT.md](GROUNDED_TEACHING_EXPERIMENT.md).

## Reconstructing meanings from cross-episode evidence

`grounded_archive=1` saves raw episode contexts, report records and identified
teacher-answer versions. `GroundedArchive` replays them from a fixed bootstrap,
carrying correlated meaning alternatives across physical-world resets. Revised
old answers can recover discarded interpretations while preserving later events
and keeping episode facts separate. Contradictions withhold answers across resets;
capacity exhaustion cannot silently discard evidence and resume authoritative
inference. Current pending decisions invalidate after accepted remote edits.

Three seeds independently pass 351 replies, 5,472 knowledge states and 113,040
receipt-version snapshots each. The actual trainer accepts 313 receipts and ends
with identical state and learned meanings with or without checkpoint probes.
Five of 23 checkpoint QAs withhold during unavailable evidence history; the final
one recovers. Sanitizer tests include correlated alternatives, old corrections,
teacher/source/version guards, clone isolation, capacity failure and mid-context
feedback. This is bounded in-memory replay, with no disk persistence or joint
updates to the fixed bootstrap. See
[GROUNDED_ARCHIVE_EXPERIMENT.md](GROUNDED_ARCHIVE_EXPERIMENT.md).

## Learned computation control: local and backprop comparison

`computation_control.h` exposes public memory status, observed actuator state,
request intent and operation costs to one policy choosing retrieve, reason, ask,
act or return. It receives terminal outcomes and actual costs, with no supplied
next-step labels or goal graph. The native local-update arm and a two-layer
backprop comparator each pass 720/720 endpoint cases across three seeds, including
32-link paths after training on 0–3. Both retain unnecessary actions. Frozen and
randomized-outcome controls perform worse; a terminal-decision-only control also
passes all endpoints, so this does not close the long-credit gate.

The new backprop implementation passes an ASAN/UBSAN finite-difference check on
all parameters. The experiment independently audits transitions, outcomes, costs
and operation-removal cases. Its state abstraction, primitive effects and
request semantics are supplied, and it is not integrated into the grounded
session. Backprop is permitted for the initial complete-learner demonstration;
the matched local-only demonstration remains a subsequent objective. Protocol,
full controls and limitations:
[COMPUTATION_CONTROL_EXPERIMENT.md](COMPUTATION_CONTROL_EXPERIMENT.md).

## Outcome-discovered primitive policies and neural rehearsal

`policy_discovery.h` proposes reusable finite policies and scores their complete
answers across several examples. It mutates decisions encountered in candidate
executions; no evaluator supplies the correct next instruction. Once a candidate
explains all training answers, execution cost ranks alternatives. Sealing keeps
only observed states and withholds on unknown states/unavailable operations.
The supplied search, binary state abstraction and ten VM operations remain fixed.

Three seeds discover following, typed equality and distinct counting from short
final-answer examples and pass 8,100 primitive tests, including 64-object counts
requiring up to 353 instructions. The discovered policies generate their own
verified short traces for the actual `LocalChoice` learner. Rehearsal preserves
earlier skills as the same neural weights acquire later ones; the matched
no-rehearsal arm loses those earlier skills. These weights then support
4,200/4,200 questions (2,400 composites) in the existing `ProcedureMemory`, without loading
teacher micro-instruction files. Direct backprop-TD remains a failed counting
baseline in its tested configuration; the working path uses symbolic policy
search and local plasticity. Their sample/compute budgets are not matched.

Independent interpreters replay all logged primitive decisions and composition
programs, and sanitizer tests cover the discovery/sealing contracts. This is a
staged primitive/procedure acquisition experiment. Language, physical actions,
evidence revision and control do not update beside it; the archive still requires
a fixed bootstrap. Protocol, controls and reproduction:
[PRIMITIVE_DISCOVERY_EXPERIMENT.md](PRIMITIVE_DISCOVERY_EXPERIMENT.md).

## Learned relational definitions and new argument roles

`concept_memory.h` adds a separate experimental memory for variable-arity
relational definitions. It induces conjunctions and shared/existential bindings
from raw fact worlds and verified query outcomes, using generic subgraph search
over a positive witness. Unambiguous learned definitions become available
predicates in subsequent induction. Head-preserving query containment removes
redundant macro expansions, and all shortest consistent alternatives are retained.
The new concept's name and head arity are provided; its definition is learned.

Three independently renamed vocabularies each acquire eight concepts with up to
four head roles, introduce new unary/ternary observed predicates, and pass 1,600
fresh final queries. Independent checks verify the learned relational semantics
as well as sampled answers. Reuse expresses up to eight base literals within a
three-local-literal budget. Without reuse, the same bound passes only 400/1,600;
a wider eight-literal flat control also passes all cases. This measures modular
representation under a bound, not an intrinsic speed advantage over flat search.

Raw evidence and versioned edits support rebuilding, recovery of alternatives,
and dependency invalidation or independent re-learning. Partial worlds withhold
negative answers, and incomplete search withholds rather than accepting a prefix.
The curriculum deliberately uses sparse witnesses and distinguishing role
contrasts. Query structure, relational logic and search remain supplied, and
teaching worlds are complete. The initial experiment runs independently of the
text reader, neural skills, computation controller, actions and fixed-prior archive.
Broader concept invention and the joint learner remain open. Protocol and audit:
[CONCEPT_GROWTH_EXPERIMENT.md](CONCEPT_GROWTH_EXPERIMENT.md).

## One relational session for concepts, language, effects and procedures

`relational_session.h` connects `ConceptMemory`, the existing `GroundedActuator`,
new variable-arity language frames and procedures derived from its own successful
executions. A shared version invalidates decisions whenever evidence changes.
Rebuild re-derives concepts and effects from raw receipts, interprets language
against the current concepts, and checks every stored procedure against current
meanings and effects. Execution feedback supplies observed transitions and one
terminal satisfaction bit; the same evidence updates effects, language and
procedural support. No correct next action or external procedure trace is supplied.

Across three interleaved curricula, the same model passes 2,304 fresh final
language queries and 144 fresh final action goals. Learned procedures pass 72/72
transfer requests under a tight prediction budget; search on identical snapshots
passes 25/72. With the normal budget, that search control also passes all final
goals. One initially failing bounded search is retained in the experiment history;
generic pruning based on learned literals and effect scopes resolves it under
the same cap. This pruning and exploration remain supplied algorithms.

An old concept contradiction invalidates dependent language and procedures;
correcting it restores the exact prior coupled models. Own-execution observation
corrections update dependent transition worlds and terminal language evidence
atomically. Stale or untrusted feedback cannot partially update the session.
The current implementation uses global replay and conservatively withholds all
language/planning while any concept is unresolved. It has no disk persistence.

This integration uses symbolic learning and controlled complete-world effect
observations. The neural primitive policy and learned computation controller are
still separate, rebuild/search scheduling remains supplied, and executed plans
in this test have at most three actions. The original byte reader and fixed-prior
archive retain their behavior. Full protocol and independent audit:
[RELATIONAL_SESSION_EXPERIMENT.md](RELATIONAL_SESSION_EXPERIMENT.md).

The follow-up `relational_control.h` now couples a learned neural operation
selector to this session. Only explicit `reason` decisions compute answers,
rebuild knowledge or expand the search allowance. Timestamped reads cannot
overwrite newer action results. Terminal controller credit and observed-world
satisfaction remain separate, with versioned evidence and exact neural replay
after corrections. Empirical TD plus backprop policy fitting passes 308/384 final
cases across three seeds; direct neural Double-Q passes 103/384. Queries pass
192/192 but actions only 116/192, and removing retrieval does not reduce success.
Control is therefore integrated but unreliable. Search and exploration remain
supplied, one motor batch is allowed per episode, and neural primitive skills
remain separate. Protocol: [RELATIONAL_CONTROL_EXPERIMENT.md](RELATIONAL_CONTROL_EXPERIMENT.md).

The [reactive follow-up](RELATIONAL_REACTIVE_EXPERIMENT.md) permits repeated
single motor steps before one terminal credit event. Failed executions remain
in the journal but do not teach invented effects; procedural induction omits
unexecuted steps, and cached planning skips already completed no-op prefixes.
The neural arm passes 354/384 final cases and 120/150 recovery cases after three
failed attempts, with successful traces up to 60 decisions. Retrieval now matters
when sensing is unavailable. An optional learned consequence model estimates
operation values from observed transition counts, with independently checked
Bellman equations; its matching controls distinguish useful nonterminal credit.
These are bounded structured experiments. Some action tasks still fail, external
world changes need episode restart, and the primitive neural learner remains
separate. Details and limits are in the linked report.

The [computational-session follow-up](COMPUTATIONAL_SESSION_EXPERIMENT.md) now
connects outcome-discovered neural primitives to this continuing model.
`PrimitiveStore` owns versioned raw lessons, supplied policy search and
self-generated local-plasticity rehearsal. Typed computational language learns
programs through the actual neural `ProcedureMemory`; relation arguments may
refer to newly learned binary concepts, materialized temporarily from the current
world. The same computed meaning evaluates hypothetical actions and observed
outcomes. Primitive/concept/evidence revisions invalidate dependent programs,
physical procedures and stale control tickets, with value-owned clone state.
Three seeds pass 1,008 questions, 120 count-dependent physical goals and 648
retained primitive cases. The learned controller interface supports these
computations and meters neural inference work, but operations in its new contract
test are forced; learned control over this combined curriculum remains untested.
An initial short curriculum failed composition despite perfect primitive-training
scores. Wider primitive lessons resolved that case; autonomous discovery of the
needed curriculum and cross-level credit remain open.

The [computational-control experiment](COMPUTATIONAL_CONTROL_EXPERIMENT.md) now
trains this combined loop without forced operation choices. Typed patterns in raw
evidence route terminal feedback correctly even before semantic compilation;
previous computational programs are reused only after revalidation against every
current lesson through the actual neural skills. Three seeds pass all 384
standard and 89 three-failure recovery cases while acquiring the components,
but only 30/36 larger goals. The primary policy explores after search exhaustion;
an optional public-inventory-size input teaches further reasoning but drops larger
success to 21/36, with some completed plans followed by retrieval loops. The
option remains off by default. All primitive checkpoints match the preceding
retention audit. General control transfer and broader representation/credit
learning remain unfinished.

The [control-transfer diagnosis](CONTROL_TRANSFER_DIAGNOSIS.md) adds optional
search-allowance retention across unchanged-scene revisions. Answers and plans
are still invalidated; changed facts or completeness reset the allowance. With
the same frozen baseline weights this resolves all six larger-goal failures,
giving 36/36 while retaining 384/384 standard and 89/89 recovery cases. The option
is not yet the training default. A generic preference-tree readout of the same
empirical values transfers worse, exposing the difference between faithfully
copying learned values and learning a useful computation policy. General
primitive repair and growing the supplied representation interfaces remain open.

The [composed-credit experiment](COMPOSED_CREDIT_EXPERIMENT.md) introduces
`ComposedSkillDiscovery`: joint micro-policy and macro-program search scored
only by whole question answers. It considers currently unavailable primitive
slots and retains partial answer fitness during generic program synthesis;
only complete teaching fits become executable learned knowledge. Candidate
search leaves live neural weights and knowledge untouched. Self-generated
successful execution traces plus old-skill rehearsal install the result into
the actual shared neural `ProcedureMemory`.

With following and equality learned beforehand, three seeds acquire missing
counting from 64 composed Boolean lessons without direct counting lessons or
external instruction traces. They pass 1,680/1,680 fresh composed tests, 324/324
standalone count tests through 64 objects and 648/648 earlier-skill tests. Half
the composed tests follow teaching of additional procedures with frozen primitive
weights. Without old-skill rehearsal, composed success falls to 280/1,680.
A separate numeric curriculum succeeds in two of three seeds. Search takes
billions of simulated instructions per run; this is not efficient online credit.

`ComposedLearner` adds versioned raw outcomes, dirty-answer withholding, checked
cache reuse, contradiction handling and withdrawal/restoration of derived
skills. A lifecycle run learns a later composition with unchanged primitive
weights and retains all three skills after restoring evidence. This typed-query
acquisition route is not yet integrated into `RelationalSession`'s language,
concepts, physical actions and learned controller. Automatic repair of an
available wrong skill, growth beyond the supplied operation/type bank and
retention across that complete continuing loop remain required.

## Things deliberately not optimised away (runbook §15)

Novelty bursts, motif K variation, strict INT4, early death, local credit, consequence-based
splitting and structural ambiguity are all live mechanisms with parameters, not benchmark switches.
