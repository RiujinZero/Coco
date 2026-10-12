# Body and circuit research log

Started 2026-09-24. **Keep this log current: every experiment, its exact settings, its numbers and its verdict.**

## Goal (user, 2026-09-24)

A complete learner as defined in the original design doc (`Laminar_200k_Organism_Design_Doc.docx`), the runbook
(`CPP_OPTIMIZATION_RUNBOOK.md`) and the handoffs (`docs/COMPLETE_LEARNER_HANDOFF.md`,
`docs/CONTINUING_LEARNER_HANDOFF.md`). Keep researching until:

1. **Every layer of a body does the job the design gives it** (table below).
2. **Bodies form circuits more capable than any single body alone.**
3. **The bodies' computation improves SKILLS beyond the sequence memory**, not just bits/byte. User, verbatim:
   "If it's not adding real computational improvement over the sequence memory it's still not working right" and
   "Not bpb, for the skills."

The complete-learner handoff's own five gates (repair/acquire computation in the continuing trainer, learned
computation choice, long-term credit and autonomous goals, growth beyond the supplied representation, one continuing
language/reasoning/action learner) apply to the structured learner (controller + Agkern), a separate model mode.
This log covers the organism's bodies and circuits, the substrate those gates will eventually need.

## The acceptance instrument: `cpp/apps/model_conformance.cpp`

Read-only, on any checkpoint. `build-task/conform.sh RUN...` (env `ITEMS=~/babylm/items_fast STRIDE=4` adds skills).
Each ARM is run from the same post-calibration state in a forked child: actual; silent bodies (L6 = 0); constant
bodies (each emits its own mean L6); constant L1/L2/L3/L5 (layer frozen at the body's own mean); no neighbour
routing; no neighbour messages into L3; no L4 return. Checks:

| Check | Claim (design doc / runbook) | Bar |
|---|---|---|
| C1 | bodies add value over the sequence memory (full model bits/byte) | >= 0.05 |
| C2 | context computation in the bodies adds value over the memory (full model) | >= 0.02 |
| C3.Lx | each hidden layer does work (body-only readout) | >= 0.01 |
| C4 | neighbour routing (L2 -> K neighbours) does work | >= 0.01 |
| C5 | neighbour messages inform L3 | >= 0.005 |
| C6 | the L4 return does work (when L4 runs) | >= 0.005 |
| C7 | early information death: admitted bodies dying before L6 | >= 10% |
| C8.Lx | no layer pinned at the clamp | saturated <= 25% |
| **S1.task** | bodies are needed for the skill (BLiMP, entity tracking, supplement) | >= 1 point |
| **S2.task** | **context computation in the bodies is needed for the skill** | >= 1 point |
| (to add) C9 | circuits: multi-body events beat the best single body | TBD |

Other instruments: `cpp/apps/model_body_probe.cpp` (per-layer saturation, sign flips, weight movement since birth,
traffic split), `cpp/apps/model_unit_check.cpp` (training curve; synthetic Markov source with known entropy, or text;
frozen vs learning held-out), `build-task/bigram.py` (n-gram references).

## Layer roles: design vs observed (update as fixed)

| Layer | Design doc role | Status (2026-09-24) |
|---|---|---|
| L1 | processes accepted input; may die early | works with structured input (C3 +0.17). Never dies (C7 fail). 55-80% saturated. |
| L2 | L1 + L4 return; route key to K neighbours; feeds L5 | did nothing while the untrained L4 return was half its input; works with the return off (+0.06). Routing it drives HURTS (C4 -0.20). |
| L3 | L1 + accepted neighbour messages | works (+0.06); neighbour messages help a little (C5 +0.02) although routing overall hurts |
| L4 | selective recurrent return / route consequence | never learns (0% weights changed); ran on almost every event (L6 saturated above recirc threshold). Off (`recirc=128`) until it has a learning rule. |
| L5 | L2 + L3 -> L6 | works (+0.18) |
| L6 | output to workspace and L4 | learns after the gain fix; 81% saturated |

## Chronology

### 0. Starting point (before 2026-09-24)
20k bodies, Strict-Small, one epoch, market off: held-out 2.932 bits/byte, BLiMP fast 52.6, entity tracking 17.96.
Default: `hidden=modulated` (three-factor), `err_gain=8`, `err_sparse=0.01`, blended-error L6/readout learning,
fixed shifts `1,2,2,3,3,3`, hops=4, L4 return on. Market variants (gated, contest, protect/throttle) all lost on
text (3.06-3.50) with BLiMP within noise; culling general bodies was not the cause (protect run: 278 culls, 3.095).

### 1. Bodies contribute nothing (probe on trained checkpoints)
- Silencing ALL bodies changed the full model by +0.015 / -0.017 bits: all prediction came from the sequence memory.
- 99% of L6 outputs at the +-127 clamp; each body a constant 32-bit sign code (0.4% flips). Hidden layers 50-90% saturated.
- Hidden weights at fixed shifts `4,...` vanish (L6 mean 0.5); the gain structure is bistable (ws-memory inputs at +-127
  saturate everything; without them the signal decays).

### 2. Fixes that worked (smoke = 4 MiB, one seed)
| Change | Smoke held-out | Note |
|---|---|---|
| default | 3.384 | bodies cost 0.4 bits vs silent |
| `body_err=1` (bodies learn their OWN error, not the post-blend residual) | 3.023 | |
| + `homeo=1` (per-body per-layer adaptive shift) | 2.943 | homeo HURTS on the clean synthetic test (per-event shift jitter); needs slow adaptation |
| + `hidden=backprop` (exact within-body gradient into the same INT4 accumulators) | 2.823 | |
| `in_struct=1` (byte features of t..t-4, word/sentence position, normalised ws memory) | ~same | inputs alone did not help until later fixes |

### 3. The output path itself was broken (`model_unit_check`, synthetic order-1 Markov source, entropy 2.64)
- Default: training loss ROSE while training (3.06 -> 3.36); frozen worse than learning (a tracker, not a model).
- `err_gain` 8 -> 2 (16-bit accumulators): frozen 2.72 (near optimal). At 8 the quantised L6 error clips at +-127 on
  most updates (sign-only noise). `err_sparse=0` also required. 8-bit accumulators drop sub-quantum updates.
- Real text, broad bodies only (158 bodies, no routing/memory), refs bigram 3.68 / trigram 3.11:
  none 3.64, **backprop 3.42, backprop+in_struct 3.36**, perturbation (XOR body's rule) 3.94, three-factor 4.34.
  => bodies CAN learn context functions; the three-factor rule is destructive; perturbation too noisy at this scale.

### 4. Full epochs (Strict-Small, 20k bodies)
| Config | Held-out | BLiMP | Supplement | Entity |
|---|---|---|---|---|
| baseline | 2.932 | 52.6 | 49.6 | 17.96 |
| homeo+body_err, three-factor / backprop / none | 2.877 / 2.797 / 2.824 | 53.0 / 52.6 / 52.2 | | 17.6 / 17.0 / 17.5 |
| body_err, err_gain=2, err_sparse=0: none / backprop / backprop+in_struct | 2.746 / 2.708 / 2.710 | 52.0 / 52.3 / 52.7 | 46.8 | 17.2 / 17.0 / 17.2 |
| + L4 return off (`recirc=128`) | **2.701** | 52.8 | 47.2 | 17.2 |
**Bits/byte improved 0.23; skills did not move.**

### 5. Layer ablations and the L4 finding
Freezing each layer to the body's own mean (body-only loss, in_struct+backprop): L1 +0.09, L2 +0.004, L3 +0.06,
L5 +0.09. L2 did nothing because half its input was the untrained, stale L4 return. With the return off: L1 +0.14,
L2 +0.06, L3 +0.06, L5 +0.13; constant bodies cost +0.094 body-only.

### 6. Conformance suite, first runs (smoke; 128 KiB held-out slice)
Best smoke (in_struct+backprop, return off): 6 of 14. C2 FAIL (+0.001), C4 FAIL (routing -0.20), C7 FAIL (0% early
death), C8 FAIL (all layers 56-82% saturated).
- `hops=0` (routing off): better loss, 1/3 fewer MACs.
- **The blend with the sequence memory is miscalibrated**: seq_alpha 0.8 -> 0.5 full 3.096 -> 2.753 on the slice;
  0.3 -> C2 passes (+0.020).
- Learned blend gate `seq_gate=1` (6 features, exact gradient of the blended loss): smoke held-out 2.363 (vs 2.428 at
  alpha 0.3 and 2.787 before); C2 +0.009. With `hidden=none` + gate: held-out 2.339 but C2 -0.018 (context computation
  hurts). body_err=0 + gate: 2.347, C2 +0.002.
- Skill arms (S1/S2) added to the suite; first runs pending (see below).

## Current best configuration (smoke)
`body_err=1 err_gain=2 err_sparse=0 in_struct=1 hidden=backprop recirc=128 hops=0 seq_gate=1`

## Open problems (ranked)
1. **S2: does context computation in bodies improve skills?** Measure first (skill arms running).
2. **Circuits (C9)**: no multi-body composition exists that beats single bodies. Routing hurts; median body sees ~100
   events (smoke), nothing is shared between bodies. The fruitfly XOR-1024 result came from ONE shared motif reused at
   every node (weight sharing = "cell types"); Laminar bodies are all independent. Candidate: typed bodies (many locks
   -> one shared-weight body type), composition via workspace/L3 messages.
3. **Saturation (C8) and early death (C7)**: need a slow homeostatic gain (EMA with hysteresis), then death thresholds
   that actually bite.
4. **L4**: needs a learning rule (credit across events) or stays off.
5. **Learning rule**: backprop works; three-factor is destructive; perturbation (the XOR body's rule) is too noisy at
   BabyLM data scale. The design wants local learning; backprop is the permitted reference first (handoff: "backprop to
   establish the complete learner exists first, then a matched non-backprop implementation").

## Code added on 2026-09-24 (all opt-in; defaults unchanged)
`hidden=backprop`, `bp_gain`, `bp_ste`; `body_err`; `homeo`, `homeo_lo`; `in_struct`, `in_scale`; `seq_gate`,
`seq_gate_lr`; market `market_contest`, `market_protect`, `market_birth_ratio`, `--gate-tol`; diagnostics
(`on_event` accessors, `l6_hook`, `layer_hook`, `diag_no_routing`, `diag_zero_msgs`, `diag_zero_return`,
`weights_changed`, `unit_hits`); apps `model_body_probe`, `model_unit_check`, `model_conformance`.
Uncommitted as of this entry.

### 7. Slow homeostasis (`homeo=2`, running averages, shift moves only on threshold crossings)
Clean harness (`model_unit_check`, err_gain=2, 16-bit accumulators):
| Setting | Synthetic (entropy 2.64) frozen | Text broad bodies, backprop, in_struct frozen |
|---|---|---|
| off | 2.719 | 3.361 |
| homeo=2 (all layers incl. L6) | 3.035 | 3.671 |
| homeo=2, homeo_sat=0.05 | 3.004 | 3.680 |
| homeo=1, hidden only (`homeo_out=0`) | 2.716 | 3.389 |
| homeo=2, hidden only | 2.718 | 3.384 |
Verdict: rescaling L6 (the emission a SHARED readout reads) is what hurt. Hidden-only gain control is harmless but
does not help: saturation of hidden layers is not what limits learning. C8 demoted to informational (the design
doc never requires unsaturated layers); C7 (early death) remains a design requirement.

### 8. Circuits: cross-body gradient through neighbour messages (`bp_route=1`)
A routed body's L3 reads its sender's L2. With bp_route the gradient at that input flows back into the sender's L2
(traces processed last to first; senders accumulate their receivers' needs), so senders learn what receivers need:
a trainable two-body circuit. Smoke + conformance running (r_h4 = routing without, r_h4bpr = with, r_h1bpr = 1 hop).

### 8 (results). Neighbour routing and the cross-body gradient
Smoke, config `body_err=1 err_gain=2 err_sparse=0 in_struct=1 hidden=backprop recirc=128 seq_gate=1`:
| Arm | Held-out | Full | C2 | C4 routing | C5 messages |
|---|---|---|---|---|---|
| routing off | 2.363 | 2.654 | +0.009 | - | - |
| routing (hops=4) | 2.463 | 2.767 | +0.015 | -0.271 | +0.032 |
| routing + `bp_route` | 2.440 | 2.731 | +0.029 | -0.211 | +0.042 |
| 1 hop + `bp_route` | 2.399 | 2.711 | +0.004 | -0.132 | +0.015 |
New arm "routed bodies silent" == "no routing" exactly (2.630 for r_h4bpr): ALL the harm is routed bodies writing
into the workspace. A routed neighbour is another context's specialist running on the wrong context. Trained WITH
routing+bp_route and evaluated without it, the model beats the never-routed one (2.630 vs 2.654).

### 9. Route consequence through L4 (`route_return=1`), the design doc's L4 role
Receivers are silent; their L6 goes through THEIR L4 back to the sender, which recomputes L2->L5->L6 and alone emits.
Backprop follows (sender -> receivers' L4 -> their body): L4 learns for the first time. `bp_route` has no effect in
this mode (needs the opposite trace order). New arm "no route consequence"; C6 measures it; new C9 (circuits: full
loss without the consequence minus with it, bar 0.01).
- K-neighbour receivers: routing becomes neutral (held-out 2.362 vs 2.363; C4 -0.004) but C9 -0.002: returns carry nothing.

### 10. Shared cell types (`shared_types=N`, `nbr_route=0`, hops=1, route_return=1)
The last N units belong to no lock; every specialist routes to one (fixed hash wiring) and gets its computation back
through L4 (fruitfly: one motif shared everywhere learned XOR-1024).
| N | Held-out | C2 | C6 (body) | C9 (full) |
|---|---|---|---|---|
| 64 | 2.382 | **+0.189** | +0.060 | -0.013 |
| 256 | 2.388 | +0.138 | +0.002 | -0.014 |
| 1024 | 2.355 | +0.096 | +0.007 | -0.012 |
First time context computation clearly matters (constant bodies cost 0.10-0.19). Circuits help the bodies' own
prediction but slightly hurt the blended one.

### 11. Literature (Consensus, 2026-09-24)
- DeepSeekMoE (Dai 2024): shared experts ALWAYS active, output ADDED to routed experts'; stops routed experts
  relearning common knowledge (2B matches GShard 2.9B). OLMoE (Muennighoff 2024): replacing a routed expert with a
  shared one slightly worse. Li 2026 (2,000 runs): expert count and granularity matter more than shared size.
- Fine-grained MoE: many small experts, SEVERAL active per token and combined; OLMoE compute-optimal ~256 experts;
  "standing committee" of experts takes most routing mass (Wang 2026). Coco: 20k experts, ~1.9 active per byte.
- kNN-LM / cache LMs: fixed lambda -> learned per-token gates (SPALM 2021, Bakhtin 2018 gate on output probabilities,
  Drozdov 2022 retrieval-quality gate +4% ppl); everyone mixes LINEARLY; TRIME (2023): train the LM WITH retrieval;
  Wang 2023: fixed interpolation helps a few tokens and hurts most.

### 12. Literature-informed changes (smoke)
| Change | Held-out | C2 | Note |
|---|---|---|---|
| geometric blend + gate (c_gate) | 2.363 | +0.009 | reference |
| `seq_linear=1` (linear mixture under the learned gate) | 2.259 | +0.001 | |
| + `body_err=2` (bodies' error weighted by their mixture responsibility = exact mixture gradient) | **2.220** | +0.001 | |
| + shared types 1024 (route_return) | **2.217** | **+0.042** | C9 -0.002, C6 +0.004 |
| `shared_always=8` (DeepSeekMoE-style always-on shared bodies emitting) | 2.693 | -0.069 | REJECTED: 8 saturated codes dominate the 32-dim workspace |
| linear + body_err=2 + shared_always=8 | 2.484 | -0.076 | rejected |

### 13. Skills under the conformance arms (smoke, every 4th item; BLiMP 3,350 items, entity 560, supplement 63)
| Model | BLiMP actual / silent / constant | S1 | S2 |
|---|---|---|---|
| c_gate (backprop) | 52.6 / 50.1 / 52.2 | +2.5 PASS | +0.4 FAIL |
| c_gatenone (no hidden learning) | 54.3 / 50.0 / 53.9 | +4.3 PASS | +0.4 FAIL |
Bodies matter for grammar (2.5-4 points over the memory alone) through WHICH body fires; their context computation
adds ~0.4 points. Entity tracking ~17% in every arm (not body-driven). Supplement too small to read.

Full epochs launched: `ss_lin` (linear + body_err=2, hops=0) and `ss_linst` (+ shared types 1024).

### 14. Longer workspace memory (hypothesis: bodies cannot see far back; ws_decay=128 spans ~2 bytes)
Smoke, linear + body_err=2, hops=0: ws_decay 128 / 224 / 240 / 250 -> held-out 2.220 / 2.222 / 2.220 / 2.220,
C2 +0.001 / +0.000 / +0.002 / +0.001. No effect: the workspace is a decayed sum of body codes and carries little
long-range structure. Long-range context for bodies needs a real carrier (e.g. word-level units, learned input
embeddings of recent words, or the memory's retrieved continuation as input).

### 15. Skills with the linear blend (smoke, every 4th item; 3 arms)
| Model | BLiMP actual / silent / constant | Entity actual / silent / constant | S2 BLiMP | S2 entity |
|---|---|---|---|---|
| l_lin2 (linear, body_err=2) | 53.0 / 51.2 / 53.4 | 20.2 / 17.9 / 20.0 | -0.4 | +0.2 |
| l_lin2st (+ shared types) | 53.0 / 52.0 / 53.8 | 20.2 / 16.1 / 19.5 | -0.8 | +0.7 |
Entity tracking now depends on bodies (+2.3 / +4.1 over silent); context computation still not (S2 fails).

### 16. Word-embedding input (`in_words=1`, Bengio-style learned embeddings of the previous 3 words) - SHELVED
Broad-body harness: 3.37-3.41 vs 3.36 without. User direction (2026-09-24): "Long range structure has to form from
multiple bodies working together not just 1" -> long range must come from cooperation across time, not wider inputs.

### 17. Float reference (`python/body_float_reference.py`, Windows torch; broad bodies, same shape/inputs, 2 MiB)
| Variant | Held-out 1 / 2 / 3 epochs |
|---|---|
| C++ organism body (INT4, accumulators, backprop), 1 pass | **3.36** |
| float clamp body, Adam | 2.905 / 2.830 / 2.797 |
| float tanh | 2.794 / 2.691 / 2.653 |
| float 4x wider | 3.044 / 2.992 / 2.977 |
| float ONE shared body + byte embedding | 3.231 / 3.163 / 3.163 |
| plain MLP (156k) | 2.819 / 2.723 / 2.696 |
| float clamp, plain SGD batch 16 | 2.823 |
| INT4 grid + float shadow (STE), Adam | 2.949 |
| **INT4 grid + plain SGD** | **2.916** |
Verdict: inside a body the SHAPE is fine and neither INT4 nor SGD costs much; the C++ update machinery (error
quantisation, shifts, lr_anneal 8x on familiar events, accumulator quanta) loses ~0.44 bits. Fixable: next item.

### 18. BUG (mine, fixed): the C10 ablation line split an if/else in make_input, so the workspace input cells were
ALWAYS zeroed in the four smoke runs of that round (tc0/tc8/tc32d224/tc0d224, first pass). Rerun after the fix
reproduces the earlier l_lin2 number exactly (2.2196). No other runs affected (the line was added in that patch).

### 19. Temporal credit through the workspace (`tc_window=N`): writers learn what later readers needed
Each byte the readers' error at the workspace input cells is sent back to the writers of the previous N bytes,
discounted by ws_decay; a writing event takes one backward pass when it leaves the window. New arm "no workspace
memory in" and check C10 (full(no ws read) - full(actual) >= 0.01).
| Smoke (linear, body_err=2) | Held-out | C10 full / body-only |
|---|---|---|
| tc off | 2.220 | +0.005 / +0.02 |
| tc_window=8 | 2.223 | **+0.021 PASS** / +0.15 |
| tc_window=32, ws_decay=224 | 2.220 | **+0.026 PASS** / +0.16 |
| ws_decay=224 only | 2.222 | +0.013 / +0.07 |
First multi-body temporal circuit that passes its check; it forms only when writers get credit from future readers.
Not yet a net gain (held-out flat; C2 ~0: constant bodies do as well, so the carried signal is WHICH bodies fired
recently, not what they computed).

### 20. Where the within-body gap comes from (float bisection, broad bodies, 2 MiB)
| Float emulation | Shuffled | Text order |
|---|---|---|
| float clamp body, SGD | 2.823 | 2.937 |
| INT4 grid + float-scaled activations, SGD | 2.916 | 3.041 |
| EXACT C++ integer format (x +-32, INT4 born -2..2, shifts 1,2,2,3,3, round+clamp +-127), SGD | 3.059 | 3.268 |
| C++ organism itself (online, text order; stoch_round, lr_shift=11) | - | **3.205** |
C++ update rule is fine (beats its own float emulation in text order). ~0.12 = online text order (replay recovers it:
replay 0.5x/1x/2x -> 3.149/3.162/3.049, but that is also extra passes); ~0.23 = the integer body FORMAT. Saturation
controls in C++ all hurt (homeo hidden-only 3.35-3.42; in_scale 8/16/64 3.23-3.28; larger shifts 3.31 / 7.79).
Format bisection in float (shift, init range, rounding, input scale) running.

## USER PROTOCOL (2026-09-24): distributed-input cooperation test - TO RUN
Verbatim intent: distribute the inputs so no single unit sees everything needed to answer. Keep a small set of
viable inter-unit connections available, allow enough recurrent processing for completed results to travel, and
supervise ONLY the final answer. Do not give intermediate units their own answer targets. Then compare the same
setup with the current usefulness-based routing and suppression enabled. Reading:
- learns with available routes but fails with usefulness-based selection -> routing/development prevents the
  cooperating computation from forming;
- fails in both, but deliberately SUPPLIED intermediate messages let the receiver succeed -> the communication
  substrate can support it; learning the intermediate representations / credit assignment is the bottleneck;
- even supplied intermediate messages fail -> investigate message delivery, mixing, timing, receiver capacity first.
Success on a distributed-input task establishes useful cooperation, not a specific hierarchy of XOR subroutines; for
that, INSPECT the messages and INTERVENE on them (weights/activity alone are not enough).
A shared reward is not a guaranteed fix (the original learner had one); published node-perturbation work finds the
standard method data-inefficient and unstable in multilayer networks.
User's leading hypothesis: the boundary between units changes what counts as useful. Inside the XOR unit, hidden
components could contribute to an answer they could not produce themselves; across units, routing, selection or
local output training may require each component to justify itself independently. (Consistent with: routed bodies
hurt when writing their own output (#8); route_gate silences bodies without their own counterfactual gain;
body_err gives every body its own answer target.)

### 21. Distributed-input cooperation protocol (`cpp/apps/coop_test.cpp`, `build-task/coop.sh`, `coop2.sh`)
Real Coco bodies (INT4 born -2..2, shifts 1,2,2,3,3,3, int8 clamp, the organism's accumulator update with stochastic
rounding); explicit wiring: sender L2 -> receiver L3 message half one tick later; own L4 -> own L2/L6 next tick.
Inputs distributed (each input unit sees its bit + 3 random distractor bits); ONLY the root's final answer supervised.
Rules: bp = exact backprop through units and ticks; np = paired node perturbation, shared reward.
Conditions: open / gated (organism route gate: counterfactual gain EMA >= 0.05, init 0.10) / local (every unit has
its own answer target) / oracle (true intermediate messages supplied) / oraclein (only input units' messages supplied).
| Task | Rule | open | GATED | local | oracle |
|---|---|---|---|---|---|
| copy2 (control) | bp / np | 100 / 100 | 100 / 100 (keeps 0->2 = 1.53 bits, cuts 1->2) | 100 / 100 | 100 / 100 |
| **xor2** | bp / np | **100 (4k) / 100 (32k)** | **50 / 50 (both links' usefulness decayed to 0)** | 100 / 100 | 100 / 100 |
| par4 | bp / np | 50 / 50 | 50 / 50 | 50 / 50 | 100 / 100 |
Interventions (xor2 open): swapping either sender's message for its opposite-bit message flips the answer 100%.
=> USER HYPOTHESIS CONFIRMED for usefulness gating: conjunctive cooperation (each message worthless alone) is cut
before it can form. Local answer targets did not block xor2. Node perturbation can organise a 2-body circuit (8x slower).
Delivery controls (bp): relay (bit -> middle -> root) 100%; relay2 (XOR in the middle, relayed) 100% (8k);
par4 oraclein (middles must compute + relay; root combines): default 50; SUMMED messages (24 cells each instead of
12) 100% (4k); 5 ticks 100% (12k); SHARED CELL TYPES (middles + root one body) 100% (24k) -> part of the failure WAS
delivery (partitioned 12-cell slots, exact tick budget).
par4 fully end-to-end: 50% under every variant (shared, summed, 5 ticks, combined; bp and np; 2 seeds; 120k).
CURRICULUM (bits 2,3 held at 0 for the first N steps = learn XOR at depth 1 first):
| par4 variant (bp) | 100% at |
|---|---|
| plain, curriculum 20k | 60k |
| summed + 5 ticks, curriculum 20k | 44k |
| shared types, curriculum 20k | 28k |
| shared + summed + 5 ticks, curriculum 20k | 24k (4k after the curriculum ends) |
| shared + summed + 5 ticks, GATED | never |
| shared + summed + 5 ticks, NODE PERTURBATION (curriculum 40k, 120k) | never |
All successful par4 runs: swapping ANY input unit's message flips the answer 100%.
Hierarchy test (middle unit 4, intervention): message change when bits 0 AND 1 flip (pair parity same) vs bit 0 alone:
shared+summed+t5 15.4 vs 56.6; plain 20.6 vs 45.1; shared-only 25.9 vs 23.5. Answer kept on double flip 100% and
swapping ONLY unit 4's message to the opposite-parity one flips the answer 100% in all three -> a genuine XOR
subroutine at the middle level in the first two; a mixed (distributed) intermediate code, read only for its parity
component, in the shared-only model.
CONCLUSIONS for the organism: (1) usefulness-based gating/suppression of routes must not decide which connections
survive while cooperation is forming (it kills conjunctive computation); (2) message delivery needs full-width
(summed) slots and enough ticks for results to travel; (3) shared cell types + learning simple before complex is
what lets depth compose (fruitfly result reproduced in Coco bodies); (4) node perturbation cannot organise depth-2
circuits here (consistent with the literature the user cited) - a better local rule is a later, separate problem.

## Steps 1-4 (user order, 2026-09-24): ungate links, full-width delivery, shared types + curriculum, local rule

### Step 1 `route_ungated=1`: no usefulness-based selection of links
Removes the three mechanisms that made each link justify itself: assoc_update (a receiver whose own output was not
consequential tightened its lock to REJECT that sender), the route-utility hop gate, and the route-gate emission cut.
### Step 2 `msg_merge=1`: full-width delivery
A receiver routed by several senders in one hop runs ONCE on the SUM of their messages (was: first sender only, the
rest dropped by dedup); its route consequence returns to every sender; the message gradient reaches every sender
(route_return: applied after the forward-order pass, since senders are processed first there).
Smoke (linear blend, body_err=2, in_struct, backprop, stoch_round, route_return):
| Arm | Held-out | C2 | C4 | C6 | C9 | C10 |
|---|---|---|---|---|---|---|
| no routing (hops=0) | 2.218 | | | | | |
| neighbour routing (gated) | 2.219 | +0.010 | -0.010 | -0.010 | -0.004 | +0.009 |
| + ungated | 2.229 | | | | | |
| + ungated + merged | 2.220 | +0.013 | -0.001 | -0.001 | -0.000 | +0.012 P |
| + ungated + merged + bp_route | 2.222 | +0.013 | -0.007 | -0.007 | -0.003 | +0.018 P |
| **shared types 64** | 2.227 | **+0.081 P** | **+0.077 P** | **+0.077 P** | **+0.012 P** | **+0.015 P** |
| shared 64 + ungated + merged + bp_route | 2.235 | +0.084 P | +0.034 P | +0.034 P | -0.005 | +0.017 P |
Shared-type links never went through the usefulness mechanisms (specialists call their shared body directly), so that
configuration was already ungated: first FULL-MODEL circuit pass (C9 +0.012), 9/17 checks. Neighbour routing: ungating
makes it neutral (neighbours are other contexts' specialists: nothing useful to add). Held-out still level with no routing.
### Step 3: shared types + simple-before-complex (`model_babylm --curriculum`: first epoch ordered childes, switchboard,
bnc_spoken, open_subtitles, simple_wiki, gutenberg; shuffled within each). Full epochs running: ss_c3_base (hops=0),
ss_c3_st (shared 64, route_return, ungated), ss_c3_cur, ss_c3_st_cur.
### Step 4: local rule. sonic-manifold-flow's original ALM rule = PC-ALM (src/manifoldflow/alm/learning/pc_alm.py):
predictive-coding / augmented-Lagrangian settling; hidden credit = top-down pressure relaxed over settle steps; its
journal: XOR never solved (constant prediction), activity_lr 0.1 x 8 steps needed ~2,000 iterations. Dendritic-basin
successors solved parity only with a local target at every node (forbidden by the protocol). Coco's forward-gradient
rule (attend.h) is Coco-original (fruitfly handoff idea).
Cooperation harness, distributed inputs, final answer only (par4: shared + summed + 5 ticks + curriculum 20k):
| Rule | xor2 | par4 |
|---|---|---|
| exact backprop | 100% (4k) | 100% (44k) |
| PC settling K=4 | 100% (4k) | - |
| PC settling K=8 | 100% (4k) | **100% (24k, faster than bp)** |
| PC settling K=16/32 | 100% (4k) | 100% (44k, == bp) |
| forward gradients, 1 tangent | fail | fail |
| forward gradients, 8 tangents | 100% (4k) | fail |
| forward gradients after bp warm start | fail | fail |
| node perturbation | 100% (32k) | fail |
PC-K model: a site more than K layer-steps from its error source gets no credit (fixed-prediction PC propagates one layer
per settle step and equals backprop after depth steps; Song et al. 2020). The original PC-ALM failure was under-settling.
Organism: `pc_settle=K` implemented (depth per trace: emitter L6/L5/L2-3/L1 = 1/2/3/4; route-consequence receiver
L4 = min(sender L2, L6)+1; message sender = receiver L3 + 1; workspace writer L6/L5/L2-3/L1 = 5/6/7/8).
Smoke (shared 64 + tc8): pc_settle=8 final state BYTE-IDENTICAL to full backprop => the organism's learning IS PC-ALM
settling with <= 8 steps, each error unit using only its own layer's weights and the error one layer above.
pc_settle 6 / 4: held-out 2.228 / 2.230 vs 2.221 (C2 +0.081 / +0.068; C10 pass at K=4).
(Caveat: symmetric feedback weights (W^T) are still used, as in PC; replacing them with learned/random feedback is a
separate, later question.)
Step 3 full epochs (linear, body_err=2, in_struct, backprop, stoch_round; S = shared 64 + route_return + ungated):
| Run | Held-out | BLiMP | Supplement | Entity |
|---|---|---|---|---|
| ss_c3_base (hops=0) | 2.025 | 52.2 | 51.2 | 19.3 |
| ss_c3_st (S) | 2.041 | 51.6 | 49.6 | 19.4 |
| ss_c3_cur (hops=0, --curriculum) | 2.081 | 52.7 | 52.4 | 17.2 |
| ss_c3_st_cur (S, --curriculum) | 2.098 | **53.4** | 52.0 | 17.7 |
One epoch: skills within ~1 point (BLiMP SE ~0.4). Curriculum ends on books, so held-out (all sources) is worse.
Next: 10-epoch runs (BabyLM budget) of base and st_cur, scored per epoch.

### 22. Long-range agreement probe (`cpp/apps/agree_probe.cpp`, `build-task/agree.sh`) - the skills gap, measured directly
Synthetic language "the NOUN FILLER^d VERB ." (12 noun pairs incl. irregular, 10 verb pairs, 25 random fillers; d = 0..8).
Trained 2 MiB (8 MiB where noted), scored frozen on minimal pairs by distance, 200 items per distance.
% correct verb preferred (d = 0 / 1 / 2 / 4 / 6 / 8):
| Config | actual |
|---|---|
| hops=0 (current best learning setup) | 91.5 / 63.5 / 54.0 / 48.0 / 53.0 / 48.5 |
| + tc_window=16, ws_decay=224 | 85.0 / 58.0 / 51.0 / 47.0 / 49.5 / 53.0 |
| shared types 64 (+/- tc16) | ~91 / 59 / 52.5 / 45-46.5 / 55 / 48-50 |
| old default config | 91.5 / 56.5 / 53.5 / 41.0 / 53.5 / 48.5 |
| tc 32-64, ws_decay 240/248, 8 MiB | all ~chance beyond d=1 |
| `ws_bands=1` (multi-timescale memory: decays 128/224/248/254 per 8 dims, per-band scale), +/- tc64, 8 MiB | ~chance beyond d=1 |
Silent bodies (memory alone) ~= actual everywhere: nothing carries number >= 2 words.
ORACLE WRITER (ORACLE=1: at every noun end, its number is written into ws_mem[31], train and test):
| oracle + bands | 93.5 / 63.0 / 64.5 / 53.5 / 62.5 / 60.0 |
| **oracle + bands + tc64** | 93.5 / **74.5 / 70.5 / 59.5 / 67.0 / 66.5** (constant bodies -> 90 / 53.5 / 53.5 / 45.5 / 51 / 50; no ws read -> chance) |
| oracle, plain decay 128 | ~chance (the write is gone within ~2 bytes) |
=> per the user's protocol: readers CAN use a supplied write across the gap (60-75%), via body computation on the
workspace; WRITERS never learn to write it. Structural reason: the workspace memory is a SUM; one write is diluted by
every later write in its band, so both its effect and its credit are tiny next to the writer's own prediction error.
Proposal (needs the user's OK: extends the design's shared workspace): gated writing (overwrite with a learned gate) for
the slow memory, so one write dominates later reads and gets undiluted credit.

### 23. Carrier options (user: "Try both and see which one gives us the most capabilities") - overnight 2026-09-24/25
Option A = gated registers (`ws_regs=1`: 4 registers; each emitting body's L6 dims 24-27 = values, 28-31 = gates
sigmoid(l6/32 + reg_bias); reg = (1-G) reg + sum g v; read in input cells 124-127; temporal credit follows the gates).
Option B = persistent activity through L4 (`persist=N`: emitters with mean|l4| >= persist_thresh stay active into the
next byte fed their own L4 return; truncated BPTT through each body's own chain).
Agreement probe (2 MiB unless noted), % correct at d = 0/1/2/4/6/8:
- A, gate bias -2/-5/-6/-8, tc16/64, curriculum, straight-through gates, 8 MiB: all ~chance beyond d=1.
- **A with ORACLE register writer** (gates bias -8): 98.5 / 94.5 / 83.5 / 80.5 / 86.0 / 81.5; +tc64: 99 / 97.5 / 85.5 / 83 / 90 /
  86.5 (constant bodies / no read -> chance). Registers work as a carrier when written.
- Guided-then-tapered oracle (oracle on 1 - progress/0.6 of nouns, never at test): chance after withdrawal (all variants).
- Register inspection (separation of register value, plural vs singular, at the verb): trained without oracle ~0.0-0.2 even at
  d=0 -> writers write NOTHING number-dependent. Writer unit test with a PERFECT error at the noun end: ~0 at bias -8.
- BUG FIX: temporal_backprop and persist_bptt applied the familiar-event anneal (lr/8) to every delayed update -> removed;
  `tc_gain` added. Then writers learn to write at d=0 (separation 1.5-2.2) but it is ERASED by d=2: every later body's own
  gate overwrites (per-body gates, starved like everything per-body).
- Option B (persist 4/8, thresh 20/40, curriculum): chance (ran with the lr/8 slip; rerun queued).
- **A2 = ONE SHARED WRITE HEAD** (`ws_regs=2`): reg = (1-g) reg + g v, g = sigmoid(w_g . ws/128 + b), v = 100 tanh(w_v . ws/128);
  exact backprop through the register recurrence over the tc window trains the head (4 x 65 floats) and sends each writer body
  its ws-dim error via temporal credit. Rationale: shared parameters learn, per-body parameters starve. Running.

### 24. Overnight results (2026-09-24 night)
- A2 shared write head in the organism: writes number at the noun (separation 2.7 at bias -1) but overwritten within 2 words;
  tapered oracle with A2: chance after withdrawal. Option B rerun with the lr fix: no usable result yet (outputs lost once).
- 10-epoch BabyLM runs: held-out plateaus ~1.98-2.00 by epoch 2; BLiMP 51-54 and entity 17.5-20 flat through epoch 5.
  More epochs do not move skills.
- Controlled harness `cpp/apps/seq_probe.cpp` (Coco INT4 bodies, one per current byte, registers + shared head, EXACT BPTT
  over the sentence, no memory/routing/blend): chance on the agreement decision even with the noun VISIBLE in a 7-byte window;
  identical decision profiles across runs (constant choice). INT4 bodies at a busy position do not learn this conditional
  (saturation / straight-through / wider shifts: unstable 34-72% at d=0).
- **CALIBRATION (float GRU reference, python/agree_gru_reference.py, full BPTT, Adam, hidden 64): at 50k sentences it ALSO
  fails beyond distance 0 (is/are d0 94-100%, d>=1 chance; verbs chance everywhere).** The agreement probe was too hard for
  its data size; failures of the organism there are NOT evidence of a Coco-specific carrier failure. Scaling the GRU to 400k
  sentences to find the data a standard gated RNN needs; the probe must be recalibrated to that before judging carriers.
- Carrier full epochs on BabyLM running (plentiful real agreement): ss_car_A2 (ws_regs=2 bias -3), ss_car_A2b1, ss_car_B
  (persist=8), ss_car_tc (control), all with tc_window=64; compare BLiMP phenomena vs ss_c3_base / ss10_base e1 (52.15).
- **GRU at 8x data (400k sentences):** is/are 100% at every distance from 160k sentences; verbs 100% at every distance from
  320k (hidden 64) / 160k (hidden 128): an abrupt phase transition. The agreement language needs ~7-15 MB even for a float
  gated RNN; all organism probes so far used 2-8 MB. Organism rerun at 16 MB (all carriers) running (build-task/agree16.sh).
- **Word 5-gram reference on Strict-Small** (python/ngram_blimp_reference.py, interpolated absolute discounting): BLiMP fast
  57.60, entity tracking 19.42 -- ~5 BLiMP points above the organism (52-53) from purely LOCAL word context. Part of the skills
  gap is local word-level context, which byte bodies (~5 bytes) do not capture. Composer levels (word units fed back, the
  earlier best BLiMP 55.6-55.7) rerun with today's learning fixes: ss_lvl1, ss_lvl2feed.
- **KEY FINDING (python/coco_body_seq_reference.py: float replica of seq_probe - Coco-shaped bodies, one per current byte,
  clamp activations, 4 registers + shared write head, full BPTT, Adam; is/are, 400k sentences):**
  | variant | d0 | d1-d8 |
  |---|---|---|
  | float weights + registers | 100 | **100 at 250k sentences** (89-96 late, some drift) |
  | float, no registers | 92.5 | chance |
  | INT4-grid weights (float shadow, STE) + registers | 55 | chance |
  => The body architecture + gated registers + shared head CAN learn long-range agreement (like a GRU). The INT4 WEIGHT GRID
  is what prevents it: coarse steps cannot express/learn the fine conditional features. C++ harness seq_probe (INT4) likewise
  stays constant even at 400k. Next: keep INT4 durable weights but add per-output-row scales (standard INT4 practice), finer
  grid, INT8 reference.
- Organism agreement probe at 16 MB (all carriers, tc64): chance beyond d=1 for none / bands / A1 / A2 (bias -1, -3, curric) / B.
- Quantized replicas (is/are, 400k): INT4 55/chance; INT4 + learned per-row scales 55/chance; finer INT4 grid 100/chance;
  **INT8 100/chance**. Every range-bounded quantized variant fails at d>=1; float succeeds. Hypothesis: a bounded weight range
  (~+-2.5 sigma of init, like INT4's [-8,7] with fixed shifts) cannot let a reader weight 4 register inputs far above the rest.
  Testing register input gain x8 / x32 (RGAIN) with INT4/INT8.
- BabyLM one-epoch carrier runs (same learning setup, hops=0): base 52.15 / 51.2 / 19.27 (BLiMP / supplement / entity);
  tc64 control 51.92 / 50.8 / 17.69; A2 bias -3 52.10 / 50.0 / 18.30; A2 bias -1 51.26 / 50.8 / 18.35 (held-out 2.024-2.029
  vs 2.025): no gain (expected given the INT4 finding). Composer levels with today's fixes: L1 52.22 / 51.6 / 18.18,
  L2 feed 52.46 / 48.0 / 18.48 (held-out 2.037 / 2.054); per phenomenon anaphor agreement 46.5 -> 52.2 / 54.2, others within
  noise. ss_car_B (persist=8) still running (slower).
- Register input gain (RGAIN), is/are 400k, one seed each: **INT4 x32 -> 100% at every distance by 300k** (94-97 later);
  INT4 x8 chance; INT8 x8 chance; float x8 chance (float x1 succeeded). Strict INT4 bodies CAN learn long-range agreement
  through registers; single-seed outcomes are dominated by an abrupt, stochastic phase transition -> 3 seeds each of
  float x1 / INT4 x32 / INT4 x1 / INT8 x32 running. Organism: `reg_scale` (register cell = reg x reg_scale/100; default 50 =
  previous behaviour) + in_scale=4 gives the ~x32 ratio; 16 MB agreement probe running (build-task/agreeG.sh).
- Organism 16 MB probe with high-gain registers (in_scale=4, reg_scale=127): A2 (bias -3/-1), A2 x8, A1: chance beyond d=1.
  The C++ harness (exact BPTT, organism-style quantised SGD) also failed at 400k. Difference to the successful Python INT4
  run = OPTIMIZER (Adam, per-weight normalised steps, vs plain SGD through accumulators). Testing Python INT4 x32 with SGD
  (lr 0.02 / 0.1 / 0.5) and the C++ harness with the same gain (fscale=4, rscale=127).
- **CORRECTION (3 seeds each, is/are 400k):** float s1 100% all d / s2 d0 only / s3 partial; INT4 s1 100% all d / s2 dead /
  s3 d0 only; INT4 x32 s1 success then COLLAPSE / s2 88-93% then collapse / s3 100%; INT8 x32 partial/dead/dead.
  INT4 is NOT the barrier (single-seed conclusions above were artefacts). Common to all: UNSTABLE learning - a stochastic
  phase transition, and collapses to a constant output (55%, even d0 lost) = dead network (hard clamp: zero gradient once
  pushed past +-1). The GRU (tanh) never collapsed. Testing: forward clamp + LEAKY backward slope 0.1 outside (design-
  compatible: kernel unchanged, only learning), and tanh reference; 3 seeds each.
- C++ harness with x32 register gain (fscale=4, rscale=127): chance; its train loss 0.93-1.87 bits vs ~0.43 in Python on
  the same language -> the C++ update path learns much less overall (cf. #20: C++ 3.2 vs float 2.9 on text).
- Python INT4 x32 with plain SGD (lr 0.02 / 0.1 / 0.5): language learned late (loss 0.43-0.55 only by ~350k at lr 0.02;
  unstable at 0.1) and NO agreement. Adam's per-weight step normalisation matters. Organism-compatible version: ROW-WISE RMS
  normalisation (one running gradient scale per neuron, 128 per body; the factored idea behind Adafactor). Testing INT4 +
  RowRMS (body weights) +/- leaky clamp, 3 seeds.
- BabyLM one epoch, option B (persist=8, tc64): held-out 2.362 (vs 2.025), BLiMP 51.64, supplement 48.0, entity 17.48: harmful as is.
- **LEAKY CLAMP GRADIENT (forward hard clamp unchanged; backward slope 0.1 outside +-1), is/are 400k, 3 seeds:**
  float 3/3 learn every distance (s3 drifts to 79-86); INT4 2/3 at 100% every distance (s1 d0 only by 400k); tanh reference
  3/3. vs hard clamp: float 1/3, INT4 1/3, with collapses. The dead-clamp instability was the main obstacle. Porting:
  `leak` in the C++ harness (running), then organism bp paths (bp_leak).
- Python INT4 + ROW-RMS (per-neuron step normalisation) for body weights: no leak 0/3 seeds, leak 0.1 1/3 learn every
  distance (Adam + leak: 2/3). Weaker than full Adam, not hopeless.
- C++ harness + leak 0.1 (any seed/lr/gain): train loss stalls at 0.776 bits (Python INT4: 0.42; Python INT4 plain SGD
  lr 0.02: 0.43-0.55) -> the C++ quantised update path loses more than the optimiser choice. The accumulator is meant to be a
  shadow weight (design doc #8), so INT4 + accumulator should behave like INT4 + float shadow. Test: C++ harness opt=1 = exact
  float shadow per synapse + Adam, INT4 weight = round(shadow) (running).
- Organism text smoke with bp_leak: 0 -> 2.217, 0.1 -> 2.230, 0.3 -> 2.206 (within noise). Organism 16 MB agreement probe
  with bp_leak running (build-task/agreeL.sh).
- Organism 16 MB probe with bp_leak 0.1 (A1, A2, A2 x32, none) and A2 without leak: chance beyond d=1; A2 registers pinned at +-100 (head gates saturated); for L_none/L_A1/L_A2 'constant bodies' == 'no ws read' == actual EXACTLY -> in the organism the sequence memory decides this task; bodies' contribution is not measurable there. Debug the learning rule in the clean C++ harness first.
- C++ harness opt=1 (exact float shadow per synapse + Adam, INT4 = round(shadow), leak 0.1; alr 0.02 / 0.01 / 0.05, 2 seeds):
  train loss 0.83-1.41 bits, decision CONSTANT at every distance incl. d0 (identical profile d0 50.5 d1 49 d2 48 d4 55 ...).
  The optimiser alone is not the discrepancy. NOTE: the Python reference prints ONE batch's loss with '\n' padding
  (padded positions are nearly free), so its "0.42 bits" is deflated; the loss comparison C++ 0.776 vs Python 0.42 was
  not like for like. The decisive difference is the decision: C++ is constant even at d0, where the plural 's' is inside
  the 3-byte window. Diagnosing per layer (DIAG=1: saturation + singular/plural separation at the space body).
- 10-epoch BabyLM runs finished. BLiMP per epoch (fast items): ss10_base 52.2 52.5 53.3 53.6 50.9 53.1 53.4 51.6 51.3 50.7;
  ss10_st_cur (shared types + curriculum) 53.4 52.5 53.2 52.2 53.5 53.7 51.8 52.1 53.0 53.3; held-out 1.98-2.01 / 2.03-2.10.
  Flat: more epochs do not move skills (confirmed to 10).
- **Harness diagnosis (DIAG=1, space body after the noun, d0; seq_probe):**
  - organism-style accumulators (lr_shift 11): the body's weights barely move in 100k sentences (weight histogram still
    the init {-2..2}; ~90 of 5.5k weights reached +-3) -> the bodies stay random features; the float readout does all the
    learning (loss 0.78 = random-feature readout). Number separation (mean |sg - pl| per cell) FALLS with training:
    l1 16 -> 6, l6 5 -> 1, while l1 saturation rises 70% -> 86%.
  - when weights do move (lr_shift 6/8, per-token Adam, minibatch-32 Adam): L1 saturates 93-98%, L3/L5 87-98%, and the
    number separation at L6 is 0-6: the information dies in saturation.
  - even a float LINEAR readout from the raw input features (direct=1) does not learn d0 (39-62%) at out_lr 0.01 per token
    -> the per-token SGD readout is too noisy for a rare conditional (the verb follows the noun in 1/9 of cases); Python
    averages 32 sentences per Adam step. The organism's readout is the same kind (lr_out 0.005 per token, SGD).
  - the C++ weight grid is 1.6-3x COARSER relative to the clamp than the Python INT4 replica (one weight step = 0.125-0.25
    of the clamp range per unit input vs 0.043-0.078): weights at +-7 overdrive every layer.
  Testing: readout/head lr /10-/30; shifts +1/+2 (sh1=2-3, shh=3-4) with minibatch Adam; Python replica at batch 1/4/32.
- **ROOT-CAUSE CANDIDATE, in the real organism (model_body_probe, new WEIGHTS_ONLY / WEIGHTS_CMP):** after 10 BabyLM epochs
  (ss10_base) the hidden layers are still at the BIRTH distribution: L1/L2/L3/L5 have 98.3-99.4% of weights in {-2..2}
  with the flat uniform birth histogram; per epoch only ~1% of hidden weights change (e9 -> e10: L1 1.06%, L2 1.17%,
  L3 0.95%, L5 1.77%; e1 -> e2 similar); L6 moves 20-31% per epoch (and 19-26% sit beyond +-2). The bodies' hidden layers
  are effectively FROZEN RANDOM FEATURES; all learning happens in L6, the float readout and the sequence memory. This is
  the common cause behind "the bodies do nothing", "layers don't do what they're supposed to" and circuits that never
  beat one body: nothing below L6 learns.
- Harness: when the hidden weights are made to move (faster accumulators / Adam), they saturate in an INPUT-INDEPENDENT way
  (l5/l6 90-100% saturated with separation 0), unlike Python float (l5 89% / l6 77% saturated but separation 17-22: a
  binary code that still varies with the input). Also found a harness bug: my C++ Adam had no bias correction (first
  hundreds of steps 3-6x too large) -> weights slammed to +-8 within 80k sentences. Fixed; faithful-mirror runs (body
  minibatch Adam + readout/head minibatch Adam `ropt=1`) running.
- Faithful C++ mirror (bias-corrected minibatch-32 Adam for bodies, alr 0.03 / 0.01, + Adam readout/head `ropt=1`; seeds
  1-2; shifts default / +1): ALL constant. alr 0.03: weights slam to -8/+7 (~1000 each of 5.6k) within 80k sentences,
  every layer 100% saturated with zero separation, train loss RISES to 1.4-2.0 bits. No leak: L6 100% saturated, no
  gradient, frozen. alr 0.01 / SGD readout: 85-99% saturated, separation ~0.
- **Bisection: python variant `cpp` = the C++ harness's exact numeric structure (+-32 inputs, integer weights init
  {-2..2} with STE and [-8,7], >> shifts 1,2,2,3,3, clamp +-127, readout of l6/128) with AUTOGRAD gradients + Adam 0.03:
  the same runaway (l2/l5/l6 100% saturated, separation 0, loss 1.3-1.6) within 16k sentences.** Not a C++ gradient bug:
  the numeric structure (grid coarseness / init / step size vs the activation range) is what fails, where the Python
  INT4 replica (init sd 3.2 grid steps, grid 2.9x finer relative to the clamp) learns. Sweeping in Python (fast).
- Harness sanity check (no registers, linear readout on the raw features + frozen random L6, Adam or SGD): after "the girl "
  p('a') stays 0.1-0.17 although no filler starts with 'a' (true p = 0): the per-byte bodies MUST be trained to separate
  this; a shared readout over random codes cannot. Direct register readout (`rdirect=1`) with frozen/slow bodies: constant.
- **Python `cpp` structure, Adam step per batch (grid units), seed 1, d0 at 80k:** 0.03 runaway; 0.01 55 (saturated);
  **0.003 85; 0.001 86; Python-scaled (x 11, shifts 1,4,4,4,4, randn init) 0.04 85.5**; 0.01 + shifts+1 77.5. So the C++
  numeric structure CAN learn with a smaller step: my C++ Adam runs (0.01-0.03) were simply too fast. The organism's
  accumulator SGD is on the other side (far too slow: 1% of hidden weights per epoch). Confirming in C++ at 0.001-0.003.
- **BUG FOUND (2026-09-25): seq_probe's training sentences had NO agreement.** `s = "\n" + sentence(.., num, verb) + " " +
  ISARE[0][num] + ...`: the operands of a chained + are unsequenced and GCC evaluated `ISARE[0][num]` / `VERB[verb][num]`
  BEFORE sentence() set num -> the verb agreed with the PREVIOUS sentence's noun. Every C++ seq_probe result in #23-24 and
  today (constant decisions even at d0, d0 below chance at times, "the C++ update path stalls", the INT4/register/optimiser
  conclusions drawn from the harness) is INVALID. The verbs-task decision eval had the same bug for the verb stem.
  agree_probe (the organism probe) builds its training text in separate statements: correct; only its 64 KB calibration
  text had the bug (fixed). The Python references were never affected. Rerunning the harness on correct data.
- **CORRECT-DATA HARNESS (seq_probe, is/are, 400k, strict INT4 bodies one per byte, 4 registers + shared float write head,
  exact BPTT, leak 0.1):**
  | learning rule (bodies) | readout | d0 | d1-d8 | train loss |
  |---|---|---|---|---|
  | organism accumulators lr_shift 11 | SGD | 82-87 | chance | 0.774 |
  | organism accumulators lr_shift 9 (= organism rate) | SGD | 55-82 | chance | 0.781 |
  | accumulators lr_shift 9, no leak | SGD | 50 | chance | 0.784 |
  | accumulators lr_shift 7 | SGD | 50 (dead, 90%+ saturated) | chance | 0.806 |
  | accumulators lr_shift 9 + direct register readout | SGD | 50-92 | chance | 0.781 |
  | accumulators lr_shift 9, no registers | SGD | 50-80 | chance | 0.861 |
  | shadow + minibatch-32 Adam 0.001 | Adam | 94.5 | chance (slower) | 0.764 |
  | **shadow + minibatch-32 Adam 0.003** | Adam | **100** | **100 at every distance from 320k** (240k: 90.5-100) | **0.737** |
  => INT4 bodies + registers CAN learn long-range agreement in C++; the organism's accumulator SGD cannot (at any rate).
  The learning rule is the bottleneck. Number separation under Adam grows through every layer (L6 sep 46 vs 1-15).
  Next: organism-compatible forms (per-body minibatch every K uses, per-token Adam, SGD readout) + seeds.
- Harness 400k, other seeds / organism-compatible forms (all leak 0.1, INT4, registers): minibatch Adam 0.003 seed 2 d0 87.5
  / seed 3 d0 97.5 + d1 63.5 (transition starting) - the phase transition is stochastic in time as in Python; per-BODY
  minibatch (Adam step every 32 / 128 uses of that body) d0 87-93; per-token Adam 1e-4 / 3e-4 d0 71 / 87.5; SGD readout
  (organism-like) + Adam bodies d0 93; no registers d0 76 (no long range possible). 1.2M-sentence runs launched.
- **Organism port: `adam=1` (adam_lr 0.003, adam_k 32)** - every backprop path (learn_backprop incl. L6 of emitting bodies,
  temporal_backprop, persist_bptt, bp_route) adds its exact float gradient to a per-synapse buffer; after every adam_k uses a
  body takes one bias-corrected Adam step on a float shadow; the INT4 weight = round(shadow); external writes (birth,
  imprint, copies) are detected and resync the shadow. Replaces the quantised L6 rule (apply_l6_error) in this mode.
  Training-time state +16 B/synapse (~1.8 GB at 20k units). Also `reg_readout=1`: the byte readout reads the registers
  directly (logits += Wr.reg/100), its error joins the register credit (tc_greg). agree_probe gained ISARE=1.
  Organism is/are probe (12 MB) running: base / adam / adam+A2 / adam+A2+reg_readout / base+A2.
- **Harness 1.2M sentences (correct data, INT4, registers, leak 0.1), decision at every distance:**
  | rule | readout | seed | result |
  |---|---|---|---|
  | global minibatch-32 Adam 0.003 | Adam | 1 | 100% from 320k |
  | same | Adam | 2 | d0 only by 1.2M |
  | same | Adam | 3 | 100% from 480k |
  | **per-body minibatch (step every 32 uses of that body)** | Adam | 1 | **100% from 960k** |
  | **same** | Adam | 2 | **93.5-98.5% at 480k, 100% at 720-960k** |
  | global minibatch Adam | SGD 0.005 (organism-like) | 1 | d0 only (unstable) |
  | per-body minibatch Adam | SGD 0.005 | 1 | d0 100, no long range |
  => the organism-compatible per-body form works (2/2); the READOUT must also be Adam (SGD readout: 0/2).
- Organism: added `ro_adam=1` (wout, register readout wreg and the ws_regs=2 write head by minibatch Adam, a step every
  ro_adam_k=512 events, lr 0.002). First organism probe (adam bodies, SGD readout, A2): register separation at the verb
  ~0 beyond d0 - consistent with the harness. Launched: organism is/are probe with adam + ro_adam (+A2, +A2+reg_readout,
  ro_adam alone); BabyLM ss_ad (adam bodies only), ss_adr (adam + ro_adam), ss_adr_A2ro (+ registers + reg_readout).
- BabyLM one epoch, the OLD accumulator rule with faster hidden learning (held-out / BLiMP / supplement / entity):
  base 2.025 / 52.15 / 51.2 / 19.27; hidden_shift -2: 2.016 / 52.56 / 51.2 / 17.73; -3: 2.063 / 51.65 / 48.0 / 18.63;
  -3 + homeo=2: 2.044 / 52.36 / 46.0 / 18.17. Speeding up the SGD accumulators does not buy skills (entity regresses) -
  consistent with the harness (accumulators never learn agreement at any rate). The Adam runs are the real test.
- User (2026-09-25): "if adam on everything works then do a 10 epoch babyLM run with full adam as a baseline". Launched
  ss10_adr = ss10_base config + bp_leak=0.1 adam=1 ro_adam=1 (bodies + readout by Adam, no registers), 10 epochs; its
  epoch 1 replaces the one-epoch ss_adr (same config/seed). Stop it if epoch 1 shows full Adam does not work.
- **Organism is/are probe (12 MB, correct data):**
  | arm | d0 | d1 | d2 | d4 | d6 | d8 | constant bodies == actual? |
  |---|---|---|---|---|---|---|---|
  | old rule (A_base) | 100 | 76 | 52 | 58 | 52 | 45 | EXACTLY equal (bodies' context computation contributes nothing) |
  | old rule + A2 | 100 | 73 | 53 | 57.5 | 52 | 47.5 | exactly equal |
  | Adam bodies, SGD readout | 100 | 65 | 48.5 | 52 | 53.5 | 50.5 | differs (bodies now compute), silent bodies d0 61 |
  | Adam bodies + A2, SGD readout | 100 | 67.5 | 61.5 | 55.5 | 53.5 | 52 | differs |
  | Adam bodies + A2 + reg_readout, SGD readout | 100 | 69.5 | 53 | 51 | 53 | 47 | differs |
  **Adam bodies + ADAM READOUT + A2 + reg_readout (B_adam_ro_A2ro): register r3 at the verb = the noun's number,
  sg -99.5 / pl +99.5, separation 3956 (d0), 72.8 (d2), 74.1 (d6); r2 separation -5 / -7.** The first organism carrier
  that holds number across distance. Accuracy arms pending.
- **LONG RANGE SOLVED IN THE ORGANISM (is/are probe, 12 MB, % correct verb preferred, d0 d1 d2 d4 d6 d8):**
  - **adam=1 ro_adam=1 ws_regs=2 reg_bias=-3 reg_readout=1 tc_window=64 bp_leak=0.1: 100 100 98 100 98.5 98.5**
  - adam + ro_adam, no registers: 100 66 55 51.5 52.5 48.5
  - adam + ro_adam + registers read only through body inputs: 100 63 51.5 51 48.5 53.5
  - ro_adam only (old accumulator bodies): 100 66.5 52 53.5 47.5 45.5
  - all SGD-readout arms (old rule / Adam bodies, with or without registers): chance beyond d1
  All four are needed: Adam bodies (the writers' L6 must encode number), Adam readout + head, the registers, and the
  direct register read. Ablation arms (constant / silent bodies, no ws read) pending.
- BabyLM now running: ss_ad (adam bodies only, 1 epoch), ss10_adr (full Adam baseline, 10 epochs, per user),
  ss10_adr_A2ro (full Adam + registers + reg_readout = the recipe that solved the organism probe, 10 epochs; replaced the
  one-epoch ss_adr_A2ro at 17 min, same config/seed).
- **Ablations of the solved organism probe (B_adam_ro_A2ro), d0 d1 d2 d4 d6 d8:** actual 100 100 98 100 98.5 98.5;
  CONSTANT BODIES 100 85 53.5 59.5 57.5 51 (long range collapses: the writer bodies' context-dependent L6, pooled into the
  workspace and written by the shared head, is what puts number into the register); NO WORKSPACE READ 96.5 83.5 66 69
  66.5 59 (the bodies at the verb also READ the carrier through their inputs, not only the direct readout); silent
  bodies 84.5 34 44 50.5 50.5 54. => long-range structure formed by MULTIPLE bodies: writers at the noun + readers at the
  verb, joined by a shared register. Contrast: ro_only (old body rule) constant bodies == actual EXACTLY.
- **BabyLM epoch 1, ss_ad (Adam BODIES only, SGD readout - the combination that failed in harness + probe):** held-out
  2.198 (base 2.025), BLiMP 53.02 (+0.9), supplement 47.6 (-3.6), entity 16.98 (-2.3). Per phenomenon vs ss_c3_base:
  npi_licensing +7.2, anaphor_agreement +3.5, ellipsis +2.3, island +2.3, filler_gap +2.0, control_raising +1.8;
  s-selection -5.0, argument_structure -2.6, subject_verb_agreement -1.4, irregular_forms -1.2. Mixed; not a result to
  keep. (scripts: build-task/phen.sh BASE EPOCH RUN EPOCH.)
- **BabyLM epoch 1, ss10_adr (FULL Adam: bodies + readout, no registers) vs old baseline:** held-out 2.187 (2.025), BLiMP
  53.27 (52.15), supplement 46.8 (51.2). Per phenomenon: anaphor_agreement +7.7, npi +5.0, control_raising +3.6,
  island +3.1, filler_gap +3.1, ellipsis +3.0, binding +1.1; s-selection -4.7, argument_structure -3.0, quantifiers
  -2.2, det_noun_agreement -1.4. Gains on structural phenomena, regressions on lexical/local ones + held-out.
  Hypothesis: with a per-body batch of 32 uses a NEW body learns nothing before its 32nd use (the old rule learned on
  every use; the market / episodic births / exact bodies rely on it). Added `adam_warm=1`: steps after 1, 2, 4, 8, 16
  uses, then every adam_k.
- **Robustness of the organism long-range result (same recipe):** is/are seed 2 (12 MB): 100 84.5 56.5 49 47 53 (no
  long range); verbs 16 MB: 88 51.5 48 50 52 51.5. => 1 of 2 seeds at 12 MB (~460k sentences); the harness transition
  came at 320k-960k sentences. Not yet robust; rerunning at 32 MB (seeds 2, 3; +/- adam_warm).
- **User plan (2026-09-25): get it working on full AdamW first, then move to Muon, then worry about local learning.**
  Backprop's resource cost is the concern: 1.6M bodies x 5632 synapses (~9B) x 16 B of Adam state ~ 144 GB vs ~25 GB RAM.
  (Idea for later: optimiser state only for recently active bodies - a fixed pool of state slots claimed on use, evicted
  when idle - so state memory scales with ACTIVE bodies; Muon needs one momentum buffer.)
- Implemented full AdamW: `adam_wd` (decoupled decay on body shadows), `ro_adam_wd` (readout, register readout, head
  weights); the sequence-memory gate now also steps with the readout Adam when ro_adam=1 (no decay on biases/gate).
  Launched ss_adamw (1 epoch, adam_wd=0.1 ro_adam_wd=0.1) next to ss_adrw (adam_warm, no decay) to see which removes the
  lexical regressions before the definitive 10-epoch AdamW baseline.
- **ON THE LIST (user, 2026-09-25): circuits must be able to increase in complexity WITHOUT BOUND by recruiting and
  layering bodies.** Today's working circuit is fixed-depth (writer bodies -> shared register -> reader bodies). To do
  once AdamW bodies learn reliably: revisit recruitment (market births / routing) and layering (composer levels,
  route_return) on bodies that now actually learn, and test that capability grows with recruited depth: tasks whose
  difficulty needs more stages (parity-N ladder as in the XOR mechanism, nested / multiple agreement, counting), with a
  check that accuracy tracks the number of recruited / layered bodies and that removing a layer of the circuit costs it.
- **BabyLM epoch 1, ss10_adr_A2ro (full Adam + A2 registers + reg_readout):** held-out 2.123 (full Adam 2.187, base
  2.025), BLiMP 53.01, supplement 48.4 (full Adam 46.8, base 51.2); ss10_adr entity 17.47 (base 19.27). Per phenomenon vs
  base: npi +7.3, anaphor +3.7, control_raising +2.8, filler_gap +2.2, island +1.1; quantifiers -3.8, s-selection -2.4,
  det_noun -1.5, irregular -1.2, argument_structure -1.0, SUBJECT_VERB_AGREEMENT -1.0. Registers help held-out and the
  supplement but the probe's long-range agreement has not reached BabyLM at epoch 1 (the probe needed ~460k clean
  agreement sentences; BabyLM's signal is far sparser). Watch epochs 2+; ss_adrw / ss_adamw decide the lexical fix.
- GPU (user asked 2026-09-25): RTX 5080 16 GB + CUDA 12.8 nvcc in WSL, but no GPU path exists (organism is CPU C++;
  Windows torch is the CPU build). Epoch times (54 MB): old rule 1352 s, full Adam 3147 s (2.3x). A single-stream GPU
  port would likely be launch-bound (tiny work per byte); GPU pays off with many parallel streams. User: not now.
- **Organism long-range robustness at 32 MB (same recipe, is/are, d0 d1 d2 d4 d6 d8):** seed 2: 100 84 49.5 50 50 47.5;
  seed 2 + adam_warm: 100 76.5 44 54 52.5 53.5; seed 3: 100 67.5 49.5 50 51 53.5 (constant bodies == actual in all).
  => the organism solved it in 1 of 4 seeds (seed 1 at 12 MB); more data does not fix the others. Unlike the harness
  (2/2 per-body-batch seeds), the organism usually does NOT find the circuit. Next: find what differs from the harness
  (sequence memory absorbing the gradient? readout step cadence? head init / reg_bias? tc_window truncation?).
- GPU port handed to a second session: docs/GPU_PORT_HANDOFF.md.
- BabyLM epoch 1, ss_adrw (full Adam + adam_warm): held-out 2.202, BLiMP 52.31, supplement 47.2 (full Adam: 2.187 /
  53.27 / 46.8). vs full Adam: anaphor +4.0, irregular +3.0; island -3.5, npi -2.6, quantifiers -2.4; s-selection and
  argument_structure NOT recovered (-0.7 / -0.1). Warm start rejected: the lexical regressions are not "new bodies learn late".
- BabyLM epoch 1, ss_adamw (full AdamW, adam_wd=0.1 ro_adam_wd=0.1): held-out 2.942 (!), BLiMP 52.61, supplement 48.4.
  Phenomena swing wildly (quantifiers +10.0, ellipsis +7.8, binding +2.7; irregular -4.2, filler_gap -3.5, det_noun -2.2)
  on a model whose text prediction is broken: decay 0.1 is far too strong (readout / shadows pulled to zero). Rejected.
  Next suspects for the lexical regressions of full Adam: readout step cadence (old SGD readout: ~54M per-event updates
  per epoch; ro_adam_k=512 gives ~105k steps) -> test ro_adam_k=32; gentle decay 0.01 on bodies only.
- Launched ss_adr_k32 (full Adam, ro_adam_k=32). Confound: binaries built after the AdamW commit also put the
  sequence-memory gate on the readout Adam (ss10_adr / ss10_adr_A2ro used per-event SGD for the gate); separate if needed.
- **Organism robustness diagnosis (is/are 12 MB, seed 2 which failed at 12 and 32 MB; d0 d1 d2 d4 d6 d8):**
  seqmem=0: 100 62 49 52 46 49.5 (seed 3: 100 72 44 48.5 55.5 52); body_err=1: 100 63 55.5 49 46.5 45;
  reg_bias=-2: 100 62 51 47 50.5 49.5; **ro_adam_k=64 (readout Adam step every 64 events instead of 512): 100 100 99.5
  100 99.5 99.5.** The readout step cadence decides whether the organism finds the circuit (the sequence memory is not
  the obstacle). Consistent with the BabyLM suspicion (lexical regressions from a rarely stepped readout). Confirming:
  k=64 seeds 1/3/4, k=32 seed 3.
- ro_adam_k=64 seed 2 ablations: actual 100 100 99.5 100 99.5 99.5; constant bodies 80 75.5 66 66 72 67; no workspace read
  100 95.5 97.5 98 97.5 97; silent 49.5 46 51.5 46 48 52.5. Register r1 = number at the verb (sg +99.9 / pl -99.8,
  separation -430 at d6). Writers' context computation carries it; here the direct register readout does most reading.
- **CORRECTION - readout cadence is NOT the fix (single-seed conclusion again).** is/are 12 MB, same recipe:
  ro_adam_k=512: seed 1 OK, seeds 2 / 3 fail (also at 32 MB); ro_adam_k=64: seeds 2 / 4 OK (100 100 99.5 100 99.5 99.5),
  seeds 1 / 3 fail (100 69 52.5 53 56 46.5 / 100 74.5 50 54 55 47.5); ro_adam_k=32 seed 3 fails. ~40% of seeds find the
  circuit regardless of cadence: an all-or-nothing, seed-dependent transition (the probe's training text is fixed; the
  seed only changes the model). Likely a chicken-and-egg start (writers get gradient only once the readout uses the
  register, which it does only once the register carries number). Testing a DISTANCE CURRICULUM (CURRIC_D: max distance
  grows 0 -> 8 over the first 80% of training; data ordering only, final-answer supervision unchanged) on failing seeds.
- Seed 4 (ro_adam_k=64) ablations: actual 100 100 99.5 100 99.5 99.5; constant bodies 100 59 56.5 57 55.5 57 (long range
  gone without the writers' context computation); no workspace read 100 92 87.5 86 82.5 85.5 (the reader bodies carry
  part of it); silent 49.5 46 50.5 47 47.5 51. Every successful seed shows the same multi-body circuit.
- Distance curriculum (CURRIC_D) on seeds 1, 3, 5 (k=64) and 2 (k=512): 0 of 4 find the circuit (d1 73.5-85, d2+ chance).
  Rejected. Tally without curriculum: 3 of 7 seed/cadence combinations.
- Hypothesis from the harness difference: the organism's write head reads the clamped SUM of all emitting bodies' L6
  (ws/128, up to +-4) where the harness head reads ONE body's L6 (+-1): a random head saturates (registers pinned at
  +-100 seen in failed runs), so success depends on the init. Added `head_norm=1` (head reads the mean emitting L6;
  writer gradient scaled to match). Testing on the failing seeds 1, 2, 3, 5.
- BabyLM epoch 1, ss_adr_k32 (full Adam, readout step every 32 events): held-out 2.192, BLiMP 52.68, supplement 47.2;
  s-selection -7.2, quantifiers -5.8, argument_structure -2.7 (vs base); ellipsis +5.0, anaphor +4.7, irregular +4.0,
  control +3.7, filler_gap +3.3. Readout cadence does not fix the lexical regressions either. Common factor of every run
  with the regressions: Adam replaced the ORIGINAL L6 rule (fast per-event quantised steps; 20-30% of L6 weights change
  per epoch) - plausibly a fast lexical co-occurrence memory (s-selection, argument structure). Added `adam_l6=0`: Adam
  for the hidden layers only, L6 keeps the original rule. Launched ss_adh (adam=1 adam_l6=0 ro_adam=1).
- head_norm=1 on seeds 1, 2, 3, 5: 0 of 4 (seed 1, which succeeds without it, fails with it). Rejected.
- **Failure anatomy (register content at the verb):** in every failed seed NO register separates singular/plural even at
  d0 (|sep| < 0.7), several pinned at +-100 - the writer side never starts; in successes one register becomes a clean
  +-99 number switch (sep 27-3956 at d0-d6). Credit to the head/writers comes only through the register readout wreg,
  which starts at ZERO (no gradient until chance correlations build up); the harness successes read registers through
  randomly initialised INT4 body weights instead. Added `reg_ro_init` (uniform +-init for wreg). Testing 0.1 on seeds 1, 2, 3, 5.
- reg_ro_init=0.1 on seeds 1, 2, 3, 5: 0 of 4 (seed 1 again fails under a change). Tally: ~3 successes in ~20
  seed/config combinations; seed 1's baseline success looks like a lucky trajectory. Stepping back to the largest
  structural difference from the (reliable) harness: FRAGMENTATION - the harness has one body per byte value (every body
  sees ~1/30 of events); the organism splits contexts over 20k bodies via the lock tree, so the writers that must encode
  number each see a sliver. Testing units=1000 / 4000 on seeds 2, 3, 5.
- User (2026-09-25): "don't let perfect be the enemy of good - we don't need all seeds to succeed, just 1". Robustness
  hunting stopped (J probes left to finish as information).
- **Did the circuit form in BabyLM? (model_body_probe REGS_PROBE=1: the agree_probe is/are language fed to a frozen
  checkpoint)** ss10_base e1 / ss10_adr e1: d0 98 / 98, d1 63 / 62, d2+ chance (no registers). ss10_adr_A2ro e1 and e2:
  d0 97.7 / 96, d1 61, d2+ chance; registers: no sg/pl separation (|sep| < 0.6) and TWO OF FOUR DEAD - r1 pinned at +100,
  r3 at -100 (the failed probe seeds showed the same pinning). Not formed by epoch 2.
- Common failure signature = pinned registers (v = 100 tanh(zv) saturated -> zero gradient forever). Adding `reg_revive`:
  a register pinned (|reg| > 99) for reg_revive consecutive events gets fresh random head weights, zeroed Adam moments and
  value 0 - a local rebirth (no labels), like the market's.
- Launched: organism probes seeds 2, 3, 5 with reg_revive=20000; BabyLM ss_rv (register recipe + reg_revive=50000, 1
  epoch; its epoch-1 checkpoint gets the REGS_PROBE check). build-task/regprobe.sh measures any checkpoint.
- Fewer bodies (units=1000 / 4000) on seeds 2, 3, 5: 0 of 6 (d1 66-72.5, d2+ chance). Fragmentation is not the obstacle.
- reg_revive=20000 probes (seeds 2, 3, 5): seed 3 100 82 73 73 68 61 (partial long range), seeds 2 / 5 fail - BUT 0
  revivals fired in all three (probe registers hover at ~+-97, under the 99 bar), so seed 3's partial result is plain
  seed variance (plain seed 3 at 12 MB had never been run). In BabyLM the dead registers sit at exactly +-100, so ss_rv
  will exercise the rule.
- BabyLM ss_rv: revival fires (5 rebirths by event 14M) but register 1 re-pins (reborn at 8.6M and again 14.1M): the rule
  treats a symptom. Cause hypothesis: the readout has NO BIAS term (and bodies have none), so a register pinned at a
  constant +-100 is a free bias - an attractor for gradient descent. Added `reg_center=T`: every reader (readout and body
  inputs) sees reg - running mean (time constant T events); a pinned register reads 0 and loses its bias value.
  Testing: probes seeds 1, 2, 3, 5 with reg_center=2000; BabyLM ss_rc (reg_center=5000, no revival).
- BabyLM epoch 1, ss_adh (Adam hidden layers only, ORIGINAL L6 rule, Adam readout): held-out **1.986** (best of any run;
  base 2.025, full Adam 2.187) but BLiMP 50.67, supplement 49.2 (npi -6.9, ellipsis -6.4, quantifiers -5.9, s-sel -3.0;
  argument_structure +2.2, irregular +1.8, binding +1.5).
- **NOISE CALIBRATION of the fast eval sets (binomial, unpaired, 95% CI of a two-run difference):** BLiMP overall (13,400
  items) +-1.2; per phenomenon +-3.5 (1,600 items) to +-6.9 (400 items: anaphor, s-selection, ellipsis, irregular);
  supplement fast has only ~250 items (+-~9). => most per-phenomenon "gains/regressions" in this log are within noise;
  only effects repeated across runs mean anything (s-selection down, npi up in most full-Adam runs). Overall BLiMP: full
  Adam +1.1 is at the edge; ss_adh -1.5 just outside. Scoring the FULL sets (BLiMP 59,875 pairs, supplement ~5,200,
  entity 3x) on the key epoch-1 checkpoints for real resolution.
- reg_center=2000 probes (seeds 2, 3, 5): 0 of 3 on accuracy, BUT the failure mode changed: WRITING now works - at d0 one
  register separates sg/pl in every seed (sep -2.3 / 3.2 / -2.4 & 2.1; before centering |sep| < 0.7) - and the number is
  LOST by d2 (sep ~0): retention fails. With reg_bias=-3 the gate writes ~5% per byte, so after ~48 filler bytes ~10% of
  the noun's value remains and fillers overwrite it; successful seeds learned to shut the gate. Testing reg_bias -5 / -7
  with centering (registers hold by default; the gate must learn to OPEN at the noun).
- **User (2026-09-25): stop the whack-a-mole - do B: bisect from the reliable harness, adding the organism's differences one
  at a time, to find what breaks circuit formation and fix it properly, then move on.** Also still open: "reliable free
  generation with no skill regressions" (the next big thing after this).
- **Bisection B setup.** The organism runs ONE body per byte (0.99 executed/byte on ss10_adr_A2ro e1; median 781 lifetime
  hits per body), so a "summed workspace" is not a difference (head_norm was a no-op). Real differences harness ->
  organism, now options in seq_probe: `ctx=h nb=N` (body chosen by a hash of the last h bytes, N bodies - the organism picks
  by a context key over a ~20k tree), `inorg=1` (organism input: 5 bytes of in_struct features + word/sentence position
  thermometers + the leaky persistent workspace (decay 0.5) at 96..123, registers at 124..127), `stream=1` (registers,
  workspace and byte history carry across sentences; the harness reset them per sentence). Remaining (later): delayed /
  truncated credit (tc_window), sequence-memory blend (organism noseq already failed). Base = per-body minibatch Adam 0.003,
  Adam readout, leak 0.1 (2/2 seeds reliable). Matrix at 1.2M sentences: base s3; ctx=3 s1/s2; ctx=5 s1; inorg s1/s2;
  stream s1/s2.
- Gate bias with centering (reg_center=2000): -5 seed 5: 100 89 90 83 75.5 77.5 (partial); -5 seed 2: 100 63 56 52 51 49.5;
  -5 seed 3: 67.5 62.5 ... (d0 broken); -7 seed 2: 100 74 54 59 48.5 50.5. 1 partial of 4 - not a fix. Organism-side
  patching stopped; the harness bisection (B) decides.
- Bisection interim (480k-720k): inorg s1 100% every distance from 480k, inorg s2 transitioning at 720k (96 80.5 81.5 64.5
  56.5 57.5); stream s1 100% from 480k, s2 not yet; base s3 d0 only (base seeds transition 720k-960k); ctx=3 s1/s2 d0 100
  and nothing beyond at 480k; ctx=5 one eval. The organism input and the continuous stream do NOT break formation; body
  choice by context is the prime suspect (pending 1.2M).
- ss_rv (BabyLM, register recipe + reg_revive=50000) epoch-1 checkpoint, REGS_PROBE: d0 97.7, d1 61.7, d2+ chance; 9
  revivals during the epoch, yet r1 / r3 are pinned at +100 / -100 again at the end, no sg/pl separation. Revival: no effect.
- **BISECTION RESULT (harness, 1.2M sentences, is/are):** base 2/3 (s3 no transition by 1.2M); stream=1 2/2 (100% at 480k
  / 1.2M); inorg=1 2/2 transition but unstable later (s1 100% 480-960k then 76.5-96 at 1.2M; s2 partial 720-960k then
  collapse); **ctx=3 0/2, ctx=5 0/1 - d0 100 and chance beyond at every checkpoint.** => CHOOSING THE BODY BY CONTEXT is
  what breaks circuit formation. Mechanism: with byte-chosen bodies ONE space body handles every post-word position (all
  register reading happens in one body; the noun-final bodies are few and frequent); with context-chosen bodies reading
  and writing are split over hundreds of bodies, each seeing a sliver - none discovers the register; and d0 becomes
  trivial (the body identity encodes "s " vs "g ") so nothing pushes toward a carrier. (Secondary: the organism input
  destabilises a formed circuit - the workspace input, to revisit.)
  Principled fix: the long-range circuit must live in SHARED bodies that see every occurrence (per byte / type), next to
  the context bodies that do local prediction - the organism's shared_types idea. Confirming in the harness (`dual=1`:
  context body + shared per-byte body each step, both into the workspace / registers / head), then port.
- Harness `dual=1`: a shared per-byte body runs next to the context-chosen body every step; both L6 sum into ws (read by
  the readout and the register head), both read the registers and both backprop. Confirmation runs (1.2M): ctx=3+dual
  seeds 1/2/3, ctx=5+dual seed 1, control ctx=3 seed 3. If dual restores formation, port it to the organism as one
  shared body per current byte value running next to the lock-selected body.
- **Design tension (user, 2026-09-25):** "if each body sees more, circuits don't form (bodies do everything individually);
  if bodies see less, more circuits form but they have no idea what's going on." Reading of the bisection: what broke
  was specialisation by SURFACE CONTEXT - the same ROLE (read the register before the verb) split over hundreds of bodies
  keyed by the preceding bytes, each seeing a sliver of the same job. The working harness is still a circuit (writers at
  the noun -> shared register -> reader before the verb); cooperation is forced because the task is distributed in time.
  Resolution to pursue: specialise by ROLE, not by input - a body sees LESS of the input (must cooperate) but EVERY
  instance of its role (learns it). The dual shared body is a stopgap in that direction (risk: it does everything, context
  bodies idle - check with ablations). Proper form: route by role / circuit state (lock tree keyed on internal state
  - ws_key / l4_key - and market credit by downstream consequence), which is also what unbounded layering needs: a
  level-k body serves a role for level k+1 and sees every instance of that role.
- dual at 480k: all arms (ctx3+dual s1-s3, ctx5+dual s1, control ctx3 s3) still chance beyond d0 (working variants
  transitioned 480k-960k; 1.2M decides). Diagnostic launched to split READER vs WRITER fragmentation: ctx=3 + rdirect
  (readout reads registers directly, bypassing fragmented readers) s1/s2, base + rdirect s1, ctx3 + dual + rdirect s1.
- **dual shared body at 1.2M:** ctx=3 alone 0/3 (s3 also fails); ctx=3 + dual 0/3; ctx=5 + dual 1/1 (100% every distance
  from 720k; 93-99 at 1.2M). Not a reliable fix. Hypothesis (unconfirmed, single seed): 3-byte context bodies are strong
  enough locally to dominate ws, so the shared body is never necessary and the long-range role is never learned; 5-byte
  context bodies are weaker, leaving work to the shared body. Waiting for the reader-vs-writer diagnostic (bisect3).
- **READER vs WRITER (harness, 1.2M):** ctx=3 + rdirect 2/2 (100% at 720k / 1.2M); ctx=3 + dual + rdirect 1/1 (960k);
  base + rdirect 1/1 (480k, faster than base). vs ctx=3 without rdirect 0/3. => the break is on the READING side:
  reading the register through context-fragmented bodies fails; context-chosen WRITERS are fine. A direct register
  readout = one shared reader that sees every instance of the role (fits "specialise by role"). The organism already has
  reg_readout=1 yet fails in most seeds -> another difference remains. Next: from ctx=3 + rdirect add inorg, stream, both.
- bisect4 interim (720k; ctx=3 + rdirect): +stream s1 100% from 480k, s2 partial; +inorg s1 partial and FADING with
  distance (82.5 76.5 64.5 53 47.5), s2 not yet; +both s1 fading (72 79.5 62 54 49.5). Hypothesis: the organism's leaky
  persistent workspace input (ws_mem, decay 0.5/byte, cells 96..123) is a competing SHORT-RANGE carrier: readers use it for
  d1-d2 and the registers (the only route to d8) are not needed there. Test launched: inorg with the workspace cells zeroed
  (`nows=1`), ctx=3 + rdirect, seeds 1/2 (+ stream s1).
- **bisect4 FINAL (ctx=3 + rdirect, 1.2M):** +stream 2/2 (720k, 1.2M); +inorg 1/2 (s1 at 1.2M); **+inorg+stream (the most
  organism-like harness) 2/2, both transitioning between 960k and 1.2M.** The "fading" partials were transitions in
  progress (workspace-shortcut hypothesis weakened). The organism input SLOWS the transition (~1.2M vs 480-720k without).
  1.2M sentences ~ 32 MB = exactly the organism runs that failed (D_s2, D_s3) -> the organism may be at the edge of its
  data need rather than broken. Decisive cheap test: organism probe at 64 MB, seeds 2 and 3 (same recipe, reg_readout).
- BabyLM epoch 1, ss_rc (register recipe + reg_center=5000): held-out 2.162, BLiMP 54.37 (highest Adam run; +2.2 vs base
  52.15, beyond the fast-set noise band), supplement 47.2, entity 17.15. REGS_PROBE: no register carries number, long range
  at chance -> the BLiMP gain is not the long-range circuit. Queue for full-set scoring before counting it.
- **User idea (2026-09-25): if the persistent workspace is the drag, see whether the model builds persistence itself, as
  the ALM dendrites do.** ALM long-range reference: persistence is never a hard-wired store - eligibility traces (bounded
  per-synapse participation records converted by later consequence), selective recurrent amplification of the one
  task-relevant state, a small persistent goal winner kept by competitive recurrence, weak bistable memory biasing a
  competent local controller. Coco's ws_mem is the opposite (fixed 0.5/byte leaky sum of every body's L6, fed to every
  body). Coco's own buildable persistence: gated registers (learned hold/write = selective recurrent gain) and the L4
  return. Added `ws_mem_in=0` (organism): cells 96..123 zero, registers still fed at 124..127, no credit through the
  unread workspace. Plan once the harness nows result is in: organism probe + BabyLM with ws_mem_in=0 - formation
  speed/reliability, whether registers take over short-range roles, and skills.
- **Harness: the persistent workspace input IS a drag.** ctx=3 + rdirect + inorg WITHOUT the workspace cells (`nows=1`):
  s1 at 960k, s2 at 720k; + stream s1 at 960k -> 3/3, 20-40% sooner than with the workspace (inorg s1 1.2M, s2 failed;
  both s1/s2 1.2M). Next: organism with ws_mem_in=0 (the model must build its own persistence - registers / L4) on the
  seeds that failed at 32 MB, then BabyLM.
- **Organism with ws_mem_in=0 (32 MB, is/are): seeds 2 / 3 fail** (100 74 49.5 49 55 44.5 / 100 69.5 54.5 54 50.5 51.5),
  although the organism-like harness without the workspace formed 3/3 by 720-960k sentences (~20-26 MB). An
  organism-only difference remains: sequence-memory blend, delayed/truncated register credit (tc_window), or body churn
  (tree growth / births). 64 MB runs (with workspace) still pending.
- User (2026-09-25): try MTP (multi-token prediction); "CLM" meant contrastive learning. Contrastive predictive coding
  (CPC) would give the carrier a direct long-range objective but is an objective on an internal state (tension with
  final-answer-only supervision) -> MTP first (output-level only). Harness `mtp=K mtp_w`: extra readout heads predict
  bytes t+2..t+K from the same ws (+ registers with rdirect); training only. Test: organism-like harness (ctx=3 rdirect
  inorg nows) with mtp=4 / 8, seeds 1 / 2 vs the same without MTP (s1 960k, s2 720k).
- The earlier organism seqmem=0 test ran at 12 MB, below what even the harness needs (~20-30 MB) - not a fair test.
  Launched: organism seqmem=0 + ws_mem_in=0 at 32 MB, seeds 2 / 3.
- **User (2026-09-25): willing to relax some rules (e.g. final-answer-only supervision) if it gets the complete learner;
  after these probes, port to the GPU (~10x CPU).** GPU branch claude/gpu-port-handoff-5bb30b (3 commits on 0861b8b):
  multi-stream trainer (S streams in lockstep over one population, CUDA Adam learner), bit-exact at S=1, up to 180k B/s
  (~9.8x); one BabyLM epoch at S=64 in 1033 s with held-out 2.127, BLiMP 53.47, supplement 51.6, entity 18.06 (= CPU run
  within noise). Supports the Adam + registers recipe (not hops / persist / homeo / market). Merge is clean (new files +
  1 line in organism.h). Port plan: commit this session's work, merge, add ws_mem_in (+ MTP / CPC if adopted) to the
  per-stream path, VERIFY circuit formation under S>1 (agree_probe at S=64, untested), then multi-seed 10-epoch BabyLM
  with REGS_PROBE per checkpoint and full-set scoring. Contrastive option (if MTP is not enough): CPC predicting a pooled
  summary of the REST OF THE SENTENCE (contrasted against other sentences' rests) from the register state.
- **Organism at 64 MB (with workspace, seeds 2 / 3): fail** (94 75 50.5 53 49.5 47 / 97 69.5 50.5 56 54 46.5). ~2.4M
  sentences, twice what the organism-like harness needs -> NOT a data shortage: an organism-only mechanism blocks
  formation. Remaining candidates: sequence-memory blend (32 MB seqmem=0 probes running), delayed/truncated register
  credit (tc_window), body churn (tree growth / births).
- Launched body-churn test: organism episodic=0 (no snapshot births) + ws_mem_in=0 at 32 MB, seeds 2 / 3.
- MTP (harness, organism-like, 1.2M runs in progress): mtp=4 s1 / s2 both 100% every distance by 720k (no MTP: 960k /
  720k) - a modest speed-up; mtp=8 slower so far (d0 only at 480k).
- **User (2026-09-25): move away from body births - several motifs with FIXED populations, only the weights inside the
  layers move; each body wires its L2 into its neighbours' L3 as governed by its motif; multi-body circuits must still
  form (generalisation).** Implemented in the organism: `motifs=h:n,...` (h=1: one body per current byte, 256; h>1: a
  hash of the last h bytes mod n; all bodies created at start; structural / episodic / market / agency forced off; each
  byte runs one body per motif) and `motif_wire=1` (the bodies active in a byte form a chain in motif order: body m's
  L3 receives body m-1's L2 as a routed neighbour message; with bp_route the receiver's gradient returns to the sender's
  L2). Bug found by the smoke run: with agency on, bodies read the agency focus copy of the input, which the motif path
  never filled -> nothing emitted; agency is now forced off in motif mode. agree_probe gained per-motif arms ("silent
  motif hK") and "no motif messages" (the L2->L3 links cut) to verify that the circuit is multi-body.
- **MTP final (harness, organism-like: ctx=3 rdirect inorg nows, 1.2M):** mtp=4 s1 / s2 and mtp=8 s1 / s2 all 4/4 at 100%
  every distance by 720k (no MTP: s1 960k, s2 720k). Modest, consistent speed-up (all seeds by 720k); stable to 1.2M.
  Worth keeping; port to the organism readout (extra heads on ws + register readout) with the GPU work.
- Organism seqmem=0 + ws_mem_in=0 at 32 MB, seeds 2 / 3: fail (100 76 46.5 54 54 50 / 100 71.5 52.5 55 50 54.5). The
  sequence memory is NOT the obstacle. Remaining: body churn (episodic=0 test running; the fixed-motif design removes
  churn entirely) and delayed / truncated register credit.
- Motif smoke (262 KB, after the agency fix): bodies emit, registers active. Launched fixed-motif organism probes (32 MB,
  is/are, seeds 2 / 3 - the seeds that fail with the address tree): motifs=1:256,2:4096,3:8192 units=12544 ws_mem_in=0,
  with and without motif_wire=1 bp_route=1; arms include per-motif silencing and "no motif messages".
- Organism episodic=0 + ws_mem_in=0 at 32 MB, seeds 2 / 3: fail (95.5 68.5 49 53.5 51.5 50.5 / 100 74 50.5 57.5 55 48).
  Only partial churn removal (lock splits / tree growth remain); the fixed-motif runs remove it entirely. If those fail,
  the delayed/truncated register credit is the last organism-only difference.
- **For the FREE-GENERATION phase (user, 2026-09-25): "Interference Search" (Al-ameen, Bad Theory Labs, Sept 2026,
  https://www.badtheorylabs.com/papers/interference-search/paper.pdf).** A decoding/reasoning-time search, not a learning
  rule: a level-synchronous frontier of explicit states - expand all live states at once, the environment executes moves,
  states reached by different paths MERGE (kept once, parents' ranks pooled as votes), a memory of already-expanded merge
  keys blocks revisits, a small trained judge prunes dead ends, the best W advance one level. Countdown (30 hard): 30/30
  vs 21 for a single chain at the same budget of 200 judged positions, 3 sequential rounds vs 23.7; merging carries most
  of the gain; a global best-first memory does WORSE than a single line (never commits to depth); parallel streams that
  could attend to each other learned nothing independent streams did not ("visibility alone did not produce sharing;
  coordination had to be an explicit operation"). Negative results: textual notes about failed attempts primed repeats;
  decode-time rewinding regenerated the same mistake. No training of the model.
  Plan for generation: a merged-frontier decoder for Coco - states = partial continuations merged by an equivalence key
  (e.g. same recent words), a memory of expanded keys to block loops / repeats, the organism's own readout (or a small
  learned coherence judge) as the judge. It cannot regress skills (no weight changes) and targets the looping / collapse
  failure of free generation. Design echo for the motifs: select bodies by a MERGE KEY (abstract "same situation")
  rather than a surface byte hash, so different spellings of one role share a body.
- **Cortical circuit motifs to replicate (user, 2026-09-25; cortical_circuit_motifs_corrected.png in the Coco root):**
  1 parallel feedforward routes -> built (every motif body reads the input and emits; motif_wire adds a serial path);
  8 selective fan-out -> star wiring (byte body's L2 to all context bodies; msg_merge summing exists) - cheap;
  2 selective recurrent excitation -> cross-byte loop (this byte's motif bodies receive the previous byte's L2 messages;
    persistence still needs the gated registers - "recurrence alone does not guarantee persistence") - moderate;
  7 dendritic integration -> an APICAL compartment: registers / feedback MULTIPLY (gate) a body's L5/L6 instead of adding
    into L1 - targets the register-READING failure and matches the ALM dendrites - top pick;
  4 feedback inhibition -> a small I population normalising the active bodies' emissions (learned divisive normalisation;
    gain control vs the recurring saturation); 5 cross-assembly competition -> soft winner-take-all among the bodies active
    in a byte; 3 feedforward inhibition -> input drives a body and its inhibitor (timing/gain); 6 disinhibition -> explicit
    control -> gate chains (the register write gate is gate-like). Inhibitory motifs need sign structure: I-bodies whose
    output is constrained to subtract / normalise (Dale's law; Coco's INT4 weights are signed but roles are not).
  Order after the fixed-motif baseline: chain vs star; apical gating of registers; cross-byte loop; feedback inhibition /
  competition. Each on the probe (formation + per-motif / no-message ablations) then BabyLM skills.
- **User plan (2026-09-25): (1) current runs report -> know the blocker; (2) port to GPU; (3) confirm no regressions on GPU
  (same config CPU vs GPU: is/are probe formation + ablations, BabyLM epoch held-out + skills); (4) then the cortical
  motifs.** Port = commit this session's work, merge claude/gpu-port-handoff-5bb30b locally (no push), add ws_mem_in,
  the fixed-motif mode (+ motif_wire) if it is the fix, and MTP to the multi-stream trainer.
- **BLOCKER FOUND: the ADDRESS TREE (body churn). Fixed motifs fix it; the user's L2->L3 wiring makes it reliable.**
  Organism is/are probe, 32 MB, seeds 2 / 3 (both failed in EVERY address-tree configuration: tree, no workspace, no
  sequence memory, no episodic births, 64 MB):
  | config | seed 2 | seed 3 |
  |---|---|---|
  | fixed motifs 1:256,2:4096,3:8192, ws_mem_in=0 | 100 100 100 100 99.5 99.5 | 100 78 51.5 52 54 55 |
  | + motif_wire=1 bp_route=1 | **100 100 100 100 99 99** | **100 100 99.5 100 99 99.5** |
  Not births as such (episodic=0 alone failed) - the tree's assignment of contexts to bodies as it grows. Fixed motifs
  3/4; wired motifs 2/2. Ablations (per-motif silencing, no motif messages) pending - the multi-body check.
- **Fixed-motif ablations (32 MB, d0 d1 d2 d4 d6 d8): the circuits are MULTI-BODY.**
  motifs s2: constant bodies 71.5 50 52.5 55.5 51.5 47; no ws read 74.5 57 54 48.5 49 53.5; silent h1 78.5 67.5 66 56.5
    57.5 52; silent h2 ~100 (unused); silent h3 78 73 71.5 62.5 62 64.5 -> byte + 3-byte motifs both needed.
  wired s2: constant 100 94 87 86 85 84.5; no ws read ~97-98.5; silent h1 100 93.5 86 88.5 84 84.5; silent h2 49.5 51 52
    45.5 47.5 52.5 (collapse, even d0); silent h3 ~100; NO MOTIF MESSAGES 100 99.5 99 99 98.5 96.5.
  wired s3: constant 36.5 61 62.5 52.5 49.5 55.5; no ws read 65.5 65.5 70 63.5 51 60.5; silent h1 44 52.5 61 51.5 52.5 57;
    silent h2 85.5 82.5 72 68.5 62.5 52; silent h3 98-94.5; NO MOTIF MESSAGES ~100.
  => every successful circuit needs TWO motifs (silencing either collapses long range; which pair varies by seed);
  cooperation runs through the shared workspace / registers, NOT the L2->L3 links (cutting messages at inference costs
  <= 2.5 points). Wired formed 2/2 vs 1/2 unwired - too few seeds to credit the wiring. Next (user plan): GPU port.
- **GPU port (2026-09-25):** committed dc20c77, merged claude/gpu-port-handoff-5bb30b locally (5c6a3e0, clean; not pushed).
  Multi-stream trainer additions: fixed motifs (P1 picks one body per motif, S1 skips admission; the trainer mirrors
  Organism's forcing of structural / episodic / agency / market off; set_mode / set_train can no longer switch structure
  back on in motif mode), ws_mem_in=0 (workspace cells zero in P1, learner gscale 0 = no workspace credit; register
  credit unchanged). motif_wire is refused for now (needs bp_route in the CUDA learner). Build script for the main working
  copy: build-gpu/build.sh (uses the GPU worktree's patched CUDA headers); agree_probe gained MS=S (train with the
  multi-stream GPU trainer, export stream 0, score with the same CPU arms) -> build-gpu/agree_ms.
  Checks running: ms_check exact (S=1 reference learner vs Organism, bit-identical) on the motif and default configs;
  GPU probe (MS=64) motif config seeds 2 / 3 vs the CPU results (s2 100 100 100 100 99.5 99.5; s3 fail).
- **Exactness after the merge (ms_check exact, S=1 reference learner vs Organism): PASS on both** - motif config
  (motifs=1:256,2:4096,3:8192 units=12544 ws_mem_in=0) 2,097,152 events bit-identical (loss, prediction, weights,
  population, every block); default config 1,048,576 events bit-identical. The merged code computes what the organism does.
- GPU motif probe (MS=64, 32 MB, seeds 2 / 3): both trained in ~10 min sharing the GPU (33.5M events each); CPU arms
  running. GPU BabyLM run gpu_motif launched: ms_babylm 64 streams, --overlap, 10 epochs, motifs=1:256,2:16384,3:32768,
  4:32768 units=82176 ws_mem_in=0 (unwired); ~131k B/s (~7x the CPU Adam run), ~7 min/epoch; each epoch_K.state scored
  on the fast sets as it lands (build-gpu/babylm_gpu.sh).
- **GPU (MS=64) motif probe = CPU result:** seed 2 100 100 100 99 99 99.5 (CPU 100 100 100 100 99.5 99.5); seed 3 fail
  (100 75 56 46.5 52 50; CPU unwired s3 also failed). Multi-stream training does not change circuit formation; ~1/3 of
  the CPU time. (Wiring - which rescued s3 on CPU - awaits bp_route in the CUDA learner.)
- **FULL-SET scores, epoch 1 (BLiMP 59,875 items: two-run CI ~+-0.6; supplement ~5.2k: ~+-1.9):**
  | run | full BLiMP | full supplement |
  |---|---|---|
  | ss10_base (old rule) | 51.88 | 49.76 |
  | ss10_adr (full Adam) | **53.41 (+1.53)** | 47.35 (-2.4, borderline) |
  | ss10_adr_A2ro (full Adam + registers) | **53.09 (+1.21)** | 49.09 (-0.7) |
  | ss_adh (Adam hidden only) | 50.89 (-0.99) | 49.58 |
  Per phenomenon (full labels, vs base): full Adam npi +5.0, anaphor +4.8, island +4.6, ellipsis +4.3, filler_gap +2.5,
  control +2.5, binding +2.2; argument_structure -3.3 (real), s-selection -1.6. + registers: npi +7.1, anaphor +3.3,
  island +2.5, ellipsis +2.2, control +1.8, binding +1.4; irregular -2.1, quantifiers -2.0, argument_structure -1.0.
  => the full-Adam BLiMP gain is REAL (structural phenomena); the remaining regression is argument structure / lexical.
  Scripts: build-task/fullrep.sh, build-task/phenfull.sh BASE E RUN E.
- GPU (MS=64) motif probe ablations, seed 2: constant bodies 88 52 53 51.5 52 47.5; no ws read 87.5 60 59 59 54 49;
  silent h1 100 97.5 97.5 95.5 95 92.5; silent h2 81 49.5 56 48.5 46 55 (collapse); silent h3 100 74.5 72.5 71.5 79 84.
  -> multi-body on GPU too (2-byte + 3-byte motifs). Wall 2542 s: ~10 min training, ~32 min CPU arm scoring (a batched
  GPU scorer would remove most of it).
- **Item A (circuit on real text), GPU motif run:** held-out 1.957 1.915 1.894 1.879 1.877 1.862 1.850 1.842 1.850
  (epochs 1-9; the CPU tree + full-Adam run ss10_adr was 1.951 at epoch 8). REGS_PROBE (is/are language on the frozen
  checkpoint) at epochs 1 / 4 / 8: d0 ~97, d1 ~60, d2+ chance, no register separates number -> the agreement circuit has
  NOT formed on real text. REGS_ABLATE (held-out with registers read as 0): motif e8 1.842 -> 1.892 (+0.051); tree+regs
  ss10_adr_A2ro e1 2.123 -> 2.480 (+0.358) -> the registers ARE used on real text, for other features (4 registers
  shared by everything; agreement's signal is sparse and varied). Levers: more signal per instance (MTP, item E) and
  more carrier capacity (with ws_mem_in=0 cells 96..123 are free: up to 32 registers).
- **OOM incident (14:30):** three scorers x 8 forking workers on the 82k-body model + the GPU trainer + the two CPU
  10-epoch runs exceeded WSL's 25 GB; the OOM killer took the scorers, gpu_motif (after epoch 9, checkpoint saved) and
  ss10_adr / ss10_adr_A2ro (at epochs 8 / 7; not restarted - superseded by GPU runs). Rule: one scorer at a time.
- WSL restarted itself after the OOM (E_UNEXPECTED); all processes gone, /tmp cleared (probe outputs were logged
  already); ~/babylm/runs intact.
- **2026-09-25 afternoon.** WSL went down again at 14:37-14:42 (the batched-scorer arm check in the other session ran it
  out of memory): gpu_strict (Strict corpus, 2 epochs) died at ~8M events -> relaunched as gpu_strict2. Merged the
  user's batched scorer (ms_score, `MultiStream::score`, no fork): 444/444 candidates byte-identical to
  model_babylm_score on the MOTIF checkpoint too; full BLiMP + supplement + entity in ~8 min per checkpoint (was hours).
  build-gpu/ms_score_seq.sh (SET=full -> full_T_eE.jsonl), build-task/fullrep2.sh RUN:E ... Runs pruned by the user
  (build-task/prune_runs.sh: 465 checkpoints, 104 -> 4.5 GB; scores / logs / configs kept).
- **Item B, full sets, gpu_motif (82k fixed-motif bodies, ws_mem_in=0, 64 streams GPU):**
  | epoch | BLiMP | supplement | entity |
  |---|---|---|---|
  | 1 | 52.24 | 50.67 | 17.63 |
  | 4 | 52.76 | 51.98 | 17.67 |
  | 8 | 52.00 | 49.83 | 18.45 |
  | 9 | 51.95 | 51.15 | 17.46 |
  Skills FLAT over epochs while held-out bpb falls 1.957 -> 1.850. Per phenomenon, e1 vs the CPU tree full-Adam run
  ss10_adr e1: argument_structure +4.0 (the Adam regression is gone), filler_gap +6.7, irregular +5.6, anaphor +2.4,
  s-selection +2.2; but npi_licensing -12.6, ellipsis -11.6, quantifiers -9.3 (vs ss10_base: -7.6 / -7.3 / -9.8). e1 -> e9:
  anaphor +7.1, npi -6.6. Size caveat: 82,176 bodies x 5,632 weights = 463M INT4 weights (4x the 20k tree models; Adam
  state 7.4 GB on the GPU) - the motif model is not parameter-matched and the extra bodies buy no BLiMP.
  Cause tests running: motif + ws_mem_in=1 (gpu_motif_ws: held-out 2.222 at e1 vs 1.957 - the decaying workspace input
  hurts the motif model's loss badly), and the GPU tree run ms64_gpu_ov on the full sets.
- **Item C, motif wiring on the GPU: implemented and exact.** Trainer: P2 routes body k's L3 from body k-1's L2
  (SRC_ROUTED, src k-1); S2 recomputes a receiver's L3..L6 with a zero message when its sender stopped at L1, and gates
  routed bodies by route utility as run_body does; P3 makes a direct job of every body past L2 when bp_route is on
  (LTrace.flags LF_OUT = emitted; message-only jobs have no output error but count as a use, as learn_backprop's
  adam_touch does) and links each receiver to its sender's job (LTrace.sender = offset); P4's ring keeps writers only.
  RefLearner: learn_backprop's order (receiver adds W3[:, A..] g3 into its sender's message; the sender adds it to g2
  before L2's live mask). CUDA: k_msg (all receivers, in parallel, before any sender: g3 depends only on the receiver's
  own error) then k_backward with the message at g2. Organism fix: a routed body whose sender stopped at L1 read that
  trace's STALE L2 (from an earlier byte); it now gets a zero message (motif_wire only; tree routing always has fresh
  senders). ms_check exact, S=1 reference vs Organism: wired + bp_route 2,097,152 events BIT-IDENTICAL; unwired motifs
  2,097,152 and the default config 1,048,576 still identical; the three configs' losses differ (2.5516 / 2.5430 wired
  without bp_route / 2.5616 unwired at 262k) so the paths are live. Dual (GPU vs reference): S=1 1 MB 79 of 534M nibbles
  flip (1.5e-7), Adam m 7e-8, head 9e-6; S=16 2 MB 1172 of 1.08e9 (1.1e-6), m 1.1e-7, head 4e-6. Unwired-motif dual at
  S=16 running for comparison (motif bodies are shared across streams far more than tree bodies: more atomics on the
  same gradient rows). Wired GPU probe (MS=64, seeds 2 / 3) queued.
- **Item H, free generation: merged-frontier decoder (apps/model_frontier.cpp, docs/FRONTIER_DECODER.md; subagent,
  merged 0ce4bcf).** Frozen checkpoint, each branch in a forked child (no weight can change: no skill regression by
  construction). gpu_motif e9, 20 prompts x 200 bytes: loops (repeated word 4-gram) greedy 9/20, sampling T=0.8 0/20,
  frontier (W4 N4 key3) 0/20; valid words 99.4 / 92.0 / 99.5%; mean log-prob/byte -0.485 / -1.061 / -0.336. BUT the
  likelihood judge turns search into RECALL: 29% of frontier bytes are written under a >=32-byte verbatim sequence-memory
  match (greedy 14%, sampling 1%), different prompts converge on the same training passages; judge copy penalty
  (copy=1) cuts it to 2% at -0.447. Merging almost never fires in free text (0.2-0.5 cross-parent merges per
  generation; 198/200 chosen states had one vote): the expanded-key MEMORY removes the loops, not the merging (unlike
  the paper). No decoder stays on the prompt's topic (the model closes the line and switches genre). Next: a judge
  that rewards dependence on the prompt (log-prob with vs without the prompt). Side finding: read-only tools build
  the model with train=1 buffers (acc16 + Adam, 8.1 GiB for 82k bodies) before load_state; train=0 gives identical
  outputs at 0.35 GiB and 20x cheaper forks - this is what made model_babylm_score's forks OOM.
- **Item B, what the workspace input does (full sets, epoch 1):**
  | run | BLiMP | supplement | entity | held-out |
  |---|---|---|---|---|
  | ms64_gpu_ov (GPU tree, ws_mem_in=1, registers) | 53.36 | 48.81 | 17.32 | 2.127 |
  | gpu_motif (82k motifs, ws_mem_in=0) | 52.24 | 50.67 | 17.63 | 1.957 |
  | gpu_motif_ws (82k motifs, ws_mem_in=1) | 53.20 | 51.00 | 16.91 | 2.222 |
  | gpu_m20 (20k motifs, ws_mem_in=0) | 52.52 | 48.59 | 17.83 | 1.966 |
  The GPU tree run equals the CPU tree full-Adam run (53.36 vs 53.41); the CPU run's argument_structure -3.3 does not
  recur on the GPU (tree 54.2, motifs 54.9 / 55.2, base 54.2): B's original regression looks like that run's variance.
  The fixed-decay workspace input is the NPI / ellipsis lever: motifs ws 0 -> 1: npi +19.8, ellipsis +14.0,
  quantifiers +3.1, irregular +2.0; but filler_gap -9.2, binding -3.0, island -2.5, anaphor -2.4, control -2.3; held-out
  much worse (2.222 vs 1.957) - skills and bpb disagree. It carries "what came earlier in the sentence" (a licensor,
  an antecedent) but as 28 loud cells at the byte cells' scale. 20k vs 82k bodies: same BLiMP (52.52 vs 52.24) and bpb
  (1.966 vs 1.957) -> USER: 20k bodies until advanced multi-body circuits are evidenced. Queued (20k): wired, ws gain 1,
  ws gain 0.25 (new ws_in_gain), each one epoch + full sets; conformance on the 20k checkpoints.
- Item C dual, unwired motifs S=16: 1573 of 1.08e9 nibbles flip (1.45e-6) = the wired rate (1.08e-6): the rate is motif
  mode's (bodies shared by many streams -> many atomic adds on the same rows), not the message path.
- Item D (apical register gating, organism only): `apical=1` gates each L5 unit by 1 + (U reg)/256 (0..2x), U = 64 INT4
  weights per body in the unused L4 block, zero at birth (starts identical to off); backward through the gate into U and
  into the registers (gws cells 28..31 -> temporal credit). Smoke at 2 MB runs; 32 MB CPU probes (seed 3 unwired failed
  without it, seed 2 formed) running. The multi-stream trainer refuses apical until it is worth porting.
- Item A dilution bisection queued (agree_probe MIX=f MIX_FILE: real BabyLM lines interleaved with the synthetic
  sentences, synthetic part fixed at 32 MB): wired probe seed 2 at 50% and 80% real text.
- **Item C, SETBACK: the wired probe does NOT form when trained with 64 GPU streams** (32 MB, ISARE): s2 90.0 48.5 48.0 49.0
  53.5 50.0; s3 100 53 56 44.5 47 49.5 (CPU organism, earlier code: s2 100 100 100 100 99 99, s3 100 100 99.5 100 99 99.5;
  unwired at 64 GPU streams DID match the CPU: s2 100 100 100 99 99 99.5). Registers separate number only at d0 (s2 r2
  3.0, s3 r0 -4.6). Exactness (S=1) and dual (S=16) pass, so the bisection: E1 today's CPU organism wired s2 (does the
  organism still form?), E2 64 streams with the CPU reference learner (GPU-specific or multi-stream?). The wired 20k
  BabyLM runs (gpu_m20w, gpu_strict_m20w) are paused (SIGSTOP) until this is settled.
- **Item A:** REGS_PROBE on every real-text model: gpu_m20 (20k motifs) d0 96.7 d1 60.7 d2+ chance; gpu_motif_ws (82k,
  ws_mem_in=1) 96.0 / 59.0 / chance; ms64_gpu_ov (GPU tree + ws) 97.0 / 61.0 / chance. No model trained on text carries
  number past one word (BLiMP subject_verb_agreement ~50 in all). REGS_ABLATE gpu_m20: 1.966 -> 2.039 (+0.072) - the
  registers are used, for other things. Dilution bisection relaunched on the UNWIRED probe (forms on the GPU).
- **Item F, conformance on gpu_m20 (20k motifs, epoch 1; lean read-only model, C5 now applies to motif_wire):** C1 PASS
  1.353 bits (bodies over the sequence memory), C2 PASS 0.132 (context computation), C3 PASS every layer (L1 0.487, L2
  0.182, L3 0.094, L5 0.444), C10 PASS 0.038 (bodies read what earlier bodies wrote); C7 FAIL (no body dies before L6);
  C8 informational (saturated L1 72%, L2-L5 ~59%, L6 39%); skill arms at stride 10 (+-2.7) inconclusive (S1 blimp -0.97,
  S2 +0.97).
- **Item G:** design written, docs/UNBOUNDED_CIRCUITS.md: recruitment by a learned ROLE CODE (the sign pattern of a
  body's L2 hashed into a fixed routed pool: same role -> same next body, a merge key), depth gated by route utility,
  messages and message gradients as motif_wire / bp_route; test = a depth ladder (parity-k, nested agreement) where the
  largest k solved must rise with the allowed hops; harness first.
- **Item A dilution, first point: the circuit FORMS with 50% real text.** Unwired probe, seed 2, 64 GPU streams, 32 MB
  synthetic + 32 MB BabyLM lines interleaved line by line: 100.0 99.5 99.0 99.5 98.5 98.5 (pure synthetic: 100 100 100 99
  99 99.5). Real text at this dose does not interfere with forming it. 80% (32 + 128 MB) running.
- **Item D, apical register gating (CPU organism probe, 32 MB, unwired motifs): NEUTRAL for formation.** seed 2 100.0
  100.0 100.0 100.0 100.0 99.5 (without: 100 100 100 100 99.5 99.5); seed 3 100.0 54.5 56.0 49.5 50.5 57.5 (without: 100 78
  51.5 52 54 55, still fails). The multiplicative gate neither breaks a forming seed nor rescues a failing one; not
  ported to the multi-stream trainer for now.
- **Item E, MTP in the organism (`mtp=K`, `mtp_w`):** heads h = 1..K on the ring slots' head input (event t-h's clamped
  workspace) predict this event's byte; their error joins event t-h's writers' delayed pass; heads step with the
  readout's minibatch Adam. 1 MB smoke (seed 3): mean head losses +1 1.51, +2 1.90, +3 2.21, +4 2.45 nats (rising with
  the horizon, as they should). 32 MB seed 3 (fails without) running.
- **Item C bisection, E1: today's CPU organism, wired + bp_route, seed 2, 32 MB: 100 100 100 100 99 99 - FORMS** (as before).
  The code changes of the day (message fix, apical/ws_in_gain/mtp off) did not break wiring; the failure is specific
  to multi-stream training (the user's point: the streams change the ORDER of events and updates). E2 (64 streams, CPU
  reference learner) stopped for memory - the S=16 dual already shows GPU = reference to float noise. E3 (16 GPU
  streams) running; diagnostic: fraction of routed motif bodies silenced by the route gate, CPU vs 64 streams (8 MB).
- **Item C diagnostic (8 MB, wired s2): the route gate is NOT the difference.** Routed bodies below the gate: CPU h2 1.8%
  h3 3.6%; 64 streams h2 2.3% h3 3.9%. But their mean route utility (counterfactual bits they gain) is 0.21 / 0.20 on the
  CPU vs 0.12 / 0.13 with 64 streams: with 64 streams the wired bodies learn to be LESS useful. E3 (16 streams) pending.
- **Word-clocked register writes (`reg_clock=1`; user: "transformers probably get some of this from the tokenizer").**
  At the byte level "the dogs in the big red house are" puts ~30 bytes between noun and verb (8 subword tokens); the write
  head decides at EVERY byte whether to overwrite, so a stored number must survive ~5x the decisions it would face per
  token, and the 64-event credit window spans ~12 words. reg_clock=1 shuts the head's gate inside words (the byte just
  read is a letter or digit): one overwrite decision per word; registers are still read at every byte; the ring's G / hg
  are 0 there, so credit passes through untouched and no learner change is needed. Organism + trainer; ms_check exact
  524,288 events bit-identical. Queued: probe unwired s3 (fails per byte) and s2, 64 GPU streams; one BabyLM epoch at 20k
  bodies -> REGS_PROBE (does number get stored on real text?) and full sets.
- **Item C RESOLVED as correctness: the wired probe FORMS with 16 GPU streams** (E3, seed 2, 32 MB: 100 100 100 99.5 100 99;
  CPU S=1 100 100 100 100 99 99; 64 GPU streams fail on s2 and s3). The GPU message-gradient path is right; the failure
  is an ORDERING effect of many streams (the user's diagnosis) that appears between S=16 and S=64. At 8 MB the 64-stream
  run already trails (d1 55 vs CPU 79) and its wired bodies' route utility is 40% lower. Leading suspect: delayed
  temporal credit - a writer's delayed pass comes 64 steps after its write, 1,024 global events at S=16 but 4,096 at
  S=64, and a wired body's input depends on its sender's L2, so its stale gradient is staler than an unwired body's
  (unwired forms at S=64). Practice: wired probes (and G's depth ladder) at S<=16; BabyLM-scale runs unwired at S=64.
- **Item B, workspace input at 20k (full sets, epoch 1):** gpu_m20ws (gain 1): held-out 2.207, BLiMP 52.8 vs 52.5 without;
  per phenomenon quantifiers +13.5, argument_structure +3.2, ellipsis +2.8, npi +2.1; filler_gap -8.4, anaphor -3.6,
  binding -2.8, control -2.6, irregular -2.4. At 82k the same switch gave npi +19.8 / ellipsis +14.0 / quantifiers +3.1.
  CAUTION: only filler_gap's drop repeats (-8.4 / -9.2); the big npi / ellipsis / quantifiers swings do not, so single
  training runs vary a lot on those phenomena - a second training seed of gpu_m20 is queued to measure the spread
  before judging per-phenomenon "regressions". Gain 0.25 (ws_in_gain): held-out 1.969 (no input 1.966, gain 1 2.207): the
  bpb cost is the input's loudness; its full sets were rejected by the old ms_score (unknown option) - rebuilt, rescoring
  (ms_score_seq.sh now logs FAILED on a scorer error instead of "scored").
- **TILE PYRAMID (user, 2026-09-25: "vector tiling" for information zooming; "straight to pyramid").** Organism only
  (`tiles=L tile_k=K tile_bodies=N`, needs fixed motifs; the trainer refuses it for now). Level-0 tiles are WORDS: at a
  word boundary (the byte just read ends a word) the workspace summary clamp(ws/128) closes a tile. A level-l tile closes
  at every 2nd level-(l-1) tile and is their MERGE by a merge body (a fixed population per level, chosen by the two
  children's sign bits: same roles -> same merge body; input [older, newer] x 32, output L6 / 32). The VIEW is the last K
  tiles of every level (fine near now, coarse further back); each view slot is read into the readout's input through a
  learned 32x32 mixing (zero at start: identical to off). Credit: the readout error reaches every tile through its
  mixing; a word tile's error goes to the writers of the event that closed it via the temporal-credit ring (like MTP);
  a merged tile accumulates its error while in view and trains its merge body when it leaves (children not credited
  through the parent yet). HOLDING IS STRUCTURAL: a tile stays in view by construction, nothing learns to keep a gate
  shut - the part that fails on real text. Mixings and views are checkpointed.
  2 MB smoke, unwired motifs, seed 3 (the seed that fails per byte), tiles=3 K=4: 100.0 98.0 85.0 74.0 53.5 45.5 (no
  tiles at 2 MB: 100.0 75.5 55.0 52.5 59.0 43.5) - number is carried up to 4 words after 2 MB; d6-8 need the merge
  bodies to keep it in the pair / quad tiles. Running: 32 MB probes s3 / s2; one CPU BabyLM epoch at 20k bodies +
  tiles=3 (cpu_m20t) -> REGS_PROBE (number to the verb on real text?) and fast-set skills.
- **Item H, prompt-dependence judge (subagent, merged): decoding CANNOT keep the model on topic - the model forgets the
  prompt within 2-4 words.** `pmi=L` scores a continuation by log p(cont | prompt) - L log p(cont | no prompt). The
  prompt's effect on each byte's log-prob: bytes 0-3 2.2-3.4 nats, 4-7 0.3-0.5, 8-15 0.04-0.14, from byte 32 at most
  0.016. Past ~2 words the PMI term is the same for every candidate: L=0.3-0.5 falls back to plain likelihood (topic
  measures at chance, no loops, 99.5-100% words, copying unchanged); L=1 rewards short words -> word salad (lp -1.30 /
  -1.42). copy=1 stays the working fix for recall. New measures: prompt GAIN, REACH (the last byte the prompt still
  moves by > 0.01 nats: 14-27 bytes for the frontier decoders), ident@1 (is the output likelier after its own prompt than
  the 19 others; second halves at chance). => H and A are the SAME limit: no mechanism holds context beyond a few words.
  REACH becomes a context-memory metric for real text - the tile pyramid is the candidate fix for both.
- **WORD CLOCK FORMS THE FAILING SEED: probe unwired s3, 64 GPU streams, reg_clock=1: 96.5 100 100 99 99 96.5** (per-byte
  writes: 100 75 56 46.5 52 50, CPU 100 78 51.5 52 54 55). One overwrite decision per word instead of per byte is what s3
  lacked. (s2 running.) MTP (mtp=4) does NOT rescue s3 (CPU: 100 71.5 53.5 47.5 51 50).
- **Dilution, second point: 80% real text PREVENTS formation** (unwired s2, 64 streams, 32 MB synthetic + 128 MB real: 100
  66.5 51.5 51.5 60 49.5; 50% formed). Real text interferes once it dominates - more epochs alone are unlikely to be enough.
- **Word clock on real text (gpu_m20rc, 20k, 1 epoch):** held-out 1.929 (gpu_m20 1.966, seed 2 1.968) but REGS_PROBE d0 97.3
  d1 61.3 then chance, and all four registers PINNED at +-100 (constants: reading them as 0 costs +0.218 bpb as a bias,
  not as information). Its bpb gain is steadier registers, not memory.
- **Item B, SEED VARIANCE (full sets, gpu_m20 seed 1 vs seed 2, same config):** BLiMP 52.52 vs 52.19 (-0.3), supplement 48.59
  vs 49.61, entity 17.83 vs 17.85; per phenomenon up to +-7 (irregular -6.9, quantifiers +5.3, anaphor -4.5, npi -3.2; the
  rest <= 1.1). => single-run per-phenomenon differences under ~7 points are training noise: the original
  "argument_structure -3.3" was within it; filler_gap's drop with the workspace input (-8.4 / -9.2 vs seed spread 1.1) looks
  real. 20k full sets: m20 52.52 / 48.59 / 17.83; word clock 52.83 / 49.47 / 17.77; ws gain 1 52.78 / 49.93 / 16.16; ws gain
  0.25 51.12 / 48.95 / 17.37 (BLiMP -1.4: the quiet input costs BLiMP though it restores bpb). Every config is within ~1 of
  the old baseline (ss10_base 51.88): no reliable regression, no reliable gain.
- Word clock, probe seed 2 (64 GPU streams): 100 95 99.5 98.5 98.5 99.5 - forms (both seeds form with reg_clock=1).
- **Memory design (user: working, short-term and long-term memory without a transformer's KV-cache blow-up):**
  docs/PYRAMID_MEMORY.md - three fixed-size tiers: short-term = the pyramid view (built); working = ~8 slots filled by
  PROMOTION BY USE (a tile about to leave the view is kept if its readers leaned on it more than the weakest slot's
  decaying use); long-term = a fixed-capacity tile store keyed by the ROLE CODE (LSH sign bits), one lookup per word,
  consolidation by averaging. No learned hold decisions anywhere; constant memory and per-byte compute in context length.
- **Pyramid tier 1 at 32 MB, probe seed 3 (CPU): 100.0 100.0 89.5 97.0 59.5 58.5** (per-byte registers 100 78 51.5 52 54 55;
  word clock at 64 GPU streams 96.5 100 100 99 99 96.5). Number is held through d4 - what the level-0 view (4 word tiles)
  covers directly - but barely survives a MERGE (d6 / d8 ~59). Cold start on the merge path: the pair / quad slots'
  mixings start at 0 -> merge bodies get almost no gradient -> they never learn to keep number -> the mixings have nothing
  to read; and v1 does not credit children through their parent. Tier 2 (a noun tile the readers leaned on at d1-3 is
  promoted when it leaves level 0 and stays readable without any merge) is the direct test: 32 MB seed 3 running.
- Pyramid bisection launched (32 MB, seed 3): `tile_avg=1` - a parent is the MEAN of its children (no merge body; any linear
  signal such as number survives; credit reaches every word tile underneath, 1/leaves each, through the ring) - vs the
  learned merges (d6 / d8 ~59). If averaging carries d6-8, the merge bodies' cold start is the bottleneck. Also running:
  tier 2 (tile_wm=8) and learned merges on seed 2.
- Pyramid tier 1 (learned merges), probe seed 2, 32 MB: 100.0 99.0 78.5 91.0 92.5 99.0 - but seed 2 also forms with
  registers alone, so this does not say which memory carries d6-8. Added arms (MEM_ARMS=1): "no tile read"
  (diag_zero_tiles) and "no register read" (diag_zero_regs); seed 2 rerun with them to attribute.
- **User: "D8 isn't really useful" - useful memory is capacity x span x retrieval on real text, not one bit over 8 words.**
  New yardsticks: BabyLM entity tracking (~17-18 today), REACH (15-27 bytes today), and a BIND-AND-RECALL probe
  (apps/recall_probe.cpp): k entity->object bindings ("the red box has the key ."), N words of random filler, then a
  question in a DIFFERENT surface form ("in the red box is the ___") so the sequence memory cannot copy; correct object vs
  another object bound in the story; table k in {1,2,4} x N in {0,10,30,60,120}; MEM_ARMS attributes it to tiles or
  registers. Prediction: linear reads of fixed slots can serve k=1 but cannot pick among k>=2 bindings (no multiplicative
  query x memory interaction) -> `tile_att=1`: a CONTENT-ADDRESSED read - query = Wq (this event's workspace / 128) after
  the bodies ran, keys = Wk tile, softmax over the live slots (view + working memory, ~20: bounded, not a KV cache), read
  = Wo sum(alpha tile) into the readout's input (Wo = 0 at start); the query's gradient joins the bodies' own error (they
  learn to ask); a tile's gradient through its value and key goes to its writers / merge body as before and counts as
  use. Running (32 MB, seed 2): registers only; pyramid + working memory; + content-addressed read.
- **Item A, MORE DATA DOES NOT FORM THE CIRCUIT.** gpu_strict_m20 (20k motifs, BabyLM STRICT, 100M words, epoch 1 = ~10
  Strict-Small epochs of text): REGS_PROBE d0 97.0, d1 57.7, then chance; register r0 WRITES number at the noun (d0
  separation 1.02: sg -38 / pl +50) but does not hold it one word. Held-out (Strict-Small set) 1.869 vs 1.966 after one
  Strict-Small epoch: more text improves prediction, not memory. => the lever is the memory architecture (pyramid tiers,
  content-addressed read), not epochs.
- **PRE-PRETRAINING (user: circuits it cannot discover without specially curated data).** Our evidence agrees: circuits form
  where the signal is dense (the synthetic language, 50% real text) and fail where it is sparse (80%, BabyLM, 10x BabyLM);
  and formal-language pre-pretraining is known to give transformers lasting syntactic biases. Schedule: FORM on curated data
  -> ANNEAL toward real text -> LIVE on real text with a little synthetic replay. The test: does a circuit formed on curated
  data survive real-text training and get used (REGS_PROBE, BLiMP subject-verb agreement ~50 today)? Tooling: agree_probe
  SAVE_STATE=path; ms_babylm --init STATE (MultiStream::reload_from_container: the learner takes the loaded weights and head).
  Running: stage 0 = the is/are language on the exact 20k BabyLM configuration + reg_clock=1 (64 GPU streams, 32 MB);
  stage 1 = one BabyLM epoch from it (gpu_m20pp) -> REGS_PROBE, REGS_ABLATE, full sets; vs gpu_m20rc (no stage 0).
  (BabyLM Strict rules restrict training data to the corpus: generated formal data is a deviation for a submission.)
- Strict epoch 1 full sets (gpu_strict_m20, 10x the text): BLiMP 52.52 (= one Strict-Small epoch), supplement 52.80 (+3.7,
  beyond noise), entity 18.32 (+0.5). More text improves prediction and the supplement, not structure-dependent skills.
- **USER (2026-09-25): BabyLM is a benchmark, not the goal. The goal is the complete learner, ending in a COHERENT TALKING
  MODEL that PERFORMS TASKS and is NOT a transformer.** Yardsticks from here: generation coherence (REACH / topic / ident@1),
  recall capacity x span, task accuracy (answer channel, bAbI-style), entity tracking; BLiMP etc. as regression checks;
  benchmark data rules do not constrain the research. The content-addressed read (tile_att) is bounded soft attention over
  ~20 slots (no token self-attention, no KV cache); a Coco-native alternative is a ROUTED read (the query's role code picks
  the slot by hash, as merge bodies are picked) if attention is to be avoided entirely.
- **cpu_m20t INVALID (my launch error):** model_babylm's default parameters are only the base set (hidden=modulated, no
  Adam); the earlier CPU runs passed the whole learning configuration explicitly, this launch passed only the motif /
  register / tile options -> it trained with the old modulated rule, no Adam, no readout Adam (the tile mixings never
  trained: held-out 3.14 with, without registers and without tiles, identical). Renamed cpu_m20t.invalid_params. Relaunched
  with the full configuration and memory v1 (tiles=3 tile_wm=8 tile_att=1): cpu_m20mem -> REGS_PROBE / ablations / REACH.
- Pyramid tier 2 (tile_wm=8) at 32 MB, probe seed 3: 100 100 93.5 88.5 52.5 55.0 (tier 1: 100 100 89.5 97 59.5 58.5) - no
  gain at d6-8. Likely cause: the working slots have per-slot read mixings, but items move between slots on promotion, so no
  slot's mixing can learn a consistent read -> working memory needs a slot-agnostic read (tile_att: running on recall).
- **AVERAGING PYRAMID HOLDS NUMBER TO d8 ON THE FAILING SEED: tile_avg=1, probe seed 3, 32 MB: 100.0 100.0 99.5 100.0 99.5
  99.5** (per-byte registers 100 78 51.5 52 54 55; learned merges 100 100 89.5 97 59.5 58.5; + working memory 100 100 93.5
  88.5 52.5 55.0). Bisection: the STRUCTURE works; the learned merge bodies were the bottleneck (cold start, no child
  credit). A mean keeps number linearly readable in the pair / quad tiles, and credit reaches every word underneath through
  the ring. Next: averaging as the pyramid's default; learned merges only as a zero-initialised correction on the mean.
- Fast-weight memory implemented (`tile_fw=1`, tile_fw_decay): each closing word writes v k^T (k = unit(Wk v)) into a
  decaying 32x32 matrix; every event reads r = A (Wq ws/128) through Wo; Wq / Wo learn from every read, Wk and the stored
  values from the words still in the level-0 view; the query's gradient joins the bodies' error. Constant cost per byte.
  Recall runs (seed 2, 32 MB): registers | pyramid + wm | pyramid + wm + slot attention | pyramid + fast weights.
- Pre-pretraining stage 0 (is/are, 32 MB, 64 GPU streams, the exact 20k BabyLM configuration + reg_clock=1): 96.5 97.0 96.5
  95.5 94.0 95.5 - formed. Stage 1 (one BabyLM epoch --init from it: gpu_m20pp) running. The memory v1 BabyLM run was
  restarted with tile_avg=1 (the learned-merge start aborted: cpu_m20mem.learned_merges_aborted).
- Attribution, pyramid (learned merges) probe seed 2, 32 MB: actual 100 99 78.5 91 92.5 99; NO TILE READ 88.5 57.0 51.5
  53.5 53.5 48.0 (d0 mostly survives, d1+ collapse: the TILES carry number, though this seed also forms with registers
  alone); no register read 49.5 71.0 59.0 46.5 48.5 53.0 (collapses even at d0: the registers act as a learned bias -
  cutting them shifts every logit - not as memory).
- **Recall baseline, registers only (seed 2, 32 MB): k=1 57 47 46 51 41; k=2 50 45 40 51 46** (filler 0 / 10 / 30 / 60 / 120
  words) - chance, even k=1 right after the binding sentence: with 4 scalar registers and a 5-byte view the model cannot
  recall which object went with an entity at any distance.
- **Pre-pretraining, first test: the circuit does NOT survive a pure real-text epoch.** gpu_m20pp (stage 0 formed 96.5 97
  96.5 95.5 94 95.5, then one BabyLM epoch --init): held-out 1.957 (no stage 0: 1.929); REGS_PROBE 97.3 60.3 49.3 55.3 51.0
  47.0, register r0 pinned at -100 again. Real text overwrote it (as at 80% dilution): the shared write head is repurposed.
  Next: REPLAY (20-50% curated data kept in the real-text phase; 50% preserved formation) - and the structural pyramid may
  simply be sturdier (no shared write head; word tiles held by construction): the memory v1 BabyLM run tests that.
- Recall, registers only, "no register read" arm: k=1 48 55 54 45 38 (chance, as actual).
- Recall, pyramid (LEARNED merges) + working memory, seed 2: k=1 56 49 54 48 44 - chance even for one binding: the object
  sits ~6 words before the question's end even with no filler ("... key . in the red box is the"), past the 4-word level-0
  view, in the merged tiles the learned merges fail to keep. Queued on the AVERAGING pyramid: linear reads / + slot
  attention / + fast weights. Replay runs queued on the GPU: BabyLM + ~30% curated agreement (23 MB prefixed to switchboard)
  from the pre-pretrained state and from scratch.
- Recall, learned merges + wm + SLOT ATTENTION: k=1 55 49 49 52 46 (chance). Diagnosis: a word tile does not know WHICH
  word it is - its content is the workspace at the word's end, trained to predict the next byte; after "the dogs" the next
  word depends on number (so number gets in: agreement works), after "the key" the next byte is " ." whatever the object
  (so nothing forces "key" in). `tile_id=c`: a word tile also carries a fixed +-c code hashed from the word's bytes (what a
  tokenizer's embedding gives a transformer); averaged parents hold a bag of their words' codes ("box has the key"), which a
  content-addressed read can match on the entity and read out the object. Queued (memory permitting): averaging pyramid
  (4 levels) + tile_id=1 + slot attention / + fast weights.
- **PRE-PRETRAINING + REPLAY: THE CIRCUIT SURVIVES REAL-TEXT TRAINING.** gpu_m20pp_rep (the pre-pretrained state, then one
  epoch of BabyLM with ~30% curated agreement sentences mixed in): held-out 1.941 (pure BabyLM from the same state 1.957,
  no stage 0 1.929); REGS_PROBE 100.0 90.7 81.3 83.3 86.7 84.7 (pure BabyLM from the same state: 97.3 60.3 49.3 55.3 51 47).
  A model trained on 70% real text holds number across 8 words. Caveat: the probe uses the replayed kind of sentence; the
  test that matters is transfer to natural agreement (BLiMP subject-verb agreement ~50 everywhere so far) - full sets
  queued. Replay from scratch (gpu_m20rep, no stage 0): held-out 1.921, probe pending.
- **Replay from scratch also forms: gpu_m20rep (BabyLM + ~30% curated from the start, no stage 0): REGS_PROBE 99.7 90.3
  79.7 84.7 91.0 92.7**, held-out 1.921 (word clock alone 1.929). What matters is keeping curated data in the mix, not a
  separate stage. Natural-agreement baseline (full BLiMP subject_verb_agreement): m20 49.0, m20 seed 2 49.6, word clock
  50.4, pre-pretrained without replay 48.3; the replay runs' full sets are queued (build-task/agree_phen.sh).
- **Replay keeps the circuit alive but it does NOT transfer to natural agreement.** gpu_m20pp_rep full sets: BLiMP 52.03,
  supplement 53.23 (+3.8 vs the word clock alone, maybe real), entity 17.79; subject_verb_agreement 50.6 (baseline 48-50),
  anaphor 59.6 (+6, within the seed spread), determiner_noun 51.8 (-3). The circuit holds number on the probe's sentences
  (81-91%) but those are "the NOUN FILLER is/are" with 12 nouns and only is/are; BLiMP's items use other nouns and verbs,
  relative clauses and distractors. It learned a NARROW circuit keyed to the curated vocabulary, not number as a general
  feature. => curated data must span the variety of natural language (many nouns and verbs, constructions, distractors),
  not only be dense; otherwise the circuit forms but stays in its lane.
- Recall on the AVERAGING pyramid without word identity: linear reads k=1 54 44 59 51 52, k=2 48 46 50 51 46, k=4 50 58 50 49
  54; + fast weights k=1 51 53 54 46 49 ...; + slot attention k=1 52 56 53 51 44 - all chance: averaging keeps what the
  tiles hold, and they hold no word identity. Word-identity runs (tile_id) pending.
- Strict run epoch 2: held-out (Strict set) 1.901 (epoch 1 1.922).
- **Transfer test launched:** build-task/make_agree_rich.py - curated agreement spanning natural variety (~150 nouns incl.
  irregular plurals, ~80 verbs with -s / has / does / is / was forms, pronouns, adjectives, prepositional attractors,
  subject / object relative clauses, conjoined subjects, questions, existentials; general templates, not BLiMP's) -> BabyLM
  + ~30% of it, from scratch (gpu_m20rich) -> full sets: does natural subject-verb agreement move?
- Replay from scratch (gpu_m20rep) full sets: BLiMP 52.88, supplement 52.95, entity 17.33; subject_verb 51.1, ANAPHOR 58.6,
  determiner_noun 52.9. Both replay runs agree: anaphor +5-6 over the no-curated runs (seed spread 4.5: suggestive - the
  circuit may transfer where the choice mirrors is/are, two forms keyed to the subject's number: himself / themselves) and
  supplement +3.5 (beyond its noise); subject-verb agreement unmoved. The rich-curriculum run tests varied nouns and verbs.
- Rich-curriculum replay (gpu_m20rich, BabyLM + ~30% varied agreement, from scratch): held-out 1.914 (best 20k run so far:
  word clock 1.929, narrow replay 1.921) but REGS_PROBE 97.3 61.0 49.0 51.7 54.0 51.3 - the long-range circuit did NOT form
  (narrow is/are replay: 85-92%). Varied data is harder to discover the circuit in; dense simple data forms it but keeps it
  narrow -> curriculum FORM NARROW -> BROADEN: start from the narrow-formed state and replay the rich data (the circuit is
  generalised, not discovered). Full sets pending.
- Rich replay full sets: BLiMP 52.76, supplement 52.04, entity 18.37; subject_verb 51.9, anaphor 52.5 - natural agreement
  unmoved by any curriculum so far.
- **What BLiMP subject-verb agreement actually tests (per paradigm, full set; build-task/sva_paradigms.py):** attractor via
  a PP ("A niece of most senators hasn't/haven't") 34-46% and via a relative clause ("This customer who had visited most
  children has/have") 36-48% - BELOW chance in every model: they DO track number but agree with the NEAREST noun (the
  classic attraction error; the Strict run with 10x text is the most attracted, 34%); simple plural items 50-62% (weak local
  morphology, esp. irregular plurals; the "The women meet / The woman meet" pairs differ in the noun, not the verb).
  Distances are 1-5 words: the failure is STRUCTURAL (head noun vs attractor), not long-range memory. For the goal
  (coherent talking + tasks) it marks a real gap - no notion of sentence structure - but a different project from memory.
- **Form narrow -> broaden ERASES the circuit:** gpu_m20nb (from gpu_m20rep, one epoch BabyLM + ~30% rich agreement): held-out
  1.903 (2 epochs total), REGS_PROBE 99.0 62.0 51.3 52.3 51.0 48.3 - once the narrow sentences leave the mix the register
  circuit is overwritten (as pure BabyLM erased the pre-pretrained one). Curriculum line summary: the REGISTER circuit forms
  only while dense narrow data is present, vanishes when it stops, and never transferred to natural agreement - a fragile,
  narrow mechanism. Effort moves to the structural memory (pyramid + word identity + retrieval), which holds by
  construction and is what coherence and tasks need.
- gpu_m20nb full sets: BLiMP 51.64, supplement 52.32, entity 18.49; subject_verb 51.8, anaphor 58.6 (kept from the narrow
  phase), determiner 53.1 - nothing moved by broadening. Launched: averaging pyramid on probe seeds 2 and 4 (robustness of
  the seed-3 result), with the memory arms.
- Recall with word identity (tile_id=1) + averaging pyramid (4 levels) + FAST WEIGHTS: k=1 54 51 53 56 44 - still chance.
  Why: neither side of the association exists. "key"'s tile is the workspace at the end of "key", produced by bodies that see
  ~5 bytes ("e key"): nothing in it says it belonged to the red box; and at the question the bodies see "s the ", so the cue
  does not carry "red box" either (the question's word tiles feed the readout, not the bodies or the cue). `tile_ctx=1`:
  CONTEXT-BEARING tiles and cues - a word tile and every attention / fast-weight cue are built from the full context vector
  (ws/128 + what the tiles were read into it), so "key" is stored as "key after red box has the" (fast-weight key from
  context, value with the object's identity) and the question can cue it. Running: ctx + fast weights / ctx + attention.
- **Memory v1 on BabyLM (cpu_m20mem: averaging pyramid 3 levels + working memory 8 + slot attention, CPU trainer, 1 epoch):**
  held-out 2.011 (CPU single-stream runs sit higher than GPU 64-stream ones at one epoch: tree 2.12-2.19, so not directly
  comparable with the 1.93-1.97 of the GPU 20k runs); REGS_PROBE 98.3 64.7 50.0 58.7 55.7 50.7 - no long-range number circuit
  on real text: the pyramid HOLDS by structure (the probe), but real text does not train the tiles to carry number (the same
  wall as the registers). Ablation (tiles read as 0) and REACH pending.
- **THE PYRAMID IS LOAD-BEARING ON REAL TEXT:** cpu_m20mem held-out 2.011; tiles read as 0 -> 2.326 (+0.316 bits/byte);
  registers read as 0 -> 2.035 (+0.024). The model relies on the pyramid's context far more than it ever used the registers
  (+0.02-0.07 elsewhere) - the tiles carry information that matters for prediction, though not the subject's number.
  REACH (does the prompt's influence extend beyond ~4 words?) running.
- **THE PYRAMID EXTENDS CONTEXT ON REAL TEXT (REACH).** cpu_m20mem vs gpu_m20 (same 20k motifs, no pyramid), 20 prompts x 200
  bytes, share of outputs whose byte log-prob the prompt still changes (> 0.01 nats), by output position:
  | | 16-31 | 32-63 | 64-127 | 128-255 | reach (bytes) |
  |---|---|---|---|---|---|
  | no pyramid, greedy | 7% | 2% | 2% | 3% | ~48 |
  | PYRAMID, greedy | 52% | 35% | 17% | 13% | 157 |
  | no pyramid, sample | 1% | 1% | 0% | 0% | ~48 |
  | PYRAMID, sample | 56% | 46% | 27% | 26% | 163 |
  The prompt's gain in log-prob per byte 0.07 -> 0.09-0.12. With the tiles read as 0 held-out costs +0.32 bits/byte: the
  context is carried by the pyramid. First Coco model whose prompt keeps shaping its output to the end of 200 bytes - the
  capability coherent talking needs. (It does not yet produce subject-verb agreement: that is structure, not span.)
- **Generation on the pyramid model (cpu_m20mem vs gpu_m20, frontier eval, 20 prompts x 200 bytes):** own-output log-prob
  per byte greedy / sample / frontier -0.38 / -0.89 / -0.28 (no pyramid -0.60 / -1.16 / -0.45); valid words (sample) 95.4%
  (92.4); median reach 180 / 199 / 174 bytes (48 / 48 / 14); ident@1 of the SECOND HALF still 0/20; the first output line is
  much shorter (13-27 bytes vs 42-71: the model ends the prompt's line and jumps into BabyLM's dominant genre, CHILDES /
  subtitle dialogue); greedy loops 7/20 (1/20). The pyramid supplies REACH and confidence; COHERENCE also needs text that
  stays on topic (BabyLM is mostly short dialogue lines) and/or a decoder objective that rewards it (the frontier judge can
  use the prompt's now-persistent influence).
- TileState refactor (organism): the pyramid's per-stream state is a TileState; tiles_read / tiles_attend / tiles_learn /
  tiles_close act on it (Organism on tst_; the multi-stream trainer will keep one per stream); ring credit through a callback;
  the attention + fast-weight query gradients now both use the readout's own error (fix). Reproduces the pre-refactor tiles
  smoke exactly (100.0 98.0 85.0 74.0 53.5 45.5).
- **The averaging pyramid holds number to d8 on every probe seed tried:** tile_avg=1 (3 levels), 32 MB: seed 2 94.5 98.5 94.0
  93.5 92.0 90.0; seed 4 100.0 100.0 100.0 100.0 100.0 99.5; seed 3 (earlier) 100.0 100.0 99.5 100.0 99.5 99.5. The structural
  memory is seed-robust where the register circuit formed on ~40% of seeds. Memory arms: NO TILE READ seed 2 73.5 72.0
  69.0 68.0 69.5 68.0 (-20 to -25: the tiles carry much of it), seed 4 93.0 92.5 92.0 92.5 90.5 88.0 (-7 to -12: seed 4's
  registers also hold number, as ~40% of seeds do without tiles); no register read ~chance on both (49.5-67.5, collapsing
  at d0 too: the register-bias effect seen before, not a memory reading).
- Recall with word identity + slot attention (tile_id=1, no context): k=1 48 48 55 51 57 - chance, as with fast weights (the
  context-bearing variants, tile_ctx=1, are running).
- **Pyramid on the GPU trainer:** tiles (averaging merges) now run in the multi-stream trainer: per-stream TileState,
  shared mixings stepped with the readout, word-tile credit as ring injections (CPU reference and CUDA kernel). ms_check
  exact 524,288 events bit-identical to the organism for memory v1 (tiles=3 wm 8 attention), tiles=4 + identity + context +
  fast weights, and tiles=4 + identity + context + attention + wm; regressions pass. Required one organism fix: the
  readout's Adam step ran before tiles_learn on its step event (every 512 events), splitting that event's tile gradient
  from its minibatch - now after it (tiles-on configurations only). Dual (GPU learner vs CPU reference): S=1 1M steps 165
  of 534M nibbles differ (3e-7), ring rel diff 2.5e-8; S=16 6,880 of 1.08G (6.4e-6), shadow mean diff 6e-4 (control
  without tiles running).
- Recall with CONTEXT-bearing tiles + identity + slot attention (tile_ctx=1, 4 levels, seed 2): k=1 56 55 48 49 55, k=2 49 60
  52 48 50, k=4 48 52 41 54 44 - still chance. Next: instrument the question position (does attention land on the object's
  tile at all?) before adding mechanism.
- **G, first test - the depth ladder (`cpp/apps/depth_probe.cpp`):** x in 0..15 and a program of k ops (+1, x3, xor 10,
  rotate) in ONE event's input; a chain of H+1 INT4 bodies (exact backprop, float shadows + Adam), each hop reading the
  input, the previous hop's message at L3 and the workspace so far; accuracy by k (chance 6.25%), 4M examples:
  | bodies | k=1 | k=2 | k=3 | k=4 | k=5 |
  |---|---|---|---|---|---|
  | 1 | 85 | 73.6 | 22 | 12.8 | 8 |
  | 2 | 100 | 97.2 | 48.6 | 15.6 | 10 |
  | 4 | 100 | 94.2 | 48.4 | 16.6 | 15.2 |
  | 8 | 100 | 88.4 | 59.6 | 26.2 | 17.8 |
  Depth helps a little, but bodies do not become one-op-per-hop steps. Bisection so far: default shifts saturate 75-90% of
  cells (sh=3 better: 3 bodies trained on k<=3 reach k=3 69.8%); a last-hop readout does not help; and a finding about the
  organism's wiring: **motif_wire's message is the sender's L2, which reads only the sender's own L1 - the incoming message
  enters at L3 and never reaches the outgoing L2, so a chain of L2 messages cannot compose (depth per byte stays ~1 layer
  beyond one body however many motifs are wired).** In the harness, composing messages (L3 / L5 / L6) do not beat L2 with a
  summed readout either (6 bodies: L2 k=3 85%, L5 77%): chains learn as wide ensembles, not serial programs. Running: TAPE
  mode (hop h sees only op h, x only at hop 0: the value must flow through the chain), per-hop and SHARED bodies.
- **Depth ladder, TAPE mode (hop h sees only op h; x only at hop 0), sh=3, 4M examples, trained on k <= bodies:**
  2 bodies solve 2-op programs perfectly (100/100, 0.009 bits); 4 bodies 100/98.8/86/34.4; 6 bodies 100/93.6/76.2/31.4;
  8 bodies 97/97/70.2/25.4; a SHARED body (one body reused at every hop) is worse (4: 100/91.6/57.2/22). Chains learn ~3
  serial steps. Cause found: the workspace a hop reads is the SUM of the earlier hops' L6, which grows with depth and clamps.
  Read modes (6 bodies, k=1..6):
  | what hop h reads | k=1 | k=2 | k=3 | k=4 | k=5 | k=6 |
  |---|---|---|---|---|---|---|
  | sum of earlier L6 (as before) | 100 | 93.6 | 76.2 | 31.4 | 15.2 | 11 |
  | **MEAN of earlier L6** | **100** | **99** | **90.2** | **62.2** | **31** | 16.6 |
  | previous hop's L6 only (a pure chain) | 4.4 | 6.4 | 5.4 | 6.2 | 5.6 | 7 |
  | no workspace; previous hop's L6 as the L3 message | 98 | 92.6 | 68 | 46.6 | 32.4 | 22.4 |
  A normalised residual stream (the mean) composes; a pure chain does not train at 6 bodies (at 4 it reaches k=3 60%) -
  the deep-learning lessons (skip paths + normalisation) hold for INT4 bodies. For the organism: its per-byte bodies run in
  parallel on the same input and motif_wire's L2 message cannot compose; a per-byte normalised residual stream between
  motif bodies (or an L6 message) is what serial composition needs. Running: mean + L6 message, role-code pools, a shared
  body, 8 bodies, longer training, and the original (non-tape) ladder with the fix.
- **Memory v1 on the GPU trainer (gpu_m20mem: gpu_m20 + averaging pyramid 3 levels + wm 8 + slot attention, 64 streams,
  1 epoch, 520 s):** held-out 1.982 (gpu_m20 1.966, seed 2 1.968: the pyramid does NOT lower next-byte loss at one epoch,
  +0.016); tiles read as 0 -> 2.334 (+0.35: load-bearing, as on the CPU); REGS_PROBE 97.0 60.7 50.0 55.3 52.0 48.7 (no
  long-range number circuit, as everywhere on real text). REACH (share of outputs the prompt still moves, by position
  16-31 / 32-63 / 64-127 / 128-255): greedy 59 / 46 / 26 / 22% (CPU pyramid 52 / 35 / 17 / 13, no pyramid 7 / 2 / 2 / 3); sample
  70 / 55 / 40 / 33%. Generation (20 prompts x 200 bytes): greedy 0/20 loops (CPU pyramid 7/20), first line 101 bytes (CPU
  pyramid 27, no pyramid 69: far less genre-hopping), topic words 0.184 and in the SECOND half 0.158 (CPU pyramid 0.053, no
  pyramid 0.053), ident@1 late 1/20, own log-prob -0.605 (no pyramid -0.607). The 64-stream pyramid model stays on its
  prompt's topic into the second half for the first time. Full sets scoring.
- **Depth ladder: a SHARED body with the normalised stream gives the clean ladder.** Tape mode, 6 hops (entry body + one
  body reused 5 times), each hop reading the mean of the earlier hops' L6: 96.6 97.0 90.8 84.4 69.6 48.2 (k=1..6); truncated
  after 5 hops k=6 falls to 7% - each hop adds one op. Per-hop bodies: 100 99 90 62 31 17; role-code pools (8 per hop): 100
  93.8 50.4 12.2 (routing fragments the data). Other variants (6 bodies): stream at L3 message cells (mean of earlier L6)
  100 98.2 97.4 71.6 38.4 21.6 (tape) / 100 97.8 89.4 47.2 24.2 15.6 (every body sees all ops); stream at both L1 and L3 worse.
  Depth that composes comes from ITERATING one circuit over a normalised residual stream, not from recruiting different
  bodies by role.
- **Recall diagnosed (recall_probe DIAG=1; context tiles + identity + slot attention, 8 MB, k=1):** at the object's close its
  word tile holds its identity - 100% linearly decodable (12 objects, chance 8.3%); at the question the slot holding it still
  does (d=0 100%, d=10 81%, d=30 47% of the 89 stories where it is still in the view). But ATTENTION AVOIDS IT: weight on the
  object's slot 0.005 vs 0.172 uniform (d=0), argmax on it 2/450; p(first letter) 0.21 (base ~0.12). Storage and holding
  work; RETRIEVAL is what fails - the query / key maps learned from next-byte error put the weight on the recent words (what
  predicts most bytes) and never learn the entity match (one answer event per story). 32 MB runs pending.
- **Item B for the pyramid: skills up, no regressions (full sets, epoch 1).** gpu_m20mem BLiMP 53.29 / supplement 50.93 /
  entity 17.75 vs gpu_m20 seeds 1 and 2: 52.52 / 48.59 / 17.83 and 52.19 / 49.61 / 17.85. Per BLiMP phenomenon against the
  two seeds' mean (flagged when beyond max(seed spread, 3)): quantifiers +10.9 (54.8 vs 41.2 / 46.5), anaphor +5.1 (57.4 vs
  54.5 / 50.0), npi +2.9; nothing falls beyond the spread (worst irregular_forms -1.5 within its 6.9 spread, islands -1.2,
  filler-gap -0.8); subject-verb agreement unchanged 49.1. With the reach and topic results above, memory v1 (the averaging
  pyramid + working memory + slot attention) is the first mechanism that improves the skills, the reach and the
  generation's topicality together, at a small next-byte cost (+0.016 bits/byte).
- **G: ONE BODY ITERATED 8 TIMES COMPUTES 8-STEP PROGRAMS.** Tape mode, entry body + one shared body run 7 more hops, each
  reading the mean of the earlier hops' L6, 8M examples: accuracy by k 92.8 92.8 97.0 94.4 90.0 80.2 62.0 37.6 (chance 6.25).
  Cutting the chain after d+1 hops leaves exactly the programs longer than d+1 at chance (d=3: k4 79 / k5 10; d=4: k5 74 /
  k6 9; d=5: k6 65 / k7 8; d=6: k7 51 / k8 10) - every hop adds one serial step. Limits: (a) running MORE hops than trained
  generalises only weakly (trained 6 hops on k<=6: a 7th hop takes k=7 from 10 to 20%); (b) a shared body needs a per-hop
  cue - with every op visible at every hop it cannot tell which step it is on (k=3 44%), while per-hop bodies manage (88%);
  (c) seed variance is large (seed 2, 6 hops: 97 92 74 55 31 15 vs 97 97 91 84 70 48); (d) a mean-reading readout is worse.
  Per-hop bodies with 4x the training (16M) also reach a clean ladder (98 95 88 78 74 58).
- motif_stream implemented in the organism, the trainer, the CPU reference learner and the CUDA learner (one round kernel
  per chain position). ms_check exact: 1,048,576 events bit-identical (motifs, stream) and 524,288 with memory v1 tiles.
  One organism fix on the way: a silent body (reached L6, did not emit) had its gradient cut at L5 by the settle-depth rule;
  with the stream its L6 carries the later bodies' error, so its depth now counts as reached.
- Recall diagnostic at 32 MB (4x the training): attention on the object's slot 0.066 (tile_id=1) / 0.141 (tile_id=2) vs 0.172
  uniform at d=0, argmax on it 30/450 / 12/450 - more training reduces the avoidance but attention never prefers the bound
  object; identity stays 100% decodable from its slot. Running: `tile_ret=128` - FIXED-KEY retrieval (organism only): an
  episodic buffer of the last 128 word tiles, each keyed at close by the frequency-weighted bag (1/sqrt(1 + running
  frequency)) of hashed codes of the 4 words before it; every event's query is the same bag over the last 4 closed words;
  softmax(10 x cosine) over the buffer, read through a learned output map (the only thing that learns to use it).
- Generation samples on the pyramid model are locally fluent word salad ("a girl with the sweets then the third and a
  fortune had been the things that people who can..."): reach and topicality moved, coherence did not - at ~2 bits/byte the
  model is far from talking-quality text; coherence will need a much better model, not a decoder trick.
- **motif_stream on BabyLM (gpu_m20s, 64 streams, 1 epoch): held-out 2.032 vs 1.966 unwired (seed 2 1.968) - WORSE by
  0.066.** (There is no finished motif_wire BabyLM run to compare: gpu_m20w was stopped.) Open: is it the 64-stream
  ordering (wired chains already failed to form at S=64 on the probe while forming at S<=16) or the mechanism on real
  text? Queued: the same at 16 streams; gpu_m20mems (pyramid + stream) and full sets pending.
- **Item F on the pyramid model (gpu_m20mem, conformance suite, fast item sets for the skill arms):** C1 PASS 0.689 (bodies
  over the sequence memory; gpu_m20 1.353 - the pyramid now carries part of what the bodies did), C2 PASS 0.148 (0.132),
  C3.L1/L2/L3/L5 PASS, C10 PASS 0.017 (0.038); C7 FAIL as before (no early deaths: fixed motifs never exit early - the check
  does not apply to motif models); C8 informational (saturation). Skill arms: S1.blimp PASS +1.34 (bodies needed for the
  skill; gpu_m20 -0.97), S2.blimp_supplement PASS +4.0 (context computation needed; gpu_m20 0.0); S2.blimp -0.07, S1 / S2
  entity and S1 supplement fail - 2 of 6 skill arms pass where gpu_m20 passed none.
- **RECALL ABOVE CHANCE FOR THE FIRST TIME - fixed-key retrieval (tile_ret=128, bag of the 4 previous words, 8 MB, seed 2):**
  k=1 68.0 73.0 67.0 71.0 63.0 (filler 0 / 10 / 30 / 60 / 120 words; every earlier variant ~50). DIAG: retrieval weight on the
  object's item 0.21 / 0.27 / 0.30 at d=0 / 10 / 30 vs 0.008 uniform, argmax on it 255 / 246 / 259 of 450 - the addressing
  works at every span inside the buffer, without any learned query. It shares the weight because a bag of context words
  cannot tell the object from the words just before it ("has" and "the" also have red / box / the in their 4 previous
  words); with 3 context words the object is never picked (0.002). Next: POSITIONAL codes (tile_ret_pos=1: a word's code
  depends on how many words back it is), so the item that FOLLOWED a context like the question's matches - an induction
  match with fixed, frequency-weighted keys. Running.
- Wide readout (ws_split: each motif's emission also read through its own weights), CPU smoke 4 MB: held-out 2.2099 vs
  2.2229 (-0.013). 8 motifs per byte (contexts 1..12 bytes) with / without it running.
- Pyramid + stream (gpu_m20mems): held-out 1.993 (pyramid alone 1.982: the stream costs +0.011 with the pyramid, +0.066
  without); registers read as 0 +0.063, tiles read as 0 +0.325. 16-stream stream / unwired pair running.
- **THE 32-DIM SUMMED WORKSPACE IS A BOTTLENECK (CPU smokes, 4 MB, one epoch, held-out):**
  | bodies per byte | shared 32-dim readout | + per-motif readout (ws_split) |
  |---|---|---|
  | 4 motifs (20k units) | 2.2229 | 2.2099 |
  | 8 motifs (contexts 1..12 bytes, 53k units) | 2.2719 (worse than 4) | **2.2073** |
  With every body summed into the same 32 numbers, more bodies per byte HURT (they interfere); read through their own
  channels, 8 motifs are the best of the four. This fits "20k matches 82k": capacity was never the limit, the readout's
  width was. Next: ws_split in the trainer and both learners (per-motif output errors), then 8 motifs + wide readout on
  the GPU at full scale.
- Positional retrieval keys, first try: weight on the object 0.000 - an off-by-one, not the idea: the query was built
  before the word ending at the current byte is closed, so at the answer every position was shifted against the stored
  key (the order-free bag hid it). Fixed: the query puts a word ending at this byte first. Rerunning.
- **Positional context keys work: retrieval lands on the bound object.** tile_ret=128 with tile_ret_pos=1 (after the
  alignment fix), k=1, 8 MB: weight on the object 0.569 / 0.754 / 0.881 at d=0 / 10 / 30 (uniform 0.008), argmax on it 324 /
  384 / 431 of 450 (72 / 85 / 96%); p(the object's first letter) 0.26 / 0.33 / 0.36 (bag keys 0.19-0.22, attention ~0.14): the
  readout has begun to use what is retrieved. An induction match with FIXED keys: nothing learned the addressing; only the
  output map learns what to do with the item. Recall scoring (k=1/2/4 x filler) running.
- motif_stream at 16 streams (gpu_m20s16): held-out 1.958 (the stream at 64: 2.032; gpu_m20 unwired at 64: 1.966). The
  unwired 16-stream control is next - if it is ~1.96, the 64-stream cost was the trainer's ordering, not the mechanism.
- **BIND-AND-RECALL SOLVED WITHIN THE BUFFER: positional fixed-key retrieval (tile_ret=128, tile_ret_pos=1, 8 MB, seed 2).**
  % correct object preferred over a distractor, filler 0 / 10 / 30 / 60 / 120 words:
  | bindings | 0 | 10 | 30 | 60 | 120 |
  |---|---|---|---|---|---|
  | k=1 | 93 | 96 | 93 | 96 | 70 |
  | k=2 | 89 | 94 | 94 | 93 | 60 |
  | k=4 | 90 | 93 | 90 | 89 | 51 |
  (bag keys at 32 MB: k=1 73 75 81 77 58; every learned-attention / fast-weight variant ~50). Up to 4 bindings recalled at
  ~90% across 60 words of filler; at 120 filler words the object has left the 128-word buffer (with k=4 the binding
  sentences add ~24 words) and recall falls to chance - the capacity limit is the buffer, by design. Constant cost per
  byte (128 x 128-dim cosines), no KV cache. The working memory tier finally RETRIEVES: fixed, frequency-weighted,
  position-coded context keys (an induction match), and only the output map learns.
- **motif_stream vs unwired, same trainer settings (BabyLM, 1 epoch, held-out):**
  | streams | unwired | motif_stream | cost |
  |---|---|---|---|
  | 16 | 1.943 | 1.958 | +0.015 |
  | 64 | 1.966 | 2.032 | +0.066 |
  The 64-stream ordering multiplies the cost ~4x, but the stream costs next-byte loss even at 16 streams: serial
  composition inside a byte is not what real text is short of at this scale (the readout's width is - see the smokes).
  (Also: 16 streams train better per epoch than 64 - 1.943 vs 1.966.) Full sets for both queued (the 64-stream stream run
  had BLiMP 53.08 vs 52.52 / 52.19).
- **motif_stream full sets:** 16 streams - unwired 52.26 / 50.08 / 17.47, stream 51.71 / 50.53 / 16.80 (BLiMP / supplement /
  entity); 64 streams - stream 53.08 / 49.64 / 17.49 vs unwired seeds 52.52 / 48.59 / 17.83 and 52.19 / 49.61 / 17.85. No
  consistent skill gain (the 64-stream BLiMP +0.7 does not replicate at 16), and it costs next-byte loss at both. Verdict
  for G on real text: the composing chain is correct (exact, dual) and composes in the harness, but byte-level text at this
  scale does not use per-byte serial depth; it stays OFF by default. G's open part is growth (depth / breadth on demand),
  which needs a task that uses depth - the harness ladder has one; real text does not show it yet.
- Retrieval on BabyLM (gpu_m20ret = memory v1 + word identity + tile_ret=128 positional): held-out 1.999 (memory v1 1.982;
  identity and retrieval are confounded here). Full sets pending. Recall with a 512-word buffer, no recency: k=1 62 70 67 72
  77, k=2 66 78 66 66 70, k=4 73 67 58 73 71 - the span reaches 120 filler words (77%) but older bindings of the same
  entities interfere at short spans (the 128 buffer: 93-96). Recency runs pending.
- Retrieval on BabyLM (gpu_m20ret, memory v1 + identity + tile_ret=128): full sets BLiMP 52.90 / supplement 48.28 / entity
  17.59 vs memory v1 53.29 / 50.93 / 17.75 - no skill gain (BLiMP items are single sentences: nothing to retrieve;
  entity tracking needs state updates, not recall). tiles read as 0 +0.28 bits/byte, registers read as 0 +0.07.
- Two heads (positional + bag) + recency 2, 128 items: recall k=1 93 94 93 96 69, k=2 90 90 86 93 66 (positional alone 93 96
  93 96 70 / 89 94 94 93 60); DIAG weight on the object 0.70 / 0.83 / 0.91 at d=0/10/30 (positional alone 0.57 / 0.75 / 0.88),
  p(first letter) 0.30 / 0.36 / 0.39. 512 items + recency 2: weight 0.32 / 0.52 / 0.65 (no recency 0.18 / 0.36 / 0.51) -
  recency helps but 2/512 per rank is too weak against older same-entity bindings. bAbI multiple choice running.
- **THE WIDE READOUT PAYS AT FULL SCALE: gpu_m20sp (gpu_m20 + ws_split, 64 streams, 1 epoch): held-out 1.916 vs 1.966
  (-0.050 bits/byte)** - the largest single held-out gain in this line of work, for zero extra bodies and V x M x 32 more
  readout weights. The 32-dim summed workspace was the bottleneck. 8 motifs with / without the wide readout running;
  full sets queued.
- 512-item retrieval buffer + recency 2, k=4: 81 79 72 80 87 (filler 0..120) - with recency the span grows with the buffer.
- bAbI multiple choice (babi_probe, 1 pass over the 7 MB bAbI dialogues, 20 questions per task), motif organism + memory
  v1 tiles + identity, NO retrieval: 34.5% overall (chance 25.0); qa1 20 (16.7), qa2 25, qa7 70 (33.3), qa18 80 (50),
  qa20 60 (14.3). Retrieval variants running.
- **bAbI with retrieval: marginal.** 1 pass, 20 questions per task: no retrieval 34.5%, positional retrieval 35.8%, two heads +
  recency 4 36.5% (chance 25.0; ~1 question = 0.25 points). Why the recall probe's success does not carry over: bAbI asks
  "Where is Mary?" in a form whose word POSITIONS do not match the fact ("Mary moved to the bathroom"), so only the bag head
  can match, and it matches EVERY sentence about Mary - the read is a blend of all the places she has been, and a weak
  recency prior (4 over 128 items) does not pick the latest. bAbI's tiny vocabulary also flattens the frequency weights.
  Running: strong recency (16, 64) with 2 passes, against a 2-pass control.
- **8 motifs + wide readout (gpu_m8sp, contexts 1..12 bytes, 53,504 units, 1 epoch): held-out 1.885** (4 motifs + wide
  1.916, gpu_m20 1.966: -0.081 in all). After ONE epoch it beats the 82k 4-motif model's third epoch (1.894). NOTE: 53.5k
  bodies exceeds the user's 20k size guideline (keep ~20k until multi-body circuits show) - run as a diagnostic of
  per-byte computation (more bodies PER BYTE, which the wide readout makes useful), not adopted; the 20k result (-0.050)
  is within the guideline. 8 motifs with the shared readout running (the smoke said worse than 4).
- **CONFIRMED AT FULL SCALE (BabyLM, 64 streams, 1 epoch, held-out bits/byte):**
  | | 4 motifs (20k) | 8 motifs (53.5k) |
  |---|---|---|
  | shared 32-dim readout (summed workspace) | 1.966 | 2.026 |
  | per-motif readout channels (ws_split) | 1.916 | **1.885** |
  With every body summed into one 32-dim workspace, doubling the bodies per byte HURTS (+0.060); read through their own
  channels, it helps (-0.031), and the wide readout alone is -0.050 at 20k. This is why "20k matched 82k" and why more
  bodies never paid: the readout's 32 inputs were the bottleneck, not capacity. Full sets queued; wide readout + memory v1
  (20k) running.
- Wide readout full sets (gpu_m20sp): BLiMP 52.48 (seeds 52.52 / 52.19), supplement 51.58 (48.59 / 49.61: +2.5), entity 17.54
  (17.83 / 17.85) - the -0.050 bits/byte comes with no skill regression beyond the spread and a supplement gain.
- bAbI, 2 passes, no retrieval: 38.2% (1 pass 34.5%). Strong-recency retrieval runs pending.
- 8 motifs + wide readout full sets (gpu_m8sp): BLiMP 52.81, supplement 51.24, entity 17.83 (gpu_m20 52.52 / 48.59 / 17.83) -
  -0.081 bits/byte with skills flat or up.
- **bAbI with strong recency: worse on qa1 (2 passes: no retrieval 38.2%, recency 16 36.2%, recency 64 37.5%; qa1 25 / 15 /
  5%).** The drop exposed SELF-MATCHING: the question's own words sit in the buffer and their bag keys share most of the
  query ("Where is Mary" -> the items "is", "Mary", "ASSISTANT" have keys built from those same words), so the bag head
  retrieves the question itself - and recency favours it, being newest. (The positional head is immune: shifted positions.)
  Fix: `tile_ret_skip` - the newest items, inside the query's own word window, cannot be retrieved ("don't retrieve the
  cue"). Running on bAbI (2 passes, two heads, with / without recency) and on recall (no harm expected: at d=0 the
  object is 7 words back).
- **Wide readout + memory v1 (gpu_m20spmem, 20k): held-out 1.913** (wide alone 1.916, memory v1 alone 1.982); tiles read as 0
  +0.205 (alone +0.35: the wide readout now carries part of it), registers read as 0 +0.109. Generation (20 prompts x 200
  bytes) mixed vs memory v1 alone: greedy loops 8/20 (0/20), second-half topic words 0.026 (0.158), first line 124 bytes
  (101), ident@1 late 3/20 (1/20), reach median 198 (189). Full sets pending.
- **Full sets, best model so far - wide readout + memory v1 at 20k (gpu_m20spmem): BLiMP 53.64, supplement 52.36, entity
  17.71** (gpu_m20 52.52 / 48.59 / 17.83; memory v1 53.29 / 50.93 / 17.75; wide alone 52.48 / 51.58 / 17.54). Per BLiMP
  phenomenon vs the two gpu_m20 seeds: quantifiers +9.0, anaphor +4.9, determiner-noun +4.8, binding +2.7, irregular +2.7;
  s-selection -3.1 (flag bar 3.0: marginal, within the ±7 per-phenomenon seed variance seen before), islands -2.7,
  subject-verb agreement unchanged 49.0. 8 motifs with the shared readout: 52.17 / 48.75 / 17.42 (loss and skills worse);
  8 motifs wide 52.81 / 51.24 / 17.83.
- Recall with cue exclusion (two heads, skip 4, recency 2): k=1 97 97 94 97 73 (without skip 93 94 93 96 69) - no harm.
- **bAbI with cue exclusion: still no gain** (2 passes: no retrieval 38.2%, skip 37.5%, skip + recency 16 37.0%; 20
  questions per task, ~1 question = 0.25 points overall). Fixed-key retrieval works where the question echoes the fact's
  phrasing (the recall probe); bAbI's reformulated questions and overwritten facts ("Where is Mary?" after several moves)
  need more than a fixed match - a read that knows WHICH word of the matched sentence answers (the location, not "moved
  to the") and that facts about an entity supersede each other. Parked; retrieval stays an option (off by default).
- REGS_PROBE on the wide-readout models: gpu_m20sp 96.3 59.7 47.3 53.0 51.3 48.3, gpu_m20spmem 97.3 60.3 49.3 53.3 50.7 49.3 -
  the wide readout does not change item A (no long-range number circuit on real text, as in every run).
- **Best 20k config trained longer (gpu_best4: wide readout + memory v1, 64 streams): held-out epoch 1 1.902, epoch 2
  1.830** - below the old 82k model's best (1.842 at epoch 8). Epochs 3-4 and full sets per epoch running.
- `tile_split` (the tiles' reads get their own readout channel too, V x 32, zero at start; organism only so far), CPU smoke 4
  MB on top of wide readout + memory v1: held-out 2.1992 vs 2.2153 (-0.016). The readout-width lesson applies to the tiles.
- **Best 20k configuration (wide readout + memory v1), 4 epochs: held-out 1.902 / 1.830 / 1.793 / 1.770** - still falling at
  epoch 4; the old 82k model plateaued at 1.842 (epoch 8, 1.850 at 9). 0.07 better than that plateau with a quarter of the
  bodies. Full sets per epoch and generation on epoch 4 running.
- **Generation on the best 20k model after 4 epochs (1.770 bits/byte):** locally far more fluent, encyclopedic ("is a list of
  famous country musicians from Florida, USA. The following other musicians had at least one number one song or album on
  country music charts: ..."), own log-prob greedy -0.54 / frontier -0.33 (1 epoch: -0.62 / -0.44); the frontier decoder
  copies 32% of 32-byte windows from training; greedy loops 9/20; but prompt-topic words fall to ~0 (1 epoch 0.105) - the
  model now writes BabyLM's Simple-Wikipedia register whatever the prompt. Lower loss bought fluency and memorisation, not
  coherence with the prompt: H needs the prompt's influence to survive the genre prior (a decoder that rewards the prompt's
  persistent influence, or conditioning the model has to learn - e.g. dialogue data where replies depend on the turn).
- **Best 20k configuration, full sets per epoch (BLiMP / supplement / entity):** e1 52.71 / 52.54 / 17.59, **e2 55.07** / 52.98 /
  17.86, e3 54.04 / 53.13 / **18.57**, e4 53.68 / **53.21** / 17.76 (gpu_m20 e1 52.52 / 48.59 / 17.83). BLiMP 55.07 at epoch 2
  is the highest in this project; after it BLiMP falls while held-out keeps improving (1.830 -> 1.770) and generation
  copies more from training - the later epochs fit the corpus rather than the grammar. Epoch 2 is the skill checkpoint.
- **6 motifs at the SAME size (20,736 bodies: 256 + 5 x 4096) + wide readout: held-out 1.914** vs 4 motifs at 20,736 + wide
  1.916 - more bodies per byte at fixed size is neutral. The 8-motif gain (1.885) came with 2.6x the bodies: with the wide
  readout, capacity now pays (it did not through the 32-dim sum - "20k matched 82k"). Whether to grow the model is the
  user's call (20k guideline); not run further.
- tile_split at full scale (gpu_ts: wide readout + memory v1 + the tiles' own channel, 1 epoch): held-out 1.928 vs 1.913 /
  1.902 for two runs of the same configuration without it (run-to-run spread ~0.011 with --overlap) - the smoke's -0.016 does
  not transfer; off. 6 motifs at 20k full sets 52.84 / 51.32 / 17.75 (same as 4).
- **PMI judge on the fluent 4-epoch model (frontier decoder, pmi = weight on -log p(continuation | no prompt)):** pmi 0 topic
  words 0.053 / 2nd half 0.000 / copied 0.32 / own log-prob -0.33; pmi 0.5: 0.105 / 0.000 / 0.29 / -0.37; pmi 1: 0.079 / 0.079 /
  0.00 / -1.36 (late gain 0.023, ident@1 late 6/20 - the prompt's influence survives to the end, but the text turns to word
  salad: "a good man. She is is a live on 1 O Irest as the fall in its bar..."). A decoding-time trade between fluency and
  prompt dependence; neither end is coherent AND on-topic. The model itself must learn to condition on its prompt.
- tile_split full sets 53.26 / 51.34 / 17.70 - inside the two same-configuration runs without it (53.64 / 52.36 / 17.71 and
  52.71 / 52.54 / 17.59; one-epoch BLiMP run-to-run noise ~1 point). Neutral on skills, slightly worse on loss: off.
  A second seed of the best configuration (2 epochs) is queued to test BLiMP ~55 at epoch 2.
- **G: DEPTH BEYOND TRAINING - an iterated body trained on RANDOM chain lengths runs longer programs than it ever saw.**
  depth_probe, tape mode, one shared body over the mean stream, trained with chains of random length (>= the program's; at
  most 6 hops) on programs of k <= 6 ops, 8M examples (`randdepth=1`). Accuracy by k (chance 6.25), by hops run:
  | hops run | k=1 | k=2 | k=3 | k=4 | k=5 | k=6 | k=7 | k=8 |
  |---|---|---|---|---|---|---|---|---|
  | 4 | 100 | 99.6 | 100 | 99.8 | 7.4 | 17.8 | 6.4 | 8 |
  | 5 | 100 | 99.6 | 100 | 99.8 | 98.6 | 5.6 | 17 | 5.4 |
  | 6 (trained max) | 100 | 99.6 | 99.8 | 99.8 | 99 | 91.8 | 5.8 | 16.6 |
  | 7 (beyond) | 100 | 99 | 99.6 | 98.4 | 98 | 91.2 | **82.8** | 7 |
  | 8 (beyond) | 97.4 | 99.2 | 91.2 | 97.6 | 85 | 91.6 | **83** | **67.4** |
  Fixed-depth training (always 6 hops): 93 95 82 72 60 30 within range, k=7 20% with a 7th hop. With random depth the body
  learns ONE step it can repeat: capability grows with the number of iterations, past the trained depth, with no new
  bodies - the unbounded axis G asks for. Trained on k <= 4 only (random depth <= 6 hops): k=5 71.8%, k=6 38.6% (lengths
  never seen) - it extrapolates in program length too, decaying. Training directly on 8 hops / k <= 8 failed to learn at
  all (75 33 19 ...): short-then-extrapolate beats learning long from scratch.
- G follow-ups (depth_probe, randdepth=1): WITHOUT a tape (every op visible at every hop) random depth does not help - 100 86
  44 15 (fixed depth 99 95 44 24): the shared body cannot tell which step it is on; the per-hop cue is essential (on real
  text it would have to come from time, the next byte, not from within a byte). Replications of the extrapolation: seed 2
  (8M) within range 91 90 93 92 89 73, k=7 60% (7 hops), k=8 37% (8 hops); seed 1 at 16M (!) 78 80 74 72 66 52, k=7 42%,
  k=8 28%; seed 1 at 8M k=7 83%, k=8 67%. The direction replicates (fixed-depth training: ~20% at k=7), the size varies a
  lot and longer training did NOT help - optimisation of the iterated body is unstable.
- Annealing test (continue the best 20k config from epoch 2 for 2 epochs, --init: optimiser moments restart): lr x0.3 held-out
  1.796 / 1.795; same lr 1.8xx / 1.814; uninterrupted 1.793 / 1.770 - the restart itself costs loss; annealing partly
  offsets it. Full sets pending.
- **Loss and skills part ways after epoch 2 (best 20k config; all three continue from the same epoch-2 state):**
  | epoch 4 | held-out | BLiMP | supplement | entity |
  |---|---|---|---|---|
  | uninterrupted | 1.770 | 53.68 | 53.21 | 17.76 |
  | restart (--init: optimiser moments reset), same lr | 1.814 | **56.06** | 55.02 | **18.24** |
  | restart, lr x0.3 | 1.795 | 54.93 | **56.64** | 17.69 |
  (epoch 3 of the restarts: 53.16 / 56.47 / 18.15 and 52.88 / 54.61 / 17.92.) Uninterrupted training keeps lowering held-out
  while the skills fall back from their epoch-2 peak; resetting the optimiser (with or without a lower rate) costs loss but
  gives the best skills yet - BLiMP 56.06 and supplement 56.64 are the project highs. Restart vs anneal is within one run's
  noise. Next: understand WHAT the uninterrupted run fits (Adam second-moment estimates adapted to frequent patterns?) -
  a schedule that keeps the skills without losing the loss.
- Caution on the restart gain - per-phenomenon swings between checkpoints are huge: epoch 2 -> 4 uninterrupted, quantifiers
  +20.2, npi -13.7, irregular -12.7; restart vs uninterrupted at epoch 4: irregular +14.7, npi +12.5, determiner-noun +6.3,
  argument structure +5.0, quantifiers -12.2, s-selection -5.9. The model flips its preference on whole paradigm families
  from checkpoint to checkpoint, so aggregate differences of 2-3 BLiMP points between single runs are partly this
  instability; the seed-2 run and the second restart calibrate it.
- **Seed 2 of the best 20k configuration:** held-out e1 1.896, e2 1.835 (seed 1 1.902 / 1.830); full sets e1 54.09 / 51.83 / 18.04,
  e2 53.82 / 52.47 / 17.67 (seed 1: 52.71 / 52.54 / 17.59 and 55.07 / 52.98 / 17.86). Seed spread at epoch 2 ~1.3 BLiMP. Two-seed
  means: e1 BLiMP 53.4 / supplement 52.2, e2 54.4 / 52.7 (gpu_m20 two-seed mean 52.36 / 49.10 / 17.84): +2 BLiMP, +3.6
  supplement, entity flat - robust. The restart gain (56.06 vs 53.68 at epoch 4, one run each) is ~2 seed spreads:
  suggestive, not established.
- **Periodic optimiser restarts hold the skills up (a second restart from the restarted run's epoch 4):** held-out e5 1.808,
  e6 1.804; full sets e5 54.44 / 55.92 / 18.23, e6 55.63 / 55.16 / 18.31. Across the four restarted checkpoints (epochs 3-6)
  supplement stays 55.0-56.5 and entity 18.2-18.3, BLiMP 53.2-56.1, vs the uninterrupted run's 53.1-53.2 supplement and 17.8
  entity at epochs 3-4 - consistent across checkpoints, so more than one run's noise. Held-out keeps improving slowly under
  restarts (1.814 -> 1.808 -> 1.804) though above the uninterrupted 1.770. Working hypothesis: Adam's long-lived moment
  estimates let the uninterrupted run specialise to frequent corpus patterns; resetting them re-spreads learning.
- **2026-09-26 morning: Muon, readout width, cortical motifs 1/3/4/5, learned fan-in - CPU smokes** (best 20k configuration,
  4 MB, one epoch, held-out bits/byte; seed-to-seed spread 0.0004: reference 2.21527 / seed 2 2.21491):
  | arm | held-out | vs reference |
  |---|---|---|
  | reference (best configuration, Adam) | 2.2153 | |
  | **Muon** (bodies' step: Nesterov momentum 0.95, 5 Newton-Schulz iterations per layer matrix), lr 0.02 / 0.06 | **2.1715 / 2.1710** | **-0.044** |
  | Muon, seed 2 (vs reference seed 2) | 2.1750 | -0.040 |
  | **Muon + motifs 3 + 4** | **2.1627** | **-0.053** |
  | motif 1, parallel feedforward routes (deep_in: the byte into L2 / L6 return cells) | 2.2215 | +0.006 |
  | motif 3, feedforward inhibition (ff_inh=0.5: drive / (1 + k mean\|input\|)) | 2.2085 | -0.007 |
  | motif 4, feedback inhibition (fb_inh=0.5: drive / (1 + k mean\|output\|)) | 2.2081 | -0.007 |
  | motifs 3 + 4 together (Adam) | 2.2159 | +0.001 |
  | motif 5, cross-assembly competition (compete=4: softmax-weighted emissions) | 2.2340 | +0.019 |
  | readout width: hidden layer (ro_hidden=256) | 2.2156 | 0 |
  | readout width: per-motif L5 channels (ws_split5) | 2.2222 | +0.007 |
  | readout width: 7 channels at the same size (h=2..4 populations halved and doubled) | 2.2180 | +0.003 |
  | motif_stream (L3 message = mean of the earlier bodies' L6) | 2.2167 | +0.001 |
  | motif_mix (per-body learned gates over the earlier bodies) | 2.2171 | +0.002 |
  Muon is the one large effect (10x the other levers, 100x the seed spread) and it replicates. With Adam the two inhibitions
  each help a little but cancel together; with Muon they add. Readout width is not the bottleneck at this size.
- **G: the learned fan-in does not form.** mix_probe on the motif_mix smoke: per receiving position the mean gates stay at
  0.50/0.50 and 0.33/0.33/0.33, effective fan-in exp(H) 1.96 of 2 and 2.93 of 3, sharp bodies (max gate >= 0.8) 1.8% / 0.2%.
  Each body's gates learn only from the events that reach that body (8192 bodies per population), so they barely move from
  the uniform start: the circuit is all-to-all at equal weights - no selective fan-in, no selective fan-out. Next: routing
  by content with projections shared by all bodies (motif_attn: query from the receiver's L1, key from each sender's L6,
  bounded to the <= 8 bodies of the event), so every event trains the routing.
- **Muon in the trainer:** RefLearner::muon_step (the organism's operations in order) and a CUDA k_muon (one block per body,
  the Newton-Schulz iterations in shared memory, weights written once from the rounded shadows). Exact at one stream vs the
  organism: muon default config 262144, muon + best config 524288, muon_lr 0.06 262144 events bit-identical; regressions
  (best config, default) bit-identical.
- **Muon at full scale (gpu_muon: the best configuration + muon=1 muon_lr=0.02, 64 streams, GPU): held-out 1.793 after ONE
  epoch** - Adam's gpu_best4 took three epochs to get there (1.902 / 1.830 / 1.793 / 1.770). GPU learner vs the CPU reference
  with Muon at one stream: 0 of 153,883,136 stepped-body nibbles differ, shadow |diff| max 1e-5 over 1M steps. Throughput
  49-62k bytes/s while sharing the machine (Adam alone: 101k) - to be re-measured alone.
- **G: content routing makes circuits fan in and out - selectively and by content - but costs loss.** motif_attn (the
  receiver's L1 query against each sender's L6 key, projections shared by all bodies, <= 8 slots per byte), smoke: held-out
  2.2247 (plain stream 2.2167, reference 2.2153); with Muon 2.1938 (Muon alone 2.1715). mix_probe over 100k held-out bytes:
  position 2 reads ONE sender in 65% of events (weight >= 0.8), which one depends on content (weight std 0.37 across events),
  effective fan-in 1.45 of 2; position 3 fan-in 2.04 of 3, 34% sharp; fan-out per event: sender 0 read by 2.15 of its 3 later
  positions, sender 1 by 1.31 of 2, sender 2 by 0.55 of 1. Compare motif_mix's per-body gates: uniform (2.93 of 3, 0.2% sharp).
  The shared projections learn a real routing; replacing the mean with it starves the receivers. Next: routing on top of the
  mean (a residual), and routing for the iterated processor's hops.
- **H, the prompt-following objective - round 1 is negative.** Expert iteration from gpu_cont2 epoch 2 (held-out 1.804):
  1200 BabyLM line-start prompts x 32 samples (len 200, T 0.8); per prompt keep the fluent candidate (no repeated 4-gram,
  >= 85% valid words, lp_self >= the prompt's median, no 3-gram copied from the prompt) with the largest gain + 3 x gain_late
  (kept: gain 0.094 / late 0.0102 vs all samples 0.071 / 0.0019); fine-tune 1/8 BabyLM + the kept text x 8, optimiser
  restarted. Held-out: select 1.811, random 1.804, none 1.802. Generation on 124 held-out prompts, paired vs the base
  (sample mode): select gain -0.003 +- 0.003, random +0.002 +- 0.003, none -0.007 +- 0.002; gain_late unchanged in every arm.
  Self-training raised lp_self (+0.23: the model grows sure of its own text) and greedy looping (loop4 +0.45 select, +0.32
  random, +0.03 none). Selecting the model's own high-PMI continuations does not teach it to keep to the prompt: the kept
  samples are as incoherent as the rest, and what transfers is sharpening. Muon epoch 1 (same loss level, 1.793): gain
  -0.004 +- 0.003, gain_late +0.001 +- 0.001 vs the Adam base - prompt dependence follows the loss level, not the optimiser.
- **Muon's skills, epoch 1 (full sets):** BLiMP **55.14** / supplement **54.08** / entity **18.68** (Adam epoch 1: seed 1 52.71 / 52.54
  / 17.59, seed 2 54.09 / 51.83 / 18.04). Entity is the project high (previous 18.31); BLiMP after one epoch equals Adam's
  epoch-2 peak (55.07). Muon improves the skills, not only the loss.
- Muon dual at 16 streams (best configuration, 2 MB): 3.6e-5 of stepped-body nibbles differ (Adam's same check: 1.1e-6),
  shadow |diff| mean 0.0025, 12173 bodies re-synced; at one stream 0 differ. Muon's update couples every entry of a layer
  (the orthogonalisation), so the float-order differences of the GPU's atomic gradient sums reach every weight of the
  layer instead of one - more rounding-boundary flips, not a kernel error (one stream is exact to the last bit).
- Round 3 G smokes: 7 channels + content routing 2.2363 vs 7 channels + plain stream 2.2301 (and 7 channels without a stream
  2.2180): the routing costs here too.
- **A model-free topicality judge for H** (scripts/topic_judge.py): corpus NPMI (co-occurrence in windows of 3 BabyLM lines,
  37k content words) between a continuation's NEW content words and the prompt's. Validated on 400 held-out prompts: real
  continuations 0.105 (second half 0.092), another line's text 0.043. Model generations on the 124 eval prompts: Adam base
  0.036 / Muon e1 0.040 (sample) - at the level of a random line: the models do not follow the prompt at all. Round 1's
  arms, paired vs the base: select -0.010 +- 0.003, random -0.007 +- 0.003, none -0.003 +- 0.003 (self-training moves the
  model AWAY from its prompts); Muon e1 +0.004 (greedy +0.009 +- 0.003, second half +0.008 +- 0.004). Best of 32 fluent
  samples BY THE JUDGE: 0.070 / second half 0.091 (as topical as real text in the second half) - a far stronger signal than
  the model's own PMI; that arm (topic) is next.
- Muon epoch 1 per BLiMP phenomenon vs the two Adam epoch-1 seeds' mean (item B, no regressions): ellipsis -0.9, binding -0.7,
  filler-gap -0.4, islands -0.4 (all inside the seed spread), control/raising +0.4, subject-verb +0.9, quantifiers +1.1,
  s-selection +2.3, determiner-noun +2.6, argument structure +2.6 (the old flag), anaphor +4.8, irregular +6.7, npi +7.3.
- **Motifs 3 + 4 halve saturation** (smokes, share of hidden outputs pinned at the rails over the epoch): reference 34.0% (seed 2
  34.1%), Muon 36.8% (lr 0.06: 47.7%), feedforward inhibition 26.6%, feedback inhibition 21.4%, both 16.5%, Muon + both 16.9%,
  competition 33.4%, content routing 35.1%. Divisive inhibition keeps the units off the rails - below conformance C8's 25%
  bound, which every model so far failed (C8.L1 71% saturated on the memory model).
- Adam regression dual at 16 streams after the Muon port: 152 of 696M nibbles differ (2.2e-7), as before the port.
- **Muon epoch 2: held-out 1.763** (Adam's 4-epoch run ended at 1.770).
- **H, the topic-judge arm is negative too:** best of 32 fluent samples by the judge (kept: topic 0.070 / late 0.091, as topical as
  real text in the second half), fine-tuned exactly like the other arms (held-out 1.807). Paired vs the base on the 124 eval
  prompts: judge topic -0.008 +- 0.003 (sample), -0.009 +- 0.003 (greedy), second half -0.004 / -0.016; model-PMI gain +0.015 +-
  0.003 (sample); greedy loop4 +0.31. Every positive-only self-training arm (PMI-select, random, topic-select) moves the
  generations AWAY from the prompt by the judge, whatever the selection: the model learns its own text's register faster
  than the selection's direction. Next: policy gradient - every sample weighted by its advantage over its prompt's other
  samples (negative weights push the off-topic continuations down; the weights sum to ~0 per prompt, no net pull toward
  the model's own text), via a per-byte learning weight in the trainer (goal_w x MS_EXTRA_W).
- Motif 6, disinhibition (`disinh=0.5`: an L6 inhibition released per motif by the 4 register cells, learned), Adam smoke: 2.2122
  (reference 2.2153: -0.003, 8x the seed spread); saturation 33.4% (unchanged).
- Motif 2 as a self-loop (`recirc=40 recirc_iters=1`: a body whose L6 is strong re-runs its L2..L6 on its own L4 return),
  Adam smoke: 2.2420 (+0.027): the L4 return is untrained here, so the loop feeds noise back. The figure's motif 2 is
  reciprocal excitation BETWEEN cells: route_return on the stream (round 5) trains that return path.
- Round 4 with Muon: disinhibition 2.1711 (Muon alone 2.1715: neutral - its small Adam gain does not survive Muon); the
  recirculation self-loop 2.1975 (+0.026, worse as with Adam). In exact merge checks the trainer's port of motifs 3 + 4 with
  Muon on the best configuration is bit-identical to the organism over 524288 events.
- **Muon epoch 3: held-out 1.719** (Adam's best, epoch 4: 1.770).
- Item A on Muon (register probe, is/are after 'the NOUN FILLER^d'): Muon epoch 2 d0 97.7 / d1 62.3 / d2 47.7 / d4 54.7 / d6 54.0 /
  d8 49.7 vs Adam epoch 4 (same held-out) 97.0 / 58.0 / 52.0 / 55.0 / 50.0 / 49.3 - the long-range number circuit still does not
  form. New with Muon: register r1 separates singular from plural right after the noun (sep -1.09 at d0; Adam's registers
  <= 0.31) but the separation decays (-0.57 d1, -0.21 d2, ~0 from d4) - the register CARRIES number briefly and does not HOLD it.
- Round 5 smokes: **motif 2 as drawn (reciprocal excitation between bodies: route_return on the stream) 2.2095** vs the plain
  stream 2.2167 / reference 2.2153 (-0.007 / -0.006; route_return trains the receivers' L4 return); motif 5 milder
  (compete=1) 2.2191 (+0.004, still worse); Muon + motifs 3 + 4 seed 2: 2.1684 (seed 1 2.1627; Muon alone seeds 2.1715 / 2.1750).
- H round 1 skills (full sets, vs the base gpu_cont2 e2 55.63 / 55.16 / 18.31): none 52.86 / 56.27 / 17.95, random 52.73 / 54.78 /
  17.85, select 52.92 / 55.09 / 17.88 - the BLiMP drop is the fine-tune procedure's (1/8 epoch after an optimiser restart; the
  arm without generated text drops the same), not the self-training's.
- Merge checks (real tree with motif_attn, mix_pi, disinh, the trainer's motifs 3 + 4, iter_hops, mix_res, k_muon padding):
  5 / 5 exact (ff+fb+muon 524288, fb 262144, regressions best+muon, best, default); dual S=16 ff+fb+muon 3.0e-5 (Muon's level).
- **Muon epoch 4: held-out 1.694** (Adam's 4-epoch run: 1.770). The run is complete; epochs 2-4 full sets scoring.
- **H policy gradient is negative too:** round 1's samples (8 per prompt) weighted by their advantage (topic judge, per prompt
  standardised, clipped +-2; 3782 positive / 5816 negative, weights summing to ~0 per prompt) through the trainer's per-byte
  learning weights; held-out 1.792 (the best of the arms); paired vs the base: judge topic -0.010 +- 0.003 (sample), -0.007 +-
  0.003 (greedy), second half -0.009 / -0.011; model-PMI gain -0.008 +- 0.003. Topic-select arm's skills 52.57 / 54.72 / 18.11
  (the procedure's drop, as the others). EVERY arm trained on the model's own continuations - positive-only (PMI, random,
  judge) or signed - loses 0.007-0.010 of topicality; the arm without generated text -0.003 +- 0.003. Reading: the
  continuations share a format (a few words, then a switch to an unrelated line) that training on them reinforces whatever
  their weights, and one sample's judge score is too noisy (within-prompt spread 0.018) to steer against it. Next: the
  real-text arms (topic-weighted vs flat replay), which carry no generated text.
- **H: topicality follows the model - Muon epoch 4 is the first generator measurably on its prompt.** Topic judge on the 124 eval
  prompts (greedy / sample): Adam base 0.033 / 0.036; Muon e1 0.042 / 0.040, e2 0.036 / 0.038, e3 0.034 / 0.035, **e4 0.053 /
  0.047**; paired vs the base e4 +0.019 +- 0.005 / +0.011 +- 0.004, second half +0.015 +- 0.006 / +0.010 +- 0.005 (a random line
  scores 0.043, real continuations 0.105). Model-PMI gain e4 0.095 (sample; base 0.072), second half 0.0077 (base 0.0023). Not
  monotonic over epochs (e2 / e3 dip) and still far from real text, but the one lever that moved prompt following is a better
  model - every fine-tuning objective on its own samples moved it the other way. Greedy loops more at e4 (loop4 0.48).
- H policy-gradient arm's skills: 52.14 / 53.20 / 18.12 (the procedure's drop; supplement -1 below the other arms).
- **H topic-weighted REAL-text replay is negative too:** 2 MB of consecutive BabyLM training lines, each weighted 0.5-2 by how
  strongly it continues its previous 3 lines' topic (judge NPMI), on top of the 1/8 replay: held-out 1.787 (the best of all
  arms); generation paired vs the base: judge topic -0.012 +- 0.003 (sample), -0.013 +- 0.003 (greedy), second half -0.012 /
  -0.018; model-PMI gain +0.010 +- 0.002 (sample). Even real text lowers the judge's topicality after the fine-tune. The flat
  control (the same text, weight 1) is queued; if it drops the same, the fine-tune's data mix (60-line blocks, mostly
  dialogue) rather than the weighting sets the effect.
- 2026-09-26 afternoon: the user asked to keep runs to ONE at a time for now; everything else was stopped and the pending
  work put in a strictly sequential, pausable queue (build-gpu/queue_one.sh; `touch ~/babylm/runs/QUEUE_PAUSE`).
- **G: the iterated processor does not extrapolate on real text.** iter_hops=4 (the byte's processor body, 0:256, run after the 4
  motif bodies on the mean stream with a per-hop cue), smokes; then held-out (heldout_ss.txt, 272 KB) with 0..8 hops at eval:
  | model | 0 | 1 | 2 | 3 | 4 (trained max) | 6 | 8 |
  |---|---|---|---|---|---|---|---|
  | Adam, random depth 0..4 | 2.217 | **2.204** | 2.206 | 2.214 | 2.219 | 2.268 | 2.342 |
  | Adam, fixed 4 hops | 2.218 | 2.217 | **2.208** | 2.222 | 2.221 | 2.270 | 2.290 |
  | Muon, random depth 0..4 | **2.157** | 2.162 | 2.160 | 2.170 | 2.177 | 2.255 | 2.369 |
  One or two hops help a little with Adam (-0.013), none with Muon; past the trained depth the loss rises steeply. The
  harness's extrapolation needed a task with serial steps and a per-hop cue from a tape; byte-level text does not reward
  serial depth per byte (as motif_stream showed). Unbounded growth on text will have to come from the time axis (memory,
  the pyramid) or from tasks that need serial steps, not from more hops per byte.
- **Muon's skills over epochs** (full sets; held-out): e1 55.14 / 54.08 / 18.68 (1.793), e2 54.96 / 54.21 / 17.79 (1.763), e3 **55.81** /
  52.98 / **18.97** (1.719), e4 54.06 / 52.28 / 18.48 (1.694); Adam gpu_best4 e2 55.07 / 52.98 / 17.86, e4 53.68 / 53.21 / 17.76.
  Muon is ahead at every epoch (entity 18.97 is the new high), but the old divergence stays: the loss keeps falling while
  BLiMP plateaus near 55 and supplement slides after epoch 2. Optimiser restarts held Adam's skills up; Muon restarts are next.
- Caveat on "Muon epoch 4 is measurably on-topic": read the outputs. Greedy decoding collapses into memorized templates for
  BOTH models (Adam base / Muon e4: 17 / 12 of 124 outputs share an identical 60-byte opening; over half of all 8-word shingles
  occur more than twice) - Muon e4 writes the same Simple-Wikipedia templates ("a song by Mario Bardi, belonging to the artistic
  movement of Magic Realism...", "the state of Wisconsin ... In 2020, 1,24 people lived there. The county seat is ...") after
  unrelated prompts. Sampled text has no template reuse but is encyclopedic word salad. The +0.011 (sample) judge gain is
  real but small and on incoherent text; free generation is not yet on-topic in any useful sense.
- H topic-weighted replay arm's skills: 52.51 / 52.30 / 17.91 (none 52.86 / 56.27 / 17.95). Round-1 summary: no fine-tuning
  objective (PMI-select, random, topic-select, policy gradient, topic-weighted real text) improved prompt following or skills;
  all share the procedure's BLiMP drop (~-2.8 vs the base) and lose topicality.
- **Muon + motifs 3 + 4 at full scale is WORSE, unlike the smokes:** gpu_muon_ffb2 (64 streams, GPU, ff_inh=0.5 fb_inh=0.5 + Muon)
  epoch 1 held-out 1.864 vs Muon alone 1.793; the training loss trails by 0.04-0.06 at every report of epoch 1 (8M: 2.020 vs
  1.990 ... 46M: 1.779 vs 1.725). In the single-stream CPU smokes the same combination was better on two seeds (2.1627 /
  2.1684 vs 2.1715 / 2.1750). The trainer's port is bit-exact at one stream and the GPU learner matches the reference, so this
  is a single-stream vs 64-stream difference, not a port error - unexplained; the natural test (ms_babylm at S=1 on the CPU
  reference, same 4 MB) was not run (wrap-up). The inhibition forward in the trainer is also slow (scalar nib_get instead
  of the vectorised kernel): epoch 1 took 55 min while sharing the machine.
  Epoch 2: 1.811 vs Muon alone 1.763 (+0.048, the same gap).
- **K bodies per byte (the user's hypothesis, 2026-09-26: 4 bodies per token is far too few; it should engage hundreds to
  thousands continuously).** Trainer limits raised (16 bodies / 8 readout channels per stream -> 256 / 256; `fm_salt` gives
  same-length populations independent hashes; exact at one stream). Test at the same size (~21k bodies), Muon, a quarter epoch
  (MS_SUBSET=4, 13.6 MB), held-out on 256 KB: K=4 (the best configuration) training loss 2.261 / 1.979 / 1.914 at 4M / 8M /
  12.6M bytes, held-out 1.932. K=256 = 1:256 + 255 populations of 80 bodies (contexts 2-8, salted), each its own readout
  channel: 2.742 at 4M - but CONFOUNDED: all 256 bodies summed into the shared 32-cell workspace that the shared readout, the
  registers and the pyramid read (64x the scale of 4 bodies). Stopped; rerun with `ws_emit_max=4` (only the first 4
  populations write the shared workspace, all 256 feed their own channel and learn only its error; exact). Throughput at
  K=256 is ~1.8k bytes/s: the per-stream 256-channel readout (an 8 MB matrix swept 2-3x per stream per byte, no reuse across
  streams) - at this width the readout must be batched across streams (one GEMM on the GPU).
- **K=256 bodies per byte (clean, ws_emit_max=4) - a clear NO for the flat version:** training loss 2.578 / 2.525 / 2.515 at 4M /
  8M / 12.6M bytes vs K=4 2.261 / 1.979 / 1.914; held-out (256 KB) 3.298 vs 1.932. It stalls near 2.5 and generalises worse
  than it trains. At the same size, 256 populations leave 80 bodies per context population (K=4: 4096-8192): the model's
  strength was bodies as fine-grained context experts, and one 5.6k-weight body cannot serve 1/80 of all contexts; and the
  256 bodies are a flat ensemble - each sees only the byte input, they meet only at the readout (no shared state, no lateral
  input, no persistence across bytes). What this does NOT test is the user's actual design: hundreds of bodies engaged
  CONTINUOUSLY on a shared state (a recurrent sheet with lateral connections and persistent activity) - that is a different
  forward pass (batched on the GPU), not a configuration of this one.
- **2026-09-26 evening: wrap-up (the user: "wrap up for now"; Coco may be superseded).** Stopped: gpu_muon_ffb2 after epoch 2
  (held-out 1.864 / 1.811 vs Muon alone 1.793 / 1.763; states kept), and the rest of the one-at-a-time queue - never run: the
  flat real-text H control (tw0), the PMI decoder on Muon epoch 4, 9 smokes (residual routing, Muon + MTP / momentum 0.9 /
  disinhibition, reciprocal excitation with Muon), conformance on the Muon models, and the second queue (a Muon optimiser
  restart, 4- and 5-level pyramids). Scripts: build-gpu/queue_one.sh, queue_two.sh.
  Where it stands: the best model is Muon on the fixed-motif configuration (gpu_muon: held-out 1.694 after 4 epochs; full sets
  up to 55.81 / 54.21 / 18.97). Open: prompt following (free generation is off-topic and incoherent; no fine-tuning objective
  helped), the long-range number circuit (registers carry, never hold), unbounded circuits (per-byte depth does not pay on
  text; content routing forms fan-in/fan-out but costs loss), local learning (not started). The last finding points at the
  next design: 4 bodies per byte is too little compute, but 256 independent bodies per byte is worse still - the bodies must
  work CONTINUOUSLY on a shared state (a recurrent sheet, batched on the GPU), not as parallel lookups.
