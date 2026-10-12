# Circuits that grow without bound (goal item G): design and first test

Status: design 2026-09-25; harness results 2026-09-26 (below, "What the depth ladder showed") revise the mechanism.

## The requirement

The user's words: circuits must be able to increase in complexity **without bound by recruiting and layering bodies**. A
fixed architecture of depth D cannot meet that. Today's fixed motifs run one body per motif per byte, so every byte gets
the same 4-body chain (with `motif_wire`, body m's L3 reads body m-1's L2). That makes depth 4 per byte, and breadth is
fixed too. The address tree grew by births, but its churn blocked circuit formation, so it was replaced by fixed motifs.

What has to be true of a design that meets the requirement:
1. **Depth on demand:** the number of bodies that compute in sequence within a byte, or across bytes, is chosen by the
   input and by learned utility. It is not a constant, and nothing in the design caps it except a compute budget.
2. **Recruitment without churn:** new capacity comes from a fixed, preallocated reservoir of bodies (no births, no tree).
   Which body is recruited is a learned, stable function of what the recruiting body computed.
3. **Composition:** a recruited body's input is the recruiting body's output (L2 -> L3, as `motif_wire` already does).
   Its gradient returns to its recruiter (`bp_route`, now exact in both learners). So a chain of k bodies is a depth-k
   network trained end to end.
4. **Capability must grow with depth:** on a task that needs k serial steps, accuracy must rise as the allowed number of
   recruitment hops rises. This is the test that separates real growth from decoration.

## Mechanism: routing by role (a learned merge key)

After a body runs, the sign pattern of its L2 (24 bits) is its **role code**. A hop recruits the body at
`pool_base + hash(role bits, hop) mod pool_n` in a routed pool of fixed size. The recruited body gets the recruiter's
L2 as its L3 message, emits into the workspace gated by its route utility (as routed bodies are today), and may recruit
in turn.

- Same role -> same next body. Different surface contexts that the recruiter maps to the same role share a body. That
  is the "merge key" (Interference Search's term): recruitment by what was computed, not by the bytes seen.
- Depth is gated by utility: hop h+1 runs only when the hop-h body's route utility exceeds `hop_gate_bits`. That is
  the counterfactual gain the organism already measures for routed bodies. Depth grows where it pays and costs nothing
  where it does not.
- The code is discrete, so recruitment itself is not differentiable. It is learned the way the tree's routing was:
  bodies learn, by gradient, to be useful to what routes to them; route utility decides whether a hop is taken; and the
  recruiter's L2 is shaped by its receivers' message gradients (`bp_route`), which moves its role code.
- Unbounded: the pool is finite, but chains can be as long as the budget allows. Growing capacity means enlarging the
  pool. That needs no retraining of existing bodies, because hashing into a larger pool only remaps roles.

This reuses what exists: routed bodies, route utility gating, counterfactual credit, L2 -> L3 messages, and
message-gradient backprop (CPU and GPU, exact). The new parts are (a) the role-code hash instead of a fixed motif
index, (b) a hop loop with utility-gated continuation in the trainer's P2 / S2, and (c) jobs linked across hops, which
`LTrace.sender` already expresses as an offset.

## Test: a depth ladder

A synthetic task where the number of serial steps is a knob, so capability can be read off against allowed depth:
- **Parity-k over marked bits:** a line is `a b c ... | ?`; the answer byte is the parity of the k bits marked in the
  line. Each extra bit is one more serial XOR. k = 1..8.
- **Nested agreement:** `the N1 that the N2 that the N3 V3 V2 V1`. Each embedding level needs one more stored-and-matched
  number; depth = nesting level.

Protocol: for max_hops H in {0, 1, 2, 4, 8}, train the organism probe (agree_probe-style, 64 GPU streams) and report
accuracy at every k. **Pass:** the largest k solved rises with H, and ablating hop h (silencing that hop's bodies) costs
exactly the k's that need it.

## Order of work

1. Harness first (cpp/apps/seq_probe.cpp: exact BPTT, minutes per run). Add role-code recruitment to its body stack and
   run the parity ladder. If capability does not track H there, the mechanism is wrong; stop before touching the organism.
2. Organism + multi-stream trainer: the hop loop in P2 / S2 (serial per stream, receivers after senders as now); jobs
   linked by `sender` offsets; ms_check exact at S=1; dual; then the ladder on the GPU probe.
3. Real text: does the model take more hops on harder bytes, and does BLiMP move?

Open questions: whether to include the hop index in the hash (distinct pools per depth vs one shared pool); how many
role bits; whether a recruited body should also read the recruiter's L6 (its output) rather than its L2.

## What the depth ladder showed (cpp/apps/depth_probe.cpp, 2026-09-26)

The harness: x in 0..15 and a program of k ops (+1, x3, xor 10, rotate-left) in one event's input; chains of INT4 bodies with
exact backprop (float shadows + Adam, as the organism's adam=1); accuracy by k (chance 6.25%). Findings, in order:

1. **motif_wire cannot compose.** Its message is the sender's L2, which is computed from the sender's own L1 only: the
   incoming message enters at L3 and never reaches the outgoing L2. A chain of L2 messages is a set of parallel functions of
   the input, not a serial computation (the GPU learner's all-receivers-in-parallel message pass is only exact because of
   this). In the harness it behaves as a wide ensemble: 6 bodies, all ops visible, 100 97.6 85 32.6 15 12.
2. **A residual stream composes only if it is normalised.** Reading the SUM of the earlier bodies' L6 grows with depth and
   clamps (6 bodies, tape: 100 93.6 76.2 31.4 15.2 11); reading the MEAN: 100 99 90.2 62.2 31 16.6; the mean at the L3 message
   cells: 100 98.2 97.4 71.6 38.4 21.6. A pure chain (each body reads only the previous one) does not train at 6-8 bodies
   (chance) - the skip path is required, as in deep learning.
3. **Iterating ONE body gives the clean ladder.** An entry body plus one body reused at every hop, reading the mean stream
   (tape: hop h is given op h): 96.6 97.0 90.8 84.4 69.6 48.2; cut after 5 hops, k=6 falls to 7% - each hop adds one op.
4. **Recruiting by role code hurts here.** Pools of 8 bodies per hop chosen by the recruiter's L2 sign bits: 100 93.8 50.4
   12.2 - routing splits the training signal over bodies that each see a fraction of the cases.

Revised mechanism for the organism (to be ported):
- **motif_stream**: the byte's motif bodies form a chain whose message is the MEAN of the earlier bodies' L6 (at the L3
  message cells), and the message gradient returns to every earlier body's L6. Unlike motif_wire this composes; its GPU
  backward needs one round per chain position (receivers' gradients depend on their own receivers').
- **An iterated processor**: after the motifs, one shared body (or a small context-chosen set) runs D more hops per byte on
  the same mean stream. Depth then costs compute, not parameters, and can be raised without new bodies - the unbounded
  axis. Depth on demand: continue while the hop changes the readout (a halting test), within a budget.
- Recruitment of *new* bodies is then a capacity question (more processors for more kinds of step), decided by evidence
  of interference, not a per-hop routing hash.

## Organism results (2026-09-26)

`motif_stream=1` (with bp_route) implements the composing chain in the organism, the multi-stream trainer and both
learners (exact at one stream; the GPU dual at the motif_wire control's noise level). On BabyLM it does not pay:
held-out +0.015 at 16 streams and +0.066 at 64 (the 64-stream update ordering multiplies the cost), skills flat. Byte-level
text at this scale does not use per-byte serial depth; the harness ladder does. It stays off.

What did pay on real text was WIDTH at the readout (`ws_split`): the summed 32-dim workspace was the bottleneck - 8 motifs
through it are worse than 4 (2.026 vs 1.966), through per-motif channels better (1.885). More bodies per byte is a real
axis of growth once each has its own channel to the readout; whether it continues (6 motifs at the same 20k size is
queued) is the next measurement for "circuits that grow by recruiting bodies".

## Depth beyond training (2026-09-26)

Training the iterated body on chains of RANDOM length (`randdepth=1`: each example runs between its program length and 6
hops) makes it depth-agnostic: within range 99-100% up to k=5 (fixed-depth training: 60% at k=5), and run for MORE hops
than it ever trained it solves longer programs it never saw - k=7 83% with 7-8 hops, k=8 67% with 8 hops. One learned step,
repeated: capability grows with iterations, not with bodies. This is the mechanism to port for unbounded depth; the open
part is WHEN to stop iterating on real text (a halting signal) and what the per-hop cue is when there is no tape.

## Fan-in and fan-out (2026-09-26 afternoon)

The user's observation: circuits did not fan out or in. Measured (cpp/apps/mix_probe.cpp):
- **Per-body gates (`motif_mix`)**: each body learns softmax weights over the earlier bodies of its byte. After a smoke epoch
  the gates are still uniform (fan-in exp(entropy) 2.93 of 3, 0.2% of bodies sharp): a body's gates learn only from the
  events that reach that body, and there are 8192 bodies per population. All-to-all at equal weights - no circuit.
- **Content routing (`motif_attn`)**: a query from the receiver's L1, a key from each sender's L6, projections SHARED by all
  bodies (every byte trains them), at most 8 slots per byte (bounded attention). The routing forms: position 2 reads one
  sender in 65% of bytes, and which one depends on the input (weight std 0.37 across bytes); position 3 fan-in 2.04 of 3;
  senders fan out to 2.15 / 1.31 / 0.55 of their later positions per byte. Selective, input-dependent fan-in and fan-out -
  motif 8 of the user's figure (a source targets the related cells: the receivers whose query matches its key).
- But the routed message REPLACING the mean costs loss (smoke: +0.008 with Adam, +0.022 with Muon). Queued: the routing as a
  residual on the mean (`mix_res`), and routing inside the iterated processor.

## The iterated processor on real text (`iter_hops`)

Ported from the harness: the population of context 0 (`motifs=...,0:256`, its body chosen by the byte) runs `iter_hops`
hops after the motif bodies, each hop on the mean stream of every earlier body of the byte (its own earlier hops included),
with a fixed per-hop cue in its return cells (the harness showed the cue is essential); `iter_rand=1` trains each byte with
a random number of hops (0..iter_hops), which made the harness body depth-agnostic. The test of unbounded depth on real text
(`heldout_eval ... iter_eval=H`): train with at most 4 hops, evaluate with 0..8 - if held-out improves past 4 hops, the
learned step keeps paying when repeated beyond its training depth.
