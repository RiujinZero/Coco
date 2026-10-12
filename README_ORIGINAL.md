# Coco — Laminar 200k Organism, native reconstruction

A faithful recreation of the processing unit described in
[`Laminar_200k_Organism_Design_Doc.docx`](Laminar_200k_Organism_Design_Doc.docx) built along
[`CPP_OPTIMIZATION_RUNBOOK.md`](CPP_OPTIMIZATION_RUNBOOK.md): a sparse, event-driven population of
up to 200,000 identical six-layer INT4 units, addressed by learned lock/key tumblers, with early
information death, motif-local K routing, local accumulator plasticity and consequence-driven
structural splitting.

Two implementations of the same organism live here and are checked against each other:

* `cpp/` — the native pipeline (one C++ call per event block, no Python in the hot path).
* `python/laminar_ref/` — the NumPy reference, the "slower faithful kernel" of runbook §15.

`docs/ARCHITECTURE.md` maps every requirement of both documents to the code and lists every
choice the documents left open.

## Build

**WSL / Linux (GCC, primary bench target)**

```bash
./scripts/build_wsl.sh
```

Uses CMake if present, otherwise calls `g++ -std=c++20 -O3 -march=native -flto -DNDEBUG` directly.
Binaries land in `build-wsl/` and `test_kernels` runs at the end. `./scripts/build_wsl.sh pgo-gen`
/ `pgo-use` give the runbook's PGO cycle.

**Windows (MSVC 2022 Build Tools, /O2 /arch:AVX2 /GL)**

```bash
powershell -ExecutionPolicy Bypass -File scripts/build_msvc.ps1
```

Note: on this machine WSL has no CMake and Windows has no `g++`; both scripts cope. Anything with
shell variables must be a script file — variables do not survive `wsl -- bash -lc '...'`.

## Run

All apps take `key=value` parameters (see `Params::from_args` in `cpp/include/laminar/config.h`).

```bash
# Milestones A/B/C: synthetic vocabulary (32 symbols, 8 contexts, 8% noise), train then frozen eval
./build-wsl/laminar_synth units=200000 train_events=120000 eval_events=16000 report=20480 log=synth.jsonl

# Milestone D: raw bytes, online next-byte prediction (mode=adaptive|frozen|nomem)
./build-wsl/laminar_bytes file=data/tinystories_100k.txt units=20000 epochs=3 mode=adaptive

# Runbook §11 stress test on the real 563 MB store, four regimes
./scripts/bench.sh 200000 100000

# Runbook §13.1 parity: NumPy reference vs native on a deterministic trace
python python/run_synth.py --units 1024 --train 3000 --eval 500 --block 1024 --trace ref.jsonl
./build-wsl/laminar_synth units=1024 train_events=3000 eval_events=500 block=1024 trace=nat.jsonl
python python/tests/compare_traces.py ref.jsonl nat.jsonl
```

`scripts/milestone_d.sh` runs the three milestone-D comparisons; `scripts/fetch_tinystories.sh`
downloads a slice if `data/` is empty.

### Optional grounded dialogue session

For the current joint structured learner, use **`laminar_train trainer=continual`**
or `bash scripts/train_coco.sh ...`. The versioned `coco_continual_v1` profile
consolidates the tested neural controller, primitive acquisition/rehearsal,
concept/language/effect learning and evidence revision in one training loop.
It supports deterministic offline replay checkpoints. Run
`bash scripts/train_coco.sh --check` for schedule, resume and correction tests.
See [the canonical trainer and curriculum interface](docs/CANONICAL_TRAINER.md),
[integration results and failures](docs/COMPOSED_SESSION_EXPERIMENT.md), and
[the next warm-start curriculum](docs/WARM_START_CURRICULUM.md).
The byte-stream grounding mode described below remains a separate model path.

`laminar_train` accepts `grounded_bootstrap=PATHS` to attach the experimental
grounded reader, evidence journal and learned correction dispatcher to its byte
stream. The bootstrap manifest names verified teaching corpora; symbolic meanings
stay frozen during ordinary stream training. Initial facts use `CONTEXT:`, later
observations use `ENVIRONMENT:`, and requests use `USER:` followed by `ASSISTANT:`.
Generated output is never evidence. Checkpoint probes use a private grounded
session and preserve the training episode.

With the preceding grounding experiments' teaching files available, run:

```bash
bash scripts/grounded_session.sh
./build-wsl/laminar_train_grounded file=logs/grounded_session/trainer.txt \
  grounded_bootstrap=logs/grounded_session/bootstrap.paths units=128 seed=0 \
  block=31 seq_window=65536 seq_bits=10 seq_consolidate=0 probe_every=1048576
```

The script builds release and sanitizer tests and audits mixed dialogue plus the
real trainer. Normal builds also include `laminar_train`. This is optional runtime
integration of bounded pretrained meanings, not full public-curriculum training
or simultaneous language/policy learning. See
[the protocol, source metadata and limits](docs/GROUNDED_SESSION_EXPERIMENT.md).

Verified observation-language teaching is now available through the separate
`grounded_feedback=FILE` interface. `bash scripts/grounded_teaching.sh` tests
learning across episode resets while byte learning runs, followed by fresh-name
transfer and corrections using the new meanings. Teacher answers must be verified
by the host; ordinary assistant text is not teaching evidence. See
[the teaching protocol and remaining limits](docs/GROUNDED_TEACHING_EXPERIMENT.md).

`grounded_archive=1` retains bounded raw teaching provenance across episodes so
old identified answers can be revised or withdrawn. `bash scripts/grounded_archive.sh`
checks recovery from earlier mistakes, episode isolation and actual trainer/probe
equivalence. See [the archive experiment](docs/GROUNDED_ARCHIVE_EXPERIMENT.md) and
[the remaining complete-learner gates](docs/FINAL_CAPABILITIES_VALIDATION.md).

`bash scripts/computation_control.sh` compares local plasticity and backprop for
choosing retrieve/reason/ask/act/return from terminal outcomes and operation costs.
Both pass the bounded endpoint test, with remaining efficiency and credit limits:
[experiment and controls](docs/COMPUTATION_CONTROL_EXPERIMENT.md).

`bash scripts/primitive_discovery.sh` tests discovery of follow/equality/count
policies from final answers, transfer to the native neural skill learner,
retention through rehearsal and composition without external skill traces. See
[results, controls and remaining integration work](docs/PRIMITIVE_DISCOVERY_EXPERIMENT.md).

`bash scripts/concept_growth.sh` tests learning relational definitions with new
argument roles, reuse in later concepts, retention and evidence correction. Three
vocabularies pass 4,800 fresh final queries; flat controls distinguish reuse under
a local rule bound from wider flat search. This is a separate symbolic module:
[concept growth and the remaining joint-learning gap](docs/CONCEPT_GROWTH_EXPERIMENT.md).

`bash scripts/relational_session.sh` tests one continuing model acquiring concepts,
language, action effects and procedures from its own executions, with versioned
evidence replay. Three seeds pass 2,304 fresh language queries and 144 fresh action
goals. That probe leaves neural skills and learned computation control separate:
[coupled learning, corrections and matched planning controls](docs/RELATIONAL_SESSION_EXPERIMENT.md).

`bash scripts/relational_control.sh` adds learned operation selection to that
continuing session. Empirical credit plus backprop policy fitting passes 308/384
final cases across three seeds (192/192 queries, 116/192 action goals), versus
103/384 for direct neural credit. Reliable actions and useful retrieval remain
open: [joint control results and controls](docs/RELATIONAL_CONTROL_EXPERIMENT.md).

`bash scripts/relational_reactive.sh` tests repeated act/observe/reason cycles,
failed motor attempts and unavailable sensing. The neural arm passes 354/384
final cases and 120/150 three-failure recovery cases; removing retrieval reduces
final success to 299/384. The full learner remains unproven:
[recovery, learned control and remaining gaps](docs/RELATIONAL_REACTIVE_EXPERIMENT.md).

`bash scripts/computational_session.sh` connects newly acquired neural primitives
to the continuing concept/language/action session. Three seeds pass 1,008 withheld
questions, 120 count-dependent physical goals and 648 retained primitive cases.
An initially insufficient curriculum and the remaining controller benchmark are
documented in [the computational-session experiment](docs/COMPUTATIONAL_SESSION_EXPERIMENT.md).

`bash scripts/computational_control.sh` trains operation selection while these
primitives, meanings, concepts and effects arrive. Three seeds pass 384/384
standard cases and 89/89 recovery cases, but only 30/36 larger goals. A separate
size-input arm learns further reasoning yet transfers less reliably:
[combined control, negative controls and remaining failures](docs/COMPUTATIONAL_CONTROL_EXPERIMENT.md).

`bash scripts/computational_progress.sh` tests retaining search allowance across
unchanged observations with those same frozen weights: 384/384 standard cases,
89/89 recovery cases and 36/36 larger goals. A matched preference-tree readout
performs worse. See [control-transfer diagnosis and remaining capability blockers](docs/CONTROL_TRANSFER_DIAGNOSIS.md).

`bash scripts/composed_credit.sh` tests acquisition of missing counting from
whole composed answers, without direct counting lessons. Three Boolean-curriculum
seeds pass 1,680/1,680 composed tests and retain earlier neural skills through
rehearsal; the numeric curriculum succeeds in two of three seeds.
`bash scripts/composed_lifecycle.sh` tests subsequent procedure learning and
versioned evidence withdrawal/restoration. See
[composed credit, search costs and remaining integration](docs/COMPOSED_CREDIT_EXPERIMENT.md).

## Results (AMD 9950X, single thread, WSL/GCC 15, AVX2 kernels)

| Test | Result |
|---|---|
| Kernel equivalence (scalar / AVX2 / expand), tree invariants | pass (`test_kernels`) |
| Native vs NumPy reference, 3,500 events with learning on | integer path 3500/3500 identical, predictions 100% agree, loss equal to 4 d.p. |
| Synthetic vocab, **1 context**, 4,096 units, 120k events | **92.19%** frozen (noise ceiling ≈ 92.25%, chance 3.1%) |
| Synthetic vocab, **8 contexts**, 4,096 units, 120k events | **79.4%** frozen, loss 1.17 nat, 242–256 live locks (256 = 32 symbols × 8 contexts) |
| … same with hidden-layer perturbation learning (`hidden=perturb perturb_clean=1`) | 78.4–80.4% (neutral; 3× forward cost) |
| Synthetic vocab, 8 contexts, **200,000 units / 563 MB store** (milestone C) | **79.4%** frozen, 243 locks; **607 KB/s training**, **1.25 MB/s frozen**, 5.6k MACs/event |
| Bench 200k, familiar | 777 KB/s, 6.2k MACs/event, 1.83 admissions/event, 2.5 GB/s weight traffic |
| Bench 200k, moderate novelty | 514 KB/s, 10.3k MACs/event |
| Bench 200k, novelty storm (256 symbols, births) | 294 KB/s, 12.2k MACs/event |
| Bench 200k, pathological full bodies | 323 KB/s, 19.4k MACs/event, 3.2 GB/s |
| **Milestone D**, TinyStories 100 KB, 50k units: adaptive locks + learning | **BPB 3.58 → 2.88** in 6 passes (2.91 in 8 before the rounding fix), acc 35 → 52%, locks 187 → 860, ≤3 active units/byte, **440–550 KB/s** |
| … same with hidden-layer perturbation learning | BPB 2.85–2.93 at pass 6 (indistinguishable from L6-only), ~130 KB/s |
| … same, locks frozen after pass 1 (weights still learn) | BPB 3.20 → 3.34 (plateaus, then drifts) |
| … same, routing/memory disabled (sensory-only locks, no L4, no neighbours) | BPB ≈ 4.1 (unigram-per-byte level) |
| TinyStories 1 MB, 200k units, first pass | BPB 3.04, 1,569 locks, 250 KB/s |
| Persistent workspace into L1 (`ws_decay=128 ws_input=1`, now default), TinyStories 100 KB | BPB 2.90 → **2.76** |
| **Rolling context**, needle recall over a 13.6 MB haystack, W = 2 MB, 800k-unit pool | **77–83% recall at every distance up to 4.2 MB** (chance 3.8%), no evictions |
| … same, 200k-unit pool | 70–76% recall up to 1.0 MB, 0% from 1.5 MB (LRU edge) |
| … same, **350k-unit pool (target)** | **70–79% recall at every distance through 2.1 MB**, 74% at 2.6 MB, 44% at 3.1 MB, 0% at 4.2 MB — a ≈2.5–3 MB rolling window |
| **Agency kernel (Agkern port) directing attention**, needle 256 KB, agency off → on | recall 53–68% → **74–81%** at every distance (learns `attend_episodic`); BPB 2.59 → 3.3 |
| **128 MB TinyStories, one pass**, baseline with frozen readout | BPB **3.06** flat (plateau at ~14k locks); online byte n-gram order 3/5/7 on the same stream: 1.84 / 1.39 / 1.20 |
| … with readout left learning (the drift bug, now fixed by default) | BPB 3.2 → 4.9 over the pass |
| … + episodic memory (600k pool) | BPB 3.90 — recall mechanism, not a language-model gain |
| Counterfactual credit for routed bodies, 32 MB | baseline BPB 3.05 → **2.97** |
| Motif packing + K screen, 32 MB | BPB 3.34 → 2.85 with the utility gate; the screen routes 98% of births to K=2 |
| ALM-style context lifecycle (nursery → admit by bits → utility → recycle), 32 MB | BPB 2.97 → **2.67**, 27.8k contexts vs 8.9k |
| **Free generation** with the Atomic-Thought sequence memory (composition), 32 MB | BPB **2.05** (1.92 with a 128 MB window); planted-phrase recall from a cue **18/32 exact** (bodies alone: 0/32) |
| **Strict 250-word test** (T=0.7, provenance rule + bubbling consensus, `scripts/strict_gen.sh`) | **380 words, 0/8 loops, 98% real words, 11.7% of bytes verbatim** — greedy decoding fails it (8/8 loops, 96% copying) |
| **Bounded memory**: 256 KB instance window + tagged consolidated class counts (`seq_consolidate=1`, `scripts/bounded_gen.sh`), 32 MB | BPB **1.75** vs 1.94 for the full 32 MB window (n-gram reference 1.84); planted recall **23/32**; 0–1/8 loops, more repetition |
| **Stay on task** (`laminar_chat`, `scripts/ontask.sh`), SmolTalk 1.7 MB, 40 held-out prompts: reply content-word overlap with the reference reply, vs chance | no steering 0.011 (chance 0.005); **goal-directed commitment to a related episode** `seq_topic_gain=4`: **0.070 (chance 0.003)**; gain 16: **0.118 (0.016)**, 100% real words. LMSYS 128 MB: 0.009 (0.012) → **0.078 (0.014)** |
| **Reasoning over supporting facts** (bAbI en-10k, `laminar_babi`, `scripts/babi.sh`, 200 held-out questions per task), baseline → relations within the episode (`seq_chain=1`) | qa1 15% → **100%**, qa2 9.5% → **100%**, qa16 23.5% → **100%**, qa3 17% → 41%, qa15 38.5% → 18% (chance 18–38%); the chain of thought is printed per question |
| **The training run** (`laminar_train`, `scripts/train_run.sh`): 650 MB mixed stream (wiki-en 512 MB + LMSYS 128 MB + SmolTalk + bAbI + CLUTRR), 200k units, general mechanisms only, probes every 64 MB | wiki-en held-out BPB 2.93 → **2.73**, 95–98% real words, loops in 2 of 20 probe prompts; learns each source's reply register, but stay-on-task, bAbI and CLUTRR stay at chance without the scaffolds — see `docs/ARCHITECTURE.md` "The training run" |
| **Learned reading** (`seq_chain=2`): no stop list, no subject/object rule, no form list — word utility, position preference and question form all credited from observed answers | bAbI original **qa1 100%, qa2 81%, qa16 98%**; shifted (unseen vocabulary and templates) **99 / 72.5 / 63.5%** (with the surprise gate; 84 / 63 / 63.5 before it) — the hand-written reading gives 100/100/100 and 100/77/63.5; the learned thought filter converges to names, objects and locations on its own |
| **Training run 2**: the same 650 MB stream with the learned reading on, credited only by the stream's own `ASSISTANT:` turns | at 64 MB: bAbI **24%** vs run 1's 8% — qa1 10/10, qa2 8/10, qa12 10/10 — with wiki BPB identical (2.926); at 128 MB it drifted to 13.5%: the penalty on non-answering chains eroded the words that answer (most turns on a mixed stream are unanswerable by a chain) |
| Learned reading, all 20 bAbI tasks trained together, penalty on → off (`seq_chain_neg=0`, now default) | qa1 15% → **74%**, qa2 1% → **84%**, qa4 0 → 66%, qa5 35 → 82%, qa12 16 → 55%, qa16 100%; **training run 3** (corrected rule) holds at 128 MB where run 2 collapsed: qa1 10/10, qa2 8/10, qa12 10/10 |
| **Stay on task with nothing hand-written** (learned reading + commitment, `seq_chain=2`) | SmolTalk 0.064–0.083 vs chance ≈ 0.015; LMSYS 128 MB, 343 held-out dialogues: **0.033 vs 0.016** (vs prompt 0.076 vs 0.027) — twice chance at scale |
| **Composition** on CLUTRR (400 held-out; answers composed along a path by a learned operator): paths only → learned key resolution (four then six candidate keys per hop, all credited; per-form selector; evidence-weighted) | 78 → **90** correct; 25.9% / 22.1% at 4 / 5 hops vs chance 13% / 14%; prose closures measured negative (kinship is not a tree); attribute equivalence by consequence starved (9 merges in 15k answers) |
| **Relations as things** (`seq_rel_gain`): (thought, words-between, thought) triples proposing the next thought, credited by the information they deliver against the organism's own prediction — from prose alone, no answers — and gated by uncertainty at generation | TinyStories 32 MB free generation: BPB unchanged (1.794), 0/8 loops, **verbatim copying 9.6% → 0.2%**, longest repeat 61 → 19 B; 2.6 M prose-credited hits at mean gain 0.47 |
| **Optimisation before the long runs** (TinyStories 4 MB, 50k units, release): nursery scan made incremental; INT8 vs INT16 accumulators; blend / sequence-memory / prefetch fusions | 139.5 → **203 KB/s** (+46%), BPB 2.0035 → 1.9935, locks unchanged; INT16 accumulators: −0.3% BPB for −31% throughput, INT8 stays; release stage shares now bodies ~32%, readout + sequence memory 33%, learn 15%, address 11% |
| … TinyStories, agency off / fixed-focus control / on | BPB 2.90 / 2.92 / 2.95; on: holds focus 39.5 events on average, `hold` 88% |
| Same synth run, MSVC 2022 build on Windows | identical accuracy/loss to the GCC build; 470 KB/s train, 850 KB/s frozen |

The doc's success signature is visible in the 200k run: loss falls while novelty (0.71 → 0.03),
accumulator updates (506 → 160/event) and INT4 level changes (11.9 → 1.3/event) all fall as the
locks refine. MACs/event stay ≈ 5.6k because at these thresholds every admitted unit survives to
L6; early death is a live mechanism (it is what killed the 1/8 route in the bug story in
`docs/ARCHITECTURE.md`) but this task does not need it.

### Milestone D against the doc's success criteria (§15)

* BPB decreases — yes, and only with adaptive locks (frozen plateaus, no-memory is flat).
* Locks become selectively more informative — yes: ~100 consequence-driven splits per pass, each
  a tumbler bit that changed the conditional next-byte distribution; no births after pass 0.
* Activity remains bounded — yes: 1.0–1.05 admissions/byte, ≤ 3 active units, ~0.03 accepted
  neighbour messages/byte.
* MACs/byte declines after novelty settles — **no, and for a structural reason.** Death thresholds
  relative to each unit's own activity history (`death_rel=`) and stronger metabolic pressure
  (`resist_up=`) were added and swept (`scripts/sweep_death.sh`): MACs/byte falls exactly in
  proportion to bytes that get no prediction at all (e.g. `death_rel=40 resist_up=6`: 4.5k MACs,
  56% of bytes emit, BPB 4.9 vs 5.6k / 99% / 3.0). On this task admission is ~1 route body per
  byte, so early death has nothing redundant to prune — a dead body is an unpredicted byte.
  Declining MACs needs redundancy to prune first (several candidate bodies per event, or
  routed neighbours doing real work), not lower thresholds. Defaults stay `death_rel=0`.
* Native online training approaches 500–1,000 KB/s — **yes**: 527–546 KB/s on bytes after the
  accumulator/evidence vectorisation (was 235–360). See `scripts/gprof_bytes.sh`.

`scripts/milestone_d.sh` reproduces the three-way comparison; `data/` holds slices copied from
`C:\path\to\gdrive\training_data\TinyStories-train.txt`.

### Working memory and the 2M-token rolling window

See `docs/ARCHITECTURE.md` ("Working memory", "Rolling context"). Short version: the persistent
workspace is the one working-memory mechanism that moved BPB; the 2M window is achieved with
*episodic locks* — exact-key locks born on surprise with a one-shot L6 imprint, evicted
least-recently-hit when the unit pool fills. `laminar_needle` is the acceptance test
(`scripts/needle.sh`). It is associative recall (complete a key seen once anywhere in the
window), not attention.

```bash
# 2 MB rolling window on the doc's 200k population (≈ 40 s)
FILE=data/tinystories_13m.txt ./scripts/needle.sh 2097152 200000 400 readout_freeze=524288 exact_only=1
```

### Attention: the agency kernel

`cpp/include/laminar/agency.h` ports the Agkern agency kernel (competition, commitment,
consequence routes, DAN-gated three-factor credit, focus field) and wires it in as the organism's
attention: its tendencies move a Gaussian focus over the L1 input and choose the attended evidence
stream; its consequence is the organism's own prediction outcome. `scripts/agency.sh` (and
`NEEDLE=1 scripts/agency.sh`) run the off / control / on comparisons. Details and port decisions
in `docs/ARCHITECTURE.md`.

### Where it differs from the design doc's 92.19%

The doc reports 92.19% on the 8-context task with a Python/NumPy reference organism. This
reconstruction reaches the noise ceiling when no structural discovery is needed (1 context) and
discovers essentially the right number of routes with 8 contexts, but lands at ~78–79%: the
remaining errors sit in locks whose context split has not earned enough evidence (rare
symbol/context pairs in the generated stream) and in routes that split late and are still
relearning. The task generator itself (permutation maps per context, 64-symbol runs) is my
reconstruction of "32 symbols / 8 contexts / 8% noise"; the original's statistics are unknown.

## Layout

```
cpp/include/laminar/   config, rng, int4 kernels, address tree + locks + evidence, telemetry, tasks, organism
cpp/src/organism.cpp   hot path, local learning, structural maintenance
cpp/apps/              laminar_synth, laminar_bytes, laminar_bench
cpp/tests/             test_kernels
python/laminar_ref/    NumPy reference (same RNG, same integer semantics)
python/run_synth.py    reference driver + trace dump
python/tests/          compare_traces.py
scripts/               build_wsl.sh, build_msvc.ps1, bench.sh, parity.sh, sweep_*.sh, diag_synth.sh, profile_bytes.sh, fetch_tinystories.sh
docs/ARCHITECTURE.md   requirement → code map, open choices, known limitations
```

## Known limitations / next steps

* Hidden-layer local learning (`hidden=perturb`, paired ±δ node perturbation with a shared reward
  difference, `perturb_clean=1` for an unperturbed committed prediction) is implemented in both
  kernels and now unbiased — fixing it exposed a floor-vs-truncate rounding bias that also improved
  L6-only learning (synth 77.8 → 79.4%, bytes BPB 3.02 → 2.97). Swept over δ, reward gain and rate
  (`scripts/sweep_perturb.sh`, `scripts/diag_perturb.sh`, `scripts/perturb_bytes.sh`), it is
  **neutral** on both tasks: with one route body per event and the split rule giving each body a
  near-constant target, L6 alone suffices and the hidden layers have nothing to add. It costs 3×
  forward. The doc's own 200k organism used L6-only learning; that remains the default here.
* L4 gets no direct error signal (its consequence lands one event later).
* The continuous (μ/σ) form of the split rule is not implemented; the discrete prototype form is.
* Single-threaded by design (runbook §10: latency first). Parallel candidate bodies and background
  structural maintenance are the obvious next steps.
* AVX-512 / VNNI kernels are not written; the AVX2 kernel uses half its lanes on 8-byte loads.
* The raw-byte path is bound by the learning stage, not the unit bodies (`scripts/profile_bytes.sh`):
  with the shared 32×256 readout the L6 error saturates and the INT4 accumulators cross a level
  ~25 times per event. That is a learning-rate question for the byte regime and should be tuned on
  real TinyStories data (`scripts/fetch_tinystories.sh`), not on this repo's own documents.
* Surprise is measured relative to the stream's running error (see `docs/ARCHITECTURE.md`); the
  margin `surprise=` and `lr_anneal=` are the two knobs that trade compute for plasticity.
