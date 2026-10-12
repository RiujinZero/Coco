# Merged-frontier decoder for free generation — 25 September 2026

Item H of the research plan: reliable free generation from a trained BabyLM checkpoint **without changing any
weights**, with a decode-time search after "Interference Search" (Al-ameen, Bad Theory Labs, Sept 2026; notes in
[BODY_CIRCUIT_RESEARCH_LOG.md](BODY_CIRCUIT_RESEARCH_LOG.md)). The model runs frozen and every search branch runs in a
forked child, so no skill can regress by construction.

**Result.** On `gpu_motif` epoch 9 (82,176 fixed-motif bodies, held-out 1.850 bits/byte), the frontier decoder
(W=4, N=4, key=3) removes loops (0/20 outputs with a repeated word 4-gram, greedy 9/20) and non-words (99.5% valid
words, sampling 92.0%), and writes far more probable text than greedy or sampling (−0.336 nats/byte; greedy −0.485,
sampling −1.061), at ~88x the model compute of plain decoding (4–9 s per 200 bytes). But the likelihood judge finds that the
cheapest bytes are the ones the sequence memory can recall: **29% of its bytes are written under a ≥32-byte verbatim
match with training text, and different prompts converge on the same memorised Simple Wikipedia passages** (30% of
4-grams shared across prompts). A copy penalty in the judge (`copy=1`) cuts verbatim copying to 2% while staying
loop-free and more probable than greedy (−0.447). No decoder keeps the prompt's topic for long: the model closes the
prompt's line within a few words and starts a new line in another genre (CHILDES transcript, wiki boilerplate). A
prompt-dependence (PMI) judge does not change that ([below](#prompt-dependence-judge-pmi--25-september-2026)): the
model's predictions depend on the prompt only for the continuation's first 2–4 words, so beyond them the judge either
reduces to the likelihood judge (L ≤ 0.5) or writes salad (L = 1).

## The decoder (`cpp/apps/model_frontier.cpp`)

```
model_frontier DIR STATE_FILE PROMPT|@PROMPTS [mode=frontier|greedy|sample] [len=200] [W=4] [N=4] [T=0.8] [key=3]
               [vote=0.03] [rep=0] [copy=0] [pmi=0] [mem=1] [stop=len|sent] [minwords=0] [maxword=20] [seed=1] [workers=8]
               [out=FILE.jsonl] [vocab=FILES] [lean=1] [ident=1] [profile=0]
```
(`pmi`, `ident` and `profile`: see [Prompt-dependence judge](#prompt-dependence-judge-pmi--25-september-2026) below.)

The model is loaded exactly as `model_babylm_score` loads it (config.json `model_params`, `load_state`,
`set_mode(false, false)`), reads a newline and the prompt as text, then writes. Free-running steps use the public
equivalent of `TaskModel::generate`: `free_run = true`, `run_block({last, 0})`, `distribution()`, `commit_outcome(v)`
(the written byte is the model's own emission: context for the sequence memory, never a recallable instance).

- **greedy / sample**: the argmax, or `TaskModel::generate`'s temperature sampler.
- **frontier**: a *state* is a continuation (the bytes written after the prompt). Each level:
  1. **expand** — each live state (≤ W) is replayed in a forked child as the model's own output, which reproduces
     the exact state the model was in when it wrote those bytes; from there N proposals each write one **word** at
     temperature T (bytes until an alphanumeric run ends, ≤ 20 bytes), each from the same state (a forked grandchild
     each, the last in place). A proposal returns its bytes and their summed log-probability under the model's actual
     distribution.
  2. **merge** — proposals whose key (the last `key` words of prompt + continuation, lowercased; alphanumeric words,
     punctuation ignored) is equal are one state: it keeps the best-scoring text and pools the votes (distinct parents
     that proposed it).
  3. **memory** — keys expanded earlier in the generation (on any branch) are never expanded again; proposals reaching
     one are dropped. If every proposal of a level were blocked, the level would be relaxed (counted; it never
     happened in 380 generations).
  4. **judge** — score = mean log-prob per byte of the whole continuation + `vote`·(votes − 1) − `rep`·repeated word
     bigrams − `copy`·(fraction of its bytes written under a ≥ `copylen` = 32-byte verbatim sequence-memory match).
     The specified judge is `copy=0 rep=0`.
  5. keep the best W; stop when the best state reaches `len` bytes (`stop=sent`: or when it is finished by a newline or
     a second sentence end, counted after `minwords` words).

**Checks.** (i) For every one of the 380 generations below, the log-probability accumulated while decoding equals the
log-probability of replaying the output as the model's own emissions from the post-prompt state (to 1e-6), so a
replayed branch is exactly the branch that was scored. (ii) `lean=1` (default) constructs the model with `train=0`:
the buffers that only learning uses — Adam's four floats per synapse (6.9 GiB at 82k bodies) and the INT16
plasticity residual (0.9 GiB) — are never allocated (`load_state` does not read them; this checkpoint stores no
residual). Greedy, sample and frontier outputs, log-probs and event counts are **bit-identical** to `lean=0`, while
resident memory drops 8.11 → 0.35 GiB, load 11–14 → 1.3 s, fork + exit 347 → 17 ms and a 120-byte frontier sample
69 → 2.3 s. Each child dirties ≤ 12 MB; at most `workers=8` children run at once.

## Setup

- Models: `gpu_motif/epoch_9.state` (fixed motifs `1:256,2:16384,3:32768,4:32768`, 82,176 bodies, registers, 1 MB
  sequence window; held-out 1.850 bits/byte) and `ss10_adr/epoch_8.state` (address tree, 20,000 units; 1.951).
- Prompts: 20 BabyLM-style openings — story, dialogue and CHILDES lines (`scripts/frontier_prompts.txt`). Each output
  is 200 bytes (the frontier stops at the first word boundary at or past 200); newlines do not stop generation
  (`stop=len`), so every decoder writes the same length. One seed.
- Metrics per output, averaged over the 20 prompts: **loops** = outputs containing a repeated word 4-gram; strict loops
  = `genscore.h`'s rule (a recurring ≥ 64-byte span or > 30% repeated 8-grams); distinct-2/3 = unique / total word
  n-grams; **log-prob/byte** of the output from the post-prompt state, read as text (teacher forcing) and replayed as
  the model's own emissions (the decoding condition); valid words = words (≥ 2 letters) seen in the BabyLM training
  text; **copied** = bytes predicted under a ≥ 32-byte verbatim sequence-memory match; **cross-prompt** = fraction of
  an output's word 4-grams that also occur in the same decoder's output for a different prompt (an attractor every
  prompt falls into); wall time of the decode alone and model byte-events per sample.
- Timing caveat: the machine was shared with another session's GPU training run, BabyLM scorer and probe jobs (load
  average ~31 on 32 threads during the runs), so wall times vary 4–9 s for the same work; events/sample is the
  load-independent cost.

## Results — `gpu_motif` epoch 9

| decoder | loops | strict loops | distinct-2 | distinct-3 | log-prob/byte read | log-prob/byte own | valid words | copied | cross-prompt | s/sample | events/sample |
|---|---|---|---|---|---|---|---|---|---|---|---|
| greedy | 9/20 | 1/20 | 0.892 | 0.941 | −0.470 | −0.485 | 0.994 | 0.141 | 0.611 | 0.01 | 200 |
| sample T=0.8 | 0/20 | 0/20 | 0.991 | 1.000 | −1.061 | −1.061 | 0.920 | 0.012 | 0.003 | 0.01 | 200 |
| **frontier** W=4 N=4 key=3 T=0.8 | **0/20** | 0/20 | 0.970 | 1.000 | **−0.336** | **−0.336** | **0.995** | 0.288 | 0.296 | 4.2 | 17,539 |
| frontier + `copy=1` | 0/20 | 0/20 | 0.980 | 1.000 | −0.446 | −0.447 | 0.993 | **0.021** | 0.142 | 5.8 | 18,016 |

Greedy is largely prompt-independent: 9 of 20 prompts fall into the same invented attractor ("…the state leagues of
the service of the Prophet Music Canada…", not in the corpus), hence cross-prompt 0.61. Sampling has no attractor but
8% non-words. The frontier's text is fluent at the word level but is stitched from memorised training sentences —
"#2 on the Alternative Airplay, #4 on the Mainstream Rock, and #55 on the "Billboard" Hot 100 charts" (one line of
`simple_wiki.train.txt`) appears in 4 of the 20 outputs.

### What each part contributes (frontier variants, same model and prompts)

| variant | loops | distinct-2 | log-prob/byte own | valid words | copied | cross-prompt | merges / generation | blocked / generation | s/sample |
|---|---|---|---|---|---|---|---|---|---|
| specified: W=4 N=4 key=3 T=0.8 | 0/20 | 0.970 | −0.336 | 0.995 | 0.288 | 0.296 | 0.4 | 6.5 | 4.2 |
| no memory (`mem=0`) | **4/20** | 0.936 | −0.306 | 0.995 | 0.334 | 0.332 | 0.2 | 0 | 5.6 |
| memory on 6 words (`key=6`) | 2/20 | 0.957 | −0.317 | 0.997 | 0.310 | 0.326 | 0.1 | 2.6 | 5.6 |
| memory on 2 words (`key=2`) | 0/20 | 1.000 | −0.411 | 0.992 | 0.166 | 0.103 | 1.3 | 31.2 | 6.2 |
| proposals at T=1.0 | 0/20 | 0.980 | −0.493 | 0.997 | 0.117 | 0.092 | 0.3 | 2.1 | 8.0 |
| copy penalty 0.5 | 0/20 | 0.975 | −0.387 | 0.993 | 0.140 | 0.209 | 0.5 | 7.0 | 8.1 |
| copy penalty 1 | 0/20 | 0.980 | −0.447 | 0.993 | 0.021 | 0.142 | 0.2 | 5.8 | 5.8 |
| *(reference: sample T=1.0)* | 0/20 | 0.994 | −1.499 | 0.829 | 0.000 | 0.000 | – | – | 0.01 |

### Second model — `ss10_adr` epoch 8 (address tree)

| decoder | loops | strict loops | distinct-2 | log-prob/byte own | valid words | copied | cross-prompt | s/sample |
|---|---|---|---|---|---|---|---|---|
| greedy | 3/20 | 0/20 | 0.969 | −0.452 | 1.000 | 0.183 | 0.717 | 0.01 |
| sample T=0.8 | 0/20 | 0/20 | 0.991 | −1.063 | 0.945 | 0.000 | 0.004 | 0.01 |
| frontier W=4 N=4 key=3 | 0/20 | 0/20 | 0.981 | −0.413 | 0.999 | 0.082 | 0.157 | 8.7 |

### Single utterances (`stop=sent`: stop at a newline or a second sentence end, `gpu_motif`)

| decoder | mean bytes | loops | log-prob/byte own | valid words | copied |
|---|---|---|---|---|---|
| greedy | 116 | 4/20 | −0.517 | 0.949 | 0.085 |
| sample T=0.8 | 104 | 0/20 | −1.121 | 0.847 | 0.008 |
| frontier | 155 | 0/20 | −0.345 | 0.986 | 0.323 |

Only 1 of 20 frontier utterances ends at a newline (greedy 10, sampling 12); 9 stop at a second sentence end and 10
run to the 200-byte cap. `minwords=4` changed no frontier output and lengthened 1 greedy and 3 sampled outputs that
had ended within three words.

## Samples (same prompt, every decoder)

Complete outputs (200 bytes, the frontier's ending at the first word boundary past 200; the last word may be cut):

Prompt `Once upon a time there was`:

```
greedy            a state of the service of the Prophet Music Canada and the state leagues of the service of the
                  Proving the same time, they were no longer be princes and people are dancing in the world, and we
                  have b
sample T=0.8      a see the produced by this is why were in the United States. They beat the age of it. One at the
                  very strong as she was a close in her was, that way the John Mary]
                  *MOT:  well you need to make a crump
frontier          one of the side of the heart is sometimes known as the "Billboard" Hot 100 charts. The song was
                  certified 2× Platinum by Music Canada and 2× Princes and play in the United States on the sitting
                  the
frontier copy=1   one of the side of the heart is sometimes known for his brightened as they were no longer be private
                  for the Austrian national team as the beginning of August 1972 Summer Olympics.
                  = = = Lillias Robertson
ss10_adr frontier still hungry.
                  *CHI:  I don't want a windy.
                  *MOT:  you're going to make the mother doll speak.]
                  *CHI:  " yeah [yes]."
                  *MOT:  " if you want what?
                  *CHI:  I want some more.
                  *CHI:  Dolly.
                  *MOT:  oh you want to say
```

Prompt `*CHI:<tab>I want the`:

```
greedy            book "The Dream of the serekh.
                  *MOT:  what do you wanna say that the same time.
                  It can be a mistake.
                  *CHI:  I want the story has a district of Broye-Vully in the canton of Vaud in Switzerland.
                  = = = Pr
sample T=0.8      pie in the ary for going to her three know that name was a majid man�}?
                  *MOT:  how about the song, vocalist Adam Gontier said "It's and then you put that we have never had
                  to water really call it pass
frontier          book.
                  *CHI:  I don't know.
                  *MOT:  yeah it is!
                  *CHI:  [laughs].
                  *CHI:  what?
                  *MOT:  what is that?
                  *CHI:  yeah.
                  *MOT:  oh that's it?
                  *CHI:  yep.
                  *MOT:  what else do you want a pencil?
                  *CHI:  yeah.
                  *FAT:  what do
frontier copy=1   book.
                  *CHI:  I don't know.
                  *MOT:  look it up.
                  *CHI:  I hope you feel good.
                  *CHI:  what is the second single from their fingers with it.
                  *MOT:  there you go.
                  *MOT:  how do you spell have some of the longest reigning
ss10_adr frontier napkin.
                  *MOT:  what are you going to use the other side.
                  *MOT:  do you think we have to go with the sand.
                  *MOT:  is he crying?
                  *CHI:  yup [yes].
                  *MOT:  what is that?
                  *CHI:  tea.
                  *MOT:  oh.
                  *CHI:  and he says
```

Prompt `It was a cold morning and`:

```
greedy            the state leagues of the serekh.
                  Given the present to a three-year before the state of the service of the Prophet Music Canada and
                  the state of the service of the Proving the same time, they were no
sample T=0.8      the needles is an uning to fly as well, Robinson competed at the 1960feeling gests thought that
                  bicy, he ran to the people leaf showing a pecked at the age of 86.
                  It seem to hear the Gray from the re
frontier          the band and began to trusted the letters the same shape, not much pleased!" said that he was a
                  member of the superground for a Grammy Award in 1988 Summer Olympics.
                  = = = Look at it.
                  *MOT:  okay.
                  *MOT:
frontier copy=1   the band and began to trusted the letters the same shape, now that she had lived as the window and
                  the little gentleman was the things that would have been made the song after the last drink:"
                  "48 hours
ss10_adr frontier delight in the United States and the looked at her with marked attention. The rest of the school in
                  time for the great did not dare to look for the Scholarship."
                  Sir John and the other judges really
```

All 380 outputs of every run above, grouped by prompt: [FRONTIER_DECODER_OUTPUTS.txt](FRONTIER_DECODER_OUTPUTS.txt)
(bytes that are not valid UTF-8 shown as U+FFFD); raw per-run JSON lines in `logs/frontier/` (gitignored).

## Findings

1. **The memory removes the loops, not the merging.** Without it 4/20 outputs repeat a word 4-gram; with a 6-word key
   2/20; with the specified 3-word key 0/20 without a single relaxation. With key = k a repeated k-word sequence is
   impossible by construction, so loops = 0 is guaranteed for k ≤ 4; the `key=6` row shows the memory still pays at a
   setting where it is not. The cost is small: −0.03 nats/byte against `mem=0`.
2. **Merging almost never happens in free text.** 0.2–0.5 cross-parent merges per 200-byte generation at key=3 (0.1
   at key=6, 1.3 at key=2), and the chosen state had a single vote in 198 of the 200 frontier generations (the two
   exceptions at key=2), so the vote bonus is inert. In the paper merging carried most of the gain on Countdown, where
   different operation orders reach the same position; here W=4 sibling states differ in their last word and rarely
   reconverge inside a 3-word key. Coarser keys merge more but block more (key=2 blocks 31 proposals per generation
   and makes every word bigram unique).
3. **The likelihood judge turns search into recall.** The most probable continuation of almost any prompt is a
   sentence the sequence memory holds verbatim (its 1 MB instance window keeps the most recently read training text;
   the recurring passages are Simple Wikipedia song, election and athlete articles). Copied bytes: greedy 14%,
   frontier 29%, 33% without the memory. A copy
   penalty in the judge is the direct fix — `copy=1` cuts copying to 2% and cross-prompt sharing from 0.30 to 0.14
   for 0.11 nats/byte, still above greedy; proposals at T=1.0 have a similar but costlier effect (12% copied,
   −0.493). The remaining shared phrases ("Austrian national team", "police chief of Springfield") are recalls
   shorter than 32 bytes.
4. **Prompt adherence is weak in every decoder.** The model ends the prompt's line within a few words and starts a new
   line from another BabyLM source; the frontier chooses whichever source is most predictable (CHILDES exchanges, wiki
   boilerplate). A judge that rewards dependence on the prompt (log p(continuation | prompt) − log p(continuation |
   no prompt), one more replay per state) is the natural next step; decoding cannot add coherence the model lacks.
5. **The frontier rarely ends a line.** Under a mean-per-byte judge a state that ended its line scores below one that
   continues into recallable text: with the specified stop rule half the utterances run to the length cap, so they are
   longer than greedy's (155 bytes vs 116) and 32% copied.
6. **Second model.** The address-tree model's frontier is also loop-free with 99.9% valid words but gains less
   likelihood over greedy (−0.413 vs −0.452) and copies less (8%) — its window recalls less.
7. **Cost.** ~590 proposals over ~37 levels per generation, each a fork of the frozen model; 17.5k model byte-events per 200 bytes
   (88x plain decoding: the continuation is replayed at every level, O(len²), minor at 200 bytes) and 4–9 s of wall
   time, dominated by forks. Keeping state processes alive across levels (no replay) and forking proposals only
   where their bytes diverge would cut both.

## Problem found outside the decoder

Every read-only tool that loads the 82k-body checkpoint with the stored `model_params` gets an **8.1 GiB** process
(the constructor allocates `acc16` whenever `p.train`, and `ad_sh/ad_m/ad_v/ad_g` whenever `p.train && p.adam`, before
`load_state` runs), and each fork of it costs ~350 ms in page-table copies — `model_babylm_score` forks once per
candidate. Constructing with `train=0` appended to the params (as `model_frontier` does) is bit-identical, 0.35 GiB
and 20x cheaper to fork. A cleaner fix in `organism.cpp` (not made here; that file is being edited elsewhere):
allocate these buffers when learning first runs.

## Reproduce

```
# WSL (no cmake); the eval script builds build-task/model_frontier if needed
cd cpp && g++ -std=c++20 -O2 -march=native -DNDEBUG -Iinclude apps/model_frontier.cpp src/organism.cpp -o ../build-task/model_frontier
build-task/model_frontier /home/user/babylm/runs/gpu_motif /home/user/babylm/runs/gpu_motif/epoch_9.state "Once upon a time there was" mode=greedy,sample,frontier copy=1
bash scripts/frontier_eval.sh                      # every run above (~25 min on a shared machine) -> logs/frontier/
python3 scripts/frontier_report.py logs/frontier/motif_main.jsonl ...     # tables (adds the cross-prompt metric)
python3 scripts/frontier_report.py --dump logs/frontier/motif_main.jsonl ...   # all outputs by prompt
```

## Prompt-dependence judge (PMI) — 25 September 2026

Finding 4 above proposed a judge that rewards dependence on the prompt. It is implemented as `pmi=L` and evaluated on
two checkpoints with new topic measures, so that "stays on topic" is measured rather than eyeballed.

**Result.** The prompt-dependence judge does not keep the decoder on topic, and with these models it cannot: the
prompt stops changing the model's predictions after the continuation's first 2–4 words. On both checkpoints the prompt
moves the log-prob of the first 4 output bytes by 2.2–3.4 nats, of bytes 8–15 by 0.04–0.14 nats and of bytes from 32
on by ≤ 0.016 nats. Beyond that point log p(byte | prompt, continuation) = log p(byte | continuation), so the PMI term
is the same for every candidate word.

- **L = 0.3–0.5:** the judge becomes (1 − L)·likelihood plus an opening bonus that fades with length, and the frontier
  writes what it wrote before. On `gpu_motif` at pmi=0.5: lp −0.344 vs −0.336, 22% copied vs 29%, prompt content
  words reused 0.000 vs 0.036 by chance, late gain 0, second half identifies its prompt 1/20 times.
- **L = 1:** the judge is blind to likelihood after the first words. It keeps the shortest continuations and writes
  word salad (lp −1.30 and −1.42, below sampling). It is also the only setting whose second half identifies its
  prompt above chance (5/20 and 8/20; chance 1/20).

No setting brings back loops (0/20 everywhere) or non-words (98.7–100% valid words), and with copy=1 copying stays at
0.2–0.8%. The cost is 2x the model compute of the plain frontier. For every fluent decoder, every measure of the topic
beyond the opening words stays near chance: prompt content words, late gain and late identification. Staying on topic
needs a model that remembers the prompt: a judge can only choose among continuations the model itself predicts.

### The judge

`pmi=L` (default 0 = off) subtracts from step 4's score L × the mean log-prob per byte of the **same continuation
replayed without the prompt**, from the state the root process keeps after the initial newline (the post-load state
before the prompt):

score = [log p(cont | prompt) − L · log p(cont | no prompt)] / bytes + vote·(votes − 1) − rep·… − copy·…

L = 1 is pointwise mutual information per byte; L < 1 keeps (1 − L) of the likelihood judge. Both log-probs are the
model's own distribution in the decoding condition (each earlier byte of the continuation replayed as its own
emission), and both accumulate word by word along a state's history. With `pmi > 0` the decode process stays in the
no-prompt state, and each expansion child does two things:
- It forks a grandchild that reads the prompt, replays the continuation and makes the N proposals, exactly as before.
- It replays the continuation without the prompt, then scores every distinct proposed word from there (a forked
  great-grandchild each, the last in place).

The vote, rep, copy, mem, key and stop options are unchanged. The final scoring of every output adds replays without
the prompt and after each of the 19 other prompts (`ident=1`). `profile=1` writes the per-byte log-prob change the
prompt makes (`dlp`) into the JSON line.

**Checks.**
1. `pmi=0` is the previous decoder, bit for bit. All 80 `gpu_motif` outputs of the main and `copy=1` runs (greedy,
   sample, frontier and frontier copy=1 on the 20 prompts) are identical to the runs in the sections above in every
   field both record: output bytes, forced and self log-probs, match statistics, and event, level, proposal, merge,
   block and vote counts (`scripts/frontier_compare.py`). On `gpu_m20` the new build equals the previous binary on 9
   outputs (3 prompts × greedy, sample, frontier, len 80).
2. For every `pmi > 0` generation, the no-prompt log-prob accumulated while decoding equals a fresh replay of the
   output without the prompt (to 1e-6), just as the with-prompt log-prob equals its replay after the prompt. So both
   chains score exactly the branch that was chosen.
3. The profile runs (greedy, sample and frontier again with `profile=1 ident=0`, built from the final source) reproduce
   the main runs of both checkpoints exactly (2 × 60 outputs, 40 shared fields each). The final source, rebuilt with
   the one-line `g++` command of [Reproduce](#reproduce), reproduces the `gpu_m20` pmi=0.5 and pmi=1 runs exactly
   (2 × 20 outputs, 44 fields each, including the no-prompt log-probs, gains, reaches and ranks).

### Topic measures

- **topic words**: the fraction of the prompt's content words whose stem reappears in the output. Content words have
  ≥ 3 letters and are not in a stop list of function words, auxiliaries, pronouns and CHILDES speaker codes; the stem
  strips a plural -s, then -ing or -ed. Prompt 5 (`*MOT: what are you doing with`) has no content words and is left
  out. The chance level is the same prompt's content words counted in the outputs the same decoder wrote for the
  other 19 prompts ("topic words, other prompts' outputs"). The measure is also taken over the output's second half
  alone.
- **gain**: the output's log-prob per byte replayed after its prompt minus replayed without it. This is the judge's
  PMI term at L = 1. **Late gain** is the same over the output's second half.
- **reach**: 1 + the last output byte whose log-prob the prompt changes by more than 0.01 nats (median over prompts).
  "Reach any" counts any change at all.
- **ident@1**: the output is more probable after its own prompt than after each of the other 19 prompts. Ties count
  against it, so a byte span the prompt no longer affects ranks last. Chance is 1/20. **ident@1 late**: the same for
  the second half alone.
- **echo**: outputs that repeat a word trigram of the prompt. **Line 1 bytes**: bytes before the output's first
  newline.

### Setup

- **Prompts and decoding:** the same 20 prompts (`scripts/frontier_prompts.txt`), 200 bytes (`stop=len`), one seed,
  W=4 N=4 key=3 T=0.8, `lean=1`.
- **Workers:** `workers=4`, so ≤ 4 computing children; the runs above used 8.
- **Machine:** shared with GPU trainings and probes (load 11–38 on 32 threads), so wall times compare only within
  this section.
- **Checkpoints:**
  - `gpu_motif` epoch 9, as above.
  - `gpu_m20` epoch 1: fixed motifs `1:256,2:4096,3:8192,4:8192`, 20,736 bodies, one epoch of the multi-stream GPU
    trainer, held-out 1.966 bits/byte.
- **Decoders:** greedy, sample T=0.8, frontier, frontier copy=1, frontier pmi=0.3 / 0.5 / 1, and frontier pmi=0.5
  copy=1.

### Results — `gpu_motif` epoch 9

| decoder | loops | strict loops | distinct-2 | distinct-3 | valid words | lp read | lp own | copied | cross-prompt | s/sample | events/sample |
|---|---|---|---|---|---|---|---|---|---|---|---|
| greedy | 9/20 | 1/20 | 0.892 | 0.941 | 0.994 | −0.470 | −0.485 | 0.141 | 0.611 | 0.01 | 200 |
| sample T=0.8 | 0/20 | 0/20 | 0.991 | 1.000 | 0.920 | −1.061 | −1.061 | 0.012 | 0.003 | 0.01 | 200 |
| frontier | 0/20 | 0/20 | 0.970 | 1.000 | 0.995 | −0.336 | −0.336 | 0.288 | 0.296 | 5.86 | 17,539 |
| frontier copy=1 | 0/20 | 0/20 | 0.980 | 1.000 | 0.993 | −0.446 | −0.447 | 0.021 | 0.142 | 6.91 | 18,016 |
| frontier pmi=0.3 | 0/20 | 0/20 | 0.979 | 1.000 | 0.996 | −0.348 | −0.349 | 0.218 | 0.234 | 15.55 | 38,288 |
| frontier pmi=0.5 | 0/20 | 0/20 | 0.977 | 1.000 | 0.997 | −0.343 | −0.344 | 0.220 | 0.325 | 15.98 | 37,707 |
| frontier pmi=1 | 0/20 | 0/20 | 0.966 | 1.000 | 0.987 | −1.295 | −1.295 | 0.003 | 0.022 | 29.66 | 60,743 |
| frontier pmi=0.5 copy=1 | 0/20 | 0/20 | 0.983 | 1.000 | 0.995 | −0.467 | −0.467 | 0.008 | 0.121 | 18.61 | 39,941 |

| decoder | topic words (own / other prompts' outputs) | topic words, 2nd half | gain | late gain | reach > 0.01 (median) | reach any (median) | ident@1 | ident@1 late | line 1 bytes | echo |
|---|---|---|---|---|---|---|---|---|---|---|
| greedy | 0.105 / 0.025 | 0.105 | 0.061 | 0.001 | 26 | 200 | 19/20 | 3/20 | 125 | 1/20 |
| sample T=0.8 | 0.000 / 0.047 | 0.000 | 0.047 | 0.000 | 97 | 200 | 11/20 | 3/20 | 108 | 0/20 |
| frontier | 0.079 / 0.041 | 0.079 | 0.057 | 0.000 | 16 | 200 | 18/20 | 2/20 | 125 | 1/20 |
| frontier copy=1 | 0.000 / 0.058 | 0.000 | 0.052 | 0.000 | 15 | 201 | 18/20 | 1/20 | 116 | 1/20 |
| frontier pmi=0.3 | 0.053 / 0.035 | 0.053 | 0.061 | 0.002 | 27 | 198 | 17/20 | 1/20 | 121 | 1/20 |
| frontier pmi=0.5 | 0.000 / 0.036 | 0.000 | 0.060 | 0.000 | 14 | 168 | 16/20 | 1/20 | 116 | 0/20 |
| frontier pmi=1 | 0.175 / 0.067 | 0.053 | 0.072 | 0.003 | 120 | 201 | 20/20 | 5/20 | 111 | 1/20 |
| frontier pmi=0.5 copy=1 | 0.105 / 0.077 | 0.000 | 0.059 | 0.000 | 22 | 167 | 18/20 | 0/20 | 111 | 1/20 |

(The greedy, sample, frontier and frontier copy=1 rows of the first table are the runs of the sections above, now
reproduced bit for bit.)

### Results — `gpu_m20` epoch 1

| decoder | loops | strict loops | distinct-2 | distinct-3 | valid words | lp read | lp own | copied | cross-prompt | s/sample | events/sample |
|---|---|---|---|---|---|---|---|---|---|---|---|
| greedy | 1/20 | 0/20 | 0.962 | 0.990 | 1.000 | −0.601 | −0.607 | 0.064 | 0.749 | 0.01 | 200 |
| sample T=0.8 | 0/20 | 0/20 | 0.995 | 1.000 | 0.924 | −1.157 | −1.157 | 0.000 | 0.000 | 0.01 | 200 |
| frontier | 0/20 | 0/20 | 0.993 | 1.000 | 0.997 | −0.446 | −0.447 | 0.078 | 0.157 | 7.89 | 18,781 |
| frontier copy=1 | 0/20 | 0/20 | 0.991 | 1.000 | 0.994 | −0.480 | −0.480 | 0.013 | 0.142 | 8.01 | 18,788 |
| frontier pmi=0.3 | 0/20 | 0/20 | 0.984 | 1.000 | 1.000 | −0.449 | −0.450 | 0.118 | 0.204 | 14.56 | 39,004 |
| frontier pmi=0.5 | 0/20 | 0/20 | 0.991 | 1.000 | 0.999 | −0.460 | −0.460 | 0.082 | 0.172 | 15.07 | 40,941 |
| frontier pmi=1 | 0/20 | 0/20 | 0.968 | 1.000 | 0.990 | −1.421 | −1.420 | 0.000 | 0.019 | 18.14 | 61,392 |
| frontier pmi=0.5 copy=1 | 0/20 | 0/20 | 0.989 | 1.000 | 0.999 | −0.496 | −0.496 | 0.002 | 0.118 | 10.20 | 40,794 |

| decoder | topic words (own / other prompts' outputs) | topic words, 2nd half | gain | late gain | reach > 0.01 (median) | reach any (median) | ident@1 | ident@1 late | line 1 bytes | echo |
|---|---|---|---|---|---|---|---|---|---|---|
| greedy | 0.105 / 0.088 | 0.053 | 0.078 | −0.001 | 48 | 48 | 18/20 | 0/20 | 69 | 0/20 |
| sample T=0.8 | 0.079 / 0.038 | 0.026 | 0.066 | 0.000 | 48 | 48 | 15/20 | 0/20 | 42 | 0/20 |
| frontier | 0.000 / 0.062 | 0.000 | 0.066 | 0.000 | 14 | 16 | 13/20 | 0/20 | 71 | 0/20 |
| frontier copy=1 | 0.000 / 0.055 | 0.000 | 0.066 | 0.000 | 15 | 20 | 14/20 | 0/20 | 63 | 0/20 |
| frontier pmi=0.3 | 0.079 / 0.061 | 0.053 | 0.073 | 0.000 | 16 | 28 | 17/20 | 0/20 | 52 | 0/20 |
| frontier pmi=0.5 | 0.053 / 0.065 | 0.053 | 0.073 | 0.000 | 13 | 14 | 17/20 | 0/20 | 60 | 0/20 |
| frontier pmi=1 | 0.228 / 0.051 | 0.053 | 0.077 | 0.004 | 154 | 159 | 19/20 | 8/20 | 59 | 0/20 |
| frontier pmi=0.5 copy=1 | 0.158 / 0.095 | 0.158 | 0.073 | 0.000 | 14 | 16 | 17/20 | 1/20 | 67 | 0/20 |

The younger, smaller model behaves like `gpu_motif` in every respect that matters here:
- Its likelihood frontier is loop-free with 99.7% valid words.
- It copies less (7.8%): a 20k-body model trained one epoch recalls less of its window.
- Its greedy decoder has the strongest attractor of all (cross-prompt 0.75): the same "compound time and er try not
  to muddle the family" line after most prompts.

### How far the prompt reaches

The table gives the mean |log p(byte | prompt, …) − log p(byte | no prompt, …)| over the 20 outputs, by the byte's
position in the output (both replays are the model's own emissions). In brackets: the share of bytes changed by more
than 0.01 nats. Source: `profile=1` runs.

| model | decoder | bytes 0–3 | bytes 4–7 | bytes 8–15 | bytes 16–31 | bytes 32–63 | bytes 64–127 | bytes 128–199 |
|---|---|---|---|---|---|---|---|---|
| `gpu_motif` | greedy | 2.378 (100%) | 0.417 (89%) | 0.140 (51%) | 0.017 (22%) | 0.016 (6%) | 0.003 (2%) | 0.001 (1%) |
| `gpu_motif` | sample T=0.8 | 2.195 (100%) | 0.389 (76%) | 0.041 (24%) | 0.010 (11%) | 0.011 (9%) | 0.005 (3%) | 0.001 (2%) |
| `gpu_motif` | frontier | 2.395 (99%) | 0.399 (84%) | 0.080 (38%) | 0.024 (16%) | 0.001 (2%) | 0.000 (0%) | 0.001 (1%) |
| `gpu_m20` | greedy | 3.365 (99%) | 0.459 (86%) | 0.107 (45%) | 0.007 (7%) | 0.001 (2%) | 0.002 (2%) | 0.002 (3%) |
| `gpu_m20` | sample T=0.8 | 3.163 (100%) | 0.294 (74%) | 0.086 (25%) | 0.002 (1%) | 0.001 (1%) | 0.000 (0%) | 0.001 (0%) |
| `gpu_m20` | frontier | 3.111 (98%) | 0.351 (80%) | 0.071 (29%) | 0.001 (1%) | 0.000 (0%) | 0.000 (0%) | 0.000 (0%) |

The prompt matters most at the first bytes. Most of that is the line start: after a bare newline, a space or a
lowercase continuation is improbable. Its effect falls to 0.3–0.5 nats at bytes 4–7 and 0.04–0.14 at bytes 8–15, and
it is almost nothing from byte 32 on. The model's predictions of a continuation depend on the prompt for about its
first 2–4 words; after that they depend only on the continuation itself (and on the sequence memory's recall).
- On `gpu_m20` the two replays predict every later byte identically after a median of 14–48 bytes ("reach any" close
  to "reach > 0.01").
- On `gpu_motif` a residual of less than 0.01 nats persists to the end ("reach any" ≈ 200), but it never matters.

This limits what a prompt-dependence judge can do. Beyond the reach, log p(byte | prompt, continuation) =
log p(byte | continuation), so every word written there adds only (1 − L) × its log-prob to the judge's numerator.

- **L < 1.** The words after the first few are ranked by likelihood, as without the judge. The prompt only re-ranks
  the opening words, through a bonus L·G/n (G = the opening gain) that all of a state's descendants share and that
  fades as n grows. So the frontier with pmi=0.3/0.5 writes the same kind of text as without it:
  - the same recall (`gpu_motif` 22% copied, `gpu_m20` 8–12%);
  - the same attractors (cross-prompt 0.17–0.33);
  - the prompt's line closed as early (first newline after 52–121 bytes vs 71–125).
- **L = 1.** Every word beyond the reach adds exactly 0 (`gpu_m20`) or noise (`gpu_motif`). A state's score becomes
  G/n, and the frontier keeps the shortest continuations:
  - 58–59 levels instead of 37–41;
  - 2.3–2.4 letters per word instead of 3.7–4.1;
  - 58–62% of words with one or two letters, instead of 22–28%.

  The output is salad made of real words (98.7–99.0% valid) that never loops. Its log-probs are far below every other
  decoder's (−1.30 and −1.42). Salad keeps the state sensitive to the prompt (reach 120–154 bytes), so it is the only
  setting whose second half identifies its prompt more often than chance. Its topic words exceed chance only in the
  opening (`gpu_m20` 0.23 vs 0.05, second half 0.05).

### Samples (two prompts, every decoder, both models)

Complete outputs; the frontier ends at the first word boundary past 200 bytes, so the last word may be cut.

Prompt `Once upon a time there was`:

```
motif greedy                   a state of the service of the Prophet Music Canada and the state leagues of the service of the
                               Proving the same time, they were no longer be princes and people are dancing in the world, and we
                               have b
motif sample T=0.8             a see the produced by this is why were in the United States. They beat the age of it. One at the
                               very strong as she was a close in her was, that way the John Mary]
                               *MOT:  well you need to make a crump
motif frontier                 one of the side of the heart is sometimes known as the "Billboard" Hot 100 charts. The song was
                               certified 2× Platinum by Music Canada and 2× Princes and play in the United States on the sitting
                               the
motif frontier copy=1          one of the side of the heart is sometimes known for his brightened as they were no longer be private
                               for the Austrian national team as the beginning of August 1972 Summer Olympics.
                               = = = Lillias Robertson
motif frontier pmi=0.5         a member of the supergroup French and moved about the new in the department of Territoire de
                               Belfort.
                               *MOT:  where did you see if you want.
                               *CHI:  okay.
                               *MOT:  what do you wanna say that night.
                               *MAR:  yeah.
motif frontier pmi=1           a lot of for a media on 1 I was, but I do to say, I go on to his stay her name on me, and as a ran
                               out of a king as I had to go to my led to see it to #1 on the way in a turns in the you go.
                               I was a
motif frontier pmi=0.5 copy=1  a lot of for a long time to be very different for the song was release in the television series in
                               his father to the police chief of Springfield. He played as a goalkeeper for a moment to see
                               anything
m20 greedy                     a war.
                               I was a bit of a bog and the compound time and er try not to muddle the family.
                               He was shot was a bit hairy of the super power from the lady and retired to the city and a people
                               who lived ther
m20 sample T=0.8               nothing seven times dunpude, as you know.
                               Lind of me.
                               So this touched so will be surned to the sat could have the laws of now.
                               Had the wation for and give to the results of this to Germine policeman,
m20 frontier                   supposed to have been here.
                               What is this?
                               Right.
                               Can I have some cash?
                               What are you gonna do, he was trying to get off with her.
                               Aye.
                               You know they're dotted beats.
                               As of course they were not particular
m20 frontier copy=1            supposed to have been here.
                               What is thing to do with all the others in the United States totally unchecked.
                               What do you think a diminished fourth when they said, that it could've had this travely information
                               of
m20 frontier pmi=0.5           supposed to have something to eat.
                               I don't have to worry about the octave.
                               You can have a minor third when they were fighting wild animals.
                               When the time for a lot of the intervals.
                               I can't remember
m20 frontier pmi=1             no shall be no run we can be a very er the zo on me?
                               He old be a bit to be padam. A Roes are a time twelfth is is a big sum an I go to hims to me that it
                               is to lot in a er the boy.
                               I can cracy there
m20 frontier pmi=0.5 copy=1    supposed to have been on the Soviet Union and therefore it was about there are two dotted beats, and
                               they were fighting with her.
                               Anyway, and I weren't gonna know what you're gonna have  doing it in
```

Prompt `It was a cold morning and`:

```
motif greedy                   the state leagues of the serekh.
                               Given the present to a three-year before the state of the service of the Prophet Music Canada and
                               the state of the service of the Proving the same time, they were no
motif sample T=0.8             the needles is an uning to fly as well, Robinson competed at the 1960feeling gests thought that
                               bicy, he ran to the people leaf showing a pecked at the age of 86.
                               It seem to hear the Gray from the re
motif frontier                 the band and began to trusted the letters the same shape, not much pleased!" said that he was a
                               member of the superground for a Grammy Award in 1988 Summer Olympics.
                               = = = Look at it.
                               *MOT:  okay.
                               *MOT:
motif frontier copy=1          the band and began to trusted the letters the same shape, now that she had lived as the window and
                               the little gentleman was the things that would have been made the song after the last drink:"
                               "48 hours
motif frontier pmi=0.5         the band and began to get to see you sure you got a constituency of Denmark. It represents the
                               region of Zealand and elects 20 years she has been in the United Kingdom and Australia, number 5 in
                               New
motif frontier pmi=1           the two not as I can to him to put it to a-ha at a rure I am to a child, for it was 6] you late of
                               it, we see and as far and as eat it to add to be so back as a local to her as far as of the part of
motif frontier pmi=0.5 copy=1  the band and began to get to see you sure you got a little cream on your money is something to me.
                               *CHI:  or Greg.
                               *CHI:  yeah.
                               *MOT:  okay.
                               *MOT:  where did you do.
                               *CHI:  I want some spoken languages. It
m20 greedy                     the compound time and er try not to muddle the family.
                               He was shot was a common market economy in some ways is the story of the West and the compound time
                               and the compound time and the come in the wo
m20 sample T=0.8               the money was repay.
                               Fine.
                               - and I've tean the street nappened the hard.
                               Then Pout this mess our of all so special.
                               The particularly as before call, I think therefore it benelaus part of best believe
m20 frontier                   that you leave Pure?
                               - I don't remember feeling more than anything is that the time better that was talking about this
                               was suggesting we give them an enormous and some of them.
                               You might be right.
                               I thought
m20 frontier copy=1            that you leave Pure?
                               - I don't remember exactly.
                               I have to take a break.
                               - What happened to Denise?
                               - I know what you can have six sixteen would have been here.
                               We're gonna have to go through the minor
m20 frontier pmi=0.5           that you do not going to talk to you about it at all, and his ears ringing still with the merred to
                               go back to the dentist thing to help you until we still got the perfect fourth and fifth.
                               You have
m20 frontier pmi=1             that you do not a day was a set up to do as a ks it an his up to  all.
                               It is a bug za, a, a,and it at I get it a week at the age of the I was so ill.
                               I had to a gid to go as I can we do with a cold to
m20 frontier pmi=0.5 copy=1    that you do not going to talk to you about you've got to remember the German ones.
                               But I've never had to tell you feel about what's better.
                               What about nineteen hundred years before Christ was born, and
```

All 320 outputs of the twelve runs (8 decoders × 2 models), grouped by prompt:
[FRONTIER_PMI_OUTPUTS.txt](FRONTIER_PMI_OUTPUTS.txt). Raw per-run JSON lines are in `logs/frontier_pmi/`
(gitignored).

### Findings

1. **A prompt-dependence judge does not keep the decoder on topic.** With L = 0.3 or 0.5, every measure of the topic
   beyond the opening words stays where the likelihood frontier has it, near chance:
   - Prompt content words reappear at about the rate they appear in other prompts' outputs (pmi=0.5: `gpu_motif`
     0.000 vs 0.036, `gpu_m20` 0.053 vs 0.065).
   - The second half gains nothing from the prompt (|late gain| ≤ 0.002 nats/byte).
   - The second half identifies its prompt 0–1 times in 20 (chance: 1).

   The whole-output gain, the judge's own term, rises by only 0.003–0.007 nats/byte, because that term lives in the
   first ~16 bytes. Identifying the prompt from the whole output is easy for every decoder (11–20 of 20), because it
   rests on those same opening bytes.
2. **It brings back no loops or non-words, and it does not stop copying.**
   - pmi=0.3/0.5 on both models: 0/20 loops and 99.5–100% valid words, at a small likelihood cost (pmi=0.5:
     `gpu_motif` −0.344 vs −0.336, `gpu_m20` −0.460 vs −0.447).
   - Copying goes down on `gpu_motif` (29% → 22%) but up on `gpu_m20` (7.8% → 8.2–11.8%). A passage recalled after
     the reach is as probable without the prompt as with it, so the judge neither favours nor penalises it.
   - `copy=1` remains the direct fix, and it combines with pmi (0.8% and 0.2% copied, lp −0.467 and −0.496).
3. **Pure PMI (L = 1) destroys fluency.** Once the prompt no longer matters, the judge can only prefer shorter
   continuations. The result is salad, mostly one- and two-letter words, at −1.30 / −1.42 nats/byte, below sampling.
   That buys the only above-chance late identification of the prompt (5/20 and 8/20).
4. **The limit is the model's memory of the prompt, not the judge.** The model's predictions depend on the prompt for
   2–4 words, so no decoder that re-ranks those predictions can carry the prompt further. Two ways forward:
   - A judge that brings the prompt back into view: for example, score each new word by how much re-reading the
     prompt just before the last few bytes raises its probability. That costs one more replay per state. It is
     untested here, and the prompt-plus-fragment junction is itself out of distribution.
   - A model whose state holds the topic for longer.

   The per-byte profile (`profile=1`: greedy outputs and two replays each, about a minute) is a cheap probe of that
   memory. It would measure a skill a circuit change should lengthen; held-out bits/byte does not measure it.
5. **Cost.** pmi > 0 doubles the model compute of the plain frontier:
   - 37.7–40.9k events per 200 bytes vs 17.5–18.8k: two replays per expansion, the prompt re-read in each, and every
     distinct proposal scored twice.
   - 15–16 s vs 6–8 s per sample at `workers=4`.
   - L = 1 adds 1.5x more levels on top (61k events).

### Reproduce

```
# WSL; builds build-task/model_frontier if needed; ~65 min on a shared machine -> logs/frontier_pmi/
bash scripts/frontier_pmi_eval.sh
REF=<logs/frontier of the runs above> bash scripts/frontier_pmi_eval.sh     # also: pmi=0 == the runs made before the judge
python3 scripts/frontier_report.py --topic logs/frontier_pmi/motif_main.jsonl ...        # quality and topic tables
python3 scripts/frontier_report.py --profile logs/frontier_pmi/motif_profile.jsonl logs/frontier_pmi/m20_profile.jsonl
python3 scripts/frontier_compare.py A.jsonl B.jsonl    # same outputs, log-probs and counts (all fields but time)?
build-task/model_frontier /home/user/babylm/runs/gpu_m20 /home/user/babylm/runs/gpu_m20/epoch_1.state "The cat is a" mode=frontier pmi=0.5 copy=1
```
