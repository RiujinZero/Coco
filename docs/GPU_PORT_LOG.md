# GPU port — log (2026-09-25)

Follows `docs/GPU_PORT_HANDOFF.md`. Branch `claude/gpu-port-handoff-5bb30b` (fast-forwarded to `0861b8b`), not pushed.

## Profile (CPU, one process, BabyLM Adam + registers config, 2 MB)
`apps/gpu_profile.cpp`: 18.4k B/s. Learning 69%, address + agency 18%, readout + sequence memory 8%, body forward ~4%.
About one body runs per byte, so a single-stream GPU port can only lose. The port is the multi-stream design.

## Design
- `include/laminar/multistream.h`, `src/multistream.cpp`: S byte streams in lockstep share one population (bodies, locks,
  readout, register readout, write head, sequence gate: an `Organism` used as the container, `friend class MultiStream`).
  Per stream: history, workspace memory, registers, sequence memory, agency kernel, running loss/prior, TC ring.
  Phases: P1 address/agency (threads), S1 births/hits (serial), P2 forwards (threads, CPU INT4 kernels), S2 thresholds,
  P3 readout/seq blend/loss/readout gradients (threads), S3 readout Adam + `Learner::step`, S4 metabolic/episodic/evidence,
  P4 workspace/head/seq capture (threads). Serial phases walk streams in index order, so S=1 is Organism's order.
- `include/laminar/batch_learner.h`: the learner interface. `src/learn_ref.cpp` = CPU reference (organism.cpp's
  arithmetic in its order); `src/gpu/learn_gpu.cu` = CUDA learner (Adam state 16 B/synapse and the TC rings on the GPU;
  kernels: body backward + gradient, Adam step, ring credit, store); `src/learn_dual.cpp` = both, compared every step.
- Minibatch semantics at S>1: all streams of a step see the step's starting weights; a body used by several streams in a
  step takes one Adam step over all its uses (per-body batch can exceed adam_k for hot bodies); readout steps stay one
  per ro_adam_k events.
- Supported config only (constructor throws otherwise): the BabyLM Adam path, hops=0, no recirculation/persist/homeo/market.

## Build (WSL)
`build-gpu/build.sh [exact|gpu|all]`. CUDA 12.8 needs host g++-13 (`-ccbin g++-13`) and, with glibc 2.43, a patched
copy of the CUDA headers (`build-gpu/cuda-inc`, `noexcept(true)` on rsqrt/sinpi/cospi in crt/math_functions.h); the
system CUDA install is untouched. The exact-check binary is built with `-ffp-contract=off`.

## Correctness
- `ms_check exact`: S=1 + CPU reference vs Organism: **5,242,880 events bit-identical** (every loss and prediction; weights
  and population identical after each of 80 blocks; crosses the readout freeze at 4M events and the population cap).
- `ms_check dual` (S=1, 1 MB): GPU vs reference, 0 differing INT4 nibbles in 153M compared (5.5k Adam-stepped bodies),
  shadow max |diff| 6e-4 grid units (mean 5e-9), ring error rel diff 2e-9, head gradients 9e-6.
- At S=128 GPU and reference runs give the same loss curve (2.2513 vs 2.2496 bits at 2 MB).
- Not yet done: skills (BLiMP / supplement / entity) of a multi-stream run vs the CPU run; agree_probe under S>1.

## Speed (`ms_check speed`, 12 threads, machine shared with the research runs)
| streams | learner | B/s | vs CPU baseline 18.4k |
|---|---|---|---|
| 1 | GPU | 2.0k | 0.11x (one GPU round trip per byte) |
| 16 | GPU | 20.2k | 1.1x |
| 64 | GPU | 58.1k | 3.2x |
| 128 | GPU | 60.0k (re-run under load: 38.1k) | 3.3x (2.1x) |
| 128 | CPU reference | 19.9k | 1.1x |

Per step at S=128 (µs): learner 290 (GPU) vs ~1000 (reference, one core); P4 230, P1 115, P3 60, S4 25.
The GPU makes the learning ~3.5x cheaper than one CPU core; the rest of the gain is multi-streaming. What limits now:
the GPU round trip is a fixed ~300 µs per step (WSL submission latency), and the per-stream CPU phases (sequence-memory
capture in P4, agency in P1) with 12 threads.

## One epoch, 64 streams, GPU (`apps/ms_babylm.cpp`, run `~/babylm/runs/ms64_gpu`)
Same corpus preparation and epoch-1 chunk order as model_babylm; checkpoint = stream 0 exported into the organism, scored
with the unchanged model_babylm_score on items_fast (build-gpu/epoch.sh).

| run | epoch time | held-out bpb | BLiMP | supplement | entity |
|---|---|---|---|---|---|
| ss10_adr_A2ro (CPU, 1 stream) | ~3150 s | 2.123 | 53.01 | 48.40 | 18.50 |
| ss10_adr (CPU, no registers) | 3147 s | 2.187 | 53.27 | 46.80 | 17.47 |
| **ms64_gpu (64 streams, GPU)** | **1033 s** | **2.127** | **53.47** | **51.60** | **18.06** |

3.0x faster per epoch with the same held-out loss and skills within noise (one seed each; supplement is 250 items).
Caveat: stream 0's sequence memory has read only 1/64 of the epoch.

## Speed work (same day)
Note: earlier "128 streams" figures on 2-4 MB were really 32-64 active streams (one 64 KiB chunk per stream minimum);
`ms_check speed` now refuses fewer chunks than streams. Figures below: 128 active streams, 16 threads, 16 MB.
1. Worker threads spin (up to 5 ms) before sleeping: waking sleeping threads under WSL cost 100-250 us per parallel
   phase. Parallel phases dropped 5-10x. 94k B/s (S=128, 12 threads, overlap off).
2. Overlap (`MultiStream::set_overlap`, `Learner::submit/wait`): the GPU learns step t while the CPU runs step t+1; a
   body's Adam step reaches the host forward one step late (off by default; S=1 exactness unaffected). 118k -> 151k B/s.
   GPU busy ~255 us/step, host wait ~10 us: the GPU is fully hidden; the trainer is CPU-bound.
3. CUDA launches on a launcher thread (submit cost 124 -> 27 us): 180k B/s (9.8x the CPU baseline) on a quieter machine.
4. Episodic imprints run in parallel after S4 (they write only the newborn's weights: exact); head weights and
   gradients ride on each step's transfers (no separate synchronous copies); readout gradient partials are summed only
   when the readout steps (S>1). Checks after all changes: S=1 exact PASS; dual (16 streams): 1 differing nibble of
   326M (a rounding flip), shadow max diff 4e-4.
   Throughput with these could not be measured cleanly: the same binary gave 126k and 106k B/s as the machine's load
   average (research probes and scoring) went from ~25 to ~30 on 32 cores.

Per step now (S=128, loaded machine, ~850 us): P1 address/agency ~210, P3 readout/seq ~155, S3 ~165 (of which submit
~40, readout Adam ~55), S4 ~110, P4 ~95, P2 ~40. The per-stream CPU phases (P1, P3, P4) are ~60% and scale with free
cores; the serial part (S1-S4) is ~300 us.

## Serial part cut, overlap epoch (final GPU optimisations before the port)
Per-stream bookkeeping moved out of the serial phases (still S=1 exact over 1 MB; dual agrees): learner inputs and
delayed-job lists built in P3, the ring store in P4, the stream's running error / prior / gain in P3 (only the lock
updates stay in S4). Serial per step ~230-290 -> ~200 us (S4 ~100 -> 57, job building ~40 -> 23).

One epoch, 64 streams, GPU, `--overlap` (run `~/babylm/runs/ms64_gpu_ov`, load average ~25-30 throughout):

| run | epoch time | held-out bpb | BLiMP | supplement | entity |
|---|---|---|---|---|---|
| ss10_adr_A2ro (CPU, 1 stream) | ~3150 s | 2.123 | 53.01 | 48.40 | 18.50 |
| ms64_gpu (64 streams, no overlap) | 1033 s | 2.127 | 53.47 | 51.60 | 18.06 |
| **ms64_gpu_ov (64 streams, overlap + all speed work)** | **468 s** | **2.117** | **52.83** | **49.20** | **18.38** |

6.7x the CPU epoch; loss and skills within the spread of the other two runs (one seed each): no regression from the
overlap. Throughput 114-122k B/s at 64 streams on the loaded machine.

Scoring is still the CPU fork-per-candidate model_babylm_score (~25 min, now longer than the epoch). A batched scorer
(one stream per candidate, learning off, each stream starting from a copy of the checkpoint's stream state) is the
next item; it must reproduce the forked scorer candidate for candidate.

## Batched scorer (`apps/ms_score.cpp`, `MultiStream::score`)
model_babylm_score's protocol without fork: load + freeze the checkpoint, read "\n" through Organism, then S candidates at
a time, each read by a stream that starts from a copy of that frozen stream state (history, workspace, registers,
sequence memory, agency). Evaluation still changes shared state that later bytes read (lock hits and error rates, unit
activity averages, route utilities): each stream keeps its own overlay of those, so no candidate sees another. No GPU
(no learning): the speed is threads instead of fork.
- Parity on ms64_gpu_ov epoch_1.state: all 38,490 candidates byte-identical to model_babylm_score's files (BLiMP 26,800,
  supplement 500, entity 11,190). 159 s for all three (64 streams, 12 threads, load ~28) vs ~25 min forked.
- Memory: ~22 MB per stream (sequence memory) plus the container; size S to the free RAM.
- `l6_hook` / `diag_zero_wsin` added for the agree_probe arms (the emitted L6 is hooked, the trace keeps the body's own,
  as in run_body). Not yet verified: the fork-vs-batched arm check crashed WSL (out of memory with 64 scoring streams next
  to the probe's own 2.1 GB model and the research runs); the agree_probe wiring (MS_SCORE=1/2) is held back until it passes.
- The trainer now refuses reg_center / reg_revive / head_norm / reg_ro_init / adam_l6=0 (Organism options it does not
  implement) instead of silently differing.

## Next
0. agree_probe arms through the batched scorer: rerun the MS_SCORE=2 check with fewer scoring streams.
1. Measure on a quiet machine (or pin threads to cores the research runs leave free).
2. Serial phases: S4 evidence / metabolic per stream could run in parallel for streams whose candidate locks do not
   overlap; job building (S3) could be done per stream in P3.
3. More epochs / seeds at S=64-128 to confirm the skills match beyond epoch 1 (with overlap on).

## The pyramid in the trainer (tiles with averaging merges, 2026-09-25)
Each stream keeps its own `TileState` (the organism's `tst_`: views per level, working memory, attention slots, fast-weight
matrix); the shared tile weights (read mixings M, attention read, fast-weight projections) live in the container organism
and step with the readout (`ro_adam_step`). Per event, in the organism's order:
- P3: `tiles_read` + `tiles_attend` just before the readout (tiles do not change within an event), the stream's tile reads
  join the readout's input; after the readout error, `tiles_learn` (at one stream straight into the organism's gradients,
  else into the worker's partials, summed when the readout steps); the queries' gradient joins dl6 before the jobs;
- word tiles' credit to the writers of the event that closed them goes to the learner as ring injections
  (`StepIn.inj`, `LInject {stream, age, g}`), applied in order before the ring's own credit (RefLearner; `k_inject` on the GPU);
- P4: `tiles_close` after the write head, before the history shift (with averaging merges it touches only the stream's own
  state, so P4 stays parallel); checkpoint export / scoring import carry the stream's TileState.
- Learned merge bodies (tile_avg=0) stay organism-only: their delayed passes would need jobs of their own.
- One ordering fix in the ORGANISM: the readout's Adam step (every ro_adam_k = 512 events) ran inside learn_l6, before
  tiles_learn, so the step event's tile gradient joined the NEXT minibatch and that event's tiles_learn saw the new mixings;
  with tiles on, the step now waits until after tiles_learn (this event's reads join its minibatch, as the trainer does at
  any S). The first exact check diverged at event 512 because of it. Configurations without tiles are unchanged.
- ms_check exact (one stream, CPU reference vs Organism, 524,288 events, every event's loss and prediction, weights and
  population per block): PASS for tiles=3 + wm 8 + attention; tiles=4 + word identity + context + fast weights; tiles=4 +
  identity + context + attention + wm. Regressions (unwired motifs, default configuration): PASS.

## motif_stream (goal item G, 2026-09-25)
The byte's motif bodies form a chain whose L3 message is the mean of the earlier bodies' L6 (x0.5, clamped); the message
gradient returns to every earlier body's L6. Unlike motif_wire's L2 message this composes, so the backward is no longer
all-receivers-in-parallel: a receiver's share depends on its own receivers'. CPU reference: a per-stream accumulator over
the stream's jobs (last to first). CUDA: `k_stream`, one round per chain position (last first), one job per stream per
round (no races); each job's read of the accumulator is kept and k_backward adds it to the job's L6 error. The trainer's
P2 assumes every earlier body reaches L6 and S2 recomputes a body whose stream changed (an earlier body died), as it does
for motif_wire's stopped senders. LTrace flags: LF_L6 (the body's L6 is in the stream), LF_STRM (its L3 read the stream;
`sender` = its position).
- ms_check exact: 1,048,576 events bit-identical (motifs + stream), 524,288 with memory v1 tiles; regressions (wired +
  bp_route, unwired, default) pass. Dual S=1: 235 of 534M nibbles differ (4.4e-7), ring rel diff 1.4e-7; S=16: 846 of
  1.08G (7.8e-7), ring 1.25e-7 - the same as the motif_wire control (786, 1.2e-7).

## ws_split and tile_ret in the trainer (2026-09-26)
- ws_split (wide readout): the trainer keeps each motif's emission (body k is motif k) and its own readout error; the
  per-motif readout weights step with the readout; LStream carries dl6m[8][F], a direct job carries its motif in `slot`
  (temporal jobs overwrite slot with their ring slot); both learners use (dl6 + dl6m[slot]) x gain (LearnConfig.split).
- tile_ret (fixed-key retrieval) runs per stream on the TileState; the output map's gradient partials reduce with the
  readout. The query sees a word ending at the current byte (tiles_attend takes hist).
- ms_check exact: ws_split 524,288 events; ws_split + stream + tiles + retrieval 524,288 - bit-identical; regressions pass.
  Dual: S=1 178 of 534M nibbles (3.3e-7); S=16 split + stream 1,048 of 1.08G (9.7e-7), as the controls.
