# Laminar 200k Organism — C++ Optimization Runbook

## Goal
Target **500–1,000 KB/s online training** and >=1 MB/s inference without changing the learning/structural rules merely to win a benchmark. Preserve event-driven sparsity, strict INT4 durable weights, local accumulators, adaptive lock/key addressing, motif-local K routing, and early information death.

## 1. Build the hot path as one native pipeline
Do not cross Python/C++ per byte, unit, or layer. One C++ call should consume a large byte/event block and execute:

`input -> adaptive address walk -> lock verify -> admitted units -> L1 -> L2/L3 -> neighbor lock tests -> L5 -> L6 -> L4/exit -> local learning`

Python should only configure runs, checkpoint, and read aggregate telemetry.

Compile baseline with `-O3 -march=native -flto -DNDEBUG`; benchmark PGO (`-fprofile-generate/use`) after correctness is frozen.

## 2. Data layout first
### Durable weights
- Pack **two signed INT4 weights per byte**.
- Store weights **unit-major, layer-contiguous** so an admitted unit touches sequential memory.
- Keep each layer's offset/length in a tiny fixed descriptor; avoid pointer-rich objects.
- Align unit/layer starts to 64-byte cache lines where practical.

### Fast plasticity
- Do **not** give all 1.12B synapses FP32 accumulators if avoidable.
- First implementation: INT8 accumulator per synapse; benchmark INT4/INT8 residuals against FP32 reference learning.
- Touch accumulators **only for synapses in active units/layers**.
- Quantized update: accumulate local evidence -> threshold crossing -> increment/decrement INT4 weight -> subtract quantum.

### Runtime state
Use structure-of-arrays (SoA), not `struct Unit { ... }` for 200k units. Separate hot arrays (address/lock metadata, thresholds, activity flags) from cold arrays (rare statistics/debug data).

## 3. Adaptive address/lock screening
This must stay cheaper than waking L1.

- Represent a lock as `(value, mask)`; verify with `((key ^ value) & mask) == 0`.
- Store adaptive address tree in flat arrays: `bit[]`, `left[]`, `right[]`, `leaf_offset[]`, `leaf_count[]`.
- A lookup should be ~10–20 cheap branch/bit decisions and end at ~1–4 candidate locks.
- Do not scan 200k locks.
- Consider branchless child selection and software prefetch only after profiling.
- Batch independent inputs if semantics allow it; otherwise keep the online path single-event but pipeline memory fetches.

## 4. Early exits are the main optimization
Never charge an admitted unit for all 5,600 parameters automatically.

Execute exactly in dependency order:
1. L1 (1,800 params)
2. exit if dead
3. L2/L3 (700 + 700)
4. cheap K-neighbor lock tests; only accepted neighbors become events
5. exit if dead
6. L5 (600)
7. exit if dead
8. L6 (800)
9. emit to workspace and/or exit
10. L4 (1,000) only when recirculation is actually active

Track survival counts for every boundary. Performance bugs should show up immediately as metabolic explosions.

## 5. INT4 kernels
Start simple and correct, then optimize the profiler's winner.

- Native scalar nibble unpack is the reference.
- Add AVX2/AVX-512 kernels that unpack blocks and perform integer dot products.
- Process blocks of 32/64 weights; accumulate in INT32.
- Precompute/pack activation representation if it reduces repeated conversion cost.
- Specialize kernels by layer size instead of one generic dynamic loop.
- Benchmark unpack-on-read versus temporarily expanding the **active unit's** layer to INT8 in scratch/cache.
- Avoid expanding the whole 560 MB model.

## 6. Motif routing
- Unit belongs to a motif whose K is fixed/configured (2,4,8,16,32,64).
- Store neighbor IDs contiguously per motif/unit.
- L2 does **cheap lock/key checks** against up to K eligible neighbors; it does not execute K bodies.
- Push only accepted neighbor events onto a compact active queue.
- Deduplicate a unit within an event/tick with an epoch counter rather than clearing a 200k boolean array every byte.

## 7. Active-event scheduler
Use preallocated arrays/ring buffers. No heap allocation in the hot path.

Suggested event record: unit ID + route/source + compact key/activity reference. Keep it <=16 bytes if possible.

Use two queues for synchronous propagation (`current`, `next`) or an epoch-tagged ring if asynchronous semantics are desired. Hard global active caps are for diagnostics only, **not** normal operation.

## 8. Training hot path
Inference and training should share the exact forward kernel.

Training adds only local work for traversed paths:
- eligibility/local perturbation state only for active synapses;
- INT accumulator update only for active synapses;
- L6 local prediction error/update;
- lock-local future-state statistics;
- structural split checks only on conflicted locks.

Do not sweep/update all weights after a sample. Apply sparse updates using the active synapse/unit list.

For paired perturbation learning, batch the + and - passes so they reuse the same admitted units/weight cache lines where valid.

## 9. Lock structural learning
Keep this off the critical path unless a lock has enough evidence.

- Maintain small running prototypes/statistics on the lock.
- Candidate tumbler becomes structural only if conditional future states diverge relative to within-group variance.
- Use raw internal future state/prototypes; no simulator identity labels.
- Split operations can be deferred to a safe boundary between event blocks; ordinary routing continues with the old lock until commit.
- Structural mutation should be rare compared with inference.

## 10. Threading strategy
First optimize **single-thread latency**. Then parallelize where causality permits.

Good candidates:
- independent training samples/streams if learning semantics permit delayed reduction;
- candidate unit bodies within the same frontier;
- motif-local neighbor screening;
- background structural maintenance/checkpointing.

Avoid locks/mutexes in the byte hot path. Prefer thread-local queues/statistics and deterministic merge points.

On a 9950X, test physical cores before SMT. Pin workers and measure NUMA/cache behavior rather than assuming more threads = faster.

## 11. Memory/cache stress test
Benchmark with the **real 200k x 5,600 packed INT4 store (~560 MB)** and random admitted unit IDs.

Test at least:
- familiar sparse stream;
- moderate novelty;
- novelty storm;
- deliberately pathological full-body activity.

Report throughput versus metabolism, not one flattering number.

## 12. Required telemetry per block
Log at least:
- bytes/events per second;
- BPB / prediction loss;
- address decisions/input;
- candidate locks/input;
- L1 admissions/input;
- L2/L3, L5, L6, L4 survivals/input;
- neighbor lock tests + accepts/input;
- MACs/input;
- accumulator updates and INT4 level changes/input;
- lock splits/births/deaths;
- mean/p95 address depth and leaf occupancy;
- active queue p50/p95/p99/max;
- weight bytes read (estimated or hardware counter);
- time by stage.

The key success signature is **loss/BPB down while MACs/input and admissions/input also trend down after novelty settles**.

## 13. Profiling order
1. Correctness against Python/NumPy reference on tiny deterministic traces.
2. `perf stat` for cycles, instructions, branches/misses, cache misses, bandwidth.
3. `perf record` / flame graph for stage-level hotspots.
4. Optimize only the dominant stage.
5. Re-run bit-for-bit/metric tolerance tests after every kernel change.

Do not infer bottlenecks from FLOP/MAC counts alone; sparse random memory traffic may dominate.

## 14. Milestones
### A — Native inference
Synthetic vocab reaches reference accuracy and native pipeline >1 MB/s when familiar.

### B — Native online learning
Same task reaches ~noise ceiling with strict INT4 + local accumulators and no Python hot path. Target >=500 KB/s if metabolism is sparse.

### C — 200k physical memory
Repeat with the actual ~560 MB packed weight store and randomly distributed active unit IDs.

### D — Raw-byte language
Start ~100 KB TinyStories. Compare:
- adaptive locks + learning;
- frozen locks;
- memory/routing disabled baseline.

Track BPB, throughput, and metabolism together. Do not scale corpus until the learning curve is real and activity remains bounded.

## 15. Things not to optimize away
Do not replace these with benchmark conveniences:
- adaptive structural ambiguity;
- novelty-induced compute bursts;
- motif K variation;
- strict INT4 durable weights;
- early information death;
- local learning/credit;
- lock splitting based on predictive consequence.

A slower faithful kernel is the reference. Fast kernels must reproduce it.
