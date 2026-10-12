# GPU port — handoff (2026-09-25)

Goal of the GPU session: make organism **training** (and ideally scoring) run on the GPU so BabyLM experiments are faster.
The research session (CPU) keeps running AdamW experiments in parallel. The user's roadmap for the learning rule:
**full AdamW first → then Muon → then local learning**. Skills (BLiMP, supplement, entity tracking) are the yardstick, not
bits-per-byte. The full research history is in `docs/BODY_CIRCUIT_RESEARCH_LOG.md` (read the end: sections 24+ and the
2026-09-25 entries).

## 1. Repo state — read before touching anything
- Branch `model-task-learner` (local, never pushed; **don't push without asking the user**). The Adam learner and all
  probes are committed in `cfdc506` — branch / make your worktree from that commit.
- `build-task/` is gitignored: the helper scripts referenced below (`launch_adam.sh`, `agreeB.sh`, `rep10.sh`, `phen.sh`,
  ...) exist only in the main working copy `C:\Users\user\Documents\projects\Coco\build-task\`.
- The research session keeps editing `organism.cpp` / `config.h`. Work in your own worktree/branch; keep your own log
  (e.g. `docs/GPU_PORT_LOG.md`) instead of editing the research log.

## 2. Machine and toolchain
- Windows 11 + WSL (Ubuntu). 32 cores, ~25 GB RAM visible in WSL. GPU: **RTX 5080, 16 GB**, driver 610.88.
  CUDA toolkit in WSL: `/usr/local/cuda-12.8/bin/nvcc` (also `/usr/bin/nvcc`). No CUDA code exists in the repo yet.
- Windows Python has torch **2.14.0+cpu** (no CUDA); WSL Python has no torch. Installing CUDA torch is a ~3 GB download:
  **ask the user before downloading anything.**
- Build (WSL, no cmake there): `g++ -std=c++20 -O2 -march=native -DNDEBUG -Iinclude apps/X.cpp src/organism.cpp -o ../build-task/X`
  from `cpp/`. Run everything through script files: `MSYS_NO_PATHCONV=1 wsl bash "/mnt/c/.../build-task/foo.sh"` —
  shell variables and loops vanish inside `wsl bash -lc '...'` from Git Bash, and Git Bash rewrites /paths without
  MSYS_NO_PATHCONV. Heredoc-written Python patches turn "\n" in C++ strings into real newlines — use the Edit tool for those.
- **Don't disturb running experiments**: `~/babylm/runs/{ss10_adr,ss10_adr_A2ro,ss_adrw,ss_adamw}` (BabyLM, some 10 epochs),
  `/tmp/agreeD/*` (organism probes), binaries `build-task/model_babylm_adam`, `agree_probe{A,B,C,D}`, `seq_probe`.
  Use your own binary names and run dirs. They use ~15 GB RAM and ~6 cores; leave headroom.
- Data: `~/babylm/BabyLM-2026-Strict-Small/` (corpus), `~/babylm/items_fast/{blimp,blimp_supplement,entity_tracking}.jsonl`
  (scoring items), `~/babylm/heldout_ss.txt`, `~/babylm/babylm-eval/` (official eval data; per-phenomenon BLiMP labels in
  `strict/evaluation_data/fast_eval/blimp_fast`). Reports: `build-task/rep10.sh RUN...`, `build-task/phen.sh BASE E RUN E`.

## 3. The organism (what must run on the GPU)
Code: `cpp/src/organism.cpp` (~2000 lines), `cpp/include/laminar/organism.h`, `config.h` (Params + Dims + parser),
`int4.h` (INT4 kernels: scalar / AVX2 / expand), `address.h` (lock tree), `sequence.h` (sequence memory).

**Body (unit)** = 5632 INT4 synapses = 2816 bytes, weights `[out][in]` row-major, layers stored in order L1,L2,L3,L5,L6,L4:

| layer | in | out | notes |
|---|---|---|---|
| L1 | IN=128 | A=16 | input cells: byte features, context, workspace ws_mem at 96..127, registers at 124..127 (ws_regs) |
| L2 | A+R2=32 | B=24 | R2 = return from L4 |
| L3 | A+B=40 | C=16 | B = neighbour (routed) L2 message |
| L5 | B+C=40 | E=16 | |
| L6 | E+R6=24 | F=32 | emission, summed into the workspace `ws` |
| L4 | F=32 | R=24 | return (route_return / persist) |

Layer op: `y = clamp8((sum_i x_i * w_ji) >> shift)`, int8 activations in ±127, shifts `{1,2,2,3,3,3}` (homeo can make
them per body). Weights int4 in [-8,7], birth uniform {-2..2}.

**Per byte event** (`process_event`, sequential stream; `run_block` feeds a chunk):
1. `make_key` / `make_input` → 64-bit key and the IN=128 input cells.
2. The lock tree (`address.h`) selects candidate bodies for the key; `run_pass` runs each (`run_body`: L1→L2/L3→L5→L6,
   optional L4), up to `max_active` traces; with hops>0 bodies route to neighbours (current BabyLM config: `hops=0`).
3. `ws` = sum of emitting bodies' L6; float readout `wout[256][32]` on ws/128 (+ `wreg` register readout) → body probs;
   blended with the sequence memory by a learned gate (`seq_linear`) → loss.
4. Learning (`learn_l6`, `learn_backprop`, `temporal_credit` ring of `tc_window`=64 events, `temporal_backprop`).
5. Structure: metabolic / market / episodic births / lock splits (branchy, sequential; **probably stays on CPU**),
   register write head (`ws_regs=2`), `ws_mem` decay, sequence-memory capture.

## 4. The learning rule to port ("full Adam/AdamW", added 2026-09-25)
Root cause found today: the original accumulator SGD left hidden layers frozen (~1% of weights change per epoch).
The replacement (all in organism.cpp, flags in config.h):
- `adam=1`: every backprop path adds the **exact float gradient** `g_j * x_i` to a per-synapse buffer `ad_g`
  (`adam_grad`); after every `adam_k`=32 uses a body takes one bias-corrected Adam step on a float **shadow** `ad_sh`
  (β 0.9/0.999, lr `adam_lr`=0.003 in INT4 grid units, `adam_wd` decoupled decay), INT4 weight = round(shadow), shadow
  clamped to [-8.49, 7.49]; external writes (birth, imprint, copies) are detected and resync the shadow (`adam_step`).
  `adam_warm=1`: young bodies step after 1,2,4,8,16 uses. Replaces the quantised L6 rule in this mode.
- `ro_adam=1`: readout `wout`, register readout `wreg`, write head `hw_g/hw_v/hb_g` and the sequence gate `seq_gw` by
  minibatch Adam, one step every `ro_adam_k`=512 events (lr 0.002, `ro_adam_wd`).
- `bp_leak=0.1`: backward slope 0.1 through saturated (±127) outputs (forward clamp unchanged). Required.
- `reg_readout=1` (+ `ws_regs=2 reg_bias=-3 tc_window=64`): the registers that carry long-range content.
- Memory: Adam state 16 B/synapse → 1.8 GB at 20k units; measured RSS 2.13 GiB per process (old rule 0.45 GiB).
  Scaling concern (user): 1.6M bodies would need ~144 GB of Adam state — Muon (one momentum buffer) and a pool of state
  for active bodies only are the planned answers.

Current BabyLM config (`F`): `body_err=2 err_gain=2 err_sparse=0 in_struct=1 hidden=backprop recirc=128 seq_gate=1
seq_linear=1 stoch_round=1 hops=0 bp_leak=0.1 adam=1 ro_adam=1` (+ `tc_window=64 ws_regs=2 reg_bias=-3 reg_readout=1`
for the register variant). Launch: `model_babylm strict-small DIR --epochs N --score ~/babylm/items_fast --workers 10 $F`
(see `build-task/launch_adam.sh`).

## 5. Speed baseline (CPU, one process, BabyLM Strict-Small epoch = 54 MB)
- Old rule: 1352 s/epoch (~40k bytes/s). Full Adam: **3147 s/epoch (~17k bytes/s)**.
- Scoring (BLiMP/supplement/entity) forks one child per candidate on CPU (`score_items`, `--workers`); embarrassingly
  parallel — a batched GPU scorer is a separate, easy win.
- Per byte the work is tiny (a handful of bodies × ~5.6k INT4 MACs forward, ~2× backward). A single-stream GPU port will
  be kernel-launch-bound and probably SLOWER than the CPU. GPU pays off with many parallel streams (batch of documents)
  sharing body weights — which matches the minibatch Adam rule (per-body gradient accumulation across streams, step at
  sync points). Open design questions: per-stream lock-tree lookups / routing, market & births at sync points, the
  sequence memory per stream, and whether learning order changes break results.

## 6. Correctness references (reproduce before claiming speed)
- Organism is/are probe (`cpp/apps/agree_probe.cpp`, env `ISARE=1`, 12 MB, BASE in `build-task/agreeB.sh` + `tc_window=64
  adam=1 ro_adam=1 ws_regs=2 reg_bias=-3 reg_readout=1`, seed 1): % correct at d0/1/2/4/6/8 = **100 100 98 100 98.5 98.5**;
  ablation "constant bodies" 100 85 53.5 59.5 57.5 51. **Not robust: seed 2 (12 and 32 MB, ± adam_warm) and seed 3
  (32 MB) stay at chance beyond d1 — 1 of 4 seeds so far.** Use seed 1 as the reproduction target, not as proof.
- Harness `cpp/apps/seq_probe.cpp` (no organism; INT4 bodies + registers + exact BPTT): `isare=1 regs=1 leak=0.1 opt=1
  batch=32 ropt=1 alr=0.003` → 100% at every distance by 320k sentences (seed 1); per-body batch `bbatch=32` 2/2 seeds.
- BabyLM epoch 1 (BLiMP / supplement / entity / held-out): old baseline 52.15 / 51.2 / 19.27 / 2.025; full Adam
  (ss10_adr) 53.27 / 46.8 / 17.47 / 2.187; full Adam + registers (ss10_adr_A2ro) 53.01 / 48.4 / – / 2.123.
- A GPU forward must match the CPU INT4 forward bit-exactly (same shifts, clamp, nibble layout); learning results should
  match within seed noise on the probe above.

## 7. Pitfalls already hit
- Unsequenced operands: `s = f(num) + ... + T[num]` evaluated `T[num]` before `f` set `num` (GCC) — this silently removed
  all agreement from the harness data for days. Build strings in separate statements.
- Scaling traps: the INT4 weight grid is coarse relative to the clamp; Adam steps ≥0.01 grid units saturate every layer
  into input-independent constants; 0.001–0.003 works.
- An SGD readout (per-event, lr 0.005) prevents long-range learning even when bodies use Adam — the readout must be Adam too.
