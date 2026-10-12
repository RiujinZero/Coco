# BabyLM 2026 — preparation (24 September 2026)

First external target: the BabyLM challenge (data-limited language modelling), with a parameter-matched model of
**20,000 bodies** (5,632 INT4 weights each ≈ 113M weights, against the challenge's ~120M-parameter baselines).
The 2026 submission deadline (20 July) has passed; the data, evaluation pipeline and leaderboard are public, so
results here are an unofficial comparison under the same rules, ahead of the 2027 round.

## Downloaded (inside WSL, `~/babylm/`)
| What | Source | Size |
|---|---|---|
| `BabyLM-2026-Strict-Small` (exactly 10,000,000 words, 6 files) | huggingface.co/datasets/BabyLM-community | 54.4 MB |
| `BabyLM-2026-Strict` (exactly 100,000,000 words) | same | 543.3 MB |
| `babylm-eval` pipeline | github.com/babylm-org/babylm-eval | 144 MB |
| `BabyLM-2026-Strict-Evals` evaluation data (full and fast splits) | huggingface.co/datasets/BabyLM-community | 146 MB |

**EWoK is gated:** its full data needs its terms accepted on Hugging Face by the account holder; only a protected
"fast" zip came with the evaluation data. It is reported once available.

## Rules that apply
At most 10 epochs over the training data (1B words exposure on Strict, 100M on Strict-Small); only the provided
corpus (Strict tracks). Zero-shot tasks compare candidate sentences by summed completion log-probability
(length-normalized only for GlobalPIQA); (Super)GLUE is by fine-tuning.

## 2026 targets (GPT-2 baselines, from the pipeline README)
| Task | Strict-Small | Strict |
|---|---|---|
| BLiMP | 65.23 | 74.73 |
| BLiMP Supplement | 57.25 | 65.00 |
| EWoK | 50.63 | 54.37 |
| Entity Tracking | 19.10 | 16.91 |
| COMPS | 51.81 | 55.85 |
| GlobalPIQA | 35.09 | 36.62 |
| (Super)GLUE average | 63.80 | 67.75 |

## Tools
- `cpp/apps/model_babylm.cpp`: training. It interleaves the six files in 64 KiB chunks at line boundaries,
  shuffled per epoch, and holds out the last 0.5% of every file for bits-per-byte tracking. The 10-epoch cap is
  enforced. Checkpoints every 16 MiB, plus one per epoch (`epoch_K.state`). Held-out scoring runs in a forked
  child, so measurement never changes the model. An existing directory resumes.
- `scripts/babylm_export.py`: exports zero-shot items through the official loader (`read_files.py`; the
  `datasets`/`aiohttp` imports it uses only for images and downloads are stubbed). Full split: BLiMP 59,875 pairs,
  BLiMP Supplement 5,218, COMPS 91,028, entity tracking 6,780.
- `cpp/apps/model_babylm_score.cpp`: byte-level zero-shot scoring. The frozen model is primed with a newline and
  forked **once per candidate**; each child reads the context, sums the completion bytes' log-probability, and
  reports it. Every candidate is therefore scored from exactly the same state, and no candidate can see another
  through sequence memory. About 120 candidates/s with 24 workers at 20k bodies.
- `scripts/babylm_report.py`: the pipeline's rules. Macro average over sub-tasks; entity tracking averages within
  its three splits first; a tie counts as the pipeline's random choice in expectation.

## Smoke test (20k bodies, 4 MB of Strict-Small, one pass)
Training 99 s (42 KB/s); held-out 4.22 → 3.38 bits/byte; **BLiMP (fast) 53.9%**, entity tracking (fast) 18.6%.
The population was full after 4 MB (19,994 of 20,000 live) and the checkpoint is 363 MB: at this size snapshot
bodies fill capacity almost immediately, so the recurrence gate and promotion (round 3) matter even more here.

## Pending before the real runs
Round-3 results (snapshot gate and promotion, workspace decay), then the re-optimization pass (memory, eviction,
hyperparameters). Then Strict-Small (10 epochs, ~2–3 h at 20k bodies), then Strict (~1 day), with every epoch
checkpoint scored. (Super)GLUE fine-tuning through the task readout is a later step.

## Body market on one Strict-Small epoch (20k bodies)
| Arm | Held-out bits/byte (16.8M / 33.6M / 50.4M / epoch end) | BLiMP fast | Entity tracking fast |
|---|---|---|---|
| market off | – / – / – / **2.932** | 52.6 | 18.0 |
| market (1% reserve, shared credit, parsimony controller) | 3.155 / 3.171 / 3.642 / 3.672 | **54.9** | 18.6 |
| market + recurrence gate | 3.008 / 3.342 / 2.993 / 3.282 | 54.0 | 18.3 |

Notes:
- The first market version reserved "two blocks" of free units, which at 20k bodies kept **82% of capacity
  empty** (4,200 live). The reserve is now `market_reserve_ppm` (1%).
- Even with that fixed, the market churns general bodies: about 205,000 general and 45,000 snapshot culls in one
  epoch for a 20k population. The old system rarely removes a general body, so its learning accumulates.
- Shared credit likely punishes general bodies in busy regions (up to 16 co-firing bodies split every gain).
- Held-out text swings ±0.3 between checkpoints, which makes the controller jumpy.

Verdict: worse text prediction, BLiMP within noise (slightly higher). The market stays off for BabyLM. It needs a
retune (no culling of admitted general bodies unless fitness is negative, plain utility, a smoothed retention
signal) before another round.

## Tokenizer test (running)
Transformers get word-sized units from a tokenizer; Coco reads bytes. The composer (`levels`) learns units online,
from surprise plus a lexicon (a unit joins once it has recurred `lvl_minc` = 5 times), with its own bodies per
level. Three arms, 3 Strict-Small epochs each at 20k bodies, fast items scored in-process at every epoch end
(`model_babylm --score`, `--no-checkpoint` because composer levels cannot be checkpointed yet): bytes only;
`levels=1`; `levels=2 lvl_feed=1`. Smoke (4 MB, `levels=1`): held-out 3.15 vs 3.38 bytes-only, BLiMP 54.7 vs 53.9.
