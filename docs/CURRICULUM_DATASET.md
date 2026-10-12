# The public-data curriculum: what was actually built

Built 2026-09-22, rebuilt the same day after `docs/CURRICULUM_REBUILD_ISSUES.md` found five real
dataset-construction bugs. **The corpus now lives at `/home/user/curriculum_v2/` inside WSL**, not
under the repo's `data/` folder: the Windows `C:` drive had ~12 GB free at rebuild time (it was
already nearly full before this project), and WSL's own filesystem has 338 GB free. Since Coco's own
trainer already runs from WSL in this project (`build-wsl/laminar_train` etc.), pointing `file=` at
an absolute WSL path is the normal way to use this from here. `PROVENANCE.json` lives alongside the
data at `/home/user/curriculum_v2/PROVENANCE.json`; this page is the durable (git-tracked) summary.

**No source required gated access.** Every registry entry this build used (S2-S7, S9-S12, S14-S15)
was checked and confirmed open; none needed an approval request or an HF gating flow.

**Total: 9.4 GB** (unchanged from the first build -- the rebuild redistributed data between stages,
mainly moving ~220k questions from Stage 2 into Stage 4, rather than adding or removing much overall).

## The five rebuild issues, and how each was verified

An external audit (`docs/CURRICULUM_REBUILD_ISSUES.md`) found these before any training used the
first build. All five are fixed in `python/curriculum/*.py` and independently re-verified, not just
re-argued:

| # | Problem | Fix | Verification |
|---|---|---|---|
| 1 | Stage 5 `tf`/`unknown` labels were picked by choosing a never-a-holder person in a fully-observed world, where that's always `false`, not `unknown` (5,423/64,952 training and 116/1,362 held-out episodes audited as contradictory) | Real partial observability: `known[obj]` tracks whether the holder is currently establishable, via unseen initial holders, explicit observation gaps (which may silently change the hidden holder), and regained observation on any later narrated transfer; corrections now retract a specific named prior misreport instead of confirming an unchallenged one | An independent script re-derives every answer from story **text alone** (not the generator's state) and diffs against the label: **0 contradictions across 66,726 training + 850 held-out questions** at full scale |
| 2 | `build_grounded_dialogue.py` dropped the `ENVIRONMENT` role in its rendered output, so a verified outcome looked identical to ordinary narration | Explicit `ENVIRONMENT:` marker in the stream; a canonical `{role, text}` test export; a new before-reveal export that cuts right after the assistant's claim and holds the true outcome back as the target | Manual inspection of rendered dialogues and before-reveal cases confirms the cut point and the held-back target are correct |
| 3 | Stage 3's paraphrase test worlds were drawn from the same CLEVR train pool every other stage trains on: 450/450 found in Stage 2 training, 449/450 in Stage 4 | Test worlds now come exclusively from CLEVR's official **val** split, the same held-out pool Stage 2/4 already use for their own tests | Exact rendered-scene text compared between the 450 Stage 3 test worlds and all 67,834 Stage 2 + 69,998 Stage 4 training worlds: **0 overlap** |
| 4 | Paraphrase templates each rendered on a fresh scene: 8,550 distinct worlds, 0 with more than one paraphrase -- no paired agreement check was possible | A shared pool of worlds is chosen once; every trained template renders on the *same* world; one of three templates is held out of training entirely (an unseen-construction test) | Checked directly: all 3,000 training worlds carry exactly 2 paired episodes; all 450 test world groups carry exactly 3 paired variants |
| 5a | Stage 2's CLEVR classifier checked only the *final* program operation, letting a `same_shape`-then-`count` question (relational, mid-program) through as an "atomic" one | `classify()` now scans the whole program; any relational/comparison function anywhere routes to Stage 4 | Full CLEVR re-run: Stage 2 went from 408,657 to 188,597 episodes, Stage 4 from 291,332 to 511,392 (~220k reclassified); the reported example now classifies as Stage 4, a plain filter-count still classifies as Stage 2 |
| 5b | The Stage 5 "immediate" budget just called the same 3-10-event generator, not a distinct bootstrap subset | `make_bootstrap_episode()`: exactly one transfer (success or fail), a genuinely minimal signal, used only for that budget | -- |
| 5c | Stage 5 "corrections" only ever confirmed an already-correct holder, never withdrew a real earlier claim | Misreports are now voiced as hedged, attributed claims and later retracted by name; a degenerate self-contradicting case (the misreport coincidentally becoming true via an intervening real transfer) is detected and skipped | Caught by the same independent checker used for issue 1, during development, before the final run |
| 6 | 71 Stage 5 temporal ("before"/"always had") examples had no textual support for their labeled owner. Root cause: `history[obj]` was seeded with the HIDDEN ground truth in `__init__`, even for objects never narrated or later lost to an observation gap | `history[obj]` now starts empty and is filled only by a narrated fact or an observed transfer; "always had" additionally requires the object be currently known, not just single-entry history; falls back to a retrieval-shaped question when an episode has no object with enough recorded history | The checker (`_check_events.py`) was extended to validate temporal questions too, with a distinct `INSUFFICIENT` verdict separate from a wrong value. A targeted 2,000-world search confirmed the bug pattern before the fix; after it, **0 contradictions and 0 INSUFFICIENT cases across 78,499 training + 1,501 held-out + 1,000 visibility-pair questions**, covering every question kind, not just tf/retrieval |
| 7 | The user's own v2 validation found 1,980 training + 29 held-out temporal errors that fix #6 missed: failed transfers still added a duplicate/spurious history entry (corrupting "before X" so X sometimes meant the CURRENT holder, not a genuinely prior one), and some "always had" answers contradicted a giver the story explicitly named, since only the RECEIVER was ever recorded into history, never the giver | `event_transfer` now records the GIVER first (only if not already the last entry -- covers both an object's first-ever mention and a post-gap reveal, where the hidden holder moved silently during a gap and this transfer is the first sentence to reveal it), then the RECEIVER only on success. A failed transfer adds at most one entry, never a duplicate | Both existing checkers turned out to share the SAME blind spot as the generator (they reconstructed history the same flawed way) and were fixed in lockstep. A second, structurally different checker (`_check_events_structural.py`) was then written: no ordered-history reconstruction at all, just raw sentence-by-sentence role scanning with consecutive-repeat deduping. Both checkers independently agree at **0 issues** on the full rebuild: 78,457 training + 1,543 held-out + 1,000 visibility-pair questions from one checker, 11,668 temporal questions scanned by raw role alone from the other |

## Per-stage contents (after the rebuild)

| Stage | Size | What's in it |
|---|---|---|
| 1: grounding | 2.0 MB | Original bAbI tasks 1, 4, 5 (unaffected by the rebuild) |
| 2: individual operations | 144 MB | CLEVR questions whose *entire* program is a single filter/relate chain ending in count/exist/query_&lt;attr&gt; (188,597 episodes, down from 408,657 pre-fix) + bAbI tasks 6,7,8,18,19 |
| 3: shared meanings | 26 MB | COGS + SCAN (diagnostic-only, unaffected) + a paraphrase generator now producing genuinely paired same-world variants on CLEVR-val-held-out worlds, one construction held out of training entirely |
| 4: composition | 910 MB | CLEVR questions with any relational/comparison operation anywhere in their program (511,392 episodes, up from 291,332) + CLEVR-CoGenT condition A/B attribute-transfer holdout |
| 5: events | 52 MB | The rebuilt transfer/correction/uncertainty generator (60,000 + 20,000-bootstrap episodes, genuine partial observability) + 500 paired visibility-loss/regained-observation test cases + ProPara + LLM-BabyBench (unaffected) + TextWorld rollouts (6,000 games, 5,664 train + 315 test, unaffected -- its own run finished cleanly while the rebuild issues were being fixed) |
| 6a: grounded dialogue | ~27 MB | Schema-Guided Dialogue (unaffected) + the grounded-dialogue generator, now with an explicit `ENVIRONMENT:` role and 1,669 before-reveal scoring cases |
| 6b: broad chat | ~8.0 GB | SmolTalk, OASST1, SQuAD 2.0, TinyStories -- all unaffected by the rebuild, output files carried over unchanged |
| oracle sidecars | 224 MB | Program/skill-family/group-id metadata -- **never fed to the trainer**, scoring/validation only |

## What's still deferred, and what's explicitly out of scope for this rebuild

Unchanged from the first build: **S1 Dyna-bAbI's own generator** (2021-pinned dependencies don't
build here; original bAbI used instead), **S8 BabyAI/MiniGrid** (not attempted), **S10's interactive
mock-service backend** (only static SGD text was reconstructed), **S13 CLUTRR** (noncommercial
license, excluded per the manifest).

`CURRICULUM_REBUILD_ISSUES.md` also describes a separate "Learner work" track -- a native C++-side
reasoner, observation-language bootstrapping, and environment-role ingestion on the trainer side. That
is explicitly *not* part of this dataset rebuild (the issues doc itself says so): a dataset adapter's
job is to produce correctly-labeled learner-facing streams with the teaching signal kept out of the
input, not to make the current trainer able to interpret every new construction. Those gaps remain
open on the learner side.

## Reproducing the rebuild

All adapters are unchanged in location (`python/curriculum/*.py`), just fixed in place. Three
verification scripts were added and are meant to be reused, not one-off. `_check_events.py` and
`_check_events_structural.py` are deliberately two DIFFERENT reconstructions of the same story text
(one builds an ordered per-object history the same way the generator does; the other scans raw
sentences for who is ever named in which role, with no ordering model at all) -- fix #7 was only
caught because the first checker shared the generator's own blind spot, so a single checker agreeing
with the generator is not sufficient evidence on its own:

```
python/curriculum/_check_events.py FILE.txt              re-derives every tf/retrieval/temporal
                                                            answer from an ordered history built from
                                                            story text alone, diffs against the file
python/curriculum/_check_events_structural.py FILE.txt   a second, independently-designed check:
                                                            no ordered history, just raw per-sentence
                                                            role scanning with repeat-deduping
python/curriculum/_check_events_jsonl.py FILE.jsonl      the first check, for a held-out jsonl export
```

To rebuild from scratch in WSL (recommended -- see the disk-space note above):

```bash
wsl bash -lc 'mkdir -p ~/curriculum_v2/{stage1,stage2,stage3,stage4,stage5,stage6,_oracle,_src,scripts}'
# copy python/curriculum/*.py into ~/curriculum_v2/scripts, fetch sources per PROVENANCE.json,
# then run each build_*.py exactly as recorded in PROVENANCE.json's "sources" section.
```

`python/mix_stream.py` (already in the repo) still assembles any subset of the resulting files into
a mixed training stream, e.g.:

```bash
python python/mix_stream.py data/curriculum_stage1.txt /path/to/curriculum_v2/stage1/babi_s1_train.txt
```

The manifest's own within-stage budget (60% current stage / 30% earlier skills / 10%
training-derived counterexamples) and broad-chat progression (10% -> 25% -> 50%) remain training-time
sampling decisions, not baked into any file.
