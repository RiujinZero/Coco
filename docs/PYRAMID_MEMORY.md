# Pyramid memory: working, short-term and long-term memory without a KV cache

Status: design, 2026-09-25. Tier 1 (the tile pyramid view) is implemented in the organism (`tiles=`). Tiers 2 and 3 are not
implemented yet.

## Why

Two separate probes hit the same limit.
- The subject's number never reaches the verb on real text (REGS_PROBE: chance beyond one word).
- The frontier decoder's prompt-dependence judge found that the model forgets its prompt within 2-4 words. The prompt
  moves each byte's log-prob by 2-3 nats over bytes 0-3, about 0.3 over bytes 4-7, and at most 0.016 from byte 32 on.

Nothing holds context. Registers can in principle, but they must learn to keep a write gate shut across every
intervening byte before any payoff arrives. On real text they do not; they pin at +-100.

A transformer holds context by keeping every past key and value and attending over all of them at every step. That
costs O(N) memory and O(N) compute per token in context length N. The user rules this out ("we don't want the blow up of
a transformer style kv cache"). The design below keeps memory and per-step compute constant in N.

## Three tiers, each with a different retention rule

| tier | holds | kept by | size | cost per step |
|---|---|---|---|---|
| short-term | the pyramid view: the last K tiles of every level (words, pairs, quads, ...) | recency, by structure | L x K tiles; 8 levels x 4 cover ~1,000 words | L x K mat-vec (32x32) |
| working | ~8 items that currently matter (a clause's subject, a question's entity, an open binding) | measured use | M slots | M reads |
| long-term | consolidated tiles from all earlier text | content address (role code) | fixed capacity C, eviction by utility | one hash lookup per word |

Nothing has to learn a hold decision, which is the step that fails today.
- Short-term holding is structural: a tile stays in view until it ages out.
- Working-memory retention is by usefulness: the model keeps what its readers leaned on.
- Long-term recall is by content: a tile's role code addresses the store.

Learning only shapes what goes into tiles (the writers and merge bodies) and how they are read (the mixings). Gradients
already do both. The weights remain the semantic long-term memory.

### Tier 1: short-term (implemented, `tiles=L tile_k=K`)
- Level-0 tiles are words: at a word boundary, the workspace summary. A level-l tile is the merge of two level-(l-1)
  tiles by a merge body, chosen from a fixed per-level population by the children's role code.
- The view holds the last K tiles of every level, each read into the readout's input by a learned 32x32 mixing.
- Credit: a word tile's error goes through the temporal-credit ring to the writers of the event that closed it. A
  merged tile trains its merge body when it leaves the view.
- 2 MB smoke (probe seed 3, which fails without it): 100 98 85 74 53.5 45.5 vs 100 75.5 55 52.5 59 43.5.

### Tier 2: working memory by promotion (next)
- When a tile is about to leave the view, compare its USE (the accumulated size of its read gradient, i.e. how much the
  readout's error depended on it) with the weakest working slot's. Promote it if it was used more; otherwise it is
  dropped (or offered to tier 3).
- Slot use decays each word (an EMA), so an item that stops mattering is eventually replaced. Use keeps accumulating while
  an item sits in working memory, so an item that keeps being read stays.
- Reads: the same readout mixing as the view. A slot's position is not meaningful, so use one shared mixing plus a
  learned per-slot bias (the same holds for a small learned query over M slots).
- Credit for a promoted tile continues as in tier 1: word tiles through the ring while their writers are in its window,
  merged tiles to their merge body when they finally leave.
- This is the counterfactual-utility idea the organism already uses for routed bodies (route utility), applied to memory
  items instead of bodies.

### Tier 3: long-term tile store (after tier 2)
- A fixed-capacity table (e.g. 64k entries) keyed by a tile's role code: the sign bits of its first dims. This is the
  same key that picks merge bodies, and a locality-sensitive hash, so similar tiles share a key.
- Write: tiles leaving working memory (and high-level merged tiles) are stored under their key. A hit averages into the
  entry (consolidation); eviction goes by the entry's own use.
- Read: when a tile closes, its key looks up the store, and the retrieved entry is read like a view slot (pattern
  completion from similar past contexts). One lookup per word, independent of how much text came before.
- Persists across documents. This is the retrieval half of what the episodic exact-key bodies tried to do, keyed by
  learned codes instead of byte hashes, so it generalises across surface forms.

## Why this does not blow up

Memory is bounded by construction: L x K + M + C tiles of 32 floats. Per-byte compute is L x K + M small mat-vecs plus one
lookup per word. Coverage grows exponentially with levels (the pyramid), and relevance, not age, decides what survives
(working memory and the store). The price is lossy compression: detail survives only where it was used, or repeated
often enough to be consolidated.

## Tests
- Short-term: REGS_PROBE (does number reach the verb on real text?) and the frontier's REACH (how far the prompt still
  moves predictions: 15-27 bytes today) on a pyramid-trained BabyLM checkpoint.
- Working: binding-and-query text beyond the view's range but used on the way ("Mary went to the kitchen. [...] Where is
  Mary?"), e.g. the bAbI tasks the answer channel already reads.
- Long-term: the same facts queried in a later document.

## Order of work
1. Evaluate tier 1 (probes at 32 MB; a CPU BabyLM epoch with REGS_PROBE and REACH).
2. Tier 2 (a small extension of the pyramid code).
3. Tier 3.
4. Port the three together to the multi-stream trainer. It needs error injection into earlier ring slots (MTP could
   share it) and merge-body jobs that carry their own output error.

## Status 2026-09-26: holding works, retrieval does not

- Memory v1 (averaging pyramid 3 levels + working memory 8 + slot attention) on BabyLM, GPU trainer: load-bearing (+0.35
  bits/byte when the tiles are read as 0), reach 22-33% of outputs still moved by the prompt at 128-255 bytes (no pyramid
  2-3%), generation stays on topic into its second half, full sets up (BLiMP +0.9, supplement +1.8, entity flat), no
  phenomenon below the seed spread. Number held to d8 on every probe seed (averaging merges).
- Bind-and-recall stays at chance for every variant (linear reads, slot attention, fast weights; with word identity; with
  context-bearing tiles). The diagnostic (recall_probe DIAG=1) locates the failure: the object's identity is 100% linearly
  decodable from its word tile at close and from the slot holding it at the question (d=0 100%, d=10 81%), but attention
  puts 0.005 of its weight on that slot against 0.172 uniform. The learned query / key maps are trained by next-byte error,
  where the recent words predict almost every byte; the answer event (one per story) never teaches the entity match.

## Next: retrieval with fixed context keys

Retrieval should not have to be discovered by gradient. A word's KEY is the bag of identity codes of the words just before
it (fixed hashed codes, a larger dimension than F); the QUERY at every event is the bag of the last few closed words; the
match is a cosine, sharpened (softmax with a fixed temperature) over the bounded slots (view + working memory), or a
fast-weight matrix A = sum lambda^age v k^T for spans past the view. Only the output map (and the values' credit) learns.
"the red box has the key ... in the red box is the ___": the key of "key" shares red / box / the with the question's last
words; a distractor bound to another entity shares only "the". Open points: down-weighting frequent words in keys (a
running frequency per hashed word, IDF-like, unsupervised); which words are written (all, or promoted by use); the
fast-weight crosstalk with many writes (key dimension, decay).

## Result 2026-09-26: retrieval with fixed context keys works (`tile_ret`)

- `tile_ret=N` keeps the last N word tiles in an episodic buffer. Each word's key, fixed when it closes, is built from the
  hashed identity codes of the tile_ret_c words before it, each weighted by 1/sqrt(1 + its running frequency); with
  `tile_ret_pos=1` (default) every position has its own code, so the match is "the item that followed a context like
  this one" (an induction match). The query at every event is the same code over the last closed words (a word ending at
  the current byte counts). softmax(beta x cosine) over the buffer; a learned output map reads the weighted items into the
  readout. Only the output map (and the items' credit to their writers) learns - the addressing is never discovered by
  gradient, which is exactly what failed with learned attention.
- Bind-and-recall (k bindings, then 0-120 filler words, then a question in a different surface form), 8 MB of stories,
  128-item buffer: k=1 93 96 93 96 70, k=2 89 94 94 93 60, k=4 90 93 90 89 51 (chance 50). The fall at 120 filler words is
  the buffer's capacity. A 512-item buffer reaches 120 words (77%) but older bindings of the same entities then compete;
  `tile_ret_recency` (newer items win among equal matches) addresses that.
- `tile_ret_heads=2` adds an order-free bag head (for paraphrased questions: "Where is Mary?" shares "Mary" with the fact
  but not its word positions). Both heads are fixed-key; one output map reads both.
- Cost per event: N cosines of 128-dim keys per head, plus N x F for the read - constant in the text's length.
- Tasks: `cpp/apps/babi_probe.cpp` scores bAbI as multiple choice after online training on the bAbI dialogues.
