#!/usr/bin/env python3
"""Prompt sets for the prompt-following objective (item H, docs/BODY_CIRCUIT_RESEARCH_LOG.md): line starts from BabyLM.

Each prompt is the first 3..8 words of a corpus line that has at least 10 words (headings, lines of fewer than 40 bytes
and lines of more than 600 bytes are skipped). The training prompts come from the training part of every file and the
evaluation prompts from the held-out part (the last 0.5% of every file, cut at a line boundary exactly as ms_babylm and
model_babylm do), an equal number from each of the six files. Written one per line with model_frontier's escapes (\\t \\n \\\\).

usage: hloop_prompts.py CORPUS_DIR OUT_TRAIN N_TRAIN OUT_EVAL N_EVAL [seed]
"""
import random
import sys

FILES = ["bnc_spoken", "childes", "gutenberg", "open_subtitles", "simple_wiki", "switchboard"]


def split(text):   # ms_babylm's held-out cut: the last 0.5%, at the newline before it
    cut = len(text) - len(text) // 200
    cut = text.rfind("\n", 0, cut + 1)
    if cut < 0:
        cut = len(text)
    return text[:cut + 1], text[cut + 1:]


def candidates(part):
    out = []
    for line in part.split("\n"):
        s = line.strip(" ")
        if len(s) < 40 or len(s) > 600 or s.startswith("=") or len(s.split()) < 10:
            continue
        out.append(s)
    return out


def prompt_of(line, rng):
    words = line.split(" ")
    n = rng.randint(3, 8)
    p = " ".join(words[:n]).rstrip()
    return p if len(p) >= 8 else " ".join(words[:n + 2]).rstrip()


def esc(s):
    return s.replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n")


def main():
    corpus, out_tr, n_tr, out_ev, n_ev = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], int(sys.argv[5])
    rng = random.Random(int(sys.argv[6]) if len(sys.argv) > 6 else 1)
    tr, ev = [], []
    for i, f in enumerate(FILES):
        text = open(f"{corpus}/{f}.train.txt", encoding="utf-8", errors="replace").read()
        train_part, held_part = split(text)
        ct, ch = candidates(train_part), candidates(held_part)
        kt = n_tr // len(FILES) + (1 if i < n_tr % len(FILES) else 0)
        ke = n_ev // len(FILES) + (1 if i < n_ev % len(FILES) else 0)
        tr += [prompt_of(s, rng) for s in rng.sample(ct, min(kt, len(ct)))]
        ev += [prompt_of(s, rng) for s in rng.sample(ch, min(ke, len(ch)))]
        print(f"{f}: {len(ct)} training lines, {len(ch)} held-out lines -> {min(kt, len(ct))} / {min(ke, len(ch))} prompts")
    rng.shuffle(tr)
    rng.shuffle(ev)
    open(out_tr, "w", encoding="utf-8").write("".join(esc(p) + "\n" for p in tr))
    open(out_ev, "w", encoding="utf-8").write("".join(esc(p) + "\n" for p in ev))
    print(f"{len(tr)} training prompts -> {out_tr}; {len(ev)} evaluation prompts -> {out_ev}")


if __name__ == "__main__":
    main()
