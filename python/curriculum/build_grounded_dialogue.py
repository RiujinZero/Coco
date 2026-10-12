"""Stage 6a: "Conversationalize worlds Coco already understands" before importing unrelated chat.

Rebuild 2026-09-22 (see docs/CURRICULUM_REBUILD_ISSUES.md #2): the previous render() dropped the
ENVIRONMENT role, folding the environment's outcome report into an untagged line that looked no
different from ordinary story context. That erases exactly the distinction the whole exercise exists
to make -- "the assistant's claim" vs "what actually happened" -- and risks a reader treating the
assistant's own predicted outcome as a verified observation.

The byte stream now uses an explicit ENVIRONMENT: marker (in addition to USER: / ASSISTANT:), and the
canonical export is a list of {"role", "text"} records with the role always present, never inferred
from content. A separate "before-reveal" test export additionally cuts each dialogue right after the
assistant's claim and right before the environment's report, with the true outcome held back as the
scoring target -- so a prediction can be scored BEFORE the outcome is revealed, per the acceptance
criteria, instead of only checked against a transcript that already contains the answer.

    python python/curriculum/build_grounded_dialogue.py OUT_DIR [--n_dialogues 8000]
"""
import argparse, json, os, random, sys
sys.path.insert(0, os.path.dirname(__file__))
from build_events import World


def make_dialogue(rng):
    w = World(rng, rng.randint(3, 5), rng.randint(2, 4))
    turns = [('CONTEXT', f"{w.holder[o]} has the {o}.") for o in w.objects]
    n_req = rng.randint(2, 6)
    for _ in range(n_req):
        o = rng.choice(w.objects)
        giver = w.holder[o]
        target = rng.choice([p for p in w.people if p != giver])
        turns.append(('USER', f"Please have {giver} give the {o} to {target}."))
        turns.append(('ASSISTANT', f"Okay, {giver} will give the {o} to {target}."))
        success = rng.random() < 0.75
        if success:
            w.holder[o] = target
            turns.append(('ENVIRONMENT', f"{giver} gives the {o} to {target}. {target} takes it."))
            turns.append(('ASSISTANT', f"Done. {target} now has the {o}."))
        else:
            turns.append(('ENVIRONMENT', f"{giver} offers the {o} to {target}, but {target} does not take it."))
            turns.append(('ASSISTANT', f"That did not work. {giver} still has the {o}."))
        if rng.random() < 0.4:
            other = rng.choice(w.objects)
            turns.append(('USER', f"Who has the {other} now?"))
            turns.append(('ASSISTANT', w.holder[other]))
    return turns


def render(turns):
    """The byte stream: CONTEXT lines carry no role tag (initial-state prefix, matching the rest of
    the curriculum's story-prefix convention); USER/ASSISTANT/ENVIRONMENT are all explicitly tagged."""
    lines = []
    for role, text in turns:
        if role == 'CONTEXT':
            lines.append(text)
        else:
            lines.append(f"{role}: {text}")
    return lines


def before_reveal_cases(turns):
    """Every (dialogue-prefix-ending-in-an-ASSISTANT-claim, the ENVIRONMENT text that follows it)
    pair in one dialogue: the prefix is what a prediction would be scored against, the environment
    text is revealed only afterward."""
    cases = []
    for i in range(1, len(turns)):
        if turns[i][0] == 'ENVIRONMENT' and turns[i - 1][0] == 'ASSISTANT':
            prefix = turns[:i]
            cases.append((render(prefix), turns[i][1]))
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out_dir')
    ap.add_argument('--n_dialogues', type=int, default=8000)
    ap.add_argument('--n_test', type=int, default=500)
    ap.add_argument('--seed', type=int, default=5)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    rng = random.Random(a.seed)
    train_path = os.path.join(a.out_dir, 'stage6_grounded_dialogue_train.txt')
    canon_path = os.path.join(a.out_dir, 'stage6_grounded_dialogue_test.jsonl')
    reveal_path = os.path.join(a.out_dir, 'stage6_grounded_dialogue_before_reveal_test.jsonl')
    n_train = n_test = n_reveal = 0
    with open(train_path, 'w', encoding='utf-8', newline='\n') as f, \
         open(canon_path, 'w', encoding='utf-8', newline='\n') as ef, \
         open(reveal_path, 'w', encoding='utf-8', newline='\n') as rf:
        for i in range(a.n_dialogues):
            turns = make_dialogue(rng)
            is_test = n_test < a.n_test and rng.random() < 0.05
            if is_test:
                ef.write(json.dumps({'dialogue_id': f'gd_{i}',
                                      'turns': [{'role': r, 'text': t} for r, t in turns]}) + '\n')
                n_test += 1
                for prefix_lines, env_text in before_reveal_cases(turns):
                    rf.write(json.dumps({'task': 'grounded_dialogue_before_reveal',
                                          'qa': [["What happens?", env_text, "", prefix_lines]]}) + '\n')
                    n_reveal += 1
            else:
                f.write('\n'.join(render(turns)) + '\n<|endofdialogue|>\n')
                n_train += 1
    print(f'grounded_dialogue: train {n_train}, test {n_test} (canonical), before-reveal cases {n_reveal}')


if __name__ == '__main__':
    main()
