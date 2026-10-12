"""TextWorld (manifest S6, core) -> Stage 5/6 action/consequence rollouts.

Runs entirely inside a Python process (no tw-make CLI, no on-disk game directory tree): generates a
game, compiles it once to a temp .z8, plays it with the GAME'S OWN walkthrough (an oracle/scripted
policy -- allowed for offline demonstration collection, "but log it": every episode's oracle bit is
recorded so it is never confused with a learner-chosen trajectory). The learner-facing stream gets
only what the manifest calls the learner stream: the environment's narration (as a USER turn, since
it is what presents the situation) and the action taken (as an ASSISTANT turn) and the environment's
resulting feedback (the next USER turn) -- never the walkthrough as a plan handed in advance, never
the quest's internal graph.

TextWorld's own dependency stack needs Linux (or the Windows Docker workaround); this runs inside
WSL, so the local machine never needs the Windows-side interpreter to see the package at all.

    /home/user/tw_venv/bin/python build_textworld.py OUT_DIR --n_games 4000
"""
import argparse, json, os, random, re, sys, tempfile

import textworld
from textworld.generator import make_game, compile_game
from textworld.generator.game import GameOptions

# every feedback string ends with a rendered status bar ("> ... -= Room =-score/moves"); the very
# first one is preceded by an ASCII banner and, critically, the game's OWN "objective" text, which
# states the walkthrough in prose ("First thing I need you to do is ... And then ... Then ...") --
# exactly the "oracle walkthrough or plan" the manifest says must never reach the learner. Both are
# stripped: the banner+objective by cutting to the first room header, the trailing status bar by
# pattern (it repeats verbatim in structure on every turn).
STATUS_BAR = re.compile(r'\n>\s*-=.*?=-\s*\d+/\d+\s*$', re.S)


def clean_feedback(fb, is_first):
    fb = STATUS_BAR.sub('', fb)
    if is_first:
        i = fb.find('-= ')
        if i >= 0:
            fb = fb[i:]
    return ' '.join(fb.split())


def one_game(seed, rng):
    opts = GameOptions()
    opts.seeds = seed
    opts.nb_rooms = rng.randint(4, 9)
    opts.nb_objects = rng.randint(6, 12)
    opts.quest_length = rng.randint(3, 6)
    opts.quest_breadth = 1
    opts.path = tempfile.mktemp(suffix='.z8')
    game = make_game(opts)
    gamefile = compile_game(game, opts)
    return game, gamefile


def play_walkthrough(game, gamefile):
    wt = game.metadata.get('walkthrough') or []
    if not wt:
        return None
    env = textworld.start(gamefile)
    gs = env.reset()
    turns = [('USER', clean_feedback(gs.feedback, is_first=True))]
    total_score = 0
    for cmd in wt:
        gs, score, done = env.step(cmd)
        turns.append(('ASSISTANT', cmd))
        fb = clean_feedback(gs.feedback or '', is_first=False)
        if fb:
            turns.append(('USER', fb))
        total_score = score
    env.close()
    return turns, total_score, game.metadata.get('max_score', total_score)


def render(turns):
    lines = []
    for role, text in turns:
        text = ' '.join(text.split())
        if not text:
            continue
        lines.append(f"{role}: {text}")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out_dir')
    ap.add_argument('--n_games', type=int, default=4000)
    ap.add_argument('--n_test', type=int, default=200)
    ap.add_argument('--seed', type=int, default=11)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    oracle_dir = os.path.join(a.out_dir, '..', '_oracle')
    os.makedirs(oracle_dir, exist_ok=True)

    rng = random.Random(a.seed)
    train_path = os.path.join(a.out_dir, 'stage5_textworld_train.txt')
    test_path = os.path.join(a.out_dir, 'stage5_textworld_test.jsonl')
    oracle_path = os.path.join(oracle_dir, 'textworld_oracle.jsonl')
    n_train = n_test = n_fail = 0
    with open(train_path, 'w', encoding='utf-8', newline='\n') as tf, \
         open(test_path, 'w', encoding='utf-8', newline='\n') as ef, \
         open(oracle_path, 'w', encoding='utf-8', newline='\n') as of:
        for i in range(a.n_games):
            seed = a.seed * 1000003 + i
            try:
                game, gamefile = one_game(seed, rng)
                result = play_walkthrough(game, gamefile)
            except Exception as e:
                n_fail += 1
                if n_fail <= 5:
                    print(f'FAIL seed={seed}: {type(e).__name__}: {e}', file=sys.stderr)
                continue
            finally:
                try:
                    os.remove(gamefile)
                except Exception:
                    pass
            if not result:
                continue
            turns, score, max_score = result
            lines = render(turns)
            if len(lines) < 4:
                continue
            of.write(json.dumps({'id': f'tw_{i}', 'seed': seed, 'score': score, 'max_score': max_score,
                                  'oracle_policy': 'walkthrough', 'n_turns': len(lines)}) + '\n')
            if n_test < a.n_test and rng.random() < 0.05:
                ef.write(json.dumps({'task': 'textworld_quest', 'lines': lines, 'score': score, 'max_score': max_score}) + '\n')
                n_test += 1
            else:
                tf.write('\n\n'.join(lines) + '\n<|endofdialogue|>\n')
                n_train += 1
            if (i + 1) % 200 == 0:
                print(f'{i + 1}/{a.n_games} games, train {n_train} test {n_test} failed {n_fail}', flush=True)
    print(f'textworld: train {n_train}, test {n_test}, failed {n_fail}')


if __name__ == '__main__':
    main()
