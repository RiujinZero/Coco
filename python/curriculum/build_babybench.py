"""LLM-BabyBench Predict (manifest S7, optional event probe) -> Stage 5 state-prediction episodes.

"Start with Predict; create grouped splits using level and seed metadata, plus task-family holdouts."
Offline precomputed BabyAI transitions: env description + initial state + an action sequence ->
target state. No interactive MiniGrid environment is needed for this offline form. Held out by
level_name (a level never seen in training appears only in the test split), which is a task-family
holdout rather than a random row split.

    python python/curriculum/build_babybench.py SRC_CSV OUT_DIR [--holdout_levels N]
"""
import argparse, csv, json, os, random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src_csv')
    ap.add_argument('out_dir')
    ap.add_argument('--holdout_levels', type=int, default=3)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    rows = list(csv.DictReader(open(a.src_csv, encoding='utf-8')))
    levels = sorted(set(r['level_name'] for r in rows))
    rng = random.Random(3)
    rng.shuffle(levels)
    holdout = set(levels[:a.holdout_levels])
    print(f'{len(levels)} levels, holding out {sorted(holdout)} entirely for test')

    train_path = os.path.join(a.out_dir, 'stage5_babybench_train.txt')
    test_path = os.path.join(a.out_dir, 'stage5_babybench_test.jsonl')
    n_train = n_test = 0
    with open(train_path, 'w', encoding='utf-8', newline='\n') as tf, \
         open(test_path, 'w', encoding='utf-8', newline='\n') as ef:
        for r in rows:
            desc = r['env_description'].strip()
            init = r['initial_state'].strip()
            actions = r['action_sequence'].strip()
            target = r['target_state'].strip()
            story = [desc, f"The agent starts at state {init}.", f"The agent takes these actions in order: {actions}."]
            q = "What is the agent's state now?"
            if r['level_name'] in holdout:
                ef.write(json.dumps({'task': f"babybench_{r['level_name']}", 'qa': [[q, target, '', story]]}) + '\n')
                n_test += 1
            else:
                tf.write('\n'.join(story) + '\nUSER: ' + q + '\n\nASSISTANT: ' + target + '\n<|endofdialogue|>\n')
                n_train += 1
    print(f'babybench: train {n_train}, test {n_test}')


if __name__ == '__main__':
    main()
