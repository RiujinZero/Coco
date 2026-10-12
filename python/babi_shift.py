"""Template + lexical shift for bAbI test stories: unseen names, places, objects, verbs and sentence templates.

Train on the ORIGINAL stream; test on these rewritten stories. If the chain is relational it holds; if it
depends on the stop list and the exact "X went to the Y" shape, it collapses.

    python python/babi_shift.py data/babi_t1-3-15-16_test.jsonl data/babi_shift_test.jsonl [seed]
"""
import sys, json, re, random

NAMES = {'Mary': 'Priya', 'John': 'Tobias', 'Daniel': 'Kwame', 'Sandra': 'Ingrid', 'Bill': 'Hiroshi', 'Fred': 'Olamide', 'Jeff': 'Matteo', 'Julie': 'Zara',
         'Winona': 'Astrid', 'Emily': 'Noor', 'Jessica': 'Lucia', 'Gertrude': 'Marisol', 'Bernhard': 'Rafael', 'Lily': 'Imani', 'Brian': 'Soren', 'Julius': 'Dmitri', 'Greg': 'Ravi'}
PLACES = {'bathroom': 'cellar', 'hallway': 'balcony', 'kitchen': 'attic', 'garden': 'library', 'office': 'workshop', 'bedroom': 'porch'}
OBJECTS = {'apple': 'lantern', 'football': 'compass', 'milk': 'ribbon'}
ANIMALS = {'wolf': 'jackal', 'wolves': 'jackals', 'cat': 'fox', 'cats': 'foxes', 'mouse': 'hare', 'mice': 'hares', 'sheep': 'goat'}
COLORS = {'white': 'crimson', 'gray': 'amber', 'green': 'violet', 'yellow': 'teal'}
SPECIES = {'swan': 'heron', 'rhino': 'tapir', 'lion': 'lynx', 'frog': 'newt'}
MOVE = ['strolled over to the', 'made her way to the', 'hurried into the', 'wandered off to the', 'headed for the', 'ended up in the']
TAKE = ['picked up the', 'grabbed hold of the', 'collected the', 'took possession of the', 'snatched the']
DROP = ['set down the', 'abandoned the', 'let go of the', 'put down the', 'released the']

def lex(s):
    def sub(m, v):
        w = m.group(0); return v.capitalize() if w[0].isupper() else v.lower()
    for d in (NAMES, PLACES, OBJECTS, ANIMALS, COLORS, SPECIES):
        for k, v in d.items(): s = re.sub(r'\b' + k + r'\b', lambda m, v=v: sub(m, v), s, flags=re.IGNORECASE)
    return s

def template(s, rng):
    m = re.match(r'^(\w+) (went|moved|journeyed|travelled|went back) to the (\w+)\.$', s)
    if m: return f'{m.group(1)} {rng.choice(MOVE)} {m.group(3)}.'
    m = re.match(r'^(\w+) (picked up|grabbed|took|got) the (\w+)( there)?\.$', s)
    if m: return f'{m.group(1)} {rng.choice(TAKE)} {m.group(3)}.'
    m = re.match(r'^(\w+) (dropped|discarded|left|put down) the (\w+)( there)?\.$', s)
    if m: return f'{m.group(1)} {rng.choice(DROP)} {m.group(3)}.'
    m = re.match(r'^(\w+) are afraid of (\w+)\.$', s)
    if m: return rng.choice([f'{m.group(1)} are terrified of {m.group(2)}.', f'{m.group(1)} fear {m.group(2)}.', f'{m.group(1)} are frightened by {m.group(2)}.'])
    m = re.match(r'^(\w+) is a (\w+)\.$', s)
    if m: return rng.choice([f'{m.group(1)} happens to be a {m.group(2)}.', f'{m.group(1)} is one of the {m.group(2)}s.'.replace('ss.', 's.'), f'{m.group(1)} is a {m.group(2)}.'])
    m = re.match(r'^(\w+) is (\w+)\.$', s)
    if m: return rng.choice([f'{m.group(1)} is coloured {m.group(2)}.', f'{m.group(1)} looks {m.group(2)}.', f'{m.group(1)} is {m.group(2)}.'])
    return s

def main():
    src, out = sys.argv[1], sys.argv[2]; rng = random.Random(int(sys.argv[3]) if len(sys.argv) > 3 else 1)
    n = 0
    with open(out, 'w', encoding='utf-8', newline='\n') as o:
        for line in open(src, encoding='utf-8'):
            r = json.loads(line)
            for qa in r['qa']:
                qa[0] = lex(qa[0]); qa[1] = lex(qa[1]); qa[3] = [lex(template(x, rng)) for x in qa[3]]
            o.write(json.dumps(r) + '\n'); n += 1
    print('rewrote', n, 'stories ->', out)

if __name__ == '__main__': main()
