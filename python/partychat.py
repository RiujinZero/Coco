"""Long-chat aggregation benchmark ("how many donuts for the party?" after dozens of turns of tangents).

A conversation mixes real chat turns (tangents: SmolTalk for training, LMSYS for testing) with a user talking about
coworkers: roles, workplace drama, and — scattered, in varied phrasing — who is coming to a party, who can't, who
brings guests, and revisions ("actually X can't come after all"). The last user turn asks a question whose answer
must be FOUND among the relevant turns and AGGREGATED: a head count (a sum that never appears in the text), a yes/no
for one person, or one person's guest count.

Honesty rules (the design must not be tuned to this set):
  * names, fact phrasings, question phrasings and distractor sources are DISJOINT between train and test;
  * test conversations are longer than any in training (tangent turns up to 60 vs 20);
  * the test file is written once and not inspected while designing mechanisms.

    python python/partychat.py data/partychat --train 6000 --test 400
Writes <out>_train.txt and <out>_test.jsonl ({"task": "<qtype>_d<bucket>", "qa": [[question, answer, "", [turns...]]]}).
"""
import sys, json, random, re, os

NAMES_TRAIN = ['Priya', 'Tom', 'Dana', 'Marcus', 'Elena', 'Kofi', 'Hana', 'Luis', 'Greta', 'Omar', 'Wen', 'Ruth', 'Ivan', 'Nia', 'Sam', 'Beth']
NAMES_TEST = ['Anika', 'Brody', 'Celeste', 'Desmond', 'Farah', 'Gideon', 'Imogen', 'Jonah', 'Keiko', 'Leandro', 'Mirela', 'Noor', 'Otis', 'Paloma']
PRON = {n: random.Random(n).choice([('she', 'her'), ('he', 'his'), ('they', 'their')]) for n in NAMES_TRAIN + NAMES_TEST}
DEPTS = ['accounting', 'sales', 'design', 'support', 'engineering', 'legal', 'marketing', 'ops']
TOPICS = ['the budget', 'the parking spots', 'the new printer', 'who ate the leftovers', 'the Q3 deadline', 'the thermostat']

YES_TR = ['{n} said {p} will come to the party.', '{n} is in for Friday.', 'Count {n} in for the party.', '{n} confirmed {p} is coming.', '{n} told me {p} will be at the party.']
YES_TE = ["{n} RSVP'd yes to the party.", 'Looks like {n} will be there Friday too.', '{n} is definitely joining us for the party.']
NO_TR = ["{n} can't make it to the party.", "{n} is out, {p} is travelling that weekend.", '{n} said no to the party.']
NO_TE = ['{n} has to skip the party, family thing.', 'Scratch {n}, {p} is sick that day.']
GUEST_TR = ['{n} is bringing {k} guests.', '{n} will bring {k} people along.']
GUEST_TE = ["{n}'s bringing {k} extra people with {pp}.", '{n} asked if {k} friends could tag along, I said yes.']
REV_NO_TR = ["Actually {n} can't come after all.", '{n} just cancelled on the party.']
REV_NO_TE = ["Update: {n} dropped out of the party.", "Ugh, {n} bailed on Friday."]
REV_YES_TR = ['{n} changed {pp} mind and is coming to the party.', 'Good news, {n} can make it after all.']
REV_YES_TE = ['{n} rearranged things and will be at the party now.']
DRAMA = ['{n} works in {d}.', '{n} and {m} argued about {t} again today.', '{n} just got promoted to lead in {d}.', '{n} has been covering for {m} all week.',
         "{n} thinks {m} took credit for {pp} slides.", '{n} moved desks to sit near {m}.']
Q_COUNT_TR = ['How many people are coming to the party?', 'How many people should I plan for at the party?']
Q_COUNT_TE = ["What's the headcount for Friday's party?", 'So how many donuts do I need if everyone at the party gets one?']
Q_YES_TR = ['Is {n} coming to the party?']
Q_YES_TE = ['Will {n} be at the party on Friday?']
Q_GUEST_TR = ['How many guests is {n} bringing?']
Q_GUEST_TE = ['How many extra people is {n} bringing?']
ACKS = ['Got it.', 'Noted!', 'Okay, thanks for the update.', "Sounds good, I'll keep track.", 'Thanks, noted.', 'Oh no, that sounds stressful.', 'Interesting!']

def chat_turns(path, rng, limit):
    """(user, assistant) pairs from a chat stream in the USER:/ASSISTANT: format."""
    text = open(path, encoding='utf-8', errors='replace').read(limit)
    pairs = re.findall(r'USER:(.*?)\n\nASSISTANT:(.*?)(?=\n\nUSER:|\n<\|endofdialogue\|>)', text, flags=re.S)
    pairs = [(u.strip(), a.strip()) for u, a in pairs if 10 < len(u) < 400 and 10 < len(a) < 600]
    rng.shuffle(pairs); return pairs

def conversation(rng, names, fr, tangents, n_tangent, test):
    Y, N, G, RN, RY = (YES_TE, NO_TE, GUEST_TE, REV_NO_TE, REV_YES_TE) if test else (YES_TR, NO_TR, GUEST_TR, REV_NO_TR, REV_YES_TR)
    people = rng.sample(names, rng.randint(4, 8))
    status, guests, facts = {}, {}, []
    for n in people:
        p, pp = PRON[n]
        roll = rng.random()
        if roll < 0.6: status[n] = True; facts.append(rng.choice(Y).format(n=n, p=p, pp=pp))
        elif roll < 0.85: status[n] = False; facts.append(rng.choice(N).format(n=n, p=p, pp=pp))
        else: status[n] = None                                  # mentioned only in drama: never said whether they come
        if status[n] and rng.random() < 0.35:
            k = rng.randint(1, 3); guests[n] = k; facts.append(rng.choice(G).format(n=n, k=k, pp=pp))
    revs = []                                                   # revisions, stated after the original fact
    for n in [x for x in people if status[x] is not None]:
        if rng.random() < 0.2:
            p, pp = PRON[n]
            if status[n]: status[n] = False; guests.pop(n, None); revs.append(rng.choice(RN).format(n=n, p=p, pp=pp))
            else: status[n] = True; revs.append(rng.choice(RY).format(n=n, p=p, pp=pp))
    drama = []
    for _ in range(rng.randint(3, 8)):
        n, m = rng.sample(people, 2)
        drama.append(rng.choice(DRAMA).format(n=n, m=m, d=rng.choice(DEPTS), t=rng.choice(TOPICS), pp=PRON[n][1]))
    items = [('fact', f) for f in facts] + [('drama', d) for d in drama]
    rng.shuffle(items); items += [('fact', r) for r in revs]   # revisions come later than what they revise
    tan = [('tangent', tangents.pop()) for _ in range(min(n_tangent, len(tangents)))]
    seq = items[:]
    for t in tan: seq.insert(rng.randint(0, len(seq)), t)      # insertion keeps the relative order of facts
    turns = []
    for kind, x in seq:
        if kind == 'tangent': turns.append(('USER', x[0])); turns.append(('ASSISTANT', x[1]))
        else: turns.append(('USER', x)); turns.append(('ASSISTANT', rng.choice(ACKS)))
    # the question
    qt = rng.choice(['count', 'count', 'yes', 'guest'])
    if qt == 'guest' and not guests: qt = 'count'
    if qt == 'count':
        q = rng.choice(Q_COUNT_TE if test else Q_COUNT_TR); a = str(sum(1 + guests.get(n, 0) for n in people if status[n]))
    elif qt == 'yes':
        n = rng.choice([x for x in people if status[x] is not None]); q = rng.choice(Q_YES_TE if test else Q_YES_TR).format(n=n); a = 'yes' if status[n] else 'no'
    else:
        n = rng.choice(list(guests)); q = rng.choice(Q_GUEST_TE if test else Q_GUEST_TR).format(n=n); a = str(guests[n])
    return turns, q, a, qt

def render(turns, q, a):
    body = ''.join(f'{r}: {x}\n\n' for r, x in turns)
    return body + 'USER: ' + q + '\n\nASSISTANT: ' + a + '\n<|endofdialogue|>\n'

def main():
    out = sys.argv[1]; args = sys.argv[2:]
    ntr = int(args[args.index('--train') + 1]) if '--train' in args else 6000
    nte = int(args[args.index('--test') + 1]) if '--test' in args else 400
    rng = random.Random(11)
    tr_tan = chat_turns('data/smoltalk.txt', rng, 2_000_000)
    te_tan = chat_turns('data/lmsys_128m.txt', random.Random(12), 20_000_000)
    with open(out + '_train.txt', 'w', encoding='utf-8', newline='\n') as f:
        for i in range(ntr):
            pool = tr_tan[:]; rng.shuffle(pool)
            turns, q, a, qt = conversation(rng, NAMES_TRAIN, None, pool, rng.randint(0, 20), False)
            f.write(render(turns, q, a))
    trng = random.Random(13)
    with open(out + '_test.jsonl', 'w', encoding='utf-8', newline='\n') as f:
        for i in range(nte):
            nt = [0, 10, 30, 60][i % 4]
            pool = te_tan[:]; trng.shuffle(pool)
            turns, q, a, qt = conversation(trng, NAMES_TEST, None, pool, nt, True)
            story = [f'{r}: {x}\n' for r, x in turns]              # the harness adds one more newline: turns end in a blank line as in training
            f.write(json.dumps({'task': f'{qt}_d{nt:02d}', 'qa': [[q, a, '', story]]}, ensure_ascii=False) + '\n')
    print(f'train {ntr} conversations ({os.path.getsize(out + "_train.txt") / 1e6:.1f} MB, tangents 0-20, SmolTalk); '
          f'test {nte} (tangents 0/10/30/60, LMSYS; unseen names, phrasings, questions)')

if __name__ == '__main__': main()
