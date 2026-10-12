"""Independent relational materialization and typed computational interpretation."""
from check_concept_growth import evaluate, require


def binding(frame, text):
    if not frame.startswith('computed:'): return None
    pattern, tokens = frame.removeprefix('computed:').split(), text.split()
    if len(pattern) != len(tokens): return None
    args, relations = [], []
    for p, value in zip(pattern, tokens):
        if p.startswith('$'):
            require(p == f'${len(args)}', 'invalid entity slot'); args.append(value)
        elif p.startswith('#'):
            if not value.isascii() or not value.isdigit() or len(value) > 9: return None
            require(p == f'#{len(args)}', 'invalid quantity slot'); args.append(value)
        elif p.startswith('%'):
            require(p == f'%{len(relations)}', 'invalid relation slot'); relations.append(value)
        elif p != value: return None
    require(len(args) == len(relations) == 2, 'invalid computed frame')
    return args, relations


def pairs(world, relation, definitions, args=()):
    if relation not in definitions:
        return {(v[0], v[1]) for p, v in world['facts'] if p == relation and len(v) == 2}
    entities = sorted(set(args) | {x for _, vs in world['facts'] for x in vs})
    return {(a, b) for a in entities for b in entities if evaluate(definitions[relation], dict(world, args=[a, b])) == 1}


def follow(relation, value):
    seen = set()
    while value not in seen:
        seen.add(value)
        nxt = sorted({b for a, b in relation if a == value})
        if not nxt: return value
        if len(nxt) != 1: return None
        value = nxt[0]
    return None


def count(relation, value): return len({b for a, b in relation if a == value})


def truth(meta, case, world):
    if not world['complete']: return -1
    a, b = case['truth']['args']
    r, s = (pairs(world, name, meta['definitions'], (a, b)) for name in (case['r'], case['s']))
    kind = case['kind']
    if kind == 0: return int(count(r, a) == count(s, b))
    x = follow(r, a)
    if kind == 1:
        y = follow(s, b)
        return int(x is not None and y is not None and x == y)
    if kind == 3: return int(x is not None and count(s, x) == int(b))
    require(kind == 2, 'invalid computational family')
    return int(x is not None and count(s, x) == count(s, b))


def program_answer(program, text, world, frame, definitions):
    b = binding(frame, text)
    if b is None or not world['complete']: return -1
    args, rel = b
    relations = [pairs(world, r, definitions, args) for r in rel]
    types = [2 if p.startswith('#') else 0 for p in frame.removeprefix('computed:').split() if p.startswith(('$', '#'))]
    inputs = [(t, int(a) if t == 2 else a) for t, a in zip(types, args)]
    regs = inputs.copy()
    for op in program:
        if op == 0: regs[0] = inputs[0]
        elif op == 1: regs[0] = inputs[1]
        elif op == 2: regs[1] = regs[0]
        elif op == 3: regs.reverse()
        elif op in (4, 5):
            if regs[0][0] != 0: return -1
            value = follow(relations[op % 2], regs[0][1])
            if value is None: return -1
            regs[0] = (0, value)
        elif op in (6, 7):
            if regs[0][0] != 0: return -1
            regs[0] = (2, count(relations[op % 2], regs[0][1]))
        elif op == 8: regs[0] = (1, int(regs[0] == regs[1]))
        elif op == 9: return regs[0][1] if regs[0][0] == 1 else -1
        else: raise AssertionError(f'unsupported computational instruction {op}')
    return -1
