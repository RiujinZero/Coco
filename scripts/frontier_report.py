"""Summary tables for model_frontier results (scripts/frontier_eval.sh -> logs/frontier/*.jsonl).

usage: python scripts/frontier_report.py FILE.jsonl [FILE.jsonl ...]          Markdown table
       python scripts/frontier_report.py --topic FILE.jsonl [FILE.jsonl ...]  quality and topic tables (pmi runs)
       python scripts/frontier_report.py --profile FILE.jsonl [...]           the prompt's effect by byte position (profile=1 runs)
       python scripts/frontier_report.py --samples "PROMPT|PROMPT" FILE.jsonl [...]   side-by-side outputs for the doc
       python scripts/frontier_report.py --dump FILE.jsonl [FILE.jsonl ...]   every output, grouped by prompt
One Markdown row per (file, mode) with the per-output means the app prints, plus one metric the app cannot compute
alone: cross-prompt 4-gram overlap, the fraction of an output's word 4-grams that also occur in the output the SAME
decoder wrote for a DIFFERENT prompt (an attractor that every prompt falls into shows up here, not in the loop rate).
--topic adds the topic measures: topic words = the fraction of the prompt's content words (>= 3 letters, not in STOP)
whose stem reappears in the output (prompts without content words are left out), also in the outputs the same decoder
wrote for the other prompts (chance level) and in the output's second half alone; echo = outputs that repeat a word
trigram of the prompt; line 1 = bytes before the output's first newline; and from the app (model replays): gain =
log-prob per byte with the prompt minus without it, late gain = the same over the output's second half, reach = 1 + the
last output byte whose log-prob the prompt changes by > 0.01 nats / at all (medians), ident@1 = outputs more probable
after their own prompt than after each of the others (late: their second half).
"""
import json
import statistics
import sys

STOP = set("""a about above after again against ago all also am an and another any are as at away back be because been
before being below between both but by can could did do does doing done don don't down during each even ever every few
for from further get gets getting go goes got had has have having he her here hers herself him himself his how i if in
into is it its itself just let like me more most much my myself never no nor not now of off oh ok okay on once one only
or other our ours ourselves out over own really said same say says she should so some such than that the their theirs
them themselves then there these they this those through to too under until up upon us very was way we well were what
when where which while who whom why will with would yeah yes you your yours yourself yourselves mot chi fat
gonna wanna""".split())


def stem(w):  # plural s, then -ing / -ed, keeping at least 3 letters
    if len(w) >= 4 and w.endswith('s') and not w.endswith('ss'):
        w = w[:-1]
    for suf in ('ing', 'ed'):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[:-len(suf)]
    return w


def strwords(b):
    return [w.decode('latin-1') for w in words(b)]


def content_words(b):
    return [w for w in dict.fromkeys(strwords(b)) if len(w) >= 3 and w.isalpha() and w not in STOP]


def topic_words(prompt, out):
    cw = content_words(raw(prompt))
    stems = {stem(w) for w in strwords(raw(out))}
    return sum(stem(w) in stems for w in cw) / len(cw) if cw else None


def topic_control(rs):
    """chance level of topic words: a prompt's content words found in the same decoder's outputs for the OTHER prompts"""
    vals = []
    for i, r in enumerate(rs):
        o = [topic_words(r['prompt'], s['output']) for j, s in enumerate(rs) if j != i]
        o = [x for x in o if x is not None]
        if o:
            vals.append(sum(o) / len(o))
    return sum(vals) / len(vals)


def topic_of(r):
    """(topic words or None, topic words in the output's second half or None, echo, line-1 bytes) of one output"""
    prompt, out = raw(r['prompt']), raw(r['output'])
    topic = topic_words(r['prompt'], r['output'])
    late = topic_words(r['prompt'], r['output'][len(r['output']) // 2:])
    p3 = set(grams(strwords(prompt), 3))
    echo = any(g in p3 for g in grams(strwords(out), 3))
    nl = out.find(b'\n')
    return topic, late, echo, (nl if nl >= 0 else len(out))


def decoder(r):
    if r['mode'] == 'greedy':
        return 'greedy'
    if r['mode'] == 'sample':
        return 'sample T=%g' % r['T']
    parts = ['frontier']
    for k, default in (('W', 4), ('N', 4), ('key', 3), ('T', 0.8), ('mem', 1), ('rep', 0), ('pmi', 0), ('copy', 0)):
        if k in r and float(r[k]) != default:
            parts.append('%s=%g' % (k, float(r[k])))
    if r.get('stop', 'len') != 'len':
        parts.append('stop=%s' % r['stop'])
    return ' '.join(parts)


def cross_prompt(rs):
    g4 = [set(grams(words(raw(r['output'])), 4)) for r in rs]
    cross = []
    for i, gi in enumerate(g4):
        other = set().union(*(g for j, g in enumerate(g4) if j != i))
        cross.append(len(gi & other) / len(gi) if gi else 0.0)
    return sum(cross) / len(rs)


def topic_tables(paths):
    groups = []
    for path in paths:
        rows = [json.loads(line) for line in open(path, encoding='utf-8')]
        tag = path.replace('\\', '/').split('/')[-1].replace('.jsonl', '')
        for dec in dict.fromkeys(decoder(r) for r in rows):
            groups.append((tag, dec, [r for r in rows if decoder(r) == dec]))
    q = ['run', 'decoder', 'n', 'loops', 'strict loops', 'distinct-2', 'distinct-3', 'valid words', 'lp read', 'lp own', 'copied',
         'cross-prompt', 's/sample', 'events/sample']
    print('| ' + ' | '.join(q) + ' |')
    print('|' + '---|' * len(q))
    for tag, dec, rs in groups:
        n = len(rs)
        mean = lambda k: sum(float(r[k]) for r in rs) / n
        print('| %s | %s | %d | %d/%d | %d/%d | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.2f | %.0f |' % (
            tag, dec, n, sum(int(r['loop4']) for r in rs), n, sum(int(r['strict_loop']) for r in rs), n, mean('distinct2'), mean('distinct3'),
            mean('valid_words'), mean('lp_forced'), mean('lp_self'), mean('copy32'), cross_prompt(rs), mean('seconds'), mean('model_bytes')))
    print()
    t = ['run', 'decoder', 'topic words', 'topic words, other prompts\' outputs', 'topic words, 2nd half', 'echo', 'line 1 bytes', 'gain', 'late gain',
         'reach >0.01 (median)', 'reach any (median)', 'ident@1', 'ident@1 late', 'mean rank']
    print('| ' + ' | '.join(t) + ' |')
    print('|' + '---|' * len(t))
    for tag, dec, rs in groups:
        n = len(rs)
        tp = [topic_of(r) for r in rs]
        tw = [x[0] for x in tp if x[0] is not None]
        tl = [x[1] for x in tp if x[1] is not None]
        ranked = [r for r in rs if 'rank' in r]
        m = len(ranked)
        print('| %s | %s | %.3f | %.3f | %.3f | %d/%d | %.0f | %.3f | %.3f | %.0f | %.0f | %s | %s | %s |' % (
            tag, dec, sum(tw) / len(tw), topic_control(rs), sum(tl) / len(tl), sum(x[2] for x in tp), n, sum(x[3] for x in tp) / n,
            sum(float(r['gain']) for r in rs) / n, sum(float(r['gain_late']) for r in rs) / n,
            statistics.median(float(r['reach01']) for r in rs), statistics.median(float(r['reach']) for r in rs),
            '%d/%d' % (sum(int(r['rank']) == 1 for r in ranked), m) if m else '-',
            '%d/%d' % (sum(int(r['rank_late']) == 1 for r in ranked), m) if m else '-',
            '%.1f' % (sum(int(r['rank']) for r in ranked) / m) if m else '-'))


def raw(s):  # the app writes every byte >= 0x80 as \\u00XX: back to the raw byte string
    return s.encode('latin-1')


def words(b):  # the app's tokenizer: runs of alphanumerics / bytes >= 0x80 (+ apostrophes inside a word), lowercased
    out, cur = [], bytearray()
    def flush():
        while cur and cur[-1] == ord("'"):
            cur.pop()
        if cur:
            out.append(bytes(cur).lower())
        cur.clear()
    for c in b:
        if chr(c).isalnum() and c < 0x80 or c >= 0x80 or (c == ord("'") and cur):
            cur.append(c)
        else:
            flush()
    flush()
    return out


def grams(w, n):
    return [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]


def main(paths):
    cols = ['file', 'mode', 'n', 'bytes', 'loop4', 'strict loop', 'distinct-2', 'distinct-3', 'lp forced', 'lp self', 'valid words',
            'copy32', 'cross-prompt 4g', 's/sample', 'events/sample', 'merged', 'blocked', 'relaxed']
    print('| ' + ' | '.join(cols) + ' |')
    print('|' + '---|' * len(cols))
    for path in paths:
        rows = [json.loads(line) for line in open(path, encoding='utf-8')]
        for mode in dict.fromkeys(r['mode'] for r in rows):
            rs = [r for r in rows if r['mode'] == mode]
            n = len(rs)
            mean = lambda k: sum(float(r[k]) for r in rs) / n
            print('| %s | %s | %d | %.0f | %.2f | %.2f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.2f | %.0f | %.1f | %.1f | %.2f |' % (
                path.replace('\\', '/').split('/')[-1].replace('.jsonl', ''), mode, n, mean('bytes'), mean('loop4'), mean('strict_loop'),
                mean('distinct2'), mean('distinct3'), mean('lp_forced'), mean('lp_self'), mean('valid_words'), mean('copy32'),
                cross_prompt(rs), mean('seconds'), mean('model_bytes'), mean('merged'), mean('blocked'), mean('relaxed')))


def label(path, r):
    tag = path.replace('\\', '/').split('/')[-1].replace('.jsonl', '')
    if r['mode'] == 'greedy':
        return '%s greedy' % tag
    if r['mode'] == 'sample':
        return '%s sample T=%g' % (tag, r['T'])
    return '%s frontier W=%d N=%d key=%d T=%g mem=%d copy=%g%s stop=%s' % (tag, r['W'], r['N'], r['key'], r['T'], r['mem'], r.get('copy', 0),
                                                                          ' pmi=%g' % r['pmi'] if r.get('pmi', 0) else '', r.get('stop', 'len'))


def dump(paths):
    runs = [(p, [json.loads(line) for line in open(p, encoding='utf-8')]) for p in paths]
    prompts = list(dict.fromkeys(r['prompt'] for r in runs[0][1]))
    print('model_frontier outputs (see docs/FRONTIER_DECODER.md), grouped by prompt; runs: %s' % ', '.join(
        p.replace('\\', '/').split('/')[-1].replace('.jsonl', '') for p in paths))
    print('Each output follows the prompt directly (read after a newline); tabs shown as 4 spaces, other control bytes as \\xNN,'
          ' invalid UTF-8 as U+FFFD.')
    for i, prompt in enumerate(prompts):
        print('=' * 100)
        print('PROMPT %d: %s' % (i + 1, raw(prompt).decode('utf-8', 'replace').replace('\t', '\\t')))
        for path, rows in runs:
            for r in rows:
                if r['prompt'] != prompt:
                    continue
                extra = ', gain %.3f, reach %d' % (r['gain'], r['reach01']) if 'gain' in r else ''
                print('--- %s  [lp/byte %.3f, copied %.2f, loop4 %d, distinct-2 %.2f%s]' % (label(path, r), r['lp_self'], r['copy32'], r['loop4'], r['distinct2'], extra))
                out = raw(r['output']).decode('utf-8', 'replace').replace('\t', '    ')
                out = ''.join(ch if ch == '\n' or (ord(ch) >= 32 and ord(ch) != 127) else '\\x%02x' % ord(ch) for ch in out)
                print('\n'.join('    ' + line for line in out.split('\n')))
        print()


def profile(paths, edges=(0, 4, 8, 16, 32, 64, 128, 256)):
    """runs made with profile=1: how much the prompt changes the log-prob of an output byte, by its position in the output
    (mean |change| in nats, and the fraction of bytes changed by more than 0.01 nats)"""
    buckets = list(zip(edges[:-1], edges[1:]))
    head = ['run', 'decoder'] + ['bytes %d-%d' % (a, b - 1) for a, b in buckets]
    print('| ' + ' | '.join(head) + ' |')
    print('|' + '---|' * len(head))
    for path in paths:
        rows = [json.loads(line) for line in open(path, encoding='utf-8')]
        tag = path.replace('\\', '/').split('/')[-1].replace('.jsonl', '')
        for dec in dict.fromkeys(decoder(r) for r in rows):
            cells = []
            for a, b in buckets:
                v = [abs(x) for r in rows if decoder(r) == dec for x in r['dlp'][a:b]]
                cells.append('%.3f (%.0f%%)' % (sum(v) / len(v), 100.0 * sum(x > 0.01 for x in v) / len(v)) if v else '-')
            print('| %s | %s | %s |' % (tag, dec, ' | '.join(cells)))


def samples(prompts, paths, width=100):
    """side-by-side blocks for the doc: per prompt, every (run, decoder) output, label left, wrapped at `width` columns"""
    import textwrap
    runs = [(p.replace('\\', '/').split('/')[-1].replace('.jsonl', ''), [json.loads(line) for line in open(p, encoding='utf-8')]) for p in paths]
    for prompt in prompts:
        print('Prompt `%s`:\n\n```' % prompt.replace('\t', '<tab>'))
        for tag, rows in runs:
            for r in rows:
                if raw(r['prompt']).decode('utf-8', 'replace') != prompt:
                    continue
                lab = '%s %s' % (tag.split('_')[0], decoder(r))
                out = raw(r['output']).decode('utf-8', 'replace').replace('\t', '  ').strip('\n')
                lines = []
                for para in out.split('\n'):
                    lines += textwrap.wrap(para.strip(), width) or ['']
                for i, line in enumerate(lines):
                    print((lab.ljust(31) if i == 0 else ' ' * 31) + line)
        print('```\n')


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8', newline='\n')
    if sys.argv[1:2] == ['--dump']:
        dump(sys.argv[2:])
    elif sys.argv[1:2] == ['--topic']:
        topic_tables(sys.argv[2:])
    elif sys.argv[1:2] == ['--samples']:   # --samples "PROMPT|PROMPT" FILE.jsonl ...  (\t in a prompt: a tab)
        samples(sys.argv[2].replace('\\t', '\t').split('|'), sys.argv[3:])
    elif sys.argv[1:2] == ['--profile']:   # runs made with profile=1
        profile(sys.argv[2:])
    else:
        main(sys.argv[1:])
