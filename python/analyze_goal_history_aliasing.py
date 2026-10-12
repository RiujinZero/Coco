"""Read-only public-history alias/backup diagnosis for the completed goal pilot.

No weights, rewards, environment state or training records are changed. Backup
checks use the final frozen warm value table, not an invented historical table.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path


def rows(path):
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            yield json.loads(line)


def analyze(pilot, diagnostic):
    histories = {}
    aliases = collections.defaultdict(lambda: {'counts': collections.Counter(), 'examples': {}, 'observation': None, 'menu': None})
    traces = collections.defaultdict(list)
    mission_results = {row['id']: row for row in rows(pilot / 'pilot_missions.jsonl')}
    for row in rows(pilot / 'pilot_trace.jsonl'):
        mission = row['mission']
        state = histories.setdefault(mission, {'requests': {}, 'closed': set(), 'failure': False})
        if row['event'] == 'public_request':
            state['requests'][row['id']] = row
        elif row['event'] == 'decision':
            # A decision with clock>D cannot close an unfinished request by D.
            expired = [goal for goal, request in state['requests'].items()
                       if goal not in state['closed'] and request['deadline'] and row['clock'] > request['deadline']]
            doomed = state['failure'] or bool(expired)
            context = 'publicly_doomed' if doomed else 'not_yet_publicly_doomed'
            key = (row['observation_bits'], tuple(row['menu']))
            group = aliases[key]
            group['counts'][context] += 1
            group['observation'] = row['observation']
            group['menu'] = row['menu']
            group['examples'].setdefault(context, {
                'mission': mission, 'clock': row['clock'], 'operation': row['operation'],
                'active': row['active_before'], 'expired_unfinished_requests': expired,
                'prior_failure': state['failure'],
            })
            traces[mission].append(row)
            if row['operation'] == 4:
                goal = row['active_before']
                request = state['requests'][goal]
                state['closed'].add(goal)
                state['failure'] |= not row['terminal_correct'] or bool(request['deadline'] and row['clock'] > request['deadline'])

    mixed = []
    for (bits, _), group in aliases.items():
        if len(group['counts']) == 2:
            mixed.append({'observation_bits': bits, **group})
    values = {row['observation_bits']: row['values'] for row in rows(diagnostic / 'diagnostic_warm_empirical.jsonl')}
    backup = collections.Counter()
    examples = []
    for name, trace in traces.items():
        if not name.startswith('warm_') or mission_results[name]['success']:
            continue
        # Exactly the first reverse suffix portion of a128-update call. Later
        # random old-receipt selections require historical RNG, so are omitted.
        for index in range(max(0, len(trace) - 128), len(trace)):
            target = -1.0
            cutoffs = 0
            for later in reversed(trace[index + 1:]):
                q = values.get(later['observation_bits'], [0.0] * 7)
                best = max(q[action] for action in later['menu'])
                if q[later['operation']] < best:
                    target = .99 * best
                    cutoffs += 1
                else:
                    target = .99 * (target - later['cost'])
            target -= trace[index]['cost']
            plain = -.99 ** (len(trace) - index - 1)
            backup['examined_updates'] += 1
            backup['nonnegative_despite_failed_mission'] += target >= 0
            backup['positive_despite_failed_mission'] += target > 0
            backup['had_greedy_cutoff'] += cutoffs > 0
            if target >= 0 and len(examples) < 12:
                examples.append({'mission': name, 'clock': trace[index]['clock'], 'operation': trace[index]['operation'],
                                 'frozen_table_watkins_target': target, 'plain_return': plain, 'cutoffs': cutoffs})
    satisfied = [row for row in mixed if row['observation'][29] == 0 and row['observation'][4] == 1 and row['observation'][6] == 1]
    return {
        'schema': 'goal_public_history_alias_diagnostic_v1',
        'scope': 'Exact public-input aliasing and frozen-final-warm-table backup check; not historical update reconstruction or a learning result.',
        'total_observation_menu_classes': len(aliases), 'mixed_history_classes': len(mixed),
        'mixed_cached_true_physical_classes': len(satisfied),
        'mixed_classes': mixed, 'frozen_table_failure_backups': dict(backup), 'backup_examples': examples,
        'interpretation': 'The32-float/menu state can omit already irreversible mission failure. A common feasible policy may still exist; this does not by itself prove no controller can solve the task. Under zero costs and discount<1, postponing an already certain failure improves return, so history can change optimal finish timing.',
        'hashes': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in
                   [pilot/'pilot_trace.jsonl', pilot/'pilot_missions.jsonl', diagnostic/'diagnostic_warm_empirical.jsonl']},
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('pilot', type=Path)
    parser.add_argument('diagnostic', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists(), 'refusing to overwrite an earlier diagnostic'
    result = analyze(args.pilot, args.diagnostic)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: value for key, value in result.items() if key not in ['mixed_classes', 'hashes', 'backup_examples']}, indent=2))
