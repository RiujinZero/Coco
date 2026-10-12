"""Independent public-state, neural-choice, physics, outcome and evidence audit.

Reads the actual native trace; reconstructs every controller input without a
hidden answer feature. A separate relational interpreter supplies evaluation
gold. No passing-score threshold is imposed: failures remain in the report.
"""
import argparse
import collections
import csv
import functools
import hashlib
import json
import math
import shlex
from pathlib import Path
from check_concept_growth import parse_world, parse_clause, evaluate, intended, oracle, require
from check_relational_session import canonical, action, effect, parse_snapshot, audit_snapshot
from relational_session_probe import names
import computational_control_oracle as computational_oracle
from check_primitive_discovery import gold as primitive_gold
from check_preference_tree import readout as tree_readout
from learned_computation_audit import LearnedComputations


class Reader:
    def __init__(self, path): self.file = path.open(); self.row = None
    def peek(self):
        if self.row is None:
            line = self.file.readline()
            self.row = shlex.split(line) if line[:2] in ('L ', 'I ', 'P ', 'K ') else [v.strip('"') for v in line.split()]
        return self.row
    def pop(self, prefix=None):
        row = self.peek(); self.row = None
        if prefix is not None: require(row[:len(prefix)] == prefix, f'expected {prefix}, got {row[:6]}')
        return row


def close(a, b): return math.isclose(a, b, rel_tol=2e-5, abs_tol=2e-6)
def same_vector(actual, expected):
    require(len(actual) == len(expected) and all(close(a, b) for a, b in zip(actual, expected)), f'public feature mismatch: {actual} != {expected}')


def guidance_rate(schedule, completed):
    full, fade = schedule
    return 1. if completed < full else max(0., (full+fade-completed)/fade) if fade else 0.


def guidance_selected(seed, episode, decision, rate):
    mask = (1 << 64)-1
    x = (((seed+1)*0x9e3779b97f4a7c15) & mask) ^ (((episode+1)*0xbf58476d1ce4e5b9) & mask) ^ (((decision+1)*0x94d049bb133111eb) & mask)
    x = ((x ^ (x >> 30))*0xbf58476d1ce4e5b9) & mask
    x = ((x ^ (x >> 27))*0x94d049bb133111eb) & mask
    x ^= x >> 31
    return (x >> 11)/2**53 < rate


def teacher_advice(o):
    if o[2]: return 1, 'refresh_model'
    if not o[0]:
        if not o[12] and (o[13] or o[21] <= o[23]): return 0, 'obtain_world'
        if not o[13]: return 2, 'obtain_world'
        return 4, 'report_unavailable'
    if not o[5]: return 1, 'evaluate_goal'
    if not o[4] or o[6] or o[8]: return 4, 'report_outcome'
    if o[9]: return 3, 'execute_plan_step'
    if o[10] and not o[20]: return 1, 'extend_plan_search'
    return 4, 'report_unavailable'


def policy(path):
    w = list(map(float, path.read_text().split()))
    width, remainder = divmod(len(w)-5, 38)
    require(not remainder and width in (32, 1316, 2632) and all(map(math.isfinite, w)), 'invalid neural weights')
    bias, output, output_bias = 32*width, 33*width, 38*width
    @functools.lru_cache(None)
    def scores(obs):
        hidden = [math.tanh(w[bias+j]+sum(w[j*32+i]*obs[i] for i in range(32))) for j in range(width)]
        return [w[output_bias+a]+sum(w[output+a*width+j]*hidden[j] for j in range(width)) for a in range(5)]
    return w, scores


def audit(directory, mode):
    meta = json.loads((directory/'oracle.json').read_text()); cases = {r['id']: r for r in meta['cases']}
    resolved_path = directory/f'{mode}.config.json'
    resolved = json.loads(resolved_path.read_text()) if resolved_path.exists() else {}
    record_observations = resolved.get('observation_records') == 'sorted_with_multiplicity'
    def public_scene(world):
        facts = sorted((p, tuple(vs)) for p, vs in world['facts']) if record_observations else canonical(world)[1]
        return world['complete'], tuple(facts)
    primitive_weight_count = resolved.get('skill_allocated_route_weights', 15360)
    require(primitive_weight_count in (15360, 50000, 100000), 'unknown primitive capacity')
    teachers = {r['id']: r for r in meta['teaching']}
    out = Reader(directory/f'{mode}.txt'); checks = collections.Counter(); metrics = collections.defaultdict(collections.Counter)
    concepts, language, transitions, executions = {}, {}, {}, {}
    primitives = {}
    version = train = updates = 0; dirty = False; model_exists = False; stage = None; checkpoints = {}; guidance = None
    initial, _ = policy(directory/f'{mode}.initial.weights'); trained, scores = None, None
    freeze_requested = mode.endswith('frozen') or resolved.get('freeze_control', False)
    shuffle_requested = mode.endswith('shuffled') or resolved.get('shuffled_credit', False)
    ablation_after = resolved.get('control_ablation_after', 0)
    require(isinstance(ablation_after, int) and ablation_after >= 0, 'invalid control ablation boundary')
    frozen = freeze_requested and train >= ablation_after
    shuffled = shuffle_requested and train >= ablation_after
    freeze_reference = None
    if freeze_requested and ablation_after:
        freeze_path = directory/f'{mode}.control_freeze.weights'
        if freeze_path.exists(): freeze_reference, _ = policy(freeze_path)
    terminal_only = mode.endswith('terminal') or resolved.get('terminal_only_credit', False)
    model_based = mode.startswith('model') and not frozen
    memory_policy = mode == 'memory' or (mode.startswith('model') and mode != 'model_neural')
    reactive = False; has_profile = False
    if out.peek()[0] == 'META':
        metadata = out.pop(['META', 'reactive']); reactive = bool(int(metadata[2])); require(metadata[3] == '32', 'unknown motor capacity'); has_profile = True
    require(not meta.get('reactive', False) or has_profile, 'missing reactive protocol metadata')
    computational = meta.get('computational', False)
    learned_models = None; current_model = None
    if (directory/f'{mode}.models.tsv').exists():
        require(computational, 'learned computation readout requires computational curriculum')
        shape = [resolved.get('skill_units', 512), resolved.get('skill_active', 16), resolved.get('skill_channels', 3)]
        learned_models = LearnedComputations(directory, mode, meta['seed'], shape)
    if computational: out.pop(['META', 'computational'])
    count_acquired = True; acquisition_rows = iter(())
    if 'composed_budget' in meta:
        out.pop(['META', 'composed', str(meta['composed_budget'])])
        if meta['composed_budget']: version = 1; dirty = True
        count_acquired = False
        with (directory/f'{mode}.acquisition.tsv').open() as f:
            acquisition_rows = iter(list(csv.DictReader(f, delimiter='\t')))
    quantitative = mode.startswith('scaled')
    if quantitative: out.pop(['META', 'quantitative'])
    tree_policy = mode.endswith('_tree'); tree_scores = None; tree_checks = {}
    if tree_policy: out.pop(['META', 'preference_tree', '12', '2'])
    preserve_search = mode == 'learned_progress'
    if preserve_search: out.pop(['META', 'evaluation_search_retention'])
    representatives = set(); rows_model = {}; learned_values = {}; seen_cases = set(); training_states = set()
    def audit_values():
        if not model_based: return
        require(set(rows_model) == set(learned_values), 'missing or invented learned control states')
        for state, rows in rows_model.items():
            for a, (count, immediate, following) in enumerate(rows):
                q = learned_values[state][a]
                if not count: require(q == 0, 'invented value without experience'); continue
                target = (immediate+.99*sum(n*max(learned_values[s]) for s, n in following.items()))/count
                require(abs(q-target) < .001, f'learned consequence value violates observed Bellman equation: {q} {target}')
                checks['control_value_equations'] += 1

    def physics(command, world):
        e = meta['effects'][command['schema']]
        return effect((e['predicate'], e['subject'], e['object'], e['kind']), command, world)
    def truth(c, world):
        if computational: return computational_oracle.truth(meta, c, world)
        raw = dict(world, args=c['truth']['args']); base = evaluate(intended(meta, c['family']), raw)
        return -1 if base < 0 else base ^ c['variant']
    def believes(c, world):
        # The controller canonicalizes observed scenes on begin/read/actions.
        # Raw teaching intentionally retains duplicates; those are a different
        # boundary and must not be silently normalized in the VM audit itself.
        visible = dict(world, facts=[[p, list(vs)] for p, vs in public_scene(world)[1]])
        return learned_models.answer(current_model, c['text'], visible) if learned_models else truth(c, world)

    # Check that neither public nor private files diverge from the independent
    # dataset metadata. The private files are evaluator inputs, never policy inputs.
    for line in (directory/'worlds.txt').read_text().splitlines():
        r = line.split(); c = cases[r[1]]
        if r[0] == 'X':
            require(list(map(int, r[2:])) == [c['failures'], int(c['sensor_unavailable'])], 'private interruption mismatch'); continue
        require(r[0] == 'W', 'unknown private world row')
        world, at = parse_world(r, 2); present, revision = map(int, r[at:at+2]); memory, at = parse_world(r, at+2)
        require(at == len(r) and canonical(world) == canonical(c['truth']) and world['args'] == names(c['truth']), 'private world mismatch')
        require(present == (c['memory'] is not None) and revision == c['memory_revision'], 'archive metadata mismatch')
        if present: require(canonical(memory) == canonical(c['memory']), 'archive world mismatch')
    private_ids = set()
    for line in (directory/'environment.txt').read_text().splitlines():
        r = line.split()
        if r[0] == 'E':
            e = meta['effects'][r[1]]; require(r[2:] == list(map(str, [e['predicate'], e['subject'], e['object'], e['kind']])), 'private effect mismatch')
        elif computational and r[0] == 'H':
            require(parse_clause(r[2:]) == (meta['definitions'][r[1]][0], [(p, tuple(v)) for p, v in meta['definitions'][r[1]][1]]), 'private derived definition mismatch')
        elif computational and r[0] == 'Q':
            c = cases[r[1]]; private_ids.add(c['id'])
            require(r[2:] == list(map(str, [c['kind'], *c['truth']['args'], c['r'], c['s']])), 'private computation mismatch')
        else:
            c = cases[r[1]]; n = int(r[2]); private_ids.add(c['id'])
            require(r[3:3+n] == c['truth']['args'] and int(r[3+n]) == c['variant'] and parse_clause(r[4+n:]) == intended(meta, c['family']), 'private meaning mismatch')
    require(private_ids == set(cases), 'missing private goals')

    def audit_case(cmd, interruptions=(), metric_stage=None):
        nonlocal version, train, updates, dirty, model_exists, count_acquired, current_model
        op = cmd[0]; metric_stage = stage if metric_stage is None else metric_stage
        interrupt_at = child_correct = child_total = performed = 0
        cid = cmd[1]; c = cases[cid]; training = op == 'TRAIN'; require(training == c['training'] == (stage is None), 'wrong split')
        require(cid not in seen_cases, 'repeated episode identity'); seen_cases.add(cid)
        known, at = parse_world(cmd, 5); text = cmd[at]; n = int(cmd[at+1]); at += 2; available = []
        for _ in range(n): a, at = action(cmd, at); available.append(a)
        require(at == len(cmd) and list(map(int, cmd[2:5])) == [int(c['effect']), c['price'], c['disabled']] and text == c['text'] and known['args'] == names(c['initial']) and public_scene(known) == public_scene(c['initial']) and available == c['available'], 'public trial differs from metadata')
        actual = c['truth']; revision = known_revision = 1; start_version = version
        cached = acted = read_failed = ask_failed = rejected = False; answer = -1; budget = 4
        plan_status = ''; plan_actions = []; plan_reused = False; decisions = motors = predictions = 0; total_cost = 0.; finished = False
        own_actions, own_worlds, own_executed, action_before = [], [], [], None
        experience = []; motor_failures = 0
        primitive_instructions = continued_reason = explored_after_budget = plan_ready_reads = 0
        max_reason_budget = 0
        operation_counts = collections.Counter(); duplicate_reads = 0
        prices = [.07, .005, .01, .01, 0] if c['price'] else [.01, .005, .06, .01, 0]
        hint_rate = guidance_rate(guidance, train) if training and guidance else 0.
        if training and guidance:
            phase = 'bootstrap' if train < guidance[0] else 'taper' if hint_rate else 'autonomous'
            g = out.pop(['G', cid, phase]); require(len(g) == 4 and close(float(g[3]), hint_rate), 'wrong assistance phase/rate')
            checks['guidance_episodes_'+phase] += 1
        def observe():
            o = [0.]*32
            o[:20] = [int(known['complete']), int(bool(known['facts'])), int(dirty), int(model_exists and not dirty), int(c['effect']), int(cached), int(cached and answer == 1), int(cached and answer == 0), int(cached and answer == -1), int(bool(plan_actions)), int(plan_status == 'search_budget'), int(plan_status in ('state_budget', 'no_plan', 'unavailable_meaning_or_world')), int(read_failed), int(ask_failed), int(acted), int(bool(available)), int(rejected), int(plan_reused), min(4, len(plan_actions))/4, math.log2(budget)/15]
            o[20] = int(reactive and motors >= 32)
            o[21:26] = [20*v for v in prices]; o[26:29] = [.0002, .4, .1]; o[31] = 1
            if quantitative:
                o[29] = math.log2(1+len(available))/math.log2(129)
                o[30] = math.log2(1+len(known['facts']))/math.log2(257)
            return o
        while out.peek()[:2] in (['S', cid], ['H', cid], ['SUSPEND', cid]):
            if out.peek()[:2] == ['SUSPEND', cid]:
                require(not training and interrupt_at < len(interruptions), 'unexpected suspension')
                after, child = interruptions[interrupt_at]
                require(decisions == after and not finished, 'wrong interruption point')
                out.pop(['SUSPEND', cid, child[1], str(after)])
                child_correct += audit_case(child, metric_stage=metric_stage+'/children')
                child_total += 1; performed += 1; interrupt_at += 1
                resumed = out.pop(['RESUME', cid, child[1]])
                visible, cursor = parse_world(resumed, 4)
                require(cursor == len(resumed) and int(resumed[3]) == known_revision and public_scene(visible) == public_scene(known), 'resume supplied hidden or changed world information')
                cached = acted = read_failed = ask_failed = rejected = False
                answer = -1; budget = 4; plan_status = ''; plan_actions = []; plan_reused = False
                checks['goal_resumptions'] += 1
                continue
            require(interrupt_at == len(interruptions) or decisions != interruptions[interrupt_at][0], 'scheduled suspension omitted')
            advice, subgoal = teacher_advice(observe())
            expected_hint = guidance_selected(meta['seed'], train, decisions, hint_rate) and advice != c['disabled']
            hinted = out.peek()[:2] == ['H', cid]
            require(hinted == expected_hint, 'missing, invented or evaluation teacher intervention')
            if hinted:
                out.pop(['H', cid, str(decisions), str(advice), 'public_control_v1', subgoal])
                checks['teacher_interventions'] += 1
            r = out.pop(['S', cid, str(decisions)]); token, chosen = map(int, r[3:5]); cost = float(r[5]); work, rebuilt, reused, was_rejected, reply = map(int, r[6:11]); status = r[11]
            require(not hinted or chosen == advice, 'controller did not execute declared hint')
            primitive_work = int(r[12]) if computational else 0
            require(len(r) == (13 if computational else 12) and primitive_work >= 0 and (chosen == 1 or primitive_work == 0), 'invalid primitive inference work')
            require(not finished and 0 <= chosen < 5 and chosen != c['disabled'] and token > 0, 'invalid choice')
            obs = list(map(float, out.pop(['O', cid, str(decisions)])[3:])); same_vector(obs, observe()); checks['public_features'] += 1
            experience.append((tuple(obs), chosen, cost))
            if not training:
                using_memory = memory_policy and tuple(obs) in learned_values
                values = tree_scores(tuple(obs)) if tree_policy else learned_values[tuple(obs)] if using_memory else scores(tuple(obs))
                best = max((a for a in range(5) if a != c['disabled']), key=lambda a: values[a])
                require(values[chosen] >= values[best]-1e-5, 'test operation differs from frozen learned selector'); checks['tree_choices' if tree_policy else 'memory_choices' if using_memory else 'neural_choices'] += 1
            old_known, old_revision = known, known_revision; old_dirty = dirty; old_budget = budget
            old_plan = (plan_status, plan_reused, plan_actions); rejected = False; emitted = []
            primitive_instructions += primitive_work
            continued_reason += chosen == 1 and old_plan[0] == 'search_budget'
            explored_after_budget += chosen == 3 and old_plan[0] == 'search_budget' and not old_plan[2]
            plan_ready_reads += chosen in (0, 2) and bool(old_plan[2])
            if chosen == 1: max_reason_budget = max(max_reason_budget, old_budget)
            if chosen == 1:
                require(rebuilt == dirty and not was_rejected, 'reason/rebuild mismatch'); dirty = False; model_exists = True; cached = True
                if learned_models and rebuilt: current_model = learned_models.rebuilt(cid, decisions, version)
                answer = believes(c, known)
                if not learned_models and rebuilt and not count_acquired and meta.get('composed_budget', 0) > 0:
                    acquisition = next(acquisition_rows)
                    require(training and int(acquisition['total']) == len(language), 'acquisition attempt does not match current training evidence')
                    count_acquired = bool(int(acquisition['acquired']))
                # The explicit no-acquisition control has no count policy.
                # Require withholding for count-dependent requests rather
                # than assuming that any rebuild learned their meaning.
                if not learned_models and not count_acquired and c['kind'] != 1: answer = -1
                require(work <= budget, 'reason prediction budget escaped')
            elif chosen in (0, 2):
                u = out.pop(['U', cid, str(decisions)]); present, stamp = map(int, u[3:5]); seen, end = parse_world(u, 5)
                expected = c['memory'] if chosen == 0 else actual; expected_stamp = c['memory_revision'] if chosen == 0 else revision
                if chosen == 2 and c.get('sensor_unavailable'): expected = None
                require(end == len(u) and present == (expected is not None) and stamp == (expected_stamp if present else 0), 'read identity mismatch')
                if present: require(public_scene(seen) == public_scene(expected), 'read world differs from source')
                accepted = present and stamp >= known_revision
                if accepted:
                    if stamp == known_revision and public_scene(seen) == public_scene(known): duplicate_reads += 1
                    known, known_revision = seen, stamp
                elif present: checks['stale_reads_rejected'] += 1
                if chosen == 0: read_failed = not accepted
                else: ask_failed = not accepted
                require(status == 'awaiting_read', 'invalid read status')
            elif chosen == 3:
                if was_rejected:
                    require((acted and not reactive) or motors >= 32 or dirty or not known['complete'] or not available, 'unexplained action rejection'); rejected = True
                else:
                    require((not acted or reactive) and motors < 32 and not dirty and known['complete'] and available, 'invalid actuation')
                    if not acted: action_before = known
                    while out.peek()[:2] == ['A', cid]:
                        arow = out.pop(['A', cid, str(decisions), str(len(emitted))]); command, cursor = action(arow, 4); after, cursor = parse_world(arow, cursor)
                        executed = bool(int(arow[cursor])) if cursor < len(arow) else True
                        require(cursor+int(cursor < len(arow)) == len(arow) and executed == (len(own_actions) >= c.get('failures', 0)), 'incorrect failure schedule')
                        require(command in available and canonical(after) == canonical(physics(command, actual) if executed else actual), 'invalid physical transition')
                        emitted.append(command); own_actions.append(command); own_worlds.append(after); own_executed.append(executed); motor_failures += not executed; actual = after; revision += 1; checks['physical_transitions'] += 1
                    expected_plan = plan_actions[:1] if reactive else plan_actions
                    require(emitted and (emitted == expected_plan if expected_plan else len(emitted) == 1), 'actuation bypassed computed ticket/exploration')
                    if plan_actions and not reactive and all(own_executed): require(believes(c, actual) == 1, 'computed plan failed learned goal')
                    if reactive: require(len(emitted) == 1, 'reactive body emitted a hidden batch')
                    acted = True; known, known_revision = actual, revision
                    require(status == 'awaiting_actions', 'invalid motor status')
            else:
                finished = True; require(reply == (answer if cached else -1) and status == 'finished', 'finish computed an unearned answer'); answer = reply
            if chosen != 1: require(work == 0 and rebuilt == 0, 'uncharged hidden reasoning')
            if chosen != 4: require(reply == -1, 'nonterminal answer exposed')
            if public_scene(old_known) != public_scene(known) or old_revision != known_revision:
                same_scene = public_scene(old_known) == public_scene(known)
                retain = preserve_search and not training and same_scene
                cached = False; answer = -1; plan_status = ''; plan_actions = []; plan_reused = False
                if retain and budget > 4: checks['retained_search_allowance'] += 1
                if not retain: budget = 4
            nrow = out.pop(['N', cid, str(decisions), str(known_revision)]); nworld, cursor = parse_world(nrow, 4); post = list(map(float, nrow[cursor:])); require(public_scene(nworld) == public_scene(known), 'visible world differs from accepted observation')
            d = out.pop(['D', cid, str(decisions)]); dstatus, dreused, count = d[3], bool(int(d[4])), int(d[5]); cursor = 6; dactions = []
            for _ in range(count): a, cursor = action(d, cursor); dactions.append(a)
            require(cursor == len(d), 'extra plan data')
            if chosen == 1:
                if c['effect'] and answer == 0 and known['complete'] and (not acted or reactive):
                    require(status == dstatus and dstatus in ('procedure', 'searched', 'search_budget', 'state_budget', 'no_plan'), 'invalid reason result')
                    if dactions:
                        predicted = known
                        for a in dactions: require(a in available, 'plan uses unavailable affordance'); predicted = physics(a, predicted)
                        require(believes(c, predicted) == 1 and work >= len(dactions) and dstatus in ('procedure', 'searched'), 'invalid computed goal plan'); checks['computed_plans'] += 1
                    if dstatus == 'search_budget': require(work == old_budget and not dactions, 'wrong exhausted budget'); budget = min(32768, budget*2)
                    require(dreused == bool(reused), 'reuse report mismatch')
                else: require(work == 0 and (dstatus, dreused, dactions) == old_plan, 'unrequested search')
                plan_status, plan_reused, plan_actions = dstatus, dreused, dactions
            else: require((dstatus, dreused, dactions) == (plan_status, plan_reused, plan_actions), 'plan changed without paid reason/world change')
            motors += len(emitted)
            same_vector(post, observe()); checks['public_features'] += 1
            expected_cost = prices[chosen]+.005*rebuilt+.00001*(work+primitive_work)+.02*len(emitted)
            require(close(cost, expected_cost), 'wrong charged operation cost')
            total_cost += cost; decisions += 1; predictions += work; operation_counts[chosen] += 1
        for after, child in interruptions[interrupt_at:]:
            out.pop(['SKIP', cid, child[1]])
            require(child[1] not in seen_cases, 'duplicate skipped child'); seen_cases.add(child[1])
            metrics[metric_stage+'/children'].update(total=1, skipped=1)
            checks['skipped_interrupted_goals'] += 1; child_total += 1
        timeout = not finished; require(decisions <= (48 if training else 128) and (not timeout or decisions == (48 if training else 128)), 'wrong execution bound')
        final_reply = -1 if timeout else answer; satisfaction = truth(c, actual); original = truth(c, c['truth'])
        correct = not timeout and ((final_reply == 1 and satisfaction == 1) if c['effect'] else (final_reply == original and canonical(actual) == canonical(c['truth'])))
        if training:
            j = out.pop(['J', cid]); evidence, reward, execution = j[2], float(j[3]), bool(int(j[4])); require(execution == acted and reward in (-1, 1), 'invalid neural receipt')
            if not shuffled: require(reward == (1 if correct else -1), 'incorrect neural terminal credit')
            if acted:
                require(evidence.startswith('execution') and evidence not in executions, 'invalid execution receipt')
                executions[evidence] = {'text': c['text'], 'before': action_before, 'actions': own_actions, 'observed': own_worlds, 'executed': own_executed, 'success': bool(satisfaction)}
                before = action_before
                for i, (a, after) in enumerate(zip(own_actions, own_worlds)):
                    transitions[f'{evidence}_step{i}'] = {'action': a, 'before': before, 'after': after, 'executed': own_executed[i]}; before = after
            require(bool(evidence) == bool(known['complete']), 'fabricated/missing terminal language evidence')
            if evidence:
                require(evidence not in language, 'duplicate terminal language receipt'); language[evidence] = {'text': c['text'], 'world': known, 'gold': satisfaction, 'family': c['family'], 'variant': c['variant']}
                version += 1; dirty = True
            train += 1; updates += 0 if frozen else 128+int(model_based)
            training_states.update(o for o, a, cost in experience)
            representatives.update(o for o, a, cost in (experience[-1:] if terminal_only else experience))
            if model_based:
                for i, (state, a, cost) in enumerate(experience):
                    if terminal_only and i+1 != len(experience): continue
                    terminal = i+1 == len(experience)
                    def ensure(s): return rows_model.setdefault(s, [[0, 0., collections.Counter()] for _ in range(5)])
                    row = ensure(state)[a]; row[0] += 1; row[1] += (reward if terminal else 0)-cost
                    if not terminal:
                        following = experience[i+1][0]; ensure(following); row[2][following] += 1
        f = out.pop(['F', cid]); require(list(map(int, f[2:8])) == [int(correct), satisfaction, final_reply, int(timeout), decisions, motors] and close(float(f[8]), total_cost), 'wrong episode score/cost')
        require(list(map(int, f[9:])) == [start_version, version, updates], 'wrong evidence/neural update count')
        if interruptions: out.pop(['NESTED', cid, str(child_correct), str(child_total), str(performed)])
        if training and has_profile and train % 200 == 0 and not frozen and mode != 'direct':
            count = 64*len(representatives); updates += count; out.pop(['CONSOLIDATE', str(train), str(count), str(updates)])
        groups = ['train'] if training else [metric_stage, f'{metric_stage}/family{c["family"]}', f'{metric_stage}/price{c["price"]}', f'{metric_stage}/intent{int(c["effect"])}', f'{metric_stage}/sensor{int(bool(c.get("sensor_unavailable")))}']
        if c['disabled'] >= 0: groups.append(f'removal{c["disabled"]}')
        for group in groups:
            m = metrics[group]; m.update(total=1, correct=int(correct), timeouts=int(timeout), decisions=decisions, motors=motors, predictions=predictions, duplicate_reads=duplicate_reads, cost=total_cost)
            m.update(motor_failures=motor_failures, recovered=int(correct and motor_failures > 0), unavailable_sensor=int(bool(c.get('sensor_unavailable'))))
            m.update(primitive_instructions=primitive_instructions, continued_reason=continued_reason,
                     explored_after_budget=explored_after_budget, plan_ready_reads=plan_ready_reads)
            m['max_reason_budget'] = max(m['max_reason_budget'], max_reason_budget)
            m.update({f'op{k}': v for k, v in operation_counts.items()}); m['max_decisions'] = max(m['max_decisions'], decisions); m['max_motors'] = max(m['max_motors'], motors)
            if correct: m['max_success_decisions'] = max(m['max_success_decisions'], decisions); m['max_success_motors'] = max(m['max_success_motors'], motors)
        checks['episodes'] += 1
        return int(correct)

    for line in (directory/'input.txt').read_text().splitlines():
        cmd = shlex.split(line); op = cmd[0]
        frozen = freeze_requested and train >= ablation_after
        shuffled = shuffle_requested and train >= ablation_after
        if op == 'GUIDANCE':
            require(guidance is None and stage is None and train == 0 and len(cmd) == 4 and cmd[1] == 'public_control_v1', 'invalid guidance declaration')
            guidance = tuple(map(int, cmd[2:]))
            require(min(guidance) >= 0 and 0 < sum(guidance) <= 8192, 'invalid guidance duration')
            out.pop(cmd+[str(meta['seed'])])
        elif op in ('C', 'L', 'T', 'P', 'M', 'N'):
            require(stage is None, 'teaching during frozen evaluation'); rid = cmd[1]; r = teachers[rid].copy()
            if op == 'C':
                world, at = parse_world(cmd, 4)
                name = r['name'] if computational else meta['heads'][r['family']]
                expected = evaluate(meta['definitions'][name], r['world']) if computational else oracle(meta, r['family'], r['world'])
                require(at == len(cmd) and cmd[2] == name and int(cmd[3]) == r['gold'] == expected and world['args'] == r['world']['args'] and canonical(world) == canonical(r['world']), 'bad concept teaching')
                r['name'] = cmd[2]; concepts[rid] = r
            elif op in ('L', 'M', 'N'):
                world, at = parse_world(cmd, 3)
                expected = truth(r, r['world']) if computational else (oracle(meta, r['family'], r['world']) ^ r['variant'])
                tail = ['2', r['r'], r['s']] if op in ('M', 'N') else []
                if op == 'N': tail += [str(len(r['quantities'])), *r['quantities']]
                require(cmd[at+1:] == tail and cmd[at] == r['text'] and world['args'] == names(r['world']) and canonical(world) == canonical(r['world']) and int(cmd[2]) == r['gold'] == expected, 'bad language teaching')
                language[rid] = r
            elif op == 'P':
                require(computational and line == r['record'], 'primitive input differs from metadata')
                task, relation, st, sv, tt, tv, atype, answer, n = map(int, cmd[2:11])
                flat = list(map(int, cmd[11:])); require(len(flat) == 3*n, 'invalid primitive fact count')
                facts = list(zip(flat[::3], flat[1::3], flat[2::3]))
                require(primitive_gold(task, (st, sv), (tt, tv), relation, facts) == (atype, answer), 'incorrect primitive outcome')
                primitives[rid] = r
                if task == 2: count_acquired = True
            else:
                a, at = action(cmd, 2); before, at = parse_world(cmd, at); after, at = parse_world(cmd, at)
                require(at == len(cmd) and a == r['action'] and canonical(before) == canonical(r['before']) and canonical(after) == canonical(r['after']) == canonical(physics(a, before)), 'bad transition teaching')
                transitions[rid] = r
            version += 1; dirty = True; out.pop(['T', op, rid, str(version)]); checks['external_receipts'] += 1
        elif op == 'CHECK':
            require(stage is None, 'nested evaluation'); stage = cmd[1]; saved_dirty, saved_exists = dirty, model_exists
            saved_model = current_model
            trained, scores = policy(directory/f'{mode}.{stage}.weights')
            if frozen:
                reference = freeze_reference if ablation_after else initial
                require(reference is not None and trained == reference, 'frozen weights changed or phase boundary missing')
                if ablation_after and train == ablation_after:
                    checks['matched_freeze_boundary_checkpoints'] += 1
                    if stage == 'stage4': checks['matched_stage4_freeze'] += 1
            else: require(trained != initial, 'learning weights did not change')
            learned_values = {}
            values_path = directory/f'{mode}.{stage}.values'
            if values_path.exists():
                for raw in values_path.read_text().splitlines():
                    v = list(map(float, raw.split())); require(len(v) == 37 and all(map(math.isfinite, v)), 'invalid control values')
                    state = tuple(v[:32]); require(state in training_states, 'control table contains an unobserved state'); learned_values[state] = v[32:]
            if frozen and ablation_after:
                require(values_path.read_bytes() == (directory/f'{mode}.control_freeze.values').read_bytes(),
                        'empirical control values changed after matched freeze')
                checks['frozen_empirical_checkpoints'] += 1
            audit_values()
            if tree_policy:
                tree_scores, tree_checks[stage] = tree_readout(directory/f'{mode}.{stage}.tree', learned_values)
            out.pop(['CHECK', stage, str(train), str(version), str(updates)])
        elif op == 'ENDCHECK':
            m = metrics[stage]; out.pop(['CHECK_RESULT', stage, str(m['correct']), str(m['total'])])
            if metrics.get(stage+'/children', {}).get('total'):
                nested = metrics[stage+'/children']; out.pop(['CHECK_NESTED', stage, str(nested['correct']), str(nested['total'])])
            rows = [out.pop(['R'])]
            while rows[-1][0] != 'END': rows.append(out.pop())
            snap, at = parse_snapshot(rows, 0); require(at == len(rows) and snap['stage'] == stage and snap['version'] == version, 'bad checkpoint snapshot')
            if computational: require(snap['primitives'] and snap['primitives'][0] == len(primitives), 'primitive evidence count differs')
            require(snap['counts'] == [len(concepts), len(language), len(transitions), len(executions), len(snap['procedures'])], 'frozen evidence counts differ')
            require(snap['ready'] == (model_exists and not dirty), 'snapshot readiness mismatch')
            if snap['ready']: checks.update(audit_snapshot(snap, concepts, language, transitions, executions, meta, 1024))
            checkpoints[stage] = {'counts': snap['counts'], 'ready': snap['ready'], 'version': version}
            if computational:
                def primitive_weights(suffix):
                    p = directory/f'{mode}.{stage}.{suffix}'
                    if not p.exists(): return None  # archived pilot predates post-probe export
                    w = list(map(float, p.read_text().split()))
                    require(len(w) == primitive_weight_count and all(map(math.isfinite, w)), 'invalid primitive neural checkpoint')
                    return w
                before, after = primitive_weights('primitive_weights'), primitive_weights('probe_primitive_weights')
                require(before is not None, 'missing primitive checkpoint')
                checkpoints[stage]['primitives'] = snap['primitives']
                checkpoints[stage]['primitive_changed_in_probe'] = before != after if after is not None else None
                if after is not None and not frozen: require(before == after, 'positive learning arm acquired primitives during evaluation')
            dirty, model_exists = saved_dirty, saved_exists; current_model = saved_model; stage = None
        elif op in ('TRAIN', 'TEST'):
            audit_case(cmd)
        elif op == 'INTERLEAVE_TEST':
            require(stage is not None, 'interleaved test outside evaluation')
            def trial_end(at):
                _, end = parse_world(cmd, at+4)
                count = int(cmd[end+1]); end += 2
                for _ in range(count): _, end = action(cmd, end)
                return end
            end = trial_end(1); parent = ['TEST']+cmd[1:end]
            count = int(cmd[end]); cursor = end+1; interruptions = []; previous = 0; ids = {parent[1]}
            require(0 < count <= 8, 'invalid interruption count')
            for _ in range(count):
                after = int(cmd[cursor]); cursor += 1; end = trial_end(cursor)
                child = ['TEST']+cmd[cursor:end]; cursor = end
                require(previous < after < 128 and child[1] not in ids, 'invalid interruption schedule')
                previous = after; ids.add(child[1]); interruptions.append((after, child))
            require(cursor == len(cmd), 'extra interleaved input')
            audit_case(parent, interruptions)
        else: raise AssertionError(f'unknown curriculum command: {op}')
    require(not out.peek() and stage is None and seen_cases == set(cases), 'unconsumed native output or omitted cases')
    if learned_models:
        learned_models.finish()
        checks['reconstructed_models'] = len(learned_models.used)
        checks['frozen_primitive_readouts'] = len(learned_models.weights)
    final, _ = policy(directory/f'{mode}.final.weights'); require(final == trained, 'evaluation changed final weights')
    if freeze_requested and train >= ablation_after:
        reference = freeze_reference if ablation_after else initial
        require(reference is not None and final == reference, 'frozen neural learner updated')
        if ablation_after:
            require(final != initial and updates >= 128*ablation_after, 'missing pre-freeze learning')
        else: require(updates == 0, 'frozen neural learner updated')
    else: require(final != initial and updates >= 128*train, 'missing neural learning')
    return {'metrics': dict(metrics), 'checks': dict(checks), 'checkpoints': checkpoints,
            'changed_parameters': sum(a != b for a, b in zip(initial, final)), 'updates': updates, 'preference_trees': tree_checks}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root', default='logs/relational_control_empirical'); parser.add_argument('--seeds', type=int, default=3); parser.add_argument('--modes', nargs='+', default=['learned', 'direct', 'frozen', 'shuffled', 'terminal']); args = parser.parse_args()
    root = Path(args.root); runs = {}
    for seed in range(args.seeds):
        for mode in args.modes:
            result = audit(root/str(seed), mode); runs[f'{seed}/{mode}'] = result
            stages = json.loads((root/str(seed)/'oracle.json').read_text()).get('stages', 8); m = result['metrics'][f'stage{stages}']
            print(f'audited seed={seed} mode={mode} final={m["correct"]}/{m["total"]}', flush=True)
    paths = [Path(p) for p in ['cpp/include/laminar/relational_control.h', 'cpp/include/laminar/relational_session.h', 'cpp/include/laminar/backprop_choice.h', 'cpp/include/laminar/concept_memory.h', 'cpp/apps/relational_control_probe.cpp', 'cpp/apps/control_probe_support.h', 'cpp/tests/test_relational_control.cpp', 'cpp/tests/test_relational_session.cpp', 'python/relational_control_probe.py', 'python/check_relational_control.py', 'python/check_relational_session.py', 'scripts/relational_control.sh', 'scripts/relational_reactive.sh']]
    paths += [Path(p) for p in ['cpp/apps/primitive_discovery_support.h', 'cpp/tests/test_computational_session.cpp', 'python/computational_control_probe.py', 'python/computational_control_oracle.py', 'python/check_primitive_discovery.py', 'scripts/computational_control.sh', 'scripts/computational_control_scale.sh']]
    paths += list(Path('cpp/include/laminar').glob('*.h'))
    paths += [Path(p) for p in ['python/check_preference_tree.py', 'python/check_computational_readout.py', 'python/diagnose_control_preferences.py', 'scripts/computational_readout.sh', 'scripts/computational_progress.sh', 'cpp/tests/test_preference_tree.cpp']]
    paths += [p for p in root.rglob('*') if p.suffix in ('.txt', '.weights', '.values', '.tree', '.json', '.primitive_weights', '.probe_primitive_weights') and p.name != 'audit.json']
    manifests = {}
    for manifest in root.glob('*source_manifest.json'):
        saved = json.loads(manifest.read_text())
        for name, expected in saved.items():
            p = Path(name); require(hashlib.sha256(p.read_bytes()).hexdigest() == expected, 'archived source changed'); paths.append(p)
        manifests[str(manifest)] = saved
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (root/'audit.json').write_text(json.dumps({'runs': runs, 'sha256': hashes, 'source_manifests': manifests}, indent=2)+'\n')
    print('independent public-state, neural policy, world, receipt and outcome audit passed')
