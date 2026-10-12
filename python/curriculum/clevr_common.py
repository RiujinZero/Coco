"""Shared CLEVR loading/verbalization for the curriculum adapters (manifest S3).

Renders the WHOLE scene as text (no query-specific filtering: every object and its full spatial
order is always given, never just the answer-bearing subset). Programs and answers are teacher-only
(used to classify episodes and to score the sidecar, never written into the learner-facing stream).

Scene text is deterministic given the scene, so identical scenes always verbalize identically —
episodes are grouped by image_index so no scene's sentences leak across a train/holdout split.
"""
import json


def describe_object(o):
    return f"a {o['size']} {o['color']} {o['material']} {o['shape']}"


def order_indices(rel, key):
    """Left-to-right / front-to-back total order of object indices, derived from the scene's own
    relationship lists (not from raw 3d coordinates, which do not match CLEVR's relation semantics
    directly): the object with the most things to its right/behind is leftmost/frontmost."""
    n = len(rel[key])
    return sorted(range(n), key=lambda i: -len(rel[key][i]))


def verbalize_scene(scene):
    objs = scene['objects']
    tags = [describe_object(o) for o in objs]
    sents = [f"There is {tags[i]}." for i in range(len(objs))]
    if len(objs) > 1:
        lr = order_indices(scene['relationships'], 'right')
        fb = order_indices(scene['relationships'], 'behind')
        sents.append("From left to right: " + ", ".join(tags[i] for i in lr) + ".")
        sents.append("From front to back: " + ", ".join(tags[i] for i in fb) + ".")
    return sents


def program_shape(program):
    """The program's function-name sequence, without its value_inputs: a stable skill/phrase family
    id (teacher-only) independent of which colors/shapes/sizes the question happens to name."""
    return tuple(p['function'] for p in program)


def program_depth(program):
    """Number of value-bearing (non-scene, non-unique) nodes: a rough composition depth."""
    return sum(1 for p in program if p['function'] not in ('scene', 'unique'))


def n_chains(program):
    """Number of independent 'scene' roots the program starts from (a comparison walks two chains)."""
    return sum(1 for p in program if p['function'] == 'scene')


def load_scenes(path):
    data = json.load(open(path, encoding='utf-8'))
    return {s['image_index']: s for s in data['scenes']}


def iter_questions(path):
    """Stream questions one at a time without holding the whole (very large) file's python objects."""
    data = json.load(open(path, encoding='utf-8'))
    for q in data['questions']:
        yield q
