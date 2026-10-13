"""Regression tests, not new experiment data.

The observation pair is transcribed from the user's HPC 160656 report.
Object/action lists and task-description fields below are synthetic fixtures.
"""
import ast
import copy
import gzip
import importlib.util
import json
import os
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('reset_basis_fixture', ROOT/'experiments/gigpo_diag/diagnostics.py')
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)

OBS = ('This room is called the workshop. In it, you see: \n'
       '\tthe agent\n\ta substance called air\n'
       '\tan inclined plane with a tin surface\n'
       '\tan inclined plane with a caesium surface\n'
       '\ta steel block\n\ta stopwatch, which is deactivated. \n'
       '\ta table. On the table is: a battery, a green wire, a orange light bulb, which is off, '
       'a orange wire, a red light bulb, which is off, a switch, which is off, '
       'a yellow light bulb, which is off, a yellow wire.\n'
       '\ta ultra low temperature freezer. The ultra low temperature freezer door is closed. \n'
       'You also see:\n\tA door to the hallway (that is open)')
OTHER = OBS.replace('\tan inclined plane with a tin surface\n\tan inclined plane with a caesium surface',
                    '\tan inclined plane with a caesium surface\n\tan inclined plane with a tin surface')


def fixture():
    observations = [OTHER if slot in (84, 86) else OBS for slot in range(80, 88)]
    infos = []
    for observation in observations:
        components = dict(observation=observation, objects=['agent', 'inclined plane'],
                          possible_actions=['look around'], score=10)
        infos.append(dict(task_id='inclined-plane-friction-named-surfaces', task_num=12,
                          variation_id=469, reset_seed=374217481, score=10,
                          task_description='synthetic task description', simplifications_requested='easy',
                          initial_state_components=components, initial_state_hash=diag.digest(components)))
    return {'anchor': observations, 'text': observations.copy()}, infos, ['10'] * 8


def records(path, kind):
    rows = []
    for file in path.glob(kind + '.pid-*.jsonl.gz'):
        with gzip.open(file, 'rt', encoding='utf-8') as stream:
            rows.extend(json.loads(line) for line in stream)
    return rows


def test_reported_reordering_passes_and_retains_exact_native_anchor_groups(tmp_path, monkeypatch):
    monkeypatch.setenv('GIGPO_DIAG_DIR', str(tmp_path))
    obs, infos, uids = fixture()
    before = copy.deepcopy((obs, infos, uids))
    diag.verify_groups(obs, infos, uids, environment='scienceworld')
    assert (obs, infos, uids) == before
    evidence = records(tmp_path, 'group_reset_text_differences')[0]
    assert evidence['observation_difference'] == 'line_order_only'
    assert evidence['full_hidden_state_verified'] is False
    assert evidence['anchor_variant_count'] == 2
    assert evidence['workers'][4]['guard_state']['anchor'] == OTHER
    # Execute the unchanged native clustering function, without GPU dependencies.
    tree = ast.parse((ROOT/'gigpo/core_gigpo.py').read_text(encoding='utf-8'))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('to_hashable', 'build_step_group')]
    from collections import Counter, defaultdict
    import uuid
    namespace = dict(np=np, uuid=uuid, defaultdict=defaultdict)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<native-anchor-clustering>', 'exec'), namespace)
    clusters = namespace['build_step_group'](np.array(obs['anchor']), np.array(uids))
    assert sorted(Counter(clusters).values()) == [2, 6]


@pytest.mark.parametrize('field,value', [
    ('task_id', 'different-task'), ('task_num', 99), ('variation_id', 470),
    ('reset_seed', 1), ('simplifications_requested', ''), ('task_description', 'different goal'),
    ('score', 20), ('objects', ['different object']), ('possible_actions', ['different action'])])
def test_real_reset_evidence_mismatch_still_rejected(field, value, tmp_path, monkeypatch):
    monkeypatch.setenv('GIGPO_DIAG_DIR', str(tmp_path))
    obs, infos, uids = fixture()
    if field in ('objects', 'possible_actions'):
        infos[4]['initial_state_components'][field] = value
    else:
        infos[4][field] = value
        if field == 'score':
            infos[4]['initial_state_components']['score'] = value
    infos[4]['initial_state_hash'] = diag.digest(infos[4]['initial_state_components'])
    with pytest.raises(RuntimeError, match='refusing advantage update'):
        diag.verify_groups(obs, infos, uids, environment='sciworld')
    assert len(records(tmp_path, 'group_guard_failures')) == 1


@pytest.mark.parametrize('problem', ['missing_seed', 'missing_components', 'missing_objects', 'bad_hash', 'bad_score'])
def test_missing_or_inconsistent_evidence_rejected_even_if_all_workers_match(problem):
    obs, infos, uids = fixture()
    for info in infos:
        if problem == 'missing_seed': info.pop('reset_seed')
        if problem == 'missing_components': info.pop('initial_state_components')
        if problem == 'missing_objects': info['initial_state_components'].pop('objects')
        if problem == 'bad_hash': info['initial_state_hash'] = 'wrong'
        if problem == 'bad_score': info['score'] = -1
    with pytest.raises(RuntimeError):
        diag.verify_groups(obs, infos, uids, environment='scienceworld')


def test_logging_disabled_or_broken_does_not_change_guard_decision(tmp_path, monkeypatch):
    obs, infos, uids = fixture()
    monkeypatch.delenv('GIGPO_DIAG_DIR', raising=False)
    diag.verify_groups(obs, infos, uids, environment='scienceworld')
    monkeypatch.setenv('GIGPO_DIAG_DIR', str(tmp_path))
    with mock.patch.object(diag, 'write', side_effect=OSError('synthetic disk error')):
        diag.verify_groups(obs, infos, uids, environment='scienceworld')
        infos[4]['variation_id'] = 999
        with pytest.raises(RuntimeError):
            diag.verify_groups(obs, infos, uids, environment='scienceworld')


def test_other_content_is_logged_as_unverified_not_certified_equivalent(tmp_path, monkeypatch):
    monkeypatch.setenv('GIGPO_DIAG_DIR', str(tmp_path))
    obs, infos, uids = fixture()
    obs['anchor'][4] = 'different rendered description'
    infos[4]['initial_state_components']['observation'] = obs['anchor'][4]
    infos[4]['initial_state_hash'] = diag.digest(infos[4]['initial_state_components'])
    diag.verify_groups(obs, infos, uids, environment='scienceworld')
    evidence = records(tmp_path, 'group_reset_text_differences')[0]
    assert evidence['observation_difference'] == 'content_difference_unverified'
    assert evidence['full_hidden_state_verified'] is False


def test_other_environments_and_cardinality_guard_unchanged():
    obs, infos, uids = fixture()
    with pytest.raises(RuntimeError):
        diag.verify_groups(obs, infos, uids)
    with pytest.raises(RuntimeError, match='cardinality'):
        diag.verify_groups(obs, infos[:-1], uids, environment='scienceworld')
