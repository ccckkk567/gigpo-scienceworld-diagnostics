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



def test_historical_line_order_remains_diagnostic_and_native_clusters_unchanged():
    obs, infos, uids = fixture()
    before = copy.deepcopy((obs, infos, uids))
    row = diag.science_reset_observations(obs, infos, uids)[0]
    assert 'known_rendering_difference' in row['classification']
    assert row['observation_variant_count'] == 2
    assert not row['full_hidden_state_verified']
    assert before == (obs, infos, uids)
    tree = ast.parse((ROOT/'gigpo/core_gigpo.py').read_text(encoding='utf-8'))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('to_hashable', 'build_step_group')]
    from collections import Counter, defaultdict
    import uuid
    namespace = dict(np=np, uuid=uuid, defaultdict=defaultdict)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<native-clustering>', 'exec'), namespace)
    clusters = namespace['build_step_group'](np.array(obs['anchor']), np.array(uids))
    assert sorted(Counter(clusters).values()) == [2, 6]


@pytest.mark.parametrize('problem', ['components', 'hash', 'version', 'score', 'auxiliary', 'description'])
def test_optional_evidence_never_has_stop_authority(problem, tmp_path, monkeypatch):
    monkeypatch.setenv('GIGPO_DIAG_DIR', str(tmp_path))
    obs, infos, uids = fixture()
    if problem == 'components': infos[1].pop('initial_state_components')
    if problem == 'hash': infos[1]['initial_state_hash'] = 'bad'
    if problem == 'version': infos[1]['reset_evidence_version'] = 999
    if problem == 'score': infos[1]['score'] = 20
    if problem == 'auxiliary': infos[1]['initial_state_components']['extra'] = 'debug'
    if problem == 'description': infos[1]['task_description'] = 'possibly changed goal'
    row = diag.verify_groups(obs, infos, uids, environment='scienceworld')[0]
    assert not row['full_hidden_state_verified']
    assert set(row['classification']) & {'evidence_unavailable', 'suspicious_semantic_difference'}
    with mock.patch.object(diag, 'write', side_effect=OSError('synthetic disk failure')):
        diag.verify_groups(obs, infos, uids, environment='scienceworld')


def test_observer_parser_failure_does_not_escape(tmp_path, monkeypatch):
    monkeypatch.setenv('GIGPO_DIAG_DIR', str(tmp_path))
    obs, infos, uids = fixture()
    with mock.patch.object(diag, 'science_reset_observations', side_effect=ValueError('synthetic parser failure')):
        assert diag.observe_science_reset(obs, infos, uids) is None
    assert diag._reset_counts['train/observer_failure_count'] == 1
