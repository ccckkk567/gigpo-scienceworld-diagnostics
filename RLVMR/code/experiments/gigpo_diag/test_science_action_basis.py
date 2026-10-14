"""Guard regressions with synthetic command aliases; no experiment results."""
import ast
import copy
import gzip
import importlib.util
import json
from pathlib import Path
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('action_basis_fixture', ROOT/'experiments/gigpo_diag/diagnostics.py')
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)


def fixture():
    infos = []
    for alias in ('blue paint in cup', 'paint on cup'):
        records = [dict(action=f'focus on {alias}', template_id=5, obj_ids=[123], type_ids=[7]),
                   dict(action=f'pour {alias} into jug', template_id=9, obj_ids=[123, 456], type_ids=[7, 8])]
        components = dict(observation='same room', objects=['blue paint', 'cup', 'jug'],
                          possible_actions=sorted(r['action'] for r in records),
                          possible_action_templates=records, score=0)
        infos.append(dict(task_id='synthetic', task_num=1, variation_id=2, reset_seed=3,
                          simplifications_requested='easy', task_description='synthetic goal', score=0,
                          reset_evidence_version=2, initial_state_components=components,
                          initial_state_hash=diag.digest(components)))
    return dict(anchor=['same room']*2), infos, [0, 0]



@pytest.mark.parametrize('mutation', ['none', 'object_id', 'argument_order', 'missing_records', 'score'])
def test_historical_structure_is_audit_evidence_not_a_gate(mutation):
    obs, infos, uids = fixture()
    c = infos[1]['initial_state_components']
    if mutation == 'object_id': c['possible_action_templates'][0]['obj_ids'][0] = 999
    if mutation == 'argument_order': c['possible_action_templates'][1]['obj_ids'].reverse()
    if mutation == 'missing_records': c.pop('possible_action_templates')
    if mutation == 'score': c['score'] = infos[1]['score'] = 10
    infos[1]['initial_state_hash'] = diag.digest(c)
    before = copy.deepcopy((obs, infos, uids))
    row = diag.science_reset_observations(obs, infos, uids)[0]
    assert not row['full_hidden_state_verified']
    assert row['alias_equivalence'] is None
    if mutation == 'missing_records': assert 'evidence_unavailable' in row['classification']
    if mutation in ('object_id', 'argument_order'): assert row['structure_difference_observed']
    assert before == (obs, infos, uids)


def test_production_only_calls_native_string_enumerations():
    tree = ast.parse((ROOT/'agent_system/environments/env_package/sciworld/envs.py').read_text(encoding='utf-8'))
    calls = [n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    assert calls.count('get_possible_objects') == 2  # reset and step
    assert calls.count('get_valid_action_object_combinations') == 2
    assert not any('LUT' in name or 'with_templates' in name for name in calls)
