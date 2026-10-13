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


def test_command_aliases_pass_without_rewriting_model_commands_or_inputs(tmp_path, monkeypatch):
    monkeypatch.setenv('GIGPO_DIAG_DIR', str(tmp_path))
    obs, infos, uids = fixture()
    before = copy.deepcopy((obs, infos, uids))
    diag.verify_groups(obs, infos, uids, environment='scienceworld')
    assert (obs, infos, uids) == before
    files = list(tmp_path.glob('group_reset_action_text_differences*.jsonl.gz'))
    with gzip.open(files[0], 'rt') as f:
        row = json.loads(next(f))
    assert row['action_variant_count'] == 2
    assert row['full_hidden_state_verified'] is False
    assert row['workers'][1]['reset_info'] == infos[1]


@pytest.mark.parametrize('problem', ['object_id', 'argument_order', 'type_id', 'template_id', 'multiplicity',
                                   'text_mismatch', 'missing_records', 'missing_ids', 'invalid_id',
                                   'invalid_type_length', 'missing_version', 'objects', 'score'])
def test_structural_or_inconsistent_evidence_still_aborts(problem):
    obs, infos, uids = fixture()
    info = infos[1]
    c = info['initial_state_components']
    rows = c['possible_action_templates']
    if problem == 'object_id': rows[0]['obj_ids'][0] = 999
    if problem == 'argument_order': rows[1]['obj_ids'].reverse()
    if problem == 'type_id': rows[0]['type_ids'][0] = 999
    if problem == 'template_id': rows[0]['template_id'] = 999
    if problem == 'multiplicity':
        rows.append(copy.deepcopy(rows[0]))
        c['possible_actions'] = sorted(r['action'] for r in rows)
    if problem == 'text_mismatch': rows[0]['action'] = 'different recorded command'
    if problem == 'missing_records': c.pop('possible_action_templates')
    if problem == 'missing_ids': rows[0].pop('obj_ids')
    if problem == 'invalid_id': rows[0]['obj_ids'] = ['123']
    if problem == 'invalid_type_length': rows[0]['type_ids'] = []
    if problem == 'missing_version': info.pop('reset_evidence_version')
    if problem == 'objects': c['objects'] = ['different object']
    if problem == 'score': c['score'] = info['score'] = 10
    info['initial_state_hash'] = diag.digest(c)
    with pytest.raises(RuntimeError, match='refusing advantage update'):
        diag.verify_groups(obs, infos, uids, environment='scienceworld')


def test_record_order_is_irrelevant_but_object_argument_order_is_retained():
    obs, infos, uids = fixture()
    c = infos[1]['initial_state_components']
    c['possible_action_templates'].reverse()
    infos[1]['initial_state_hash'] = diag.digest(c)
    diag.verify_groups(obs, infos, uids, environment='scienceworld')


def test_no_fallback_for_missing_v2_evidence_even_when_every_worker_is_equal():
    obs, infos, uids = fixture()
    for i in infos:
        i['initial_state_components'].pop('possible_action_templates')
        i['initial_state_hash'] = diag.digest(i['initial_state_components'])
    with pytest.raises(RuntimeError):
        diag.verify_groups(obs, infos, uids, environment='scienceworld')


def test_diagnostic_failure_cannot_affect_comparison(tmp_path, monkeypatch):
    obs, infos, uids = fixture()
    monkeypatch.delenv('GIGPO_DIAG_DIR', raising=False)
    diag.verify_groups(obs, infos, uids, environment='scienceworld')
    monkeypatch.setenv('GIGPO_DIAG_DIR', str(tmp_path))
    with mock.patch.object(diag, 'write', side_effect=OSError('synthetic disk error')):
        diag.verify_groups(obs, infos, uids, environment='scienceworld')
        infos[1]['variation_id'] = 9
        with pytest.raises(RuntimeError):
            diag.verify_groups(obs, infos, uids, environment='scienceworld')


def test_worker_uses_exactly_one_native_enumeration_at_reset_and_unchanged_step_api():
    tree = ast.parse((ROOT/'agent_system/environments/env_package/sciworld/envs.py').read_text(encoding='utf-8'))
    calls = [n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr.startswith('get_valid_action_object_combinations')]
    assert sorted(calls) == ['get_valid_action_object_combinations', 'get_valid_action_object_combinations_with_templates']
