"""Object identity guard regressions; synthetic fixtures are not task outcomes."""
import copy
import gzip
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('object_basis_fixture', ROOT/'experiments/gigpo_diag/diagnostics.py')
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)


def fixture():
    infos = []
    for alias in ('pride and prejudice on shelf', 'book by jane austen on shelf'):
        records = {'7': {'123': alias, 'desc': 'Book'}, '8': {'456': 'shelf', 'desc': 'Shelf'}}
        actions = [dict(action=f'look at {alias}', template_id=5, obj_ids=[123], type_ids=[7])]
        components = dict(observation='same room', objects=diag.science_object_evidence(records)[0],
                          possible_object_referents=records, possible_actions=[actions[0]['action']],
                          possible_action_templates=actions, score=0)
        infos.append(dict(task_id='synthetic', task_num=24, variation_id=141, reset_seed=3,
                          simplifications_requested='easy', task_description='synthetic goal', score=0,
                          reset_evidence_version=3, initial_state_components=components,
                          initial_state_hash=diag.digest(components)))
    return dict(anchor=['same room'] * 2), infos, [0, 0]


def test_aliases_pass_and_all_raw_policy_inputs_are_preserved(tmp_path, monkeypatch):
    monkeypatch.setenv('GIGPO_DIAG_DIR', str(tmp_path))
    obs, infos, uids = fixture()
    original = copy.deepcopy((obs, infos, uids))
    diag.verify_groups(obs, infos, uids, environment='scienceworld')
    assert (obs, infos, uids) == original
    files = list(tmp_path.glob('group_reset_object_text_differences*.jsonl.gz'))
    with gzip.open(files[0], 'rt', encoding='utf-8') as f:
        row = json.loads(next(f))
    assert row['object_variant_count'] == 2
    assert row['full_hidden_state_verified'] is False
    assert row['workers'][1]['reset_info'] == infos[1]


@pytest.mark.parametrize('problem', ['object_id', 'type_id', 'missing_mapping', 'empty_mapping',
                                   'missing_object', 'duplicate_id', 'bad_id', 'bad_type', 'bad_alias',
                                   'missing_desc', 'mismatched_text', 'hash', 'version', 'action_id'])
def test_real_or_inconsistent_evidence_still_rejected(problem):
    obs, infos, uids = fixture()
    c = infos[1]['initial_state_components']
    r = c['possible_object_referents']
    if problem == 'object_id': r['7']['999'] = r['7'].pop('123')
    if problem == 'type_id': r['9'] = r.pop('7')
    if problem == 'missing_mapping': c.pop('possible_object_referents')
    if problem == 'empty_mapping': c['possible_object_referents'] = {}
    if problem == 'missing_object':
        r.pop('8')
        c['objects'] = diag.science_object_evidence(r)[0]
    if problem == 'duplicate_id': r['8']['123'] = 'another alias'
    if problem == 'bad_id': r['7']['bad'] = r['7'].pop('123')
    if problem == 'bad_type': r['bad'] = r.pop('7')
    if problem == 'bad_alias': r['7']['123'] = None
    if problem == 'missing_desc': r['7'].pop('desc')
    if problem == 'mismatched_text': c['objects'] = ['invented alias']
    if problem == 'version': infos[1]['reset_evidence_version'] = 2
    if problem == 'action_id': c['possible_action_templates'][0]['obj_ids'] = [999]
    infos[1]['initial_state_hash'] = diag.digest(c) if problem != 'hash' else 'corrupt'
    with pytest.raises(RuntimeError, match='refusing advantage update'):
        diag.verify_groups(obs, infos, uids, environment='scienceworld')


def test_missing_mapping_fails_even_if_every_worker_matches():
    obs, infos, uids = fixture()
    for info in infos:
        info['initial_state_components'].pop('possible_object_referents')
        info['initial_state_hash'] = diag.digest(info['initial_state_components'])
    with pytest.raises(RuntimeError):
        diag.verify_groups(obs, infos, uids, environment='scienceworld')


def test_duplicate_aliases_do_not_collapse_distinct_objects_and_java_order_is_preserved():
    r = {'7': {'1': 'same', '2': 'same', '3': '\ue000', '4': '\U00010000', 'desc': 'fixture'}}
    aliases, identities = diag.science_object_evidence(r)
    assert aliases == ['same', 'same', '\U00010000', '\ue000']
    assert identities == [(1, 7), (2, 7), (3, 7), (4, 7)]


def test_historical_v2_does_not_retroactively_certify_alias_differences():
    obs, infos, uids = fixture()
    for info in infos:
        info['reset_evidence_version'] = 2
        c = info['initial_state_components']
        c.pop('possible_object_referents')
        info['initial_state_hash'] = diag.digest(c)
    with pytest.raises(RuntimeError):
        diag.verify_groups(obs, infos, uids, environment='scienceworld')
