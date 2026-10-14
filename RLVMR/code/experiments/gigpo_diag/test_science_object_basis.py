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



@pytest.mark.parametrize('mutation', ['none', 'id', 'missing_mapping', 'hash'])
def test_historical_book_aliases_or_unstable_ids_cannot_abort(mutation):
    obs, infos, uids = fixture()
    c = infos[1]['initial_state_components']
    if mutation == 'id': c['possible_object_referents']['7']['999'] = c['possible_object_referents']['7'].pop('123')
    if mutation == 'missing_mapping': c.pop('possible_object_referents')
    infos[1]['initial_state_hash'] = 'bad' if mutation == 'hash' else diag.digest(c)
    before = copy.deepcopy((obs, infos, uids))
    row = diag.science_reset_observations(obs, infos, uids)[0]
    assert row['alias_equivalence'] is None
    assert row['initial_state_status'] == 'unverified'
    assert 'objects' in row['differing_components']
    assert before == (obs, infos, uids)


def test_historical_decoder_keeps_multiplicity():
    aliases, identities = diag.science_object_evidence({'7': {'1': 'same', '2': 'same', 'desc': 'fixture'}})
    assert aliases == ['same', 'same']
    assert identities == [(1, 7), (2, 7)]
