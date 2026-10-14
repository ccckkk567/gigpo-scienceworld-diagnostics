"""Fault injection, not experiment outcomes. No CUDA/model dependencies."""
import copy
import importlib.util
import os
from pathlib import Path
import random
import sys
import types
from unittest import mock

import numpy as np
import pytest
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.gigpo_diag import diagnostics as diag, science_contract as c


def requests(n=8):
    return [dict(task_id=24, task_name='mendelian-genetics-unknown-plant', variation_id=141,
                 simplifications='easy', env_step_limit=100, reset_seed=1562125877,
                 assignment_token='run/train/3/5', group_slot=5, reset_seq=3, worker_slot=i,
                 contract_policy=c.POLICY) for i in range(n)]


def receipts(expected):
    return [{**x, 'load_reset_returned': True} for x in expected]


@pytest.mark.parametrize('field,value', [('task_id', 23), ('task_name', 'old-task'), ('variation_id', 140),
                                      ('simplifications', ''), ('env_step_limit', 200), ('reset_seed', 1),
                                      ('reset_seq', 2), ('assignment_token', 'stale'), ('worker_slot', 9)])
def test_all_workers_loading_the_same_wrong_request_is_rejected(field, value):
    expected = requests()
    loaded = receipts(expected)
    for row in loaded: row[field] = value
    with pytest.raises(RuntimeError, match='assignment_mismatch'):
        c.validate_assignment_receipts(expected, loaded)


def test_actual_uid_mapping_train_eval_replacement_and_stale_receipts():
    expected = requests(16)
    for i in range(8, 16):
        expected[i].update(assignment_token='run/train/3/6', group_slot=6)
    loaded = receipts(expected)
    trajectories = [f't{i}' for i in range(16)]
    ids = ['a']*8 + ['b']*8
    c.validate_collector_binding(expected, loaded, ids, trajectories, 8)
    for bad_ids, bad_trajs, size in [(['a']*16, trajectories, 8), (['a','b']*8, trajectories, 8),
                                     (ids, ['same']*16, 8), (ids, trajectories, 4)]:
        with pytest.raises(RuntimeError, match='uid_binding_mismatch'):
            c.validate_collector_binding(expected, loaded, bad_ids, bad_trajs, size)
    c.validate_collector_binding(expected[:1], loaded[:1], ['eval'], ['t'], 1)
    with pytest.raises(RuntimeError, match='assignment_mismatch'):
        c.validate_collector_binding(expected, list(reversed(loaded)), ids, trajectories, 8)


def test_required_inputs_and_reviewed_semantics_remain_fatal():
    good = dict(available_actions='native text', task_description='goal', possible_actions=['look around'], score=0)
    c.required_inputs('obs', good, reset=True)
    for key in ('available_actions', 'task_description', 'possible_actions'):
        bad = good.copy(); bad.pop(key)
        with pytest.raises(RuntimeError, match='required_input'): c.required_inputs('obs', bad, reset=True)
    with pytest.raises(RuntimeError, match='required_input'):
        c.required_inputs('obs', {**good, 'score': float('nan')})
    for reason in ('confirmed target changed', 'confirmed key object changed'):
        findings = [dict(task_name=requests()[0]['task_name'], variation_id=141, reason=reason,
                         evidence_ref='synthetic reviewed fixture, not natural outcome')]
        with pytest.raises(RuntimeError, match='confirmed_semantics'):
            c.enforce_confirmed_violations(requests(), findings)


def worker_module():
    spec = importlib.util.spec_from_file_location('routing_worker', ROOT/'agent_system/environments/env_package/sciworld/envs.py')
    module = importlib.util.module_from_spec(spec)
    gym = types.ModuleType('gymnasium'); gym.Env = object
    with mock.patch.dict(sys.modules, {'gymnasium': gym}): spec.loader.exec_module(module)
    return module


class Sim:
    calls = []
    native_failure = False
    stale_load = False
    def __init__(self, *args, **kwargs):
        self.envStepLimit = kwargs['envStepLimit']
        self.server = self
    def get_task_names(self): return ['zero', 'one', 'two', 'three']
    def load(self, name, variation, simplifications):
        self.calls.append(('load', name, variation, simplifications))
        self.taskName, self.variationIdx, self.simplificationStr = name, variation, simplifications
        if self.stale_load: self.variationIdx = -1
    def reset(self): self.calls.append('reset'); return 'state', {'score': 0}
    def step(self, action):
        self.calls.append(('step', action))
        if self.native_failure: raise ValueError('native step failed')
        return 'next', 0, False, {'score': 0}
    def get_task_description(self): self.calls.append('description'); return 'goal'
    def get_possible_actions(self): self.calls.append('action_templates'); return ['look around']
    def get_possible_objects(self): self.calls.append('objects'); return ['book alias']
    def get_valid_action_object_combinations(self): self.calls.append('actions'); return ['look around']
    def getScore(self): raise ValueError('optional getter unavailable')
    def close(self): pass


class Pipe:
    def __init__(self, commands): self.commands=iter(commands); self.results=[]
    def recv(self): return next(self.commands)
    def send(self, value): self.results.append(value)
    def close(self): pass


def run_worker(module):
    request = {**requests(1)[0], 'task_id': 2, 'task_name': 'two', 'variation_id': 5}
    pipe = Pipe([('reset', request), ('step', 'look around'), ('close', None)])
    sci = types.ModuleType('scienceworld'); sci.ScienceWorldEnv = Sim
    with mock.patch.dict(sys.modules, {'scienceworld': sci}):
        module._worker(pipe, 0, [], 'easy', 100, None, variations_idx=[[2,5]])
    return pipe


def test_worker_native_path_no_diagnostic_rng_or_reward_changes(tmp_path, monkeypatch):
    module = worker_module()
    saved = random.getstate()
    try:
        monkeypatch.delenv('GIGPO_DIAG_DIR', raising=False)
        Sim.calls = []
        before = run_worker(module)
        state = random.getstate(); calls = Sim.calls[:]
        monkeypatch.setenv('GIGPO_DIAG_DIR', str(tmp_path))
        Sim.calls = []
        after = run_worker(module)
        assert state == random.getstate() and calls == Sim.calls
        assert calls == [('load', 'two', 5, 'easy'), 'reset', 'description', 'action_templates', 'objects', 'actions',
                         ('step', 'look around'), 'action_templates', 'objects', 'actions']
        for key in ('available_actions', 'task_description', 'possible_actions', 'score', 'load_receipt'):
            assert before.results[1][1][key] == after.results[1][1][key]
        assert before.results[1][0] == after.results[1][0]
        assert before.results[2] == after.results[2]
        assert after.results[2][3]['strict_success'] is None
        assert after.results[2][1] == 0  # Native reference reward unchanged.
        with mock.patch.object(diag, 'digest', side_effect=ValueError('bad diagnostic hash')):
            run_worker(module)
        with mock.patch.object(Sim, 'native_failure', True):
            with pytest.raises(ValueError, match='native step failed'): run_worker(module)
        with mock.patch.object(Sim, 'stale_load', True):
            with pytest.raises(RuntimeError, match='assignment_mismatch'): run_worker(module)
    finally: random.setstate(saved)


def test_dead_gateway_is_not_optional_label_failure():
    from py4j.protocol import Py4JNetworkError
    env = types.SimpleNamespace(server=types.SimpleNamespace(getScore=mock.Mock(side_effect=Py4JNetworkError('dead JVM'))))
    with pytest.raises(Py4JNetworkError): c.optional_strict_label(env, {})


def test_parent_rng_replacement_order_and_sampler_resume_unchanged():
    module = worker_module()
    class ParentPipe:
        def send(self, message): self.payload = message[1]
        def recv(self): return 'obs', dict(available_actions='actions', possible_actions=[], load_receipt={**self.payload, 'load_reset_returned': True})
    env = module.SciWorldMultiProcessEnv.__new__(module.SciWorldMultiProcessEnv)
    env.group_n=2; env.env_num=2; env.num_processes=4; env.split='train'
    env.variations_idx=[[1,7],[2,5],[3,2]]; env._rng=np.random.RandomState(0); env._reset_count=0
    env._task_names=['zero','one','two','three']; env._assignment_prefix='run/train'
    env.simplifications_preset='easy'; env.env_step_limit=100; env._confirmed_violations=[]
    env._parent_remotes=[ParentPipe() for _ in range(4)]; env.prev_available_actions=[None]*4; env.prev_possible_actions=[None]*4; env._closed=True
    reference=np.random.RandomState(0)
    for _ in range(3):
        indices=reference.choice(3, size=2, replace=True); seeds=reference.randint(0, 2**31-1, size=2)
        env.reset()
        for group in range(2):
            for slot in (group*2, group*2+1):
                p=env.expected_assignments[slot]
                assert [p['task_id'], p['variation_id']] == env.variations_idx[indices[group]]
                assert p['reset_seed'] == seeds[group]
        assert np.array_equal(reference.get_state()[1], env._rng.get_state()[1])
        assert reference.get_state()[2:] == env._rng.get_state()[2:]
    saved=env.sampler_state(); env.reset(); expected=copy.deepcopy(env.expected_assignments)
    env.restore_sampler_state(saved); env.reset(); assert env.expected_assignments == expected


def test_native_anchor_support_deduplicates_rows_and_counts_trajectories(tmp_path, monkeypatch):
    import torch
    monkeypatch.setenv('GIGPO_DIAG_DIR', str(tmp_path))
    monkeypatch.setattr(diag, '_adv_ids', ['t1:0','t1:0','t1:1','t1:2','t2:0'])
    ae = torch.ones((5,1)); step = torch.tensor([[0.],[0.],[0.],[2.],[0.]])
    diag.capture_core(['episode']*5, ['single']*3+['cross']*2,
                      torch.zeros((5,1)), torch.zeros(5), ae, step, ae+step, torch.ones((5,1)), 1.0)
    assert diag._pending['diag/adv/unique_active_steps'] == 4
    assert diag._pending['diag/anchor/support_known_steps'] == 4
    assert diag._pending['diag/anchor/cross_trajectory_supported_steps'] == 2
    assert diag._pending['diag/anchor/cross_trajectory_support_rate'] == .5
    assert diag._pending['diag/adv/AS_nonzero_supported_steps'] == 1
    assert diag._pending['diag/adv/AS_supported_denominator'] == 2
