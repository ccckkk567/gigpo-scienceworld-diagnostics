"""Small routing contracts; no world equivalence, RNG, or diagnostic dependency."""
from collections import Counter, defaultdict
import json
import math
import os

POLICY = 'science-routing-v4'


def fail(kind, detail):
    # Always visible, including with diagnostics disabled or disk logging broken.
    print('SCIENCE_CONTRACT_FAILURE ' + json.dumps({'kind': kind, 'count': 1, 'detail': detail}), flush=True)
    raise RuntimeError(f'ScienceWorld {kind}: {detail}')


def native_action_inputs(actions, objects):
    for key, value in [('action templates', actions), ('objects', objects)]:
        if not isinstance(value, list) or any(not isinstance(x, str) for x in value):
            fail('required_input', 'native ' + key)


def required_inputs(observation, info, *, reset=False):
    for name, value in [('observation', observation), ('available_actions', info.get('available_actions'))]:
        if not isinstance(value, str) or not value.strip():
            fail('required_input', name)
    if reset and (not isinstance(info.get('task_description'), str) or not info['task_description'].strip()):
        fail('required_input', 'task_description')
    actions = info.get('possible_actions')
    if not isinstance(actions, list) or any(not isinstance(x, str) for x in actions):
        fail('required_input', 'possible_actions')
    if not reset:
        score = info.get('score')
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            fail('required_input', 'native step score required by reference reward')


def load_receipt(env, request, task_names, task_num, variation, simplifications, budget, worker_slot):
    """Called after native load/reset succeeded, with arguments used at that call.

    Wrapper attributes confirm Python routing, not hidden JVM/world equivalence.
    """
    actual = dict(task_id=task_num, task_name=task_names[task_num], variation_id=variation,
                  simplifications=simplifications, env_step_limit=budget, worker_slot=worker_slot,
                  reset_seed=int(request['reset_seed']), assignment_token=request['assignment_token'],
                  group_slot=request['group_slot'], reset_seq=request['reset_seq'],
                  load_reset_returned=True, contract_policy=POLICY)
    for attr, key in [('taskName', 'task_name'), ('variationIdx', 'variation_id'),
                      ('simplificationStr', 'simplifications'), ('envStepLimit', 'env_step_limit')]:
        if hasattr(env, attr) and getattr(env, attr) != actual[key]:
            fail('assignment_mismatch', f'worker {worker_slot}: wrapper {attr} disagrees with load arguments')
    return actual


def validate_assignment_receipts(expected, receipts):
    if len(expected) != len(receipts) or not expected:
        fail('assignment_mismatch', 'receipt cardinality')
    for index, (request, receipt) in enumerate(zip(expected, receipts)):
        if not isinstance(receipt, dict):
            fail('assignment_mismatch', f'missing receipt at slot {index}')
        for key, value in request.items():
            if receipt.get(key) != value:
                fail('assignment_mismatch', f'slot {index}: {key}, requested={value!r}, loaded={receipt.get(key)!r}')
        if receipt.get('load_reset_returned') is not True:
            fail('assignment_mismatch', f'load/reset did not complete at slot {index}')


def validate_collector_binding(expected, receipts, episode_uids, traj_uids, group_size):
    validate_assignment_receipts(expected, receipts)
    if not (len(receipts) == len(episode_uids) == len(traj_uids)) or group_size < 1:
        fail('uid_binding_mismatch', 'collector cardinality')
    tokens_to_uid, uids_to_token = defaultdict(set), defaultdict(set)
    counts = Counter()
    trajectories = set()
    for slot, (receipt, uid, trajectory) in enumerate(zip(receipts, episode_uids, traj_uids)):
        token = receipt['assignment_token']
        if receipt['worker_slot'] != slot or not isinstance(uid, str) or not uid or not isinstance(trajectory, str) or not trajectory or trajectory in trajectories:
            fail('uid_binding_mismatch', f'slot/UID/trajectory mismatch at slot {slot}')
        trajectories.add(trajectory)
        tokens_to_uid[token].add(uid)
        uids_to_token[uid].add(token)
        counts[token] += 1
    if any(len(v) != 1 for v in tokens_to_uid.values()) or any(len(v) != 1 for v in uids_to_token.values()):
        fail('uid_binding_mismatch', 'actual collector UID must map one-to-one to assignment token')
    if any(n != group_size for n in counts.values()):
        fail('uid_binding_mismatch', f'expected {group_size} workers per token')


def optional_strict_label(env, info):
    """No retries. Only the two optional getters/conversions are isolated."""
    try:
        score, completed = env.server.getScore(), env.server.getCompleted()
        if type(completed) is not bool or isinstance(score, bool) or not math.isfinite(float(score)):
            raise ValueError('invalid optional native completion/score')
        info.update(native_score_unrounded=float(score), native_completed=completed,
                    strict_success=completed and float(score) >= 1.0)
    except Exception as exc:
        # A dead gateway is an environment fault, even if noticed by an optional getter.
        from py4j.protocol import Py4JNetworkError, Py4JJavaError
        if isinstance(exc, (Py4JNetworkError, Py4JJavaError, ConnectionError, EOFError, OSError)):
            raise
        info.update(native_score_unrounded=None, native_completed=None, strict_success=None,
                    strict_label_error=f'{type(exc).__name__}: {exc}')
    info['strict_definition'] = 'ScienceWorld 1.2.3 Java completed and raw score >= 1; read-only, no reward override'


def confirmed_violations():
    """Operator-reviewed findings only; never infer semantics from text/UUIDs.

    An explicit finding blocks the matching assignment, rather than dropping it.
    No registry is installed by default. Bad operator configuration is an error.
    """
    path = os.environ.get('GIGPO_SCIENCE_CONFIRMED_VIOLATIONS')
    if not path:
        return []
    with open(path, encoding='utf-8') as source:
        rows = json.load(source)
    for row in rows:
        if not all(row.get(k) for k in ('task_name', 'reason', 'evidence_ref')) or type(row.get('variation_id')) is not int:
            fail('confirmed_semantics', 'reviewed finding requires task_name, variation_id, reason, evidence_ref')
    return rows


def enforce_confirmed_violations(assignments, findings):
    for request in assignments:
        for finding in findings:
            if (request['task_name'], request['variation_id']) == (finding['task_name'], finding['variation_id']):
                fail('confirmed_semantics', finding)
