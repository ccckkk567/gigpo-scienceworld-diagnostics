"""Evidence comparisons only. Never normalize or replace actual actor inputs."""
from collections import Counter
import hashlib
import json


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def boundary_evidence(info, prompt, actions):
    keys = ('taskName','variationIdx','simplificationStr','score','native_completed',
            'moves','look','inv','taskDesc','available_actions','possible_actions')
    return dict(info={k:info.get(k) for k in keys}, rendered_prompt=prompt, actions=actions)


def compare(prefix, replayed, info, prompt, a_reached):
    src = prefix['source_steps']; ref = src[-1]['native_info']
    audit = dict(prefix_id=prefix['prefix_id'], action_prefix_length=len(src), actions_replayed=len(replayed),
        task_assignment_match=info.get('taskName') == prefix['task_id'],
        variation_match=info.get('variationIdx') == prefix['variation_id'],
        simplifications_match=info.get('simplificationStr') == 'easy',
        history_comparison=[r['executed_action'] for r in replayed] == prefix['full_executed_action_prefix'],
        score_sequence_matches=[r['score_after'] for r in replayed] == [r['score_post'] for r in src],
        termination_matches=[r['done_after'] for r in replayed] == [r['done_raw'] for r in src],
        action_feedback_matches=[r['observation_after'] for r in replayed] == [r['observation_post'] for r in src],
        stage_A_verified=a_reached is True,
        stage_B_not_completed=bool(replayed and replayed[-1]['done_after'] is False and info.get('score',100)<100),
        rendered_prompt_comparison=prompt == prefix['post_A_rendered_prompt'],
        remaining_budget_comparison=30-len(replayed) == prefix['remaining_policy_steps'],
        replay_failure_step=None, replay_failure_reason=None,
        objects_structure_comparison='not_observed; native strings only, no additional JVM enumeration')
    key_fields=('task_assignment_match','variation_match','simplifications_match','history_comparison',
                'score_sequence_matches','termination_matches','remaining_budget_comparison')
    observed = ('look','inv','taskDesc','moves','available_actions','possible_actions','native_completed')
    missing=[k for k in observed if info.get(k) is None or ref.get(k) is None]
    audit['missing_boundary_fields']=missing
    audit['boundary_field_equal']={k:info.get(k)==ref.get(k) for k in observed}
    actions = info.get('possible_actions'); expected = ref.get('possible_actions')
    audit['available_actions_comparison'] = ('same_multiset' if isinstance(actions,list) and isinstance(expected,list)
        and Counter(actions)==Counter(expected) else 'different_or_missing')
    audit['boundary_evidence_hash']=digest(boundary_evidence(info,prompt,[r['executed_action'] for r in replayed]))
    if not all(audit[k] for k in key_fields): quality='mismatch'
    elif not audit['stage_A_verified'] or not audit['stage_B_not_completed']: quality='replay_failed'
    elif not missing and all(audit['boundary_field_equal'].values()) and audit['rendered_prompt_comparison'] and audit['action_feedback_matches']:
        quality='verified_match'
    else:
        # No stable object identity/structural API is queried: aliases alone cannot prove structural equivalence.
        quality='ambiguous'
    audit['state_match_quality']=quality
    return audit


def paired_eligible(p, h):
    return (p.get('state_match_quality') == h.get('state_match_quality') == 'verified_match'
        and p.get('boundary_evidence_hash') is not None
        and p.get('boundary_evidence_hash') == h.get('boundary_evidence_hash')
        and p.get('status') == h.get('status') == 'completed'
        and p.get('B_confirmed_completed') is not None and h.get('B_confirmed_completed') is not None)
