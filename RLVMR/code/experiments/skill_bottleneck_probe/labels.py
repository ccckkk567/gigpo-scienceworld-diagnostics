"""Task-specific, evidence-limited labels. No oracle actions or reward changes."""
TASK = 'lifespan-longest-lived-then-shortest-lived'
VERSION = 'lifespan-ordered-focus-v1'
HINT = ('According to the previous environment feedback, the first subgoal of the original task '
        'has been completed. Continue with the remaining part of the original task.')
DIRECT_B_HINT = ('According to the previous environment feedback, the first subgoal of the original task '
                 'has been completed. Your current objective is to find and focus on the animal with the shortest life span.')


def condition_hint(condition, protocol):
    if condition == 'H': return protocol['stage_hint']
    if condition == 'D': return protocol['direct_B_hint']
    if condition in ('N', 'P'): return None
    raise ValueError(f'Unknown study condition: {condition}')


def transition(action, feedback, before, after, done, already_a=False, a_action=None):
    focus = action.lower().strip().startswith('focus on ')
    recognized = feedback.lower().strip().startswith('you focus on ')
    score_known = all(isinstance(x, (float, int)) and not isinstance(x, bool) for x in (before, after))
    a = None if not score_known else bool(not already_a and focus and recognized and
        before in (0, 17, 33) and after in (50, 67, 83) and after - before == 50 and done is False)
    b = None if not score_known else bool(already_a and focus and recognized and after == 100 and done is True)
    failures = []
    if 'no known action matches' in feedback.lower(): failures.append('invalid_action')
    if already_a and focus and recognized and after == -100: failures.append('wrong_entity_selection')
    # Exact repeat is observable. Different aliases are not proof of a different object.
    if already_a and focus and a_action and action.strip().lower() == a_action.strip().lower():
        failures.append('repeat_completed_subgoal')
    return dict(A_transition_event=a, B_transition_event=b,
                B_attempted=bool(already_a and focus), B_correct_action=b,
                failure_event=failures, label_version=VERSION,
                label_evidence_source='TaskLifeSpan+GoalSequence+native_focus_feedback+score+done')


def update_of(row):
    import re
    m = re.match(r'u(\d+)-(train_rollout|reference_eval)-', row.get('rollout_batch_id', row.get('analysis_step_id', '')))
    if m: return int(m[1]) + (m[2] == 'train_rollout')
    return None
