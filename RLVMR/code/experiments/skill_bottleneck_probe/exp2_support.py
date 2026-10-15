"""Pure Exp2 trial identity, conservative labels, and natural-prefix capture."""
from pathlib import Path
from replay import digest

VERSION = 'skill-bottleneck-exp2-v2'
PILOT_COMMIT = '1ccc08529b59cfd967f632a6d9ab84e9ad9bc68b'
SEMANTICS = ('task_id','task_num','simplifications','max_policy_steps','env_step_limit',
             'temperature','top_p','top_k','max_response_tokens','max_prompt_tokens',
             'generation_seed_rule','natural_reset_seed_rule','label_version','hint_placement','resource_adaptation')


def make_trials(protocol, prefixes, checkpoint, run_id):
    rows=[]
    for p in prefixes:
        for seed in protocol['sampling_seeds']:
            for condition in ('P','H','D'):
                rows.append(dict(condition=condition,variation_id=p['variation_id'],prefix_id=p['prefix_id'],
                                 prefix_origin='historical',reset_seed=p['reset_seed'],sample_seed=seed))
    for v in protocol['natural_variations']:
        for seed in protocol['sampling_seeds']:
            rows.append(dict(condition='N',variation_id=v,prefix_id=None,prefix_origin='none',
                             reset_seed=20261015+v,sample_seed=seed))
    for r in rows:
        r.update(trial_id=f"{checkpoint}-{r['condition']}-{r['prefix_id'] or r['variation_id']}-{r['sample_seed']}",
                 run_id=run_id,checkpoint_id=checkpoint,task_id=protocol['task_id'],exp2=True,
                 prompt_protocol_id=VERSION,dataset_role='train_diagnostic',
                 simplification_profile=protocol['simplifications'],max_steps=protocol['max_policy_steps'])
    return rows


def fingerprints(rows):
    return sorted((Path(r['path']).name,r['bytes'],r['sha256']) for r in rows)


def trial_key(item, protocol, prefix, identity):
    condition=item['condition']
    semantic={k:protocol.get(k) for k in SEMANTICS}
    semantic['hint']=protocol.get('stage_hint') if condition=='H' else protocol.get('direct_B_hint') if condition=='D' else None
    return digest(dict(execution_contract='pilot-v1-compatible-PHN;exp2-D-v1',
        checkpoint=item['checkpoint_id'],model=fingerprints(identity['source_model_fingerprints']),
        tokenizer_revision=identity['tokenizer_revision'],template=identity['tokenizer_chat_template_sha256'],
        placeholder=identity['placeholder'],native=identity['native_identity']['source_sha256_lf'],
        jar=identity['jar_identity']['sha256'],inference={k:v for k,v in identity['resolved_inference_config'].items()
                                                      if k not in ('model','tokenizer')},
        condition=condition,variation=item['variation_id'],reset_seed=item['reset_seed'],sample_seed=item['sample_seed'],
        semantics=semantic,prefix=digest(prefix) if prefix else None))


def first_class(row):
    if not row or row.get('task_phase_before')!='post_A': return 'other_or_unknown'
    failures=row.get('failure_event',[])
    if row.get('B_transition_event') is True: return 'B_success'
    if 'repeat_completed_subgoal' in failures: return 'repeat_A'
    if 'invalid_action' in failures: return 'invalid_action'
    if 'wrong_entity_selection' in failures: return 'wrong_focus'
    action=row.get('executed_action','').strip().lower()
    if action.startswith(('go to ','look at ','look around','examine ','inventory')):
        return 'non_focus_navigation_or_inspection'
    return 'other_or_unknown'


def enrich(episode, records):
    post=[r for r in records if r.get('phase')=='policy' and r.get('task_phase_before')=='post_A']
    first=post[0] if post else None
    focus=next((r for r in post if r.get('executed_action','').strip().lower().startswith('focus on ')),None)
    a_index=episode.get('A_reached_step')
    known=sum(first_class(r)!='other_or_unknown' for r in post)
    return dict(first_postA_action=first.get('executed_action') if first else None,
        first_postA_action_class=first_class(first),
        first_postA_score_pre=first.get('score_before') if first else None,
        first_postA_score_post=first.get('score_after') if first else None,
        first_postA_done=first.get('done_after') if first else None,
        first_postA_wrong_focus=('wrong_entity_selection' in first.get('failure_event',[])) if first else None,
        first_focus_after_A=focus.get('executed_action') if focus else None,first_focus_class=first_class(focus),
        steps_to_first_focus_after_A=focus['step_index']-a_index if focus and a_index is not None else None,
        policy_steps_before_A=a_index+1 if a_index is not None else None,
        continuation_steps=len(post),remaining_budget_at_A=episode.get('remaining_steps_at_A'),
        known_label_count=known,unknown_label_count=len(post)-known,
        manual_text_B_action_A=None,manual_label_status='not_reviewed; never inferred from automatic keyword matching')


def natural_prefix(item, records, reset_info, prompt, ids, protocol):
    a_index=next((i for i,r in enumerate(records) if r.get('A_transition_event') is True),None)
    if a_index is None or a_index+1>=protocol['max_policy_steps']: return None
    rows=records[:a_index+1]; last=rows[-1]
    if last['done_after'] or last.get('B_transition_event') is True: return None
    actions=[r['executed_action'] for r in rows]
    prefix_id='op-'+digest([item['checkpoint_id'],item['variation_id'],item['reset_seed'],actions,prompt])[:16]
    return dict(prefix_id=prefix_id,prefix_origin='onpolicy_natural',source_checkpoint=item['checkpoint_id'],
        source_run_id=item['run_id'],origin_natural_trial_id=item['trial_id'],source_trajectory_id=item['trial_id'],
        source_update=None,source_policy_version=item['checkpoint_id'],source_step_id=f"{item['trial_id']}:{a_index}",
        task_id=protocol['task_id'],task_num=protocol['task_num'],variation_id=item['variation_id'],
        reset_seed=item['reset_seed'],sample_seed=item['sample_seed'],split='train_diagnostic',
        source_load_receipt=reset_info.get('load_receipt'),step_count_at_A=len(rows),
        remaining_policy_steps=protocol['max_policy_steps']-len(rows),score_before_A=last['score_before'],
        score_after_A=last['score_after'],A_event_evidence=last,B_not_yet_completed_evidence=dict(done=False,score=last['score_after']),
        full_executed_action_prefix=actions,source_steps=[dict(analysis_step_id=f"{item['trial_id']}:{r['step_index']}",
            step_index=r['step_index'],raw_response=r['raw_response'],executed_command=r['executed_action'],
            rendered_prompt=r['rendered_prompt'],observation_post=r['observation_after'],score_pre=r['score_before'],
            score_post=r['score_after'],done_raw=r['done_after'],native_info=r['native_info']) for r in rows],
        post_A_rendered_prompt=prompt,post_A_prompt_token_ids=ids,
        prefix_selection_rule='first A prefix per checkpoint/variation in (variation,sample_seed,trial_id) order; no B outcome input')


def select_natural(candidates, cap):
    selected=[]; seen=set()
    for p in sorted(candidates,key=lambda p:(p['variation_id'],p['sample_seed'],p['origin_natural_trial_id'])):
        if p['variation_id'] in seen: continue
        seen.add(p['variation_id']); selected.append(p)
        if len(selected)>=cap: break
    return selected if cap else []


def branch_trials(protocol,prefixes,checkpoint,run_id):
    rows=[]
    for p in prefixes:
        for seed in protocol['onpolicy_branch_seeds']:
            for condition in ('P','D'):
                rows.append(dict(trial_id=f"{checkpoint}-{condition}-{p['prefix_id']}-{seed}",
                    checkpoint_id=checkpoint,condition=condition,prefix_id=p['prefix_id'],
                    prefix_origin='onpolicy_natural',origin_natural_trial_id=p['origin_natural_trial_id'],
                    variation_id=p['variation_id'],reset_seed=p['reset_seed'],sample_seed=seed,
                    run_id=run_id,task_id=protocol['task_id'],prompt_protocol_id=VERSION,
                    dataset_role='train_diagnostic',simplification_profile=protocol['simplifications'],
                    max_steps=protocol['max_policy_steps'],exp2=True))
    return rows
