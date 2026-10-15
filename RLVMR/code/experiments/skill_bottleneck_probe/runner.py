"""One model per process; independent native JVM per trial; no training or optimizer."""
import argparse
import gzip
import importlib.metadata
import json
import re
import time
import traceback
import zipfile
from pathlib import Path
from labels import transition, VERSION
from replay import compare, digest
from native import manager, Renderer


def plain(x):
    if hasattr(x,'tolist'): return x.tolist()
    if isinstance(x,Path): return str(x)
    raise TypeError(type(x).__name__)


def write(path, obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=plain),encoding='utf-8')


def append(path, obj):
    opener=gzip.open if str(path).endswith('.gz') else open
    # Each record is a complete gzip member; completed evidence survives interruption.
    with opener(path,'at',encoding='utf-8') as f:
        f.write(json.dumps(obj,ensure_ascii=False,default=plain)+'\n')


def jar_identity(reference):
    from scienceworld.constants import JAR_PATH
    import hashlib
    path=Path(JAR_PATH)
    with path.open('rb') as f:
        h=hashlib.sha256()
        for b in iter(lambda:f.read(4*1024**2),b''): h.update(b)
    with zipfile.ZipFile(path) as z:
        classes={k:hashlib.sha256(z.read(k)).hexdigest() if k in z.namelist() else None for k in reference['reference_classes']}
    return dict(path=str(path),sha256=h.hexdigest(),classes=classes,
        label_semantics_verified=classes==reference['reference_classes'],
        scienceworld_version=importlib.metadata.version('scienceworld'))


def trial(spec, protocol, prefix, renderer, llm, out, labels_verified):
    from vllm import SamplingParams
    env=None; records=[]; reset_info=None; a=False; a_index=None; a_action=None; b=False
    label_unknown=not labels_verified; phase_valid=labels_verified; done=False; info={}; obs={}; audit=None; generated=0
    episode={**spec,'status':'running','state_match_quality':'not_applicable' if prefix is None else 'not_run',
        'A_reached':None,'B_confirmed_completed':None,'strict_task_success':None,'termination_reason':None,
        'navigation_loop_after_A':None,'failure_type':[],'label_unknown':label_unknown}
    try:
        mgr,env=manager(protocol,spec['variation_id'],spec['reset_seed'],spec['trial_id'])
        obs,infos=mgr.reset(); info=plain(infos[0]) if hasattr(infos[0],'tolist') else infos[0]
        reset_info=info
        append(out/'resets.jsonl.gz',dict(trial_id=spec['trial_id'],observation=obs,info=info))
        episode.update(task_description=info['task_description'],native_score_initial=info.get('score'))

        def act(raw, phase, prompt, ids, generation=None):
            nonlocal obs,info,done,a,b,a_index,a_action,label_unknown,phase_valid
            before_obs=obs; before_info=info; before_done=done; was_a=a
            next_obs,rewards,dones,infos=mgr.step([raw]); new_info=infos[0]; done=bool(dones[0])
            action=new_info['diag_executed_action']; feedback=next_obs['anchor'][0]
            lab=transition(action,feedback,info.get('score'),new_info.get('score'),done,a,a_action)
            if action.strip().lower()=='reset task': phase_valid=False
            if was_a and isinstance(info.get('score'),(int,float)) and isinstance(new_info.get('score'),(int,float)):
                if 0 <= new_info['score'] < info['score']: phase_valid=False
            if not phase_valid:
                for k in ('A_transition_event','B_transition_event','B_correct_action'): lab[k]=None
                lab['B_attempted']=None; lab['failure_event']=[]
            if lab['A_transition_event'] is None or lab['B_transition_event'] is None: label_unknown=True
            if lab['A_transition_event']: a=True; a_index=len(records); a_action=action
            if lab['B_transition_event']: b=True
            m=re.search(r'<action>(.*?)</action>',raw,re.S)
            r=dict(trial_id=spec['trial_id'],checkpoint_id=spec['checkpoint_id'],condition=spec['condition'],
                prefix_id=spec['prefix_id'],step_index=len(records),phase=phase,rendered_prompt=prompt,prompt_token_ids=ids,
                raw_response=raw,parsed_action=m.group(1).strip() if m else None,executed_action=action,
                fallback_used=m is None,parse_status='native_format_valid' if bool(new_info['is_action_valid']) else 'native_format_invalid',
                observation_before=before_obs['anchor'][0],observation_after=feedback,
                score_before=before_info.get('score'),score_after=new_info.get('score'),done_before=before_done,done_after=done,
                native_reward=float(rewards[0]),native_info=new_info,available_inputs_before=before_info.get('available_actions'),
                available_actions_before=before_info.get('possible_actions'),world_move_count=new_info.get('moves'),
                remaining_budget=protocol['max_policy_steps']-len(records)-1,
                steps_since_A=len(records)-a_index if a_index is not None else None,
                event_stage=('unknown' if label_unknown else 'A_success_step' if lab['A_transition_event'] else
                    ('post_A_1_to_3' if was_a and len(records)-a_index<=3 else 'post_A_later' if was_a else 'pre_A')),
                task_phase_before='unknown' if label_unknown else ('post_A' if was_a else 'pre_A'),
                task_phase_after='unknown' if label_unknown else ('complete' if b else ('post_A' if a else 'pre_A')),
                generation=generation,**lab)
            append(out/'steps.jsonl.gz',r); records.append(r); obs=next_obs; info=new_info

        if prefix:
            for old in prefix['source_steps']:
                prompt,ids=renderer.render(obs)
                act(old['raw_response'],'replay',prompt,ids)
                if records[-1]['executed_action']!=old['executed_command'] or done: break
            boundary_prompt,boundary_ids=renderer.render(obs)
            audit=compare(prefix,records,info,boundary_prompt,a and phase_valid)
            audit.update(trial_id=spec['trial_id'],checkpoint_id=spec['checkpoint_id'],condition=spec['condition'],
                post_A_rendered_prompt=boundary_prompt,post_A_prompt_token_ids=boundary_ids,
                source_prompt_tokens_equal=boundary_ids==prefix['post_A_prompt_token_ids'])
            append(out/'replay_audit.jsonl',audit)
            episode.update(state_match_quality=audit['state_match_quality'],boundary_evidence_hash=audit['boundary_evidence_hash'])
            if audit['state_match_quality'] in ('mismatch','replay_failed'):
                episode.update(status='replay_blocked',termination_reason=audit['state_match_quality'])
                return
        for index in range(len(records),protocol['max_policy_steps']):
            hint=protocol['stage_hint'] if spec['condition']=='H' else None
            prompt,ids=renderer.render(obs,hint)
            seed=spec['sample_seed']+1009*index
            params=SamplingParams(temperature=protocol['temperature'],top_p=protocol['top_p'],top_k=protocol['top_k'],
                max_tokens=protocol['max_response_tokens'],seed=seed)
            # Save the real input before inference, including generation errors/timeouts.
            append(out/'decisions.jsonl.gz',dict(trial_id=spec['trial_id'],step_index=index,
                rendered_prompt=prompt,prompt_token_ids=ids,sampling_seed=seed))
            result=llm.generate([{'prompt_token_ids':ids}],params,use_tqdm=False)[0]
            answer=result.outputs[0]
            raw=renderer.tokenizer.decode(list(answer.token_ids),skip_special_tokens=True)
            act(raw,'policy',prompt,ids,dict(seed=seed,response_token_ids=list(answer.token_ids),finish_reason=answer.finish_reason))
            generated+=1
            if done: break
        episode.update(status='completed',termination_reason='native_done' if done else 'policy_budget_exhausted')
    except Exception as exc:
        episode.update(status='error',termination_reason='execution_error',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
        if prefix and audit is None:
            append(out/'replay_audit.jsonl',dict(trial_id=spec['trial_id'],prefix_id=prefix['prefix_id'],
                state_match_quality='replay_failed',replay_failure_step=len(records),replay_failure_reason=episode['error']))
            episode['state_match_quality']='replay_failed'
    finally:
        if env: env.close()
        completed=episode['status']=='completed'
        post=[r for r in records if r['task_phase_before']=='post_A' and r['phase']=='policy']
        events=[e for r in records for e in r['failure_event']]
        known=completed and not label_unknown
        a_result=True if a else (False if known else None)
        b_result=True if b else (False if known else None)
        strict=info.get('strict_success') if completed else None
        episode.update(A_reached=a_result,A_reached_step=a_index,
            A_reached_evidence=records[a_index] if a_index is not None else None,
            post_A_eligible=bool(a and a_index+1<protocol['max_policy_steps']) if a else (False if known else None),
            B_attempted=any(r['B_attempted'] for r in post) if (a and known) else (True if any(r['B_attempted'] for r in post) else None),
            B_correct_action=b_result,B_confirmed_completed=b_result,strict_task_success=strict,
            reference_task_success=bool(info['won']) if completed and 'won' in info else None,
            native_score_after_A=records[a_index]['score_after'] if a_index is not None else None,
            native_score_final=info.get('score'),episode_done=done if reset_info is not None else None,
            num_policy_steps=generated,total_policy_budget_used=len(records),replayed_policy_steps=len(records)-generated,
            remaining_steps_at_A=protocol['max_policy_steps']-a_index-1 if a_index is not None else None,
            steps_after_A=len(post),failure_stage=('after_A' if a else 'before_A') if known and not b else ('none' if b else 'unknown'),
            failure_type=sorted(set(events)),wrong_focus_after_A=True if 'wrong_entity_selection' in events else (False if known else None),
            repeat_A_after_A=True if 'repeat_completed_subgoal' in events else (False if known else None),
            no_match_after_A=True if any('invalid_action' in r['failure_event'] for r in post) else (False if known else None),
            budget_exhausted=episode['termination_reason']=='policy_budget_exhausted',
            state_unverified=prefix is not None and episode['state_match_quality']!='verified_match',label_unknown=label_unknown,
            first_B_attempt_step=next((r['step_index'] for r in post if r['B_attempted']),None),
            B_first_continuation_action_correct=(post[0]['B_correct_action'] if post else None))
        append(out/'episodes.jsonl.gz',episode)
        print(json.dumps({k:episode.get(k) for k in ('trial_id','status','A_reached','B_confirmed_completed','state_match_quality','error')}),flush=True)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--spec',type=Path,required=True); args=ap.parse_args()
    spec=json.loads(args.spec.read_text()); out=Path(spec['output']); out.mkdir(parents=True,exist_ok=True)
    protocol=spec['protocol']; manifest={**spec,'checkpoint_load_status':'not_loaded','status':'starting','start_time':time.time()}
    try:
        from transformers import AutoTokenizer
        from vllm import LLM
        tokenizer=AutoTokenizer.from_pretrained(spec['base_model'],local_files_only=True)
        renderer=Renderer(tokenizer,protocol,Path(spec['placeholder']))
        identity=jar_identity(spec['native_identity']); manifest['jar_identity']=identity
        manifest['tokenizer_chat_template_sha256']=digest(tokenizer.chat_template)
        manifest['placeholder']=renderer.placeholder
        config=dict(model=spec['inference_model'],tokenizer=spec['base_model'],dtype='bfloat16',
            tensor_parallel_size=1,enforce_eager=True,max_model_len=protocol['max_prompt_tokens']+protocol['max_response_tokens'],
            gpu_memory_utilization=0.35,enable_chunked_prefill=False,disable_log_stats=True)
        manifest['resolved_inference_config']=config; write(out/'run_manifest.json',manifest)
        llm=LLM(**config)
        manifest['actual_loaded_model_path']=llm.llm_engine.model_config.model
        manifest.update(checkpoint_load_status='loaded',status='running'); write(out/'run_manifest.json',manifest)
        prefixes={p['prefix_id']:p for p in spec['prefixes']}
        for item in spec['trials']:
            trial(item,protocol,prefixes.get(item['prefix_id']),renderer,llm,out,identity['label_semantics_verified'])
        manifest['status']='finished'
    except Exception as exc:
        manifest.update(status='failed',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
        raise
    finally:
        manifest['end_time']=time.time(); write(out/'run_manifest.json',manifest)


if __name__=='__main__': main()
