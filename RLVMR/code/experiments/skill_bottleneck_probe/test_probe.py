import copy
import gzip
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from labels import transition
from replay import compare, paired_eligible
from study import select, trials, validate_study_pool, trial_summary, reuse_base, save, sha
from prepare import source_load_receipt
from analyze import ratio

HERE=Path(__file__).parent


class ProbeTests(unittest.TestCase):
    def test_pilot_keeps_later_checkpoints_after_failure_but_returns_partial_exit(self):
        import study
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'shared'; root.mkdir(); out=root/'analysis/pilot'
            for index,run in enumerate(study.RUNS):
                folder=root/'logs/gigpo-reproduction/scienceworld'/run
                cp=folder/'checkpoints'; cp.mkdir(parents=True)
                save(folder/'RUN_MANIFEST.json',{'code_commit':'fixture'})
                (cp/'latest_checkpointed_iteration.txt').write_text('110' if index==0 else '150')
                for step in ([50,110] if index==0 else [150]):
                    actor=cp/f'global_step_{step}/actor'; actor.mkdir(parents=True)
                    for rank in range(2):
                        with (actor/f'model_world_size_2_rank_{rank}.pt').open('wb') as f: f.truncate(1024**2+1)
            base=root/'data/models/Qwen2.5-1.5B-Instruct-989aa798'; base.mkdir(parents=True)
            (base/'model.safetensors').write_bytes(b'fixture; not real weights')
            def process(cmd,**kwargs):
                if Path(cmd[1]).name=='fsdp_export.py':
                    export=Path(cmd[cmd.index('--target-dir')+1])
                    if export.name=='u50': raise subprocess.CalledProcessError(1,cmd)
                    export.mkdir(); (export/'model.safetensors').write_bytes(b'fixture export')
                elif Path(cmd[1]).name=='runner.py':
                    spec=json.loads(Path(cmd[-1]).read_text()); target=Path(spec['output'])
                    with gzip.open(target/'episodes.jsonl.gz','wt',encoding='utf-8') as f:
                        for i,row in enumerate(spec['trials']):
                            status='error' if spec['checkpoint_id']=='u110' and i==0 else 'completed'
                            f.write(json.dumps({**row,'status':status})+'\n')
                return SimpleNamespace(returncode=0)
            argv=['study.py','run','--root',str(root),'--native-code',str(HERE.parents[1]),
                  '--output',str(out),'--work',str(Path(tmp)/'exports')]
            with patch.object(sys,'argv',argv),patch.object(study,'git',return_value='fixture_commit'), \
                 patch.object(study.subprocess,'run',side_effect=process):
                self.assertEqual(study.main(),2)
            manifest=json.loads((out/'RUN_MANIFEST.json').read_text())
            self.assertEqual(manifest['status'],'pilot_partial')
            self.assertEqual({c['checkpoint_id']:c['status'] for c in manifest['checkpoints']},
                             {'base':'finished','u50':'failed','u110':'partial','u150':'finished'})

    def test_complete_base_reuse_preserves_evidence_and_rejects_partial_or_changed_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); old=root/'old'; out=root/'new'; (old/'base').mkdir(parents=True); out.mkdir()
            for name in ('protocol.json','prefixes.jsonl'):
                (old/name).write_text('{}\n'); (out/name).write_text('{}\n')
            manifest=dict(protocol={'fixed':True},source_sha256_lf={'native':'hash'},base_model_revision='revision',
                          tokenizer_revision='revision',git_commit='prior_commit')
            save(old/'RUN_MANIFEST.json',{**manifest,'checkpoints':[dict(checkpoint_id='base',status='finished',
                 source_model_fingerprints=[{'sha256':'weight_hash'}],inference_model_fingerprints=[{'sha256':'weight_hash'}])]})
            save(old/'base/run_manifest.json',dict(status='finished',checkpoint_load_status='loaded',actual_loaded_model_path='/base'))
            for name in ('steps.jsonl.gz','decisions.jsonl.gz','resets.jsonl.gz','invocation.json','replay_audit.jsonl'):
                (old/'base'/name).write_bytes(b'fixture evidence')
            expected=[{'trial_id':'base-N-0-1'},{'trial_id':'base-P-prefix-1'}]
            def episodes(rows):
                with gzip.open(old/'base/episodes.jsonl.gz','wt',encoding='utf-8') as f:
                    for r in rows: f.write(json.dumps(r)+'\n')
            complete=[{**r,'status':'completed','run_id':'old'} for r in expected]
            episodes(complete)
            c={'path':'/base'}; before=sha(old/'base/episodes.jsonl.gz')
            reuse_base(old,out,manifest,c,expected)
            self.assertEqual(c['status'],'reused_completed')
            self.assertEqual(sha(out/'base/episodes.jsonl.gz'),before)
            self.assertEqual(sha(old/'base/episodes.jsonl.gz'),before)
            self.assertEqual(c['reused_from']['git_commit'],'prior_commit')
            with self.assertRaisesRegex(ValueError,'protocol'):
                reuse_base(old,out,{**manifest,'protocol':{'fixed':False}},c,expected)
            for bad in (complete[:1],complete+complete[:1],[complete[0],{**complete[1],'status':'error'}]):
                episodes(bad)
                self.assertFalse(trial_summary(old/'base',expected)['complete'])
                with self.assertRaisesRegex(ValueError,'incomplete'):
                    reuse_base(old,out,manifest,c,expected)

    def test_historical_native_task_index_and_real_training_pool(self):
        protocol=json.loads((HERE/'data/protocol.json').read_text())
        prefixes=[json.loads(s) for s in (HERE/'data/prefixes.jsonl').read_text(encoding='utf-8').splitlines()]
        pool_path=HERE.parents[1]/'agent_system/environments/env_package/sciworld/variations_idx/L0_idx.json'
        pool=json.loads(pool_path.read_text())['train']
        self.assertEqual(protocol['task_num'],18)
        validate_study_pool(protocol,prefixes,pool)
        for p in prefixes:
            self.assertEqual(p['source_load_receipt']['task_id'],18)
            self.assertIn([18,p['variation_id']],pool)
        with self.assertRaisesRegex(ValueError,'recorded task_num=18'):
            validate_study_pool({**protocol,'task_num':22},prefixes,pool)
        with self.assertRaisesRegex(ValueError,'999999'):
            validate_study_pool({**protocol,'natural_variations':[999999]},prefixes,pool)

    def test_receipt_binds_source_episode_task_variation_seed(self):
        p,_,_=self.fixture()
        episode=dict(task_id=p['task_id'],variation_id=p['variation_id'],reset_seed=p['reset_seed'],traj_uid=p['source_trajectory_id'])
        reset={'reset_info':{'task_num':p['task_num'],'load_receipt':p['source_load_receipt']}}
        self.assertEqual(source_load_receipt(episode,reset)['task_id'],18)
        with self.assertRaisesRegex(ValueError,'disagrees'):
            source_load_receipt({**episode,'variation_id':999999},reset)

    def test_stage_requires_native_event_not_score_or_claim(self):
        self.assertFalse(transition('look around','I completed the first goal',33,83,False)['A_transition_event'])
        self.assertFalse(transition('go to outside','You move to outside',0,33,False)['A_transition_event'])
        self.assertTrue(transition('focus on crocodile egg','You focus on the crocodile egg.',33,83,False)['A_transition_event'])
        self.assertFalse(transition('focus on crocodile egg','No known action matches',33,83,False)['A_transition_event'])
        self.assertIsNone(transition('focus on egg','You focus on egg',None,83,False)['A_transition_event'])

    def test_wrong_repeat_and_real_b(self):
        r=transition('focus on crocodile egg','You focus on crocodile egg.',83,-100,True,True,'focus on crocodile egg')
        self.assertEqual(set(r['failure_event']),{'wrong_entity_selection','repeat_completed_subgoal'})
        self.assertFalse(r['B_correct_action'])
        self.assertTrue(transition('focus on fly','You focus on fly.',83,100,True,True)['B_transition_event'])

    def fixture(self):
        p=json.loads((HERE/'data/prefixes.jsonl').read_text(encoding='utf-8').splitlines()[0])
        rr=[dict(executed_action=s['executed_command'],score_after=s['score_post'],done_after=s['done_raw'],
            observation_after=s['observation_post']) for s in p['source_steps']]
        return p,rr,copy.deepcopy(p['source_steps'][-1]['native_info'])

    def test_historical_prefix_boundary_and_no_future_step(self):
        p,rr,info=self.fixture()
        self.assertEqual(len(rr)+p['remaining_policy_steps'],30)
        self.assertEqual(len(rr),p['step_count_at_A'])
        self.assertFalse(p['source_termination_status'])
        result=compare(p,rr,info,p['post_A_rendered_prompt'],True)
        self.assertEqual(result['state_match_quality'],'verified_match')

    def test_alias_drift_auxiliary_and_route_mismatch_local(self):
        p,rr,info=self.fixture()
        info['possible_actions']=info['possible_actions']+['focus on an alternate alias']
        result=compare(p,rr,info,p['post_A_rendered_prompt']+' alternate alias',True)
        self.assertEqual(result['state_match_quality'],'ambiguous')
        info['variationIdx']+=1
        self.assertEqual(compare(p,rr,info,p['post_A_rendered_prompt'],True)['state_match_quality'],'mismatch')
        self.assertEqual(compare(p,rr,p['source_steps'][-1]['native_info'],p['post_A_rendered_prompt'],False)['state_match_quality'],'replay_failed')

    def test_pair_does_not_mix_different_observed_boundaries(self):
        p={'state_match_quality':'verified_match','boundary_evidence_hash':'a','status':'completed','B_confirmed_completed':False}
        self.assertTrue(paired_eligible(p,p))
        self.assertFalse(paired_eligible(p,{**p,'boundary_evidence_hash':'b'}))
        self.assertFalse(paired_eligible(p,{**p,'state_match_quality':'ambiguous'}))
        self.assertFalse(paired_eligible(p,{**p,'B_confirmed_completed':None}))

    def test_inventory_does_not_select_incomplete_abandoned_120(self):
        inv=[{'eligible':True,'step':110,'checkpoint_id':'u110'}, {'eligible':False,'step':120,'checkpoint_id':'u120'}]
        r=select('u120',inv)
        self.assertEqual(r['checkpoint_id'],'u110'); self.assertIsNotNone(r['substitution_reason'])

    def test_manifest_n_p_h_pairing_and_unknown_denominators(self):
        protocol=json.loads((HERE/'data/protocol.json').read_text())
        prefixes=[json.loads(s) for s in (HERE/'data/prefixes.jsonl').read_text(encoding='utf-8').splitlines()]
        rr=trials(protocol,prefixes,'u50','example')
        self.assertEqual(len(rr),16); self.assertEqual(sum(r['condition']=='N' for r in rr),8)
        for p in [r for r in rr if r['condition']=='P']:
            h=next(r for r in rr if r['condition']=='H' and r['prefix_id']==p['prefix_id'] and r['sample_seed']==p['sample_seed'])
            self.assertEqual(p['reset_seed'],h['reset_seed'])
        result=ratio('u50','N','reach_A',[{'A':True},{'A':False},{'A':None}],'A')
        self.assertEqual((result['numerator'],result['denominator'],result['unknown']),(1,2,1))
        self.assertIsNone(ratio('u50','P','B',[],'B')['rate'])

    def test_runner_records_true_success_and_local_execution_error(self):
        import runner
        protocol=json.loads((HERE/'data/protocol.json').read_text()); protocol['max_policy_steps']=3
        spec=trials(protocol,[],'base','fixture')[0]
        class Environment:
            index=0
            def reset(self):
                return {'text':['start'],'anchor':['start']},[{'score':0,'task_description':'fixture'}]
            def step(self, raw):
                self.index+=1; final=self.index==2
                info=dict(score=100 if final else 50,diag_executed_action='focus on fly' if final else 'focus on animal',
                    is_action_valid=1,strict_success=final,won=final,moves=self.index)
                return {'text':['next'],'anchor':['You focus on animal.']},[10.0 if final else 0.0],[final],[info]
            def close(self): pass
        renderer=SimpleNamespace(render=lambda obs,hint=None:('prompt',[1]),
            tokenizer=SimpleNamespace(decode=lambda ids,skip_special_tokens:'<think>x</think><action>focus on animal</action>'))
        llm=SimpleNamespace(generate=lambda *a,**kw:[SimpleNamespace(outputs=[SimpleNamespace(token_ids=[2],finish_reason='stop')])])
        vllm=SimpleNamespace(SamplingParams=lambda **kw:kw)
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp); env=Environment()
            with patch.dict('sys.modules',{'vllm':vllm}),patch.object(runner,'manager',return_value=(env,env)):
                runner.trial(spec,protocol,None,renderer,llm,out,True)
            with gzip.open(out/'episodes.jsonl.gz','rt',encoding='utf-8') as f: episode=json.loads(f.readline())
            self.assertTrue(episode['A_reached']); self.assertTrue(episode['B_confirmed_completed'])
            self.assertEqual(episode['num_policy_steps'],2); self.assertEqual(episode['A_reached_step'],0)
            env=Environment()
            with patch.dict('sys.modules',{'vllm':vllm}),patch.object(runner,'manager',return_value=(env,env)),patch.object(env,'step',side_effect=RuntimeError('fixture native failure')):
                runner.trial({**spec,'trial_id':'failed-fixture'},protocol,None,renderer,llm,out,True)
            with gzip.open(out/'episodes.jsonl.gz','rt',encoding='utf-8') as f: failed=json.loads(list(f)[-1])
            self.assertEqual(failed['status'],'error'); self.assertIsNone(failed['A_reached']); self.assertIsNone(failed['B_confirmed_completed'])


if __name__=='__main__': unittest.main()
