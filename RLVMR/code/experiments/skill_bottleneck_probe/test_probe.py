import copy
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from labels import transition
from replay import compare, paired_eligible
from study import select, trials
from analyze import ratio

HERE=Path(__file__).parent


class ProbeTests(unittest.TestCase):
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
