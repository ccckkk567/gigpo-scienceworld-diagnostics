"""Fault-injection and protocol regression tests; no GPU, weights, or live JVM."""
import copy
import gzip
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from labels import HINT, DIRECT_B_HINT, condition_hint
from study import save, csv_write
from exp2_support import make_trials, trial_key, first_class, enrich, select_natural, branch_trials, PILOT_COMMIT
from exp2_runtime import EvidenceSource, read_records, write_records, complete_evidence, run_trials
from exp2_analyze import analyze
import runner

HERE=Path(__file__).parent


def protocol(): return json.loads((HERE/'data/protocol_v2.json').read_text())


class Renderer:
    tokenizer=types.SimpleNamespace(decode=lambda ids,**kw:''.join(chr(x) for x in ids))
    def render(self,obs,hint=None):
        text=obs['text'][0]+('\n\n'+hint if hint else '')
        return text,list(map(ord,text))


class FakeLLM:
    def generate(self,prompts,params,**kw):
        text=''.join(chr(x) for x in prompts[0]['prompt_token_ids'])
        action='focus on baby dragonfly' if DIRECT_B_HINT in text else 'focus on crocodile egg'
        answer=types.SimpleNamespace(token_ids=list(map(ord,'<think>fixture</think><action>'+action+'</action>')),finish_reason='stop')
        return [types.SimpleNamespace(outputs=[answer])]


class Environment:
    def __init__(self,p,v,seed,token): self.p=p;self.v=v;self.seed=seed;self.token=token;self.n=0;self.score=0
    def info(self,action='',done=False):
        return dict(taskName=self.p['task_id'],variationIdx=self.v,simplificationStr='easy',score=self.score,
            native_completed=done,moves=self.n,look='outside',inv='',taskDesc='longest then shortest',
            available_actions=['focus on OBJ'],possible_actions=['focus on crocodile egg','focus on baby dragonfly'],
            diag_executed_action=action,is_action_valid=True,strict_success=self.score==100,won=self.score==100,
            task_description='longest then shortest',load_receipt=dict(task_id=18,task_name=self.p['task_id'],
                variation_id=self.v,reset_seed=self.seed,simplifications='easy',load_reset_returned=True))
    def reset(self): return dict(anchor=['outside'],text=['outside']),[self.info()]
    def step(self,raws):
        action=re.search(r'<action>(.*?)</action>',raws[0]).group(1);self.n+=1
        self.score=50 if self.n==1 else 100 if action=='focus on baby dragonfly' else -100
        done=self.n>1;feedback='You focus on '+action.removeprefix('focus on ')+'.'
        return dict(anchor=[feedback],text=[feedback]),[float(self.score==100)],[done],[self.info(action,done)]
    def close(self): pass


def fixture_manager(*args):
    env=Environment(*args);return env,env


def fake_vllm(): return types.SimpleNamespace(SamplingParams=lambda **kw:types.SimpleNamespace(**kw))


def identity(p,trials,prefixes):
    return dict(checkpoint_id='u110',checkpoint_load_status='loaded',actual_loaded_model_path='/exports/u110',
        inference_model='/exports/u110',source_model_fingerprints=[dict(path='/cp/model.pt',bytes=123,sha256='weight')],
        inference_model_fingerprints=[dict(path='/exports/u110/model.safetensors',bytes=12,sha256='export')],
        tokenizer_revision='revision',tokenizer_chat_template_sha256='template',placeholder={'prompt':'original'},
        native_identity={'source_sha256_lf':{'native':'unchanged'}},jar_identity={'sha256':'jar','label_semantics_verified':True},
        resolved_inference_config={'model':'/exports/u110','tokenizer':'/base','dtype':'bfloat16'},protocol=p,
        prefixes=prefixes,trials=trials)


class Exp2Tests(unittest.TestCase):
    def run_natural(self,root):
        p=protocol();item=next(r for r in make_trials(p,[],'u110','fixture') if r['variation_id']==0)
        root.mkdir()
        with patch.object(runner,'manager',side_effect=fixture_manager),patch.dict('sys.modules',{'vllm':fake_vllm()}):
            runner.trial(item,p,None,Renderer(),FakeLLM(),root,True)
        return p,item,read_records(root/'natural_prefix_candidates.jsonl.gz')[0]

    def test_frozen_design_and_unchanged_P_H_inputs(self):
        p=protocol();old=json.loads((HERE/'data/protocol.json').read_text())
        prefixes=[json.loads(s) for s in (HERE/'data/prefixes.jsonl').read_text().splitlines()]
        rows=[r for c in p['checkpoint_ids'] for r in make_trials(p,prefixes,c,'new')]
        self.assertEqual(len(rows),96);self.assertEqual(len({r['trial_id'] for r in rows}),96)
        self.assertEqual(sum(r['condition']=='N' for r in rows),24)
        self.assertEqual({r['variation_id'] for r in rows if r['condition']=='N'},{0,40,50,55})
        obs={'text':['unchanged actor history']};renderer=Renderer()
        self.assertEqual(p['stage_hint'],HINT);self.assertEqual(p['direct_B_hint'],DIRECT_B_HINT)
        for c in ('P','H','N'):
            self.assertEqual(renderer.render(obs,condition_hint(c,p)),renderer.render(obs,condition_hint(c,old)))
        d=renderer.render(obs,condition_hint('D',p))[0]
        self.assertIn(DIRECT_B_HINT,d);self.assertNotIn('dragonfly',d);self.assertEqual(obs['text'][0],'unchanged actor history')
        self.assertEqual(prefixes[2]['full_executed_action_prefix'][-1],'focus on baby baby brown bear')

    def test_real_trial_loop_capture_replay_D_and_first_action_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);p,item,prefix=self.run_natural(root/'N')
            natural=read_records(root/'N/episodes.jsonl.gz')[0]
            self.assertFalse(natural['B_confirmed_completed']);self.assertEqual(natural['first_postA_action_class'],'repeat_A')
            self.assertEqual(len(read_records(root/'N/steps.jsonl.gz')),2)
            self.assertEqual(prefix['remaining_policy_steps'],29)
            self.assertEqual(len(prefix['source_steps']),1)  # Never includes natural later failure/answer.
            for condition in ('P','D'):
                out=root/condition;out.mkdir();spec={**item,'condition':condition,'prefix_id':prefix['prefix_id'],
                    'prefix_origin':'onpolicy_natural','trial_id':condition}
                with patch.object(runner,'manager',side_effect=fixture_manager),patch.dict('sys.modules',{'vllm':fake_vllm()}):
                    runner.trial(spec,p,prefix,Renderer(),FakeLLM(),out,True)
                ep=read_records(out/'episodes.jsonl.gz')[0];records=read_records(out/'steps.jsonl.gz')
                self.assertEqual(ep['state_match_quality'],'verified_match')
                self.assertEqual(ep['B_confirmed_completed'],condition=='D')
                self.assertEqual(ep['first_postA_action_class'],'B_success' if condition=='D' else 'repeat_A')
                self.assertNotIn(DIRECT_B_HINT,records[0]['rendered_prompt'])
                self.assertEqual(DIRECT_B_HINT in records[1]['rendered_prompt'],condition=='D')
                self.assertEqual(records[1]['generation']['seed'],item['sample_seed']+1009)

    def test_pilot_subset_reuse_and_changed_model_generation_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);p,item,prefix=self.run_natural(root/'N')
            src=root/'pilot';target=src/'u110';target.mkdir(parents=True)
            spec={**item,'condition':'P','prefix_origin':'historical','prefix_id':prefix['prefix_id'],'trial_id':'u110-P-fixture'}
            with patch.object(runner,'manager',side_effect=fixture_manager),patch.dict('sys.modules',{'vllm':fake_vllm()}):
                runner.trial(spec,p,prefix,Renderer(),FakeLLM(),target,True)
            old={**p,'protocol_version':'skill-bottleneck-v1.1-routing'};old.pop('direct_B_hint')
            m=identity(old,[spec],[prefix]);save(target/'run_manifest.json',m)
            save(src/'RUN_MANIFEST.json',dict(git_commit=PILOT_COMMIT,protocol_version=old['protocol_version'],experiment_id='original'))
            new=identity(p,[spec],[prefix]);source=EvidenceSource(src,'u110',{'execution_source_sha256':{}})
            dest=root/'import';dest.mkdir();key=trial_key(spec,p,prefix,new)
            self.assertTrue(source.attempt(spec,prefix,key,dest)[0])
            self.assertEqual(read_records(dest/'steps.jsonl.gz'),read_records(target/'steps.jsonl.gz'))
            self.assertEqual(read_records(dest/'episodes.jsonl.gz')[0]['run_id'],'fixture')
            changed=copy.deepcopy(new);changed['source_model_fingerprints'][0]['sha256']='other'
            reject=root/'reject';reject.mkdir()
            self.assertFalse(source.attempt(spec,prefix,trial_key(spec,p,prefix,changed),reject)[0])
            self.assertFalse(source.attempt(spec,prefix,trial_key(spec,{**p,'temperature':.8},prefix,new),reject)[0])
            self.assertFalse((reject/'episodes.jsonl.gz').exists())
            parts={name:read_records(target/name) for name in ('episodes.jsonl.gz','steps.jsonl.gz','resets.jsonl.gz','decisions.jsonl.gz','replay_audit.jsonl')}
            self.assertTrue(complete_evidence(parts,prefix));parts['steps.jsonl.gz']=parts['steps.jsonl.gz'][:-1]
            self.assertFalse(complete_evidence(parts,prefix))

    def test_natural_selection_has_no_later_outcome_filter_and_cap(self):
        candidates=[dict(prefix_id=f'{v}-{s}',variation_id=v,sample_seed=s,origin_natural_trial_id=f'{v}-{s}',
                         reset_seed=v,B_later_success=(s==2)) for v in (55,0,40,50) for s in (2,1)]
        selected=select_natural(candidates,2)
        self.assertEqual([(p['variation_id'],p['sample_seed']) for p in selected],[(0,1),(40,1)])
        self.assertEqual(select_natural([],4),[]);self.assertEqual(select_natural(candidates,0),[])
        branch=branch_trials(protocol(),selected,'u110','new')
        self.assertEqual(len(branch),8);self.assertEqual({r['sample_seed'] for r in branch},{20261115,20261116})

    def test_exp2_resume_reuses_complete_trials_and_rejects_error_markers(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);p,item,prefix=self.run_natural(root/'N')
            src=root/'prior';folder=src/'u110/trials'/item['trial_id'];folder.mkdir(parents=True)
            for name in ('episodes.jsonl.gz','steps.jsonl.gz','decisions.jsonl.gz','resets.jsonl.gz','natural_prefix_candidates.jsonl.gz'):
                write_records(folder/name,read_records(root/'N'/name))
            save(folder/'trial_spec.json',item)
            m=identity(p,[item],[]);execution={'runner.py':'same-code'}
            save(src/'u110/run_manifest.json',m)
            save(src/'RUN_MANIFEST.json',dict(git_commit='exp2-commit',protocol_version=p['protocol_version'],
                execution_source_sha256=execution,experiment_id='prior'))
            source=EvidenceSource(src,'u110',{'execution_source_sha256':execution})
            dest=root/'import';dest.mkdir();key=trial_key(item,p,None,m)
            self.assertTrue(source.attempt(item,None,key,dest)[0])
            self.assertEqual(read_records(dest/'natural_prefix_candidates.jsonl.gz')[0]['source_run_id'],'fixture')
            save(folder/'execution_failure.json',dict(error='fixture interrupted teardown'))
            reject=root/'reject';reject.mkdir()
            self.assertEqual(source.attempt(item,None,key,reject),(False,'source_trial_has_execution_or_import_failure'))
            with self.assertRaisesRegex(ValueError,'execution code changed'):
                EvidenceSource(src,'u110',{'execution_source_sha256':{'runner.py':'different'}})

    def test_driver_continues_after_missing_exact_checkpoint_and_preserves_trial_plan(self):
        import exp2
        import study
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'shared';root.mkdir();out=root/'analysis/exp2'
            # Only u150 exists: there must be no fallback from u110 to another model.
            folder=root/'logs/gigpo-reproduction/scienceworld'/study.RUNS[1]
            actor=folder/'checkpoints/global_step_150/actor';actor.mkdir(parents=True)
            save(folder/'RUN_MANIFEST.json',{'code_commit':'fixture'})
            (actor.parent.parent/'latest_checkpointed_iteration.txt').write_text('150')
            for rank in range(2):
                with (actor/f'model_world_size_2_rank_{rank}.pt').open('wb') as f: f.truncate(1024**2+1)
            called=[]
            def process(cmd,**kwargs):
                called.append(Path(cmd[1]).name)
                if Path(cmd[1]).name=='fsdp_export.py':
                    export=Path(cmd[cmd.index('--target-dir')+1]);export.mkdir()
                    (export/'model.safetensors').write_bytes(b'fixture export, not model weights')
                elif Path(cmd[1]).name=='runner.py':
                    spec=json.loads(Path(cmd[-1]).read_text())
                    self.assertEqual(spec['checkpoint_id'],'u150');self.assertEqual(spec['onpolicy_prefix_cap'],4)
                    self.assertEqual(len(spec['trials']),48)
                    target=Path(spec['output'])
                    for item in spec['trials']:
                        trialdir=target/'trials'/item['trial_id'];trialdir.mkdir(parents=True)
                        save(trialdir/'trial_spec.json',item)
                        write_records(trialdir/'episodes.jsonl.gz',[{**item,'status':'completed','A_reached':False,
                            'B_confirmed_completed':False,'strict_task_success':False,'post_A_eligible':False,
                            'state_match_quality':'ambiguous' if item['condition']!='N' else 'not_applicable'}])
                    save(target/'branch_selection.json',dict(status='not_triggered',candidates=0,trials=[],selected_prefix_ids=[]))
                return types.SimpleNamespace(returncode=0)
            argv=['exp2.py','--root',str(root),'--native-code',str(HERE.parents[1]),
                  '--output',str(out),'--work',str(Path(temp)/'exports')]
            with patch.object(sys,'argv',argv),patch.object(exp2,'git',return_value='fixture_commit'), \
                 patch.object(exp2.subprocess,'run',side_effect=process):
                self.assertEqual(exp2.main(),2)
            m=json.loads((out/'RUN_MANIFEST.json').read_text())
            self.assertEqual(m['status'],'exp2_partial');self.assertEqual(m['summary']['core_planned'],96)
            self.assertEqual(m['summary']['core_completed'],48)
            self.assertEqual({c['checkpoint_id']:c['status'] for c in m['checkpoints']},{'u110':'failed','u150':'finished'})
            self.assertEqual(called,['fsdp_export.py','runner.py'])
            self.assertTrue((out/'reports/final_or_partial.md').exists())
            self.assertIsNone(m['summary']['plot_error'])

    def test_local_replay_failure_continues_and_analysis_keeps_NA_exclusions(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);p=protocol()
            prefixes=[json.loads(s) for s in (HERE/'data/prefixes.jsonl').read_text().splitlines()]
            planned=make_trials(p,prefixes,'u110','test')
            out=root/'u110';out.mkdir();spec=dict(checkpoint_id='u110',experiment_id='test',protocol=p,
                prefixes=prefixes,trials=planned,reuse_sources=[],onpolicy_prefix_cap=2,execution_source_sha256={})
            called=[]
            def execute(item,protocol,prefix,renderer,llm,folder,labels):
                called.append(item['trial_id']);blocked=item['prefix_id']==p['primary_prefixes'][2]
                ep={**item,'status':'replay_blocked' if blocked else 'completed','A_reached':item['condition']!='N',
                    'post_A_eligible':item['condition']!='N','A_reached_step':None,'state_match_quality':'mismatch' if blocked else 'verified_match',
                    'B_confirmed_completed':None if blocked else False,'strict_task_success':False,
                    'B_first_continuation_action_correct':None,'repeat_A_after_A':None,'wrong_focus_after_A':None,
                    'steps_after_A':0,'episode_done':False,'boundary_evidence_hash':'same'}
                write_records(folder/'episodes.jsonl.gz',[ep])
            run_trials(spec,Renderer(),None,out,{'label_semantics_verified':True},identity(p,planned,prefixes),execute)
            self.assertEqual(len(called),48)
            self.assertEqual(json.loads((out/'branch_selection.json').read_text())['status'],'not_triggered')
            csv_write(root/'study_manifest.csv',planned)
            save(root/'RUN_MANIFEST.json',dict(protocol=p,protocol_version=p['protocol_version'],git_commit='fixture'))
            summary=analyze(root)
            self.assertEqual(summary['core_completed'],39);self.assertEqual(summary['branch_planned'],0)
            import csv
            with (root/'natural_by_checkpoint.csv').open() as f: n=list(csv.DictReader(f))
            self.assertTrue(all(r['B_given_A_denominator']=='0' and r['B_given_A_rate']=='' for r in n))
            with (root/'postA_by_prefix.csv').open() as f: rr=list(csv.DictReader(f))
            excluded=[r for r in rr if r['checkpoint']=='u110' and r['variation_id']=='50']
            self.assertTrue(all(r['excluded']=='3' and r['B_denominator']=='0' for r in excluded))


if __name__=='__main__': unittest.main()
