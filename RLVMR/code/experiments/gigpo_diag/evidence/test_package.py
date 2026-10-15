import gzip
import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('pack',Path(__file__).with_name('package.py'))
pack=importlib.util.module_from_spec(spec)
spec.loader.exec_module(pack)


class EvidenceTests(unittest.TestCase):
    def test_update_mapping(self):
        self.assertEqual(pack.update_of('steps_numeric',{'rollout_batch_id':'u45-train_rollout-3'}),46)
        self.assertEqual(pack.update_of('episodes',{'rollout_batch_id':'u45-reference_eval-3'}),45)
        self.assertEqual(pack.update_of('advantages',{'policy_version':44}),45)
        self.assertEqual(pack.update_of('science_reset_observations',{'policy_version':110,'sample_kind':'train'}),111)
        self.assertEqual(pack.update_of('science_reset_observations',{'policy_version':110,'sample_kind':'eval'}),110)
        self.assertIsNone(pack.update_of('optimizer',{'policy_version':0,'optimizer_step_local':100}))

    def test_active_gzip_and_fixed_size(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'a.gz'
            a=gzip.compress(b'{"a":1}\n')
            p.write_bytes(a+gzip.compress(b'{"a":2}\n')[:-5])
            issues=[]
            self.assertEqual(list(pack.lines(p,len(a),issues)),[(1,b'{"a":1}')])
            self.assertFalse(issues)
            issues=[]
            self.assertEqual(len(list(pack.lines(p,p.stat().st_size,issues))),2)
            self.assertIn('incomplete_gzip_member_at_snapshot_boundary',issues)

    def test_snapshot_copy_and_hash(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'source'; q=Path(td)/'snapshot'
            p.write_bytes(b'firstlater')
            result=pack.snapshot_file(p,q,5,0)
            self.assertEqual(q.read_bytes(),b'first')
            self.assertEqual(result['sha256'],pack.hash_file(q))
            self.assertEqual(p.read_bytes(),b'firstlater')

    def test_export_leaves_source_unchanged_and_excludes_weights(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/'runs'; src=root/'scienceworld-161254-test'
            (src/'diagnostics').mkdir(parents=True)
            (src/'RUN_MANIFEST.json').write_text(json.dumps({'code_commit':'fixture-only'}))
            (src/'checkpoints').mkdir()
            (src/'checkpoints'/'weights.pt').write_bytes(b'not a model')
            def write(kind,rows):
                with gzip.open(src/'diagnostics'/(kind+'.pid-1.jsonl.gz'),'wt') as f:
                    for r in rows:
                        f.write(json.dumps(r)+'\n')
            common={'policy_version':0,'rollout_batch_id':'u0-train_rollout-1','sample_kind':'train_rollout','episode_group_uid':'g','task_id':'fixture','detail_selected':True}
            reset=[dict(common,traj_uid=str(i)) for i in range(2)]
            write('group_resets',reset)
            write('episodes',[dict(r,active_steps=1,strict_success=False,reference_success=False) for r in reset])
            steps=[dict(r,analysis_step_id='u0-train_rollout-1:'+r['traj_uid']+':0',step_index=0) for r in reset]
            write('detailed_steps',steps); write('steps_numeric',steps)
            write('updates',[{'policy_update':1,'native_actor_update_returned':True},{'policy_update':3,'native_actor_update_returned':True}])
            later=dict(common,rollout_batch_id='u1-train_rollout-2',episode_group_uid='partial',traj_uid='later',active_steps=2,strict_success=True)
            write('group_resets',reset+[later])
            write('episodes',[dict(r,active_steps=1,strict_success=False,reference_success=False) for r in reset]+[later])
            before={str(p):pack.hash_file(p) for p in src.rglob('*') if p.is_file()}
            out=Path(td)/'export'
            subprocess.run([sys.executable,str(Path(__file__).with_name('package.py')),'--root',str(root),'--output',str(out),'--expected-group-size','2'],check=True,stdout=subprocess.DEVNULL)
            self.assertEqual(before,{str(p):pack.hash_file(p) for p in src.rglob('*') if p.is_file()})
            self.assertFalse(list(out.rglob('*.pt')))
            m=json.loads((out/'PACKAGE_MANIFEST.json').read_text())
            self.assertEqual(m['complete_text_groups'],1)
            self.assertEqual(m['partial_text_groups'],1)
            self.assertEqual(m['completed_updates_missing_metrics'],[1,3])
            with (out/'reanalysis'/'task_update_stage_metrics.csv').open() as f:
                self.assertFalse(any(r['axis']=='update' and r['period']=='2' for r in csv.DictReader(f)))
            with (out/'FILE_MANIFEST.csv').open() as f:
                for r in csv.DictReader(f):
                    self.assertNotEqual(r['path'],'FILE_MANIFEST.csv')
                    self.assertEqual(r['sha256'],pack.hash_file(out/r['path']))
            self.assertEqual((out/'raw_snapshot'/'diagnostics'/'episodes.pid-1.jsonl.gz').read_bytes(),(src/'diagnostics'/'episodes.pid-1.jsonl.gz').read_bytes())
            self.assertTrue(out.with_suffix('.tar.gz').exists())

    def test_resume_overlap_all_record_types_and_missing_new_no_fallback(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/'runs'
            names=('scienceworld-161254-fixture','scienceworld-resume161254-fixture')
            data_by_run={}
            for name,job,updates in ((names[0],'161254',range(1,121)),(names[1],'162461',range(111,151))):
                src=root/name; (src/'diagnostics').mkdir(parents=True)
                (src/'RUN_MANIFEST.json').write_text(json.dumps(dict(job_id=job,code_commit='fixture',run_id=name)))
                (src/'checkpoints'/'global_step_150'/'actor').mkdir(parents=True)
                (src/'checkpoints'/'global_step_150'/'actor'/'optim_world_size_2_rank_0.pt').write_bytes(b'exclude weights')
                (src/'swanlog').mkdir(); (src/'swanlog'/'local.swanlab').write_bytes(b'opaque tracker evidence')
                data={kind:[] for kind in ('updates','metrics','group_resets','episodes','steps_numeric','detailed_steps','advantages','reward_inputs','anchor_summaries','science_reset_observations','optimizer','tracker')}
                for u in updates:
                    data['updates'].append(dict(policy_update=u,native_actor_update_returned=True,run_id=name))
                    data['metrics'].append(dict(global_step=u,metrics={'sentinel':int(job)},run_id=name))
                    data['science_reset_observations'].append(dict(policy_version=u-1,sample_kind='train',run_id=name))
                    for i in range(2):
                        c=dict(run_id=name,policy_version=u-1,rollout_batch_id=f'u{u-1}-train_rollout-1',sample_kind='train_rollout',episode_group_uid='g',traj_uid=str(i),task_id='fixture',detail_selected=True)
                        data['group_resets'].append(c)
                        data['episodes'].append(dict(c,active_steps=1,reference_success=job=='162461',strict_success=False))
                        step=dict(c,analysis_step_id=f'u{u-1}-train_rollout-1:{i}:0',step_index=0,action_signature=f'action{i}',format_valid_native=True)
                        data['steps_numeric'].append(step); data['detailed_steps'].append(step)
                        a=dict(run_id=name,policy_version=u-1,analysis_step_id=step['analysis_step_id'],AE=i+1,AS=i+1,joint=2*(i+1),step_weight=1,actual_step_return=i,step_group_id='anchor')
                        data['advantages'].extend([a,a])  # harmless training padding
                        for stage,val in [('before_invalid_penalty',1),('after_invalid_penalty',0.9)]:
                            data['reward_inputs'].append(dict(a,stage=stage,token_score_sum=val,step_return=val))
                    data['anchor_summaries'].append(dict(run_id=name,policy_version=u-1,step_group_id='anchor'))
                # Startup validation and lifecycle records must not impersonate update 110.
                if job=='162461':
                    data['metrics'].append(dict(global_step=110,metrics={'sentinel':-999},run_id=name))
                data['optimizer'].append(dict(policy_version=0,optimizer_step_local=7,run_id=name))
                data['tracker'].append(dict(policy_version=0,state='initialized',run_id=name))
                for kind,rs in data.items():
                    with gzip.open(src/'diagnostics'/(kind+'.pid-1.jsonl.gz'),'wt') as f:
                        for r in rs:f.write(json.dumps(r)+'\n')
                data_by_run[name]=data
            before={str(p):pack.hash_file(p) for p in root.rglob('*') if p.is_file()}
            logs=Path(td)/'logs'; logs.mkdir()
            for job in ('161254','162461'):
                (logs/f'ggp-scienceworld-train-{job}.out').write_text('fixture slurm output '+job)
            args=[sys.executable,str(Path(__file__).with_name('package.py')),'--root',str(root),
                  '--slurm-log-root',str(logs),
                  '--segment',names[0],'0','110','--segment',names[1],'111','150',
                  '--require-through','150','--expected-group-size','2','--source-mib-per-second','0']
            out=Path(td)/'merged'
            result=subprocess.run(args+['--output',str(out)],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            m=json.loads((out/'PACKAGE_MANIFEST.json').read_text())
            self.assertEqual(m['completed_updates'],list(range(1,151)))
            self.assertEqual(m['coverage_status'],'complete')
            self.assertEqual(m['complete_text_groups'],150)
            self.assertEqual(m['partial_text_groups'],0)
            self.assertEqual(m['advantage_conflicting_step_ids'],0)
            self.assertFalse(list(out.rglob('*.pt')))
            for p in (out/'records').glob('*.gz'):
                with gzip.open(p,'rt') as f:
                    for line in f:
                        r=json.loads(line); u=pack.update_of(p.name.split('.')[0],r)
                        self.assertEqual(r['run_id'],names[u>110])
            with (out/'reanalysis'/'unique_step_advantages.csv').open() as f:
                adv=list(csv.DictReader(f))
            self.assertEqual(len(adv),300)
            self.assertTrue(all(r['training_row_copies']=='2' for r in adv))
            with (out/'reanalysis'/'anchor_groups.csv').open() as f:
                self.assertEqual(len(list(csv.DictReader(f))),150)
            with (out/'reanalysis'/'reward_boundary_deltas.csv').open() as f:
                self.assertTrue(all(r['status']=='paired' for r in csv.DictReader(f)))
            self.assertEqual(before,{str(p):pack.hash_file(p) for p in root.rglob('*') if p.is_file()})
            self.assertTrue((out/'auxiliary_records'/names[1]/'optimizer.jsonl.gz').exists())
            self.assertTrue((out/'figures'/'outcomes.svg').exists())
            self.assertEqual(len(m['tracker_sources']),2)
            self.assertEqual((out/'raw_snapshot'/'slurm_logs'/'ggp-scienceworld-train-162461.out').read_text(),'fixture slurm output 162461')
            # Missing resumed 115 must stay missing although original 115 exists.
            src=root/names[1]
            for kind,key in [('updates','policy_update'),('metrics','global_step')]:
                with gzip.open(src/'diagnostics'/(kind+'.pid-1.jsonl.gz'),'wt') as f:
                    for r in data_by_run[names[1]][kind]:
                        if r[key]!=115:f.write(json.dumps(r)+'\n')
            out2=Path(td)/'missing'
            result=subprocess.run(args+['--output',str(out2)],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            m=json.loads((out2/'PACKAGE_MANIFEST.json').read_text())
            self.assertEqual(m['missing_required_updates'],[115])
            self.assertEqual(m['missing_required_metrics'],[115])
            self.assertTrue(out2.with_suffix('.tar.gz').exists())


if __name__=='__main__':
    unittest.main()
