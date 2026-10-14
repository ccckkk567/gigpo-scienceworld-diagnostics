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


if __name__=='__main__':
    unittest.main()
