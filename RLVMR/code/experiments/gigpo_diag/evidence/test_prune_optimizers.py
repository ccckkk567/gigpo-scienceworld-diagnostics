import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('prune',Path(__file__).with_name('prune_optimizers.py'))
prune=importlib.util.module_from_spec(spec); spec.loader.exec_module(prune)


class CleanupTests(unittest.TestCase):
    def test_only_old_optimizers_removed_across_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'logs'; root.mkdir()
            def create(env,run,step,pointer=True):
                c=root/env/(env+'-qwen25-15b-gigpo-'+run)/'checkpoints'
                p=c/f'global_step_{step}'; (p/'actor').mkdir(parents=True)
                (p/'data.pt').write_bytes(b'data')
                for kind in ('model','optim','extra_state'):
                    for rank in range(2):
                        (p/'actor'/f'{kind}_world_size_2_rank_{rank}.pt').write_bytes(b'fixture')
                if pointer:
                    (c/'latest_checkpointed_iteration.txt').write_text(str(step))
                return p
            old=create('alfworld','old',10)
            latest=create('alfworld','resume',20)
            inflight=create('alfworld','resume',30,False)
            sci=create('scienceworld','one',10)
            # AppWorld is never a cleanup target, even if it has similarly named files.
            app=create('appworld','one',10)
            subprocess.run([sys.executable,str(Path(__file__).with_name('prune_optimizers.py')),'--root',str(root),'--audit',str(Path(tmp)/'audit'),'--delete'],check=True,stdout=subprocess.DEVNULL)
            self.assertFalse(list((old/'actor').glob('optim_*')))
            self.assertEqual(len(list((old/'actor').glob('model_*'))),2)
            self.assertEqual(len(list((old/'actor').glob('extra_state_*'))),2)
            for p in (latest,inflight,sci,app):
                self.assertEqual(len(list((p/'actor').glob('optim_*'))),2)

    def test_missing_pointer_does_not_authorize_deletion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            p=root/'scienceworld'/'scienceworld-qwen25-15b-gigpo-one'/'checkpoints'/'global_step_10'/'actor'
            p.mkdir(parents=True)
            (p/'optim_world_size_1_rank_0.pt').write_bytes(b'fixture')
            self.assertEqual(prune.plan(root,'scienceworld')[1],[])


if __name__=='__main__':
    unittest.main()
