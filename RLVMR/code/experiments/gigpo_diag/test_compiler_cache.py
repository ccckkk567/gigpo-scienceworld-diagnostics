"""Cache placement and subprocess propagation; no GPU compilation or rollouts."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

spec = importlib.util.spec_from_file_location('compiler_cache', Path(__file__).with_name('compiler_cache.py'))
cache = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cache)


class CompilerCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.env = mock.patch.dict(os.environ, {
            'GIGPO_COMPILER_CACHE_ROOT': str(self.root / 'node-local' / 'compiler-cache'),
            'GIGPO_RUN_DIR': str(self.root / 'scratch' / 'run'),
        })
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_unconfigured_recipe_does_not_override_existing_caches(self):
        os.environ.pop('GIGPO_COMPILER_CACHE_ROOT')
        before = dict(os.environ)
        self.assertEqual(cache.worker_compiler_cache_env('actor_0'), {})
        self.assertEqual(dict(os.environ), before)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_workers_and_groups_are_isolated_and_records_match(self):
        names = ['groupAWorkerDict_0:0', 'groupAWorkerDict_0:1', 'groupBWorkerDict_0:0']
        envs = [cache.worker_compiler_cache_env(name) for name in names]
        paths = [path for env in envs for path in env.values()]
        self.assertEqual(len(set(paths)), 6)
        for path in paths:
            self.assertTrue(Path(path).is_dir())
            self.assertTrue(Path(path).is_relative_to(self.root / 'node-local'))
        records = [json.loads(p.read_text()) for p in (self.root / 'scratch/run/compiler_cache_workers').glob('*.json')]
        self.assertEqual({r['worker_name'] for r in records}, set(names))
        self.assertEqual({r['worker_name']: r['env_vars'] for r in records}, dict(zip(names, envs)))

    def test_reconfiguration_keeps_existing_cache_and_parent_environment(self):
        before = dict(os.environ)
        first = cache.worker_compiler_cache_env('actor_0')
        artifact = Path(first['TRITON_CACHE_DIR']) / '__triton_launcher.so'
        artifact.write_bytes(b'fixture')
        self.assertEqual(cache.worker_compiler_cache_env('actor_0'), first)
        self.assertEqual(artifact.read_bytes(), b'fixture')
        self.assertEqual(dict(os.environ), before)

    def test_child_processes_use_separate_caches_despite_inherited_defaults(self):
        # Same kernel key, separate processes: one worker's cache publication
        # cannot replace another's. Exercise OS environment propagation.
        code = """import json,os
from pathlib import Path
p=Path(os.environ['TRITON_CACHE_DIR'])/'same-kernel'
p.mkdir(parents=True,exist_ok=True)
(p/'__triton_launcher.so').write_text(os.environ['TEST_WORKER'])
print(json.dumps({k:os.environ[k] for k in ('TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR','TMPDIR')}))
"""
        processes = []
        envs = []
        for name in ('actor_0', 'actor_1'):
            env = dict(os.environ, TORCHINDUCTOR_CACHE_DIR='old/shared', TRITON_CACHE_DIR='old/shared',
                       TMPDIR=str(self.root / 'scratch'), TEST_WORKER=name)
            override = cache.worker_compiler_cache_env(name)
            env.update(override)
            envs.append(override)
            processes.append(subprocess.Popen([sys.executable, '-c', code], env=env, stdout=subprocess.PIPE,
                                               stderr=subprocess.PIPE, text=True))
        for index, process in enumerate(processes):
            stdout, stderr = process.communicate(timeout=20)
            self.assertEqual(process.returncode, 0, stderr)
            self.assertEqual(json.loads(stdout), dict(envs[index], TMPDIR=str(self.root / 'scratch')))
            artifact = Path(envs[index]['TRITON_CACHE_DIR']) / 'same-kernel/__triton_launcher.so'
            self.assertEqual(artifact.read_text(), f'actor_{index}')

    def test_relative_root_is_rejected(self):
        os.environ['GIGPO_COMPILER_CACHE_ROOT'] = 'relative/path'
        with self.assertRaisesRegex(ValueError, 'absolute'):
            cache.worker_compiler_cache_env('actor_0')

    def test_unwritable_cache_fails_before_worker_creation(self):
        blocked = self.root / 'file-not-directory'
        blocked.write_text('fixture')
        os.environ['GIGPO_COMPILER_CACHE_ROOT'] = str(blocked)
        with self.assertRaises(OSError):
            cache.worker_compiler_cache_env('actor_0')


if __name__ == '__main__':
    unittest.main()
