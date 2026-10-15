import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import archive_checkpoints as a


class FakeHub:
    """Content-addressed Hub fixture; revisions are immutable snapshots."""
    def __init__(self):
        self.revisions = {'0': {}}
        self.upload_calls = 0
        self.corrupt = None
        self.after_upload = None

    def head(self):
        return str(len(self.revisions) - 1)

    def info(self, paths, revision):
        found = {}
        for path in paths:
            data = self.revisions[revision].get(path)
            if data is None: continue
            sha = hashlib.sha256(data).hexdigest()
            blob = hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()
            lfs = SimpleNamespace(sha256=sha) if path.endswith('.pt') else None
            found[path] = SimpleNamespace(path=path, size=len(data), lfs=lfs, blob_id=blob)
        return found

    def upload(self, files, revision, message):
        self.upload_calls += 1
        if revision != self.head(): raise RuntimeError('Parent changed')
        snapshot = self.revisions[revision].copy()
        for name, path in files.items():
            data = Path(path).read_bytes()
            if self.corrupt and self.corrupt in name: data = b'x' * len(data)
            snapshot[name] = data
        new = str(len(self.revisions)); self.revisions[new] = snapshot
        if self.after_upload: self.after_upload(files)
        return new


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name, 'scienceworld'); self.root.mkdir()
        self.state = Path(self.tmp.name, 'audit')
        self.old = self.create(a.RUNS[0], 110)
        self.partial = self.create(a.RUNS[0], 120, partial=True)
        self.latest = self.create(a.RUNS[1], 150)
        self.hub = FakeHub()

    def tearDown(self): self.tmp.cleanup()

    def create(self, run, step, partial=False):
        cp = self.root / run / 'checkpoints' / f'global_step_{step}'
        cp.mkdir(parents=True)
        names = ['actor/model_world_size_2_rank_1.pt'] if partial else a.REQUIRED + ['environment_sampler.json']
        for name in names:
            p = cp / name; p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes((f'{step}/{name}: fixture').encode())
        if not partial:
            (cp.parent / 'latest_checkpointed_iteration.txt').write_text(str(step))
        return cp

    def run_archive(self, delete=True):
        with contextlib.redirect_stdout(io.StringIO()):
            a.run(self.root.resolve(), self.state, self.hub, delete)

    def test_upload_verify_delete_retains_latest_and_restart_skips_upload(self):
        # Historical optimizer cleanup is supported, without inventing resumability.
        for p in (self.old / 'actor').glob('optim_*'): p.unlink()
        original = {str(p.relative_to(self.root)): p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        self.run_archive()
        self.assertFalse(self.old.exists()); self.assertFalse(self.partial.exists())
        for name in a.REQUIRED: self.assertTrue((self.latest / name).is_file())
        archived = self.hub.revisions[self.hub.head()]
        for path, data in original.items():
            if '/global_step_' in path:
                self.assertEqual(archived['checkpoints/' + path], data)
        plan = json.loads((self.state / 'inventory.json').read_text())
        self.assertTrue(next(e for e in plan['entries'] if e['step'] == 110)['missing_resume_files'])
        calls = self.hub.upload_calls
        self.run_archive()
        self.assertEqual(calls, self.hub.upload_calls)

    def test_same_size_remote_corruption_stops_deletion(self):
        self.hub.corrupt = 'global_step_110/'
        with self.assertRaisesRegex(RuntimeError, 'verification failed'): self.run_archive()
        self.assertTrue((self.old / 'data.pt').exists())
        self.assertTrue(self.partial.exists()); self.assertTrue(self.latest.exists())

    def test_changed_source_after_upload_stops_deletion(self):
        def change(files):
            if any('global_step_110/' in p for p in files):
                (self.old / 'data.pt').write_bytes(b'changed source')
        self.hub.after_upload = change
        with self.assertRaisesRegex(RuntimeError, 'changed or disappeared'): self.run_archive()
        self.assertTrue((self.old / 'data.pt').exists())

    def test_interrupted_cleanup_can_resume_from_verified_receipt(self):
        self.run_archive(delete=False)
        (self.old / 'data.pt').unlink()  # interruption after unlink, before final audit event
        calls = self.hub.upload_calls
        self.run_archive()
        self.assertEqual(calls, self.hub.upload_calls)
        self.assertFalse(self.old.exists()); self.assertTrue(self.latest.exists())

    def test_missing_retained_optimizer_blocks_all_uploads_and_cleanup(self):
        (self.latest / 'actor/optim_world_size_2_rank_0.pt').unlink()
        with self.assertRaisesRegex(RuntimeError, 'not structurally complete'): self.run_archive()
        self.assertEqual(self.hub.upload_calls, 0); self.assertTrue(self.old.exists())

    def test_retained_changes_after_verified_backup_blocks_old_cleanup(self):
        def change(files):
            if any('global_step_110/' in p for p in files):
                (self.latest / 'data.pt').write_bytes(b'changed latest')
        self.hub.after_upload = change
        with self.assertRaisesRegex(RuntimeError, 'changed or disappeared'): self.run_archive()
        self.assertTrue(self.old.exists())

    def test_different_remote_content_never_overwritten(self):
        self.run_archive(delete=False)
        path = next(p for p in self.hub.revisions[self.hub.head()] if 'global_step_110/' in p)
        self.hub.revisions[self.hub.head()][path] = b'conflicting remote'
        calls = self.hub.upload_calls
        with self.assertRaisesRegex(RuntimeError, 'conflicts'): self.run_archive()
        self.assertEqual(calls, self.hub.upload_calls); self.assertTrue(self.old.exists())

    def test_scope_excludes_unrelated_runs_and_rejects_escape(self):
        other = self.root / 'unrelated' / 'checkpoints/global_step_10/data.pt'
        other.parent.mkdir(parents=True); other.write_bytes(b'untouched')
        self.run_archive()
        self.assertEqual(other.read_bytes(), b'untouched')
        with self.assertRaisesRegex(RuntimeError, 'Unsafe relative'): a.checked_path(self.root, '../outside')


if __name__ == '__main__': unittest.main()
