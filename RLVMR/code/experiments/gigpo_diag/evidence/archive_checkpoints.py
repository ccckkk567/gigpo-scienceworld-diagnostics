"""Archive the two completed ScienceWorld runs; verify public bytes before pruning.

No tensor deserialization, conversion, training imports or changes to rewards.
The persistent inventory supports restarting after an upload or partial cleanup.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import time

RUNS = (
    'scienceworld-qwen25-15b-gigpo-161254-20261014T023812Z',
    'scienceworld-qwen25-15b-gigpo-resume161254-20261014T161429Z',
)
REPO = 'Fyvoid/scienceworld-gigpo-qwen2.5-1.5b-checkpoints'
RETAIN_STEP = 150
REQUIRED = ['data.pt'] + [f'actor/{kind}_world_size_2_rank_{rank}.pt'
    for kind in ('model', 'optim', 'extra_state') for rank in range(2)]


def write_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)


def checked_path(root, relative):
    rel = PurePosixPath(relative)
    if rel.is_absolute() or '..' in rel.parts or '\\' in relative or ':' in relative:
        raise RuntimeError('Unsafe relative path: ' + relative)
    p = root.joinpath(*rel.parts)
    if root not in p.resolve().parents or any(x.is_symlink() for x in (p, *p.parents)):
        raise RuntimeError('Symlink or escaped path: ' + str(p))
    return p


def stamp(path):
    s = path.stat()
    if not path.is_file():
        raise RuntimeError('Not a regular file: ' + str(path))
    return {'size': s.st_size, 'mtime_ns': s.st_mtime_ns, 'inode': s.st_ino}


def files_in(cp, root):
    files = {}
    for p in sorted(cp.rglob('*')):
        checked_path(root, p.relative_to(root).as_posix())
        if p.is_file():
            files[p.relative_to(cp).as_posix()] = stamp(p)
        elif not p.is_dir():
            raise RuntimeError('Unsupported filesystem entry: ' + str(p))
    return files


def initial_plan(root):
    entries = []
    for run in RUNS:
        cpdir = checked_path(root, run + '/checkpoints')
        if not cpdir.is_dir():
            raise RuntimeError('Missing expected run: ' + run)
        for cp in sorted(cpdir.glob('global_step_*')):
            if not re.fullmatch(r'global_step_\d+', cp.name) or not cp.is_dir():
                raise RuntimeError('Unexpected checkpoint entry: ' + str(cp))
            step = int(cp.name.rsplit('_', 1)[1])
            if step > RETAIN_STEP:
                raise RuntimeError('Newer checkpoint exists; update archive scope first')
            content = files_in(cp, root)
            if not content:
                continue
            missing = [p for p in REQUIRED if content.get(p, {}).get('size', 0) <= 0]
            entries.append({'path': cp.relative_to(root).as_posix(), 'step': step,
                            'keep': step == RETAIN_STEP, 'files': content,
                            'missing_resume_files': missing})
    retained = [e for e in entries if e['keep']]
    if not retained or any(e['missing_resume_files'] for e in retained):
        raise RuntimeError('Step 150 is not structurally complete; delete nothing')
    # Latest first: back up the retained recovery point before deleting any old file.
    entries.sort(key=lambda e: (not e['keep'], e['step'], e['path']))
    return {'schema': 1, 'repo': REPO, 'root': str(root), 'entries': entries}


def validate_plan(plan, root):
    if plan.get('schema') != 1 or plan.get('repo') != REPO or plan.get('root') != str(root):
        raise RuntimeError('Archive state belongs to a different source/repository')
    for e in plan['entries']:
        parts = PurePosixPath(e['path']).parts
        if len(parts) != 3 or parts[0] not in RUNS or parts[1] != 'checkpoints':
            raise RuntimeError('Invalid checkpoint scope')
        if parts[2] != f"global_step_{e['step']}" or not 0 <= e['step'] <= RETAIN_STEP:
            raise RuntimeError('Invalid checkpoint step')
        if e['keep'] != (e['step'] == RETAIN_STEP):
            raise RuntimeError('Invalid retention policy')
        for name in e['files']:
            checked_path(root, e['path'] + '/' + name)
    retained = [e for e in plan['entries'] if e['keep']]
    if not retained or any(any(e['files'].get(p, {}).get('size', 0) <= 0 for p in REQUIRED) for e in retained):
        raise RuntimeError('Inventory does not contain a complete retained checkpoint')


def unchanged(root, entry, allow_missing=False):
    cp = checked_path(root, entry['path'])
    actual = files_in(cp, root) if cp.exists() else {}
    if set(actual) - set(entry['files']):
        raise RuntimeError('Checkpoint gained files: ' + entry['path'])
    for name, expected in entry['files'].items():
        if name not in actual and allow_missing:
            continue
        if actual.get(name) != {k: expected[k] for k in ('size', 'mtime_ns', 'inode')}:
            raise RuntimeError('Checkpoint changed or disappeared: ' + entry['path'] + '/' + name)


def guard_retained(root, plan):
    for run in RUNS:
        c = checked_path(root, run + '/checkpoints')
        for p in c.glob('global_step_*'):
            if re.fullmatch(r'global_step_\d+', p.name) and int(p.name.rsplit('_', 1)[1]) > RETAIN_STEP:
                raise RuntimeError('Newer checkpoint appeared; cleanup stopped')
    for e in plan['entries']:
        if e['keep']:
            unchanged(root, e)
            pointer = checked_path(root, str(PurePosixPath(e['path']).parent / 'latest_checkpointed_iteration.txt'))
            if pointer.read_text().strip() != str(RETAIN_STEP):
                raise RuntimeError('Retained checkpoint pointer changed')


def hash_file(path, expected):
    before = stamp(path)
    if before != {k: expected[k] for k in ('size', 'mtime_ns', 'inode')}:
        raise RuntimeError('Source changed before hashing')
    sha = hashlib.sha256()
    git = hashlib.sha1(f"blob {before['size']}\0".encode())
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            sha.update(block); git.update(block)
    if stamp(path) != before:
        raise RuntimeError('Source changed while hashing')
    return {'sha256': sha.hexdigest(), 'git_blob_sha1': git.hexdigest()}


def matches(remote, expected):
    if remote.size != expected['size']:
        return False
    if remote.lfs:
        return remote.lfs.sha256 == expected['sha256']
    return remote.blob_id == expected['git_blob_sha1']


class Hub:
    def __init__(self):
        from huggingface_hub import HfApi, get_token
        token = get_token()
        if not token:
            raise RuntimeError('HF_TOKEN is missing; no local files were deleted')
        self.api = HfApi(endpoint='https://huggingface.co', token=token)
        self.public = HfApi(endpoint='https://huggingface.co', token=False)
        user = self.api.whoami()
        if user.get('name') != 'Fyvoid':
            raise RuntimeError('HF_TOKEN is not for Fyvoid; stopping')
        if user.get('auth', {}).get('accessToken', {}).get('role') == 'read':
            raise RuntimeError('HF_TOKEN is read-only; configure a write token and resubmit')
        self.api.create_repo(REPO, repo_type='model', private=False, exist_ok=True)
        if self.api.repo_info(REPO).private:
            raise RuntimeError('Existing repository is private; refusing to change visibility')

    def head(self):
        return self.public.repo_info(REPO).sha

    def info(self, paths, revision):
        result = {}
        for start in range(0, len(paths), 100):
            for item in self.public.get_paths_info(REPO, paths[start:start+100], revision=revision):
                result[item.path] = item
        return result

    def upload(self, files, revision, message):
        from huggingface_hub import CommitOperationAdd
        return self.api.create_commit(REPO,
            operations=[CommitOperationAdd(path_in_repo=k, path_or_fileobj=str(v)) for k, v in files.items()],
            commit_message=message, parent_commit=revision, num_threads=2).oid


def publish(hub, sources, expected, message):
    revision = hub.head()
    found = hub.info(list(expected), revision)
    # Never silently replace a different existing checkpoint at the same path.
    for name, remote in found.items():
        if not matches(remote, expected[name]):
            raise RuntimeError('Remote file conflicts with source: ' + name)
    missing = {k: sources[k] for k in expected if k not in found}
    if missing:
        revision = hub.upload(missing, revision, message)
    verified = hub.info(list(expected), revision)
    if set(verified) != set(expected) or any(not matches(verified[k], v) for k, v in expected.items()):
        raise RuntimeError('Public content verification failed; no cleanup for this checkpoint')
    return revision


def run(root, state, hub, delete):
    state.mkdir(parents=True, exist_ok=True)
    plan_path = state / 'inventory.json'
    plan = json.loads(plan_path.read_text()) if plan_path.exists() else initial_plan(root)
    validate_plan(plan, root)
    guard_retained(root, plan)
    write_json(plan_path, plan)
    receipt_path = state / 'receipts.json'
    receipts = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    deleted = 0
    def event(**kw):
        data = {'time': time.time(), **kw}
        with (state / 'actions.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(data) + '\n'); f.flush(); os.fsync(f.fileno())
        print(json.dumps(data), flush=True)
    event(action='start', repo=REPO, retained=[e['path'] for e in plan['entries'] if e['keep']],
          delete_enabled=delete,
          planned_cleanup_GiB=sum(v['size'] for e in plan['entries'] if not e['keep']
                                  for v in e['files'].values()) / 2**30,
          inventoried_GiB=sum(v['size'] for e in plan['entries'] for v in e['files'].values()) / 2**30)
    readme = Path(__file__).with_name('HF_CHECKPOINT_README.md')
    readme_meta = stamp(readme); readme_meta.update(hash_file(readme, readme_meta))
    publish(hub, {'README.md': readme}, {'README.md': readme_meta}, 'Document native checkpoint archive')
    for e in plan['entries']:
        previously_verified = e['path'] in receipts
        unchanged(root, e, allow_missing=previously_verified and not e['keep'])
        for name, meta in e['files'].items():
            if 'sha256' not in meta:
                event(action='hash', checkpoint=e['path'], file=name, bytes=meta['size'])
                meta.update(hash_file(checked_path(root, e['path'] + '/' + name), meta))
                write_json(plan_path, plan)
        prefix = 'checkpoints/' + e['path']
        sources = {prefix + '/' + n: checked_path(root, e['path'] + '/' + n) for n in e['files']}
        expected = {prefix + '/' + n: v for n, v in e['files'].items()}
        event(action='upload_or_verify', checkpoint=e['path'])
        revision = publish(hub, sources, expected, 'Archive ' + e['path'])
        unchanged(root, e, allow_missing=previously_verified and not e['keep'])
        # Public per-checkpoint manifest excludes credentials and absolute HPC paths.
        manifest = {'source': e['path'], 'step': e['step'], 'retained_on_hpc': e['keep'],
                    'missing_resume_files': e['missing_resume_files'], 'data_revision': revision,
                    'files': {n: {k: v[k] for k in ('size', 'sha256', 'git_blob_sha1')} for n, v in e['files'].items()}}
        # A verified checkpoint must keep the same data revision in its immutable manifest on retries.
        mf = state / (e['path'].replace('/', '__') + '.json')
        if not mf.exists():
            write_json(mf, manifest)
        meta = stamp(mf); meta.update(hash_file(mf, meta))
        manifest_name = 'manifests/' + mf.name
        manifest_rev = publish(hub, {manifest_name: mf}, {manifest_name: meta}, 'Index checkpoint archive')
        receipts[e['path']] = {'data_revision': revision, 'manifest_revision': manifest_rev,
                               'public_hashes_verified': True, 'verified_at': time.time()}
        write_json(receipt_path, receipts)
        event(action='verified', checkpoint=e['path'], revision=revision, keep=e['keep'])
        if not delete or e['keep']:
            continue
        guard_retained(root, plan)
        unchanged(root, e, allow_missing=previously_verified)
        # Only unlink the inventoried, unchanged regular files. Never recursively remove a tree.
        for name, meta in e['files'].items():
            p = checked_path(root, e['path'] + '/' + name)
            if not p.exists():
                continue
            guard_retained(root, plan)
            if stamp(p) != {k: meta[k] for k in ('size', 'mtime_ns', 'inode')}:
                raise RuntimeError('File changed immediately before deletion')
            event(action='delete_intent', checkpoint=e['path'], file=name, bytes=meta['size'], revision=revision)
            p.unlink(); deleted += meta['size']
            event(action='deleted', checkpoint=e['path'], file=name, bytes=meta['size'])
        cp = checked_path(root, e['path'])
        if cp.exists():
            for d in sorted((p for p in cp.rglob('*') if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
                if not any(d.iterdir()): d.rmdir()
            if not any(cp.iterdir()): cp.rmdir()
    result = {'deleted_bytes_this_job': deleted, 'deleted_GiB_this_job': deleted / 2**30,
              'repo': 'https://huggingface.co/' + REPO,
              'retained': [e['path'] for e in plan['entries'] if e['keep']]}
    write_json(state / 'result.json', result)
    event(action='finished', **result)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--state', type=Path, required=True)
    ap.add_argument('--delete', action='store_true')
    args = ap.parse_args()
    root = args.root.resolve(strict=True); state = args.state.resolve()
    if root == state or root in state.parents:
        raise RuntimeError('Archive state must be outside the source logs')
    state.mkdir(parents=True, exist_ok=True)
    import fcntl  # CPU-node execution only; no simultaneous archive jobs.
    with (state / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(root, state, Hub(), args.delete)


if __name__ == '__main__':
    main()
