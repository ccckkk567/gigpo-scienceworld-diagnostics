"""Compiler-cache placement for this single-node recipe; no model operations."""
import hashlib
import json
import os
from pathlib import Path


def worker_compiler_cache_env(worker_name):
    """Set each Ray actor's cache paths before it imports Torch or Triton."""
    root = os.environ.get('GIGPO_COMPILER_CACHE_ROOT')
    if not root:
        return {}
    root = Path(root)
    if not root.is_absolute():
        raise ValueError('GIGPO_COMPILER_CACHE_ROOT must be an absolute path')
    # Include the complete actor name (group and rank), not CUDA device index:
    # Ray can expose a different physical GPU as device 0 in each process.
    identity = hashlib.sha256(worker_name.encode('utf-8')).hexdigest()
    directory = root.resolve() / 'workers' / identity
    env = {
        'TORCHINDUCTOR_CACHE_DIR': str(directory / 'torchinductor'),
        'TRITON_CACHE_DIR': str(directory / 'triton'),
    }
    for path in env.values():
        Path(path).mkdir(parents=True, exist_ok=True)
    run_dir = os.environ.get('GIGPO_RUN_DIR')
    if run_dir:
        records = Path(run_dir) / 'compiler_cache_workers'
        records.mkdir(parents=True, exist_ok=True)
        (records / f'{identity}.json').write_text(
            json.dumps({'worker_name': worker_name, 'env_vars': env}, indent=2),
            encoding='utf-8',
        )
    return env
