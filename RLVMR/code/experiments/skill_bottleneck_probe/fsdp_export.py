"""Run the locked native FSDP merger without its unused Megatron dependency."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
IMPORT = 'from verl.utils.megatron_utils import get_model_checkpoint_path, get_hf_model_checkpoint_path'
MEGATRON = 'def convert_megatron_checkpoints_to_hfmodes():\n'


def lazy_megatron_import(source):
    if source.count(IMPORT) != 1 or source.count(MEGATRON) != 1:
        raise ValueError('Native merger layout changed; adapter needs review')
    # Keep every FSDP operation unchanged; the import is needed only by the other backend.
    return source.replace(IMPORT + '\n', '').replace(MEGATRON, MEGATRON + '    ' + IMPORT + '\n')


def main():
    ap = argparse.ArgumentParser()
    for name in ('native-code', 'hf-model-path', 'local-dir', 'target-dir', 'metadata'):
        ap.add_argument('--' + name, type=Path, required=True)
    args = ap.parse_args()
    path = args.native_code / 'scripts/model_merger.py'
    source = path.read_text(encoding='utf-8')
    digest = hashlib.sha256(source.encode()).hexdigest()
    expected = json.loads((HERE / 'native_identity.json').read_text())['source_sha256_lf']['scripts/model_merger.py']
    if digest != expected:
        raise ValueError('Native merger differs from reviewed source')
    patched = lazy_megatron_import(source)
    args.metadata.write_text(json.dumps(dict(native_merger=str(path), native_sha256_lf=digest,
        adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        executed_source_sha256=hashlib.sha256(patched.encode()).hexdigest(),
        change='Move Megatron-only import inside convert_megatron_checkpoints_to_hfmodes; FSDP body unchanged',
        backend='fsdp', source_checkpoint=str(args.local_dir), target=str(args.target_dir)), indent=2), encoding='utf-8')
    sys.argv = [str(path), '--backend', 'fsdp', '--hf_model_path', str(args.hf_model_path),
                '--local_dir', str(args.local_dir), '--target_dir', str(args.target_dir)]
    exec(compile(patched, str(path), 'exec'), {'__name__': '__main__', '__file__': str(path)})


if __name__ == '__main__':
    main()
