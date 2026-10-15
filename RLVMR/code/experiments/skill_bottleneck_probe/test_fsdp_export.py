"""CPU regression of the actual native merger, with synthetic two-rank input shards."""
import ast
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import fsdp_export

NATIVE = Path(__file__).resolve().parents[2]


class ExportTests(unittest.TestCase):
    def test_only_megatron_import_moves_and_fsdp_ast_is_unchanged(self):
        source=(NATIVE/'scripts/model_merger.py').read_text(encoding='utf-8')
        before=ast.parse(source); after=ast.parse(fsdp_export.lazy_megatron_import(source))
        moved=next(n for n in before.body if isinstance(n,ast.ImportFrom) and n.module=='verl.utils.megatron_utils')
        before.body.remove(moved)
        megatron=next(n for n in before.body if isinstance(n,ast.FunctionDef) and n.name=='convert_megatron_checkpoints_to_hfmodes')
        megatron.body.insert(0,moved)
        self.assertEqual(ast.dump(before),ast.dump(after))

    def test_native_fsdp_merge_without_megatron_exports_exact_bfloat16_tensors(self):
        import torch
        import torch.distributed._tensor as dt
        from transformers import Qwen2Config, Qwen2ForCausalLM
        from safetensors.torch import load_file
        config=Qwen2Config(vocab_size=16,hidden_size=8,intermediate_size=16,num_hidden_layers=1,
                          num_attention_heads=2,num_key_value_heads=2,tie_word_embeddings=False)
        model=Qwen2ForCausalLM(config)
        state={k:torch.arange(v.numel(),dtype=torch.float32).reshape(v.shape)/100 for k,v in model.state_dict().items()}
        expected={k:v.bfloat16() for k,v in state.items()}
        class SyntheticDTensor:
            def __init__(self,value,placement):
                self._local_tensor=value; self.placements=(placement,)
                self.device_mesh=SimpleNamespace(mesh=torch.tensor([0,1]),mesh_dim_names=('fsdp',))
        class BlockMegatron:
            def find_spec(self,fullname,*args):
                if fullname.startswith(('megatron','verl.utils.megatron_utils')):
                    raise AssertionError('FSDP must not import Megatron')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); base=root/'base'; source=root/'actor'; source.mkdir()
            config.architectures=['Qwen2ForCausalLM']; config.save_pretrained(base)
            for rank in range(2): (source/f'model_world_size_2_rank_{rank}.pt').write_bytes(b'fixture')
            def load(path,**kwargs):
                rank=int(Path(path).stem.rsplit('_',1)[1]); result={}
                for k,v in state.items():
                    placement=dt.Replicate() if v.ndim==1 else dt.Shard(0)
                    local=v.clone() if placement.is_replicate() else v.chunk(2,dim=0)[rank].clone()
                    result[k]=SyntheticDTensor(local,placement)
                return result
            args=['fsdp_export.py','--native-code',str(NATIVE),'--hf-model-path',str(base),
                  '--local-dir',str(source),'--target-dir',str(root/'export'),'--metadata',str(root/'metadata.json')]
            source_hash=fsdp_export.hashlib.sha256((NATIVE/'scripts/model_merger.py').read_bytes()).hexdigest()
            with patch.object(sys,'argv',args),patch.object(sys,'meta_path',[BlockMegatron(),*sys.meta_path]), \
                 patch.object(dt,'DTensor',SyntheticDTensor),patch.object(torch,'load',side_effect=load), \
                 contextlib.redirect_stdout(io.StringIO()):
                fsdp_export.main()
            actual=load_file(root/'export/model.safetensors')
            self.assertEqual(set(actual),set(expected))
            for k in expected: torch.testing.assert_close(actual[k],expected[k],rtol=0,atol=0)
            self.assertEqual(source_hash,fsdp_export.hashlib.sha256((NATIVE/'scripts/model_merger.py').read_bytes()).hexdigest())
            self.assertEqual(json.loads((root/'metadata.json').read_text())['backend'],'fsdp')


if __name__=='__main__': unittest.main()
