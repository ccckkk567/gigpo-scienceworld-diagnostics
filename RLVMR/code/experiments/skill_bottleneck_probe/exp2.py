"""Existing-checkpoint Exp2 entrypoint; no training and no source mutations."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from study import HERE, RUNS, REVISION, save, sha, git, fingerprint, csv_write, validate_study_pool
from exp2_support import VERSION, make_trials

EXECUTION_FILES=('runner.py','native.py','labels.py','replay.py','fsdp_export.py','study.py',
                 'exp2.py','exp2_runtime.py','exp2_support.py')


def inventory(root):
    rows=[]
    for cid,run in zip(('u110','u150'),RUNS):
        folder=root/'logs/gigpo-reproduction/scienceworld'/run
        path=folder/'checkpoints'/f'global_step_{cid[1:]}'
        row=dict(checkpoint_id=cid,requested_checkpoint=cid,path=str(path),origin_run=run,eligible=False,
                 optimizers_required=False,substitution_reason=None)
        try:
            manifest=json.loads((folder/'RUN_MANIFEST.json').read_text())
            row['source_code_commit']=manifest.get('code_commit')
            row['latest_pointer']=int((folder/'checkpoints/latest_checkpointed_iteration.txt').read_text().strip())
            row['model_shards_present']=all((path/'actor'/f'model_world_size_2_rank_{i}.pt').is_file() and
                (path/'actor'/f'model_world_size_2_rank_{i}.pt').stat().st_size>1024**2 for i in range(2))
            row['eligible']=row['model_shards_present'] and row['latest_pointer']>=int(cid[1:])
            if not row['eligible']: row['error']='Required exact checkpoint missing/incomplete; no substitution'
        except Exception as exc: row['error']=f'{type(exc).__name__}: {exc}'
        rows.append(row)
    return rows


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,required=True);ap.add_argument('--native-code',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--work',type=Path,required=True)
    ap.add_argument('--reuse-pilot',type=Path);ap.add_argument('--resume-from',type=Path)
    args=ap.parse_args();root=args.root.resolve();native=args.native_code.resolve();out=args.output.resolve();work=args.work.resolve()
    for protected in (root/'logs',root/'data',root/'envs',root/'code',native,HERE):
        if out==protected or protected in out.parents: raise ValueError('Output overlaps protected source')
    if out.exists(): raise ValueError('Use a new output directory; --resume-from imports completed trials read-only')
    if work==root or root in work.parents: raise ValueError('Model exports must be node-local, outside shared root')
    out.mkdir(parents=True);work.mkdir(parents=True,exist_ok=False)
    protocol=json.loads((HERE/'data/protocol_v2.json').read_text())
    all_prefixes=[json.loads(s) for s in (HERE/'data/prefixes.jsonl').read_text(encoding='utf-8').splitlines()]
    prefixes=[next(p for p in all_prefixes if p['prefix_id']==pid) for pid in protocol['primary_prefixes']]
    identity=json.loads((HERE/'native_identity.json').read_text())
    execution={name:hashlib.sha256((HERE/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for name in EXECUTION_FILES}
    manifest=dict(experiment_id=out.name,git_commit=git(HERE,'rev-parse','HEAD'),
        branch=git(HERE,'branch','--show-current'),native_git_commit=git(native,'rev-parse','HEAD'),
        protocol_version=VERSION,protocol=protocol,execution_source_sha256=execution,
        base_model_revision=REVISION,tokenizer_revision=REVISION,dataset_role='train_diagnostic',
        status='inventory',start_time=time.time(),expected_core_trials=96,maximum_branch_trials=16,
        errors_summary=[],checkpoints=[],reuse_sources=[str(p.resolve()) for p in (args.resume_from,args.reuse_pilot) if p])
    save(out/'RUN_MANIFEST.json',manifest)
    try:
        for source,destination in [('data/protocol_v2.json','protocol.json'),('data/prefixes.jsonl','prefix_manifest.jsonl'),
                                   ('PROTOCOL_V2.md','PROTOCOL_V2.md'),('labels_spec.md','labels_spec.md'),('CONFIG_DIFF_V2.md','CONFIG_DIFF.md')]:
            shutil.copyfile(HERE/source,out/destination)
        snapshot=out/'source_code';snapshot.mkdir()
        for name in (*EXECUTION_FILES,'exp2_analyze.py','exp2.sbatch'): shutil.copyfile(HERE/name,snapshot/name)
        (snapshot/'data').mkdir()
        for name in ('protocol_v2.json','prefixes.jsonl'): shutil.copyfile(HERE/'data'/name,snapshot/'data'/name)
        for name in ('native_identity.json','PROTOCOL_V2.md','CONFIG_DIFF_V2.md','labels_spec.md'):
            shutil.copyfile(HERE/name,snapshot/name)
        sources={p:hashlib.sha256((native/p).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for p in identity['source_sha256_lf']}
        manifest['source_sha256_lf']=sources
        if sources!=identity['source_sha256_lf']: raise ValueError('Native execution code differs from locked Pilot semantics')
        pool=json.loads((native/'agent_system/environments/env_package/sciworld/variations_idx/L0_idx.json').read_text())['train']
        validate_study_pool(protocol,prefixes,pool)
        checkpoints=inventory(root);manifest['checkpoints']=checkpoints
        csv_write(out/'checkpoint_inventory.csv',checkpoints)
        csv_write(out/'study_manifest.csv',[r for c in checkpoints for r in make_trials(protocol,prefixes,c['checkpoint_id'],out.name)])
        csv_write(out/'variation_manifest.csv',[dict(variation_id=v,task_num=18,task_id=protocol['task_id'],
            dataset_role='train_diagnostic',sampling_seeds=json.dumps(protocol['sampling_seeds'])) for v in protocol['natural_variations']])
        (out/'REPRODUCE.md').write_text('# Actual invocation\n\n```text\n'+repr(sys.argv)+'\n```\n\n'
            'Resume into a NEW output using --resume-from OLD. Only complete, identity-compatible trial evidence is imported. '
            'Original outputs and weights are never modified. Trial execution status and task success are distinct.\n',encoding='utf-8')
        env=os.environ.copy();env['PYTHONPATH']=str(native);env['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'
        base=root/'data/models/Qwen2.5-1.5B-Instruct-989aa798'
        remaining_prefixes=protocol['onpolicy_max_prefixes'];manifest['status']='running'
        for c in checkpoints:
            cid=c['checkpoint_id'];target=out/cid;target.mkdir()
            try:
                if not c['eligible']: raise FileNotFoundError(c.get('error','Checkpoint unavailable'))
                path=Path(c['path']);c['source_model_fingerprints']=fingerprint([path/'actor'/f'model_world_size_2_rank_{i}.pt' for i in range(2)])
                exported=work/cid
                cmd=[sys.executable,str(HERE/'fsdp_export.py'),'--native-code',str(native),'--hf-model-path',str(base),
                     '--local-dir',str(path/'actor'),'--target-dir',str(exported),'--metadata',str(target/'conversion_manifest.json')]
                with (target/'conversion.log').open('w') as log:
                    subprocess.run(cmd,env=env,cwd=target,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=900)
                c['inference_model_fingerprints']=fingerprint(sorted(exported.glob('*.safetensors')))
                if not c['inference_model_fingerprints']: raise FileNotFoundError('Export has no model weights')
                spec=dict(exp2=True,checkpoint_id=cid,checkpoint_path=c['path'],checkpoint_origin_run=c['origin_run'],
                    source_model_fingerprints=c['source_model_fingerprints'],inference_model_fingerprints=c['inference_model_fingerprints'],
                    experiment_id=out.name,git_commit=manifest['git_commit'],base_model_revision=REVISION,tokenizer_revision=REVISION,
                    base_model=str(base),inference_model=str(exported),output=str(target),protocol=protocol,prefixes=prefixes,
                    native_identity=identity,execution_source_sha256=execution,trials=make_trials(protocol,prefixes,cid,out.name),
                    reuse_sources=manifest['reuse_sources'],
                    onpolicy_prefix_cap=min(remaining_prefixes,protocol['onpolicy_u110_max_prefixes'] if cid=='u110' else remaining_prefixes),
                    placeholder=str(root/'data/gigpo-reproduction/scienceworld/placeholders/text/train.parquet'))
                save(target/'invocation.json',spec);save(out/'RUN_MANIFEST.json',manifest)
                with (target/'inference.log').open('w') as log:
                    result=subprocess.run([sys.executable,str(HERE/'runner.py'),'--spec',str(target/'invocation.json')],
                                          env=env,cwd=target,stdout=log,stderr=subprocess.STDOUT)
                c.update(status='finished' if result.returncode==0 else 'failed',exit_code=result.returncode)
                if result.returncode: raise RuntimeError(f'Inference exited {result.returncode}; see {cid}/inference.log')
            except Exception as exc:
                c.update(status='failed',error=f'{type(exc).__name__}: {exc}');save(target/'failure.json',c)
                manifest['errors_summary'].append(dict(checkpoint=cid,error=c['error']))
            selection=target/'branch_selection.json'
            if selection.exists(): remaining_prefixes-=len(json.loads(selection.read_text())['selected_prefix_ids'])
            save(out/'RUN_MANIFEST.json',manifest)
        from exp2_analyze import analyze
        summary=analyze(out,plots=True)
        complete=summary['core_completed']==96 and summary['branch_completed']==summary['branch_planned']
        complete=complete and all(c['status']=='finished' for c in checkpoints) and not summary['read_errors']
        manifest['status']='exp2_complete' if complete else 'exp2_partial'
        manifest['summary']=summary;save(out/'RUN_MANIFEST.json',manifest)
        print(json.dumps(dict(run_status=manifest['status'],**summary),ensure_ascii=False),flush=True)
        return 0 if complete else 2
    except Exception as exc:
        manifest['status']='failed';manifest['errors_summary'].append(f'{type(exc).__name__}: {exc}');raise
    finally:
        manifest['end_time']=time.time();save(out/'RUN_MANIFEST.json',manifest)


if __name__=='__main__': sys.exit(main())
