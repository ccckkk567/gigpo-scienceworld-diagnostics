"""Inventory and pilot entrypoint. Source trees/checkpoints are read-only."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
RUNS=['scienceworld-qwen25-15b-gigpo-161254-20261014T023812Z',
      'scienceworld-qwen25-15b-gigpo-resume161254-20261014T161429Z']
REVISION='989aa7980e4cf806f80c7fef2b1adb7bc71aa306'


def save(path, data): path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''): h.update(block)
    return h.hexdigest()


def fingerprint(paths):
    result=[]
    for path in paths:
        before=path.stat(); digest=sha(path); after=path.stat()
        if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns): raise RuntimeError(f'Source changed during read: {path}')
        result.append(dict(path=str(path),bytes=after.st_size,mtime_ns=after.st_mtime_ns,sha256=digest))
    return result


def git(path, *args):
    return subprocess.check_output(['git','-C',str(path),*args],text=True).strip()


def checkpoint_inventory(root):
    records=[]
    for run in RUNS:
        folder=root/'logs/gigpo-reproduction/scienceworld'/run
        manifest=json.loads((folder/'RUN_MANIFEST.json').read_text())
        cp=folder/'checkpoints'; pointer=cp/'latest_checkpointed_iteration.txt'
        last=int(pointer.read_text().strip()) if pointer.exists() else None
        for p in sorted(cp.glob('global_step_*')):
            try: step=int(p.name.rsplit('_',1)[1])
            except ValueError: continue
            models=[p/'actor'/f'model_world_size_2_rank_{i}.pt' for i in range(2)]
            files_ok=all(f.is_file() and f.stat().st_size>1024**2 for f in models)
            lineage_ok=(step<=110 if run==RUNS[0] else 110<step<=150)
            records.append(dict(checkpoint_id=f'u{step}',step=step,path=str(p),origin_run=run,
                source_code_commit=manifest.get('code_commit'),lineage_eligible=lineage_ok,
                model_shards_present=files_ok,latest_pointer=last,
                eligible=files_ok and lineage_ok and last is not None and step<=last,
                availability_scope='file inventory only; actual deserialization occurs in pilot',
                optimizers_required=False))
    return records


def select(requested, inventory):
    target=int(requested[1:]); available=[r for r in inventory if r['eligible']]
    if not available: return None
    row=min(available,key=lambda r:(abs(r['step']-target),r['step']))
    return {**row,'requested_checkpoint':requested,
        'substitution_reason':None if row['step']==target else 'requested point absent/incomplete; nearest eligible lineage checkpoint by step (lower breaks ties)'}


def validate_study_pool(protocol, prefixes, pool):
    task_num=protocol['task_num']; task_name=protocol['task_id']
    for p in prefixes:
        receipt=p.get('source_load_receipt',{})
        expected=dict(task_id=task_num,task_name=task_name,variation_id=p['variation_id'],
                      reset_seed=p['reset_seed'],simplifications=protocol['simplifications'],load_reset_returned=True)
        if p.get('task_num')!=task_num or p['task_id']!=task_name or any(receipt.get(k)!=v for k,v in expected.items()):
            raise ValueError(f"Study routing disagrees with original load receipt for prefix {p['prefix_id']}: "
                             f"requested task_num={task_num}, recorded task_num={receipt.get('task_id')}")
    chosen=protocol['natural_variations']+protocol.get('reserve_variations',[])+[p['variation_id'] for p in prefixes]
    missing=sorted({v for v in chosen if [task_num,v] not in pool})
    if missing: raise ValueError(f'Study task_num={task_num} ({task_name}) variations outside original train pool: {missing}')


def trials(protocol, prefixes, checkpoint, run_id):
    result=[]
    for v in protocol['natural_variations']:
        for seed in protocol['sampling_seeds']:
            result.append(dict(condition='N',variation_id=v,prefix_id=None,reset_seed=20261015+v,sample_seed=seed))
    for p in prefixes:
        if p['prefix_id'] not in protocol['primary_prefixes']: continue
        for seed in protocol['sampling_seeds']:
            for condition in ('P','H'):
                result.append(dict(condition=condition,variation_id=p['variation_id'],prefix_id=p['prefix_id'],
                    reset_seed=p['reset_seed'],sample_seed=seed))
    for r in result:
        key=f"{checkpoint}-{r['condition']}-{r['prefix_id'] or r['variation_id']}-{r['sample_seed']}"
        r.update(trial_id=key,run_id=run_id,checkpoint_id=checkpoint,task_id=protocol['task_id'],
            prompt_protocol_id=protocol['protocol_version'],dataset_role='train_diagnostic',
            simplification_profile=protocol['simplifications'],max_steps=protocol['max_policy_steps'])
    return result


def csv_write(path, rows):
    if not rows: return
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r))); w.writeheader(); w.writerows(rows)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('command',choices=['inventory','run'])
    ap.add_argument('--root',type=Path,required=True); ap.add_argument('--native-code',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True); ap.add_argument('--work',type=Path)
    args=ap.parse_args(); root=args.root.resolve(); out=args.output.resolve(); native=args.native_code.resolve()
    # Prevent outputs from targeting either source tree or any checkpoint/log directory.
    for protected in (native,root/'logs',root/'data',root/'envs'):
        if out==protected or protected in out.parents: raise ValueError('Output overlaps protected source')
    if out.exists(): raise ValueError('Use a new output directory; no overwrite or implicit resume')
    out.mkdir(parents=True)
    protocol=json.loads((HERE/'data/protocol.json').read_text())
    prefixes=[json.loads(s) for s in (HERE/'data/prefixes.jsonl').read_text(encoding='utf-8').splitlines()]
    identity=json.loads((HERE/'native_identity.json').read_text())
    manifest=dict(experiment_id=out.name,protocol_version=protocol['protocol_version'],project='ScienceWorld GiGPO skill probe',
        git_commit=git(HERE,'rev-parse','HEAD'),native_git_commit=git(native,'rev-parse','HEAD'),
        base_model_revision=REVISION,tokenizer_revision=REVISION,model_family='Qwen2.5-1.5B-Instruct',
        start_time=time.time(),status='inventory',errors_summary=[],task_manifest_hash=sha(HERE/'data/protocol.json'),
        dataset_role='train_diagnostic',simplifications='easy',source_policy='161254 <=110;162461 111..150',
        checkpoints=[],protocol=protocol)
    save(out/'RUN_MANIFEST.json',manifest)
    try:
        for file in ('protocol.json','prefixes.jsonl','selection_audit.json','inventory_report.md','training_background.csv','A_stage_advantages.csv'):
            (out/file).write_bytes((HERE/'data'/file).read_bytes())
        for file in ('labels_spec.md','CONFIG_DIFF.md'):
            (out/file).write_bytes((HERE/file).read_bytes())
        sources={p:hashlib.sha256((native/p).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for p in identity['source_sha256_lf']}
        manifest['source_sha256_lf']=sources
        if sources!=identity['source_sha256_lf']: raise ValueError('Native source differs from locked training semantics; review this study configuration')
        pool=json.loads((native/'agent_system/environments/env_package/sciworld/variations_idx/L0_idx.json').read_text())['train']
        validate_study_pool(protocol,prefixes,pool)
        inv=checkpoint_inventory(root); csv_write(out/'checkpoint_inventory.csv',inv)
        base=root/'data/models/Qwen2.5-1.5B-Instruct-989aa798'
        checkpoint_specs=[dict(checkpoint_id='base',requested_checkpoint='base',path=str(base),origin_run='pretrained')]
        checkpoint_specs += [select(c,inv) or dict(checkpoint_id=c,requested_checkpoint=c,path=None,origin_run=None) for c in protocol['checkpoint_ids'][1:]]
        manifest['checkpoints']=checkpoint_specs
        csv_write(out/'study_manifest.csv',[r for c in checkpoint_specs for r in trials(protocol,prefixes,c['checkpoint_id'],out.name)])
        with (out/'inventory_report.md').open('a',encoding='utf-8') as f:
            f.write('\n\n## HPC file inventory\n\n'+json.dumps(checkpoint_specs,indent=2)+'\n\nAvailability is not proof of successful model loading.\n')
        save(out/'RUN_MANIFEST.json',manifest)
        if args.command=='inventory': manifest['status']='inventory_complete'; return
        if args.work is None: raise ValueError('--work must be a fresh node-local directory')
        work=args.work.resolve()
        if work==root or root in work.parents: raise ValueError('Export work directory must be node-local, outside shared source root')
        work.mkdir(parents=True,exist_ok=False)
        manifest['status']='running'; env=os.environ.copy(); env['PYTHONPATH']=str(native)
        env['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'  # Trusted own FSDP files; native merger rank-0 call has no explicit flag.
        (out/'reproduce_commands.md').write_text('# Actual invocation\n\n'+repr(sys.argv)+'\n',encoding='utf-8')
        used=set()
        for c in checkpoint_specs:
            cid=c['checkpoint_id']; target=out/cid
            if cid in used:
                c['status']='duplicate_substitution_not_rerun'; continue
            used.add(cid); target.mkdir()
            try:
                if not c['path']: raise FileNotFoundError('No eligible model checkpoint')
                path=Path(c['path'])
                model_files=sorted(path.glob('*.safetensors')) if cid=='base' else [path/'actor'/f'model_world_size_2_rank_{i}.pt' for i in range(2)]
                if not model_files: raise FileNotFoundError('Model weights absent')
                c['source_model_fingerprints']=fingerprint(model_files)
                exported=base if cid=='base' else work/cid
                if cid!='base':
                    cmd=[sys.executable,str(native/'scripts/model_merger.py'),'--backend','fsdp','--hf_model_path',str(base),
                        '--local_dir',str(path/'actor'),'--target_dir',str(exported)]
                    with (target/'conversion.log').open('w') as log: subprocess.run(cmd,env=env,cwd=target,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=900)
                c['inference_model_fingerprints']=fingerprint(sorted(exported.glob('*.safetensors')))
                if not c['inference_model_fingerprints']: raise FileNotFoundError('Export has no model weights')
                spec=dict(checkpoint_id=cid,checkpoint_path=c['path'],checkpoint_origin_run=c['origin_run'],
                    source_model_fingerprints=c['source_model_fingerprints'],inference_model_fingerprints=c['inference_model_fingerprints'],
                    experiment_id=out.name,git_commit=manifest['git_commit'],base_model_revision=REVISION,tokenizer_revision=REVISION,
                    base_model=str(base),inference_model=str(exported),output=str(target),protocol=protocol,
                    prefixes=prefixes,native_identity=identity,trials=trials(protocol,prefixes,cid,out.name),
                    placeholder=str(root/'data/gigpo-reproduction/scienceworld/placeholders/text/train.parquet'))
                save(target/'invocation.json',spec)
                cmd=[sys.executable,str(HERE/'runner.py'),'--spec',str(target/'invocation.json')]
                with (target/'inference.log').open('w') as log: result=subprocess.run(cmd,env=env,cwd=target,stdout=log,stderr=subprocess.STDOUT)
                c.update(status='finished' if result.returncode==0 else 'failed',exit_code=result.returncode)
            except Exception as exc:
                c.update(status='failed',error=f'{type(exc).__name__}: {exc}'); save(target/'failure.json',c)
                manifest['errors_summary'].append(dict(checkpoint=cid,error=c['error']))
            save(out/'RUN_MANIFEST.json',manifest)
        manifest['status']='pilot_finished_review_per_checkpoint_and_trial_status'
        subprocess.run([sys.executable,str(HERE/'analyze.py'),str(out)],check=True)
    except Exception as exc:
        manifest['status']='failed'; manifest['errors_summary'].append(str(exc)); raise
    finally:
        manifest['end_time']=time.time(); save(out/'RUN_MANIFEST.json',manifest)


if __name__=='__main__': main()
