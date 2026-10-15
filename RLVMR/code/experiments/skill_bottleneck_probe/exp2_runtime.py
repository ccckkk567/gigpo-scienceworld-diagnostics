"""Exp2 sequential execution on one loaded model, with read-only evidence reuse."""
import gzip
import json
import traceback
from collections import defaultdict
from pathlib import Path
from exp2_support import (VERSION, PILOT_COMMIT, trial_key, enrich, natural_prefix,
                          select_natural, branch_trials)
from study import save, fingerprint

PARTS=('episodes.jsonl.gz','steps.jsonl.gz','decisions.jsonl.gz','resets.jsonl.gz','replay_audit.jsonl',
       'natural_prefix_candidates.jsonl.gz','prefix_capture_errors.jsonl')


def read_records(path, errors=None):
    rows=[]
    if not path.exists(): return rows
    try:
        opener=gzip.open if path.suffix=='.gz' else open
        with opener(path,'rt',encoding='utf-8') as f:
            for line in f:
                try: rows.append(json.loads(line))
                except ValueError as exc:
                    if errors is None: raise
                    errors.append(dict(file=str(path),error=str(exc)))
    except (OSError,EOFError,ValueError) as exc:
        if errors is None: raise
        errors.append(dict(file=str(path),error=str(exc)))
    return rows


def write_records(path,rows):
    if not rows: return
    opener=gzip.open if path.suffix=='.gz' else open
    with opener(path,'wt',encoding='utf-8') as f:
        for row in rows: f.write(json.dumps(row,ensure_ascii=False)+'\n')


def complete_evidence(parts, prefix):
    eps=parts.get('episodes.jsonl.gz',[]); steps=parts.get('steps.jsonl.gz',[])
    if len(eps)!=1 or eps[0].get('status')!='completed': return False
    if len(parts.get('resets.jsonl.gz',[]))!=1: return False
    if [r.get('step_index') for r in steps]!=list(range(eps[0].get('total_policy_budget_used',-1))): return False
    policy=[r for r in steps if r.get('phase')=='policy']
    decisions=parts.get('decisions.jsonl.gz',[])
    if len(policy)!=eps[0].get('num_policy_steps') or len(decisions)!=len(policy): return False
    if not all(all(k in r for k in ('raw_response','executed_action','observation_after','score_after','done_after',
                                   'rendered_prompt','prompt_token_ids','native_info')) for r in steps): return False
    if any(d.get('step_index')!=s['step_index'] or d.get('rendered_prompt')!=s['rendered_prompt'] or
           d.get('prompt_token_ids')!=s['prompt_token_ids'] for d,s in zip(decisions,policy)): return False
    return not prefix or len(parts.get('replay_audit.jsonl',[]))==1


class EvidenceSource:
    def __init__(self,root,cid,current_spec):
        self.root=Path(root); self.cid=cid; self.cache={}; self.errors=[]; self.file_hashes={}
        self.top=json.loads((self.root/'RUN_MANIFEST.json').read_text())
        self.identity=json.loads((self.root/cid/'run_manifest.json').read_text())
        if self.identity.get('checkpoint_load_status')!='loaded': raise ValueError('Source model was not loaded')
        if (self.identity.get('actual_loaded_model_path')!=self.identity.get('inference_model')
                or not self.identity.get('inference_model_fingerprints')):
            raise ValueError('Source loaded-model identity is incomplete or inconsistent')
        self.pilot=self.top.get('git_commit')==PILOT_COMMIT and self.top.get('protocol_version')=='skill-bottleneck-v1.1-routing'
        if not self.pilot:
            if self.top.get('protocol_version')!=VERSION: raise ValueError('Unrecognized source protocol/code')
            if self.top.get('execution_source_sha256')!=current_spec['execution_source_sha256']:
                raise ValueError('Exp2 execution code changed; do not silently reuse previous results')
        self.old_protocol=self.identity['protocol']
        self.prefixes={p['prefix_id']:p for p in self.identity['prefixes']}
        self.old_specs={p['trial_id']:p for p in self.identity['trials']}
        if not self.pilot:
            for p in read_records(self.root/cid/'onpolicy_prefixes.jsonl'):
                self.prefixes[p['prefix_id']]=p
        if self.pilot:
            # One pass per source file/checkpoint, not one full scan per reused trial.
            for name in PARTS:
                path=self.root/cid/name
                if not path.exists(): continue
                before=path.stat(); rows=read_records(path,self.errors); after=path.stat()
                if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
                    raise ValueError('Source evidence changed during read')
                grouped=defaultdict(list)
                for row in rows: grouped[row['trial_id']].append(row)
                self.cache[name]=grouped
                self.file_hashes[name]=fingerprint([path])[0]

    def attempt(self,item,prefix,key,target):
        tid=item['trial_id']
        if self.pilot:
            old_item=self.old_specs.get(tid)
            if old_item is None: return False,'trial_not_in_source'
            parts={name:group.get(tid,[]) for name,group in self.cache.items()}
            hashes=self.file_hashes
        else:
            folder=self.root/self.cid/'trials'/tid
            if not (folder/'trial_spec.json').exists(): return False,'trial_not_in_source'
            if any((folder/name).exists() for name in ('import_failure.json','execution_failure.json')):
                return False,'source_trial_has_execution_or_import_failure'
            old_item=json.loads((folder/'trial_spec.json').read_text())
            parts={name:read_records(folder/name,self.errors) for name in PARTS}
            hashes={name:fingerprint([folder/name])[0] for name in PARTS if (folder/name).exists()}
        old_prefix=self.prefixes.get(old_item.get('prefix_id'))
        if trial_key(old_item,self.old_protocol,old_prefix,self.identity)!=key: return False,'compatibility_key_differs'
        if not complete_evidence(parts,prefix): return False,'source_raw_evidence_incomplete_or_execution_not_completed'
        # Preserve original rows, run IDs, responses and prompts; annotate provenance separately.
        for name,rows in parts.items(): write_records(target/name,rows)
        save(target/'evidence_manifest.json',dict(trial_id=tid,compatibility_key=key,reused=True,
            source_run_id=self.top.get('experiment_id',self.root.name),source_trial_id=tid,
            source_output=str(self.root),source_git_commit=self.top['git_commit'],source_files=hashes,
            source_read_errors=self.errors,execution_status='reused',original_execution_status='completed'))
        return True,'compatible_complete_trial'


def candidate_from_trial(folder,item,protocol):
    existing=read_records(folder/'natural_prefix_candidates.jsonl.gz',[])
    if existing: return existing[0]
    # Compatible Pilot N has the next decision's real, unhinted rendered input.
    episodes=read_records(folder/'episodes.jsonl.gz',[])
    steps=read_records(folder/'steps.jsonl.gz',[])
    if not episodes: return None
    index=next((r['step_index'] for r in steps if r.get('A_transition_event') is True and not r.get('done_after')),None)
    if index is None: return None
    decision=next((r for r in read_records(folder/'decisions.jsonl.gz',[]) if r['step_index']==index+1),None)
    resets=read_records(folder/'resets.jsonl.gz',[])
    if not decision or not resets: return None
    return natural_prefix({**item,'run_id':episodes[0]['run_id']},steps[:index+1],resets[0]['info'],
                          decision['rendered_prompt'],decision['prompt_token_ids'],protocol)


def run_trials(spec,renderer,llm,out,jar,manifest,execute):
    sources=[]; reuse_notes=[]; prefix_map={p['prefix_id']:p for p in spec['prefixes']}
    for source in spec.get('reuse_sources',[]):
        try: sources.append(EvidenceSource(source,spec['checkpoint_id'],spec))
        except Exception as exc: reuse_notes.append(dict(source=source,error=f'{type(exc).__name__}: {exc}',action='run new trials'))
    protocol=spec['protocol']; counts=defaultdict(int); candidates=[]

    def run_one(item,prefix):
        target=out/'trials'/item['trial_id']; target.mkdir(parents=True,exist_ok=False)
        save(target/'trial_spec.json',item)
        key=trial_key(item,protocol,prefix,manifest); reused=False
        for source in sources:
            try:
                reused,reason=source.attempt(item,prefix,key,target)
                if reason!='trial_not_in_source': reuse_notes.append(dict(trial_id=item['trial_id'],source=str(source.root),reason=reason))
                if reused: break
            except Exception as exc:
                # Validate before copying; unexpected copy failure is a local evidence error.
                if (target/'episodes.jsonl.gz').exists():
                    save(target/'import_failure.json',dict(error=str(exc)))
                    counts['import_failed']+=1
                    return None
                reuse_notes.append(dict(trial_id=item['trial_id'],source=str(source.root),error=str(exc)))
        if not reused:
            try: execute(item,protocol,prefix,renderer,llm,target,jar['label_semantics_verified'])
            except Exception as exc:
                failure=dict(**item,status='error',termination_reason='uncaught_execution_error',
                             error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
                save(target/'execution_failure.json',failure)
                if not (target/'episodes.jsonl.gz').exists(): write_records(target/'episodes.jsonl.gz',[failure])
        episodes=read_records(target/'episodes.jsonl.gz',[])
        episode=episodes[0] if len(episodes)==1 else None
        if not reused:
            save(target/'evidence_manifest.json',dict(trial_id=item['trial_id'],compatibility_key=key,reused=False,
                source_run_id=spec['experiment_id'],source_trial_id=item['trial_id'],source_output=str(out.parent),
                execution_status=episode.get('status','interrupted') if episode else 'interrupted'))
        if episode:
            counts['reused' if reused else episode['status']]+=1
            save(target/'analysis_fields.json',enrich(episode,read_records(target/'steps.jsonl.gz',[])))
        save(out/'progress.json',dict(counts=dict(counts),last_trial=item['trial_id']))
        return episode

    for item in spec['trials']:
        episode=run_one(item,prefix_map.get(item['prefix_id']))
        if item['condition']=='D' and item['prefix_id']==protocol['primary_prefixes'][0] and item['sample_seed']==protocol['sampling_seeds'][0]:
            # Actual D is a pipeline smoke check, not a success-rate gate.
            save(out/'smoke_status.json',dict(trial_id=item['trial_id'],
                status='executed' if episode and episode.get('status')=='completed' else 'partial',
                replay_quality=episode.get('state_match_quality') if episode else None,
                B_success=episode.get('B_confirmed_completed') if episode else None,success_not_required=True))
        if item['condition']=='N':
            try:
                candidate=candidate_from_trial(out/'trials'/item['trial_id'],item,protocol)
                if candidate: candidates.append(candidate)
            except Exception as exc: reuse_notes.append(dict(trial_id=item['trial_id'],prefix_capture_error=str(exc)))
    selected=select_natural(candidates,spec['onpolicy_prefix_cap'])
    write_records(out/'natural_prefix_candidates.jsonl',candidates)
    write_records(out/'onpolicy_prefixes.jsonl',selected)
    branch=branch_trials(protocol,selected,spec['checkpoint_id'],spec['experiment_id'])
    save(out/'branch_selection.json',dict(status='selected' if selected else 'not_triggered',
        candidates=len(candidates),selected_prefix_ids=[p['prefix_id'] for p in selected],planned_trials=len(branch),
        rule='first source per variation sorted by variation/sample_seed/trial_id; u110 cap2, u150 fills remaining global cap4; no outcome filter',
        trials=branch))
    for item in branch:
        run_one(item,next(p for p in selected if p['prefix_id']==item['prefix_id']))
    save(out/'reuse_audit.json',reuse_notes)
    manifest.update(exp2_execution_counts=dict(counts),onpolicy_prefixes=len(selected),onpolicy_planned_trials=len(branch))
