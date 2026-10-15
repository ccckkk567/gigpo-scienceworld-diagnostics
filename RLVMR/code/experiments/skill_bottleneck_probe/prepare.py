"""Read the existing three-part evidence package; freeze an outcome-independent pilot protocol."""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import zipfile
from labels import TASK, HINT, VERSION, transition, update_of


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def rows(z, suffix):
    name = next(n for n in z.namelist() if n.endswith(suffix))
    with gzip.open(z.open(name), 'rt', encoding='utf-8') as f:
        for line in f: yield json.loads(line)


def dump_csv(path, records):
    records = list(records)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in records for k in r)))
        w.writeheader(); w.writerows(records)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--archives', type=Path, required=True)
    ap.add_argument('--native-code', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    z = zipfile.ZipFile(args.archives / '01-analysis-and-selected-records.zip')
    episodes = [r for r in rows(z, '/records/episodes.jsonl.gz') if r['task_id'] == TASK and r['sample_kind'] == 'train_rollout']
    eps = {(r['run_id'], r['traj_uid']): r for r in episodes}
    groups=defaultdict(list)
    for ep in episodes: groups[ep['run_id'],ep['episode_group_uid']].append(ep)
    numeric = defaultdict(list)
    for r in rows(z, '/records/steps_numeric.jsonl.gz'):
        if (r['run_id'], r['traj_uid']) in eps: numeric[r['run_id'], r['traj_uid']].append(r)
    aevents = {}; stages = defaultdict(Counter)
    for key, ep in eps.items():
        u = update_of(ep); stage = next(s for s, end in [('1-35',35), ('36-75',75), ('76-110',110), ('111-150',150)] if u <= end)
        stages[stage]['episodes'] += 1; stages[stage]['strict_success'] += ep.get('strict_success') is True
        for r in sorted(numeric[key], key=lambda r:r['step_index']):
            lab = transition(r['command_short'], r['feedback_short'], r['score_pre'], r['score_post'], r['done_raw'])
            if lab['A_transition_event']:
                aevents[key] = r; stages[stage]['A_reached'] += 1; break
    ids = {(r['run_id'], r['analysis_step_id']): r for r in aevents.values()}
    advantages = {}
    for r in rows(z, '/records/advantages.jsonl.gz'):
        key = (r['run_id'],r['analysis_step_id'])
        if key in ids:
            if key in advantages and any(advantages[key].get(k) != r.get(k) for k in ('AE','AS','joint')):
                raise ValueError('Conflicting duplicate advantage rows')
            advantages[key] = r
    advantage_rows=[]
    for key,r in aevents.items():
        ep=eps[key]; group=groups[ep['run_id'],ep['episode_group_uid']]
        complete=len(group)==8 and all(type(e.get('strict_success')) is bool for e in group)
        returns=[e.get('env_reward_sum') for e in group]
        advantage_rows.append(dict(run_id=r['run_id'],analysis_step_id=r['analysis_step_id'],update=update_of(r),
            variation_id=r['variation_id'],command=r['command_short'],score_pre=r['score_pre'],score_post=r['score_post'],
            episode_group_uid=ep['episode_group_uid'],group_episode_count=len(group),
            group_strict_all_fail=not any(e['strict_success'] for e in group) if complete else None,
            group_environment_return_varies=len(set(returns))>1 if len(group)==8 and all(x is not None for x in returns) else None,
            environment_return_note='env_reward_sum only; not discounted reward with format penalty',
            **{k:advantages.get((r['run_id'],r['analysis_step_id']),{}).get(k) for k in ('AE','AS','joint')}))
    dump_csv(args.out / 'A_stage_advantages.csv',advantage_rows)
    dump_csv(args.out / 'training_background.csv', [dict(stage=s, **{k:v[k] for k in ('episodes','A_reached','strict_success')}) for s,v in stages.items()])
    # Read only full, preselected episode groups; no selection on B outcomes.
    full = zipfile.ZipFile(args.archives / '02-complete-trajectory-groups.zip')
    candidates = []
    for name in sorted(n for n in full.namelist() if n.endswith('/group_status.json')):
        status = json.loads(full.read(name)); folder = name.rsplit('/',1)[0]
        if not all(status.get(k) is True for k in ('complete_episode_group','complete_text_group','update_completed')): continue
        group_eps = list(rows(full, folder + '/episodes.jsonl.gz'))
        keys = {(e['run_id'],e['traj_uid']) for e in group_eps} & set(aevents)
        if not keys: continue
        detailed = defaultdict(list)
        for r in rows(full, folder + '/detailed_steps.jsonl.gz'):
            key = (r['run_id'],r['traj_uid'])
            if key in keys: detailed[key].append(r)
        for key, rr in detailed.items():
            rr.sort(key=lambda r:r['step_index']); stop = aevents[key]['step_index']
            prefix = [r for r in rr if r['step_index'] <= stop]
            boundary = next((r for r in rr if r['step_index'] == stop+1), None)
            if [r['step_index'] for r in prefix] != list(range(stop+1)) or boundary is None or stop+1 >= 30: continue
            ep = eps[key]
            ident = hashlib.sha256(('/'.join(key) + '/' + str(stop)).encode()).hexdigest()[:16]
            # Keep only pre-A decisions plus the next decision INPUT. Never future responses.
            candidates.append(dict(prefix_id=ident, source_run_id=key[0], source_trajectory_id=key[1],
                source_step_id=aevents[key]['analysis_step_id'], source_policy_version=ep['policy_version'],
                source_checkpoint=None, source_checkpoint_note='exact rollout policy; not assumed to be a saved checkpoint',
                source_update=update_of(ep), task_id=TASK, variation_id=ep['variation_id'], reset_seed=ep['reset_seed'],
                split='train_diagnostic', step_count_at_A=stop+1, remaining_policy_steps=30-stop-1,
                score_before_A=prefix[-1]['score_pre'], score_after_A=prefix[-1]['score_post'],
                source_termination_status=prefix[-1]['done_raw'], A_event_evidence=aevents[key],
                B_not_yet_completed_evidence='native score <100 and done=false after verified A focus',
                full_executed_action_prefix=[r['executed_command'] for r in prefix],
                source_steps=[{k:r.get(k) for k in ('analysis_step_id','step_index','raw_response','executed_command',
                    'rendered_prompt','observation_post','score_pre','score_post','done_raw','native_info')} for r in prefix],
                post_A_rendered_prompt=boundary['rendered_prompt'], post_A_prompt_token_ids=boundary['prompt_token_ids'],
                group_status=status, source_archive=Path(full.filename).name, source_group=folder,
                prefix_selection_rule='earliest update, variation, trajectory; distinct variations first; no filtering on B',
                replay_status='not_run'))
    candidates.sort(key=lambda p:(p['source_update'],p['variation_id'],p['source_trajectory_id']))
    selected=[]; seen=set()
    for p in candidates:
        if p['variation_id'] not in seen: selected.append(p); seen.add(p['variation_id'])
        if len(selected)==4: break
    for i,p in enumerate(selected): p['selection_role']='pilot' if i<2 else 'reserve'
    with (args.out / 'prefixes.jsonl').open('w',encoding='utf-8') as f:
        for p in selected: f.write(json.dumps(p,ensure_ascii=False)+'\n')
    # Runtime adapter checks the canonical task table. Offline task number comes from recorded reset receipts where possible.
    task_table=json.loads((args.native_code / 'agent_system/environments/env_package/sciworld/ScienceWorld/scienceworld/tasks.json').read_text())
    task_num=next(int(k) for k,v in task_table.items() if isinstance(v,dict) and v.get('task_name')==TASK) if isinstance(task_table,dict) else next(i for i,v in enumerate(task_table) if v['task_name']==TASK)
    pool=json.loads((args.native_code / 'agent_system/environments/env_package/sciworld/variations_idx/L0_idx.json').read_text())['train']
    variations=sorted({int(v) for t,v in pool if int(t)==task_num})
    if len(variations)<4: raise ValueError('Not enough training variations')
    protocol=dict(protocol_version='skill-bottleneck-v1',task_id=TASK,task_num=task_num,
        natural_variations=variations[:4], reserve_variations=variations[4:8],
        variation_selection='lowest four IDs in the original L0 training pool, fixed before evaluation',
        sampling_seeds=[20261015,20261016], max_policy_steps=30, env_step_limit=100, simplifications='easy',
        dataset_role='train_diagnostic', checkpoint_ids=['base','u50','u110','u150'],
        temperature=0.4, top_p=1.0, top_k=-1, max_response_tokens=1024, max_prompt_tokens=6000,
        stage_hint=HINT,label_version=VERSION, primary_prefixes=[p['prefix_id'] for p in selected[:2]],
        reserve_prefixes=[p['prefix_id'] for p in selected[2:]],
        paired_primary_quality=['verified_match'], no_weight_or_optimizer_writes=True)
    protocol['hint_placement'] = 'append to original user content on each H continuation decision only'
    protocol['natural_reset_seed_rule'] = '20261015 + variation_id; shared across checkpoints and samples'
    protocol['generation_seed_rule'] = 'sample_seed + 1009 * absolute_policy_step_index'
    protocol['resource_adaptation'] = 'one GPU, tensor_parallel_size=1, eager inference, sequential trials'
    write_json(args.out / 'protocol.json',protocol)
    write_json(args.out / 'selection_audit.json',dict(task_episodes=len(episodes),A_events=len(aevents),
        full_text_prefix_candidates=len(candidates),selected_prefixes=len(selected),
        advantages_found=len(advantages),source_lineage='161254 updates 1..110;162461 updates111..150',
        candidate_ids=[p['prefix_id'] for p in candidates], training_stages=stages))
    for name in ['checkpoint_index.csv','PACKAGE_MANIFEST.json']:
        n=next(n for n in z.namelist() if n.endswith('/'+name)); (args.out/name).write_bytes(z.read(n))
    (args.out/'inventory_report.md').write_text(
        '# Local evidence inventory (not an HPC execution report)\n\n'
        f'Task: `{TASK}`. The merged mainline has {len(episodes)} target episodes, {len(aevents)} '
        f'evidence-supported A events, and {len(advantages)} linked historical advantage records. '
        f'{len(candidates)} prefixes have contiguous full text through A and the next decision input.\n\n'
        'Lineage: job 161254 updates 1–110; resumed job 162461 updates 111–150. '
        'Updates beyond 110 from the abandoned original branch of training are excluded.\n\n'
        'Pilot checkpoints: base, u50, u110, u150. Model files are on HPC; the package only contains their index. '
        'No checkpoint has been loaded locally for this study. No study rollout has run.\n\n'
        'Natural variations are 0,1,2,3, the first four original L0 train IDs. Primary historical prefixes '
        'use variations 40 and 55, selected chronologically before evaluation, without filtering on future B. '
        'N and P therefore use different variation sets; their aggregate gap is descriptive, not a matched causal comparison. '
        'P/H pairing requires verified replay and equal recorded boundary evidence.\n\n'
        'Historical evidence is sampled by complete episode group. Only ten candidate prefixes have full text, '
        'not all 58 A events. Exact rollout policy versions are retained; a saved checkpoint at that update is not assumed. '
        'Replay, current checkpoint availability, loaded model identity, and conditional B ability remain unverified.\n', encoding='utf-8')
    with (args.out/'inventory_report.md').open('a',encoding='utf-8') as f:
        f.write('\nHistorical A-step advantage nonzero counts (exact recorded value != 0): '+
            json.dumps({k:sum(r[k]!=0 for r in advantage_rows if r[k] is not None) for k in ('AE','AS','joint')})+
            f'. Matched A events in verified eight-episode all-failure groups: {sum(r["group_strict_all_fail"] is True for r in advantage_rows)}/{len(advantage_rows)}. '+
            'These observations describe the original sampled training groups, and do not prove why a later checkpoint succeeds or fails.\n')
    print(json.dumps({'out':str(args.out),'episodes':len(episodes),'A':len(aevents),'candidate_prefixes':len(candidates),
                      'selected':[(p['prefix_id'],p['variation_id'],p['source_update']) for p in selected],
                      'natural_variations':protocol['natural_variations']}))


if __name__=='__main__': main()
