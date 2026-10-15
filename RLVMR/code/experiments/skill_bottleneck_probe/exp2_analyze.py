"""Read actual Exp2 evidence, retaining exclusions, unknowns and separate origins."""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys
from exp2_runtime import read_records, write_records
from exp2_support import enrich, make_trials
from replay import paired_eligible
from study import save, csv_write

CONDITIONS=('P','H','D')
CHECKPOINTS=('u110','u150')


def metric(rows,key):
    known=[r for r in rows if type(r.get(key)) is bool]
    return dict(numerator=sum(r[key] for r in known),denominator=len(known),unknown=len(rows)-len(known),
                rate=sum(r[key] for r in known)/len(known) if known else None)


def attach(row,name,values,key):
    row.update({name+'_'+k:v for k,v in metric(values,key).items()})


def collect(root,errors):
    result=[]; steps={}; audits=[]; seen=set()
    for folder in sorted(root.glob('u*/trials/*')):
        if not (folder/'trial_spec.json').exists(): continue
        spec=json.loads((folder/'trial_spec.json').read_text())
        ep=read_records(folder/'episodes.jsonl.gz',errors)
        if len(ep)!=1:
            errors.append(dict(trial_id=spec['trial_id'],reason='missing_or_multiple_episode_rows',rows=len(ep)))
            ep=[dict(status='interrupted',B_confirmed_completed=None,A_reached=None)]
        if spec['trial_id'] in seen: raise ValueError('Duplicate logical trial ID')
        seen.add(spec['trial_id'])
        record=ep[0]; rr=read_records(folder/'steps.jsonl.gz',errors)
        provenance=json.loads((folder/'evidence_manifest.json').read_text()) if (folder/'evidence_manifest.json').exists() else {}
        row={**record,**enrich(record,rr),'trial_id':spec['trial_id'],
             'checkpoint_id':spec['checkpoint_id'],'condition':spec['condition'],
             'variation_id':spec['variation_id'],'sample_seed':spec['sample_seed'],
             'prefix_id':spec.get('prefix_id'),'prefix_origin':spec['prefix_origin'],
             'origin_natural_trial_id':spec.get('origin_natural_trial_id'),
             'analysis_run_id':root.name,'reused':provenance.get('reused',False),
             'source_run_id':provenance.get('source_run_id',record.get('run_id')),
             'source_trial_id':provenance.get('source_trial_id',spec['trial_id']),
             'compatibility_key':provenance.get('compatibility_key'),
             'evidence_ref':folder.relative_to(root).as_posix()}
        if (folder/'import_failure.json').exists(): row['status']='import_failed'
        if (folder/'execution_failure.json').exists(): row['status']='error'
        result.append(row);steps[row['trial_id']]=rr
        for audit in read_records(folder/'replay_audit.jsonl',errors):
            audits.append({**audit,'prefix_origin':spec['prefix_origin'],'evidence_ref':row['evidence_ref']})
    return result,steps,audits


def matched_rows(planned,lookup,conditions):
    groups=defaultdict(dict)
    for p in planned: groups[p['checkpoint_id'],p['prefix_id'],p['sample_seed']][p['condition']]=p
    rows=[]
    for (cid,pid,seed),items in groups.items():
        actual={cond:lookup.get(items.get(cond,{}).get('trial_id'),{}) for cond in conditions}
        row=dict(checkpoint=cid,prefix_id=pid,sample_seed=seed,variation_id=next(iter(items.values()))['variation_id'])
        hashes=[r.get('boundary_evidence_hash') for r in actual.values()]
        row['same_boundary']=len(hashes)==len(conditions) and None not in hashes and len(set(hashes))==1
        row['all_conditions_eligible']=all(paired_eligible(actual[conditions[0]],actual[c]) for c in conditions[1:])
        for cond,r in actual.items():
            row.update({cond+'_success':r.get('B_confirmed_completed'),cond+'_first_action':r.get('first_postA_action'),
                        cond+'_first_class':r.get('first_postA_action_class'),cond+'_replay_quality':r.get('state_match_quality'),
                        cond+'_execution_status':r.get('status','not_run'),cond+'_reused':r.get('reused'),
                        cond+'_source_run_id':r.get('source_run_id'),cond+'_source_trial_id':r.get('source_trial_id'),
                        cond+'_response_prompt_ref':r.get('evidence_ref','')+'/steps.jsonl.gz' if r else None})
        for left,right in [('P','H'),('P','D'),('H','D')]:
            if left not in conditions or right not in conditions: continue
            l,r=actual[left],actual[right];ok=paired_eligible(l,r)
            row[f'{left}_{right}_eligible']=ok
            row[f'{left}_to_{right}_flip']=(f"{int(l['B_confirmed_completed'])}->{int(r['B_confirmed_completed'])}" if ok else None)
        rows.append(row)
    return rows


def analyze(root,plots=False):
    root=Path(root);errors=[]
    manifest=json.loads((root/'RUN_MANIFEST.json').read_text());protocol=manifest['protocol']
    planned=[]
    if (root/'study_manifest.csv').exists():
        with (root/'study_manifest.csv').open(encoding='utf-8',newline='') as f: planned=list(csv.DictReader(f))
        for r in planned:
            r['variation_id']=int(r['variation_id']);r['sample_seed']=int(r['sample_seed'])
    elif (root/'prefix_manifest.jsonl').exists():
        prefixes=read_records(root/'prefix_manifest.jsonl',errors)
        planned=[r for cid in CHECKPOINTS for r in make_trials(protocol,prefixes,cid,root.name)]
    rows,steps,audits=collect(root,errors);lookup={r['trial_id']:r for r in rows}
    write_records(root/'episodes.jsonl.gz',rows)
    write_records(root/'replay_audit.jsonl.gz',audits)
    csv_write(root/'steps_index.csv',[dict(trial_id=r['trial_id'],steps=len(steps[r['trial_id']]),
                                         path=r['evidence_ref']+'/steps.jsonl.gz') for r in rows] or [dict(status='not_run')])
    historical=[r for r in rows if r['prefix_origin']=='historical'];natural=[r for r in rows if r['condition']=='N']
    post=[]
    for cid in CHECKPOINTS:
        for pid in protocol['primary_prefixes']:
            for cond in CONDITIONS:
                target=[p for p in planned if p['checkpoint_id']==cid and p.get('prefix_id')==pid and p['condition']==cond]
                rr=[r for r in historical if r['checkpoint_id']==cid and r['prefix_id']==pid and r['condition']==cond]
                valid=[r for r in rr if r.get('status')=='completed' and r.get('state_match_quality')=='verified_match']
                row=dict(checkpoint=cid,prefix_id=pid,variation_id=target[0]['variation_id'] if target else None,condition=cond,
                    planned=len(target),recorded=len(rr),verified_completed=len(valid),excluded=len(rr)-len(valid),
                    not_run=len(target)-len(rr),reused=sum(r['reused'] for r in rr),
                    quality_counts=json.dumps(Counter(r.get('state_match_quality') for r in rr),sort_keys=True))
                for name,key in [('B','B_confirmed_completed'),('first_B','B_first_continuation_action_correct'),
                                 ('repeat_A','repeat_A_after_A'),('wrong_focus','wrong_focus_after_A')]: attach(row,name,valid,key)
                post.append(row)
    csv_write(root/'postA_by_prefix.csv',post)
    aggregate=[]
    for cid in CHECKPOINTS:
        for cond in CONDITIONS:
            rr=[r for r in post if r['checkpoint']==cid and r['condition']==cond];known=[r for r in rr if r['B_denominator']]
            n=sum(r['B_numerator'] for r in rr);d=sum(r['B_denominator'] for r in rr)
            aggregate.append(dict(checkpoint=cid,condition=cond,macro_rate=sum(r['B_rate'] for r in known)/len(known) if known else None,
                macro_prefixes=len(known),planned_prefixes=len(protocol['primary_prefixes']),trial_numerator=n,trial_denominator=d,
                trial_rate=n/d if d else None,unknown=sum(r['B_unknown'] for r in rr),excluded=sum(r['excluded'] for r in rr),
                not_run=sum(r['not_run'] for r in rr)))
    csv_write(root/'postA_macro_and_trial.csv',aggregate)
    pairs=matched_rows([p for p in planned if p['condition']!='N'],lookup,CONDITIONS)
    csv_write(root/'P_H_D_matched.csv',pairs or [dict(status='not_run')])
    natural_table=[]
    for cid in CHECKPOINTS:
        for v in protocol['natural_variations']:
            rr=[r for r in natural if r['checkpoint_id']==cid and r['variation_id']==v]
            valid=[r for r in rr if r.get('status')=='completed']
            a=[r for r in valid if r.get('A_reached') is True and r.get('post_A_eligible') is True]
            row=dict(checkpoint=cid,variation_id=v,planned=len(protocol['sampling_seeds']),recorded=len(rr),
                     completed=len(valid),excluded=len(rr)-len(valid),not_run=len(protocol['sampling_seeds'])-len(rr),
                     post_A_eligible=len(a),post_A_steps=json.dumps([r['steps_after_A'] for r in a if r.get('episode_done') is True]))
            for name,key,denom in [('reach_A','A_reached',valid),('full_success','strict_task_success',valid),
                                  ('B_given_A','B_confirmed_completed',a),('first_wrong_focus','first_postA_wrong_focus',a),
                                  ('repeat_A','repeat_A_after_A',a)]: attach(row,name,denom,key)
            natural_table.append(row)
    csv_write(root/'natural_by_checkpoint.csv',natural_table)
    branches=[];branch_planned=[];selections=[]
    for cid in CHECKPOINTS:
        path=root/cid/'branch_selection.json'
        selection=json.loads(path.read_text()) if path.exists() else {'status':'not_run','trials':[],'candidates':None}
        selections.append(dict(checkpoint=cid,**selection));branch_planned.extend(selection['trials'])
        if selection['trials']:
            for row in matched_rows(selection['trials'],lookup,('P','D')):
                source=next(p for p in selection['trials'] if p['prefix_id']==row['prefix_id'])
                branches.append(dict(status='recorded_or_partial',origin_natural_trial_id=source['origin_natural_trial_id'],**row))
        else: branches.append(dict(checkpoint=cid,status=selection['status'],candidate_prefixes=selection['candidates']))
    csv_write(root/'onpolicy_branch_results.csv',branches)
    failure_table=[]
    for r in rows:
        if r['condition']=='N' and r.get('A_reached') is not True: continue
        failure_table.append(dict(checkpoint=r['checkpoint_id'],prefix_origin=r['prefix_origin'],condition=r['condition'],
            prefix_id=r['prefix_id'],variation_id=r['variation_id'],trial_id=r['trial_id'],execution_status=r.get('status'),
            replay_quality=r.get('state_match_quality'),first_postA_action=r.get('first_postA_action'),
            first_postA_action_class=r.get('first_postA_action_class'),first_focus_class=r.get('first_focus_class'),
            known_label_count=r['known_label_count'],unknown_label_count=r['unknown_label_count'],
            manual_text_B_action_A=None,manual_label_status='not_reviewed',evidence_ref=r['evidence_ref']))
    csv_write(root/'failure_classification.csv',failure_table or [dict(status='no_post_A_records')])
    identities=[]
    for cid in CHECKPOINTS:
        p=root/cid/'run_manifest.json';m=json.loads(p.read_text()) if p.exists() else {}
        identities.append(dict(checkpoint=cid,checkpoint_load_status=m.get('checkpoint_load_status','not_loaded'),
            actual_loaded_model_path=m.get('actual_loaded_model_path'),checkpoint_source=m.get('checkpoint_path'),
            exported_weight_hashes=json.dumps([r['sha256'] for r in m.get('inference_model_fingerprints',[])]),
            tokenizer_revision=m.get('tokenizer_revision'),prompt_template_hash=m.get('tokenizer_chat_template_sha256')))
    csv_write(root/'checkpoint_identity.csv',identities)
    core_ids={r['trial_id'] for r in planned};branch_ids={r['trial_id'] for r in branch_planned}
    summary=dict(core_planned=len(planned),core_recorded=sum(r['trial_id'] in core_ids for r in rows),
        core_completed=sum(r['trial_id'] in core_ids and r.get('status')=='completed' for r in rows),
        reused_core=sum(r['trial_id'] in core_ids and r['reused'] for r in rows),
        branch_planned=len(branch_ids),branch_completed=sum(r['trial_id'] in branch_ids and r.get('status')=='completed' for r in rows),
        historical_verified=sum(r['verified_completed'] for r in post),matched_PHD=sum(r['all_conditions_eligible'] for r in pairs),
        execution_status_counts=dict(Counter(r.get('status') for r in rows)),read_errors=errors,plot_error=None)
    report(root,manifest,summary,post,aggregate,natural_table,pairs,branches,rows,steps)
    if plots:
        try: plot(root,post,natural_table,failure_table)
        except Exception as exc: summary['plot_error']=f'{type(exc).__name__}: {exc}'
    save(root/'ANALYSIS_SUMMARY.json',summary)
    return summary


def report(root,manifest,summary,post,aggregate,natural,pairs,branches,rows,steps):
    folder=root/'reports';folder.mkdir(exist_ok=True);cases=root/'cases';cases.mkdir(exist_ok=True)
    text=f"# ScienceWorld Exp2 — actual results / partial delivery\n\nCode: `{manifest['git_commit']}`. Protocol: `{manifest['protocol_version']}`.\n\n"
    text+=f"Core execution completed: {summary['core_completed']}/{summary['core_planned']}; reused: {summary['reused_core']}. Branch execution completed: {summary['branch_completed']}/{summary['branch_planned']}. Completed means execution, not task success.\n\n"
    text+='Checkpoint/runtime errors: '+json.dumps(manifest.get('errors_summary',[]),ensure_ascii=False)+'.\n\n'
    text+='Evidence read errors: '+json.dumps(summary['read_errors'],ensure_ascii=False)+'.\n\n'
    text+='## Historical post-A: prefix first\n\n| Checkpoint | Variation / prefix | Condition | B success | Unknown | Excluded | Not run |\n|---|---|---|---|---|---|---|\n'
    for r in post:
        text+=f"| {r['checkpoint']} | {r['variation_id']} / {r['prefix_id']} | {r['condition']} | {r['B_numerator']}/{r['B_denominator']} | {r['B_unknown']} | {r['excluded']} | {r['not_run']} |\n"
    text+='\nZero denominators are NA. Main denominators require completed execution, verified observed replay and known B label. Ambiguous, mismatched, blocked and missing trials are not silently failures.\n\n'
    text+='## Equal-prefix macro vs pooled trials\n\n| Checkpoint | Condition | Macro | Covered prefixes | Trial success |\n|---|---|---|---|---|\n'
    for r in aggregate:
        rate='NA' if r['macro_rate'] is None else f"{r['macro_rate']:.3f}"
        text+=f"| {r['checkpoint']} | {r['condition']} | {rate} | {r['macro_prefixes']}/{r['planned_prefixes']} | {r['trial_numerator']}/{r['trial_denominator']} |\n"
    text+='\nRepeated seeds share prefixes; they are not independent tasks. Macro means average over prefixes with an evaluable denominator, with coverage shown.\n\n'
    text+='## Natural rollout on the same variation set\n\n| Checkpoint | Variation | A reached | B given A | Full success |\n|---|---|---|---|---|\n'
    for r in natural:
        text+=f"| {r['checkpoint']} | {r['variation_id']} | {r['reach_A_numerator']}/{r['reach_A_denominator']} | {r['B_given_A_numerator']}/{r['B_given_A_denominator']} | {r['full_success_numerator']}/{r['full_success_denominator']} |\n"
    text+='\nN and historical post-A now share variation IDs, not full histories or identical states. Their difference is not a matched causal effect. See onpolicy_branch_results.csv for the separate own-policy prefix P/D comparison, including not_triggered/not_run.\n\n'
    text+='## Matched behavioral evidence and competing interpretations\n\n'
    for cid in CHECKPOINTS:
        counts=Counter(r.get('P_to_D_flip') for r in pairs if r['checkpoint']==cid and r.get('P_D_eligible'))
        d_success=[r for r in post if r['checkpoint']==cid and r['condition']=='D' and r['B_numerator']>0]
        text+=f"- {cid}: P→D paired outcomes {dict(counts)}; D successes cover {len(d_success)} historical prefixes: {[r['variation_id'] for r in d_success]}.\n"
    text+='''
Interpretation must start from the per-prefix table and actual actions. D success across several prefixes together with paired P failures would support a gap in autonomous B deployment; it would not isolate stage recognition from goal salience, attention or prompt-distribution effects. If verified D also fails broadly, weak conditional B competence remains a live explanation. If outcomes are concentrated in one prefix, describe fragile context dependence rather than transferable B mastery. Mixed results are a valid stopping outcome; no automatic further seed/checkpoint/prompt search.

Same generation seed is not token-level causal control. Verified replay is relative to available native observations, not a proof of full hidden-world equality. Historical AE/AS remain separate background; no study advantages or new rewards were computed. Manual text/action inconsistency tags remain not_reviewed until an actual response and reliable entity evidence are inspected.

## First post-A actions and cases

failure_classification.csv separates the first policy action after A, first focus and exact-repeat evidence. An exact repeated A takes precedence in the exclusive first-action class but can also be a wrong-focus event. No-match is environment execution feedback, not automatically a formatting error. Optional semantic navigation/state-recognition tags remain unknown.

Case selection is descriptive: first completed post-A trial per checkpoint/origin/condition, up to eight. This does not select the data used for rates. Each case shows at most the first three post-A policy actions; complete raw inputs, responses and feedback remain in its referenced trial directory.
'''
    used=set();chosen=[]
    for r in sorted(rows,key=lambda r:(r['checkpoint_id'],r['prefix_origin'],r['condition'],str(r['prefix_id']),r['sample_seed'])):
        key=r['checkpoint_id'],r['prefix_origin'],r['condition']
        if key not in used and r.get('status')=='completed' and r.get('first_postA_action') is not None:
            used.add(key);chosen.append(r)
        if len(chosen)==8: break
    for r in chosen:
        body=f"# {r['trial_id']}\n\nEvidence: `{r['evidence_ref']}`. Source run: `{r['source_run_id']}`; reused={r['reused']}.\n\n"
        for s in [s for s in steps[r['trial_id']] if s.get('phase')=='policy' and s.get('task_phase_before')=='post_A'][:3]:
            body+=f"## Index {s['step_index']}\n\nAction: `{s['executed_action']}`; score {s['score_before']} → {s['score_after']}; done={s['done_after']}.\n\n```text\n{s['raw_response']}\n```\n\nFeedback:\n\n```text\n{s['observation_after']}\n```\n\n"
        (cases/(r['trial_id']+'.md')).write_text(body,encoding='utf-8')
        text+=f"\n- [{r['trial_id']}](../cases/{r['trial_id']}.md)"
    text+='\n\n## Reproduction and source evidence\n\nSee RUN_MANIFEST.json, protocol.json, PROTOCOL_V2.md, checkpoint_identity.csv, and source_code/. Per-trial trial_spec.json and evidence_manifest.json record actual/reused identities. The original row run_id is preserved; analysis_run_id identifies this summary. Raw steps are indexed in steps_index.csv, including actual P/H/D rendered prompts. Read errors are listed in ANALYSIS_SUMMARY.json.\n'
    (folder/'final_or_partial.md').write_text(text,encoding='utf-8')


def plot(root,post,natural,failures):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=root/'figures';folder.mkdir(exist_ok=True)
    colors={'P':'#266A91','H':'#D4803B','D':'#328170'}
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    for ax,cid in zip(axes,CHECKPOINTS):
        prefixes=list(dict.fromkeys(r['prefix_id'] for r in post if r['checkpoint']==cid))
        for i,pid in enumerate(prefixes):
            for cond,offset in [('P',-.24),('H',0),('D',.24)]:
                r=next(r for r in post if r['checkpoint']==cid and r['prefix_id']==pid and r['condition']==cond)
                if r['B_denominator']:
                    ax.bar(i+offset,r['B_rate'],.22,color=colors[cond],label=cond if i==0 else None)
                    ax.text(i+offset,r['B_rate']+.03,f"{r['B_numerator']}/{r['B_denominator']}",ha='center',fontsize=8)
                else: ax.text(i+offset,.03,'NA',ha='center',fontsize=8)
        ax.set(title=cid,ylim=(0,1.22),xticks=range(len(prefixes)),xticklabels=[next(r['variation_id'] for r in post if r['prefix_id']==pid) for pid in prefixes],xlabel='Historical variation',ylabel='B success')
        ax.legend(frameon=False,ncol=3)
    fig.suptitle('Exp2: prefix-specific P / H / D (verified, known denominators)');fig.tight_layout();fig.savefig(folder/'postA_by_prefix.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,cid in zip(axes,CHECKPOINTS):
        rr=[r for r in natural if r['checkpoint']==cid]
        for i,r in enumerate(rr):
            for name,off,col in [('reach_A',-.18,'#537793'),('full_success',.18,'#328170')]:
                d=r[name+'_denominator'];n=r[name+'_numerator'];v=n/d if d else 0
                ax.bar(i+off,v,.32,color=col,label=name if i==0 else None)
                ax.text(i+off,v+.03,f'{n}/{d}' if d else 'NA',ha='center',fontsize=9)
        ax.set(title=cid,ylim=(0,1.2),xticks=range(len(rr)),xticklabels=[r['variation_id'] for r in rr],xlabel='Natural variation');ax.legend(frameon=False)
    fig.suptitle('Natural A reach and full completion; same variation is not same history');fig.tight_layout();fig.savefig(folder/'natural_funnel.png',dpi=160);plt.close(fig)
    classes=['B_success','repeat_A','wrong_focus','invalid_action','non_focus_navigation_or_inspection','other_or_unknown']
    fig,ax=plt.subplots(figsize=(12,5));labels=[]
    for cid in CHECKPOINTS:
        for cond in CONDITIONS: labels.append((cid,cond))
    bottoms=[0]*len(labels)
    for label in classes:
        counts=[sum(r.get('first_postA_action_class')==label and r['checkpoint']==cid and r['condition']==cond and
                    r['prefix_origin']=='historical' and r['execution_status']=='completed' and r['replay_quality']=='verified_match' for r in failures) for cid,cond in labels]
        ax.bar(range(len(labels)),counts,bottom=bottoms,label=label);bottoms=[a+b for a,b in zip(bottoms,counts)]
    ax.set(xticks=range(len(labels)),xticklabels=[c+'\n'+d for c,d in labels],ylabel='Trials',title='First post-A action class (historical, verified completed trials)')
    ax.legend(bbox_to_anchor=(1.02,1),loc='upper left',frameon=False);fig.tight_layout();fig.savefig(folder/'first_postA_class.png',dpi=160);plt.close(fig)


if __name__=='__main__': print(json.dumps(analyze(Path(sys.argv[1]),plots=True),ensure_ascii=False))
