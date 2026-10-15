"""Analyze actual study records, with explicit unknowns and strict matched denominators."""
import csv
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys
from replay import paired_eligible


def read(path, errors):
    try:
        opener=gzip.open if path.suffix=='.gz' else open
        with opener(path,'rt',encoding='utf-8') as f:
            for n,line in enumerate(f,1):
                try: yield json.loads(line)
                except ValueError as exc: errors.append(f'{path}:{n}: {exc}')
    except (OSError,EOFError) as exc: errors.append(f'{path}: {exc}')


def table(path, rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        fields=list(dict.fromkeys(k for r in rows for k in r))
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)


def ratio(checkpoint, condition, metric, rows, field):
    known=[r for r in rows if type(r.get(field)) is bool]
    numerator=sum(r[field] for r in known); denominator=len(known)
    return dict(checkpoint=checkpoint,condition=condition,metric=metric,numerator=numerator,
        denominator=denominator,unknown=len(rows)-denominator,rate=numerator/denominator if denominator else None)


def main(root):
    errors=[]; episodes=[r for p in sorted(root.glob('*/episodes.jsonl.gz')) for r in read(p,errors)]
    ids=[r['trial_id'] for r in episodes]
    if len(set(ids))!=len(ids): raise ValueError('Duplicate trial IDs: refusing ambiguous aggregation')
    metrics=[]; pairs=[]; failures=Counter(); event_trials=defaultdict(set); phases=Counter(); coverage=[]
    checks=list(dict.fromkeys(r['checkpoint_id'] for r in episodes))
    for c in checks:
        records=[r for r in episodes if r['checkpoint_id']==c]
        natural=[r for r in records if r['condition']=='N']
        eligible=[r for r in natural if r.get('A_reached') is True and r.get('post_A_eligible') is True]
        metrics.extend([ratio(c,'N','reach_A',natural,'A_reached'),ratio(c,'N','full_success',natural,'strict_task_success'),
            ratio(c,'N','B_attempt_given_A',eligible,'B_attempted'),ratio(c,'N','B_success_given_A',eligible,'B_confirmed_completed')])
        for cond in ('P','H'):
            rr=[r for r in records if r['condition']==cond]
            verified=[r for r in rr if r['state_match_quality']=='verified_match' and r['status']=='completed']
            metrics.append(ratio(c,cond,'fixed_postA_B_success_verified',verified,'B_confirmed_completed'))
            metrics.append(ratio(c,cond,'fixed_postA_B_correct_verified',verified,'B_correct_action'))
            auxiliary=[r for r in rr if r['state_match_quality']=='ambiguous' and r['status']=='completed']
            metrics.append(ratio(c,cond,'fixed_postA_B_success_ambiguous_auxiliary',auxiliary,'B_confirmed_completed'))
            coverage.append(dict(checkpoint=c,condition=cond,recorded=len(rr),verified_completed=len(verified),
                excluded=len(rr)-len(verified),quality_counts=json.dumps(Counter(r['state_match_quality'] for r in rr))))
        grouped=defaultdict(dict)
        for r in records:
            if r['condition'] in ('P','H'): grouped[r['prefix_id'],r['sample_seed']][r['condition']]=r
        matched=[]
        for (prefix,seed),r in grouped.items():
            p=r.get('P',{}); h=r.get('H',{}); ok=paired_eligible(p,h)
            row=dict(checkpoint=c,prefix_id=prefix,sample_seed=seed,primary_pair_eligible=ok,
                P_quality=p.get('state_match_quality'),H_quality=h.get('state_match_quality'),
                same_boundary=p.get('boundary_evidence_hash') is not None and p.get('boundary_evidence_hash')==h.get('boundary_evidence_hash'),
                P_B_success=p.get('B_confirmed_completed'),H_B_success=h.get('B_confirmed_completed'),
                P_first_action_correct=p.get('B_first_continuation_action_correct'),H_first_action_correct=h.get('B_first_continuation_action_correct'),
                P_repeat_A=p.get('repeat_A_after_A'),H_repeat_A=h.get('repeat_A_after_A'),
                P_wrong_focus=p.get('wrong_focus_after_A'),H_wrong_focus=h.get('wrong_focus_after_A'))
            pairs.append(row)
            if ok: matched.append(row)
        delta=sum(int(r['H_B_success'])-int(r['P_B_success']) for r in matched)
        metrics.append(dict(checkpoint=c,condition='H-P',metric='stage_hint_sensitivity',numerator=delta,
            denominator=len(matched),unknown=len(grouped)-len(matched),rate=delta/len(matched) if matched else None))
    case_dir=root/'cases'; case_dir.mkdir(exist_ok=True)
    chosen=[]
    for c in checks:
        rr=[r for r in episodes if r['checkpoint_id']==c]
        # Deterministic descriptive examples, not a success estimate.
        for cond in ('N','P','H'):
            candidate=next((r for r in rr if r['condition']==cond),None)
            if candidate: chosen.append(candidate)
    case_records=defaultdict(list); case_ids={r['trial_id'] for r in chosen}
    for p in sorted(root.glob('*/steps.jsonl.gz')):
        for r in read(p,errors):
            if r['trial_id'] in case_ids: case_records[r['trial_id']].append(r)
            if r['phase']!='policy': continue
            for error in r['failure_event']:
                key=(r['checkpoint_id'],r['condition'],r.get('event_stage',r['task_phase_before']),error)
                failures[key]+=1; event_trials[key].add(r['trial_id'])
            phases[r['checkpoint_id'],r['condition'],r['task_phase_before']]+=1
    for r in episodes:
        if r.get('budget_exhausted'):
            key=(r['checkpoint_id'],r['condition'],r['failure_stage'],'budget_exhausted')
            failures[key]+=1; event_trials[key].add(r['trial_id'])
    table(root/'metrics_by_checkpoint.csv',metrics)
    table(root/'prefix_matched_results.csv',pairs)
    table(root/'replay_coverage.csv',coverage)
    table(root/'failure_types_by_checkpoint.csv',[dict(checkpoint=k[0],condition=k[1],phase=k[2],failure_type=k[3],
        events=n,episodes=len(event_trials[k])) for k,n in failures.items()])
    for ep in chosen:
        rows=case_records[ep['trial_id']]
        text=f"# {ep['trial_id']}\n\nOutcome: {ep.get('termination_reason')}; replay: {ep.get('state_match_quality')}.\n\n"
        for r in rows:
            text+=f"## Policy index {r['step_index']} ({r['phase']})\n\nAction: `{r['executed_action']}`\n\nScore: {r['score_before']} → {r['score_after']}; done={r['done_after']}; A={r['A_transition_event']}; B={r['B_transition_event']}.\n\n{r['observation_after']}\n\n"
        text+='Full prompts, responses, token IDs and native inputs are in this trial’s steps.jsonl.gz records.\n'
        (case_dir/(ep['trial_id']+'.md')).write_text(text,encoding='utf-8')
    fig=root/'figures'; fig.mkdir(exist_ok=True)
    plot_error=None
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        for filename,title,selected in [
            ('F01','Natural capability (known outcomes)',[r for r in metrics if r['condition']=='N']),
            ('F02','Fixed post-A capability (verified replay only)',[r for r in metrics if r['condition'] in ('P','H') and r['metric']=='fixed_postA_B_success_verified']),
            ('F03','Stage-hint sensitivity (matched P/H pairs)',[r for r in metrics if r['condition']=='H-P'])]:
            if not selected: continue
            figure,ax=plt.subplots(figsize=(max(8,len(selected)*0.65),5))
            xs=range(len(selected)); values=[r['rate'] if r['rate'] is not None else 0 for r in selected]
            bars=ax.bar(xs,values)
            for bar,r in zip(bars,selected):
                if r['rate'] is None: bar.set_alpha(0); label='unknown (0 denominator)'
                else: label=f"{r['numerator']}/{r['denominator']}"
                ax.text(bar.get_x()+bar.get_width()/2,max(bar.get_height(),0)+0.02,label,ha='center',fontsize=8,rotation=45)
            ax.set_xticks(list(xs),[f"{r['checkpoint']} {r['condition']}\n{r['metric']}" for r in selected],rotation=60,ha='right',fontsize=8)
            ax.set_title(title); ax.set_ylim((-1 if filename=='F03' else 0),1.2); figure.tight_layout()
            figure.savefig(fig/(filename+'.png'),dpi=160); plt.close(figure)
    except Exception as exc: plot_error=str(exc)
    reports=root/'reports'; reports.mkdir(exist_ok=True)
    manifest=json.loads((root/'RUN_MANIFEST.json').read_text())
    text=(f"# Pilot evidence report\n\nCode: `{manifest['git_commit']}`. Recorded trials: {len(episodes)}; "
        f"status counts: {dict(Counter(r['status'] for r in episodes))}. Target was 64 trials; unrecorded trials are not failures or successes.\n\n"
        "This is a train-diagnostic pilot, not a held-out evaluation. N uses variations 0–3; P/H use 40 and 55. "
        "Their aggregate gap is descriptive, not a same-variation comparison. Stage hints change inputs and do not establish a causal GiGPO defect.\n\n"
        "F01/F02: metrics_by_checkpoint.csv includes numerators, denominators and unknown counts. "
        "F03: prefix_matched_results.csv includes all P/H attempts and an explicit primary eligibility flag. "
        "Primary pairs require both replays verified and identical recorded boundary evidence. "
        "F04: failure_types_by_checkpoint.csv is the error-event table; labels overlap. F05: cases/ contains the first recorded trial per checkpoint/condition.\n\n"
        "A focus after A is only an operational B-attempt proxy; it can be an exact repeat of A. B-correct is conservatively confirmed by native completion, not an oracle target list. "
        "Ambiguous replays remain auxiliary records and never enter primary rates. A missing stable object identity cannot be upgraded to structural equivalence from strings.\n\n")
    for r in metrics:
        text+=f"- {r['checkpoint']} {r['condition']} {r['metric']}: {r['numerator']}/{r['denominator']}, unknown={r['unknown']}.\n"
    text+='\n## Limits and next conclusion\n\n'
    if not episodes: text+='No study episodes were produced. Check conversion/inference logs; conditional ability is unmeasured.\n'
    else:
        text+='Use the measured N A reach rate to describe access to post-A states; use only verified P and matched P/H to describe conditional B ability and hint sensitivity. '
        text+='Where a denominator is zero, that bottleneck remains unresolved. This pilot does not estimate multi-seed significance or prove a credit-assignment mechanism. Review the linked error evidence before choosing an expansion.\n'
    text+=f'\nRecord-reading errors: {json.dumps(errors)}\nPlot limitation: {plot_error}\n'
    (reports/'pilot.md').write_text(text,encoding='utf-8'); (reports/'final_or_partial.md').write_text(text,encoding='utf-8')
    print(json.dumps({'recorded_trials':len(episodes),'reports':str(reports),'read_errors':errors,'plot_error':plot_error}))


if __name__=='__main__': main(Path(sys.argv[1]))
