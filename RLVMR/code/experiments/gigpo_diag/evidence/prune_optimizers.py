"""Delete ONLY older ALFWorld/ScienceWorld FSDP optimizer shards, with an audit."""
import argparse
import json
import os
from pathlib import Path
import re
import time

OPT = re.compile(r'optim_world_size_(\d+)_rank_(\d+)\.pt$')


def regular(p, root):
    return p.is_file() and not any(x.is_symlink() for x in (p, *p.parents)) and root in p.resolve().parents


def complete(cp, root):
    actor = cp/'actor'
    opts = list(actor.glob('optim_world_size_*_rank_*.pt'))
    if not opts or not regular(cp/'data.pt',root) or not (cp/'data.pt').stat().st_size:
        return False
    matches = [OPT.fullmatch(p.name) for p in opts]
    if not all(matches):
        return False
    worlds = {int(m[1]) for m in matches}
    if len(worlds)!=1:
        return False
    world=worlds.pop()
    if world<1 or {int(m[2]) for m in matches}!=set(range(world)):
        return False
    return all(regular(actor/f'{kind}_world_size_{world}_rank_{rank}.pt',root) and
               (actor/f'{kind}_world_size_{world}_rank_{rank}.pt').stat().st_size>0
               for rank in range(world) for kind in ('model','optim','extra_state'))


def plan(root, env):
    runs=[]; keep=[]; skipped=[]
    for run in sorted((root/env).glob(env+'-qwen25-15b-gigpo-*')):
        cpdir=run/'checkpoints'
        pointer=cpdir/'latest_checkpointed_iteration.txt'
        try:
            if not regular(pointer,root):
                raise ValueError('missing or symlinked checkpoint pointer')
            step=int(pointer.read_text().strip())
            if step<0:
                raise ValueError('negative step')
            cp=cpdir/f'global_step_{step}'
            runs.append((cpdir,step))
            if complete(cp,root):
                keep.append((step,cp))
            else:
                skipped.append({'path':str(cp),'reason':'latest pointer target structurally incomplete'})
        except (OSError,ValueError) as e:
            skipped.append({'path':str(cpdir),'reason':str(e)})
    if not keep:
        return [],[],skipped+[{'reason':'no structurally complete latest checkpoint; delete nothing'}]
    newest=max(step for step,_ in keep)
    retained=[cp for step,cp in keep if step==newest]
    targets=[]
    for cpdir,finished in runs:
        for cp in sorted(cpdir.glob('global_step_*')):
            m=re.fullmatch(r'global_step_(\d+)',cp.name)
            if not m or int(m[1])>=newest or int(m[1])>finished:
                continue
            if not complete(cp,root):
                skipped.append({'path':str(cp),'reason':'old checkpoint structurally incomplete; left untouched'})
                continue
            for role in ('actor','critic'):
                for p in sorted((cp/role).glob('optim_world_size_*_rank_*.pt')):
                    if OPT.fullmatch(p.name) and regular(p,root):
                        s=p.stat()
                        targets.append({'path':str(p),'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'inode':s.st_ino})
    return retained,targets,skipped


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--audit',type=Path,required=True)
    ap.add_argument('--delete',action='store_true')
    args=ap.parse_args()
    root=args.root.resolve(strict=True)
    audit=args.audit.resolve()
    if root==audit or root in audit.parents:
        raise SystemExit('Audit must be outside training logs')
    audit.mkdir(parents=True,exist_ok=False)
    with (audit/'actions.jsonl').open('w',encoding='utf-8') as log:
        def record(event):
            log.write(json.dumps(event,ensure_ascii=False)+'\n'); log.flush(); os.fsync(log.fileno())
        total=0
        for env in ('scienceworld','alfworld'):
            retained,targets,skipped=plan(root,env)
            summary={'environment':env,'retained':[str(p) for p in retained],'targets':targets,'skipped':skipped}
            (audit/(env+'-plan.json')).write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
            print(json.dumps({'environment':env,'retained':summary['retained'],'optimizer_files':len(targets),'planned_GiB':sum(x['bytes'] for x in targets)/2**30,'skipped':len(skipped)}),flush=True)
            for item in targets:
                p=Path(item['path'])
                if not all(complete(cp,root) for cp in retained):
                    raise RuntimeError('Retained checkpoint changed; stopping cleanup')
                if not regular(p,root):
                    record({'action':'skip_changed',**item}); continue
                s=p.stat()
                if (s.st_size,s.st_mtime_ns,s.st_ino)!=(item['bytes'],item['mtime_ns'],item['inode']):
                    record({'action':'skip_changed',**item}); continue
                record({'action':'delete_intent' if args.delete else 'dry_run',**item})
                if args.delete:
                    p.unlink()
                    total+=item['bytes']
                    record({'action':'deleted',**item})
        result={'deleted_bytes':total,'deleted_GiB':total/2**30,'audit':str(audit),'finished_unix':time.time()}
        record({'action':'finished',**result})
        print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
