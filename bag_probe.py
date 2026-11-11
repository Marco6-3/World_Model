"""Matched physical continuation of frozen held-out slip handoffs."""
import json
from hashlib import sha256
from pathlib import Path
import numpy as np
import torch
from bag_common import COLLAR_Z, THRESHOLD
from bag_learning import SPLITS, clean_observations, decide, load_models
from bag_audit import vertical_extent
from bag_packing import dump


def physical_probe(task,args):
    root=args.model_root or args.out
    models=load_models(root,args.device)
    checkpoint_hashes={p.name:sha256(p.read_bytes()).hexdigest() for p in (root/'trained').glob('*.pt')}
    source_hashes={name:sha256((Path(__file__).resolve().parent/name).read_bytes()).hexdigest()
                   for name in ('bag_scene.py','bag_learning.py','bag_common.py','mamba_wm.py')}
    result_path=args.out/'guidance_probe_physical.json'
    previous=json.loads(result_path.read_text()) if result_path.exists() else {}
    if previous and (previous.get('checkpoint_hashes')!=checkpoint_hashes or previous.get('planner_source_hashes')!=source_hashes):
        raise ValueError('Physical probe checkpoints/planner changed; archive the previous probe first')
    rows=previous.get('rows',[])
    done={(r['file'],r['particles'],r['method']) for r in rows}
    for path in sorted((root/'dataset').glob('*slip_withdraw_reopen.npz')):
        if int(path.name.split('_')[0]) not in SPLITS[2]:continue
        with np.load(path,allow_pickle=False) as raw:
            meta=json.loads(raw['metadata'].item())
            _,commands,_=clean_observations(raw['observations'],raw['action'],meta)
            learned=raw['learn'].copy()
            slip=next(int(e['t']) for e in meta['events'] if e['event']=='collar_slip')
            reference=raw['frames'][slip].copy()
        for k in (2,4,8,16):
            for method in ('GPC-RANK','Ours tau=0.7'):
                if (path.name,k,method) in done:continue
                task.reset(meta['mode'],meta['order'],meta['seed'])
                for t,action in enumerate(commands[:slip+1]):
                    task.step(action,learn=bool(learned[t]),perturb=(t==slip))
                prefix_error=float(np.abs(np.asarray(task.frames)[-1,1:14]-reference[1:14]).max())
                if prefix_error>.002:raise ValueError(f'Frozen handoff replay differs by {prefix_error*1000:.2f}mm: {path.name}')
                rng=torch.Generator().manual_seed(16000+meta['seed'])
                start=task.tick;collapsed=False;withdraw=None;reopen=None;safe=False
                while task.tick-start<216:
                    actions,_=decide(models,task.history(),method,rng,particles=k,world_kind='full')
                    for action in actions[:min(6,216-(task.tick-start))]:
                        previous=task.env.aperture
                        task.step(action,learn=True)
                        collapsed|=task.env.aperture<.070
                        bottom=task.poses()[0,2]-vertical_extent(task.poses()[None],0)[0]
                        peak=max(float(r['state'][3]) for r in task.trace)
                        if withdraw is None and bottom>COLLAR_Z+.05:
                            withdraw=task.tick;safe=bool(peak<THRESHOLD)
                        if withdraw is not None and collapsed and task.env.attached and task.env.aperture>.095:
                            reopen=task.tick
                        if reopen is not None or previous-task.env.aperture>.006:break
                    if reopen is not None:break
                row=dict(file=path.name,source_sha256=sha256(path.read_bytes()).hexdigest(),
                         method=method,particles=k,seed=meta['seed'],mode=meta['mode'],order=meta['order'],
                         prefix_max_coordinate_error_m=prefix_error,slipped=task.slipped,
                         safe_withdrawal=safe,withdraw_tick=withdraw,reopen_tick=reopen,
                         withdraw_reopen=bool(withdraw is not None and reopen is not None),
                         collision=bool(max(float(r['state'][3]) for r in task.trace)>=THRESHOLD))
                rows.append(row)
                name=f"{meta['seed']}_{meta['mode']}_{meta['order']}_{k}_{method.split()[0]}.npz"
                row['trace_file']='probe_traces/'+name
                task.save(args.out/row['trace_file'],row)
                dump(result_path,dict(protocol_version='stage_c2',rows=rows,
                     checkpoint_hashes=checkpoint_hashes,planner_source_hashes=source_hashes,
                     complete=len(rows)==96,protocol='Replay frozen physical actions to first post-slip observation; paired full WM, K and noise. Local withdrawal/reopen, no packing claim.'))
                print(f"Budget probe {len(rows)}/96 K={k} {method}: safe_withdrawal={safe}, reopen={reopen is not None}",flush=True)
