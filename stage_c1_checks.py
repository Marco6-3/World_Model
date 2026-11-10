"""Inspect physical C.1 data and acceptance targets without outcome rewriting."""
from pathlib import Path
import json
import numpy as np
from bag_common import history_indices

def check(out):
    out=Path(out);rows=[];histories={};violations=[]
    for path in sorted((out/'dataset').glob('*.npz')):
        with np.load(path) as d:
            m=json.loads(d['metadata'].item());rows.append(m)
            if m.get('protocol_version')!='stage_c1':violations.append(path.name+': stale protocol')
            obs=d['observations'];stage=d['stage'];learn=d['learn']
            if np.any(obs[:-1][stage==0,23]!=1) or np.any(obs[:-1][stage==1,24]!=1):
                violations.append(path.name+': item sequence is not fixed A then B')
            if np.any(obs[:-1][stage==1,28:31]):violations.append(path.name+': phase1 cue leak')
            t=m['learn_ranges'][1][0]
            histories[m['seed'],m['mode'],m['method'],m['order']]=obs[history_indices(t)][-4:]
            slips=[e['t'] for e in m['events'] if e['event']=='collar_slip']
            if slips and not np.allclose(d['action'][slips[0]:slips[0]+9,6],.038):
                violations.append(path.name+': forced finger feedback mismatch')
    pairs=[]
    for (seed,mode,method,order),h in histories.items():
        if order==0 and (seed,mode,method,1) in histories:
            delta=np.abs(h-histories[seed,mode,method,1])
            pairs.append(dict(seed=seed,mode=mode,scheme=method,max_h4_difference=float(delta.max()),
                              per_channel_max=delta.max(0).tolist()))
    summary={s:dict(n=len(a:=[r for r in rows if r['method']==s]),success=sum(r['success'] for r in a),
                    collision=sum(r['collision'] for r in a)) for s in sorted({r['method'] for r in rows})}
    report=dict(episodes=len(rows),schemes=summary,violations=violations,paired_h4_actual_differences=pairs,
                note='Floating-point physical histories are measured, not assumed identical. No evaluation labels are passed into policies.')
    recovery=[r for r in rows if r['method']=='slip_withdraw_reopen']
    report['recovery_physics']={k:sum(bool(r[k]) for r in recovery) for k in ('slipped','slip_effective','safe_recovery')}
    (out/'observation_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='paired_h4_actual_differences'},indent=2))
    if violations:raise RuntimeError('C.1 observation audit failed')
    return report

if __name__=='__main__':check(Path(__file__).resolve().parent/'outputs/stage_c_bag')
