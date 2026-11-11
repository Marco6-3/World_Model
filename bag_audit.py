"""Independent final outcome audit from recorded poses, not controller event names."""
from pathlib import Path
from hashlib import sha256
import json
import numpy as np
from bag_common import SIZE, SLOTS, COLLAR_Z, OCCLUDE_Z, THRESHOLD, slot_pos
from bag_packing import dump


def vertical_extent(poses,item):
    w,x,y,z=poses[:,item,3:7].T
    row=np.stack((2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)),-1)
    return np.abs(row)@(SIZE[item]/2)


def recovery_start(row,frames,stage,learn):
    """Use events when present; also detect early physical collar collapse."""
    slips=[int(e['t']) for e in row['events'] if e['event']=='collar_slip']
    if slips:return slips[0],'collar_slip'
    if row.get('perturbed',False) or row.get('slip_episode',False):
        collapsed=np.flatnonzero((stage==0)&learn&(frames[:,1]<.015))
        if len(collapsed):return int(collapsed[0]),'measured_collapse_below_15mm'
    return None,None


def audit(out,data,*,write=True):
    out=Path(out);records=[]
    keys=('second_slot','second_slot_correct','withdraw_reopen','safe_recovery')
    for row in data['episodes']:
        path=out/row['trace_file']
        with np.load(path,allow_pickle=False) as raw:
            frames,poses,stage,learn,attached=(raw[k] for k in ('frames','poses','stage','learn','attached'))
        assert np.isfinite(frames).all() and np.isfinite(poses).all()
        peak=float(frames[:,17].max())
        assert bool(peak>=THRESHOLD)==row['collision']
        extents=[vertical_extent(poses,i) for i in (0,1)]
        placed=[bool(np.linalg.norm(poses[-1,i,:2]-slot_pos(i,int(row['order']) if i==0 else 1-int(row['order']))[:2])<.035 and
                     poses[-1,i,2]-extents[i][-1]>=.009 and poses[-1,i,2]+extents[i][-1]<.085) for i in (0,1)]
        assert bool(all(placed) and frames[-1,1]<.035)==row['success']
        row.setdefault('online_measurements',{k:row.get(k) for k in keys})
        first=0;second=1
        pos=poses[:,second,:3]
        entry=(stage==1)&learn&(np.abs(pos[:,1])<.05)
        entry&=(pos[:,2]-extents[second]<COLLAR_Z+.005)&(pos[:,2]+extents[second]>COLLAR_Z-.025)
        indices=np.flatnonzero(entry)
        chosen=occupied=None;occluded=False
        if len(indices):
            i=int(indices[0]);chosen=int(pos[i,0]>0)
            previous=poses[i,first,:3]
            occluded=bool(previous[2]<OCCLUDE_Z and abs(previous[0])<.095 and abs(previous[1])<.08)
            if occluded:occupied=int(previous[0]>0)
        start,start_source=recovery_start(row,frames,stage,learn)
        slips=[start] if start is not None else []
        withdraw=reopen=None;collapsed=False
        if slips:
            for i in range(int(slips[0]),len(frames)):
                if stage[i]!=0:break
                collapsed|=frames[i,1]<.07
                if withdraw is None and poses[i,first,2]-extents[first][i]>COLLAR_Z+.05:
                    withdraw=i+1
                if withdraw is not None and collapsed and attached[i] and frames[i,1]>.095:
                    reopen=i+1;break
        recovered=bool(withdraw is not None and reopen is not None)
        grazing=False;early_peak=None
        if slips:
            start=int(slips[0]);stop=withdraw if withdraw is not None else int(np.flatnonzero(stage!=0)[0]) if np.any(stage!=0) else len(frames)
            early_peak=float(frames[start:max(start+1,stop),18].max())
            grazing=bool(early_peak>=THRESHOLD)
        first_replan=None;pre_reaction=False
        if slips:
            clock=int(row['learn_ranges'][0][0])
            for commit in row['committed_steps']:
                clock+=int(commit)
                if clock>slips[0]:first_replan=clock;break
            if first_replan is not None:
                pre_reaction=bool(np.any(frames[slips[0]:first_replan,17]>=THRESHOLD))
        row.update(second_slot=chosen,second_goal_slot_correct=bool(chosen==1-int(row['order'])),
                   occupied_slot_at_second_entry=occupied,first_occluded_at_second_entry=occluded,
                   second_slot_correct=bool(chosen is not None and occupied is not None and chosen!=occupied),
                   withdraw_reopen=recovered,safe_recovery=bool(recovered and row['success'] and not row['collision']),
                   recovery_measurement=dict(second_slot=chosen,occupied_slot=occupied,withdraw_tick=withdraw,
                                             recovery_start_tick=start,recovery_start_source=start_source,
                                             reopen_tick=reopen,withdraw_reopen=recovered),
                   measurement_version='stage_c2_oriented_pose_event_or_physical_collapse',
                   first_replan_after_slip_tick=first_replan,collision_before_first_reaction=pre_reaction,
                   collar_grazing_before_withdrawal=grazing,collar_peak_before_withdrawal_Ns=early_peak,
                   trace_sha256=sha256(path.read_bytes()).hexdigest())
        records.append({'key':[row[k] for k in ('method','mode','order','seed','perturbed')],
                        'online':row['online_measurements'],'audited':{k:row[k] for k in keys}})
    if write:
        dump(out/'metric_audit.json',{'criterion':'actual occupied slot; oriented lower face 50 mm above collar; measured collapse and re-grasp',
                                      'episodes':len(records),'records':records})
        dump(out/'results.json',data)
    return data
