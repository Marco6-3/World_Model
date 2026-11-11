"""Generate auditable Stage C tables and a trace-backed, standalone paper figure."""
from pathlib import Path
from itertools import product
from hashlib import sha256
import json
import shutil
import subprocess
import sys
import numpy as np
from bag_packing import METHODS, MODES, dump
from bag_common import COLLAR_Z, SIZE, THRESHOLD


def table(headers,rows):
    return '\n'.join('| '+' | '.join(map(str,row))+' |' for row in (headers,['---']*len(headers),*rows))


def count(rows,key):
    return f"{sum(bool(r[key]) for r in rows)}/{len(rows)}"


def latency(rows):
    values=[v for row in rows for v in row['decision_ms']]
    return float(np.median(values))


def merge(out):
    shards=sorted(out.glob('c2_seed*/results.json'))
    if shards:
        rows=[];seeds=[]
        for file in shards:
            part=json.loads(file.read_text(encoding='utf-8'));seeds.extend(part['seeds'])
            for row in part['episodes']:
                row['trace_file']=file.parent.name+'/'+row['trace_file'];rows.append(row)
        if len(seeds)!=len(set(seeds)):raise ValueError('Overlapping evaluation seeds')
        data={k:v for k,v in part.items() if k not in ('episodes','seeds')}
        data.update(seeds=sorted(seeds),episodes=rows,shards=[str(p.relative_to(out)) for p in shards])
        dump(out/'results.json',data)
    else:data=json.loads((out/'results.json').read_text(encoding='utf-8'))
    protocol=json.loads((out/'protocol.json').read_text(encoding='utf-8'))
    if sorted(data['seeds'])!=sorted(protocol['closed_loop_seeds']):raise ValueError('Incomplete preregistered seed matrix')
    if any(r["perturbed"] and r.get("perturbation_version")!="forced_finger_release_v2" for r in data["episodes"]):
        raise ValueError("Perturbation audit failed: physical finger release v2 is required")
    expected=set(product(METHODS,MODES,(0,1),data['seeds'],(False,True)))
    actual=[(r['method'],r['mode'],r['order'],r['seed'],r['perturbed']) for r in data['episodes']]
    if set(actual)!=expected or len(actual)!=len(expected):raise ValueError('Incomplete or duplicate Stage C matrix')
    if len({json.dumps(r['checkpoint_hashes'],sort_keys=True) for r in data['episodes']})!=1:
        raise ValueError('Mixed checkpoint versions')
    current={p.name:sha256(p.read_bytes()).hexdigest() for p in (out/'trained').glob('*.pt')}
    if any(r['checkpoint_hashes']!=current for r in data['episodes']):raise ValueError('Current checkpoints differ from evaluation')
    sources={name:sha256((Path(__file__).resolve().parent/name).read_bytes()).hexdigest()
             for name in ('bag_scene.py','bag_learning.py','bag_common.py','mamba_wm.py')}
    if any(r.get('planner_source_hashes')!=sources for r in data['episodes']):raise ValueError('Current planner differs from evaluation')
    if any(r.get('protocol_version')!='stage_c2' for r in data['episodes']):raise ValueError('Stale pre-C.2 episode')
    return data


def paired_recovery(rows,ours,baseline):
    selected=[r for r in rows if r['perturbed']]
    lookup={(r['method'],r['seed'],r['mode'],r['order']):int(r['safe_recovery']) for r in selected}
    seeds=sorted({r['seed'] for r in selected})
    differences=np.array([np.mean([lookup[ours,s,m,o]-lookup[baseline,s,m,o]
                                  for m in MODES for o in (0,1)]) for s in seeds])
    draws=np.random.default_rng(731).integers(0,len(seeds),(10000,len(seeds)))
    lo,hi=np.quantile(differences[draws].mean(-1),[.025,.975])
    return {'mean':float(differences.mean()),'ci95':[float(lo),float(hi)],'seed_clusters':len(seeds)}


def figure(out,data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,
                         'axes.labelsize':10,'axes.spines.top':False,'axes.spines.right':False})
    protocol=json.loads((out/'protocol.json').read_text())
    case=protocol['figure_case']
    selected=[];trace=[]
    for method in case['methods']:
        row=next(r for r in data['episodes'] if r['method']==method and r['mode']==case['mode']
                 and r['order']==case['order'] and r['seed']==case['seed'] and r['perturbed']==case['perturbed'])
        selected.append(row)
        with np.load(out/row['trace_file']) as raw:
            trace.append({k:raw[k] for k in ('frames','learn','stage','poses','observations')})
    fig=plt.figure(figsize=(18,12),facecolor='white')
    grid=fig.add_gridspec(3,3,height_ratios=[1.55,1,1],hspace=.37,wspace=.26)
    colors={'right':'#28577c','left':'#b48b31','A':'#c76d3b','B':'#528a8e'}
    minimum=min([0.]+[float(t['frames'][:,14][t['frames'][:,15]>.5].min())*1000
                         for t in trace if np.any(t['frames'][:,15]>.5)])
    points=np.concatenate([t['frames'][t['learn'],2:14].reshape(-1,3) for t in trace])
    lo=np.minimum(np.array([-.11,-.23,0.]),points.min(0)-.025)
    hi=np.maximum(np.array([.11,.26,.28]),points.max(0)+.025)
    manifests=[]
    for col,(row,raw) in enumerate(zip(selected,trace)):
        frames=raw['frames'];ax=fig.add_subplot(grid[0,col],projection='3d')
        x=np.array([-.098,.098,.098,-.098,-.098]);y=np.array([-.087,-.087,.087,.087,-.087])
        for z in (.014,COLLAR_Z):ax.plot(x,y,np.full(5,z),color='#a5aaae',lw=1)
        for xx,yy in zip(x[:4],y[:4]):ax.plot([xx,xx],[yy,yy],[.014,COLLAR_Z],color='#c0c4c7',lw=.7)
        for target in (-.048,.048):ax.scatter([target],[0],[.014],marker='s',s=65,facecolors='none',edgecolors='#626970')
        visible=np.where(raw['learn'][:,None],frames[:,2:14],np.nan)
        ax.plot(*visible[:,0:3].T,color=colors['left'],lw=1.25,label='Left gripper',ls='--')
        ax.plot(*visible[:,3:6].T,color=colors['right'],lw=1.2,label='Right gripper')
        for i,key in enumerate(('A','B')):
            pos=frames[:,8+3*i:11+3*i]
            ax.plot(*np.where(raw['learn'][:,None],pos,np.nan).T,color=colors[key],lw=1.9,label=f'Item {key}',ls='-' if i==0 else ':')
            centre=pos[-1];half=SIZE[i]/2
            w,qx,qy,qz=raw['poses'][-1,i,3:7]
            rotation=np.array([[1-2*(qy*qy+qz*qz),2*(qx*qy-w*qz),2*(qx*qz+w*qy)],
                               [2*(qx*qy+w*qz),1-2*(qx*qx+qz*qz),2*(qy*qz-w*qx)],
                               [2*(qx*qz-w*qy),2*(qy*qz+w*qx),1-2*(qx*qx+qy*qy)]])
            vertices=(np.array([[sx,sy,sz] for sx in (-1,1) for sy in (-1,1) for sz in (-1,1)])*half)@rotation.T+centre
            faces=[[vertices[j] for j in ids] for ids in ((0,1,3,2),(4,5,7,6),(0,1,5,4),(2,3,7,6),(0,2,6,4),(1,3,7,5))]
            ax.add_collection3d(Poly3DCollection(faces,facecolor=colors[key],edgecolor='#555555',alpha=.45,lw=.4))
        ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),zlim=(lo[2],hi[2]),xlabel='x (m)',ylabel='y (m)',zlabel='z (m)')
        ax.view_init(elev=26,azim=-61);ax.set_box_aspect(hi-lo)
        ax.set_xticks([-.10,0,.10]);ax.set_yticks([-.20,0,.20]);ax.set_zticks([0,.10,.20])
        label={'BC H4':'BC (H=4)','GPC-OPT':'GPC-OPT (M=20)','Ours tau=0.9':'Ours (256D, adaptive tau <= 0.9)'}[row['method']]
        ax.set_title(label+f"\nPacking: {'success' if row['success'] else 'failure'} | Collision: {'yes' if row['collision'] else 'no'}",pad=12)
        if col==0:handles,labels=ax.get_legend_handles_labels()
        slips=[e['t'] for e in row['events'] if e['event']=='collar_slip']
        origin=slips[0] if slips else row['learn_ranges'][0][0]
        times=frames[:,0]-origin*data['control_dt']
        window=(times>=-1)&(times<=8)
        aper=fig.add_subplot(grid[1,col]);clear=fig.add_subplot(grid[2,col])
        aper.plot(times[window],frames[window,1]*1000,color=colors['right'],lw=1.7)
        aper.axhline(70,color='#666666',ls=':',lw=1,label='70 mm diagnostic level')
        aper.set(xlim=(-1,8),ylim=(-10,175),ylabel='Collar aperture (mm)',xlabel='Time from slip (s)' if slips else 'Time from insertion start (s)')
        aper.set_title('Measured collar response'+('' if slips else ' — slip not triggered'))
        occupancy=frames[:,15]>.5
        values=np.where(occupancy,frames[:,14]*1000,np.nan)
        clear.plot(times[window],values[window],color=colors['right'],lw=1.7,label='Occupied-throat clearance')
        if not np.any(occupancy&window):
            clear.text(.5,.65,'No occupied-throat samples',transform=clear.transAxes,ha='center',color='#666666')
        clear.axhline(0,color='#555555',lw=1)
        clear.axhline(28,color='#777777',ls=':',lw=1,label='28 mm planning margin')
        collision=(frames[:,17]>=THRESHOLD)&window
        rug=min(-25,minimum-8)
        clear.scatter(times[collision],np.full(int(collision.sum()),rug+2),marker='x',s=22,color=colors['A'],label='Contact events (rug)')
        clear.set(xlim=(-1,8),ylim=(rug,70),ylabel='Effective clearance (mm)',xlabel='Time from slip (s)' if slips else 'Time from insertion start (s)')
        clear.set_title(f"Withdraw/reopen: {'yes' if row['withdraw_reopen'] else 'no'} | Safe recovery: {'yes' if row['safe_recovery'] else 'no'}")
        for axis in (aper,clear):
            axis.grid(axis='y',color='#dddddd',lw=.6)
            if slips:axis.axvline(0,color='#777777',ls='--',lw=.9)
            measured=row.get('recovery_measurement',{})
            for key,color,style in (('withdraw_tick',colors['left'],'--'),('reopen_tick',colors['B'],':')):
                tick=measured.get(key)
                if tick is not None:
                    t=(tick-origin)*data['control_dt']
                    if -1<=t<=8:axis.axvline(t,color=color,ls=style,lw=1.1)
        if col==0:clear.legend(loc='upper right',fontsize=8,frameon=False)
        manifests.append({'method':row['method'],'trace_file':row['trace_file'],
                          'trace_sha256':sha256((out/row['trace_file']).read_bytes()).hexdigest(),
                          'origin_control_tick':origin,'slip_triggered':bool(slips),
                          'success':row['success'],'collision':row['collision'],'safe_recovery':row['safe_recovery']})
    fig.suptitle('Stage C.2 | Conformal risk-adaptive in-flow guidance',fontsize=17,y=.985)
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.955),ncol=4,frameon=False)
    overall=[]
    for method in case['methods']:
        population=[r for r in data['episodes'] if r['method']==method and r['perturbed']]
        overall.append(f"{method}: {sum(r['safe_recovery'] for r in population)}/{len(population)}")
    overall_note='Full-matrix safe recovery | '+'; '.join(overall) if all(len([r for r in data['episodes'] if r['method']==m and r['perturbed']])>=18 for m in case['methods']) else 'Full-matrix rates are reported separately.'
    fig.text(.5,.025,'Fixed paired case: elastic_fast, A-left then B-right, seed 2000, slip enabled. Top: measured learned-control trajectories and final poses (not camera frames).\n'
             'Lower panels: 9 s display windows; untriggered cases use insertion start. Clearance shown only while occupied; contact marks are a rug.\n'
             'Dashed gold: measured withdrawal; dotted teal: measured reopening. Outcomes and unsuccessful recoveries are preserved.\n'+overall_note,ha='center',fontsize=9)
    fig.subplots_adjust(top=.88,bottom=.105,left=.055,right=.98)
    fig.savefig(out/'bag_packing_demo.png',dpi=260,facecolor='white')
    fig.savefig(out/'bag_packing_demo.pdf',facecolor='white')
    plt.close(fig)
    dump(out/'figure_manifest.json',dict(selection='predeclared paired case, no outcome-based selection',cases=manifests,
                                       png_pixels=[4680,3120]))


def build_report(out):
    out=Path(out);data=merge(out)
    from bag_audit import audit
    data=audit(out,data);rows=data['episodes']
    hashes={p.name:sha256(p.read_bytes()).hexdigest() for p in (out/'trained').glob('*.pt')}
    timing_path=out/'isolated_timing.json'
    cached=json.loads(timing_path.read_text()) if timing_path.exists() else {}
    if cached.get('checkpoint_hashes')!=hashes or cached.get('planner_source_hashes')!=rows[0]['planner_source_hashes']:
        from bag_timing import measure
        measure(out)
    timing=json.loads((out/'isolated_timing.json').read_text())
    train=json.loads((out/'training.json').read_text(encoding='utf-8'))
    if train.get('protocol_version')!='stage_c2':raise ValueError('C.2 training manifest required')
    for source in train['files']:
        if sha256((out/'dataset'/source['file']).read_bytes()).hexdigest()!=source['sha256']:
            raise ValueError('Frozen training dataset changed: '+source['file'])
    from bag_learning import load_windows,SPLITS
    windows,source_files=load_windows(out,SPLITS[0]);valid=windows['expert'].sum(-1)>=6
    dump(out/'supervision_audit.json',dict(protocol_version='stage_c2',world_windows=len(windows['obs']),
         flow_expert_windows=int(valid.sum()),recovery_windows=int(windows['recovery'].sum()),
         retained_recovery_windows=int(windows['retained_recovery'].sum()),
         flow_eligible_recovery_windows=int((valid&windows['recovery']).sum()),files=source_files,
         excluded_schemes=['direct_insert','premature_release','exploration'],
         command_imputation='9 forced-release frames held at preceding command; raw sensors and NPZ immutable'))
    memory=json.loads((out/'memory_diagnostic.json').read_text(encoding='utf-8'))
    calibration=json.loads((out/'calibration.json').read_text(encoding='utf-8'))
    normal=[r for r in rows if not r['perturbed']];recovery=[r for r in rows if r['perturbed']]
    c1=[];c2=[];mode_table=[];stats={}
    for method in METHODS:
        a=[r for r in normal if r['method']==method];b=[r for r in recovery if r['method']==method]
        c1.append([method,len(a),count(a,'success'),count(a,'second_slot_correct'),
                   sum(r['second_slot'] is not None for r in a),count(a,'collision'),f'{latency(a):.2f}'])
        safe_no_timeout=sum(r['safe_recovery'] and not any(r['insertion_budget_exhausted']) for r in b)
        c2.append([method,len(b),count(b,'slipped'),count(b,'slip_effective'),count(b,'success'),count(b,'withdraw_reopen'),count(b,'safe_recovery'),f'{safe_no_timeout}/{len(b)}',count(b,'collision'),count(b,'collision_before_first_reaction'),sum(any(r['insertion_budget_exhausted']) for r in b),f'{latency(b):.2f}'])
        stats[method]={'normal':{'n':len(a),'success':sum(r['success'] for r in a),'collision':sum(r['collision'] for r in a),
                                'second_slot_correct':sum(r['second_slot_correct'] for r in a),'p50_ms':latency(a)},
                       'recovery':{'n':len(b),'triggered':sum(r['slipped'] for r in b),'effective':sum(r['slip_effective'] for r in b),'success':sum(r['success'] for r in b),'safe_recovery':sum(r['safe_recovery'] for r in b),'safe_recovery_no_timeout':safe_no_timeout}}
        for mode in MODES:
            selected=[r for r in a if r['mode']==mode]
            mode_table.append([method,mode,count(selected,'success'),count(selected,'second_slot_correct'),count(selected,'collision')])
    comparisons={}
    for ours in ('Ours tau=0.7','Ours tau=0.9'):
        comparisons[ours]={other:paired_recovery(rows,ours,other) for other in METHODS if not other.startswith('Ours tau=')}
    dump(out/'summary.json',dict(methods=stats,recovery_differences=comparisons,total_episodes=len(rows)))
    figure(out,data)
    from paper_fig_stage_c import main as paper_figure
    paper_figure(out)
    dataset_rows=[]
    for scheme in ('direct_insert','spread_and_hold','slip_withdraw_reopen','exploration'):
        source=[f for f in train['files'] if f['scheme']==scheme]
        dataset_rows.append([scheme,len(source),sum(f['success'] for f in source),sum(f['collision'] for f in source)])
    model_rows=[]
    conformal_rows=[]
    for kind,record in train['models'].items():
        test=record['heldout']
        c=record['conformal']
        label={'full':'Full (256D)','short':'Short-MLP (H4)','four':'4D-WM'}[kind]
        conformal_rows.append([label,f"{c['clearance_mae_m']*1000:.2f}",f"{c['conformal_q_m']*1000:.2f}",
                               f"{c['conformal_coverage']:.2%}",c['calibration_windows'],c['test_windows']])
        model_rows.append([kind,record['past_dimension'],record['future_dimension'],record['parameters'],
                           f"{test['safety_mae'][0]*1000:.2f}",f"{test['mode_accuracy']:.1%}",
                           f"{test['occupied_slot_accuracy']:.1%}" if test['occupied_slot_accuracy'] is not None else '—',test['occupied_slot_windows']])
    memory_rows=[[name,result['heldout']['n'],f"{result['heldout']['accuracy']:.1%}"] for name,result in memory['models'].items()]
    comparison_rows=[[ours,other,f"{v['mean']:+.1%}",f"[{v['ci95'][0]:+.1%}, {v['ci95'][1]:+.1%}]"]
                     for ours,items in comparisons.items() for other,v in items.items()]
    full=stats['Ours tau=0.9'];baseline=stats['BC H4']
    acceptance=[]
    for m in ('Ours tau=0.7','Ours tau=0.9'):
        n=stats[m]['normal'];r=stats[m]['recovery']
        acceptance.extend([[m+' normal success 18/18',n['success'],n['success']==18],
                           [m+' second slot 18/18',n['second_slot_correct'],n['second_slot_correct']==18],
                           [m+' normal collisions 0/18',n['collision'],n['collision']==0],
                           [m+' safe recovery >=17/18',r['safe_recovery'],r['safe_recovery']>=17]])
        best=max(stats[b]['recovery']['safe_recovery'] for b in METHODS if not b.startswith('Ours tau='))
        acceptance.append([m+' safe recovery > every baseline',r['safe_recovery']-best,r['safe_recovery']>best])
        exhausted=sum(any(row['insertion_budget_exhausted']) for row in rows if row['method']==m)
        acceptance.append([m+' no insertion budget exhaustion (hover check)',exhausted,exhausted==0])
        ratio=timing['summary']['GPC-OPT']['p50_ms']/timing['summary'][m]['p50_ms']
        acceptance.append([m+' GPC-OPT latency / Ours >=8',ratio,ratio>=8])
    for m in ('BC H4','Ours Short-MLP H4'):
        n=stats[m]['normal']
        acceptance.extend([[m+' second slot approximately 9/18 (+/-1)',n['second_slot_correct'],8<=n['second_slot_correct']<=10],
                           [m+' packing <=9/18',n['success'],n['success']<=9]])
    for m,maximum in (('GPC-RANK',11),('Ours 4D-WM',12),('BC H4',6),('Joint-FM',6),('Ours Short-MLP H4',6)):
        v=stats[m]['recovery']['safe_recovery']
        acceptance.append([m+f' safe recovery <={maximum}/18',v,v<=maximum])
    for kind in ('full','short','four'):
        h=train['models'][kind]['heldout'];mae=h['safety_mae'][0]*1000;acc=h['occupied_slot_accuracy']
        if kind=='full':acceptance.extend([['full clearance MAE <5 mm',mae,mae<5],['full mode accuracy >90%',h['mode_accuracy'],h['mode_accuracy']>.9],['full slots >=98%',acc,acc>=.98]])
        elif kind=='short':acceptance.append(['short slots 45%-55%',acc,.45<=acc<=.55])
        else:acceptance.append(['four clearance MAE >10 mm',mae,mae>10])
        c=train['models'][kind]['conformal']
        acceptance.append([kind+' held-out conformal coverage >=90%',c['conformal_coverage'],c['conformal_coverage']>=.9])
    probe=json.loads((out/'guidance_probe.json').read_text(encoding='utf-8'))
    if probe.get('checkpoint_hashes')!=hashes or probe.get('planner_source_hashes')!=rows[0]['planner_source_hashes']:
        raise ValueError('Stale predictive particle-budget probe; rerun guidance_vs_ranking_probe')
    probe_rows=[]
    physical_path=out/'guidance_probe_physical.json'
    physical=json.loads(physical_path.read_text()) if physical_path.exists() else {}
    if not physical.get('complete'):
        with (out/'physical_probe_c2.log').open('w',encoding='utf-8') as log:
            subprocess.run([sys.executable,str(Path(__file__).with_name('bag_packing.py')),'probe',
                            '--out',str(out.resolve()),'--device','cuda'],stdout=log,stderr=subprocess.STDOUT,check=True)
        physical=json.loads(physical_path.read_text())
    if physical.get('checkpoint_hashes')!=hashes:raise ValueError('Stale physical probe checkpoints')
    if physical.get('planner_source_hashes')!=rows[0]['planner_source_hashes']:raise ValueError('Stale physical probe planner')
    if len(physical['rows'])!=96:raise ValueError('Incomplete physical particle-budget matrix')
    coverage_note=''
    diagnosis_path=out/'conformal_diagnostics.json'
    if diagnosis_path.exists():
        diagnosis=json.loads(diagnosis_path.read_text())
        if abs(diagnosis['q_m']-train['models']['full']['conformal']['conformal_q_m'])>1e-8:
            raise ValueError('Stale conformal subgroup diagnosis')
        groups=diagnosis['rows'];lowest=min((r for r in groups if r['group']=='seed'),key=lambda r:r['coverage'])
        rec=next(r for r in groups if r['group']=='recovery' and r['label']=='recovery')
        coverage_note=(f"Full分组事后复核：种子{lowest['label']}整窗口覆盖率为{lowest['coverage']:.2%}；"
                       f"恢复相关窗口覆盖率为{rec['coverage']:.2%}。这些留出分组仅用于解释误差，未用于修改校准分位数。")
    for k in (2,4,8,16):
        rank_values=[r['min_clear_lb_m'] for r in probe['rows'] if r['particles']==k and r['method']=='GPC-RANK']
        for method in ('GPC-RANK','Ours tau=0.7'):
            p=[r for r in probe['rows'] if r['particles']==k and r['method']==method]
            live=[r for r in physical['rows'] if r['particles']==k and r['method']==method]
            gain=(np.mean([r['min_clear_lb_m'] for r in p])-np.mean(rank_values))*1000
            probe_rows.append([method,k,len(p),f"{np.mean([r['min_clear_lb_m'] for r in p])*1000:.2f}",f'{gain:+.2f}',
                               count(p,'commanded_withdrawal'),count(live,'safe_withdrawal') if live else '未运行',
                               count(live,'withdraw_reopen') if live else '未运行'])
    dump(out/'acceptance.json',dict(checks=[dict(target=t,actual=v,passed=bool(p)) for t,v,p in acceptance],
         all_passed=all(p for _,_,p in acceptance),note='Thresholds are preregistered targets, never substituted for measurements'))
    failed=[target for target,_,passed in acceptance if not passed]
    verdict=(("**硬性验收全部通过。** " if not failed else "**硬性验收未全部通过。** 未通过："+'；'.join(failed)+'。\n\n')+
             f"实测 Ours τ=0.9 常规双物品成功 {full['normal']['success']}/{full['normal']['n']}、"
             f"夹碰 {full['normal']['collision']}/{full['normal']['n']}、安全滑脱恢复 {full['recovery']['safe_recovery']}/{full['recovery']['n']}。"
             f"BC 对应为 {baseline['normal']['success']}/{baseline['normal']['n']}、{baseline['normal']['collision']}/{baseline['normal']['n']}、"
             f"{baseline['recovery']['safe_recovery']}/{baseline['recovery']['n']}。GPC-OPT 安全恢复为 {stats['GPC-OPT']['recovery']['safe_recovery']}/{stats['GPC-OPT']['recovery']['n']}。"
             + ("本轮未验证 Full 对所有基线的安全恢复领先。" if max(stats[m]['recovery']['safe_recovery'] for m in ('Ours tau=0.7','Ours tau=0.9')) <= max(stats[m]['recovery']['safe_recovery'] for m in METHODS if not m.startswith('Ours tau=')) else "恢复差异仍需更多独立种子验证。"))
    if full['recovery']['safe_recovery_no_timeout']<full['recovery']['safe_recovery']:
        verdict+=f" τ=0.9中，安全恢复且未耗尽插入预算为{full['recovery']['safe_recovery_no_timeout']}/{full['recovery']['n']}；其余成功包含超时后公共释放的重力装包，不能解释为已完全消除悬停。"
    late_bc=[r for r in normal if r['method']=='BC H4' and r['success'] and not r['second_slot_correct']]
    late_bc_note=(f"BC H4有{len(late_bc)}个常规回合初次二次分槽错误而最终姿态仍合格，"
                  f"其中{sum(r['collision'] for r in late_bc)}个发生夹碰。因此最终装包成功率与初次正确分槽率不同；两项均按原定义保留。")
    lines=['# Stage C.2 · 双臂动态弹性袋口多物品收纳实验报告','',verdict,'',
      '## Stage C.2 验收','',table(['目标','实测','达到目标'],[[t,(f'{v:.2%}' if any(k in t for k in ('coverage','accuracy','slots')) else f'{v:.4g}') if isinstance(v,float) else v,'是' if p else '否'] for t,v,p in acceptance]),'',
      '上述表格核对文档预期的性能曲线；不会为使弱基线落入指定区间而人为降低其性能。恢复率高于各基线的计数比较不等于已达到统计显著性，配对小样本区间另列。','',
      '这里GPC为共享BagFlow和本地世界模型的RANK/OPT操作基线，不声称完整复现原论文的模型与训练协议。Flow/Joint-FM 训练使用目标槽位 one-hot；推理时第一阶段读取尚可见的指令，第二阶段仅使用各自历史编码器预测的空闲槽位概率。BC、Joint-FM、Short-MLP、GPC 使用 H4 编码器；Full、4D 使用长历史编码器。训练标签不进入评测策略。'
      '风险统一使用 clear_lb=预测净空−验证集保形分位数，28 mm 单步裕度，常规18 mm/紧急35 mm终端裕度；安全门为sigmoid((clear_lb−18 mm)/0.5 mm)。安全时保留0.35×safe_gate×(右臂高度−55 mm)线性项，并加0.35×stopgrad(safe_gate)×((右臂高度−55 mm)/100 mm)²平滑下插势能，克服持物先验的悬停平衡点；其门控停止梯度以防优化器通过降低预测净空关闭进度代价。风险增大时关闭推进并拉向240 mm以上。紧急状态另外按实测左指到袋柄的距离平滑设置重抓夹爪目标（远32 mm/近8 mm）及左臂重撑势能；这些解析代价对全部WM规划方法共用，不覆盖最终动作。该修正在独立开发种子1100上针对长时间张爪等待提出，开发回合不计入主表。所有粒子从同一时序高斯分布出发，只保留第二阶段左右对称探索偏置。流内梯度统一归一化为L2范数3（零梯度保持零），避免24步均值令向下吸引梯度消失；步长由自适应增益控制。τ=0.7检查一步欧拉外推后风险最优粒子的最小保形下界；低于18 mm或孔径突降时提前至0.7并用高增益6.0，否则使用温和0.45；若历史孔径在曾打开后坍塌，则重开后到实际下插前仍按0.7介入，安全时增益仍为0.45；仅初始闭口不会触发这一状态。τ=0.9表示常规安全时的最晚介入时间。主表所有多粒子方法K=16；GPC-OPT保留K=1和20次端点梯度迭代。','',
      '## 实现与适用范围','',
      '场景由 IsaacLab/PhysX 的真实刚体、两个 Y 轴滑动关节、弹簧阻尼和双唇耦合构成。袋身为刚性碰撞代理与柔性外观配色，不是布料有限元仿真。'
      '左臂在实际末端接近撑口柄且闭爪后，通过有界柔顺抓持力拉动后唇；前唇由被动耦合机构撑开。滑脱解除这一抓持并施加向内速度扰动，所有方法使用相同物理逻辑。'
      '沿用的 ALOHA 资产日志仍有部分 visual 引用未解析警告；关节与物理接触正常运行，未将其声称为已验证的相机渲染。右臂使用 ALOHA 关节和接触抓持异构盒体。抓取、释放、两次搬运前缀和最终关口是共同脚本，学习策略控制入袋及紧急响应；释放交接由共同几何触发器判断，它不选择或修正目标槽位。不把它描述为端到端自主任务。','',
      '槽位中心 x=−0.048/+0.048 m。ItemA 尺寸 36×42×24 mm，ItemB 为 28×30×42 mm 方柱。'
      '已入袋物品中心 z<65 mm 时，其位置、当前槽位占用和两个槽位可见性同时清零，避免掩码位置泄漏占用侧。'
      '策略输入没有真实 ξ、缓存占用标签、真实目标槽位或任务成功标志。当前物品身份与阶段仍可见。','',
      '**重要识别限制：**本轮固定 A→B 抓取顺序，order 决定 A 左/B 右或 A 右/B 左；pre_entry 均从 x=0 开始，第二阶段指令清零。'
      '这消除了固定身份映射，但真实动力学可能留下微小短历史差异；不能先验宣称 H4 输入完全相同或其成绩必定为 50%。下文另列真实回放占用读出准确率和独立、平衡的合成遮挡记忆容量诊断。','',
      '物理步长 1/120 s，控制每 4 个物理步更新；预测 24 步、最多执行 6 步；若一个控制步观测到孔径下降超过 6 mm，则所有方法统一提前重规划，动作仍由各自策略决定。每件物品最多执行 216 个学习控制步。'
      '长历史为 64 个观测 token，覆盖约 14.8 s，最后四帧连续；Short-History/Short-MLP 只读取这四帧。'
      '规范给出的 arange 示例实际产生67个token；本实现以60个稀疏点加末尾4个连续点保持64-token架构，首尾覆盖444帧（443个时间间隔）。'
      'CUDA 使用本机 RTX 5060 与 torch '+train['torch']+'。物理在 CPU 上执行并在规划期间暂停；决策延迟不是已验证的硬实时闭环保证。','',
      '指定的`C:\\isaacsim_env\\phase2_cuda\\Scripts\\python.exe`实际不存在；本轮使用`C:\\isaacsim_env\\Scripts\\python.exe`，加载`C:\\isaacsim_env\\phase2_cuda`中的CUDA PyTorch。GPU可用性和设备名称已实测，执行器差异记录在protocol.json与training.json。','',
      '## 袋口回缩标定','',
      '下表沿用首轮自由回缩标定，弹簧、阻尼和袋口几何参数本轮未改动；它不是本轮新增的闭环成绩。','',table(['模态','初始孔径 mm','自由回缩低于 70 mm 所需 s'],
      [[r['mode'],f"{1000*r['initial_aperture_m']:.2f}",f"{r['time_below_70mm_s']:.3f}" if r['time_below_70mm_s'] is not None else '2 s 观察窗内未达到'] for r in calibration]),'',
      '70 mm 是展示回缩时间差的诊断水平，不是接触冲量阈值。安全状态的净空结合物体旋转包围盒的水平余量、物体包围球和末端 12 mm 半径代理的垂向分离，形成三维有效余量下界；并非精确网格距离。'
      '碰撞以实际接触传感器物理步冲量合计 ≥0.004 N·s 判定，包含袋口与右臂/物品接触及两物品之间接触；左臂预期撑口和袋口自接触不计为非预期夹碰。','',
      '## 数据与训练','',
      f"共 {len(train['files'])} 个已冻结数据回合，按种子分成训练 {train['split_seeds'][0]}、验证 {train['split_seeds'][1]}、离线留出 {train['split_seeds'][2]}。"
      '三档动力学×两种顺序×四种轨迹覆盖全部组合；探索包含松手、滑脱和错误槽位动作。Flow/Joint-FM只监督成功spread_and_hold与稀疏slip_withdraw_reopen的无夹碰专家前缀；direct_insert与premature_release整条禁止专家监督。与恢复段重叠的窗口按预先固定的文件名哈希采样保留15%（逐回合向下取整），世界模型仍使用100%窗口。'
      '早期未实际撤回的恢复标定轨迹已归档，未混入最终训练集。','',
      table(['轨迹类型','回合','装包成功','出现夹碰'],dataset_rows),'',
      f"训练保留{len(windows['obs'])}个世界模型窗口、{int(valid.sum())}个Flow/Joint-FM专家窗口；恢复相关{int(windows['recovery'].sum())}个窗口中固定采样保留{int(windows['retained_recovery'].sum())}个（逐回合向下取整），其余仅用于世界模型。",'',
      f"Flow 和 Joint-FM 各训练 {train['flow_updates']} 步；每种世界模型训练 {train['world_updates']} 步。"
      'Full 的 Mamba-Past 和 Mamba-Future 都是 256D。仅迁移 Stage B 256D checkpoint 中形状兼容的未来主干层；新 256D 历史编码器及任务读出重新训练，迁移清单与哈希见 training.json。'
      'Short-MLP 只替换历史编码器；4D-WM 保留长历史编码器而将未来动力学压缩到 4D，因此是未来表示维度对照，不是参数量匹配对照。','',
      table(['变体','Past D','Future D','WM 参数数','留出净空 MAE mm','ξ 准确率','遮挡占用读出准确率','占用窗口数'],model_rows),'',
      '遮挡槽位读出主指标只在第二阶段学习控制开始前取一次，排除专家后续目标相关动作反向泄漏槽位；完整占用窗口读出另保存在 training.json。固定物品身份捷径已移除；回放成绩仍不能替代闭环分槽结果。','',
      '### 流内梯度引导 vs. 纯采样排序：算力预算与先验稀缺度分析','',
      '校准仅使用验证种子1004/1005，测试种子1006/1007不参与分位数选择。每窗口分数取24步净空高估的最大值，使用ceil((N+1)×0.9)次序统计量；预测净空按推理所用的隐动态后验混合，避免真实模态标签进入校准推理。覆盖率是整窗口同时覆盖率。重叠窗口存在相关性，90%标称值不能直接解释为任意闭环状态或整回合的无条件安全保证。','',
      table(['世界模型','留出净空MAE mm','90%校准q mm','留出实测覆盖率','校准窗口','测试窗口'],conformal_rows),'',
      coverage_note,'',
      '粒子预算表在全部12个留出滑脱首响应交接状态上配对比较。此表两方法统一使用Full世界模型、同K、同噪声种子，以隔离流内引导的影响；主表GPC使用H4，其记忆差异不应归因于引导。最小保形净空为选中24步轨迹的预测值；末端指令达到240 mm仅是命令代理指标。物理撤回率来自重放同一冻结轨迹前缀后实际执行策略，采用同一物品旋转包围盒下表面高于袋口50 mm且从前缀至撤回无夹碰；重撑额外要求实际孔径恢复95 mm以上并重新抓持。该局部诊断不替代Table C2的完整装包安全恢复率。','',
      table(['方法（同Full WM）','K','状态数','平均最小预测clear_lb mm','相对排序净空提升 mm','末端指令撤回','物理无碰撞撤回','物理撤回重撑'],probe_rows),'',
      '### 独立 GPU 规划计时','',
      '所有仿真分片结束后，以 12 个相同留出第二次入袋历史逐方法计时；预热、轮换方法顺序，显式同步 CUDA，包含动作返回 CPU。此表与下方含模拟器负载的闭环延迟分开报告。','',
      table(['方法','独立 P50 ms','独立均值 ms'],[[m,f"{timing['summary'][m]['p50_ms']:.2f}",f"{timing['summary'][m]['mean_ms']:.2f}"] for m in METHODS]),'',
      '## Table C1 · 常规双物品闭环','',
      f"独立仿真种子 {data['seeds']}，每方法跨三模态、两顺序共 {len(normal)//len(METHODS)} 回合。"
      '成功要求两件物品都位于规定槽位且最终孔径 <35 mm；碰撞独立计数。二次分槽在学习控制期间物品实际进入袋口咽喉时判定，要求第一件物品确实在遮挡槽位中且选择其实际空闲侧；未到达第二通道或第一件未进入遮挡槽位计为未正确，分母不删失败回合。'
      '最终表格使用完整姿态复核后的 results.json；在线近似值保留在 online_measurements，逐项变化记录于 metric_audit.json，原始日志与物理轨迹不删除。计时含观测编码、CUDA 规划和动作返回 CPU；主矩阵最多使用两个独立种子进程并行，部分时段另有局部物理预算诊断共享GPU；闭环延迟受并行及模拟器负载影响；算法效率应参考所有仿真结束后的独立计时表。','',
      table(['方法','N','双物品成功','二次分槽正确','到达第二通道','夹碰','决策 P50 ms'],c1),'',
      late_bc_note,'',
      '## Table C2 · Collar-Slip 扰动恢复','',
      '在第一件物品接近袋口时触发一次相同状态条件的滑脱，解除撑口抓持 0.3 s，同时将左指张开目标强制设为 38 mm 指间半距并推动袋口向内回缩，以防真实手指接触继续夹持撑口柄。本轮复用192条冻结v2轨迹并重训全部模型，未重新collect。物理故障仍在BagScene强制开指，但策略last_action保持自身命令。旧数据中故障覆盖的九帧命令无法精确恢复，在加载时保持故障前命令值，原始NPZ与真实袋口传感器不改写。恢复示教先垂直撤至 z=245 mm 再重撑；occupied 提前至物品下表面低于170 mm。'
      '安全恢复采用严格门槛：用记录的物品姿态重建旋转包围盒，同一件物品的实际下表面撤到袋口上方至少 50 mm，袋口曾收缩到 70 mm 以下后由左臂重新撑到 95 mm 以上，最终双物品装包成功且全回合无夹碰。'
      '后续拿起第二件物品不能充当撤回。扰动回合即使缺少collar_slip事件，只要记录显示第一阶段学习期间孔径坍塌到15 mm以下，仍可按同一物品实际撤回、重新抓持和重撑审计，消除早期坍塌的假阴性。“实际回缩”指触发后 1 s 内实测孔径低于 70 mm。“首次响应前夹碰”按控制步核对：真实滑脱发生后、下一次规划读取新观测前已经发生夹碰；这类回合仍计入失败分母，但不能单独据此归因于规划质量。脚本动作名称不参与恢复成功判定。插入预算耗尽单独列出：超时后公共脚本释放物品仍可能靠重力落入正确槽位，因此最终成功不自动证明悬停已消除。另列“安全恢复且无超时”是事后严格控制诊断，不替换原预注册安全恢复定义。首轮数据与模型整体归档，本轮常规和扰动组均重新评测。','',
      table(['方法','N','触发滑脱','实际回缩','双物品成功','实测撤回并重撑','安全恢复','安全恢复且无超时','夹碰','首次响应前夹碰','插入预算耗尽','决策 P50 ms'],c2),'',
      '### 撤回初段袋口擦碰','',
      '从滑脱触发到同一物品实际下表面高于袋口50 mm，检查袋口接触传感器峰值冲量是否达到0.004 N·s；未撤离则检查至第一阶段结束。该局部指标不替代全回合安全恢复，未触发滑脱单列。','',
      table(['方法','触发滑脱','撤离前袋口擦碰'],[[m,count(b:=[r for r in recovery if r['method']==m],'slipped'),count(b,'collar_grazing_before_withdrawal')] for m in METHODS]),'',
      '### 配对恢复率差异','',
      '以种子为簇、同模态/顺序配对，做 10,000 次簇 bootstrap。只有三组独立种子，区间是小样本百分位诊断；全零或相同的簇可能导致退化区间，这不表示没有不确定性，不足以确立普遍或强统计意义的领先。','',
      table(['Ours','对照','安全恢复率差','95% 簇 bootstrap 区间'],comparison_rows),'',
      '## 独立遮挡记忆容量诊断','',
      '这是明确标注的合成、平衡输入诊断，不是新的物理成功回合，也不是已训练策略的消融成绩。'
      '前 12 个 token 给出左右占用线索，随后 52 个 token 对两标签完全相同；当前物品身份、顺序和动作捷径均移除。'
      '两种独立分类器重新训练 400 步，用它检验记忆容量，而不把其优势归因到主任务。','',
      table(['诊断分类器','留出样本','准确率'],memory_rows),'',
      '## Figure · 实测轨迹与袋口净空','',
      '图固定选择测试前声明的 elastic_fast、A→B、seed=2000、开启滑脱的配对回合，对比 BC、GPC-OPT 与 Ours τ=0.9。'
      '没有按成功或失败筛选案例；图上保留实测结果，并同时标注全矩阵安全恢复率，避免用单例代替统计。三维部分重建学习控制时段的轨迹及最终物品姿态，不是相机渲染截图。PNG 为 4680×3120，并提供矢量 PDF。','',
      '![Stage C measured trajectories](stage_c_bag/bag_packing_demo.png)','',
      '[论文图 PNG](stage_c_bag/paper/fig_stage_c_bag.png) · [PDF](stage_c_bag/paper/fig_stage_c_bag.pdf) · [SVG](stage_c_bag/paper/fig_stage_c_bag.svg) · [来源清单](stage_c_bag/paper/fig_stage_c_bag_manifest.json)。论文图的逐回合Wilson区间仅作描述，忽略种子簇相关性；配对差异以本报告的簇bootstrap为准。星号标出含插入超时的安全恢复。','',
      '## 按模态完整结果','',table(['方法','模态','成功','二次分槽正确','夹碰'],mode_table),'',
      '## 复现与证据','',
      '本轮入口为 `bag_packing.py fit --flow-steps 5000 --world-steps 3500` → 三个种子的 `evaluate` → `report`。报告在缺少完整物理预算诊断时自动执行 `probe`。本轮没有重新执行 `calibrate` 或 `collect`。本轮Stage C.2复用192条冻结物理轨迹，训练5000步Flow/Joint-FM和每种3500步世界模型，并重新评测288回合。结果按实际数据报告，未达到的验收目标不改写为通过。主矩阵仅使用c2_seed*_final目录；早期部分矩阵因开发组仍悬停而中止，日志保留但不混入最终结果。'
      '并行评测分片使用 `--out outputs/stage_c_bag/c2_seed2000_final --model-root outputs/stage_c_bag --seeds 2000`，其他种子同理。'
      '训练/推理默认 CUDA；所有路径、数据划分、维度、预算、迁移参数和 checkpoint 哈希均保留。','',
      '[协议](stage_c_bag/protocol.json) · [监督与数据审计](stage_c_bag/supervision_audit.json) · [最终姿态指标审计](stage_c_bag/metric_audit.json) · '
      '[训练](stage_c_bag/training.json) · [逐回合结果](stage_c_bag/results.json) · [汇总](stage_c_bag/summary.json) · '
      '[回缩标定](stage_c_bag/calibration.json) · [图像来源](stage_c_bag/figure_manifest.json) · '
      '[保形分组诊断](stage_c_bag/conformal_diagnostics.json) · [保形与预算诊断](stage_c_bag/guidance_probe.json) · [物理预算诊断](stage_c_bag/guidance_probe_physical.json) · [独立计时](stage_c_bag/isolated_timing.json) · '
      '[记忆诊断](stage_c_bag/memory_diagnostic.json) · [PDF 图](stage_c_bag/bag_packing_demo.pdf)','']
    report='\n'.join(lines)
    root=out.parent
    (root/'StageC_Bag_Packing_实验报告.md').write_text(report,encoding='utf-8')
    summary='\n\n'.join(['## Stage C.2 · 双臂动态弹性袋口收纳',verdict,
          '真实 PhysX 弹簧滑动袋口；Full 使用 256D Past/Future。固定身份到槽位映射已移除，槽位提示来自各方法自己的历史编码器；记忆效果以实测为准。',
          '### 常规闭环',table(['方法','N','双物品成功','二次分槽正确','到达第二通道','夹碰','P50 ms'],c1),
          '### 滑脱恢复',table(['方法','N','触发滑脱','实际回缩','双物品成功','撤回重撑','安全恢复','安全恢复且无超时','夹碰','首次响应前夹碰','插入预算耗尽','P50 ms'],c2),
          '[完整 Stage C 报告](StageC_Bag_Packing_实验报告.md) · [逐回合 JSON](stage_c_bag/results.json)',
          '![Stage C measured trajectories](stage_c_bag/bag_packing_demo.png)'])
    backup=out/'previous_reports';backup.mkdir(exist_ok=True)
    for file in (root/'Phase2_GPC_Mamba_实验报告.md',root/'evaluation/results.md'):
        if not (backup/file.name).exists():shutil.copy2(file,backup/file.name)
        text=file.read_text(encoding='utf-8')
        start='<!-- STAGE_C_BAG_BEGIN -->';end='<!-- STAGE_C_BAG_END -->'
        if start in text:
            a=text.index(start);b=text.index(end,a)+len(end);text=text[:a]+text[b:]
        section=summary
        if file.parent.name=='evaluation':
            section=section.replace('(StageC_Bag_Packing_', '(../StageC_Bag_Packing_').replace('(stage_c_bag/','(../stage_c_bag/')
        file.write_text(text.rstrip()+'\n\n'+start+'\n'+section+'\n'+end+'\n',encoding='utf-8')
    print('Stage C reports and figure written; all outcomes retained',flush=True)
