"""Matched-input timing after all Stage C simulator jobs finish."""
import json
import sys
if 'torch' not in sys.modules:sys.path.insert(0,r'C:\isaacsim_env\phase2_cuda')
from pathlib import Path
from time import perf_counter
from hashlib import sha256
import numpy as np
import torch
from bag_packing import METHODS, dump
from bag_common import history_indices
from bag_learning import load_models, decide, clean_observations


def _load_heldout_histories(out):
    histories=[];sources=[]
    for path in sorted((out/'dataset').glob('*spread_and_hold.npz')):
        if int(path.name.split('_')[0]) not in (1006,1007):continue
        with np.load(path) as raw:
            metadata=json.loads(raw['metadata'].item())
            t=metadata['learn_ranges'][1][0]
            obs,_,_=clean_observations(raw['observations'],raw['action'],metadata)
            histories.append(obs[history_indices(t)])
            sources.append({'file':path.name,'control_tick':t,'sha256':sha256(path.read_bytes()).hexdigest()})
    assert len(histories)==12
    return histories,sources


def _benchmark_mode(models,histories,*,use_cuda_graph=True):
    for method in METHODS:
        decide(models,histories[0],method,torch.Generator().manual_seed(9100),use_cuda_graph=use_cuda_graph)
    torch.cuda.synchronize()
    rows=[]
    for i,history in enumerate(histories):
        order=list(METHODS[i%len(METHODS):])+list(METHODS[:i%len(METHODS)])
        for method in order:
            rng=torch.Generator().manual_seed(9100+i)
            torch.cuda.synchronize();start=perf_counter()
            decide(models,history,method,rng,use_cuda_graph=use_cuda_graph)
            torch.cuda.synchronize()
            rows.append(dict(method=method,history_index=i,ms=(perf_counter()-start)*1000))
        print(f"Isolated bag timing (cuda_graph={use_cuda_graph}) {i+1}/12",flush=True)
    summary={}
    for m in METHODS:
        vals=[r['ms'] for r in rows if r['method']==m]
        summary[m]={
            'p50_ms':float(np.median(vals)),
            'mean_ms':float(np.mean(vals)),
            'p95_ms':float(np.percentile(vals,95)),
            'max_ms':float(np.max(vals)),
            'ticks_at_30hz_p50':max(1,int(np.ceil(np.median(vals)/(1000.0/30.0)))),
        }
    return rows,summary


def measure(out,*,target_file='isolated_timing.json',use_cuda_graph=True):
    out=Path(out)
    models=load_models(out,'cuda')
    histories,sources=_load_heldout_histories(out)
    rows,summary=_benchmark_mode(models,histories,use_cuda_graph=use_cuda_graph)
    hashes={p.name:sha256(p.read_bytes()).hexdigest() for p in (out/'trained').glob('*.pt')}
    source_hashes={name:sha256((Path(__file__).resolve().parent/name).read_bytes()).hexdigest()
                   for name in ('bag_scene.py','bag_learning.py','bag_common.py','mamba_wm.py')}
    dump(out/target_file,dict(device=torch.cuda.get_device_name(),torch=str(torch.__version__),
         use_cuda_graph=use_cuda_graph,checkpoint_hashes=hashes,planner_source_hashes=source_hashes,
         protocol='12 identical held-out second-entry histories per method; warm-up; rotating order; synchronized wall clock',
         histories=sources,rows=rows,summary=summary))


def measure_realtime_comparison(out,*,target_file='realtime_timing.json'):
    out=Path(out)
    models=load_models(out,'cuda')
    histories,sources=_load_heldout_histories(out)
    graph_rows,graph_summary=_benchmark_mode(models,histories,use_cuda_graph=True)
    eager_rows,eager_summary=_benchmark_mode(models,histories,use_cuda_graph=False)
    baseline_path=out/'isolated_timing.json'
    baseline_summary=json.loads(baseline_path.read_text(encoding='utf-8'))['summary'] if baseline_path.exists() else None
    speedups={}
    for m in METHODS:
        entry={'eager_parallel_scan_p50_ms':eager_summary[m]['p50_ms'],
               'cuda_graph_p50_ms':graph_summary[m]['p50_ms'],
               'cuda_graph_p95_ms':graph_summary[m]['p95_ms'],
               'ticks_at_30hz_p50':graph_summary[m]['ticks_at_30hz_p50']}
        if baseline_summary and m in baseline_summary:
            base_p50=float(baseline_summary[m]['p50_ms'])
            entry['pre_optimization_p50_ms']=base_p50
            entry['speedup_vs_pre_optimization']=round(base_p50/max(graph_summary[m]['p50_ms'],1e-6),2)
        speedups[m]=entry
    hashes={p.name:sha256(p.read_bytes()).hexdigest() for p in (out/'trained').glob('*.pt')}
    source_hashes={name:sha256((Path(__file__).resolve().parent/name).read_bytes()).hexdigest()
                   for name in ('bag_scene.py','bag_learning.py','bag_common.py','mamba_wm.py')}
    dump(out/target_file,dict(device=torch.cuda.get_device_name(),torch=str(torch.__version__),
         control_dt_ms=1000.0*4.0/120.0,checkpoint_hashes=hashes,planner_source_hashes=source_hashes,
         protocol='12 identical held-out second-entry histories per method; warm-up; rotating order; synchronized wall clock; compares pre-optimization eager loop, parallel-scan eager, and parallel-scan + CUDA Graph',
         histories=sources,summary=speedups,cuda_graph={'summary':graph_summary,'rows':graph_rows},
         eager_parallel_scan={'summary':eager_summary,'rows':eager_rows}))


if __name__=='__main__':
    measure_realtime_comparison(Path(__file__).resolve().parent/'outputs/stage_c_bag')
