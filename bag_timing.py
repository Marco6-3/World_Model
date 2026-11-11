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


def measure(out):
    out=Path(out)
    models=load_models(out,'cuda')
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
    for method in METHODS:decide(models,histories[0],method,torch.Generator().manual_seed(9100))
    rows=[]
    for i,history in enumerate(histories):
        order=list(METHODS[i%len(METHODS):])+list(METHODS[:i%len(METHODS)])
        for method in order:
            rng=torch.Generator().manual_seed(9100+i)
            torch.cuda.synchronize();start=perf_counter()
            decide(models,history,method,rng)
            torch.cuda.synchronize()
            rows.append(dict(method=method,history_index=i,ms=(perf_counter()-start)*1000))
        print(f'Isolated bag timing {i+1}/12',flush=True)
    summary={m:{'p50_ms':float(np.median([r['ms'] for r in rows if r['method']==m])),
                'mean_ms':float(np.mean([r['ms'] for r in rows if r['method']==m]))} for m in METHODS}
    hashes={p.name:sha256(p.read_bytes()).hexdigest() for p in (out/'trained').glob('*.pt')}
    source_hashes={name:sha256((Path(__file__).resolve().parent/name).read_bytes()).hexdigest()
                   for name in ('bag_scene.py','bag_learning.py','bag_common.py','mamba_wm.py')}
    dump(out/'isolated_timing.json',dict(device=torch.cuda.get_device_name(),torch=str(torch.__version__),checkpoint_hashes=hashes,planner_source_hashes=source_hashes,
         protocol='12 identical held-out second-entry histories per method; warm-up; rotating order; synchronized wall clock',
         histories=sources,rows=rows,summary=summary))
