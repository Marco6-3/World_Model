"""Sequential C.1 execution; stop on failed commands, retain all logs and outcomes."""
from pathlib import Path
import sys
import subprocess
import json
from datetime import datetime

root=Path(__file__).resolve().parent
out=root/'outputs/stage_c_bag'
def run(name,args):
    (out/'run_status.json').write_text(json.dumps(dict(stage=name,start=datetime.now().isoformat(),command=args)),encoding='utf-8')
    with (out/(name+'.log')).open('w',encoding='utf-8') as f:
        subprocess.run([sys.executable,*args],cwd=root,stdout=f,stderr=subprocess.STDOUT,check=True)

if __name__=='__main__':
    run('collect',['bag_packing.py','collect'])
    from stage_c1_checks import check
    data=check(out)
    assert data['episodes']==192
    for key in ('spread_and_hold','slip_withdraw_reopen'):
        s=data['schemes'][key]
        if s!={'n':48,'success':48,'collision':0}:
            raise RuntimeError('Expert collection target failed: '+key+' '+str(s))
    run('fit',['bag_packing.py','fit','--flow-steps','5000','--world-steps','3500'])
    for seed in (2000,2001,2002):
        run('evaluate_'+str(seed),['bag_packing.py','evaluate','--out',str(out/f'seed{seed}'),
             '--model-root',str(out),'--seeds',str(seed),'--execution-context','stage_c1_sequential'])
    run('timing',['-c',"import bag_packing,torch; torch.set_num_threads(1); torch.set_float32_matmul_precision('high'); from bag_timing import measure; measure(bag_packing.OUT)"])
    run('report',['bag_packing.py','report'])
    (out/'run_status.json').write_text(json.dumps(dict(stage='complete',time=datetime.now().isoformat())),encoding='utf-8')
