"""CUDA conditional flows, 256D selective memory, and action-conditioned bag dynamics."""
from pathlib import Path
import sys
if 'torch' not in sys.modules:sys.path.insert(0,r'C:\isaacsim_env\phase2_cuda')
import json
from hashlib import sha256
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from bag_packing import MODES, dump
from bag_common import OBS_DIM, ACTION_DIM, HISTORY, HORIZON, history_indices, THRESHOLD, measure_trace, effective_clearance
from mamba_wm import UnifiedMambaWorldModel
from learn import flow_loss, temporal_noise, safety_loss
from inference import flow_search, Search

ACTION_MEAN = [-.048,-.14,.165,0.,.06,.14,.018,.018]
ACTION_STD = [.06,.08,.06,.07,.13,.10,.014,.014]
STATE_SCALE = [.10,1.,1.,.05]
GEO_SCALE = [.15,1.,.10,.25,.25,.10,.25,.25,1.,1.]
SPLITS = ([1000,1001,1002,1003],[1004,1005],[1006,1007])


class Residual(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.n1=nn.GroupNorm(8,width);self.n2=nn.GroupNorm(8,width)
        self.c1=nn.Conv1d(width,width,3,padding=1);self.c2=nn.Conv1d(width,width,3,padding=1)
        self.cond=nn.Linear(width,width*2)
    def forward(self,x,c):
        scale,bias=self.cond(c).chunk(2,-1)
        y=self.c1(F.silu(self.n1(x)))
        return x+self.c2(F.silu(self.n2(y)*(1+scale[:,:,None])+bias[:,:,None]))


class BagFlow(nn.Module):
    def __init__(self,channels=8):
        super().__init__()
        self.channels=channels
        self.register_buffer('freq',torch.exp(torch.linspace(0,-9.21,32)))
        self.condition=nn.Sequential(nn.Linear(4*OBS_DIM+64+2,128),nn.SiLU(),nn.Linear(128,128))
        self.embed=nn.Conv1d(channels,128,3,padding=1)
        self.blocks=nn.ModuleList([Residual(128) for _ in range(3)])
        self.out=nn.Conv1d(128,channels,3,padding=1)
    def forward(self,x,tau,obs,stage=None):
        t=tau*self.freq[None]*1000
        if stage is None:raise ValueError('Flow requires observation-derived slot hint at inference')
        c=self.condition(torch.cat((obs[:,-4:].flatten(1),t.sin(),t.cos(),stage),-1))
        z=self.embed(x.transpose(1,2))
        for layer in self.blocks:z=layer(z,c)
        return self.out(F.silu(z)).transpose(1,2)


class ShortPast(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Flatten(1),nn.Linear(4*OBS_DIM,256),nn.SiLU(),nn.Linear(256,256))
        self.posterior=nn.Linear(256,3)
    def forward(self,obs):
        b=self.net(obs[:,-4:]);return b,self.posterior(b)


class FourFuture(nn.Module):
    def __init__(self):
        super().__init__()
        self.cell=nn.Sequential(nn.Linear(4+8,64),nn.SiLU(),nn.Linear(64,4))
    def forward(self,z,action):
        rows=[]
        for a in action.unbind(1):
            z=z+.15*self.cell(torch.cat((z,a),-1));rows.append(z)
        return torch.stack(rows,1)


class BagWorld(UnifiedMambaWorldModel):
    def __init__(self,kind='full'):
        super().__init__(OBS_DIM,8,10,d_z=4 if kind=='four' else 256,belief_dim=256,context_dim=3)
        self.kind=kind
        if kind=='short':self.past=ShortPast()
        if kind=='four':self.future=FourFuture()
        self.slots=nn.Linear(256,2)


class Models(nn.Module):
    def __init__(self,kind='full'):
        super().__init__()
        self.flow=BagFlow();self.joint=BagFlow(12);self.world=BagWorld(kind)
        self.register_buffer('mean',torch.zeros(OBS_DIM));self.register_buffer('std',torch.ones(OBS_DIM))
        self.register_buffer('am',torch.tensor(ACTION_MEAN));self.register_buffer('asd',torch.tensor(ACTION_STD))
        self.register_buffer('ss',torch.tensor(STATE_SCALE));self.register_buffer('gs',torch.tensor(GEO_SCALE))
    def encode(self,obs):return (obs-self.mean)/self.std
    def actions(self,a):return (a-self.am)/self.asd
    def physical(self,a):return a*self.asd+self.am


def load_windows(out,seeds):
    items=[];file_stats=[]
    for path in sorted((out/'dataset').glob('*.npz')):
        if int(path.name.split('_')[0]) not in seeds:continue
        with np.load(path,allow_pickle=False) as data:
            metadata=json.loads(data['metadata'].item())
            obs=data['observations'];action=data['action'];states=data['state'];truth=data['truth']
            expert=data['expert'];learn=data['learn'];stage=data['stage']
            file_stats.append(dict(file=path.name,sha256=sha256(path.read_bytes()).hexdigest(),
                                   success=metadata['success'],collision=metadata['collision'],
                                   mode=metadata['mode'],order=metadata['order'],scheme=metadata['method']))
            handoff=int(metadata['learn_ranges'][1][0])
            times=np.unique(np.r_[np.flatnonzero(learn)[::3],handoff])
            for t in times:
                ix=np.minimum(np.arange(t,t+HORIZON),len(action)-1)
                hist=obs[history_indices(t)]
                mask=expert[ix] & learn[ix] & (stage[ix]==stage[t]) & bool(metadata["success"])
                # Once a demonstrated suffix fails, it cannot be reintroduced as expert supervision.
                mask=np.logical_and.accumulate(mask)
                items.append(dict(obs=hist,action=action[ix],state=states[ix],geo=truth[ix],
                                  slots=truth[max(0,t-1),-2:],expert=mask,stage=stage[t],
                                  second_entry=bool(t==handoff),
                                  hint=np.eye(2,dtype=np.float32)[int(metadata['order']) if stage[t]==0 else 1-int(metadata['order'])],
                                  mode=MODES.index(metadata['mode']),seed=metadata['seed']))
    if not items:raise ValueError(f'No bag data for {seeds}')
    return {k:torch.as_tensor(np.asarray([r[k] for r in items])) for k in items[0]},file_stats


def transfer_future(world):
    path=Path(__file__).resolve().parent/'outputs/phase2_1/spring/world_256.pt'
    source=torch.load(path,map_location='cpu',weights_only=True)['weights']
    target=world.state_dict();copied=[]
    for key,value in source.items():
        name=key.removeprefix('world.')
        if key.startswith('world.future.') and name in target and target[name].shape==value.shape:
            target[name]=value;copied.append(name)
    world.load_state_dict(target)
    return dict(source=str(path),sha256=sha256(path.read_bytes()).hexdigest(),copied_parameters=copied,
                note='Shape-compatible future backbone only; new 256D Past and task readouts are trained afresh')


@torch.no_grad()
def validate(model,data):
    errors=[];slot_correct=[];replay_correct=[];mode_correct=[]
    for ix in torch.arange(len(data['obs']),device=model.mean.device).split(32):
        obs=model.encode(data['obs'][ix]);a=model.actions(data['action'][ix])
        stage=F.one_hot(data['stage'][ix].long(),3).float()
        b,logits=model.world.encode(obs)
        xi=F.one_hot(data['mode'][ix].long(),3).float()
        z=model.world.latent(b,stage,a,xi)
        pred=model.world.decode_safety(z)*model.ss
        errors.append((pred-data['state'][ix]).abs())
        occupied=(data['slots'][ix].sum(-1)==1)
        replay_correct.extend((model.world.slots(b)[occupied].argmax(-1)==data['slots'][ix][occupied].argmax(-1)).tolist())
        # Evaluate memory BEFORE the expert's target-dependent actions can reveal its choice.
        mask=occupied & data['second_entry'][ix].bool()
        slot_correct.extend((model.world.slots(b)[mask].argmax(-1)==data['slots'][ix][mask].argmax(-1)).tolist())
        mode_correct.extend((logits.argmax(-1)==data['mode'][ix]).tolist())
    return dict(windows=len(data['obs']),safety_mae=torch.cat(errors).mean((0,1)).tolist(),
                occupied_slot_accuracy=float(np.mean(slot_correct)) if slot_correct else None,
                occupied_slot_windows=len(slot_correct),mode_accuracy=float(np.mean(mode_correct)),
                occupied_slot_scope='second-entry handoff, before target-dependent learned actions',
                all_occupied_replay_accuracy=float(np.mean(replay_correct)) if replay_correct else None,
                all_occupied_replay_windows=len(replay_correct))


def fit(args):
    # Recompute outcome metadata from recorded geometry before freezing dataset hashes.
    audits=[]
    for path in sorted((args.out/'dataset').glob('*.npz')):
        with np.load(path,allow_pickle=False) as raw:
            arrays={k:raw[k] for k in raw.files}
        metadata=json.loads(arrays['metadata'].item())
        horizontal=arrays.get('horizontal_clearance',arrays['state'][:,0].copy())
        arrays['horizontal_clearance']=horizontal.copy()
        for t,frame in enumerate(arrays['frames']):
            item=0 if arrays['stage'][t]==0 else 1
            clear=effective_clearance(horizontal[t],frame[8+3*item:11+3*item],frame[5:8],item)
            arrays['state'][t,0]=clear;arrays['frames'][t,14]=clear
        metadata['clearance_definition']='3D lower-bound: XY oriented-envelope margin plus vertical object bounding sphere and 12 mm tip proxy'
        from bag_audit import vertical_extent
        from bag_common import COLLAR_Z
        for item in (0,1):
            extent=vertical_extent(arrays['poses'],item);pos=arrays['poses'][:,item,:3]
            occupied=(pos[:,2]-extent<COLLAR_Z+.045)&(pos[:,2]+extent>COLLAR_Z-.018)&(np.abs(pos[:,1])<.10)
            mask=arrays['stage']==item
            arrays['state'][mask,1]=occupied[mask];arrays['frames'][mask,15]=occupied[mask]
        slips=[e['t'] for e in metadata['events'] if e['event']=='collar_slip']
        measurement=measure_trace(arrays['frames'],arrays['stage'],arrays['learn'],arrays['attached'],
                                  metadata['order'],slips[0] if slips else None)
        previous={k:metadata.get(k) for k in ('second_slot','second_slot_correct','withdraw_reopen','safe_recovery')}
        metadata.update(measurement)
        metadata['safe_recovery']=bool(measurement['withdraw_reopen'] and metadata['success'] and not metadata['collision'])
        metadata['measurement_version']='stage_c1_physical_entry_and_same_item_recovery'
        arrays['metadata']=json.dumps(metadata)
        temporary=path.with_suffix('.tmp')
        with temporary.open('wb') as stream:np.savez_compressed(stream,**arrays)
        temporary.replace(path)
        audits.append(dict(file=path.name,previous=previous,measured={k:metadata[k] for k in previous}))
    dump(args.out/'dataset_measurement_audit.json',audits)
    train_info=[load_windows(args.out,seeds) for seeds in SPLITS]
    data=[{k:v.to(args.device) for k,v in part.items()} for part,_ in train_info]
    train,val,test=data
    folder=args.out/'trained';folder.mkdir(exist_ok=True)
    torch.manual_seed(731)
    model=Models().to(args.device)
    model.mean.copy_(train['obs'].reshape(-1,OBS_DIM).mean(0))
    model.std.copy_(train['obs'].reshape(-1,OBS_DIM).std(0).clamp_min(.025))
    expert_ix=(train['expert'].sum(-1)>=6).nonzero().flatten()
    if len(expert_ix)<50:raise ValueError(f'Only {len(expert_ix)} valid expert windows; repair collection before training')
    opt=torch.optim.AdamW(list(model.flow.parameters())+list(model.joint.parameters()),lr=2e-4)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,args.flow_steps)
    for step in range(1,args.flow_steps+1):
        ix=expert_ix[torch.randint(len(expert_ix),(128,),device=args.device)]
        obs=model.encode(train['obs'][ix]);a=model.actions(train['action'][ix]);mask=train['expert'][ix]
        stage=F.one_hot(train['stage'][ix].long(),3).float()
        joint=torch.cat((a,train['state'][ix]/model.ss),-1)
        loss=flow_loss(model.flow,a,obs,train['hint'][ix],8,mask,lambda_smooth=0.)
        jloss=flow_loss(model.joint,joint,obs,train['hint'][ix],8,mask,lambda_smooth=0.)
        opt.zero_grad(set_to_none=True);(loss+jloss).backward()
        nn.utils.clip_grad_norm_(list(model.flow.parameters())+list(model.joint.parameters()),2.)
        opt.step();schedule.step()
        if step%200==0 or step==1:print(f'Flow {step}/{args.flow_steps}: {loss.item():.4f}, joint={jloss.item():.4f}',flush=True)
    common={k:v.detach().cpu() for k,v in model.state_dict().items() if not k.startswith('world.')}
    torch.save(common,folder/'flows.pt')
    report=dict(device=args.device,torch=str(torch.__version__),gpu=torch.cuda.get_device_name(),
                split_seeds=SPLITS,flow_updates=args.flow_steps,world_updates=args.world_steps,
                expert_windows=len(expert_ix),files=[f for _,files in train_info for f in files],models={})
    for kind in ('full','short','four'):
        torch.manual_seed(731)
        model=Models(kind).to(args.device);model.load_state_dict(common,strict=False)
        model.flow.requires_grad_(False);model.joint.requires_grad_(False)
        transfer=transfer_future(model.world) if kind!='four' else None
        pos=train['state'][...,3]>=THRESHOLD
        weight=((pos.numel()-pos.sum())/pos.sum().clamp_min(1)).sqrt().clamp(1,12)
        model.world.event_log_weight.copy_(weight.log())
        opt=torch.optim.AdamW(model.world.parameters(),lr=5e-4,weight_decay=1e-4)
        schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,args.world_steps,eta_min=3e-5)
        for step in range(1,args.world_steps+1):
            ix=torch.randint(len(train['obs']),(48,),device=args.device)
            obs=model.encode(train['obs'][ix]);a=model.actions(train['action'][ix])
            stage=F.one_hot(train['stage'][ix].long(),3).float();xi=F.one_hot(train['mode'][ix].long(),3).float()
            b,logits=model.world.encode(obs);z=model.world.latent(b,stage,a,xi)
            loss=safety_loss(model.world.decode_safety(z,raw=True),train['state'][ix]/model.ss,weight)
            loss+=F.smooth_l1_loss(model.world.g_state(z),train['geo'][ix]/model.gs)
            loss+=.4*F.binary_cross_entropy_with_logits(model.world.slots(b),train['slots'][ix])
            loss+=.3*F.cross_entropy(logits,train['mode'][ix].long())
            opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.world.parameters(),5.)
            opt.step();schedule.step()
            if step%100==0 or step==1:print(f'{kind} WM {step}/{args.world_steps}: {loss.item():.4f}',flush=True)
        model.eval()
        report['models'][kind]=dict(past_dimension=256,future_dimension=4 if kind=='four' else 256,
             parameters=sum(p.numel() for p in model.world.parameters()),transfer=transfer,
             validation=validate(model,val),heldout=validate(model,test))
        torch.save(model.state_dict(),folder/f'{kind}.pt')
        dump(args.out/'training.json',report)
    memory_probe(args.out,train_info)


def load_models(out,device):
    result={}
    for kind in ('full','short','four'):
        model=Models(kind).to(device).eval().requires_grad_(False)
        model.load_state_dict(torch.load(out/'trained'/f'{kind}.pt',map_location=device,weights_only=True))
        result[kind]=model
    return result


def safety_risk(state,action):
    clear,occupied,progress,impulse=state.unbind(-1)
    pinch=4500.*occupied*F.relu(.028-clear).square()+80.*impulse
    safe_gate=torch.sigmoid((clear-.018)/.004)
    withdraw=(1-safe_gate)*(action[...,5]-.18)*1.2
    cost=(pinch-.18*safe_gate*progress-withdraw).mean(-1)
    closing=F.relu((clear[:,-12]-clear[:,-1])/12)
    projected=clear[:,-1]-closing*24*(1-progress[:,-1].clamp(0,1))
    return cost+2800*F.relu(.035-projected).square()


def decide(models,history,method,rng):
    kind='short' if method in ('BC H4','Joint-FM','Ours Short-MLP H4') else 'four' if method=='Ours 4D-WM' else 'full'
    model=models[kind];joint=method=='Joint-FM'
    configs={'BC H4':Search(1),'Joint-FM':Search(1,tau_start=.7),'GPC-RANK':Search(16),
             'GPC-OPT':Search(1,20),'Ours Short-MLP H4':Search(8,tau_start=.7),
             'Ours 4D-WM':Search(8,tau_start=.7),'Ours tau=0.7':Search(8,tau_start=.7),
             'Ours tau=0.9':Search(8,tau_start=.9)}
    config=configs[method]
    raw=torch.as_tensor(history,dtype=torch.float32,device=model.mean.device)[None]
    obs=model.encode(raw);stage=raw[:,-1,25:28]
    flow=model.joint if joint else model.flow
    noise=temporal_noise((config.particles,HORIZON,flow.channels),generator=rng).to(obs.device)
    with torch.no_grad():
        belief,logits=model.world.encode(obs);prob=logits.softmax(-1)[0]
        slots=model.world.slots(belief).sigmoid()[0]
        hint=(1-slots)[None] if int(stage.argmax())==1 else raw[:,-1,28:30]
        hint=hint/hint.sum(-1,keepdim=True).clamp_min(1e-6)
    emergency=bool(raw[0,-1,4]<.085 or raw[0,-1,4]-raw[0,-2,4]<-.006)
    if config.particles>1 and int(stage.argmax())==1:
        noise[:config.particles//2,:,3]-=.35
        noise[config.particles//2:,:,3]+=.35
    if emergency and config.particles>=8:
        # Shared, observation-only candidate priors for every multi-particle planner.
        noise[:2,:,5]+=1.2
        noise[:2,:,4]-=.4
        noise[:2,:,6]+=.5
        noise[2:4,:,1]-=.5
        noise[2:4,:,6]-=.5
    def field(x,t):return flow(x,x.new_full((len(x),1),t),obs.expand(len(x),-1,-1),hint.expand(len(x),-1))
    def project(x):
        action=model.physical(x[...,:8])
        low=action.new_tensor([-.09,-.23,.08,-.09,-.06,.04,.006,.006])
        high=action.new_tensor([.02,-.04,.24,.09,.25,.25,.038,.038])
        action=torch.maximum(low,torch.minimum(high,action))
        previous=raw[:,-1,-8:-2].expand(len(x),-1);steps=[]
        for a in action.unbind(1):
            delta=(a[:,:6]-previous).reshape(-1,2,3)
            previous=previous+(delta*(.0075/delta.norm(dim=-1,keepdim=True).clamp_min(.0075))).flatten(1)
            steps.append(torch.cat((previous,a[:,6:]),-1))
        normalized=model.actions(torch.stack(steps,1))
        return torch.cat((normalized,x[...,8:]),-1) if joint else normalized
    def risk(x):
        nonlocal belief,prob,slots
        physical=model.physical(x[...,:8])
        if joint:return safety_risk(x[...,8:]*model.ss,physical)
        if belief is None:
            with torch.no_grad():
                belief,logits=model.world.encode(obs);prob=logits.softmax(-1)[0]
                slots=model.world.slots(belief).sigmoid()[0]
        n=len(x)
        z=model.world.latent(belief.expand(n*3,-1),stage.expand(n*3,-1),
                            x.repeat_interleave(3,0),torch.eye(3,device=x.device).repeat(n,1))
        safety=model.world.decode_safety(z)*model.ss
        base=safety_risk(safety,physical.repeat_interleave(3,0)).reshape(n,3)@prob
        pos=physical[...,3]
        occupied_overlap=slots[0]*torch.exp(-((pos+.048)/.025).square())+slots[1]*torch.exp(-((pos-.048)/.025).square())
        target=.048*(slots[0]-slots[1]) if int(stage.argmax())==1 else (.048 if raw[0,-1,29]>.5 else -.048)
        return base+35*(pos-target).square().mean(-1)+4*occupied_overlap[:,-8:].mean(-1)
    remaining=max(1,10-round(config.tau_start*10))
    guidance=.5*10/remaining
    if emergency and config.tau_start<1:guidance=max(guidance,1.2 if config.tau_start==.7 else 2.5)
    result=flow_search(noise,field,risk,project,config,opt_lr=.08,guidance=guidance)
    prediction=int(slots.argmax()) if slots is not None else None
    return model.physical(result[...,:8]).cpu().numpy(),dict(predicted_occupied_slot=prediction)


def memory_probe(out,train_info):
    """Balanced observation-only diagnostic; never passed off as a physical rollout."""
    rows=[]
    for split,(data,_) in zip(('train','validation','heldout'),train_info):
        mask=data['slots'].sum(-1)==1
        source=data['obs'][mask].numpy()
        examples=[];labels=[]
        for hist in source[::4]:
            # Retain early item-position evidence, erase all shortcuts after the blackout.
            visible=np.where((hist[:,14]>0)&(hist[:,18]>0))[0]
            if not len(visible):continue
            for side in (0,1):
                x=np.zeros((64,5),np.float32)
                x[:12,0]=-.048 if side==0 else .048
                x[:12,1]=hist[:12,4];x[:12,2]=1
                x[12:,3]=1
                x[:,4]=np.arange(64)/64
                examples.append(x);labels.append(side)
        rows.append((torch.tensor(np.asarray(examples),device='cuda'),torch.tensor(labels,device='cuda')))
    from mamba_wm import MambaPast
    report={'type':'balanced counterfactual observation diagnostic, not physical task success',
            'main_task_shortcut':'Identity mapping removed in Stage C.1; this remains a separate synthetic diagnostic', 'models':{}}
    for name in ('Short-MLP H4','Mamba-Past 256D'):
        torch.manual_seed(932)
        model=(nn.Sequential(nn.Flatten(1),nn.Linear(20,128),nn.SiLU(),nn.Linear(128,2))
               if name.startswith('Short') else MambaPast(5,256,2)).cuda()
        opt=torch.optim.AdamW(model.parameters(),lr=.001)
        for _ in range(400):
            x,y=rows[0];ix=torch.randint(len(x),(64,),device='cuda')
            pred=model(x[ix,-4:]) if name.startswith('Short') else model(x[ix])[1]
            loss=F.cross_entropy(pred,y[ix]);opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        model.eval();values={}
        with torch.no_grad():
            for split,(x,y) in zip(('train','validation','heldout'),rows):
                predictions=[]
                for chunk in x.split(64):predictions.append(model(chunk[:,-4:]) if name.startswith('Short') else model(chunk)[1])
                values[split]={'n':len(y),'accuracy':float((torch.cat(predictions).argmax(-1)==y).float().mean())}
        report['models'][name]=values
    dump(out/'memory_diagnostic.json',report)
