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
from inference import Search, clipped_descent, flow_search

ACTION_MEAN = [-.048,-.14,.165,0.,.06,.14,.018,.018]
ACTION_STD = [.06,.08,.06,.07,.13,.10,.014,.014]
STATE_SCALE = [.10,1.,1.,.05]
GEO_SCALE = [.15,1.,.10,.25,.25,.10,.25,.25,1.,1.]
SPLITS = ([1000,1001,1002,1003],[1004,1005],[1006,1007])
RECOVERY_FRACTION = .15
PARTICLES = 16


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
        self.register_buffer('conformal_q',torch.zeros(()))
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
        self._planners={}
    def encode(self,obs):return (obs-self.mean)/self.std
    def actions(self,a):return (a-self.am)/self.asd
    def physical(self,a):return a*self.asd+self.am


def clean_observations(observations, actions, metadata):
    """Impute only the nine forced-release frames in frozen v2 recordings.

    v2 overwrote the command before recording it, so its original value cannot
    be recovered. Hold the pre-fault command; never change the physical sensors.
    Explicit withdrawal/regrasp commands outside that interval remain intact.
    """
    obs=observations.copy(); action=actions.copy(); corrected=0
    for event in metadata['events']:
        if event['event']!='collar_slip':continue
        start=int(event['t']);stop=min(start+9,len(action))
        command=float(obs[start,-2])
        action[start:stop,6]=command
        obs[start+1:stop+1,-2]=command
        corrected+=stop-start
    return obs,action,corrected


def load_windows(out,seeds):
    items=[];file_stats=[]
    for path in sorted((out/'dataset').glob('*.npz')):
        if int(path.name.split('_')[0]) not in seeds:continue
        with np.load(path,allow_pickle=False) as data:
            metadata=json.loads(data['metadata'].item())
            obs,action,corrected=clean_observations(data['observations'],data['action'],metadata)
            states=data['state'];truth=data['truth']
            expert=data['expert'];learn=data['learn'];stage=data['stage']
            file_stats.append(dict(file=path.name,sha256=sha256(path.read_bytes()).hexdigest(),
                                   success=metadata['success'],collision=metadata['collision'],
                                   mode=metadata['mode'],order=metadata['order'],scheme=metadata['method'],
                                   imputed_command_frames=corrected))
            handoff=int(metadata['learn_ranges'][1][0])
            times=np.unique(np.r_[np.flatnonzero(learn)[::3],handoff])
            scheme=metadata['method']
            slips=[int(e['t']) for e in metadata['events'] if e['event']=='collar_slip']
            resumes=[int(e['t'])+24 for e in metadata['events'] if e['event']=='resume_entry']
            recovery_start=slips[0] if slips else len(action)
            recovery_stop=resumes[0] if resumes else recovery_start
            recovery_times=[int(t) for t in times if t<recovery_stop and t+HORIZON>recovery_start]
            # Fixed, outcome-independent sampling, shared by Flow and Joint-FM.
            sample_rng=np.random.default_rng(int(sha256(path.name.encode()).hexdigest()[:8],16))
            retained=set(sample_rng.choice(recovery_times,int(np.floor(len(recovery_times)*RECOVERY_FRACTION)),replace=False).tolist())
            file_stats[-1].update(recovery_windows=len(recovery_times),retained_recovery_windows=len(retained))
            for t in times:
                ix=np.minimum(np.arange(t,t+HORIZON),len(action)-1)
                hist=obs[history_indices(t)]
                mask=expert[ix] & learn[ix] & (stage[ix]==stage[t]) & bool(metadata["success"])
                if scheme not in ('spread_and_hold','slip_withdraw_reopen'):mask[:]=False
                recovery=bool(t in recovery_times)
                if scheme=='slip_withdraw_reopen' and recovery and int(t) not in retained:mask[:]=False
                # Once a demonstrated suffix fails, it cannot be reintroduced as expert supervision.
                mask=np.logical_and.accumulate(mask)
                items.append(dict(obs=hist,action=action[ix],state=states[ix],geo=truth[ix],
                                  slots=truth[max(0,t-1),-2:],expert=mask,stage=stage[t],
                                  second_entry=bool(t==handoff),
                                  recovery=recovery,retained_recovery=bool(int(t) in retained),
                                  hint=np.eye(2,dtype=np.float32)[int(metadata['order']) if stage[t]==0 else 1-int(metadata['order'])],
                                  mode=MODES.index(metadata['mode']),seed=metadata['seed']))
    if not items:raise ValueError(f'No bag data for {seeds}')
    return {k:torch.as_tensor(np.asarray([r[k] for r in items])) for k in items[0]},file_stats


@torch.no_grad()
def clearance_scores(model,data):
    """Simultaneous horizon scores under the inferred dynamics posterior."""
    scores=[];errors=[]
    for ix in torch.arange(len(data['obs']),device=model.mean.device).split(32):
        obs=model.encode(data['obs'][ix]);a=model.actions(data['action'][ix])
        stage=F.one_hot(data['stage'][ix].long(),3).float()
        b,logits=model.world.encode(obs);n=len(ix)
        z=model.world.latent(b.repeat_interleave(3,0),stage.repeat_interleave(3,0),
                            a.repeat_interleave(3,0),torch.eye(3,device=obs.device).repeat(n,1))
        clear=(model.world.decode_safety(z)[...,0]*model.ss[0]).reshape(n,3,HORIZON)
        predicted=(clear*logits.softmax(-1)[...,None]).sum(1)
        error=predicted-data['state'][ix,:,0]
        scores.append(error.max(-1).values);errors.append(error.abs().flatten())
    return torch.cat(scores),torch.cat(errors)


@torch.no_grad()
def calibrate_conformal(model,val,test,alpha=.10):
    scores,_=clearance_scores(model,val);n=len(scores)
    rank=int(np.ceil((n+1)*(1-alpha)))
    if rank>n:raise ValueError('Insufficient calibration windows for finite split conformal quantile')
    q=scores.sort().values[rank-1].clamp_min(0)
    model.world.conformal_q.copy_(q)
    heldout,errors=clearance_scores(model,test)
    return dict(alpha=alpha,nominal_coverage=1-alpha,calibration_windows=n,quantile_rank=rank,
                conformal_q_m=float(q),validation_coverage=float((scores<=q).float().mean()),
                conformal_coverage=float((heldout<=q).float().mean()),test_windows=len(heldout),
                clearance_mae_m=float(errors.mean()),score='max over all 24 steps of posterior-mean predicted minus measured clearance',
                scope='Empirical held-out simultaneous-window coverage; overlapping windows are dependent, no unconditional episode guarantee')


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
    # Frozen C.1 physical traces stay byte-identical; clean command leakage in memory.
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
                protocol_version='stage_c2',python=sys.executable,recovery_fraction=RECOVERY_FRACTION,
                world_windows=len(train['obs']),expert_windows=len(expert_ix),files=[f for _,files in train_info for f in files],models={})
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
             validation=validate(model,val),heldout=validate(model,test),
             conformal=calibrate_conformal(model,val,test))
        torch.save(model.state_dict(),folder/f'{kind}.pt')
        dump(args.out/'training.json',report)
    memory_probe(args.out,train_info)
    guidance_vs_ranking_probe(args.out)


def load_models(out,device):
    result={}
    for kind in ('full','short','four'):
        model=Models(kind).to(device).eval().requires_grad_(False)
        model.load_state_dict(torch.load(out/'trained'/f'{kind}.pt',map_location=device,weights_only=True))
        result[kind]=model
    return result


def safety_risk(state,action,conformal_q=0.,*,emergency=False):
    clear,occupied,progress,impulse=state.unbind(-1)
    clear_lb=clear-conformal_q
    pinch=4500.*occupied*F.relu(.028-clear_lb).square()+80.*impulse
    safe_gate=torch.sigmoid((clear_lb-.018)/.0005)
    # Aperture collapse disables downward attraction until the collar reopens.
    if torch.is_tensor(emergency):
        safe_gate=safe_gate*(1.-emergency)
        terminal_margin=.018+.017*emergency
    else:
        if emergency:safe_gate=safe_gate*0
        terminal_margin=.035 if emergency else .018
    withdraw=4.*(1-safe_gate)*F.relu(.240-action[...,5])
    insert=.35*safe_gate*(action[...,5]-.055)
    # The linear term has a shallow holding equilibrium under a strong flow
    # prior. A smooth metric-scaled attraction removes that equilibrium once
    # the conformal gate opens. Stop-gradient on its gate prevents reducing
    # predicted clearance merely to switch off the progress potential.
    insert=insert+.35*safe_gate.detach()*((action[...,5]-.055)/.10).square()
    cost=(pinch-.18*safe_gate*progress+withdraw+insert).mean(-1)
    closing=F.relu((clear_lb[:,-12]-clear_lb[:,-1])/12)
    projected=clear_lb[:,-1]-closing*24*(1-progress[:,-1].clamp(0,1))
    return cost+2800*F.relu(terminal_margin-projected).square()


def flow_descent(gradient):
    """Normalize the risk direction so a horizon mean cannot erase guidance.

    A clipped-only gradient of the 0.35 insertion potential is O(1/H), far
    smaller than the learned holding velocity. Preserve its direction and use
    the adaptive gain for its magnitude, identically for all in-flow variants.
    """
    norm=gradient.flatten(1).norm(dim=1).view(-1,1,1)
    return 3.*gradient/norm.clamp_min(1e-8)


class CUDAGraphPlanner:
    """Static-buffer CUDA Graph execution for sub-tick (30 Hz) guided planning."""

    def __init__(self,model,joint:bool,particles:int):
        self.model=model
        self.joint=joint
        self.particles=particles
        self.flow=model.joint if joint else model.flow
        self.channels=self.flow.channels
        dev=model.mean.device
        self.low=torch.tensor([-.09,-.23,.08,-.09,-.06,.04,.006,.006],device=dev)
        self.high=torch.tensor([.02,-.04,.24,.09,.25,.25,.038,.038],device=dev)
        self.eye3_K=torch.eye(3,device=dev).repeat(particles,1)
        self.eye3_1=torch.eye(3,device=dev)
        self.handle_xz=torch.tensor([-.048,0.,.175],device=dev)
        self.handle_mask=torch.tensor([1.,0.,1.],device=dev)
        self.handle_y=torch.tensor([0.,1.,0.],device=dev)
        self.raw_buf=torch.zeros((1,HISTORY,OBS_DIM),device=dev)
        self.obs_buf=torch.zeros((1,HISTORY,OBS_DIM),device=dev)
        self.stage_buf=torch.zeros((1,3),device=dev)
        self.belief_buf=torch.zeros((1,256),device=dev)
        self.prob_buf=torch.zeros((3,),device=dev)
        self.slots_buf=torch.zeros((2,),device=dev)
        self.hint_buf=torch.zeros((1,2),device=dev)
        self.emerg_buf=torch.zeros((),device=dev)
        self.tau_buf=torch.zeros((1,1),device=dev)
        self.guidance_buf=torch.zeros((),device=dev)
        self.xK_buf=torch.zeros((particles,HORIZON,self.channels),device=dev)
        self.x1_buf=torch.zeros((1,HORIZON,self.channels),device=dev)
        self._build_graphs()

    def _project(self,x):
        action=self.model.physical(x[...,:8])
        action=torch.maximum(self.low,torch.minimum(self.high,action))
        lg=self.emerg_buf*action[...,6:7]+(1.-self.emerg_buf)*action[...,6:7].clamp(0.,.012)
        action=torch.cat((action[...,:6],lg,action[...,7:]),-1)
        previous=self.raw_buf[:,-1,-8:-2].expand(len(x),-1);steps=[]
        for a in action.unbind(1):
            delta=(a[:,:6]-previous).reshape(-1,2,3)
            previous=previous+(delta*(.0075/delta.norm(dim=-1,keepdim=True).clamp_min(.0075))).flatten(1)
            steps.append(torch.cat((previous,a[:,6:]),-1))
        normalized=self.model.actions(torch.stack(steps,1))
        return torch.cat((normalized,x[...,8:]),-1) if self.joint else normalized

    def _predict(self,x,eye3):
        n=len(x)
        z=self.model.world.latent(self.belief_buf.expand(n*3,-1),self.stage_buf.expand(n*3,-1),
                                  x[...,:8].repeat_interleave(3,0),eye3)
        return self.model.world.decode_safety(z)*self.model.ss

    def _risk_and_bounds(self,x,eye3):
        n=len(x)
        physical=self.model.physical(x[...,:8])
        if self.joint:
            bounds=x[...,8]*self.model.ss[0]
            base=safety_risk(x[...,8:]*self.model.ss,physical,0.,emergency=self.emerg_buf)
            return base,bounds
        raw_last=self.raw_buf[0,-1]
        safety=self._predict(x,eye3)
        safety_3=safety.reshape(n,3,HORIZON,4)
        bounds=(safety_3[...,0]*self.prob_buf[None,:,None]).sum(1)-self.model.world.conformal_q
        base=safety_risk(safety,physical.repeat_interleave(3,0),self.model.world.conformal_q,emergency=self.emerg_buf).reshape(n,3)@self.prob_buf
        pos=physical[...,3]
        occupied_overlap=self.slots_buf[0]*torch.exp(-((pos+.048)/.025).square())+self.slots_buf[1]*torch.exp(-((pos-.048)/.025).square())
        is_stage1=(self.stage_buf[0].argmax()==1).float()
        cue_right=(raw_last[29]>.5).float()
        target=is_stage1*(.048*(self.slots_buf[0]-self.slots_buf[1]))+(1.-is_stage1)*(.096*cue_right-.048)
        cost=base+35.*(pos-target).square().mean(-1)+4.*occupied_overlap[:,-8:].mean(-1)
        handle=self.handle_xz*self.handle_mask+(raw_last[0]-.100)*self.handle_y
        near=torch.sigmoid((.026-(raw_last[5:8]-handle).norm())/.004)
        gap=.032*(1.-near)+.008*near
        holding=near*torch.sigmoid((.016-raw_last[-2])/.002)
        target_y=handle[1]*(1.-holding)-.185*holding
        emerg_cost=2000.*(physical[...,6]-gap).square().mean(-1)+.5*((physical[...,1]-target_y)/.060).square().mean(-1)
        return cost+self.emerg_buf*emerg_cost,bounds

    def _field(self,x,tau_scalar):
        n=len(x)
        return self.flow(x,tau_scalar.expand(n,1),self.obs_buf.expand(n,-1,-1),self.hint_buf.expand(n,-1))

    @staticmethod
    def _capture(fn):
        stream=torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):fn()
        torch.cuda.current_stream().wait_stream(stream)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out=fn()
        return graph,out

    def _build_graphs(self):
        def encode_and_prior7():
            with torch.no_grad():
                obs=self.model.encode(self.raw_buf);self.obs_buf.copy_(obs)
                stage=self.raw_buf[:,-1,25:28];self.stage_buf.copy_(stage)
                b,logits=self.model.world.encode(obs)
                self.belief_buf.copy_(b);self.prob_buf.copy_(logits.softmax(-1)[0])
                slots=self.model.world.slots(b).sigmoid()[0];self.slots_buf.copy_(slots)
                is_s1=(stage[0].argmax()==1).float()
                hint=is_s1*(1.-slots)[None]+(1.-is_s1)*self.raw_buf[:,-1,28:30]
                self.hint_buf.copy_(hint/hint.sum(-1,keepdim=True).clamp_min(1e-6))
                x=self.xK_buf
                for idx in range(7):
                    t=x.new_full((len(x),1),idx/10)
                    x=x+self.flow(x,t,obs.expand(len(x),-1,-1),self.hint_buf.expand(len(x),-1))/10
                return x
        self.g_prior7,self.out_prior7=self._capture(encode_and_prior7)

        def euler_K():
            with torch.no_grad():
                return self.xK_buf+self._field(self.xK_buf,self.tau_buf)/10
        self.g_eulerK,self.out_eulerK=self._capture(euler_K)

        def eval_ep_K():
            with torch.no_grad():
                v=self._field(self.xK_buf,self.tau_buf)
                ep=self.xK_buf+(1.-self.tau_buf[:,:,None])*v
                scores,bounds=self._risk_and_bounds(ep,self.eye3_K)
                return scores,bounds,self.xK_buf+v/10
        self.g_evalK,self.out_evalK=self._capture(eval_ep_K)

        def guided_1():
            with torch.enable_grad():
                cur=self.x1_buf.detach().requires_grad_(True)
                v=self._field(cur,self.tau_buf)
                ep=cur+(1.-self.tau_buf[:,:,None])*v
                scores,_=self._risk_and_bounds(ep,self.eye3_1)
                grad,=torch.autograd.grad(scores.sum(),cur)
            with torch.no_grad():
                return cur.detach()+(v.detach()-self.guidance_buf*flow_descent(grad.detach()))/10
        self.g_guided1,self.out_guided1=self._capture(guided_1)

        def proj_1():
            with torch.no_grad():
                p=self._project(self.x1_buf)
                _,bounds=self._risk_and_bounds(p,self.eye3_1)
                return p,bounds
        self.g_proj1,self.out_proj1=self._capture(proj_1)

        def proj_rank_K():
            with torch.no_grad():
                p=self._project(self.xK_buf)
                scores,bounds=self._risk_and_bounds(p,self.eye3_K)
                return p,scores,bounds
        self.g_projK,self.out_projK=self._capture(proj_rank_K)

        def opt_step_1():
            with torch.enable_grad():
                cur=self.x1_buf.detach().requires_grad_(True)
                scores,_=self._risk_and_bounds(cur,self.eye3_1)
                grad,=torch.autograd.grad(scores.sum(),cur)
            with torch.no_grad():
                return self._project(cur.detach()-.08*grad.detach())
        self.g_opt1,self.out_opt1=self._capture(opt_step_1)


def _decide_cuda_graph(model,history,config,joint,kind,rng,diagnostics):
    key=(joint,config.particles)
    planner=model._planners.get(key)
    if planner is None:
        planner=CUDAGraphPlanner(model,joint,config.particles)
        model._planners[key]=planner
    noise=temporal_noise((config.particles,HORIZON,planner.channels),generator=rng)
    hist_np=np.asarray(history,dtype=np.float32)
    stage_idx=int(np.argmax(hist_np[-1,25:28]))
    if config.particles>1 and stage_idx==1:
        noise[:config.particles//2,:,3]-=.35
        noise[config.particles//2:,:,3]+=.35
    emergency=bool(hist_np[-1,4]<.085 or hist_np[-1,4]-hist_np[-2,4]<-.006)
    aperture=hist_np[:,4]
    opened=np.maximum.accumulate(aperture>.095)
    recovering=bool(((aperture<.070)&opened).any() and hist_np[-1,10]>.170)
    planner.raw_buf[0].copy_(torch.from_numpy(hist_np))
    planner.xK_buf.copy_(noise)
    planner.emerg_buf.fill_(1.0 if emergency else 0.0)
    planner.g_prior7.replay()
    start=round(config.tau_start*10);guidance=.45
    intervention_min=None;adaptive=False
    if start==10:
        planner.xK_buf.copy_(planner.out_prior7)
        for idx in (7,8,9):
            planner.tau_buf.fill_(idx/10)
            planner.g_eulerK.replay()
            planner.xK_buf.copy_(planner.out_eulerK)
        if config.particles>1:
            planner.g_projK.replay()
            p_all,scores,bounds=planner.out_projK
            best=int(scores.argmin())
            res_norm=p_all[best:best+1]
            min_lb=float(bounds[best].min())
        else:
            planner.x1_buf.copy_(planner.xK_buf)
            planner.g_proj1.replay()
            res_norm=planner.out_proj1[0].clone()
            for _ in range(config.opt_steps):
                planner.x1_buf.copy_(res_norm)
                planner.g_opt1.replay()
                res_norm.copy_(planner.out_opt1)
            min_lb=float(planner.out_proj1[1][0].min())
    else:
        planner.xK_buf.copy_(planner.out_prior7)
        planner.tau_buf.fill_(.7)
        planner.g_evalK.replay()
        scores,bounds,x_after_7=planner.out_evalK
        best=int(scores.argmin())
        intervention_min=float(bounds[best].min())
        adaptive=emergency or intervention_min<.018
        if adaptive:start=7;guidance=6.
        elif recovering:start=7
        planner.guidance_buf.fill_(guidance)
        if start==7:
            planner.x1_buf.copy_(planner.xK_buf[best:best+1])
            for idx in (7,8,9):
                planner.tau_buf.fill_(idx/10)
                planner.g_guided1.replay()
                planner.x1_buf.copy_(planner.out_guided1)
        else:
            planner.xK_buf.copy_(x_after_7)
            planner.tau_buf.fill_(.8)
            planner.g_eulerK.replay()
            planner.xK_buf.copy_(planner.out_eulerK)
            planner.tau_buf.fill_(.9)
            planner.g_evalK.replay()
            best9=int(planner.out_evalK[0].argmin())
            planner.x1_buf.copy_(planner.xK_buf[best9:best9+1])
            planner.g_guided1.replay()
            planner.x1_buf.copy_(planner.out_guided1)
        planner.g_proj1.replay()
        res_norm=planner.out_proj1[0]
        min_lb=float(planner.out_proj1[1][0].min())
    prediction=int(planner.slots_buf.argmax())
    details=dict(predicted_occupied_slot=prediction,particles=config.particles,world_kind=kind,
                 emergency=emergency,recovering=recovering,adaptive_intervention=bool(adaptive or recovering),
                 tau_actual=start/10,guidance=guidance,intervention_min_clear_lb_m=intervention_min)
    if diagnostics:
        if joint or config.opt_steps>0:
            with torch.no_grad():
                safety=planner._predict(res_norm,planner.eye3_1).reshape(1,3,HORIZON,4)
                min_lb=float(((safety[...,0]*planner.prob_buf[None,:,None]).sum(1)-model.world.conformal_q).min())
        details['min_clear_lb_m']=min_lb
    return model.physical(res_norm[0,:,:8]).cpu().numpy(),details


def decide(models,history,method,rng,*,particles=None,world_kind=None,diagnostics=False,use_cuda_graph=True):
    kind=world_kind or ('short' if method in ('BC H4','Joint-FM','GPC-RANK','GPC-OPT','Ours Short-MLP H4') else 'four' if method=='Ours 4D-WM' else 'full')
    model=models[kind];joint=method=='Joint-FM'
    configs={'BC H4':Search(1),'Joint-FM':Search(1,tau_start=.7),'GPC-RANK':Search(PARTICLES),
             'GPC-OPT':Search(1,20),'Ours Short-MLP H4':Search(PARTICLES,tau_start=.7),
             'Ours 4D-WM':Search(PARTICLES,tau_start=.7),'Ours tau=0.7':Search(PARTICLES,tau_start=.7),
             'Ours tau=0.9':Search(PARTICLES,tau_start=.9)}
    config=configs[method]
    if particles is not None:config=Search(particles,config.opt_steps,config.tau_start)
    if use_cuda_graph and model.mean.is_cuda and round(config.tau_start*10) in (7,9,10):
        return _decide_cuda_graph(model,history,config,joint,kind,rng,diagnostics)
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
    # A collapse after an observed open collar remains a recovery state until
    # insertion resumes. Initial closed frames before the first spread do not
    # trigger it. This uses aperture history only, never simulator fault events.
    aperture=raw[0,:,4]
    opened=(aperture>.095).int().cummax(0).values.bool()
    recovering=bool(((aperture<.070)&opened).any() and raw[0,-1,10]>.170)
    if config.particles>1 and int(stage.argmax())==1:
        noise[:config.particles//2,:,3]-=.35
        noise[config.particles//2:,:,3]+=.35
    def field(x,t):return flow(x,x.new_full((len(x),1),t),obs.expand(len(x),-1,-1),hint.expand(len(x),-1))
    def project(x):
        action=model.physical(x[...,:8])
        low=action.new_tensor([-.09,-.23,.08,-.09,-.06,.04,.006,.006])
        high=action.new_tensor([.02,-.04,.24,.09,.25,.25,.038,.038])
        action=torch.maximum(low,torch.minimum(high,action))
        if not emergency:
            action=torch.cat((action[...,:6],action[...,6:7].clamp(0.,.012),action[...,7:]),-1)
        previous=raw[:,-1,-8:-2].expand(len(x),-1);steps=[]
        for a in action.unbind(1):
            delta=(a[:,:6]-previous).reshape(-1,2,3)
            previous=previous+(delta*(.0075/delta.norm(dim=-1,keepdim=True).clamp_min(.0075))).flatten(1)
            steps.append(torch.cat((previous,a[:,6:]),-1))
        normalized=model.actions(torch.stack(steps,1))
        return torch.cat((normalized,x[...,8:]),-1) if joint else normalized
    def predict(x):
        n=len(x)
        z=model.world.latent(belief.expand(n*3,-1),stage.expand(n*3,-1),
                            x[...,:8].repeat_interleave(3,0),torch.eye(3,device=x.device).repeat(n,1))
        return model.world.decode_safety(z)*model.ss
    def risk(x,*,safety=None):
        nonlocal belief,prob,slots
        physical=model.physical(x[...,:8])
        if joint:return safety_risk(x[...,8:]*model.ss,physical,emergency=emergency)
        if belief is None:
            with torch.no_grad():
                belief,logits=model.world.encode(obs);prob=logits.softmax(-1)[0]
                slots=model.world.slots(belief).sigmoid()[0]
        n=len(x)
        if safety is None:safety=predict(x)
        base=safety_risk(safety,physical.repeat_interleave(3,0),model.world.conformal_q,emergency=emergency).reshape(n,3)@prob
        pos=physical[...,3]
        occupied_overlap=slots[0]*torch.exp(-((pos+.048)/.025).square())+slots[1]*torch.exp(-((pos-.048)/.025).square())
        target=.048*(slots[0]-slots[1]) if int(stage.argmax())==1 else (.048 if raw[0,-1,29]>.5 else -.048)
        cost=base+35*(pos-target).square().mean(-1)+4*occupied_overlap[:,-8:].mean(-1)
        if emergency:
            # Smooth, observation-only regrasp potential, shared by ALL WM
            # planners. Closing a nearby handle must not wait for a rare flow
            # sample; the model's future clearance still supplies causal risk.
            handle=raw.new_tensor([-.048,0.,.175]);handle[1]=raw[0,-1,0]-.100
            near=torch.sigmoid((.026-(raw[0,-1,5:8]-handle).norm())/.004)
            gap=.032*(1-near)+.008*near
            holding=near*torch.sigmoid((.016-raw[0,-1,-2])/.002)
            target_y=handle[1]*(1-holding)-.185*holding
            cost+=2000*(physical[...,6]-gap).square().mean(-1)
            cost+=.5*((physical[...,1]-target_y)/.060).square().mean(-1)
        return cost
    # Inspect the posterior-predictive conformal bound at tau=.7 for both
    # nominal intervention times. Dangerous states bring tau=.9 forward to .7.
    x=noise;start=round(config.tau_start*10);guidance=.45
    intervention_min=None;adaptive=False
    for index in range(10):
        tau=index/10
        v7=None;best7=None
        if index==7 and start<10:
            with torch.no_grad():
                v7=field(x,.7)
                endpoint=x+.3*v7
                if joint:
                    bounds=endpoint[...,8]*model.ss[0]
                    scores7=risk(endpoint)
                else:
                    safety=predict(endpoint)
                    bounds=(safety.reshape(len(x),3,HORIZON,4)[...,0]*prob[None,:,None]).sum(1)-model.world.conformal_q
                    scores7=risk(endpoint,safety=safety)
                best7=int(scores7.argmin())
                intervention_min=float(bounds[best7].min())
                adaptive=emergency or intervention_min<.018
                if adaptive:start=7;guidance=6.
                elif recovering:start=7
        if index==start and len(x)>1:
            if best7 is not None:
                x=x[best7:best7+1]
            else:
                with torch.no_grad():x=x[risk(x+(1-tau)*field(x,tau)).argmin().reshape(1)]
        if index>=start:
            with torch.enable_grad():
                current=x.detach().requires_grad_(True)
                velocity=field(current,tau)
                endpoint=current+(1-tau)*velocity
                grad,=torch.autograd.grad(risk(endpoint).sum(),current)
            with torch.no_grad():x=current+(velocity-guidance*flow_descent(grad))/10
        else:
            with torch.no_grad():x=x+(v7 if v7 is not None else field(x,tau))/10
    with torch.no_grad():x=project(x)
    for _ in range(config.opt_steps):
        with torch.enable_grad():
            current=x.detach().requires_grad_(True)
            grad,=torch.autograd.grad(risk(current).sum(),current)
        with torch.no_grad():x=project(current-.08*grad)
    if len(x)>1:
        with torch.no_grad():x=x[risk(x).argmin().reshape(1)]
    result=x[0]
    prediction=int(slots.argmax()) if slots is not None else None
    details=dict(predicted_occupied_slot=prediction,particles=config.particles,world_kind=kind,
                 emergency=emergency,recovering=recovering,adaptive_intervention=bool(adaptive or recovering),tau_actual=start/10,
                 guidance=guidance,intervention_min_clear_lb_m=intervention_min)
    if diagnostics:
        with torch.no_grad():
            safety=predict(x).reshape(1,3,HORIZON,4)
            details['min_clear_lb_m']=float(((safety[...,0]*prob[None,:,None]).sum(1)-model.world.conformal_q).min())
    return model.physical(result[...,:8]).cpu().numpy(),details


def guidance_vs_ranking_probe(out):
    """Paired frozen handoff predictions; physical withdrawal is measured separately."""
    models=load_models(out,'cuda');rows=[];sources=[]
    for path in sorted((out/'dataset').glob('*slip_withdraw_reopen.npz')):
        if int(path.name.split('_')[0]) not in SPLITS[2]:continue
        with np.load(path,allow_pickle=False) as data:
            meta=json.loads(data['metadata'].item())
            obs,_,_=clean_observations(data['observations'],data['action'],meta)
            slips=[int(e['t']) for e in meta['events'] if e['event']=='collar_slip']
            if not slips:continue
            t=slips[0]+1;hist=obs[history_indices(t)]
        sources.append(dict(file=path.name,sha256=sha256(path.read_bytes()).hexdigest(),tick=t))
        for k in (2,4,8,16):
            for method in ('GPC-RANK','Ours tau=0.7'):
                action,details=decide(models,hist,method,torch.Generator().manual_seed(16000+meta['seed']),
                                      particles=k,world_kind='full',diagnostics=True)
                rows.append(dict(file=path.name,particles=k,method=method,
                                 min_clear_lb_m=details['min_clear_lb_m'],
                                 commanded_withdrawal=bool(action[-1,5]>=.240),
                                 max_right_z_m=float(action[:,5].max()),**{key:details[key] for key in ('tau_actual','guidance','emergency')}))
    dump(out/'guidance_probe.json',dict(protocol_version='stage_c2',sources=sources,rows=rows,
         checkpoint_hashes={p.name:sha256(p.read_bytes()).hexdigest() for p in (out/'trained').glob('*.pt')},
         planner_source_hashes={name:sha256((Path(__file__).resolve().parent/name).read_bytes()).hexdigest()
                               for name in ('bag_scene.py','bag_learning.py','bag_common.py','mamba_wm.py')},
         protocol='Paired seeds, same full world model and K; all 12 held-out first post-slip observations. Predicted bounds and command withdrawal are proxies, not physical success.',
         recovery_fraction=RECOVERY_FRACTION))


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
