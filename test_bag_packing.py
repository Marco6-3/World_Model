"""Regression tests for causal history, occlusion, architecture and risk gradients."""
import unittest
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import bag_packing
import numpy as np
import torch
from bag_common import (history_indices, visible_contents, measure_trace, effective_clearance,
                        slot_pos, SIZE, FLOOR, CONTROL_DT, wallclock_lag_ticks, align_action_chunk)
from bag_learning import BagWorld, Models, decide, safety_risk, clean_observations, calibrate_conformal, load_windows, flow_descent
from mamba_wm import MambaBlock


class BagTests(unittest.TestCase):
    def test_mask_cannot_reveal_hidden_slot(self):
        poses=np.array([[-.048,0,.03,1,0,0,0],[.048,.22,.10,1,0,0,0]])
        left=visible_contents(poses,[1,0])
        poses[0,0]=.048
        right=visible_contents(poses,[0,1])
        np.testing.assert_array_equal(left,right)
        np.testing.assert_array_equal(left[:4],0)
        np.testing.assert_array_equal(left[8:],0)

    def test_causal_long_history_and_true_h4_suffix(self):
        for t in (0,3,70,500):
            indices=history_indices(t)
            self.assertEqual(len(indices),64)
            self.assertLessEqual(indices.max(),t)
            np.testing.assert_array_equal(indices[-4:],np.maximum(0,np.arange(t-3,t+1)))
        self.assertGreater(500-history_indices(500)[0],300)

    def test_required_dimensions(self):
        full=BagWorld('full');four=BagWorld('four')
        self.assertEqual(full.past.embed.out_features,256)
        self.assertEqual(full.initialize.out_features,256)
        self.assertEqual(four.initialize.out_features,4)
        self.assertEqual(four.g_state[0].in_features,4)
        self.assertEqual(four.g_safe[0].in_features,4)

    def test_fault_command_imputation_preserves_sensors_and_later_commands(self):
        obs=np.arange(24*39,dtype=np.float32).reshape(24,39)
        obs[:,-2]=.008;obs[4:13,-2]=.038;obs[16:,-2]=.032
        action=np.zeros((23,8),np.float32);action[:,6]=.008;action[3:12,6]=.038;action[15:,6]=.032
        clean,a,n=clean_observations(obs,action,{'events':[{'t':3,'event':'collar_slip'}]})
        self.assertEqual(n,9)
        np.testing.assert_array_equal(clean[:,:5],obs[:,:5])
        np.testing.assert_allclose(clean[4:13,-2],.008)
        np.testing.assert_array_equal(clean[16:,-2],obs[16:,-2])
        np.testing.assert_allclose(a[3:12,6],.008)
        self.assertAlmostEqual(float(obs[4,-2]),.038,places=6)

    def test_conformal_uses_finite_sample_order_statistic_and_frozen_test(self):
        model=torch.nn.Module();model.world=torch.nn.Module()
        model.world.register_buffer('conformal_q',torch.tensor(0.))
        scores=torch.arange(1,10,dtype=torch.float32)/1000
        heldout=torch.tensor([.001,.009,.010])
        with patch('bag_learning.clearance_scores',side_effect=[(scores,scores),(heldout,heldout)]):
            result=calibrate_conformal(model,{}, {})
        self.assertEqual(result['quantile_rank'],9)
        self.assertAlmostEqual(result['conformal_q_m'],.009,places=6)
        self.assertAlmostEqual(result['conformal_coverage'],2/3,places=6)

    def test_reopened_safe_collar_risk_attracts_downward_insertion(self):
        state=torch.tensor([.028,0.,.4,0.]).repeat(1,24,1)
        action=torch.zeros(1,24,8);action[...,5]=.245;action.requires_grad_(True)
        grad,=torch.autograd.grad(safety_risk(state,action,.008).sum(),action)
        self.assertTrue((grad[...,5]>0).all())
        action=torch.zeros(1,24,8,requires_grad=True)
        grad,=torch.autograd.grad(safety_risk(state,action,.008,emergency=True).sum(),action)
        self.assertTrue((grad[...,5]<0).all())

    def test_negative_schemes_never_supervise_flow_but_keep_world_windows(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);(root/'dataset').mkdir()
            for scheme in ('direct_insert','premature_release','spread_and_hold'):
                meta=dict(success=True,collision=False,method=scheme,events=[],
                          mode='relaxed',order=0,seed=1000,learn_ranges=[[0,40],[40,80]])
                stage=np.r_[np.zeros(40),np.ones(40)]
                np.savez(root/'dataset'/f'1000_{scheme}.npz',metadata=json.dumps(meta),
                         observations=np.zeros((81,39),np.float32),action=np.zeros((80,8),np.float32),
                         state=np.zeros((80,4),np.float32),truth=np.zeros((80,10),np.float32),
                         expert=np.ones(80,dtype=bool),learn=np.ones(80,dtype=bool),stage=stage)
                if scheme!='spread_and_hold':
                    data,_=load_windows(root,[1000]);self.assertFalse(data['expert'].any())
            data,files=load_windows(root,[1000])
            self.assertTrue(data['expert'].any());self.assertEqual(len(files),3)

    def test_audit_accepts_early_collapse_without_slip_event(self):
        from bag_audit import audit
        with TemporaryDirectory() as folder:
            frames=np.zeros((6,20));frames[:,1]=[.01,.03,.12,.12,.12,.02]
            poses=np.zeros((6,2,7));poses[:,:,3]=1
            poses[:,0,:3]=[-.048,0,.16];poses[1:3,0,2]=.20;poses[3:,0,:3]=slot_pos(0,0)
            poses[:,1,:3]=[.048,.22,.20];poses[4,1,:3]=[.048,0,.13];poses[5,1,:3]=slot_pos(1,1)
            np.savez(Path(folder)/'trace.npz',frames=frames,poses=poses,
                     stage=[0,0,0,0,1,1],learn=np.ones(6,dtype=bool),attached=np.ones(6,dtype=bool))
            row=dict(trace_file='trace.npz',success=True,collision=False,perturbed=True,
                     events=[],learn_ranges=[[0,4],[4,6]],committed_steps=[1]*6,
                     method='Ours tau=0.7',seed=2000,mode='relaxed',order=0)
            result=audit(Path(folder),{'episodes':[row]},write=False)['episodes'][0]
            self.assertTrue(result['safe_recovery'])
            self.assertEqual(result['recovery_measurement']['recovery_start_source'],'measured_collapse_below_15mm')

    def test_recovery_thinning_keeps_all_world_windows(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);(root/'dataset').mkdir()
            meta=dict(success=True,collision=False,method='slip_withdraw_reopen',
                      events=[dict(t=20,event='collar_slip'),dict(t=90,event='resume_entry')],
                      mode='relaxed',order=0,seed=1000,learn_ranges=[[0,140],[140,188]])
            np.savez(root/'dataset'/'1000_slip.npz',metadata=json.dumps(meta),
                     observations=np.zeros((189,39),np.float32),action=np.zeros((188,8),np.float32),
                     state=np.zeros((188,4),np.float32),truth=np.zeros((188,10),np.float32),
                     expert=np.ones(188,dtype=bool),learn=np.ones(188,dtype=bool),
                     stage=np.r_[np.zeros(140),np.ones(48)])
            scarce,files=load_windows(root,[1000])
            with patch('bag_learning.RECOVERY_FRACTION',1.):full,_=load_windows(root,[1000])
            self.assertEqual(len(scarce['obs']),len(full['obs']))
            self.assertLess(int(scarce['expert'].sum()),int(full['expert'].sum()))
            self.assertEqual(files[0]['retained_recovery_windows'],int(files[0]['recovery_windows']*.15))
            a,_=load_windows(root,[1000]);torch.testing.assert_close(a['expert'],scarce['expert'])

    def test_flow_guidance_preserves_direction_under_horizon_scaling(self):
        g=torch.randn(2,24,8)
        torch.testing.assert_close(flow_descent(g),flow_descent(g/24))
        torch.testing.assert_close(flow_descent(g).flatten(1).norm(dim=1),torch.full((2,),3.))
        self.assertTrue(torch.isfinite(flow_descent(torch.zeros_like(g))).all())

    def test_c1_item_geometry_independent_of_slot_assignment(self):
        for item in (0,1):
            self.assertEqual(slot_pos(item,0)[2],FLOOR+SIZE[item,2]/2)
            self.assertEqual(slot_pos(item,1)[2],FLOOR+SIZE[item,2]/2)
        frames=np.zeros((2,20));frames[:,11:14]=[-.048,0,.13]
        result=measure_trace(frames,np.ones(2),np.ones(2,dtype=bool),np.zeros(2),1,None)
        self.assertTrue(result['second_slot_correct'])
        self.assertEqual(500-history_indices(500)[0],443)

    def test_c1_risk_rewards_vertical_escape_before_contact(self):
        state=torch.tensor([.002,1.,.8,0.]).repeat(1,24,1)
        action=torch.zeros(1,24,8,requires_grad=True)
        grad,=torch.autograd.grad(safety_risk(state,action).sum(),action)
        self.assertTrue((grad[...,5]<0).all())

    def test_future_prefix_causality(self):
        torch.manual_seed(10)
        world=BagWorld().cuda().eval()
        b=torch.randn(1,256,device='cuda');context=torch.tensor([[1.,0,0]],device='cuda')
        actions=torch.randn(1,24,8,device='cuda');altered=actions.clone();altered[:,12:]+=3
        with torch.no_grad():
            first=world.latent(b,context,actions,context)
            second=world.latent(b,context,altered,context)
        torch.testing.assert_close(first[:,:12],second[:,:12])
        self.assertGreater(float((first[:,12:]-second[:,12:]).abs().max()),.001)

    def test_nonzero_clearance_risk_gradient(self):
        state=torch.tensor([.005,1.,.5,0.]).repeat(2,24,1).requires_grad_(True)
        grad,=torch.autograd.grad(safety_risk(state,torch.zeros(2,24,8)).sum(),state)
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(float(grad[...,0].abs().sum()),.01)

    def test_slot_metric_requires_learned_physical_entry(self):
        frames=np.zeros((2,20));frames[:,11:14]=[-.048,0,.13]
        result=measure_trace(frames,np.ones(2),np.array([False,True]),np.zeros(2),0,None)
        self.assertEqual(result['second_slot'],0)
        self.assertFalse(result['second_slot_correct'])
        result=measure_trace(frames,np.ones(2),np.zeros(2,dtype=bool),np.zeros(2),0,None)
        self.assertIsNone(result['second_slot'])

    def test_second_item_pickup_is_not_first_item_recovery(self):
        frames=np.zeros((4,20));frames[:,10]=.03;frames[2:,13]=.24
        frames[:2,1]=.02;frames[2:,1]=.14
        result=measure_trace(frames,np.array([0,0,1,1]),np.ones(4,dtype=bool),np.ones(4),0,0)
        self.assertFalse(result['withdraw_reopen'])

    def test_vertical_withdrawal_clears_a_closed_xy_projection(self):
        near=effective_clearance(-.02,[0,0,.13],[0,0,.15],0)
        withdrawn=effective_clearance(-.02,[0,0,.25],[0,0,.28],0)
        self.assertLess(near,0)
        self.assertGreater(withdrawn,.05)

    def test_mamba_parallel_scan_matches_sequential_transition_and_gradients(self):
        torch.manual_seed(42)
        block=MambaBlock(d_model=32,d_state=8,d_conv=4,expand=2)
        for length in (1,12,24,64):
            u_par=torch.randn(3,length,32,requires_grad=True)
            u_seq=u_par.detach().clone().requires_grad_(True)
            out_par=block(u_par)
            x,gate=block.in_proj(u_seq).chunk(2,dim=-1)
            x=torch.nn.functional.silu(block.conv(x.transpose(1,2))[...,:length].transpose(1,2))
            decay,drive,read=block.select(x).split((block.inner,block.d_state,block.d_state),dim=-1)
            state=x.new_zeros((len(x),block.inner,block.d_state))
            a=-block.A_log.exp()
            rows=[]
            for xt,dt,bt,ct in zip(*(v.unbind(1) for v in (x,decay,drive,read)),strict=True):
                state,y=block._transition(xt,dt,bt,ct,state,a)
                rows.append(y)
            out_seq=block.norm(u_seq+block.out_proj(torch.stack(rows,1)*torch.nn.functional.silu(gate)))
            torch.testing.assert_close(out_par,out_seq,atol=1e-5,rtol=1e-5)
            g_par,=torch.autograd.grad(out_par.square().sum(),u_par)
            g_seq,=torch.autograd.grad(out_seq.square().sum(),u_seq)
            torch.testing.assert_close(g_par,g_seq,atol=1e-5,rtol=1e-5)

    def test_cuda_graph_planner_matches_eager_decide(self):
        if not torch.cuda.is_available():
            self.skipTest('CUDA required for CUDAGraphPlanner test')
        torch.manual_seed(123)
        models={k:Models(k).cuda().eval().requires_grad_(False) for k in ('full','short','four')}
        hist=np.zeros((64,39),dtype=np.float32)
        hist[:,4]=.110;hist[-1,4]=.065;hist[-1,10]=.19;hist[-1,26]=1.;hist[-1,28:30]=[.5,.5]
        for method in ('BC H4','Joint-FM','GPC-RANK','GPC-OPT','Ours tau=0.7','Ours tau=0.9'):
            a_graph,d_graph=decide(models,hist,method,torch.Generator().manual_seed(77),diagnostics=True,use_cuda_graph=True)
            a_eager,d_eager=decide(models,hist,method,torch.Generator().manual_seed(77),diagnostics=True,use_cuda_graph=False)
            np.testing.assert_allclose(a_graph,a_eager,atol=1e-4,rtol=1e-4)
            self.assertEqual(d_graph['predicted_occupied_slot'],d_eager['predicted_occupied_slot'])
            self.assertEqual(d_graph['tau_actual'],d_eager['tau_actual'])
            self.assertAlmostEqual(d_graph['min_clear_lb_m'],d_eager['min_clear_lb_m'],places=4)

    def test_wallclock_lag_and_chunk_alignment(self):
        self.assertEqual(wallclock_lag_ticks(14.44,CONTROL_DT),1)
        self.assertEqual(wallclock_lag_ticks(17.25,CONTROL_DT),1)
        self.assertEqual(wallclock_lag_ticks(64.60,CONTROL_DT),2)
        self.assertEqual(wallclock_lag_ticks(182.75,CONTROL_DT),6)
        self.assertEqual(wallclock_lag_ticks(780.0,CONTROL_DT),24)
        chunk=np.arange(24*8,dtype=np.float32).reshape(24,8)
        aligned1,off1=align_action_chunk(chunk,1)
        self.assertEqual(off1,1);self.assertEqual(len(aligned1),23)
        np.testing.assert_array_equal(aligned1[0],chunk[1])
        aligned24,off24=align_action_chunk(chunk,24)
        self.assertEqual(off24,23);self.assertEqual(len(aligned24),1)
        np.testing.assert_array_equal(aligned24[0],chunk[23])


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
