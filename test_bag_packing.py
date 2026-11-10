"""Regression tests for causal history, occlusion, architecture and risk gradients."""
import unittest
import bag_packing
import numpy as np
import torch
from bag_common import history_indices, visible_contents, measure_trace, effective_clearance, slot_pos, SIZE, FLOOR
from bag_learning import BagWorld, safety_risk


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


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
