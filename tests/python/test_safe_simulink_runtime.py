"""Online history, controller parity, and oracle isolation at the MATLAB bridge."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
AVAILABLE=importlib.util.find_spec('torch') is not None and importlib.util.find_spec('scipy') is not None
if AVAILABLE:
    import torch
    from test_safe_training_export import checkpoint_fixture
    from export_augmented_saits_barrier import export_checkpoint
    from saits_barrier_simulink import Controller, output_layout


@unittest.skipUnless(AVAILABLE,'Torch and SciPy required')
class TestSimulinkRuntime(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory()
        cls.folder=Path(cls.temp.name)
        checkpoint=cls.folder/'best.pt'
        torch.save(checkpoint_fixture(),checkpoint)
        cls.deployment=export_checkpoint(checkpoint,cls.folder/'deployment')

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.runtime=Controller(self.deployment,[30,35,-7])

    def healthy(self,time=0):
        return self.runtime.step([0,0,-7],[0,0,0],[0,0,0],[0,0,0],[7]*7+[0]*7,True,time)

    def test_exported_reference_neural_qp_and_allocation_agree(self):
        result=json.loads(self.runtime.reference_check())
        self.assertTrue(result['passed'],result)

    def test_fixed_packet_matches_executed_commands_and_masks(self):
        packet=self.healthy()
        self.assertEqual(packet.shape,(113,))
        self.assertTrue(np.isfinite(packet).all())
        np.testing.assert_array_equal(packet[63:94],1)
        np.testing.assert_array_equal(packet[18:32],[7]*7+[0]*7)
        self.assertEqual(packet[103],1)
        self.assertEqual(packet[104],0)
        self.assertEqual(output_layout()['diagnostics'],(103,113))

    def test_unavailable_input_is_never_read_and_anchor_does_not_reset(self):
        class Poison:
            def __array__(self,*args,**kwargs):
                raise AssertionError('Hidden LiDAR was read')
        self.healthy()
        packet=self.runtime.step([.2,.1,-7],[0]*3,[0]*3,[0]*3,Poison(),False,.1)
        np.testing.assert_array_equal(packet[63:77],0)
        np.testing.assert_allclose(packet[46:49],[.2,.1,0],atol=1e-7)
        self.assertAlmostEqual(packet[62],.1,places=6)
        self.assertEqual(packet[103],0); self.assertEqual(packet[104],1)

    def test_recovery_resets_age_and_displacement(self):
        self.healthy()
        self.runtime.step([.2,.1,-7],[0]*3,[0]*3,[0]*3,None,False,.1)
        packet=self.runtime.step([.3,.2,-7],[0]*3,[0]*3,[0]*3,[7]*7+[0]*7,True,.2)
        np.testing.assert_array_equal(packet[46:49],0)
        self.assertEqual(packet[62],0)
        self.assertTrue(self.runtime.mask[0,-1,:14].all())

    def test_duplicate_tick_is_idempotent_and_reset_is_reproducible(self):
        first=self.healthy()
        history=self.runtime.history.clone()
        duplicate=self.runtime.step([1,2,3],[0]*3,[0]*3,[0]*3,None,False,0)
        np.testing.assert_array_equal(first,duplicate)
        torch.testing.assert_close(history,self.runtime.history,rtol=0,atol=0)
        self.runtime.reset()
        repeated=self.healthy()
        indices=[i for i in range(113) if i!=110]
        np.testing.assert_allclose(first[indices],repeated[indices],rtol=0,atol=0)

    def test_invalid_scan_clock_and_motion_fail_explicitly(self):
        self.healthy()
        with self.assertRaisesRegex(ValueError,'10-Hz'):
            self.healthy(.2)
        self.runtime.reset()
        with self.assertRaisesRegex(ValueError,'nonfinite'):
            self.runtime.step([float('nan'),0,-7],[0]*3,[0]*3,[0]*3,[7]*14,True,0)

    def test_initially_invalid_barrier_requests_stop_without_claiming_safety(self):
        packet=self.runtime.step([0,0,-7],[0]*3,[0]*3,[0]*3,[.1]*7+[1]*7,True,0)
        self.assertEqual(packet[105],0)  # feasible
        self.assertEqual(packet[109],1)  # failure latch -> MATLAB Stop Simulation
        self.assertLess(packet[111],0)

    def test_diagnostic_continuation_uses_bounded_nominal_and_logs_executed_residual(self):
        self.runtime=Controller(self.deployment,[30,35,-7],continue_after_failure=True)
        captured={}
        original=self.runtime.barrier.forward
        def record(*args,**kwargs):
            result=original(*args,**kwargs); captured.update(result); return result
        self.runtime.barrier.forward=record
        packet=self.runtime.step([0,0,-7],[0]*3,[0]*3,[0]*3,[.1]*7+[1]*7,True,0)
        self.assertEqual(packet[105],0)
        self.assertEqual(packet[109],1)
        # Emergency braking at rest would request zero; diagnostic mode
        # follows the perceived-sector nominal with its upstream slew limit.
        self.assertLess(packet[97],-.09)
        self.assertLessEqual(np.max(np.abs(packet[97:99])),.100001)
        selected=torch.tensor([[packet[97],packet[98],packet[107]]],dtype=torch.float64)
        violation=(captured['constraint_matrix']@selected.unsqueeze(-1)).squeeze(-1)-captured['constraint_rhs']
        self.assertAlmostEqual(packet[106],float(violation.clamp_min(0).max()),places=7)
        next_packet=self.runtime.step([0,0,-7],[0]*3,[0]*3,[0]*3,[.1]*7+[1]*7,True,.1)
        self.assertEqual(next_packet[109],1)
        self.assertLessEqual(np.max(np.abs(next_packet[97:99]-packet[97:99])),.100001)


if __name__=='__main__': unittest.main()
