"""Dense reconstruction, chronological windows, stage-one artifacts and export."""
import ast
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
import warnings

import matplotlib
matplotlib.use('Agg')
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from dense_saits_single_episode import (BEAMS,FEATURES,DenseSaitsReconstructor,build_dataset,
    build_train_val_tensors,validate_config,train_saits_baseline,fit_normalizer)
from export_dense_saits_matlab import export_episode,resolve_training_run
from test_cnn_mlp_single_episode_notebook import fixture


class TestDenseSaitsSingleEpisode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads(); torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls): torch.set_num_threads(cls.threads)

    def config(self,root):
        config=json.loads((ROOT/'configs/dense_saits_single_episode.json').read_text())
        config['paths'].update(repo_root=str(root),output_root=str(root/'runs'))
        config['data']['episode_path']='fixture-episode'
        config['model'].update(window_size=4,d_model=8,n_heads=2,d_ff=16,head_width=8,dropout=0)
        config['training'].update(epochs=1,batch_size=4,save_every_n_epochs=1)
        config['scheduler']['t_max_epochs']=1
        config['runtime'].update(device='cpu',num_workers=0)
        config['plots']['dpi']=40
        config['faults'].update(initial_healthy_scans=2,short_outage_scans=[2,2],long_outage_scans=[5,5],recovery_scans=1)
        return config

    def data(self,root):
        raw,_,_=fixture(root,count=20)
        cfg=self.config(root)
        corpus=build_dataset(root,str(raw),.8,.2,split_policy='chronological',verify_hashes=True)
        train,val,norm=build_train_val_tensors(corpus,cfg)
        return cfg,corpus,train,val,norm

    def test_notebook_preserves_section_flow_full_scan_scope_and_no_saved_outputs(self):
        baseline=json.loads((ROOT/'notebooks/train_cnn_mlp_baseline_single_episode.ipynb').read_text())
        notebook=json.loads((ROOT/'notebooks/train_saits_baseline_single_episode.ipynb').read_text())
        self.assertEqual(len(baseline['cells']),len(notebook['cells']))
        self.assertEqual([c['cell_type'] for c in baseline['cells']],[c['cell_type'] for c in notebook['cells']])
        for c in notebook['cells']:
            if c['cell_type']=='code':
                ast.parse(''.join(c['source'])); self.assertEqual(c['outputs'],[])
        text=''.join(''.join(c['source']) for c in notebook['cells'])
        self.assertIn('captureSaitsSingleEpisodeSingleUav.m',text)
        self.assertNotIn('train_stage_two',text)

    def test_split_boundary_has_no_training_or_future_scan_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg,corpus,train,val,norm=self.data(Path(tmp))
            raw,mask,target,hidden=val.raw_window(0)
            self.assertFalse(mask[:-1].any())
            self.assertTrue(mask[-1].all())
            np.testing.assert_array_equal(target,corpus.scans[16].T.reshape(-1))
            self.assertEqual(norm['fit_scan_ids'],corpus.splits['train'])
            _,mask,_,_=train.raw_window(16+2)
            self.assertFalse(mask[-1,:BEAMS].any())
            raw,_,_,_=train.raw_window(16+2)
            self.assertTrue((raw[-1,:BEAMS]==0).all())
            _,long_mask,_,_=train.raw_window(32+6)
            self.assertFalse(long_mask[:,:BEAMS].any())

    def test_normalizer_excludes_validation_and_masked_values_cannot_leak(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg,corpus,train,val,norm=self.data(Path(tmp))
            corpus.states[16:]=999999
            self.assertEqual(norm,fit_normalizer(corpus))
            values,mask,_,_=train[18]
            model=DenseSaitsReconstructor(cfg['model']).eval()
            a=model(values[None],mask[None])['prediction']
            b=model(values.masked_fill(~mask,float('nan'))[None],mask[None])['prediction']
            torch.testing.assert_close(a,b,atol=0,rtol=0)
            self.assertEqual(a.shape,(1,10740))
            a.square().mean().backward()
            self.assertGreater(float(model.embedding_first.weight.grad.abs().sum()),0)

    def test_training_and_export_have_selected_weights_provenance_and_dense_fixtures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); cfg,corpus,train,val,norm=self.data(root)
            model,history,run=train_saits_baseline(DenseSaitsReconstructor(cfg['model']),train,val,
                config=cfg,split_ids=corpus.splits,source_metadata=corpus.source_metadata,normalizer=norm)
            for name in ('best.pt','last.pt','config.json','normalizer.json','source.json','summary.json','metrics.csv','metrics.json','splits.json',
                         'plots/loss_curves.png','plots/learning_rate.png','plots/validation_scan.png','validation_predictions.npz'):
                self.assertGreater((run/name).stat().st_size,0,name)
            summary=json.loads((run/'summary.json').read_text())
            self.assertEqual(summary['stage'],'reconstruction-only'); self.assertFalse(summary['barrier_enabled'])
            self.assertEqual(summary['status'],'stage1_completed')
            from export_cnn_mlp_matlab import resolve_training_run as resolve_cnn_run
            with self.assertRaisesRegex(ValueError,'No completed'):
                resolve_cnn_run(root/'runs',corpus.episode_id)
            pointer=export_episode(root,corpus.episode_id,training_run=run)
            from scipy.io import loadmat
            exported=loadmat(pointer['model_path'],simplify_cells=True)
            self.assertEqual(exported['model']['nTargets'],10740)
            self.assertEqual(exported['fixtures']['predictedRanges'].shape[:2],(15,716))
            self.assertEqual(resolve_training_run(root/'runs',corpus.episode_id),run)
            self.assertEqual(export_episode(root,corpus.episode_id,training_run=run)['model_path'],pointer['model_path'])
            with self.assertRaisesRegex(ValueError,'another episode'):
                resolve_training_run(root/'runs','other-id',run)

    def test_configuration_requires_fields_and_valid_splits(self):
        cfg=self.config(ROOT)
        for field,value in [('train_fraction',1.),('episode_path','')]:
            bad=copy.deepcopy(cfg); bad['data'][field]=value
            with self.assertRaises(ValueError): validate_config(bad)
        bad=copy.deepcopy(cfg); del bad['training']['learning_rate']
        with self.assertRaisesRegex(ValueError,'learning_rate'): validate_config(bad)

    def test_training_rejects_changed_fault_configuration_before_allocating_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg,corpus,train,val,norm=self.data(Path(tmp))
            cfg['faults']['recovery_scans']+=1
            with self.assertRaisesRegex(ValueError,'Rebuild datasets'):
                train_saits_baseline(DenseSaitsReconstructor(cfg['model']),train,val,config=cfg,
                    split_ids=corpus.splits,source_metadata=corpus.source_metadata,normalizer=norm)


class TestDenseSaitsRunSelection(unittest.TestCase):
    """A corrupt/interrupted results folder must not block an intact run."""
    episode_id='selection-fixture'

    def write_run(self,root,name,created='2026-11-13T05:00:00+00:00',**changes):
        from dense_saits_single_episode import SCHEMA
        folder=root/name; folder.mkdir()
        (folder/'source.json').write_text(json.dumps({'episode_id':self.episode_id}),encoding='utf-8')
        summary=dict(schema_version=SCHEMA,episode_id=self.episode_id,status='stage1_completed',created_at_utc=created)
        summary.update(changes)
        (folder/'summary.json').write_text(json.dumps(summary),encoding='utf-8')
        (folder/'best.pt').write_bytes(b'opaque-checkpoint-not-loaded-during-discovery')
        return folder

    def test_null_summary_does_not_block_latest_completed_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            self.write_run(root,'old',created='2026-11-12T22:00:00+00:00')
            broken=self.write_run(root,'broken')
            (broken/'summary.json').write_bytes(b'\x00'*1738)
            selected=self.write_run(root,'latest')
            self.write_run(root,'interrupted',created='2026-11-13T06:00:00+00:00',status='interrupted')
            before=(broken/'summary.json').read_bytes()
            with self.assertWarnsRegex(UserWarning,'summary.json'):
                self.assertEqual(resolve_training_run(root,self.episode_id),selected)
            self.assertEqual((broken/'summary.json').read_bytes(),before)

    def test_explicit_bad_selection_never_falls_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); self.write_run(root,'good')
            broken=self.write_run(root,'broken'); (broken/'summary.json').write_text('',encoding='utf-8')
            with self.assertRaises(ValueError) as caught:
                resolve_training_run(root,self.episode_id,broken)
            self.assertIn(str(broken/'summary.json'),str(caught.exception))
            self.assertIn('Cannot read JSON metadata',str(caught.exception))

    def test_skips_bad_source_nonobject_summary_timestamp_and_missing_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); selected=self.write_run(root,'good')
            source=self.write_run(root,'bad_source'); (source/'source.json').write_bytes(b'\xff')
            summary=self.write_run(root,'bad_object'); (summary/'summary.json').write_text('[]',encoding='utf-8')
            self.write_run(root,'bad_time',created='not-a-date')
            self.write_run(root,'naive_time',created='2026-11-13T08:00:00')
            incomplete=self.write_run(root,'missing_weights'); (incomplete/'best.pt').unlink()
            with warnings.catch_warnings(record=True) as emitted:
                warnings.simplefilter('always')
                self.assertEqual(resolve_training_run(root,self.episode_id),selected)
            self.assertEqual(len(emitted),5)
            for folder in (source,summary,incomplete):
                with self.assertRaises(ValueError): resolve_training_run(root,self.episode_id,folder)

    def test_other_episode_summary_is_not_read_and_wrong_model_is_not_selected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); selected=self.write_run(root,'good')
            other=self.write_run(root,'other_episode')
            (other/'source.json').write_text(json.dumps({'episode_id':'different-episode'}),encoding='utf-8')
            (other/'summary.json').write_bytes(b'\x00')
            self.write_run(root,'cnn',schema_version='foreign-model',created='2026-11-14T00:00:00+00:00')
            with warnings.catch_warnings(record=True) as emitted:
                warnings.simplefilter('always')
                self.assertEqual(resolve_training_run(root,self.episode_id),selected)
            self.assertEqual(len(emitted),0)

    def test_no_intact_run_reports_corrupt_metadata_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); broken=self.write_run(root,'broken')
            (broken/'summary.json').write_text('{',encoding='utf-8')
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                with self.assertRaises(ValueError) as caught: resolve_training_run(root,self.episode_id)
            self.assertIn('No completed',str(caught.exception))
            self.assertIn(str(broken/'summary.json'),str(caught.exception))


if __name__=='__main__': unittest.main()
