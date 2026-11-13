"""Full-resolution, reconstruction-only SAITS for one recorded episode.

Two DMSA imputation blocks are architectural blocks, not two training stages.
Derived attention layers retain the MIT attribution in saits_endpoint_model.py
and LICENSES/SAITS-MIT.txt. No barrier, QP, rollout or controller training.
"""
from __future__ import annotations
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import random
import sys
from time import perf_counter
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from saits_endpoint_model import _DMSALayer

SCHEMA = 'ares-dense-saits-stage1-single-episode-v1'
BEAMS, FEATURES = 10740, 10752


class DenseSaitsReconstructor(nn.Module):
    """SAITS endpoint prediction with linear-cost dense heads, not F² heads."""
    def __init__(self, model_config):
        super().__init__()
        self.model_config = copy.deepcopy(model_config)
        c = model_config
        steps, width = c['window_size'], c['d_model']
        if steps < 2 or width % 2 or width % c['n_heads']:
            raise ValueError('Invalid window/attention dimensions.')
        positions = torch.arange(steps).float()[:, None]
        frequency = torch.exp(torch.arange(0, width, 2).float() * (-math.log(10000.) / width))
        encoding = torch.zeros(steps, width)
        encoding[:, 0::2], encoding[:, 1::2] = torch.sin(positions * frequency), torch.cos(positions * frequency)
        self.register_buffer('position_encoding', encoding[None])
        self.register_buffer('allowed_attention', ~torch.eye(steps, dtype=torch.bool), persistent=False)
        self.embedding_first = nn.Linear(2 * FEATURES, width)
        self.embedding_second = nn.Linear(2 * FEATURES, width)
        self.first_dropout = nn.Dropout(c['dropout'])
        self.first_block = nn.ModuleList([_DMSALayer(width, c['n_heads'], c['d_ff'], c['dropout']) for _ in range(c['n_layers'])])
        self.second_block = nn.ModuleList([_DMSALayer(width, c['n_heads'], c['d_ff'], c['dropout']) for _ in range(c['n_layers'])])
        self.reduce_first = nn.Linear(width, FEATURES)
        self.reduce_second_beta = nn.Linear(width, c['head_width'])
        self.reduce_second_gamma = nn.Linear(c['head_width'], BEAMS)
        self.gate_first = nn.Linear(FEATURES + steps, c['head_width'])
        self.gate_second = nn.Linear(c['head_width'], BEAMS)

    def forward(self, values, observed_mask):
        if values.shape[1:] != (self.model_config['window_size'], FEATURES) or values.shape != observed_mask.shape:
            raise ValueError('Expected normalized [batch,window,10752] history and masks.')
        mask = observed_mask.to(values.dtype)
        filled = values.masked_fill(~observed_mask.bool(), 0)
        hidden = self.first_dropout(self.embedding_first(torch.cat((filled, mask), -1)) + self.position_encoding)
        for layer in self.first_block:
            hidden, _ = layer(hidden, self.allowed_attention)
        first = self.reduce_first(hidden)
        completed = mask * filled + (1 - mask) * first
        hidden = self.embedding_second(torch.cat((completed, mask), -1)) + self.position_encoding
        for i, layer in enumerate(self.second_block):
            hidden, attention = layer(hidden, self.allowed_attention, need_weights=i == len(self.second_block)-1)
        second = self.reduce_second_gamma(F.relu(self.reduce_second_beta(hidden[:, -1])))
        mixing = torch.sigmoid(self.gate_second(F.relu(self.gate_first(torch.cat((mask[:, -1], attention.mean(1)[:, -1]), -1)))))
        first = first[:, -1, :BEAMS]
        prediction = (1 - mixing) * second + mixing * first
        return {'prediction': prediction * 7, 'stage1': first * 7, 'stage2': second * 7}


def encode_sensor_state(states):
    states = np.asarray(states, dtype=np.float32)
    angles = states[..., 6:9]
    return np.concatenate((states[..., :6], np.stack((np.sin(angles), np.cos(angles)), -1).reshape(*states.shape[:-1], 6)), -1)


def normalize_history(raw, mask, normalizer):
    values = np.asarray(raw, np.float32).copy()
    mask = np.asarray(mask, bool)
    if values.shape != mask.shape or values.shape[-1] != FEATURES or not np.isfinite(values[mask]).all():
        raise ValueError('Invalid observed dense history.')
    values[~mask] = 0
    values[..., :BEAMS] /= np.float32(7)
    values[..., BEAMS:] = ((values[..., BEAMS:].astype(np.float64) - np.asarray(normalizer['context_mean'])) / np.asarray(normalizer['context_std'])).astype(np.float32)
    values[~mask] = 0
    return values


def fit_normalizer(corpus):
    start, stop = corpus.split_ranges['train']
    encoded = encode_sensor_state(corpus.states[start:stop]).astype(np.float64)
    std = encoded.std(0); std[std < 1e-6] = 1
    return dict(context_mean=encoded.mean(0).tolist(), context_std=std.tolist(), distance_scale_m=7.,
                fit_scan_ids=corpus.splits['train'], schema_version='ares-dense-saits-normalizer-v1')


def fault_availability(count, profile, seed, faults):
    available = np.ones(count, dtype=bool)
    if profile == 'healthy': return available
    rng = np.random.default_rng(seed)
    start = min(faults['initial_healthy_scans'], max(1, count // 4))
    lengths = faults['long_outage_scans'] if profile == 'long' else faults['short_outage_scans']
    while start < count:
        stop = min(count, start + int(rng.integers(lengths[0], lengths[1]+1)))
        available[start:stop] = False
        start = stop + faults['recovery_scans']
    return available


class DenseEpisodeWindows(Dataset):
    """Build causal windows on demand; split boundaries never share histories."""
    def __init__(self, corpus, split, normalizer, config):
        self.corpus, self.split, self.normalizer = corpus, split, normalizer
        self.start, self.stop = corpus.split_ranges[split]
        self.window = config['model']['window_size']
        self.augmentation_spec = dict(window_size=self.window,seed=config['run']['seed'],faults=copy.deepcopy(config['faults']))
        self.profiles = tuple(config['faults']['profiles'])
        count = self.stop - self.start
        self.availability = [fault_availability(count, p, config['run']['seed'] + 1009*i + (0 if split=='train' else 99991), config['faults']) for i,p in enumerate(self.profiles)]
        self.context = encode_sensor_state(corpus.states[self.start:self.stop])
        self.ranges = corpus.scans[self.start:self.stop].transpose(0,2,1).reshape(count,BEAMS)

    def __len__(self): return (self.stop - self.start) * len(self.profiles)

    def raw_window(self, index):
        count = self.stop - self.start
        profile, endpoint = divmod(int(index), count)
        if not 0 <= profile < len(self.profiles): raise IndexError(index)
        first = max(0, endpoint - self.window + 1)
        length = endpoint - first + 1
        raw = np.zeros((self.window, FEATURES), np.float32)
        mask = np.zeros(raw.shape, bool)
        raw[-length:,:BEAMS] = self.ranges[first:endpoint+1]
        raw[-length:,BEAMS:] = self.context[first:endpoint+1]
        mask[-length:,:BEAMS] = self.availability[profile][first:endpoint+1,None]
        mask[-length:,BEAMS:] = True
        raw[~mask] = 0  # discard hidden labels BEFORE returning any predictor
        return raw, mask, self.ranges[endpoint].copy(), ~mask[-1,:BEAMS]

    def __getitem__(self, index):
        raw, mask, target, hidden = self.raw_window(index)
        return tuple(torch.from_numpy(np.ascontiguousarray(v)) for v in (normalize_history(raw,mask,self.normalizer),mask,target,hidden))


def build_train_val_tensors(corpus, config):
    """Same notebook step as CNN; use lazy windows to avoid dense duplication."""
    normalizer = fit_normalizer(corpus)
    return DenseEpisodeWindows(corpus,'train',normalizer,config), DenseEpisodeWindows(corpus,'validation',normalizer,config), normalizer


def validate_config(config):
    for section in ('paths','run','data','model','training','runtime','scheduler','plots','faults'):
        if not isinstance(config.get(section),dict): raise ValueError(f'Missing CONFIG.{section}')
    required=dict(paths=('repo_root','output_root'),run=('name','seed'),data=('episode_path','train_fraction','val_fraction','split_policy','verify_hashes'),
        model=('window_size','d_model','n_heads','d_ff','n_layers','head_width','dropout'),
        training=('optimizer','loss','epochs','batch_size','learning_rate','weight_decay','betas','eps','patience','save_every_n_epochs','auxiliary_weight','observed_weight','clip_grad_norm','early_stopping'),
        runtime=('device','gpu_index','amp_dtype','num_workers','pin_memory','persistent_workers','cudnn_benchmark','matmul_precision','fused_adamw'),
        scheduler=('name','t_max_epochs','min_lr'),plots=('dpi','scan_index','range_min_m','range_max_m','colormap'),
        faults=('profiles','initial_healthy_scans','short_outage_scans','long_outage_scans','recovery_scans'))
    for section,keys in required.items():
        for key in keys:
            if key not in config[section]: raise ValueError(f'Missing CONFIG.{section}.{key}')
    def check(value,path):
        if value is None or isinstance(value,str) and not value.strip(): raise ValueError(f'Fill {path}')
        if isinstance(value,float) and not math.isfinite(value): raise ValueError(f'Nonfinite {path}')
        if isinstance(value,dict):
            for k,v in value.items(): check(v,path+'.'+k)
        elif isinstance(value,(list,tuple)):
            if not value: raise ValueError(f'Empty {path}')
            for i,v in enumerate(value): check(v,path+'.'+str(i))
    check(config,'CONFIG')
    d,m,t,r,s,f = [config[k] for k in ('data','model','training','runtime','scheduler','faults')]
    if d['split_policy']!='chronological' or not 0<d['train_fraction']<1 or not math.isclose(d['train_fraction']+d['val_fraction'],1):
        raise ValueError('Use chronological train/validation fractions summing to one.')
    for section,keys in ((m,('window_size','d_model','n_heads','d_ff','n_layers','head_width')), (t,('epochs','batch_size','patience'))):
        for key in keys:
            if isinstance(section[key],bool) or not isinstance(section[key],int) or section[key]<1: raise ValueError(f'{key} must be positive integer')
    if m['window_size']<2 or m['d_model']%2 or m['d_model']%m['n_heads'] or not 0<=m['dropout']<1: raise ValueError('Invalid attention dimensions/dropout')
    if r['device'] not in ('cuda','cpu') or r['amp_dtype'] not in ('bf16','fp16'): raise ValueError('Invalid runtime device/precision')
    if t['optimizer']!='adamw' or t['loss']!='mse' or s['name']!='cosine' or s['t_max_epochs']<t['epochs']: raise ValueError('Use AdamW/MSE and cosine covering the full run')
    if not 0<=s['min_lr']<t['learning_rate'] or t['weight_decay']<0 or t['auxiliary_weight']<0 or t['observed_weight']<0: raise ValueError('Invalid optimization settings')
    if not set(f['profiles'])<= {'healthy','intermittent','long'} or len(set(f['profiles']))!=len(f['profiles']) or not set(f['profiles'])&{'intermittent','long'}: raise ValueError('Profiles need at least one outage condition')
    for key in ('short_outage_scans','long_outage_scans'):
        a,b=f[key]
        if not isinstance(a,int) or not isinstance(b,int) or not 1<=a<=b: raise ValueError('Invalid outage lengths')
    if f['initial_healthy_scans']<1 or f['recovery_scans']<1: raise ValueError('Fault healthy/recovery lengths must be positive')
    if r['num_workers']<0 or r['persistent_workers'] and r['num_workers']==0: raise ValueError('Invalid loader workers')
    if t['clip_grad_norm']<=0 or t['eps']<=0 or len(t['betas'])!=2 or any(not 0<=b<1 for b in t['betas']): raise ValueError('Invalid optimizer parameters')
    if not config['run']['name'] or any(c in config['run']['name'] for c in '<>:"/\\|?*'): raise ValueError('Use a Windows-safe run name')
    for section,key in [('run','seed'),('runtime','gpu_index'),('runtime','num_workers'),('training','save_every_n_epochs'),('plots','scan_index')]:
        value=config[section][key]
        if isinstance(value,bool) or not isinstance(value,int) or value<0: raise ValueError(f'{section}.{key} must be a nonnegative integer')
    for section,key in [('data','verify_hashes'),('training','early_stopping'),('runtime','pin_memory'),('runtime','persistent_workers'),('runtime','cudnn_benchmark'),('runtime','fused_adamw')]:
        if not isinstance(config[section][key],bool): raise ValueError(f'{section}.{key} must be boolean')
    if r['matmul_precision'] not in ('highest','high','medium'): raise ValueError('Invalid matmul precision')
    if not 0<=config['plots']['range_min_m']<config['plots']['range_max_m'] or config['plots']['dpi']<1: raise ValueError('Invalid plot settings')
    json.dumps(config,allow_nan=False)
    return config


def seed_training(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def write_json(path, value):
    path=Path(path); temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8'); os.replace(temporary,path)


def save_checkpoint(path, value):
    path=Path(path); temporary=path.with_name(path.name+'.tmp')
    torch.save(value,temporary); os.replace(temporary,path)


def source_hashes():
    return {name: hashlib.sha256(Path(__file__).with_name(name).read_text(encoding='utf-8').encode()).hexdigest()
            for name in ('dense_saits_single_episode.py','saits_endpoint_model.py')}


def _autocast(device,config):
    return torch.autocast(device.type,dtype=torch.bfloat16 if config['runtime']['amp_dtype']=='bf16' else torch.float16,enabled=device.type=='cuda')


def _loss(output,target,hidden,config):
    weights=torch.where(hidden,torch.ones_like(target),torch.full_like(target,config['training']['observed_weight']))
    def term(prediction): return ((prediction.float()-target).square()*weights).sum()/weights.sum().clamp_min(1)
    return term(output['prediction']) + config['training']['auxiliary_weight']*.5*(term(output['stage1'])+term(output['stage2']))


@torch.inference_mode()
def evaluate(model,loader,config,device):
    model.eval(); squared=torch.zeros((),device=device); absolute=squared.clone(); total=0; returning=0; outside=0
    persistence_error=squared.clone(); prediction_rows=[]; target_rows=[]; hidden_rows=[]
    for values,mask,target,hidden in loader:
        values,mask,target,hidden=[v.to(device,non_blocking=True) for v in (values,mask,target,hidden)]
        with _autocast(device,config): prediction=model(values,mask)['prediction'].float()
        if not torch.isfinite(prediction).all(): raise FloatingPointError('Nonfinite validation predictions')
        difference=(prediction-target)[hidden]
        squared+=difference.square().sum(); absolute+=difference.abs().sum(); total+=int(hidden.sum())
        returning+=int(((target<7)&hidden).sum()); outside+=int((((prediction<0)|(prediction>7))&hidden).sum())
        seen=mask[:,:,:BEAMS]; last=seen.shape[1]-1-seen.flip(1).long().argmax(1)
        baseline=values[:,:,:BEAMS].gather(1,last[:,None]).squeeze(1)*7
        baseline=torch.where(seen.any(1),baseline,torch.full_like(baseline,7))
        persistence_error+=((baseline-target)[hidden]).square().sum()
        prediction_rows.append(prediction.cpu().numpy()); target_rows.append(target.cpu().numpy()); hidden_rows.append(hidden.cpu().numpy())
    if not total: raise ValueError('Validation requires artificially hidden targets')
    return dict(mse=float(squared/total),mae=float(absolute/total),persistence_mse=float(persistence_error/total),
                hidden_beams=total,returning_hidden_beams=returning,no_return_only=returning==0,outside_physical_bounds=outside), (np.concatenate(prediction_rows),np.concatenate(target_rows),np.concatenate(hidden_rows))


def train_saits_baseline(model,train_data,val_data,*,config,split_ids,source_metadata,normalizer):
    config=copy.deepcopy(validate_config(config)); t,r,s=[config[k] for k in ('training','runtime','scheduler')]
    if model.model_config!=config['model']: raise ValueError('Model and configured architecture differ')
    if train_data.split!='train' or val_data.split!='validation' or train_data.corpus is not val_data.corpus: raise ValueError('Use disjoint folds from one selected episode')
    corpus=train_data.corpus
    if split_ids!=corpus.splits or source_metadata!=corpus.source_metadata: raise ValueError('Selected scan provenance changed')
    if normalizer!=fit_normalizer(corpus): raise ValueError('Normalizer must be fitted only to native training rows')
    if train_data.normalizer!=normalizer or val_data.normalizer!=normalizer: raise ValueError('Dataset normalization differs from the saved transform')
    expected_spec=dict(window_size=config['model']['window_size'],seed=config['run']['seed'],faults=config['faults'])
    if train_data.augmentation_spec!=expected_spec or val_data.augmentation_spec!=expected_spec: raise ValueError('Rebuild datasets after changing window, seed or fault settings')
    if resolve_episode_directory(config['paths']['repo_root'],config['data']['episode_path']).name!=corpus.episode_id:
        raise ValueError('Configuration episode differs from loaded data')
    if corpus.split_ranges['train'][1]!=math.floor(len(corpus.states)*config['data']['train_fraction']):
        raise ValueError('Rebuild datasets after changing chronological fractions')
    if file_sha256(source_metadata['npz_path'])!=source_metadata['npz_sha256']:
        raise ValueError('Episode changed since loading')
    if config['plots']['scan_index']>=len(val_data): raise ValueError('plots.scan_index exceeds validation dataset')
    device=torch.device(r['device'],r['gpu_index']) if r['device']=='cuda' else torch.device('cpu')
    if device.type=='cuda':
        if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; select the GPU kernel')
        torch.cuda.set_device(device)
        if r['amp_dtype']=='bf16' and not torch.cuda.is_bf16_supported(): raise ValueError('Choose FP16 on this GPU')
    seed_training(config['run']['seed']); model=model.to(device)
    torch.set_float32_matmul_precision(r['matmul_precision']); torch.backends.cudnn.benchmark=r['cudnn_benchmark']
    generator=torch.Generator().manual_seed(config['run']['seed'])
    options=dict(batch_size=t['batch_size'],num_workers=r['num_workers'],persistent_workers=r['persistent_workers'],pin_memory=r['pin_memory'] and device.type=='cuda')
    train_loader=DataLoader(train_data,shuffle=True,generator=generator,**options); val_loader=DataLoader(val_data,shuffle=False,**options)
    optimizer=torch.optim.AdamW(model.parameters(),lr=t['learning_rate'],weight_decay=t['weight_decay'],betas=tuple(t['betas']),eps=t['eps'],fused=r['fused_adamw'] and device.type=='cuda')
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,s['t_max_epochs'],eta_min=s['min_lr'])
    scaler=torch.amp.GradScaler(device.type,enabled=device.type=='cuda' and r['amp_dtype']=='fp16')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    run_dir=Path(config['paths']['output_root']).expanduser().resolve()/f"{config['run']['name']}_{stamp}_{uuid4().hex[:8]}"
    run_dir.mkdir(parents=True,exist_ok=False)
    logger=logging.Logger('dense_saits',logging.INFO)
    for handler in (logging.StreamHandler(sys.stdout),logging.FileHandler(run_dir/'training.log',encoding='utf-8')):
        logger.addHandler(handler)
    keys=('epoch','train_mse','train_rmse','val_mse','val_rmse','val_mae','learning_rate','next_learning_rate','elapsed_s','gpu_peak_allocated_mib','stale_epochs','improved')
    history={k:[] for k in keys}; best=float('inf'); best_epoch=None; stale=0
    metadata=dict(source_metadata,schema_version=SCHEMA,model_type='dense-saits-stage1-single-episode',stage='reconstruction-only',
                  created_at_utc=datetime.now(timezone.utc).isoformat(),status='running',torch_version=str(torch.__version__),
                  gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else None,source_hashes=source_hashes(),
                  validation_metric_scope='artificially hidden beams only; physical metre units; no output clipping',
                  train_profiles=list(train_data.profiles),barrier_enabled=False,closed_loop_training=False)
    for name,value in [('config',config),('source',source_metadata),('splits',split_ids),('normalizer',normalizer),('summary',metadata),('architecture',dict(schema_version=SCHEMA,model_config=config['model']))]: write_json(run_dir/(name+'.json'),value)
    try:
        with (run_dir/'metrics.csv').open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=keys); writer.writeheader()
            for epoch in range(1,t['epochs']+1):
                started=perf_counter(); model.train(); train_sum=torch.zeros((),device=device); n=0; lr=optimizer.param_groups[0]['lr']
                if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
                for values,mask,target,hidden in train_loader:
                    values,mask,target,hidden=[v.to(device,non_blocking=True) for v in (values,mask,target,hidden)]
                    optimizer.zero_grad(set_to_none=True)
                    with _autocast(device,config): output=model(values,mask); loss=_loss(output,target,hidden,config)
                    if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training loss')
                    scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(),t['clip_grad_norm'],error_if_nonfinite=not scaler.is_enabled())
                    scaler.step(optimizer); scaler.update()
                    train_sum+=(output['prediction'].detach().float()-target).square().sum(); n+=target.numel()
                metrics,_=evaluate(model,val_loader,config,device)
                improved=metrics['mse']<best
                if improved: best,best_epoch,stale=metrics['mse'],epoch,0
                else: stale+=1
                scheduler.step()
                train_mse=float(train_sum/n)
                row=dict(epoch=epoch,train_mse=train_mse,train_rmse=math.sqrt(train_mse),val_mse=metrics['mse'],val_rmse=math.sqrt(metrics['mse']),val_mae=metrics['mae'],learning_rate=lr,next_learning_rate=optimizer.param_groups[0]['lr'],elapsed_s=perf_counter()-started,gpu_peak_allocated_mib=torch.cuda.max_memory_allocated(device)/2**20 if device.type=='cuda' else 0,stale_epochs=stale,improved=improved)
                for k in keys: history[k].append(row[k])
                writer.writerow(row); stream.flush(); write_json(run_dir/'metrics.json',history)
                payload=dict(schema_version=SCHEMA,epoch=epoch,best_epoch=best_epoch,best_val_mse=best,model_state_dict=model.state_dict(),model_config=model.model_config,
                    optimizer_state_dict=optimizer.state_dict(),scheduler_state_dict=scheduler.state_dict(),scaler_state_dict=scaler.state_dict(),torch_rng_state=torch.get_rng_state(),
                    cuda_rng_states=torch.cuda.get_rng_state_all() if device.type=='cuda' else [],loader_rng_state=generator.get_state(),config=config,history=history,
                    source_metadata=source_metadata,split_ids=split_ids,normalizer=normalizer,source_hashes=metadata['source_hashes'])
                save_checkpoint(run_dir/'last.pt',dict(payload,checkpoint_role='last'))
                if improved: save_checkpoint(run_dir/'best.pt',dict(payload,checkpoint_role='best'))
                if t['save_every_n_epochs'] and epoch%t['save_every_n_epochs']==0: save_checkpoint(run_dir/f'epoch_{epoch:04d}.pt',payload)
                logger.info('epoch=%03d/%03d | train_mse=%.5f | val_hidden_mse=%.5f | val_mae=%.5f | lr=%.2e | %.1fs | %s',epoch,t['epochs'],train_mse,best if improved else metrics['mse'],metrics['mae'],lr,row['elapsed_s'],'BEST' if improved else 'stale='+str(stale))
                if t['early_stopping'] and stale>=t['patience']: break
        selected=torch.load(run_dir/'best.pt',map_location=device,weights_only=True); model.load_state_dict(selected['model_state_dict']); model.eval()
        metrics,arrays=evaluate(model,val_loader,config,device)
        np.savez_compressed(run_dir/'validation_predictions.npz',prediction=arrays[0],target=arrays[1],hidden_mask=arrays[2],endpoint_scan_indices=np.tile(np.arange(val_data.start,val_data.stop),len(val_data.profiles)),profile_names=np.repeat(val_data.profiles,val_data.stop-val_data.start))
        # Model-specific completion states also keep the older CNN resolver
        # (which recursively searches results/) from selecting a SAITS run.
        metadata.update(status='stage1_completed' if len(history['epoch'])==t['epochs'] else 'stage1_early_stopped',best_epoch=best_epoch,best_val_mse=best,completed_epochs=len(history['epoch']),validation_metrics=metrics)
        write_json(run_dir/'summary.json',metadata); plot_run_history(run_dir,show=False); plot_validation_example(model,val_data,run_dir,show=False)
        logger.info('Saved %s',run_dir)
        return model,history,run_dir
    except BaseException as error:
        metadata.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=str(error),completed_epochs=len(history['epoch']))
        write_json(run_dir/'summary.json',metadata); raise
    finally:
        for handler in list(logger.handlers): handler.close(); logger.removeHandler(handler)


def resolve_episode_directory(repo_root, episode_path):
    """Accept an episode ID, data/raw/<ID>, or an absolute raw episode folder."""
    root = Path(repo_root).expanduser().resolve()
    raw_root = (root / "data" / "raw").resolve()
    value = Path(episode_path).expanduser()
    if not value.is_absolute():
        value = raw_root / value if len(value.parts) == 1 else root / value
    value = value.resolve()
    if value.parent != raw_root:
        raise ValueError("Select one episode folder directly inside repo_root/data/raw.")
    return value


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_dataset(repo_root, episode_path, train_fraction, val_fraction, *,
                  split_policy, verify_hashes):
    """Load only the selected episode and freeze disjoint chronological scan folds."""
    root = Path(repo_root).expanduser().resolve()
    episode_dir = resolve_episode_directory(root, episode_path)
    manifest_path = episode_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing raw episode manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    episode_id = episode_dir.name
    if str(manifest["runId"]) != episode_id:
        raise ValueError("The raw folder name differs from the manifest episode ID.")
    if split_policy != "chronological":
        raise ValueError("This single-path notebook uses a chronological scan split.")
    if not (math.isfinite(train_fraction) and math.isfinite(val_fraction)
            and train_fraction > 0 and val_fraction > 0
            and math.isclose(train_fraction + val_fraction, 1, rel_tol=0, abs_tol=1e-12)):
        raise ValueError("Positive train/validation fractions must sum to one.")

    # The raw folder supplies identity; the existing Python export supplies aligned scans.
    npz_path = root / "data" / "processed" / "lidar_episode_v1" / episode_id / "episode.npz"
    if not npz_path.is_file():
        raise FileNotFoundError(
            f"No Python export for selected raw episode: {npz_path}. "
            "Export this episode to the existing lidar_episode_v1 format first."
        )
    npz_sha256 = file_sha256(npz_path)
    report_path = None
    expected_count = None
    if verify_hashes:
        matches = []
        for candidate in sorted((root / "data" / "batches").glob("*/python_export_qa.json")):
            report = json.loads(candidate.read_text(encoding="utf-8-sig"))
            entries = [item for item in report.get("files", []) if item["runId"] == episode_id]
            if entries:
                if report.get("allExportChecksPassed") is not True:
                    raise ValueError(f"Selected episode's export audit did not pass: {candidate}")
                matches.extend((candidate, entry) for entry in entries)
        if not matches:
            raise ValueError(
                "No export checksum report found for this episode. Supply its export audit "
                "or explicitly set data.verify_hashes=False for an unaudited export."
            )
        if any(entry["npzSha256"] != npz_sha256 for _, entry in matches):
            raise ValueError("Selected episode's NPZ checksum differs from its export report.")
        if len({int(entry["scanCount"]) for _, entry in matches}) != 1:
            raise ValueError("Export reports disagree on this episode's scan count.")
        report_path, entry = matches[0]
        expected_count = int(entry["scanCount"])

    fields = ("time_s", "sample_index", "position_ned_m", "velocity_ned_mps",
              "attitude_ned_rpy_rad", "lidar_ranges_m", "lidar_valid_return_mask",
              "lidar_availability_mask", "lidar_missing_mask", "lidar_enabled")
    with np.load(npz_path, allow_pickle=False) as archive:
        required = set(fields) | {"run_id", "schema_version", "metadata_json"}
        if not required.issubset(archive.files):
            raise ValueError(f"Selected NPZ is missing fields: {required - set(archive.files)}")
        if (str(archive["run_id"].item()) != episode_id
                or str(archive["schema_version"].item()) != "ares-python-lidar-episode-v1"):
            raise ValueError("Selected NPZ episode identity or schema is invalid.")
        export_metadata = json.loads(str(archive["metadata_json"].item()))
        if (export_metadata["lidar_grid_size"] != [15, 716]
                or export_metadata["lidar_channel_order"] !=
                "elevation-index-fastest-then-azimuth-index"):
            raise ValueError("Selected episode has incompatible LiDAR beam ordering.")
        episode = {name: archive[name] for name in fields}

    times = np.asarray(episode["time_s"], dtype=np.float64)
    count = len(times)
    if count < 2 or times.shape != (count,) or not np.isfinite(times).all():
        raise ValueError("An episode needs at least two finite timestamps.")
    if not np.allclose(np.diff(times), 0.1, rtol=0, atol=1e-6):
        raise ValueError("Expected strictly increasing, aligned 10 Hz scan timestamps.")
    if expected_count is not None and count != expected_count:
        raise ValueError("Episode scan count differs from its export report.")
    if not np.array_equal(episode["sample_index"], np.arange(1, count + 1)):
        raise ValueError("Episode sample indices must be contiguous and one-based.")
    for field in ("position_ned_m", "velocity_ned_mps", "attitude_ned_rpy_rad"):
        if episode[field].shape != (count, 3) or not np.isfinite(episode[field]).all():
            raise ValueError(f"Invalid state array: {field}")
    ranges = episode["lidar_ranges_m"]
    if ranges.shape != (count, 15 * 716) or not np.isfinite(ranges).all():
        raise ValueError("Expected finite full-resolution [T, 10740] ranges.")
    for field in ("lidar_valid_return_mask", "lidar_availability_mask", "lidar_missing_mask"):
        if episode[field].shape != ranges.shape or not np.isin(episode[field], [0, 1]).all():
            raise ValueError(f"Invalid LiDAR mask: {field}")
    if (episode["lidar_enabled"].shape != (count,)
            or not np.equal(episode["lidar_enabled"], True).all()
            or not episode["lidar_availability_mask"].all()
            or episode["lidar_missing_mask"].any()):
        raise ValueError("Selected episode must contain measured, healthy LiDAR.")
    maximum = float(manifest["static"]["maximum_sensor_range_m"])
    valid = episode["lidar_valid_return_mask"].astype(bool)
    if (not math.isclose(maximum, 7.0) or not ((ranges >= 0) & (ranges <= maximum)).all()
            or not (ranges[valid] > 0).all() or not (ranges[~valid] == maximum).all()):
        raise ValueError("Invalid recorded range or no-return encoding.")

    states = np.column_stack((
        episode["position_ned_m"], episode["velocity_ned_mps"],
        episode["attitude_ned_rpy_rad"],
    )).astype(np.float32, copy=False)
    scans = ranges.reshape(count, 716, 15).transpose(0, 2, 1).astype(np.float32, copy=False)
    boundary = int(math.floor(count * train_fraction))
    if boundary < 1 or boundary >= count:
        raise ValueError("The chosen fractions leave an empty train or validation fold.")
    split_ranges = {"train": [0, boundary], "validation": [boundary, count]}
    splits = {
        name: [f"{episode_id}:{index + 1}" for index in range(start, stop)]
        for name, (start, stop) in split_ranges.items()
    }
    source_metadata = {
        "episode_id": episode_id, "episode_path": str(episode_dir),
        "npz_path": str(npz_path), "npz_sha256": npz_sha256,
        "manifest_sha256": file_sha256(manifest_path),
        "checksum_verified": verify_hashes,
        "checksum_report": str(report_path) if report_path is not None else None,
        "split_policy": split_policy, "total_scans": count,
        "split_ranges_zero_based_end_exclusive": split_ranges,
        "validation_scope": "later scans of the same path; not unseen-path evaluation",
    }
    return SimpleNamespace(
        episode_id=episode_id, states=states, scans=scans, times=times,
        splits=splits, split_ranges=split_ranges, source_metadata=source_metadata,
        diagnostics={name: {
            "scan_count": stop - start,
            "first_time_s": float(times[start]), "last_time_s": float(times[stop - 1]),
        } for name, (start, stop) in split_ranges.items()},
    )


def plot_run_history(run_dir, *, show):
    """Read persisted metrics; works after restarting the notebook kernel."""
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    history = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    if not history["epoch"]:
        raise ValueError("This run has no completed epochs to plot.")
    plots = run_dir / "plots"
    plots.mkdir(exist_ok=True)
    best_index = int(np.argmin(history["val_mse"]))
    best_epoch = history["epoch"][best_index]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for ax, keys, title, ylabel in (
        (axes[0], ("train_mse", "val_mse"), "Training and validation loss", "MSE (mÂ²)"),
        (axes[1], ("train_rmse", "val_rmse"), "Training and validation range error", "RMSE (m)"),
    ):
        for key, label in zip(keys, ("Train", "Validation")):
            ax.plot(history["epoch"], history[key], label=label)
        ax.axvline(best_epoch, color="gray", linestyle="--",
                   label=f"Best validation: epoch {best_epoch}")
        ax.set(title=title, xlabel="Epoch", ylabel=ylabel)
        ax.grid(alpha=0.25)
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend()
    fig.savefig(plots / "loss_curves.png", dpi=config["plots"]["dpi"])
    if show:
        plt.show()
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3), constrained_layout=True)
    ax.plot(history["epoch"], history["learning_rate"], label="LR used for epoch")
    ax.set(title="Cosine learning-rate schedule", xlabel="Epoch", ylabel="Learning rate")
    ax.grid(alpha=0.25)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend()
    fig.savefig(plots / "learning_rate.png", dpi=config["plots"]["dpi"])
    if show:
        plt.show()
    plt.close(fig)


def plot_validation_example(model, val_data, run_dir, *, show):
    """Compare a measured scan with the best model's prediction."""
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    p = config["plots"]
    index = p["scan_index"]
    if index >= len(val_data):
        raise ValueError("plots.scan_index exceeds the validation dataset.")
    # Prefer a hidden endpoint over a healthy-profile example.
    hidden_indices = [i for i in range(len(val_data)) if val_data.raw_window(i)[3].any()]
    index = hidden_indices[min(index, len(hidden_indices)-1)]
    values, mask, target, _ = val_data[index]
    device = next(model.parameters()).device
    model.eval()
    with torch.inference_mode(), _autocast(device, config):
        flat = model(values[None].to(device), mask[None].to(device))["prediction"][0].float().cpu().numpy()
    prediction = flat.reshape(716, 15).T
    measured = target.numpy().reshape(716, 15).T
    error = np.abs(prediction - measured)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)
    for ax, values, title in zip(
        axes, (measured, prediction, error),
        ("Measured range", "Predicted range (best checkpoint)", "Absolute range error"),
    ):
        kwargs = dict(vmin=p["range_min_m"], vmax=p["range_max_m"])
        if ax is axes[2]:
            kwargs = dict(vmin=0, vmax=max(float(error.max()), 1e-9))
        im = ax.imshow(values, aspect="auto", origin="lower",
                       cmap=p["colormap"], **kwargs)
        ax.set(title=title, xlabel="Azimuth bin", ylabel="Elevation bin")
        fig.colorbar(im, ax=ax, label="Meters")
    fig.suptitle(f"Validation sample {index} | MAE = {error.mean():.4f} m")
    plots = run_dir / "plots"
    plots.mkdir(exist_ok=True)
    fig.savefig(plots / "validation_scan.png", dpi=p["dpi"])
    if show:
        plt.show()
    plt.close(fig)
