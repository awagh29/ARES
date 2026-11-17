"""Export selected full-scan stage-one SAITS to native MATLAB numeric weights."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import uuid
import warnings

import numpy as np
from scipy.io import savemat
import torch
from dense_saits_single_episode import (SCHEMA as TRAINING_SCHEMA, BEAMS, FEATURES,
    DenseSaitsReconstructor, build_dataset, build_train_val_tensors, normalize_history, source_hashes)
from export_saits_matlab import _linear, _layer

SCHEMA='ares-dense-saits-single-episode-mat-v1'

def require(condition,message):
    if not condition: raise ValueError(message)

def read_json(path):
    """Read metadata strictly, identifying the file rather than only a column."""
    path=Path(path)
    try:
        value=json.loads(path.read_text(encoding='utf-8-sig'))
    except (OSError, UnicodeError, ValueError) as error:
        raise ValueError(f"Cannot read JSON metadata '{path}': {error}") from error
    require(isinstance(value,dict),f"JSON metadata must be an object: {path}")
    return value

def sha256(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def atomic_json(path,value):
    path=Path(path); temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8'); os.replace(temporary,path)

def resolve_training_run(results_root,episode_id,training_run=None):
    """Discover intact completed runs; an explicit selection never falls back.

    The shared results tree also contains incomplete/interrupted experiments.
    Corruption in one candidate must not prevent discovery of other intact
    runs. This does not relax selected-checkpoint integrity checks in export.
    """
    explicit=training_run is not None
    candidates=[Path(training_run).resolve()] if explicit else sorted(p.parent for p in Path(results_root).rglob('source.json'))
    matches=[]
    skipped=[]
    for folder in candidates:
        try:
            if not (folder/'source.json').is_file() or not (folder/'summary.json').is_file():
                if explicit:
                    missing=[str(folder/name) for name in ('source.json','summary.json') if not (folder/name).is_file()]
                    raise ValueError('Selected run lacks source/summary metadata: '+', '.join(missing))
                continue
            source=read_json(folder/'source.json')
            # Do not even open another episode's possibly incomplete summary.
            if source.get('episode_id')!=episode_id:
                if explicit: raise ValueError('Selected run belongs to another episode: '+str(folder))
                continue
            summary=read_json(folder/'summary.json')
            valid=(summary.get('schema_version')==TRAINING_SCHEMA
                   and summary.get('status') in ('stage1_completed','stage1_early_stopped'))
            if not valid:
                if explicit: raise ValueError('Select a completed reconstruction-only dense SAITS run for this episode: '+str(folder))
                continue
            require(summary.get('episode_id')==episode_id,'Source/summary episode IDs differ: '+str(folder/'summary.json'))
            require((folder/'best.pt').is_file(),'Completed run has no best.pt: '+str(folder/'best.pt'))
            created=summary.get('created_at_utc')
            try:
                require(isinstance(created,str),'created_at_utc must be a timestamp string')
                created=datetime.fromisoformat(created)
                require(created.utcoffset() is not None,'created_at_utc must include its timezone')
            except ValueError as error:
                raise ValueError(f"Invalid created_at_utc in '{folder/'summary.json'}': {error}") from error
            matches.append((created.astimezone(timezone.utc),str(folder),folder))
        except ValueError as error:
            if explicit: raise
            message=f"Skipping unusable training-run metadata in '{folder}': {error}"
            skipped.append(message)
            warnings.warn(message,UserWarning,stacklevel=2)
    detail=(' Skipped candidates: '+'; '.join(skipped)) if skipped and not matches else ''
    require(bool(matches),f'No completed dense SAITS stage-one model for {episode_id} under {results_root}; train the notebook or explicitly select an intact completed run.'+detail)
    return max(matches)[2]

def pack_model(checkpoint,model):
    state=checkpoint['model_state_dict']; c=checkpoint['model_config']; norm=checkpoint['normalizer']
    first=np.empty((1,c['n_layers']),object); second=np.empty_like(first)
    for i in range(c['n_layers']):
        first[0,i]=_layer(state,f'first_block.{i}'); second[0,i]=_layer(state,f'second_block.{i}')
    result=dict(schemaVersion=SCHEMA,episodeId=checkpoint['source_metadata']['episode_id'],
        nSteps=c['window_size'],nFeatures=FEATURES,nTargets=BEAMS,dModel=c['d_model'],nHeads=c['n_heads'],
        headDim=c['d_model']//c['n_heads'],nLayers=c['n_layers'],contextMean=np.asarray(norm['context_mean']).reshape(1,12),
        contextStd=np.asarray(norm['context_std']).reshape(1,12),distanceScale=np.float32(7),
        positionEncoding=model.position_encoding[0].cpu().numpy(),firstBlock=first,secondBlock=second,
        predictionClipped=False,gridSize=np.array([[15,716]],np.int32),historyOrder='past-to-present; current row last',
        beamOrder='elevation-index-fastest-then-azimuth-index')
    for field,prefix in [('embeddingFirst','embedding_first'),('embeddingSecond','embedding_second'),('reduceFirst','reduce_first'),
        ('reduceSecondBeta','reduce_second_beta'),('reduceSecondGamma','reduce_second_gamma'),('gateFirst','gate_first'),('gateSecond','gate_second')]:
        result[field]=_linear(state,prefix)
    return result

def export_episode(project_root,episode_id,*,training_run=None,results_root=None,output_root=None):
    require(bool(re.fullmatch('[A-Za-z0-9-]+',episode_id)),'Invalid episode ID')
    root=Path(project_root).resolve()
    folder=resolve_training_run(results_root or root/'results',episode_id,training_run)
    cp_path=folder/'best.pt'; cp=torch.load(cp_path,map_location='cpu',weights_only=True)
    config,source,summary=read_json(folder/'config.json'),read_json(folder/'source.json'),read_json(folder/'summary.json')
    require(cp.get('schema_version')==TRAINING_SCHEMA and cp.get('checkpoint_role')=='best','Selected stage-one best.pt required')
    require(cp['config']==config and cp['source_metadata']==source and cp['normalizer']==read_json(folder/'normalizer.json'),'Checkpoint and saved metadata differ')
    require(cp['epoch']==cp['best_epoch']==summary['best_epoch'],'Checkpoint is not the selected best-validation weights')
    require(cp['source_hashes']==source_hashes(),'Architecture/training source changed since this checkpoint')
    require(source['episode_id']==episode_id,'Checkpoint episode differs')
    require(sha256(root/'data/raw'/episode_id/'manifest.json')==source['manifest_sha256'],'Raw manifest changed since training')
    require(sha256(root/'data/processed/lidar_episode_v1'/episode_id/'episode.npz')==source['npz_sha256'],'Training episode NPZ changed')
    corpus=build_dataset(root,episode_id,config['data']['train_fraction'],config['data']['val_fraction'],split_policy='chronological',verify_hashes=config['data']['verify_hashes'])
    train,val,norm=build_train_val_tensors(corpus,config)
    require(norm==cp['normalizer'] and cp['split_ids']==corpus.splits,'Normalizer or chronological split changed')
    model=DenseSaitsReconstructor(cp['model_config']).eval(); model.load_state_dict(cp['model_state_dict'],strict=True)
    require(all(value.dtype==torch.float32 and bool(torch.isfinite(value).all()) for value in model.state_dict().values()),
            'Export requires finite FP32 model tensors')
    count=val.stop-val.start
    windows=[val.raw_window(count*p+count//2)[:2] for p in range(len(val.profiles))]
    raw,mask=[v.copy() for v in windows[-1]]
    raw[:,:BEAMS]=0; mask[:,:BEAMS]=False; windows.append((raw,mask))
    raw,mask=[v.copy() for v in windows[-1]]
    raw[:-1]=0; mask[:-1]=False; windows.append((raw,mask))
    outputs=[]
    with torch.inference_mode():
        for raw,mask in windows:
            out=model(torch.from_numpy(normalize_history(raw,mask,norm)[None]),torch.from_numpy(mask[None]))['prediction'][0].numpy()
            outputs.append(out.reshape(716,15).T)
    require(all(np.isfinite(value).all() for value in outputs),'Reference inference produced nonfinite predictions')
    fixtures=dict(rawHistory=np.stack([v[0] for v in windows],-1),observedMask=np.stack([v[1] for v in windows],-1),
                  predictedRanges=np.stack(outputs,-1),absoluteTolerance=5e-4)
    checkpoint_sha=sha256(cp_path); exporter_sha=sha256(__file__)
    version=hashlib.sha256((checkpoint_sha+exporter_sha+json.dumps(source_hashes(),sort_keys=True)).encode()).hexdigest()[:16]
    episode_folder=Path(output_root).resolve()/episode_id if output_root else root/'data/models/saits_dense_single_episode'/episode_id
    episode_folder.mkdir(parents=True,exist_ok=True); destination=episode_folder/version
    if destination.exists():
        exported=read_json(destination/'export_manifest.json')
        require(exported['checkpoint_sha256']==checkpoint_sha and exported['model_mat_sha256']==sha256(destination/'model.mat'),'Published export was modified')
    else:
        staging=episode_folder/('.building_'+uuid.uuid4().hex); staging.mkdir()
        savemat(staging/'model.mat',dict(model=pack_model(cp,model),fixtures=fixtures),do_compression=True)
        exported=dict(schema_version=SCHEMA,episode_id=episode_id,training_run_dir=str(folder),checkpoint_path=str(cp_path),
            checkpoint_sha256=checkpoint_sha,model_mat_sha256=sha256(staging/'model.mat'),source_manifest_sha256=source['manifest_sha256'],
            source_npz_sha256=source['npz_sha256'],best_epoch=cp['best_epoch'],validation_mse=cp['best_val_mse'],
            model_source_sha256=source_hashes(),exporter_sha256=exporter_sha,history_length=config['model']['window_size'],
            matlab_parity_verified=False,barrier_enabled=False,closed_loop_training=False)
        atomic_json(staging/'export_manifest.json',exported); os.rename(staging,destination)
    pointer=dict(exported,model_path=str(destination/'model.mat'),export_dir=str(destination))
    atomic_json(episode_folder/'deployment.json',pointer)
    return pointer

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,required=True); parser.add_argument('--episode-id',required=True)
    parser.add_argument('--training-run',type=Path); parser.add_argument('--results-root',type=Path)
    args=parser.parse_args(); torch.set_num_threads(1)
    print(export_episode(args.project_root,args.episode_id,training_run=args.training_run,results_root=args.results_root)['model_path'])

if __name__=='__main__': main()
