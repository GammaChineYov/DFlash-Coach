#!/usr/bin/env python3
"""Validate an immutable plan, its artifacts, disk capacity and mentor connectivity."""
import argparse
import json
import os
from pathlib import Path
import subprocess
from common import atomic_json,digest,load_config,read_rows
from mentor import read_key,review


def check(config_path,api_smoke=False):
    cfg=load_config(config_path);run=Path(cfg['run_dir']);root=Path(__file__).resolve().parents[1]
    plan=json.loads((run/'plan.json').read_text())
    if cfg!=plan['config']:raise ValueError('Configuration changed; prepare a new run_dir')
    for key in ('production_config','production_launcher'):
        if cfg.get(key) and digest(cfg[key])!=plan[key+'_sha256']:raise ValueError(key+' changed')
    for key in ('target','initial_draft'):
        path=cfg['target_model'] if key=='target' else cfg['initial_draft']
        if digest(path)!=plan[key+'_sha256']:raise ValueError(key+' weights changed')
    if cfg.get('seed_checkpoint') and digest(cfg['seed_checkpoint'])!=plan['seed_checkpoint_sha256']:
        raise ValueError('Seed checkpoint changed')
    for name,expected in plan['source_hashes'].items():
        if digest(Path(cfg['dataset'])/name)!=expected:raise ValueError('Input changed: '+name)
    for name,expected in plan['code_hashes'].items():
        if digest(root/name)!=expected:raise ValueError('Code changed: '+name)
    for name,expected in plan['binaries'].items():
        if digest(name)!=expected:raise ValueError('Binary changed: '+name)
    for shard in plan['shards']:
        if digest(shard['path'])!=shard['sha256']:raise ValueError('Shard changed')
    for name,key in (('monitor','monitor_sha256'),('benchmark-test','benchmark_sha256')):
        if digest(run/'data'/(name+'.jsonl'))!=plan[key]:raise ValueError('Holdout changed')
    tokens=json.loads((run/'evidence/token-verification.json').read_text())
    if tokens['invalid'] or tokens['mismatches'] or tokens['rows']!=plan['tokenization_rows']:
        raise ValueError('Prompt token verification failed')
    free=os.statvfs(run).f_bavail*os.statvfs(run).f_frsize
    val=sum((r['prompt_tokens']+cfg['max_new_tokens']+1)*32768 for r in read_rows(run/'data/monitor.jsonl'))
    checkpoint=int(cfg.get('checkpoint_bytes_estimate',7500000000))
    required_total=val+int(cfg['shard_feature_budget_gib']*2**30)+2*checkpoint+Path(cfg['initial_draft']).stat().st_size
    allocated=sum(p.stat().st_size for sub in ('features','shards','outputs') for p in (run/sub).rglob('*') if p.is_file() and not p.is_symlink())
    required=max(0,required_total-allocated)
    if free<required:raise RuntimeError(f'Need {required/2**30:.1f} GiB free; available {free/2**30:.1f}')
    read_key(cfg['mentor'])
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',**cfg.get('environment',{}))
    if cfg.get('gguf_python_dir'):env['PYTHONPATH']=cfg['gguf_python_dir']+os.pathsep+env.get('PYTHONPATH','')
    code='import json,torch,numpy,transformers,gguf;print(json.dumps({"torch":torch.__version__,"numpy":numpy.__version__,"transformers":transformers.__version__}))'
    versions=json.loads(subprocess.check_output([cfg['python'],'-c',code],env=env,text=True))
    report={'status':'cpu_preflight_passed','free_gib':round(free/2**30,2),'required_gib':round(required/2**30,2),
        'versions':versions,'gpu_training_tested':False,'capture_tested':False,'plan_sha256':digest(run/'plan.json')}
    if api_smoke:
        report['mentor_setup']=review(cfg['mentor'],run,'setup-smoke',{'phase':'setup_only','training_started':False,
            'tasks':plan['splits'],'shards':len(plan['shards']),'epoch_limit':cfg.get('max_epochs')})
        if report['mentor_setup']['action']!='continue':raise RuntimeError('Mentor setup was not accepted')
        report['mentor_api_tested']=True
    atomic_json(run/'evidence/preflight.json',report)
    return cfg,plan,report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--api-smoke',action='store_true')
    a=p.parse_args();print(json.dumps(check(a.config,a.api_smoke)[2],indent=2))
