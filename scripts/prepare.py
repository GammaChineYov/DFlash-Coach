#!/usr/bin/env python3
"""Freeze normalized train/validation/test inputs into a bounded capture plan."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
from common import atomic_json,digest,load_config,read_rows,shard_rows,validate_splits,write_rows,command_argv


def normalize(row,split):
    row=dict(row)
    if not re.fullmatch(r'[A-Za-z0-9_-]+',row['id']):
        raise ValueError('Sample IDs must contain only letters, digits, underscores and hyphens')
    if row.get('dataset_split',row.get('split',split)) not in (split, 'eval' if split=='validation' else split):
        raise ValueError('A record has the wrong split')
    for name in ('project','source_path','source_sha256','prompt','language','sampling_group'):
        if not isinstance(row.get(name),str) or not row[name]:raise ValueError('Missing string field: '+name)
    if not re.fullmatch(r'[0-9a-f]{64}',row['source_sha256']):raise ValueError('Invalid source_sha256')
    if type(row.get('prompt_tokens')) is not int or row['prompt_tokens']<=0:raise ValueError('Invalid prompt_tokens')
    row.update(split=split,dataset_split=split,source_group=row.get('source_group',row['project']+':'+row['source_path']),
               length_bucket=row.get('length_bucket','compact' if row['prompt_tokens']<=1536 else 'extended'))
    return row


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args();cfg=load_config(args.config)
    run=Path(cfg['run_dir']);source=Path(cfg['dataset'])
    if (run/'plan.json').exists() or (run/'evidence/run-state.json').exists():
        raise SystemExit('Run already exists; use a new run_dir or resume it')
    groups={split:[normalize(r,split) for r in read_rows(source/(split+'.jsonl'))]
            for split in ('train','validation','test')}
    tasks=sum(groups.values(),[]);counts=validate_splits(tasks)
    if any(not rows for rows in groups.values()):raise ValueError('All three splits must be nonempty')
    # Identical prompts and source hashes cannot cross split boundaries.
    seen={}
    for row in tasks:
        for key in (('prompt',hashlib.sha256(row['prompt'].encode()).hexdigest()),('source',row['source_sha256'])):
            old=seen.setdefault(key,row['split'])
            if old!=row['split']:raise ValueError('A prompt or source hash crosses splits')
    def choose(split,count):
        rows=sorted(groups[split],key=lambda r:hashlib.sha256(('monitor:'+r['id']).encode()).hexdigest())
        if len(rows)<count:raise ValueError(f'{split} needs at least {count} records')
        return rows[:count]
    train=groups['train'];monitor=choose('validation',cfg['validation_monitor_rows'])
    benchmark=choose('test',cfg['benchmark_test_rows'])
    if cfg['validation_monitor_rows']<16:raise ValueError('The trainer requires at least 16 validation records')
    maximum=max(r['prompt_tokens'] for r in train+monitor+benchmark)
    if maximum+cfg['max_new_tokens']+1>cfg['capture_context']:raise ValueError('capture_context is too small')
    for name in ('data','evidence','outputs','features','shards','configs'):(run/name).mkdir(parents=True,exist_ok=True)
    for name,rows in [('train-all',train),('monitor',monitor),('benchmark-test',benchmark),
                      ('validation-all',groups['validation']),('test-all',groups['test']),('tokenize-all',train+monitor+benchmark)]:
        rows=[{**r,'split':'eval' if r['dataset_split']=='validation' else r['dataset_split']} for r in rows]
        write_rows(run/'data'/(name+'.jsonl'),rows)
    ordered=sorted(train,key=lambda r:hashlib.sha256(('train:'+r['id']).encode()).hexdigest())
    shards=[]
    for index,rows in enumerate(shard_rows(ordered,cfg['max_shard_rows'],int(cfg['shard_feature_budget_gib']*2**30),cfg['max_new_tokens'])):
        key=f'e00-s{index:04d}';path=run/'data/shards'/(key+'.jsonl');write_rows(path,rows)
        shards.append({'key':key,'path':str(path),'sha256':digest(path),'rows':len(rows),
            'steps':math.ceil(len(rows)/cfg['batch_size']),
            'feature_bytes_upper_bound':sum((r['prompt_tokens']+cfg['max_new_tokens']+1)*32768 for r in rows)})
    root=Path(__file__).resolve().parents[1]
    code_files=[root/'coach.py',*sorted((root/'scripts').glob('*.py')),*sorted((root/'source').glob('*.py'))]
    binaries={str(Path(cfg[key])):digest(cfg[key]) for key in ('capture_binary','tokenize_binary')}
    server=command_argv(cfg)[0]
    binaries[server]=digest(server)
    plan={'version':1,'config':cfg,'splits':counts,'train_tasks':len(train),'shards':shards,
        'source_hashes':{split+'.jsonl':digest(source/(split+'.jsonl')) for split in groups},
        'binaries':binaries,'code_hashes':{str(p.relative_to(root)):digest(p) for p in code_files},
        'target_sha256':digest(cfg['target_model']),'initial_draft_sha256':digest(cfg['initial_draft']),
        'monitor_sha256':digest(run/'data/monitor.jsonl'),'benchmark_sha256':digest(run/'data/benchmark-test.jsonl'),
        'tokenization_rows':len(train)+len(monitor)+len(benchmark),'max_prompt_tokens':maximum,
        'steps_per_epoch_estimate':sum(s['steps'] for s in shards),
        'production_config_sha256':digest(cfg['production_config']) if cfg.get('production_config') else None,
        'production_launcher_sha256':digest(cfg['production_launcher']) if cfg.get('production_launcher') else None}
    if cfg.get('seed_checkpoint'):
        if not cfg.get('seed_plan_sha256') or type(cfg.get('seed_step')) is not int:
            raise ValueError('A seed checkpoint requires seed_plan_sha256 and seed_step')
        plan['seed_checkpoint_sha256']=digest(cfg['seed_checkpoint'])
    atomic_json(run/'plan.json',plan)
    atomic_json(run/'evidence/prepared.json',{'status':'prepared_not_started','train_tasks':len(train),
        'shards':len(shards),'max_prompt_tokens':maximum,'plan_sha256':digest(run/'plan.json')})
    print(json.dumps({'status':'prepared_not_started','splits':counts,'shards':len(shards),'max_prompt_tokens':maximum}))


if __name__=='__main__':main()
