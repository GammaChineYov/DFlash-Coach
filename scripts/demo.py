#!/usr/bin/env python3
"""Offline synthetic demonstration: no GPU, model files, network or API key."""
import argparse
from pathlib import Path
import hashlib
import json
from common import atomic_json,shard_rows,write_rows
from mentor import apply_decision
from strategy import trend_summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,default=Path('runs/demo'));a=p.parse_args()
    if a.run_dir.exists():raise SystemExit('Choose a fresh demo --run-dir')
    rows=[{'id':f'synthetic_{i:03d}','prompt_tokens':80+i,'project':'example',
           'source_path':f'file_{i}.txt','source_sha256':hashlib.sha256(f'synthetic:{i}'.encode()).hexdigest(),
           'split':'train','prompt':f'SYNTHETIC DEMO INPUT {i}. No source data.'} for i in range(24)]
    shards=list(shard_rows(rows,8,64*2**20,32))
    state={'simulation':True,'learning_rate':2.5e-6,'history':[],'committed':[]}
    for index,shard in enumerate(shards):
        key=f'demo-s{index:04d}';write_rows(a.run_dir/'data'/(key+'.jsonl'),shard)
        point={'steps':(index+1)*2,'learning_rate':state['learning_rate'],
               'metrics':{'loss':[2.2,2.1,2.1][index],'prefix_tokens':1.5}}
        state['history'].append(point);state['committed'].append(key)
        decision={'action':'stop' if index==2 else 'continue',
                  'reason':'Synthetic offline decision; no remote model was called.',
                  'learning_rate':1e-6 if index==1 else None}
        apply_decision(state,decision);atomic_json(a.run_dir/'evidence/mentor'/(key+'.json'),
            {'simulation':True,'decision':decision})
    state.update(status='simulation_completed',training_performed=False,network_requests=0,
                 best_step=4,validation_trend=trend_summary(state['history'],4))
    atomic_json(a.run_dir/'evidence/run-state.json',state)
    print(json.dumps({'status':state['status'],'samples':len(rows),'shards':len(shards),
          'training_performed':False,'network_requests':0,'run_dir':str(a.run_dir)},indent=2))


if __name__=='__main__':main()
