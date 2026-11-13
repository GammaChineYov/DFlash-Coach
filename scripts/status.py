#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from common import load_config


def main():
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--config',type=Path);g.add_argument('--run-dir',type=Path);a=p.parse_args()
    run=Path(load_config(a.config)['run_dir']) if a.config else a.run_dir
    state=run/'evidence/run-state.json';plan=run/'plan.json'
    if not state.exists():
        print(json.dumps({'status':'prepared_not_started' if plan.exists() else 'not_prepared'}));return
    s=json.loads(state.read_text());out={k:s.get(k) for k in ('status','stage','epoch','next_shard','shard','learning_rate',
        'error','training_finished','training_stop_reason','production_restored','candidate_installed','simulation') if k in s}
    f=run/'outputs/training.json'
    if f.exists():
        t=json.loads(f.read_text());out['training']={k:t.get(k) for k in ('steps','best_step','best','last','train_examples')}
    f=run/'evidence/benchmark-measurements.jsonl'
    if f.exists():out['benchmark_requests_completed']=sum(bool(x.strip()) for x in f.read_text().splitlines())
    out['last_mentor_decision']=s.get('last_mentor_decision')
    print(json.dumps(out,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
