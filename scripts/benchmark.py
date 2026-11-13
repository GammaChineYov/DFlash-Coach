#!/usr/bin/env python3
"""Compare control, installed parent and candidate on reserved test prompts."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import time
import urllib.request
from urllib.parse import urlparse
from common import atomic_json, read_rows, load_config, command_argv

SERVER = None
API_URL = "http://127.0.0.1:1238"


def api(path, data=None, timeout=120):
    request = urllib.request.Request(API_URL.rstrip('/') + path,
        data=None if data is None else json.dumps(data).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def stop():
    global SERVER
    if SERVER and SERVER.poll() is None:
        os.killpg(SERVER.pid, signal.SIGTERM)
        try:
            SERVER.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(SERVER.pid, signal.SIGKILL)
            SERVER.wait(timeout=10)
    SERVER = None


def variant_command(original, draft):
    argv = list(original)
    for option, value in (('--port', str(urlparse(API_URL).port or 1238)), ('--spec-type', 'none' if draft is None else 'draft-dflash')):
        argv[argv.index(option) + 1] = value
    if draft is None:
        out, i = [], 0
        while i < len(argv):
            if argv[i].startswith('--spec-draft-'):
                i += 2
            else:
                out.append(argv[i])
                i += 1
        return out
    argv[argv.index('--spec-draft-model') + 1] = str(draft)
    return argv


def main():
    global SERVER, API_URL
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    a = p.parse_args()
    cfg = load_config(a.config)
    API_URL = cfg.get("benchmark_url","http://127.0.0.1:1238")
    run = Path(cfg['run_dir'])
    rows = list(read_rows(run / 'data/benchmark-test.jsonl'))
    assert rows and all(r['dataset_split'] == 'test' for r in rows)
    argv = command_argv(cfg)
    env = dict(os.environ, **cfg.get('environment',{}))
    env['CUDA_VISIBLE_DEVICES']=','.join(cfg['gpu_uuids'])
    candidate = run / 'outputs' / cfg['output_name']
    measurements = []
    evidence_path = run / 'evidence/benchmark-measurements.jsonl'
    # Restarting the benchmark intentionally reruns it completely; no partial comparisons.
    evidence_path.write_text('')
    def interrupted(signum, frame):
        raise KeyboardInterrupt('Benchmark interrupted')
    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(sig, interrupted)
    try:
        for label, draft in (('no_draft', None), ('parent', cfg['initial_draft']), ('candidate', candidate)):
            with (run / 'evidence' / ('benchmark-' + label + '.log')).open('w') as log:
                SERVER = subprocess.Popen(variant_command(argv, draft), env=env, stdout=log,
                                          stderr=subprocess.STDOUT, start_new_session=True)
                deadline = time.monotonic() + 180
                while time.monotonic() < deadline and SERVER.poll() is None:
                    try:
                        if api('/health', timeout=1).get('status') == 'ok':
                            break
                    except (OSError, ValueError):
                        pass
                    time.sleep(1)
                else:
                    raise RuntimeError('Benchmark server did not become healthy: ' + label)
                api('/completion', {'prompt': rows[0]['prompt'], 'n_predict': 32, 'temperature': 0, 'cache_prompt': False})
                for repeat in range(2):
                    ordered = rows if repeat == 0 else list(reversed(rows))
                    for row in ordered:
                        begin = time.monotonic()
                        result = api('/completion', {'prompt': row['prompt'], 'n_predict': 512,
                            'temperature': 0, 'top_k': 1, 'top_p': 1, 'min_p': 0,
                            'repeat_penalty': 1, 'seed': 731, 'cache_prompt': False})
                        text = result.get('content', '')
                        timings = result['timings']
                        assert timings.get('cache_n', 0) == 0
                        measurement = {'variant': label, 'id': row['id'], 'repeat': repeat,
                            'language': row['language'], 'length_bucket': row['length_bucket'],
                            'output_sha256': hashlib.sha256(text.encode()).hexdigest(), 'content': text,
                            'timings': timings, 'wall_seconds': time.monotonic() - begin}
                        measurements.append(measurement)
                        with evidence_path.open('a') as stream:
                            stream.write(json.dumps(measurement, ensure_ascii=False) + '\n')
                stop()
    finally:
        stop()
    by_variant = defaultdict(dict)
    for row in measurements:
        by_variant[row['variant']][(row['id'], row['repeat'])] = row
    summary = {'test_tasks': len(rows), 'repeats': 2, 'max_new_tokens': 512, 'variants': {},
               'candidate_installed': False, 'longest_test_prompt_tokens': max(r['prompt_tokens'] for r in rows),
               'context_note': 'This benchmark does not validate the full 262144-token window.'}
    for label, records in by_variant.items():
        values = list(records.values())
        tokens = sum(x['timings']['predicted_n'] for x in values)
        seconds = sum(x['timings']['predicted_ms'] for x in values) / 1000
        summary['variants'][label] = {'requests': len(values), 'new_tokens': tokens,
            'decode_tps': tokens / seconds if seconds else 0,
            'wall_tps': tokens / sum(x['wall_seconds'] for x in values),
            'same_as_parent': sum(x['output_sha256'] == by_variant['parent'][k]['output_sha256'] for k, x in records.items()),
            'same_as_no_draft': sum(x['output_sha256'] == by_variant['no_draft'][k]['output_sha256'] for k, x in records.items()),
            'median_tps_by_language': {lang: statistics.median(x['timings']['predicted_per_second'] for x in values if x['language'] == lang) for lang in ('ru', 'en')}}
    parent = summary['variants']['parent']
    child = summary['variants']['candidate']
    summary['decode_speedup_vs_parent_percent'] = (child['decode_tps'] / parent['decode_tps'] - 1) * 100
    summary['candidate_eligible_for_manual_review'] = child['same_as_parent'] == child['requests'] and child['decode_tps'] >= parent['decode_tps']
    atomic_json(run / 'outputs/benchmark-summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
