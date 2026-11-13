#!/usr/bin/env python3
"""Explicitly launched, resumable shard capture/training with a mentor veto."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import urllib.request
from common import atomic_json, digest, metric_guard, read_rows, write_rows, load_config
from mentor import review, apply_decision
from preflight import check
from strategy import pending_strategy, trend_summary

CHILD = None

def stop_child():
    global CHILD
    if CHILD and CHILD.poll() is None:
        os.killpg(CHILD.pid, signal.SIGTERM)
        try:
            CHILD.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(CHILD.pid, signal.SIGKILL)
            CHILD.wait(timeout=10)
    CHILD = None


def local_api(path, data=None, port=1234, timeout=10, base_url=None):
    req = urllib.request.Request((base_url or f'http://127.0.0.1:{port}').rstrip('/') + path,
        data=None if data is None else json.dumps(data).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def capture_command(cfg, rows):
    maximum = max(row['prompt_tokens'] for row in rows)
    assert maximum + cfg['max_new_tokens'] <= cfg['capture_context']
    return [cfg['capture_binary'], '-m', cfg['target_model'], '--device', ','.join('CUDA'+str(i) for i in range(len(cfg['gpu_uuids']))),
            '--split-mode', 'none', '--gpu-layers', 'all', '--override-tensor',
            '^per_layer_token_embd[.]weight$=CUDA1,.*=CUDA0' if len(cfg['gpu_uuids'])>1 else '.*=CUDA0', '--lazy-mode', 'off', '--no-host',
            '--fit', 'off', '-c', str(cfg['capture_context']), '-b', '512', '-ub', '256',
            '--flash-attn', 'on', '--cache-type-k', 'bf16', '--cache-type-v', 'bf16']


def run_stage(cmd, env, log, cfg):
    global CHILD
    with Path(log).open('a') as stream:
        CHILD = subprocess.Popen(cmd, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        while CHILD.poll() is None:
            fs = os.statvfs(cfg['run_dir'])
            if fs.f_bavail <= 0:
                stop_child()
                raise RuntimeError('Filesystem is full; progress retained')
            time.sleep(2)
        rc = CHILD.returncode
        CHILD = None
    if rc:
        raise RuntimeError(f'Stage failed ({rc}); see {log}')


def verify_features(rows, directory, max_new):
    for row in rows:
        p = Path(directory) / (row['id'] + '.json')
        meta = json.loads(p.read_text())
        assert meta['prompt'] == row['prompt'] and meta['source_sha256'] == row['source_sha256']
        assert meta['feature_dtype'] == 'float16' and meta['max_new_tokens'] == max_new
        assert meta['feature_shape'] == [len(meta['input_ids']), 8, 2048]
        assert meta['prompt_n'] == row['prompt_tokens']
        assert p.with_suffix('.f16').stat().st_size == len(meta['input_ids']) * 32768


def capture(cfg, prompts, directory, env, log):
    rows = list(read_rows(prompts))
    directory = Path(directory)
    directory.mkdir(exist_ok=True)
    # Existing metadata is reusable only after exact prompt/hash/length/size verification.
    for row in rows:
        if (directory / (row['id'] + '.json')).exists():
            verify_features([row], directory, cfg['max_new_tokens'])
    run_stage(capture_command(cfg, rows), dict(env, DFLASH_PROMPTS=str(prompts),
              DFLASH_FEATURES=str(directory)), log, cfg)
    verify_features(rows, directory, cfg['max_new_tokens'])
    return rows


def service_control(action, service, scope='user'):
    if not service:
        return
    command = ['/usr/bin/systemctl']
    if scope == 'user':
        command += ['--user']
    elif scope != 'system':
        raise ValueError('production_scope must be user or system')
    command += [action, service]
    if scope == 'system' and os.geteuid() != 0:
        command = ['sudo', '-n'] + command
    subprocess.run(command, check=True)


def restore(cfg, marker):
    stop_child()
    if marker.exists():
        service_control('start', cfg['production_service'], cfg.get('production_scope','user'))
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            try:
                if local_api('/health', timeout=1, base_url=cfg.get('production_url')).get('status') == 'ok':
                    marker.unlink()
                    return
            except (OSError, ValueError):
                pass
            time.sleep(1)
        raise RuntimeError('target model service restart did not become healthy')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--start-authorized', action='store_true')
    p.add_argument('--restore-only', action='store_true')
    a = p.parse_args()
    cfg = load_config(a.config)
    run = Path(cfg['run_dir'])
    marker = run / 'evidence/restore-needed'
    if a.restore_only:
        restore(cfg, marker)
        return
    if not a.start_authorized:
        raise SystemExit('Prepared only. Full capture/training requires the explicit --start-authorized flag.')
    lock_path = run / 'evidence/trial.lock'
    lock = os.fdopen(os.open(lock_path, os.O_RDONLY | os.O_CREAT, 0o600), 'r')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    cfg, plan, _ = check(a.config)
    state_path = run / 'evidence/run-state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {'status': 'running', 'epoch': cfg.get('initial_epoch',0), 'next_shard': 0, 'committed': [], 'learning_rate': cfg['learning_rate'], 'history': []}
    if state['status'] == 'completed':
        print('Run is already completed; candidate remains uninstalled.')
        lock.close()
        return
    def interrupted(signum, frame):
        raise KeyboardInterrupt('Training interrupted; restoring target model')
    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(sig, interrupted)
    if cfg.get('production_service'):
        prefix = ['systemctl'] + (['--user'] if cfg.get('production_scope','user') == 'user' else [])
        if subprocess.run(prefix + ['is-active','--quiet',cfg['production_service']]).returncode:
            raise RuntimeError('The configured production service must initially be active')
        assert not any(slot.get('is_processing') for slot in local_api('/slots',base_url=cfg.get('production_url')))
    env = dict(os.environ, **cfg.get('environment', {}))
    env.update(CUDA_VISIBLE_DEVICES=','.join(cfg['gpu_uuids']),CUDA_DEVICE_ORDER='PCI_BUS_ID',
        DFLASH_F16='1',DFLASH_MAX_NEW=str(cfg['max_new_tokens']),
        DFLASH_MAX_PROMPT=str(plan['max_prompt_tokens']))
    if cfg.get('gguf_python_dir'):
        env['PYTHONPATH']=cfg['gguf_python_dir']+os.pathsep+env.get('PYTHONPATH','')
    if cfg.get('production_service'):
        marker.write_text('restore configured inference service on exit\n')
    previous_error = state.get('error')
    recovery_context = None
    recovery_file = run / 'evidence/recovery-review.json'
    if previous_error and recovery_file.exists():
        candidate_context = json.loads(recovery_file.read_text())
        if candidate_context.get('failed_shard') == state.get('shard'):
            recovery_context = candidate_context
    state.update(status='running', error=None, started_unix=state.get('started_unix') or time.time())
    atomic_json(state_path, state)
    try:
        if not state.get('training_finished'):
            strategy = pending_strategy(cfg, run, state)
            if strategy:
                decision = strategy['decision']
                state.setdefault('applied_strategy_requests', []).append(strategy['request_id'])
                state['last_strategy_review'] = strategy
            else:
                phase = 'resume_training' if state['committed'] else 'before_training'
                report_path = run / 'outputs/training.json'
                resume_metrics = json.loads(report_path.read_text()) if report_path.exists() else None
                decision = review(cfg['mentor'], run, f'{phase}-e{state["epoch"]}-s{state["next_shard"]}',
                    {'phase': phase, 'cpu_preflight': 'passed', 'shards_per_epoch': len(plan['shards']),
                     'steps_per_epoch_estimate': plan['steps_per_epoch_estimate'], 'epoch_limit': cfg.get('max_epochs'),
                     'learning_rate': state['learning_rate'], 'validation_monitor_rows': cfg['validation_monitor_rows'],
                     'recent_history': state['history'][-32:],
                     'continuation': cfg.get('continuation'),
                     'seed_checkpoint': cfg.get('seed_step'),
                     'previous_stage_error_present': bool(previous_error),
                     'technical_recovery_report_available': recovery_context is not None,
                     'validation_trend': trend_summary(state['history'], resume_metrics['best_step']) if resume_metrics else None})
            if not apply_decision(state, decision):
                if not (run / 'outputs/training.json').exists():
                    state.update(status='completed', stage='mentor_finished_before_training', mentor_final=decision,
                                 candidate_installed=False, finished_unix=time.time())
                    return
                state.update(training_finished=True, training_stop_reason=decision['reason'])
            atomic_json(state_path, state)
        service_control('stop', cfg['production_service'], cfg.get('production_scope','user'))
        for committed_key in state['committed']:
            old_features = run / 'shards' / committed_key / 'features'
            if old_features.exists():
                shutil.rmtree(old_features)
        monitor = run / 'features/monitor'
        capture(cfg, run / 'data/monitor.jsonl', monitor, env, run / 'evidence/capture-monitor.log')
        training_finished = state.get('training_finished', False)
        metrics = json.loads((run / 'outputs/training.json').read_text()) if (run / 'outputs/training.json').exists() else None
        while not training_finished:
            epoch = state['epoch']
            # Each epoch visits the whole train corpus, in an epoch-specific shard order.
            templates = sorted(plan['shards'], key=lambda row: (row.get('priority',1), hashlib.sha256(
                (str(epoch) + row['key']).encode()).hexdigest()))
            for index in range(state['next_shard'], len(templates)):
                template = templates[index]
                shard = {**template, 'key': f'e{epoch:04d}-' + template['key'].split('-')[-1]}
                work = run / 'shards' / shard['key']
                for sub in ('data', 'features', 'evidence'):
                    (work / sub).mkdir(parents=True, exist_ok=True)
                state.update(stage='capture', shard=shard['key'])
                atomic_json(state_path, state)
                shard_started = time.time()
                rows = capture(cfg, shard['path'], work / 'features', env, work / 'evidence/capture.log')
                capture_seconds = time.time() - shard_started
                val = list(read_rows(run / 'data/monitor.jsonl'))
                write_rows(work / 'data/all.jsonl', rows + val)
                for row in val:
                    for suffix in ('.json', '.f16'):
                        dest = work / 'features' / (row['id'] + suffix)
                        if not dest.exists():
                            dest.symlink_to(monitor / dest.name)
                state['stage'] = 'training'
                atomic_json(state_path, state)
                cmd = [cfg['python'], str(Path(__file__).with_name('train_shard.py')),
                    '--run-dir', str(work), '--state-dir', str(run / 'outputs'), '--plan-hash', digest(run / 'plan.json'),
                    '--stage-key', shard['key'], '--target-model', cfg['target_model'],
                    '--initial-draft', cfg['initial_draft'],
                    '--steps', str(cfg.get('seed_step',0)+(epoch+1-cfg.get('initial_epoch',0))*plan['steps_per_epoch_estimate']),
                    '--batch-size', str(cfg['batch_size']), '--context', str(cfg['training_context']),
                    '--learning-rate', str(state['learning_rate']), '--pool-field', 'sampling_group',
                    '--min-train', '1', '--min-eval', '16', '--min-total', '17',
                    '--output-name', cfg['output_name'],
                    '--model-name', cfg['model_name']]
                if cfg.get('seed_checkpoint'):
                    cmd += ['--seed-checkpoint',cfg['seed_checkpoint'],'--seed-plan-hash',cfg['seed_plan_sha256'],
                            '--seed-step',str(cfg['seed_step'])]
                run_stage(cmd, dict(env, CUDA_VISIBLE_DEVICES=cfg['gpu_uuids'][0]),
                          work / 'evidence/training.log', cfg)
                metrics = json.loads((run / 'outputs/training.json').read_text())
                assert metrics['stage_key'] == shard['key']
                metric_guard(metrics['baseline'], metrics['last'])
                atomic_json(work / 'evidence/committed-training.json', metrics)
                point = {'shard': shard['key'], 'epoch': epoch + 1, 'steps': metrics['steps'],
                         'learning_rate': metrics['learning_rate'], 'metrics': metrics['last'],
                         'capture_seconds':capture_seconds,'shard_seconds_before_mentor':time.time()-shard_started,
                         'amp_overflow_retries': metrics.get('amp_overflow_retries', 0)}
                # Commit coverage only after the mentor decision, so a failed API call can be retried.
                decision = review(cfg['mentor'], run, shard['key'], {'phase': 'after_shard',
                    'continuation':cfg.get('continuation'),
                    'shard': shard['key'], 'epoch': epoch + 1, 'completed_shards_in_epoch': index + 1,
                    'shards_per_epoch': len(templates), 'epoch_limit': cfg.get('max_epochs'),
                    'steps': metrics['steps'], 'learning_rate': metrics['learning_rate'],
                    'proposed_next_learning_rate': state['learning_rate'],
                    'baseline': metrics['baseline'], 'current': metrics['last'], 'best': metrics['best'],
                    'recent_history': state['history'][-32:],
                    'validation_trend': trend_summary(state['history'] + [point], metrics['best_step']),
                    'train_examples': metrics['train_examples'],
                    'short_generation_exclusions': len(rows) - metrics['train_examples'],
                    'gpu_peak_gib': metrics['peak_memory_gib'],
                    'amp_overflow_retries': metrics.get('amp_overflow_retries', 0),
                    'amp_final_scale': metrics.get('amp_final_scale'), 'errors': []})
                keep_training = apply_decision(state, decision)
                state['history'].append(point)
                state['committed'].append(shard['key'])
                state.update(next_shard=index + 1, stage='checkpoint_committed', last_mentor_decision=decision)
                state['updated_unix']=time.time()
                atomic_json(state_path, state)
                shutil.rmtree(work / 'features')
                if not keep_training:
                    state['training_stop_reason'] = decision['reason']
                    training_finished = True
                    state['training_finished'] = True
                    atomic_json(state_path, state)
                    break
            if not training_finished and cfg.get('max_epochs') is not None and epoch+1-cfg.get('initial_epoch',0)>=cfg['max_epochs']:
                training_finished=True
                state.update(training_finished=True,training_stop_reason='Configured max_epochs reached')
                atomic_json(state_path,state)
            if not training_finished:
                decision = review(cfg['mentor'], run, f'epoch-{epoch:04d}-complete', {'phase': 'after_epoch',
                    'continuation':cfg.get('continuation'),
                    'epochs_completed': epoch + 1, 'epoch_limit': cfg.get('max_epochs'),
                    'steps_per_epoch_estimate': plan['steps_per_epoch_estimate'],
                    'learning_rate': state['learning_rate'], 'baseline': metrics['baseline'],
                    'current': metrics['last'], 'best': metrics['best'], 'recent_history': state['history'][-32:],
                    'validation_trend': trend_summary(state['history'], metrics['best_step']),
                    'instruction': 'Decide whether another full pass over the configured train split is useful, within configured limits.'})
                if not apply_decision(state, decision):
                    state['training_stop_reason'] = decision['reason']
                    training_finished = True
                    state['training_finished'] = True
                else:
                    state.update(epoch=epoch + 1, next_shard=0, stage='next_epoch')
                state['last_mentor_decision'] = decision
                atomic_json(state_path, state)
        state['stage'] = 'benchmark'
        atomic_json(state_path, state)
        run_stage([cfg['python'], str(Path(__file__).with_name('benchmark.py')), '--config', str(a.config)],
                  env, run / 'evidence/benchmark.log', cfg)
        summary = json.loads((run / 'outputs/benchmark-summary.json').read_text())
        decision = review(cfg['mentor'], run, 'final-benchmark', {'phase': 'final_benchmark', **summary})
        state.update(status='completed', stage='candidate_ready_for_review', mentor_final=decision,
                     candidate_installed=False, finished_unix=time.time())
    except BaseException as error:
        state.update(status='stopped', error=str(error), candidate_installed=False)
        raise
    finally:
        try:
            restore(cfg, marker)
            state['production_restored'] = True
            state['production_config_unchanged'] = not cfg.get('production_config') or digest(cfg['production_config']) == plan['production_config_sha256']
            if not state['production_config_unchanged']:
                state.update(status='stopped', error='Production config changed during training')
        finally:
            try:
                atomic_json(state_path, state)
            finally:
                lock.close()


if __name__ == '__main__':
    main()
