"""Small filesystem and planning helpers; no GPU or network side effects on import."""
import hashlib
import json
import math
import os
from pathlib import Path
import shutil


def load_config(path):
    """Resolve filesystem settings relative to the configuration file."""
    path = Path(path).resolve()
    cfg = json.loads(path.read_text())
    root = Path(__file__).resolve().parents[1]
    defaults = {'base_dir': str(root), 'python': 'python3', 'gpu_uuids': ['0'],
        'production_service': None, 'production_config': None, 'production_launcher': None,
        'production_scope': 'user', 'production_url': 'http://127.0.0.1:1234',
        'benchmark_url': 'http://127.0.0.1:1238', 'initial_epoch': 0,
        'max_shard_rows': 256, 'shard_feature_budget_gib': 8,
        'validation_monitor_rows': 256, 'benchmark_test_rows': 128,
        'max_new_tokens': 320, 'capture_context': 14336, 'training_context': 384,
        'batch_size': 4, 'learning_rate': 2.5e-6,
        'output_name': 'draft-candidate-F16.gguf', 'model_name': 'DFlash-Coach-candidate'}
    cfg = {**defaults, **cfg}
    for name in ('base_dir','run_dir','dataset','target_model','initial_draft','capture_binary',
                 'tokenize_binary','server_binary','production_config','production_launcher','seed_checkpoint','gguf_python_dir'):
        if cfg.get(name):
            value = Path(os.path.expandvars(os.path.expanduser(cfg[name])))
            cfg[name] = str((path.parent / value).resolve() if not value.is_absolute() else value.resolve())
    executable = cfg['python']
    if '/' in executable:
        value = Path(executable).expanduser()
        cfg['python'] = str((path.parent/value).resolve() if not value.is_absolute() else value.resolve())
    else:
        cfg['python'] = shutil.which(executable) or executable
    mentor = dict(cfg.get('mentor') or {})
    if mentor.get('key_file'):
        value = Path(mentor['key_file']).expanduser()
        mentor['key_file'] = str((path.parent/value).resolve() if not value.is_absolute() else value.resolve())
    cfg['mentor'] = mentor
    if not Path(cfg['output_name']).name == cfg['output_name']:
        raise ValueError('output_name must be a filename')
    return cfg


def command_argv(cfg):
    """Expand a configured argument list without a shell."""
    values = {k: str(v) for k,v in cfg.items() if isinstance(v,(str,int,float))}
    return [str(value).format_map(values) for value in cfg['server_argv']]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.chmod(0o600)
    if os.geteuid() == 0:
        owner = path.parent.stat()
        os.chown(temp, owner.st_uid, owner.st_gid)
    temp.replace(path)


def read_rows(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    path.chmod(0o600)
    if os.geteuid() == 0:
        owner = path.parent.stat()
        os.chown(path, owner.st_uid, owner.st_gid)


def shard_rows(rows, max_rows, budget_bytes, max_new):
    shard, used = [], 0
    for row in rows:
        size = (row['prompt_tokens'] + max_new + 1) * 8 * 2048 * 2
        if size > budget_bytes:
            raise ValueError('Single example exceeds shard feature budget')
        if shard and (len(shard) >= max_rows or used + size > budget_bytes):
            yield shard
            shard, used = [], 0
        shard.append(row)
        used += size
    if shard:
        yield shard


def validate_splits(tasks):
    ids, families, sources = set(), {}, {}
    counts = {}
    for row in tasks:
        if row['id'] in ids:
            raise ValueError('Duplicate task id')
        ids.add(row['id'])
        split = row['split']
        if split not in ('train', 'validation', 'test'):
            raise ValueError('Unexpected split')
        counts[split] = counts.get(split, 0) + 1
        for table, key in ((families, row['source_group']),
                           (sources, (row['project'], row['source_path']))):
            previous = table.setdefault(key, split)
            if previous != split:
                raise ValueError('Related source occurs across dataset splits')
    return counts


def metric_guard(baseline, current):
    # Quality judgments belong to the mentor; local code rejects invalid numerics only.
    if any(not math.isfinite(float(x)) for x in current.values()):
        raise ValueError('Non-finite validation metric')
