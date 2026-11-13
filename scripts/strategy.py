"""Summarize validation drift and apply a provider-backed strategy review once."""
import json
from pathlib import Path
import re
import statistics
from common import digest
from mentor import validate_decision

METRICS = ('loss', 'ce', 'selector_ce', 'accuracy', 'prefix_tokens', 'top16_recall')


def trend_summary(history, best_step):
    if not history:
        return {'checkpoints': 0, 'steps_without_new_best': 0, 'shards_without_new_best': 0}
    def window(rows):
        if not rows:
            return None
        return {'count': len(rows), 'first_step': rows[0]['steps'], 'last_step': rows[-1]['steps'],
                'mean': {key: statistics.mean(r['metrics'][key] for r in rows)
                         for key in METRICS if all(key in r['metrics'] for r in rows)}}
    result = {'checkpoints': len(history), 'best_step': best_step,
            'steps_without_new_best': max(0, history[-1]['steps'] - best_step),
            'shards_without_new_best': sum(r['steps'] > best_step for r in history),
            'previous_10': window(history[-20:-10]), 'latest_10': window(history[-10:]),
            'monitor_reused_for_checkpoint_selection': True,
            'independent_test_evaluation_available': False}
    rate = history[-1].get('learning_rate')
    if rate is not None:
        first = len(history) - 1
        while first > 0 and history[first - 1].get('learning_rate') == rate:
            first -= 1
        result.update(current_learning_rate=rate, shards_at_current_learning_rate=len(history)-first,
                      since_latest_learning_rate_change=window(history[first:]))
    return result


def pending_strategy(cfg, run, state):
    run = Path(run)
    pointer = run / 'evidence/strategy-control.json'
    if not pointer.exists():
        return None
    control = json.loads(pointer.read_text())
    key = control['review_key']
    if not isinstance(key, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', key):
        raise ValueError('Invalid strategy review key')
    if control['plan_sha256'] != digest(run / 'plan.json'):
        raise ValueError('Strategy review refers to a different data plan')
    record = json.loads((run / 'evidence/mentor' / (key + '.json')).read_text())
    request_id = record.get('request_id')
    if not request_id or record.get('evidence', {}).get('phase') != 'strategy_review':
        raise ValueError('Strategy review has no provider request or strategy evidence')
    if record.get('model_requested') != cfg['mentor']['model']:
        raise ValueError('Strategy review provider does not match the configured mentor')
    if request_id in state.get('applied_strategy_requests', []):
        return None
    return {'request_id': request_id, 'review_key': key,
            'decision': validate_decision(record['decision']),
            'source_step': record['evidence'].get('current_step')}
