import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from common import atomic_json, digest
from strategy import pending_strategy, trend_summary


class TrendTests(unittest.TestCase):
    def history(self):
        return [{'steps': i * 64, 'metrics': {'loss': 3.8 if i <= 10 else 4.0, 'prefix_tokens': 2.5}}
                for i in range(1, 21)]

    def test_rolling_drift_and_best_age_are_exposed(self):
        value = trend_summary(self.history(), 640)
        self.assertAlmostEqual(value['previous_10']['mean']['loss'], 3.8)
        self.assertAlmostEqual(value['latest_10']['mean']['loss'], 4.0)
        self.assertEqual(value['shards_without_new_best'], 10)
        self.assertEqual(value['steps_without_new_best'], 640)
        self.assertFalse(value['independent_test_evaluation_available'])

    def test_new_rate_evidence_is_separated_from_old_plateau(self):
        history = self.history()
        for row in history:row['learning_rate'] = 5e-6
        for row in history[-2:]:row['learning_rate'] = 2.5e-6
        value = trend_summary(history, 640)
        self.assertEqual(value['shards_at_current_learning_rate'], 2)
        self.assertEqual(value['since_latest_learning_rate_change']['count'], 2)
        self.assertEqual(value['current_learning_rate'], 2.5e-6)

    def test_initial_history_is_supported(self):
        self.assertEqual(trend_summary([], 0)['checkpoints'], 0)
        self.assertIsNone(trend_summary(self.history()[:1], 64)['previous_10'])


class StrategyTests(unittest.TestCase):
    def prepare(self, root):
        atomic_json(root / 'plan.json', {'data': 'frozen'})
        atomic_json(root / 'evidence/strategy-control.json', {'review_key': 'review-1', 'plan_sha256': digest(root / 'plan.json')})
        record = {'request_id': 'actual-provider-id', 'model_requested': 'deepseek-flash',
                  'evidence': {'phase': 'strategy_review', 'current_step': 64},
                  'decision': {'action': 'continue', 'reason': 'plateau', 'learning_rate': 2.5e-6}}
        atomic_json(root / 'evidence/mentor/review-1.json', record)
        return {'mentor': {'model': 'deepseek-flash'}}

    def test_approved_provider_decision_is_available_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);cfg = self.prepare(root)
            value = pending_strategy(cfg, root, {})
            self.assertEqual(value['decision']['learning_rate'], 2.5e-6)
            self.assertIsNone(pending_strategy(cfg, root, {'applied_strategy_requests': [value['request_id']]}))

    def test_other_data_plan_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);cfg = self.prepare(root)
            atomic_json(root / 'plan.json', {'data': 'changed'})
            with self.assertRaisesRegex(ValueError, 'different data plan'):
                pending_strategy(cfg, root, {})

    def test_path_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);cfg = self.prepare(root)
            atomic_json(root / 'evidence/strategy-control.json', {'review_key': '../bad', 'plan_sha256': digest(root / 'plan.json')})
            with self.assertRaisesRegex(ValueError, 'review key'):
                pending_strategy(cfg, root, {})

    def test_unrelated_provider_record_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);cfg = self.prepare(root)
            cfg['mentor']['model'] = 'other-model'
            with self.assertRaisesRegex(ValueError, 'provider'):
                pending_strategy(cfg, root, {})
