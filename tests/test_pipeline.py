import io
import json
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.commands.benchmark import variant_command
from scripts.coach_core.files import atomic_json
from scripts.coach_core.metrics import metric_guard
from scripts.coach_core.planning import shard_rows, validate_splits
from scripts.mentoring.review import apply_decision, review, validate_decision


class PlanningTests(unittest.TestCase):
    def row(self, ident, split="train", family=None):
        return {
            "id": ident,
            "split": split,
            "source_group": family or ident,
            "project": "p",
            "source_path": (family or ident) + ".txt",
            "prompt_tokens": 100,
        }

    def test_reject_duplicate_ids(self):
        with self.assertRaises(ValueError):
            validate_splits([self.row("a"), self.row("a")])

    def test_reject_family_across_splits(self):
        with self.assertRaises(ValueError):
            validate_splits([self.row("a", family="x"), self.row("b", "test", "x")])

    def test_reject_source_across_splits(self):
        rows = [self.row("a"), self.row("b", "validation")]
        rows[1]["source_path"] = rows[0]["source_path"]
        with self.assertRaises(ValueError):
            validate_splits(rows)

    def test_valid_splits(self):
        self.assertEqual(validate_splits([self.row("a"), self.row("b", "test")]), {"train": 1, "test": 1})

    def test_shards_respect_memory_bound(self):
        rows = [self.row(str(i)) for i in range(9)]
        per_row = 421 * 32768
        shards = list(shard_rows(rows, 99, per_row * 2, 320))
        self.assertEqual([len(x) for x in shards], [2, 2, 2, 2, 1])
        self.assertEqual([x["id"] for s in shards for x in s], [r["id"] for r in rows])

    def test_shards_respect_count_bound(self):
        self.assertEqual([len(x) for x in shard_rows([self.row(str(i)) for i in range(7)], 3, 10**9, 320)], [3, 3, 1])

    def test_oversized_row_is_rejected(self):
        with self.assertRaises(ValueError):
            list(shard_rows([self.row("a")], 256, 1, 320))

    def test_atomic_file_replaces_previous_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "state.json"
            atomic_json(p, {"next_shard": 1})
            atomic_json(p, {"next_shard": 2})
            self.assertEqual(json.loads(p.read_text()), {"next_shard": 2})
            self.assertEqual(p.stat().st_mode & 0o777, 0o600)
            self.assertFalse(p.with_suffix(".json.tmp").exists())


class MetricTests(unittest.TestCase):
    baseline = {"loss": 4.0, "prefix_tokens": 2.0}

    def test_improvement(self):
        metric_guard(self.baseline, {"loss": 3.9, "prefix_tokens": 2.1})

    def test_loss_regression_is_left_to_mentor(self):
        metric_guard(self.baseline, {"loss": 40.0, "prefix_tokens": 2.1})

    def test_prefix_regression_is_left_to_mentor(self):
        metric_guard(self.baseline, {"loss": 3.9, "prefix_tokens": 0.0})

    def test_nan(self):
        with self.assertRaises(ValueError):
            metric_guard(self.baseline, {"loss": float("nan"), "prefix_tokens": 2.0})


class MentorTests(unittest.TestCase):
    def test_valid_continue_and_stop(self):
        for action in ("continue", "stop"):
            self.assertEqual(
                validate_decision({"action": action, "reason": "metrics", "learning_rate": None}),
                {"action": action, "reason": "metrics", "learning_rate": None},
            )

    def test_command_injection_cannot_become_action(self):
        with self.assertRaises(ValueError):
            validate_decision({"action": "execute", "reason": "run command", "learning_rate": None})

    def test_extra_fields_rejected(self):
        with self.assertRaises(ValueError):
            validate_decision({"action": "continue", "reason": "ok", "learning_rate": None, "command": "anything"})

    def test_empty_reason_rejected(self):
        with self.assertRaises(ValueError):
            validate_decision({"action": "stop", "reason": "", "learning_rate": None})

    def response(self):
        return io.BytesIO(
            json.dumps(
                {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "content": json.dumps({"action": "continue", "reason": "ok", "learning_rate": 1e-6})
                            },
                        }
                    ],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 20},
                    "model": "deepseek-flash",
                }
            ).encode()
        )

    def cfg(self):
        return {
            "model": "deepseek-flash",
            "endpoint": "https://api.deepseek.com/v1/chat/completions",
            "reasoning_effort": "high",
            "timeout_seconds": 600,
        }

    def test_large_recorded_usage_does_not_limit_requests(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            atomic_json(
                run / "evidence/mentor-usage.json",
                {"requests": 1000000, "prompt_tokens": 10**12, "completion_tokens": 10**12},
            )
            with (
                patch("scripts.mentoring.review.read_key", return_value="hidden"),
                patch("scripts.mentoring.client.urllib.request.urlopen", return_value=self.response()) as network,
            ):
                decision = review(self.cfg(), run, "chunk", {"phase": "after_shard"})
                self.assertEqual(decision["learning_rate"], 1e-6)
                body = json.loads(network.call_args.args[0].data)
                self.assertNotIn("max_tokens", body)
                self.assertEqual(body["thinking"]["type"], "enabled")
                self.assertEqual(body["reasoning_effort"], "high")

    def test_transient_retry_continues_past_previous_three_attempt_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            errors = [urllib.error.HTTPError("https://api.deepseek.com", 429, "rate", None, None) for _ in range(4)]
            with (
                patch("scripts.mentoring.review.read_key", return_value="hidden"),
                patch("scripts.mentoring.client.time.sleep"),
                patch("scripts.mentoring.client.urllib.request.urlopen", side_effect=errors + [self.response()]) as network,
            ):
                self.assertEqual(review(self.cfg(), Path(tmp), "chunk", {})["action"], "continue")
                self.assertEqual(network.call_count, 5)

    def test_mentor_can_change_learning_rate(self):
        state = {"learning_rate": 5e-6}
        self.assertTrue(apply_decision(state, {"action": "continue", "reason": "adapt", "learning_rate": 1e-7}))
        self.assertEqual(state["learning_rate"], 1e-7)

    def test_no_artificial_learning_rate_range(self):
        for rate in (1e-12, 0.1):
            validate_decision({"action": "continue", "reason": "mentor chooses", "learning_rate": rate})

    def test_invalid_learning_rate_rejected(self):
        for rate in (0, -1, float("nan"), float("inf"), True, "1e-6"):
            with self.assertRaises(ValueError):
                validate_decision({"action": "continue", "reason": "bad", "learning_rate": rate})

    def test_cached_decision_does_not_repeat_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            atomic_json(
                run / "evidence/mentor/chunk.json",
                {"decision": {"action": "continue", "reason": "ok", "learning_rate": None}, "evidence": {}},
            )
            with patch("scripts.mentoring.client.urllib.request.urlopen") as network:
                self.assertEqual(review({}, run, "chunk", {})["action"], "continue")
                network.assert_not_called()


class BenchmarkTests(unittest.TestCase):
    argv = [
        "runtime",
        "--port",
        "1234",
        "--spec-type",
        "draft-dflash",
        "--spec-draft-model",
        "parent.gguf",
        "--spec-draft-device",
        "CUDA0",
        "--spec-draft-n-max",
        "3",
        "--slots",
    ]

    def test_control_has_no_draft_parameters(self):
        command = variant_command(self.argv, None)
        self.assertFalse(any(x.startswith("--spec-draft-") for x in command))
        self.assertEqual(command[command.index("--spec-type") + 1], "none")
        self.assertEqual(command[command.index("--port") + 1], "1238")
        self.assertIn("--slots", command)

    def test_candidate_uses_its_own_file(self):
        command = variant_command(self.argv, "candidate.gguf")
        self.assertEqual(command[command.index("--spec-draft-model") + 1], "candidate.gguf")
        self.assertIn("parent.gguf", self.argv)


if __name__ == "__main__":
    unittest.main()
