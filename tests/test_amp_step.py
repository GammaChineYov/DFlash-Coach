import unittest

try:
    import sys
    from pathlib import Path

    import torch
    from torch import nn

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.training.amp_step import amp_train_step
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Run AMP tests with the training Python environment")
class AmpStepTests(unittest.TestCase):
    def test_overflow_retries_same_batch_and_updates_optimizer_once(self):
        model = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        scaler = torch.amp.GradScaler("cpu", init_scale=65536)
        inputs = torch.full((2, 2), 4.0)
        before = model.weight.detach().clone()
        events = []
        calls = []

        def compute():
            calls.append(inputs)
            with torch.autocast("cpu", dtype=torch.float16):
                loss = model(inputs).float().square().mean()
            return loss, {"value": float(loss.detach())}

        loss, metrics, retries = amp_train_step(
            compute, model.parameters(), optimizer, scaler, lambda count, old, new: events.append((count, old, new))
        )
        self.assertGreater(retries, 0)
        self.assertEqual(len(calls), retries + 1)
        self.assertTrue(all(batch is inputs for batch in calls))
        self.assertTrue(all(new < old for _, old, new in events))
        self.assertEqual(int(optimizer.state[model.weight]["step"]), 1)
        self.assertTrue(torch.isfinite(model.weight).all())
        self.assertFalse(torch.equal(before, model.weight))

    def test_nonfinite_loss_leaves_weights_and_optimizer_unchanged(self):
        model = nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters())
        scaler = torch.amp.GradScaler("cpu")
        before = [p.detach().clone() for p in model.parameters()]
        with self.assertRaisesRegex(RuntimeError, "Non-finite training loss"):
            amp_train_step(lambda: (torch.tensor(float("nan")), {}), model.parameters(), optimizer, scaler)
        self.assertFalse(optimizer.state)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, model.parameters(), strict=False)))

    def test_successful_scale_state_can_resume(self):
        model = nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scaler = torch.amp.GradScaler("cpu", init_scale=8)

        def compute():
            return model(torch.ones(1, 2)).square().mean(), {}

        _, _, retries = amp_train_step(compute, model.parameters(), optimizer, scaler)
        self.assertEqual(retries, 0)
        restored = torch.amp.GradScaler("cpu")
        restored.load_state_dict(scaler.state_dict())
        self.assertEqual(restored.get_scale(), scaler.get_scale())
        amp_train_step(compute, model.parameters(), optimizer, restored)
        self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))
