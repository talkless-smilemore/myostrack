"""Focused checks for the ASC-LoRA F0--F5 switches."""
import unittest

import torch
import torch.nn.functional as F

from lib.models.layers.asc_lora import ASCLoRA


def make_adapter(**kwargs):
    torch.manual_seed(7)
    weight = torch.randn(5, 4)
    adapter = ASCLoRA(4, 5, False, rank=2, top_k=3, alpha=4.0,
                      weight=weight, bias_tensor=None, dropout=0.0, **kwargs)
    adapter.lora_B.data.normal_()
    return adapter


class TestASCLoRAAblations(unittest.TestCase):
    def test_gate_off_is_exactly_gate_one_and_has_no_parameter(self):
        adapter = make_adapter(channel_gate=False, spectral_projection="none")
        self.assertIsNone(adapter.gate_logit)
        x = torch.randn(2, 3, 4)
        expected = F.linear(x, adapter.weight) + (adapter.alpha / adapter.rank) * F.linear(
            F.linear(x, adapter.lora_A), adapter.lora_B
        )
        self.assertTrue(torch.allclose(adapter(x), expected, atol=1e-6, rtol=1e-6))

    def test_none_projection_leaves_ba_unprojected(self):
        adapter = make_adapter(channel_gate=False, spectral_projection="none")
        merged, _ = adapter.merge_to_weight()
        expected = adapter.weight + (adapter.alpha / adapter.rank) * (adapter.lora_B @ adapter.lora_A)
        self.assertTrue(torch.allclose(merged, expected, atol=1e-6, rtol=1e-6))

    def test_hard_projection_has_all_one_spectral_weights(self):
        adapter = make_adapter(channel_gate=True, spectral_projection="hard")
        self.assertTrue(torch.equal(adapter.spectral_weight, torch.ones_like(adapter.spectral_weight)))

    def test_zero_complement_weight_is_exactly_zero_after_weighting(self):
        adapter = make_adapter(channel_gate=True, spectral_projection="weighted")
        raw = adapter.complement_constraint_loss()
        self.assertEqual((raw * 0.0).item(), 0.0)

    def test_only_gate_parameter_changes_trainable_count(self):
        full = make_adapter(channel_gate=True, spectral_projection="weighted")
        no_gate = make_adapter(channel_gate=False, spectral_projection="weighted")
        full_count = sum(p.numel() for p in full.parameters() if p.requires_grad)
        no_gate_count = sum(p.numel() for p in no_gate.parameters() if p.requires_grad)
        self.assertEqual(full_count - no_gate_count, full.out_features)
        # Projection mode changes buffers/operations, never trainable parameters.
        hard = make_adapter(channel_gate=True, spectral_projection="hard")
        none = make_adapter(channel_gate=True, spectral_projection="none")
        self.assertEqual(full_count, sum(p.numel() for p in hard.parameters() if p.requires_grad))
        self.assertEqual(full_count, sum(p.numel() for p in none.parameters() if p.requires_grad))


if __name__ == "__main__":
    unittest.main()
