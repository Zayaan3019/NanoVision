"""
Unit tests for the small ResNet9-style CNN baseline (used only as an honest
comparison point against ViT-from-scratch — see AUDIT.md §5, item 14).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from nanovision.model.cnn import BaselineCNN, CNNConfig


def _tiny_cfg() -> CNNConfig:
    return CNNConfig(widths=(8, 16, 32, 64), n_classes=10)


class TestBaselineCNN:
    def test_output_shape(self):
        model = BaselineCNN(_tiny_cfg())
        x = torch.randn(4, 3, 32, 32)
        out = model(x)
        assert out.shape == (4, 10)

    def test_output_finite(self):
        model = BaselineCNN(_tiny_cfg())
        x = torch.randn(2, 3, 32, 32)
        assert torch.isfinite(model(x)).all()

    def test_eval_mode_deterministic(self):
        model = BaselineCNN(_tiny_cfg()).eval()
        x = torch.randn(3, 3, 32, 32)
        y1, y2 = model(x), model(x)
        assert torch.allclose(y1, y2)

    def test_gradients_flow(self):
        model = BaselineCNN(_tiny_cfg())
        x = torch.randn(2, 3, 32, 32)
        labels = torch.zeros(2, dtype=torch.long)
        loss = nn.CrossEntropyLoss()(model(x), labels)
        loss.backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"

    def test_loss_decreases(self):
        torch.manual_seed(0)
        model = BaselineCNN(_tiny_cfg())
        opt = model.configure_optimizer(lr=0.05, weight_decay=5e-4)
        x = torch.randn(8, 3, 32, 32)
        labels = torch.zeros(8, dtype=torch.long)

        losses = []
        for _ in range(5):
            opt.zero_grad()
            loss = nn.CrossEntropyLoss()(model(x), labels)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        assert losses[-1] < losses[0]

    def test_config_roundtrip(self, tmp_path):
        cfg = _tiny_cfg()
        p = tmp_path / "cnn_cfg.json"
        cfg.save(p)
        loaded = CNNConfig.load(p)
        assert loaded.widths == cfg.widths
        assert loaded.n_classes == cfg.n_classes

    def test_count_params_matches_manual_sum(self):
        model = BaselineCNN(_tiny_cfg())
        assert model.count_params() == sum(p.numel() for p in model.parameters())

    def test_optimizer_two_groups_no_decay_on_bn_and_bias(self):
        model = BaselineCNN(_tiny_cfg())
        opt = model.configure_optimizer(lr=0.1, weight_decay=5e-4)
        assert len(opt.param_groups) == 2
        assert opt.param_groups[0]["weight_decay"] == 5e-4
        assert opt.param_groups[1]["weight_decay"] == 0.0
