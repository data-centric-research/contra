"""SoTTA: Robust Test-Time Adaptation on Noisy Data Streams (NeurIPS 2023).

Reference: Gong et al., "SoTTA: Robust Test-Time Adaptation on Noisy Data
Streams," in NeurIPS, 2023.

Core idea:
  1. High-confidence Uniform-class Sampling (HUS): maintain a memory bank of
     high-confidence samples with uniform class distribution to reduce the
     effect of noisy/corrupted inputs and class imbalance during adaptation.
  2. Entropy-Sharpness Minimization (ESM): combine entropy minimization with
     Sharpness-Aware Minimization (SAM) to seek flat minima that are more
     robust to distribution shifts in the test stream.
"""

from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F


def configure_model(model):
    """Configure model: train mode, freeze all except BN affine params."""
    model.train()
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.requires_grad_(True)
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return model


def collect_params(model):
    """Collect BN affine parameters and their names."""
    params, names = [], []
    for nm, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            for np_, p in m.named_parameters():
                if np_ in ["weight", "bias"] and p.requires_grad:
                    params.append(p)
                    names.append(f"{nm}.{np_}")
    return params, names


class SoTTA(nn.Module):
    """SoTTA adapts a model via HUS + ESM during test time.

    Every forward pass filters the batch through the HUS memory bank, then
    performs one (or more) ESM update steps on the filtered samples.
    """

    def __init__(self, model, optimizer, steps=1, episodic=False,
                 conf_threshold=0.9, num_classes=10, bank_size=64,
                 rho=0.05):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self.episodic = episodic
        self.conf_threshold = conf_threshold
        self.num_classes = num_classes
        self.bank_size = bank_size
        self.rho = rho

        self.model_state = deepcopy(model.state_dict())
        self.optimizer_state = deepcopy(optimizer.state_dict())

        self.memory_bank = []
        self.memory_labels = []

    def forward(self, x):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs = self.forward_and_adapt(x)
        return outputs

    def reset(self):
        self.model.load_state_dict(self.model_state, strict=True)
        self.optimizer.load_state_dict(self.optimizer_state)
        self.memory_bank = []
        self.memory_labels = []

    # ------------------------------------------------------------------
    # HUS: High-confidence Uniform-class Sampling
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _update_memory(self, x):
        """Add high-confidence samples to memory, keep class-uniform."""
        logits = self.model(x)
        probs = F.softmax(logits, dim=1)
        conf, pseudo = probs.max(dim=1)

        mask = conf >= self.conf_threshold
        if mask.sum() == 0:
            return

        hc_x = x[mask]
        hc_labels = pseudo[mask]

        for i in range(hc_x.size(0)):
            self.memory_bank.append(hc_x[i].detach())
            self.memory_labels.append(hc_labels[i].item())

        if len(self.memory_bank) > self.bank_size:
            self._balance_memory()

    def _balance_memory(self):
        """Keep at most bank_size entries with uniform class distribution."""
        per_class = max(1, self.bank_size // self.num_classes)
        class_buckets = {c: [] for c in range(self.num_classes)}

        for feat, lbl in zip(self.memory_bank, self.memory_labels):
            if len(class_buckets[lbl]) < per_class:
                class_buckets[lbl].append(feat)

        self.memory_bank = []
        self.memory_labels = []
        for c in range(self.num_classes):
            for feat in class_buckets[c]:
                self.memory_bank.append(feat)
                self.memory_labels.append(c)

    def _get_memory_batch(self):
        """Return stacked memory tensor, or None if empty."""
        if len(self.memory_bank) == 0:
            return None
        return torch.stack(self.memory_bank)

    # ------------------------------------------------------------------
    # ESM: Entropy-Sharpness Minimization (SAM-style)
    # ------------------------------------------------------------------
    @torch.enable_grad()
    def _esm_step(self, x):
        """One ESM update: entropy loss with sharpness-aware perturbation."""
        logits = self.model(x)
        loss = softmax_entropy(logits).mean()

        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            self._trainable_params(), max_norm=1e6)
        if grad_norm == 0:
            self.optimizer.zero_grad()
            return loss.item()

        with torch.no_grad():
            saved = []
            for p in self._trainable_params():
                if p.grad is not None:
                    e_w = self.rho * p.grad / (grad_norm + 1e-12)
                    p.add_(e_w)
                    saved.append(e_w)
                else:
                    saved.append(torch.zeros_like(p))

        self.optimizer.zero_grad()
        logits_adv = self.model(x)
        loss_adv = softmax_entropy(logits_adv).mean()
        loss_adv.backward()

        with torch.no_grad():
            for p, e_w in zip(self._trainable_params(), saved):
                p.sub_(e_w)

        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss_adv.item()

    def _trainable_params(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    # ------------------------------------------------------------------
    # Main adaptation loop
    # ------------------------------------------------------------------
    @torch.enable_grad()
    def forward_and_adapt(self, x):
        self._update_memory(x)

        mem_x = self._get_memory_batch()
        if mem_x is not None and mem_x.size(0) >= 2:
            self._esm_step(mem_x)

        with torch.no_grad():
            outputs = self.model(x)
        return outputs


def softmax_entropy(logits):
    """Entropy of softmax distribution from logits."""
    return -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
