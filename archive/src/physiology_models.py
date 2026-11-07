"""Small downstream experiments; CalibSleep-inspired, not author checkpoints."""

import torch
from torch import nn
from torch.nn import functional as F

CALIB_REVISION = "a6fb8f26725010c6434dfb3b35540f5591f63921"


def supervised_contrastive(z, y, temperature=0.1, hard=False, ids=None):
    """SupCon, excluding self/duplicate epoch pairs; bounded hard-negative weights."""
    z = F.normalize(z.float(), dim=-1)
    scores = z @ z.T / temperature
    valid = ~torch.eye(len(y), dtype=torch.bool, device=y.device)
    if ids is not None:
        valid &= ids[:, None] != ids[None, :]
    positive = (y[:, None] == y[None, :]) & valid
    negative = (y[:, None] != y[None, :]) & valid
    usable = positive.any(1) & negative.any(1)
    if not usable.any():
        raise ValueError(
            "Contrastive minibatch has no distinct positive/negative pairs"
        )
    weights = torch.ones_like(scores)
    if hard:
        a, b = y[:, None], y[None, :]
        pairs = (
            ((a == 1) & ((b == 0) | (b == 2) | (b == 4)))
            | ((b == 1) & ((a == 0) | (a == 2) | (a == 4)))
            | (((a == 2) & (b == 3)) | ((a == 3) & (b == 2)))
        )
        # Detached similarity, capped at 2x: no unbounded focus on noisy labels.
        weights += pairs * torch.sigmoid((z.detach() @ z.detach().T) / 0.2)
    denominator = torch.logsumexp(
        (scores + weights.log()).masked_fill(~valid, -torch.inf), dim=1
    )
    logp = scores - denominator[:, None]
    per_anchor = -(logp.masked_fill(~positive, 0).sum(1) / positive.sum(1).clamp_min(1))
    return per_anchor[usable].mean()


class EmbeddingClassifier(nn.Module):
    def __init__(self, features: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(features, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.GELU(),
        )
        self.classifier = nn.Linear(64, 5)
        self.projection = nn.Sequential(nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 32))

    def forward(self, x):
        h = self.body(x)
        return self.classifier(h), self.projection(h)


class ResidualCalibration(nn.Module):
    def __init__(self, features: int):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(features, 64), nn.GELU(), nn.Dropout(0.2))
        self.out = nn.Linear(64, 5)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, base):
        delta = 0.5 * self.out(self.body(x)).tanh()
        return base.clamp_min(1e-7).log() + delta, delta


class DualStream(nn.Module):
    """Three modality-specific encoders; GPU STFT and bidirectional calibration.

    Adaptation of CalibSleep CMC, Eq4–9. Strided channel-specific CNNs and small
    spectral convs replace author CNN/BiGRU/TimesNet; no reproduction claim.
    """

    def __init__(self, calibrated: bool):
        super().__init__()
        self.calibrated = calibrated
        self.time = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(1, 16, 25, stride=5, padding=12),
                    nn.GELU(),
                    nn.Conv1d(16, 32, 9, stride=5, padding=4),
                    nn.GELU(),
                    nn.AdaptiveAvgPool1d(24),
                )
                for _ in range(3)
            ]
        )
        self.spectral = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(65, 32, 3, padding=1),
                    nn.GELU(),
                    nn.Conv1d(32, 32, 5, padding=2),
                    nn.GELU(),
                    nn.AdaptiveAvgPool1d(24),
                )
                for _ in range(3)
            ]
        )
        self.project_time, self.project_freq = nn.Linear(96, 64), nn.Linear(96, 64)
        self.norm_time, self.norm_freq = nn.LayerNorm(64), nn.LayerNorm(64)
        if calibrated:
            self.tf = nn.MultiheadAttention(64, 4, batch_first=True, dropout=0.1)
            self.ft = nn.MultiheadAttention(64, 4, batch_first=True, dropout=0.1)
            self.gate = nn.Linear(128, 64)
        self.classifier = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 5))
        self.register_buffer("stft_window", torch.hann_window(128))

    def forward(self, x):
        # Input raw uV / 50, 50Hz; no CPU spectrogram in the training hot path.
        b = len(x)
        with torch.autocast(device_type=x.device.type, enabled=False):
            stft = torch.stft(
                x.float().reshape(-1, 1500),
                n_fft=128,
                hop_length=50,
                window=self.stft_window,
                return_complex=True,
            )
            spec = stft.abs().square().log1p().reshape(b, 3, 65, -1)
        t = torch.cat(
            [m(x[:, c : c + 1]) for c, m in enumerate(self.time)], 1
        ).transpose(1, 2)
        f = torch.cat(
            [m(spec[:, c]) for c, m in enumerate(self.spectral)], 1
        ).transpose(1, 2)
        t, f = (
            self.norm_time(self.project_time(t)),
            self.norm_freq(self.project_freq(f)),
        )
        alignment = x.new_zeros(())
        if self.calibrated:
            at, _ = self.tf(t, f, f, need_weights=False)
            af, _ = self.ft(f, t, t, need_weights=False)
            t, f = t + at, f + af
            a = self.gate(torch.cat([t.mean(1), f.mean(1)], -1)).sigmoid()
            h = a * t.mean(1) + (1 - a) * f.mean(1)
            alignment = (
                (
                    F.normalize(t.mean(1).float(), dim=-1)
                    - F.normalize(f.mean(1).float(), dim=-1)
                )
                .square()
                .mean()
            )
            combined = torch.cat([h, (t - f).mean(1)], -1)
        else:
            combined = torch.cat([t.mean(1), f.mean(1)], -1)
        return self.classifier(combined), alignment


def transition_penalty(probabilities, labels, valid_pairs, permitted):
    """Soft expected cost, only real adjacent epochs; training-fold prior only."""
    if not valid_pairs.any():
        return probabilities.sum() * 0
    pair = probabilities[:, :-1, :, None] * probabilities[:, 1:, None, :]
    loss = (pair * (1 - permitted)).sum((-1, -2))
    # Downweight actual transitions instead of erasing short annotated N1 runs.
    weight = torch.where(labels[:, :-1] == labels[:, 1:], 1.0, 0.25)
    return (loss * weight)[valid_pairs].mean()
