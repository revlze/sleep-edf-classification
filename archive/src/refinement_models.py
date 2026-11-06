"""Small residual gates and temporal heads on fixed pretrained representations."""

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

PRIOR = (0.5, 0.25, 0.25)


def confidence_features(p: np.ndarray) -> np.ndarray:
    """Three experts, fixed W/N1/N2/N3/REM order; no IDs or target labels."""
    if p.ndim != 3 or p.shape[1:] != (3, 5):
        raise ValueError("Expected N x 3 x 5 expert probabilities")
    ordered = np.sort(p, axis=-1)
    entropy = -(p * np.log(np.maximum(p, 1e-7))).sum(-1)
    chunks = [
        p.reshape(len(p), 15),
        ordered[:, :, -1],
        entropy,
        ordered[:, :, -1] - ordered[:, :, -2],
    ]
    for a, b in ((0, 1), (0, 2), (1, 2)):
        chunks.extend(
            [
                np.abs(p[:, a] - p[:, b]).sum(1, keepdims=True),
                (p[:, a].argmax(1) != p[:, b].argmax(1))[:, None],
            ]
        )
    return np.concatenate(chunks, axis=1).astype(np.float32)


def segment_bounds(meta: dict) -> tuple[np.ndarray, np.ndarray]:
    """Half-open bounds for every row, breaking on record changes AND missing epochs."""
    n = len(meta["ids"])
    cuts = (
        np.flatnonzero(
            (meta["record_id"][1:] != meta["record_id"][:-1])
            | (np.diff(meta["epoch_index"]) != 1)
        )
        + 1
    )
    starts, ends = np.r_[0, cuts], np.r_[cuts, n]
    return np.repeat(starts, ends - starts), np.repeat(ends, ends - starts)


class WindowBank:
    """Compact arrays stay on GPU; gather windows there, without Python row loading."""

    def __init__(self, x, experts, meta, device):
        self.x = torch.as_tensor(x, dtype=torch.float32, device=device)
        self.experts = torch.as_tensor(experts, dtype=torch.float32, device=device)
        self.y = torch.as_tensor(
            meta.get("labels", np.full(len(x), -1)), dtype=torch.long, device=device
        )
        starts, ends = segment_bounds(meta)
        self.starts = torch.as_tensor(starts, device=device)
        self.ends = torch.as_tensor(ends, device=device)
        self.device = device

    def batch(self, centers, width):
        centers = torch.as_tensor(centers, dtype=torch.long, device=self.device)
        left = (width - 1) // 2
        begin = torch.maximum(centers - left, self.starts[centers])
        end = torch.minimum(centers + (width - left), self.ends[centers])
        lengths = end - begin
        idx = begin[:, None] + torch.arange(width, device=self.device)
        mask = idx < end[:, None]
        idx = torch.minimum(idx, (end - 1)[:, None])
        x = self.x[idx] * mask[:, :, None]
        p, y = self.experts[idx], self.y[idx]
        return x, p, y, mask, centers - begin, lengths


class TemporalBody(nn.Module):
    def __init__(self, features: int, architecture: str, hidden: int = 64):
        super().__init__()
        self.architecture = architecture
        self.project = nn.Linear(features, hidden)
        if architecture == "bigru":
            self.rnn = nn.GRU(
                hidden, hidden // 2, batch_first=True, bidirectional=True, num_layers=1
            )
        else:
            self.convs = nn.ModuleList(
                [
                    nn.Conv1d(hidden, hidden, 3, padding=d, dilation=d)
                    for d in (1, 2, 4, 8, 16, 32)
                ]
            )
            self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in self.convs])
        self.drop = nn.Dropout(0.15)

    def forward(self, x, mask, lengths):
        h = torch.nn.functional.gelu(self.project(x)) * mask[:, :, None]
        if self.architecture == "bigru":
            # Right padding is packed so the backward GRU never consumes fake epochs.
            packed = pack_padded_sequence(
                h, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            h, _ = self.rnn(packed)
            h, _ = pad_packed_sequence(h, batch_first=True, total_length=x.shape[1])
        else:
            for conv, norm in zip(self.convs, self.norms, strict=True):
                z = conv(h.transpose(1, 2)).transpose(1, 2)
                h = norm(h + self.drop(torch.nn.functional.gelu(z))) * mask[:, :, None]
        return h


class ResidualHead(nn.Module):
    def __init__(self, features: int, config: dict):
        super().__init__()
        self.gate = config["model"]["kind"] == "gate"
        architecture = config["model"]["architecture"]
        hidden = config["model"]["hidden"]
        self.temporal = architecture in ("tcn", "bigru")
        if self.temporal:
            self.body = TemporalBody(features, architecture, hidden)
            last_features = hidden
        elif architecture == "linear":
            self.body = nn.Identity()
            last_features = features
        else:
            self.body = nn.Sequential(
                nn.Linear(features, hidden),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(hidden, hidden),
                nn.GELU(),
            )
            last_features = hidden
        self.head = nn.Linear(last_features, 3 if self.gate else 5)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.register_buffer("prior", torch.tensor(PRIOR))

    def forward(self, x, p, mask, lengths):
        z = self.body(x, mask, lengths) if self.temporal else self.body(x)
        delta = self.head(z).float()
        if self.gate:
            weights = (self.prior.log() + delta).softmax(-1)
            prob = (p * weights[:, :, :, None]).sum(2).clamp_min(1e-7)
            kl = (weights * (weights.clamp_min(1e-7).log() - self.prior.log())).sum(-1)
            return prob.log(), kl, weights
        base = (p * self.prior[None, None, :, None]).sum(2).clamp_min(1e-7)
        return base.log() + delta, delta.square().mean(-1), None
