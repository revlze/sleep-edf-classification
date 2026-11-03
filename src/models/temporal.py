"""Small residual correction on log probabilities; zero-initialized output delta."""

import torch
from torch import nn


class ResidualTemporalCNN(nn.Module):
    radius = 3

    def __init__(self, width: int = 16, dropout: float = 0.1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(5, width, 3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(width, width, 3, padding=2, dilation=2),
            nn.GELU(),
        )
        self.delta = nn.Conv1d(width, 5, 1)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward(self, base_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        delta = self.delta(self.features(base_logits))
        return base_logits + delta, delta


def configure_finetuning(model: nn.Module, strategy: str) -> list[dict]:
    if strategy not in ("classifier", "partial", "full"):
        raise ValueError(strategy)
    for parameter in model.parameters():
        parameter.requires_grad_(strategy == "full")
    if strategy != "full":
        for parameter in model.segment_classifier.parameters():
            parameter.requires_grad_(True)
    if strategy == "partial":
        for block in list(model.decoders)[-2:]:
            for parameter in block.parameters():
                parameter.requires_grad_(True)
    return [
        {"name": name, "shape": list(p.shape), "parameters": p.numel()}
        for name, p in model.named_parameters()
        if p.requires_grad
    ]


def training_mode(model: nn.Module, finetune: bool) -> None:
    model.train()
    if finetune:
        # Small windows/batches: freeze running moments even in full FT.
        # Affine BN weights still train when their module is unfrozen.
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm) or not any(
                p.requires_grad for p in module.parameters()
            ):
                module.eval()
