"""Original AnySleep / U-Sleep standalone implementations, strict loading."""

from pathlib import Path

import numpy as np
import torch
from scipy.special import softmax

from src.models.pretrained_assets import official_module, resolve_asset
from src.pretrained_signals import pair_vote


class AnySleepAdapter:
    right_boundary_alignment = 16

    def __init__(
        self,
        family: str,
        checkpoint: str,
        cache: Path,
        inputs: Path | None,
        device: str,
    ):
        self.family = family
        self.device = torch.device(device)
        source = f"{family}_no_hydra.py"
        source_path = resolve_asset(source, cache, inputs)
        checkpoint_path = resolve_asset(checkpoint, cache, inputs)
        module = official_module(source_path)
        cls = module.AnySleep if family == "anysleep" else module.USleep
        # Author constructors use weights_only=True and strict load_state_dict.
        self.model = cls(path=str(checkpoint_path)).eval().to(self.device)
        self.assets = [source, checkpoint]

    def probabilities(self, signal: np.ndarray) -> np.ndarray:
        n = len(signal) // 3840
        pairs = [(0, 1, 2)] if self.family == "anysleep" else [(0, 2), (1, 2)]
        predictions = []
        with torch.inference_mode():
            for channels in pairs:
                x = torch.from_numpy(np.ascontiguousarray(signal[:, channels]))
                logits = self.model(x.unsqueeze(0).to(self.device))
                if logits.shape != (1, n, 5):
                    raise ValueError(f"Unexpected {self.family} output: {logits.shape}")
                predictions.append(softmax(logits[0].float().cpu().numpy(), axis=-1))
        return (
            predictions[0]
            if len(predictions) == 1
            else pair_vote(np.stack(predictions))
        )


class AnySleepEnsemble:
    right_boundary_alignment = 16

    def __init__(
        self, checkpoints: list[str], cache: Path, inputs: Path | None, device: str
    ):
        self.members = [
            AnySleepAdapter("anysleep", c, cache, inputs, device) for c in checkpoints
        ]
        self.assets = list(
            dict.fromkeys(a for model in self.members for a in model.assets)
        )

    def probabilities(self, signal: np.ndarray) -> np.ndarray:
        return np.mean(
            [member.probabilities(signal) for member in self.members], axis=0
        )
