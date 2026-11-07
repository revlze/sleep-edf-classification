"""Complete released SleepFM stager, BAS-only montage, masked missing modalities."""

from pathlib import Path

import numpy as np
import torch

from src.models.pretrained_assets import resolve_asset, sleepfm_parameters
from src.pretrained_signals import aggregate_patches


class SleepFMAdapter:
    def __init__(self, cache: Path, inputs: Path | None, device: str):
        from braindecode.models import SleepFMStager
        from safetensors.torch import load_file

        self.device = torch.device(device)
        self.assets = ["sleepfm-config.json", "sleepfm-stager.safetensors"]
        config_path, checkpoint_path = [
            resolve_asset(name, cache, inputs) for name in self.assets
        ]
        self.model = SleepFMStager(**sleepfm_parameters(config_path))
        # Complete 180-tensor revision: no missing-key fallback / random head.
        state = load_file(str(checkpoint_path))
        if len(state) != 180:
            raise ValueError("Expected complete 180-tensor SleepFMStager checkpoint")
        self.model.load_state_dict(state, strict=True)
        self.model.eval().to(self.device)

    def probabilities(self, signal: np.ndarray) -> np.ndarray:
        n = len(signal) // 3840
        if n * 6 > 8196:
            raise ValueError("SleepFM staging position table exceeded")
        x = torch.from_numpy(np.ascontiguousarray(signal.T)).unsqueeze(0)
        with torch.inference_mode():
            logits = self.model(x.to(self.device))
        if logits.shape != (1, 5, n * 6):
            raise ValueError(f"Unexpected SleepFM output: {logits.shape}")
        return aggregate_patches(logits[0].float().cpu().numpy().T)
