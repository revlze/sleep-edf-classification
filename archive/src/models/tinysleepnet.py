"""PhysioEx released one-channel TinySleepNet, with original class definitions."""

import json
from pathlib import Path

import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt
from scipy.special import softmax
from torch import nn

from src import series_assets  # noqa: F401
from src.models.anysleep_variants import definitions
from src.models.pretrained_assets import resolve_asset


def tiny_preprocess(x: np.ndarray, channel: int) -> np.ndarray:
    if channel not in (0, 1) or x.ndim != 3 or x.shape[1:] != (3000, 3):
        raise ValueError("TinySleepNet requires one EEG channel")
    # PhysioEx EDF reader returns µV. raw preset = BP .3–40 Hz, order 5, then
    # resample to 100 Hz (already 100 here); no z-score/RobustScaler/clipping.
    sos = butter(5, [0.3 / 50, 40 / 50], btype="band", output="sos")
    return (
        sosfiltfilt(sos, x[:, :, channel].reshape(-1).astype(np.float64))
        .astype(np.float32)
        .reshape(-1, 3000)
    )


class TinySleepNetAdapter:
    def __init__(self, cache: Path, inputs: Path | None, device: str):
        self.assets = [
            "tinysleepnet-source.py",
            "tinysleepnet-config.json",
            "tinysleepnet.pt",
        ]
        paths = [resolve_asset(name, cache, inputs) for name in self.assets]
        config = json.loads(paths[1].read_text())
        required = {"in_chan": 1, "sf": 100, "n_classes": 5}
        if (
            any(config["model_kwargs"].get(k) != v for k, v in required.items())
            or config["training"]["sequence_length"] != 20
            or config["training"]["pipeline_preset"] != "raw"
        ):
            raise ValueError("TinySleepNet checkpoint contract differs")
        namespace = {"torch": torch, "nn": nn}
        definitions(
            paths[0], namespace, {"FeatureExtractor", "Classifier", "TinySleepNet"}
        )
        self.model = namespace["TinySleepNet"](**config["model_kwargs"])
        state = torch.load(paths[2], map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.device = torch.device(device)
        self.model.eval().to(self.device)

    def probabilities(self, epochs: np.ndarray, batch_size: int = 16) -> np.ndarray:
        """20-epoch rolling windows; score each epoch at its rightmost available position.

        Native LSTM is unidirectional and resets on every forward. First 19 epochs
        are taken from the first window, later epochs from windows ending there.
        Short continuous segments run at their actual length, no invented signals.
        """
        n = len(epochs)
        if epochs.shape != (n, 3000) or n == 0:
            raise ValueError("Invalid EEG epoch shape")
        out = np.empty((n, 5), dtype=np.float32)
        width = min(20, n)
        with torch.inference_mode():
            features = []
            for left in range(0, n, 64):
                batch = torch.from_numpy(epochs[left : left + 64].copy())[:, None].to(
                    self.device
                )
                features.append(self.model.feature_extractor(batch).float().cpu())
            features = torch.cat(features)
            logits = self.model.clf(features[:width].unsqueeze(0).to(self.device))
            if logits.shape != (1, width, 5):
                raise ValueError("TinySleepNet output/class axis mismatch")
            out[:width] = softmax(logits[0].float().cpu().numpy(), axis=-1)
            for left in range(20, n, batch_size):
                ends = list(range(left, min(n, left + batch_size)))
                windows = torch.stack([features[end - 19 : end + 1] for end in ends])
                logits = self.model.clf(windows.to(self.device))
                out[ends] = softmax(logits[:, -1].float().cpu().numpy(), axis=-1)
        return out
