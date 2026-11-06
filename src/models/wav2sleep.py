"""Original pinned wav2sleep-eog with the author's missing-modality interface."""

import importlib
import json
import sys
import types
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import torch
from omegaconf import OmegaConf

from src.models.pretrained_assets import verify_asset
from src.storage import atomic_path, file_lock

MANIFEST = json.loads(
    (Path(__file__).parents[1] / "refinement_assets.json").read_text()
)


def official_files(cache: Path):
    root = cache / "wav2sleep" / MANIFEST["source_revision"]
    for name, info in MANIFEST["files"].items():
        target = root / name
        with file_lock(target.with_suffix(target.suffix + ".lock"), wait=True):
            if not target.exists():
                with atomic_path(target) as temporary:
                    with (
                        urlopen(info["url"], timeout=120) as response,
                        temporary.open("wb") as out,
                    ):
                        while block := response.read(1024 * 1024):
                            out.write(block)
                    verify_asset(temporary, info["sha256"])
            verify_asset(target, info["sha256"])
    return root


def architecture(root):
    # Import only four verified pure-PyTorch source files. No Lightning/Hydra
    # training framework, EDF readers or upstream __init__.py side effects.
    name = "sleep_edf_official_wav2sleep"
    for suffix, folder in (("", root), (".models", root / "models")):
        package = types.ModuleType(name + suffix)
        package.__path__ = [str(folder)]
        sys.modules[name + suffix] = package
    module = importlib.import_module(name + ".models.wav2sleep")
    cfg = OmegaConf.to_container(OmegaConf.load(root / "config.yaml"), resolve=True)
    enc = dict(cfg["signal_encoders"])
    mix = dict(cfg["epoch_mixer"])
    seq = dict(cfg["sequence_mixer"])
    for item in (enc, mix, seq):
        item.pop("_target_")
    if (
        enc["signal_map"] != {"EOG-L": "EOG-L", "EOG-R": "EOG-R"}
        or cfg["num_classes"] != 5
    ):
        raise ValueError("Unexpected wav2sleep checkpoint channel/class contract")
    return module.Wav2Sleep(
        module.SignalEncoders(**enc),
        module.MultiModalAttentionEmbedder(**mix),
        module.SequenceCNN(**seq),
        num_classes=5,
    )


def eog_preprocess(raw):
    """Author noncausal recipe: time interpolation to4096/epoch, per-signal z-score.

    Fixed normalization of this continuous record segment only. Horizontal EOG is
    a differential derivation, mapped explicitly to the available EOG-L slot;
    EOG-R is omitted using native subset support, never duplicated or imputed.
    """
    signal = raw[:, :, 2].reshape(-1).astype(np.float64)
    target = np.arange(1, len(raw) * 4096 + 1, dtype=np.float64) * 30 / 4096
    values = np.interp(
        target, np.arange(len(signal)) / 100, signal, left=0.0, right=0.0
    ).astype(np.float32)
    # torch.std uses correction=1 in the official ParquetDataset implementation.
    values = (values - values.mean()) / max(float(values.std(ddof=1)), 1e-8)
    return values


class Wav2SleepAdapter:
    def __init__(self, cache, device):
        root = official_files(cache)
        self.model = architecture(root)
        self.model.load_state_dict(
            torch.load(root / "state_dict.pth", map_location="cpu", weights_only=True),
            strict=True,
        )
        self.model.eval().to(device)
        self.device = device
        self.provenance = {
            **MANIFEST,
            "input": "horizontal EOG -> EOG-L only; native EOG-R omission",
            "limitation": "differential horizontal derivation differs from author EOG-L; no claim of identical electrode reference",
            "preprocessing": "linear time interpolation4096/epoch, per-contiguous-segment sample z-score; full variable-length segment",
        }

    @torch.inference_mode()
    def probabilities(self, raw):
        values = eog_preprocess(raw)
        # Native API consumes [batch, continuous samples]; its multimodal mixer
        # accepts a dictionary with a single available signal.
        logits = self.model({"EOG-L": torch.from_numpy(values[None]).to(self.device)})
        if logits.shape != (1, len(raw), 5):
            raise ValueError("wav2sleep output shape differs")
        return logits[0].float().softmax(-1).cpu().numpy()
