"""Pinned original AnySleep variants; HF soft vote follows author mean-logits code."""

import ast
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy.special import softmax
from torch import nn
from torch.nn import functional as F

from src import series_assets  # noqa: F401 -- register pinned official assets
from src.models.pretrained_assets import official_module, resolve_asset


def aggregate_high_frequency(
    logits: np.ndarray, frequency: int, voting: str
) -> np.ndarray:
    if (
        frequency not in (1, 4, 8)
        or logits.ndim != 2
        or logits.shape[1] != 5
        or len(logits) % frequency
    ):
        raise ValueError("Invalid high-frequency logits/grid")
    if not np.isfinite(logits).all():
        raise ValueError("Nonfinite logits")
    grouped = logits.reshape(-1, frequency, 5)
    if voting == "soft":
        # Official script: mean raw logits, then argmax. Softmax retains that argmax
        # and supplies probabilities for our scoring/ensemble interface.
        return softmax(grouped.mean(axis=1), axis=1).astype(np.float32)
    if voting == "hard":
        return np.eye(5, dtype=np.float32)[grouped.argmax(axis=2)].mean(axis=1)
    raise ValueError(voting)


def definitions(path: Path, namespace: dict, names: set[str]) -> None:
    """Keep exact upstream class/function ASTs; skip training/demo imports only."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [
        n
        for n in tree.body
        if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names
    ]
    if {n.name for n in nodes} != names:
        raise ValueError(f"Official source definitions changed: {path}")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)


class _InferenceBase(nn.Module):
    """Replace Hydra persistence only; model classes and forward ASTs are unchanged."""

    def __init__(self, seed=None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

    def load(self, path):
        if path is not None:
            raise ValueError("Load only verified weights, outside the Hydra loader")


def fusion_model(
    variant: str, frequency: int, cache: Path, inputs: Path | None, state: dict
):
    files = [
        "anysleep-fusion-components.py",
        "anysleep-fusion-utils.py",
        "anysleep-init.py",
        f"anysleep-{variant}-fusion.py",
    ]
    paths = {name: resolve_asset(name, cache, inputs) for name in files}
    config = SimpleNamespace(data=SimpleNamespace(sampling_rate=128, epoch_duration=30))
    namespace = {
        "np": np,
        "torch": torch,
        "nn": nn,
        "F": F,
        "BaseModel": _InferenceBase,
        "Config": SimpleNamespace(get=lambda: config),
        "logger": logging.getLogger(__name__),
    }
    definitions(
        paths[files[0]],
        namespace,
        {"USleepEncoderBlock", "USleepDecoderBlock", "SegmentClassifier"},
    )
    definitions(paths[files[1]], namespace, {"calc_filters"})
    definitions(paths[files[2]], namespace, {"_weights_init"})
    cls = "AnySleepEF" if variant == "early" else "AnySleepLF"
    definitions(paths[files[3]], namespace, {cls, "ChannelAttention"})
    params = {
        "att_filters": int(state["channel_attention.encoder.0.weight"].shape[0]),
        "att_hidden_size": int(state["channel_attention.mlps.0.0.weight"].shape[0]),
    }
    if variant == "early":
        params["num_virt_channels"] = int(state["encoders.0.enc.0.weight"].shape[1])
    model = namespace[cls](path=None, sleep_stage_frequency=frequency, **params)
    model.released_architecture_parameters = params
    return model, files


class AnySleepVariant:
    right_boundary_alignment = 16
    samples_per_epoch = 3840

    def __init__(self, recipe: dict, cache: Path, inputs: Path | None, device: str):
        self.device = torch.device(device)
        self.frequency = int(recipe.get("frequency", 1))
        self.voting = recipe.get("voting", "soft")
        if self.frequency not in (1, 4, 8) or self.voting not in ("soft", "hard"):
            raise ValueError("Supported high-frequency settings: 1/4/8, soft/hard")
        variant = recipe.get("variant", "base")
        checkpoint = recipe["checkpoint"]
        expected = "anysleep-" + (f"{variant}-fusion-" if variant != "base" else "")
        if checkpoint not in {f"{expected}run{i}.pth" for i in (1, 2, 3)}:
            raise ValueError("Checkpoint/architecture mismatch")
        state = torch.load(
            resolve_asset(checkpoint, cache, inputs),
            map_location="cpu",
            weights_only=True,
        )
        if variant == "base":
            source = "anysleep_no_hydra.py"
            module = official_module(resolve_asset(source, cache, inputs))
            self.model = module.AnySleep(
                path=None, sleep_stage_frequency=self.frequency
            )
            self.assets = [source]
        elif variant in ("early", "late"):
            self.model, self.assets = fusion_model(
                variant, self.frequency, cache, inputs, state["state_dict"]
            )
        else:
            raise ValueError(variant)
        self.model.load_state_dict(state["state_dict"], strict=True)
        self.assets.append(checkpoint)
        self.model.eval().to(self.device)

    def logits(self, signal: np.ndarray) -> np.ndarray:
        n = len(signal) // 3840
        with torch.inference_mode():
            x = (
                torch.from_numpy(np.ascontiguousarray(signal))
                .unsqueeze(0)
                .to(self.device)
            )
            logits = self.model(x)
        if logits.shape != (1, n * self.frequency, 5):
            raise ValueError(f"High-frequency alignment: {logits.shape}")
        return logits[0].float().cpu().numpy()

    def probabilities(self, signal: np.ndarray) -> np.ndarray:
        return aggregate_high_frequency(
            self.logits(signal), self.frequency, self.voting
        )
