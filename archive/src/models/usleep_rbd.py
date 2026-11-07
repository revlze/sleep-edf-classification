"""Official RBD/PD U-Sleep with our existing019 preprocessing and128/64 blocks.
Author source/weights: jesperstroem/U-Sleep-for-RBD-PD, non-commercial research.
The older017 used AnySleep's usleep-run1, a different source and checkpoint.
"""

import hashlib
from pathlib import Path

import numpy as np
import torch

from src.models.pretrained_assets import official_module, verify_asset
from src.pretrained_signals import contiguous_segments, epoch_blocks, preprocess_segment

REVISION = "19e72b632b1bd87650f7d560f42aa32d47bd9a9b"
CHECKPOINT_SHA = "5d7abe5e6799b697cda823657ef5d61a07ebbece3224bd5e863cae1983f6bd00"
ARCH_SHA = "9579133211737df86b5eb8f6c89e414fc612be52233fb88c8a09ed89cfb51336"
STATE_SHA = "4cb5e1f0fbec21b0823cf03e0d7fbc292cbab6b3ce3bade14b19e81f1b4a669d"


def state_sha(state: dict) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        tensor = tensor.detach().cpu().contiguous()
        for value in (name, str(tuple(tensor.shape)), str(tensor.dtype)):
            digest.update(value.encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def load_model(assets: Path, device: torch.device):
    checkpoint = assets / "Pretrained_Model.ckpt"
    architecture = assets / "author_usleep.py"
    verify_asset(checkpoint, CHECKPOINT_SHA)
    verify_asset(architecture, ARCH_SHA)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state, hp = saved["state_dict"], saved["hyper_parameters"]
    parameters = {
        k: hp[k] for k in ("initial_filters", "complexity_factor", "progression_factor")
    }
    if (
        len(state) != 265
        or state_sha(state) != STATE_SHA
        or not hp["include_eog"]
        or parameters
        != dict(initial_filters=5, complexity_factor=1.67, progression_factor=2)
        or not all(k.startswith("model.") for k in state)
    ):
        raise ValueError("Not the pinned official Pretrained_Model checkpoint")
    model = official_module(architecture).USleep(num_channels=2, **parameters)
    model.load_state_dict(
        {k.removeprefix("model."): v for k, v in state.items()}, strict=True
    )
    return model.eval().to(device)


def preprocess(raw: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, list[slice]]:
    """Reuse019 preprocessing independently inside every continuous segment."""
    if raw.shape != (len(indices), 3000, 3):
        raise ValueError("Expected100Hz 30s epochs, two EEG and horizontal EOG")
    segments = contiguous_segments(indices)
    normalized = np.empty((len(raw), 3, 3840), dtype=np.float32)
    for segment in segments:
        signal = preprocess_segment(raw[segment], "anysleep")
        normalized[segment] = signal.reshape(-1, 3840, 3).transpose(0, 2, 1)
    return normalized, segments


@torch.inference_mode()
def forward(model, batch: np.ndarray, device: torch.device, epochs: int) -> np.ndarray:
    logits = model(torch.from_numpy(batch).to(device))
    if logits.shape != (1, 5, epochs) or not torch.isfinite(logits).all():
        raise ValueError(
            f"Invalid native logits shape/alignment: {tuple(logits.shape)}"
        )
    return logits.softmax(1)[0].T.cpu().numpy()


def predict(model, normalized, segments, device, progress, prefix):
    """Our128-core/64-halo protocol, no overlap weighting or full-record pass.

    Keep only the core output from each block. The16-epoch encoder grid is
    preserved by the same epoch_blocks helper used for019. Both EEG/EOG pairs
    use the identical blocks and one official checkpoint; mean softmax pairs.
    """
    pairs = np.empty((len(normalized), 2, 5), np.float32)
    coverage = np.zeros((len(normalized), 2), np.int64)
    blocks = []
    for segment in segments:
        length = segment.stop - segment.start
        for a, b, left, right in epoch_blocks(length, 128, 64, right_alignment=16):
            blocks.append(
                (
                    segment.start + a,
                    segment.start + b,
                    segment.start + left,
                    segment.start + right,
                )
            )
    details = []
    for block_number, (a, b, left, right) in enumerate(blocks, 1):
        for pair, eeg in enumerate((0, 1)):
            progress.update(
                detail=f"{prefix} · блок128/halo64 {block_number}/{len(blocks)} · EEG{eeg + 1} · core{left}:{right}"
            )
            batch = np.ascontiguousarray(
                normalized[a:b][:, [eeg, 2]].transpose(1, 0, 2).reshape(1, 2, -1)
            )
            probabilities = forward(model, batch, device, b - a)
            pairs[left:right, pair] = probabilities[left - a : right - a]
            coverage[left:right, pair] += 1
        details.append(
            {"input_start": a, "input_stop": b, "core_start": left, "core_stop": right}
        )
    if not np.all(coverage == 1):
        raise ValueError("Each epoch/pair must receive exactly one core prediction")
    return (
        pairs.mean(axis=1),
        {"pair_probabilities": pairs, "core_coverage": coverage},
        details,
    )
