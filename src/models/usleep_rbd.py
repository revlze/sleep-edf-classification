"""Official RBD/PD U-Sleep checkpoint; independent adapter, no competitor code.

Inference recipe reference: baffledhedgehog/sleep-edf-u-sleep-v4, version4.
Author source/weights: jesperstroem/U-Sleep-for-RBD-PD, non-commercial research.
The older017 used AnySleep's usleep-run1, a different source and checkpoint.
"""

import gc
import hashlib
from pathlib import Path

import numpy as np
import torch
from scipy.signal import butter, resample_poly, sosfiltfilt

from src.models.pretrained_assets import official_module, verify_asset
from src.pretrained_signals import contiguous_segments

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
    """Float64 DC removal, gapwise128Hz/highpass, RECORD median/IQR then clip.

    Record-level normalization is intentional: it matches the referenced recipe;
    filter/resampling and temporal inference never cross a missing-epoch boundary.
    """
    if raw.shape != (len(indices), 3000, 3):
        raise ValueError("Expected100Hz 30s epochs, two EEG and horizontal EOG")
    segments = contiguous_segments(indices)
    result = np.empty((len(raw), 3, 3840), dtype=np.float32)
    highpass = butter(2, 0.1, btype="highpass", fs=128, output="sos")
    for channel in range(3):
        signal = raw[:, :, channel].reshape(-1).astype(np.float64)
        signal -= signal.mean(dtype=np.float64)
        filtered = np.empty(len(raw) * 3840, dtype=np.float64)
        for segment in segments:
            a, b = segment.start, segment.stop
            filtered[a * 3840 : b * 3840] = sosfiltfilt(
                highpass, resample_poly(signal[a * 3000 : b * 3000], 32, 25)
            )
        q25, median, q75 = np.percentile(filtered, [25, 50, 75])
        scale = q75 - q25
        if scale < 10 * np.finfo(np.float64).eps:
            scale = 1.0
        filtered -= median
        filtered /= scale
        iqr = float(np.subtract(*np.percentile(filtered, [75, 25])))
        np.clip(filtered, -20 * iqr, 20 * iqr, out=filtered)
        result[:, channel] = filtered.reshape(len(raw), 3840)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite normalized signals")
    return result, segments


@torch.inference_mode()
def forward(model, batch: np.ndarray, device: torch.device, epochs: int) -> np.ndarray:
    logits = model(torch.from_numpy(batch).to(device))
    if logits.shape != (1, 5, epochs) or not torch.isfinite(logits).all():
        raise ValueError(
            f"Invalid native logits shape/alignment: {tuple(logits.shape)}"
        )
    return logits.softmax(1)[0].T.cpu().numpy()


def predict(model, normalized, segments, device, progress, prefix):
    """64/32 Hann softmax mean, true full segments, then contexts and EEG pairs.

    No AMP, changed batch sizes or labels. OOM retries the SAME full segment on CPU,
    records that fallback explicitly, and returns the model to the original device.
    """
    n = len(normalized)
    weighted = np.zeros((n, 2, 5), np.float64)
    counts = np.zeros(n, np.float64)
    full = np.empty((n, 2, 5), np.float32)
    hann = np.maximum(0.05, np.hanning(64)).astype(np.float32)
    details = []
    total = sum(2 * (len(range(s.start, s.stop, 32)) + 1) for s in segments)
    done = 0
    for segment in segments:
        for start in range(segment.start, segment.stop, 32):
            end = min(start + 64, segment.stop)
            for pair, eeg in enumerate((0, 1)):
                progress.update(
                    detail=f"{prefix} · окна64/32 · EEG{eeg + 1} · forward {done + 1}/{total}"
                )
                batch = np.zeros((1, 2, 64 * 3840), np.float32)
                batch[0, :, : (end - start) * 3840] = (
                    normalized[start:end][:, [eeg, 2]].transpose(1, 0, 2).reshape(2, -1)
                )
                p = forward(model, batch, device, 64)
                weighted[start:end, pair] += (
                    p[: end - start] * hann[: end - start, None]
                )
                done += 1
            counts[start:end] += hann[: end - start]
        for pair, eeg in enumerate((0, 1)):
            progress.update(
                detail=f"{prefix} · full segment{segment.start}:{segment.stop} · EEG{eeg + 1} · forward {done + 1}/{total}"
            )
            batch = np.ascontiguousarray(
                normalized[segment][:, [eeg, 2]].transpose(1, 0, 2).reshape(1, 2, -1)
            )
            fallback = False
            try:
                p = forward(model, batch, device, segment.stop - segment.start)
            except torch.cuda.OutOfMemoryError:
                fallback = True
            if fallback:
                gc.collect()
                torch.cuda.empty_cache()
                progress.update(
                    detail=f"{prefix} · CUDA OOM: тот же полный сегмент на CPU; контекст не обрезается"
                )
                model.to("cpu")
                p = forward(
                    model, batch, torch.device("cpu"), segment.stop - segment.start
                )
                model.to(device)
            full[segment, pair] = p
            details.append(
                {
                    "start": segment.start,
                    "stop": segment.stop,
                    "eeg": eeg,
                    "full_cpu_oom_fallback": fallback,
                }
            )
            done += 1
    if not (counts > 0).all():
        raise ValueError("An epoch received no Hann window")
    windows = (weighted / counts[:, None, None]).astype(np.float32)
    # Preserve float32 reduction order of the reference recipe; no renormalization.
    probabilities = (0.5 * (windows + full)).mean(axis=1)
    return (
        probabilities,
        {
            "window_pair_probabilities": windows,
            "full_pair_probabilities": full,
            "hann_weight_sum": counts,
        },
        details,
    )
