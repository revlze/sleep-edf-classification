"""Author preprocessing per contiguous segment; explicit 30 s epoch alignment."""

from collections.abc import Iterator

import numpy as np
from scipy.signal import resample_poly
from sklearn.preprocessing import RobustScaler


def contiguous_segments(indices: np.ndarray) -> list[slice]:
    indices = np.asarray(indices)
    if indices.ndim != 1 or len(indices) == 0 or np.any(np.diff(indices) <= 0):
        raise ValueError("Expected nonempty strictly increasing epoch indices")
    bounds = np.r_[0, np.flatnonzero(np.diff(indices) != 1) + 1, len(indices)]
    return [slice(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:], strict=True)]


def preprocess_segment(x: np.ndarray, family: str) -> np.ndarray:
    """NPZ µV -> author's volts -> 128 Hz; fit only this continuous segment.

    AnySleep example: polyphase resampling, ±20 IQR clipping, RobustScaler.
    SleepFM: upstream np.interp grid, then channel mean/std (ddof=0), no clip.
    Both normalizations remove the physical scale; volts match EDF readers.
    """
    if x.ndim != 3 or x.shape[1:] != (3000, 3):
        raise ValueError("Expected epochs x 3000 x 3")
    signal = x.reshape(-1, 3).astype(np.float64) * 1e-6
    if family in ("anysleep", "usleep"):
        signal = resample_poly(signal, 128, 100, axis=0)
        iqr = np.percentile(signal, 75, axis=0) - np.percentile(signal, 25, axis=0)
        signal = np.clip(signal, -20 * iqr, 20 * iqr)
        signal = RobustScaler().fit_transform(signal)
    elif family == "sleepfm":
        duration = len(signal) / 100
        old = np.linspace(0, duration, len(signal), endpoint=False)
        new = np.linspace(0, duration, len(x) * 3840, endpoint=False)
        signal = np.stack([np.interp(new, old, s) for s in signal.T], axis=1)
        std = signal.std(axis=0)
        signal = (signal - signal.mean(axis=0)) / np.where(std == 0, 1, std)
    else:
        raise ValueError(f"Unknown preprocessing family: {family}")
    if signal.shape != (len(x) * 3840, 3) or not np.isfinite(signal).all():
        raise ValueError("Invalid preprocessed signal")
    return np.asarray(signal, dtype=np.float32)


def epoch_blocks(
    n: int, core: int, halo: int, *, right_alignment: int = 1
) -> Iterator[tuple[int, int, int, int]]:
    """Yield input [a:b], output [left:right]. Every epoch is emitted once."""
    if n <= 0 or core <= 0 or halo < 0 or right_alignment <= 0:
        raise ValueError("Invalid block dimensions")
    for left in range(0, n, core):
        right = min(left + core, n)
        a, b = max(0, left - halo), min(n, right + halo)
        # Author U-Net left-pads odd pool inputs. Every cropped input must have
        # the same length modulo 4096 samples as the complete segment. At 128 Hz
        # this is 16 epochs. Extend with REAL right context by at most 15 epochs.
        # Never synthesize padding or move an output epoch.
        if a % right_alignment:
            raise ValueError("Block start is not on the encoder grid")
        b += (n - b) % right_alignment
        yield a, b, left, right


def aggregate_patches(logits: np.ndarray) -> np.ndarray:
    """SleepFM: mean of six softmax distributions, never argmax before pooling."""
    from scipy.special import softmax

    if logits.ndim != 2 or logits.shape[1] != 5 or len(logits) % 6:
        raise ValueError("Expected exactly six 5-second logits per epoch")
    return softmax(logits, axis=1).reshape(-1, 6, 5).mean(axis=1)


def pair_vote(probabilities: np.ndarray) -> np.ndarray:
    """Author U-Sleep hard majority; ties select lower class (scipy.stats.mode)."""
    if probabilities.ndim != 3 or probabilities.shape[0] != 2:
        raise ValueError("Expected two EEG/EOG pairs")
    return np.eye(5, dtype=np.float32)[probabilities.argmax(axis=-1)].mean(axis=0)
