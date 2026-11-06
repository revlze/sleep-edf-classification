"""Frozen AnySleep inference: vary temporal extent, never weights or precision."""

import gc
from collections import defaultdict

import numpy as np
import torch
from scipy.special import softmax

from src.pretrained_signals import epoch_blocks


class ContextOOM(RuntimeError):
    """A batch of one with the requested temporal extent does not fit."""


def blocks(n: int, inference: dict) -> list[tuple[int, int, int, int]]:
    if inference["mode"] == "full_segment":
        return [(0, n, 0, n)]
    return list(
        epoch_blocks(
            n, inference["core_epochs"], inference["halo_epochs"], right_alignment=16
        )
    )


@torch.inference_mode()
def predict_segment(
    signal: torch.Tensor, models: list, inference: dict, max_batch: int, detail
) -> tuple[np.ndarray, dict]:
    """Reuse preprocessed input; batch only equal lengths; OOM changes batch, not time."""
    n = len(signal) // 3840
    if signal.shape != (n * 3840, 3) or n < 1 or max_batch < 1:
        raise ValueError("Expected nonempty 128 Hz, 30-second, three-channel signal")
    windows = blocks(n, inference)
    groups = defaultdict(list)
    for window in windows:
        a, b, _, _ = window
        groups[b - a].append(window)
    output, batch_sizes = [], []
    # Sequential checkpoints limit activation memory. Models themselves are small.
    for run, model in enumerate(models, 1):
        prob = np.empty((n, 5), np.float32)
        covered = np.zeros(n, np.int32)
        completed = 0
        for same_length in groups.values():
            size = 1 if inference["mode"] == "full_segment" else max_batch
            offset = 0
            while offset < len(same_length):
                take = same_length[offset : offset + size]
                detail(
                    f"AnySleep run{run}/{len(models)} · блоки {completed}/{len(windows)} · "
                    f"batch={len(take)}, вход={take[0][1] - take[0][0]} эпох"
                )
                batch = logits = None
                try:
                    batch = torch.stack(
                        [signal[a * 3840 : b * 3840] for a, b, _, _ in take]
                    )
                    # Original019 is float32 without autocast. Keep that contract.
                    with torch.autocast(device_type=signal.device.type, enabled=False):
                        logits = model(batch)
                    expected = (len(take), take[0][1] - take[0][0], 5)
                    if tuple(logits.shape) != expected:
                        raise ValueError(
                            f"AnySleep output {tuple(logits.shape)} != {expected}"
                        )
                    values = softmax(logits.float().cpu().numpy(), axis=-1)
                except torch.cuda.OutOfMemoryError:
                    del batch, logits
                    gc.collect()
                    torch.cuda.empty_cache()
                    if len(take) == 1:
                        raise ContextOOM(
                            f"{inference['mode']}, run{run}/{len(models)}, {take[0][1] - take[0][0]} epochs, "
                            "batch=1 float32: GPU OOM; no temporal fallback"
                        ) from None
                    size = max(1, len(take) // 2)
                    continue
                for j, (a, _, left, right) in enumerate(take):
                    prob[left:right] = values[j, left - a : right - a]
                    covered[left:right] += 1
                batch_sizes.append(len(take))
                completed += len(take)
                offset += len(take)
                del batch, logits
        if not np.all(covered == 1) or not np.isfinite(prob).all():
            raise ValueError("Missing, repeated or nonfinite epoch predictions")
        output.append(prob)
    return np.mean(output, axis=0), {
        "epochs": n,
        "windows_per_run": len(windows),
        "runs": len(models),
        "actual_batch_sizes": sorted(set(batch_sizes)),
        "input_epochs": [b - a for a, b, _, _ in windows],
        "temporal_fallback": False,
        "precision": "float32, autocast disabled",
    }
