"""Pinned PhysioEx logit voting, separate from the existing rolling adapter."""

import hashlib

import numpy as np
import torch

LENGTH = 20
REFERENCE_REVISION = "66b24320aa39bc635e304b0891eff41131ca54a5"
REFERENCE_SOURCE_SHA256 = (
    "22b3113e2f1a2f08810a27b8de335fb3975077412559cc94d5f97a63eb19ceab"
)
REFERENCE_URL = f"https://github.com/guidogagl/physioex/blob/{REFERENCE_REVISION}/physioex/train/trainer.py#L572"


def backbone_sha256(model) -> str:
    """Actual CNN parameters AND buffers; equal head-only backbones share a cache."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.feature_extractor.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str((array.shape, array.dtype)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


@torch.inference_mode()
def logits_voting(classifier, features, *, window_batch=512, detail=None):
    """One continuous segment, L20 complete windows, exact offset0..19 ordering.

    The author skips segments shorter than20. Those receive the original native
    short rolling output here, explicitly marked with official vote_counts=0.
    No zero/reflected/padded signal epochs or votes are invented.
    """
    if features.ndim != 2 or len(features) == 0 or window_batch < 1:
        raise ValueError(
            "Expected nonempty epoch x CNN-features matrix and positive batch"
        )
    n, device = len(features), features.device
    classifier.eval()
    if n < LENGTH:
        logits = classifier(features[None])[0].float()
        if logits.shape != (n, 5) or not torch.isfinite(logits).all():
            raise ValueError("Invalid short-segment logits")
        probabilities = logits.softmax(-1).cpu().numpy()
        return {
            "probabilities": probabilities,
            "rolling_probabilities": probabilities.copy(),
            "mean_logits": logits.cpu().numpy(),
            "vote_counts": np.zeros(n, np.int32),
            "short_fallback": np.ones(n, bool),
        }
    votes = torch.zeros((n, 5), dtype=torch.float32, device=device)
    counts = torch.zeros(n, dtype=torch.int32, device=device)
    rolling_logits = torch.empty_like(votes)
    done, total = 0, n - LENGTH + 1
    positions = torch.arange(LENGTH, device=device)
    for offset in range(LENGTH):
        usable = n - offset - (n - offset) % LENGTH
        if usable == 0:
            continue
        starts = torch.arange(offset, offset + usable, LENGTH, device=device)
        for start in range(0, len(starts), window_batch):
            batch_starts = starts[start : start + window_batch]
            indices = batch_starts[:, None] + positions
            logits = classifier(features[indices]).float()
            if (
                logits.shape != (len(batch_starts), LENGTH, 5)
                or not torch.isfinite(logits).all()
            ):
                raise ValueError("Invalid voting logits/class axis")
            # Within an offset every destination index is unique; offsets retain
            # the author's accumulation order, without nondeterministic collisions.
            flat = indices.reshape(-1)
            votes[flat] += logits.reshape(-1, 5)
            counts[flat] += 1
            rolling_logits[batch_starts + LENGTH - 1] = logits[:, -1]
            if offset == 0 and start == 0:
                rolling_logits[:LENGTH] = logits[0]
            done += len(batch_starts)
            if detail is not None:
                detail(
                    f"LSTM voting: окна {done}/{total}, offset {offset + 1}/{LENGTH}, batch<={window_batch}"
                )
    if torch.any(counts == 0) or done != total:
        raise ValueError("Official windows did not cover every real epoch")
    mean_logits = votes / counts[:, None]
    return {
        "probabilities": mean_logits.softmax(-1).cpu().numpy(),
        "rolling_probabilities": rolling_logits.softmax(-1).cpu().numpy(),
        "mean_logits": mean_logits.cpu().numpy(),
        "vote_counts": counts.cpu().numpy(),
        "short_fallback": np.zeros(n, bool),
    }
