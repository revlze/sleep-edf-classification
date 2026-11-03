"""Fixed probability ensembles and a discriminative transition-penalized decoder."""

import numpy as np
from scipy.special import logsumexp

from src.evaluation import validate_probabilities


def weighted_probabilities(
    members: list[np.ndarray], weights: list[float]
) -> np.ndarray:
    if (
        not members
        or len(members) != len(weights)
        or any(w < 0 for w in weights)
        or not np.isclose(sum(weights), 1)
    ):
        raise ValueError("Ensemble weights must be nonnegative and sum to one")
    n = len(members[0])
    for member in members:
        validate_probabilities(member, n)
    result = sum(
        w * p.astype(np.float64) for w, p in zip(weights, members, strict=True)
    )
    result = result.astype(np.float32)
    validate_probabilities(result, n)
    return result


def sequence_slices(records: np.ndarray, indices: np.ndarray) -> list[slice]:
    if len(records) != len(indices) or not len(records):
        raise ValueError("Invalid sequence metadata")
    boundaries = (
        np.flatnonzero((records[1:] != records[:-1]) | (np.diff(indices) != 1)) + 1
    )
    bounds = np.r_[0, boundaries, len(records)]
    return [slice(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:], strict=True)]


def transition_prior(
    y, records, indices, alpha: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Train labels only; Laplace smoothing allows every transition."""
    if alpha <= 0 or np.any((y < 0) | (y > 4)):
        raise ValueError("Need valid labels and strictly positive pseudocount")
    counts = np.full((5, 5), alpha, dtype=np.float64)
    initial = np.full(5, alpha, dtype=np.float64)
    for segment in sequence_slices(records, indices):
        values = y[segment]
        initial[values[0]] += 1
        np.add.at(counts, (values[:-1], values[1:]), 1)
    return counts / counts.sum(axis=1, keepdims=True), initial / initial.sum()


def decode_chain(
    probabilities: np.ndarray,
    transition: np.ndarray,
    initial: np.ndarray,
    strength: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Score = sum log posterior + strength*(log pi + sum log A).

    This is a discriminative chain score, NOT a generative HMM emission model.
    Return both Viterbi MAP path and normalized forward/backward marginals.
    Their argmax need not agree; evaluate/export the saved MAP labels explicitly.
    """
    n = len(probabilities)
    validate_probabilities(probabilities, n)
    if (
        strength < 0
        or transition.shape != (5, 5)
        or np.any(transition <= 0)
        or np.any(initial <= 0)
    ):
        raise ValueError("Invalid transition prior/strength")
    if strength == 0:
        return probabilities.argmax(axis=1).astype(np.int64), probabilities.copy()
    emission = np.log(np.maximum(probabilities.astype(np.float64), 1e-12))
    edge, start = strength * np.log(transition), strength * np.log(initial)
    best = np.empty((n, 5))
    forward = np.empty_like(best)
    pointers = np.zeros((n, 5), dtype=np.int64)
    best[0] = forward[0] = emission[0] + start
    for t in range(1, n):
        values = best[t - 1, :, None] + edge
        pointers[t] = values.argmax(axis=0)
        best[t] = emission[t] + values.max(axis=0)
        forward[t] = emission[t] + logsumexp(forward[t - 1, :, None] + edge, axis=0)
    path = np.empty(n, dtype=np.int64)
    path[-1] = best[-1].argmax()
    for t in range(n - 1, 0, -1):
        path[t - 1] = pointers[t, path[t]]
    backward = np.zeros_like(best)
    for t in range(n - 2, -1, -1):
        backward[t] = logsumexp(
            edge + emission[t + 1][None, :] + backward[t + 1][None, :], axis=1
        )
    posterior = forward + backward
    posterior = np.exp(posterior - logsumexp(posterior, axis=1, keepdims=True)).astype(
        np.float32
    )
    validate_probabilities(posterior, n)
    return path, posterior


def viterbi_postprocess(probabilities, records, indices, transition, initial, strength):
    path = np.empty(len(probabilities), dtype=np.int64)
    posterior = np.empty_like(probabilities)
    for segment in sequence_slices(records, indices):
        path[segment], posterior[segment] = decode_chain(
            probabilities[segment], transition, initial, strength
        )
    return path, posterior


def stage_diagnostics(predicted, records, indices, truth=None) -> dict:
    lengths = [[] for _ in range(5)]
    (
        transitions,
        possible,
        n1_short,
        n1_short_retained,
        truth_changes,
        truth_changes_retained,
    ) = 0, 0, 0, 0, 0, 0
    for segment in sequence_slices(records, indices):
        values = predicted[segment]
        changes = np.diff(values) != 0
        transitions += int(changes.sum())
        possible += len(values) - 1
        bounds = np.r_[0, np.flatnonzero(changes) + 1, len(values)]
        for a, b in zip(bounds[:-1], bounds[1:], strict=True):
            lengths[int(values[a])].append(int(b - a))
        if truth is not None:
            y = truth[segment]
            real = np.diff(y) != 0
            truth_changes += int(real.sum())
            truth_changes_retained += int((real & changes).sum())
            bounds = np.r_[0, np.flatnonzero(real) + 1, len(y)]
            for a, b in zip(bounds[:-1], bounds[1:], strict=True):
                if y[a] == 1 and b - a <= 2:
                    n1_short += int(b - a)
                    n1_short_retained += int((values[a:b] == 1).sum())
    return {
        "transition_count": transitions,
        "adjacent_pairs": possible,
        "stage_epoch_counts": np.bincount(predicted, minlength=5).tolist(),
        "stage_run_lengths_epochs": lengths,
        "stage_run_durations_seconds": [[n * 30 for n in stage] for stage in lengths],
        "short_n1_truth_epochs": n1_short if truth is not None else None,
        "short_n1_retained_epochs": n1_short_retained if truth is not None else None,
        "truth_boundary_count": truth_changes if truth is not None else None,
        "truth_boundaries_retained_exact_epoch": truth_changes_retained
        if truth is not None
        else None,
    }
