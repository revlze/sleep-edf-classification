"""Paired validation diagnostics on identical IDs, without fitting or test labels."""

import numpy as np

from src.evaluation import classification_metrics
from src.postprocessing import sequence_slices


def transition_distances(meta, pair=None):
    """Distance to either real epoch adjoining a true transition; -1 if none."""
    distances = np.full(len(meta["ids"]), -1, np.int64)
    for segment in sequence_slices(meta["record_id"], meta["epoch_index"]):
        y = meta["labels"][segment]
        changes = np.flatnonzero(y[1:] != y[:-1]) + 1
        if pair is not None:
            a, b = pair
            changes = changes[
                ((y[changes - 1] == a) & (y[changes] == b))
                | ((y[changes - 1] == b) & (y[changes] == a))
            ]
        if not len(changes):
            continue
        edges = np.unique(np.r_[changes - 1, changes])
        positions = np.arange(len(y))
        at = np.searchsorted(edges, positions)
        distances[segment] = np.minimum(
            np.abs(positions - edges[np.clip(at, 0, len(edges) - 1)]),
            np.abs(positions - edges[np.clip(at - 1, 0, len(edges) - 1)]),
        )
    return distances


def paired_counts(y, before, after, mask):
    return {
        "epochs": int(mask.sum()),
        "baseline_errors": int(((before != y) & mask).sum()),
        "voting_errors": int(((after != y) & mask).sum()),
        "errors_fixed": int(((before != y) & (after == y) & mask).sum()),
        "errors_introduced": int(((before == y) & (after != y) & mask).sum()),
        "prediction_changes": int(((before != after) & mask).sum()),
    }


def comparison(values, baseline, *, fallback=None):
    if not np.array_equal(values["ids"], baseline["ids"]) or not np.array_equal(
        values["labels"], baseline["labels"]
    ):
        raise ValueError("Comparison epoch IDs or target labels differ")
    y = values["labels"]
    old, new = baseline["probabilities"].argmax(1), values["probabilities"].argmax(1)
    full = np.ones(len(y), bool)
    a = classification_metrics(y, baseline["probabilities"])
    b = classification_metrics(y, values["probabilities"])
    result = {
        "baseline_metrics": a,
        "voting_metrics": b,
        "delta_macro_f1": b["macro_f1"] - a["macro_f1"],
        **paired_counts(y, old, new, full),
        "by_true_class": {},
        "transition_slices": {},
        "subjects": [],
    }
    for label, name in enumerate(("W", "N1", "N2", "N3", "REM")):
        result["by_true_class"][name] = paired_counts(y, old, new, y == label)
    distance = transition_distances(values)
    masks = {
        "near_any_transition_d_le_1": (distance >= 0) & (distance <= 1),
        "near_any_transition_d_le_2": (distance >= 0) & (distance <= 2),
        "stable_d_ge_5_or_no_transition": (distance < 0) | (distance >= 5),
    }
    for pair in ((1, 2), (2, 3)):
        d = transition_distances(values, pair)
        masks[f"near_{pair[0]}_{pair[1]}_transition_d_le_1"] = (d >= 0) & (d <= 1)
    for name, mask in masks.items():
        result["transition_slices"][name] = paired_counts(y, old, new, mask)
    result["transition_distance_definition"] = (
        "min real epoch distance to either side of a true label transition, within consecutive epochs of a single record; -1 means no transition in segment"
    )
    result["confusion_directions"] = {}
    for truth, pred in ((1, 2), (2, 1), (2, 3), (3, 2)):
        result["confusion_directions"][f"true_{truth}_pred_{pred}"] = {
            "baseline": int(((y == truth) & (old == pred)).sum()),
            "voting": int(((y == truth) & (new == pred)).sum()),
        }
    for subject in sorted(set(values["subject_id"])):
        mask = values["subject_id"] == subject
        result["subjects"].append(
            {"subject_id": str(subject), **paired_counts(y, old, new, mask)}
        )
    if fallback is not None:
        covered = ~fallback
        result["short_segment_fallback_epochs"] = int(fallback.sum())
        result["official_voting_covered_only"] = {
            **paired_counts(y, old, new, covered),
            "baseline_metrics": classification_metrics(
                y[covered], baseline["probabilities"][covered]
            )
            if covered.any()
            else None,
            "voting_metrics": classification_metrics(
                y[covered], values["probabilities"][covered]
            )
            if covered.any()
            else None,
        }
    return result
