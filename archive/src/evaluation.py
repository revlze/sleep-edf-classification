"""Фиксированный порядок классов и сопоставимые метрики классификации."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)


def validate_probabilities(probabilities: np.ndarray, n: int) -> None:
    if probabilities.shape != (n, 5) or not np.isfinite(probabilities).all():
        raise ValueError("Expected finite N x 5 probabilities")
    if np.any(probabilities < 0) or not np.allclose(
        probabilities.sum(axis=1), 1, atol=1e-5
    ):
        raise ValueError("Invalid probability distribution")


def classification_metrics(y: np.ndarray, probabilities: np.ndarray) -> dict:
    validate_probabilities(probabilities, len(y))
    pred = probabilities.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        y, pred, labels=list(range(5)), zero_division=0
    )
    return {
        "macro_f1": float(
            f1_score(y, pred, labels=list(range(5)), average="macro", zero_division=0)
        ),
        "f1": f1.tolist(),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "support": support.tolist(),
        "log_loss": float(log_loss(y, probabilities, labels=list(range(5)))),
        "accuracy": float(accuracy_score(y, pred)),
        "confusion_matrix": confusion_matrix(y, pred, labels=list(range(5))).tolist(),
    }


def subject_metrics(y, probabilities, subjects) -> list[dict]:
    return [
        {
            "subject_id": str(s),
            **classification_metrics(y[subjects == s], probabilities[subjects == s]),
        }
        for s in sorted(set(subjects))
    ]


def comparison_table(output_root, experiment_ids):
    """Две строки на подход: fixed/tuned, OOF для выбора и отдельная выданная val."""
    import json
    from pathlib import Path

    import polars as pl

    rows = []
    fold_fingerprint = None
    label_fingerprint = None
    for experiment_id in experiment_ids:
        path = Path(output_root) / experiment_id / "metrics.json"
        provenance = json.loads(
            (path.parent / "provenance.json").read_text(encoding="utf-8")
        )
        current_fold = provenance.get("fold_sha256")
        if fold_fingerprint is None and current_fold is not None:
            fold_fingerprint = current_fold
        elif current_fold is not None and current_fold != fold_fingerprint:
            raise ValueError("Cannot compare different fold assignments/datasets")
        metrics = json.loads(path.read_text(encoding="utf-8"))
        current_labels = provenance.get("labels_sha256")
        if current_labels is None:
            from omegaconf import OmegaConf

            snapshot = OmegaConf.load(path.parent / "resolved.yaml")
            current_labels = snapshot.runtime.get("labels_sha256")
        if label_fingerprint is None:
            label_fingerprint = current_labels
        elif current_labels is not None and current_labels != label_fingerprint:
            raise ValueError("Cannot compare different label datasets")
        single_fit = metrics.get("evaluation_protocol") == "train_val_fixed"
        for regime in ("fixed",) if single_fit else ("fixed", "tuned"):
            cv = {} if single_fit else metrics[regime]
            validation = metrics["val"][regime]
            row = {
                "experiment_id": experiment_id,
                "regime": regime,
                "C": cv.get("C"),
                "params": json.dumps(
                    validation.get("params", {})
                    if single_fit
                    else cv.get("params", {"C": cv.get("C")}),
                    sort_keys=True,
                ),
                "oof_macro_f1": None if single_fit else cv["oof"]["macro_f1"],
                "fold_f1_std": None if single_fit else cv["fold_f1_std"],
                "oof_log_loss": None if single_fit else cv["oof"]["log_loss"],
                "evaluation_protocol": "train_val_fixed"
                if single_fit
                else "group_cv_then_val",
                "val_macro_f1": validation["macro_f1"],
                "val_log_loss": validation["log_loss"],
            }
            row.update(
                {
                    f"val_f1_{stage}": value
                    for stage, value in zip(
                        ("w", "n1", "n2", "n3", "rem"), validation["f1"], strict=True
                    )
                }
            )
            rows.append(row)
    return pl.DataFrame(rows)
