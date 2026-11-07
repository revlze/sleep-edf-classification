"""Unified validation/test artifacts with explicit labels for structured decoders."""

from pathlib import Path

import numpy as np

from src.evaluation import classification_metrics
from src.journal import upsert_result
from src.models.pretrained_assets import sha256
from src.postprocessing import stage_diagnostics
from src.pretrained import save_json
from src.series_data import save_prediction, submission_frame
from src.storage import atomic_path


def prediction_metrics(values: dict) -> dict:
    y, prob = values["labels"], values["probabilities"]
    result = classification_metrics(y, prob)
    if "predicted_labels" in values:
        path = values["predicted_labels"]
        hard = classification_metrics(y, np.eye(5)[path])
        hard["log_loss"] = result["log_loss"]
        hard["marginal_argmax_macro_f1"] = result["macro_f1"]
        hard["label_semantics"] = (
            "explicit decoded MAP path; log_loss uses normalized chain marginals"
        )
        return hard
    return result


def evaluate_values(values: dict) -> dict:
    result = prediction_metrics(values)
    result["subjects"] = []
    for person in sorted(set(values["subject_id"])):
        mask = values["subject_id"] == person
        result["subjects"].append(
            {
                "subject_id": str(person),
                **prediction_metrics({k: v[mask] for k, v in values.items()}),
            }
        )
    result["diagnostics"] = stage_diagnostics(
        values.get("predicted_labels", values["probabilities"].argmax(1)),
        values["record_id"],
        values["epoch_index"],
        values["labels"],
    )
    return result


def ensure_submission(folder: Path, values: dict, data) -> str:
    """Recreate a missing CSV without inference; reject an altered completed CSV."""
    import polars as pl

    expected = submission_frame(values, data.sample, data.smoke)
    target = folder / ("submission_smoke.csv" if data.smoke else "submission.csv")
    if target.exists():
        if not pl.read_csv(target).equals(expected):
            raise ValueError("Saved submission differs from aligned test predictions")
    else:
        with atomic_path(target) as temporary:
            expected.write_csv(temporary)
    return sha256(target)


def write_split(folder: Path, split: str, values: dict, data, provenance: dict):
    if split == "test" and "predicted_labels" not in values:
        values = {
            **values,
            "predicted_labels": values["probabilities"].argmax(axis=1).astype(np.int64),
        }
    save_prediction(folder / f"{split}_predictions.npz", values, provenance)
    if split == "val":
        metrics = evaluate_values(values)
        save_json(folder / "validation_metrics.json", metrics)
        return metrics
    if split == "test":
        ensure_submission(folder, values, data)
        save_json(
            folder / "test_diagnostics.json",
            stage_diagnostics(
                values.get("predicted_labels", values["probabilities"].argmax(1)),
                values["record_id"],
                values["epoch_index"],
            ),
        )
    return None


def journal(
    config_path: Path,
    config: dict,
    metrics: dict,
    folder: Path,
    project: Path,
    *,
    status: str = "complete",
    evaluation_set: str = "provided_val_then_unlabelled_test",
):
    validation = metrics["validation"]
    row = {
        "experiment_id": config["experiment_id"],
        "config": f"configs/{config_path.name}",
        "model": config["model"]["kind"],
        "summary": config["summary"],
        "seed": config["seed"],
        "evaluation_set": evaluation_set,
        "macro_f1": validation["macro_f1"],
        "train_seconds": metrics.get("training_seconds", ""),
        "infer_seconds": metrics["inference_seconds"],
        "status": status,
        "artefacts_path": folder.as_posix(),
    }
    row.update(
        {
            f"f1_{s}": f
            for s, f in zip(
                ("w", "n1", "n2", "n3", "rem"), validation["f1"], strict=True
            )
        }
    )
    upsert_result(project / "experiments.csv", row)
