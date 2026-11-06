"""Один fit фиксированного LightGBM на train, затем только оценка val."""

import hashlib
from pathlib import Path
from time import perf_counter

import numpy as np
import polars as pl
from omegaconf import OmegaConf

from src.evaluation import classification_metrics, subject_metrics
from src.experiment import update_journal
from src.models.boosting import fit_booster, predict, save_model
from src.progress import Progress
from src.storage import atomic_path, file_lock
from src.training.tabular import save_json


def run_fixed_tree(cfg: dict, *, cache_only: bool = False) -> dict:
    from src.training.boosting import prepare_data, read_json

    out = Path(cfg["paths"]["output_root"])
    folder = out / cfg["experiment_id"]
    with file_lock(out / ".export.lock", shared=True):
        with file_lock(folder / ".run.lock"):
            x, meta, names, train, folds, cache, provenance = prepare_data(
                cfg, require_folds=False
            )
            if folds is not None or x.shape[1] != 855:
                raise ValueError("Expected 855 cached features and no CV folds")
            provenance["evaluation_protocol"] = "train_val_fixed"
            if cache_only:
                return provenance
            cfg["runtime"].update(
                {
                    "labels_sha256": provenance["labels_sha256"],
                    "cache_objects_sha256": provenance["cache_objects_sha256"],
                }
            )
            snapshot = folder / "resolved.yaml"
            if (
                snapshot.exists()
                and OmegaConf.to_container(OmegaConf.load(snapshot), resolve=True)
                != cfg
            ):
                raise ValueError(
                    "Existing result has different settings/data/code. Preserve it and use a separate output root."
                )
            if (folder / "metrics.json").exists():
                metrics = read_json(folder / "metrics.json")
                update_journal(cfg, metrics, folder)
                print(f"Уже завершён: {folder}", flush=True)
                return metrics
            OmegaConf.save(OmegaConf.create(cfg), snapshot)
            save_json(folder / "provenance.json", provenance)
            save_json(folder / "feature_schema.json", {"names": names, "cache": cache})
            y = meta["label"].to_numpy()
            iterations = cfg["training"]["iterations"]
            threads = cfg["training"]["threads"]
            with Progress(
                "Один фиксированный LightGBM", iterations, "итерация"
            ) as progress:
                # Deliberately one call. No eval_set, validation labels, CV or callbacks for early stopping.
                model, count, seconds = fit_booster(
                    x[train],
                    y[train],
                    family="lightgbm",
                    params=cfg["model"]["params"],
                    weight=cfg["model"]["class_weight"],
                    seed=cfg["seed"],
                    threads=threads,
                    iterations=iterations,
                    progress=progress,
                )
                progress.update(
                    detail=f"Fit завершён ({count} итераций); предсказание val"
                )
                start = perf_counter()
                prob = predict(model, x[~train], "lightgbm", threads)
                infer_seconds = perf_counter() - start
                validation = {
                    "params": cfg["model"]["params"],
                    "iterations": count,
                    **classification_metrics(y[~train], prob),
                    "subjects": subject_metrics(
                        y[~train], prob, meta["subject_id"].to_numpy()[~train]
                    ),
                }
                checkpoint = folder / "fixed.txt"
                with atomic_path(checkpoint) as temporary:
                    save_model(model, temporary, "lightgbm")
                save_json(
                    folder / "fixed.model.json",
                    {
                        "model_file": checkpoint.name,
                        "family": "lightgbm",
                        "feature_names": names,
                        "iterations": count,
                        "requested_iterations": iterations,
                        "params": cfg["model"]["params"],
                        "class_order": [0, 1, 2, 3, 4],
                        "model_sha256": hashlib.sha256(
                            checkpoint.read_bytes()
                        ).hexdigest(),
                    },
                )
                with atomic_path(folder / "fixed_predictions.npz") as temporary:
                    np.savez_compressed(
                        temporary,
                        val_ids=meta["id"].to_numpy()[~train].astype(str),
                        val_probabilities=prob,
                    )
                pl.DataFrame(
                    {
                        "feature": names,
                        "importance": model.booster_.feature_importance(
                            importance_type="gain"
                        ),
                    }
                ).sort("importance", descending=True).write_csv(
                    folder / "fixed_importance.csv"
                )
            metrics = {
                "evaluation_protocol": "train_val_fixed",
                "val": {"fixed": validation},
                "fit_seconds": seconds,
                "inference_seconds": infer_seconds,
                "requested_iterations": iterations,
                "actual_iterations": count,
                "selection_note": "Parameters and 420 iterations transferred from 011_tree. One fit on all train; no search/CV/OOF or validation-based early stopping.",
            }
            save_json(folder / "metrics.json", metrics)
            update_journal(cfg, metrics, folder)
            print(f"Завершён единственный fit: {folder}", flush=True)
            return metrics
