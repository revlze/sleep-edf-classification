"""Subject-independent CV, внутренние trials C и финальные refit-модели."""

import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits
from tqdm.auto import tqdm

from src.evaluation import classification_metrics, subject_metrics
from src.models.linear import make_model
from src.preprocessing import make_preprocessor
from src.progress import stage


def save_json(path: Path, value: dict) -> None:
    temp = path.with_suffix(".tmp")
    temp.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    temp.replace(path)


def grouped_folds(
    metadata: pl.DataFrame, path: Path, seed: int, n_splits: int
) -> np.ndarray:
    """Сохранить назначения людей; проверить неизменность labels и набора эпох."""
    if path.exists():
        saved = pl.read_csv(path)
        joined = metadata.select("id", "subject_id", "label").join(
            saved,
            on=["id", "subject_id", "label"],
            how="left",
            maintain_order="left",
            validate="1:1",
        )
        if joined["fold"].null_count() or saved.height != metadata.height:
            raise ValueError("Stored folds do not match current data")
        folds = joined["fold"].to_numpy()
    else:
        y = metadata["label"].to_numpy()
        groups = metadata["subject_id"].to_numpy()
        folds = np.full(len(y), -1, dtype=int)
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold, (_, validation) in enumerate(cv.split(np.zeros(len(y)), y, groups)):
            folds[validation] = fold
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata.select("id", "subject_id", "label").with_columns(
            pl.Series("fold", folds)
        ).write_csv(path)
    if set(folds) != set(range(n_splits)):
        raise ValueError("Unexpected fold IDs")
    check = (
        metadata.select("subject_id")
        .with_columns(pl.Series("fold", folds))
        .group_by("subject_id")
        .agg(pl.col("fold").n_unique())
    )
    if check["fold"].max() != 1:
        raise ValueError("Subject leakage in folds")
    return folds


def fit_pipeline(
    x: np.ndarray, y: np.ndarray, names: list[str], c: float, config: dict
) -> tuple[Pipeline, bool]:
    pipeline = Pipeline(
        [
            (
                "preprocess",
                make_preprocessor(names, config["preprocessing"], config["seed"]),
            ),
            ("model", make_model(c, y, config["model"], config["seed"])),
        ]
    )
    with (
        warnings.catch_warnings(record=True) as recorded,
        threadpool_limits(limits=config["training"]["threads"]),
    ):
        warnings.simplefilter("always", ConvergenceWarning)
        pipeline.fit(x, y)
    converged = not any(issubclass(w.category, ConvergenceWarning) for w in recorded)
    if not np.array_equal(pipeline.named_steps["model"].classes_, np.arange(5)):
        raise ValueError("Incorrect model class order")
    return pipeline, converged


def evaluate_trial(x, y, subjects, names, folds, c, config, folder):
    token = format(c, ".12g").replace(".", "p").replace("-", "m")
    metric_file = folder / f"C_{token}.json"
    prob_file = folder / f"C_{token}.npz"
    if metric_file.exists() and prob_file.exists():
        tqdm.write(f"C={c:g}: готовый trial из кэша")
        result = json.loads(metric_file.read_text(encoding="utf-8"))
        with np.load(prob_file, allow_pickle=False) as z:
            prob = z["probabilities"]
        classification_metrics(y, prob)
        return result, prob
    started = time.perf_counter()
    prob = np.full((len(y), 5), np.nan, dtype=np.float32)
    fold_metrics = []
    for fold in tqdm(
        sorted(set(folds)), desc=f"C={c:g}: folds", unit="fold", leave=False
    ):
        train = folds != fold
        valid = ~train
        with stage(f"C={c:g}, fold={fold + 1}: preprocessing + fit"):
            model, converged = fit_pipeline(x[train], y[train], names, c, config)
        with (
            stage(f"C={c:g}, fold={fold + 1}: predict"),
            threadpool_limits(limits=config["training"]["threads"]),
        ):
            prob[valid] = model.predict_proba(x[valid])
        metrics = classification_metrics(y[valid], prob[valid])
        fold_metrics.append({"fold": int(fold), "converged": converged, **metrics})
        tqdm.write(
            f"C={c:g} fold={fold} macro-F1={metrics['macro_f1']:.4f} converged={converged}",
        )
    result = {
        "C": c,
        "oof": classification_metrics(y, prob),
        "folds": fold_metrics,
        "fold_f1_std": float(np.std([f["macro_f1"] for f in fold_metrics])),
        "converged": all(f["converged"] for f in fold_metrics),
        "seconds": time.perf_counter() - started,
        "subjects": subject_metrics(y, prob, subjects),
    }
    temp = prob_file.with_suffix(".tmp.npz")
    np.savez_compressed(temp, probabilities=prob)
    temp.replace(prob_file)
    save_json(metric_file, result)
    return result, prob


def run_linear(
    x: np.ndarray,
    metadata: pl.DataFrame,
    names: list[str],
    config: dict,
    folder: Path,
    folds: np.ndarray,
) -> dict:
    train = metadata["split"].to_numpy() == "train"
    val = ~train
    if set(metadata["split"]) != {"train", "val"}:
        raise ValueError("Only train/val accepted for training")
    tx = x[train]
    y = metadata["label"].to_numpy()[train]
    subjects = metadata["subject_id"].to_numpy()[train]
    trial_folder = folder / "trials"
    trial_folder.mkdir(exist_ok=True)
    fixed = float(config["model"]["C"])
    candidates = list(dict.fromkeys([fixed, *map(float, config["search"]["C"])]))
    results = []
    probabilities = {}
    for c in tqdm(candidates, desc="Поиск C (без возможного уточнения)", unit="trial"):
        result, prob = evaluate_trial(
            tx, y, subjects, names, folds, c, config, trial_folder
        )
        results.append(result)
        probabilities[c] = prob
    eligible = [r for r in results if r["converged"]]
    if not eligible:
        raise RuntimeError(
            "No converged trials; increase max_iter and rerun with explicit settings"
        )

    def rank(r):
        return (-r["oof"]["macro_f1"], r["fold_f1_std"], r["C"])

    best = min(eligible, key=rank)
    # Exactly one optional decade extension; no unbounded search.
    if config["search"]["refine_boundary"] and best["C"] in (
        min(candidates),
        max(candidates),
    ):
        c = best["C"] / 10 if best["C"] == min(candidates) else best["C"] * 10
        tqdm.write(f"Уточнение границы: один дополнительный trial C={c:g}")
        result, prob = evaluate_trial(
            tx, y, subjects, names, folds, c, config, trial_folder
        )
        results.append(result)
        probabilities[c] = prob
        if result["converged"]:
            best = min([best, result], key=rank)
    baseline = next(r for r in results if r["C"] == fixed)
    if not baseline["converged"]:
        raise RuntimeError("Fixed baseline did not converge; comparison is incomplete")
    pl.DataFrame(
        [
            {
                "C": r["C"],
                "oof_macro_f1": r["oof"]["macro_f1"],
                "fold_f1_std": r["fold_f1_std"],
                "log_loss": r["oof"]["log_loss"],
                "converged": r["converged"],
                "seconds": r["seconds"],
            }
            for r in results
        ]
    ).write_csv(folder / "search.csv")
    validation = {}
    refit_seconds = 0.0
    inference_seconds = 0.0
    fitted = {}
    for regime, c in tqdm(
        [("fixed", fixed), ("tuned", best["C"])], desc="Refit и val", unit="модель"
    ):
        start = time.perf_counter()
        if c not in fitted:
            with stage(f"{regime}, C={c:g}: fit на всём train"):
                fitted[c], converged = fit_pipeline(tx, y, names, c, config)
            if not converged:
                raise RuntimeError(f"Full train refit did not converge: C={c}")
            refit_seconds += time.perf_counter() - start
        pipeline = fitted[c]
        start = time.perf_counter()
        with (
            stage(f"{regime}: predict val"),
            threadpool_limits(limits=config["training"]["threads"]),
        ):
            vp = pipeline.predict_proba(x[val]).astype(np.float32)
        inference_seconds += time.perf_counter() - start
        validation[regime] = {
            "C": c,
            **classification_metrics(metadata["label"].to_numpy()[val], vp),
            "subjects": subject_metrics(
                metadata["label"].to_numpy()[val],
                vp,
                metadata["subject_id"].to_numpy()[val],
            ),
        }
        tqdm.write(f"{regime}: сохранение модели и вероятностей")
        joblib.dump(
            {
                "pipeline": pipeline,
                "feature_names": names,
                "features": config["features"],
                "experiment_id": config["experiment_id"],
                "regime": regime,
                "C": c,
            },
            folder / f"{regime}.joblib",
        )
        np.savez_compressed(
            folder / f"{regime}_predictions.npz",
            train_ids=metadata["id"].to_numpy()[train].astype(str),
            oof=probabilities[c],
            val_ids=metadata["id"].to_numpy()[val].astype(str),
            val_probabilities=vp,
        )
    metrics = {
        "fixed": baseline,
        "tuned": best,
        "val": validation,
        "refit_seconds": refit_seconds,
        "inference_seconds": inference_seconds,
        "search_seconds": sum(r["seconds"] for r in results),
        "selection_note": "OOF used for selection; not an unbiased estimate after search",
    }
    save_json(folder / "metrics.json", metrics)
    return metrics
