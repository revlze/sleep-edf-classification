"""Train-only embedding classifiers and conservative binary soft specialists."""

import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from src.evaluation import classification_metrics, validate_probabilities
from src.expansion_data import physiological_features, shared_folds
from src.models.pretrained_assets import sha256
from src.pretrained import save_json, save_npz
from src.series_data import save_prediction
from src.storage import atomic_path


def predict_probabilities(model, x: np.ndarray) -> np.ndarray:
    # LightGBM auto-generates feature names even for an all-NumPy pipeline. The
    # column order is fixed by feature_matrix; suppress only this cosmetic warning.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
            category=UserWarning,
        )
        return model.predict_proba(x)


def soft_correction(
    base: np.ndarray, positive: np.ndarray, target: str, strength: float
) -> np.ndarray:
    validate_probabilities(base, len(base))
    if (
        positive.shape != (len(base),)
        or not np.isfinite(positive).all()
        or np.any((positive < 0) | (positive > 1))
        or not 0 <= strength <= 1
    ):
        raise ValueError("Invalid binary specialist probabilities/strength")
    if strength == 0:
        return base.copy()
    output = base.astype(np.float64).copy()
    if target == "n1":
        p = (1 - strength) * base[:, 1] + strength * positive
        other = [0, 2, 3, 4]
        old_mass = base[:, other].sum(1)
        conditional = np.divide(
            base[:, other],
            old_mass[:, None],
            out=np.full((len(base), 4), 0.25),
            where=old_mass[:, None] > 0,
        )
        output[:, other] = conditional * (1 - p[:, None])
        output[:, 1] = p
    elif target == "n23":
        mass = base[:, 2] + base[:, 3]
        old = np.divide(base[:, 3], mass, out=np.full(len(base), 0.5), where=mass > 0)
        p = (1 - strength) * old + strength * positive
        output[:, 3], output[:, 2] = mass * p, mass * (1 - p)
    else:
        raise ValueError(target)
    validate_probabilities(output, len(base))
    return output.astype(np.float32)


def feature_matrix(config, engine, split: str, feature_cache: Path | None):
    recipe = config["model"]["features"]
    chunks = [engine.embeddings("tiny", split)]
    if "physio" in recipe:
        chunks.append(physiological_features(engine.data, split, feature_cache))
    if "anysleep" in recipe:
        chunks.append(engine.embeddings("anysleep", split))
    if "probabilities" in recipe:
        chunks.append(engine.baseline("053", split)["probabilities"])
    output = np.concatenate(chunks, axis=1).astype(np.float32)
    output[~np.isfinite(output)] = np.nan
    return output


def estimator(config, trial: dict, threads: int):
    balanced = "balanced" if config["model"]["kind"] == "specialist" else None
    common = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
    if config["model"]["classifier"] == "lr":
        return Pipeline(
            [
                *common,
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=trial["C"],
                        max_iter=1000,
                        solver="lbfgs",
                        class_weight=balanced,
                        random_state=config["seed"],
                    ),
                ),
            ]
        )
    return Pipeline(
        [
            *common,
            (
                "model",
                LGBMClassifier(
                    n_estimators=300,
                    learning_rate=0.04,
                    num_leaves=trial["num_leaves"],
                    max_depth=-1,
                    min_child_samples=100,
                    colsample_bytree=0.8,
                    reg_lambda=5.0,
                    class_weight=balanced,
                    random_state=config["seed"],
                    n_jobs=threads,
                    verbosity=-1,
                    deterministic=True,
                    force_col_wise=True,
                ),
            ),
        ]
    )


def binary_data(y: np.ndarray, target: str):
    if target == "n1":
        return np.ones(len(y), dtype=bool), (y == 1).astype(np.int64)
    if target == "n23":
        return np.isin(y, [2, 3]), (y == 3).astype(np.int64)
    raise ValueError(target)


def fit_tabular(
    config,
    engine,
    folder: Path,
    trained: Path,
    feature_cache: Path | None,
    threads: int,
    contract: str,
):
    started = time.perf_counter()
    model_path = trained / "classifier.joblib"
    manifest_path = trained / "classifier.json"
    oof_path = folder / "oof_predictions.npz"
    selection_path = folder / "selection.json"
    if (
        manifest_path.exists()
        and selection_path.exists()
        and oof_path.with_suffix(".json").exists()
        and (folder / "training_timing.json").exists()
    ):
        info = json.loads(manifest_path.read_text())
        oof_info = json.loads(oof_path.with_suffix(".json").read_text())
        selected = json.loads(selection_path.read_text())
        if (
            info["contract"] != contract
            or info["sha256"] != sha256(model_path)
            or oof_info["contract"] != contract
            or oof_info["prediction_sha256"] != sha256(oof_path)
            or oof_info["selection"] != selected
            or info["trial"] != selected["trial"]
        ):
            raise ValueError("Completed tabular refit/OOF contract differs")
        engine.detail(
            f"{config['experiment_id']}: reuse complete refit and OOF; train embeddings not needed"
        )
        model = joblib.load(model_path)
        timing = json.loads((folder / "training_timing.json").read_text())
        return model, selected, timing["fit_seconds"]
    x = feature_matrix(config, engine, "train", feature_cache)
    meta, folds = engine.data.metadata("train"), shared_folds(engine.data, engine.cache)
    specialist = config["model"]["kind"] == "specialist"
    if specialist:
        eligible, y = binary_data(meta["labels"], config["model"]["target"])
        baseline = engine.baseline("053", "train")["probabilities"]
    else:
        eligible, y = np.ones(len(folds), dtype=bool), meta["labels"]
    results, selected, cv_seconds = [], None, 0.0
    for number, trial in enumerate(config["search"]["trials"]):
        file = folder / "trials" / f"{number}.npz"
        record = file.with_suffix(".json")
        if record.exists():
            info = json.loads(record.read_text())
            if (
                info["contract"] != contract
                or info["trial"] != trial
                or info["sha256"] != sha256(file)
            ):
                raise ValueError("Tabular trial resume differs")
            with np.load(file, allow_pickle=False) as stored:
                oof = stored["probabilities"]
        else:
            fit_started = time.perf_counter()
            oof = np.empty((len(y), 2 if specialist else 5), dtype=np.float32)
            for fold in range(3):
                engine.detail(
                    f"{config['experiment_id']}: trial {number + 1}/{len(config['search']['trials'])}, fold {fold + 1}/3"
                )
                train = (folds != fold) & eligible
                held = folds == fold
                model = estimator(config, trial, threads)
                with threadpool_limits(limits=threads):
                    model.fit(x[train], y[train])
                    p = predict_probabilities(model, x[held])
                if not np.array_equal(model.classes_, np.arange(oof.shape[1])):
                    raise ValueError("Training fold lacks a required class")
                oof[held] = p
            save_npz(file, probabilities=oof, ids=meta["ids"])
            info = {
                "contract": contract,
                "trial": trial,
                "sha256": sha256(file),
                "fit_predict_seconds": time.perf_counter() - fit_started,
            }
            save_json(record, info)
        cv_seconds += info["fit_predict_seconds"]
        if specialist:
            strengths = []
            for alpha in config["model"]["strengths"]:
                p = soft_correction(
                    baseline, oof[:, 1], config["model"]["target"], alpha
                )
                strengths.append(
                    {"strength": alpha, **classification_metrics(meta["labels"], p)}
                )
            # Conservative ties prefer no correction, then lower strength.
            optimum = max(strengths, key=lambda r: (r["macro_f1"], -r["strength"]))
            score = optimum["macro_f1"]
            row = {
                "trial": trial,
                "macro_f1": score,
                "strength": optimum["strength"],
                "strength_search": strengths,
                "binary_report": classification_report(
                    y[eligible],
                    oof[eligible].argmax(1),
                    labels=[0, 1],
                    output_dict=True,
                    zero_division=0,
                ),
                "binary_log_loss": log_loss(y[eligible], oof[eligible], labels=[0, 1]),
                "folds": [
                    classification_metrics(
                        meta["labels"][folds == f],
                        soft_correction(
                            baseline[folds == f],
                            oof[folds == f, 1],
                            config["model"]["target"],
                            optimum["strength"],
                        ),
                    )
                    for f in range(3)
                ],
            }
        else:
            row = {
                "trial": trial,
                **classification_metrics(y, oof),
                "folds": [
                    classification_metrics(y[folds == f], oof[folds == f])
                    for f in range(3)
                ],
            }
            score = row["macro_f1"]
        results.append(row)
        if selected is None or score > selected["macro_f1"]:
            selected, best_oof = row, oof.copy()
        save_json(
            folder / "search.json",
            {
                "selection_set": "train subject-grouped OOF only",
                "trials": results,
                "best": selected,
            },
        )
    model_path = trained / "classifier.joblib"
    manifest_path = trained / "classifier.json"
    if manifest_path.exists():
        info = json.loads(manifest_path.read_text())
        if (
            info["contract"] != contract
            or info["sha256"] != sha256(model_path)
            or info["trial"] != selected["trial"]
        ):
            raise ValueError("Saved classifier contract differs")
        model = joblib.load(model_path)
    else:
        fit_started = time.perf_counter()
        model = estimator(config, selected["trial"], threads)
        with threadpool_limits(limits=threads):
            model.fit(x[eligible], y[eligible])
        with atomic_path(model_path) as temporary:
            joblib.dump(model, temporary)
        info = {
            "contract": contract,
            "trial": selected["trial"],
            "sha256": sha256(model_path),
            "features": x.shape[1],
            "fit_seconds": time.perf_counter() - fit_started,
        }
        save_json(manifest_path, info)
    if specialist:
        save_npz(
            folder / "binary_oof.npz",
            ids=meta["ids"],
            labels=y,
            eligible=eligible,
            probabilities=best_oof,
        )
        p = soft_correction(
            baseline, best_oof[:, 1], config["model"]["target"], selected["strength"]
        )
    else:
        p = best_oof
    save_prediction(
        folder / "oof_predictions.npz",
        {**meta, "probabilities": p},
        {"contract": contract, "selection": selected},
    )
    save_json(folder / "selection.json", selected)
    seconds = cv_seconds + info["fit_seconds"]
    save_json(
        folder / "training_timing.json",
        {
            "fit_seconds": seconds,
            "current_request_stage_seconds": time.perf_counter() - started,
            "semantics": "all CV trials + full refit; frozen embeddings preparation excluded",
        },
    )
    return model, selected, seconds


def tabular_predict(
    config, engine, split, model, selected, feature_cache, folder, threads
):
    x = feature_matrix(config, engine, split, feature_cache)
    with threadpool_limits(limits=threads):
        p = predict_probabilities(model, x).astype(np.float32)
    meta = engine.data.metadata(split)
    if config["model"]["kind"] == "specialist":
        target = config["model"]["target"]
        if split != "test":
            eligible, y = binary_data(meta["labels"], target)
            save_json(
                folder / f"{split}_binary_metrics.json",
                {
                    "eligible_epochs": int(eligible.sum()),
                    "report": classification_report(
                        y[eligible],
                        p[eligible].argmax(1),
                        labels=[0, 1],
                        output_dict=True,
                        zero_division=0,
                    )
                    if np.any(eligible)
                    else None,
                    "log_loss": log_loss(y[eligible], p[eligible], labels=[0, 1])
                    if np.any(eligible)
                    else None,
                },
            )
        save_npz(folder / f"{split}_binary.npz", ids=meta["ids"], probabilities=p)
        p = soft_correction(
            engine.baseline("053", split)["probabilities"],
            p[:, 1],
            target,
            selected["strength"],
        )
    return {**meta, "probabilities": p}
