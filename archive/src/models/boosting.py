"""CPU LightGBM/CatBoost: единые веса, фиксированные пять классов, native models."""

import os
from pathlib import Path
from time import perf_counter

import numpy as np

from src.models.linear import class_weights


def resolve_threads(value: int | str) -> int:
    available = os.process_cpu_count() or os.cpu_count() or 1
    if value == "auto":
        return max(1, available - 1)
    value = int(value)
    if value == -1:
        return available
    if value < 1 or value > available:
        raise ValueError(f"threads must be 1..{available}, -1 or auto")
    return value


def macro_f1(y: np.ndarray, prob: np.ndarray) -> float:
    counts = np.bincount(5 * y.astype(int) + prob.argmax(axis=1), minlength=25).reshape(
        5, 5
    )
    denominator = counts.sum(axis=0) + counts.sum(axis=1)
    return float(
        np.divide(
            2 * np.diag(counts), denominator, out=np.zeros(5), where=denominator != 0
        ).mean()
    )


def fit_booster(
    x: np.ndarray,
    y: np.ndarray,
    *,
    family: str,
    params: dict,
    weight: str,
    seed: int,
    threads: int,
    iterations: int,
    valid=None,
    patience: int = 100,
    progress=None,
    max_seconds=None,
):
    weights = class_weights(y, weight)
    started = perf_counter()
    if family == "lightgbm":
        import lightgbm as lgb

        model = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=5,
            n_estimators=iterations,
            random_state=seed,
            n_jobs=threads,
            verbosity=-1,
            deterministic=True,
            force_col_wise=True,
            class_weight=weights,
            **params,
        )

        def metric(labels, probabilities):
            loss = -np.log(
                np.maximum(
                    probabilities[np.arange(len(labels)), labels.astype(int)], 1e-15
                )
            ).mean()
            return [
                ("macro_f1", macro_f1(labels, probabilities), True),
                ("log_loss", float(loss), False),
            ]

        def callback(env):
            if progress:
                progress.update(
                    1 if progress.unit == "итерация" else 0,
                    detail=f"LightGBM: итерация {env.iteration + 1}/{iterations}",
                )
            if max_seconds is not None and perf_counter() - started >= max_seconds:
                raise lgb.callback.EarlyStopException(
                    env.iteration, env.evaluation_result_list
                )

        callbacks = [callback]
        kwargs = {}
        if valid is not None:
            kwargs.update(eval_X=valid[0], eval_y=valid[1], eval_metric=metric)
            if max_seconds is None:
                callbacks.append(
                    lgb.early_stopping(patience, first_metric_only=True, verbose=False)
                )
        model.set_params(metric="None")
        model.fit(x, y, callbacks=callbacks, **kwargs)
        count = model.best_iteration_ or model.booster_.current_iteration()
    elif family == "catboost":
        from catboost import CatBoostClassifier

        class Callback:
            def after_iteration(self, info):
                if progress:
                    progress.update(
                        1 if progress.unit == "итерация" else 0,
                        detail=f"CatBoost: итерация {info.iteration}/{iterations}",
                    )
                return max_seconds is None or perf_counter() - started < max_seconds

        model = CatBoostClassifier(
            loss_function="MultiClass",
            eval_metric="TotalF1:average=Macro"
            if weights is None
            else "TotalF1:average=Macro;use_weights=false",
            iterations=iterations,
            random_seed=seed,
            thread_count=threads,
            class_weights=None if weights is None else [weights[i] for i in range(5)],
            task_type="CPU",
            boosting_type="Plain",
            nan_mode="Min",
            allow_writing_files=False,
            verbose=False,
            **params,
        )
        kwargs = {"callbacks": [Callback()]}
        if valid is not None:
            kwargs.update(eval_set=valid, use_best_model=max_seconds is None)
            if max_seconds is None:
                kwargs["early_stopping_rounds"] = patience
        model.fit(x, y, **kwargs)
        count = model.tree_count_
    else:
        raise ValueError(f"Unsupported tree family: {family}")
    if not np.array_equal(model.classes_, np.arange(5)):
        raise ValueError("Expected classes W, N1, N2, N3, REM")
    return model, int(count), perf_counter() - started


def save_model(model, path: Path, family: str) -> None:
    if family == "lightgbm":
        model.booster_.save_model(str(path))
    else:
        model.save_model(str(path))


def load_model(path: Path, family: str):
    if family == "lightgbm":
        import lightgbm as lgb

        return lgb.Booster(model_file=str(path))
    from catboost import CatBoostClassifier

    model = CatBoostClassifier()
    model.load_model(str(path))
    return model


def predict(model, x, family: str, threads: int) -> np.ndarray:
    prob = (
        model.predict(x, num_threads=threads)
        if family == "lightgbm" and not hasattr(model, "predict_proba")
        else (
            model.predict_proba(x, num_threads=threads)
            if family == "lightgbm"
            else model.predict_proba(x, thread_count=threads)
        )
    )
    return np.asarray(prob, dtype=np.float32)
