"""Multinomial Logistic Regression с весами, рассчитанными внутри train-fold."""

import numpy as np
from sklearn.linear_model import LogisticRegression


def class_weights(y: np.ndarray, mode: str) -> dict[int, float] | None:
    if mode == "none":
        return None
    exponent = {"sqrt": 0.5, "balanced": 1.0}[mode]
    counts = np.bincount(y, minlength=5)
    if np.any(counts == 0):
        raise ValueError("Every training fold must contain all five classes")
    p = counts / counts.sum()
    weights = p ** (-exponent)
    weights /= np.dot(p, weights)
    return dict(enumerate(weights.tolist()))


def make_model(c: float, y: np.ndarray, config: dict, seed: int) -> LogisticRegression:
    return LogisticRegression(
        C=c,
        solver=config["solver"],
        l1_ratio=config["l1_ratio"],
        class_weight=class_weights(y, config["class_weight"]),
        max_iter=config["max_iter"],
        tol=config["tol"],
        random_state=seed,
    )
