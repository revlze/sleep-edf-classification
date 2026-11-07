"""Обучаемые преобразования признаков, не исходных EEG/EOG сигналов."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, RobustScaler


class LogPower(BaseEstimator, TransformerMixin):
    """Логарифм только мощностей и отношений; набор колонок задан схемой."""

    def __init__(self, indices=(), epsilon=1e-12):
        self.indices = indices
        self.epsilon = epsilon

    def fit(self, x, y=None):
        self.n_features_in_ = x.shape[1]
        return self

    def transform(self, x):
        result = np.array(x, copy=True)
        if self.indices:
            result[:, self.indices] = np.log10(
                np.maximum(result[:, self.indices], self.epsilon)
            )
        return result


def make_preprocessor(names: list[str], config: dict, seed: int) -> Pipeline:
    steps = []
    if config["transform"] == "log_robust":
        indices = tuple(
            i for i, n in enumerate(names) if "total_power" in n or "ratio_" in n
        )
        steps.append(("log_power", LogPower(indices, config["epsilon"])))
    elif config["transform"] != "quantile":
        raise ValueError("Unknown feature transform")
    steps.append(("impute", SimpleImputer(strategy="median", keep_empty_features=True)))
    if config["transform"] == "quantile":
        scaler = QuantileTransformer(
            n_quantiles=config["n_quantiles"],
            output_distribution="uniform",
            subsample=config["quantile_subsample"],
            random_state=seed,
        )
    else:
        scaler = RobustScaler(quantile_range=tuple(config["robust_quantiles"]))
    steps.append(("scale", scaler))
    return Pipeline(steps)
