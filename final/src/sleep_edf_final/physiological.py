"""106 признаков по семействам Van Der Donckt, без фильтрации и EMG.

Это адаптация, а не побитовая репликация tsfresh/YASA. FFT moments используют
индексы rFFT и математически центральные моменты (Pearson kurtosis). Fourier
entropy сохраняет отдельный Welch-256 рецепт tsfresh. Bandpower — Welch-500,
Hamming, median, Simpson; частотные полосы включают обе границы как в YASA.
"""

import warnings

import antropy as ant
import numpy as np
from scipy.integrate import simpson
from scipy.signal import welch
from scipy.stats import iqr, kurtosis, skew

BANDS = (
    (0.4, 1, "sdelta"),
    (1, 4, "fdelta"),
    (4, 8, "theta"),
    (8, 12, "alpha"),
    (12, 16, "sigma"),
    (16, 30, "beta"),
)
EXTRACTOR_VERSION = "linear106_v1"


def feature_names() -> list[str]:
    names = []
    common = [
        "std",
        "iqr",
        "skew",
        "kurtosis",
        "zero_crossings",
        "hjorth_mobility",
        "hjorth_complexity",
        "higuchi_fd",
        "petrosian_fd",
        "permutation_entropy",
    ]
    common += [f"binned_entropy_{b}" for b in (5, 10, 30, 60)]
    common += [f"fft_{m}" for m in ("centroid", "variance", "skew", "kurtosis")]
    common += [f"fourier_entropy_{b}" for b in (2, 3, 5, 10, 30, 60, 100)]
    common += ["total_power"] + [f"relative_{b[2]}" for b in BANDS]
    for c in range(3):
        names += [f"ch{c}__{n}" for n in common]
        if c < 2:
            names += [
                f"ch{c}__{n}"
                for n in (
                    "sum_fdelta_theta",
                    "ratio_alpha_theta",
                    "ratio_delta_beta",
                    "ratio_delta_sigma",
                    "ratio_delta_theta",
                )
            ]
    return names


def histogram_entropy(x: np.ndarray, bins: int) -> float:
    if not np.isfinite(x).all():
        return np.nan
    p = np.histogram(x, bins=bins)[0] / len(x)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def extract_epoch_features(x: np.ndarray, spec: dict) -> np.ndarray:
    """Пакет (N,T,3) в µV -> (N,106); вырожденные признаки остаются NaN."""
    if x.ndim != 3 or x.shape[2] != 3 or not np.isfinite(x).all():
        raise ValueError("Expected finite (N,T,3) input in microvolts")
    outputs = []
    eps = spec["epsilon"]
    for c in range(3):
        a = np.ascontiguousarray(x[:, :, c], dtype=np.float64)
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)
            mobility, complexity = ant.hjorth_params(a, axis=1)
            values = [
                a.std(axis=1),
                iqr(a, axis=1),
                skew(a, axis=1),
                kurtosis(a, axis=1),
                ant.num_zerocross(a, axis=1),
                mobility,
                complexity,
            ]
            values += [
                np.array([ant.higuchi_fd(row, kmax=spec["higuchi_kmax"]) for row in a]),
                ant.petrosian_fd(a, axis=1),
                np.array(
                    [
                        ant.perm_entropy(
                            row,
                            order=spec["permutation_order"],
                            delay=1,
                            normalize=False,
                        )
                        for row in a
                    ]
                ),
            ]
            values += [
                np.array([histogram_entropy(row, b) for row in a])
                for b in (5, 10, 30, 60)
            ]
            fft = np.abs(np.fft.rfft(a, axis=1))
            bins = np.arange(fft.shape[1])
            total = fft.sum(axis=1)
            centroid = np.sum(fft * bins, axis=1) / total
            centered = bins[None, :] - centroid[:, None]
            variance = np.sum(fft * centered**2, axis=1) / total
            third = np.sum(fft * centered**3, axis=1) / total / variance**1.5
            fourth = np.sum(fft * centered**4, axis=1) / total / variance**2
            third[variance < 0.5] = np.nan
            fourth[variance < 0.5] = np.nan
            values += [centroid, variance, third, fourth]
            _, small = welch(
                a, nperseg=256, noverlap=128, window="hann", average="mean", axis=1
            )
            small = small / small.max(axis=1, keepdims=True)
            values += [
                np.array([histogram_entropy(row, b) for row in small])
                for b in (2, 3, 5, 10, 30, 60, 100)
            ]
            f, p = welch(
                a,
                fs=100,
                nperseg=spec["welch_nperseg"],
                noverlap=spec["welch_noverlap"],
                window=spec["welch_window"],
                average=spec["welch_average"],
                detrend="constant",
                scaling="density",
                axis=1,
            )
            mask = (f >= 0.4) & (f <= 30)
            power = simpson(p[:, mask], x=f[mask], axis=1)
            bands = np.stack(
                [
                    simpson(
                        p[:, (f >= lo) & (f <= hi)], x=f[(f >= lo) & (f <= hi)], axis=1
                    )
                    for lo, hi, _ in BANDS
                ],
                axis=1,
            )
            relative = bands / np.where(power > eps, power, np.nan)[:, None]
            values += [power, *relative.T]
            if c < 2:
                delta = relative[:, 0] + relative[:, 1]

                def ratio(num, den):
                    return num / np.where(den > eps, den, np.nan)

                values += [
                    relative[:, 1] + relative[:, 2],
                    ratio(relative[:, 3], relative[:, 2]),
                    ratio(delta, relative[:, 5]),
                    ratio(delta, relative[:, 4]),
                    ratio(delta, relative[:, 2]),
                ]
        outputs.append(np.column_stack(values))
    result = np.column_stack(outputs).astype(np.float32)
    result[~np.isfinite(result)] = np.nan
    if result.shape[1] != len(feature_names()):
        raise RuntimeError("Feature schema mismatch")
    return result
