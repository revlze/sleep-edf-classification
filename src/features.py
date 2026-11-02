"""106 признаков по семействам Van Der Donckt, без фильтрации и EMG.

Это адаптация, а не побитовая репликация tsfresh/YASA. FFT moments используют
индексы rFFT и математически центральные моменты (Pearson kurtosis). Fourier
entropy сохраняет отдельный Welch-256 рецепт tsfresh. Bandpower — Welch-500,
Hamming, median, Simpson; частотные полосы включают обе границы как в YASA.
"""

import warnings
from pathlib import Path

import antropy as ant
import numpy as np
import polars as pl
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


def extract_record_windows(
    x: np.ndarray,
    indices: np.ndarray,
    selected: np.ndarray,
    spec: dict,
    *,
    long: bool,
    progress=None,
    record_id: str = "",
) -> dict[str, np.ndarray]:
    """Каждое уникальное 60 s окно вычисляется один раз для left60/right60."""
    from src.datasets import window_indices

    plan = (
        {"left60": (1, 0), "right60": (0, 1), "center90": (1, 1)}
        if long
        else {"base": (0, 0)}
    )
    values = {
        name: np.full((len(selected), 106), np.nan, dtype=np.float32) for name in plan
    }
    for width in sorted({left + right + 1 for left, right in plan.values()}):
        mappings = {}
        for name, (left, right) in plan.items():
            if left + right + 1 != width:
                continue
            rows, valid = window_indices(indices, left, right)
            positions = np.flatnonzero(valid[selected])
            mappings[name] = (positions, rows[selected[positions], 0])
        starts = np.unique(
            np.concatenate([row_starts for _, row_starts in mappings.values()])
        )
        features = np.empty((len(starts), 106), dtype=np.float32)
        for begin in range(0, len(starts), spec["batch_size"]):
            end = min(begin + spec["batch_size"], len(starts))
            if progress is not None:
                progress.update(
                    detail=f"{record_id}: окно {width * 30} s, {begin}/{len(starts)}"
                )
            batch = np.concatenate(
                [x[starts[begin:end] + j] for j in range(width)], axis=1
            )
            features[begin:end] = extract_epoch_features(batch, spec)
        for name, (positions, row_starts) in mappings.items():
            values[name][positions] = features[np.searchsorted(starts, row_starts)]
    return values


def build_feature_table(
    root: Path,
    cache_root: Path,
    labels: pl.DataFrame,
    spec: dict,
    mode: str,
    *,
    smoke: bool = False,
    test: bool = False,
) -> tuple[np.ndarray, pl.DataFrame, list[str], dict]:
    """Извлечь по записям; кэш 30 s общий, длинные окна вычисляются по требованию."""
    import hashlib
    import json
    from importlib.metadata import version
    from time import perf_counter

    from src.data import align_labels, load_record
    from src.datasets import add_context
    from src.progress import Progress
    from src.storage import (
        atomic_path,
        canonical_fingerprint,
        file_lock,
        load_or_create_npz,
    )

    started = perf_counter()
    root = Path(root)
    cache_root = Path(cache_root)
    if not labels["id"].str.starts_with("full_").all():
        raise ValueError("Expected full_psg_v1 epoch IDs with the full_ prefix")
    source_hash = hashlib.sha256(
        Path(__file__).read_bytes()
        + Path(__file__).with_name("datasets.py").read_bytes()
    ).hexdigest()
    source_hash = canonical_fingerprint(source_hash, "features")
    signature = {
        "version": EXTRACTOR_VERSION,
        "source": source_hash,
        "spec": spec,
        "revision": "full_psg_v1",
        "labels_sha256": hashlib.sha256((root / "labels.csv").read_bytes()).hexdigest(),
        "packages": {n: version(n) for n in ("numpy", "scipy", "antropy")},
        "smoke": smoke,
    }
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[
        :20
    ]
    folder = cache_root / key
    folder.mkdir(parents=True, exist_ok=True)
    definition = folder / "definition.json"
    if not definition.exists():
        with file_lock(folder / "definition.lock", wait=True):
            if not definition.exists():
                with atomic_path(definition) as temporary:
                    temporary.write_text(
                        json.dumps(signature, indent=2), encoding="utf-8"
                    )
    if json.loads(definition.read_text(encoding="utf-8")) != signature:
        raise ValueError("Feature cache definition mismatch")
    if test:
        if smoke:
            raise ValueError("Test inference cannot use smoke sampling")
        parts = []
        for path in sorted((root / "epochs/test").glob("*.npz")):
            rec = load_record(path, signals=False)
            parts.append(
                pl.DataFrame(
                    {
                        "id": rec["ids"],
                        "label": pl.Series([None] * len(rec["ids"]), dtype=pl.Int64),
                        "split": "test",
                        "record_id": path.stem,
                        "epoch_index": rec["epoch_index"],
                        "subject_id": pl.Series(
                            [None] * len(rec["ids"]), dtype=pl.String
                        ),
                    }
                )
            )
        if not parts:
            raise ValueError("No test recordings")
        labels = pl.concat(parts)
    chosen_labels = labels
    if smoke:
        train = sorted(set(labels.filter(pl.col("split") == "train")["subject_id"]))[:9]
        val = sorted(set(labels.filter(pl.col("split") == "val")["subject_id"]))[:2]
        chosen_labels = labels.filter(pl.col("subject_id").is_in(train + val))
    matrices = []
    metadata = []
    hits = 0
    records = (
        chosen_labels.select("split", "record_id").unique().sort("split", "record_id")
    )
    with Progress("Признаки: все записи", records.height, "зап.") as progress:
        for split, record_id in records.iter_rows():
            progress.update(detail=f"{split}/{record_id}: метаданные")
            path = root / "epochs" / split / f"{record_id}.npz"
            rec = load_record(path, signals=False)
            if not np.char.startswith(rec["ids"], "full_").all():
                raise ValueError(f"Expected full_psg_v1 epoch IDs: {path}")
            y = None if test else align_labels(rec, labels, record_id, split)
            selected = np.arange(len(rec["ids"]))
            if smoke:
                centers = [
                    int(
                        np.flatnonzero(y == stage)[len(np.flatnonzero(y == stage)) // 2]
                    )
                    for stage in range(5)
                    if np.any(y == stage)
                ]
                selected = np.unique(
                    np.concatenate(
                        [np.arange(max(0, i - 5), min(len(y), i + 6)) for i in centers]
                    )
                )
            stat = path.stat()
            stamp = hashlib.sha256(
                f"{split}:{stat.st_mtime_ns}:{stat.st_size}".encode()
            ).hexdigest()[:16]

            signals = None

            def obtain(long=False):
                nonlocal hits, signals
                target = (
                    folder / f"{record_id}_{stamp}_{'long' if long else 'epoch'}.npz"
                )

                def create():
                    nonlocal signals
                    if signals is None:
                        progress.update(detail=f"{record_id}: загрузка сигналов")
                        signals = load_record(path)["x"]
                    values = extract_record_windows(
                        signals,
                        rec["epoch_index"],
                        selected,
                        spec,
                        long=long,
                        progress=progress,
                        record_id=record_id,
                    )
                    progress.update(detail=f"{record_id}: сохранение кэша")
                    return {"ids": rec["ids"][selected], **values}

                progress.update(detail=f"{record_id}: кэш / ожидание вычисления")
                values, hit = load_or_create_npz(target, create)
                if not np.array_equal(values.pop("ids"), rec["ids"][selected]):
                    raise ValueError("Cache IDs mismatch")
                hits += int(hit)
                return values

            base = obtain()["base"]
            long_values = obtain(True) if mode == "multires" else None
            # Neighbor availability is evaluated on the retained grid (also in smoke runs).
            matrix, columns = add_context(
                base, rec["epoch_index"][selected], feature_names(), mode, long_values
            )
            matrices.append(matrix)
            order = pl.DataFrame({"id": rec["ids"][selected]})
            metadata.append(
                order.join(
                    labels, on="id", how="left", validate="1:1", maintain_order="left"
                )
            )
            progress.update(1)
    return (
        np.concatenate(matrices),
        pl.concat(metadata),
        columns,
        {
            "key": key,
            "cache_hits": hits,
            "seconds": perf_counter() - started,
            "definition": signature,
        },
    )
