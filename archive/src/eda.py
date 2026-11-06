"""Воспроизводимые сводки EDA. Статистики сигналов вычисляются только train/val."""

import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy.signal import welch
from tqdm.auto import tqdm

from src.data import CHANNELS, align_labels, load_labels, load_record


def contiguous_runs(indices: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Длины одноимённых стадий; разрыв сетки всегда завершает серию."""
    boundaries = np.r_[
        0,
        np.flatnonzero((np.diff(indices) != 1) | (np.diff(labels) != 0)) + 1,
        len(indices),
    ]
    return np.column_stack((labels[boundaries[:-1]], np.diff(boundaries)))


def audit_dataset(
    root: Path, *, seed: int = 42, spectra_per_class_record: int = 3
) -> dict:
    """Полный проход train/val; test проверяется только по метаданным и заголовку x.

    Welch: 4 s Hann, overlap 50%, density µV²/Hz. Спектры сохраняются как
    среднее выборочных эпох внутри записи и класса, а не как спектр всей базы.
    """
    import zipfile

    labels = load_labels(root)
    manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
    if manifest["dataset_revision"] != "full_psg_v1":
        raise ValueError("Unsupported dataset revision")
    entries = {item["path"]: item for item in manifest["files"]}
    for filename in (
        "labels.csv",
        "sample_submission.csv",
        "team_info.csv",
        "data_description.md",
    ):
        content = (root / filename).read_bytes()
        if hashlib.sha256(content).hexdigest() != entries[filename]["sha256"]:
            raise ValueError(f"Manifest checksum mismatch: {filename}")
    rng = np.random.default_rng(seed)
    records, quality, runs, spectra, examples = [], [], [], [], {}
    transitions = {
        split: np.zeros((5, 5), dtype=np.int64) for split in ("train", "val")
    }
    seen, test_ids = set(), []
    expected_files = {
        name for name in entries if name.startswith("epochs/") and name.endswith(".npz")
    }
    actual_files = set()
    for split in ("train", "val", "test"):
        for path in tqdm(sorted((root / "epochs" / split).glob("*.npz")), desc=split):
            relative = path.relative_to(root).as_posix()
            actual_files.add(relative)
            if (
                relative not in entries
                or path.stat().st_size != entries[relative]["bytes"]
            ):
                raise ValueError(f"File size/manifest mismatch: {relative}")
            rec = load_record(path, signals=split != "test")
            ids, idx = rec["ids"], rec["epoch_index"]
            if seen.intersection(ids.tolist()):
                raise ValueError("IDs overlap between recordings")
            seen.update(ids.tolist())
            gap = np.diff(idx) != 1
            records.append(
                {
                    "split": split,
                    "record_id": path.stem,
                    "epochs": len(ids),
                    "retained_hours": len(ids) / 120,
                    "span_hours": (idx[-1] - idx[0] + 1) / 120,
                    "gaps": int(gap.sum()),
                    "missing_inside": int(np.sum(np.diff(idx) - 1)),
                    "segments": int(gap.sum()) + 1,
                }
            )
            if split == "test":
                with zipfile.ZipFile(path) as archive, archive.open("x.npy") as stream:
                    version = np.lib.format.read_magic(stream)
                    if version == (1, 0):
                        shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                    elif version == (2, 0):
                        shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
                    else:
                        raise ValueError(f"Unsupported NPY header version: {version}")
                if shape != (len(ids), 3000, 3) or dtype != np.float32:
                    raise ValueError(f"Invalid test signal header: {path}")
                test_ids.extend(ids.tolist())
                continue
            y = align_labels(rec, labels, path.stem, split)
            x = rec["x"]
            rms = np.sqrt(np.mean(np.square(x, dtype=np.float64), axis=1))
            std = np.std(x, axis=1, dtype=np.float64)
            peaks = np.max(np.abs(x), axis=1)
            repeated = np.mean(np.diff(x, axis=1) == 0, axis=1)
            for c, channel in enumerate(CHANNELS):
                quality.append(
                    pl.DataFrame(
                        {
                            "id": ids,
                            "split": split,
                            "record_id": path.stem,
                            "label": y,
                            "channel": channel,
                            "rms_uv": rms[:, c],
                            "std_uv": std[:, c],
                            "max_abs_uv": peaks[:, c],
                            "constant": std[:, c] == 0,
                            "repeated_fraction": repeated[:, c],
                        }
                    )
                )
            np.add.at(transitions[split], (y[:-1][~gap], y[1:][~gap]), 1)
            r = contiguous_runs(idx, y)
            runs.append(
                pl.DataFrame(
                    {
                        "split": split,
                        "record_id": path.stem,
                        "label": r[:, 0],
                        "epochs": r[:, 1],
                    }
                )
            )
            for label in range(5):
                candidates = np.flatnonzero(y == label)
                if not len(candidates):
                    continue
                selected = rng.choice(
                    candidates,
                    min(len(candidates), spectra_per_class_record),
                    replace=False,
                )
                f, p = welch(
                    x[selected],
                    fs=100,
                    window="hann",
                    nperseg=400,
                    noverlap=200,
                    axis=1,
                    scaling="density",
                )
                spectra.append(
                    {
                        "split": split,
                        "record_id": path.stem,
                        "label": label,
                        "n": len(selected),
                        "psd": p.mean(axis=0),
                    }
                )
                if split == "train" and label not in examples:
                    i = int(selected[0])
                    examples[label] = {"id": str(ids[i]), "x": x[i].copy()}
    if actual_files != expected_files:
        raise ValueError(f"Missing/extra NPZ: {actual_files ^ expected_files}")
    submission = pl.read_csv(root / "sample_submission.csv")
    if submission.columns != ["id", "label"] or submission["id"].to_list() != test_ids:
        raise ValueError("Submission order/IDs mismatch")
    if set(labels["id"]) != seen - set(test_ids):
        raise ValueError("Label coverage mismatch")
    return {
        "labels": labels,
        "records": pl.DataFrame(records),
        "quality": pl.concat(quality),
        "runs": pl.concat(runs),
        "transitions": transitions,
        "frequency": f,
        "spectra": spectra,
        "examples": examples,
        "seed": seed,
    }


BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 12),
    "sigma": (12, 16),
    "beta": (16, 30),
}


def bandpowers(frequency: np.ndarray, psd: np.ndarray) -> dict[str, np.ndarray]:
    """Интеграл PSD по полуинтервалам; относительная мощность к 0.5–40 Hz.

    Маска включается в полный массив перед trapezoid, поэтому сумма непересекающихся
    полос аддитивна на сетке FFT; границы распределяются половинными трапециями.
    Ось частоты предпоследняя, последняя — канал.
    """
    total = np.trapezoid(
        psd * ((frequency >= 0.5) & (frequency < 40))[..., None], frequency, axis=-2
    )
    result = {}
    for name, (low, high) in BANDS.items():
        power = np.trapezoid(
            psd * ((frequency >= low) & (frequency < high))[..., None],
            frequency,
            axis=-2,
        )
        result[name] = power
        result[f"relative_{name}"] = np.divide(
            power, total, out=np.zeros_like(power), where=total > 0
        )
    return result


def gap_details(
    indices: np.ndarray, labels: np.ndarray
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Точные пропуски и непрерывные сегменты на исходной сетке, без заполнения."""
    stops = np.flatnonzero(np.diff(indices) != 1)
    gaps = pl.DataFrame(
        {
            "before_index": indices[stops],
            "after_index": indices[stops + 1],
            "missing_epochs": indices[stops + 1] - indices[stops] - 1,
            "start_hours": (indices[stops] + 1) / 120,
            "before_stage": labels[stops],
            "after_stage": labels[stops + 1],
        }
    )
    boundaries = np.r_[0, stops + 1, len(indices)]
    segments = pl.DataFrame(
        {
            "first_index": indices[boundaries[:-1]],
            "last_index": indices[boundaries[1:] - 1],
            "epochs": np.diff(boundaries),
        }
    )
    return gaps, segments


def sleep_structure(labels: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Крайние W удаляются только в диагностической копии; без sleep epochs — пустое ядро."""
    enriched = labels.with_columns(
        pl.when(pl.col("label") != 0)
        .then(pl.col("epoch_index"))
        .min()
        .over("record_id")
        .alias("first_sleep"),
        pl.when(pl.col("label") != 0)
        .then(pl.col("epoch_index"))
        .max()
        .over("record_id")
        .alias("last_sleep"),
        pl.col("epoch_index").max().over("record_id").alias("last_index"),
    ).with_columns(
        (
            (pl.col("epoch_index") >= pl.col("first_sleep"))
            & (pl.col("epoch_index") <= pl.col("last_sleep"))
        )
        .fill_null(False)
        .alias("in_sleep_span"),
        (pl.col("epoch_index") / (pl.col("last_index") + 1) * 20)
        .floor()
        .cast(pl.Int64)
        .alias("position_bin"),
    )
    fraction = enriched.group_by("split", "subject_id", "record_id").agg(
        pl.len().alias("epochs"),
        (pl.col("label") == 0).mean().alias("wake_fraction"),
        pl.col("in_sleep_span").sum().alias("sleep_span_epochs"),
    )
    profile = enriched.group_by("split", "subject_id", "position_bin").agg(
        (pl.col("label") == 0).mean().alias("wake_fraction"), pl.len().alias("epochs")
    )
    counts = (
        pl.concat(
            [
                enriched.with_columns(pl.lit("full").alias("scope")),
                enriched.filter("in_sleep_span").with_columns(
                    pl.lit("sleep_span").alias("scope")
                ),
            ]
        )
        .group_by("scope", "split", "label")
        .len()
        .with_columns(
            (pl.col("len") / pl.col("len").sum().over("scope", "split")).alias(
                "fraction"
            )
        )
    )
    regions = (
        enriched.with_columns(
            pl.when(pl.col("position_bin") < 4)
            .then(pl.lit("first20%"))
            .when(pl.col("position_bin") >= 16)
            .then(pl.lit("last20%"))
            .otherwise(pl.lit("middle60%"))
            .alias("region")
        )
        .group_by("split", "subject_id", "region")
        .agg((pl.col("label") == 0).mean().alias("wake_fraction"))
    )
    return {
        "record_wake": fraction,
        "position": profile,
        "class_scope": counts,
        "regions": regions,
    }


def signal_followup(
    root: Path, *, seeds: tuple[int, int] = (42, 137), per_stage: int = 30
) -> dict:
    """Все REM и две выборки остальных стадий; статистики людей внутри стадии.

    RMS/IQR/p95_abs: эпохи первой выборки, для REM — все эпохи. Peak EOG и частота
    превышений вычисляются по всем эпохам всех train/val. Пороги 500/800 µV только
    диагностические, не правила исключения. Ничего не вычисляется по test.
    """
    labels = load_labels(root)
    features, psds, eog, excerpts = [], [], [], {}
    random = [np.random.default_rng(seed) for seed in seeds]
    for meta in tqdm(
        labels.select("split", "record_id", "subject_id")
        .unique()
        .sort("split", "record_id")
        .iter_rows(named=True),
        total=128,
        desc="Signal follow-up",
    ):
        path = root / "epochs" / meta["split"] / f"{meta['record_id']}.npz"
        rec = load_record(path)
        y = align_labels(rec, labels, meta["record_id"], meta["split"])
        x = rec["x"]
        peak = np.max(np.abs(x[:, :, 2]), axis=1)
        for label in range(5):
            candidates = np.flatnonzero(y == label)
            if not len(candidates):
                continue
            eog.append(
                {
                    **meta,
                    "label": label,
                    "epochs": len(candidates),
                    "over500": int((peak[candidates] > 500).sum()),
                    "over800": int((peak[candidates] > 800).sum()),
                    "max_uv": float(peak[candidates].max()),
                }
            )
            for sample, rng in enumerate(random):
                if label == 4 and sample == 1:
                    continue  # All REM: no random sampling to repeat.
                selected = (
                    candidates
                    if label == 4
                    else rng.choice(
                        candidates, min(per_stage, len(candidates)), replace=False
                    )
                )
                freq, p = welch(
                    x[selected],
                    fs=100,
                    nperseg=400,
                    noverlap=200,
                    window="hann",
                    axis=1,
                    scaling="density",
                )
                psds.append(
                    {
                        **meta,
                        "label": label,
                        "sample": sample,
                        "n": len(selected),
                        "mean": p.mean(axis=0),
                        "median": np.median(p, axis=0),
                    }
                )
                if sample:
                    continue
                q = np.percentile(x[selected], [25, 75], axis=1)
                bands = bandpowers(freq, p)
                for c, channel in enumerate(CHANNELS):
                    rows = {
                        "id": rec["ids"][selected],
                        **meta,
                        "label": label,
                        "channel": channel,
                        "rms_uv": np.sqrt(
                            np.mean(
                                np.square(x[selected, :, c], dtype=np.float64), axis=1
                            )
                        ),
                        "iqr_uv": q[1, :, c] - q[0, :, c],
                        "p95_abs_uv": np.percentile(
                            np.abs(x[selected, :, c]), 95, axis=1
                        ),
                    }
                    rows.update({name: value[:, c] for name, value in bands.items()})
                    features.append(pl.DataFrame(rows))
        if meta["record_id"] in ("SC4242", "SC4592"):
            extreme = int(np.argmax(peak))
            typical = int(np.argmin(np.abs(peak - np.median(peak))))
            excerpts[meta["record_id"]] = {
                "x": x[[typical, extreme]].copy(),
                "ids": rec["ids"][[typical, extreme]],
                "labels": y[[typical, extreme]],
                "peaks": peak[[typical, extreme]],
            }
    metrics = pl.concat(features)
    columns = [
        "rms_uv",
        "iqr_uv",
        "p95_abs_uv",
        *BANDS,
        *[f"relative_{b}" for b in BANDS],
    ]
    subjects = metrics.group_by("split", "subject_id", "label", "channel").agg(
        pl.len().alias("sampled_epochs"), *[pl.col(c).median() for c in columns]
    )
    return {
        "features": metrics,
        "subjects": subjects,
        "psds": psds,
        "eog": pl.DataFrame(eog),
        "excerpts": excerpts,
        "frequency": freq,
        "seeds": seeds,
        "per_stage": per_stage,
    }


def inspect_spectral_records(root: Path) -> dict:
    """Адресная проверка REM-хвостов и устойчивых узких пиков, без удаления эпох."""
    labels = load_labels(root)
    details, examples, peak_rows = [], {}, []
    for record_id in ("SC4261", "SC4172", "SC4272", "SC4501"):
        split = "train" if record_id == "SC4501" else "val"
        rec = load_record(root / "epochs" / split / f"{record_id}.npz")
        y = align_labels(rec, labels, record_id, split)
        for stage in range(5):
            ids = np.flatnonzero(y == stage)
            if not len(ids):
                continue
            # All REM in anomalous records; evenly spaced <=100 epochs for peak controls.
            chosen = (
                ids
                if stage == 4
                else ids[np.linspace(0, len(ids) - 1, min(100, len(ids)), dtype=int)]
            )
            f, p = welch(rec["x"][chosen], fs=100, nperseg=400, noverlap=200, axis=1)
            median = np.median(p, axis=0)
            hz = 27 if record_id == "SC4501" else 26.75
            i = int(np.argmin(np.abs(f - hz)))
            for c, channel in enumerate(CHANNELS):
                baseline = np.median(
                    np.r_[median[i - 4 : i, c], median[i + 1 : i + 5, c]]
                )
                peak_rows.append(
                    {
                        "record_id": record_id,
                        "split": split,
                        "label": stage,
                        "channel": channel,
                        "epochs": len(chosen),
                        "peak_hz": hz,
                        "local_ratio": float(median[i, c] / max(baseline, 1e-12)),
                    }
                )
            if record_id in ("SC4261", "SC4172") and stage == 4:
                band = (f >= 10) & (f < 40)
                power = np.trapezoid(p[:, band, 1], f[band], axis=1)
                n = max(1, int(np.ceil(len(power) * 0.05)))
                details.append(
                    {
                        "record_id": record_id,
                        "rem_epochs": len(power),
                        "mean_power10_40": float(power.mean()),
                        "median_power10_40": float(np.median(power)),
                        "top5_percent_epochs_power_share": float(
                            np.sort(power)[-n:].sum() / power.sum()
                        ),
                    }
                )
                selected = [
                    int(np.argmin(np.abs(power - np.median(power)))),
                    int(np.argmax(power)),
                ]
                examples[record_id] = {
                    "x": rec["x"][chosen[selected]].copy(),
                    "ids": rec["ids"][chosen[selected]],
                    "power": power[selected],
                    "psd": p[selected, :, 1],
                }
    return {
        "details": pl.DataFrame(details),
        "peak_controls": pl.DataFrame(peak_rows),
        "examples": examples,
        "frequency": f,
    }
