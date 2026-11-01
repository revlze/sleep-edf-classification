"""Загрузка поставки Sleep-EDF без предположений о порядке строк меток."""

from pathlib import Path

import numpy as np
import polars as pl

CHANNELS = ("EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal")
STAGES = ("W", "N1", "N2", "N3", "REM")


def load_labels(root: Path) -> pl.DataFrame:
    """Прочитать метки и проверить схему, классы и разделение людей."""
    labels = pl.read_csv(root / "labels.csv")
    required = {"id", "label", "split", "record_id", "epoch_index", "subject_id"}
    if not required.issubset(labels.columns):
        raise ValueError(f"Missing label columns: {required - set(labels.columns)}")
    if labels.select(pl.any_horizontal(pl.all().is_null()).any()).item():
        raise ValueError("Null values in labels")
    if labels["id"].n_unique() != labels.height:
        raise ValueError("Duplicate label IDs")
    if labels["label"].dtype not in (pl.Int64, pl.Int32) or not set(
        labels["label"]
    ).issubset(range(5)):
        raise ValueError("Labels must be integer classes 0..4")
    if set(labels["split"]) != {"train", "val"}:
        raise ValueError("Expected train and val labels")
    if (
        labels.group_by("subject_id").agg(pl.col("split").n_unique())["split"].max()
        != 1
    ):
        raise ValueError("Subject overlap across splits")
    if (
        labels.group_by("record_id")
        .agg(pl.col("subject_id").n_unique())["subject_id"]
        .max()
        != 1
    ):
        raise ValueError("Record maps to multiple subjects")
    return labels


def load_record(path: Path, *, signals: bool = True) -> dict[str, np.ndarray]:
    """Прочитать запись; signals=False не распаковывает массив x (например, test)."""
    with np.load(path, allow_pickle=False) as archive:
        required = {"x", "ids", "epoch_index", "fs", "channel_names"}
        if set(archive.files) != required:
            raise ValueError(f"Unexpected keys: {path}")
        record = {key: archive[key] for key in required if key != "x"}
        if signals:
            record["x"] = archive["x"]
    ids, indices = record["ids"], record["epoch_index"]
    if (
        ids.ndim != 1
        or indices.shape != ids.shape
        or not np.issubdtype(indices.dtype, np.integer)
    ):
        raise ValueError(f"Invalid metadata arrays: {path}")
    if (
        len(ids) == 0
        or len(set(ids.tolist())) != len(ids)
        or np.any(np.diff(indices) <= 0)
        or indices[0] < 0
    ):
        raise ValueError(f"Duplicate IDs or invalid epoch order: {path}")
    if (
        np.asarray(record["fs"]).item() != 100
        or tuple(record["channel_names"].tolist()) != CHANNELS
    ):
        raise ValueError(f"Unexpected sampling rate or channels: {path}")
    if signals:
        x = record["x"]
        if x.shape != (len(ids), 3000, 3) or x.dtype != np.float32:
            raise ValueError(f"Unexpected signal shape or dtype: {path}")
        if not np.isfinite(x).all():
            raise ValueError(f"Non-finite signal: {path}")
    return record


def align_labels(
    record: dict[str, np.ndarray], labels: pl.DataFrame, record_id: str, split: str
) -> np.ndarray:
    """Соединить по ID и проверить запись, split и исходную временную сетку."""
    order = pl.DataFrame({"id": record["ids"], "expected_index": record["epoch_index"]})
    joined = order.join(
        labels, on="id", how="left", maintain_order="left", validate="1:1"
    )
    if (
        joined["label"].null_count()
        or joined.filter(
            (pl.col("record_id") != record_id)
            | (pl.col("split") != split)
            | (pl.col("epoch_index") != pl.col("expected_index"))
        ).height
    ):
        raise ValueError(f"Label metadata mismatch: {record_id}")
    if labels.filter(
        (pl.col("record_id") == record_id) & (pl.col("split") == split)
    ).height != len(order):
        raise ValueError(f"Extra/missing labels: {record_id}")
    return joined["label"].to_numpy()
