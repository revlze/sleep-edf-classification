"""Чтение проверенного кэша без обращения к сигналам и без пересчёта признаков."""

import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import polars as pl

from src.datasets import add_context
from src.features import EXTRACTOR_VERSION, feature_names
from src.progress import Progress
from src.storage import canonical_fingerprint


def read_cached_features(
    cache_root: Path,
    labels: pl.DataFrame,
    spec: dict,
    context: str,
    *,
    smoke: bool = False,
    test: bool = False,
    data_root: Path | None = None,
) -> tuple[np.ndarray, pl.DataFrame, list[str], dict]:
    """ID и схема обязательны; mtime исходных NPZ может меняться при переносе."""
    from src import features

    source = hashlib.sha256(
        Path(features.__file__).read_bytes()
        + Path(features.__file__).with_name("datasets.py").read_bytes()
    ).hexdigest()
    # Hash the original CSV separately in the caller; Polars reserialization differs.
    expected_hash = spec["labels_sha256"]
    extractor = {k: v for k, v in spec.items() if k != "labels_sha256"}
    matches = []
    for path in sorted(cache_root.glob("*/definition.json")):
        definition = json.loads(path.read_text(encoding="utf-8"))
        if (
            definition.get("version") == EXTRACTOR_VERSION
            and definition.get("revision") == "full_psg_v1"
            and definition.get("spec") == extractor
            and definition.get("labels_sha256") == expected_hash
            and definition.get("smoke") == smoke
            and definition.get("packages")
            == {n: version(n) for n in ("numpy", "scipy", "antropy")}
            and canonical_fingerprint(definition["source"], "features")
            == canonical_fingerprint(source, "features")
        ):
            matches.append((path.parent, definition))
    if len(matches) != 1:
        raise ValueError(
            f"Expected one compatible {'smoke' if smoke else 'full'} cache in {cache_root}; found {len(matches)}. No features were extracted."
        )
    folder, definition = matches[0]
    if smoke:
        subjects = []
        for split, count in (("train", 9), ("val", 2)):
            subjects.extend(
                sorted(set(labels.filter(pl.col("split") == split)["subject_id"]))[
                    :count
                ]
            )
        labels = labels.filter(pl.col("subject_id").is_in(subjects))
    if test:
        # Cache IDs provide test metadata; no labels or common test statistics are read.
        records = sorted({p.name.rsplit("_", 2)[0] for p in folder.glob("*_epoch.npz")})
        records = [r for r in records if r not in set(labels["record_id"])]
    else:
        records = sorted(set(labels["record_id"]))
    matrices, metadata, objects = [], [], []

    def read(record: str, suffix: str):
        candidates = list(folder.glob(f"{record}_*_{suffix}.npz"))
        if len(candidates) != 1:
            raise ValueError(
                f"Cache {record}/{suffix}: expected one object, found {len(candidates)}; extraction disabled"
            )
        path = candidates[0]
        with np.load(path, allow_pickle=False) as z:
            result = {key: z[key] for key in z.files}
        objects.append(
            {"name": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        )
        return result

    with Progress("Готовый кэш: все записи", len(records), "зап.") as progress:
        for record in records:
            progress.update(detail=record)
            base = read(record, "epoch")
            ids = base.pop("ids")
            if (
                base["base"].dtype != np.float32
                or len(ids) != len(set(ids.tolist()))
                or base["base"].shape
                != (
                    len(ids),
                    106,
                )
            ):
                raise ValueError(f"Invalid cache IDs/shape: {record}")
            if test:
                from src.data import load_record

                if data_root is None:
                    raise ValueError("Test metadata requires data_root")
                header = load_record(
                    data_root / "epochs/test" / f"{record}.npz", signals=False
                )
                if not np.array_equal(header["ids"], ids):
                    raise ValueError("Test cache/header IDs differ")
                indices = header["epoch_index"]
                meta = pl.DataFrame(
                    {
                        "id": ids,
                        "record_id": record,
                        "epoch_index": indices,
                        "split": "test",
                    }
                )
            else:
                expected = labels.filter(pl.col("record_id") == record)
                if not smoke and set(ids.tolist()) != set(expected["id"]):
                    raise ValueError(f"Cache coverage differs from labels: {record}")
                meta = pl.DataFrame({"id": ids}).join(
                    expected, on="id", how="left", validate="1:1", maintain_order="left"
                )
                if meta["label"].null_count() or meta["record_id"].null_count():
                    raise ValueError(f"Cache IDs are not in labels: {record}")
            if np.any(np.diff(meta["epoch_index"].to_numpy()) <= 0):
                raise ValueError(f"Cache is not in chronological order: {record}")
            long = None
            if context == "multires":
                long = read(record, "long")
                if not np.array_equal(long.pop("ids"), ids):
                    raise ValueError("Long/base cache IDs differ")
                if set(long) != {"left60", "right60", "center90"} or any(
                    v.shape != (len(ids), 106) or v.dtype != np.float32
                    for v in long.values()
                ):
                    raise ValueError("Invalid long-window cache")
            matrix, names = add_context(
                base["base"],
                meta["epoch_index"].to_numpy(),
                feature_names(),
                context,
                long,
            )
            if np.isinf(matrix).any():
                raise ValueError("Infinite cached features")
            matrices.append(matrix)
            metadata.append(meta)
            progress.update(1)
    if not matrices:
        raise ValueError("Empty cache selection")
    digest = hashlib.sha256(json.dumps(objects, sort_keys=True).encode()).hexdigest()
    base_digest = hashlib.sha256(
        json.dumps(
            [o for o in objects if o["name"].endswith("_epoch.npz")], sort_keys=True
        ).encode()
    ).hexdigest()
    return (
        np.concatenate(matrices),
        pl.concat(metadata),
        names,
        {
            "key": folder.name,
            "definition": definition,
            "objects_sha256": digest,
            "objects": objects,
            "base_objects_sha256": base_digest,
            "cache_only": True,
        },
    )
