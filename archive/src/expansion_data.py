"""Read-only portable features, shared subject folds and bounded record caches."""

import hashlib
import json
from copy import copy
from pathlib import Path

import numpy as np
import scipy
from sklearn.model_selection import GroupKFold

from src.data import load_record
from src.models.pretrained_assets import sha256
from src.models.tinysleepnet import tiny_preprocess
from src.pretrained import save_json, save_npz
from src.pretrained_signals import contiguous_segments
from src.series_data import SeriesData, fingerprint
from src.storage import atomic_path, canonical_fingerprint, file_lock


class ExpansionData(SeriesData):
    """Training smoke samples real sleep-onset windows rather than hours of Wake.

    Training labels determine smoke sampling only. Full runs and official val/test
    retain the original selection; smoke is never an experiment of quality.
    """

    def record_metadata(self, path: Path, split: str) -> dict:
        if not self.smoke or split != "train":
            return super().record_metadata(path, split)
        full = copy(self)
        full.smoke = False
        meta = SeriesData.record_metadata(full, path, split)
        asleep = np.flatnonzero(meta["labels"] != 0)
        start = (
            max(0, int(asleep[0]) - min(20, self.smoke_epochs // 10))
            if len(asleep)
            else 0
        )
        start = min(start, max(0, len(meta["ids"]) - self.smoke_epochs))
        return {k: v[start : start + self.smoke_epochs] for k, v in meta.items()}


def selected_signal(data, path: Path, split: str) -> np.ndarray:
    """Align raw rows by epoch ID, including non-prefix training smoke windows."""
    meta = data.record_metadata(path, split)
    raw = load_record(path, signals=True)
    if np.array_equal(raw["ids"][: len(meta["ids"])], meta["ids"]):
        return raw["x"][: len(meta["ids"])]
    positions = {str(name): i for i, name in enumerate(raw["ids"])}
    indices = np.array([positions[str(name)] for name in meta["ids"]])
    if not np.array_equal(raw["epoch_index"][indices], meta["epoch_index"]):
        raise ValueError("Selected raw signal epoch alignment differs")
    return raw["x"][indices]


def shared_folds(data, root: Path) -> np.ndarray:
    metadata = data.metadata("train")
    contract = {
        "data": data.identity,
        "ids": fingerprint(metadata["ids"].tolist()),
        "subjects": fingerprint(metadata["subject_id"].tolist()),
        "method": "GroupKFold3",
        "smoke": data.smoke,
    }
    target = root / "folds" / fingerprint(contract) / "folds.npz"
    with file_lock(target.with_suffix(".lock"), wait=True):
        if target.exists():
            info = json.loads(target.with_suffix(".json").read_text())
            if info["contract"] != contract or info["sha256"] != sha256(target):
                raise ValueError("Saved folds contract/checksum differs")
            with np.load(target, allow_pickle=False) as stored:
                if not np.array_equal(stored["ids"], metadata["ids"]):
                    raise ValueError("Saved folds IDs differ")
                result = stored["folds"]
        else:
            result = np.full(len(metadata["ids"]), -1, dtype=np.int8)
            for fold, (train, held) in enumerate(
                GroupKFold(3).split(result, groups=metadata["subject_id"])
            ):
                if set(metadata["subject_id"][train]) & set(
                    metadata["subject_id"][held]
                ):
                    raise ValueError("Subject leakage")
                result[held] = fold
            save_npz(target, ids=metadata["ids"], folds=result)
            save_json(
                target.with_suffix(".json"),
                {"contract": contract, "sha256": sha256(target)},
            )
    if set(result.tolist()) != {0, 1, 2}:
        raise ValueError("Invalid fold coverage")
    for fold in range(3):
        if set(metadata["subject_id"][result == fold]) & set(
            metadata["subject_id"][result != fold]
        ):
            raise ValueError("Subject leakage in saved folds")
    return result


class RecordCache:
    """One record in RAM, disk .npy memmaps, preprocessing never crosses gaps."""

    def __init__(self, data, root: Path):
        self.data, self.root = data, root

    def get(self, path: Path, split: str, family: str) -> np.ndarray:
        meta = self.data.record_metadata(path, split)
        digest = self.data.verify(path)
        contract = {
            "input": digest,
            "data": self.data.identity,
            "ids": fingerprint(meta["ids"].tolist()),
            "family": family,
            "recipe": "tiny-bp03-40-order5-v1" if family == "tiny" else "raw-uV-v1",
            "implementation": sha256(Path(tiny_preprocess.__code__.co_filename))
            if family == "tiny"
            else "raw-float32-no-transform",
            "packages": {"numpy": np.__version__, "scipy": scipy.__version__},
        }
        target = self.root / "signals" / fingerprint(contract) / "signal.npy"
        with file_lock(target.with_suffix(".lock"), wait=True):
            if not target.exists():
                raw = selected_signal(self.data, path, split)
                if family == "tiny":
                    signal = np.empty((len(raw), 3000), dtype=np.float32)
                    for segment in contiguous_segments(meta["epoch_index"]):
                        signal[segment] = tiny_preprocess(raw[segment], 0)
                elif family == "sleepgpt":
                    signal = np.asarray(raw, dtype=np.float32)
                else:
                    raise ValueError(family)
                with atomic_path(target) as temporary:
                    with temporary.open("wb") as stream:
                        np.save(stream, signal, allow_pickle=False)
                save_json(
                    target.with_suffix(".json"), {**contract, "sha256": sha256(target)}
                )
        info = json.loads(target.with_suffix(".json").read_text())
        if any(info[k] != v for k, v in contract.items()):
            raise ValueError("Signal cache contract differs")
        # Verify once per process, not once per training mini-batch.
        if not hasattr(self, "verified"):
            self.verified = set()
        if target not in self.verified:
            if sha256(target) != info["sha256"]:
                raise ValueError("Signal cache checksum differs")
            self.verified.add(target)
        return np.load(target, mmap_mode="r", allow_pickle=False)


def physiological_features(data, split: str, cache: Path | None) -> np.ndarray:
    """Read existing 106-feature files; do not import/re-run the extractor.

    Package versions describe creation, not reading; the definition and array IDs
    are retained in the new experiment provenance. No cache writes on Input.
    """
    if cache is None:
        raise FileNotFoundError(
            "D3/D4/E require --feature-cache with the existing 106-feature cache"
        )
    definitions = list(cache.rglob("definition.json"))
    if (cache / "definition.json").exists():
        definitions = [cache / "definition.json"]
    candidates = []
    for definition in definitions:
        content = definition.read_text(encoding="utf-8")
        info = json.loads(content)
        if (
            info.get("version") == "linear106_v1"
            and info.get("revision") == "full_psg_v1"
            and not info.get("smoke")
        ):
            candidates.append(definition)
    if len(candidates) != 1:
        raise ValueError("Specify exactly one full_psg_v1 linear106 cache directory")
    root = candidates[0].parent
    definition = json.loads(candidates[0].read_text())
    source = Path(__file__).parent
    expected_source = hashlib.sha256(
        (source / "features.py").read_bytes() + (source / "datasets.py").read_bytes()
    ).hexdigest()
    if definition["labels_sha256"] != data.labels_hash or canonical_fingerprint(
        definition["source"], "features"
    ) != canonical_fingerprint(expected_source, "features"):
        raise ValueError("Physiological cache data/extractor contract differs")
    wanted = data.metadata(split)["ids"]
    collected = {}
    for path in root.rglob("*_epoch.npz"):
        with np.load(path, allow_pickle=False) as stored:
            if (
                "base" not in stored
                or stored["base"].ndim != 2
                or stored["base"].shape[1] != 106
            ):
                raise ValueError(f"Not a 106-feature cache: {path}")
            for name, row in zip(
                stored["ids"].astype(str), stored["base"], strict=True
            ):
                if name in collected:
                    raise ValueError("Duplicate physiological cache epoch")
                collected[name] = row
    missing = set(wanted) - collected.keys()
    if missing:
        raise ValueError(
            f"Physiological cache missing {len(missing)} {split} epochs; extraction is never automatic"
        )
    return np.stack([collected[name] for name in wanted]).astype(np.float32)
