"""Read-only cached sources for four-expert integration; no old backbone inference."""

import hashlib
import json
import shutil
from functools import cache
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from src.expansion_data import shared_folds
from src.models.pretrained_assets import asset_manifest, sha256
from src.pretrained import ROOT
from src.refinement_data import InputStore
from src.series_data import SeriesData, align_prediction, fingerprint

PRIOR4 = (1 / 3, 1 / 3, 1 / 6, 1 / 6)
EXPERTS4 = (156, 19, 82, 85)
REFERENCES = (19, 82, 85, 99, 102, 106, 113, 156, 157)


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def name(number: int) -> str:
    return f"{number:03d}_pretrained"


def normalize(p):
    p = np.asarray(p, np.float32)
    if not np.isfinite(p).all() or np.any(p < 0) or np.any(p.sum(-1) <= 0):
        raise ValueError("Invalid expert probabilities")
    return p / p.sum(-1, keepdims=True)


class Sources:
    def __init__(self, root: Path, data):
        self.root, self.data = root.resolve(), data
        self.manifest = read(self.root / "bundle.json")
        if (
            self.manifest.get("kind") != "usleep_integration_inputs_v1"
            or self.manifest["data"] != data.identity
        ):
            raise ValueError("Need matching sleep-edf-usleep-integration-inputs.zip")
        self.used = {}

    def path(self, relative: str) -> Path:
        target = (self.root / relative).resolve()
        if (
            not target.is_relative_to(self.root)
            or relative not in self.manifest["files"]
        ):
            raise ValueError(f"Unknown/unsafe Input file: {relative}")
        if relative not in self.used:
            if sha256(target) != self.manifest["files"][relative]:
                raise ValueError(f"Input SHA differs: {relative}")
            self.used[relative] = self.manifest["files"][relative]
        return target

    @cache
    def prediction(self, number: int, split: str):
        path = self.path(f"results/{name(number)}/{split}_predictions.npz")
        with np.load(path, allow_pickle=False) as z:
            values = {
                k: z[k] for k in ("ids", "probabilities", "labels") if k in z.files
            }
        return align_prediction(values, self.data.metadata(split))

    def verify_prefix(self, prefix: str):
        for relative in self.manifest["files"]:
            if relative.startswith(prefix):
                self.path(relative)


class TrainingInputs:
    """Facade for the existing102/106 training code, now with four experts."""

    def __init__(self, data, args, progress, sources: Sources, cache_root: Path):
        self.data, self.args, self.progress, self.sources = (
            data,
            args,
            progress,
            sources,
        )
        self.cache = cache_root
        self.memo = {}
        self.old = InputStore([sources.root / "results"], data)
        self.cache.mkdir(parents=True, exist_ok=True)
        # shared_folds uses writable lock files; never lock/write a Kaggle Input.
        for relative in sources.manifest["files"]:
            if relative.startswith("training_cache/folds/"):
                source = sources.path(relative)
                target = self.cache / Path(relative).relative_to("training_cache")
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists() and sha256(target) != sha256(source):
                    raise ValueError("Cached folds differ")
                if not target.exists():
                    shutil.copyfile(source, target)
        self.folds = shared_folds(data, self.cache)

    def preflight(self):
        """Check nested sources BEFORE paying for U-Sleep train inference."""
        self.sources.verify_prefix("results/")
        self.sources.verify_prefix("features/")
        self.sources.verify_prefix("training_cache/")
        meta = self.data.metadata("train")
        p019 = self.old.baseline019("train", meta)
        if p019 is None:
            raise FileNotFoundError("Saved train019 is required; no recomputation")
        original = normalize(
            np.stack(
                [
                    p019,
                    self.old.expert_oof("082_pretrained", meta, self.folds),
                    self.old.expert_oof("085_pretrained", meta, self.folds),
                ],
                axis=1,
            ).astype(np.float32)
        )
        self.memo["original3"] = original
        nested = []
        for outer in range(3):
            relative = f"training_cache/nested/outer_{outer}.npz"
            path = self.sources.path(relative)
            info = read(self.sources.path(relative.replace(".npz", ".json")))
            expected = fingerprint(
                {
                    "data": self.data.identity,
                    "ids": fingerprint(meta["ids"].tolist()),
                    "protocol": "inner-group2-fixed5-v1",
                    "folds": self.folds.tolist(),
                    "original_oof": hashlib.sha256(original.tobytes()).hexdigest(),
                    "outer": outer,
                }
            )
            if info["contract"] != expected or info["sha256"] != sha256(path):
                raise ValueError(
                    "Nested OOF source contract differs; refusing expert retraining"
                )
            with np.load(path, allow_pickle=False) as z:
                if not np.array_equal(z["ids"], meta["ids"]):
                    raise ValueError("Nested OOF ID order differs")
                p = z["experts"]
            if p.shape != original.shape or not np.array_equal(
                p[self.folds == outer], original[self.folds == outer]
            ):
                raise ValueError(
                    "Nested outer-held predictions differ from original expert OOF"
                )
            nested.append(p)
        self.memo["nested3"] = nested
        # All representation shards must exist; there is no extraction fallback.
        index = {}
        for relative in self.sources.manifest["files"]:
            if not relative.startswith("embeddings/") or not relative.endswith(".json"):
                continue
            info = read(self.sources.path(relative))
            family, record, split = (
                info["family"],
                Path(relative).stem,
                Path(relative).parent.name,
            )
            if info["data"] != self.data.identity or info["smoke"]:
                raise ValueError("Representation dataset/smoke differs")
            key = (family, split, record)
            if key in index:
                raise ValueError("Duplicate representation shard")
            index[key] = (relative, info)
        for family in ("tiny", "anysleep"):
            for split in ("train", "val", "test"):
                if any(
                    (family, split, p.stem) not in index for p in self.data.paths(split)
                ):
                    raise FileNotFoundError(
                        f"Missing {family}/{split} embeddings; no new inference allowed"
                    )
        self.embedding_index = index

    def set_usleep(self, p):
        p = normalize(p)
        self.memo[("experts", "train")] = normalize(
            np.concatenate([p[:, None], self.memo["original3"]], axis=1)
        )
        self.memo["nested"] = [
            normalize(np.concatenate([p[:, None], old], axis=1))
            for old in self.memo.pop("nested3")
        ]

    def experts(self, split):
        key = ("experts", split)
        if key not in self.memo:
            if split == "train":
                raise RuntimeError("U-Sleep train probabilities not prepared")
            self.memo[key] = normalize(
                np.stack(
                    [
                        self.sources.prediction(n, split)["probabilities"]
                        for n in EXPERTS4
                    ],
                    axis=1,
                )
            )
        return self.memo[key]

    def embeddings(self, split):
        key = ("embeddings", split)
        if key in self.memo:
            return self.memo[key]
        chunks = []
        for family in ("tiny", "anysleep"):
            parts = []
            expected_assets = asset_manifest(
                [
                    "tinysleepnet.pt",
                    "tinysleepnet-source.py",
                    "tinysleepnet-config.json",
                ]
                if family == "tiny"
                else ["anysleep-run1.pth", "anysleep_no_hydra.py"]
            )
            layer = (
                "clf.encode contextual128 rolling20"
                if family == "tiny"
                else "segment_classifier.segment_classifier.2 AvgPool before 5class Conv"
            )
            for record in self.data.paths(split):
                relative, info = self.embedding_index[(family, split, record.stem)]
                expected_input = self.data.hashes[
                    record.relative_to(self.data.root).as_posix()
                ]
                if (
                    info["assets"] != expected_assets
                    or info["layer"] != layer
                    or info["input"] != expected_input
                ):
                    raise ValueError("Frozen embedding weights/layer/source differ")
                path = self.sources.path(relative.replace(".json", ".npz"))
                if sha256(path) != info["sha256"]:
                    raise ValueError("Frozen embedding SHA differs")
                meta = self.data.record_metadata(record, split)
                with np.load(path, allow_pickle=False) as z:
                    if not np.array_equal(z["ids"], meta["ids"]):
                        raise ValueError("Embedding epoch IDs differ")
                    parts.append(z["embeddings"])
            chunks.append(np.concatenate(parts))
        result = np.concatenate(chunks, axis=1).astype(np.float32)
        if result.shape[1] != 134:
            raise ValueError("Expected original frozen Tiny128 + AnySleep6 embeddings")
        self.memo[key] = result
        return result


def pack_input(target: Path):
    """Package existing caches only; no feature extraction, model fits or inference."""
    old = ROOT / "archive/artefacts"
    files = {}
    for number in REFERENCES:
        folder = ROOT / "artefacts" / name(number)
        if not (folder / "val_predictions.npz").exists():
            folder = old / name(number)
        for filename in (
            "provenance.json",
            "resolved.yaml",
            "selection.json",
            "oof_predictions.npz",
            "oof_predictions.json",
            "val_predictions.npz",
            "val_predictions.json",
            "test_predictions.npz",
            "test_predictions.json",
        ):
            p = folder / filename
            if p.is_file():
                files[f"results/{name(number)}/{filename}"] = p
        for split in ("val", "test"):
            if f"results/{name(number)}/{split}_predictions.npz" not in files:
                raise FileNotFoundError(f"Missing {name(number)} {split}")
    for p in (old / "predictions").glob("*/train.json"):
        if read(p).get("recipe") == {"kind": "baseline019"}:
            for f in (p, p.with_suffix(".npz")):
                files[f"results/predictions/{p.parent.name}/{f.name}"] = f
    for definition in (old / "cache/features").glob("*/definition.json"):
        content = read(definition)
        if content.get("version") == "linear106_v1" and not content.get("smoke"):
            for p in definition.parent.iterdir():
                if p.is_file() and (
                    p.name == "definition.json" or p.name.endswith("_epoch.npz")
                ):
                    files[f"features/{p.name}"] = p
    cache = old / "cache/refinement/full"
    for prefix in ("folds", "nested", "embeddings"):
        for p in (cache / prefix).rglob("*"):
            if p.is_file() and p.suffix in (".npz", ".json"):
                relative = p.relative_to(cache).as_posix()
                files[
                    relative if prefix == "embeddings" else f"training_cache/{relative}"
                ] = p
    for filename in ("Pretrained_Model.ckpt", "author_usleep.py"):
        files[f"assets/{filename}"] = ROOT / "docs/usleep_rbd_source" / filename
    data_root = ROOT / "data/raw"
    if not (data_root / "MANIFEST.json").exists():
        data_root = ROOT / "archive/data/raw"
    manifest = {
        "kind": "usleep_integration_inputs_v1",
        "data": SeriesData(data_root).identity,
        "files": {k: sha256(v) for k, v in files.items()},
        "purpose": "saved expert probabilities, original nested OOF/folds, frozen Tiny128+Any6 embeddings,106 features and official U-Sleep assets; no old backbone inference/training",
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(target, "x", ZIP_DEFLATED, compresslevel=1) as z:
        for relative, path in files.items():
            z.write(path, relative)
        z.writestr("bundle.json", json.dumps(manifest, indent=2))
    print(
        f"Integration Input: {target} ({target.stat().st_size / 2**20:.1f}MiB), {len(files)} files",
        flush=True,
    )
