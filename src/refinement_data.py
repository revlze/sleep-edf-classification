"""Verified read-through inputs and shared expert/representation caches."""

import json
from functools import cache
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from src.expansion_data import ExpansionData, RecordCache, selected_signal, shared_folds
from src.expansion_engine import ExpansionEngine
from src.models.pretrained_assets import sha256
from src.pretrained import ROOT, save_json
from src.series_data import (
    align_prediction,
    fingerprint,
    load_prediction,
    save_prediction,
)
from src.storage import atomic_path


class RefinementData(ExpansionData):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        import polars as pl

        train = set(self.labels.filter(pl.col("split") == "train")["subject_id"])
        held = set(self.labels.filter(pl.col("split") == "val")["subject_id"])
        if train & held:
            raise ValueError("Train/val subject overlap in supplied metadata")

    @cache
    def record_metadata(self, path: Path, split: str) -> dict:
        return super().record_metadata(path, split)


def subset(values: dict, meta: dict) -> dict:
    wanted = set(meta["ids"].tolist())
    select = np.array([str(x) in wanted for x in values["ids"]])
    return align_prediction({k: v[select] for k, v in values.items()}, meta)


class InputStore:
    """Read-only unpacked Input directories; code/checkpoints never execute on discovery."""

    def __init__(self, roots: list[Path], data):
        self.roots, self.data = list(dict.fromkeys(roots)), data
        self.files = list(
            dict.fromkeys(
                p
                for root in self.roots
                if root.exists()
                for p in root.rglob("*.json")
                if "source" not in p.parts and "smoke" not in p.parts
            )
        )
        self.used = {}
        self.alignment = {}

    def experiment(self, name: str, filename: str) -> Path | None:
        for root in self.roots:
            direct = root / name / filename
            if direct.is_file():
                return direct
        for info in self.files:
            if info.parent.name == name and (info.parent / filename).is_file():
                return info.parent / filename
        return None

    def prediction(self, name, split, meta):
        path = self.experiment(name, f"{split}_predictions.npz")
        if path is None:
            return None
        if (
            name == "019_pretrained"
            and split == "val"
            and not path.with_suffix(".json").exists()
        ):
            from src.series_data import read_legacy_019

            result = read_legacy_019([path.parent.parent], self.data, "val")
            if result is None:
                raise ValueError("Missing original019 validation provenance")
            self.used[str(path)] = sha256(path)
            return result[0]
        info = json.loads(path.with_suffix(".json").read_text())
        if info.get("data") != self.data.identity or info.get("smoke", False):
            raise ValueError(f"{name}: source dataset/smoke differs")
        if info["prediction_sha256"] != sha256(path):
            raise ValueError(f"Corrupt saved prediction: {path}")
        if name in ("082_pretrained", "085_pretrained"):
            original = json.loads((path.parent / "provenance.json").read_text())
            if (
                info.get("contract") != original["contract"]
                or info.get("config") != original["config"]
            ):
                raise ValueError(
                    f"{name}: OOF and final checkpoint recipes are not aligned"
                )
        with np.load(path, allow_pickle=False) as saved:
            values = {k: saved[k] for k in saved.files}
        self.used[str(path)] = info["prediction_sha256"]
        return subset(values, meta)

    def expert_oof(self, name, meta, folds):
        path = self.experiment(name, "oof_predictions.npz")
        if path is None:
            raise FileNotFoundError(
                f"Need original complete {name}/oof_predictions.npz"
            )
        folder = path.parent
        provenance = json.loads((folder / "provenance.json").read_text())
        info = json.loads(path.with_suffix(".json").read_text())
        expected = OmegaConf.to_container(
            OmegaConf.load(ROOT / "configs" / f"{name}.yaml"), resolve=True
        )
        if (
            provenance["data"] != self.data.identity
            or provenance["smoke"]
            or provenance["config"] != expected
            or info["contract"] != provenance["contract"]
            or info["prediction_sha256"] != sha256(path)
        ):
            raise ValueError(f"{name}: original OOF config/provenance differs")
        with np.load(path, allow_pickle=False) as saved:
            original = {k: saved[k] for k in saved.files}
        source_folds = dict(zip(original["ids"].tolist(), info["folds"], strict=True))
        # Smoke has different selected subjects: verify original full fold subjects
        # separately, then use fresh strict nested folds for smoke below.
        if not self.data.smoke and not np.array_equal(
            [source_folds[x] for x in meta["ids"]], folds
        ):
            raise ValueError("082/085 OOF folds differ from shared subject folds")
        self.used[str(path)] = info["prediction_sha256"]
        selected = json.loads((folder / "selection.json").read_text())
        self.alignment[name] = {
            "oof_contract": info["contract"],
            "fold_best_epochs": info.get("best_epochs"),
            "refit_epochs": selected.get("refit_epochs"),
            "downstream_train": "original grouped OOF, never refit train predictions",
            "provided_val_test": "original full-train refit predictions with same config/contract",
            "nested": "inner2 fixed5 epochs; outer held original OOF; smaller training sample and different checkpoint-selection budget than final experts",
        }
        return subset(original, meta)["probabilities"]

    def baseline019(self, split, meta):
        saved = self.prediction("019_pretrained", split, meta)
        if saved is not None:
            return saved["probabilities"]
        for path in self.files:
            if path.name != f"{split}.json":
                continue
            info = json.loads(path.read_text())
            if (
                info.get("recipe") != {"kind": "baseline019"}
                or info.get("data") != self.data.identity
            ):
                continue
            payload = path.with_suffix(".npz")
            with np.load(payload, allow_pickle=False) as saved:
                if not set(meta["ids"]).issubset(set(saved["ids"].tolist())):
                    continue
                values = {k: saved[k] for k in saved.files}
            if info["prediction_sha256"] != sha256(payload):
                raise ValueError("019 cached probability checksum differs")
            self.used[str(payload)] = info["prediction_sha256"]
            return subset(values, meta)["probabilities"]
        return None


class SharedInputs:
    def __init__(self, data, args, progress):
        self.data, self.args, self.progress = data, args, progress
        self.cache = args.cache_root / ("smoke" if data.smoke else "full")
        self.cache.mkdir(parents=True, exist_ok=True)
        self.store = InputStore([*args.input_root, args.output_root], data)
        self.engine = ExpansionEngine(
            data,
            args.output_root,
            self.cache,
            args.checkpoint_cache,
            args.checkpoint_root,
            args.input_root,
            args.device,
        )
        self.engine.progress = progress
        self.memo = {}

    def baseline(self, split):
        key = ("019", split)
        if key not in self.memo:
            meta = self.data.metadata(split)
            p = self.store.baseline019(split, meta)
            if p is None:
                self.progress.stage(
                    "Подготовка: 019", f"{split}: кэш не найден, frozen inference"
                )
                p = self.engine.baseline("019", split)["probabilities"]
            self.memo[key] = p
        return self.memo[key]

    def experts(self, split):
        key = ("experts", split)
        if key not in self.memo:
            meta = self.data.metadata(split)
            if split == "train":
                folds = shared_folds(self.data, self.cache)
                ps = [
                    self.store.expert_oof(f"{n}_pretrained", meta, folds)
                    for n in ("082", "085")
                ]
            else:
                ps = []
                for n in ("082", "085"):
                    values = self.store.prediction(f"{n}_pretrained", split, meta)
                    if values is None:
                        raise FileNotFoundError(
                            f"Need {n} saved {split} predictions; will not silently retrain"
                        )
                    ps.append(values["probabilities"])
            p = np.stack([self.baseline(split), *ps], axis=1).astype(np.float32)
            p /= p.sum(-1, keepdims=True)
            self.memo[key] = p
        return self.memo[key]

    def embeddings(self, split):
        key = ("embeddings", split)
        if key in self.memo:
            return self.memo[key]
        chunks = []
        for family in ("tiny", "anysleep"):
            parts = []
            for record in self.data.paths(split):
                meta = self.data.record_metadata(record, split)
                candidates = [
                    p
                    for p in self.store.files
                    if p.stem == record.stem and p.parent.name == split
                ]
                found = None
                for p in candidates:
                    info = json.loads(p.read_text())
                    if (
                        info.get("family") != family
                        or info.get("data") != self.data.identity
                        or info.get("smoke")
                        or info.get("input") != self.data.verify(record)
                    ):
                        continue
                    # Compare pinned source weights AND actual pre-classifier layer.
                    from src.models.pretrained_assets import asset_manifest

                    names = (
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
                    if (
                        info.get("assets") != asset_manifest(names)
                        or info.get("layer") != layer
                    ):
                        continue
                    target = p.with_suffix(".npz")
                    if sha256(target) != info["sha256"]:
                        raise ValueError("Embedding checksum differs")
                    with np.load(target, allow_pickle=False) as saved:
                        positions = {x: i for i, x in enumerate(saved["ids"].tolist())}
                        if not set(meta["ids"]).issubset(positions):
                            continue
                        found = saved["embeddings"][[positions[x] for x in meta["ids"]]]
                    self.store.used[str(target)] = info["sha256"]
                    break
                if found is None:
                    parts = []
                    break
                parts.append(found)
            if parts:
                self.progress.stage(
                    "Подготовка embeddings",
                    f"{family}/{split}: готовый проверенный Input",
                )
                chunks.append(np.concatenate(parts))
            else:
                self.progress.stage(
                    "Подготовка embeddings", f"{family}/{split}: извлечение один раз"
                )
                chunks.append(extract_embeddings(self, family, split))
        self.memo[key] = np.concatenate(chunks, axis=1).astype(np.float32)
        return self.memo[key]


def extract_embeddings(shared, family, split):
    """Same original representation and context, larger vectorized GPU batches."""
    import torch

    from src.models.anysleep_variants import AnySleepVariant
    from src.models.pretrained_assets import asset_manifest
    from src.models.tinysleepnet import TinySleepNetAdapter, tiny_preprocess
    from src.pretrained import save_npz
    from src.pretrained_signals import (
        contiguous_segments,
        epoch_blocks,
        preprocess_segment,
    )
    from src.series_engine import base_member

    names = (
        ["tinysleepnet.pt", "tinysleepnet-source.py", "tinysleepnet-config.json"]
        if family == "tiny"
        else ["anysleep-run1.pth", "anysleep_no_hydra.py"]
    )
    contract = {
        "family": family,
        "data": shared.data.identity,
        "implementation": shared.engine.implementation,
        "assets": asset_manifest(names),
        "layer": "clf.encode contextual128 rolling20"
        if family == "tiny"
        else "segment_classifier.segment_classifier.2 AvgPool before 5class Conv",
        "smoke": shared.data.smoke,
        "limit": shared.data.smoke_epochs if shared.data.smoke else None,
    }
    root = shared.cache / "embeddings" / fingerprint(contract)
    outputs, adapter = [], None
    for number, path in enumerate(shared.data.paths(split), 1):
        meta = shared.data.record_metadata(path, split)
        target = root / split / f"{path.stem}.npz"
        if target.with_suffix(".json").exists():
            info = json.loads(target.with_suffix(".json").read_text())
            if info["sha256"] != sha256(target) or info["input"] != shared.data.verify(
                path
            ):
                raise ValueError("Embedding record resume differs")
            with np.load(target, allow_pickle=False) as saved:
                if not np.array_equal(saved["ids"], meta["ids"]):
                    raise ValueError("Embedding IDs differ")
                outputs.append(saved["embeddings"])
            continue
        if adapter is None:
            adapter = (
                TinySleepNetAdapter(
                    shared.args.checkpoint_cache,
                    shared.args.checkpoint_root,
                    shared.args.device,
                )
                if family == "tiny"
                else AnySleepVariant(
                    base_member(1),
                    shared.args.checkpoint_cache,
                    shared.args.checkpoint_root,
                    shared.args.device,
                )
            )
        raw = selected_signal(shared.data, path, split)
        parts = []
        for part in contiguous_segments(meta["epoch_index"]):

            def detail(message):
                shared.progress.update(
                    detail=f"embeddings {family}/{split}, запись {number}/{len(shared.data.paths(split))}, {path.stem}: {message}"
                )

            with torch.inference_mode():
                if family == "tiny":
                    signal = tiny_preprocess(raw[part], 0)
                    features = []
                    for start in range(0, len(signal), 256):
                        detail(f"CNN {start}/{len(signal)} эпох, batch256")
                        x = torch.from_numpy(
                            signal[start : start + 256, None].copy()
                        ).to(shared.args.device)
                        features.append(adapter.model.feature_extractor(x))
                    features = torch.cat(features)
                    width = min(20, len(signal))
                    out = [
                        adapter.model.clf.encode(features[:width][None])[0]
                        .float()
                        .cpu()
                        .numpy()
                    ]
                    for start in range(20, len(signal), 256):
                        detail(f"LSTM {start}/{len(signal)} окон, batch256")
                        ends = torch.arange(
                            start,
                            min(len(signal), start + 256),
                            device=shared.args.device,
                        )
                        idx = ends[:, None] + torch.arange(
                            -19, 1, device=shared.args.device
                        )
                        out.append(
                            adapter.model.clf.encode(features[idx])[:, -1]
                            .float()
                            .cpu()
                            .numpy()
                        )
                    parts.append(np.concatenate(out))
                else:
                    signal = preprocess_segment(raw[part], "anysleep")
                    blocks = list(
                        epoch_blocks(len(raw[part]), 128, 64, right_alignment=16)
                    )
                    groups = {}
                    for block in blocks:
                        groups.setdefault(block[1] - block[0], []).append(block)
                    signal_gpu = torch.from_numpy(np.ascontiguousarray(signal)).to(
                        shared.args.device
                    )
                    captured = []
                    layer = adapter.model.segment_classifier.segment_classifier[2]
                    handle = layer.register_forward_hook(
                        lambda module, args, result: captured.append(result)
                    )
                    result = None
                    completed = 0
                    try:
                        for group in groups.values():
                            for start in range(0, len(group), 4):
                                batch = group[start : start + 4]
                                detail(
                                    f"AnySleep блоки {completed}/{len(blocks)}, batch{len(batch)}, исходный context128+64"
                                )
                                x = torch.stack(
                                    [
                                        signal_gpu[a * 3840 : b * 3840]
                                        for a, b, _, _ in batch
                                    ]
                                )
                                captured.clear()
                                adapter.model(x)
                                if len(captured) != 1 or captured[0].shape[0] != len(
                                    batch
                                ):
                                    raise ValueError(
                                        "AnySleep embedding hook shape differs"
                                    )
                                z = captured[0].float().cpu().numpy()
                                if result is None:
                                    result = np.empty(
                                        (len(raw[part]), z.shape[1]), np.float32
                                    )
                                for i, (a, b, left, right) in enumerate(batch):
                                    result[left:right] = z[i, :, left - a : right - a].T
                                completed += len(batch)
                    finally:
                        handle.remove()
                    parts.append(result)
        embedding = np.concatenate(parts)
        save_npz(target, ids=meta["ids"], embeddings=embedding)
        save_json(
            target.with_suffix(".json"),
            {**contract, "input": shared.data.verify(path), "sha256": sha256(target)},
        )
        outputs.append(embedding)
    del adapter
    shared.engine.release()
    return np.concatenate(outputs)


class FastRecords:
    """One memory-mapped pool; no metadata/hash/record switching inside a batch."""

    family = "tiny"

    def __init__(self, data, cache: Path, progress, read_roots=()):
        self.data, self.rows, self.current = data, [], None
        self.cache, self.progress = RecordCache(data, cache), progress
        self.read_roots = list(read_roots)
        offset = 0
        for path in data.paths("train"):
            meta = data.record_metadata(path, "train")
            self.rows.append((path, meta, offset))
            offset += len(meta["ids"])
        meta = data.metadata("train")
        key = fingerprint(
            {
                "data": data.identity,
                "ids": fingerprint(meta["ids"].tolist()),
                "recipe": "tiny-bp03-40-order5-v1",
            }
        )[:24]
        target = cache / "pools" / key / "tiny.npy"
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            with atomic_path(target) as temporary:
                pool = np.lib.format.open_memmap(
                    temporary, mode="w+", dtype="float32", shape=(offset, 3000)
                )
                for i, (path, record, start) in enumerate(self.rows):
                    progress.stage(
                        "Подготовка сигналов",
                        f"train запись {i + 1}/{len(self.rows)}; один проход preprocessing",
                    )
                    pool[start : start + len(record["ids"])] = self.signal(path)
                pool.flush()
                del pool
            save_json(
                target.with_suffix(".json"), {"shape": [offset, 3000], "key": key}
            )
        self.pool = np.load(target, mmap_mode="r", allow_pickle=False)
        if self.pool.shape != (offset, 3000):
            raise ValueError("Signal pool shape differs")
        self.offsets = np.asarray([row[2] for row in self.rows])
        self.labels = meta["labels"]

    def signal(self, path, split="train"):
        return self.cache.get(path, split, "tiny")

    def ends(self, permitted):
        from src.refinement_models import segment_bounds

        meta = self.data.metadata("train")
        starts, _ = segment_bounds(meta)
        idx = np.flatnonzero(permitted & (np.arange(len(starts)) - starts >= 19))
        # All epochs of a record belong to the same subject/fold.
        if np.any(~permitted[idx - 19]):
            raise ValueError("Training window crosses permitted subset")
        record = np.searchsorted(self.offsets, idx, side="right") - 1
        return np.stack([record, idx - self.offsets[record]], axis=1)

    def batch(self, pairs):
        import torch

        ends = self.offsets[pairs[:, 0]] + pairs[:, 1]
        idx = ends[:, None] + np.arange(-19, 1)
        x = np.asarray(self.pool[idx])[:, :, None, :].copy()
        return torch.from_numpy(x), torch.from_numpy(self.labels[idx].astype(np.int64))


def nested_experts(shared: SharedInputs):
    """Inner cross-fitting prevents base-model label leakage into outer gate CV.

    082/085 outer held predictions are reused verbatim in full runs. Only the
    missing inner OOF predictions are trained, once for ALL gate/temporal models.
    Inner checkpoints run for a fixed budget, never stopping on outer held labels.
    """
    import hashlib

    from sklearn.model_selection import GroupKFold

    from src.training.expansion import fit_one
    from src.training.refinement import tiny_predict

    data, args = shared.data, shared.args
    meta = data.metadata("train")
    folds = shared_folds(data, shared.cache)
    original = shared.experts("train")
    outputs, records = [], None
    for outer in range(3):
        mask = folds != outer
        target = shared.cache / "nested" / f"outer_{outer}.npz"
        contract = fingerprint(
            {
                "data": data.identity,
                "ids": fingerprint(meta["ids"].tolist()),
                "protocol": "inner-group2-fixed5-v1",
                "folds": folds.tolist(),
                "original_oof": hashlib.sha256(original.tobytes()).hexdigest(),
                "outer": outer,
            }
        )
        if target.with_suffix(".json").exists():
            info = json.loads(target.with_suffix(".json").read_text())
            if info["contract"] != contract or info["sha256"] != sha256(target):
                raise ValueError("Nested source cache contract differs")
            with np.load(target, allow_pickle=False) as saved:
                outputs.append(saved["experts"])
            continue
        if records is None:
            records = FastRecords(data, shared.cache, shared.progress)
        p = original.copy()
        outer_indices = np.flatnonzero(mask)
        inner_folds = np.full(len(folds), -1)
        for inner, (_, held) in enumerate(
            GroupKFold(2).split(outer_indices, groups=meta["subject_id"][mask])
        ):
            inner_folds[outer_indices[held]] = inner
        for inner in range(2):
            train_mask, held_mask = mask & (inner_folds != inner), inner_folds == inner
            if set(meta["subject_id"][train_mask]) & set(
                meta["subject_id"][~mask | held_mask]
            ):
                raise ValueError("Nested expert subject leakage")
            for member, name in enumerate(("082", "085"), start=1):
                root = (
                    args.trained_root
                    / ("smoke" if data.smoke else "full")
                    / "nested"
                    / f"outer_{outer}"
                    / f"inner_{inner}"
                    / name
                )
                pred = root / "held.npz"
                if pred.with_suffix(".json").exists():
                    info = json.loads(pred.with_suffix(".json").read_text())
                    if info["contract"] != contract or info[
                        "prediction_sha256"
                    ] != sha256(pred):
                        raise ValueError("Nested expert partial resume differs")
                    values = load_prediction(
                        pred, {k: v[held_mask] for k, v in meta.items()}
                    )
                else:
                    cfg = OmegaConf.to_container(
                        OmegaConf.load(ROOT / "configs" / f"{name}_pretrained.yaml"),
                        resolve=True,
                    )
                    shared.progress.stage(
                        "Подготовка nested OOF",
                        f"outer {outer + 1}/3, inner {inner + 1}/2, эксперт {name}",
                    )
                    shared.progress.scope = f"inner fit {outer * 4 + inner * 2 + member}/12 · outer {outer + 1}/3 inner {inner + 1}/2 эксперт {name}"
                    model, _ = fit_one(
                        cfg,
                        shared.engine,
                        records,
                        train_mask,
                        None,
                        root,
                        cfg["training"]["epochs"],
                        contract,
                        refit=True,
                    )
                    values = tiny_predict(
                        model,
                        records,
                        shared,
                        "train",
                        held_mask,
                        folder=root / "records",
                        contract=contract,
                    )
                    save_prediction(
                        pred,
                        values,
                        {
                            "contract": contract,
                            "outer": outer,
                            "inner": inner,
                            "train_subjects": sorted(
                                set(meta["subject_id"][train_mask])
                            ),
                        },
                    )
                    del model
                    shared.engine.release()
                p[held_mask, member] = values["probabilities"]
        from src.pretrained import save_npz

        save_npz(target, experts=p, ids=meta["ids"])
        save_json(
            target.with_suffix(".json"),
            {
                "contract": contract,
                "sha256": sha256(target),
                "outer": outer,
                "semantics": "outer held uses original 082/085 OOF; outer fit uses inner2 OOF; no outer held labels in expert fitting",
            },
        )
        outputs.append(p)
    shared.progress.scope = ""
    return outputs
