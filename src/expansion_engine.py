"""New frozen inference and true pre-classifier embeddings, with record resume."""

import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch

from src.expansion_assets import MANIFEST
from src.expansion_data import selected_signal
from src.models.anysleep_variants import AnySleepVariant
from src.models.expansion_official import (
    RobustAdapter,
    SleePyCoAdapter,
    SpectralAdapter,
)
from src.models.pretrained_assets import sha256
from src.models.tinysleepnet import TinySleepNetAdapter, tiny_preprocess
from src.pretrained import save_json, save_npz
from src.pretrained_signals import contiguous_segments, epoch_blocks, preprocess_segment
from src.series_data import (
    align_prediction,
    fingerprint,
    load_prediction,
    save_prediction,
)
from src.series_engine import PredictionEngine, base_member, base_recipe
from src.storage import file_lock

TINY = {"kind": "tiny", "channel": 0, "checkpoint": "tinysleepnet.pt"}
BASE = {"kind": "mean", "members": [base_recipe(), TINY], "weights": [0.5, 0.5]}


def implementation() -> str:
    files = [
        *Path(__file__).parent.glob("expansion_*.py"),
        Path(__file__).parent / "models/expansion_official.py",
        Path(__file__).parent / "models/sleepgpt.py",
        Path(__file__).parent / "training/expansion.py",
        *[
            Path(__file__).parent / name
            for name in (
                "series_data.py",
                "series_engine.py",
                "pretrained_signals.py",
                "models/tinysleepnet.py",
                "models/anysleep_variants.py",
                "models/anysleep.py",
                "models/pretrained_assets.py",
                "series_assets.py",
                "evaluation.py",
                "data.py",
            )
        ],
        Path(__file__).parents[1] / "environments/pretrained/uv.lock",
    ]
    return fingerprint(
        {
            "files": {
                p.name: hashlib.sha256(
                    p.read_bytes().replace(b"\r\n", b"\n")
                ).hexdigest()
                for p in files
                if p.exists()
            },
            "assets": MANIFEST,
        }
    )


class ExpansionEngine:
    def __init__(
        self,
        data,
        output: Path,
        cache: Path,
        checkpoints: Path,
        inputs: Path | None,
        results: list[Path],
        device: str,
    ):
        self.data, self.root, self.cache = data, output, cache
        self.checkpoints, self.inputs, self.results, self.device = (
            checkpoints,
            inputs,
            results,
            device,
        )
        self.legacy = PredictionEngine(
            data, output, checkpoints, inputs, results, device
        )
        self.progress = None
        self.failed_models = {}
        self.implementation = implementation()

    def detail(self, value: str):
        if self.progress is not None:
            self.progress.update(detail=value)

    def baseline(self, name: str, split: str) -> dict:
        recipes = {"019": base_recipe(), "041": TINY, "053": BASE}
        recipe = recipes[name]
        full_key = fingerprint(
            {
                "recipe": recipe,
                "data": self.data.identity,
                "smoke": False,
                "smoke_epochs": None,
                "implementation": self.legacy.implementation,
            }
        )[:24]
        locations = [
            f"{name}_pretrained/{split}_predictions",
            f"predictions/{full_key}/{split}",
        ]
        metadata = self.data.metadata(split)
        for root in self.results:
            payload = info = None
            for relative in locations:
                if root.is_dir() and (root / f"{relative}.json").exists():
                    info = json.loads((root / f"{relative}.json").read_text())
                    payload = (root / f"{relative}.npz").read_bytes()
                elif root.is_file() and root.suffix == ".zip":
                    with ZipFile(root) as zipped:
                        if f"{relative}.json" in zipped.namelist():
                            info = json.loads(zipped.read(f"{relative}.json"))
                            payload = zipped.read(f"{relative}.npz")
                if payload is not None:
                    break
            if payload is None:
                continue
            if (
                info.get("data") != self.data.identity
                or info.get("recipe") != recipe
                or info.get("prediction_sha256") != hashlib.sha256(payload).hexdigest()
            ):
                raise ValueError(f"{name} cached baseline contract differs: {root}")
            with np.load(io.BytesIO(payload), allow_pickle=False) as stored:
                values = {k: stored[k] for k in stored.files}
            if self.data.smoke:
                wanted = set(metadata["ids"])
                select = np.array([str(n) in wanted for n in values["ids"]])
                values = {k: v[select] for k, v in values.items()}
            self.detail(f"{name} {split}: reuse verified Input")
            return align_prediction(values, metadata)
        if name == "053":
            a, b = self.baseline("019", split), self.baseline("041", split)
            return {
                **metadata,
                "probabilities": (a["probabilities"] + b["probabilities"]) / 2,
            }
        if self.data.smoke and split == "train":
            return self.small_baseline(name, split)
        self.legacy.progress = self.progress
        return self.legacy.get(recipe, split)

    def small_baseline(self, name, split):
        """Original frozen models on the same real non-prefix training smoke."""
        meta = self.data.metadata(split)
        contract = {
            "name": name,
            "data": self.data.identity,
            "ids": fingerprint(meta["ids"].tolist()),
            "implementation": self.implementation,
            "smoke": True,
        }
        target = (
            self.root / "expansion_baselines" / fingerprint(contract) / f"{split}.npz"
        )
        with file_lock(target.with_suffix(".lock"), wait=True):
            if target.with_suffix(".json").exists():
                if json.loads(target.with_suffix(".json").read_text())[
                    "prediction_sha256"
                ] != sha256(target):
                    raise ValueError("Smoke baseline cache differs")
                return load_prediction(target, meta)
            if name == "019":
                sums = []
                for run in (1, 2, 3):
                    model = AnySleepVariant(
                        base_member(run), self.checkpoints, self.inputs, self.device
                    )
                    sums.append(self.baseline_records(model, split, "anysleep"))
                    del model
                p = np.mean(sums, axis=0)
            elif name == "041":
                model = TinySleepNetAdapter(self.checkpoints, self.inputs, self.device)
                p = self.baseline_records(model, split, "tiny")
                del model
            else:
                raise ValueError(name)
            values = {**meta, "probabilities": p}
            save_prediction(target, values, contract)
            self.release()
            return values

    def baseline_records(self, model, split, family):
        outputs = []
        for path in self.data.paths(split):
            meta = self.data.record_metadata(path, split)
            raw = selected_signal(self.data, path, split)
            p = np.empty((len(raw), 5), dtype=np.float32)
            for part in contiguous_segments(meta["epoch_index"]):
                self.detail(f"train smoke baseline {family}: {path.stem}")
                if family == "tiny":
                    p[part] = model.probabilities(tiny_preprocess(raw[part], 0))
                else:
                    signal = preprocess_segment(raw[part], "anysleep")
                    for a, b, left, right in epoch_blocks(
                        len(raw[part]), 128, 64, right_alignment=16
                    ):
                        chunk = model.probabilities(signal[a * 3840 : b * 3840])
                        p[part.start + left : part.start + right] = chunk[
                            left - a : right - a
                        ]
            outputs.append(p)
        return np.concatenate(outputs)

    def release(self):
        self.legacy.release_models()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def frozen(self, recipe: dict, split: str) -> dict:
        key = fingerprint(recipe)
        if key in self.failed_models:
            raise RuntimeError(
                f"This source already failed in this process: {self.failed_models[key]}"
            )
        try:
            return self._frozen(recipe, split)
        except Exception as error:
            self.failed_models[key] = repr(error)
            raise

    def _frozen(self, recipe: dict, split: str) -> dict:
        contract = {
            "recipe": recipe,
            "data": self.data.identity,
            "implementation": self.implementation,
            "smoke": self.data.smoke,
            "limit": self.data.smoke_epochs if self.data.smoke else None,
        }
        folder = self.root / "expansion_predictions" / fingerprint(contract)[:24]
        merged = folder / f"{split}.npz"
        with file_lock(merged.with_suffix(".lock"), wait=True):
            if merged.with_suffix(".json").exists():
                if json.loads(merged.with_suffix(".json").read_text())[
                    "prediction_sha256"
                ] != sha256(merged):
                    raise ValueError("Frozen cache checksum mismatch")
                return load_prediction(merged, self.data.metadata(split))
            name = recipe["name"]
            self.detail(f"{name}: download/load original checkpoint")
            if name.startswith("sleepyco_"):
                model = SleePyCoAdapter(
                    name.split("_")[1], self.checkpoints, self.inputs, self.device
                )
            elif name == "robust":
                model = RobustAdapter(self.checkpoints, self.inputs, self.device)
            else:
                model = SpectralAdapter(
                    name, self.checkpoints, self.inputs, self.device
                )
            records = []
            for path in self.data.paths(split):
                metadata = self.data.record_metadata(path, split)
                target = folder / "records" / split / f"{path.stem}.npz"
                if target.with_suffix(".json").exists():
                    info = json.loads(target.with_suffix(".json").read_text())
                    if info["prediction_sha256"] != sha256(target) or info[
                        "input"
                    ] != self.data.verify(path):
                        raise ValueError("Record cache checksum mismatch")
                    values = load_prediction(target, metadata)
                else:
                    self.detail(f"{name} {split}: {path.stem}")
                    raw = selected_signal(self.data, path, split)
                    p = np.empty((len(raw), 5), dtype=np.float32)
                    for segment in contiguous_segments(metadata["epoch_index"]):
                        p[segment] = model.probabilities(raw[segment], self.progress)
                    values = {**metadata, "probabilities": p}
                    save_prediction(
                        target,
                        values,
                        {
                            **contract,
                            "input": self.data.verify(path),
                            "model": model.provenance,
                        },
                    )
                    del raw
                records.append(values)
                if self.progress is not None:
                    self.progress.update(len(metadata["ids"]))
            values = {k: np.concatenate([r[k] for r in records]) for k in records[0]}
            save_prediction(merged, values, {**contract, "model": model.provenance})
            del model
            self.release()
            return values

    def embeddings(self, family: str, split: str) -> np.ndarray:
        from src.models.pretrained_assets import asset_manifest

        assets = (
            ["tinysleepnet.pt", "tinysleepnet-source.py", "tinysleepnet-config.json"]
            if family == "tiny"
            else ["anysleep-run1.pth", "anysleep_no_hydra.py"]
        )
        contract = {
            "family": family,
            "data": self.data.identity,
            "implementation": self.implementation,
            "assets": asset_manifest(assets),
            "layer": "clf.encode contextual128 rolling20"
            if family == "tiny"
            else "segment_classifier.segment_classifier.2 AvgPool before 5class Conv",
            "smoke": self.data.smoke,
            "limit": self.data.smoke_epochs if self.data.smoke else None,
        }
        root = self.cache / "embeddings" / fingerprint(contract)
        outputs, model = [], None
        for path in self.data.paths(split):
            metadata = self.data.record_metadata(path, split)
            target = root / split / f"{path.stem}.npz"
            with file_lock(target.with_suffix(".lock"), wait=True):
                if not target.exists():
                    if model is None:
                        model = (
                            TinySleepNetAdapter(
                                self.checkpoints, self.inputs, self.device
                            )
                            if family == "tiny"
                            else AnySleepVariant(
                                base_member(1),
                                self.checkpoints,
                                self.inputs,
                                self.device,
                            )
                        )
                    raw = selected_signal(self.data, path, split)
                    chunks = []
                    for segment in contiguous_segments(metadata["epoch_index"]):
                        self.detail(
                            f"{family} embeddings {split}: {path.stem}, {segment.start}/{len(raw)}"
                        )
                        chunks.append(
                            tiny_embeddings(model, tiny_preprocess(raw[segment], 0))
                            if family == "tiny"
                            else anysleep_embeddings(
                                model, preprocess_segment(raw[segment], "anysleep")
                            )
                        )
                    embedding = np.concatenate(chunks)
                    save_npz(target, ids=metadata["ids"], embeddings=embedding)
                    save_json(
                        target.with_suffix(".json"),
                        {
                            **contract,
                            "input": self.data.verify(path),
                            "sha256": sha256(target),
                        },
                    )
                    del raw
                info = json.loads(target.with_suffix(".json").read_text())
                if info["sha256"] != sha256(target) or info[
                    "input"
                ] != self.data.verify(path):
                    raise ValueError("Embedding cache checksum differs")
                with np.load(target, allow_pickle=False) as stored:
                    if not np.array_equal(stored["ids"], metadata["ids"]):
                        raise ValueError("Embedding IDs differ")
                    outputs.append(stored["embeddings"])
            if self.progress is not None:
                self.progress.update(len(metadata["ids"]))
        del model
        self.release()
        return np.concatenate(outputs)


def tiny_embeddings(adapter, epochs: np.ndarray) -> np.ndarray:
    features = []
    model, device = adapter.model, adapter.device
    with torch.inference_mode():
        for start in range(0, len(epochs), 64):
            features.append(
                model.feature_extractor(
                    torch.from_numpy(epochs[start : start + 64].copy())[:, None].to(
                        device
                    )
                )
                .float()
                .cpu()
            )
        features = torch.cat(features)
        width = min(20, len(epochs))
        output = [
            model.clf.encode(features[:width].unsqueeze(0).to(device))[0]
            .float()
            .cpu()
            .numpy()
        ]
        for start in range(20, len(epochs), 32):
            ends = range(start, min(len(epochs), start + 32))
            windows = torch.stack([features[end - 19 : end + 1] for end in ends]).to(
                device
            )
            output.append(model.clf.encode(windows)[:, -1].float().cpu().numpy())
    return np.concatenate(output)


def anysleep_embeddings(adapter, signal: np.ndarray) -> np.ndarray:
    captured = []
    layer = adapter.model.segment_classifier.segment_classifier[2]
    handle = layer.register_forward_hook(
        lambda module, args, result: captured.append(
            result.detach().float().cpu().numpy()
        )
    )
    parts = []
    try:
        for a, b, left, right in epoch_blocks(
            len(signal) // 3840, 128, 64, right_alignment=16
        ):
            captured.clear()
            adapter.logits(signal[a * 3840 : b * 3840])
            if (
                len(captured) != 1
                or captured[0].shape[0] != 1
                or captured[0].shape[2] != b - a
                or captured[0].shape[1] <= 5
            ):
                raise ValueError("AnySleep internal epoch embedding shape differs")
            parts.append(captured[0][0, :, left - a : right - a].T.copy())
    finally:
        handle.remove()
    return np.concatenate(parts)
