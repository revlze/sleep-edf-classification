"""Verified saved sources, frozen Tiny checkpoints, shared signal/CNN caches."""

import json
from functools import cache
from pathlib import Path

import numpy as np
import scipy
import torch
from omegaconf import OmegaConf

from src.expansion_data import selected_signal
from src.models.pretrained_assets import asset_manifest, sha256
from src.models.tiny_voting import backbone_sha256
from src.models.tinysleepnet import TinySleepNetAdapter, tiny_preprocess
from src.pretrained import ROOT, save_json
from src.pretrained_signals import contiguous_segments
from src.refinement_data import subset
from src.series_data import SeriesData, fingerprint, read_legacy_019
from src.storage import atomic_path, file_lock
from src.training.expansion import parameter_manifest, restore_state


class VotingData(SeriesData):
    @cache
    def record_metadata(self, path: Path, split: str) -> dict:
        return super().record_metadata(path, split)


class VotingInputs:
    def __init__(self, roots, data):
        manifests = [
            p / "bundle.json"
            for p in roots
            if (p / "bundle.json").is_file()
            and json.loads((p / "bundle.json").read_text(encoding="utf-8")).get("kind")
            == "tiny_voting_inputs_v1"
        ]
        if len(manifests) != 1:
            raise ValueError("Attach exactly one sleep-edf-voting-inputs bundle")
        self.root, self.data = manifests[0].parent.resolve(), data
        self.manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
        if self.manifest["data"] != data.identity:
            raise ValueError("Voting Input dataset identity differs")
        self.verified, self.memo = set(), {}

    def path(self, relative):
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root) or relative not in self.manifest["files"]:
            raise ValueError(f"Unregistered voting Input file: {relative}")
        if relative not in self.verified:
            if sha256(path) != self.manifest["files"][relative]:
                raise ValueError(f"Voting Input checksum mismatch: {relative}")
            self.verified.add(relative)
        return path

    def source_json(self, name, filename):
        return json.loads(
            self.path(f"results/{name}/{filename}").read_text(encoding="utf-8")
        )

    def baseline(self, name, split):
        key = (name, split)
        if key in self.memo:
            return self.memo[key]
        meta = self.data.metadata(split)
        path = self.path(f"results/{name}/{split}_predictions.npz")
        if name == "019_pretrained" and split == "val":
            for filename in ("provenance.json", "resolved.yaml"):
                self.path(f"results/{name}/{filename}")
            result = read_legacy_019([self.root / "results"], self.data, "val")
            if result is None:
                raise ValueError("Original019 validation provenance missing")
            values = result[0]
        else:
            info = self.source_json(name, f"{split}_predictions.json")
            if info.get("data") != self.data.identity or info.get("smoke", False):
                raise ValueError(f"Not a full source prediction: {name}/{split}")
            if (
                info["prediction_sha256"]
                != self.manifest["files"][f"results/{name}/{path.name}"]
            ):
                raise ValueError("Source prediction metadata SHA differs")
            with np.load(path, allow_pickle=False) as saved:
                values = subset({k: saved[k] for k in saved.files}, meta)
        self.memo[key] = values
        return values

    def tiny(self, source, device):
        assets = self.root / "assets"
        for name in (
            "tinysleepnet.pt",
            "tinysleepnet-source.py",
            "tinysleepnet-config.json",
        ):
            self.path("assets/" + name)
        adapter = TinySleepNetAdapter(assets, assets, device)
        model = adapter.model
        provenance = {
            "source": source,
            "official_assets": asset_manifest(adapter.assets),
            "mode": "eval, no gradients",
        }
        if source != "041_pretrained":
            cfg = OmegaConf.to_container(
                OmegaConf.load(ROOT / "configs" / f"{source}.yaml"), resolve=True
            )
            original = self.source_json(source, "provenance.json")
            selected = self.source_json(source, "selection.json")
            if (
                original["config"] != cfg
                or original["data"] != self.data.identity
                or original["smoke"]
            ):
                raise ValueError("Fine-tuned source recipe differs")
            path = self.path(f"trained/{source}/fit/best.pt")
            summary = json.loads(
                self.path(f"trained/{source}/fit/training_summary.json").read_text()
            )
            if summary["checkpoint_sha256"] != sha256(path):
                raise ValueError("Fine-tuned best checkpoint SHA differs")
            if cfg["model"]["strategy"] == "classifier":
                model.requires_grad_(False)
                model.clf.clf.requires_grad_(True)
            saved = torch.load(path, map_location="cpu", weights_only=True)
            if (
                saved["contract"] != original["contract"]
                or saved["manifest"] != parameter_manifest(model)
                or saved["epoch"] != selected["refit_epochs"]
            ):
                raise ValueError(
                    "Fine-tuned checkpoint identity/parameters/refit epoch differs"
                )
            restore_state(model, saved["model"])
            provenance.update(
                checkpoint_sha256=sha256(path),
                source_contract=saved["contract"],
                epoch=saved["epoch"],
            )
        model.eval().requires_grad_(False)
        provenance["backbone_sha256"] = backbone_sha256(model)
        return model, provenance


class CNNCache:
    """Disk .npy and vectorized GPU batches; shared for identical actual backbones."""

    def __init__(self, data, root, read_roots, device, progress, batch_size=256):
        (
            self.data,
            self.root,
            self.read_roots,
            self.device,
            self.progress,
            self.batch_size,
        ) = data, root, read_roots, device, progress, batch_size
        self.mode = "smoke" if data.smoke else "full"
        self.verified = set()
        self.used = {}

    def signal(self, path, split):
        meta = self.data.record_metadata(path, split)
        contract = {
            "data": self.data.identity,
            "input": self.data.hashes[path.relative_to(self.data.root).as_posix()],
            "ids": fingerprint(meta["ids"].tolist()),
            "preprocessing": "Tiny041 BP0.3-40Hz order5, EEG Fpz-Cz uV, contiguous segments",
            "scipy": scipy.__version__,
        }
        target = (
            self.root
            / self.mode
            / "signals"
            / fingerprint(contract)
            / f"{path.stem}.npy"
        )
        with file_lock(target.with_suffix(".lock"), wait=True):
            if not target.with_suffix(".json").exists():
                self.progress.detail(
                    "сигнал: проверка manifest, распаковка NPZ и preprocessing один раз"
                )
                self.data.verify(path)
                raw = selected_signal(self.data, path, split)
                values = np.empty((len(raw), 3000), np.float32)
                for segment in contiguous_segments(meta["epoch_index"]):
                    values[segment] = tiny_preprocess(raw[segment], 0)
                with atomic_path(target) as temporary:
                    with temporary.open("wb") as stream:
                        np.save(stream, values, allow_pickle=False)
                save_json(
                    target.with_suffix(".json"),
                    {"contract": contract, "sha256": sha256(target)},
                )
        info = json.loads(target.with_suffix(".json").read_text())
        if info["contract"] != contract:
            raise ValueError("Signal cache contract differs")
        if target not in self.verified:
            if sha256(target) != info["sha256"]:
                raise ValueError("Signal cache checksum differs")
            self.verified.add(target)
        return np.load(target, mmap_mode="r", allow_pickle=False)

    def get(self, model, source, path, split):
        meta = self.data.record_metadata(path, split)
        contract = {
            "data": self.data.identity,
            "input": self.data.hashes[path.relative_to(self.data.root).as_posix()],
            "ids": fingerprint(meta["ids"].tolist()),
            "backbone": source["backbone_sha256"],
            "recipe": "tiny-original-CNN2048-eval-float32-v1",
            "torch": torch.__version__,
            "scipy": scipy.__version__,
        }
        relative = (
            Path(self.mode) / "cnn" / fingerprint(contract) / split / f"{path.stem}.npy"
        )
        # Include record contract in key: stale smoke prefixes cannot overwrite full.
        target = self.root / relative
        candidates = [*(root / relative for root in self.read_roots), target]
        existing = next(
            (p for p in candidates if p.with_suffix(".json").is_file()), None
        )
        if existing is None:
            self.progress.detail("ожидание/подготовка общего CNN-кэша")
            with file_lock(target.with_suffix(".lock"), wait=True):
                if not target.with_suffix(".json").exists():
                    signal = self.signal(path, split)
                    output = np.empty((len(signal), 2048), np.float32)
                    with torch.inference_mode():
                        for start in range(0, len(signal), self.batch_size):
                            self.progress.detail(
                                f"CNN: эпохи {start}/{len(signal)}, batch<={self.batch_size}"
                            )
                            values = torch.from_numpy(
                                np.array(
                                    signal[start : start + self.batch_size, None],
                                    copy=True,
                                )
                            ).to(self.device)
                            features = model.feature_extractor(values).float()
                            if (
                                features.shape != (len(values), 2048)
                                or not torch.isfinite(features).all()
                            ):
                                raise ValueError("Invalid CNN features")
                            output[start : start + len(values)] = features.cpu().numpy()
                    with atomic_path(target) as temporary:
                        with temporary.open("wb") as stream:
                            np.save(stream, output, allow_pickle=False)
                    save_json(
                        target.with_suffix(".json"),
                        {
                            "contract": contract,
                            "sha256": sha256(target),
                            "shape": list(output.shape),
                        },
                    )
            existing = target
        info = json.loads(existing.with_suffix(".json").read_text())
        if info["contract"] != contract:
            raise ValueError("CNN cache contract differs")
        if existing not in self.verified:
            if sha256(existing) != info["sha256"]:
                raise ValueError("CNN cache payload checksum differs")
            self.verified.add(existing)
        values = np.load(existing, mmap_mode="r", allow_pickle=False)
        if values.shape != (len(meta["ids"]), 2048):
            raise ValueError("CNN cache epoch/feature dimensions differ")
        self.used[str(existing)] = {
            "sha256": info["sha256"],
            "backbone": contract["backbone"],
        }
        return values
