"""Shared DAG of frozen predictions: each checkpoint/split is cached independently."""

import hashlib
import io
import json
import time
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch

from src.data import load_record
from src.evaluation import validate_probabilities
from src.models.anysleep_variants import AnySleepVariant, aggregate_high_frequency
from src.models.pretrained_assets import asset_manifest, sha256
from src.models.tinysleepnet import TinySleepNetAdapter, tiny_preprocess
from src.postprocessing import weighted_probabilities
from src.pretrained_signals import contiguous_segments, epoch_blocks, preprocess_segment
from src.series_data import (
    SeriesData,
    fingerprint,
    load_prediction,
    read_legacy_019,
    save_prediction,
)
from src.storage import file_lock


def base_member(run: int) -> dict:
    return {
        "kind": "anysleep",
        "variant": "base",
        "checkpoint": f"anysleep-run{run}.pth",
        "frequency": 1,
        "voting": "soft",
    }


def base_recipe() -> dict:
    return {"kind": "baseline019"}


class PredictionEngine:
    def __init__(
        self,
        data: SeriesData,
        root: Path,
        checkpoint_cache: Path,
        checkpoint_input: Path | None,
        result_inputs: list[Path],
        device: str,
    ):
        self.data, self.root = data, root
        self.checkpoint_cache, self.checkpoint_input = (
            checkpoint_cache,
            checkpoint_input,
        )
        self.result_inputs, self.device = result_inputs, device
        self.progress = None
        self.models = {}
        self.trained = {}
        self.used = {}
        self.peak_gpu_bytes = 0
        self.source_files = [
            Path(__file__),
            Path(__file__).with_name("series_data.py"),
            Path(__file__).with_name("pretrained_signals.py"),
            Path(__file__).parent / "models/anysleep_variants.py",
            Path(__file__).parent / "models/tinysleepnet.py",
            Path(__file__).with_name("series_assets.py"),
            Path(__file__).with_name("postprocessing.py"),
            Path(__file__).with_name("data.py"),
            Path(__file__).parent / "models/anysleep.py",
            Path(__file__).parent / "models/pretrained_assets.py",
        ]
        self.implementation = fingerprint(
            {
                p.name: hashlib.sha256(
                    p.read_bytes().replace(b"\r\n", b"\n")
                ).hexdigest()
                for p in self.source_files
            }
        )

    def import_cached(self, key, recipe, split, metadata):
        relative = f"{'smoke/' if self.data.smoke else ''}predictions/{key}/{split}"
        for root in self.result_inputs:
            if root.is_dir() and (root / f"{relative}.json").exists():
                info = json.loads((root / f"{relative}.json").read_text())
                payload = (root / f"{relative}.npz").read_bytes()
            elif root.is_file() and root.suffix == ".zip":
                with ZipFile(root) as archive:
                    if f"{relative}.json" not in archive.namelist():
                        continue
                    info = json.loads(archive.read(f"{relative}.json"))
                    payload = archive.read(f"{relative}.npz")
            else:
                continue
            if (
                info["recipe"] != recipe
                or info["data"] != self.data.identity
                or info["implementation"] != self.implementation
                or info["prediction_sha256"] != hashlib.sha256(payload).hexdigest()
            ):
                raise ValueError("Input probability cache contract/checksum differs")
            from src.series_data import align_prediction

            with np.load(io.BytesIO(payload), allow_pickle=False) as stored:
                values = {k: stored[k] for k in stored.files}
            return align_prediction(values, metadata), {
                **info,
                "reused_input": str(root),
            }
        return None

    def key(self, recipe: dict) -> str:
        return fingerprint(
            {
                "recipe": recipe,
                "data": self.data.identity,
                "smoke": self.data.smoke,
                "smoke_epochs": self.data.smoke_epochs if self.data.smoke else None,
                "implementation": self.implementation,
            }
        )[:24]

    def detail(self, value: str):
        if self.progress is not None:
            self.progress.update(detail=value)

    def get(self, recipe: dict, split: str) -> dict:
        key = self.key(recipe)
        folder = self.root / "predictions" / key
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / f"{split}.npz"
        metadata = self.data.metadata(split)
        with file_lock(folder / f"{split}.lock", wait=True):
            if target.with_suffix(".json").exists():
                info = json.loads(target.with_suffix(".json").read_text())
                if (
                    info["prediction_sha256"] != sha256(target)
                    or info["recipe"] != recipe
                    or info["data"] != self.data.identity
                ):
                    raise ValueError("Cached probability contract/checksum differs")
                self.used[f"{key}/{split}"] = info
                return load_prediction(target, metadata)
            self.detail(f"{split}: {recipe.get('checkpoint', recipe['kind'])}")
            started = time.perf_counter()
            kind = recipe["kind"]
            extra = {}
            imported = self.import_cached(key, recipe, split, metadata)
            if imported is not None:
                values, extra = imported
            elif kind == "baseline019":
                legacy = read_legacy_019(self.result_inputs, self.data, split)
                if legacy is not None:
                    values, extra = legacy
                else:
                    members = [self.get(base_member(i), split) for i in (1, 2, 3)]
                    values = {
                        **metadata,
                        # Preserve original 019 float32 mean, including test extension.
                        "probabilities": np.mean(
                            [m["probabilities"] for m in members], axis=0
                        ),
                    }
            elif kind == "mean":
                members = [self.get(member, split) for member in recipe["members"]]
                values = {
                    **metadata,
                    "probabilities": weighted_probabilities(
                        [m["probabilities"] for m in members], recipe["weights"]
                    ),
                }
            elif kind == "anysleep" and recipe.get("voting") == "hard":
                soft_recipe = {**recipe, "voting": "soft"}
                soft_values = self.get(soft_recipe, split)
                values = {
                    **metadata,
                    "probabilities": soft_values["hard_probabilities"],
                }
                extra = {"raw_inference_reused_from_soft_node": self.key(soft_recipe)}
            elif kind in ("anysleep", "tiny", "trained_anysleep"):
                values, extra = self._backbone(recipe, split, folder)
            else:
                raise ValueError(f"Unknown prediction node: {kind}")
            info = {
                "recipe": recipe,
                "data": self.data.identity,
                "implementation": self.implementation,
                "seconds_this_request": time.perf_counter() - started,
                "device": self.device,
                **extra,
            }
            save_prediction(target, values, info)
            self.used[f"{key}/{split}"] = json.loads(
                target.with_suffix(".json").read_text()
            )
            return values

    def _backbone(self, recipe: dict, split: str, folder: Path) -> tuple[dict, dict]:
        metadata = self.data.metadata(split)
        if self.progress is not None:
            self.progress.add_total(len(metadata["ids"]))
        key = self.key(recipe)
        model = None
        results, records = [], []
        for path in self.data.paths(split):
            record_metadata = self.data.record_metadata(path, split)
            raw_hash = self.data.verify(path)
            target = folder / "records" / split / f"{path.stem}.npz"
            info_path = target.with_suffix(".json")
            if info_path.exists():
                info = json.loads(info_path.read_text())
                if info["input_sha256"] != raw_hash or info[
                    "prediction_sha256"
                ] != sha256(target):
                    raise ValueError("Per-record prediction checksum differs")
                result = load_prediction(target, record_metadata)
                if self.progress is not None:
                    self.progress.update(len(result["ids"]))
            else:
                if model is None:
                    if recipe["kind"] == "trained_anysleep":
                        model = self.trained[recipe["weights_sha256"]]
                    elif key not in self.models:
                        self.models[key] = (
                            TinySleepNetAdapter(
                                self.checkpoint_cache,
                                self.checkpoint_input,
                                self.device,
                            )
                            if recipe["kind"] == "tiny"
                            else AnySleepVariant(
                                recipe,
                                self.checkpoint_cache,
                                self.checkpoint_input,
                                self.device,
                            )
                        )
                    if model is None:
                        model = self.models[key]
                self.detail(
                    f"{split}: {recipe.get('checkpoint', recipe['kind'])}; {path.stem}"
                )
                raw = load_record(path)
                n = len(record_metadata["ids"])
                x, indices = raw["x"][:n], raw["epoch_index"][:n]
                prob = np.empty((n, 5), dtype=np.float32)
                hard = np.empty_like(prob) if recipe.get("frequency", 1) > 1 else None
                if self.device.startswith("cuda"):
                    torch.cuda.synchronize(self.device)
                    torch.cuda.reset_peak_memory_stats(self.device)
                started = time.perf_counter()
                for segment in contiguous_segments(indices):
                    if recipe["kind"] == "tiny":
                        signal = tiny_preprocess(x[segment], recipe["channel"])
                        prob[segment] = model.probabilities(signal)
                        if self.progress is not None:
                            self.progress.update(segment.stop - segment.start)
                    else:
                        signal = preprocess_segment(x[segment], "anysleep")
                        count = segment.stop - segment.start
                        for a, b, left, right in epoch_blocks(
                            count, 128, 64, right_alignment=16
                        ):
                            chunk = signal[a * 3840 : b * 3840]
                            if hard is not None:
                                logits = model.logits(chunk)
                                p = aggregate_high_frequency(
                                    logits, recipe["frequency"], "soft"
                                )
                                votes = aggregate_high_frequency(
                                    logits, recipe["frequency"], "hard"
                                )
                                hard[segment.start + left : segment.start + right] = (
                                    votes[left - a : right - a]
                                )
                            else:
                                p = model.probabilities(chunk)
                            validate_probabilities(p, b - a)
                            prob[segment.start + left : segment.start + right] = p[
                                left - a : right - a
                            ]
                            if self.progress is not None:
                                self.progress.update(right - left)
                if self.device.startswith("cuda"):
                    torch.cuda.synchronize(self.device)
                assets = (
                    asset_manifest(model.assets) if hasattr(model, "assets") else []
                )
                info = {
                    "input_sha256": raw_hash,
                    "seconds": time.perf_counter() - started,
                    "peak_gpu_bytes": torch.cuda.max_memory_allocated(self.device)
                    if self.device.startswith("cuda")
                    else None,
                    "assets": assets,
                    "device": self.device,
                    "recipe": recipe,
                    "architecture_parameters": getattr(
                        model.model, "released_architecture_parameters", {}
                    ),
                }
                self.peak_gpu_bytes = max(
                    self.peak_gpu_bytes, info["peak_gpu_bytes"] or 0
                )
                result = {**record_metadata, "probabilities": prob}
                if hard is not None:
                    result["hard_probabilities"] = hard
                save_prediction(target, result, info)
                info = json.loads(info_path.read_text())
                del raw, x
            results.append(result)
            records.append({"record_id": path.stem, **info})
        values = {k: np.concatenate([r[k] for r in results]) for k in results[0]}
        return values, {
            "records": records,
            "inference_seconds": sum(r["seconds"] for r in records),
            "peak_gpu_bytes": max((r["peak_gpu_bytes"] or 0) for r in records),
        }

    def release_models(self):
        self.models.clear()
        self.trained.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
