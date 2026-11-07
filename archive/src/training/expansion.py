"""Bounded grouped fine-tuning from verified weights, resumable best/last and refit."""

import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.evaluation import classification_metrics
from src.expansion_data import RecordCache, shared_folds
from src.models.pretrained_assets import sha256
from src.models.sleepgpt import SleepGPTStager
from src.models.tinysleepnet import TinySleepNetAdapter
from src.pretrained import save_json
from src.pretrained_signals import contiguous_segments
from src.series_data import fingerprint, load_prediction, save_prediction
from src.storage import atomic_path, file_lock


def native(value):
    """Checkpoint metadata must remain weights_only=True compatible."""
    if isinstance(value, dict):
        return {
            k.item() if isinstance(k, np.generic) else k: native(v)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [native(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_checkpoint(path: Path, payload: dict):
    with atomic_path(path) as temporary:
        torch.save(native(payload), temporary)


def loss_values(logits, labels, weights=None, gamma=0.0):
    flat, y = logits.reshape(-1, 5).float(), labels.reshape(-1)
    loss = F.cross_entropy(flat, y, weight=weights, reduction="none")
    if gamma:
        pt = flat.softmax(-1).gather(1, y[:, None]).squeeze(1)
        loss = (1 - pt).pow(gamma) * loss
    return loss.mean()


def parameter_manifest(model) -> dict:
    rows = [
        {"name": n, "shape": list(p.shape), "parameters": p.numel()}
        for n, p in model.named_parameters()
        if p.requires_grad
    ]
    return {
        "total": sum(p.numel() for p in model.parameters()),
        "trainable": sum(r["parameters"] for r in rows),
        "entries": rows,
    }


def trainable_state(model):
    # Include all buffers (e.g. BatchNorm); frozen parameters are reconstructed from
    # the pinned original checkpoint. This keeps SleepGPT last.pt below 2GB.
    names = {n for n, p in model.named_parameters() if p.requires_grad} | {
        n for n, _ in model.named_buffers()
    }
    return {n: v.detach().cpu() for n, v in model.state_dict().items() if n in names}


def restore_state(model, state):
    expected = set(trainable_state(model))
    if set(state) != expected:
        raise ValueError("Checkpoint trainable parameter/buffer names differ")
    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys or any(n in expected for n in result.missing_keys):
        raise ValueError("Incomplete trained state")


class TrainingRecords:
    def __init__(self, data, cache: Path, family: str):
        self.data, self.family = data, family
        self.cache = RecordCache(data, cache)
        self.rows = []
        offset = 0
        for path in data.paths("train"):
            metadata = data.record_metadata(path, "train")
            self.rows.append((path, metadata, offset))
            offset += len(metadata["ids"])
        self.current = None

    def signal(self, path, split="train"):
        key = (path, split)
        if self.current != key:
            self.array = self.cache.get(path, split, self.family)
            self.current = key
        return self.array

    def ends(self, permitted: np.ndarray):
        candidates = []
        width = 20 if self.family == "tiny" else 1
        for record, (_, meta, offset) in enumerate(self.rows):
            for part in contiguous_segments(meta["epoch_index"]):
                for end in range(part.start + width - 1, part.stop):
                    if permitted[offset + end] and np.all(
                        permitted[offset + end - width + 1 : offset + end + 1]
                    ):
                        candidates.append((record, end))
        if not candidates:
            raise ValueError("No complete training windows for this fold")
        return np.asarray(candidates, dtype=np.int64)

    def batch(self, pairs: np.ndarray):
        x, y = [], []
        width = 20 if self.family == "tiny" else 1
        for record, end in pairs:
            path, meta, _ = self.rows[int(record)]
            signal = self.signal(path)
            if self.family == "tiny":
                x.append(np.array(signal[end - width + 1 : end + 1, None], copy=True))
                y.append(meta["labels"][end - width + 1 : end + 1])
            else:
                x.append(np.array(signal[end], copy=True))
                y.append(meta["labels"][end])
        return torch.from_numpy(np.stack(x)), torch.as_tensor(
            np.asarray(y), dtype=torch.long
        )


def make_model(config: dict, engine):
    family = config["model"]["kind"]
    if family == "tiny_finetune":
        adapter = TinySleepNetAdapter(engine.checkpoints, engine.inputs, engine.device)
        model = adapter.model
        if config["model"]["strategy"] == "classifier":
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            for parameter in model.clf.clf.parameters():
                parameter.requires_grad_(True)
        return model
    return SleepGPTStager(
        engine.checkpoints, engine.inputs, config["model"]["strategy"]
    ).to(engine.device)


def training_mode(model, config):
    model.train()
    if config["model"]["kind"] == "tiny_finetune":
        if config["model"]["strategy"] == "classifier":
            model.feature_extractor.eval()
            model.clf.rnn.eval()
            model.clf.rnn_dropout.eval()
        # Small batches: keep published running statistics, allow affine gradients
        # in full FT. This is an explicit downstream recipe, not author retraining.
        for layer in model.modules():
            if isinstance(layer, nn.modules.batchnorm._BatchNorm):
                layer.eval()


def infer_record(model, signal, family: str, device: str, batch_size=16) -> np.ndarray:
    model.eval()
    result = np.empty((len(signal), 5), dtype=np.float32)
    with torch.inference_mode():
        if family == "tiny":
            features = []
            for start in range(0, len(signal), 64):
                x = torch.from_numpy(
                    np.array(signal[start : start + 64, None], copy=True)
                ).to(device)
                features.append(model.feature_extractor(x).float().cpu())
            features = torch.cat(features)
            width = min(len(signal), 20)
            result[:width] = (
                model.clf(features[:width].unsqueeze(0).to(device))[0]
                .float()
                .softmax(-1)
                .cpu()
                .numpy()
            )
            for start in range(20, len(signal), batch_size):
                ends = range(start, min(len(signal), start + batch_size))
                x = torch.stack([features[e - 19 : e + 1] for e in ends]).to(device)
                result[list(ends)] = (
                    model.clf(x)[:, -1].float().softmax(-1).cpu().numpy()
                )
        else:
            for start in range(0, len(signal), batch_size):
                x = torch.from_numpy(
                    np.array(signal[start : start + batch_size], copy=True)
                ).to(device)
                with torch.autocast(
                    device_type=torch.device(device).type,
                    enabled=device.startswith("cuda"),
                ):
                    result[start : start + len(x)] = (
                        model(x).float().softmax(-1).cpu().numpy()
                    )
    return result


def predict(
    model,
    records: TrainingRecords,
    engine,
    split: str,
    permitted=None,
    per_record_limit=None,
    resume_folder: Path | None = None,
    resume_contract: str | None = None,
) -> dict:
    parts = []
    metadata = engine.data.metadata(split)
    offset = 0
    for path in engine.data.paths(split):
        meta = engine.data.record_metadata(path, split)
        n = len(meta["ids"])
        if permitted is not None and not np.any(permitted[offset : offset + n]):
            offset += n
            continue
        target = (
            resume_folder / f"{path.stem}.npz" if resume_folder is not None else None
        )
        if target is not None and target.with_suffix(".json").exists():
            info = json.loads(target.with_suffix(".json").read_text())
            if (
                per_record_limit is not None
                or info["contract"] != resume_contract
                or info["input"] != engine.data.verify(path)
                or info["prediction_sha256"] != sha256(target)
            ):
                raise ValueError("Trained record prediction resume differs")
            parts.append(load_prediction(target, meta))
            offset += n
            engine.detail(f"{split}: reuse trained record {path.stem}")
            continue
        signal = records.signal(path, split)
        p = np.empty((n, 5), dtype=np.float32)
        selected = np.zeros(n, dtype=bool)
        for part in contiguous_segments(meta["epoch_index"]):
            engine.detail(f"{split}: {path.stem}, {part.start}/{n}")
            cores = [(part.start, part.stop)]
            if per_record_limit and n > per_record_limit:
                quota = max(1, int(per_record_limit * (part.stop - part.start) / n))
                chunks = min(4, max(1, quota // 16))
                width = min(part.stop - part.start, max(1, quota // chunks))
                starts = np.unique(
                    np.linspace(part.start, part.stop - width, chunks, dtype=int)
                )
                cores = [(int(a), int(a) + width) for a in starts]
            for left, right in cores:
                context_left = (
                    max(part.start, left - 19) if records.family == "tiny" else left
                )
                out = infer_record(
                    model,
                    signal[context_left:right],
                    records.family,
                    engine.device,
                    batch_size=16 if records.family == "tiny" else 2,
                )
                p[left:right] = out[left - context_left :]
                selected[left:right] = True
        values = {
            **{k: v[selected] for k, v in meta.items()},
            "probabilities": p[selected],
        }
        parts.append(values)
        if target is not None:
            if per_record_limit is not None or resume_contract is None:
                raise ValueError("Only complete record predictions may resume")
            save_prediction(
                target,
                values,
                {"contract": resume_contract, "input": engine.data.verify(path)},
            )
        offset += n
    if not parts:
        raise ValueError("Empty evaluation subset")
    values = {k: np.concatenate([part[k] for part in parts]) for k in parts[0]}
    if (
        per_record_limit is None
        and permitted is None
        and not np.array_equal(values["ids"], metadata["ids"])
    ):
        raise ValueError("Inference ordering mismatch")
    return values


def fit_one(
    config,
    engine,
    records,
    train_mask,
    held_mask,
    folder: Path,
    epochs: int,
    contract: str,
    refit=False,
):
    settings = config["training"]
    engine.detail(
        f"{config['experiment_id']} {folder.name}: load original weights / resume"
    )
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    model = make_model(config, engine)
    manifest = parameter_manifest(model)
    save_json(folder / "trainable_parameters.json", manifest)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=settings["lr"],
        weight_decay=settings["weight_decay"],
    )
    scaler = torch.amp.GradScaler("cuda", enabled=engine.device.startswith("cuda"))
    generator = torch.Generator().manual_seed(config["seed"])
    ends = records.ends(train_mask)
    labels = engine.data.metadata("train")["labels"][train_mask]
    weights = None
    if settings["class_weight"] == "sqrt":
        counts = np.bincount(labels, minlength=5)
        w = np.maximum(counts, 1) ** -0.5
        weights = torch.tensor(w / w.mean(), dtype=torch.float32, device=engine.device)
    start, history, best, best_epoch, stale = 0, [], -1.0, 0, 0
    last, best_path = folder / "last.pt", folder / "best.pt"
    if last.exists():
        saved = torch.load(last, map_location=engine.device, weights_only=True)
        if saved["contract"] != contract or saved["manifest"] != manifest:
            raise ValueError(
                "Training resume contract differs; use an isolated output root"
            )
        restore_state(model, saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        generator.set_state(saved["generator"].cpu())
        torch.set_rng_state(saved["torch_rng"].cpu())
        if saved["cuda_rng"] is not None and engine.device.startswith("cuda"):
            torch.cuda.set_rng_state(saved["cuda_rng"].cpu(), engine.device)
        start, history, best, best_epoch, stale = (
            saved["epoch"],
            saved["history"],
            saved["best"],
            saved["best_epoch"],
            saved["stale"],
        )
    steps = 2 if engine.data.smoke else settings["steps_per_epoch"]
    epochs = 1 if engine.data.smoke else epochs
    for epoch in range(start, epochs):
        if not refit and stale >= settings["patience"]:
            break
        epoch_started = time.perf_counter()
        training_mode(model, config)
        total_loss = 0.0
        for step in range(steps):
            engine.detail(
                f"{config['experiment_id']} {folder.name}: epoch {epoch + 1}/{epochs}, optimizer step {step + 1}/{steps}, preparing batch / training"
            )
            optimizer.zero_grad(set_to_none=True)
            for _ in range(settings["accumulation"]):
                sample = torch.randint(
                    len(ends), (settings["batch_size"],), generator=generator
                ).numpy()
                x, y = records.batch(ends[sample])
                x, y = x.to(engine.device), y.to(engine.device)
                with torch.autocast(
                    device_type=torch.device(engine.device).type,
                    enabled=engine.device.startswith("cuda"),
                ):
                    logits = model(x)
                    loss = (
                        loss_values(logits, y, weights, settings["focal_gamma"])
                        / settings["accumulation"]
                    )
                scaler.scale(loss).backward()
                total_loss += loss.item()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                settings["grad_clip"],
            )
            scaler.step(optimizer)
            scaler.update()
            engine.detail(
                f"{config['experiment_id']} {folder.name}: epoch {epoch + 1}/{epochs}, optimizer step {step + 1}/{steps}"
            )
        score = None
        held_metrics = None
        if not refit:
            held = predict(
                model,
                records,
                engine,
                "train",
                held_mask,
                settings["early_stop_epochs_per_record"],
            )
            held_metrics = classification_metrics(held["labels"], held["probabilities"])
            score = held_metrics["macro_f1"]
        history.append(
            {
                "epoch": epoch + 1,
                "loss": total_loss / steps,
                "macro_f1": score,
                "held_metrics": held_metrics,
                "seconds": time.perf_counter() - epoch_started,
            }
        )
        improved = refit or score > best + settings["min_delta"]
        if improved:
            best, best_epoch, stale = score if score is not None else 0.0, epoch + 1, 0
        else:
            stale += 1
        payload = {
            "contract": contract,
            "manifest": manifest,
            "model": trainable_state(model),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "generator": generator.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(engine.device)
            if engine.device.startswith("cuda")
            else None,
            "epoch": epoch + 1,
            "history": history,
            "best": best,
            "best_epoch": best_epoch,
            "stale": stale,
        }
        save_checkpoint(last, payload)
        if improved:
            save_checkpoint(
                best_path,
                {k: payload[k] for k in ("contract", "manifest", "model", "epoch")},
            )
        save_json(folder / "history.json", history)
    selected = torch.load(best_path, map_location="cpu", weights_only=True)
    if selected["contract"] != contract:
        raise ValueError("Best checkpoint contract differs")
    restore_state(model, selected["model"])
    save_json(
        folder / "training_summary.json",
        {
            "best_epoch": selected["epoch"],
            "epochs_completed": len(history),
            "fit_seconds": sum(row["seconds"] for row in history),
            "training_windows": len(ends),
            "class_counts": np.bincount(labels, minlength=5).tolist(),
            "class_weights": weights.detach().cpu().tolist()
            if weights is not None
            else None,
            "effective_window_batch": settings["batch_size"] * settings["accumulation"],
            "early_stopping": "fixed label-independent blocks spread across each held-out subject record, with real past context; full held fold scored once at best",
            "refit": refit,
            "checkpoint_sha256": sha256(best_path),
        },
    )
    return model, selected["epoch"]


def neural_experiment(config, engine, trained_root: Path, folder: Path, contract: str):
    started = time.perf_counter()
    family = "tiny" if config["model"]["kind"] == "tiny_finetune" else "sleepgpt"
    records = TrainingRecords(engine.data, engine.cache, family)
    folds = shared_folds(engine.data, engine.cache)
    metadata = engine.data.metadata("train")
    oof = np.empty((len(folds), 5), dtype=np.float32)
    best_epochs = []
    trained = (
        trained_root
        / ("smoke" if engine.data.smoke else "full")
        / config["experiment_id"]
    )
    for fold in range(3):
        destination = folder / f"fold_{fold}_oof.npz"
        fold_root = trained / f"fold_{fold}"
        if destination.with_suffix(".json").exists():
            info = json.loads(destination.with_suffix(".json").read_text())
            if info["contract"] != contract or info["prediction_sha256"] != sha256(
                destination
            ):
                raise ValueError("OOF resume differs")
            values = load_prediction(
                destination, {k: v[folds == fold] for k, v in metadata.items()}
            )
            best_epoch = info["best_epoch"]
        else:
            with file_lock(fold_root / "fit.lock", wait=True):
                model, best_epoch = fit_one(
                    config,
                    engine,
                    records,
                    folds != fold,
                    folds == fold,
                    fold_root,
                    config["training"]["epochs"],
                    contract,
                )
                values = predict(
                    model,
                    records,
                    engine,
                    "train",
                    folds == fold,
                    resume_folder=folder / "fold_prediction_records" / str(fold),
                    resume_contract=fingerprint(
                        {"contract": contract, "best": sha256(fold_root / "best.pt")}
                    ),
                )
                save_prediction(
                    destination,
                    values,
                    {
                        "contract": contract,
                        "best_epoch": best_epoch,
                        "fold": fold,
                        "checkpoint_sha256": sha256(fold_root / "best.pt"),
                    },
                )
                del model
                gc.collect()
                engine.release()
        oof[folds == fold] = values["probabilities"]
        best_epochs.append(int(best_epoch))
    save_prediction(
        folder / "oof_predictions.npz",
        {**metadata, "probabilities": oof},
        {"contract": contract, "folds": folds.tolist(), "best_epochs": best_epochs},
    )
    epochs = max(1, int(np.median(best_epochs)))
    save_json(
        folder / "selection.json",
        {
            "fold_best_epochs": best_epochs,
            "refit_epochs": epochs,
            "selection": "median best epoch; provided val/test never select epochs",
            "oof": classification_metrics(metadata["labels"], oof),
            "fold_metrics": [
                classification_metrics(metadata["labels"][folds == f], oof[folds == f])
                for f in range(3)
            ],
        },
    )
    with file_lock(trained / "fit" / "fit.lock", wait=True):
        model, _ = fit_one(
            config,
            engine,
            records,
            np.ones(len(folds), dtype=bool),
            None,
            trained / "fit",
            epochs,
            contract,
            refit=True,
        )
    fit_seconds = sum(
        json.loads(p.read_text())["fit_seconds"]
        for p in trained.glob("*/training_summary.json")
    )
    save_json(
        folder / "training_timing.json",
        {
            "fit_seconds": fit_seconds,
            "current_request_stage_seconds": time.perf_counter() - started,
            "fit_semantics": "completed optimization epochs plus early-stop evaluation; full OOF inference and preparation excluded",
        },
    )
    return model, records, fit_seconds, trained
