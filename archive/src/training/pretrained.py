"""Opt-in temporal correction CV and bounded AnySleep fine-tuning; no official-val fit."""

import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from torch.nn import functional as F

from src.data import load_record
from src.evaluation import classification_metrics, subject_metrics
from src.models.anysleep_variants import AnySleepVariant
from src.models.pretrained_assets import resolve_asset, sha256
from src.models.temporal import ResidualTemporalCNN, configure_finetuning, training_mode
from src.postprocessing import sequence_slices
from src.pretrained import save_json
from src.pretrained_signals import contiguous_segments, epoch_blocks, preprocess_segment
from src.progress import Progress
from src.series_data import fingerprint
from src.series_engine import base_recipe
from src.storage import atomic_path, file_lock


def save_torch(path: Path, value: dict):
    with atomic_path(path) as temporary:
        torch.save(value, temporary)


def training_identity(engine) -> dict:
    files = [Path(__file__), Path(__file__).parents[1] / "models/temporal.py"]
    return {
        "data": engine.data.identity,
        "prediction_implementation": engine.implementation,
        "training_implementation": hashlib.sha256(
            b"".join(p.read_bytes().replace(b"\r\n", b"\n") for p in files)
        ).hexdigest(),
        "smoke": engine.data.smoke,
        "smoke_epochs": engine.data.smoke_epochs if engine.data.smoke else None,
    }


def author_groups(subjects: np.ndarray, cache: Path, inputs: Path | None) -> np.ndarray:
    split = yaml.safe_load(
        resolve_asset("anysleep-author-split.yaml", cache, inputs).read_text()
    )
    people = {"SC4" + str(record)[:2] for record in split["train"]["sedf-sc"]}
    return np.asarray(
        ["author_train" if s in people else "outside_author_train" for s in subjects]
    )


def grouped_report(y, prob, subjects, groups) -> dict:
    return {
        g: {
            "subjects": len(set(subjects[groups == g])),
            "epochs": int((groups == g).sum()),
            **classification_metrics(y[groups == g], prob[groups == g]),
        }
        for g in sorted(set(groups))
    }


class WindowDataset:
    """Windows selected only from the given training people and true continuous segments."""

    def __init__(
        self,
        metadata: dict,
        subjects: set[str],
        arrays: dict[str, np.ndarray],
        kind: str,
        core: int,
        halo: int,
    ):
        self.metadata, self.arrays, self.kind, self.core, self.halo = (
            metadata,
            arrays,
            kind,
            core,
            halo,
        )
        self.windows = []
        for segment in sequence_slices(metadata["record_id"], metadata["epoch_index"]):
            if str(metadata["subject_id"][segment.start]) not in subjects:
                continue
            n = segment.stop - segment.start
            for a, b, left, right in epoch_blocks(
                n, core, halo, right_alignment=16 if kind == "finetune" else 1
            ):
                self.windows.append((segment.start, a, b, left, right))
        if not self.windows:
            raise ValueError("No training windows in requested subject subset")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        start, a, b, left, right = self.windows[index]
        y = self.metadata["labels"][start + a : start + b].copy()
        y[: left - a] = -100
        y[right - a :] = -100
        if self.kind == "correction":
            x = np.log(
                np.maximum(self.arrays["probabilities"][start + a : start + b], 1e-8)
            ).T.copy()
        else:
            record = str(self.metadata["record_id"][start])
            rec_start = self.arrays["record_starts"][record]
            x = (
                np.asarray(
                    self.arrays[record][start + a - rec_start : start + b - rec_start]
                )
                .reshape(-1, 3)
                .copy()
            )
        return torch.from_numpy(x.astype(np.float32)), torch.from_numpy(
            y.astype(np.int64)
        )


def collate_correction(items):
    n = max(len(y) for _, y in items)
    x = torch.zeros(len(items), 5, n)
    y = torch.full((len(items), n), -100, dtype=torch.long)
    for i, (signal, target) in enumerate(items):
        x[i, :, : signal.shape[1]] = signal
        y[i, : len(target)] = target
    return x, y


def fit(
    model,
    dataset: WindowDataset,
    evaluate,
    settings: dict,
    folder: Path,
    device: str,
    seed: int,
    contract: dict,
    finetune: bool = False,
    fixed_epochs: int | None = None,
):
    """Atomic model/optimizer/scaler/RNG/step checkpoints; resume the last saved step."""
    folder.mkdir(parents=True, exist_ok=True)
    token = fingerprint(contract)
    last, best = folder / "last.pt", folder / "best.pt"
    done = folder / "complete.json"
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=settings["lr"],
        weight_decay=settings["weight_decay"],
    )
    enabled = device.startswith("cuda") and settings["mixed_precision"]
    scaler = torch.amp.GradScaler("cuda", enabled=enabled)
    epochs = int(fixed_epochs if fixed_epochs is not None else settings["epochs"])
    steps = int(settings["steps_per_epoch"])
    if epochs <= 0 or steps <= 0 or settings["batch_size"] < 1:
        raise ValueError("Training epochs/steps/batch size must be positive")
    generator = torch.Generator().manual_seed(seed)
    epoch, next_step, order, history, best_score, best_epoch, bad = (
        0,
        0,
        None,
        [],
        float("-inf"),
        0,
        0,
    )
    loss_sum, loss_count, elapsed_before = 0.0, 0, 0.0
    if last.exists():
        saved = torch.load(last, map_location="cpu", weights_only=True)
        if saved["contract"] != token:
            raise ValueError(
                "Training checkpoint differs; keep it and use a separate output directory"
            )
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        epoch, next_step, order = saved["epoch"], saved["next_step"], saved["order"]
        history, best_score, best_epoch, bad = (
            saved["history"],
            saved["best_score"],
            saved["best_epoch"],
            saved["bad"],
        )
        generator.set_state(saved["generator"])
        torch.set_rng_state(saved["torch_rng"])
        loss_sum, loss_count = saved["loss_sum"], saved["loss_count"]
        elapsed_before = saved["elapsed_seconds"]
        if device.startswith("cuda") and saved["cuda_rng"] is not None:
            torch.cuda.set_rng_state(saved["cuda_rng"], device)
    if done.exists():
        completed = json.loads(done.read_text())
        if completed["contract"] != token or sha256(best) != completed["sha256"]:
            raise ValueError("Completed training checkpoint checksum/contract differs")
        model.load_state_dict(
            torch.load(best, map_location=device, weights_only=True)["model"],
            strict=True,
        )
        return completed

    def checkpoint():
        save_torch(
            last,
            {
                "contract": token,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "next_step": next_step,
                "order": order,
                "history": history,
                "best_score": best_score,
                "best_epoch": best_epoch,
                "bad": bad,
                "generator": generator.get_state(),
                "torch_rng": torch.get_rng_state(),
                "loss_sum": loss_sum,
                "loss_count": loss_count,
                "elapsed_seconds": elapsed_before + time.perf_counter() - started,
                "cuda_rng": torch.cuda.get_rng_state(device)
                if device.startswith("cuda")
                else None,
            },
        )

    started = time.perf_counter()
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    with Progress(f"Обучение {folder.name}", epochs * steps, "шаг") as progress:
        progress.update(epoch * steps + next_step)
        stopped = evaluate is not None and bad >= settings["patience"]
        if stopped:
            progress.add_total(-(epochs - epoch) * steps)
        while epoch < epochs and not stopped:
            if order is None:
                order = torch.randint(
                    len(dataset), (steps, settings["batch_size"]), generator=generator
                )
            training_mode(model, finetune)
            losses = []
            while next_step < steps:
                items = [dataset[int(i)] for i in order[next_step]]
                x, y = (
                    (items[0][0].unsqueeze(0), items[0][1].unsqueeze(0))
                    if finetune
                    else collate_correction(items)
                )
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type="cuda" if device.startswith("cuda") else "cpu",
                    enabled=enabled,
                ):
                    if finetune:
                        logits = model(x).transpose(1, 2)
                        penalty = 0.0
                    else:
                        logits, delta = model(x)
                        valid = y != -100
                        penalty = (
                            settings["residual_l2"]
                            * delta.transpose(1, 2)[valid].square().mean()
                        )
                    loss = F.cross_entropy(logits, y, ignore_index=-100) + penalty
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training loss")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    settings["gradient_clip"],
                )
                scaler.step(optimizer)
                scaler.update()
                losses.append(float(loss.detach()))
                loss_sum += losses[-1]
                loss_count += 1
                next_step += 1
                progress.update(
                    1, detail=f"epoch {epoch + 1}/{epochs}; loss={losses[-1]:.4f}"
                )
                if next_step % settings["checkpoint_every_steps"] == 0:
                    checkpoint()
            model.eval()
            progress.update(
                detail=f"epoch {epoch + 1}: внутренняя оценка"
                if evaluate is not None
                else f"epoch {epoch + 1}: сохранение refit"
            )
            evaluation = evaluate(model) if evaluate is not None else None
            history.append(
                {
                    "epoch": epoch + 1,
                    "mean_loss": loss_sum / loss_count if loss_count else None,
                    "validation": evaluation,
                }
            )
            score = (
                evaluation["macro_f1"] if evaluation is not None else float(epoch + 1)
            )
            if score > best_score:
                best_score, best_epoch, bad = score, epoch + 1, 0
                save_torch(
                    best,
                    {
                        "contract": token,
                        "model": model.state_dict(),
                        "epoch": best_epoch,
                    },
                )
            else:
                bad += 1
            epoch += 1
            loss_sum, loss_count = 0.0, 0
            next_step, order = 0, None
            checkpoint()
            save_json(folder / "history.json", history)
            if evaluate is not None and bad >= settings["patience"]:
                progress.add_total(-(epochs - epoch) * steps)
                progress.update(detail=f"early stopping; best epoch {best_epoch}")
                break
    model.load_state_dict(
        torch.load(best, map_location=device, weights_only=True)["model"], strict=True
    )
    result = {
        "best_epoch": best_epoch,
        "epochs_completed": epoch,
        "seconds_this_session": time.perf_counter() - started,
        "seconds_recorded_total": elapsed_before + time.perf_counter() - started,
        "peak_gpu_bytes": torch.cuda.max_memory_allocated(device)
        if device.startswith("cuda")
        else None,
        "best_checkpoint": str(best),
        "sha256": sha256(best),
        "history": history,
        "contract": token,
    }
    save_json(done, result)
    return result


def correction_predict(model, base: dict, device: str) -> np.ndarray:
    out = np.empty_like(base["probabilities"])
    model.eval()
    with torch.inference_mode():
        for segment in sequence_slices(base["record_id"], base["epoch_index"]):
            n = segment.stop - segment.start
            for a, b, left, right in epoch_blocks(n, 1024, model.radius):
                p = base["probabilities"][segment.start + a : segment.start + b]
                x = (
                    torch.from_numpy(np.log(np.maximum(p, 1e-8)).T.copy())
                    .unsqueeze(0)
                    .to(device)
                )
                logits, _ = model(x)
                out[segment.start + left : segment.start + right] = (
                    logits[0, :, left - a : right - a].softmax(0).T.cpu().numpy()
                )
    return out


def signal_cache(data, cache: Path) -> dict:
    metadata = data.metadata("train")
    key = fingerprint(
        {
            "data": data.identity,
            "smoke": data.smoke,
            "limit": data.smoke_epochs if data.smoke else None,
            "preprocessing": sha256(
                Path(__file__).parents[1] / "pretrained_signals.py"
            ),
        }
    )[:24]
    folder = cache / "signals" / key
    folder.mkdir(parents=True, exist_ok=True)
    arrays = {"record_starts": {}}
    offset = 0
    with Progress("Кэш сигналов train", len(metadata["ids"]), "эпох") as progress:
        for path in data.paths("train"):
            n = len(data.record_metadata(path, "train")["ids"])
            data.verify(path)
            target = folder / f"{path.stem}.npy"
            with file_lock(target.with_suffix(".lock"), wait=True):
                if not target.exists():
                    raw = load_record(path)
                    x = np.empty((n, 3840, 3), dtype=np.float32)
                    for segment in contiguous_segments(raw["epoch_index"][:n]):
                        x[segment] = preprocess_segment(
                            raw["x"][segment], "anysleep"
                        ).reshape(-1, 3840, 3)
                    with atomic_path(target) as temp:
                        with temp.open("wb") as stream:
                            np.save(stream, x, allow_pickle=False)
                    del raw, x
            arrays[path.stem] = np.load(target, mmap_mode="r", allow_pickle=False)
            if arrays[path.stem].shape != (n, 3840, 3):
                raise ValueError("Signal cache shape mismatch")
            arrays["record_starts"][path.stem] = offset
            offset += n
            progress.update(n, detail=path.stem)
    return arrays


def fine_predict(
    model, metadata, arrays, subjects: set[str], device: str
) -> tuple[np.ndarray, np.ndarray]:
    selected = np.isin(metadata["subject_id"], list(subjects))
    indices = np.flatnonzero(selected)
    out = np.empty((selected.sum(), 5), dtype=np.float32)
    position = 0
    model.eval()
    with torch.inference_mode():
        for segment in sequence_slices(metadata["record_id"], metadata["epoch_index"]):
            if str(metadata["subject_id"][segment.start]) not in subjects:
                continue
            name = str(metadata["record_id"][segment.start])
            offset = arrays["record_starts"][name]
            n = segment.stop - segment.start
            for a, b, left, right in epoch_blocks(n, 128, 64, right_alignment=16):
                x = (
                    np.asarray(
                        arrays[name][
                            segment.start + a - offset : segment.start + b - offset
                        ]
                    )
                    .reshape(-1, 3)
                    .copy()
                )
                logits = model(torch.from_numpy(x).unsqueeze(0).to(device))
                out[position + left : position + right] = (
                    logits[0, left - a : right - a].float().softmax(-1).cpu().numpy()
                )
            position += n
    return indices, out


def train_correction(
    config: dict, engine, folder: Path, checkpoint_root: Path, device: str
):
    settings = copy.deepcopy(config["training"])
    if engine.data.smoke:
        settings.update(
            epochs=1, steps_per_epoch=2, batch_size=2, checkpoint_every_steps=1
        )
    engine.release_models()
    with Progress("Подготовка P_base на train", 0, "эпох") as progress:
        engine.progress = progress
        base = engine.get(base_recipe(), "train")
    engine.progress = None
    engine.release_models()
    meta = engine.data.metadata("train")
    groups = author_groups(
        meta["subject_id"], engine.checkpoint_cache, engine.checkpoint_input
    )
    people = np.asarray(meta["subject_id"], dtype=str)
    folds = GroupKFold(n_splits=settings["folds"])
    oof = np.empty_like(base["probabilities"])
    assignments = np.full(len(people), -1, dtype=np.int64)
    training_results = []
    model_params = config["model"]
    for fold, (train_idx, valid_idx) in enumerate(
        folds.split(people, meta["labels"], people)
    ):
        train_people = set(people[train_idx])
        valid_people = set(people[valid_idx])
        if train_people & valid_people:
            raise ValueError("Subject leakage in correction CV")
        torch.manual_seed(config["seed"] + fold)
        model = ResidualTemporalCNN(model_params["width"], model_params["dropout"]).to(
            device
        )
        dataset = WindowDataset(
            meta,
            train_people,
            base,
            "correction",
            settings["core_epochs"],
            settings["halo_epochs"],
        )
        subset = {k: v[valid_idx] for k, v in base.items()}

        def evaluate(current):
            p = correction_predict(current, subset, device)
            return {
                "macro_f1": classification_metrics(subset["labels"], p)["macro_f1"],
                "author_groups": grouped_report(
                    subset["labels"], p, subset["subject_id"], groups[valid_idx]
                ),
            }

        contract = {
            "config": config,
            "inputs": training_identity(engine),
            "fold": fold,
            "train_people": sorted(train_people),
            "valid_people": sorted(valid_people),
        }
        with file_lock(checkpoint_root / config["experiment_id"] / f"fold{fold}.lock"):
            result = fit(
                model,
                dataset,
                evaluate,
                settings,
                checkpoint_root / config["experiment_id"] / f"fold{fold}",
                device,
                config["seed"] + fold,
                contract,
            )
        oof[valid_idx] = correction_predict(model, subset, device)
        assignments[valid_idx] = fold
        training_results.append(result)
    if np.any(assignments < 0):
        raise ValueError("Missing correction OOF rows")
    from src.pretrained import save_npz

    save_npz(
        folder / "oof_predictions.npz", **meta, probabilities=oof, fold=assignments
    )
    save_json(
        folder / "oof_metrics.json",
        {
            "protocol": "three_subject_grouped_folds; frozen pretrained encoder may have seen author_train people",
            "baseline": classification_metrics(meta["labels"], base["probabilities"]),
            "corrected": classification_metrics(meta["labels"], oof),
            "author_groups": grouped_report(meta["labels"], oof, people, groups),
            "subjects": subject_metrics(meta["labels"], oof, people),
            "folds": [
                {
                    "fold": i,
                    "train_subjects": sorted(set(people[assignments != i])),
                    "valid_subjects": sorted(set(people[assignments == i])),
                }
                for i in range(settings["folds"])
            ],
        },
    )
    # Refit epochs chosen exclusively from training-CV early stopping.
    epochs = max(1, int(np.median([r["best_epoch"] for r in training_results])))
    torch.manual_seed(config["seed"] + 100)
    final = ResidualTemporalCNN(model_params["width"], model_params["dropout"]).to(
        device
    )
    dataset = WindowDataset(
        meta,
        set(people),
        base,
        "correction",
        settings["core_epochs"],
        settings["halo_epochs"],
    )
    contract = {
        "config": config,
        "inputs": training_identity(engine),
        "refit_epochs": epochs,
        "subjects": sorted(set(people)),
    }
    with file_lock(checkpoint_root / config["experiment_id"] / "final.lock"):
        result = fit(
            final,
            dataset,
            None,
            settings,
            checkpoint_root / config["experiment_id"] / "final",
            device,
            config["seed"] + 100,
            contract,
            fixed_epochs=epochs,
        )
    info = {
        "protocol": "train_group_cv_then_full_train_refit; no official-val fit",
        "folds": training_results,
        "refit": result,
        "refit_epochs": epochs,
        "base_train_key": engine.key(base_recipe()),
    }
    save_json(folder / "training.json", info)
    return final, info


def train_finetune(
    config: dict,
    engine,
    folder: Path,
    checkpoint_root: Path,
    signal_root: Path,
    device: str,
):
    engine.release_models()
    data = engine.data
    meta = data.metadata("train")
    people = np.asarray(meta["subject_id"], dtype=str)
    groups = author_groups(people, engine.checkpoint_cache, engine.checkpoint_input)
    train_idx, valid_idx = next(
        GroupShuffleSplit(
            n_splits=1,
            test_size=config["training"]["holdout_fraction"],
            random_state=config["seed"],
        ).split(people, meta["labels"], people)
    )
    train_people, valid_people = set(people[train_idx]), set(people[valid_idx])
    if train_people & valid_people:
        raise ValueError("Fine-tuning holdout subject overlap")
    arrays = signal_cache(data, signal_root)
    recipe = {
        "kind": "anysleep",
        "variant": "base",
        "checkpoint": "anysleep-run1.pth",
        "frequency": 1,
        "voting": "soft",
    }
    torch.manual_seed(config["seed"])
    adapter = AnySleepVariant(
        recipe, engine.checkpoint_cache, engine.checkpoint_input, device
    )
    model = adapter.model
    params = configure_finetuning(model, config["model"]["strategy"])
    save_json(
        folder / "trainable_parameters.json",
        {
            "strategy": config["model"]["strategy"],
            "trainable_parameters": sum(p["parameters"] for p in params),
            "entries": params,
            "batch_norm_policy": "frozen running moments; selected affine parameters train",
        },
    )
    settings = copy.deepcopy(config["training"])
    if data.smoke:
        settings.update(
            epochs=1, steps_per_epoch=2, batch_size=1, checkpoint_every_steps=1
        )
    dataset = WindowDataset(
        meta,
        train_people,
        arrays,
        "finetune",
        settings["core_epochs"],
        settings["halo_epochs"],
    )

    def evaluate(current):
        idx, p = fine_predict(current, meta, arrays, valid_people, device)
        return {
            "macro_f1": classification_metrics(meta["labels"][idx], p)["macro_f1"],
            "author_groups": grouped_report(
                meta["labels"][idx], p, people[idx], groups[idx]
            ),
        }

    with Progress("Внутренняя оценка исходного run1", 1, "проверка") as progress:
        progress.update(detail=f"{len(valid_people)} испытуемых; исходные веса")
        baseline = evaluate(model)
        progress.update(1)
    save_json(
        folder / "internal_validation.json",
        {
            "train_subjects": sorted(train_people),
            "validation_subjects": sorted(valid_people),
            "provided_val_used_for_fit": False,
            "baseline": baseline,
            "author_group_counts": {
                "train": {
                    g: len(set(people[train_idx][groups[train_idx] == g]))
                    for g in sorted(set(groups))
                },
                "validation": {
                    g: len(set(people[valid_idx][groups[valid_idx] == g]))
                    for g in sorted(set(groups))
                },
            },
        },
    )
    contract = {
        "config": config,
        "inputs": training_identity(engine),
        "train_people": sorted(train_people),
        "valid_people": sorted(valid_people),
    }
    with file_lock(checkpoint_root / config["experiment_id"] / "fit.lock"):
        result = fit(
            model,
            dataset,
            evaluate,
            settings,
            checkpoint_root / config["experiment_id"] / "fit",
            device,
            config["seed"],
            contract,
            finetune=True,
        )
    model.eval()
    info = {
        "protocol": "subject_group_holdout_best_checkpoint; no full-train refit; no official-val fit",
        "fit": result,
        "starting_checkpoint": recipe["checkpoint"],
        "starting_assets": adapter.assets,
        "baseline_internal": baseline,
        "best_internal": evaluate(model),
    }
    save_json(folder / "training.json", info)
    return adapter, info
