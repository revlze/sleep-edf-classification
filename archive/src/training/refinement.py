"""Subject-grouped residual training and coverage-driven TinySleepNet fine-tuning."""

import json
import math
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.evaluation import classification_metrics
from src.expansion_data import physiological_features, shared_folds
from src.models.pretrained_assets import sha256
from src.models.tinysleepnet import TinySleepNetAdapter
from src.pretrained import save_json
from src.pretrained_signals import contiguous_segments
from src.refinement_data import FastRecords, nested_experts
from src.refinement_models import ResidualHead, WindowBank, confidence_features
from src.series_data import load_prediction, save_prediction
from src.training.expansion import save_checkpoint, training_mode


def inputs(config, shared, split, experts):
    recipe = config["model"].get("features", "confidence")
    chunks = [confidence_features(experts)]
    if recipe == "physio":
        key = ("physio", split)
        if key not in shared.memo:
            shared.progress.stage(
                "Подготовка 106 признаков", f"{split}: готовый кэш, без извлечения"
            )
            shared.memo[key] = physiological_features(
                shared.data, split, shared.args.feature_cache
            )
        chunks.append(shared.memo[key])
    elif recipe == "embeddings":
        chunks.append(shared.embeddings(split))
    return np.concatenate(chunks, axis=1)


def scaler_fit(x, mask):
    train = np.where(np.isfinite(x[mask]), x[mask], np.nan)
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(train, axis=0)
        std = np.nanstd(train, axis=0)
    mean = np.nan_to_num(mean).astype(np.float32)
    std = np.where(np.isfinite(std) & (std > 1e-5), std, 1).astype(np.float32)
    return mean, std


def transform(x, mean, std):
    return np.clip((np.where(np.isfinite(x), x, mean) - mean) / std, -12, 12).astype(
        np.float32
    )


@torch.inference_mode()
def head_predict(model, bank, indices, width, batch_size, progress, title):
    model.eval()
    parts, gates = [], []
    batches = math.ceil(len(indices) / batch_size)
    for i, start in enumerate(range(0, len(indices), batch_size)):
        progress.update(
            detail=f"{title}: batch {i + 1}/{batches}, эпохи {start}/{len(indices)}"
        )
        x, p, y, mask, center, lengths = bank.batch(
            indices[start : start + batch_size], width
        )
        with torch.autocast(
            device_type=torch.device(bank.device).type,
            enabled=str(bank.device).startswith("cuda"),
        ):
            logits, _, g = model(x, p, mask, lengths)
        row = torch.arange(len(x), device=x.device)
        parts.append(logits[row, center].float().softmax(-1).cpu().numpy())
        if g is not None:
            gates.append(g[row, center].float().cpu().numpy())
    return np.concatenate(parts), np.concatenate(gates) if gates else None


def head_fit(
    config,
    shared,
    x,
    experts,
    train_mask,
    held_mask,
    root,
    contract,
    epochs,
    refit=False,
):
    settings = config["training"]
    torch.manual_seed(config["seed"])
    meta = shared.data.metadata("train")
    mean, std = scaler_fit(x, train_mask)
    bank = WindowBank(transform(x, mean, std), experts, meta, shared.args.device)
    model = ResidualHead(x.shape[1], config).to(shared.args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"]
    )
    amp = str(shared.args.device).startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    generator = np.random.default_rng(config["seed"])
    start, history, best, stale, best_epoch = 0, [], -1.0, 0, 0
    last, best_path = root / "last.pt", root / "best.pt"
    if last.exists():
        old = torch.load(last, map_location=shared.args.device, weights_only=True)
        if old["contract"] != contract:
            raise ValueError("Residual training resume contract differs")
        model.load_state_dict(old["model"])
        optimizer.load_state_dict(old["optimizer"])
        scaler.load_state_dict(old["scaler"])
        generator.bit_generator.state = old["numpy_rng"]
        torch.set_rng_state(old["torch_rng"].cpu())
        if amp:
            torch.cuda.set_rng_state(old["cuda_rng"].cpu(), shared.args.device)
        start, history, best, stale, best_epoch = (
            old["epoch"],
            old["history"],
            old["best"],
            old["stale"],
            old["best_epoch"],
        )
    indices = np.flatnonzero(train_mask)
    width = config["model"]["context"]
    batch_size = settings["batch_size"]
    stride = settings["center_stride"]
    if shared.data.smoke:
        epochs = 1
    for epoch in range(start, epochs):
        if not refit and stale >= settings["patience"]:
            break
        began = time.perf_counter()
        # Cycle offsets across epochs: long-context training covers every epoch,
        # rather than repeatedly selecting the same sparse center positions.
        centers = indices[epoch % stride :: stride].copy()
        generator.shuffle(centers)
        if shared.data.smoke:
            centers = centers[: 2 * min(batch_size, 16)]
        batches = math.ceil(len(centers) / batch_size)
        model.train()
        total, count = 0.0, 0
        for i, offset in enumerate(range(0, len(centers), batch_size)):
            shared.progress.update(
                detail=f"{root.name}: обучение, эпоха {epoch + 1}/{epochs}, batch {i + 1}/{batches}"
            )
            x_batch, p, y, mask, center, lengths = bank.batch(
                centers[offset : offset + batch_size], width
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=torch.device(shared.args.device).type, enabled=amp
            ):
                logits, penalty, _ = model(x_batch, p, mask, lengths)
                row = torch.arange(len(y), device=y.device)
                loss = F.cross_entropy(logits[row, center].float(), y[row, center])
                if config["model"].get("center_supervision", False):
                    positions = torch.arange(width, device=y.device)[None, :]
                    weights = (
                        1 + (positions - center[:, None]).abs()
                    ).float().reciprocal() * mask
                    weights[row, center] = 0
                    neighbor = F.cross_entropy(
                        logits.float().reshape(-1, 5), y.reshape(-1), reduction="none"
                    ).reshape_as(y)
                    loss = loss + 0.2 * (
                        neighbor * weights
                    ).sum() / weights.sum().clamp_min(1)
                loss = loss + settings["regularization"] * penalty[row, center].mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach()) * len(y)
            count += len(y)
        score = None
        if not refit:
            held = np.flatnonzero(held_mask)
            # Fixed label-independent grid throughout held records; final OOF is full.
            held = held[:: max(1, len(held) // (512 if shared.data.smoke else 12000))]
            prob, _ = head_predict(
                model,
                bank,
                held,
                width,
                batch_size,
                shared.progress,
                f"{root.name}: early stopping (train held)",
            )
            score = classification_metrics(meta["labels"][held], prob)["macro_f1"]
        improved = refit or score > best + 1e-4
        if improved:
            best, best_epoch, stale = score if score is not None else 0.0, epoch + 1, 0
        else:
            stale += 1
        history.append(
            {
                "epoch": epoch + 1,
                "loss": total / max(count, 1),
                "held_macro_f1": score,
                "seconds": time.perf_counter() - began,
                "centers": count,
                "batch_size": batch_size,
            }
        )
        state = {
            "contract": contract,
            "model": model.state_dict(),
            "mean": torch.from_numpy(mean),
            "std": torch.from_numpy(std),
            "epoch": epoch + 1,
            "best_epoch": best_epoch,
            "best": best,
            "stale": stale,
            "history": history,
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "numpy_rng": generator.bit_generator.state,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(shared.args.device) if amp else None,
        }
        save_checkpoint(last, state)
        if improved:
            save_checkpoint(
                best_path,
                {k: state[k] for k in ("contract", "model", "mean", "std", "epoch")},
            )
        save_json(root / "history.json", history)
    chosen = torch.load(best_path, map_location=shared.args.device, weights_only=True)
    model.load_state_dict(chosen["model"])
    save_json(
        root / "training_summary.json",
        {
            "fit_seconds": sum(r["seconds"] for r in history),
            "best_epoch": chosen["epoch"],
            "history": history,
        },
    )
    return model, bank, (mean, std), int(chosen["epoch"])


def residual_experiment(config, shared, folder, trained, contract):
    meta = shared.data.metadata("train")
    folds = shared_folds(shared.data, shared.cache)
    original = shared.experts("train")
    if "nested" not in shared.memo:
        shared.memo["nested"] = nested_experts(shared)
    oof, epochs = np.empty((len(folds), 5), np.float32), []
    for fold in range(3):
        target = folder / f"fold_{fold}_oof.npz"
        root = trained / f"fold_{fold}"
        shared.progress.stage(
            f"OOF fold {fold + 1}/3",
            "нормализация только по train fold; nested expert probabilities",
        )
        shared.progress.scope = f"OOF fold {fold + 1}/3"
        if target.with_suffix(".json").exists():
            info = json.loads(target.with_suffix(".json").read_text())
            if info["contract"] != contract or info["prediction_sha256"] != sha256(
                target
            ):
                raise ValueError("Residual OOF resume differs")
            values = load_prediction(
                target, {k: v[folds == fold] for k, v in meta.items()}
            )
            best_epoch = info["best_epoch"]
        else:
            p = shared.memo["nested"][fold]
            x = inputs(config, shared, "train", p)
            shared.progress.stage(
                f"OOF fold {fold + 1}/3", "обучение head на готовых входах"
            )
            model, bank, _, best_epoch = head_fit(
                config,
                shared,
                x,
                p,
                folds != fold,
                folds == fold,
                root,
                contract,
                config["training"]["epochs"],
            )
            prob, gates = head_predict(
                model,
                bank,
                np.flatnonzero(folds == fold),
                config["model"]["context"],
                config["training"]["batch_size"],
                shared.progress,
                "полный OOF held inference",
            )
            values = {
                **{k: v[folds == fold] for k, v in meta.items()},
                "probabilities": prob,
            }
            if gates is not None:
                save_gate(folder / f"fold_{fold}_gate.npz", values, gates)
            save_prediction(
                target,
                values,
                {
                    "contract": contract,
                    "best_epoch": best_epoch,
                    "protocol": "outer GroupKFold3; inner GroupKFold2 source cross-fitting",
                },
            )
            del model, bank
        oof[folds == fold] = values["probabilities"]
        epochs.append(best_epoch)
    save_prediction(
        folder / "oof_predictions.npz",
        {**meta, "probabilities": oof},
        {"contract": contract, "folds": folds.tolist(), "fold_best_epochs": epochs},
    )
    refit_epochs = max(1, int(np.median(epochs)))
    save_json(
        folder / "selection.json",
        {
            "refit_epochs": refit_epochs,
            "fold_best_epochs": epochs,
            "oof": classification_metrics(meta["labels"], oof),
            "selection_set": "train OOF only",
            "source_alignment": "nested inner OOF for fold fitting; original 082/085 OOF for outer held and final refit; final expert checkpoints only at val/test",
        },
    )
    shared.progress.stage(
        "Refit на всём train", f"{refit_epochs} эпох по медиане внутренних folds"
    )
    shared.progress.scope = "refit"
    x = inputs(config, shared, "train", original)
    model, bank, scale, _ = head_fit(
        config,
        shared,
        x,
        original,
        np.ones(len(folds), bool),
        None,
        trained / "fit",
        contract,
        refit_epochs,
        refit=True,
    )
    del bank
    fit_seconds = sum(
        json.loads(p.read_text())["fit_seconds"]
        for p in trained.glob("*/training_summary.json")
    )
    return model, scale, fit_seconds


def save_gate(path, values, gates):
    from src.pretrained import save_npz

    save_npz(path, ids=values["ids"], weights=gates)
    save_json(
        path.with_suffix(".json"),
        {
            "experts": ["019", "082", "085"],
            "mean": gates.mean(0).tolist(),
            "std": gates.std(0).tolist(),
            "quantiles": np.quantile(gates, [0.05, 0.5, 0.95], axis=0).tolist(),
            "sha256": sha256(path),
        },
    )


def residual_predict(config, shared, split, model, scale, folder):
    meta = shared.data.metadata(split)
    p = shared.experts(split)
    x = inputs(config, shared, split, p)
    shared.progress.stage(
        "Validation" if split == "val" else "Test", "head inference на готовых входах"
    )
    bank = WindowBank(transform(x, *scale), p, meta, shared.args.device)
    prob, gates = head_predict(
        model,
        bank,
        np.arange(len(x)),
        config["model"]["context"],
        config["training"]["batch_size"],
        shared.progress,
        f"{split} inference",
    )
    values = {**meta, "probabilities": prob}
    if gates is not None:
        save_gate(folder / f"{split}_gate.npz", values, gates)
    return values


@torch.inference_mode()
def tiny_record(model, signal, device, detail, epoch_batch=256, window_batch=256):
    """Extract CNN features once, keep on GPU, vectorize rolling LSTM windows."""
    model.eval()
    chunks = []
    for start in range(0, len(signal), epoch_batch):
        detail(f"CNN epochs {start}/{len(signal)}")
        x = torch.from_numpy(
            np.array(signal[start : start + epoch_batch, None], copy=True)
        ).to(device)
        chunks.append(model.feature_extractor(x))
    features = torch.cat(chunks)
    width = min(20, len(signal))
    parts = [model.clf(features[:width][None])[0].float().softmax(-1).cpu().numpy()]
    for start in range(20, len(signal), window_batch):
        detail(f"LSTM windows {start}/{len(signal)}")
        ends = torch.arange(
            start, min(len(signal), start + window_batch), device=device
        )
        idx = ends[:, None] + torch.arange(-19, 1, device=device)
        parts.append(model.clf(features[idx])[:, -1].float().softmax(-1).cpu().numpy())
    return np.concatenate(parts)


def tiny_predict(
    model, records, shared, split, permitted=None, folder=None, contract=None
):
    parts, offset = [], 0
    for i, path in enumerate(shared.data.paths(split)):
        meta = shared.data.record_metadata(path, split)
        n = len(meta["ids"])
        wanted = (
            np.ones(n, bool) if permitted is None else permitted[offset : offset + n]
        )
        offset += n
        if not wanted.any():
            continue
        target = folder / f"{path.stem}.npz" if folder else None
        if target and target.with_suffix(".json").exists():
            info = json.loads(target.with_suffix(".json").read_text())
            if info["contract"] != contract or info["prediction_sha256"] != sha256(
                target
            ):
                raise ValueError("Tiny prediction partial resume differs")
            values = load_prediction(target, meta)
        else:
            signal = records.signal(path, split)
            prob = np.empty((n, 5), np.float32)
            for segment in contiguous_segments(meta["epoch_index"]):

                def detail(message):
                    shared.progress.update(
                        detail=f"{split} запись {i + 1}/{len(shared.data.paths(split))}, {path.stem}: {message}"
                    )

                prob[segment] = tiny_record(
                    model, signal[segment], shared.args.device, detail
                )
            values = {**meta, "probabilities": prob}
            if target:
                save_prediction(
                    target,
                    values,
                    {"contract": contract, "input": shared.data.verify(path)},
                )
        parts.append({k: v[wanted] for k, v in values.items()})
    return {k: np.concatenate([v[k] for v in parts]) for k in parts[0]}


def coverage_centers(records, permitted, rng):
    """Cover all 20-epoch tiles, then subject-balanced sleep/random supplements."""
    meta = records.data.metadata("train")
    rows = records.ends(permitted)
    global_ends = records.offsets[rows[:, 0]] + rows[:, 1]
    coverage = []
    for record in np.unique(rows[:, 0]):
        candidates = np.flatnonzero(rows[:, 0] == record)
        # Each contiguous component receives its first/last window plus stride20.
        splits = np.split(
            candidates, np.flatnonzero(np.diff(global_ends[candidates]) != 1) + 1
        )
        for block in splits:
            coverage.extend(np.unique(np.r_[block[::20], block[-1]]).tolist())
    # Equal expected contribution per subject in supplements. Half the budget is
    # unconstrained random windows (including all-W stretches), half sleep/transitions.
    subjects = meta["subject_id"][global_ends]
    groups = {s: np.flatnonzero(subjects == s) for s in sorted(set(subjects))}
    labels = meta["labels"]
    extra = []
    per_subject = max(2, math.ceil(len(coverage) / len(groups)))
    for candidates in groups.values():
        ends = global_ends[candidates]
        idx = ends[:, None] + np.arange(-19, 1)
        y = labels[idx]
        focused = candidates[
            ((y != 0).sum(1) >= 5) | np.any(y[:, 1:] != y[:, :-1], axis=1)
        ]
        extra.extend(rng.choice(candidates, per_subject // 2, replace=True).tolist())
        extra.extend(
            rng.choice(
                focused if len(focused) else candidates,
                per_subject - per_subject // 2,
                replace=True,
            ).tolist()
        )
    chosen = np.r_[coverage, extra]
    rng.shuffle(chosen)
    return rows[chosen], {
        "coverage_windows": len(coverage),
        "subject_balanced_supplements": len(extra),
        "random_fraction_of_supplements": 0.5,
        "subjects": len(groups),
        "limitation": "mandatory full-record coverage follows record duration; equal-subject supplements prevent long records owning the entire budget",
    }


def tiny_batch_size(model, records, permitted, shared, settings, root):
    """One real GPU backward estimates activation memory; no parameter update."""
    requested = settings["batch_size"]
    if (
        not shared.args.device.startswith("cuda")
        or shared.data.smoke
        or not settings.get("auto_batch", False)
    ):
        return requested
    rng_cpu = torch.get_rng_state()
    rng_cuda = torch.cuda.get_rng_state(shared.args.device)
    training_mode(model, {"model": {"kind": "tiny_finetune", "strategy": "full"}})
    probe = min(16, requested)
    candidates = records.ends(permitted)
    try:
        while True:
            shared.progress.update(
                detail=f"авто-batch: один пробный backward на {probe} окнах; параметры не обновляются"
            )
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(shared.args.device)
            baseline = torch.cuda.memory_allocated(shared.args.device)
            free, _ = torch.cuda.mem_get_info(shared.args.device)
            x = y = logits = loss = None
            try:
                x, y = records.batch(candidates[:probe])
                x, y = x.to(shared.args.device), y.to(shared.args.device)
                with torch.autocast("cuda", enabled=True):
                    logits = model(x)
                    loss = F.cross_entropy(logits.reshape(-1, 5).float(), y.reshape(-1))
                loss.backward()
                torch.cuda.synchronize(shared.args.device)
                extra = max(
                    torch.cuda.max_memory_allocated(shared.args.device) - baseline, 1
                )
                estimate = int(max(0, free - 256 * 2**20) * 0.65 * len(y) / extra)
                chosen = max(1, min(settings.get("max_batch_size", 128), estimate))
                chosen = max(1, chosen // 8 * 8) if chosen >= 8 else chosen
                save_json(
                    root / "batch_calibration.json",
                    {
                        "probe_windows": len(y),
                        "probe_extra_bytes": extra,
                        "free_bytes": free,
                        "chosen_windows": chosen,
                        "policy": "65% free memory estimate; max128; halve on real OOM",
                        "optimizer_steps_during_probe": 0,
                    },
                )
                return chosen
            except torch.cuda.OutOfMemoryError:
                if probe <= 1:
                    raise
                probe = max(1, probe // 2)
            finally:
                model.zero_grad(set_to_none=True)
                del x, y, logits, loss
                torch.cuda.empty_cache()
    finally:
        torch.set_rng_state(rng_cpu)
        torch.cuda.set_rng_state(rng_cuda, shared.args.device)


def tiny_fit(
    config, shared, records, train_mask, held_mask, root, contract, epochs, refit=False
):
    torch.manual_seed(config["seed"])
    model = TinySleepNetAdapter(
        shared.args.checkpoint_cache, shared.args.checkpoint_root, shared.args.device
    ).model
    settings = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"]
    )
    amp = shared.args.device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    rng = np.random.default_rng(config["seed"])
    history, start, stale, best, best_epoch = [], 0, 0, -1.0, 0
    batch_size = settings["batch_size"]
    last, best_path = root / "last.pt", root / "best.pt"
    if last.exists():
        state = torch.load(last, map_location=shared.args.device, weights_only=True)
        if state["contract"] != contract:
            raise ValueError("Tiny refit contract differs")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        rng.bit_generator.state = state["numpy_rng"]
        torch.set_rng_state(state["torch_rng"].cpu())
        if amp:
            torch.cuda.set_rng_state(state["cuda_rng"].cpu(), shared.args.device)
        history, start, stale, best, best_epoch, batch_size = (
            state["history"],
            state["epoch"],
            state["stale"],
            state["best"],
            state["best_epoch"],
            state["batch_size"],
        )
    else:
        batch_size = tiny_batch_size(model, records, train_mask, shared, settings, root)
    if shared.data.smoke:
        epochs = 1
    for epoch in range(start, epochs):
        if not refit and stale >= settings["patience"]:
            break
        began = time.perf_counter()
        pairs, sampling = coverage_centers(records, train_mask, rng)
        if shared.data.smoke:
            pairs = pairs[:8]
        pair_subjects = shared.data.metadata("train")["subject_id"][
            records.offsets[pairs[:, 0]] + pairs[:, 1]
        ]
        _, inverse, counts = np.unique(
            pair_subjects, return_inverse=True, return_counts=True
        )
        sample_weights = 1.0 / counts[inverse]
        sample_weights /= sample_weights.mean()
        sampling["loss_weighting"] = (
            "inverse sampled window count per subject; equal total subject contribution"
        )
        training_mode(model, {"model": {"kind": "tiny_finetune", "strategy": "full"}})
        loss_sum, count, offset = 0.0, 0, 0
        while offset < len(pairs):
            shared.progress.update(
                detail=f"{root.name}: эпоха {epoch + 1}/{epochs}, окна {offset}/{len(pairs)}, batch={batch_size}; полное покрытие + balanced sleep/random"
            )
            optimizer.zero_grad(set_to_none=True)
            try:
                x, y = records.batch(pairs[offset : offset + batch_size])
                if amp:
                    x, y = x.pin_memory(), y.pin_memory()
                x, y = (
                    x.to(shared.args.device, non_blocking=True),
                    y.to(shared.args.device, non_blocking=True),
                )
                with torch.autocast(
                    device_type=torch.device(shared.args.device).type, enabled=amp
                ):
                    logits = model(x)
                    ce = (
                        F.cross_entropy(
                            logits.reshape(-1, 5).float(),
                            y.reshape(-1),
                            reduction="none",
                        )
                        .reshape_as(y)
                        .mean(1)
                    )
                    weights = torch.as_tensor(
                        sample_weights[offset : offset + len(y)],
                        dtype=ce.dtype,
                        device=ce.device,
                    )
                    loss = (ce * weights).mean()
                scaler.scale(loss).backward()
            except torch.cuda.OutOfMemoryError:
                optimizer.zero_grad(set_to_none=True)
                if batch_size <= 1:
                    raise
                batch_size //= 2
                del x, y
                torch.cuda.empty_cache()
                continue
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach()) * len(y)
            count += len(y)
            offset += len(y)
        score = None
        if not refit:
            # Whole held subjects, reduced deterministic segments for interim ES;
            # reuse the old evaluator's fixed spread blocks, not a Wake prefix.
            from src.training.expansion import predict

            values = predict(
                model,
                records,
                shared.engine,
                "train",
                held_mask,
                per_record_limit=128 if shared.data.smoke else 256,
            )
            score = classification_metrics(values["labels"], values["probabilities"])[
                "macro_f1"
            ]
        improved = refit or score > best + 1e-4
        if improved:
            best, stale, best_epoch = score if score is not None else 0.0, 0, epoch + 1
        else:
            stale += 1
        history.append(
            {
                "epoch": epoch + 1,
                "loss": loss_sum / max(count, 1),
                "held_macro_f1": score,
                "seconds": time.perf_counter() - began,
                "optimizer_steps": math.ceil(count / batch_size),
                "windows": count,
                "sampling": sampling,
                "batch_size": batch_size,
            }
        )
        state = {
            "contract": contract,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "numpy_rng": rng.bit_generator.state,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(shared.args.device) if amp else None,
            "epoch": epoch + 1,
            "best_epoch": best_epoch,
            "best": best,
            "stale": stale,
            "history": history,
            "batch_size": batch_size,
        }
        save_checkpoint(last, state)
        if improved:
            save_checkpoint(
                best_path, {k: state[k] for k in ("contract", "model", "epoch")}
            )
        save_json(root / "history.json", history)
    state = torch.load(best_path, map_location=shared.args.device, weights_only=True)
    model.load_state_dict(state["model"])
    save_json(
        root / "training_summary.json",
        {
            "fit_seconds": sum(r["seconds"] for r in history),
            "best_epoch": state["epoch"],
        },
    )
    return model, int(state["epoch"])


def tiny_experiment(config, shared, folder, trained, contract):
    records = FastRecords(shared.data, shared.cache, shared.progress)
    folds = shared_folds(shared.data, shared.cache)
    meta = shared.data.metadata("train")
    oof, best_epochs = np.empty((len(folds), 5), np.float32), []
    for fold in range(3):
        shared.progress.stage(
            f"Tiny OOF fold {fold + 1}/3", "subject-balanced sampling и полное покрытие"
        )
        shared.progress.scope = f"Tiny OOF fold {fold + 1}/3"
        target = folder / f"fold_{fold}_oof.npz"
        if target.with_suffix(".json").exists():
            info = json.loads(target.with_suffix(".json").read_text())
            if info["contract"] != contract or info["prediction_sha256"] != sha256(
                target
            ):
                raise ValueError("Tiny OOF resume differs")
            values = load_prediction(
                target, {k: v[folds == fold] for k, v in meta.items()}
            )
            epoch = info["best_epoch"]
        else:
            model, epoch = tiny_fit(
                config,
                shared,
                records,
                folds != fold,
                folds == fold,
                trained / f"fold_{fold}",
                contract,
                config["training"]["epochs"],
            )
            values = tiny_predict(
                model,
                records,
                shared,
                "train",
                folds == fold,
                folder / "oof_records" / str(fold),
                contract,
            )
            save_prediction(target, values, {"contract": contract, "best_epoch": epoch})
            del model
        oof[folds == fold] = values["probabilities"]
        best_epochs.append(epoch)
    save_prediction(
        folder / "oof_predictions.npz",
        {**meta, "probabilities": oof},
        {"contract": contract, "folds": folds.tolist()},
    )
    epochs = max(1, int(np.median(best_epochs)))
    save_json(
        folder / "selection.json",
        {
            "refit_epochs": epochs,
            "fold_best_epochs": best_epochs,
            "selection_set": "train subject folds only",
            "oof": classification_metrics(meta["labels"], oof),
        },
    )
    shared.progress.stage("Tiny refit на всём train", f"{epochs} эпох по train OOF")
    shared.progress.scope = "Tiny refit"
    model, _ = tiny_fit(
        config,
        shared,
        records,
        np.ones(len(folds), bool),
        None,
        trained / "fit",
        contract,
        epochs,
        refit=True,
    )
    seconds = sum(
        json.loads(p.read_text())["fit_seconds"]
        for p in trained.glob("*/training_summary.json")
    )
    return model, records, seconds
