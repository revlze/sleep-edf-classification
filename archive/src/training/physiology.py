"""Bounded grouped CV/refit for frozen features and compact signal encoders."""

import json
import math
import time

import numpy as np
import torch
from torch.nn import functional as F

from src.evaluation import classification_metrics
from src.expansion_data import shared_folds
from src.models.pretrained_assets import sha256
from src.physiology_models import (
    DualStream,
    EmbeddingClassifier,
    ResidualCalibration,
    supervised_contrastive,
    transition_penalty,
)
from src.pretrained import save_json
from src.refinement_models import segment_bounds
from src.series_data import save_prediction
from src.storage import atomic_path
from src.training.expansion import save_checkpoint
from src.training.refinement import scaler_fit, transform


def transition_prior(meta, permitted):
    """Soft fold-local empirical plausibility; no hard clinical rules."""
    valid = (
        (meta["record_id"][1:] == meta["record_id"][:-1])
        & (np.diff(meta["epoch_index"]) == 1)
        & permitted[1:]
        & permitted[:-1]
    )
    counts = np.ones((5, 5), np.float32)
    np.add.at(counts, (meta["labels"][:-1][valid], meta["labels"][1:][valid]), 1)
    # sqrt moderates penalties on rare transitions; diagonal always allowed.
    prior = np.sqrt(counts / counts.max(1, keepdims=True))
    np.fill_diagonal(prior, 1)
    return prior


def signal_blocks(meta, indices, width=4):
    starts, ends = segment_bounds(meta)
    permitted = np.zeros(len(starts), bool)
    permitted[indices] = True
    chosen = []
    for begin in np.flatnonzero(np.r_[True, np.diff(starts) != 0]):
        if not permitted[begin]:
            continue
        block = np.arange(begin, ends[begin], width)
        chosen.extend(block.tolist())
    chosen = np.asarray(chosen, dtype=np.int64)
    rows = chosen[:, None] + np.arange(width)
    mask = rows < ends[chosen, None]
    rows = np.minimum(rows, ends[chosen, None] - 1)
    if not np.all(permitted[rows]):
        raise ValueError("Signal training block crosses a permitted subject set")
    return rows, mask


class SignalBlocks(torch.utils.data.Dataset):
    def __init__(self, pool, rows, mask):
        self.pool, self.rows, self.mask = pool, rows, mask

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return (
            np.array(self.pool[self.rows[index]], dtype=np.float32),
            self.rows[index],
            self.mask[index],
        )


def batch_indices(y, permitted, batch_size, rng, balanced):
    indices = np.flatnonzero(permitted)
    if not balanced:
        rng.shuffle(indices)
        return [indices[s : s + batch_size] for s in range(0, len(indices), batch_size)]
    groups = [indices[y[indices] == c] for c in range(5)]
    available = [g for g in groups if len(g) >= 2]
    if len(available) < 2:
        raise ValueError(
            "Need at least two training stages with distinct epochs for SupCon smoke"
        )
    n = max(2, batch_size // len(available))
    return [
        np.concatenate(
            [rng.choice(g, min(n, len(g)), replace=False) for g in available]
        )
        for _ in range(math.ceil(len(indices) / batch_size))
    ]


def make_model(kind, features, settings):
    if kind == "contrastive":
        return EmbeddingClassifier(features)
    if kind == "residual_calibration":
        return ResidualCalibration(features)
    return DualStream(settings["calibration"])


@torch.inference_mode()
def predict(model, kind, x, base, indices, device, progress, batch=4096):
    model.eval()
    chunks = []
    batch = min(batch, 512) if kind == "dual_stream" else batch
    for s in range(0, len(indices), batch):
        if hasattr(progress, "work"):
            progress.work(s // batch, math.ceil(len(indices) / batch))
        rows = indices[s : s + batch]
        progress.update(
            detail=f"inference batch {s // batch + 1}/{math.ceil(len(indices) / batch)} · {s}/{len(indices)} эпох"
        )
        a = (
            x[torch.as_tensor(rows, device=device)]
            if isinstance(x, torch.Tensor)
            else torch.from_numpy(np.array(x[rows], dtype=np.float32)).to(device)
        )
        with torch.autocast(
            device_type=torch.device(device).type,
            enabled=str(device).startswith("cuda"),
        ):
            result = (
                model(a, base[rows]) if kind == "residual_calibration" else model(a)
            )
        chunks.append(result[0].float().softmax(-1).cpu().numpy())
    return np.concatenate(chunks)


def fit(config, inputs, x, base, permitted, held, root, contract, epochs, refit=False):
    import gc

    while True:
        try:
            return fit_epoch_resume(
                config, inputs, x, base, permitted, held, root, contract, epochs, refit
            )
        except RuntimeError as exc:
            if not str(exc).startswith("Retry CUDA OOM"):
                raise
            gc.collect()
            torch.cuda.empty_cache()
            inputs.progress.update(
                detail=f"{root.name}: CUDA OOM, автоматическое уменьшение batch; повтор с последней сохранённой эпохи"
            )


def fit_epoch_resume(
    config, inputs, x, base, permitted, held, root, contract, epochs, refit=False
):
    args, data, progress = inputs.args, inputs.data, inputs.progress
    kind, settings = config["model"]["kind"], config["training"]
    device = args.device
    meta = data.metadata("train")
    torch.manual_seed(config["seed"])
    rng = np.random.default_rng(config["seed"])
    mean, std = (None, None) if kind == "dual_stream" else scaler_fit(x, permitted)
    values = (
        x
        if kind == "dual_stream"
        else torch.as_tensor(transform(x, mean, std), device=device)
    )
    base_tensor = None if base is None else torch.as_tensor(base, device=device)
    y = torch.as_tensor(meta["labels"], device=device, dtype=torch.long)
    model = make_model(kind, 0 if mean is None else x.shape[1], config["model"]).to(
        device
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"]
    )
    amp = str(device).startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    batch = settings["batch_size"]
    cap_file = root / "oom.json"
    if cap_file.exists():
        batch = min(batch, json.loads(cap_file.read_text())["retry_batch_size"])
    start, history, best, stale, best_epoch = 0, [], -1.0, 0, 0
    last, best_path = root / "last.pt", root / "best.pt"
    if last.exists():
        old = torch.load(last, map_location=device, weights_only=True)
        if old["contract"] != contract:
            raise ValueError("Training resume contract differs")
        model.load_state_dict(old["model"])
        optimizer.load_state_dict(old["optimizer"])
        scaler.load_state_dict(old["scaler"])
        torch.set_rng_state(old["torch_rng"].cpu())
        if amp:
            torch.cuda.set_rng_state(old["cuda_rng"].cpu(), device)
        rng.bit_generator.state = old["numpy_rng"]
        start, history, best, stale, best_epoch, batch = [
            old[k] for k in ("epoch", "history", "best", "stale", "best_epoch", "batch")
        ]
        if cap_file.exists():
            batch = min(batch, json.loads(cap_file.read_text())["retry_batch_size"])
    prior = torch.as_tensor(transition_prior(meta, permitted), device=device)
    counts = np.bincount(meta["labels"][permitted], minlength=5)
    ce_weights = torch.tensor(
        counts / max(1, counts.sum()) * max(1, np.count_nonzero(counts)),
        device=device,
        dtype=torch.float32,
    )
    if data.smoke:
        epochs = 1
    for epoch in range(start, epochs):
        if not refit and stale >= settings["patience"]:
            break
        began = time.perf_counter()
        model.train()
        totals = torch.zeros(4, device=device)
        distribution = torch.zeros(5, device=device, dtype=torch.long)
        class_min = torch.full((5,), 2**31 - 1, device=device, dtype=torch.long)
        class_max = torch.zeros(5, device=device, dtype=torch.long)
        if kind == "dual_stream":
            rows, mask = signal_blocks(meta, np.flatnonzero(permitted))
            order = rng.permutation(len(rows))
            if data.smoke:
                order = order[:2]
            loader = torch.utils.data.DataLoader(
                SignalBlocks(x, rows[order], mask[order]),
                batch_size=max(1, batch // 4),
                num_workers=0 if data.smoke or not amp else 2,
                pin_memory=amp,
                persistent_workers=False,
            )
            iterable = loader
            batches = len(loader)
        else:
            batches_list = batch_indices(
                meta["labels"], permitted, batch, rng, kind == "contrastive"
            )
            if data.smoke:
                batches_list = batches_list[:2]
            iterable = batches_list
            batches = len(batches_list)
        for number, item in enumerate(iterable):
            if hasattr(progress, "work"):
                progress.work(number, batches)
            progress.update(
                detail=f"{root.name} · эпоха {epoch + 1}/{epochs} · batch {number + 1}/{batches} · batch_size {batch}"
            )
            optimizer.zero_grad(set_to_none=True)
            if kind == "dual_stream":
                a, indices, mask = item
                a = a.to(device, non_blocking=True).flatten(0, 1)
                indices = indices.to(device)
                mask = mask.to(device)
                labels = y[indices]
            else:
                indices = torch.as_tensor(item, device=device)
                a, labels = values[indices], y[indices]
                mask = None
            try:
                with torch.autocast(device_type=torch.device(device).type, enabled=amp):
                    logits, aux = (
                        model(a, base_tensor[indices])
                        if kind == "residual_calibration"
                        else model(a)
                    )
                logits = logits.float()
                extra = logits.sum() * 0
                trans = extra
                if kind == "dual_stream":
                    logits = logits.reshape(*indices.shape, 5)
                    ce = F.cross_entropy(logits[mask], labels[mask])
                    extra = aux.float() * config["model"].get("alignment_weight", 0)
                    if config["model"].get("transition", False):
                        trans = (
                            transition_penalty(
                                logits.softmax(-1),
                                labels,
                                mask[:, 1:] & mask[:, :-1],
                                prior,
                            )
                            * 0.02
                        )
                    used = labels[mask]
                else:
                    ce = F.cross_entropy(
                        logits,
                        labels,
                        weight=ce_weights if kind == "contrastive" else None,
                    )
                    used = labels
                    if kind == "contrastive" and config["model"]["contrastive_weight"]:
                        extra = (
                            supervised_contrastive(
                                aux,
                                labels,
                                config["model"]["temperature"],
                                config["model"]["hard_negatives"],
                                indices,
                            )
                            * config["model"]["contrastive_weight"]
                        )
                    if kind == "residual_calibration":
                        pb = base_tensor[indices].clamp_min(1e-7)
                        kl = (
                            (
                                logits.log_softmax(-1).exp()
                                * (logits.log_softmax(-1) - pb.log())
                            )
                            .sum(-1)
                            .mean()
                        )
                        extra = 0.1 * kl + 0.01 * aux.square().mean()
                loss = ce + extra + trans
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite training loss")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            except torch.cuda.OutOfMemoryError:
                # Never silently skip a batch or change optimizer state mid-epoch.
                # Persist desired smaller batch, restore last epoch and retry run.
                save_json(
                    root / "oom.json",
                    {"retry_batch_size": max(4, batch // 2), "epoch": epoch + 1},
                )
                if batch <= 4:
                    raise RuntimeError("CUDA OOM even at batch4") from None
                raise RuntimeError("Retry CUDA OOM") from None
            totals += torch.stack(
                [ce.detach(), extra.detach(), trans.detach(), ce.detach().new_ones(())]
            )
            counts_batch = torch.bincount(used.detach().reshape(-1), minlength=5)
            distribution += counts_batch
            class_min = torch.minimum(class_min, counts_batch)
            class_max = torch.maximum(class_max, counts_batch)
        totals = totals.cpu().numpy()
        distribution = distribution.cpu().numpy()
        score = None
        if not refit:
            held_indices = np.flatnonzero(held)
            held_indices = held_indices[:: max(1, len(held_indices) // 12000)]
            pp = predict(
                model, kind, values, base_tensor, held_indices, device, progress
            )
            score = classification_metrics(meta["labels"][held_indices], pp)["macro_f1"]
        improved = refit or score > best + 1e-4
        if improved:
            best, best_epoch, stale = score or 0.0, epoch + 1, 0
        else:
            stale += 1
        history.append(
            {
                "epoch": epoch + 1,
                "ce": float(totals[0] / totals[3]),
                "auxiliary": float(totals[1] / totals[3]),
                "transition": float(totals[2] / totals[3]),
                "class_counts": distribution.tolist(),
                "batch_class_min": class_min.cpu().tolist(),
                "batch_class_max": class_max.cpu().tolist(),
                "held_macro_f1": score,
                "seconds": time.perf_counter() - began,
                "batch_size": batch,
            }
        )
        state = {
            "contract": contract,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch + 1,
            "history": history,
            "best": best,
            "stale": stale,
            "best_epoch": best_epoch,
            "batch": batch,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(device) if amp else None,
            "numpy_rng": rng.bit_generator.state,
            "mean": None if mean is None else torch.from_numpy(mean),
            "std": None if std is None else torch.from_numpy(std),
            "transition_prior": prior,
        }
        save_checkpoint(last, state)
        if improved:
            save_checkpoint(
                best_path,
                {
                    k: state[k]
                    for k in (
                        "contract",
                        "model",
                        "epoch",
                        "mean",
                        "std",
                        "transition_prior",
                    )
                },
            )
        save_json(root / "history.json", history)
    chosen = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(chosen["model"])
    save_json(
        root / "training_summary.json",
        {
            "best_epoch": chosen["epoch"],
            "fit_seconds": sum(r["seconds"] for r in history),
            "checkpoint_sha256": sha256(best_path),
            "transition_prior": prior.cpu().tolist(),
        },
    )
    return model, (mean, std), int(chosen["epoch"]), sum(r["seconds"] for r in history)


def neural_experiment(
    config, inputs, x, base, folder, trained, contract, fold_inputs=None
):
    meta = inputs.data.metadata("train")
    folds = shared_folds(inputs.data, inputs.args.cache_root)
    oof = np.empty((len(folds), 5), np.float32)
    best_epochs = []
    seconds = 0.0
    for fold in range(3):
        permitted, held = folds != fold, folds == fold
        xx, bb = x, base
        if fold_inputs is not None:
            xx, bb, permitted, held = fold_inputs(fold)
        inputs.progress.stage(
            f"OOF fold {fold + 1}/3", "train-only normalization and fitting"
        )
        model, scale, epoch, elapsed = fit(
            config,
            inputs,
            xx,
            bb,
            permitted,
            held,
            trained / f"fold_{fold}",
            contract,
            config["training"]["epochs"],
        )
        mean, std = scale
        values = (
            xx
            if mean is None
            else torch.as_tensor(transform(xx, mean, std), device=inputs.args.device)
        )
        baseline = (
            None if bb is None else torch.as_tensor(bb, device=inputs.args.device)
        )
        oof[held] = predict(
            model,
            config["model"]["kind"],
            values,
            baseline,
            np.flatnonzero(held),
            inputs.args.device,
            inputs.progress,
        )
        best_epochs.append(epoch)
        seconds += elapsed
        del model, values, baseline
    save_prediction(
        folder / "oof_predictions.npz",
        {**meta, "probabilities": oof},
        {
            "contract": contract,
            "folds": folds.tolist(),
            "fold_best_epochs": best_epochs,
        },
    )
    epochs = max(1, int(np.median(best_epochs)))
    save_json(
        folder / "selection.json",
        {
            "oof": classification_metrics(meta["labels"], oof),
            "fold_metrics": [
                classification_metrics(meta["labels"][folds == f], oof[folds == f])
                for f in range(3)
            ],
            "refit_epochs": epochs,
            "fold_best_epochs": best_epochs,
            "selection_set": "train-only grouped CV",
            "protocol": "honest four-role grouped CV, final fit on original full OOF"
            if fold_inputs
            else "subject GroupKFold3",
        },
    )
    inputs.progress.stage("Refit на всём train", f"{epochs} эпох")
    model, scale, _, elapsed = fit(
        config,
        inputs,
        x,
        base,
        np.ones(len(folds), bool),
        None,
        trained / "fit",
        contract,
        epochs,
        refit=True,
    )
    return model, scale, seconds + elapsed


def tree_probabilities(model, x, threads):
    raw = np.asarray(model.booster_.predict(x, num_threads=threads))
    if raw.ndim == 1:
        raw = np.column_stack([1 - raw, raw])
    aligned = np.zeros((len(x), 5), np.float32)
    aligned[:, model.classes_.astype(int)] = raw
    return aligned


def tree_experiment(config, inputs, x, folder, trained, contract):
    import joblib
    from lightgbm import LGBMClassifier

    meta = inputs.data.metadata("train")
    y = meta["labels"]
    folds = shared_folds(inputs.data, inputs.args.cache_root)
    trials = []
    total = 0.0
    for trial, params in enumerate(config["model"]["trials"]):
        pp = np.empty((len(y), 5), np.float32)
        trial_fit_seconds = 0.0
        for fold in range(3):
            root = trained / f"trial_{trial}_fold_{fold}"
            file = root / "classifier.joblib"
            allowed = folds != fold
            inputs.progress.stage(
                f"OOF fold {fold + 1}/3",
                f"LightGBM trial {trial + 1}/{len(config['model']['trials'])}",
            )
            if file.exists() and file.with_suffix(".json").exists():
                info = json.loads(file.with_suffix(".json").read_text())
                if info["contract"] != contract or info["sha256"] != sha256(file):
                    raise ValueError("Tree resume differs")
                model = joblib.load(file)
                fit_seconds = info["fit_seconds"]
            else:
                settings = {
                    **params,
                    "n_jobs": inputs.args.threads,
                    "verbosity": -1,
                    "random_state": config["seed"],
                }
                if inputs.data.smoke:
                    settings["n_estimators"] = 3
                model = LGBMClassifier(**settings)

                def callback(env):
                    inputs.progress.work(
                        env.iteration + 1, settings["n_estimators"], "iter"
                    )
                    inputs.progress.update(
                        detail=f"LightGBM trial {trial + 1}/2 · fold {fold + 1}/3 · iteration {env.iteration + 1}/{settings['n_estimators']}"
                    )

                counts = np.bincount(y[allowed], minlength=5)
                weights = np.maximum(counts, 1) ** -0.5
                started = time.perf_counter()
                model.fit(
                    x[allowed],
                    y[allowed],
                    sample_weight=weights[y[allowed]],
                    callbacks=[callback],
                )
                fit_seconds = time.perf_counter() - started
                root.mkdir(parents=True, exist_ok=True)
                with atomic_path(file) as temporary:
                    joblib.dump(model, temporary)
                save_json(
                    file.with_suffix(".json"),
                    {
                        "contract": contract,
                        "sha256": sha256(file),
                        "fit_seconds": fit_seconds,
                    },
                )
            trial_fit_seconds += fit_seconds
            pp[~allowed] = tree_probabilities(model, x[~allowed], inputs.args.threads)
        metrics = classification_metrics(y, pp)
        elapsed = trial_fit_seconds
        total += elapsed
        trials.append(
            {
                "params": params,
                "macro_f1": metrics["macro_f1"],
                "metrics": metrics,
                "fold_metrics": [
                    classification_metrics(y[folds == f], pp[folds == f])
                    for f in range(3)
                ],
                "seconds": elapsed,
            }
        )
        save_prediction(
            folder / f"trial_{trial}_oof.npz",
            {**meta, "probabilities": pp},
            {"contract": contract, "folds": folds.tolist()},
        )
    chosen = int(np.argmax([r["macro_f1"] for r in trials]))
    source = folder / f"trial_{chosen}_oof.npz"
    with np.load(source, allow_pickle=False) as z:
        pp = z["probabilities"]
    save_prediction(
        folder / "oof_predictions.npz",
        {**meta, "probabilities": pp},
        {"contract": contract, "folds": folds.tolist()},
    )
    save_json(
        folder / "selection.json",
        {
            "oof": trials[chosen]["metrics"],
            "selected_trial": chosen,
            "trials": trials,
            "selection_set": "pooled train OOF macro-F1; two predeclared trials, not unbiased nested estimate",
        },
    )
    inputs.progress.stage("Refit на всём train", "LightGBM, выбранный OOF trial")
    final = trained / "fit" / "classifier.joblib"
    if final.exists() and final.with_suffix(".json").exists():
        info = json.loads(final.with_suffix(".json").read_text())
        if info["contract"] != contract or info["sha256"] != sha256(final):
            raise ValueError("Tree refit checkpoint differs")
        model = joblib.load(final)
        refit_seconds = info["fit_seconds"]
    else:
        model = LGBMClassifier(
            **trials[chosen]["params"],
            n_jobs=inputs.args.threads,
            verbosity=-1,
            random_state=config["seed"],
        )
        if inputs.data.smoke:
            model.set_params(n_estimators=3)
        counts = np.bincount(y, minlength=5)
        started = time.perf_counter()

        def refit_callback(env):
            inputs.progress.work(env.iteration + 1, model.n_estimators, "iter")
            inputs.progress.update(
                detail=f"LightGBM refit · iteration {env.iteration + 1}/{model.n_estimators}"
            )

        model.fit(
            x,
            y,
            sample_weight=(np.maximum(counts, 1) ** -0.5)[y],
            callbacks=[refit_callback],
        )
        refit_seconds = time.perf_counter() - started
        final.parent.mkdir(parents=True, exist_ok=True)
        with atomic_path(final) as temporary:
            joblib.dump(model, temporary)
        save_json(
            final.with_suffix(".json"),
            {
                "contract": contract,
                "sha256": sha256(final),
                "fit_seconds": refit_seconds,
            },
        )
    return model, None, total + refit_seconds
