"""160–165: saved-probability ablations first, then frozen-expert Gate4/TCN4."""

import argparse
import gc
import hashlib
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from src.context_run import diagnostics, export
from src.data import load_record
from src.integration_data import (
    EXPERTS4,
    PRIOR4,
    Sources,
    TrainingInputs,
    name,
    pack_input,
    read,
)
from src.journal import initialize_journal
from src.models.pretrained_assets import sha256
from src.models.usleep_rbd import CHECKPOINT_SHA, load_model, predict, preprocess
from src.pretrained import ROOT, save_json, snapshot
from src.progress import Progress
from src.refinement_run import StageProgress
from src.series_data import SeriesData, fingerprint, load_prediction, save_prediction
from src.series_results import evaluate_values, journal, write_split
from src.storage import file_lock
from src.training.refinement import residual_experiment, residual_predict


def implementation():
    paths = [
        "integration_run.py",
        "integration_data.py",
        "refinement_models.py",
        "training/refinement.py",
        "models/usleep_rbd.py",
        "pretrained_signals.py",
        "series_results.py",
        "expansion_data.py",
    ]
    return hashlib.sha256(
        b"".join((ROOT / "src" / p).read_bytes().replace(b"\r\n", b"\n") for p in paths)
    ).hexdigest()


def config(number):
    path = ROOT / "configs" / f"{name(number)}.yaml"
    cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if cfg["experiment_id"] != name(number):
        raise ValueError("Config ID mismatch")
    return cfg


def prepare_folder(root, cfg, sources):
    folder = root / cfg["experiment_id"]
    folder.mkdir(parents=True, exist_ok=True)
    contract = fingerprint(
        {
            "config": cfg,
            "implementation": implementation(),
            "input_manifest": sha256(sources.root / "bundle.json"),
            "data": sources.data.identity,
        }
    )
    target = folder / "provenance.json"
    if target.exists() and read(target)["contract"] != contract:
        raise ValueError(
            f"Resume contract differs: {folder}; preserve it and use a new OUTPUT_ROOT"
        )
    save_json(
        target,
        {
            "contract": contract,
            "data": sources.data.identity,
            "config": cfg,
            "implementation": implementation(),
            "test_labels_read": False,
        },
    )
    OmegaConf.save(OmegaConf.create(cfg), folder / "resolved.yaml")
    return folder, contract


def simplex_projection(raw):
    """Euclidean projection of each row onto the five-class probability simplex."""
    raw = np.asarray(raw, np.float64)
    if raw.ndim != 2 or raw.shape[1] != 5 or not np.isfinite(raw).all():
        raise ValueError("Expected finite N x 5 scores")
    ordered = np.sort(raw, axis=1)[:, ::-1]
    threshold = (np.cumsum(ordered, axis=1) - 1) / np.arange(1, 6)
    active = (ordered > threshold).sum(axis=1) - 1
    theta = threshold[np.arange(len(raw)), active]
    projected = np.maximum(raw - theta[:, None], 0)
    return (projected / projected.sum(axis=1, keepdims=True)).astype(np.float32)


def base4(p):
    return np.einsum("nek,e->nk", p.astype(np.float64), np.asarray(PRIOR4))


def save_metrics(root, cfg, folder, seconds, training_seconds=0.0, extra=None):
    metrics = {
        "status": "complete",
        "validation": read(folder / "validation_metrics.json"),
        "inference_seconds": float(seconds),
        "training_seconds": float(training_seconds),
        **(extra or {}),
    }
    save_json(folder / "metrics.json", metrics)
    journal(
        ROOT / "configs" / f"{cfg['experiment_id']}.yaml", cfg, metrics, folder, root
    )
    return metrics


def compare_controls(folder, values, sources, quick_root=None):
    for number in (113, 156, 157):
        save_json(
            folder / f"comparison_{number:03d}.json",
            diagnostics(values, sources.prediction(number, "val")),
        )
    if quick_root is not None:
        for number in range(160, 165):
            target = quick_root / name(number) / "val_predictions.npz"
            baseline = load_prediction(target, sources.data.metadata("val"))
            save_json(
                folder / f"comparison_{number:03d}.json", diagnostics(values, baseline)
            )


def quick(args, data, sources):
    root = args.output_root / "phase1"
    root.mkdir(parents=True, exist_ok=True)
    initialize_journal(root / "experiments.csv", ROOT / "experiments.csv")
    snapshot(root)
    rows = []
    with Progress(
        "Phase1: быстрые смеси, без inference и обучения", 10, "конфигурация×split"
    ) as progress:
        for position, number in enumerate(range(160, 165), 1):
            cfg = config(number)
            folder, contract = prepare_folder(root, cfg, sources)
            begin = time.perf_counter()
            for split in ("val", "test"):
                progress.update(
                    detail=f"{name(number)} [{position}/5] · {split}: Base4 и сохранённые поправки"
                )
                target = folder / f"{split}_predictions.npz"
                if target.exists():
                    info = read(target.with_suffix(".json"))
                    if info["contract"] != contract or info[
                        "prediction_sha256"
                    ] != sha256(target):
                        raise ValueError("Quick resume predictions differ")
                    values = load_prediction(target, data.metadata(split))
                else:
                    p = np.stack(
                        [
                            sources.prediction(n, split)["probabilities"]
                            for n in EXPERTS4
                        ],
                        axis=1,
                    )
                    base = base4(p)
                    oldbase = sources.prediction(99, split)["probabilities"].astype(
                        np.float64
                    )
                    gate = (
                        sources.prediction(102, split)["probabilities"].astype(
                            np.float64
                        )
                        - oldbase
                    )
                    temporal = (
                        sources.prediction(106, split)["probabilities"].astype(
                            np.float64
                        )
                        - oldbase
                    )
                    weights = cfg["model"]["correction_weights"]
                    correction = weights[0] * gate + weights[1] * temporal
                    raw = base + correction
                    projected = simplex_projection(raw)
                    adjustment = np.abs(projected.astype(np.float64) - raw).sum(1)
                    save_json(
                        folder / f"{split}_correction_diagnostics.json",
                        {
                            "projection": "Euclidean projection onto five-class simplex",
                            "negative_rows": int((raw < 0).any(1).sum()),
                            "negative_values": int((raw < 0).sum()),
                            "minimum_raw_value": float(raw.min()),
                            "projection_l1_mean": float(adjustment.mean()),
                            "projection_l1_max": float(adjustment.max()),
                            "projection_l1_quantiles": np.quantile(
                                adjustment, [0, 0.5, 0.9, 0.99, 1]
                            ).tolist(),
                            "correction_l1_mean": float(
                                np.abs(correction).sum(1).mean()
                            ),
                            "gate_delta_l1_mean": float(np.abs(gate).sum(1).mean()),
                            "tcn_delta_l1_mean": float(np.abs(temporal).sum(1).mean()),
                            "source_inputs": dict(sources.used),
                        },
                    )
                    values = {**data.metadata(split), "probabilities": projected}
                    write_split(
                        folder,
                        split,
                        values,
                        data,
                        {"contract": contract, "data": data.identity},
                    )
                if split == "val":
                    compare_controls(folder, values, sources)
                progress.update(1)
            prior = (
                read(folder / "metrics.json")
                if (folder / "metrics.json").exists()
                else None
            )
            seconds = (
                prior["inference_seconds"] if prior else time.perf_counter() - begin
            )
            metrics = save_metrics(root, cfg, folder, seconds)
            rows.append(
                {
                    "id": name(number),
                    "macro_f1": metrics["validation"]["macro_f1"],
                    "f1": metrics["validation"]["f1"],
                    "test_ready": (folder / "submission.csv").exists(),
                }
            )
    for n in (113, 156, 157):
        metric = evaluate_values(sources.prediction(n, "val"))
        rows.append(
            {
                "id": name(n),
                "control": True,
                "macro_f1": metric["macro_f1"],
                "f1": metric["f1"],
            }
        )
    save_json(
        root / "phase1_report.json",
        {
            "complete": True,
            "selection_split": "validation only",
            "results": rows,
            "training_performed": False,
            "inference_performed": False,
        },
    )
    print(json.dumps(rows, indent=2), flush=True)


def usleep_train(args, data, sources, progress, cache_root):
    root = cache_root / "usleep156"
    root.mkdir(parents=True, exist_ok=True)
    source_cfg = config(156)
    if source_cfg["inference"]["recipe_id"] != "own019_blocks128_halo64_v2":
        raise ValueError("U-Sleep train must use our exact156 recipe")
    contract = fingerprint(
        {
            "config156": source_cfg,
            "data": data.identity,
            "checkpoint": CHECKPOINT_SHA,
            "implementation": {
                p: sha256(ROOT / "src" / p)
                for p in ("models/usleep_rbd.py", "pretrained_signals.py")
            },
        }
    )
    target = root / "train_predictions.npz"
    if target.exists():
        info = read(target.with_suffix(".json"))
        if info["contract"] != contract or info["prediction_sha256"] != sha256(target):
            raise ValueError("Saved U-Sleep train inference differs")
        return load_prediction(target, data.metadata("train")), info[
            "inference_seconds"
        ]
    for asset in ("Pretrained_Model.ckpt", "author_usleep.py"):
        sources.path(f"assets/{asset}")
    model = load_model(sources.root / "assets", torch.device(args.device))
    rows, seconds = [], 0.0
    paths = data.paths("train")
    progress.config("U-Sleep train (frozen; до двух heads)", 0, 2)
    for i, path in enumerate(paths, 1):
        meta = data.record_metadata(path, "train")
        cached = root / "records" / path.name
        cached.parent.mkdir(parents=True, exist_ok=True)
        if cached.exists():
            info = read(cached.with_suffix(".json"))
            if info["contract"] != contract or info["prediction_sha256"] != sha256(
                cached
            ):
                raise ValueError("Partial U-Sleep train cache differs")
            row = load_prediction(cached, meta)
        else:
            begin = time.perf_counter()
            progress.stage(
                "U-Sleep train: подготовка сигналов",
                f"запись{i}/{len(paths)}; без обучения",
            )
            data.verify(path)
            record = load_record(path)
            normalized, segments = preprocess(record["x"], record["epoch_index"])
            del record
            progress.stage(
                "U-Sleep train: frozen inference",
                f"запись{i}/{len(paths)}; core128/halo64",
            )
            p, _, _ = predict(
                model,
                normalized,
                segments,
                torch.device(args.device),
                progress,
                f"train запись{i}/{len(paths)}",
            )
            row = {**meta, "probabilities": p}
            save_prediction(
                cached,
                row,
                {
                    "contract": contract,
                    "inference_seconds": time.perf_counter() - begin,
                },
            )
            info = read(cached.with_suffix(".json"))
            del normalized
        rows.append(row)
        seconds += info["inference_seconds"]
    values = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
    save_prediction(
        target,
        values,
        {
            "contract": contract,
            "inference_seconds": seconds,
            "training": False,
            "same_pipeline_as": "156_pretrained",
        },
    )
    del model
    gc.collect()
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return values, seconds


def component(cfg, shared, folder, trained, contract):
    folder.mkdir(parents=True, exist_ok=True)
    existing = folder / "contract.json"
    if existing.exists() and read(existing)["contract"] != contract:
        raise ValueError("Head resume contract differs")
    save_json(
        existing,
        {
            "contract": contract,
            "config": cfg,
            "experts": list(EXPERTS4),
            "nested_sources": "cached inner2 OOF for outer train; original outer-held082/085 OOF; frozen156/019",
            "embeddings": "original frozen author Tiny128 and AnySleep6; never FT082/085 embeddings",
        },
    )
    OmegaConf.save(OmegaConf.create(cfg), folder / "resolved.yaml")
    if (folder / "metrics.json").exists():
        for split in ("val", "test", "oof"):
            path = folder / f"{split}_predictions.npz"
            if read(path.with_suffix(".json"))["prediction_sha256"] != sha256(path):
                raise ValueError("Completed head predictions differ")
        return read(folder / "metrics.json")
    model, scale, train_seconds = residual_experiment(
        cfg, shared, folder, trained, contract
    )
    seconds = 0.0
    for split in ("val", "test"):
        target = folder / f"{split}_predictions.npz"
        if target.exists():
            info = read(target.with_suffix(".json"))
            if info["contract"] != contract or info["prediction_sha256"] != sha256(
                target
            ):
                raise ValueError("Partial head predictions differ")
            values = load_prediction(target, shared.data.metadata(split))
            seconds += info["seconds"]
        else:
            begin = time.perf_counter()
            values = residual_predict(cfg, shared, split, model, scale, folder)
            elapsed = time.perf_counter() - begin
            write_split(
                folder,
                split,
                values,
                shared.data,
                {"contract": contract, "seconds": elapsed},
            )
            seconds += elapsed
        if split == "val":
            save_json(folder / "validation_metrics.json", evaluate_values(values))
    metrics = {
        "training_seconds": train_seconds,
        "inference_seconds": seconds,
        "validation": read(folder / "validation_metrics.json"),
    }
    save_json(folder / "metrics.json", metrics)
    del model
    gc.collect()
    if str(shared.args.device).startswith("cuda"):
        torch.cuda.empty_cache()
    return metrics


def train(args, data, sources):
    if not args.allow_training:
        raise ValueError("Phase2 requires --allow-training")
    quick_root = args.output_root / "phase1"
    if (
        not (quick_root / "phase1_report.json").exists()
        or not read(quick_root / "phase1_report.json")["complete"]
    ):
        raise ValueError("Complete and export Phase1 before Phase2")
    root = args.output_root / "phase2"
    root.mkdir(parents=True, exist_ok=True)
    initialize_journal(root / "experiments.csv", quick_root / "experiments.csv")
    snapshot(root)
    cfg = config(165)
    folder, contract = prepare_folder(root, cfg, sources)
    args.feature_cache = sources.root / "features"
    with Progress(
        "Phase2: U-Sleep train → Gate4 CV/refit → TCN4 CV/refit", 0, "этап"
    ) as raw_progress:
        progress = StageProgress(raw_progress)
        progress.stage(
            "Проверка готовых кэшей",
            "без старых backbone inference или expert training",
        )
        shared = TrainingInputs(data, args, progress, sources, root / "cache")
        shared.preflight()
        values, prepare_seconds = usleep_train(
            args, data, sources, progress, root / "cache"
        )
        shared.set_usleep(values["probabilities"])
        summaries = {}
        for pos, role in enumerate(("gate", "tcn"), 1):
            settings = cfg["components"][role]
            settings = {**settings, "experiment_id": name(165), "seed": cfg["seed"]}
            progress.config(f"165 {role.upper()}4", pos, 2)
            progress.plan = list(range(1, 9))
            summaries[role] = component(
                settings,
                shared,
                folder / "components" / role,
                folder / "checkpoints" / role,
                fingerprint({"ensemble": contract, "component": settings}),
            )
        begin = time.perf_counter()
        for split in ("train", "val", "test"):
            progress.stage("Сохранение", f"{split}: Base4 + Gate4 + TCN4")
            source_split = "oof" if split == "train" else split
            meta = data.metadata(split)
            ps = [
                load_prediction(
                    folder / "components" / role / f"{source_split}_predictions.npz",
                    meta,
                )["probabilities"]
                for role in ("gate", "tcn")
            ]
            p = (
                0.5 * base4(shared.experts(split))
                + 0.25 * ps[0].astype(np.float64)
                + 0.25 * ps[1].astype(np.float64)
            )
            p = (p / p.sum(1, keepdims=True)).astype(np.float32)
            values = {**meta, "probabilities": p}
            if split == "train":
                save_prediction(
                    folder / "oof_predictions.npz",
                    values,
                    {
                        "contract": contract,
                        "folds": shared.folds.tolist(),
                        "protocol": "subject grouped3, cached nested expert cross-fitting; U-Sleep and original embeddings frozen",
                    },
                )
                save_json(folder / "oof_metrics.json", evaluate_values(values))
            else:
                write_split(
                    folder,
                    split,
                    values,
                    data,
                    {"contract": contract, "data": data.identity},
                )
                if split == "val":
                    compare_controls(folder, values, sources, quick_root)
        seconds = (
            time.perf_counter()
            - begin
            + sum(m["inference_seconds"] for m in summaries.values())
        )
        metrics = save_metrics(
            root,
            cfg,
            folder,
            seconds,
            sum(m["training_seconds"] for m in summaries.values()),
            {
                "usleep_train_inference_seconds": prepare_seconds,
                "components": summaries,
                "expert_order": list(EXPERTS4),
                "prior_weights": list(PRIOR4),
                "provided_val_used_for_training": False,
                "limits": "reused validation and author-pretraining overlap; nested expert models have smaller training sets than final refits",
            },
        )
        save_json(
            root / "phase2_report.json",
            {
                "complete": True,
                "id": name(165),
                "macro_f1": metrics["validation"]["macro_f1"],
                "f1": metrics["validation"]["f1"],
                "test_ready": (folder / "submission.csv").exists(),
                "components": summaries,
            },
        )
        print(json.dumps(read(root / "phase2_report.json"), indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("quick", "train", "all"), default="all")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--input-root", type=Path)
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "artefacts/usleep-integration"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=max(1, os.cpu_count() or 2))
    parser.add_argument("--allow-training", action="store_true")
    parser.add_argument("--pack-input", type=Path)
    parser.add_argument("--export", type=Path)
    args = parser.parse_args()
    if args.pack_input:
        pack_input(args.pack_input)
        return
    if args.export:
        if args.phase == "all":
            parser.error("Export quick or train independently")
        export(
            args.output_root / ("phase1" if args.phase == "quick" else "phase2"),
            args.export,
        )
        return
    if not args.data_root or not args.input_root or args.threads < 1:
        parser.error("Need data/Input and positive threads")
    args.output_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    data = SeriesData(args.data_root)
    sources = Sources(args.input_root, data)
    with file_lock(args.output_root / ".integration.lock", wait=True):
        if args.phase in ("quick", "all"):
            quick(args, data, sources)
            if args.phase == "all":
                # CLI all, like notebook, guarantees a recoverable quick ZIP first.
                target = args.output_root / f"phase1-{time.time_ns()}.zip"
                export(args.output_root / "phase1", target)
        if args.phase in ("train", "all"):
            try:
                train(args, data, sources)
            except Exception as error:
                failure = {
                    "complete": False,
                    "error": repr(error),
                    "traceback": traceback.format_exc(),
                }
                save_json(args.output_root / "phase2" / "last_error.json", failure)
                save_json(args.output_root / "phase2" / "phase2_report.json", failure)
                raise


if __name__ == "__main__":
    main()
