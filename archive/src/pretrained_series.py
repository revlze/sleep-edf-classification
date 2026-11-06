"""Staged pretrained experiments: frozen inference, postprocessing, opt-in training."""

import argparse
import hashlib
import json
import os
import re
import shutil
import time
import traceback
from pathlib import Path

import polars as pl
import torch
from omegaconf import OmegaConf

from src import series_assets  # noqa: F401
from src.models.pretrained_assets import ASSETS, sha256
from src.postprocessing import transition_prior, viterbi_postprocess
from src.pretrained import (
    ROOT,
    export_results,
    git_state,
    hardware,
    save_json,
    snapshot,
)
from src.progress import Progress
from src.series_data import SeriesData, fingerprint, load_prediction
from src.series_engine import PredictionEngine, base_recipe
from src.series_results import (
    ensure_submission,
    evaluate_values,
    journal,
    prediction_metrics,
    write_split,
)
from src.storage import canonical_fingerprint, file_lock
from src.training.pretrained import (
    author_groups,
    correction_predict,
    train_correction,
    train_finetune,
)


def source_digest() -> str:
    files = [
        *sorted((ROOT / "src").glob("series_*.py")),
        Path(__file__),
        ROOT / "src/postprocessing.py",
        ROOT / "src/models/anysleep_variants.py",
        ROOT / "src/models/tinysleepnet.py",
        ROOT / "src/models/temporal.py",
        ROOT / "src/training/pretrained.py",
        ROOT / "src/pretrained_signals.py",
        ROOT / "src/pretrained.py",
        ROOT / "src/data.py",
        ROOT / "src/evaluation.py",
        ROOT / "src/storage.py",
        ROOT / "src/journal.py",
        ROOT / "src/models/pretrained_assets.py",
        ROOT / "environments/pretrained/uv.lock",
    ]
    digest = hashlib.sha256(
        b"".join(p.read_bytes().replace(b"\r\n", b"\n") for p in files)
    ).hexdigest()
    return canonical_fingerprint(digest, "pretrained_series")


def load_config(path: Path) -> dict:
    config = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if config["experiment_id"] != path.stem or not re.fullmatch(
        r"\d{3}_pretrained", path.stem
    ):
        raise ValueError("Experiment ID/config path mismatch")
    if (
        config["model"]["class_order"] != ["W", "N1", "N2", "N3", "REM"]
        or config["data"]["version"] != "full_psg_v1"
    ):
        raise ValueError("Class order/dataset revision mismatch")
    if config["mode"] not in ("zero_shot", "postprocessing", "correction", "finetune"):
        raise ValueError("Unknown execution mode")
    if config["training"]["enabled"] != (config["mode"] in ("correction", "finetune")):
        raise ValueError("Training switch/mode mismatch")
    if (
        config["inference"]["core_epochs"] != 128
        or config["inference"]["halo_epochs"] != 64
        or config["inference"]["precision"] != "float32"
    ):
        raise ValueError("Frozen inference is fixed at core128/halo64, float32")
    if config["mode"] == "finetune" and (
        config["training"]["batch_size"] != 1
        or config["model"]["recipe"]["checkpoint"] != "anysleep-run1.pth"
    ):
        raise ValueError("Current fine-tuning uses batch1 and original run1")
    settings = config["training"]
    if config["mode"] == "correction" and (
        settings["halo_epochs"] < 3 or settings["folds"] < 2
    ):
        raise ValueError(
            "Correction needs halo >= receptive radius 3 and grouped folds >= 2"
        )
    if config["mode"] == "finetune" and (
        not 0 < settings["holdout_fraction"] < 1
        or settings["core_epochs"] <= 0
        or settings["core_epochs"] % 16
        or settings["halo_epochs"] % 16
        or settings["halo_epochs"] < 0
    ):
        raise ValueError(
            "Fine-tuning needs a subject holdout and phase-aligned windows"
        )
    if (
        config["provenance"]["anysleep_revision"]
        != "899bfff284f3eb3e7448bc90990f2230757470a2"
    ):
        raise ValueError("Unpinned AnySleep source")
    return config


def recursive_assets(recipe: dict) -> list[dict]:
    names = []
    if recipe["kind"] == "anysleep":
        names.append(recipe["checkpoint"])
    elif recipe["kind"] == "tiny":
        names.extend(
            ["tinysleepnet.pt", "tinysleepnet-config.json", "tinysleepnet-source.py"]
        )
    elif recipe["kind"] == "baseline019":
        names.extend(f"anysleep-run{i}.pth" for i in (1, 2, 3))
    elif recipe["kind"] == "mean":
        return [asset for node in recipe["members"] for asset in recursive_assets(node)]
    return [
        {"name": name, "url": ASSETS[name][0], "sha256": ASSETS[name][1]}
        for name in names
    ]


def run_config(
    path: Path,
    config: dict,
    args,
    data: SeriesData,
    engine: PredictionEngine,
    environment: dict,
):
    if config["training"]["enabled"] and not args.allow_training:
        raise ValueError(
            "Training is opt-in: pass --allow-training and explicitly select correction/finetune mode"
        )
    folder = engine.root / config["experiment_id"]
    folder.mkdir(parents=True, exist_ok=True)
    with (
        file_lock(engine.root / ".pretrained-export.lock", wait=True, shared=True),
        file_lock(folder / "run.lock"),
    ):
        contract = {
            "config": config,
            "data": data.identity,
            "source": source_digest(),
            "smoke": data.smoke,
            "smoke_epochs": data.smoke_epochs if data.smoke else None,
        }
        token = fingerprint(contract)
        contract_path = folder / "series_contract.json"
        if (
            contract_path.exists()
            and json.loads(contract_path.read_text())["sha256"] != token
        ):
            raise ValueError(
                "Existing result has another contract; keep it and choose a new output directory"
            )
        if not contract_path.exists():
            save_json(contract_path, {"sha256": token, **contract})
        if (folder / "complete.json").exists():
            completed = json.loads((folder / "complete.json").read_text())
            for split in ("val", "test"):
                values = load_prediction(
                    folder / f"{split}_predictions.npz", data.metadata(split)
                )
                if (
                    sha256(folder / f"{split}_predictions.npz")
                    != completed[f"{split}_sha256"]
                ):
                    raise ValueError("Completed predictions changed")
            submission_hash = ensure_submission(folder, values, data)
            if submission_hash != completed["submission_sha256"]:
                raise ValueError("Completed submission checksum changed")
            if not data.smoke:
                journal(
                    path,
                    config,
                    json.loads((folder / "metrics.json").read_text()),
                    folder,
                    ROOT,
                )
            print(
                config["experiment_id"],
                "готовый val/test проверен; inference пропущен",
                flush=True,
            )
            return
        snapshot(folder)
        resolved = {
            **config,
            "runtime": {
                "data_root": str(data.root),
                "output_root": str(engine.root),
                "device": args.device,
                "smoke": data.smoke,
                "smoke_epochs": data.smoke_epochs,
                "trained_checkpoint_root": str(args.trained_root),
            },
        }
        OmegaConf.save(OmegaConf.create(resolved), folder / "resolved.yaml")
        engine.used = {}
        engine.peak_gpu_bytes = 0
        trained, training_info = None, None
        train_start = time.perf_counter()
        recipe = config["model"]["recipe"]
        checkpoint_root = args.trained_root / ("smoke" if data.smoke else "full")
        if config["mode"] == "correction":
            trained, training_info = train_correction(
                config, engine, folder, checkpoint_root, args.device
            )
        elif config["mode"] == "finetune":
            trained, training_info = train_finetune(
                config, engine, folder, checkpoint_root, args.signal_cache, args.device
            )
            state_sha = training_info["fit"]["sha256"]
            engine.trained[state_sha] = trained
            recipe = {
                "kind": "trained_anysleep",
                "weights_sha256": state_sha,
                "source_checkpoint": "anysleep-run1.pth",
            }
        training_seconds = time.perf_counter() - train_start if training_info else None
        transition, initial = None, None
        if config["mode"] == "postprocessing":
            # Full permitted train labels even for diagnostic smoke, never val/test.
            labels = data.labels.filter(pl.col("split") == "train").sort(
                "record_id", "epoch_index"
            )
            transition, initial = transition_prior(
                labels["label"].to_numpy(),
                labels["record_id"].to_numpy(),
                labels["epoch_index"].to_numpy(),
                config["model"]["alpha"],
            )
            save_json(
                folder / "transition_prior.json",
                {
                    "matrix": transition.tolist(),
                    "initial": initial.tolist(),
                    "fitted_split": "train",
                    "labels_sha256": data.labels_hash,
                    "pseudocount": config["model"]["alpha"],
                    "strength": config["model"]["strength"],
                    "score": "sum(log posterior) + strength*(log pi + sum log A); discriminative, not a generative HMM",
                },
            )
        if args.device.startswith("cuda"):
            torch.cuda.synchronize(args.device)
            torch.cuda.reset_peak_memory_stats(args.device)
        inference_start = time.perf_counter()
        validation = None
        counts = sum(len(data.metadata(s)["ids"]) for s in ("val", "test"))
        with Progress(
            config["experiment_id"] + ": validation → test", counts, "эпох"
        ) as progress:
            engine.progress = progress
            for split in ("val", "test"):
                target = folder / f"{split}_predictions.npz"
                if target.with_suffix(".json").exists():
                    info = json.loads(target.with_suffix(".json").read_text())
                    if info["contract"] != token or info["prediction_sha256"] != sha256(
                        target
                    ):
                        raise ValueError("Partial result checksum/contract differs")
                    values = load_prediction(target, data.metadata(split))
                else:
                    values = engine.get(recipe, split)
                    if config["mode"] == "postprocessing":
                        path_labels, marginal = viterbi_postprocess(
                            values["probabilities"],
                            values["record_id"],
                            values["epoch_index"],
                            transition,
                            initial,
                            config["model"]["strength"],
                        )
                        values = {
                            **values,
                            "probabilities": marginal,
                            "predicted_labels": path_labels,
                        }
                    elif config["mode"] == "correction":
                        corrected = correction_predict(trained, values, args.device)
                        values = {**values, "probabilities": corrected}
                provenance = {
                    "contract": token,
                    "data": data.identity,
                    "recipe": recipe,
                    "class_order": ["W", "N1", "N2", "N3", "REM"],
                    "checkpoint_assets": recursive_assets(config["model"]["recipe"]),
                    "dependencies": engine.used,
                    "training": training_info,
                    "split": split,
                }
                # Validation is written/checked before any unlabelled test inference.
                result = write_split(folder, split, values, data, provenance)
                if split == "val":
                    validation = result
                    groups = author_groups(
                        values["subject_id"],
                        engine.checkpoint_cache,
                        engine.checkpoint_input,
                    )
                    save_json(
                        folder / "author_group_metrics.json",
                        {
                            g: {
                                "subjects": len(set(values["subject_id"][groups == g])),
                                "epochs": int((groups == g).sum()),
                                **prediction_metrics(
                                    {k: v[groups == g] for k, v in values.items()}
                                ),
                            }
                            for g in sorted(set(groups))
                        },
                    )
                    save_json(
                        folder / "status.json",
                        {"validation": "complete", "test": "pending"},
                    )
                progress.update(
                    len(values["ids"]),
                    detail=f"{split}: сохранено {len(values['ids'])} эпох",
                )
        engine.progress = None
        if args.device.startswith("cuda"):
            torch.cuda.synchronize(args.device)
        metrics = {
            "smoke": data.smoke,
            "validation": validation,
            "evaluation_protocol": "provided_val_then_unlabelled_test",
            "inference_seconds": time.perf_counter() - inference_start,
            "inference_time_scope": "wall time of this request incl cache IO, excludes training; per-node native times in provenance",
            "test_epochs": len(data.metadata("test")["ids"]),
            "peak_gpu_bytes": max(
                engine.peak_gpu_bytes, torch.cuda.max_memory_allocated(args.device)
            )
            if args.device.startswith("cuda")
            else None,
        }
        if training_seconds is not None:
            metrics["training_seconds"] = training_seconds
        save_json(folder / "metrics.json", metrics)
        save_json(
            folder / "provenance.json",
            {
                "git": git_state(),
                "source_sha256": source_digest(),
                "environment": environment,
                "data": data.identity,
                "training": training_info,
                "dependency_probabilities": engine.used,
                "limitation": "val used repeatedly; AnySleep author train overlaps course subjects; see author_group_metrics.json",
            },
        )
        save_json(
            folder / "status.json", {"validation": "complete", "test": "complete"}
        )
        save_json(
            folder / "complete.json",
            {
                "val_sha256": sha256(folder / "val_predictions.npz"),
                "test_sha256": sha256(folder / "test_predictions.npz"),
                "submission_sha256": ensure_submission(
                    folder,
                    load_prediction(
                        folder / "test_predictions.npz", data.metadata("test")
                    ),
                    data,
                ),
                "contract": token,
            },
        )
        if not data.smoke:
            journal(path, config, metrics, folder, ROOT)
        print(
            config["experiment_id"],
            "SMOKE" if data.smoke else "VAL+TEST",
            f"macro-F1={validation['macro_f1']:.5f}",
            flush=True,
        )


def complete_baseline(args, data: SeriesData, engine: PredictionEngine):
    """Add missing test output to 019, preserving original val files/config/metrics."""
    folder = engine.root / "019_pretrained"
    folder.mkdir(parents=True, exist_ok=True)
    with (
        file_lock(engine.root / ".pretrained-export.lock", wait=True, shared=True),
        file_lock(folder / "baseline_test.lock"),
    ):
        with Progress(
            "019: сохранённый val → новый test",
            sum(len(data.metadata(s)["ids"]) for s in ("val", "test")),
            "эпох",
        ) as progress:
            engine.progress = progress
            validation = engine.get(base_recipe(), "val")
            save_json(
                folder / "baseline_validation_reference.json",
                {
                    "cached_predictions": f"predictions/{engine.key(base_recipe())}/val.npz",
                    "validation_metrics": evaluate_values(validation),
                    "data": data.identity,
                    "note": "Original 019 validation/config/provenance files are not overwritten",
                },
            )
            progress.update(len(validation["ids"]), detail="val 019 проверен")
            test = engine.get(base_recipe(), "test")
            write_split(
                folder,
                "test",
                test,
                data,
                {
                    "data": data.identity,
                    "recipe": base_recipe(),
                    "dependencies": engine.used,
                    "class_order": ["W", "N1", "N2", "N3", "REM"],
                    "extension_of": "019_pretrained",
                },
            )
            progress.update(len(test["ids"]), detail="test 019 и submission сохранены")
    engine.progress = None
    engine.release_models()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", type=Path, nargs="*")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-root", type=Path, default=ROOT / "artefacts")
    parser.add_argument(
        "--checkpoint-cache",
        type=Path,
        default=ROOT / "artefacts/checkpoints/pretrained",
    )
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--result-input", type=Path, action="append", default=[])
    parser.add_argument(
        "--signal-cache", type=Path, default=ROOT / "artefacts/cache/neural"
    )
    parser.add_argument(
        "--trained-root", type=Path, default=ROOT / "artefacts/trained-checkpoints"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--threads", type=int, default=max(1, (os.cpu_count() or 2) - 1)
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-epochs", type=int, default=48)
    parser.add_argument("--allow-training", action="store_true")
    parser.add_argument(
        "--complete-baseline",
        action="store_true",
        help="Add test probabilities/submission to original 019, reusing archived val",
    )
    parser.add_argument("--export", type=Path)
    args = parser.parse_args()
    if args.export:
        export_results(
            args.output_root,
            args.export,
            extra_roots=("predictions", "series_failures.json"),
        )
        return
    if (not args.configs and not args.complete_baseline) or args.data_root is None:
        parser.error("Provide configs and --data-root")
    configs = [(path.resolve(), load_config(path)) for path in args.configs]
    if any(c["training"]["enabled"] for _, c in configs) and not args.allow_training:
        parser.error(
            "Training requires --allow-training; default A–D never train a neural model"
        )
    torch.set_num_threads(max(1, args.threads))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    environment = hardware(args.device)
    data = SeriesData(args.data_root, args.smoke, args.smoke_epochs)
    namespace = args.output_root / "smoke" if args.smoke else args.output_root
    engine = PredictionEngine(
        data,
        namespace,
        args.checkpoint_cache,
        args.checkpoint_root,
        [*args.result_input, args.output_root],
        args.device,
    )
    failures = []
    if args.complete_baseline:
        try:
            complete_baseline(args, data, engine)
        except Exception as error:
            failure = {
                "experiment_id": "019_pretrained",
                "error": repr(error),
                "traceback": traceback.format_exc(),
            }
            failures.append(failure)
            save_json(
                namespace / "019_pretrained" / "test_extension_error.json", failure
            )
            print(
                "019 test extension:",
                error,
                "; независимые эксперименты продолжаются",
                flush=True,
            )
        finally:
            engine.progress = None
            engine.release_models()
    for path, config in configs:
        try:
            run_config(path, config, args, data, engine, environment)
        except Exception as error:
            failure = {
                "experiment_id": config["experiment_id"],
                "error": repr(error),
                "traceback": traceback.format_exc(),
            }
            failures.append(failure)
            save_json(namespace / config["experiment_id"] / "last_error.json", failure)
            print(
                config["experiment_id"],
                "ошибка:",
                error,
                "; следующие эксперименты продолжаются",
                flush=True,
            )
        finally:
            engine.progress = None
            engine.release_models()
    save_json(namespace / "series_failures.json", failures)
    shutil.copy2(ROOT / "experiments.csv", args.output_root / "experiments.csv")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
