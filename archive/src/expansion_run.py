"""Prepared 058–098 series: frozen models, fixed ensembles and opt-in training."""

import argparse
import gc
import json
import os
import re
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from src.evaluation import classification_metrics
from src.expansion_assets import config_sources
from src.expansion_data import ExpansionData, shared_folds
from src.expansion_engine import ExpansionEngine
from src.expansion_tabular import fit_tabular, tabular_predict
from src.models.pretrained_assets import sha256
from src.pretrained import (
    ROOT,
    export_results,
    git_state,
    hardware,
    save_json,
    snapshot,
)
from src.progress import Progress
from src.series_data import fingerprint, load_prediction
from src.series_results import ensure_submission, evaluate_values, journal, write_split
from src.storage import file_lock
from src.training.expansion import neural_experiment, predict


class DetailProgress:
    """Child work updates one stage detail, never creates nested bars."""

    def __init__(self, progress):
        self.progress = progress

    def update(self, n=0, *, detail=None):
        if detail is not None:
            self.progress.update(detail=detail)

    def add_total(self, n):
        pass


def load_config(path: Path) -> dict:
    config = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if (
        config["experiment_id"] != path.stem
        or not re.fullmatch(r"\d{3}_pretrained", config["experiment_id"])
        or config["data"]["version"] != "full_psg_v1"
    ):
        raise ValueError("Invalid expansion configuration")
    if config["model"]["kind"] not in {
        "official",
        "ensemble",
        "adaptive_ensemble",
        "tiny_finetune",
        "embedding_classifier",
        "specialist",
        "sleepgpt",
    }:
        raise ValueError("Unknown expansion model kind")
    if config["evaluation"]["class_order"] != ["W", "N1", "N2", "N3", "REM"]:
        raise ValueError("Invalid class order")
    return config


def compare(values, baseline):
    y = values["labels"]
    new, old = values["probabilities"].argmax(1), baseline["probabilities"].argmax(1)
    result = {
        "disagreement": int(np.sum(new != old)),
        "baseline_errors_fixed": int(np.sum((new == y) & (old != y))),
        "new_errors": int(np.sum((new != y) & (old == y))),
        "delta_macro_f1": classification_metrics(y, values["probabilities"])["macro_f1"]
        - classification_metrics(y, baseline["probabilities"])["macro_f1"],
    }
    result["subjects"] = [
        {
            "subject_id": str(s),
            "disagreement": int(np.sum((new != old) & (values["subject_id"] == s))),
            "errors_fixed": int(
                np.sum((new == y) & (old != y) & (values["subject_id"] == s))
            ),
            "new_errors": int(
                np.sum((new != y) & (old == y) & (values["subject_id"] == s))
            ),
        }
        for s in sorted(set(values["subject_id"]))
    ]
    return result


def component_diagnostics(values, engine, split):
    baseline = engine.baseline("053", split)
    result = {"against_053": compare(values, baseline)}
    if split == "train":
        result["protocol"] = (
            "train grouped OOF of fitted model; author-pretraining overlap is a separate limitation"
        )
    result["ensemble_probes"] = []
    for weight in (0.25, 0.5):
        p = (1 - weight) * baseline["probabilities"] + weight * values["probabilities"]
        result["ensemble_probes"].append(
            {"candidate_weight": weight, **classification_metrics(values["labels"], p)}
        )
    return result


def adaptive_selection(config, engine, folder):
    target = folder / "selection.json"
    if target.exists():
        return json.loads(target.read_text())
    meta = engine.data.metadata("train")
    folds = shared_folds(engine.data, engine.cache)
    base = engine.baseline("053", "train")["probabilities"]
    rows = []
    for recipe in config["model"]["candidates"]:
        try:
            p = engine.frozen(recipe, "train")["probabilities"]
            mixture = 0.75 * base + 0.25 * p
            rows.append(
                {
                    "recipe": recipe,
                    "macro_f1": classification_metrics(meta["labels"], mixture)[
                        "macro_f1"
                    ],
                    "fold_macro_f1": [
                        classification_metrics(
                            meta["labels"][folds == f], mixture[folds == f]
                        )["macro_f1"]
                        for f in range(3)
                    ],
                }
            )
        except Exception as error:
            save_json(
                folder / f"candidate_{recipe['name']}_error.json",
                {"error": repr(error), "traceback": traceback.format_exc()},
            )
    if len(rows) < 2:
        raise RuntimeError(
            "Adaptive ensemble requires at least two successful train-level frozen candidates"
        )
    rows.sort(key=lambda r: (-r["macro_f1"], r["recipe"]["name"]))
    selected = {
        "selection_set": "train only; frozen predictions, pooled and grouped diagnostic scores; no local fitting",
        "trials": rows,
        "selected": [r["recipe"] for r in rows[:2]],
        "weights": [0.5, 0.25, 0.25],
    }
    save_json(target, selected)
    return selected


def run_config(path, config, args, engine, environment):
    folder = engine.root / config["experiment_id"]
    folder.mkdir(parents=True, exist_ok=True)
    feature_provenance = None
    if "physio" in config["model"].get("features", []):
        if args.feature_cache is None:
            raise FileNotFoundError(
                "This configuration needs --feature-cache; other configurations remain independent"
            )
        definitions = sorted(args.feature_cache.rglob("definition.json"))
        objects = sorted(args.feature_cache.rglob("*_epoch.npz"))
        feature_provenance = {
            "definitions": {
                str(p.relative_to(args.feature_cache)): json.loads(p.read_text())
                for p in definitions
            },
            "objects": {
                str(p.relative_to(args.feature_cache)): sha256(p) for p in objects
            },
        }
    contract = fingerprint(
        {
            "config": config,
            "checkpoint_sources": config_sources(config),
            "data": engine.data.identity,
            "implementation": engine.implementation,
            "smoke": engine.data.smoke,
            "limit": engine.data.smoke_epochs if engine.data.smoke else None,
            "feature_cache": feature_provenance,
        }
    )
    with (
        file_lock(args.output_root / ".pretrained-export.lock", wait=True, shared=True),
        file_lock(folder / ".run.lock", wait=False),
    ):
        provenance_file = folder / "provenance.json"
        if (
            provenance_file.exists()
            and json.loads(provenance_file.read_text())["contract"] != contract
        ):
            raise ValueError(
                "Existing experiment differs; preserve it and use another output root"
            )
        provenance = {
            "contract": contract,
            "data": engine.data.identity,
            "implementation": engine.implementation,
            "environment": environment,
            "git": git_state(),
            "class_order": config["evaluation"]["class_order"],
            "config": config,
            "checkpoint_sources": config_sources(config),
            "feature_cache": feature_provenance,
            "smoke": engine.data.smoke,
        }
        if not provenance_file.exists():
            save_json(provenance_file, provenance)
        else:
            provenance = json.loads(provenance_file.read_text())
        OmegaConf.save(OmegaConf.create(config), folder / "resolved.yaml")
        save_json(
            folder / "runtime.json",
            {
                "data_root": str(args.data_root),
                "output_root": str(args.output_root),
                "trained_root": str(args.trained_root),
                "cache_root": str(args.cache_root),
                "checkpoint_cache": str(args.checkpoint_cache),
                "checkpoint_root": str(args.checkpoint_root)
                if args.checkpoint_root
                else None,
                "result_inputs": [str(p) for p in args.result_input],
                "threads": args.threads,
                "device": args.device,
                "smoke": engine.data.smoke,
                "smoke_epochs": engine.data.smoke_epochs if engine.data.smoke else None,
            },
        )
        snapshot(folder)
        metrics_path = folder / "metrics.json"
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text())
            for split in ("val", "test"):
                prediction = folder / f"{split}_predictions.npz"
                if json.loads(prediction.with_suffix(".json").read_text())[
                    "prediction_sha256"
                ] != sha256(prediction):
                    raise ValueError("Completed prediction checksum differs")
                values = load_prediction(prediction, engine.data.metadata(split))
                if split == "test":
                    ensure_submission(folder, values, engine.data)
            if not engine.data.smoke:
                journal(path, config, metrics, folder, args.output_root)
            (folder / "last_error.json").unlink(missing_ok=True)
            print(
                config["experiment_id"], "complete: verified results reused", flush=True
            )
            return
        if engine.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(engine.device)
        kind = config["model"]["kind"]
        training_seconds = None
        model = records = selected = None
        if kind in ("tiny_finetune", "sleepgpt"):
            model, records, training_seconds, trained = neural_experiment(
                config, engine, args.trained_root, folder, contract
            )
        elif kind in ("embedding_classifier", "specialist"):
            trained = (
                args.trained_root
                / ("smoke" if engine.data.smoke else "full")
                / config["experiment_id"]
            )
            model, selected, training_seconds = fit_tabular(
                config,
                engine,
                folder,
                trained,
                args.feature_cache,
                args.threads,
                contract,
            )
        elif kind == "adaptive_ensemble":
            selected = adaptive_selection(config, engine, folder)
        elapsed, validation = {}, None
        for split in ("val", "test"):
            target = folder / f"{split}_predictions.npz"
            if target.with_suffix(".json").exists():
                info = json.loads(target.with_suffix(".json").read_text())
                if info["contract"] != contract or info["prediction_sha256"] != sha256(
                    target
                ):
                    raise ValueError("Partial result contract/checksum differs")
                values = load_prediction(target, engine.data.metadata(split))
                seconds = info["seconds"]
            else:
                start = time.perf_counter()
                engine.detail(f"{config['experiment_id']}: {split}")
                if kind == "official":
                    values = engine.frozen(config["model"]["recipe"], split)
                elif kind in ("ensemble", "adaptive_ensemble"):
                    if kind == "adaptive_ensemble":
                        members = [
                            engine.baseline("053", split),
                            *[engine.frozen(r, split) for r in selected["selected"]],
                        ]
                        weights = selected["weights"]
                    else:
                        members = [
                            engine.baseline(m["baseline"], split)
                            if "baseline" in m
                            else engine.frozen(m["recipe"], split)
                            for m in config["model"]["members"]
                        ]
                        weights = config["model"]["weights"]
                    probability = sum(
                        float(w) * m["probabilities"]
                        for m, w in zip(members, weights, strict=True)
                    )
                    values = {
                        **engine.data.metadata(split),
                        "probabilities": probability.astype(np.float32),
                    }
                elif kind in ("tiny_finetune", "sleepgpt"):
                    values = predict(
                        model,
                        records,
                        engine,
                        split,
                        resume_folder=folder / "refit_prediction_records" / split,
                        resume_contract=fingerprint(
                            {
                                "contract": contract,
                                "best": sha256(trained / "fit/best.pt"),
                            }
                        ),
                    )
                else:
                    values = tabular_predict(
                        config,
                        engine,
                        split,
                        model,
                        selected,
                        args.feature_cache,
                        folder,
                        args.threads,
                    )
                if engine.device.startswith("cuda"):
                    torch.cuda.synchronize(engine.device)
                seconds = time.perf_counter() - start
                write_split(
                    folder,
                    split,
                    values,
                    engine.data,
                    {
                        **provenance,
                        "environment": environment,
                        "git": git_state(),
                        "seconds": seconds,
                    },
                )
            elapsed[split] = seconds
            if split == "val":
                validation = evaluate_values(values)
                save_json(folder / "validation_metrics.json", validation)
                save_json(
                    folder / "comparison_053.json",
                    component_diagnostics(values, engine, split),
                )
                if kind == "tiny_finetune":
                    save_json(
                        folder / "comparison_041.json",
                        compare(values, engine.baseline("041", split)),
                    )
            else:
                ensure_submission(folder, values, engine.data)
        if kind in ("tiny_finetune", "sleepgpt", "embedding_classifier", "specialist"):
            oof = load_prediction(
                folder / "oof_predictions.npz", engine.data.metadata("train")
            )
            save_json(folder / "oof_metrics.json", evaluate_values(oof))
            save_json(
                folder / "oof_comparison_053.json",
                component_diagnostics(oof, engine, "train"),
            )
            if kind == "tiny_finetune":
                save_json(
                    folder / "oof_comparison_041.json",
                    compare(oof, engine.baseline("041", "train")),
                )
        result = {
            "smoke": engine.data.smoke,
            "val_epochs": len(engine.data.metadata("val")["ids"]),
            "test_epochs": len(engine.data.metadata("test")["ids"]),
            "validation": validation,
            "training_seconds": training_seconds,
            "inference_seconds": sum(elapsed.values()),
            "split_seconds": elapsed,
            "peak_gpu_bytes": torch.cuda.max_memory_allocated(engine.device)
            if engine.device.startswith("cuda")
            else None,
            "contract": contract,
        }
        save_json(metrics_path, result)
        if not engine.data.smoke:
            journal(path, config, result, folder, args.output_root)
            target = args.output_root / "submissions" / f"{config['experiment_id']}.csv"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(folder / "submission.csv", target)
        (folder / "last_error.json").unlink(missing_ok=True)
        del model, records
        gc.collect()
        print(
            config["experiment_id"],
            "smoke complete"
            if engine.data.smoke
            else f"complete val macro-F1={validation['macro_f1']:.5f}",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="*", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "artefacts/expansion"
    )
    parser.add_argument(
        "--cache-root", type=Path, default=ROOT / "artefacts/cache/expansion"
    )
    parser.add_argument(
        "--checkpoint-cache",
        type=Path,
        default=ROOT / "artefacts/checkpoints/expansion",
    )
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument(
        "--trained-root", type=Path, default=ROOT / "artefacts/expansion-trained"
    )
    parser.add_argument("--feature-cache", type=Path)
    parser.add_argument("--result-input", type=Path, action="append", default=[])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--threads", type=int, default=max(1, (os.cpu_count() or 2) - 1)
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-epochs", type=int, default=48)
    parser.add_argument("--allow-training", action="store_true")
    parser.add_argument("--export", type=Path)
    args = parser.parse_args()
    if args.export:
        export_results(
            args.output_root,
            args.export,
            extra_roots=(
                "predictions",
                "expansion_predictions",
                "expansion_baselines",
                "expansion_failures.json",
                "submissions",
            ),
        )
        return
    if not args.configs or args.data_root is None:
        parser.error("Provide configs and --data-root")
    configs = [(p.resolve(), load_config(p)) for p in args.configs]
    if any(c["training"]["enabled"] for _, c in configs) and not args.allow_training:
        parser.error("C/D/E/F require explicit --allow-training")
    args.output_root.mkdir(parents=True, exist_ok=True)
    with file_lock(args.output_root / "experiments.csv.lock", wait=True):
        if not args.output_root.joinpath("experiments.csv").exists():
            shutil.copy2(ROOT / "experiments.csv", args.output_root / "experiments.csv")
    torch.set_num_threads(args.threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    environment = hardware(args.device)
    data = ExpansionData(args.data_root, args.smoke, args.smoke_epochs)
    root = args.output_root / "smoke" if args.smoke else args.output_root
    engine = ExpansionEngine(
        data,
        root,
        args.cache_root,
        args.checkpoint_cache,
        args.checkpoint_root,
        [*args.result_input, args.output_root],
        args.device,
    )
    failures = []
    with Progress("Новая серия pretrained", len(configs), "эксп.") as progress:
        engine.progress = DetailProgress(progress)
        for path, config in configs:
            failed = False
            try:
                run_config(path, config, args, engine, environment)
            except Exception as error:
                failed = True
                failure = {
                    "experiment_id": config["experiment_id"],
                    "error": repr(error),
                    "traceback": traceback.format_exc(),
                }
                save_json(root / config["experiment_id"] / "last_error.json", failure)
                failures.append(failure)
                print(
                    config["experiment_id"],
                    "failed:",
                    str(error),
                    "; независимые эксперименты продолжаются",
                    flush=True,
                )
            finally:
                engine.release()
                gc.collect()
                status = "ошибка" if failed else "готово"
                progress.update(1, detail=f"{config['experiment_id']}: {status}")
    save_json(root / "expansion_failures.json", failures)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
