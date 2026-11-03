"""Ограниченный поиск бустингов, возобновляемые folds и финальный refit."""

import hashlib
import json
import platform
import re
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import polars as pl
from omegaconf import OmegaConf

from src.cached_features import read_cached_features
from src.data import load_labels
from src.evaluation import classification_metrics, subject_metrics
from src.experiment import PROJECT_ROOT, read_git_state, update_journal
from src.models.boosting import (
    fit_booster,
    load_model,
    predict,
    resolve_threads,
    save_model,
)
from src.progress import Progress
from src.storage import atomic_path, canonical_fingerprint, file_lock
from src.training.tabular import grouped_folds, save_json


def tree_fingerprint() -> str:
    modules = (
        "training/boosting.py",
        "training/fixed_tree.py",
        "models/boosting.py",
        "cached_features.py",
        "features.py",
        "datasets.py",
        "data.py",
        "evaluation.py",
        "storage.py",
        "experiment.py",
        "inference.py",
        "run.py",
        "models/linear.py",
        "progress.py",
    )
    paths = [PROJECT_ROOT / "src" / name for name in modules] + [
        PROJECT_ROOT / "uv.lock"
    ]
    return hashlib.sha256(
        b"".join(p.read_bytes().replace(b"\r\n", b"\n") for p in paths)
    ).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def tree_source_compatible(saved: str, current: str) -> bool:
    return canonical_fingerprint(saved, "tree") == canonical_fingerprint(
        current, "tree"
    )


def load_tree_config(
    path: str | Path,
    *,
    data_root=None,
    output_root=None,
    cache_root=None,
    fold_root=None,
    threads=None,
    smoke: bool = False,
) -> dict:
    path = Path(path).resolve()
    cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if cfg.get("retired", False):
        raise ValueError(
            "This experiment plan is cancelled. Use 015_tree for the single fixed LightGBM run."
        )
    if cfg["experiment_id"] != path.stem or not re.fullmatch(r"\d{3}_tree", path.stem):
        raise ValueError("Tree ID must match config filename")
    cfg["config_path"] = str(path)
    for name, override in (
        ("data_root", data_root),
        ("output_root", output_root),
        ("cache_root", cache_root),
        ("fold_root", fold_root),
    ):
        if name == "fold_root" and name not in cfg["paths"]:
            continue
        cfg["paths"][name] = str(
            (PROJECT_ROOT / Path(override or cfg["paths"][name])).resolve()
        )
    cfg["training"]["threads"] = resolve_threads(
        threads if threads is not None else cfg["training"]["threads"]
    )
    cfg["runtime"] = {"smoke": smoke, "source_fingerprint": tree_fingerprint()}
    if cfg["training"].get("mode") == "fixed":
        if cfg["data_revision"] != "full_psg_v1" or cfg["seed"] != 42:
            raise ValueError("Expected full_psg_v1 and seed 42")
        if cfg["model"]["type"] != "lightgbm" or cfg["model"]["class_weight"] != "sqrt":
            raise ValueError("Fixed run requires LightGBM and sqrt weights")
        if cfg.get("search") or cfg.get("cv") or cfg.get("parent_candidates") or smoke:
            raise ValueError(
                "Fixed run has no search, CV, parent selection or smoke sampling"
            )
        if (
            not isinstance(cfg["training"]["iterations"], int)
            or cfg["training"]["iterations"] < 1
        ):
            raise ValueError("A positive fixed iteration count is required")
        return cfg
    if (
        cfg["data_revision"] != "full_psg_v1"
        or cfg["cv"]["n_splits"] != 3
        or cfg["seed"] != 42
        or cfg["cv"]["strategy"] != "StratifiedGroupKFold"
    ):
        raise ValueError(
            "Expected full_psg_v1 and the existing seed-42 three-fold protocol"
        )
    if cfg["search"]["screen_fold"] not in range(3) or cfg["search"]["top_k"] not in (
        1,
        2,
    ):
        raise ValueError("Invalid screening budget")
    return cfg


def candidates(cfg: dict) -> list[dict]:
    result = []
    for params in [cfg["model"]["params"], *cfg["search"]["candidates"]]:
        if params not in result:
            result.append(params)
    if len(result) > 4:
        raise ValueError("At most four candidates per experiment")
    return result


def prepare_data(
    cfg: dict,
    *,
    require_folds: bool = True,
) -> tuple[
    np.ndarray, pl.DataFrame, list[str], np.ndarray, np.ndarray | None, dict, dict
]:
    root = Path(cfg["paths"]["data_root"])
    labels = load_labels(root)
    label_hash = hashlib.sha256((root / "labels.csv").read_bytes()).hexdigest()
    x, meta, names, cache = read_cached_features(
        Path(cfg["paths"]["cache_root"]),
        labels,
        {**cfg["features"]["extractor"], "labels_sha256": label_hash},
        cfg["features"]["context"],
        smoke=cfg["runtime"]["smoke"],
    )
    train = meta["split"].to_numpy() == "train"
    if set(meta["split"]) != {"train", "val"}:
        raise ValueError("Training accepts train/val only")
    if not require_folds:
        folds = None
        fold_path = None
    elif cfg["runtime"]["smoke"]:
        fold_root = Path(cfg["paths"]["output_root"]) / "smoke/splits"
    elif require_folds:
        fold_root = Path(cfg["paths"]["fold_root"])
    if require_folds:
        fold_path = fold_root / f"groups_{cfg['seed']}_{label_hash[:12]}.csv"
    if require_folds and not fold_path.exists() and not cfg["runtime"]["smoke"]:
        raise FileNotFoundError(
            f"Copy the existing LR fold assignments: {fold_path}. New full-data folds will not be generated."
        )
    if require_folds and cfg["runtime"]["smoke"]:
        with file_lock(fold_path.with_suffix(".csv.lock"), wait=True):
            folds = grouped_folds(
                meta.filter(pl.col("split") == "train"), fold_path, cfg["seed"], 3
            )
    elif require_folds:
        # Existing full-data folds are immutable, including a read-only Kaggle Input.
        folds = grouped_folds(
            meta.filter(pl.col("split") == "train"), fold_path, cfg["seed"], 3
        )
    provenance = {
        "fold_file": str(fold_path) if fold_path is not None else None,
        "fold_sha256": hashlib.sha256(fold_path.read_bytes()).hexdigest()
        if fold_path is not None
        else None,
        "labels_sha256": label_hash,
        "cache_key": cache["key"],
        "cache_objects_sha256": cache["objects_sha256"],
        "base_objects_sha256": cache["base_objects_sha256"],
        "source_fingerprint": cfg["runtime"]["source_fingerprint"],
        "smoke": cfg["runtime"]["smoke"],
        "python": platform.python_version(),
        "packages": {
            n: version(n)
            for n in ("numpy", "scipy", "scikit-learn", "lightgbm", "catboost")
        },
        **read_git_state(PROJECT_ROOT),
    }
    print(
        f"Готовая матрица {x.shape}, {x.nbytes / 2**20:.0f} MiB; CPU-потоков: {cfg['training']['threads']}",
        flush=True,
    )
    return x, meta, names, train, folds, cache, provenance


def select_parent(cfg, namespace, provenance):
    if not cfg.get("parent_candidates"):
        return cfg
    parents = []
    for name in cfg["parent_candidates"]:
        folder = namespace / name
        metrics = read_json(folder / "metrics.json")
        previous = read_json(folder / "provenance.json")
        parent = OmegaConf.to_container(
            OmegaConf.load(folder / "resolved.yaml"), resolve=True
        )
        if any(
            previous[k] != provenance[k]
            for k in (
                "source_fingerprint",
                "fold_sha256",
                "labels_sha256",
                "base_objects_sha256",
            )
        ):
            raise ValueError(f"Incompatible parent provenance: {name}")
        if (
            parent["seed"] != cfg["seed"]
            or parent["features"]["extractor"] != cfg["features"]["extractor"]
            or parent["features"]["context"] != "neighbors"
        ):
            raise ValueError(f"Incompatible parent settings: {name}")
        parents.append((name, metrics, parent, folder))
    name, metrics, parent, folder = min(
        parents,
        key=lambda p: (
            -p[1]["tuned"]["oof"]["macro_f1"],
            p[1]["tuned"]["fold_f1_std"],
            p[0],
        ),
    )
    cfg["model"] = {**parent["model"], "params": metrics["tuned"]["params"]}
    pool = candidates(parent)
    shortlist = read_json(folder / "screening.json")["shortlist"]
    cfg["search"]["candidates"] = [
        pool[i] for i in shortlist if pool[i] != cfg["model"]["params"]
    ][:1]
    cfg["selected_parent"] = {
        "experiment_id": name,
        "criterion": "tuned pooled OOF macro-F1",
        "score": metrics["tuned"]["oof"]["macro_f1"],
    }
    return cfg


def fold_trial(
    x, y, subjects, ids, names, folds, cfg, folder, trial_id, params, fold, progress
):
    trial = folder / "trials" / f"trial_{trial_id:03d}"
    trial.mkdir(parents=True, exist_ok=True)
    target = trial / f"fold_{fold}.json"
    probs = trial / f"fold_{fold}.npz"
    valid = folds == fold
    if target.exists() and probs.exists():
        result = read_json(target)
        with np.load(probs, allow_pickle=False) as z:
            if not np.array_equal(z["ids"], ids[valid]) or result["params"] != params:
                raise ValueError("Stored fold does not match current trial")
            prob = z["probabilities"]
        classification_metrics(y[valid], prob)
        progress.update(1, detail=f"trial {trial_id}, fold {fold + 1}: готов")
        return result, prob
    progress.update(detail=f"trial {trial_id}, fold {fold + 1}: подготовка")
    model, count, seconds = fit_booster(
        x[~valid],
        y[~valid],
        family=cfg["model"]["type"],
        params=params,
        weight=cfg["model"]["class_weight"],
        seed=cfg["seed"],
        threads=cfg["training"]["threads"],
        iterations=cfg["training"]["max_iterations"],
        valid=(x[valid], y[valid]),
        patience=cfg["training"]["patience"],
        progress=progress,
    )
    start = perf_counter()
    prob = predict(model, x[valid], cfg["model"]["type"], cfg["training"]["threads"])
    result = {
        "fold": fold,
        "params": params,
        "iterations": count,
        "seconds": seconds,
        "infer_seconds": perf_counter() - start,
        **classification_metrics(y[valid], prob),
        "subjects": subject_metrics(y[valid], prob, subjects[valid]),
    }
    with atomic_path(probs) as temporary:
        np.savez_compressed(temporary, ids=ids[valid], probabilities=prob)
    save_json(target, result)
    progress.update(1, detail=f"trial {trial_id}, fold {fold + 1}: завершён")
    return result, prob


def full_trial(
    x, y, subjects, ids, names, folds, cfg, folder, trial_id, params, progress
):
    prob = np.empty((len(y), 5), dtype=np.float32)
    results = []
    for fold in range(3):
        result, prob[folds == fold] = fold_trial(
            x,
            y,
            subjects,
            ids,
            names,
            folds,
            cfg,
            folder,
            trial_id,
            params,
            fold,
            progress,
        )
        results.append(result)
    result = {
        "trial": trial_id,
        "params": params,
        "folds": results,
        "oof": classification_metrics(y, prob),
        "subjects": subject_metrics(y, prob, subjects),
        "fold_f1_std": float(np.std([r["macro_f1"] for r in results])),
        "seconds": sum(r["seconds"] for r in results),
    }
    save_json(folder / "trials" / f"trial_{trial_id:03d}" / "summary.json", result)
    return result, prob


def refit_models(
    x, meta, names, train, cfg, folder, baseline, best, base_oof, best_oof
):
    y = meta["label"].to_numpy()
    family = cfg["model"]["type"]
    threads = cfg["training"]["threads"]
    validation = {}
    seconds = infer_seconds = 0.0
    previous = None
    with Progress("Финальные модели и val", 2, "модель") as progress:
        for regime, result, oof in (
            ("fixed", baseline, base_oof),
            ("tuned", best, best_oof),
        ):
            iterations = max(
                1, int(np.median([r["iterations"] for r in result["folds"]]))
            )
            key = (json.dumps(result["params"], sort_keys=True), iterations)
            progress.update(
                detail=f"{regime}: fit на всём train, {iterations} итераций"
            )
            if previous is None or previous[0] != key:
                model, _, duration = fit_booster(
                    x[train],
                    y[train],
                    family=family,
                    params=result["params"],
                    weight=cfg["model"]["class_weight"],
                    seed=cfg["seed"],
                    threads=threads,
                    iterations=iterations,
                    progress=progress,
                )
                seconds += duration
                previous = (key, model)
            else:
                model = previous[1]
            start = perf_counter()
            prob = predict(model, x[~train], family, threads)
            infer_seconds += perf_counter() - start
            validation[regime] = {
                "params": result["params"],
                "iterations": iterations,
                **classification_metrics(y[~train], prob),
                "subjects": subject_metrics(
                    y[~train], prob, meta["subject_id"].to_numpy()[~train]
                ),
            }
            extension = "txt" if family == "lightgbm" else "cbm"
            model_path = folder / f"{regime}.{extension}"
            with atomic_path(model_path) as temporary:
                save_model(model, temporary, family)
            save_json(
                folder / f"{regime}.model.json",
                {
                    "model_file": model_path.name,
                    "family": family,
                    "feature_names": names,
                    "iterations": iterations,
                    "params": result["params"],
                    "class_order": [0, 1, 2, 3, 4],
                    "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
                },
            )
            with atomic_path(folder / f"{regime}_predictions.npz") as temporary:
                np.savez_compressed(
                    temporary,
                    train_ids=meta["id"].to_numpy()[train].astype(str),
                    oof=oof,
                    val_ids=meta["id"].to_numpy()[~train].astype(str),
                    val_probabilities=prob,
                )
            values = (
                model.booster_.feature_importance(importance_type="gain")
                if family == "lightgbm"
                else model.get_feature_importance(thread_count=threads)
            )
            pl.DataFrame({"feature": names, "importance": values}).sort(
                "importance", descending=True
            ).write_csv(folder / f"{regime}_importance.csv")
            progress.update(1)
    selection = read_json(folder / "selection.json")
    metrics = {
        "fixed": baseline,
        "tuned": best,
        "val": validation,
        "search_seconds": selection["search_seconds"],
        "refit_seconds": seconds,
        "inference_seconds": infer_seconds,
        "selection_note": "OOF and early stopping used for selection; not an unbiased estimate. Provided val unused for selection.",
    }
    save_json(folder / "metrics.json", metrics)
    return metrics


def benchmark(x, meta, train, folds, cfg, out):
    """One fold, bounded iteration/time budget, whole-subject subsets; no journal."""
    tx = x[train]
    data = meta.filter(pl.col("split") == "train")
    labels = data["label"].to_numpy()
    groups = data["subject_id"].to_numpy()
    fold = cfg["search"]["screen_fold"]

    def subset(mask, limit):
        selected = np.zeros(len(labels), dtype=bool)
        for person in sorted(set(groups[mask])):
            selected |= mask & (groups == person)
            if selected.sum() >= limit:
                break
        return selected

    tr = subset(folds != fold, cfg["benchmark"]["train_epochs"])
    va = subset(folds == fold, cfg["benchmark"]["val_epochs"])
    with Progress(
        "Короткий замер", cfg["benchmark"]["iterations"], "итерация"
    ) as progress:
        model, count, duration = fit_booster(
            tx[tr],
            labels[tr],
            family=cfg["model"]["type"],
            params=cfg["model"]["params"],
            weight=cfg["model"]["class_weight"],
            seed=cfg["seed"],
            threads=cfg["training"]["threads"],
            iterations=cfg["benchmark"]["iterations"],
            valid=(tx[va], labels[va]),
            progress=progress,
            max_seconds=cfg["benchmark"]["max_seconds"],
        )
        progress.update(
            max(0, count - progress.n), detail="Замер завершён; не эксперимент качества"
        )
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "benchmark_only": True,
        "config": cfg,
        "train_epochs": int(tr.sum()),
        "validation_epochs": int(va.sum()),
        "features": tx.shape[1],
        "fold": fold,
        "actual_iterations": count,
        "train_subjects": sorted(set(groups[tr].tolist())),
        "validation_subjects": sorted(set(groups[va].tolist())),
        "fit_seconds": duration,
        "seconds_per_iteration_including_setup": duration / count,
        "note": "Whole-subject subsets of one fold. Includes quantization/setup; cannot linearly predict full CV runtime.",
    }
    name = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    save_json(out / f"{cfg['experiment_id']}_{name}.json", report)
    print(
        json.dumps(
            {k: v for k, v in report.items() if k != "config"},
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return report


def run_tree(
    config_path: str | Path,
    *,
    stage="all",
    data_root=None,
    output_root=None,
    cache_root=None,
    fold_root=None,
    threads=None,
    smoke: bool = False,
) -> dict:
    cfg = load_tree_config(
        config_path,
        data_root=data_root,
        output_root=output_root,
        cache_root=cache_root,
        fold_root=fold_root,
        threads=threads,
        smoke=smoke,
    )
    if cfg["training"].get("mode") == "fixed":
        if stage not in ("all", "fixed", "cache"):
            raise ValueError(
                "This config permits only --stage fixed (or cache); no CV/search/benchmark"
            )
        from src.training.fixed_tree import run_fixed_tree

        return run_fixed_tree(cfg, cache_only=stage == "cache")
    if stage == "fixed":
        raise ValueError("--stage fixed requires an explicitly fixed configuration")
    out = Path(cfg["paths"]["output_root"])
    namespace = out / "smoke" if smoke else out
    # Benchmark/cache checks do not occupy an experiment ID.
    with file_lock(out / ".export.lock", shared=True):
        x, meta, names, train, folds, cache, provenance = prepare_data(cfg)
        cfg["runtime"].update(
            {
                k: provenance[k]
                for k in ("fold_sha256", "labels_sha256", "cache_objects_sha256")
            }
        )
        if stage == "cache":
            return provenance
        if stage == "benchmark":
            if cfg["model"]["type"] == "auto":
                raise ValueError("Benchmark a concrete LightGBM/CatBoost config")
            return benchmark(x, meta, train, folds, cfg, out / "benchmarks")
        cfg = select_parent(cfg, namespace, provenance)
        if cfg["model"]["type"] not in ("lightgbm", "catboost") or cfg["model"][
            "class_weight"
        ] not in ("none", "sqrt"):
            raise ValueError("Unsupported model/weights")
        pool = candidates(cfg)
        folder = namespace / cfg["experiment_id"]
        with file_lock(folder / ".run.lock"):
            snapshot = folder / "resolved.yaml"
            cfg["runtime"].update(
                {
                    k: provenance[k]
                    for k in ("fold_sha256", "labels_sha256", "cache_objects_sha256")
                }
            )
            if (
                snapshot.exists()
                and OmegaConf.to_container(OmegaConf.load(snapshot), resolve=True)
                != cfg
            ):
                raise ValueError(
                    "Existing experiment settings/data/code differ. Preserve old results and use a separate output root."
                )
            OmegaConf.save(OmegaConf.create(cfg), snapshot)
            save_json(folder / "provenance.json", provenance)
            save_json(folder / "feature_schema.json", {"names": names, "cache": cache})
            if (folder / "metrics.json").exists():
                metrics = read_json(folder / "metrics.json")
                if not smoke:
                    update_journal(cfg, metrics, folder)
                print(f"Уже завершён: {folder}", flush=True)
                return metrics
            tx = x[train]
            tm = meta.filter(pl.col("split") == "train")
            y, ids, subjects = (
                tm["label"].to_numpy(),
                tm["id"].to_numpy().astype(str),
                tm["subject_id"].to_numpy(),
            )
            if stage == "baseline":
                with Progress("Базовая CV", 3, "fold") as progress:
                    return full_trial(
                        tx,
                        y,
                        subjects,
                        ids,
                        names,
                        folds,
                        cfg,
                        folder,
                        0,
                        pool[0],
                        progress,
                    )[0]
            if stage in ("all", "screen", "search"):
                with Progress("Screening: один fold", len(pool), "trial") as progress:
                    scored = [
                        fold_trial(
                            tx,
                            y,
                            subjects,
                            ids,
                            names,
                            folds,
                            cfg,
                            folder,
                            i,
                            p,
                            cfg["search"]["screen_fold"],
                            progress,
                        )[0]
                        for i, p in enumerate(pool)
                    ]
                shortlist = sorted(
                    range(len(pool)),
                    key=lambda i: (-scored[i]["macro_f1"], scored[i]["log_loss"], i),
                )[: cfg["search"]["top_k"]]
                save_json(
                    folder / "screening.json",
                    {
                        "shortlist": shortlist,
                        "candidates": pool,
                        "results": scored,
                        "fold": cfg["search"]["screen_fold"],
                    },
                )
                if stage == "screen":
                    return {"shortlist": shortlist}
                results, probabilities = {}, {}
                selected = sorted(set([0, *shortlist]))
                with Progress(
                    "Поиск: полная групповая CV", len(selected) * 3, "fold"
                ) as progress:
                    for i in selected:
                        results[i], probabilities[i] = full_trial(
                            tx,
                            y,
                            subjects,
                            ids,
                            names,
                            folds,
                            cfg,
                            folder,
                            i,
                            pool[i],
                            progress,
                        )
                best_id = min(
                    results,
                    key=lambda i: (
                        -results[i]["oof"]["macro_f1"],
                        results[i]["fold_f1_std"],
                        i,
                    ),
                )
                search_seconds = sum(
                    read_json(p)["seconds"]
                    for p in (folder / "trials").glob("*/fold_*.json")
                )
                save_json(
                    folder / "selection.json",
                    {
                        "best_trial": best_id,
                        "search_seconds": search_seconds,
                        "criterion": "pooled OOF macro-F1; then fold std; then candidate order",
                    },
                )
                pl.DataFrame(
                    [
                        {
                            "trial": i,
                            "params": json.dumps(r["params"], sort_keys=True),
                            "oof_macro_f1": r["oof"]["macro_f1"],
                            "fold_f1_std": r["fold_f1_std"],
                            "log_loss": r["oof"]["log_loss"],
                        }
                        for i, r in results.items()
                    ]
                ).write_csv(folder / "search.csv")
                if stage == "search":
                    return results[best_id]
            else:
                best_id = read_json(folder / "selection.json")["best_trial"]
                results, probabilities = {}, {}
                with Progress(
                    "Загрузка завершённых CV", len(set([0, best_id])) * 3, "fold"
                ) as progress:
                    for i in sorted(set([0, best_id])):
                        for fold in range(3):
                            if not (
                                folder
                                / "trials"
                                / f"trial_{i:03d}"
                                / f"fold_{fold}.json"
                            ).exists():
                                raise ValueError("Run --stage search before refit")
                            if not (
                                folder
                                / "trials"
                                / f"trial_{i:03d}"
                                / f"fold_{fold}.npz"
                            ).exists():
                                raise ValueError(
                                    "Stored OOF probabilities missing; run --stage search before refit"
                                )
                        results[i], probabilities[i] = full_trial(
                            tx,
                            y,
                            subjects,
                            ids,
                            names,
                            folds,
                            cfg,
                            folder,
                            i,
                            pool[i],
                            progress,
                        )
            del tx
            metrics = refit_models(
                x,
                meta,
                names,
                train,
                cfg,
                folder,
                results[0],
                results[best_id],
                probabilities[0],
                probabilities[best_id],
            )
            if not smoke:
                update_journal(cfg, metrics, folder)
            print(f"Завершён: {folder}", flush=True)
            return metrics


def predict_tree_submission(
    checkpoint: str | Path, *, data_root=None, cache_root=None, output=None
) -> Path:
    from src.inference import assemble_submission

    checkpoint = Path(checkpoint).resolve()
    folder = checkpoint.parent
    cfg = OmegaConf.to_container(OmegaConf.load(folder / "resolved.yaml"), resolve=True)
    if cfg["runtime"]["smoke"] or not tree_source_compatible(
        cfg["runtime"]["source_fingerprint"], tree_fingerprint()
    ):
        raise ValueError("Smoke checkpoint or source/environment mismatch")
    if not (folder / "metrics.json").exists():
        raise ValueError("Experiment is incomplete")
    bundle = read_json(folder / f"{checkpoint.stem}.model.json")
    if (
        bundle["model_file"] != checkpoint.name
        or hashlib.sha256(checkpoint.read_bytes()).hexdigest() != bundle["model_sha256"]
    ):
        raise ValueError("Checkpoint checksum mismatch")
    root = Path(data_root or cfg["paths"]["data_root"])
    target = Path(
        output or PROJECT_ROOT / "submissions" / f"{cfg['experiment_id']}.csv"
    )
    with (
        file_lock(Path(cfg["paths"]["output_root"]) / ".export.lock", shared=True),
        file_lock(target.with_suffix(".csv.lock")),
    ):
        if target.exists():
            raise FileExistsError(target)
        label_hash = hashlib.sha256((root / "labels.csv").read_bytes()).hexdigest()
        x, meta, names, cache = read_cached_features(
            Path(cache_root or cfg["paths"]["cache_root"]),
            load_labels(root),
            {**cfg["features"]["extractor"], "labels_sha256": label_hash},
            cfg["features"]["context"],
            test=True,
            data_root=root,
        )
        schema = read_json(folder / "feature_schema.json")
        if (
            names != bundle["feature_names"]
            or cache["key"] != schema["cache"]["key"]
            or bundle["class_order"] != [0, 1, 2, 3, 4]
        ):
            raise ValueError("Feature/class schema differs")
        model = load_model(checkpoint, bundle["family"])
        probabilities = predict(model, x, bundle["family"], cfg["training"]["threads"])
        submission = assemble_submission(
            meta["id"].to_numpy(),
            probabilities,
            pl.read_csv(root / "sample_submission.csv"),
        )
        with atomic_path(target) as temp:
            submission.write_csv(temp)
            with atomic_path(target.with_suffix(".meta.yaml")) as meta_temp:
                OmegaConf.save(
                    OmegaConf.create(
                        {
                            "experiment_id": cfg["experiment_id"],
                            "checkpoint": str(checkpoint),
                            "sha256": bundle["model_sha256"],
                            "class_order": bundle["class_order"],
                            "cache_key": cache["key"],
                        }
                    ),
                    meta_temp,
                )
    return target
