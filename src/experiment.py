"""Запуск экспериментов из YAML, воспроизводимость и независимый CSV-журнал."""

import csv
import hashlib
import json
import re
import subprocess
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from src.data import load_labels
from src.features import build_feature_table
from src.training.tabular import grouped_folds, run_linear, save_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def exclusive_file(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        stream = path.open("x", encoding="utf-8")
    except FileExistsError as error:
        raise RuntimeError(
            f"Another run or stale lock: {path}. Run experiments sequentially; inspect before removing stale lock."
        ) from error
    try:
        stream.write("Running\n")
        stream.close()
        yield
    finally:
        path.unlink(missing_ok=True)


def source_fingerprint() -> str:
    # Unrelated EDA/CNN modules do not invalidate the LR pipeline.
    modules = (
        "data.py",
        "features.py",
        "datasets.py",
        "preprocessing.py",
        "evaluation.py",
        "models/linear.py",
        "training/tabular.py",
        "experiment.py",
        "run.py",
        "inference.py",
        "progress.py",
    )
    paths = [PROJECT_ROOT / "src" / name for name in modules] + [
        PROJECT_ROOT / "uv.lock"
    ]
    return hashlib.sha256(b"".join(p.read_bytes() for p in paths)).hexdigest()


def read_git_state(root: Path) -> dict:
    """Git необязателен для загруженного на Kaggle архива исходников."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=root,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except OSError, subprocess.CalledProcessError:
        return {"git_commit": None, "git_dirty": None}
    return {"git_commit": commit, "git_dirty": dirty}


def load_config(
    path: str | Path,
    *,
    data_root: str | Path | None = None,
    output_root: str | Path | None = None,
    smoke: bool = False,
) -> dict:
    path = Path(path).resolve()
    config = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if config["experiment_id"] != path.stem or not re.fullmatch(
        r"\d{3}_[a-z]+", path.stem
    ):
        raise ValueError("Config ID must match NNN_family.yaml")
    for key, value in [("data_root", data_root), ("output_root", output_root)]:
        raw = value or config["paths"][key]
        resolved = Path(raw)
        config["paths"][key] = str(
            (PROJECT_ROOT / resolved).resolve()
            if not resolved.is_absolute()
            else resolved.resolve()
        )
    config["paths"]["cache_root"] = str(
        Path(config["paths"]["output_root"]) / "cache/features"
    )
    config["runtime"] = {"smoke": smoke, "source_fingerprint": source_fingerprint()}
    config["config_path"] = str(path)
    return config


def select_parent(config: dict, namespace: Path) -> dict:
    candidates = config.get("parent_candidates", [])
    if not candidates:
        return config
    parents = []
    for name in candidates:
        folder = namespace / name
        if not (folder / "metrics.json").exists():
            raise ValueError(f"Run parent {name} first")
        resolved = OmegaConf.to_container(
            OmegaConf.load(folder / "resolved.yaml"), resolve=True
        )
        if (
            resolved["runtime"]["source_fingerprint"]
            != config["runtime"]["source_fingerprint"]
        ):
            raise ValueError("Parent code/environment differs; do not mix experiments")
        metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
        parents.append((metrics["tuned"]["oof"]["macro_f1"], name, resolved))
    score, name, parent = sorted(parents, key=lambda p: (-p[0], p[1]))[0]
    if (
        parent["seed"] != config["seed"]
        or parent["paths"]["data_root"] != config["paths"]["data_root"]
        or parent["features"]["extractor"] != config["features"]["extractor"]
    ):
        raise ValueError("Incompatible parent settings")
    config["preprocessing"] = parent["preprocessing"].copy()
    config["model"]["class_weight"] = parent["model"]["class_weight"]
    config["selected_parent"] = {
        "experiment_id": name,
        "tuned_oof_macro_f1": score,
        "selection": "OOF only; no val selection",
    }
    return config


def update_journal(config: dict, metrics: dict, folder: Path) -> None:
    path = PROJECT_ROOT / "experiments.csv"
    with exclusive_file(path.with_suffix(".csv.lock")):
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            fields = list(reader.fieldnames)
            rows = list(reader)
        for extra in ("best_c", "oof_macro_f1", "fixed_oof_macro_f1", "fold_f1_std"):
            if extra not in fields:
                fields.append(extra)
        result = metrics["val"]["tuned"]
        row = {
            "experiment_id": config["experiment_id"],
            "config": str(Path(config["config_path"]).relative_to(PROJECT_ROOT)),
            "model": "LogisticRegression",
            "summary": f"{config['features']['context']}; {config['preprocessing']['transform']}; weights={config['model']['class_weight']}",
            "seed": config["seed"],
            "evaluation_set": "provided_val_after_train_group_cv",
            "macro_f1": result["macro_f1"],
            "train_seconds": metrics["search_seconds"] + metrics["refit_seconds"],
            "infer_seconds": metrics["inference_seconds"],
            "status": "complete",
            "artefacts_path": str(folder),
            "best_c": metrics["tuned"]["C"],
            "oof_macro_f1": metrics["tuned"]["oof"]["macro_f1"],
            "fixed_oof_macro_f1": metrics["fixed"]["oof"]["macro_f1"],
            "fold_f1_std": metrics["tuned"]["fold_f1_std"],
        }
        row.update(
            dict(
                zip(
                    ("f1_w", "f1_n1", "f1_n2", "f1_n3", "f1_rem"),
                    result["f1"],
                    strict=True,
                )
            )
        )
        rows = [r for r in rows if r["experiment_id"] != row["experiment_id"]] + [row]
        temp = path.with_suffix(".csv.tmp")
        with temp.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        temp.replace(path)


def run_experiment(
    config_path: str | Path,
    *,
    data_root: str | Path | None = None,
    output_root: str | Path | None = None,
    smoke: bool = False,
    prepare: bool = False,
) -> dict:
    config = load_config(
        config_path, data_root=data_root, output_root=output_root, smoke=smoke
    )
    out = Path(config["paths"]["output_root"])
    out.mkdir(parents=True, exist_ok=True)
    namespace = out / "smoke" if smoke else out
    with exclusive_file(out / ".linear.lock"):
        # Feature preparation is independent of parent model selection.
        if not prepare:
            config = select_parent(config, namespace)
        if (
            config["model"]["type"] != "logistic_regression"
            or config["model"]["solver"] != "lbfgs"
            or config["model"]["l1_ratio"] != 0
        ):
            raise ValueError("This runner supports multinomial L2 LR with lbfgs only")
        if config["cv"]["strategy"] != "StratifiedGroupKFold":
            raise ValueError("Unknown CV strategy")
        if config["features"]["context"] not in ("none", "neighbors", "multires"):
            raise ValueError("Unknown context")
        if config["data_revision"] != "full_psg_v1" or config["cv"]["n_splits"] != 3:
            raise ValueError("Expected full_psg_v1 and three folds")
        if not prepare and (
            config["preprocessing"]["transform"] not in ("quantile", "log_robust")
            or config["model"]["class_weight"] not in ("none", "sqrt", "balanced")
        ):
            raise ValueError("Unresolved/invalid approach settings")
        if any(float(c) <= 0 for c in config["search"]["C"]):
            raise ValueError("C must be positive")
        labels = load_labels(Path(config["paths"]["data_root"]))
        label_hash = hashlib.sha256(
            (Path(config["paths"]["data_root"]) / "labels.csv").read_bytes()
        ).hexdigest()
        config["runtime"]["labels_sha256"] = label_hash
        config["runtime"]["manifest_sha256"] = hashlib.sha256(
            (Path(config["paths"]["data_root"]) / "MANIFEST.json").read_bytes()
        ).hexdigest()
        folder = namespace / config["experiment_id"]
        if not prepare:
            folder.mkdir(parents=True, exist_ok=True)
            snapshot = OmegaConf.create(config)
            if (folder / "resolved.yaml").exists():
                old = OmegaConf.to_container(
                    OmegaConf.load(folder / "resolved.yaml"), resolve=True
                )
                if old != config:
                    raise ValueError(
                        "Existing experiment has different settings; use a new ID/output root"
                    )
                if (folder / "metrics.json").exists():
                    metrics = json.loads(
                        (folder / "metrics.json").read_text(encoding="utf-8")
                    )
                    if not smoke:
                        update_journal(config, metrics, folder)
                    print(f"Already complete: {folder}")
                    return metrics
            else:
                OmegaConf.save(snapshot, folder / "resolved.yaml")
        x, metadata, names, cache = build_feature_table(
            Path(config["paths"]["data_root"]),
            Path(config["paths"]["cache_root"]),
            labels,
            config["features"]["extractor"],
            config["features"]["context"],
            smoke=smoke,
        )
        print(
            f"Features: {x.shape}; matrix={x.nbytes / 2**20:.1f} MiB; cache hits={cache['cache_hits']}",
            flush=True,
        )
        if prepare:
            return cache
        save_json(
            folder / "feature_schema.json",
            {"names": names, "cache": cache, "nan_fraction": float(np.isnan(x).mean())},
        )
        fold_metadata = metadata.filter(metadata["split"] == "train")
        fold_path = (
            namespace / "splits" / f"groups_{config['seed']}_{label_hash[:12]}.csv"
        )
        folds = grouped_folds(
            fold_metadata, fold_path, config["seed"], config["cv"]["n_splits"]
        )
        save_json(
            folder / "provenance.json",
            {
                "fold_file": str(fold_path),
                "fold_sha256": hashlib.sha256(fold_path.read_bytes()).hexdigest(),
                "source_fingerprint": config["runtime"]["source_fingerprint"],
                **read_git_state(PROJECT_ROOT),
                "smoke": smoke,
                "cache_key": cache["key"],
            },
        )
        metrics = run_linear(x, metadata, names, config, folder, folds)
        if not smoke:
            update_journal(config, metrics, folder)
        print(f"Completed: {folder}", flush=True)
        return metrics
