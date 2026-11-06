"""124–146: physiology, contrastive heads, calibrated signals and honest fusion."""

import argparse
import csv
import gc
import json
import shutil
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import torch
from omegaconf import OmegaConf

from src.models.pretrained_assets import sha256
from src.physiology_data import PhysiologyInputs, prepare_residual_inputs
from src.physiology_features import raw_pool, record_features
from src.pretrained import ROOT, git_state, hardware, save_json, snapshot
from src.progress import Progress
from src.refinement_data import RefinementData
from src.refinement_run import StageProgress
from src.series_data import fingerprint, load_prediction, save_prediction
from src.series_results import ensure_submission, evaluate_values, journal, write_split
from src.storage import file_lock
from src.training.physiology import (
    neural_experiment,
    predict,
    tree_experiment,
    tree_probabilities,
)
from src.training.refinement import transform
from src.voting_reports import comparison


class PhysiologyProgress(StageProgress):
    """One stage display; bar counts actual records/batches, never convergence."""

    def stage(self, name, detail=""):
        self.work(0, 0, "этап")
        super().stage(name, detail)

    def work(self, completed, total, unit="batch"):
        with self.progress._lock:
            if (
                completed < self.progress.n
                or total != self.progress.total
                or unit != self.progress.unit
            ):
                self.progress.started = time.monotonic()
            self.progress.n = completed
            self.progress.total = total
            self.progress.unit = unit


def feature_job(parameters):
    data, path, split, root, mode, channel = parameters
    # BLAS is bounded in launcher; each native detector uses one worker.
    return record_features(data, path, split, root, mode, channel)


def feature_worker_init(root, smoke, epochs, threads):
    global _WORKER_DATA, _WORKER_LIMIT
    import logging

    import numba
    from threadpoolctl import threadpool_limits

    torch.set_num_threads(threads)
    numba.set_num_threads(threads)
    _WORKER_LIMIT = threadpool_limits(limits=threads)
    logging.getLogger("yasa").setLevel(logging.ERROR)
    _WORKER_DATA = RefinementData(root, smoke, epochs)


def feature_worker(job):
    path, split, root, mode, channel = job
    return record_features(_WORKER_DATA, path, split, root, mode, channel)


class Context:
    def __init__(self, data, args, progress):
        self.data, self.args, self.progress = data, args, progress
        self.inputs = PhysiologyInputs(data, args, progress)
        self.mode = "smoke" if data.smoke else "full"
        self.namespace = args.output_root / "smoke" if data.smoke else args.output_root
        self.memo = {}

    def features(self, split, kind="events", channel=0):
        key = (split, kind, channel)
        if key not in self.memo:
            paths = self.data.paths(split)
            jobs = [
                (self.data, p, split, self.args.cache_root, kind, channel)
                for p in paths
            ]
            self.progress.stage(
                "Подготовка", f"{kind}/{split}: общий кэш; записи 0/{len(paths)}"
            )
            results = []
            self.progress.work(0, len(paths), "записей")
            # Record-level CPU parallelism, one progress display for all records.
            if self.args.workers > 1 and not self.data.smoke:
                with ProcessPoolExecutor(
                    max_workers=min(self.args.workers, len(paths)),
                    mp_context=get_context("spawn"),
                    initializer=feature_worker_init,
                    initargs=(
                        self.data.root,
                        self.data.smoke,
                        self.data.smoke_epochs,
                        max(1, self.args.threads // self.args.workers),
                    ),
                ) as pool:
                    lightweight = [
                        (p, split, self.args.cache_root, kind, channel) for p in paths
                    ]
                    for i, result in enumerate(
                        pool.map(feature_worker, lightweight), 1
                    ):
                        results.append(result)
                        self.progress.work(i, len(paths), "записей")
                        self.progress.update(
                            detail=f"{kind}/{split}: записи {i}/{len(paths)}"
                        )
            else:
                for i, job in enumerate(jobs, 1):
                    self.progress.update(
                        detail=f"{kind}/{split}: запись {i}/{len(paths)} · YASA native extraction"
                    )
                    results.append(feature_job(job))
                    self.progress.work(i, len(paths), "записей")
            self.memo[key] = np.concatenate([r[0] for r in results])
            save_json(
                self.namespace / f"{kind}_{split}_feature_sources.json",
                [r[1] for r in results],
            )
        return self.memo[key]

    def matrix(self, cfg, split, components=None):
        kind = cfg["model"]["kind"]
        if kind == "contrastive":
            return self.inputs.embeddings(split), None
        if kind == "dual_stream":
            return raw_pool(self.data, split, self.args.cache_root, self.progress), None
        if kind == "event_tree":
            x = self.features(split)
            if cfg["model"]["handcrafted"]:
                x = np.concatenate([x, self.inputs.physio(split)], 1)
            return x, None
        if kind == "residual_calibration":
            return self.inputs.residual(
                split,
                self.features(split) if cfg["model"]["events"] else None,
                components,
            )
        raise ValueError(kind)

    def candidate(self, name, split):
        if name == "113_pretrained":
            return self.inputs.prediction(name, split)
        folder = self.namespace / name
        if not (folder / "metrics.json").exists():
            raise FileNotFoundError(
                f"{name}: complete smoke/full source required before fixed ensemble"
            )
        target = folder / (
            "oof_predictions.npz" if split == "train" else f"{split}_predictions.npz"
        )
        info = json.loads(target.with_suffix(".json").read_text())
        provenance = json.loads((folder / "provenance.json").read_text())
        if (
            info["prediction_sha256"] != sha256(target)
            or info["contract"] != provenance["contract"]
        ):
            raise ValueError("Candidate prediction integrity differs")
        return load_prediction(target, self.data.metadata(split))["probabilities"]


def implementation():
    paths = [
        *ROOT.glob("src/physiology_*.py"),
        ROOT / "src/training/physiology.py",
        ROOT / "src/series_data.py",
        ROOT / "src/series_results.py",
        ROOT / "environments/pretrained/uv.lock",
        *(
            ROOT / relative
            for relative in (
                "src/data.py",
                "src/evaluation.py",
                "src/storage.py",
                "src/expansion_data.py",
                "src/refinement_data.py",
                "src/refinement_models.py",
                "src/training/refinement.py",
                "src/models/tinysleepnet.py",
                "src/models/pretrained_assets.py",
                "src/pretrained_signals.py",
            )
        ),
    ]
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in paths}


def memory_usage():
    try:
        import resource

        return {
            "peak_parent_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * 1024,
            "peak_child_rss_bytes": resource.getrusage(
                resource.RUSAGE_CHILDREN
            ).ru_maxrss
            * 1024,
            "scope": "Linux process high-water RSS; child peak is not simultaneous total",
        }
    except ImportError:
        return {"scope": "CPU RSS not collected on this platform"}


def save_ablation_comparisons(ctx: Context) -> None:
    pairs = [
        ("127_pretrained", "126_pretrained"),
        ("128_pretrained", "127_pretrained"),
        ("130_pretrained", "129_pretrained"),
        ("131_pretrained", "130_pretrained"),
        ("133_tree", "132_tree"),
        ("135_pretrained", "134_pretrained"),
    ]
    for candidate, baseline in pairs:
        if not all(
            (ctx.namespace / n / "metrics.json").exists() for n in (candidate, baseline)
        ):
            continue
        for split in ("train", "val"):
            target = (
                ctx.namespace
                / candidate
                / "comparisons"
                / f"{split}_vs_{baseline}.json"
            )
            if target.exists():
                continue
            meta = ctx.data.metadata(split)
            report = comparison(
                {**meta, "probabilities": ctx.candidate(candidate, split)},
                {**meta, "probabilities": ctx.candidate(baseline, split)},
            )
            report["candidate_metrics"] = report.pop("voting_metrics")
            for counts in report["confusion_directions"].values():
                counts["candidate"] = counts.pop("voting")
            save_json(
                target,
                {
                    "candidate": candidate,
                    "baseline": baseline,
                    "evaluation_set": "train OOF"
                    if split == "train"
                    else "provided val",
                    **report,
                },
            )


def run_config(path: Path, cfg: dict, ctx: Context) -> None:
    args, data, progress = ctx.args, ctx.data, ctx.progress
    name = cfg["experiment_id"]
    folder = ctx.namespace / name
    folder.mkdir(parents=True, exist_ok=True)
    contract = fingerprint(
        {
            "config": cfg,
            "data": data.identity,
            "smoke": data.smoke,
            "smoke_epochs": data.smoke_epochs if data.smoke else None,
            "implementation": implementation(),
            "inputs": ctx.inputs.manifest["files"],
        }
    )
    provenance = {
        "config": cfg,
        "contract": contract,
        "cpu_memory_at_start": memory_usage(),
        "data": data.identity,
        "smoke": data.smoke,
        "git": git_state(),
        "environment": hardware(args.device),
        "implementation": implementation(),
    }
    old = folder / "provenance.json"
    if old.exists() and json.loads(old.read_text())["contract"] != contract:
        raise ValueError(
            "Experiment resume contract differs; preserve old results and choose separate output root"
        )
    if not old.exists():
        save_json(old, provenance)
    OmegaConf.save(OmegaConf.create(cfg), folder / "resolved.yaml")
    if (folder / "metrics.json").exists():
        for split in ("val", "test"):
            target = folder / f"{split}_predictions.npz"
            if json.loads(target.with_suffix(".json").read_text())[
                "prediction_sha256"
            ] != sha256(target):
                raise ValueError("Saved predictions checksum differs")
            value = load_prediction(target, data.metadata(split))
            if split == "test":
                ensure_submission(folder, value, data)
        if not data.smoke:
            journal(
                path,
                cfg,
                json.loads((folder / "metrics.json").read_text()),
                folder,
                args.output_root,
            )
        save_ablation_comparisons(ctx)
        return
    kind = cfg["model"]["kind"]
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(args.device)
    trained = args.trained_root / ctx.mode / name
    training_seconds = None
    model, scale = None, None
    selected = None
    if kind == "selected_ensemble":
        groups = cfg["model"]["candidate_groups"]
        selected = []
        records = []
        for group in groups:
            choices = []
            for n in group:
                source = ctx.namespace / n / "selection.json"
                if not source.exists() or not (source.parent / "metrics.json").exists():
                    raise FileNotFoundError(
                        f"Need complete {n}, including train OOF, before selection"
                    )
                score = json.loads(source.read_text())["oof"]["macro_f1"]
                choices.append((score, n))
            score, n = max(choices)
            selected.append(n)
            records.append({"candidates": choices, "selected": n, "score": score})
        save_json(
            folder / "selection.json",
            {
                "selection_set": "train OOF only",
                "groups": records,
                "no_val_test_selection": True,
            },
        )
    elif kind not in ("yasa_staging", "fixed_ensemble"):
        x, base = ctx.matrix(cfg, "train")
        if kind == "event_tree":
            model, scale, training_seconds = tree_experiment(
                cfg, ctx.inputs, x, folder, trained, contract
            )
        else:
            fold_inputs = None
            if kind == "residual_calibration":
                root = args.cache_root / ctx.mode / "role_stack"
                if not (root / "ready.json").exists():
                    raise FileNotFoundError(
                        "First RUN_GROUP='fusion_prepare': honest A/B/C/D probabilities, no naive re-CV of102/106 OOF"
                    )

                def fold_inputs(fold):
                    file = root / f"outer_{fold}.npz"
                    if json.loads(file.with_suffix(".json").read_text())[
                        "sha256"
                    ] != sha256(file):
                        raise ValueError("Role input checksum differs")
                    with np.load(file, allow_pickle=False) as z:
                        if not np.array_equal(z["ids"], data.metadata("train")["ids"]):
                            raise ValueError("Role epoch IDs differ")
                        xx, bb = ctx.matrix(cfg, "train", z["components"])
                        return xx, bb, z["calibration_fit"], z["held"]

            model, scale, training_seconds = neural_experiment(
                cfg, ctx.inputs, x, base, folder, trained, contract, fold_inputs
            )
    times = {}
    for split in ("val", "test"):
        progress.stage(
            "Validation" if split == "val" else "Test",
            "inference, сохранение probabilities",
        )
        meta = data.metadata(split)
        start = time.perf_counter()
        target = folder / f"{split}_predictions.npz"
        previous = None
        if target.with_suffix(".json").exists():
            previous = json.loads(target.with_suffix(".json").read_text())
            if previous["contract"] != contract or previous[
                "prediction_sha256"
            ] != sha256(target):
                raise ValueError("Partial split resume differs")
            p = load_prediction(target, meta)["probabilities"]
        elif kind == "yasa_staging":
            p = ctx.features(split, "yasa", cfg["model"]["channel"])
        elif kind in ("fixed_ensemble", "selected_ensemble"):
            names = cfg["model"].get("members") or ["113_pretrained", *selected]
            p = sum(
                w * ctx.candidate(n, split)
                for n, w in zip(names, cfg["model"]["weights"], strict=True)
            )
        else:
            xx, bb = ctx.matrix(cfg, split)
            if kind == "event_tree":
                p = tree_probabilities(model, xx, args.threads)
            else:
                mean, std = scale
                xx = (
                    xx
                    if mean is None
                    else torch.as_tensor(transform(xx, mean, std), device=args.device)
                )
                bb = None if bb is None else torch.as_tensor(bb, device=args.device)
                p = predict(
                    model,
                    kind,
                    xx,
                    bb,
                    np.arange(len(meta["ids"])),
                    args.device,
                    progress,
                )
        p = np.asarray(p, np.float32)
        p /= p.sum(1, keepdims=True)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize(args.device)
        times[split] = (
            previous["seconds"] if previous is not None else time.perf_counter() - start
        )
        values = {**meta, "probabilities": p}
        if previous is None:
            write_split(
                folder, split, values, data, {**provenance, "seconds": times[split]}
            )
        elif split == "test":
            ensure_submission(folder, values, data)
        if split == "val":
            before = ctx.inputs.prediction("113_pretrained", split)
            report = comparison(values, {**meta, "probabilities": before})
            report["candidate_metrics"] = report.pop("voting_metrics")
            for counts in report["confusion_directions"].values():
                counts["candidate"] = counts.pop("voting")
            save_json(folder / "comparison_113.json", report)
        if kind == "residual_calibration":
            delta = np.log(np.maximum(p, 1e-7)) - np.log(
                np.maximum(ctx.inputs.residual(split)[1], 1e-7)
            )
            save_json(
                folder / f"{split}_corrections.json",
                {
                    "log_probability_delta_mean": delta.mean(0).tolist(),
                    "log_probability_delta_std": delta.std(0).tolist(),
                    "quantiles": np.quantile(delta, [0.05, 0.5, 0.95], axis=0).tolist(),
                },
            )
    progress.stage("Сохранение", "метрики, журнал, checkpoints")
    if kind in ("fixed_ensemble", "selected_ensemble"):
        members = cfg["model"].get("members") or ["113_pretrained", *selected]
        if all(
            n == "113_pretrained"
            or (ctx.namespace / n / "oof_predictions.npz").exists()
            for n in members
        ):
            from src.evaluation import classification_metrics

            train = data.metadata("train")
            pooled = sum(
                w * ctx.candidate(n, "train")
                for n, w in zip(members, cfg["model"]["weights"], strict=True)
            )
            save_prediction(
                folder / "oof_predictions.npz",
                {**train, "probabilities": pooled},
                {
                    "contract": contract,
                    "protocol": "fixed blend of aligned train OOF; reused for component selection, not nested unbiased estimate",
                },
            )
            score = classification_metrics(train["labels"], pooled)
            baseline = classification_metrics(
                train["labels"], ctx.inputs.prediction("113_pretrained", "train")
            )
            selection = (
                json.loads((folder / "selection.json").read_text())
                if (folder / "selection.json").exists()
                else {}
            )
            selection.update(
                oof=score,
                baseline113_oof=baseline,
                recommended_by_train_oof=score["macro_f1"] > baseline["macro_f1"],
                members=members,
                weights=cfg["model"]["weights"],
            )
            save_json(folder / "selection.json", selection)
    if kind == "selected_ensemble":
        families = {
            "contrastive": ["127_pretrained", "128_pretrained"],
            "calib": ["129_pretrained", "130_pretrained", "131_pretrained"],
            "events": ["132_tree", "133_tree"],
        }
        summary = {}
        for family, names in families.items():
            if not all((ctx.namespace / n / "selection.json").exists() for n in names):
                summary[family] = {
                    "status": "not all candidates completed; no selection"
                }
                continue
            scores = [
                (
                    json.loads((ctx.namespace / n / "selection.json").read_text())[
                        "oof"
                    ]["macro_f1"],
                    n,
                )
                for n in names
            ]
            _, chosen = max(scores)
            summary[family] = {
                "train_oof_scores": scores,
                "chosen": chosen,
                "existing_75_25_mixture": f"{136 + int(chosen[:3]) - 124:03d}_pretrained",
            }
        save_json(
            ctx.namespace / "family_selection.json",
            {
                "selection_set": "train OOF only",
                "families": summary,
                "F1_F2_F3": "reuse existing chosen75/25 experiment IDs; no duplicate configurations",
            },
        )
    validation = evaluate_values(
        load_prediction(folder / "val_predictions.npz", data.metadata("val"))
    )
    if (folder / "oof_predictions.npz").exists():
        save_json(
            folder / "oof_metrics.json",
            evaluate_values(
                load_prediction(folder / "oof_predictions.npz", data.metadata("train"))
            ),
        )
    metrics = {
        "validation": validation,
        "smoke": data.smoke,
        "training_seconds": training_seconds,
        "cpu_memory": memory_usage(),
        "inference_seconds": sum(times.values()),
        "split_seconds": times,
        "contract": contract,
        "peak_gpu_bytes": torch.cuda.max_memory_allocated(args.device)
        if args.device.startswith("cuda")
        else None,
        "memory_note": "GPU peak is per process; native feature extraction runs in bounded CPU workers",
        "timing_note": "training_seconds sums saved fits/epochs; inference includes cache acquisition; separate prepare history records preprocessing wall time",
    }
    if kind == "event_tree":
        with (ROOT / "experiments.csv").open(encoding="utf-8", newline="") as stream:
            controls = [
                r
                for r in csv.DictReader(stream)
                if r["experiment_id"]
                in ("005_linear", "007_linear", "011_tree", "015_tree")
            ]
        save_json(
            folder / "tabular_controls.json",
            {
                "candidate_provided_val": validation,
                "historical_provided_val_rows": controls,
                "source": "immutable project experiments.csv; not train OOF or Kaggle test scores",
            },
        )
    save_json(folder / "metrics.json", metrics)
    if not data.smoke:
        journal(path, cfg, metrics, folder, args.output_root)
        dest = args.output_root / "submissions" / f"{name}.csv"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(folder / "submission.csv", dest)
    (folder / "last_error.json").unlink(missing_ok=True)
    save_ablation_comparisons(ctx)
    del model
    # Immutable event/YASA arrays stay in RAM for other configs in this process.
    gc.collect()
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()


def export_folder(root, target, exclude=()):
    root.mkdir(parents=True, exist_ok=True)
    if "signal_pool" in exclude:
        save_json(
            root / "cache_manifest.json",
            {
                "kind": "physiology_cache_v1",
                "excludes": "raw signals and temporary signal pools",
            },
        )
    files = [
        p
        for p in root.rglob("*")
        if p.is_file()
        and not set(p.relative_to(root).parts) & set(exclude)
        and p.suffix not in (".lock",)
        and not (p.name.startswith(".") and ".tmp" in p.name)
        and p.resolve() != target.resolve()
    ]
    with ZipFile(target, "w", ZIP_DEFLATED, compresslevel=4) as z:
        for p in files:
            z.write(p, p.relative_to(root).as_posix())
    print(
        f"ZIP: {target.name}, {target.stat().st_size / 2**20:.1f} MiB, {len(files)} files"
    )


def pack_input(target):
    files = {}
    for name in (
        "019_pretrained",
        "082_pretrained",
        "085_pretrained",
        "099_pretrained",
        "102_pretrained",
        "106_pretrained",
        "113_pretrained",
    ):
        folder = ROOT / "artefacts" / name
        for f in folder.glob("*"):
            if f.is_file() and (
                f.name in ("provenance.json", "resolved.yaml", "selection.json")
                or f.name.startswith(
                    (
                        "val_predictions.",
                        "test_predictions.",
                        "oof_predictions.",
                        "train_predictions.",
                    )
                )
            ):
                files[f"results/{name}/{f.name}"] = f
    # Frozen019 training probability cache from the existing refinement bundle.
    source = ROOT / "artefacts/cachepacks/sleep-edf-refinement-inputs.zip"
    extra = {}
    with ZipFile(source) as z:
        for n in z.namelist():
            if (
                n.startswith("results/predictions/")
                or n.endswith("definition.json")
                or n.endswith("_epoch.npz")
            ):
                extra["legacy/" + n] = z.read(n)
    for f in (ROOT / "artefacts/cache/refinement/full/embeddings").rglob("*"):
        if f.is_file() and f.suffix in (".json", ".npz"):
            files[
                "representations/"
                + f.relative_to(ROOT / "artefacts/cache/refinement").as_posix()
            ] = f
    for outer in range(3):
        for n in ("082", "085"):
            root = (
                ROOT
                / f"artefacts/trained-checkpoints/full/nested/outer_{outer}/inner_0/{n}"
            )
            for name in (
                "best.pt",
                "held.json",
                "held.npz",
                "trainable_parameters.json",
            ):
                files[f"nested/outer_{outer}/inner_0/{n}/{name}"] = root / name
    for name in (
        "tinysleepnet.pt",
        "tinysleepnet-source.py",
        "tinysleepnet-config.json",
    ):
        files["assets/" + name] = ROOT / "docs/expansion-assets" / name
    missing = [str(p) for p in files.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError(str(missing))
    import hashlib

    manifest = {
        "kind": "physiology_inputs_v1",
        "data": RefinementData(ROOT / "data/raw").identity,
        "files": {
            **{n: sha256(p) for n, p in files.items()},
            **{n: hashlib.sha256(b).hexdigest() for n, b in extra.items()},
        },
        "purpose": "saved full134 embeddings,106 features, trueOOF and six existing nested experts; no new inference",
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(target, "w", ZIP_DEFLATED, compresslevel=4) as z:
        for n, p in files.items():
            z.write(p, n)
        for n, b in extra.items():
            z.writestr(n, b)
        z.writestr("bundle.json", json.dumps(manifest, indent=2))
    print(
        f"Input ZIP {target}: {target.stat().st_size / 2**20:.1f} MiB, {len(manifest['files'])} files"
    )


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("configs", nargs="*", type=Path)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/raw")
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "artefacts/physiology"
    )
    parser.add_argument(
        "--cache-root", type=Path, default=ROOT / "artefacts/cache/physiology"
    )
    parser.add_argument(
        "--trained-root", type=Path, default=ROOT / "artefacts/physiology-trained"
    )
    parser.add_argument("--input-root", type=Path, action="append", default=[])
    parser.add_argument(
        "--checkpoint-cache", type=Path, default=Path("/tmp/physiology-checkpoints")
    )
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--feature-cache", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-epochs", type=int, default=160)
    parser.add_argument("--allow-training", action="store_true")
    parser.add_argument("--prepare", choices=("features", "signals", "fusion"))
    parser.add_argument("--report-tag", default="main")
    parser.add_argument("--export", type=Path)
    parser.add_argument("--export-cache", type=Path)
    parser.add_argument("--export-trained", type=Path)
    parser.add_argument("--pack-input", type=Path)
    args = parser.parse_args()
    if args.pack_input:
        pack_input(args.pack_input)
        return
    for target, root, exclude in (
        (args.export, args.output_root, ()),
        (args.export_cache, args.cache_root, ("signal_pool", "pools", "signals")),
        (args.export_trained, args.trained_root, ()),
    ):
        if target:
            export_folder(root, target, exclude)
            return
    if not args.input_root:
        parser.error("--input-root must point to the new verified physiology bundle")
    torch.set_num_threads(args.threads)
    args.checkpoint_root = args.input_root[0] / "assets"
    if args.feature_cache is None:
        candidates = list(args.input_root[0].rglob("definition.json"))
        if len(candidates) != 1:
            raise ValueError("Expected one106-feature definition in bundle")
        args.feature_cache = candidates[0].parent
    data = RefinementData(args.data_root, args.smoke, args.smoke_epochs)
    namespace = args.output_root / "smoke" if args.smoke else args.output_root
    namespace.mkdir(parents=True, exist_ok=True)
    # One shared source snapshot, never one per configuration/fold.
    marker = args.output_root / "implementation.json"
    current = implementation()
    if marker.exists() and json.loads(marker.read_text()) != current:
        raise ValueError("Output root belongs to another code version")
    if not marker.exists():
        snapshot(args.output_root)
        save_json(marker, current)
    failures = []
    with Progress("Physiology: подготовка", total=0, unit="этап") as raw:
        progress = PhysiologyProgress(raw)
        ctx = Context(data, args, progress)
        if args.prepare:
            started = time.perf_counter()
            progress.config("prepare", 1, 1)
            if args.prepare == "features":
                for split in ("train", "val", "test"):
                    ctx.features(split)
            elif args.prepare == "signals":
                for split in ("train", "val", "test"):
                    raw_pool(data, split, args.cache_root, progress)
            else:
                if not args.allow_training:
                    raise ValueError(
                        "fusion prepare fits cheap heads; --allow-training required"
                    )
                prepare_residual_inputs(ctx.inputs)
            report = namespace / f"prepare_{args.prepare}_history.json"
            with file_lock(report.with_suffix(".lock"), wait=True):
                history = json.loads(report.read_text()) if report.exists() else []
                history.append(
                    {
                        "prepare": args.prepare,
                        "seconds": time.perf_counter() - started,
                        "smoke": data.smoke,
                        "data": data.identity,
                        "cpu_memory": memory_usage(),
                        "timestamp_unix": time.time(),
                        "semantics": "wall time for this invocation; repeated cache-only invocations retained separately",
                    }
                )
                save_json(report, history)
        for i, path in enumerate(args.configs, 1):
            cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
            if cfg["experiment_id"] != path.stem:
                raise ValueError("Config filename/ID mismatch")
            if cfg["training"]["enabled"] and not args.allow_training:
                raise ValueError("--allow-training required")
            progress.config(path.stem, i, len(args.configs))
            progress.plan = (
                list(range(1, 9)) if cfg["training"]["enabled"] else [1, 6, 7, 8]
            )
            try:
                run_config(path, cfg, ctx)
            except Exception as exc:
                folder = namespace / path.stem
                folder.mkdir(parents=True, exist_ok=True)
                failure = {
                    "experiment_id": path.stem,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
                save_json(folder / "last_error.json", failure)
                failures.append(failure)
                print(
                    f"{path.stem} failed: {exc}; независимые эксперименты продолжаются"
                )
                gc.collect()
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()
    save_json(namespace / f"physiology_failures-{args.report_tag}.json", failures)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
