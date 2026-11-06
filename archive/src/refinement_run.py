"""100–117: learned experts, frozen long-context heads and coverage Tiny FT."""

import argparse
import gc
import hashlib
import json
import os
import re
import shutil
import time
import traceback
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import torch
from omegaconf import OmegaConf

from src.expansion_data import selected_signal
from src.models.pretrained_assets import sha256
from src.models.wav2sleep import MANIFEST, Wav2SleepAdapter
from src.pretrained import (
    ROOT,
    export_results,
    git_state,
    hardware,
    save_json,
    snapshot,
)
from src.pretrained_signals import contiguous_segments
from src.progress import Progress
from src.refinement_data import RefinementData, SharedInputs, nested_experts
from src.refinement_models import PRIOR
from src.series_data import fingerprint, load_prediction, save_prediction
from src.series_results import ensure_submission, evaluate_values, journal, write_split
from src.storage import file_lock
from src.training.refinement import (
    residual_experiment,
    residual_predict,
    tiny_experiment,
    tiny_predict,
)


class StageProgress:
    """One display with group/config/phase/fold/batch and actual elapsed times."""

    def __init__(self, progress):
        self.progress, self.name, self.position, self.total = (
            progress,
            "подготовка",
            0,
            0,
        )
        self.phase, self.scope, self.phase_name = 1, "", "подготовка"
        self.started = self.phase_started = time.monotonic()
        self.plan = list(range(1, 9))

    def config(self, name, position, total):
        self.name, self.position, self.total = name, position, total
        number = int(name[:3]) if name[:3].isdigit() else None
        self.plan = (
            list(range(1, 9))
            if number in [*range(100, 105), *range(106, 111), 117]
            else [1, 6, 7, 8]
            if number is not None
            else [1]
        )
        self.started = time.monotonic()
        self.scope = ""
        self.stage("Подготовка", "загрузка готовых входов")

    def stage(self, name, detail=""):
        if name != self.phase_name:
            self.phase_started = time.monotonic()
        self.phase_name = name
        if "fold" in name:
            match = re.search(r"fold (\d)", name)
            self.phase = 1 + int(match[1]) if match else 2
        elif "refit" in name.lower():
            self.phase = 5
        elif name == "Validation":
            self.phase = 6
        elif name == "Test":
            self.phase = 7
        elif name == "Сохранение":
            self.phase = 8
        else:
            self.phase = 1
        self.update(detail=detail)

    def update(self, n=0, *, detail=None):
        elapsed = int(time.monotonic() - self.started)
        phase_elapsed = int(time.monotonic() - self.phase_started)
        phase = self.plan.index(self.phase) + 1 if self.phase in self.plan else 1
        text = (
            f"{self.name} [{self.position}/{self.total}] · этап {phase}/{len(self.plan)}: {self.phase_name} · "
            f"конфигурация {elapsed // 60}:{elapsed % 60:02d}, этап {phase_elapsed // 60}:{phase_elapsed % 60:02d} · "
            f"{self.scope} · {detail or ''}"
        )
        self.progress.update(detail=text)

    def add_total(self, n):
        pass


def load_config(path):
    cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if (
        cfg["experiment_id"] != path.stem
        or not re.fullmatch(r"\d{3}_pretrained", path.stem)
        or cfg["data"]["version"] != "full_psg_v1"
        or cfg["evaluation"]["class_order"] != ["W", "N1", "N2", "N3", "REM"]
        or cfg["model"]["kind"]
        not in (
            "gate",
            "temporal_residual",
            "coverage_tiny",
            "wav2sleep",
            "refinement_ensemble",
        )
    ):
        raise ValueError("Invalid refinement config")
    return cfg


def implementation():
    paths = [
        *ROOT.glob("src/refinement*.*"),
        ROOT / "src/training/refinement.py",
        ROOT / "src/models/wav2sleep.py",
        ROOT / "environments/pretrained/uv.lock",
    ]
    return fingerprint(
        {
            str(p.relative_to(ROOT)): hashlib.sha256(
                p.read_bytes().replace(b"\r\n", b"\n")
            ).hexdigest()
            for p in paths
        }
    )


def selected_member(shared, label, namespace):
    ids = range(100, 105) if label == "BestGate" else range(106, 111)
    rows = []
    for number in ids:
        root = namespace / f"{number:03d}_pretrained"
        if (
            not (root / "metrics.json").exists()
            or not (root / "selection.json").exists()
        ):
            raise FileNotFoundError(
                f"{label}: first complete {root.name}; selection uses only train OOF"
            )
        selection = json.loads((root / "selection.json").read_text())
        rows.append(
            {
                "id": root.name,
                "oof_macro_f1": selection["oof"]["macro_f1"],
                "selection_sha256": sha256(root / "selection.json"),
            }
        )
    rows.sort(key=lambda r: (-r["oof_macro_f1"], r["id"]))
    return rows[0]["id"], rows


def ensemble_values(config, shared, namespace, split, folder):
    members, selection, ps = [], {}, []
    meta = shared.data.metadata(split)
    for member in config["model"]["members"]:
        if member == "B":
            ps.append((shared.experts(split) * np.array(PRIOR)[None, :, None]).sum(1))
            members.append("099_pretrained")
        elif member in ("019", "082", "085"):
            ps.append(shared.experts(split)[:, ("019", "082", "085").index(member)])
            members.append(member + "_pretrained")
        else:
            name = member
            if member.startswith("Best"):
                name, rows = selected_member(shared, member, namespace)
                selection[member] = {
                    "selected": name,
                    "candidates": rows,
                    "criterion": "train OOF macro-F1 only",
                }
            path = namespace / name / f"{split}_predictions.npz"
            if not path.exists():
                raise FileNotFoundError(f"First run {name}; missing {path.name}")
            info = json.loads(path.with_suffix(".json").read_text())
            if info["prediction_sha256"] != sha256(path):
                raise ValueError("Ensemble member checksum differs")
            ps.append(load_prediction(path, meta)["probabilities"])
            members.append(name)
    save_json(
        folder / "ensemble_selection.json",
        {
            "members": members,
            "weights": config["model"]["weights"],
            "selection": selection,
        },
    )
    p = sum(
        w * p.astype(np.float64)
        for w, p in zip(config["model"]["weights"], ps, strict=True)
    )
    p /= p.sum(1, keepdims=True)
    return {**meta, "probabilities": p.astype(np.float32)}


def wav_predict(model, shared, split, folder, contract):
    parts = []
    for i, path in enumerate(shared.data.paths(split)):
        meta = shared.data.record_metadata(path, split)
        target = folder / "records" / split / f"{path.stem}.npz"
        if target.with_suffix(".json").exists():
            info = json.loads(target.with_suffix(".json").read_text())
            if info["contract"] != contract or info["prediction_sha256"] != sha256(
                target
            ):
                raise ValueError("wav2sleep record resume differs")
            values = load_prediction(target, meta)
        else:
            raw = selected_signal(shared.data, path, split)
            p = np.empty((len(raw), 5), np.float32)
            for j, part in enumerate(contiguous_segments(meta["epoch_index"])):
                shared.progress.update(
                    detail=f"{split}: запись {i + 1}/{len(shared.data.paths(split))}, {path.stem}, сегмент {j + 1}, {part.stop - part.start} эпох; preprocessing и GPU inference"
                )
                p[part] = model.probabilities(raw[part])
            values = {**meta, "probabilities": p}
            save_prediction(
                target,
                values,
                {
                    "contract": contract,
                    "model": model.provenance,
                    "input": shared.data.verify(path),
                },
            )
        parts.append(values)
    return {k: np.concatenate([v[k] for v in parts]) for k in parts[0]}


def run_config(path, config, shared, namespace, environment):
    args, data = shared.args, shared.data
    folder = namespace / config["experiment_id"]
    folder.mkdir(parents=True, exist_ok=True)
    kind = config["model"]["kind"]
    # Register immutable source identities before saving the numerical contract.
    if kind != "wav2sleep":
        for split in ("train", "val", "test"):
            shared.experts(split)
    sources = dict(shared.store.used)
    source_hashes = (
        {
            s: hashlib.sha256(shared.experts(s).tobytes()).hexdigest()
            for s in ("train", "val", "test")
        }
        if kind != "wav2sleep"
        else {}
    )
    feature_sources = None
    if config["model"].get("features") == "physio":
        if args.feature_cache is None:
            raise FileNotFoundError(
                "102 requires the existing full106 feature cache in Input"
            )
        feature_sources = {
            p.name: sha256(p)
            for p in args.feature_cache.rglob("*")
            if p.is_file()
            and (p.name == "definition.json" or p.name.endswith("_epoch.npz"))
        }
    contract = fingerprint(
        {
            "config": config,
            "data": data.identity,
            "implementation": implementation(),
            "smoke": data.smoke,
            "smoke_epochs": data.smoke_epochs if data.smoke else None,
            # Keys are portable logical content hashes, not Kaggle mount paths.
            "inputs": source_hashes,
            "features": feature_sources,
            "assets": MANIFEST if kind == "wav2sleep" else None,
        }
    )
    provenance_path = folder / "provenance.json"
    if (
        provenance_path.exists()
        and json.loads(provenance_path.read_text())["contract"] != contract
    ):
        raise ValueError(
            "Existing experiment contract differs; preserve it, use a separate output root"
        )
    provenance = {
        "contract": contract,
        "config": config,
        "data": data.identity,
        "smoke": data.smoke,
        "environment": environment,
        "git": git_state(),
        "source_predictions": sources,
        "expert_alignment": getattr(shared.store, "alignment", {}),
        "feature_sources": feature_sources,
        "implementation": implementation(),
        "class_order": config["evaluation"]["class_order"],
    }
    if not provenance_path.exists():
        save_json(provenance_path, provenance)
    OmegaConf.save(OmegaConf.create(config), folder / "resolved.yaml")
    snapshot(folder)
    if (folder / "metrics.json").exists():
        for split in ("val", "test"):
            target = folder / f"{split}_predictions.npz"
            if json.loads(target.with_suffix(".json").read_text())[
                "prediction_sha256"
            ] != sha256(target):
                raise ValueError("Completed prediction checksum differs")
        metrics = json.loads((folder / "metrics.json").read_text())
        ensure_submission(
            folder,
            load_prediction(folder / "test_predictions.npz", data.metadata("test")),
            data,
        )
        if not data.smoke:
            journal(path, config, metrics, folder, args.output_root)
        return
    trained = (
        args.trained_root
        / ("smoke" if data.smoke else "full")
        / config["experiment_id"]
    )
    model, records, scale, seconds = None, None, None, None
    if kind in ("gate", "temporal_residual"):
        model, scale, seconds = residual_experiment(
            config, shared, folder, trained, contract
        )
    elif kind == "coverage_tiny":
        model, records, seconds = tiny_experiment(
            config, shared, folder, trained, contract
        )
    elif kind == "wav2sleep":
        model = Wav2SleepAdapter(args.checkpoint_cache, args.device)
    elapsed, validation = {}, None
    for split in ("val", "test"):
        shared.progress.scope = ""
        shared.progress.stage(
            "Validation" if split == "val" else "Test",
            "готовые кэши -> inference -> сохранение",
        )
        target = folder / f"{split}_predictions.npz"
        if target.with_suffix(".json").exists():
            info = json.loads(target.with_suffix(".json").read_text())
            if info["contract"] != contract or info["prediction_sha256"] != sha256(
                target
            ):
                raise ValueError("Partial split resume differs")
            values = load_prediction(target, data.metadata(split))
            elapsed[split] = info["seconds"]
        else:
            start = time.perf_counter()
            if kind in ("gate", "temporal_residual"):
                values = residual_predict(config, shared, split, model, scale, folder)
            elif kind == "coverage_tiny":
                values = tiny_predict(
                    model,
                    records,
                    shared,
                    split,
                    folder=folder / "refit_records" / split,
                    contract=contract,
                )
            elif kind == "wav2sleep":
                values = wav_predict(model, shared, split, folder, contract)
            else:
                values = ensemble_values(config, shared, namespace, split, folder)
            if args.device.startswith("cuda"):
                torch.cuda.synchronize(args.device)
            elapsed[split] = time.perf_counter() - start
            write_split(
                folder, split, values, data, {**provenance, "seconds": elapsed[split]}
            )
        if split == "val":
            validation = evaluate_values(values)
            save_json(folder / "validation_metrics.json", validation)
        else:
            ensure_submission(folder, values, data)
    shared.progress.stage(
        "Сохранение", "метрики, probabilities, submission и experiments.csv"
    )
    if (folder / "oof_predictions.npz").exists():
        save_json(
            folder / "oof_metrics.json",
            evaluate_values(
                load_prediction(folder / "oof_predictions.npz", data.metadata("train"))
            ),
        )
    metrics = {
        "smoke": data.smoke,
        "validation": validation,
        "training_seconds": seconds,
        "inference_seconds": sum(elapsed.values()),
        "split_seconds": elapsed,
        "val_epochs": len(data.metadata("val")["ids"]),
        "test_epochs": len(data.metadata("test")["ids"]),
        "peak_gpu_bytes": torch.cuda.max_memory_allocated(args.device)
        if args.device.startswith("cuda")
        else None,
        "contract": contract,
    }
    save_json(folder / "metrics.json", metrics)
    if not data.smoke:
        journal(path, config, metrics, folder, args.output_root)
        destination = (
            args.output_root / "submissions" / f"{config['experiment_id']}.csv"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(folder / "submission.csv", destination)
    (folder / "last_error.json").unlink(missing_ok=True)
    del model, records


def pack_input(target):
    """Small portable bundle of already available sources; never runs inference."""
    files = {}
    artefacts = ROOT / "artefacts"
    for name in ("019_pretrained", "082_pretrained", "085_pretrained"):
        folder = artefacts / name
        for filename in (
            "provenance.json",
            "resolved.yaml",
            "selection.json",
            "oof_predictions.npz",
            "oof_predictions.json",
            "val_predictions.npz",
            "val_predictions.json",
            "test_predictions.npz",
            "test_predictions.json",
        ):
            path = folder / filename
            if path.exists():
                files[f"results/{name}/{filename}"] = path
    for path in (artefacts / "predictions").glob("*/train.json"):
        info = json.loads(path.read_text())
        if info.get("recipe") == {"kind": "baseline019"}:
            with np.load(path.with_suffix(".npz"), allow_pickle=False) as stored:
                if len(stored["ids"]) < 300000:
                    continue
            for candidate in (path, path.with_suffix(".npz")):
                files[
                    "results/predictions/"
                    + candidate.parent.name
                    + "/"
                    + candidate.name
                ] = candidate
    for path in (artefacts / "cache/features").glob("*/definition.json"):
        definition = json.loads(path.read_text())
        if definition.get("version") == "linear106_v1" and not definition.get("smoke"):
            for candidate in path.parent.glob("*"):
                if candidate.is_file() and (
                    candidate.name == "definition.json"
                    or candidate.name.endswith("_epoch.npz")
                ):
                    files["features/" + candidate.name] = candidate
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    with ZipFile(target, "x", compression=ZIP_DEFLATED, compresslevel=1) as zipped:
        for name, path in files.items():
            zipped.write(path, name)
        zipped.writestr(
            "bundle.json",
            json.dumps(
                {
                    "files": list(files),
                    "purpose": "019 + 082/085 OOF and val/test, full106 features",
                    "missing": "embeddings and preprocessed signals were excluded from old exports; compute once unless separately saved",
                    "trained_expert_checkpoints_required": False,
                },
                indent=2,
            ),
        )
    print(
        f"Input bundle: {target}, {target.stat().st_size / 2**20:.1f} MiB, {len(files)} files"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", type=Path, nargs="*")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "artefacts/refinement"
    )
    parser.add_argument(
        "--cache-root", type=Path, default=ROOT / "artefacts/cache/refinement"
    )
    parser.add_argument(
        "--trained-root", type=Path, default=ROOT / "artefacts/refinement-trained"
    )
    parser.add_argument(
        "--checkpoint-cache",
        type=Path,
        default=ROOT / "artefacts/checkpoints/refinement",
    )
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--input-root", type=Path, action="append", default=[])
    parser.add_argument("--feature-cache", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 2))
    parser.add_argument("--allow-training", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-epochs", type=int, default=160)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--report-tag", choices=("gpu0", "gpu1"))
    parser.add_argument("--export", type=Path)
    parser.add_argument("--pack-input", type=Path)
    args = parser.parse_args()
    if args.pack_input:
        pack_input(args.pack_input)
        return
    if args.export:
        export_results(
            args.output_root,
            args.export,
            extra_roots=(
                "refinement_failures.json",
                "refinement_failures-gpu0.json",
                "refinement_failures-gpu1.json",
                "prepare_report.json",
                "submissions",
                "predictions",
            ),
        )
        return
    configs = [(p.resolve(), load_config(p)) for p in args.configs]
    if not args.data_root or (not configs and not args.prepare):
        parser.error("Provide --data-root and configs, or --prepare")
    if (
        args.prepare or any(c["training"]["enabled"] for _, c in configs)
    ) and not args.allow_training:
        parser.error("Training/nested preparation requires --allow-training")
    if args.threads < 1:
        parser.error("--threads must be positive")
    args.output_root.mkdir(parents=True, exist_ok=True)
    if not (args.output_root / "experiments.csv").exists():
        shutil.copy2(ROOT / "experiments.csv", args.output_root / "experiments.csv")
    torch.set_num_threads(args.threads)
    environment = hardware(args.device)
    data = RefinementData(args.data_root, args.smoke, args.smoke_epochs)
    namespace = args.output_root / "smoke" if args.smoke else args.output_root
    namespace.mkdir(parents=True, exist_ok=True)
    failures = []
    with Progress(
        f"Refinement {args.device}: конфигурации", len(configs) or 1, "конф."
    ) as progress:
        stage = StageProgress(progress)
        shared = SharedInputs(data, args, stage)
        if args.prepare:
            prepared_started = time.perf_counter()
            stage.config("Общая подготовка (12 inner expert fits один раз)", 1, 1)
            for split in ("train", "val", "test"):
                shared.experts(split)
                shared.embeddings(split)
            nested_experts(shared)
            save_json(
                namespace / "prepare_report.json",
                {
                    "smoke": data.smoke,
                    "data": data.identity,
                    "used": shared.store.used,
                    "nested_fits": 12,
                    "prepare_seconds": time.perf_counter() - prepared_started,
                    "cache": str(shared.cache),
                },
            )
            progress.update(1)
        for number, (path, cfg) in enumerate(configs, 1):
            stage.config(cfg["experiment_id"], number, len(configs))
            try:
                if args.device.startswith("cuda"):
                    torch.cuda.reset_peak_memory_stats(args.device)
                with file_lock(
                    namespace / cfg["experiment_id"] / ".run.lock", wait=False
                ):
                    run_config(path, cfg, shared, namespace, environment)
                print(
                    f"{cfg['experiment_id']}: готово ({'smoke' if args.smoke else 'val/test'})",
                    flush=True,
                )
            except Exception as error:
                failure = {
                    "experiment_id": cfg["experiment_id"],
                    "error": repr(error),
                    "traceback": traceback.format_exc(),
                }
                save_json(namespace / cfg["experiment_id"] / "last_error.json", failure)
                failures.append(failure)
                print(
                    f"{cfg['experiment_id']}: ошибка: {error}; независимые конфигурации продолжаются",
                    flush=True,
                )
            finally:
                shared.engine.release()
                gc.collect()
                progress.update(1)
    report_name = (
        f"refinement_failures-{args.report_tag}.json"
        if args.report_tag
        else "refinement_failures.json"
    )
    save_json(namespace / report_name, failures)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
