"""118–123: inference-only PhysioEx Tiny voting and fixed saved-probability blends."""

import argparse
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
import polars as pl
import torch
from omegaconf import OmegaConf

from src.models.pretrained_assets import sha256
from src.models.tiny_voting import (
    REFERENCE_REVISION,
    REFERENCE_SOURCE_SHA256,
    REFERENCE_URL,
    logits_voting,
)
from src.pretrained import (
    ROOT,
    export_results,
    git_state,
    hardware,
    save_json,
    save_npz,
    snapshot,
)
from src.pretrained_signals import contiguous_segments
from src.progress import Progress
from src.series_data import fingerprint, load_prediction, save_prediction
from src.series_results import ensure_submission, evaluate_values, journal, write_split
from src.storage import file_lock
from src.voting_data import CNNCache, VotingData, VotingInputs
from src.voting_reports import comparison

BASELINES = (
    "041_pretrained",
    "082_pretrained",
    "085_pretrained",
    "053_pretrained",
    "099_pretrained",
    "113_pretrained",
)


class VotingProgress:
    def __init__(self, bar):
        self.bar = bar
        self.name, self.position, self.total = "подготовка", 0, 0
        self.started = self.phase_started = time.monotonic()
        self.phase_name, self.phase_number, self.context = "подготовка", 1, ""

    def start(self, name, position, total):
        self.name, self.position, self.total = name, position, total
        self.phase_total = 1 if name.startswith("CNN ") else 4
        self.started = time.monotonic()
        self.stage("Подготовка", 1)

    def stage(self, name, number):
        self.phase_name, self.phase_number, self.context = name, number, ""
        self.phase_started = time.monotonic()
        self.detail("")

    def detail(self, message):
        elapsed = int(time.monotonic() - self.started)
        phase = int(time.monotonic() - self.phase_started)
        self.bar.update(
            detail=f"{self.name} [{self.position}/{self.total}] · этап {self.phase_number}/{getattr(self, 'phase_total', 4)} {self.phase_name} · "
            f"конфигурация {elapsed // 60}:{elapsed % 60:02d}, этап {phase // 60}:{phase % 60:02d} · {self.context} · {message}"
        )


def load_config(path):
    cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if (
        cfg["experiment_id"] != path.stem
        or not re.fullmatch(r"\d{3}_pretrained", path.stem)
        or cfg["data"]["version"] != "full_psg_v1"
        or cfg["training"] != {"enabled": False}
        or cfg["evaluation"]["class_order"] != ["W", "N1", "N2", "N3", "REM"]
        or cfg["model"]["kind"] not in ("tiny_voting", "saved_voting_ensemble")
    ):
        raise ValueError("Invalid inference-only voting configuration")
    if cfg["model"]["kind"] == "tiny_voting":
        if (
            cfg["model"]["source"] not in BASELINES[:3]
            or cfg["inference"]["length"] != 20
        ):
            raise ValueError("Voting must use a registered frozen source and L20")
    else:
        weights = cfg["model"]["weights"]
        if (
            len(weights) != len(cfg["model"]["members"])
            or any(w < 0 for w in weights)
            or not np.isclose(sum(weights), 1)
        ):
            raise ValueError("Invalid fixed mixture weights")
    return cfg


def implementation():
    paths = [
        *ROOT.glob("src/voting_*.py"),
        ROOT / "src/models/tiny_voting.py",
        *[
            ROOT / "src" / p
            for p in (
                "models/tinysleepnet.py",
                "models/pretrained_assets.py",
                "series_assets.py",
                "series_data.py",
                "series_results.py",
                "data.py",
                "evaluation.py",
                "pretrained_signals.py",
                "training/expansion.py",
            )
        ],
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


def checked_prediction(path, meta, contract=None):
    info = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if info["prediction_sha256"] != sha256(path) or (
        contract is not None and info["contract"] != contract
    ):
        raise ValueError(f"Saved prediction contract/SHA differs: {path}")
    return load_prediction(path, meta)


def model_split(config, model, source, data, cnn, split, folder, contract, progress):
    records, seconds = [], 0.0
    for number, path in enumerate(data.paths(split), 1):
        meta = data.record_metadata(path, split)
        target = folder / "records" / split / f"{path.stem}.npz"
        progress.context = (
            f"{split}: запись {number}/{len(data.paths(split))}, {path.stem}"
        )
        if target.with_suffix(".json").exists():
            info = json.loads(target.with_suffix(".json").read_text())
            if info["contract"] != contract or info["sha256"] != sha256(target):
                raise ValueError("Voting record resume differs")
            with np.load(target, allow_pickle=False) as saved:
                values = {k: saved[k] for k in saved.files}
            if not np.array_equal(values["ids"], meta["ids"]):
                raise ValueError("Voting record resume IDs differ")
            progress.detail("готовая запись переиспользована")
            seconds += info["seconds"]
        else:
            started = time.perf_counter()
            features = cnn.get(model, source, path, split)
            segments = contiguous_segments(meta["epoch_index"])
            parts = []
            for index, segment in enumerate(segments, 1):
                signal = torch.from_numpy(np.array(features[segment], copy=True)).to(
                    cnn.device
                )

                def detail(message):
                    progress.detail(f"сегмент {index}/{len(segments)}: {message}")

                parts.append(
                    logits_voting(
                        model.clf,
                        signal,
                        window_batch=config["inference"]["window_batch"],
                        detail=detail,
                    )
                )
                del signal
            values = {
                **meta,
                **{k: np.concatenate([p[k] for p in parts]) for k in parts[0]},
            }
            elapsed = time.perf_counter() - started
            save_npz(target, **values)
            save_json(
                target.with_suffix(".json"),
                {
                    "contract": contract,
                    "sha256": sha256(target),
                    "seconds": elapsed,
                    "input": data.hashes[path.relative_to(data.root).as_posix()],
                    "source": source,
                },
            )
            seconds += elapsed
        records.append(values)
    combined = {k: np.concatenate([r[k] for r in records]) for k in records[0]}
    expected = data.metadata(split)
    if not np.array_equal(combined["ids"], expected["ids"]):
        raise ValueError("Merged voting IDs differ from complete selected split")
    rolling = {**expected, "probabilities": combined.pop("rolling_probabilities")}
    aux = {
        "ids": expected["ids"],
        **{
            k: combined.pop(k) for k in ("vote_counts", "short_fallback", "mean_logits")
        },
    }
    return combined, rolling, aux, seconds


def blend_split(config, inputs, namespace, data, split):
    ps, covered = [], []
    meta = data.metadata(split)
    for member in config["model"]["members"]:
        if member == "019_pretrained":
            ps.append(inputs.baseline(member, split)["probabilities"])
        else:
            path = namespace / member / f"{split}_predictions.npz"
            ps.append(checked_prediction(path, meta)["probabilities"])
            aux_path = namespace / member / f"{split}_voting_counts.npz"
            info = json.loads(aux_path.with_suffix(".json").read_text())
            if sha256(aux_path) != info["sha256"]:
                raise ValueError("Ensemble coverage checksum differs")
            with np.load(aux_path, allow_pickle=False) as saved:
                if not np.array_equal(saved["ids"], meta["ids"]):
                    raise ValueError("Ensemble coverage IDs differ")
                covered.append(saved["short_fallback"])
    p = sum(
        w * v.astype(np.float64)
        for w, v in zip(config["model"]["weights"], ps, strict=True)
    )
    p /= p.sum(1, keepdims=True)
    fallback = np.logical_or.reduce(covered)
    return {**meta, "probabilities": p.astype(np.float32)}, {
        "ids": meta["ids"],
        "short_fallback": fallback,
    }


def compare_saved(config, inputs, namespace, data, folder, contract):
    meta = data.metadata("val")
    values = checked_prediction(folder / "val_predictions.npz", meta, contract)
    with np.load(folder / "val_voting_counts.npz", allow_pickle=False) as saved:
        fallback = saved["short_fallback"]
    rows = []
    for name in BASELINES:
        baseline = inputs.baseline(name, "val")
        result = comparison(values, baseline, fallback=fallback)
        result.update(
            reference_id=name,
            experiment_id=config["experiment_id"],
            evaluation_set="provided_val_only",
        )
        save_json(folder / "comparisons" / f"{name}.json", result)
        rows.append(
            {
                "baseline": name,
                "val_macro_f1": result["voting_metrics"]["macro_f1"],
                "baseline_macro_f1": result["baseline_metrics"]["macro_f1"],
                "delta_macro_f1": result["delta_macro_f1"],
                "errors_fixed": result["errors_fixed"],
                "errors_introduced": result["errors_introduced"],
            }
        )
    pl.DataFrame(rows).write_csv(folder / "comparisons.csv")
    if config["model"]["kind"] == "tiny_voting":
        paired = checked_prediction(
            folder / "val_rolling_predictions.npz", meta, contract
        )
        save_json(
            folder / "paired_same_inputs.json",
            comparison(values, paired, fallback=fallback),
        )


def run_config(path, config, args, data, inputs, namespace, progress, cnn, environment):
    folder = namespace / config["experiment_id"]
    folder.mkdir(parents=True, exist_ok=True)
    contract = fingerprint(
        {
            "config": config,
            "data": data.identity,
            "implementation": implementation(),
            "input_bundle": fingerprint(inputs.manifest),
            "smoke": data.smoke,
            "smoke_epochs": data.smoke_epochs if data.smoke else None,
        }
    )
    provenance_path = folder / "provenance.json"
    if (
        provenance_path.exists()
        and json.loads(provenance_path.read_text())["contract"] != contract
    ):
        raise ValueError(
            "Existing voting result differs; preserve it and use a separate output root"
        )
    provenance = {
        "contract": contract,
        "config": config,
        "data": data.identity,
        "smoke": data.smoke,
        "implementation": implementation(),
        "git": git_state(),
        "environment": environment,
        "class_order": config["evaluation"]["class_order"],
        "reference_revision": REFERENCE_REVISION,
        "reference_url": REFERENCE_URL,
        "reference_source_sha256": REFERENCE_SOURCE_SHA256,
        "input_bundle_sha256": fingerprint(inputs.manifest),
        "snapshot": "../source; one shared immutable code snapshot per output root",
    }
    if not provenance_path.exists():
        save_json(provenance_path, provenance)
    OmegaConf.save(OmegaConf.create(config), folder / "resolved.yaml")
    complete = (folder / "metrics.json").exists()
    if complete:
        metrics = json.loads((folder / "metrics.json").read_text())
        if metrics["contract"] != contract:
            raise ValueError("Completed metric contract differs")
        for split in ("val", "test"):
            values = checked_prediction(
                folder / f"{split}_predictions.npz", data.metadata(split), contract
            )
            aux_path = folder / f"{split}_voting_counts.npz"
            info = json.loads(aux_path.with_suffix(".json").read_text())
            if info["contract"] != contract or info["sha256"] != sha256(aux_path):
                raise ValueError("Completed coverage checksum differs")
            if config["model"]["kind"] == "tiny_voting":
                checked_prediction(
                    folder / f"{split}_rolling_predictions.npz",
                    data.metadata(split),
                    contract,
                )
            if split == "test":
                ensure_submission(folder, values, data)
        if not data.smoke:
            journal(path, config, metrics, folder, args.output_root)
        (folder / "last_error.json").unlink(missing_ok=True)
        progress.detail(
            "завершённые результаты проверены и переиспользованы без перезаписи"
        )
        return
    model = source = None
    if not complete and config["model"]["kind"] == "tiny_voting":
        progress.detail(
            "загрузка строго проверенного исходного/refit checkpoint; обучение выключено"
        )
        model, source = inputs.tiny(config["model"]["source"], args.device)
        save_json(folder / "checkpoint.json", source)
    times = {}
    for number, split in enumerate(("val", "test"), 2):
        progress.stage("Validation" if split == "val" else "Test", number)
        meta = data.metadata(split)
        target = folder / f"{split}_predictions.npz"
        aux_path = folder / f"{split}_voting_counts.npz"
        if target.with_suffix(".json").exists():
            values = checked_prediction(target, meta, contract)
            info = json.loads(aux_path.with_suffix(".json").read_text())
            if info["contract"] != contract or info["sha256"] != sha256(aux_path):
                raise ValueError("Voting coverage resume differs")
            times[split] = json.loads(target.with_suffix(".json").read_text())[
                "seconds"
            ]
            if config["model"]["kind"] == "tiny_voting":
                checked_prediction(
                    folder / f"{split}_rolling_predictions.npz", meta, contract
                )
        else:
            if config["model"]["kind"] == "tiny_voting":
                values, rolling, aux, elapsed = model_split(
                    config, model, source, data, cnn, split, folder, contract, progress
                )
                save_prediction(
                    folder / f"{split}_rolling_predictions.npz",
                    rolling,
                    {
                        **provenance,
                        "mode": "unchanged rolling recipe, same features and epoch IDs",
                    },
                )
                old = inputs.baseline(config["model"]["source"], split)
                max_diff = float(
                    np.max(np.abs(rolling["probabilities"] - old["probabilities"]))
                )
                agreement = {
                    "same_epoch_ids": bool(np.array_equal(rolling["ids"], old["ids"])),
                    "max_absolute_probability_difference": max_diff,
                    "argmax_changes": int(
                        (
                            rolling["probabilities"].argmax(1)
                            != old["probabilities"].argmax(1)
                        ).sum()
                    ),
                    "source": config["model"]["source"],
                    "mode": "smoke preprocessing/window endpoint differs from full source"
                    if data.smoke
                    else "same full source recipe",
                }
                # Smoke truncates filtering/window boundaries: use paired rolling
                # there. Full runs must reproduce the actual saved source recipe.
                if not data.smoke and not np.allclose(
                    rolling["probabilities"], old["probabilities"], atol=2e-5, rtol=2e-4
                ):
                    raise ValueError(
                        f"Rolling source reproduction differs (max {max_diff:.3g}); check checkpoint/preprocessing"
                    )
                save_json(folder / f"{split}_rolling_agreement.json", agreement)
            else:
                start = time.perf_counter()
                values, aux = blend_split(config, inputs, namespace, data, split)
                elapsed = time.perf_counter() - start
            save_npz(aux_path, **aux)
            save_json(
                aux_path.with_suffix(".json"),
                {
                    "contract": contract,
                    "sha256": sha256(aux_path),
                    "fallback_epochs": int(aux["short_fallback"].sum()),
                    "fallback_policy": "native short rolling, official vote count0; never crosses record/gap",
                },
            )
            write_split(folder, split, values, data, {**provenance, "seconds": elapsed})
            times[split] = elapsed
        if split == "test":
            ensure_submission(folder, values, data)
    progress.stage("Сохранение и сравнение", 4)
    compare_saved(config, inputs, namespace, data, folder, contract)
    validation = evaluate_values(
        checked_prediction(
            folder / "val_predictions.npz", data.metadata("val"), contract
        )
    )
    save_json(folder / "validation_metrics.json", validation)
    metrics = {
        "smoke": data.smoke,
        "validation": validation,
        "training_seconds": None,
        "inference_seconds": sum(times.values()),
        "split_seconds": times,
        "val_epochs": len(data.metadata("val")["ids"]),
        "test_epochs": len(data.metadata("test")["ids"]),
        "peak_gpu_bytes": torch.cuda.max_memory_allocated(args.device)
        if args.device.startswith("cuda")
        else None,
        "contract": contract,
        "timing_semantics": "joint voting+paired rolling and CNN cache acquisition; separate prepare stage recorded in prepare_report.json",
    }
    save_json(folder / "metrics.json", metrics)
    save_json(folder / "cnn_cache_used.json", cnn.used)
    if not data.smoke:
        journal(path, config, metrics, folder, args.output_root)
        destination = (
            args.output_root / "submissions" / f"{config['experiment_id']}.csv"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(folder / "submission.csv", destination)
    (folder / "last_error.json").unlink(missing_ok=True)
    del model


def pack_input(target):
    files = {}
    for name in ("019_pretrained", *BASELINES):
        for filename in (
            "provenance.json",
            "resolved.yaml",
            "selection.json",
            "val_predictions.npz",
            "val_predictions.json",
            "test_predictions.npz",
            "test_predictions.json",
        ):
            path = ROOT / "artefacts" / name / filename
            if path.exists():
                files[f"results/{name}/{filename}"] = path
    for name in ("082_pretrained", "085_pretrained"):
        for filename in (
            "best.pt",
            "training_summary.json",
            "trainable_parameters.json",
        ):
            files[f"trained/{name}/fit/{filename}"] = (
                ROOT / "artefacts/trained-checkpoints/full" / name / "fit" / filename
            )
    for filename in (
        "tinysleepnet.pt",
        "tinysleepnet-source.py",
        "tinysleepnet-config.json",
    ):
        candidates = [
            ROOT / "docs/expansion-assets" / filename,
            ROOT / "docs/series_assets" / filename,
        ]
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            raise FileNotFoundError(f"Missing official asset {filename}")
        from src.models.pretrained_assets import ASSETS, verify_asset

        verify_asset(path, ASSETS[filename][1])
        files["assets/" + filename] = path
    data = VotingData(ROOT / "data/raw")
    manifest = {
        "kind": "tiny_voting_inputs_v1",
        "data": data.identity,
        "files": {name: sha256(p) for name, p in files.items()},
        "reference_revision": REFERENCE_REVISION,
        "purpose": "frozen source checkpoints and saved rolling/reference predictions only; no training",
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(target, "x", compression=ZIP_DEFLATED, compresslevel=1) as zipped:
        for name, path in files.items():
            zipped.write(path, name)
        zipped.writestr("bundle.json", json.dumps(manifest, indent=2))
    print(
        f"Voting Input: {target}, {target.stat().st_size / 2**20:.1f} MiB, {len(files)} files",
        flush=True,
    )


def export_cache(root, read_roots, target):
    files = {}
    for directory in [*read_roots, root]:
        if not directory.exists():
            continue
        for path in directory.rglob("*"):
            relative = path.relative_to(directory)
            if (
                not path.is_file()
                or path.suffix not in (".json", ".npy")
                or "signals" in relative.parts
                or ".tmp" in path.name
            ):
                continue
            name = relative.as_posix()
            if name in files and sha256(files[name]) != sha256(path):
                raise ValueError(
                    f"Different cache payloads at the same portable path: {name}"
                )
            files[name] = path
    if not files:
        raise FileNotFoundError("No CNN cache to export")
    if target.resolve().is_relative_to(root.resolve()):
        raise ValueError("Cache ZIP must be outside cache root")
    with ZipFile(target, "x", compression=ZIP_DEFLATED, compresslevel=1) as zipped:
        for name, path in sorted(files.items()):
            zipped.write(path, name)
    print(
        f"Reusable CNN cache: {target} ({len(files)} files; signals excluded)",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="*", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-root", type=Path, default=ROOT / "artefacts/voting")
    parser.add_argument(
        "--cache-root", type=Path, default=ROOT / "artefacts/cache/voting"
    )
    parser.add_argument("--input-root", type=Path, action="append", default=[])
    parser.add_argument("--cnn-input", type=Path, action="append", default=[])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 2))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-epochs", type=int, default=160)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--report-tag", choices=("gpu0", "gpu1"))
    parser.add_argument("--export", type=Path)
    parser.add_argument("--export-cache", type=Path)
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
                "source",
                "code_snapshot.json",
                "submissions",
                "prepare_report.json",
                "voting_failures.json",
                "voting_failures-gpu0.json",
                "voting_failures-gpu1.json",
                "input_manifest.json",
            ),
        )
        return
    if args.export_cache:
        export_cache(args.cache_root, args.cnn_input, args.export_cache)
        return
    if (
        args.data_root is None
        or (not args.configs and not args.prepare)
        or args.threads < 1
    ):
        parser.error(
            "Provide --data-root, input bundle and configs or --prepare, threads>=1"
        )
    configs = [(p.resolve(), load_config(p)) for p in args.configs]
    args.output_root.mkdir(parents=True, exist_ok=True)
    with file_lock(args.output_root / ".init.lock", wait=True):
        if not (args.output_root / "experiments.csv").exists():
            shutil.copy2(ROOT / "experiments.csv", args.output_root / "experiments.csv")
        info = args.output_root / "code_snapshot.json"
        if (
            info.exists()
            and json.loads(info.read_text())["implementation"] != implementation()
        ):
            raise ValueError(
                "Output root belongs to different numerical code; preserve it and choose another root"
            )
        snapshot(args.output_root)
        if not info.exists():
            save_json(info, {"implementation": implementation(), "git": git_state()})
    torch.set_num_threads(args.threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    environment = hardware(args.device)
    data = VotingData(args.data_root, args.smoke, args.smoke_epochs)
    inputs = VotingInputs(args.input_root, data)
    save_json(args.output_root / "input_manifest.json", inputs.manifest)
    namespace = args.output_root / "smoke" if args.smoke else args.output_root
    namespace.mkdir(parents=True, exist_ok=True)
    args.cache_root.mkdir(parents=True, exist_ok=True)
    save_json(
        args.cache_root / "cache_manifest.json",
        {"kind": "tiny_voting_cnn_v1", "data": data.identity, "features": 2048},
    )
    failures = []
    with Progress(
        f"Voting {args.device}: конфигурации", len(configs) or 2, "конф."
    ) as bar:
        progress = VotingProgress(bar)
        cnn = CNNCache(data, args.cache_root, args.cnn_input, args.device, progress)
        if args.prepare:
            started = time.perf_counter()
            backbones = []
            for number, source in enumerate(("041_pretrained", "085_pretrained"), 1):
                progress.start(f"CNN {source}", number, 2)
                model, info = inputs.tiny(source, args.device)
                for split in ("val", "test"):
                    for index, path in enumerate(data.paths(split), 1):
                        progress.context = f"{split}: запись {index}/{len(data.paths(split))}, {path.stem}"
                        cnn.get(model, info, path, split)
                backbones.append(info["backbone_sha256"])
                del model
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()
                bar.update(1)
            save_json(
                namespace / "prepare_report.json",
                {
                    "smoke": data.smoke,
                    "seconds": time.perf_counter() - started,
                    "backbones": backbones,
                    "cnn_cache_used": cnn.used,
                    "training_performed": False,
                },
            )
        for index, (path, config) in enumerate(configs, 1):
            progress.start(config["experiment_id"], index, len(configs))
            try:
                if args.device.startswith("cuda"):
                    torch.cuda.reset_peak_memory_stats(args.device)
                with file_lock(
                    namespace / config["experiment_id"] / ".run.lock", wait=False
                ):
                    run_config(
                        path,
                        config,
                        args,
                        data,
                        inputs,
                        namespace,
                        progress,
                        cnn,
                        environment,
                    )
                print(
                    config["experiment_id"],
                    "готово (smoke)" if args.smoke else "готово (val/test)",
                    flush=True,
                )
            except Exception as error:
                failure = {
                    "experiment_id": config["experiment_id"],
                    "error": repr(error),
                    "traceback": traceback.format_exc(),
                }
                failures.append(failure)
                save_json(
                    namespace / config["experiment_id"] / "last_error.json", failure
                )
                print(
                    config["experiment_id"],
                    "ошибка:",
                    str(error),
                    "; независимые конфигурации продолжаются",
                    flush=True,
                )
            finally:
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()
                bar.update(1)
    filename = (
        f"voting_failures-{args.report_tag}.json"
        if args.report_tag
        else "voting_failures.json"
    )
    save_json(namespace / filename, failures)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
