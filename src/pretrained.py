"""Zero-shot validation CLI; independent of the Python 3.14 tabular environment."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import polars as pl
import torch
from omegaconf import OmegaConf

from src.data import CHANNELS, STAGES, align_labels, load_labels, load_record
from src.evaluation import (
    classification_metrics,
    subject_metrics,
    validate_probabilities,
)
from src.journal import upsert_result
from src.models.pretrained_assets import (
    ANYSLEEP_REVISION,
    ASSETS,
    SLEEPFM_REVISION,
    asset_manifest,
    sha256,
)
from src.pretrained_signals import contiguous_segments, epoch_blocks, preprocess_segment
from src.progress import Progress
from src.storage import atomic_path, file_lock

ROOT = Path(__file__).resolve().parents[1]


def save_json(path: Path, value: dict | list) -> None:
    with atomic_path(path) as temporary:
        temporary.write_text(
            json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8"
        )


def save_npz(path: Path, **arrays) -> None:
    with atomic_path(path) as temporary:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)


def source_digest() -> str:
    paths = [
        ROOT / "src" / name
        for name in (
            "pretrained.py",
            "pretrained_signals.py",
            "models/anysleep.py",
            "models/sleepfm.py",
            "models/pretrained_assets.py",
            "data.py",
            "evaluation.py",
            "storage.py",
            "journal.py",
            "progress.py",
        )
    ] + [ROOT / "environments/pretrained/uv.lock"]
    return hashlib.sha256(
        b"".join(p.read_bytes().replace(b"\r\n", b"\n") for p in paths)
    ).hexdigest()


def hardware(device: str) -> dict:
    result = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": device,
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "numpy",
                "scipy",
                "scikit-learn",
                "polars",
                "braindecode",
                "safetensors",
            )
        },
    }
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable in inference environment")
        torch.cuda.set_device(torch.device(device))
        props = torch.cuda.get_device_properties(device)
        result.update(
            gpu=props.name,
            gpu_bytes=props.total_memory,
            capability=list(torch.cuda.get_device_capability(device)),
        )
        # Test this exact build on the selected GPU before downloading weights.
        probe = torch.ones(8, device=device)
        if probe.sum().item() != 8:
            raise RuntimeError("CUDA probe failed")
    return result


def git_state() -> dict:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True
            )
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def make_adapter(config: dict, cache: Path, inputs: Path | None, device: str):
    family = config["model"]["family"]
    if family == "sleepfm":
        from src.models.sleepfm import SleepFMAdapter

        return SleepFMAdapter(cache, inputs, device)
    if family not in ("anysleep", "usleep"):
        raise ValueError(f"Unknown family: {family}")
    from src.models.anysleep import AnySleepAdapter, AnySleepEnsemble

    if isinstance(config["model"]["checkpoint"], list):
        return AnySleepEnsemble(config["model"]["checkpoint"], cache, inputs, device)
    return AnySleepAdapter(family, config["model"]["checkpoint"], cache, inputs, device)


def infer_segment(
    adapter, signal: np.ndarray, core: int, halo: int, progress=None
) -> np.ndarray:
    if signal.ndim != 2 or signal.shape[1] != 3 or len(signal) % 3840:
        raise ValueError(
            "Input must contain complete 30-second epochs and three channels"
        )
    n = len(signal) // 3840
    out = np.empty((n, 5), dtype=np.float32)
    for a, b, left, right in epoch_blocks(
        n, core, halo, right_alignment=getattr(adapter, "right_boundary_alignment", 1)
    ):
        prob = adapter.probabilities(signal[a * 3840 : b * 3840])
        validate_probabilities(prob, b - a)
        out[left:right] = prob[left - a : right - a]
        if progress is not None:
            progress.update(right - left)
    validate_probabilities(out, n)
    return out


def boundary_check(adapter, signal: np.ndarray, family: str) -> dict:
    """Measure context sensitivity, not an equality claim for global attention/LSTM."""
    n = min(len(signal) // 3840, 64 if family != "sleepfm" else 60)
    signal = signal[: n * 3840]
    whole = adapter.probabilities(signal)
    core, halo = (16, 16) if family != "sleepfm" else (20, 20)
    blocked = infer_segment(adapter, signal, core, halo)
    delta = np.abs(whole - blocked)
    return {
        "epochs": n,
        "probe_core": core,
        "probe_halo": halo,
        "mean_abs_probability_change": float(delta.mean()),
        "max_abs_probability_change": float(delta.max()),
        "argmax_disagreement": float(np.mean(whole.argmax(1) != blocked.argmax(1))),
        "note": "Diagnostic of global-context/position effects; chunked inference is approximate.",
    }


def validate_config(config: dict) -> None:
    family = config["model"]["family"]
    inference = config["inference"]
    core, halo = inference["core_epochs"], inference["halo_epochs"]
    alignment = 10 if family == "sleepfm" else 16
    if core <= 0 or halo < 0 or core % alignment or halo % alignment:
        raise ValueError(f"Block core/halo must align to {alignment} epochs")
    if family == "sleepfm" and (core + 2 * halo) * 6 > 8196:
        raise ValueError("SleepFM block exceeds position table")
    if config["data"]["version"] != "full_psg_v1":
        raise ValueError("Unsupported dataset version")
    if config["model"]["class_order"] != list(STAGES):
        raise ValueError("Class-order mismatch")
    if config["training"]["enabled"] or config["data"]["split"] != "val":
        raise ValueError("Only zero-shot validation is implemented")
    expected_checkpoint = {
        "anysleep": [f"anysleep-run{i}.pth" for i in (1, 2, 3)],
        "usleep": ["usleep-run1.pth"],
        "sleepfm": ["sleepfm-stager.safetensors"],
    }[family]
    checkpoints = config["model"]["checkpoint"]
    checkpoints = checkpoints if isinstance(checkpoints, list) else [checkpoints]
    if not checkpoints or any(c not in expected_checkpoint for c in checkpoints):
        raise ValueError("Unsupported checkpoint")
    hashes = [ASSETS[name][1] for name in checkpoints]
    expected_hash = (
        hashes if isinstance(config["model"]["checkpoint"], list) else hashes[0]
    )
    if config["model"]["checkpoint_sha256"] != expected_hash:
        raise ValueError("Checkpoint SHA-256 differs from pinned assets")
    expected_revision = SLEEPFM_REVISION if family == "sleepfm" else ANYSLEEP_REVISION
    if config["provenance"]["checkpoint_revision"] != expected_revision:
        raise ValueError("Checkpoint revision differs from pinned assets")
    p = config["preprocessing"]
    expected_normalization = (
        "channel_mean_std_ddof0_no_clip"
        if family == "sleepfm"
        else "clip_plusminus_20IQR_then_RobustScaler"
    )
    expected_aggregation = (
        "mean_six_patch_softmax"
        if family == "sleepfm"
        else "two_pair_hard_vote_tie_lowest_class"
        if family == "usleep"
        else "equal_checkpoint_softmax_mean"
        if len(checkpoints) > 1
        else "softmax"
    )
    if (
        p["sampling_rate"] != 128
        or p["scope"] != "per_contiguous_segment"
        or p["normalization"] != expected_normalization
        or p["units_conversion"] != "microvolts_to_volts"
        or inference["precision"] != "float32"
        or inference["aggregation"] != expected_aggregation
        or config["data"]["input_channels"] != list(CHANNELS)
        or config["data"]["input_units"] != "microvolts"
        or p["resampling"]
        != (
            "upstream_linear_interpolation"
            if family == "sleepfm"
            else "scipy_resample_poly_128_100"
        )
    ):
        raise ValueError("Config must describe the implemented author recipe exactly")


def check_cached(path: Path, contract: str, record: dict, input_hash: str) -> dict:
    with np.load(path, allow_pickle=False) as stored:
        result = {key: stored[key] for key in stored.files}
    if str(result["contract"]) != contract or str(result["input_sha256"]) != input_hash:
        raise ValueError(
            f"Saved record belongs to a different run/input: {path}; keep it and use a new output directory"
        )
    if not np.array_equal(result["ids"], record["ids"]) or not np.array_equal(
        result["epoch_index"], record["epoch_index"]
    ):
        raise ValueError(f"Cached epoch alignment mismatch: {path}")
    validate_probabilities(result["probabilities"], len(record["ids"]))
    return result


def snapshot(folder: Path) -> None:
    """Small self-contained source snapshot, no env, raw signals or model weights."""
    target = folder / "source"
    if target.exists():
        return
    target.mkdir()
    shutil.copytree(
        ROOT / "src", target / "src", ignore=shutil.ignore_patterns("__pycache__")
    )
    shutil.copytree(ROOT / "configs", target / "configs")
    shutil.copy2(ROOT / "experiments.csv", target / "experiments.csv")
    env = target / "environments/pretrained"
    env.mkdir(parents=True)
    for name in ("pyproject.toml", "uv.lock", ".python-version"):
        shutil.copy2(ROOT / "environments/pretrained" / name, env / name)


def update_pretrained_journal(
    config_path: Path, config: dict, metrics: dict, folder: Path
) -> None:
    row = {
        "experiment_id": config["experiment_id"],
        "config": f"configs/{config_path.name}",
        "model": config["model"]["name"],
        "summary": config["summary"],
        "seed": config["seed"],
        "evaluation_set": "zero_shot_provided_val",
        "macro_f1": metrics["validation"]["macro_f1"],
        "train_seconds": "",
        "infer_seconds": metrics["inference_seconds"],
        "status": "complete",
        "artefacts_path": str(folder),
    }
    row.update(
        {
            f"f1_{stage.lower()}": v
            for stage, v in zip(STAGES, metrics["validation"]["f1"], strict=True)
        }
    )
    upsert_result(ROOT / "experiments.csv", row)


def run_model(config_path: Path, args) -> None:
    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    validate_config(config)
    name = config["experiment_id"]
    if name != config_path.stem or not re.fullmatch(r"\d{3}_pretrained", name):
        raise ValueError("Experiment ID must match NNN_pretrained.yaml")
    family = config["model"]["family"]
    data = args.data_root.resolve()
    namespace = args.output_root / "smoke" if args.smoke else args.output_root
    folder = namespace / name
    folder.mkdir(parents=True, exist_ok=True)
    with (
        file_lock(args.output_root / ".pretrained-export.lock", wait=True, shared=True),
        file_lock(folder / "run.lock"),
    ):
        _run_model(config_path, config, args, data, folder, family)


def _run_model(
    config_path: Path, config: dict, args, data: Path, folder: Path, family: str
) -> None:
    env = hardware(args.device)
    torch.set_num_threads(max(1, args.threads))
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    labels = load_labels(data)
    labels_hash = sha256(data / "labels.csv")
    manifest = data / "MANIFEST.json"
    manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
    if manifest_data["dataset_revision"] != config["data"]["version"]:
        raise ValueError("Manifest dataset revision mismatch")
    expected_hashes = {item["path"]: item["sha256"] for item in manifest_data["files"]}
    if labels_hash != expected_hashes["labels.csv"]:
        raise ValueError("labels.csv differs from the original manifest")
    source = source_digest()
    smoke_epochs = getattr(args, "smoke_epochs", None)
    if smoke_epochs is None:
        smoke_epochs = (
            2 * config["inference"]["core_epochs"]
            + 2 * config["inference"]["halo_epochs"]
            + (15 if family in ("anysleep", "usleep") else 0)
        )
    if smoke_epochs <= 0:
        raise ValueError("--smoke-epochs must be positive")
    contract_data = {
        "config": config,
        "source": source,
        "labels": labels_hash,
        "smoke": args.smoke,
        "smoke_epochs": smoke_epochs if args.smoke else None,
    }
    contract = hashlib.sha256(
        json.dumps(contract_data, sort_keys=True).encode()
    ).hexdigest()
    state_path = folder / "contract.json"
    if state_path.exists():
        if json.loads(state_path.read_text())["sha256"] != contract:
            raise ValueError(
                "Different config/source/data in output directory; keep existing results and choose another --output-root"
            )
    else:
        save_json(state_path, {"sha256": contract, **contract_data})
    snapshot(folder)
    resolved = {
        **config,
        "runtime": {
            "data_root": str(data),
            "output_root": str(args.output_root.resolve()),
            "checkpoint_cache": str(args.checkpoint_cache.resolve()),
            "checkpoint_root": str(args.checkpoint_root.resolve())
            if args.checkpoint_root
            else None,
            "device": args.device,
            "smoke": args.smoke,
            "smoke_epochs": smoke_epochs if args.smoke else None,
        },
    }
    with atomic_path(folder / "resolved.yaml") as temp:
        OmegaConf.save(OmegaConf.create(resolved), temp)
    paths = sorted((data / "epochs/val").glob("*.npz"))
    if {p.stem for p in paths} != set(
        labels.filter(pl.col("split") == "val")["record_id"]
    ):
        raise ValueError("Validation records do not match labels")
    if args.smoke:
        paths = paths[:1]
    # A compact ZIP omits duplicated record caches for completed runs.
    # Validate that final result directly rather than recomputing all records.
    if (folder / "metrics.json").exists():
        previous = json.loads((folder / "provenance.json").read_text())
        if previous["predictions_sha256"] != sha256(folder / "val_predictions.npz"):
            raise ValueError("Completed predictions checksum mismatch")
        actual = [{"record_id": p.stem, "sha256": sha256(p)} for p in paths]
        expected = [
            {"record_id": r["record_id"], "sha256": r["sha256"]}
            for r in previous["records"]
        ]
        if actual != expected:
            raise ValueError("Input hashes changed since completed run")
        with np.load(folder / "val_predictions.npz", allow_pickle=False) as saved:
            expected_ids, expected_y = [], []
            for path in paths:
                h = load_record(path, signals=False)
                labels_aligned = align_labels(h, labels, path.stem, "val")
                size = min(smoke_epochs, len(h["ids"])) if args.smoke else len(h["ids"])
                expected_ids.append(h["ids"][:size])
                expected_y.append(labels_aligned[:size])
            if not np.array_equal(
                saved["ids"], np.concatenate(expected_ids)
            ) or not np.array_equal(saved["labels"], np.concatenate(expected_y)):
                raise ValueError("Completed prediction alignment mismatch")
            validate_probabilities(saved["probabilities"], len(saved["ids"]))
        metrics = json.loads((folder / "metrics.json").read_text())
        if not args.smoke:
            update_pretrained_journal(config_path, config, metrics, folder)
        print(
            f"{config['experiment_id']}: завершённый результат проверен, повторный inference пропущен",
            flush=True,
        )
        return
    headers = [load_record(p, signals=False) for p in paths]
    if args.smoke:
        headers = [
            {
                k: v[:smoke_epochs] if k in ("ids", "epoch_index") else v
                for k, v in h.items()
            }
            for h in headers
        ]
    count = sum(len(h["ids"]) for h in headers)
    records = folder / "records"
    records.mkdir(exist_ok=True)
    results, provenance_records = [], []
    adapter = None
    with Progress(
        f"{config['experiment_id']}: zero-shot val", count, "эпох"
    ) as progress:
        for path, header in zip(paths, headers, strict=True):
            progress.update(detail=f"{path.stem}: проверка / чтение")
            raw_hash = sha256(path)
            if raw_hash != expected_hashes[f"epochs/val/{path.name}"]:
                raise ValueError(f"NPZ differs from manifest: {path.name}")
            cached = records / f"{path.stem}.npz"
            if cached.exists():
                result = check_cached(cached, contract, header, raw_hash)
                progress.update(
                    len(header["ids"]), detail=f"{path.stem}: восстановлено"
                )
            else:
                if adapter is None:
                    progress.update(detail="Загрузка проверенных весов")
                    adapter = make_adapter(
                        config, args.checkpoint_cache, args.checkpoint_root, args.device
                    )
                    save_json(folder / "assets.json", asset_manifest(adapter.assets))
                record = load_record(path)
                y = align_labels(record, labels, path.stem, "val")
                if args.smoke:
                    record = {
                        k: v[:smoke_epochs] if k in ("ids", "epoch_index", "x") else v
                        for k, v in record.items()
                    }
                    y = y[:smoke_epochs]
                if args.device.startswith("cuda"):
                    torch.cuda.reset_peak_memory_stats(args.device)
                    torch.cuda.synchronize(args.device)
                start = time.perf_counter()
                probabilities = np.empty((len(record["ids"]), 5), dtype=np.float32)
                for segment in contiguous_segments(record["epoch_index"]):
                    progress.update(
                        detail=f"{path.stem}: preprocessing {segment.start}:{segment.stop}"
                    )
                    signal = preprocess_segment(record["x"][segment], family)
                    if args.smoke and not (folder / "boundary_check.json").exists():
                        save_json(
                            folder / "boundary_check.json",
                            boundary_check(adapter, signal, family),
                        )
                    progress.update(
                        detail=f"{path.stem}: inference {segment.start}:{segment.stop}"
                    )
                    probabilities[segment] = infer_segment(
                        adapter,
                        signal,
                        config["inference"]["core_epochs"],
                        config["inference"]["halo_epochs"],
                        progress,
                    )
                    del signal
                if args.device.startswith("cuda"):
                    torch.cuda.synchronize(args.device)
                seconds = time.perf_counter() - start
                peak = (
                    torch.cuda.max_memory_allocated(args.device)
                    if args.device.startswith("cuda")
                    else 0
                )
                validate_probabilities(probabilities, len(record["ids"]))
                result = {
                    "ids": record["ids"],
                    "epoch_index": record["epoch_index"],
                    "probabilities": probabilities,
                    "labels": y,
                    "contract": np.array(contract),
                    "input_sha256": np.array(raw_hash),
                    "inference_seconds": np.array(seconds),
                    "peak_gpu_bytes": np.array(peak),
                    "environment_json": np.array(json.dumps(env)),
                }
                save_npz(cached, **result)
                del record
            # Recheck labels on resume, including full alignment/record metadata.
            full_header = load_record(path, signals=False)
            y = align_labels(full_header, labels, path.stem, "val")[
                : len(result["ids"])
            ]
            if not np.array_equal(result["labels"], y):
                raise ValueError("Saved labels mismatch")
            results.append(result)
            provenance_records.append(
                {
                    "record_id": path.stem,
                    "sha256": raw_hash,
                    "epochs": len(result["ids"]),
                    "environment": json.loads(str(result["environment_json"])),
                }
            )
    ids = np.concatenate([r["ids"] for r in results])
    y = np.concatenate([r["labels"] for r in results])
    prob = np.concatenate([r["probabilities"] for r in results])
    if len(set(ids.tolist())) != len(ids):
        raise ValueError("Duplicate output IDs")
    lookup = pl.DataFrame({"id": ids}).join(
        labels.select("id", "subject_id"),
        on="id",
        how="left",
        maintain_order="left",
        validate="1:1",
    )
    if lookup["subject_id"].null_count():
        raise ValueError("Missing subject IDs")
    subjects = lookup["subject_id"].to_numpy()
    metrics = {
        "evaluation_protocol": "zero_shot_provided_val",
        "smoke": args.smoke,
        "validation": classification_metrics(y, prob),
        "epochs": len(ids),
        "inference_seconds": sum(float(r["inference_seconds"]) for r in results),
        "peak_gpu_bytes": max(int(r["peak_gpu_bytes"]) for r in results),
        "timing_scope": "per-record preprocessing + inference, excludes downloads/model load; smoke includes boundary probes",
    }
    save_npz(
        folder / "val_predictions.npz",
        ids=ids,
        labels=y,
        probabilities=prob,
        subject_id=subjects,
    )
    save_json(folder / "subjects.json", subject_metrics(y, prob, subjects))
    save_json(
        folder / "provenance.json",
        {
            "git": git_state(),
            "environment": env,
            "source_sha256": source,
            "labels_sha256": labels_hash,
            "manifest_sha256": sha256(manifest) if manifest.exists() else None,
            "records": provenance_records,
            "class_order": list(STAGES),
            "checkpoint": json.loads((folder / "assets.json").read_text()),
            "predictions_sha256": sha256(folder / "val_predictions.npz"),
            "external_sources": config["provenance"],
            "evaluation_limit": config["evaluation"]["limitation"],
        },
    )
    save_json(folder / "metrics.json", metrics)
    if not args.smoke:
        update_pretrained_journal(config_path, config, metrics, folder)
    print(
        f"{config['experiment_id']}: {'SMOKE, не эксперимент качества' if args.smoke else 'val'} macro-F1={metrics['validation']['macro_f1']:.5f}; {metrics['inference_seconds']:.1f} s",
        flush=True,
    )
    if adapter is not None:
        del adapter
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def export_results(root: Path, archive: Path) -> None:
    """Complete runs: final predictions only; incomplete runs retain record resume files."""
    root, archive = root.resolve(), archive.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    if archive.is_relative_to(root) or archive.exists():
        raise ValueError("ZIP must be new and outside output root")
    with file_lock(root / ".pretrained-export.lock", wait=True):
        _export_results(root, archive)


def _export_results(root: Path, archive: Path) -> None:
    with atomic_path(archive) as temp:
        with ZipFile(temp, "w", compression=ZIP_DEFLATED, compresslevel=6) as out:
            for path in sorted(root.rglob("*")):
                if path.is_symlink():
                    raise ValueError("Unexpected symlink in results")
                if (
                    not path.is_file()
                    or (
                        path.suffix in (".lock", ".tmp", ".pyc")
                        and path.name != "uv.lock"
                    )
                    or ".tmp." in path.name
                ):
                    continue
                relative = path.relative_to(root)
                parts = relative.parts
                experiment = (
                    parts[1] if parts[0] == "smoke" and len(parts) > 1 else parts[0]
                )
                if not re.fullmatch(
                    r"\d{3}_pretrained", experiment
                ) and relative.as_posix() not in (
                    "experiments.csv",
                    "failures.json",
                    "smoke_failures.json",
                    "notebook_failures.json",
                ):
                    continue
                if (
                    "records" in relative.parts
                    and (path.parent.parent / "metrics.json").exists()
                ):
                    continue
                if (
                    path.suffix
                    not in (
                        ".json",
                        ".yaml",
                        ".toml",
                        ".lock",
                        ".csv",
                        ".npz",
                        ".py",
                        ".txt",
                    )
                    and path.name != ".python-version"
                ):
                    continue
                out.write(path, relative.as_posix())


def main() -> None:
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
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--threads", type=int, default=max(1, (os.cpu_count() or 2) - 1)
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--smoke-epochs",
        type=int,
        help="Optional smaller local smoke; default exercises a full-size interior block",
    )
    parser.add_argument("--export", type=Path)
    args = parser.parse_args()
    if args.export:
        export_results(args.output_root, args.export)
        return
    if not args.configs or args.data_root is None:
        parser.error("Provide configs and --data-root")
    failures = []
    for config in args.configs:
        try:
            run_model(config.resolve(), args)
        except Exception as error:
            failure = {
                "config": str(config),
                "error": repr(error),
                "traceback": traceback.format_exc(),
                "time_utc": datetime.now(UTC).isoformat(),
            }
            failures.append(failure)
            namespace = args.output_root / "smoke" if args.smoke else args.output_root
            if re.fullmatch(r"\d{3}_pretrained", config.stem):
                save_json(namespace / config.stem / "last_error.json", failure)
            print(
                f"Ошибка {config.name}: {error}. Остальные модели продолжат работу.",
                flush=True,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    args.output_root.mkdir(parents=True, exist_ok=True)
    save_json(
        args.output_root / ("smoke_failures.json" if args.smoke else "failures.json"),
        failures,
    )
    shutil.copy2(ROOT / "experiments.csv", args.output_root / "experiments.csv")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
