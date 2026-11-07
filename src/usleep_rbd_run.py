"""156–159: official pretrained RBD/PD U-Sleep and three fixed113 mixtures."""

import argparse
import hashlib
import json
import os
import time
import traceback
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import torch
from omegaconf import OmegaConf

from src.context_run import diagnostics, export
from src.data import load_record
from src.journal import initialize_journal
from src.models.pretrained_assets import sha256, verify_asset
from src.models.usleep_rbd import (
    ARCH_SHA,
    CHECKPOINT_SHA,
    REVISION,
    load_model,
    predict,
    preprocess,
)
from src.pretrained import ROOT, save_json, save_npz, snapshot
from src.progress import Progress
from src.series_data import SeriesData, align_prediction
from src.series_results import evaluate_values, journal, write_split
from src.storage import file_lock

IDS = range(156, 160)


def name(number: int) -> str:
    return f"{number:03d}_pretrained"


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def implementation() -> str:
    paths = [ROOT / "src/usleep_rbd_run.py", ROOT / "src/models/usleep_rbd.py"]
    paths += [ROOT / "configs" / f"{name(n)}.yaml" for n in IDS]
    paths += [
        ROOT / "src" / p
        for p in (
            "series_data.py",
            "series_results.py",
            "context_run.py",
            "data.py",
            "pretrained_signals.py",
        )
    ]
    return hashlib.sha256(
        b"".join(p.read_bytes().replace(b"\r\n", b"\n") for p in paths)
    ).hexdigest()


class Runner:
    def __init__(self, args, progress):
        self.args, self.progress = args, progress
        self.data = SeriesData(args.data_root, args.smoke, args.smoke_epochs)
        self.root = args.output_root / "smoke" if args.smoke else args.output_root
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest = read(args.input_root / "bundle.json")
        if (
            self.manifest["kind"] != "usleep_rbd_inputs_v1"
            or self.manifest["data"] != self.data.identity
        ):
            raise ValueError("Need matching sleep-edf-usleep-rbd-inputs.zip")
        for relative, digest in self.manifest["files"].items():
            path = (args.input_root / relative).resolve()
            if not path.is_relative_to(args.input_root.resolve()):
                raise ValueError("Unsafe Input path")
            verify_asset(path, digest)
        identity = {
            "implementation": implementation(),
            "input": sha256(args.input_root / "bundle.json"),
            "data": self.data.identity,
            "smoke": args.smoke,
            "smoke_epochs": args.smoke_epochs if args.smoke else None,
            "device": args.device,
        }
        prior = self.root / "run_identity.json"
        if prior.exists() and read(prior) != identity:
            raise ValueError(
                "Preserve previous output; use a new OUTPUT_ROOT for this implementation/Input"
            )
        save_json(prior, identity)
        self.cfg = {
            n: OmegaConf.to_container(
                OmegaConf.load(ROOT / "configs" / f"{name(n)}.yaml"), resolve=True
            )
            for n in IDS
        }
        for n, config in self.cfg.items():
            if config["experiment_id"] != name(n) or config["training"]["enabled"]:
                raise ValueError("Frozen156–159 only")
            folder = self.root / name(n)
            folder.mkdir(exist_ok=True)
            OmegaConf.save(OmegaConf.create(config), folder / "resolved.yaml")
        initialize_journal(self.root / "experiments.csv", ROOT / "experiments.csv")
        self.seconds = {n: 0.0 for n in IDS}
        self.model = None
        self.failures = []

    def reference(self, split: str) -> dict:
        path = (
            self.args.input_root / "results/113_pretrained" / f"{split}_predictions.npz"
        )
        with np.load(path, allow_pickle=False) as z:
            values = {
                k: z[k] for k in ("ids", "probabilities", "labels") if k in z.files
            }
        meta = self.data.metadata(split)
        if self.data.smoke:
            subset = np.isin(values["ids"], meta["ids"])
            values = {k: v[subset] for k, v in values.items()}
        return align_prediction(values, meta)

    def standalone(self, split: str) -> dict:
        folder = self.root / name(156)
        records = []
        contexts = {"window": [], "full": []}
        for i, path in enumerate(self.data.paths(split), 1):
            meta = self.data.record_metadata(path, split)
            cached = folder / "records" / split / path.name
            cached.parent.mkdir(parents=True, exist_ok=True)
            if cached.exists():
                verify_asset(
                    cached, read(cached.with_suffix(".json"))["prediction_sha256"]
                )
                with np.load(cached, allow_pickle=False) as z:
                    row = align_prediction(
                        {k: z[k] for k in ("ids", "probabilities")}, meta
                    )
            else:
                begin = time.perf_counter()
                self.data.verify(path)
                record = load_record(path)
                self.progress.update(
                    detail=f"156 · {split} · запись{i}/{len(self.data.paths(split))} · этап1/3: preprocessing100→128Hz"
                )
                normalized, segments = preprocess(
                    record["x"][: len(meta["ids"])], meta["epoch_index"]
                )
                del record
                p, views, details = predict(
                    self.model,
                    normalized,
                    segments,
                    torch.device(self.args.device),
                    self.progress,
                    f"156 · {split} · запись{i}/{len(self.data.paths(split))} · этап2/3",
                )
                row = {**meta, "probabilities": p}
                save_npz(cached, **row, **views)
                elapsed = time.perf_counter() - begin
                save_json(
                    cached.with_suffix(".json"),
                    {
                        "seconds": elapsed,
                        "input_sha256": self.data.hashes[
                            path.relative_to(self.data.root).as_posix()
                        ],
                        "full_segments": details,
                        "prediction_sha256": sha256(cached),
                    },
                )
                del normalized, views
            self.seconds[156] += read(cached.with_suffix(".json"))["seconds"]
            records.append(row)
            if split == "val":
                with np.load(cached, allow_pickle=False) as z:
                    contexts["window"].append(z["window_pair_probabilities"].mean(1))
                    contexts["full"].append(z["full_pair_probabilities"].mean(1))
            self.progress.update(
                1,
                detail=f"156 · {split} · запись{i} сохранена · следующий этап: метрики/фиксированные смеси",
            )
        result = {k: np.concatenate([r[k] for r in records]) for k in records[0]}
        if split == "val":
            save_json(
                folder / "context_validation_metrics.json",
                {
                    mode: evaluate_values(
                        {**result, "probabilities": np.concatenate(parts)}
                    )
                    for mode, parts in contexts.items()
                },
            )
        return align_prediction(result, self.data.metadata(split))

    def save(self, n: int, split: str, values: dict, baseline: dict):
        folder = self.root / name(n)
        write_split(
            folder,
            split,
            values,
            self.data,
            {
                "implementation": implementation(),
                "checkpoint_sha256": CHECKPOINT_SHA,
                "data": self.data.identity,
                "source113_sha256": self.manifest["files"][
                    f"results/113_pretrained/{split}_predictions.npz"
                ],
            },
        )
        if split == "val":
            save_json(folder / "comparison.json", diagnostics(values, baseline))
        if split == "test":
            metrics = {
                "status": "complete",
                "validation": read(folder / "validation_metrics.json"),
                "training_seconds": 0,
                "inference_seconds": self.seconds[n],
                "test_epochs": len(values["ids"]),
                "smoke": self.data.smoke,
                "hardware": {
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "device": self.args.device,
                    "gpu": torch.cuda.get_device_name()
                    if self.args.device.startswith("cuda")
                    else None,
                },
                "checkpoint_sha256": CHECKPOINT_SHA,
                "class_order": ["W", "N1", "N2", "N3", "REM"],
            }
            save_json(folder / "metrics.json", metrics)
            if not self.data.smoke:
                journal(
                    ROOT / "configs" / f"{name(n)}.yaml",
                    self.cfg[n],
                    metrics,
                    folder,
                    self.root,
                )

    def run(self):
        # Validate BOTH saved113 ID sets before spending GPU time.
        references = {s: self.reference(s) for s in ("val", "test")}
        self.model = load_model(
            self.args.input_root / "assets", torch.device(self.args.device)
        )
        allowed = False
        for split in ("val", "test"):
            try:
                values = self.standalone(split)
                self.save(156, split, values, references[split])
                if split == "val":
                    score = evaluate_values(values)["macro_f1"]
                    allowed = (
                        self.data.smoke
                        or abs(score - 0.93657518) <= 0.02
                        or self.args.allow_score_mismatch
                    )
                    check = {
                        "validation_macro_f1": score,
                        "external_reported_macro_f1": 0.93657518,
                        "tolerance": 0.02,
                        "passed": allowed,
                        "smoke_is_not_quality_validation": self.data.smoke,
                        "explicit_override": self.args.allow_score_mismatch,
                        "checkpoint_sha256": CHECKPOINT_SHA,
                        "architecture_sha256": ARCH_SHA,
                        "class_mapping": "identity W,N1,N2,N3,REM",
                        "preprocessing": "F64 DC, gapwise resample/highpass; RECORD median/IQR; clip transformed20IQR",
                        "averaging": "FP32 mean contexts, then mean EEG pairs; no post-renormalization",
                        "on_mismatch": "Inspect checkpoint/preprocessing/class mapping/gaps/alignment and window/full diagnostics; blends withheld unless explicitly overridden",
                    }
                    save_json(self.root / "standalone_validation_check.json", check)
                    print(
                        f"156 standalone val macro-F1={score:.6f}; blend gate={allowed}",
                        flush=True,
                    )
                if self.args.group == "all" and allowed:
                    for n in range(157, 160):
                        begin = time.perf_counter()
                        w = self.cfg[n]["model"]["usleep_weight"]
                        p = w * values["probabilities"].astype(np.float64) + (
                            1 - w
                        ) * references[split]["probabilities"].astype(np.float64)
                        p = (p / p.sum(1, keepdims=True)).astype(np.float32)
                        self.seconds[n] += time.perf_counter() - begin
                        self.save(n, split, {**values, "probabilities": p}, values)
            except Exception as error:
                failure = {
                    "split": split,
                    "error": repr(error),
                    "traceback": traceback.format_exc(),
                }
                self.failures.append(failure)
                save_json(self.root / name(156) / f"{split}_last_error.json", failure)
                print(
                    f"156 {split} failed: {error}; completed files preserved",
                    flush=True,
                )
                if split == "val":
                    break
        rows = []
        for n in IDS:
            folder = self.root / name(n)
            if (folder / "metrics.json").exists():
                m = read(folder / "metrics.json")
                rows.append(
                    {
                        "id": name(n),
                        "val_macro_f1": m["validation"]["macro_f1"],
                        "test_ready": (
                            folder
                            / (
                                "submission_smoke.csv"
                                if self.data.smoke
                                else "submission.csv"
                            )
                        ).exists(),
                        "seconds": m["inference_seconds"],
                    }
                )
        report = {
            "smoke": self.data.smoke,
            "results": rows,
            "failures": self.failures,
            "blends_allowed": allowed,
            "selection_split": "validation only",
            "best_validation": max(rows, key=lambda r: r["val_macro_f1"])
            if rows
            else None,
            "all_requested_complete": len(rows)
            == (4 if self.args.group == "all" else 1),
        }
        save_json(self.root / "usleep_rbd_report.json", report)
        print(json.dumps(report, indent=2), flush=True)
        if self.failures:
            raise RuntimeError(
                "Series has errors; export preserved results and inspect usleep_rbd_report.json"
            )


def pack_input(target: Path):
    data_root = ROOT / "data/raw"
    if not (data_root / "MANIFEST.json").exists():
        data_root = ROOT / "archive/data/raw"
    artefacts = ROOT / "artefacts/113_pretrained"
    if not (artefacts / "val_predictions.npz").exists():
        artefacts = ROOT / "archive/artefacts/113_pretrained"
    files = {
        f"results/113_pretrained/{split}_predictions.npz": artefacts
        / f"{split}_predictions.npz"
        for split in ("val", "test")
    }
    files.update(
        {
            f"assets/{n}": ROOT / "docs/usleep_rbd_source" / n
            for n in ("Pretrained_Model.ckpt", "author_usleep.py")
        }
    )
    verify_asset(files["assets/Pretrained_Model.ckpt"], CHECKPOINT_SHA)
    verify_asset(files["assets/author_usleep.py"], ARCH_SHA)
    for relative in list(files):
        if relative.startswith("results/"):
            sidecar = files[relative].with_suffix(".json")
            if sidecar.exists() and read(sidecar).get("prediction_sha256") != sha256(
                files[relative]
            ):
                raise ValueError("Historical113 prediction SHA differs")
    manifest = {
        "kind": "usleep_rbd_inputs_v1",
        "data": SeriesData(data_root).identity,
        "files": {k: sha256(v) for k, v in files.items()},
        "author_revision": REVISION,
        "author_url": "https://github.com/jesperstroem/U-Sleep-for-RBD-PD",
        "author_terms": "non-commercial research only",
        "recipe_reference": "https://www.kaggle.com/code/baffledhedgehog/sleep-edf-u-sleep-v4?scriptVersionId=142584439",
        "purpose": "official author model and checkpoint plus saved113 val/test only; no competitor implementation, signal caches or new training",
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(target, "x", ZIP_DEFLATED, compresslevel=4) as z:
        for relative, path in files.items():
            z.write(path, relative)
        z.writestr("bundle.json", json.dumps(manifest, indent=2))
    print(f"Input: {target} ({target.stat().st_size / 2**20:.1f}MiB)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--input-root", type=Path)
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "artefacts/usleep-rbd"
    )
    parser.add_argument("--group", choices=("all", "standalone"), default="all")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=max(1, os.cpu_count() or 2))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-epochs", type=int, default=4)
    parser.add_argument("--allow-score-mismatch", action="store_true")
    parser.add_argument("--pack-input", type=Path)
    parser.add_argument("--export", type=Path)
    args = parser.parse_args()
    if args.pack_input:
        pack_input(args.pack_input)
        return
    if args.export:
        export(args.output_root, args.export)
        return
    if not args.data_root or not args.input_root or args.threads < 1:
        parser.error("Need data/input roots and positive threads")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(args.threads)
    torch.manual_seed(2070)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    with file_lock(args.output_root / ".usleep-rbd.lock", wait=True):
        snapshot(args.output_root)
        data = SeriesData(args.data_root, args.smoke, args.smoke_epochs)
        total = sum(len(data.paths(s)) for s in ("val", "test"))
        with Progress(
            "U-Sleep RBD/PD: val → test, затем готовые смеси", total, "запись"
        ) as progress:
            Runner(args, progress).run()


if __name__ == "__main__":
    main()
