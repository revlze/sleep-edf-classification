"""147–155: AnySleep context ablation, saved B and explicitly diagnostic113 blends."""

import argparse
import gc
import hashlib
import json
import os
import shutil
import time
import traceback
from functools import cache
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import torch
from omegaconf import OmegaConf

from src.context_inference import ContextOOM, predict_segment
from src.data import load_record
from src.evaluation import classification_metrics
from src.journal import initialize_journal
from src.models.anysleep import AnySleepAdapter
from src.models.pretrained_assets import ASSETS, asset_manifest, sha256, verify_asset
from src.pretrained import ROOT, git_state, hardware, save_json, snapshot
from src.pretrained_signals import contiguous_segments, preprocess_segment
from src.progress import Progress
from src.series_data import (
    align_prediction,
    fingerprint,
    read_legacy_019,
    save_prediction,
)
from src.series_results import ensure_submission, evaluate_values, journal, write_split
from src.storage import file_lock
from src.voting_reports import comparison, transition_distances

REFERENCES = (19, 82, 85, 99, 102, 106, 113)
ASSET_NAMES = ["anysleep_no_hydra.py", *[f"anysleep-run{i}.pth" for i in (1, 2, 3)]]


def name(number: int) -> str:
    return f"{number:03d}_pretrained"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def configs() -> dict[int, dict]:
    result = {}
    for number in range(147, 156):
        cfg = OmegaConf.to_container(
            OmegaConf.load(ROOT / "configs" / f"{name(number)}.yaml"), resolve=True
        )
        if cfg["experiment_id"] != name(number) or cfg["training"]["enabled"]:
            raise ValueError("Only frozen context configurations147–155 are allowed")
        result[number] = cfg
    return result


def implementation() -> str:
    files = [
        "src/context_run.py",
        "src/context_inference.py",
        "src/pretrained_signals.py",
        "src/models/anysleep.py",
        "src/models/pretrained_assets.py",
        "src/series_data.py",
        "src/series_results.py",
        "src/voting_reports.py",
        "src/evaluation.py",
        "environments/pretrained/uv.lock",
    ]
    return hashlib.sha256(
        b"".join((ROOT / p).read_bytes().replace(b"\r\n", b"\n") for p in files)
    ).hexdigest()


class References:
    def __init__(self, root: Path, data):
        self.root, self.data = root, data
        self.manifest = read(root / "bundle.json")
        if (
            self.manifest["kind"] != "anysleep_context_inputs_v1"
            or self.manifest["data"] != data.identity
        ):
            raise ValueError("Need matching sleep-edf-anysleep-context-inputs bundle")
        self.used = {}

    def path(self, relative: str) -> Path:
        p = self.root / relative
        expected = self.manifest["files"].get(relative)
        if not expected or not p.is_file():
            raise FileNotFoundError(f"Missing context Input: {relative}")
        if relative not in self.used:
            if sha256(p) != expected:
                raise ValueError(f"Input checksum differs: {relative}")
            self.used[relative] = expected
        return p

    @cache
    def prediction(self, number: int, split: str) -> dict:
        meta = self.data.metadata(split)
        p = self.path(f"results/{name(number)}/{split}_predictions.npz")
        if number == 19 and split == "val":
            for f in ("provenance.json", "resolved.yaml"):
                self.path(f"results/{name(number)}/{f}")
            original = read_legacy_019([self.root / "results"], self.data, split)
            if original is None:
                raise ValueError("Audited original019 validation is required")
            return original[0]
        with np.load(p, allow_pickle=False) as z:
            # Rebuild metadata; historical object subject IDs are never deserialized.
            keys = ("ids", "probabilities") + (("labels",) if split != "test" else ())
            values = {k: z[k] for k in keys if k in z.files}
        if self.data.smoke:
            subset = np.isin(values["ids"], meta["ids"])
            values = {k: v[subset] for k, v in values.items()}
        return align_prediction(values, meta)


def diagnostics(values: dict, baseline: dict) -> dict:
    result = comparison(values, baseline)

    # Existing paired reports name the new model 'voting'; do not mislabel this series.
    def rename(obj):
        if isinstance(obj, dict):
            return {k.replace("voting", "candidate"): rename(v) for k, v in obj.items()}
        return [rename(v) for v in obj] if isinstance(obj, list) else obj

    result = rename(result)
    d = transition_distances(values)
    masks = {
        "near_any_transition_d_le_1": (d >= 0) & (d <= 1),
        "near_any_transition_d_le_2": (d >= 0) & (d <= 2),
        "stable_d_ge_5_or_no_transition": (d < 0) | (d >= 5),
    }
    for pair in ((1, 2), (2, 3)):
        distance = transition_distances(values, pair)
        masks[f"near_{pair[0]}_{pair[1]}_transition_d_le_1"] = (distance >= 0) & (
            distance <= 1
        )
    for key, mask in masks.items():
        result["transition_slices"][key].update(
            {
                "baseline_metrics": classification_metrics(
                    values["labels"][mask], baseline["probabilities"][mask]
                )
                if mask.any()
                else None,
                "candidate_metrics": classification_metrics(
                    values["labels"][mask], values["probabilities"][mask]
                )
                if mask.any()
                else None,
            }
        )
    p, q = values["probabilities"], baseline["probabilities"]
    diff = np.abs(p - q)
    result["probability_changes"] = {
        "mean_abs": float(diff.mean()),
        "max_abs": float(diff.max()),
        "mean_row_l1": float(diff.sum(1).mean()),
        "row_l1_quantiles": np.quantile(diff.sum(1), [0.5, 0.9, 0.99, 1]).tolist(),
        "mean_JS_divergence": float(
            (
                (p * np.log(np.maximum(p, 1e-12) / np.maximum((p + q) / 2, 1e-12))).sum(
                    1
                )
                + (
                    q * np.log(np.maximum(q, 1e-12) / np.maximum((p + q) / 2, 1e-12))
                ).sum(1)
            ).mean()
            / 2
        ),
    }
    return result


class Runner:
    def __init__(self, args, progress):
        from src.series_data import SeriesData

        self.args, self.progress, self.configs = args, progress, configs()
        self.data = SeriesData(args.data_root, args.smoke, args.smoke_epochs)
        self.refs = References(args.input_root, self.data)
        self.namespace = args.output_root / "smoke" if args.smoke else args.output_root
        self.namespace.mkdir(parents=True, exist_ok=True)
        self.code = implementation()
        self.models = None
        self.environment = hardware(args.device)
        self.failures = []
        for n, cfg in self.configs.items():
            folder = self.folder(n)
            folder.mkdir(parents=True, exist_ok=True)
            provenance = {
                "config": cfg,
                "data": self.data.identity,
                "smoke": args.smoke,
                "smoke_epochs": args.smoke_epochs if args.smoke else None,
                "implementation": self.code,
                "assets": asset_manifest(ASSET_NAMES),
                "input_manifest_sha256": sha256(args.input_root / "bundle.json"),
            }
            contract = fingerprint(provenance)
            path = folder / "provenance.json"
            if path.exists() and read(path)["contract"] != contract:
                raise ValueError(
                    f"{name(n)} resume contract differs; preserve old output and choose another root"
                )
            if not path.exists():
                save_json(
                    path,
                    {
                        **provenance,
                        "contract": contract,
                        "environment": self.environment,
                        "git": git_state(),
                    },
                )
            OmegaConf.save(OmegaConf.create(cfg), folder / "resolved.yaml")
        initialize_journal(
            args.output_root / "experiments.csv", ROOT / "experiments.csv"
        )

    def folder(self, number):
        return self.namespace / name(number)

    def contract(self, number):
        return read(self.folder(number) / "provenance.json")["contract"]

    def stage(self, description, total):
        with self.progress._lock:
            self.progress.description = description
            self.progress.n = 0
            self.progress.total = total
            self.progress.started = time.monotonic()

    def cached(self, number, split, meta, *, record=None):
        folder = self.folder(number)
        p = folder / (
            f"records/{split}/{record}.npz" if record else f"{split}_predictions.npz"
        )
        if not p.with_suffix(".json").exists():
            return None
        info = read(p.with_suffix(".json"))
        if info["contract"] != self.contract(number) or info[
            "prediction_sha256"
        ] != sha256(p):
            raise ValueError(f"Corrupt/incompatible context output: {p}")
        with np.load(p, allow_pickle=False) as z:
            values = {k: z[k] for k in z.files}
        return align_prediction(values, meta), info

    def failure(self, number, split, exc, record_id, segment_index):
        status = "skipped_oom" if isinstance(exc, ContextOOM) else "failed"
        row = {
            "experiment_id": name(number),
            "split": split,
            "record_id": record_id,
            "segment_index": segment_index,
            "status": status,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "no_temporal_fallback": True,
        }
        save_json(self.folder(number) / f"{split}_status.json", row)
        for dependent in (number + 3, number + 6):
            save_json(
                self.folder(dependent) / f"{split}_status.json",
                {
                    "status": "skipped_dependency",
                    "source": name(number),
                    "error": f"{name(number)} {split}: {status}; no replacement probabilities",
                },
            )
        self.failures.append(row)
        print(
            f"{name(number)} {split}: {status}: {exc}; остальные варианты продолжаются",
            flush=True,
        )
        gc.collect()
        if self.args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    def adapters(self):
        if self.models is None:
            for asset in ASSET_NAMES:
                self.refs.path("assets/" + asset)
            self.models = [
                AnySleepAdapter(
                    "anysleep",
                    f"anysleep-run{i}.pth",
                    self.args.output_root / "unused-assets",
                    self.args.input_root / "assets",
                    self.args.device,
                ).model
                for i in (1, 2, 3)
            ]
        return self.models

    def infer(self, split: str, numbers: list[int]) -> list[int]:
        meta = self.data.metadata(split)
        active, parts, times, done = [], {}, {}, []
        for n in numbers:
            old = self.cached(n, split, meta)
            if old is not None:
                self.finish(n, split, old[0], old[1]["seconds"])
                done.append(n)
            else:
                active.append(n)
                parts[n] = []
                times[n] = 0.0
        paths = self.data.paths(split)
        self.stage(
            f"AnySleep · {'SMOKE' if self.data.smoke else 'FULL'} · {split.upper()}",
            len(paths) * len(active),
        )
        for record_number, path in enumerate(paths, 1):
            record = self.data.record_metadata(path, split)
            pending = []
            for n in active:
                old = self.cached(n, split, record, record=path.stem)
                if old is not None:
                    parts[n].append(old[0])
                    times[n] += old[1]["seconds"]
                    self.progress.update(
                        1,
                        detail=f"{name(n)} · запись {record_number}/{len(paths)} · готовый record cache",
                    )
                else:
                    pending.append(n)
            if not pending:
                continue
            if self.models is None:
                self.progress.update(
                    detail="Загрузка трёх официальных AnySleep checkpoints, без обучения"
                )
                started = time.perf_counter()
                self.adapters()
                save_json(
                    self.namespace / "model_setup.json",
                    {
                        "seconds": time.perf_counter() - started,
                        "assets": asset_manifest(ASSET_NAMES),
                    },
                )
            self.progress.update(
                detail=f"этап 1/3: NPZ и preprocessing · {split}, запись {record_number}/{len(paths)} {path.stem}"
            )
            self.data.verify(path)
            raw = load_record(path)["x"][: len(record["ids"])]
            segments = contiguous_segments(record["epoch_index"])
            buffers = {n: [] for n in pending}
            elapsed = {n: 0.0 for n in pending}
            details = {n: [] for n in pending}
            prep_seconds = 0.0
            for si, segment in enumerate(segments, 1):
                if not pending:
                    break
                started = time.perf_counter()
                signal = preprocess_segment(raw[segment], "anysleep")
                device_signal = torch.from_numpy(signal).to(self.args.device)
                prep_seconds += time.perf_counter() - started
                for n in pending.copy():
                    cfg = self.configs[n]
                    prefix = f"{name(n)} [{n - 146}/3] · этап 2/3: forward · запись {record_number}/{len(paths)}, сегмент {si}/{len(segments)} · "
                    started = time.perf_counter()
                    try:
                        if self.args.device.startswith("cuda"):
                            torch.cuda.reset_peak_memory_stats(self.args.device)
                        p, info = predict_segment(
                            device_signal,
                            self.adapters(),
                            cfg["inference"],
                            self.args.max_batch,
                            lambda text: self.progress.update(detail=prefix + text),
                        )
                        if self.args.device.startswith("cuda"):
                            torch.cuda.synchronize(self.args.device)
                            info["peak_gpu_allocated_bytes"] = (
                                torch.cuda.max_memory_allocated(self.args.device)
                            )
                        elapsed[n] += time.perf_counter() - started
                        details[n].append(info)
                        buffers[n].append(p)
                    except Exception as exc:
                        self.failure(n, split, exc, path.stem, si)
                        pending.remove(n)
                        active.remove(n)
                del device_signal, signal
            del raw
            prep_report = (
                self.namespace / f"shared_preprocessing/{split}/{path.stem}.json"
            )
            prep_history = read(prep_report) if prep_report.exists() else []
            prep_history.append(
                {
                    "seconds": prep_seconds,
                    "timestamp_unix": time.time(),
                    "epochs": len(record["ids"]),
                    "modes": [name(n) for n in buffers],
                    "semantics": "segment resampling/scaling and CPU->device copy shared across modes; one invocation, including resume attempts",
                }
            )
            save_json(prep_report, prep_history)
            for n in pending:
                values = {**record, "probabilities": np.concatenate(buffers[n])}
                info = {
                    "contract": self.contract(n),
                    "input_sha256": self.data.verify(path),
                    "seconds": elapsed[n],
                    "shared_preprocessing_report": prep_report.relative_to(
                        self.namespace
                    ).as_posix(),
                    "segments": details[n],
                    "preprocessing_reused_across_modes_and_runs": True,
                }
                save_prediction(
                    self.folder(n) / f"records/{split}/{path.stem}.npz", values, info
                )
                parts[n].append(values)
                times[n] += elapsed[n]
                self.progress.update(
                    1,
                    detail=f"{name(n)} · этап 3/3: сохранение · запись {record_number}/{len(paths)}",
                )
        for n in active:
            self.progress.update(
                detail=f"{name(n)} · финальная проверка ID, метрики и сохранение {split}"
            )
            values = {k: np.concatenate([p[k] for p in parts[n]]) for k in parts[n][0]}
            values = align_prediction(values, meta)
            self.finish(n, split, values, times[n])
            done.append(n)
        return sorted(done)

    def finish(self, number: int, split: str, values: dict, seconds: float):
        folder, cfg = self.folder(number), self.configs[number]
        validation = write_split(
            folder,
            split,
            values,
            self.data,
            {
                "contract": self.contract(number),
                "data": self.data.identity,
                "smoke": self.data.smoke,
                "seconds": seconds,
                "config": cfg,
                "class_order": ["W", "N1", "N2", "N3", "REM"],
            },
        )
        baseline = cfg["evaluation"]["baseline_id"]
        if split == "val":
            save_json(
                folder / f"comparison_{name(baseline)}.json",
                diagnostics(values, self.refs.prediction(baseline, split)),
            )
        old = (
            read(folder / "metrics.json") if (folder / "metrics.json").exists() else {}
        )
        timings = old.get("split_seconds", {})
        timings[split] = seconds
        complete = (folder / "test_predictions.json").exists()
        metrics = {
            "smoke": self.data.smoke,
            "validation": validation or read(folder / "validation_metrics.json"),
            "training_seconds": None,
            "inference_seconds": sum(timings.values()),
            "split_seconds": timings,
            "status": "complete" if complete else "validation_only",
            "val_epochs": len(self.data.metadata("val")["ids"]),
            "test_epochs": len(values["ids"])
            if split == "test"
            else old.get("test_epochs"),
            "diagnostic_only": cfg["model"]["kind"] == "diagnostic_context113",
            "timing_semantics": "forward wall time; preprocessing in separate shared_preprocessing records, not multiplied by model count; blends exclude source inference",
        }
        save_json(folder / "metrics.json", metrics)
        save_json(
            folder / f"{split}_status.json",
            {"status": "complete", "contract": self.contract(number)},
        )
        if split == "test":
            ensure_submission(folder, values, self.data)
            target = self.namespace / "submissions" / f"{name(number)}.csv"
            target.parent.mkdir(exist_ok=True)
            shutil.copy2(
                folder
                / ("submission_smoke.csv" if self.data.smoke else "submission.csv"),
                target,
            )
        if not self.data.smoke:
            journal(
                ROOT / "configs" / f"{name(number)}.yaml",
                cfg,
                metrics,
                folder,
                self.args.output_root,
                status=metrics["status"],
                evaluation_set="provided_val_then_unlabelled_test"
                if complete
                else "provided_val_context_selection",
            )

    def blends(self, split: str, numbers: list[int]):
        meta = self.data.metadata(split)
        self.stage(
            f"AnySleep · {split.upper()} · смеси и парные сравнения", len(numbers) * 2
        )
        for source in numbers:
            self.progress.update(
                detail=f"{name(source + 3)}: обычный B, готовые probabilities; без forward"
            )
            new = self.cached(source, split, meta)[0]["probabilities"]
            started = time.perf_counter()
            b = (
                0.5 * new.astype(np.float64)
                + 0.25 * self.refs.prediction(82, split)["probabilities"]
                + 0.25 * self.refs.prediction(85, split)["probabilities"]
            )
            b = (b / b.sum(1, keepdims=True)).astype(np.float32)
            self.finish(
                source + 3,
                split,
                {**meta, "probabilities": b},
                time.perf_counter() - started,
            )
            self.progress.update(1)
            self.progress.update(
                detail=f"{name(source + 6)}: диагностическая113, прежние102/106; без forward"
            )
            started = time.perf_counter()
            diagnostic = (
                0.5 * b.astype(np.float64)
                + 0.25 * self.refs.prediction(102, split)["probabilities"]
                + 0.25 * self.refs.prediction(106, split)["probabilities"]
            )
            diagnostic = (diagnostic / diagnostic.sum(1, keepdims=True)).astype(
                np.float32
            )
            self.finish(
                source + 6,
                split,
                {**meta, "probabilities": diagnostic},
                time.perf_counter() - started,
            )
            self.progress.update(1)

    def select(self, available: list[int]):
        baseline = self.refs.prediction(99, "val")
        rows = [
            {
                "source": "019_pretrained",
                "ensemble": "099_pretrained",
                "macro_f1": evaluate_values(baseline)["macro_f1"],
                "cost_order": 0,
            }
        ]
        for n in available:
            rows.append(
                {
                    "source": name(n),
                    "ensemble": name(n + 3),
                    "macro_f1": read(self.folder(n + 3) / "validation_metrics.json")[
                        "macro_f1"
                    ],
                    "cost_order": n - 146,
                }
            )
        rows.sort(key=lambda r: (-r["macro_f1"], r["cost_order"]))
        result = {
            "criterion": "validation macro-F1 of simple B ensemble; baseline099 eligible; exact ties prefer lower context cost",
            "selected": rows[0],
            "candidates": rows,
            "test_labels_used": False,
            "diagnostic113_excluded_from_selection": True,
            "limitation": "reused validation; no independent final performance estimate",
        }
        save_json(self.namespace / "selection.json", result)
        return result

    def run(self):
        save_json(
            self.namespace / "baseline_validation.json",
            {
                name(n): evaluate_values(self.refs.prediction(n, "val"))
                for n in (19, 99, 113)
            },
        )
        numbers = list(range(147, 150))
        if self.args.require_smoke and not self.data.smoke:
            numbers = [
                n
                for n in numbers
                if (
                    self.args.output_root
                    / "smoke"
                    / name(n)
                    / "validation_metrics.json"
                ).exists()
            ]
            if not numbers:
                raise RuntimeError(
                    "No successful structural smoke; full inference not started"
                )
        for split in ("val",):
            available = self.infer(split, numbers)
            self.blends(split, available)
        selection = self.select(available)  # Persisted before any new test inference.
        test_numbers = available
        if not self.data.smoke and self.args.test_policy == "selected":
            selected = selection["selected"]["source"]
            test_numbers = [n for n in available if name(n) == selected]
        elif not self.data.smoke and self.args.test_policy == "none":
            test_numbers = []
        successful_test = self.infer("test", test_numbers) if test_numbers else []
        self.blends("test", successful_test)
        save_json(
            self.namespace / "context_report.json",
            {
                "smoke": self.data.smoke,
                "validation_sources": [name(n) for n in available],
                "test_sources": [name(n) for n in successful_test],
                "test_policy": self.args.test_policy,
                "failures": self.failures,
                "input_files_used": self.refs.used,
                "all_requested_test_completed": set(successful_test)
                == set(test_numbers),
                "unavailable_validation_sources": [
                    name(n) for n in range(147, 150) if n not in available
                ],
                "full_segment_never_replaced_by_blocks": True,
            },
        )
        print("Val selection:", selection["selected"], flush=True)
        print(
            f"Val sources: {len(available)}/3; test sources: {len(successful_test)}/{len(test_numbers)}; failures/skips: {len(self.failures)}",
            flush=True,
        )


def pack_input(target: Path):
    from src.series_data import SeriesData

    files = {}
    for n in REFERENCES:
        folder = ROOT / "artefacts" / name(n)
        for f in folder.iterdir():
            if f.name in ("provenance.json", "resolved.yaml") or f.name.startswith(
                ("val_predictions.", "test_predictions.")
            ):
                if f.is_file():
                    files[f"results/{name(n)}/{f.name}"] = f
    for asset in ASSET_NAMES:
        p = ROOT / "docs/expansion-assets" / asset
        verify_asset(p, ASSETS[asset][1])
        files["assets/" + asset] = p
    for n in REFERENCES:
        for split in ("val", "test"):
            if f"results/{name(n)}/{split}_predictions.npz" not in files:
                raise FileNotFoundError(f"Need saved {name(n)} {split}")
    manifest = {
        "kind": "anysleep_context_inputs_v1",
        "data": SeriesData(ROOT / "data/raw").identity,
        "files": {n: sha256(p) for n, p in files.items()},
        "purpose": "three pinned AnySleep weights and saved019/082/085/099/102/106/113 val/test; no signals, embeddings, features or training weights",
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(target, "x", ZIP_DEFLATED, compresslevel=4) as z:
        for n, p in files.items():
            z.write(p, n)
        z.writestr("bundle.json", json.dumps(manifest, indent=2))
    print(
        f"Input bundle: {target} · {target.stat().st_size / 2**20:.1f} MiB · {len(files)} files"
    )


def export(root: Path, target: Path):
    if not root.is_dir() or target.resolve().is_relative_to(root.resolve()):
        raise ValueError("Export outside an existing result directory")
    # Preserve uv.lock! Exclude only runtime locks/temp files, not every *.lock.
    files = [
        p
        for p in root.rglob("*")
        if p.is_file()
        and (not p.name.endswith(".lock") or p.name == "uv.lock")
        and ".tmp." not in p.name
    ]
    target.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(target, "x", ZIP_DEFLATED, compresslevel=4) as z:
        with Progress("Накопительный ZIP контекста", len(files), "файл") as progress:
            for p in files:
                if p.is_symlink() or not p.resolve().is_relative_to(root.resolve()):
                    raise ValueError("Unsafe result path")
                z.write(p, p.relative_to(root))
                progress.update(1)
    print(f"Результаты: {target} ({target.stat().st_size / 2**20:.1f} MiB)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--input-root", type=Path)
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "artefacts/anysleep-context"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--threads", type=int, default=max(1, (os.cpu_count() or 2) - 1)
    )
    parser.add_argument("--max-batch", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-epochs", type=int, default=160)
    parser.add_argument("--require-smoke", action="store_true")
    parser.add_argument(
        "--test-policy", choices=("all", "selected", "none"), default="all"
    )
    parser.add_argument("--pack-input", type=Path)
    parser.add_argument("--export", type=Path)
    args = parser.parse_args()
    if args.pack_input:
        pack_input(args.pack_input)
        return
    if args.export:
        export(args.output_root, args.export)
        return
    if (
        not args.data_root
        or not args.input_root
        or args.threads < 1
        or args.max_batch < 1
    ):
        parser.error("Need data/input roots, positive threads and max-batch")
    torch.set_num_threads(args.threads)
    args.output_root.mkdir(parents=True, exist_ok=True)
    with file_lock(args.output_root / ".context.lock", wait=True):
        metadata = {
            "implementation": implementation(),
            "input_manifest_sha256": sha256(args.input_root / "bundle.json"),
        }
        info = args.output_root / "implementation.json"
        if info.exists() and read(info) != metadata:
            raise ValueError(
                "Output belongs to another code/Input; preserve it and use a new output root"
            )
        save_json(info, metadata)
        snapshot(args.output_root)
        with Progress("AnySleep context: подготовка", 0, "запись×режим") as progress:
            Runner(args, progress).run()


if __name__ == "__main__":
    main()
