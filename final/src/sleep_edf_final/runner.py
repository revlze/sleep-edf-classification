"""Frozen113 inference from raw test; optional099 export. No training or test labels."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from .heads import ResidualHead, WindowBank, confidence_features
from .io import (
    FINAL_ROOT,
    input_records,
    labels_csv,
    load_record,
    read_json,
    save_json,
    save_npz,
    sha256,
    verify_record,
    write_submission,
)
from .models import FrozenBackbones, verify_weights
from .progress import Progress


def configure(device, threads):
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_device(device)
        if "T4" not in torch.cuda.get_device_name(device):
            raise RuntimeError("Final GPU profile is Tesla T4; choose T4 x2 in Kaggle")
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "device": device,
        "gpu": torch.cuda.get_device_name(device)
        if device.startswith("cuda")
        else None,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "threads": threads,
        "deterministic_algorithms": True,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "packages": {
            n: importlib.metadata.version(n)
            for n in ("numpy", "scipy", "scikit-learn", "omegaconf")
        },
    }


def implementation():
    files = sorted((FINAL_ROOT / "src").rglob("*.py")) + [
        FINAL_ROOT / "run.py",
        FINAL_ROOT / "compute_features.py",
        FINAL_ROOT / "feature_spec.json",
        FINAL_ROOT / "uv.lock",
        FINAL_ROOT / "environments/features/uv.lock",
        FINAL_ROOT / "configs/113_pretrained.yaml",
        FINAL_ROOT / "configs/099_pretrained.yaml",
    ]
    return hashlib.sha256(
        b"".join(p.read_bytes().replace(b"\r\n", b"\n") for p in files)
    ).hexdigest()


@torch.inference_mode()
def head_prediction(config, number, inputs, experts, meta, device, progress):
    checkpoint = torch.load(
        FINAL_ROOT / "weights" / config["checkpoint"],
        map_location="cpu",
        weights_only=True,
    )
    model = ResidualHead(inputs.shape[1], config).to(device).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    mean, std = checkpoint["mean"].numpy(), checkpoint["std"].numpy()
    x = np.clip(
        (np.where(np.isfinite(inputs), inputs, mean) - mean) / std, -12, 12
    ).astype(np.float32)
    bank = WindowBank(x, experts, meta, device)
    output = []
    width = config["model"]["context"]
    batch = config["batch_size"]
    for left in range(0, len(x), batch):
        progress.update(
            detail=f"head{number}: batch {left // batch + 1}/{(len(x) + batch - 1) // batch}, context{width}; inference only"
        )
        xx, p, _, mask, center, lengths = bank.batch(
            np.arange(left, min(len(x), left + batch)), width
        )
        with torch.autocast(
            device_type=torch.device(device).type, enabled=device.startswith("cuda")
        ):
            logits, _, _ = model(xx, p, mask, lengths)
        row = torch.arange(len(xx), device=device)
        output.append(logits[row, center].float().softmax(-1).cpu().numpy())
    return np.concatenate(output)


def assemble(raw_experts, experts, gate, temporal):
    """Keep the two historical rounding contracts distinct."""
    b_for113 = (experts * np.array([0.5, 0.25, 0.25])[None, :, None]).sum(1)
    p113 = (
        0.5 * b_for113
        + 0.25 * gate.astype(np.float64)
        + 0.25 * temporal.astype(np.float64)
    )
    p113 = (p113 / p113.sum(1, keepdims=True)).astype(np.float32)
    p099 = (
        raw_experts.astype(np.float64) * np.array([0.5, 0.25, 0.25])[None, :, None]
    ).sum(1)
    p099 = (p099 / p099.sum(1, keepdims=True)).astype(np.float32)
    return p113, p099


def run(args):
    started = time.perf_counter()
    weights = verify_weights()
    env = configure(args.device, args.threads)
    cfg = OmegaConf.to_container(
        OmegaConf.load(FINAL_ROOT / "configs/113_pretrained.yaml"), resolve=True
    )
    paths = input_records(args.data_root)
    if args.smoke_epochs:
        paths = paths[:1]
    identity = {
        "implementation": implementation(),
        "weights": sha256(FINAL_ROOT / "weights/manifest.json"),
        "data": sha256(FINAL_ROOT / "data_contract.json"),
        "device": args.device,
        "smoke_epochs": args.smoke_epochs,
        "deterministic": True,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    contract_path = args.output_root / "contract.json"
    if contract_path.exists() and read_json(contract_path) != identity:
        raise ValueError(
            "Output belongs to another version/profile; preserve it and choose a new output directory"
        )
    save_json(contract_path, identity)
    feature_root = args.output_root / "cache/physiological"
    feature_command = [
        str(args.feature_python),
        str(FINAL_ROOT / "compute_features.py"),
        "--data-root",
        str(args.data_root),
        "--output-root",
        str(feature_root),
        "--workers",
        str(args.feature_workers),
    ]
    if args.smoke_epochs:
        feature_command += ["--smoke-epochs", str(args.smoke_epochs)]
    subprocess.run(
        feature_command,
        check=True,
        env={
            **os.environ,
            "OMP_NUM_THREADS": str(max(1, args.threads // args.feature_workers)),
            "MKL_NUM_THREADS": str(max(1, args.threads // args.feature_workers)),
            "OPENBLAS_NUM_THREADS": str(max(1, args.threads // args.feature_workers)),
        },
    )
    feature_seconds = time.perf_counter() - started
    all_ids = []
    records = []
    epochs = []
    parts = []
    physio = []
    setup_started = time.perf_counter()
    with Progress(
        "Этап2/3: frozen backbones из raw test", len(paths), "запись"
    ) as progress:
        progress.update(
            detail="Загрузка3 AnySleep, оригинального Tiny и082/085 refit checkpoints"
        )
        models = FrozenBackbones(args.device)
        setup_seconds = time.perf_counter() - setup_started
        for number, path in enumerate(paths, 1):
            input_hash = verify_record(args.data_root, path)
            target = args.output_root / f"cache/backbones/{path.stem}.npz"
            info = target.with_suffix(".json")
            progress.update(
                detail=f"{path.stem} [{number}/{len(paths)}]: чтение сигналов и preprocessing"
            )
            record = load_record(path)
            if args.smoke_epochs:
                record = {
                    k: (
                        v[: args.smoke_epochs]
                        if k in ("x", "ids", "epoch_index")
                        else v
                    )
                    for k, v in record.items()
                }
            if info.exists() and target.exists():
                state = read_json(info)
                if (
                    state["identity"] != identity
                    or state["input_sha256"] != input_hash
                    or state["sha256"] != sha256(target)
                ):
                    raise ValueError("Backbone resume checksum/contract mismatch")
                with np.load(target, allow_pickle=False) as z:
                    if not np.array_equal(z["ids"], record["ids"]):
                        raise ValueError(
                            "Backbone cache epoch IDs differ from raw record"
                        )
                    result = {k: z[k] for k in z.files if k != "ids"}
            else:
                began = time.perf_counter()
                result = models.record(
                    record["x"],
                    record["epoch_index"],
                    lambda s: progress.update(
                        detail=f"{path.stem} [{number}/{len(paths)}] · " + s
                    ),
                )
                save_npz(target, ids=record["ids"], **result)
                save_json(
                    info,
                    {
                        "identity": identity,
                        "input_sha256": input_hash,
                        "sha256": sha256(target),
                        "seconds": time.perf_counter() - began,
                    },
                )
            feature_path = feature_root / f"{path.stem}.npz"
            if (
                sha256(feature_path)
                != read_json(feature_path.with_suffix(".json"))["sha256"]
            ):
                raise ValueError("Feature file checksum mismatch")
            with np.load(feature_path, allow_pickle=False) as z:
                if not np.array_equal(z["ids"], record["ids"]):
                    raise ValueError("Feature epoch alignment mismatch")
                physio.append(z["features"])
            all_ids.append(record["ids"].astype(str))
            records.append(np.full(len(record["ids"]), path.stem))
            epochs.append(record["epoch_index"])
            parts.append(result)
            progress.update(1)
        del models
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    meta = {
        "ids": np.concatenate(all_ids),
        "record_id": np.concatenate(records),
        "epoch_index": np.concatenate(epochs),
    }
    arrays = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
    raw_experts = np.stack(
        [arrays["p019"], arrays["p082"], arrays["p085"]], axis=1
    ).astype(np.float32)
    experts = raw_experts.copy()
    experts /= experts.sum(-1, keepdims=True)
    confidence = confidence_features(experts)
    with Progress("Этап3/3: heads102/106 и финальные CSV", 3, "шаг") as progress:
        gate = head_prediction(
            cfg["heads"]["102"],
            102,
            np.concatenate([confidence, np.concatenate(physio)], axis=1),
            experts,
            meta,
            args.device,
            progress,
        )
        progress.update(1)
        temporal = head_prediction(
            cfg["heads"]["106"],
            106,
            np.concatenate([confidence, arrays["embeddings"]], axis=1),
            experts,
            meta,
            args.device,
            progress,
        )
        progress.update(1)
        p113, p099 = assemble(raw_experts, experts, gate, temporal)
        save_npz(
            args.output_root / "component_probabilities.npz",
            **meta,
            p019=arrays["p019"],
            p082=arrays["p082"],
            p085=arrays["p085"],
            p102=gate,
            p106=temporal,
        )
        outputs = [(113, "submission.csv", p113)]
        if args.output_second_submission:
            outputs.append((99, "submission_2.csv", p099))
        comparisons = {}
        for number, filename, p in outputs:
            write_submission(
                args.output_root / filename,
                meta["ids"],
                p,
                args.data_root / "sample_submission.csv",
                smoke=bool(args.smoke_epochs),
            )
            save_npz(
                args.output_root / f"{number:03}_probabilities.npz",
                **meta,
                probabilities=p,
            )
            expected = labels_csv(FINAL_ROOT / "submissions" / filename)
            actual = labels_csv(args.output_root / filename)
            mismatches = [
                {"id": i, "expected": expected.get(i), "actual": label}
                for i, label in actual.items()
                if expected.get(i) != label
            ]
            comparisons[str(number)] = {
                "compared_ids": len(actual),
                "label_mismatches": len(mismatches),
                "examples": mismatches[:20],
                "output_sha256": sha256(args.output_root / filename),
                "reference_sha256": sha256(FINAL_ROOT / "submissions" / filename),
                "semantics": "comparison only; reference predictions never replace inference outputs",
            }
        progress.update(
            1, detail="CSV созданы; проверка всех пар id,label с выбранными отправками"
        )
    failures = sum(v["label_mismatches"] for v in comparisons.values())
    report = {
        "status": "STRUCTURAL_SMOKE"
        if args.smoke_epochs
        else ("EXACT_MATCH" if failures == 0 else "LABEL_MISMATCH"),
        "training": False,
        "raw_test_inference": True,
        "environment": env,
        "feature_environment": read_json(feature_root / "environment.json"),
        "feature_stage_seconds": feature_seconds,
        "model_setup_seconds": setup_seconds,
        "total_seconds": time.perf_counter() - started,
        "epochs": len(meta["ids"]),
        "output_second_submission": args.output_second_submission,
        "comparisons": comparisons,
        "weights_manifest_sha256": sha256(FINAL_ROOT / "weights/manifest.json"),
        "identity": identity,
        "test_labels_read": False,
        "input_metadata": read_json(FINAL_ROOT / "data_contract.json")["source_input"],
    }
    save_json(args.output_root / "run_report.json", report)
    save_json(args.output_root / "weights_manifest.json", weights)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if not args.smoke_epochs and failures:
        raise RuntimeError(
            "Final labels differ from selected submission. Outputs preserved for diagnosis; no label corrections or fallback applied."
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--feature-python", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=max(1, os.cpu_count() or 2))
    parser.add_argument("--feature-workers", type=int, default=2)
    parser.add_argument("--output-second-submission", action="store_true")
    parser.add_argument("--smoke-epochs", type=int, default=0)
    args = parser.parse_args()
    if args.threads < 1 or args.feature_workers < 1 or args.smoke_epochs < 0:
        parser.error("Positive threads/workers; nonnegative smoke epochs required")
    run(args)
