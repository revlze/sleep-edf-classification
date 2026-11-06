"""Generate106 features from raw test signals in the ORIGINAL Python3.14 environment."""

import argparse
import importlib.metadata
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import numpy as np

from sleep_edf_final.io import (
    FINAL_ROOT,
    input_records,
    load_record,
    read_json,
    save_json,
    save_npz,
    sha256,
    verify_record,
)
from sleep_edf_final.physiological import extract_epoch_features
from sleep_edf_final.progress import Progress


def one_record(parameters):
    root, path, output, limit = parameters
    input_hash = verify_record(root, path)
    target = output / f"{path.stem}.npz"
    settings = read_json(FINAL_ROOT / "feature_spec.json")
    contract = {
        "input_sha256": input_hash,
        "settings": settings,
        "source_sha256": sha256(FINAL_ROOT / "src/sleep_edf_final/physiological.py"),
        "limit": limit,
    }
    if target.exists() and target.with_suffix(".json").exists():
        info = read_json(target.with_suffix(".json"))
        if info["contract"] != contract or info["sha256"] != sha256(target):
            raise ValueError(f"Feature resume conflict: {target}")
        return path.stem, True
    started = time.perf_counter()
    raw = load_record(path)
    x = raw["x"][:limit] if limit else raw["x"]
    spec = settings["spec"]
    parts = []
    for left in range(0, len(x), spec["batch_size"]):
        parts.append(extract_epoch_features(x[left : left + spec["batch_size"]], spec))
    save_npz(target, ids=raw["ids"][: len(x)], features=np.concatenate(parts))
    save_json(
        target.with_suffix(".json"),
        {
            "contract": contract,
            "sha256": sha256(target),
            "seconds": time.perf_counter() - started,
        },
    )
    return path.stem, False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--smoke-epochs", type=int, default=0)
    args = parser.parse_args()
    settings = read_json(FINAL_ROOT / "feature_spec.json")
    versions = {n: importlib.metadata.version(n) for n in settings["packages"]}
    if versions != settings["packages"]:
        raise RuntimeError(f"Feature environment differs: {versions}")
    paths = input_records(args.data_root)
    if args.smoke_epochs:
        paths = paths[:1]
    args.output_root.mkdir(parents=True, exist_ok=True)
    save_json(
        args.output_root / "environment.json",
        {"python": sys.version, "packages": versions},
    )
    jobs = [(args.data_root, p, args.output_root, args.smoke_epochs) for p in paths]
    with Progress(
        "Этап1/3: физиологические106 признаки из сигналов", len(paths), "запись"
    ) as progress:
        with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
            for future in as_completed([pool.submit(one_record, job) for job in jobs]):
                record, cached = future.result()
                progress.update(
                    1,
                    detail=f"{record}: "
                    + ("кэш текущего запуска" if cached else "вычислено из raw test"),
                )


if __name__ == "__main__":
    main()
