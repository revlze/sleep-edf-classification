"""Bounded CPU smoke of the ACTUAL Linux bundle inside `unshare --net`.

This checks cold restoration and real four-epoch inference, not full T4 reproducibility.
Run as: unshare --net python3 final/offline/check_no_network.py [arguments].
"""

import argparse
import importlib.util
import json
import socket
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-root", type=Path, required=True)
    parser.add_argument("--runtime-input", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    with socket.socket() as probe:
        probe.settimeout(2)
        error = probe.connect_ex(("1.1.1.1", 443))
    if not error:
        raise RuntimeError("Network is reachable: run this check inside unshare --net")
    spec = importlib.util.spec_from_file_location(
        "offline_runtime", args.final_root / "offline_runtime.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    runtimes, manifest = module.prepare(
        args.runtime_input,
        args.cache_root,
        args.final_root,
        lambda s: print(s, flush=True),
    )
    env = module.execution_env()
    env.update(
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
        PYTHONIOENCODING="utf-8",
    )
    subprocess.run(
        [
            str(runtimes["inference"]),
            "-I",
            str(args.final_root / "run.py"),
            "--data-root",
            str(args.data_root),
            "--output-root",
            str(args.output_root),
            "--feature-python",
            str(runtimes["features"]),
            "--device",
            "cpu",
            "--threads",
            "2",
            "--feature-workers",
            "1",
            "--smoke-epochs",
            "4",
            "--output-second-submission",
        ],
        env=env,
        check=True,
    )
    report = json.loads((args.output_root / "run_report.json").read_text())
    if report["status"] != "STRUCTURAL_SMOKE" or report["epochs"] != 4:
        raise RuntimeError("Offline smoke did not complete")
    if set(report["comparisons"]) != {"157", "113"}:
        raise RuntimeError("Missing optional second export")
    proof = {
        "network_connect_errno": error,
        "namespace": str(Path("/proc/self/ns/net").readlink()),
        "offline_cold_restoration": True,
        "real_checkpoints_loaded": True,
        "inference_epochs": 4,
        "both_csv_created": True,
        "python_versions": {
            k: v["python_version"] for k, v in manifest["profiles"].items()
        },
        "scope": "Linux CPU bootstrap/smoke only; full T4 id,label verification remains pending",
    }
    (args.output_root / "offline_bootstrap_proof.json").write_text(
        json.dumps(proof, indent=2) + "\n"
    )
    print("OFFLINE CHECK PASSED:", json.dumps(proof), flush=True)


if __name__ == "__main__":
    main()
