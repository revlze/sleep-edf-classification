"""Package public inference Input or compact inference outputs."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parent.parent
FINAL = ROOT / "final"
EXCLUDE = {".venv", "__pycache__", ".pytest_cache", ".ruff_cache", "outputs", "bundles"}


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def public_files():
    return [
        p
        for p in FINAL.rglob("*")
        if p.is_file()
        and not set(p.relative_to(FINAL).parts) & EXCLUDE
        and not p.name.endswith((".pyc", ".tmp"))
    ]


def bundle(target):
    files = public_files()
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True
            )
        )
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    manifest = {
        "kind": "sleep_edf_final_v1",
        "solutions": ["113_pretrained", "099_pretrained"],
        "source_commit": revision,
        "source_dirty": dirty,
        "files": {"final/" + p.relative_to(FINAL).as_posix(): sha(p) for p in files},
        "purpose": "raw-test frozen inference; weights and code; reference CSVs are comparison-only; no old probability or feature caches",
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(target, "x", ZIP_DEFLATED, compresslevel=4) as z:
        for p in files:
            z.write(p, "final/" + p.relative_to(FINAL).as_posix())
        z.writestr("bundle.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    print(
        f"Kaggle Input: {target} ({target.stat().st_size / 2**20:.1f}MiB); SHA256={sha(target)}"
    )


def results(source, target):
    if not (source / "run_report.json").exists():
        raise FileNotFoundError("No completed/diagnostic run report")
    report = json.loads((source / "run_report.json").read_text())
    excluded = (
        set()
        if report["output_second_submission"]
        else {"submission_2.csv", "099_probabilities.npz"}
    )
    with ZipFile(target, "x", ZIP_DEFLATED, compresslevel=4) as z:
        for p in source.iterdir():
            if (
                p.is_file()
                and p.name not in excluded
                and p.suffix in (".csv", ".json", ".npz")
            ):
                z.write(p, p.name)
    print(f"Results: {target}; SHA256={sha(target)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["bundle", "results"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--results-root", type=Path)
    args = parser.parse_args()
    if args.action == "bundle":
        bundle(args.output or FINAL / "bundles/sleep-edf-final-inputs.zip")
    else:
        if args.output is None or args.results_root is None:
            parser.error("results needs --output and --results-root")
        results(args.results_root, args.output)


if __name__ == "__main__":
    main()
