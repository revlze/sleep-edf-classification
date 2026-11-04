"""Pinned official sources and weights for the expansion; no implicit weight fallback."""

import hashlib
import json
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

from src.models.pretrained_assets import sha256
from src.storage import atomic_path, file_lock

MANIFEST_PATH = Path(__file__).with_suffix(".json")
MANIFEST = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def checked(path: Path, checksum: str) -> Path:
    if not path.is_file() or sha256(path) != checksum:
        raise ValueError(f"Official file missing or checksum mismatch: {path}")
    return path


def fetch(entry: dict, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(target.with_suffix(target.suffix + ".lock"), wait=True):
        if not target.exists():
            with atomic_path(target) as temporary:
                if "drive_id" in entry:
                    import gdown

                    result = gdown.download(
                        id=entry["drive_id"], output=str(temporary), quiet=True
                    )
                    if result is None:
                        raise RuntimeError(
                            "Official Google Drive checkpoint is unavailable"
                        )
                else:
                    with (
                        urlopen(entry["url"], timeout=120) as response,
                        temporary.open("wb") as stream,
                    ):
                        while chunk := response.read(1024 * 1024):
                            stream.write(chunk)
                checked(temporary, entry["sha256"])
        return checked(target, entry["sha256"])


def asset(name: str, cache: Path, inputs: Path | None = None) -> Path:
    entry = MANIFEST["assets"][name]
    if inputs is not None and (inputs / name).is_file():
        return checked(inputs / name, entry["sha256"])
    target = cache / name
    if "archive" not in entry:
        return fetch(entry, target)
    with file_lock(target.with_suffix(target.suffix + ".lock"), wait=True):
        if not target.exists():
            archive = asset(entry["archive"], cache, inputs)
            with ZipFile(archive) as zipped:
                payload = zipped.read(entry["member"])
            if hashlib.sha256(payload).hexdigest() != entry["sha256"]:
                raise ValueError("Checkpoint member checksum differs")
            with atomic_path(target) as temporary:
                temporary.write_bytes(payload)
        return checked(target, entry["sha256"])


def package(name: str, cache: Path, inputs: Path | None = None) -> Path:
    entry = MANIFEST["packages"][name]
    relative_root = Path("sources") / name / entry["revision"]
    root = cache / relative_root
    for relative, info in entry["files"].items():
        source = inputs / relative_root / relative if inputs is not None else None
        target = root / relative
        if source is not None and source.is_file():
            checked(source, info["sha256"])
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                with atomic_path(target) as temporary:
                    temporary.write_bytes(source.read_bytes())
            checked(target, info["sha256"])
        else:
            fetch(info, target)
    return root


def provenance(names: list[str], packages: list[str]) -> dict:
    return {
        "weights": {n: MANIFEST["assets"][n] for n in names},
        "source_packages": {p: MANIFEST["packages"][p] for p in packages},
    }


def config_sources(config: dict) -> dict:
    """Explicit weights/revisions for every dependency of this configuration."""
    from src.models.pretrained_assets import asset_manifest

    model = config["model"]
    recipes = (
        [model["recipe"]]
        if model["kind"] == "official"
        else model.get("candidates", [])
    )
    recipes += [m["recipe"] for m in model.get("members", []) if "recipe" in m]
    selected = {}
    for recipe in recipes:
        name = recipe["name"]
        if name.startswith("sleepyco_"):
            selected[name] = provenance(
                [
                    f"sleepyco-{name.split('_')[1]}-fold01.pth",
                    f"sleepyco-{name.split('_')[1]}.zip",
                ],
                ["sleepyco"],
            )
        elif name == "robust":
            selected[name] = provenance(["robust-leave-sleepedf-out.gz"], ["robust"])
        else:
            selected[name] = provenance(
                [f"{name}-phan.pt", f"{name}-phan-config.json"], ["physioex"]
            )
    if model["kind"] == "sleepgpt":
        selected["sleepgpt"] = provenance(["sleepgpt.ckpt"], ["sleepgpt"])
    return {
        "official_models": selected,
        "baseline_and_tiny_dependencies": asset_manifest(
            [
                "anysleep_no_hydra.py",
                *[f"anysleep-run{i}.pth" for i in (1, 2, 3)],
                "tinysleepnet.pt",
                "tinysleepnet-source.py",
                "tinysleepnet-config.json",
            ]
        ),
    }
