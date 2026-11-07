"""Pinned official assets. Never accept a pointer, partial download or changed weights."""

import hashlib
import importlib.util
import json
from pathlib import Path
from urllib.request import urlopen

from src.storage import atomic_path, file_lock

ANYSLEEP_REVISION = "899bfff284f3eb3e7448bc90990f2230757470a2"
SLEEPFM_REVISION = "95903b507c183bc6154e7a1bda064d6b043d5b8a"
ANYSLEEP_URL = (
    f"https://raw.githubusercontent.com/dslaborg/AnySleep/{ANYSLEEP_REVISION}"
)
SLEEPFM_URL = (
    f"https://huggingface.co/braindecode/SleepFMStager/resolve/{SLEEPFM_REVISION}"
)
ASSETS = {
    "anysleep_no_hydra.py": (
        f"{ANYSLEEP_URL}/examples/anysleep_no_hydra.py",
        "b94a9b62216a663e2923138b42c9439ccfa7b94cc8ca95729e2548c80afe785f",
    ),
    "usleep_no_hydra.py": (
        f"{ANYSLEEP_URL}/examples/usleep_no_hydra.py",
        "343968388a78ffc795ff2a3bd29659fe4a5c8a10bd7a4d1a443dd5761a89552a",
    ),
    "anysleep-run1.pth": (
        f"{ANYSLEEP_URL}/models/anysleep-run1.pth",
        "9f305c0f81a58d36092bb88b04832a48f0f4dae940f31486cd2284e9c659483f",
    ),
    "anysleep-run2.pth": (
        f"{ANYSLEEP_URL}/models/anysleep-run2.pth",
        "dde1e762718787e6705c0169f30cc43be2a7681fba82e83a5a01c11c70987902",
    ),
    "anysleep-run3.pth": (
        f"{ANYSLEEP_URL}/models/anysleep-run3.pth",
        "ac846afe53ccd9761455b53279a5102cbde92904526cbad6778b3b4ea91581d1",
    ),
    "usleep-run1.pth": (
        f"{ANYSLEEP_URL}/models/usleep-run1.pth",
        "eb88c447b543918f2e16a2ce3f4d40e9ae975b07b34f4709568d3a66ff2287ee",
    ),
    "sleepfm-stager.safetensors": (
        f"{SLEEPFM_URL}/model.safetensors",
        "6a0cdbee2fc6eaf5383cc85984f0b30f4911ac8ca6254ddd5bb1fbecdad3039a",
    ),
    "sleepfm-config.json": (
        f"{SLEEPFM_URL}/config.json",
        "24dbc797e59609d99abdc307c2b8ba02f4947bfad32fbe4165f474a288a7e63a",
    ),
}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_asset(path: Path, expected: str) -> None:
    with path.open("rb") as stream:
        if stream.read(128).startswith(b"version https://git-lfs.github.com/spec"):
            raise ValueError(f"Git LFS pointer instead of weights: {path}")
    if sha256(path) != expected:
        raise ValueError(f"Checkpoint/source SHA-256 mismatch: {path}")


def resolve_asset(name: str, cache: Path, inputs: Path | None = None) -> Path:
    """Inputs contain the listed flat filenames; missing assets download officially."""
    url, expected = ASSETS[name]
    if inputs is not None and (inputs / name).is_file():
        path = inputs / name
        verify_asset(path, expected)
        return path
    target = cache / name
    target.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(target.with_suffix(target.suffix + ".lock"), wait=True):
        if not target.exists():
            with atomic_path(target) as temporary:
                with urlopen(url, timeout=120) as response, temporary.open("wb") as out:
                    while block := response.read(1024 * 1024):
                        out.write(block)
                verify_asset(temporary, expected)
        verify_asset(target, expected)
    return target


def official_module(path: Path):
    """Execute only the hash-verified standalone official architecture."""
    spec = importlib.util.spec_from_file_location(f"sleep_edf_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def asset_manifest(names: list[str]) -> list[dict]:
    return [
        {"name": name, "url": ASSETS[name][0], "sha256": ASSETS[name][1]}
        for name in names
    ]


def sleepfm_parameters(path: Path) -> dict:
    import torch

    params = json.loads(path.read_text())
    params.pop("braindecode_version")
    if params.pop("activation") != "torch.nn.modules.activation.ELU":
        raise ValueError("Unexpected SleepFM activation")
    params.update(
        activation=torch.nn.ELU,
        n_chans=3,
        n_outputs=5,
        n_times=38400,
        sfreq=128,
        channel_modalities=["BAS"] * 3,
        on_non_divisible="error",
    )
    return params
