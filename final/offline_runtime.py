"""Restore Linux runtimes using ONLY local archives and stdlib venv; no installers."""

import contextlib
import hashlib
import json
import os
import platform
import subprocess
import tarfile
import time
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

KIND = "sleep_edf_offline_runtime_v1"


def canonical_sha(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


class RuntimeInput:
    def __init__(self, path):
        self.path = Path(path)
        if self.path.is_dir():
            self.manifest = json.loads(
                (self.path / "offline_manifest.json").read_text()
            )
        else:
            with ZipFile(self.path) as archive:
                self.manifest = json.loads(archive.read("offline_manifest.json"))
        if self.manifest.get("kind") != KIND:
            raise ValueError("Wrong offline runtime Input")

    @contextlib.contextmanager
    def stream(self, relative):
        rel = PurePosixPath(relative)
        if (
            rel.is_absolute()
            or ".." in rel.parts
            or "\\" in relative
            or ":" in relative
        ):
            raise ValueError("Unsafe runtime path")
        if self.path.is_dir():
            path = (self.path / relative).resolve()
            if not path.is_relative_to(self.path.resolve()):
                raise ValueError("Runtime archive is outside Input")
            with path.open("rb") as stream:
                yield stream
        else:
            with ZipFile(self.path) as archive, archive.open(relative) as stream:
                yield stream


def discover(input_root):
    found = []
    for path in Path(input_root).rglob("offline_manifest.json"):
        if json.loads(path.read_text()).get("kind") == KIND:
            found.append(path.parent)
    if not found:
        for path in Path(input_root).rglob("*.zip"):
            with ZipFile(path) as archive:
                if "offline_manifest.json" in archive.namelist():
                    if (
                        json.loads(archive.read("offline_manifest.json")).get("kind")
                        == KIND
                    ):
                        found.append(path)
    if len(found) != 1:
        raise RuntimeError(
            f"Need exactly one sleep-edf-final-runtime-linux Input; found {found}"
        )
    return found[0]


class HashingReader:
    def __init__(self, stream):
        self.stream, self.digest, self.count = stream, hashlib.sha256(), 0

    def read(self, size=-1):
        data = self.stream.read(size)
        self.digest.update(data)
        self.count += len(data)
        return data


def extract_checked(source, relative, target, notify):
    """Stream from read-only Input/ZIP once; execute nothing until SHA is verified."""
    expected = source.manifest["files"][relative]
    files, size, refreshed = 0, 0, time.monotonic()
    with source.stream(relative) as raw:
        reader = HashingReader(raw)
        with tarfile.open(fileobj=reader, mode="r|gz") as archive:
            for member in archive:
                name = PurePosixPath(member.name)
                if (
                    name.is_absolute()
                    or ".." in name.parts
                    or "\\" in member.name
                    or ":" in member.name
                ):
                    raise ValueError("Unsafe path inside runtime tar")
                # Builder dereferences links: restore only regular files/directories.
                if not (member.isfile() or member.isdir()):
                    raise ValueError("Unexpected link/device inside runtime tar")
                filtered = tarfile.data_filter(member, str(target))
                destination = (Path(target) / filtered.name).resolve()
                if not destination.is_relative_to(Path(target).resolve()):
                    raise ValueError("Runtime tar escapes extraction directory")
                if member.isdir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as src, destination.open("wb") as out:
                    while chunk := src.read(2**20):
                        out.write(chunk)
                        size += len(chunk)
                        if time.monotonic() - refreshed >= 7:
                            notify(
                                f"{Path(relative).name}: {size / 2**20:.0f}/{expected['uncompressed_bytes'] / 2**20:.0f}MiB распаковано"
                            )
                            refreshed = time.monotonic()
                destination.chmod(filtered.mode)
                files += 1
        while reader.read(2**20):
            pass
    if (
        reader.digest.hexdigest() != expected["sha256"]
        or reader.count != expected["bytes"]
    ):
        raise ValueError(f"Runtime archive checksum mismatch: {relative}")
    if files != expected["regular_files"] or size != expected["uncompressed_bytes"]:
        raise ValueError(f"Runtime archive inventory mismatch: {relative}")


def execution_env():
    env = dict(os.environ)
    for key in (
        "PYTHONHOME",
        "PYTHONPATH",
        "VIRTUAL_ENV",
        "CONDA_PREFIX",
        "UV_PROJECT_ENVIRONMENT",
    ):
        env.pop(key, None)
    env.update(
        PYTHONNOUSERSITE="1",
        UV_OFFLINE="1",
        UV_PYTHON_DOWNLOADS="never",
        PIP_NO_INDEX="1",
    )
    return env


def verify_profile(python, profile):
    code = """import importlib.metadata as m,json,sys
print(json.dumps({'python':sys.version.split()[0], 'packages':{
d.metadata['Name'].lower().replace('_','-'):d.version for d in m.distributions()}}))"""
    actual = json.loads(
        subprocess.check_output(
            [str(python), "-I", "-c", code], env=execution_env(), text=True
        )
    )
    if (
        actual["python"] != profile["python_version"]
        or actual["packages"] != profile["packages"]
    ):
        raise ValueError("Restored runtime versions differ from locked Linux build")
    return actual


def prepare(source_path, cache_root, final_root, notify=print):
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError(
            "Offline runtime targets Linux x86_64 (Kaggle); do not run Windows binaries"
        )
    source = RuntimeInput(source_path)
    expected_locks = {
        "inference": canonical_sha(Path(final_root) / "uv.lock"),
        "features": canonical_sha(Path(final_root) / "environments/features/uv.lock"),
    }
    if source.manifest["source_locks"] != expected_locks:
        raise ValueError("Runtime Input was built for different lockfiles")
    manifest_hash = hashlib.sha256(
        json.dumps(source.manifest, sort_keys=True).encode()
    ).hexdigest()
    root = Path(cache_root) / manifest_hash[:16]
    ready = root / "ready.json"
    if root.exists() and not ready.exists():
        raise RuntimeError(
            f"Incomplete runtime extraction preserved at {root}; choose a fresh RUNTIME_CACHE"
        )
    if not ready.exists():
        root.mkdir(parents=True)
        for index, (name, profile) in enumerate(source.manifest["profiles"].items(), 1):
            notify(
                f"Окружение {index}/2: Python{profile['python_version']}, локальная распаковка"
            )
            extract_checked(source, profile["python_archive"], root, notify)
            base = (
                root / profile["python_prefix"] / "bin" / ("python" + profile["minor"])
            )
            env_root = root / ("env-" + name)
            subprocess.run(
                [
                    str(base),
                    "-I",
                    "-m",
                    "venv",
                    "--without-pip",
                    "--symlinks",
                    str(env_root),
                ],
                check=True,
                env=execution_env(),
            )
            notify(
                f"Окружение {index}/2: восстановление готовых пакетов {name}; установка из сети не выполняется"
            )
            extract_checked(
                source,
                profile["site_archive"],
                env_root / "lib" / ("python" + profile["minor"]),
                notify,
            )
            verify_profile(env_root / "bin/python", profile)
        ready.write_text(
            json.dumps(
                {"runtime_manifest_sha256": manifest_hash, "offline": True}, indent=2
            )
        )
    result = {}
    for name, profile in source.manifest["profiles"].items():
        python = root / ("env-" + name) / "bin/python"
        verify_profile(python, profile)
        result[name] = python
    notify("Оба окружения готовы; сеть для финального запуска не требуется")
    return result, source.manifest
