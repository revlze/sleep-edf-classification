"""ONLINE preparation on Linux; Kaggle consumes the resulting bundle OFFLINE.

Both environments are installed by pinned uv sync --locked --no-dev. Ship managed
Python and installed site-packages separately; recreate venv paths after extraction.
No Windows environment or absolute build-time venv path is reused in Kaggle.
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

UV_VERSION = "0.12.15"
UV_ASSET = "uv-x86_64-unknown-linux-gnu.tar.gz"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical_sha(path):
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def download(url, target):
    if target.exists():
        return
    request = urllib.request.Request(
        url, headers={"User-Agent": "SleepEDF-offline-builder"}
    )
    temporary = target.with_suffix(target.suffix + ".part")
    with (
        urllib.request.urlopen(request, timeout=120) as src,
        temporary.open("wb") as out,
    ):
        while chunk := src.read(2**20):
            out.write(chunk)
    temporary.replace(target)


def execute(command, env):
    print("RUN:", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), env=env, check=True)


def tar_directory(source, destination, prefix):
    if destination.exists() and destination.with_suffix(".json").exists():
        old = json.loads(destination.with_suffix(".json").read_text())
        if old["sha256"] == sha(destination):
            return old
        raise ValueError("Existing runtime archive checksum differs")
    count, size = 0, 0

    def keep(info):
        nonlocal count, size
        if "__pycache__" in Path(info.name).parts or info.name.endswith(
            (".pyc", ".pyo")
        ):
            return None
        if info.isfile():
            count += 1
            size += info.size
        return info

    temporary = destination.with_suffix(destination.suffix + ".part")
    print(f"PACK: {prefix}", flush=True)
    with tarfile.open(temporary, "w:gz", compresslevel=1, dereference=True) as tar:
        tar.add(source, arcname=prefix, filter=keep)
    temporary.replace(destination)
    result = {
        "sha256": sha(destination),
        "bytes": destination.stat().st_size,
        "regular_files": count,
        "uncompressed_bytes": size,
    }
    destination.with_suffix(".json").write_text(json.dumps(result, indent=2))
    print(f"PACKED: {prefix}: {destination.stat().st_size / 2**20:.1f} MiB", flush=True)
    return result


def write_bundle(artifacts, output, manifest):
    # Opaque extensions prevent Dataset import from unpacking nested gzip files.
    manifest = json.loads(json.dumps(manifest))
    manifest["files"] = {
        (name if name.endswith(".bin") else name + ".bin"): value
        for name, value in manifest["files"].items()
    }
    for profile in manifest["profiles"].values():
        for key in ("python_archive", "site_archive"):
            if not profile[key].endswith(".bin"):
                profile[key] += ".bin"
    manifest["payload_format"] = (
        "gzip tar stored as .bin to prevent automatic Dataset extraction"
    )
    (artifacts / "offline_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "x", compression=ZIP_STORED, allowZip64=True) as zipped:
        for relative in manifest["files"]:
            zipped.write(artifacts / Path(relative).name.removesuffix(".bin"), relative)
        zipped.write(artifacts / "offline_manifest.json", "offline_manifest.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-root", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != "linux" or platform.machine() != "x86_64":
        parser.error("Build on Linux x86_64, e.g. WSL Ubuntu; not Windows")
    root, work = args.final_root.resolve(), args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    locks = {
        "inference": canonical_sha(root / "uv.lock"),
        "features": canonical_sha(root / "environments/features/uv.lock"),
    }
    contract = work / "build_contract.json"
    if contract.exists() and json.loads(contract.read_text()) != locks:
        raise ValueError(
            "Work dir belongs to other locks; preserve it and choose a new dir"
        )
    contract.write_text(json.dumps(locks, indent=2))
    release_url = (
        f"https://api.github.com/repos/astral-sh/uv/releases/tags/{UV_VERSION}"
    )
    request = urllib.request.Request(
        release_url, headers={"User-Agent": "SleepEDF-offline-builder"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        release = json.load(response)
    asset = next(a for a in release["assets"] if a["name"] == UV_ASSET)
    expected = asset.get("digest", "")
    if not expected.startswith("sha256:"):
        raise ValueError(
            "Pinned uv release asset has no SHA256; refuse unverified tool"
        )
    uv_archive = work / UV_ASSET
    download(asset["browser_download_url"], uv_archive)
    if sha(uv_archive) != expected.removeprefix("sha256:"):
        raise ValueError("uv release checksum mismatch")
    tool = work / "tool"
    tool.mkdir(exist_ok=True)
    with tarfile.open(uv_archive) as tar:
        tar.extractall(tool, filter="data")
    uv = next(tool.rglob("uv"))
    env = {
        **os.environ,
        "UV_CACHE_DIR": str(work / "uv-cache"),
        "UV_PYTHON_INSTALL_DIR": str(work / "managed-python"),
        "UV_PYTHON_BIN_DIR": str(work / "python-bin"),
        "UV_LINK_MODE": "copy",
    }
    execute([uv, "python", "install", "3.12.12", "3.14.2", "--no-bin"], env)
    artifacts = work / "artifacts"
    artifacts.mkdir(exist_ok=True)
    manifest = {
        "kind": "sleep_edf_offline_runtime_v1",
        "os": "linux",
        "arch": "x86_64",
        "source_locks": locks,
        "build_uv": {
            "version": UV_VERSION,
            "sha256": sha(uv_archive),
            "url": asset["browser_download_url"],
        },
        "profiles": {},
        "files": {},
        "runtime_network": "not needed; local extraction and stdlib venv only",
    }
    inspection = """import importlib.metadata as m,json,sys,sysconfig
rows={}
for d in m.distributions():
 rows[d.metadata['Name'].lower().replace('_','-')]={'version':d.version,'wheel':d.read_text('WHEEL')}
print(json.dumps({'python':sys.version.split()[0],'base_prefix':sys.base_prefix,
 'site_packages':sysconfig.get_path('purelib'),'distributions':rows}))"""
    for profile, version, minor, project in [
        ("inference", "3.12.12", "3.12", root),
        ("features", "3.14.2", "3.14", root / "environments/features"),
    ]:
        python = subprocess.check_output(
            [uv, "python", "find", "--managed-python", version], env=env, text=True
        ).strip()
        profile_env = {**env, "UV_PROJECT_ENVIRONMENT": str(work / ("venv-" + profile))}
        execute(
            [
                uv,
                "sync",
                "--project",
                project,
                "--locked",
                "--no-dev",
                "--python",
                python,
            ],
            profile_env,
        )
        venv_python = Path(profile_env["UV_PROJECT_ENVIRONMENT"]) / "bin/python"
        details = json.loads(
            subprocess.check_output(
                [venv_python, "-I", "-c", inspection], text=True, env=env
            )
        )
        if details["python"] != version:
            raise ValueError("Unexpected Python patch version")
        for d in details["distributions"].values():
            for line in (d["wheel"] or "").splitlines():
                if line.startswith("Tag:") and "manylinux_2_" in line:
                    floor = int(line.split("manylinux_2_", 1)[1].split("_", 1)[0])
                    if floor > 35:
                        raise ValueError(
                            f"Wheel needs newer glibc than Kaggle Ubuntu22: {line}"
                        )
        tag = "python" + minor.replace(".", "")
        python_name, site_name = tag + ".tar.gz", profile + "-site.tar.gz"
        manifest["files"]["runtime/" + python_name] = tar_directory(
            Path(details["base_prefix"]), artifacts / python_name, tag
        )
        manifest["files"]["runtime/" + site_name] = tar_directory(
            Path(details["site_packages"]), artifacts / site_name, "site-packages"
        )
        manifest["profiles"][profile] = {
            "python_version": version,
            "minor": minor,
            "python_archive": "runtime/" + python_name,
            "python_prefix": tag,
            "site_archive": "runtime/" + site_name,
            "packages": {n: d["version"] for n, d in details["distributions"].items()},
        }
    manifest["built_unix"] = time.time()
    write_bundle(artifacts, args.output, manifest)
    print(
        f"READY: {args.output} ({args.output.stat().st_size / 2**30:.2f} GiB), SHA256={sha(args.output)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
