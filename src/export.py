"""Один ZIP со всеми результатами, кэшами и снимком кода для выгрузки с Kaggle."""

import argparse
import hashlib
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from src.progress import Progress
from src.storage import file_lock


def export_results(
    output_root: Path,
    archive: Path,
    *,
    cache_root: Path | None = None,
    fold_root: Path | None = None,
) -> Path:
    root = output_root.resolve(strict=True)
    with file_lock(root / ".export.lock"):
        return _export_results(
            root, archive, cache_root=cache_root, fold_root=fold_root
        )


def _export_results(
    output_root: Path, archive: Path, *, cache_root=None, fold_root=None
) -> Path:
    """Упаковать каталог целиком потоково, без окружения и исходных сигналов."""
    root = output_root.resolve(strict=True)
    target = archive.resolve()
    if target.is_relative_to(root):
        raise ValueError("Archive must be outside output_root")
    if (root / ".linear.lock").exists():
        raise RuntimeError("Training lock exists; stop training before exporting")
    if target.exists():
        raise FileExistsError(target)
    sources = [(root, Path("."))]
    if cache_root is not None:
        sources.append((Path(cache_root).resolve(strict=True), Path("cache/features")))
    if fold_root is not None:
        sources.append((Path(fold_root).resolve(strict=True), Path("splits")))
    files = {}
    for source, prefix in sources:
        for path in sorted(source.rglob("*")):
            if (
                not path.is_file()
                or path.name.endswith(".lock")
                or ".tmp." in path.name
            ):
                continue
            if path.is_symlink() or not path.resolve().is_relative_to(source):
                raise ValueError("Source contains links outside its directory")
            name = prefix / path.relative_to(source)
            if name in files and files[name].resolve() != path.resolve():
                if (
                    hashlib.sha256(files[name].read_bytes()).digest()
                    != hashlib.sha256(path.read_bytes()).digest()
                ):
                    raise ValueError(f"Conflicting archive entry: {name}")
            files[name] = path
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    temporary.touch(exist_ok=False)
    try:
        with ZipFile(
            temporary, "w", compression=ZIP_DEFLATED, compresslevel=1
        ) as bundle:
            with Progress("Архив результатов", len(files), "файл") as progress:
                for name, path in files.items():
                    bundle.write(path, Path(root.name) / name)
                    progress.update(1)
        temporary.rename(target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    print(f"Готово: {target} ({target.stat().st_size / 2**20:.1f} MiB)", flush=True)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--fold-root", type=Path)
    args = parser.parse_args()
    export_results(
        args.output_root,
        args.archive,
        cache_root=args.cache_root,
        fold_root=args.fold_root,
    )


if __name__ == "__main__":
    main()
