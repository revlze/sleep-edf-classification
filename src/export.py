"""Один ZIP со всеми результатами, кэшами и снимком кода для выгрузки с Kaggle."""

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from tqdm.auto import tqdm


def export_results(output_root: Path, archive: Path) -> Path:
    """Упаковать каталог целиком потоково, без окружения и исходных сигналов."""
    root = output_root.resolve(strict=True)
    target = archive.resolve()
    if target.is_relative_to(root):
        raise ValueError("Archive must be outside output_root")
    if (root / ".linear.lock").exists():
        raise RuntimeError("Training lock exists; stop training before exporting")
    if target.exists():
        raise FileExistsError(target)
    files = sorted(p for p in root.rglob("*") if p.is_file())
    if any(p.is_symlink() or not p.resolve().is_relative_to(root) for p in files):
        raise ValueError("Output contains links outside the results directory")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    temporary.touch(exist_ok=False)
    try:
        with ZipFile(
            temporary, "w", compression=ZIP_DEFLATED, compresslevel=1
        ) as bundle:
            for path in tqdm(files, desc="Архив результатов", unit="файл"):
                bundle.write(path, Path(root.name) / path.relative_to(root))
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
    args = parser.parse_args()
    export_results(args.output_root, args.archive)


if __name__ == "__main__":
    main()
