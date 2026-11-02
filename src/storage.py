"""Блокировки ресурсов и атомарная публикация файлов на Windows/macOS/Linux."""

import json
import os
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np

if os.name == "nt":
    import portalocker
else:
    import fcntl


@contextmanager
def file_lock(
    path: Path, *, wait: bool = False, shared: bool = False
) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        if os.name == "nt":
            operation = portalocker.LOCK_SH if shared else portalocker.LOCK_EX
            try:
                portalocker.lock(
                    stream, operation | (0 if wait else portalocker.LOCK_NB)
                )
            except portalocker.exceptions.LockException as error:
                raise RuntimeError(f"Ресурс занят другим процессом: {path}") from error
        else:
            operation = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
            try:
                fcntl.flock(stream, operation | (0 if wait else fcntl.LOCK_NB))
            except BlockingIOError as error:
                raise RuntimeError(f"Ресурс занят другим процессом: {path}") from error
        try:
            yield
        finally:
            if os.name == "nt":
                portalocker.unlock(stream)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)
    # Не удаляем файл: иначе другой процесс может заблокировать другой inode.


@contextmanager
def atomic_path(target: Path) -> Iterator[Path]:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=f".tmp{target.suffix}", dir=target.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        yield temporary
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def load_or_create_npz(
    target: Path, create: Callable[[], dict[str, np.ndarray]]
) -> tuple[dict[str, np.ndarray], bool]:
    def read() -> dict[str, np.ndarray]:
        with np.load(target, allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}

    if target.exists():
        return read(), True
    with file_lock(target.with_suffix(".npz.lock"), wait=True):
        if target.exists():
            return read(), True
        values = create()
        with atomic_path(target) as temporary:
            np.savez_compressed(temporary, **values)
        return values, False


def canonical_fingerprint(value: str, kind: str) -> str:
    """Только явно проверенные пары версий с одинаковыми вычислениями."""
    aliases = json.loads(
        Path(__file__)
        .with_name("source_compatibility.json")
        .read_text(encoding="utf-8")
    )
    return aliases[kind].get(value, value)
