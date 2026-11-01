"""Индикация долгих этапов без изменения вычислений модели."""

from contextlib import contextmanager
from threading import Event, Thread

from tqdm.auto import tqdm


@contextmanager
def stage(description: str):
    """Показывать время ожидания; без выдуманного процента или ETA для fit/I/O."""
    stopped = Event()
    with tqdm(
        desc=description,
        total=None,
        bar_format="{desc} | прошло {elapsed} | {postfix}",
        postfix="выполняется; ETA неизвестно",
        leave=False,
    ) as bar:

        def refresh():
            while not stopped.wait(1):
                bar.refresh()

        worker = Thread(target=refresh, daemon=True)
        worker.start()
        try:
            yield
        finally:
            stopped.set()
            worker.join()
