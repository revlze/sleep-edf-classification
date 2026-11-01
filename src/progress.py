"""Один индикатор этапа; обновление не чаще одного раза в семь секунд."""

import json
import os
import sys
from threading import Event, Lock, Thread
from time import monotonic

from tqdm import tqdm

PREFIX = "__SLEEP_PROGRESS__"
INTERVAL = 7.0


class Progress:
    def __init__(self, description: str, total: int, unit: str):
        self.description = description
        self.total = total
        self.unit = unit
        self.n = 0
        self.detail = ""
        self.started = monotonic()
        self._lock = Lock()
        self._stop = Event()
        self._width = 0

    def update(self, n: int = 0, *, detail: str | None = None) -> None:
        with self._lock:
            self.n += n
            if detail is not None:
                self.detail = detail

    def add_total(self, n: int) -> None:
        with self._lock:
            self.total += n

    def _render(self, status: str = "running") -> None:
        with self._lock:
            state = {
                "description": self.description,
                "n": self.n,
                "total": self.total,
                "unit": self.unit,
                "elapsed": monotonic() - self.started,
                "detail": self.detail,
                "status": status,
            }
        if os.environ.get("SLEEP_PROGRESS_JSON") == "1":
            print(PREFIX + json.dumps(state, ensure_ascii=False), flush=True)
        else:
            text = (
                tqdm.format_meter(
                    self.n,
                    self.total,
                    state["elapsed"],
                    prefix=self.description,
                    unit=self.unit,
                    ascii=True,
                )
                + f" | {self.detail}"
            )
            sys.stderr.write("\r" + text.ljust(self._width))
            self._width = len(text)
            if status != "running":
                sys.stderr.write("\n")
            sys.stderr.flush()

    def __enter__(self):
        self._render()

        def refresh():
            while not self._stop.wait(INTERVAL):
                self._render()

        self._worker = Thread(target=refresh, daemon=True)
        self._worker.start()
        return self

    def __exit__(self, kind, value, traceback):
        self._stop.set()
        self._worker.join()
        self._render("failed" if kind else "completed")
