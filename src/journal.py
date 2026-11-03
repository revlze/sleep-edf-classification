"""Atomic upsert shared by all experiment families; no ML dependencies."""

import csv
from collections.abc import Sequence
from pathlib import Path

from src.storage import atomic_path, file_lock


def upsert_result(path: Path, row: dict, extra_fields: Sequence[str] = ()) -> None:
    with file_lock(path.with_suffix(".csv.lock"), wait=True):
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            fields, rows = list(reader.fieldnames), list(reader)
        for field in [*extra_fields, *row]:
            if field not in fields:
                fields.append(field)
        rows = [r for r in rows if r["experiment_id"] != row["experiment_id"]] + [row]
        with atomic_path(path) as temporary:
            with temporary.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
