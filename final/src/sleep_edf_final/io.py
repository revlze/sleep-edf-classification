"""Checksummed input/output. No label or training-file access in final inference."""

import csv
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np

FINAL_ROOT = Path(__file__).resolve().parents[2]
CHANNELS = ("EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def save_npz(path: Path, **values) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as f:
        temporary = Path(f.name)
    try:
        np.savez_compressed(temporary, **values)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_record(path: Path, *, signals: bool = True) -> dict:
    with np.load(path, allow_pickle=False) as z:
        if set(z.files) != {"x", "ids", "epoch_index", "fs", "channel_names"}:
            raise ValueError(f"Unexpected raw NPZ keys: {path}")
        record = {k: z[k] for k in z.files if signals or k != "x"}
    ids, indices = record["ids"], record["epoch_index"]
    if (
        ids.ndim != 1
        or indices.shape != ids.shape
        or not np.issubdtype(indices.dtype, np.integer)
        or len(ids) == 0
        or len(set(ids.tolist())) != len(ids)
        or np.any(np.diff(indices) <= 0)
    ):
        raise ValueError(f"Invalid record IDs/epoch order: {path}")
    if int(record["fs"]) != 100 or tuple(record["channel_names"].tolist()) != CHANNELS:
        raise ValueError("Unexpected sampling rate/channel order")
    if signals and (
        record["x"].dtype != np.float32
        or record["x"].shape != (len(ids), 3000, 3)
        or not np.isfinite(record["x"]).all()
    ):
        raise ValueError(f"Invalid raw signals: {path}")
    return record


def input_records(root: Path) -> list[Path]:
    contract = read_json(FINAL_ROOT / "data_contract.json")
    if (
        sha256(root / "MANIFEST.json") != contract["manifest_sha256"]
        or sha256(root / "sample_submission.csv") != contract["sample_sha256"]
    ):
        raise ValueError("Input is not the fixed full_psg_v1 edition")
    paths = sorted((root / "epochs/test").glob("*.npz"))
    if {p.relative_to(root).as_posix() for p in paths} != set(contract["files"]):
        raise ValueError("Missing/extra test records")
    return paths


def verify_record(root: Path, path: Path) -> str:
    expected = read_json(FINAL_ROOT / "data_contract.json")["files"][
        path.relative_to(root).as_posix()
    ]
    if sha256(path) != expected:
        raise ValueError(f"Raw input checksum mismatch: {path}")
    return expected


def labels_csv(path: Path) -> dict[str, int]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["id", "label"]:
            raise ValueError(f"Wrong submission schema: {path}")
        rows = list(reader)
    result = {r["id"]: int(r["label"]) for r in rows}
    if len(result) != len(rows) or any(v not in range(5) for v in result.values()):
        raise ValueError("Duplicate submission IDs/invalid classes")
    return result


def write_submission(
    path: Path, ids, probabilities, sample: Path, *, smoke=False
) -> None:
    if probabilities.shape != (len(ids), 5) or not np.isfinite(probabilities).all():
        raise ValueError("Invalid final probabilities")
    predictions = dict(zip(ids.tolist(), probabilities.argmax(1).tolist(), strict=True))
    order = list(labels_csv(sample))  # Template labels are not used.
    if smoke:
        order = [i for i in order if i in predictions]
    if set(order) != set(predictions) or len(ids) != len(predictions):
        raise ValueError("Submission IDs do not match sample")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "label"])
        writer.writerows((i, predictions[i]) for i in order)
