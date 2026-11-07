"""Aligned per-record probabilities, manifest verification and conservative cache reuse."""

import hashlib
import io
import json
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import numpy as np
import polars as pl

from src.data import align_labels, load_labels, load_record
from src.evaluation import validate_probabilities
from src.models.pretrained_assets import sha256
from src.pretrained import save_json, save_npz


def fingerprint(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class SeriesData:
    def __init__(self, root: Path, smoke: bool = False, smoke_epochs: int = 48):
        if smoke_epochs <= 0:
            raise ValueError("Smoke epoch limit must be positive")
        self.root, self.smoke, self.smoke_epochs = root.resolve(), smoke, smoke_epochs
        self.labels = load_labels(self.root)
        manifest_path = self.root / "MANIFEST.json"
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest["dataset_revision"] != "full_psg_v1":
            raise ValueError("Expected full_psg_v1")
        self.hashes = {f["path"]: f["sha256"] for f in self.manifest["files"]}
        self.labels_hash = sha256(self.root / "labels.csv")
        if self.labels_hash != self.hashes["labels.csv"]:
            raise ValueError("Labels differ from manifest")
        self.sample = pl.read_csv(self.root / "sample_submission.csv")
        if self.sample.columns != ["id", "label"] or self.sample[
            "id"
        ].n_unique() != len(self.sample):
            raise ValueError("Invalid sample_submission schema/IDs")
        self.sample_hash = sha256(self.root / "sample_submission.csv")
        if self.sample_hash != self.hashes["sample_submission.csv"]:
            raise ValueError("sample_submission differs from manifest")
        self.identity = {
            "version": "full_psg_v1",
            "manifest_sha256": sha256(manifest_path),
            "labels_sha256": self.labels_hash,
            "sample_sha256": self.sample_hash,
        }
        self._tables, self._headers, self._verified = {}, {}, set()

    def paths(self, split: str) -> list[Path]:
        if split not in ("train", "val", "test"):
            raise ValueError(split)
        paths = sorted((self.root / "epochs" / split).glob("*.npz"))
        if not paths:
            raise FileNotFoundError(f"No {split} records")
        if self.smoke and split == "train":
            subject = {
                r["record_id"]: r["subject_id"]
                for r in self.labels.select("record_id", "subject_id")
                .unique()
                .to_dicts()
            }
            selected, seen = [], set()
            for path in paths:
                if subject[path.stem] not in seen:
                    selected.append(path)
                    seen.add(subject[path.stem])
                if len(selected) == 6:
                    break
            return selected
        return paths[:1] if self.smoke else paths

    def header(self, path: Path) -> dict:
        if path not in self._headers:
            self._headers[path] = load_record(path, signals=False)
        return self._headers[path]

    def verify(self, path: Path) -> str:
        expected = self.hashes[path.relative_to(self.root).as_posix()]
        if path not in self._verified:
            if sha256(path) != expected:
                raise ValueError(f"NPZ differs from manifest: {path.name}")
            self._verified.add(path)
        return expected

    def record_metadata(self, path: Path, split: str) -> dict:
        record = self.header(path)
        n = (
            min(len(record["ids"]), self.smoke_epochs)
            if self.smoke
            else len(record["ids"])
        )
        values = {
            "ids": np.asarray(record["ids"][:n], dtype=str),
            "record_id": np.full(n, path.stem),
            "epoch_index": record["epoch_index"][:n],
        }
        if split != "test":
            values["labels"] = align_labels(record, self.labels, path.stem, split)[:n]
            row = self.labels.filter(pl.col("record_id") == path.stem)
            values["subject_id"] = np.full(n, str(row["subject_id"][0]))
        else:
            # No inferred original identity and no test labels, including placeholders.
            values["subject_id"] = np.full(n, "")
        return values

    def metadata(self, split: str) -> dict:
        if split not in self._tables:
            paths = self.paths(split)
            rows = [self.record_metadata(path, split) for path in paths]
            result = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
            if len(set(result["ids"].tolist())) != len(result["ids"]):
                raise ValueError("Duplicate epoch IDs")
            if not self.smoke:
                expected = (
                    set(self.sample["id"])
                    if split == "test"
                    else set(self.labels.filter(pl.col("split") == split)["id"])
                )
                if set(result["ids"].tolist()) != expected:
                    raise ValueError(f"Missing/extra {split} epochs")
            self._tables[split] = result
        return self._tables[split]


def align_prediction(values: dict, metadata: dict) -> dict:
    ids = np.asarray(values["ids"], dtype=str)
    if len(set(ids.tolist())) != len(ids):
        raise ValueError("Duplicate saved prediction IDs")
    validate_probabilities(values["probabilities"], len(ids))
    order = {name: i for i, name in enumerate(ids.tolist())}
    wanted = metadata["ids"].tolist()
    if set(wanted) != set(ids.tolist()):
        raise ValueError("Prediction IDs do not match requested split exactly")
    idx = np.asarray([order[name] for name in wanted])
    result = {**metadata, "probabilities": values["probabilities"][idx]}
    if "predicted_labels" in values:
        result["predicted_labels"] = values["predicted_labels"][idx]
    if "hard_probabilities" in values:
        validate_probabilities(values["hard_probabilities"], len(ids))
        result["hard_probabilities"] = values["hard_probabilities"][idx]
    if (
        "labels" in values
        and "labels" in metadata
        and not np.array_equal(values["labels"][idx], metadata["labels"])
    ):
        raise ValueError("Prediction target labels differ")
    return result


def load_prediction(path: Path, metadata: dict) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        values = {k: archive[k] for k in archive.files}
    return align_prediction(values, metadata)


def read_legacy_019(
    inputs: list[Path], data: SeriesData, split: str
) -> tuple[dict, dict] | None:
    """Verify archived baseline before reuse. Never import executable code from Input."""
    if split != "val":
        return None
    metadata = data.metadata(split)
    candidates = []
    for root in inputs:
        if root.is_file() and root.suffix == ".zip":
            with ZipFile(root) as archive:
                paths = [
                    n
                    for n in archive.namelist()
                    if n.endswith("019_pretrained/val_predictions.npz")
                    and "smoke" not in PurePosixPath(n).parts
                ]
                for path in paths:
                    prefix = path.removesuffix("val_predictions.npz")
                    candidates.append(
                        (
                            archive.read(path),
                            json.loads(archive.read(prefix + "provenance.json")),
                            archive.read(prefix + "resolved.yaml"),
                            f"{root}!{path}",
                        )
                    )
        elif root.is_dir():
            found = list(root.glob("019_pretrained/val_predictions.npz"))
            if (
                root.name == "019_pretrained"
                and (root / "val_predictions.npz").exists()
            ):
                found.append(root / "val_predictions.npz")
            for path in found:
                candidates.append(
                    (
                        path.read_bytes(),
                        json.loads(path.with_name("provenance.json").read_text()),
                        path.with_name("resolved.yaml").read_bytes(),
                        str(path),
                    )
                )
    if not candidates:
        return None
    from omegaconf import OmegaConf

    from src.models.pretrained_assets import ASSETS

    seen = set()
    accepted = []
    for content, provenance, resolved, location in candidates:
        digest = hashlib.sha256(content).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        cfg = OmegaConf.to_container(OmegaConf.create(resolved.decode()), resolve=True)
        if cfg.get("experiment_id") != "019_pretrained" or cfg["runtime"]["smoke"]:
            raise ValueError("Only complete production 019 may be reused")
        if (
            cfg["model"]["checkpoint"] != [f"anysleep-run{i}.pth" for i in (1, 2, 3)]
            or cfg["model"]["class_order"] != ["W", "N1", "N2", "N3", "REM"]
            or cfg["inference"]["core_epochs"] != 128
            or cfg["inference"]["halo_epochs"] != 64
            or cfg["preprocessing"]["normalization"]
            != "clip_plusminus_20IQR_then_RobustScaler"
        ):
            raise ValueError("019 preprocessing/checkpoint recipe differs")
        expected_assets = {
            f"anysleep-run{i}.pth": ASSETS[f"anysleep-run{i}.pth"][1] for i in (1, 2, 3)
        }
        actual = {a["name"]: a["sha256"] for a in provenance["checkpoint"]}
        if (
            provenance["source_sha256"]
            != "28fc485db2e55e6177c74ec8c061de7ff3f7cb26d2ccfcafb590f9fdbb00714e"
        ):
            raise ValueError(
                "019 source is not the audited original inference implementation"
            )
        if (
            any(actual.get(k) != v for k, v in expected_assets.items())
            or provenance["labels_sha256"] != data.labels_hash
            or provenance["manifest_sha256"] != data.identity["manifest_sha256"]
            or provenance["predictions_sha256"] != digest
        ):
            raise ValueError("019 provenance/checksum mismatch")
        with np.load(io.BytesIO(content), allow_pickle=False) as stored:
            # Historical object subject_id is not deserialized; metadata is rebuilt by ID.
            values = {k: stored[k] for k in ("ids", "labels", "probabilities")}
        if data.smoke:
            idx = np.flatnonzero(np.isin(values["ids"], metadata["ids"]))
            values = {k: v[idx] for k, v in values.items()}
        accepted.append(
            (
                align_prediction(values, metadata),
                {
                    "legacy_019": location,
                    "sha256": digest,
                    "source_sha256": provenance["source_sha256"],
                    "git": provenance["git"],
                },
            )
        )
    first = accepted[0]
    if any(
        not np.array_equal(first[0]["probabilities"], item[0]["probabilities"])
        for item in accepted[1:]
    ):
        raise ValueError(
            "Different 019 probability runs found; select one Input explicitly"
        )
    return first


def submission_frame(
    values: dict, sample: pl.DataFrame, smoke: bool = False
) -> pl.DataFrame:
    validate_probabilities(values["probabilities"], len(values["ids"]))
    ids = values["ids"].tolist()
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate submission IDs")
    expected = (
        sample.filter(pl.col("id").is_in(ids)).select("id")
        if smoke
        else sample.select("id")
    )
    if set(ids) != set(expected["id"]):
        raise ValueError("Submission ID set differs from sample")
    predicted = np.asarray(
        values.get("predicted_labels", values["probabilities"].argmax(1))
    )
    if (
        predicted.shape != (len(ids),)
        or not np.issubdtype(predicted.dtype, np.integer)
        or np.any((predicted < 0) | (predicted > 4))
    ):
        raise ValueError("Submission labels must be integers 0..4")
    return expected.join(
        pl.DataFrame({"id": values["ids"], "label": predicted.astype(np.int64)}),
        on="id",
        how="left",
        maintain_order="left",
        validate="1:1",
    )


def save_prediction(path: Path, values: dict, provenance: dict) -> None:
    validate_probabilities(values["probabilities"], len(values["ids"]))
    if any(np.asarray(v).dtype.hasobject for v in values.values()):
        raise ValueError("Prediction NPZ must not require pickle")
    save_npz(path, **values)
    save_json(
        path.with_suffix(".json"), {**provenance, "prediction_sha256": sha256(path)}
    )
