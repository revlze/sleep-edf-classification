"""Фиксированный инференс checkpoint и submission в точном порядке sample CSV."""

import argparse
import hashlib
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from omegaconf import OmegaConf

from src.data import load_labels
from src.evaluation import validate_probabilities
from src.experiment import PROJECT_ROOT, exclusive_file, source_fingerprint
from src.features import build_feature_table


def assemble_submission(ids, probabilities, sample: pl.DataFrame) -> pl.DataFrame:
    ids = np.asarray(ids)
    if len(ids) != len(set(ids.tolist())) or sample["id"].n_unique() != sample.height:
        raise ValueError("Duplicate prediction/sample IDs")
    if set(ids.tolist()) != set(sample["id"].to_list()):
        raise ValueError("Prediction IDs must exactly match sample")
    # Validate distributions without using any target labels.
    validate_probabilities(probabilities, len(ids))
    predicted = pl.DataFrame(
        {"id": ids, "label": probabilities.argmax(axis=1).astype(np.int64)}
    )
    return sample.select("id").join(
        predicted, on="id", how="left", maintain_order="left", validate="1:1"
    )


def predict_submission(
    checkpoint: str | Path,
    *,
    data_root: str | Path | None = None,
    output: str | Path | None = None,
) -> Path:
    checkpoint = Path(checkpoint).resolve()
    config = OmegaConf.to_container(
        OmegaConf.load(checkpoint.parent / "resolved.yaml"), resolve=True
    )
    if config["runtime"]["smoke"]:
        raise ValueError("Smoke models cannot produce production submissions")
    if source_fingerprint() != config["runtime"]["source_fingerprint"]:
        raise ValueError("Source/dependency fingerprint differs from checkpoint")
    bundle = joblib.load(checkpoint)
    root = Path(data_root or config["paths"]["data_root"]).resolve()
    cache = Path(config["paths"]["cache_root"])
    target = Path(
        output or PROJECT_ROOT / "submissions" / f"{config['experiment_id']}.csv"
    ).resolve()
    if target.exists():
        raise FileExistsError(target)
    with exclusive_file(Path(config["paths"]["output_root"]) / ".linear.lock"):
        x, metadata, names, definition = build_feature_table(
            root,
            cache,
            load_labels(root),
            bundle["features"]["extractor"],
            bundle["features"]["context"],
            test=True,
        )
        import json

        expected = json.loads(
            (checkpoint.parent / "feature_schema.json").read_text(encoding="utf-8")
        )
        if (
            names != bundle["feature_names"]
            or definition["key"] != expected["cache"]["key"]
        ):
            raise ValueError("Feature extractor/data recipe differs from checkpoint")
        probabilities = bundle["pipeline"].predict_proba(x).astype(np.float32)
        sample = pl.read_csv(root / "sample_submission.csv")
        if sample.columns != ["id", "label"]:
            raise ValueError("Unexpected sample schema")
        submission = assemble_submission(
            metadata["id"].to_numpy(), probabilities, sample
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        submission.write_csv(target)
        with checkpoint.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        OmegaConf.save(
            OmegaConf.create(
                {
                    "experiment_id": config["experiment_id"],
                    "regime": bundle["regime"],
                    "C": bundle["C"],
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": digest,
                    "feature_key": definition["key"],
                    "data_revision": config["data_revision"],
                    "csv": str(target),
                }
            ),
            target.with_suffix(".meta.yaml"),
        )
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--data-root")
    parser.add_argument("--output")
    args = parser.parse_args()
    print(
        predict_submission(
            args.checkpoint, data_root=args.data_root, output=args.output
        )
    )


if __name__ == "__main__":
    main()
