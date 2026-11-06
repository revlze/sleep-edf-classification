"""Recover index/publication after a bookkeeping failure; never run a model."""

import argparse
import json
import shutil
from pathlib import Path

from src.journal import initialize_journal
from src.models.pretrained_assets import sha256
from src.pretrained import ROOT
from src.series_results import journal
from src.storage import atomic_path


def recover_results(output: Path, project: Path = ROOT) -> list[str]:
    initialize_journal(output / "experiments.csv", project / "experiments.csv")
    recovered = []
    for folder in sorted(output.glob("[0-9][0-9][0-9]_*")):
        metrics_path, provenance_path = (
            folder / "metrics.json",
            folder / "provenance.json",
        )
        submission = folder / "submission.csv"
        if not all(p.is_file() for p in (metrics_path, provenance_path, submission)):
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        config = provenance["config"]
        if metrics.get("smoke") or provenance.get("smoke"):
            continue
        if (
            config["experiment_id"] != folder.name
            or metrics["contract"] != provenance["contract"]
        ):
            raise ValueError(f"Saved result identity differs: {folder.name}")
        journal(
            project / "configs" / f"{folder.name}.yaml", config, metrics, folder, output
        )
        destination = output / "submissions" / f"{folder.name}.csv"
        if destination.exists():
            if sha256(destination) != sha256(submission):
                raise ValueError(
                    f"Existing submission differs; preserved: {destination}"
                )
        else:
            with atomic_path(destination) as temporary:
                shutil.copyfile(submission, temporary)
        error_path = folder / "last_error.json"
        if error_path.exists():
            error = json.loads(error_path.read_text(encoding="utf-8")).get("error", "")
            if "experiments.csv" in error and "No such file or directory" in error:
                error_path.unlink()
        recovered.append(folder.name)
    for report_path in output.glob("*failures*.json"):
        entries = json.loads(report_path.read_text(encoding="utf-8"))
        if not isinstance(entries, list):
            continue
        remaining = [
            entry
            for entry in entries
            if not (
                entry.get("experiment_id") in recovered
                and "experiments.csv" in entry.get("error", "")
                and "No such file or directory" in entry.get("error", "")
            )
        ]
        if len(remaining) != len(entries):
            with atomic_path(report_path) as temporary:
                temporary.write_text(
                    json.dumps(remaining, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
    return recovered


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    recovered = recover_results(args.output_root)
    print(
        f"Журнал восстановлен: {len(recovered)} завершённых конфигураций; обучение и inference не запускались."
    )


if __name__ == "__main__":
    main()
