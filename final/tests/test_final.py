import ast
import json

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from sleep_edf_final.heads import ResidualHead, WindowBank
from sleep_edf_final.io import FINAL_ROOT, labels_csv, write_submission
from sleep_edf_final.models import restore_tiny, tiny_model, verify_weights


def test_selected_states_restore_strictly():
    manifest = verify_weights()
    assert len(manifest["weights"]) == 9
    assert all(row["bytes"] < 50 * 2**20 for row in manifest["weights"])
    original = tiny_model()
    import copy

    classifier = restore_tiny(copy.deepcopy(original), 82)
    changed = [
        k
        for k, v in original.state_dict().items()
        if not torch.equal(v, classifier.state_dict()[k])
    ]
    assert set(changed) == {"clf.clf.weight", "clf.clf.bias"}
    full = restore_tiny(copy.deepcopy(original), 85)
    assert not torch.equal(
        full.feature_extractor.cnn[0][1].weight,
        original.feature_extractor.cnn[0][1].weight,
    )
    cfg = OmegaConf.to_container(
        OmegaConf.load(FINAL_ROOT / "configs/113_pretrained.yaml"), resolve=True
    )
    for number, size in [(102, 136), (106, 164)]:
        state = torch.load(
            FINAL_ROOT / f"weights/{number:03}_best.pt",
            map_location="cpu",
            weights_only=True,
        )
        model = ResidualHead(size, cfg["heads"][str(number)])
        model.load_state_dict(state["model"], strict=True)
        assert len(state["mean"]) == size and np.isfinite(state["std"].numpy()).all()


def test_sample_id_order_and_missing_ids(tmp_path):
    sample = tmp_path / "sample.csv"
    sample.write_text("id,label\nb,0\na,0\n")
    path = tmp_path / "result.csv"
    write_submission(
        path, np.array(["a", "b"]), np.eye(5, dtype=np.float32)[[2, 4]], sample
    )
    assert list(labels_csv(path).items()) == [("b", 4), ("a", 2)]
    with pytest.raises(ValueError):
        write_submission(path, np.array(["a"]), np.eye(5)[[0]], sample)


def test_context_stops_at_record_and_gap():
    meta = {
        "ids": np.arange(7).astype(str),
        "record_id": np.array(["a"] * 4 + ["b"] * 3),
        "epoch_index": np.array([0, 1, 5, 6, 0, 1, 2]),
    }
    bank = WindowBank(
        np.ones((7, 1), np.float32), np.ones((7, 3, 5), np.float32) / 5, meta, "cpu"
    )
    x, _, _, mask, _, lengths = bank.batch(np.arange(7), 20)
    assert lengths.tolist() == [2, 2, 2, 2, 3, 3, 3]
    assert not x[~mask].any()


def test_published_notebook_has_outputs_and_default_primary_only():
    nb = json.loads(
        (FINAL_ROOT / "notebooks/06_final.ipynb").read_text(encoding="utf-8")
    )
    settings = {}
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        assert cell["execution_count"] is not None
        assert not any(o.get("output_type") == "error" for o in cell["outputs"])
        for node in ast.parse("".join(cell["source"])).body:
            if (
                isinstance(node, ast.Assign)
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "OUTPUT_SECOND_SUBMISSION"
            ):
                settings["second"] = ast.literal_eval(node.value)
    assert settings["second"] is False


def test_false_flag_excludes_old_secondary_outputs_from_zip(tmp_path):
    import importlib.util
    from zipfile import ZipFile

    spec = importlib.util.spec_from_file_location(
        "final_package", FINAL_ROOT / "package.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / "outputs"
    source.mkdir()
    (source / "run_report.json").write_text(
        json.dumps({"output_second_submission": False})
    )
    for name in ["submission.csv", "submission_2.csv", "113_probabilities.npz"]:
        (source / name).write_text("previous output")
    target = tmp_path / "results.zip"
    module.results(source, target)
    with ZipFile(target) as z:
        assert "submission.csv" in z.namelist()
        assert (
            "submission_2.csv" not in z.namelist()
            and "113_probabilities.npz" not in z.namelist()
        )
