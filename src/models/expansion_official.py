"""Original published model definitions, strict checkpoints, explicit context recipes."""

import ast
import contextlib
import importlib
import io
import math
import sys
import tarfile
import types
from pathlib import Path

import numpy as np
import torch
from scipy.signal import get_window, spectrogram
from torch import nn
from torch.nn import functional as F

from src.expansion_assets import asset, package, provenance
from src.models.tinysleepnet import tiny_preprocess


def definitions(path: Path, namespace: dict) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    selected = []
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            selected.append(node)
        elif isinstance(node, ast.Assign):
            try:
                ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
            selected.append(node)
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"),
        namespace,
    )


def numpy_namespace() -> dict:
    return {"torch": torch, "nn": nn, "F": F, "np": np, "math": math}


def coverage_windows(n: int, length: int, stride: int) -> list[tuple[int, int]]:
    if min(n, length, stride) < 1 or stride > length:
        raise ValueError("Invalid context window")
    if n <= length:
        return [(0, n)]
    starts = list(range(0, n - length + 1, stride))
    if starts[-1] != n - length:
        starts.append(n - length)
    return [(a, a + length) for a in starts]


def spectral_preprocess(x: np.ndarray) -> np.ndarray:
    """PhysioEx pinned seqsleepnet preset, per-epoch 29x129 power-dB spectra."""
    epochs = tiny_preprocess(x, 0)
    _, _, power = spectrogram(
        epochs.astype(np.float64),
        fs=100,
        window=get_window("hamming", 200),
        nperseg=200,
        noverlap=100,
        nfft=256,
        axis=-1,
    )
    return (
        (10 * np.log10(power + np.finfo(np.float32).eps))
        .transpose(0, 2, 1)
        .astype(np.float32)[:, None]
    )


class SpectralAdapter:
    def __init__(self, name: str, cache: Path, inputs: Path | None, device: str):
        import json

        names = {
            "seqsleepnet": "seqsleepnet-phan",
            "lseqsleepnet": "lseqsleepnet-phan",
            "sleeptransformer": "sleeptransformer-phan",
        }
        if name not in names:
            raise ValueError(name)
        basename = names[name]
        self.config = json.loads(
            asset(basename + "-config.json", cache, inputs).read_text()
        )
        source = package("physioex", cache, inputs)
        namespace = numpy_namespace()
        if name == "lseqsleepnet":
            definitions(source / "physioex/models/seqsleepnet.py", namespace)
        definitions(source / f"physioex/models/{name}.py", namespace)
        cls = self.config["model_class"].split(":")[-1]
        self.model = namespace[cls](**self.config["model_kwargs"])
        state = torch.load(
            asset(basename + ".pt", cache, inputs),
            map_location="cpu",
            weights_only=True,
        )
        self.model.load_state_dict(state, strict=True)
        self.device = torch.device(device)
        self.model.eval().to(self.device)
        self.length = self.config["training"]["sequence_length"]
        self.provenance = provenance(
            [basename + ".pt", basename + "-config.json"], ["physioex"]
        )

    def probabilities(self, raw: np.ndarray, progress=None) -> np.ndarray:
        spec = spectral_preprocess(raw)
        total = np.zeros((len(raw), 5), dtype=np.float64)
        count = np.zeros(len(raw), dtype=np.int64)
        with torch.inference_mode():
            for left, right in coverage_windows(
                len(raw), self.length, max(1, self.length // 2)
            ):
                x = (
                    torch.from_numpy(spec[left:right].copy())
                    .unsqueeze(0)
                    .to(self.device)
                )
                logits = self.model(x)
                if logits.shape != (1, right - left, 5):
                    raise ValueError("Spectral model epoch/class axis differs")
                total[left:right] += logits[0].float().softmax(-1).cpu().numpy()
                count[left:right] += 1
                if progress is not None:
                    progress.update(
                        detail=f"spectral context {right}/{len(raw)} epochs"
                    )
        if np.any(count == 0):
            raise ValueError("Uncovered epoch")
        return (total / count[:, None]).astype(np.float32)


class SleePyCoAdapter:
    def __init__(self, population: str, cache: Path, inputs: Path | None, device: str):
        import json

        if population not in ("edf2018", "shhs"):
            raise ValueError(population)
        source = package("sleepyco", cache, inputs)
        dataset = "Sleep-EDF-2018" if population == "edf2018" else "SHHS"
        config_path = (
            source
            / f"configs/SleePyCo-Transformer_SL-10_numScales-3_{dataset}_freezefinetune.json"
        )
        self.config = json.loads(config_path.read_text())
        namespace = numpy_namespace()
        for relative in (
            "models/sleepyco.py",
            "models/classifiers.py",
            "models/main_model.py",
        ):
            definitions(source / relative, namespace)
        with contextlib.redirect_stdout(io.StringIO()):
            self.model = namespace["MainModel"](self.config)
        checkpoint = f"sleepyco-{population}-fold01.pth"
        state = torch.load(
            asset(checkpoint, cache, inputs), map_location="cpu", weights_only=True
        )
        state = {k.removeprefix("module."): v for k, v in state.items()}
        self.model.load_state_dict(state, strict=True)
        self.device = torch.device(device)
        self.model.eval().to(self.device)
        self.provenance = provenance([checkpoint], ["sleepyco"])

    def probabilities(self, raw: np.ndarray, progress=None) -> np.ndarray:
        # Official EDF preprocessing reads physical µV directly, no filtering/scaling.
        # Target is the last epoch of each past-10 window. At a segment start use
        # native variable-length prefixes (1..9); never invent missing EEG epochs.
        result = np.empty((len(raw), 5), dtype=np.float32)
        with torch.inference_mode():
            groups = [[end] for end in range(min(9, len(raw)))]
            groups += [
                list(range(start, min(len(raw), start + 4)))
                for start in range(9, len(raw), 4)
            ]
            for ends in groups:
                signal = np.stack(
                    [
                        raw[max(0, end - 9) : end + 1, :, 0].reshape(1, -1)
                        for end in ends
                    ]
                ).copy()
                outputs = self.model(torch.from_numpy(signal).to(self.device))
                logits = torch.stack(outputs).sum(
                    0
                )  # Author test_custom: sum scale logits.
                if logits.shape != (len(ends), 5):
                    raise ValueError("SleePyCo classifier output differs")
                result[ends] = logits.float().softmax(-1).cpu().numpy()
                if progress is not None:
                    progress.update(detail=f"SleePyCo epoch {ends[-1] + 1}/{len(raw)}")
        return result


class RobustAdapter:
    def __init__(self, cache: Path, inputs: Path | None, device: str):
        import json

        source = package("robust", cache, inputs)
        prefix = "_sleepedf_official_robust"
        if prefix not in sys.modules:
            module = types.ModuleType(prefix)
            module.__path__ = [str(source / "robust_sleep_net")]
            sys.modules[prefix] = module
        net_module = importlib.import_module(prefix + ".models.modulo_net.net")
        self.normalization_module = importlib.import_module(
            prefix + ".models.modulo_net.normalization"
        )
        name = "robust-leave-sleepedf-out.gz"
        with tarfile.open(asset(name, cache, inputs)) as archive:

            def read_json(n):
                return json.load(archive.extractfile(n))

            self.groups = read_json("groups.json")
            normalizers = read_json("normalization_parameters.json")
            for group in normalizers["signals"]:
                for op in group["normalization"]:
                    for key in ("mu", "sigma"):
                        if key in op["args"]:
                            op["args"][key] = torch.tensor(op["args"][key])
            with contextlib.redirect_stdout(io.StringIO()):
                self.model = net_module.ModuloNet(
                    self.groups,
                    read_json("features.json"),
                    normalizers,
                    read_json("net_params.json"),
                )
            state = torch.load(
                io.BytesIO(archive.extractfile("state.torch").read()),
                map_location="cpu",
                weights_only=True,
            )
            self.model.load_state_dict(state, strict=True)
        self.device = torch.device(device)
        self.model.device = self.device
        self.model.eval().to(self.device)
        self.provenance = provenance([name], ["robust"])

    def probabilities(self, raw: np.ndarray, progress=None) -> np.ndarray:
        from scipy.signal import filtfilt, iirfilter, resample_poly

        # Original inference_on_h5: BP .2-30Hz order2, 60Hz, robust normalization,
        # then 900s zero context at each true segment boundary (author recipe).
        b, a = iirfilter(
            2, [0.2 * 2 / 100, 30 * 2 / 100], btype="bandpass", ftype="butter"
        )
        signal = resample_poly(filtfilt(b, a, raw.reshape(-1, 3), axis=0), 3, 5, axis=0)
        median = np.percentile(signal, 50, axis=0)
        lower, upper = (
            median - np.percentile(signal, 25, axis=0),
            np.percentile(signal, 75, axis=0) - median,
        )
        signal = np.clip(signal, -40 * lower, 40 * upper)
        scale = np.percentile(signal, 75, axis=0) - np.percentile(signal, 25, axis=0)
        signal = (signal - np.median(signal, axis=0)) / np.where(
            scale == 0, 1e-5, scale
        )
        epochs = np.pad(
            signal.astype(np.float32).reshape(len(raw), 1800, 3),
            ((30, 30), (0, 0), (0, 0)),
        )
        total = np.zeros((len(epochs), 5), dtype=np.float64)
        count = np.zeros(len(epochs), dtype=np.int64)
        with torch.inference_mode():
            for start in range(0, len(epochs) - 20, 8):
                starts = range(start, min(len(epochs) - 20, start + 8))
                x = torch.from_numpy(
                    np.stack(
                        [epochs[left : left + 21].transpose(0, 2, 1) for left in starts]
                    ).copy()
                ).to(self.device)
                output = self.model({"signals": {"eeg": x}, "features": {}})[0]
                p = output.reshape(len(starts), 21, 5).float().softmax(-1).cpu().numpy()
                for row, left in enumerate(starts):
                    total[left : left + 21] += p[row]
                    count[left : left + 21] += 1
                if progress is not None:
                    progress.update(
                        detail=f"RobustSleepNet context {starts[-1] + 21}/{len(epochs)}"
                    )
        return (total[30:-30] / count[30:-30, None]).astype(np.float32)
