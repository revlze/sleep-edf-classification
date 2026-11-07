"""Pinned frozen backbones. Preserve research batches, floating point and rolling."""

import ast
import copy
import hashlib

import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt
from scipy.special import softmax
from torch import nn

from .io import FINAL_ROOT, read_json, sha256
from .preprocessing import contiguous_segments, epoch_blocks, preprocess_segment
from .vendor.anysleep_no_hydra import AnySleep
from .vendor.usleep_rbd import USleep


def verify_weights():
    manifest = read_json(FINAL_ROOT / "weights/manifest.json")
    for row in manifest["weights"]:
        p = FINAL_ROOT / "weights" / row["file"]
        if sha256(p) != row["sha256"]:
            raise ValueError(f"Weight checksum mismatch: {p}")
    for filename, expected in manifest["vendor_files"].items():
        source = (
            (FINAL_ROOT / "src/sleep_edf_final/vendor" / filename)
            .read_bytes()
            .replace(b"\r\n", b"\n")
        )
        if hashlib.sha256(source).hexdigest() != expected:
            raise ValueError(f"Vendored source checksum mismatch: {filename}")
    return manifest


def tiny_model():
    path = FINAL_ROOT / "src/sleep_edf_final/vendor/tinysleepnet_source.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {"FeatureExtractor", "Classifier", "TinySleepNet"}
    nodes = [
        n
        for n in tree.body
        if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names
    ]
    if {n.name for n in nodes} != names:
        raise ValueError("Tiny source definitions differ")
    namespace = {"torch": torch, "nn": nn}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    cfg = read_json(path.with_name("tinysleepnet-config.json"))
    model = namespace["TinySleepNet"](**cfg["model_kwargs"])
    model.load_state_dict(
        torch.load(
            FINAL_ROOT / "weights/tinysleepnet.pt",
            map_location="cpu",
            weights_only=True,
        ),
        strict=True,
    )
    return model.eval()


def restore_tiny(model, number):
    checkpoint = torch.load(
        FINAL_ROOT / f"weights/{number:03}_best.pt",
        map_location="cpu",
        weights_only=True,
    )
    if number == 82:
        expected = {n for n, _ in model.named_buffers()} | {
            "clf.clf.weight",
            "clf.clf.bias",
        }
        if set(checkpoint["model"]) != expected:
            raise ValueError("082 partial-state contract differs")
        model.load_state_dict(checkpoint["model"], strict=False)
    else:
        model.load_state_dict(checkpoint["model"], strict=True)
    return model.eval()


def usleep_model(device):
    checkpoint = torch.load(
        FINAL_ROOT / "weights/Pretrained_Model.ckpt",
        map_location="cpu",
        weights_only=True,
    )
    hp, state = checkpoint["hyper_parameters"], checkpoint["state_dict"]
    parameters = {
        k: hp[k] for k in ("initial_filters", "complexity_factor", "progression_factor")
    }
    if (
        not hp["include_eog"]
        or len(state) != 265
        or parameters
        != {"initial_filters": 5, "complexity_factor": 1.67, "progression_factor": 2}
        or not all(k.startswith("model.") for k in state)
    ):
        raise ValueError("Unexpected official U-Sleep checkpoint")
    model = USleep(num_channels=2, **parameters)
    model.load_state_dict(
        {k.removeprefix("model."): v for k, v in state.items()}, strict=True
    )
    return model.eval().to(device)


@torch.inference_mode()
def usleep_probabilities(model, signal, device, detail):
    n = len(signal) // 3840
    normalized = signal.reshape(n, 3840, 3).transpose(0, 2, 1)
    pairs = np.empty((n, 2, 5), np.float32)
    for a, b, left, right in epoch_blocks(n, 128, 64, right_alignment=16):
        for pair, eeg in enumerate((0, 1)):
            detail(f"U-Sleep EEG{eeg + 1}/2, эпохи {left}:{right}/{n}, core128/halo64")
            batch = np.ascontiguousarray(
                normalized[a:b][:, [eeg, 2]].transpose(1, 0, 2).reshape(1, 2, -1)
            )
            logits = model(torch.from_numpy(batch).to(device))
            if logits.shape != (1, 5, b - a) or not torch.isfinite(logits).all():
                raise ValueError("U-Sleep temporal/class alignment differs")
            p = logits.softmax(1)[0].T.cpu().numpy()
            pairs[left:right, pair] = p[left - a : right - a]
    return pairs.mean(axis=1)


def tiny_preprocess(x):
    sos = butter(5, [0.3 / 50, 40 / 50], btype="band", output="sos")
    return (
        sosfiltfilt(sos, x[:, :, 0].reshape(-1).astype(np.float64))
        .astype(np.float32)
        .reshape(-1, 3000)
    )


@torch.inference_mode()
def tiny_probabilities(model, signal, device, detail):
    features = []
    for left in range(0, len(signal), 64):
        detail(f"CNN64: {left}/{len(signal)} эпох")
        x = torch.from_numpy(np.array(signal[left : left + 64, None], copy=True)).to(
            device
        )
        features.append(model.feature_extractor(x).float().cpu())
    features = torch.cat(features)
    width = min(20, len(signal))
    p = np.empty((len(signal), 5), np.float32)
    p[:width] = (
        model.clf(features[:width][None].to(device))[0]
        .float()
        .softmax(-1)
        .cpu()
        .numpy()
    )
    for left in range(20, len(signal), 16):
        detail(f"LSTM rolling16: {left}/{len(signal)} эпох")
        ends = range(left, min(len(signal), left + 16))
        windows = torch.stack([features[e - 19 : e + 1] for e in ends]).to(device)
        p[list(ends)] = model.clf(windows)[:, -1].float().softmax(-1).cpu().numpy()
    return p


@torch.inference_mode()
def tiny_embedding(model, signal, device, detail):
    features = []
    for left in range(0, len(signal), 256):
        detail(f"original Tiny embeddings CNN256: {left}/{len(signal)}")
        x = torch.from_numpy(signal[left : left + 256, None].copy()).to(device)
        features.append(model.feature_extractor(x))
    features = torch.cat(features)
    width = min(20, len(signal))
    output = [model.clf.encode(features[:width][None])[0].float().cpu().numpy()]
    for start in range(20, len(signal), 256):
        detail(f"original Tiny embeddings LSTM256: {start}/{len(signal)}")
        ends = torch.arange(start, min(len(signal), start + 256), device=device)
        idx = ends[:, None] + torch.arange(-19, 1, device=device)
        output.append(model.clf.encode(features[idx])[:, -1].float().cpu().numpy())
    return np.concatenate(output)


@torch.inference_mode()
def any_probabilities(models, signal, detail):
    n = len(signal) // 3840
    result = np.empty((n, 5), np.float32)
    for a, b, left, right in epoch_blocks(n, 128, 64, right_alignment=16):
        ps = []
        for run, model in enumerate(models, 1):
            detail(
                f"AnySleep run{run}/3, выход {left}:{right}/{n}, core128/halo64, batch1"
            )
            logits = model(signal[a * 3840 : b * 3840][None])
            if tuple(logits.shape) != (1, b - a, 5):
                raise ValueError("AnySleep output shape")
            ps.append(softmax(logits[0].float().cpu().numpy(), axis=-1))
        result[left:right] = np.mean(ps, axis=0)[left - a : right - a]
    return result


@torch.inference_mode()
def any_embedding(model, signal, detail):
    n = len(signal) // 3840
    groups = {}
    for window in epoch_blocks(n, 128, 64, right_alignment=16):
        groups.setdefault(window[1] - window[0], []).append(window)
    captured = []
    result = np.empty((n, 6), np.float32)
    hook = model.segment_classifier.segment_classifier[2].register_forward_hook(
        lambda _, args, value: captured.append(value)
    )
    try:
        for group in groups.values():
            for left in range(0, len(group), 4):
                batch = group[left : left + 4]
                detail(f"AnySleep run1 embeddings6: batch{len(batch)}, core128/halo64")
                x = torch.stack([signal[a * 3840 : b * 3840] for a, b, _, _ in batch])
                captured.clear()
                model(x)
                if len(captured) != 1:
                    raise ValueError("Embedding hook did not capture one layer")
                z = captured[0].float().cpu().numpy()
                for j, (a, _, start, end) in enumerate(batch):
                    result[start:end] = z[j, :, start - a : end - a].T
    finally:
        hook.remove()
    return result


class FrozenBackbones:
    def __init__(self, device):
        self.device = device
        self.any = [
            AnySleep(path=str(FINAL_ROOT / f"weights/anysleep-run{i}.pth"))
            .eval()
            .to(device)
            for i in (1, 2, 3)
        ]
        self.original = tiny_model()
        self.classifier = restore_tiny(copy.deepcopy(self.original), 82).to(device)
        self.full = restore_tiny(copy.deepcopy(self.original), 85).to(device)
        self.original = self.original.to(device)
        self.usleep = usleep_model(device)
        for model in [
            *self.any,
            self.original,
            self.classifier,
            self.full,
            self.usleep,
        ]:
            model.requires_grad_(False)

    def record(self, raw, indices, detail):
        p19 = []
        p82 = []
        p85 = []
        tiny = []
        anysix = []
        p156 = []
        for si, part in enumerate(contiguous_segments(indices), 1):

            def status(message):
                detail(f"сегмент {si}: " + message)

            eeg = tiny_preprocess(raw[part])
            p82.append(
                tiny_probabilities(
                    self.classifier, eeg, self.device, lambda s: status("082 " + s)
                )
            )
            p85.append(
                tiny_probabilities(
                    self.full, eeg, self.device, lambda s: status("085 " + s)
                )
            )
            tiny.append(tiny_embedding(self.original, eeg, self.device, status))
            prepared = np.ascontiguousarray(preprocess_segment(raw[part], "anysleep"))
            signal = torch.from_numpy(prepared).to(self.device)
            p19.append(any_probabilities(self.any, signal, status))
            anysix.append(any_embedding(self.any[0], signal, status))
            p156.append(
                usleep_probabilities(self.usleep, prepared, self.device, status)
            )
        return {
            "p019": np.concatenate(p19),
            "p082": np.concatenate(p82),
            "p085": np.concatenate(p85),
            "p156": np.concatenate(p156),
            "embeddings": np.concatenate(
                [np.concatenate(tiny), np.concatenate(anysix)], axis=1
            ),
        }
