"""Segment-safe native YASA staging and cached, label-free event features."""

import json
from functools import cache
from importlib.metadata import version
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly, welch

from src.expansion_data import selected_signal
from src.models.pretrained_assets import sha256
from src.pretrained import save_json, save_npz
from src.pretrained_signals import contiguous_segments
from src.series_data import fingerprint
from src.storage import atomic_path, file_lock

YASA_MODEL = "clf_eeg+eog_lgb_0.5.0.joblib"
# Official artifact distributed in the pinned PyPI wheel, no custom classifier.
YASA_SHA256 = "86d726a3e6aa6195197ffbc9765f5e194c74358295ef331998c21f8badcebcc3"


@cache
def yasa_classifier() -> Path:
    import yasa

    path = Path(yasa.__file__).parent / "classifiers" / YASA_MODEL
    if version("yasa") != "0.7.0" or sha256(path) != YASA_SHA256:
        raise ValueError("Pinned YASA0.7.0 EEG+EOG classifier checksum differs")
    return path


@cache
def native_classifier():
    import warnings

    import joblib

    with warnings.catch_warnings(record=True) as messages:
        warnings.simplefilter("always")
        classifier = joblib.load(yasa_classifier())
    classifier._sleep_edf_load_warnings = [str(w.message) for w in messages]
    classifier.set_params(n_jobs=1)
    return classifier


def yasa_segment(signal: np.ndarray, channel: int) -> np.ndarray:
    import logging

    import mne
    import yasa

    logging.getLogger("yasa").setLevel(logging.ERROR)

    # MNE physical units are volts. YASA converts Raw to uV internally.
    samples = signal[:, :, [channel, 2]].transpose(2, 0, 1).reshape(2, -1) * 1e-6
    if len(signal) == 1:
        # Pinned YASA sliding_window requires n_samples > window, strictly.
        # One terminal sample (not another epoch) enables the single-epoch case.
        samples = np.concatenate([samples, samples[:, -1:]], axis=1)
    raw = mne.io.RawArray(
        samples,
        mne.create_info(["EEG", "EOG"], 100, ["eeg", "eog"]),
        verbose=False,
    )
    stage = yasa.SleepStaging(raw, eeg_name="EEG", eog_name="EOG", metadata=None)
    with np.errstate(divide="ignore", invalid="ignore"):
        stage.fit()
    if len(signal) == 1:
        # YASA's time_norm=times/times[-1] is 0/0 for a single epoch. Explicit
        # boundary repair to 0 (segment start); no padding or joining segments.
        stage._features.loc[:, "time_norm"] = 0.0
    classifier = native_classifier()
    stage._validate_predict(classifier)
    columns = ["W", "N1", "N2", "N3", "R"]
    if set(classifier.classes_) != set(columns):
        raise ValueError("Unexpected native YASA class order/epoch coverage")
    features = stage.get_features()[classifier.feature_name_]
    output = classifier.predict_proba(features)
    if len(output) != len(signal):
        raise ValueError("Unexpected native YASA epoch coverage")
    return output[:, [list(classifier.classes_).index(c) for c in columns]].astype(
        np.float32
    )


def assign_events(events, epochs: int, kind: str) -> tuple[np.ndarray, list[str]]:
    """Count/mean at peak epoch; occupancy split by exact interval overlap."""
    names = [
        f"{kind}_{v}"
        for v in (
            "count",
            "seconds",
            "occupancy",
            "amplitude",
            "frequency",
            "position_mean",
            "position_std",
        )
    ]
    output = np.zeros((epochs, len(names)), np.float32)
    if events is None or len(events) == 0:
        return output, names
    peak = "Peak" if kind == "spindle" else "NegPeak"
    amplitude = "Amplitude" if kind == "spindle" else "PTP"
    rows = [[] for _ in range(epochs)]
    for _, event in events.iterrows():
        start, end, center = (
            float(event["Start"]),
            float(event["End"]),
            float(event[peak]),
        )
        if (
            not np.isfinite([start, end, center]).all()
            or not 0 <= start <= center <= end <= epochs * 30 + 1e-5
        ):
            raise ValueError("Event timestamps outside the continuous segment")
        for epoch in range(max(0, int(start // 30)), min(epochs, int(end // 30) + 1)):
            output[epoch, 1] += max(
                0, min(end, (epoch + 1) * 30) - max(start, epoch * 30)
            )
        epoch = min(epochs - 1, int(center // 30))
        rows[epoch].append(
            [float(event[amplitude]), float(event["Frequency"]), center % 30]
        )
    for epoch, values in enumerate(rows):
        if values:
            a = np.asarray(values)
            output[epoch, [0, 3, 4, 5, 6]] = [len(a), *a.mean(0), a[:, 2].std()]
    output[:, 2] = np.minimum(output[:, 1] / 30, 1)
    return output, names


def event_segment(signal: np.ndarray) -> tuple[np.ndarray, list[str]]:
    import logging

    import yasa

    logging.getLogger("yasa").setLevel(logging.ERROR)

    chunks, names = [], []
    for channel in (0, 1):
        x = signal[:, :, channel].reshape(-1).astype(np.float64)
        # No hypnogram is supplied: events are detected without stage labels.
        for kind, detect in (
            ("spindle", yasa.spindles_detect),
            ("slow", yasa.sw_detect),
        ):
            result = detect(x, sf=100, verbose="ERROR")
            values, columns = assign_events(
                None if result is None else result.summary(), len(signal), kind
            )
            chunks.append(values)
            names += [f"eeg{channel}_{c}" for c in columns]
    # Six 5s subwindows, each of three raw channels. Absolute uV² band powers.
    windows = signal.transpose(0, 2, 1).reshape(len(signal), 3, 6, 500)
    frequencies, psd = welch(windows, fs=100, nperseg=256, axis=-1)
    for channel in range(3):
        for low, high, band in (
            (0.5, 4, "delta"),
            (4, 8, "theta"),
            (8, 12, "alpha"),
            (12, 16, "sigma"),
        ):
            mask = (frequencies >= low) & (frequencies < high)
            # Boolean indexing of a single axis preserves N x 6.
            power = np.log1p(
                np.trapezoid(psd[:, channel][..., mask], frequencies[mask], axis=-1)
            )
            chunks.append(
                np.column_stack(
                    [
                        power.mean(1),
                        power.std(1),
                        power.max(1),
                        np.abs(np.diff(power, axis=1)).mean(1),
                    ]
                )
            )
            names += [
                f"ch{channel}_{band}_{s}" for s in ("mean", "std", "max", "change")
            ]
    eog = signal[:, :, 2]
    chunks.append(
        np.column_stack(
            [
                eog.std(1),
                np.ptp(eog, axis=1),
                np.abs(np.diff(eog, axis=1)).mean(1),
                (np.diff(np.signbit(eog), axis=1)).mean(1),
            ]
        )
    )
    names += ["eog_std", "eog_ptp", "eog_speed", "eog_zero_crossing"]
    output = np.concatenate(chunks, axis=1).astype(np.float32)
    if not np.isfinite(output).all():
        raise ValueError("Non-finite event features")
    return output, names


def record_features(data, path: Path, split: str, root: Path, mode: str, channel=0):
    meta = data.record_metadata(path, split)
    contract = {
        "data": data.identity,
        "raw": data.verify(path),
        "ids": fingerprint(meta["ids"].tolist()),
        "kind": mode,
        "channel": channel if mode == "yasa" else None,
        "yasa": "0.7.0",
        "recipe": "native-segments-events-v1",
        "implementation": sha256(Path(__file__)),
    }
    target = (
        root
        / ("smoke" if data.smoke else "full")
        / mode
        / fingerprint(contract)
        / f"{path.stem}.npz"
    )
    with file_lock(target.with_suffix(".lock"), wait=True):
        if not target.with_suffix(".json").exists():
            signal = selected_signal(data, path, split)
            parts, columns = [], None
            segments = contiguous_segments(meta["epoch_index"])
            for part in segments:
                if mode == "yasa":
                    parts.append(yasa_segment(signal[part], channel))
                else:
                    values, columns = event_segment(signal[part])
                    parts.append(values)
            values = np.concatenate(parts)
            save_npz(target, ids=meta["ids"], values=values)
            save_json(
                target.with_suffix(".json"),
                {
                    "contract": contract,
                    "sha256": sha256(target),
                    "columns": columns,
                    "segments": len(segments),
                    "single_epoch_segments": int(
                        sum(part.stop - part.start == 1 for part in segments)
                    ),
                    "classifier_load_warnings": native_classifier()._sleep_edf_load_warnings
                    if mode == "yasa"
                    else [],
                    "single_epoch_yasa_time_norm": "one terminal sample; time_norm0; pinned strict-window/0-div boundary repair",
                },
            )
    info = json.loads(target.with_suffix(".json").read_text())
    if info["contract"] != contract or sha256(target) != info["sha256"]:
        raise ValueError("Event/YASA record cache differs")
    with np.load(target, allow_pickle=False) as saved:
        if not np.array_equal(saved["ids"], meta["ids"]):
            raise ValueError("Event/YASA record IDs differ")
        return saved["values"], info


def raw_pool(data, split: str, cache: Path, progress):
    """Once-per-record disk pool, no NPZ decoding in the training batch loop."""
    meta = data.metadata(split)
    contract = {
        "data": data.identity,
        "ids": fingerprint(meta["ids"].tolist()),
        "recipe": "raw-uV-resample-poly-100to50-div50-clip20-float16-v1",
    }
    target = (
        cache
        / ("smoke" if data.smoke else "full")
        / "signal_pool"
        / fingerprint(contract)
        / f"{split}.npy"
    )
    with file_lock(target.with_suffix(".lock"), wait=True):
        if not target.with_suffix(".json").exists():
            with atomic_path(target) as temporary:
                pool = np.lib.format.open_memmap(
                    temporary,
                    mode="w+",
                    dtype=np.float16,
                    shape=(len(meta["ids"]), 3, 1500),
                )
                offset = 0
                paths = data.paths(split)
                for i, path in enumerate(paths):
                    progress.work(i, len(paths), "записей")
                    progress.update(
                        detail=f"{split}: signal pool, запись {i + 1}/{len(paths)}"
                    )
                    data.verify(path)
                    signal = selected_signal(data, path, split)
                    # Resample each epoch: no filtering across gaps or other epochs.
                    values = resample_poly(signal, 1, 2, axis=1).transpose(0, 2, 1)
                    pool[offset : offset + len(values)] = np.clip(values / 50, -20, 20)
                    offset += len(values)
                if offset != len(pool):
                    raise ValueError("Signal pool epoch coverage differs")
                pool.flush()
                del pool
            save_json(
                target.with_suffix(".json"),
                {"contract": contract, "sha256": sha256(target)},
            )
    if json.loads(target.with_suffix(".json").read_text())["contract"] != contract:
        raise ValueError("Signal pool resume differs")
    pool = np.load(target, mmap_mode="r", allow_pickle=False)
    if pool.shape != (len(meta["ids"]), 3, 1500) or pool.dtype != np.float16:
        raise ValueError("Signal pool shape/dtype differs")
    return pool
