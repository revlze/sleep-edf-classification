"""Verified reuse of saved representations and honest role-split stacking inputs."""

import json
from functools import cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
from sklearn.model_selection import GroupKFold

from src.expansion_data import physiological_features, shared_folds
from src.models.pretrained_assets import sha256
from src.pretrained import ROOT, save_json, save_npz
from src.refinement_data import InputStore
from src.refinement_models import confidence_features
from src.series_data import align_prediction, fingerprint


class PhysiologyInputs:
    def __init__(self, data, args, progress):
        self.data, self.args, self.progress = data, args, progress
        self.bundle = Path(args.input_root[0])
        self.manifest = json.loads((self.bundle / "bundle.json").read_text())
        if (
            self.manifest["kind"] != "physiology_inputs_v1"
            or self.manifest["data"] != data.identity
        ):
            raise ValueError("New physiology input bundle/data identity required")
        self.verified = set()
        self.store = InputStore([self.bundle], data)
        self.shared = SimpleNamespace(
            data=data, args=args, progress=progress, memo={}, embeddings=self.embeddings
        )

    def path(self, relative: str) -> Path:
        path = self.bundle / relative
        expected = self.manifest["files"].get(relative)
        if expected is None or not path.is_file():
            raise FileNotFoundError(f"Required portable input missing: {relative}")
        if relative not in self.verified:
            if sha256(path) != expected:
                raise ValueError(f"Portable input checksum differs: {relative}")
            self.verified.add(relative)
        return path

    @cache
    def prediction(self, name: str, split: str) -> np.ndarray:
        meta = self.data.metadata(split)
        if split != "train":
            value = self.store.prediction(name, split, meta)
            if value is None:
                raise FileNotFoundError(f"Missing {name}/{split} probabilities")
            return value["probabilities"]
        if name == "019_pretrained":
            p = self.store.baseline019(split, meta)
            if p is None:
                raise FileNotFoundError("Need saved train019, no automatic inference")
            return p
        if name in ("099_pretrained", "113_pretrained"):
            components = (
                [
                    ("019_pretrained", 0.5),
                    ("082_pretrained", 0.25),
                    ("085_pretrained", 0.25),
                ]
                if name.startswith("099")
                else [
                    ("099_pretrained", 0.5),
                    ("102_pretrained", 0.25),
                    ("106_pretrained", 0.25),
                ]
            )
            return sum(w * self.prediction(n, split) for n, w in components)
        self.path(f"results/{name}/oof_predictions.npz")
        path = self.path(f"results/{name}/oof_predictions.json")
        info = json.loads(path.read_text())
        original = json.loads(self.path(f"results/{name}/provenance.json").read_text())
        if (
            info["contract"] != original["contract"]
            or original["data"] != self.data.identity
            or original["smoke"]
        ):
            raise ValueError(f"{name}: original train OOF provenance differs")
        expected = OmegaConf.to_container(
            OmegaConf.load(ROOT / "configs" / (name + ".yaml")), resolve=True
        )
        if original["config"] != expected:
            raise ValueError(
                f"{name}: upstream recipe differs from current pinned configuration"
            )
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as z:
            values = {k: z[k] for k in z.files}
        folds = np.asarray(info["folds"])
        ids = values["ids"]
        for f in range(3):
            if set(values["subject_id"][folds == f]) & set(
                values["subject_id"][folds != f]
            ):
                raise ValueError("Original OOF subject leakage")
        lookup = {str(i): j for j, i in enumerate(ids)}
        take = np.array([lookup[str(i)] for i in meta["ids"]])
        if not self.data.smoke and not np.array_equal(
            folds[take], shared_folds(self.data, self.args.cache_root)
        ):
            raise ValueError("Upstream OOF folds differ from this series")
        return align_prediction({k: v[take] for k, v in values.items()}, meta)[
            "probabilities"
        ]

    @cache
    def embedding_index(self):
        result = {}
        for relative in self.manifest["files"]:
            if relative.startswith("representations/") and relative.endswith(".json"):
                path = self.path(relative)
                info = json.loads(path.read_text())
                key = (info["family"], path.parent.name, path.stem)
                if key in result:
                    raise ValueError("Duplicate saved embedding record")
                result[key] = (relative, info)
        return result

    @cache
    def embeddings(self, split: str) -> np.ndarray:
        from src.models.pretrained_assets import asset_manifest

        chunks = []
        for family in ("tiny", "anysleep"):
            names = (
                [
                    "tinysleepnet.pt",
                    "tinysleepnet-source.py",
                    "tinysleepnet-config.json",
                ]
                if family == "tiny"
                else ["anysleep-run1.pth", "anysleep_no_hydra.py"]
            )
            layer = (
                "clf.encode contextual128 rolling20"
                if family == "tiny"
                else "segment_classifier.segment_classifier.2 AvgPool before 5class Conv"
            )
            parts = []
            for record in self.data.paths(split):
                key = (family, split, record.stem)
                if key not in self.embedding_index():
                    raise FileNotFoundError(
                        f"Missing saved embedding {key}; automatic extraction is disabled"
                    )
                relative, info = self.embedding_index()[key]
                if (
                    info.get("data") != self.data.identity
                    or info.get("smoke")
                    or info.get("input") != self.data.verify(record)
                    or info.get("assets") != asset_manifest(names)
                    or info.get("layer") != layer
                ):
                    raise ValueError(
                        f"Saved embedding data/weights/layer differs: {key}"
                    )
                path = self.path(relative[:-5] + ".npz")
                if info["sha256"] != self.manifest["files"][relative[:-5] + ".npz"]:
                    raise ValueError("Embedding content provenance differs")
                meta = self.data.record_metadata(record, split)
                with np.load(path, allow_pickle=False) as z:
                    lookup = {str(v): i for i, v in enumerate(z["ids"])}
                    if len(lookup) != len(z["ids"]) or not set(meta["ids"]).issubset(
                        lookup
                    ):
                        raise ValueError("Embedding epoch IDs differ")
                    values = z["embeddings"][[lookup[str(i)] for i in meta["ids"]]]
                if (
                    values.shape != (len(meta["ids"]), 128 if family == "tiny" else 6)
                    or not np.isfinite(values).all()
                ):
                    raise ValueError("Invalid saved pre-classifier representation")
                parts.append(values)
            chunks.append(np.concatenate(parts))
        x = np.concatenate(chunks, 1).astype(np.float32)
        if x.shape[1] != 134:
            raise ValueError(
                "Expected Tiny contextual128 + AnySleep pre-classifier6, not probabilities"
            )
        return x

    @cache
    def physio(self, split):
        return physiological_features(self.data, split, self.args.feature_cache)

    def residual(self, split, event_features=None, components=None):
        p = (
            np.stack(
                [
                    self.prediction(n, split)
                    for n in ("099_pretrained", "102_pretrained", "106_pretrained")
                ],
                1,
            )
            if components is None
            else components
        )
        base = (p * np.array([0.5, 0.25, 0.25], np.float32)[None, :, None]).sum(1)
        x = np.concatenate([confidence_features(p), base], 1)
        if event_features is not None:
            x = np.concatenate([x, event_features], 1)
        return x, base


def role_masks(meta: dict, folds: np.ndarray, outer: int):
    """A=frozen expert fit; B=upstream head fit; C=calibration fit; D=held.

    A matches existing outer/inner0 experts. B/C partition their held subjects.
    No calibration-fit or outer-held labels enter expert/head fitting.
    """
    outer_train = np.flatnonzero(folds != outer)
    inner = np.full(len(folds), -1)
    for i, (_, held) in enumerate(
        GroupKFold(2).split(outer_train, groups=meta["subject_id"][outer_train])
    ):
        inner[outer_train[held]] = i
    expert_fit = inner == 1
    remaining = np.flatnonzero(inner == 0)
    head_indices, calibration_indices = next(
        GroupKFold(2).split(remaining, groups=meta["subject_id"][remaining])
    )
    head_fit, calibration_fit = np.zeros(len(folds), bool), np.zeros(len(folds), bool)
    (
        head_fit[remaining[head_indices]],
        calibration_fit[remaining[calibration_indices]],
    ) = True, True
    held = folds == outer
    masks = [expert_fit, head_fit, calibration_fit, held]
    groups = [set(meta["subject_id"][m]) for m in masks]
    if any(groups[i] & groups[j] for i in range(4) for j in range(i)) or not np.all(
        np.sum(masks, axis=0) == 1
    ):
        raise ValueError("Stacking role split is not a disjoint subject partition")
    return masks


def prepare_residual_inputs(inputs: PhysiologyInputs):
    """Reuse six saved nested experts; fit six cheap heads, never refit backbones.

    Honest role-split CV is deliberately smaller than the final upstream fits.
    This transport limitation is stored, not mislabeled as exact nested 113 CV.
    """
    import torch

    from src.expansion_data import RecordCache
    from src.models.tinysleepnet import TinySleepNetAdapter
    from src.pretrained_signals import contiguous_segments
    from src.training.expansion import parameter_manifest, restore_state
    from src.training.refinement import head_fit, head_predict, tiny_record
    from src.training.refinement import inputs as head_inputs

    data, args, progress = inputs.data, inputs.args, inputs.progress
    meta = data.metadata("train")
    folds = shared_folds(data, args.cache_root)
    mode = "smoke" if data.smoke else "full"
    root = args.cache_root / mode / "role_stack"
    root.mkdir(parents=True, exist_ok=True)
    signal_cache = RecordCache(data, args.cache_root / mode)
    for outer in range(3):
        a, b, c, d = role_masks(meta, folds, outer)
        roles = {
            name: sorted(set(meta["subject_id"][mask]))
            for name, mask in zip(
                ("expert_fit", "head_fit", "calibration_fit", "held"),
                (a, b, c, d),
                strict=True,
            )
        }
        contract = fingerprint(
            {
                "data": data.identity,
                "ids": meta["ids"].tolist(),
                "roles": roles,
                "recipe": "honest-role-stack-v1",
                "source_files": inputs.manifest["files"],
            }
        )
        target = root / f"outer_{outer}.npz"
        if target.exists():
            info = json.loads(target.with_suffix(".json").read_text())
            if info["contract"] != contract or info["sha256"] != sha256(target):
                raise ValueError("Honest role stack resume differs")
            continue
        progress.stage(
            "Подготовка fusion",
            f"outer {outer + 1}/3: A эксперты → B heads → C correction → D held",
        )
        p = np.stack([inputs.prediction("019_pretrained", "train")] * 3, 1).copy()
        for member, name in enumerate(("082_pretrained", "085_pretrained"), 1):
            progress.scope = (
                f"fusion outer {outer + 1}/3 · frozen source {member}/2 {name}"
            )
            relative = f"nested/outer_{outer}/inner_0/{name[:3]}/best.pt"
            saved = torch.load(
                inputs.path(relative), map_location="cpu", weights_only=True
            )
            source = json.loads(
                inputs.path(
                    f"nested/outer_{outer}/inner_0/{name[:3]}/held.json"
                ).read_text()
            )
            if saved["contract"] != source["contract"] or (
                not data.smoke and set(source["train_subjects"]) & set(roles["held"])
            ):
                raise ValueError(
                    "Nested expert checkpoint identity/training subjects differ"
                )
            # Full bundle roles refer to all58 subjects; smoke is structural only.
            if not data.smoke and set(source["train_subjects"]) != set(
                roles["expert_fit"]
            ):
                raise ValueError("Existing nested expert does not match role A")
            model = TinySleepNetAdapter(
                args.checkpoint_cache, args.checkpoint_root, args.device
            ).model
            if name.startswith("082"):
                model.requires_grad_(False)
                model.clf.clf.requires_grad_(True)
            if saved["manifest"] != parameter_manifest(model):
                raise ValueError("Nested expert trainable manifest differs")
            protected = (
                {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                    if not k.startswith("clf.clf.")
                }
                if member == 1
                else None
            )
            restore_state(model, saved["model"])
            model.eval().requires_grad_(False)
            cached = root / f"outer_{outer}_{name}_sources.npz"
            if cached.exists():
                info = json.loads(cached.with_suffix(".json").read_text())
                if info["contract"] != contract or info["sha256"] != sha256(cached):
                    raise ValueError("Frozen role source cache differs")
                with np.load(cached, allow_pickle=False) as z:
                    if not np.array_equal(z["ids"], meta["ids"]):
                        raise ValueError("Role source IDs differ")
                    p[:, member] = z["probabilities"]
            elif member == 1:
                # Classifier-only082 has precisely the original CNN/RNN, including
                # all BN buffers. Reuse contextual128 before its changed Linear.
                if any(
                    not torch.equal(v, model.state_dict()[k].detach().cpu())
                    for k, v in protected.items()
                ):
                    raise ValueError(
                        "Nested082 changed frozen encoder; cached128 cannot be used"
                    )
                embedding = inputs.embeddings("train")[:, :128]
                with torch.inference_mode():
                    for start in range(0, len(embedding), 4096):
                        values = torch.as_tensor(
                            embedding[start : start + 4096], device=args.device
                        )
                        p[start : start + len(values), member] = (
                            model.clf.clf(values).float().softmax(-1).cpu().numpy()
                        )
            else:
                # Existing nested held predictions already cover B/C. Only D
                # needs new frozen085 inference; no CNN replay for saved rows.
                existing = inputs.path(
                    f"nested/outer_{outer}/inner_0/{name[:3]}/held.npz"
                )
                if source["prediction_sha256"] != sha256(existing):
                    raise ValueError("Nested held probability checksum differs")
                if not data.smoke:
                    with np.load(existing, allow_pickle=False) as z:
                        lookup = {str(v): i for i, v in enumerate(z["ids"])}
                        for i in np.flatnonzero(b | c):
                            p[i, member] = z["probabilities"][
                                lookup[str(meta["ids"][i])]
                            ]
                offset = 0
                paths = data.paths("train")
                for number, path in enumerate(paths):
                    record_meta = data.record_metadata(path, "train")
                    n = len(record_meta["ids"])
                    needed = (b | c | d) if data.smoke else d
                    if np.any(needed[offset : offset + n]):
                        progress.update(
                            detail=f"fusion outer {outer + 1}/3 · frozen085 onlyD · запись {number + 1}/{len(paths)}"
                        )
                        signal = signal_cache.get(path, "train", "tiny")
                        for segment in contiguous_segments(record_meta["epoch_index"]):
                            p[
                                offset + segment.start : offset + segment.stop, member
                            ] = tiny_record(
                                model,
                                signal[segment],
                                args.device,
                                lambda text: progress.update(detail=text),
                            )
                    offset += n
            if not cached.exists():
                save_npz(cached, ids=meta["ids"], probabilities=p[:, member])
                save_json(
                    cached.with_suffix(".json"),
                    {"contract": contract, "sha256": sha256(cached)},
                )
            del model
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
        components = [np.sum(p * np.array([0.5, 0.25, 0.25])[None, :, None], 1)]
        for head_index, name in enumerate(("102_pretrained", "106_pretrained"), 1):
            progress.scope = f"fusion outer {outer + 1}/3 · upstream head {head_index}/2 {name}, train-roleB only"
            cfg = OmegaConf.to_container(
                OmegaConf.load(ROOT / "configs" / (name + ".yaml")), resolve=True
            )
            fixed_epochs = json.loads(
                inputs.path(f"results/{name}/selection.json").read_text()
            )["refit_epochs"]
            # Budget fixed from a previously completed recipe, no D/C labels used
            # for selection or scaling; C and D are only predicted by this head.
            model, bank, _, _ = head_fit(
                cfg,
                inputs.shared,
                head_inputs(cfg, inputs.shared, "train", p),
                p,
                b,
                None,
                args.trained_root / mode / "role_stack" / f"outer_{outer}" / name,
                contract,
                fixed_epochs,
                refit=True,
            )
            values, _ = head_predict(
                model,
                bank,
                np.arange(len(p)),
                cfg["model"]["context"],
                cfg["training"]["batch_size"],
                progress,
                name,
            )
            components.append(values)
            del model, bank
        save_npz(
            target,
            ids=meta["ids"],
            components=np.stack(components, 1).astype(np.float32),
            calibration_fit=c,
            held=d,
        )
        save_json(
            target.with_suffix(".json"),
            {
                "contract": contract,
                "sha256": sha256(target),
                "roles": roles,
                "limitation": "Honest four-role grouped CV, smaller expert/head/calibration training sets than final refit; not exact nested retraining of113",
                "smoke_limitation": "Original full nested experts may include smoke-C/B subjects; smoke is structural only"
                if data.smoke
                else None,
            },
        )
    save_json(
        root / "ready.json", {"data": data.identity, "smoke": data.smoke, "folds": 3}
    )
    progress.scope = ""
