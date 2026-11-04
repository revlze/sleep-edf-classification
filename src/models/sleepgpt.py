"""Published SleepGPT large encoder, original EDF channel mapping, trainable staging head."""

import contextlib
import io
from functools import partial
from pathlib import Path

import torch
from einops import rearrange
from torch import nn
from torch.utils.checkpoint import checkpoint

from src.expansion_assets import asset, package, provenance
from src.models.expansion_official import definitions, numpy_namespace


def unsupported_drop_path(*args, **kwargs):
    raise ValueError("This fixed downstream recipe requires drop_path_rate=0")


class SleepGPTStager(nn.Module):
    """Only the new 1536→5 head is random; every encoder tensor loads strictly."""

    def __init__(self, cache: Path, inputs: Path | None, strategy: str):
        super().__init__()
        if strategy not in ("frozen", "partial"):
            raise ValueError(strategy)
        self.strategy = strategy
        source = package("sleepgpt", cache, inputs)
        saved = torch.load(
            asset("sleepgpt.ckpt", cache, inputs), map_location="cpu", weights_only=True
        )
        config = dict(saved["hyper_parameters"]["config"])
        config.update(drop_path_rate=0.0, actual_channels="EDF", mask_ratio=0.0)
        namespace = numpy_namespace()
        namespace.update(
            partial=partial,
            rearrange=rearrange,
            List=list,
            trunc_normal_=nn.init.trunc_normal_,
            DropPath=unsupported_drop_path,
            register_model=lambda cls: cls,
            rank_zero_info=lambda *args: None,
        )
        definitions(source / "main/modules/multiway_transformer.py", namespace)
        with contextlib.redirect_stdout(io.StringIO()):
            self.encoder = namespace[config["model_arch"]](config=config)
        state = saved["state_dict"]
        encoder_state = {
            k.removeprefix("transformer."): v
            for k, v in state.items()
            if k.startswith("transformer.")
        }
        self.encoder.load_state_dict(encoder_state, strict=True)
        if any(
            not k.startswith(
                (
                    "transformer.",
                    "token_type_embeddings.",
                    "Masked_docoder.",
                    "Masked_docoder_fft.",
                )
            )
            for k in state
        ):
            raise ValueError(
                "Unexpected published pretraining module; review checkpoint"
            )
        self.token_type_embeddings = nn.Embedding(2, self.encoder.num_features)
        self.token_type_embeddings.load_state_dict(
            {"weight": state["token_type_embeddings.weight"]}, strict=True
        )
        self.classifier = nn.Linear(2 * self.encoder.num_features, 5)
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for parameter in self.classifier.parameters():
            parameter.requires_grad_(True)
        if strategy == "partial":
            for module in [*self.encoder.blocks[-2:], self.encoder.norm]:
                for parameter in module.parameters():
                    parameter.requires_grad_(True)
        self.provenance = {
            **provenance(["sleepgpt.ckpt"], ["sleepgpt"]),
            "channels": {"EOG horizontal": 3, "EEG Fpz-Cz": 5, "EEG Pz-Oz": 7},
            "embedding": "concat normalized time CLS and frequency CLS; original last two cross-modal FFN blocks",
            "discarded_modules": ["Masked_docoder", "Masked_docoder_fft"],
            "new_trained_head": "Linear(1536,5)",
            "strategy": strategy,
            "drop_path_rate": 0.0,
        }
        self.register_buffer(
            "channel_mu",
            torch.tensor(
                [
                    -0.06846,
                    0.19104,
                    0.38937,
                    -2.0938,
                    0.0016496,
                    -0.000048439,
                    0.00081125,
                    0.000071748,
                ]
            ),
            persistent=False,
        )
        self.register_buffer(
            "channel_std",
            torch.tensor(
                [34.6887, 34.9556, 23.2826, 35.4035, 26.8738, 4.9272, 25.1366, 3.6142]
            ),
            persistent=False,
        )
        self.train(False)

    def train(self, mode: bool = True):
        super().train(mode)
        if hasattr(self, "encoder") and self.strategy == "frozen":
            self.encoder.eval()
        return self

    def encode(self, raw: torch.Tensor) -> torch.Tensor:
        if raw.ndim != 3 or raw.shape[1:] != (3000, 3):
            raise ValueError("SleepGPT input must be B×3000×3, µV at 100Hz")
        wave = raw.new_zeros((len(raw), 8, 3000))
        wave[:, 3], wave[:, 5], wave[:, 7] = raw[:, :, 2], raw[:, :, 0], raw[:, :, 1]
        wave = (wave - self.channel_mu[None, :, None]) / self.channel_std[None, :, None]
        presence = raw.new_zeros((len(raw), 8))
        presence[:, [3, 5, 7]] = 1
        with torch.autocast(device_type=raw.device.type, enabled=False):
            fft, _ = self.encoder.get_fft(wave.float(), presence, aug=False)
        result = self.encoder.embed([wave, fft], presence, mask=False)
        x, boundary = result["x"], result["x_len"]
        x = torch.cat(
            (
                x[:, :boundary] + self.token_type_embeddings.weight[0],
                x[:, boundary:] + self.token_type_embeddings.weight[1],
            ),
            dim=1,
        )
        mask = torch.ones((len(raw), x.shape[1]), dtype=torch.bool, device=raw.device)
        for block in self.encoder.blocks:
            call = partial(
                block, mask=mask, modality_type="tf", relative_position_bias=None
            )
            if self.training and any(p.requires_grad for p in block.parameters()):
                x = checkpoint(call, x, use_reentrant=False)
            else:
                x = call(x)
        x = self.encoder.norm(x)
        return torch.cat((x[:, 0], x[:, boundary]), dim=-1)

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode(raw))

    def trainable_manifest(self) -> dict:
        entries = [
            {"name": n, "shape": list(p.shape), "parameters": p.numel()}
            for n, p in self.named_parameters()
            if p.requires_grad
        ]
        return {
            "strategy": self.strategy,
            "trainable": sum(p["parameters"] for p in entries),
            "total": sum(p.numel() for p in self.parameters()),
            "entries": entries,
        }
