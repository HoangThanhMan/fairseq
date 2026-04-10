# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn


def _resolve_decoder_layers(model: nn.Module):
    candidates = [
        ("decoder", "layers"),
        ("module", "decoder", "layers"),
        ("model", "decoder", "layers"),
        ("module", "model", "decoder", "layers"),
    ]

    for chain in candidates:
        current = model
        found = True
        for attr in chain:
            if not hasattr(current, attr):
                found = False
                break
            current = getattr(current, attr)
        if found:
            return current

    raise ValueError(
        "Could not locate decoder layers on model. Expected one of "
        "model.decoder.layers or model.model.decoder.layers."
    )


class FFNIntervention(object):
    """A single FFN intervention rule for one decoder layer."""

    def __init__(
        self,
        layer_idx: int,
        scale: float = 1.0,
        bias: float = 0.0,
        neurons: Optional[Sequence[int]] = None,
        clamp_min: Optional[float] = None,
        clamp_max: Optional[float] = None,
    ):
        self.layer_idx = int(layer_idx)
        self.scale = float(scale)
        self.bias = float(bias)
        self.neurons = list(neurons) if neurons is not None else None
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def apply(self, activation: torch.Tensor) -> torch.Tensor:
        if self.neurons is None:
            updated = activation * self.scale + self.bias
            if self.clamp_min is not None or self.clamp_max is not None:
                updated = torch.clamp(updated, min=self.clamp_min, max=self.clamp_max)
            return updated

        index = torch.tensor(self.neurons, device=activation.device, dtype=torch.long)
        selected = activation.index_select(-1, index)
        selected = selected * self.scale + self.bias
        if self.clamp_min is not None or self.clamp_max is not None:
            selected = torch.clamp(selected, min=self.clamp_min, max=self.clamp_max)

        updated = activation.clone()
        updated.index_copy_(-1, index, selected)
        return updated


class FFNInterventionManager(object):
    """Register forward hooks on decoder FFN `fc1` to intervene activations."""

    def __init__(
        self,
        model: nn.Module,
        interventions: Union[Dict[int, dict], Sequence[FFNIntervention]],
    ):
        self.model = model
        self._decoder_layers = _resolve_decoder_layers(model)
        self._hooks = []
        self._interventions = self._normalize_interventions(interventions)

    def _normalize_interventions(
        self,
        interventions: Union[Dict[int, dict], Sequence[FFNIntervention]],
    ) -> Dict[int, List[FFNIntervention]]:
        grouped = defaultdict(list)

        if isinstance(interventions, dict):
            for layer_idx, spec in interventions.items():
                if isinstance(spec, FFNIntervention):
                    grouped[int(layer_idx)].append(spec)
                    continue

                if not isinstance(spec, dict):
                    raise TypeError(
                        "Intervention dict values must be FFNIntervention or dict specs."
                    )

                grouped[int(layer_idx)].append(
                    FFNIntervention(
                        layer_idx=int(layer_idx),
                        scale=spec.get("scale", 1.0),
                        bias=spec.get("bias", 0.0),
                        neurons=spec.get("neurons"),
                        clamp_min=spec.get("clamp_min"),
                        clamp_max=spec.get("clamp_max"),
                    )
                )
        else:
            for spec in interventions:
                if not isinstance(spec, FFNIntervention):
                    raise TypeError(
                        "Interventions must be FFNIntervention instances or dict specs."
                    )
                grouped[int(spec.layer_idx)].append(spec)

        return dict(grouped)

    def _validate_layer_index(self, layer_idx: int):
        if layer_idx < 0 or layer_idx >= len(self._decoder_layers):
            raise ValueError(
                "Layer index {} out of range for decoder with {} layers".format(
                    layer_idx, len(self._decoder_layers)
                )
            )

        layer = self._decoder_layers[layer_idx]
        if not hasattr(layer, "fc1"):
            raise ValueError(
                "Decoder layer {} has no fc1 module for FFN intervention".format(
                    layer_idx
                )
            )

    def _make_hook(self, layer_idx: int):
        layer_interventions = self._interventions[layer_idx]

        def hook_fn(module, inp, out):
            if not torch.is_tensor(out):
                raise TypeError("Expected Tensor output from fc1 hook")

            updated = out
            for intervention in layer_interventions:
                updated = intervention.apply(updated)
            return updated

        return hook_fn

    def register_hooks(self):
        self.clear()
        for layer_idx in sorted(self._interventions.keys()):
            self._validate_layer_index(layer_idx)
            layer = self._decoder_layers[layer_idx]
            handle = layer.fc1.register_forward_hook(self._make_hook(layer_idx))
            self._hooks.append(handle)

    def clear(self):
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

    def __enter__(self):
        self.register_hooks()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.clear()


class CETTExtractor(object):
    """Collect decoder FFN activations and compute CETT statistics."""

    def __init__(self, model: nn.Module, layer_indices: Iterable[int], eps: float = 1e-10):
        self.model = model
        self.layer_indices = [int(i) for i in layer_indices]
        self.eps = float(eps)

        self._decoder_layers = _resolve_decoder_layers(model)
        self._hooks = []
        self._intermediate = {}

        for layer_idx in self.layer_indices:
            self._validate_layer_index(layer_idx)

    def _validate_layer_index(self, layer_idx: int):
        if layer_idx < 0 or layer_idx >= len(self._decoder_layers):
            raise ValueError(
                "Layer index {} out of range for decoder with {} layers".format(
                    layer_idx, len(self._decoder_layers)
                )
            )

        layer = self._decoder_layers[layer_idx]
        required = ["fc1", "fc2", "activation_fn"]
        missing = [name for name in required if not hasattr(layer, name)]
        if missing:
            raise ValueError(
                "Decoder layer {} missing required attributes: {}".format(
                    layer_idx, ", ".join(missing)
                )
            )

    def _make_hook(self, layer_idx: int):
        def hook_fn(module, inp, out):
            layer = self._decoder_layers[layer_idx]
            self._intermediate[layer_idx] = layer.activation_fn(out).detach()

        return hook_fn

    def register_hooks(self):
        self.clear()
        for layer_idx in self.layer_indices:
            layer = self._decoder_layers[layer_idx]
            handle = layer.fc1.register_forward_hook(self._make_hook(layer_idx))
            self._hooks.append(handle)

    def clear(self):
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
        self._intermediate.clear()

    @torch.no_grad()
    def compute_cett(self, *model_args, **model_kwargs) -> Dict[int, np.ndarray]:
        self._intermediate.clear()
        self.model(*model_args, **model_kwargs)

        cett_per_layer = {}
        for layer_idx in self.layer_indices:
            if layer_idx not in self._intermediate:
                raise RuntimeError(
                    "No FFN activations were captured for layer {}. "
                    "Ensure hooks are registered before calling compute_cett.".format(
                        layer_idx
                    )
                )

            z = self._intermediate[layer_idx]
            layer = self._decoder_layers[layer_idx]
            w_down = layer.fc2.weight
            b_down = layer.fc2.bias

            h_full = torch.matmul(z, w_down.transpose(0, 1))
            if b_down is not None:
                h_full = h_full + b_down
            h_norm = torch.norm(h_full, dim=-1, keepdim=True).clamp(min=self.eps)

            w_col_norms = torch.norm(w_down, dim=0)
            view_shape = [1] * (z.dim() - 1) + [w_col_norms.size(0)]
            h_j_norm = z.abs() * w_col_norms.view(*view_shape)

            cett_per_layer[layer_idx] = (h_j_norm / h_norm).cpu().numpy()

        return cett_per_layer

    @staticmethod
    def aggregate_cett(cett_per_layer: Dict[int, np.ndarray], method: str = "mean") -> np.ndarray:
        if method not in {"mean", "max"}:
            raise ValueError("Unsupported aggregation method: {}".format(method))

        parts = []
        for layer_idx in sorted(cett_per_layer.keys()):
            arr = np.asarray(cett_per_layer[layer_idx])

            if arr.ndim <= 1:
                reduced = arr
            else:
                axes = tuple(range(arr.ndim - 1))
                reduced = arr.max(axis=axes) if method == "max" else arr.mean(axis=axes)

            parts.append(np.asarray(reduced).reshape(-1))

        if not parts:
            return np.array([], dtype=np.float32)
        return np.concatenate(parts, axis=0)

    def __enter__(self):
        self.register_hooks()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.clear()


__all__ = [
    "CETTExtractor",
    "FFNIntervention",
    "FFNInterventionManager",
]
