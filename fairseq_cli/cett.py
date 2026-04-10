#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import ast
import json
import logging
import os

import numpy as np
import torch

from fairseq import checkpoint_utils, utils
from fairseq.models import CETTExtractor, FFNIntervention, FFNInterventionManager


def get_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Compute CETT from decoder FFN layers with optional FFN interventions. "
            "Input must be a torch serialized dict containing model forward kwargs "
            "or a full sample with a `net_input` field."
        )
    )
    parser.add_argument("--path", required=True, help="path(s) to model file(s), colon separated")
    parser.add_argument("--sample-file", required=True, help="torch file containing model inputs")
    parser.add_argument(
        "--layers",
        required=True,
        help="comma-separated decoder layer indices for CETT extraction, e.g. 0,1,5",
    )
    parser.add_argument(
        "--intervention-file",
        default=None,
        help=(
            "optional JSON file for FFN interventions. "
            "Accepted formats: list of dicts with `layer_idx`, or dict keyed by layer index."
        ),
    )
    parser.add_argument(
        "--ablate-layer",
        action="append",
        type=int,
        default=[],
        help="optionally zero-out all neurons in a decoder FFN layer (repeatable)",
    )
    parser.add_argument(
        "--aggregate",
        default="none",
        choices=["none", "mean", "max"],
        help="optional CETT aggregation method across non-feature dimensions",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="output path (.npz for per-layer values, .npy for aggregate only)",
    )
    parser.add_argument(
        "--model-overrides",
        default="{}",
        type=str,
        help="a dictionary used to override model args at generation",
    )
    parser.add_argument(
        "--checkpoint-suffix",
        default="",
        type=str,
        help="checkpoint suffix to load, if any",
    )
    parser.add_argument(
        "--checkpoint-shard-count",
        default=1,
        type=int,
        help="number of checkpoint shards",
    )
    parser.add_argument("--fp16", action="store_true", help="use half precision")
    parser.add_argument("--cpu", action="store_true", help="run on CPU")
    return parser


def _parse_layers(raw_layers):
    values = [v.strip() for v in raw_layers.split(",") if v.strip()]
    if not values:
        raise ValueError("--layers must not be empty")
    return [int(v) for v in values]


def _extract_forward_kwargs(sample):
    if not isinstance(sample, dict):
        raise TypeError("Sample file must contain a dict")

    if "net_input" in sample:
        forward_kwargs = sample["net_input"]
    elif "forward_kwargs" in sample:
        forward_kwargs = sample["forward_kwargs"]
    else:
        forward_kwargs = sample

    if not isinstance(forward_kwargs, dict):
        raise TypeError("Extracted model inputs must be a dict")
    return forward_kwargs


def _load_interventions(path):
    if path is None:
        return []

    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)

    interventions = []
    if isinstance(raw, dict):
        items = raw.items()
        for layer_idx, spec in items:
            if not isinstance(spec, dict):
                raise TypeError("Intervention spec must be a dict")
            interventions.append(
                FFNIntervention(
                    layer_idx=int(layer_idx),
                    scale=spec.get("scale", 1.0),
                    bias=spec.get("bias", 0.0),
                    neurons=spec.get("neurons"),
                    clamp_min=spec.get("clamp_min"),
                    clamp_max=spec.get("clamp_max"),
                )
            )
    elif isinstance(raw, list):
        for spec in raw:
            if not isinstance(spec, dict):
                raise TypeError("Intervention list items must be dicts")
            if "layer_idx" not in spec:
                raise ValueError("Each intervention item must contain `layer_idx`")
            interventions.append(
                FFNIntervention(
                    layer_idx=int(spec["layer_idx"]),
                    scale=spec.get("scale", 1.0),
                    bias=spec.get("bias", 0.0),
                    neurons=spec.get("neurons"),
                    clamp_min=spec.get("clamp_min"),
                    clamp_max=spec.get("clamp_max"),
                )
            )
    else:
        raise TypeError("Intervention file must contain a dict or list")

    return interventions


def _move_to_device(sample, device):
    return utils.apply_to_sample(
        lambda tensor: tensor.to(device) if torch.is_tensor(tensor) else tensor,
        sample,
    )


def main(args):
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
    )
    logger = logging.getLogger("fairseq_cli.cett")

    layer_indices = _parse_layers(args.layers)
    model_overrides = ast.literal_eval(args.model_overrides)

    use_cuda = torch.cuda.is_available() and not args.cpu
    device = torch.device("cuda") if use_cuda else torch.device("cpu")

    models, _cfg, _task = checkpoint_utils.load_model_ensemble_and_task(
        utils.split_paths(args.path),
        arg_overrides=model_overrides,
        strict=(args.checkpoint_shard_count == 1),
        suffix=args.checkpoint_suffix,
        num_shards=args.checkpoint_shard_count,
    )
    if len(models) == 0:
        raise RuntimeError("No model was loaded")

    if len(models) > 1:
        logger.warning("Received an ensemble. Using the first model for CETT extraction.")
    model = models[0]

    if args.fp16:
        model.half()
    model.eval()
    model.to(device)

    sample = torch.load(args.sample_file, map_location="cpu")
    forward_kwargs = _extract_forward_kwargs(sample)
    forward_kwargs = _move_to_device(forward_kwargs, device)

    interventions = _load_interventions(args.intervention_file)
    for layer_idx in args.ablate_layer:
        interventions.append(FFNIntervention(layer_idx=layer_idx, scale=0.0))

    extractor = CETTExtractor(model, layer_indices)
    intervention_manager = None
    if interventions:
        intervention_manager = FFNInterventionManager(model, interventions)

    try:
        extractor.register_hooks()
        if intervention_manager is not None:
            intervention_manager.register_hooks()

        cett_per_layer = extractor.compute_cett(**forward_kwargs)
    finally:
        extractor.clear()
        if intervention_manager is not None:
            intervention_manager.clear()

    aggregate = None
    if args.aggregate != "none":
        aggregate = CETTExtractor.aggregate_cett(cett_per_layer, method=args.aggregate)

    ext = os.path.splitext(args.output)[1].lower()
    if ext == ".npy":
        if aggregate is None:
            raise ValueError(".npy output requires --aggregate mean|max")
        np.save(args.output, aggregate)
    else:
        payload = {}
        for layer_idx, value in cett_per_layer.items():
            payload["layer_{}".format(layer_idx)] = value
        if aggregate is not None:
            payload["aggregate"] = aggregate
        np.savez(args.output, **payload)

    logger.info("Saved CETT output to %s", args.output)


def cli_main():
    parser = get_parser()
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()
