# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.nn as nn

from fairseq.models import CETTExtractor, FFNIntervention, FFNInterventionManager


class _DummyDecoderLayer(nn.Module):
    def __init__(self, embed_dim, ffn_dim):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.activation_fn = torch.relu

        nn.init.constant_(self.fc1.weight, 0.5)
        nn.init.constant_(self.fc1.bias, 0.25)
        nn.init.constant_(self.fc2.weight, 0.2)
        nn.init.constant_(self.fc2.bias, 0.0)


class _DummyDecoder(nn.Module):
    def __init__(self, num_layers, embed_dim, ffn_dim):
        super().__init__()
        self.layers = nn.ModuleList(
            [_DummyDecoderLayer(embed_dim, ffn_dim) for _ in range(num_layers)]
        )


class _DummyModel(nn.Module):
    def __init__(self, num_layers=2, embed_dim=8, ffn_dim=16):
        super().__init__()
        self.decoder = _DummyDecoder(num_layers, embed_dim, ffn_dim)

    def forward(self, x):
        for layer in self.decoder.layers:
            x = layer.fc1(x)
            x = layer.activation_fn(x)
            x = layer.fc2(x)
        return x


class TestFFNCett(unittest.TestCase):
    def test_cett_extractor_shapes_and_aggregation(self):
        model = _DummyModel(num_layers=3, embed_dim=8, ffn_dim=16)
        extractor = CETTExtractor(model, layer_indices=[0, 2])

        with extractor:
            x = torch.ones(2, 4, 8)
            cett_per_layer = extractor.compute_cett(x)

        self.assertEqual(sorted(cett_per_layer.keys()), [0, 2])
        self.assertEqual(cett_per_layer[0].shape, (2, 4, 16))
        self.assertEqual(cett_per_layer[2].shape, (2, 4, 16))

        agg_mean = CETTExtractor.aggregate_cett(cett_per_layer, method="mean")
        agg_max = CETTExtractor.aggregate_cett(cett_per_layer, method="max")

        self.assertEqual(tuple(agg_mean.shape), (32,))
        self.assertEqual(tuple(agg_max.shape), (32,))

    def test_ffn_intervention_changes_output(self):
        model = _DummyModel(num_layers=2, embed_dim=8, ffn_dim=16)
        x = torch.ones(2, 4, 8)

        baseline = model(x)

        interventions = [FFNIntervention(layer_idx=0, scale=0.0)]
        manager = FFNInterventionManager(model, interventions)
        with manager:
            modified = model(x)

        self.assertFalse(torch.allclose(baseline, modified))

    def test_invalid_layer_index_raises(self):
        model = _DummyModel(num_layers=2, embed_dim=8, ffn_dim=16)
        manager = FFNInterventionManager(model, [FFNIntervention(layer_idx=8)])

        with self.assertRaises(ValueError):
            manager.register_hooks()


if __name__ == "__main__":
    unittest.main()
