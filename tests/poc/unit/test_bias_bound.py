# SPDX-License-Identifier: Apache-2.0
"""Real e_score_correction_bias magnitudes vs the grouped-forcing bound.

The forcing formula assigns chosen experts logits top_k..1; under sigmoid
scoring the lowest-ranked chosen expert scores sigmoid(1), and a floor
expert flips into the selection when its bias exceeds a chosen expert's
bias by more than that gap (see test_grouped_forcing boundary test).

The spreads below were measured from the published checkpoints
(HTTP-range reads of every e_score_correction_bias tensor, all MoE
layers, max over layers of per-layer max-min). Re-measure if a checkpoint
revision changes.

The selection override is NOT installed (it collapsed honest/fraud
separability — see PR #2); the engine's selection over the forced ladder
is again the production path, so these bounds GATE CORRECTNESS on every
model whose scoring adds e_score_correction_bias.
"""
import math

import pytest

SIGMOID_GAP = 1.0 / (1.0 + math.exp(-1.0))  # sigmoid(min forced value = 1)

MEASURED_MAX_SPREAD = {
    # repo: (max per-layer spread, layers measured)
    "deepseek-ai/DeepSeek-V3": (0.2208, 59),
    "moonshotai/Kimi-K2-Instruct": (0.7832, 60),
}


def test_bound_derives_from_forcing_formula():
    """The bound is sigmoid of the SMALLEST forced logit value (1). If the
    formula's value ladder changes, this constant — and both pins below —
    must be re-derived, and the change is a consensus change."""
    assert abs(SIGMOID_GAP - 0.7311) < 1e-4


def test_deepseek_v3_bias_within_forcing_bound():
    """V3's learned biases stay well under the gap: seeded selection cannot
    be overridden by bias on any layer. Grouped forcing is sound for V3."""
    spread, _ = MEASURED_MAX_SPREAD["deepseek-ai/DeepSeek-V3"]
    assert spread < SIGMOID_GAP - 0.25  # comfortable margin


def test_kimi_k2_bias_exceeds_forcing_bound():
    """Kimi-K2's worst layer has bias spread 0.7832 > sigmoid(1) = 0.7311:
    logit forcing alone cannot survive it. With the selection override
    retired (separability, PR #2), this bound is a LIVE production blocker
    for Kimi-K2 until the forcing values are strengthened (spec change,
    joint sign-off) or the checkpoint changes. If this test FAILS the
    checkpoint changed: re-measure."""
    spread, _ = MEASURED_MAX_SPREAD["moonshotai/Kimi-K2-Instruct"]
    assert spread > SIGMOID_GAP
