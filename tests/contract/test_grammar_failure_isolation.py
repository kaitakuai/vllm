# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""What a request sees after its own grammar has been disabled.

When ``XgrammarGrammar`` gives up on a request it sets ``_grammar_failed`` and
stops constraining it. The two methods below are what the rest of the engine
calls afterwards, and both are shared with other requests: the bitmask is one
tensor with a row per request, reused across steps.

Neither test needs a GPU or a running engine -- ``_grammar_failed`` short-
circuits before the matcher is touched, so the grammar object is constructed
without one.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("xgrammar")

from vllm.v1.structured_output.backend_xgrammar import (  # noqa: E402
    XgrammarGrammar,
)


def _failed_grammar() -> XgrammarGrammar:
    """A grammar in the failed state, with no matcher behind it."""
    grammar = object.__new__(XgrammarGrammar)
    grammar._grammar_failed = True
    return grammar


def test_failed_grammar_does_not_inherit_the_previous_row() -> None:
    """A disabled grammar must clear its row, not leave the last mask in it.

    Rows are reused between steps and between requests. Returning without
    writing leaves whatever occupied the row before -- in a shared batch,
    another request's grammar -- silently constraining this request to an
    unrelated set of tokens.
    """
    vocab_bits = 4
    bitmask = torch.zeros((2, vocab_bits), dtype=torch.int32)
    # Row 1 carries a leftover mask admitting a single token.
    bitmask[1].fill_(0)
    bitmask[1][0] = 0b0010

    _failed_grammar().fill_bitmask(bitmask, 1)

    assert torch.equal(bitmask[1], torch.full((vocab_bits,), -1, dtype=torch.int32)), (
        "a disabled grammar left a stale mask in its row"
    )
    assert torch.equal(bitmask[0], torch.zeros(vocab_bits, dtype=torch.int32)), (
        "filling one row must not disturb another request's row"
    )


def test_failed_grammar_accepts_draft_tokens() -> None:
    """A disabled grammar must not reject the speculative tokens it is given.

    ``validate_tokens`` returns the accepted prefix. A failed matcher's
    verdicts are meaningless, and rejecting on them takes speculative
    acceptance to zero -- a silent throughput collapse rather than an error.
    """
    tokens = [11, 22, 33]

    assert _failed_grammar().validate_tokens(tokens) == tokens
