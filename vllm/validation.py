# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Enforced-token support for replay-based inference validation.

A validator replays a previously produced token sequence through this engine
and compares the resulting logprobs against the originals. The ids arrive in
the request body, so they are untrusted: they end up in an embedding lookup,
where an out-of-range value is not a rejected request but a device-side
assert that takes the worker process -- and every other request on it -- down.
"""

from typing import Any

from pydantic import BaseModel, Field

# Ceilings on the replay payload, applied before any of it reaches the engine.
# They do not bound the request body (that belongs to the proxy), but they do
# stop a large body from being materialised as a huge number of models, and
# they keep the per-position fan-out finite.
MAX_ENFORCED_TOKENS = 32768
MAX_TOP_TOKENS_PER_POSITION = 64

# Prefix vLLM uses when a request asks for token ids instead of text.
_TOKEN_ID_PREFIX = "token_id:"


def _parse_token_id(token: str) -> int | None:
    """Return the id a token string denotes, or None if it denotes text.

    Only a plain decimal is accepted, so forms that ``int()`` would otherwise
    take -- underscores, non-ASCII digits, surrounding whitespace, a sign --
    are treated as text and tokenized instead of being silently reinterpreted
    as an id.
    """
    body = (
        token[len(_TOKEN_ID_PREFIX) :] if token.startswith(_TOKEN_ID_PREFIX) else token
    )
    # isascii() matters: isdigit() alone is true for non-ASCII digits, so "٣"
    # would resolve to id 3 instead of being tokenized as the text it is.
    return int(body) if body.isascii() and body.isdigit() else None


class EnforcedToken(BaseModel):
    token: str
    top_tokens: list[str] = Field(
        default_factory=list, max_length=MAX_TOP_TOKENS_PER_POSITION
    )
    token_id: int | None = Field(default=None, exclude=True)
    top_token_ids: list[int] = Field(default_factory=list, exclude=True)

    def encode(self, tokenizer) -> None:
        """Resolve the token strings in this position to token ids.

        Accepts the ``token_id:N`` form that vLLM emits for a request with
        ``return_tokens_as_token_ids``, which is how a validator should ask
        for ids rather than text: it is the engine's own contract, it round
        trips through ``resolve_token_id_placeholder``, and it does not
        require changing what every other client sees.

        A bare decimal string is also accepted, since that is what earlier
        validators send. Anything else is treated as text and tokenized.
        """
        parsed = _parse_token_id(self.token)
        if parsed is not None:
            self.token_id = parsed
            self.top_token_ids = [
                tid
                for tid in (_parse_token_id(t) for t in self.top_tokens)
                if tid is not None
            ]
            return

        # Text token: tokenize. A string that does not resolve to at least one
        # id is left out rather than mapped to id 0, which would otherwise be
        # indistinguishable from a genuine id 0 downstream.
        ids = tokenizer.encode(self.token, add_special_tokens=False)
        self.token_id = ids[0] if ids else None
        self.top_token_ids = []
        for t in self.top_tokens:
            t_ids = tokenizer.encode(t, add_special_tokens=False)
            if t_ids:
                self.top_token_ids.append(t_ids[0])


class EnforcedTokens(BaseModel):
    tokens: list[EnforcedToken] = Field(max_length=MAX_ENFORCED_TOKENS)

    def encode(self, tokenizer) -> None:
        for token in self.tokens:
            token.encode(tokenizer)

    @classmethod
    def from_content(cls, content: list[dict[str, Any]]) -> "EnforcedTokens":
        tokens = []
        for position in content:
            token = position["token"]
            top_tokens = [x["token"] for x in position["top_logprobs"]]
            tokens.append(EnforcedToken(token=token, top_tokens=top_tokens))
        return cls(tokens=tokens)

    def get_enforced_token_ids(self) -> list[int]:
        if not self.tokens:
            raise ValueError("Enforced tokens are not encoded")
        ids = [token.token_id for token in self.tokens]
        if any(tid is None for tid in ids):
            raise ValueError("Enforced tokens are not encoded")
        return [tid for tid in ids if tid is not None]

    def detect_logprobs_mode(self, threshold: float = 0.10) -> str | None:
        """Classify original inference logprobs mode from top_token_ids.

        In processed-logprobs results, empty top-k slots are padded with the
        lowest vocab IDs (0-3 for Qwen: ``!``, ``"``, ``#``, ``$``) at
        logprob -9999. This makes ~75% of top_token_ids entries < 4.
        In raw-logprobs results, only ~0.2% of entries are < 4.

        Must be called after encode().

        Returns ``'raw_logprobs'``, ``'processed_logprobs'``, or ``None``
        if there is insufficient data (fewer than 10 top-token entries).
        """
        total = 0
        low_id_count = 0
        for t in self.tokens:
            for tid in t.top_token_ids:
                total += 1
                if tid < 4:
                    low_id_count += 1
        if total < 10:
            return None
        ratio = low_id_count / total
        return "processed_logprobs" if ratio > threshold else "raw_logprobs"


def validate_enforced_token_ids(token_ids: list[int], vocab_size: int) -> None:
    """Reject replay ids the model cannot embed.

    Raises ``ValueError`` so the caller can return a 400. Without this an
    out-of-range id reaches the embedding lookup and aborts the worker.

    Negative values are rejected wholesale: the sampler reserves ``-1`` as the
    "no enforcement at this position" sentinel, so accepting it from a client
    would silently disable enforcement instead of replaying.
    """
    for position, token_id in enumerate(token_ids):
        if not isinstance(token_id, int) or isinstance(token_id, bool):
            raise ValueError(
                f"enforced token at position {position} is not an integer: {token_id!r}"
            )
        if token_id < 0 or token_id >= vocab_size:
            raise ValueError(
                f"enforced token at position {position} is out of range for "
                f"this model: {token_id} not in [0, {vocab_size})"
            )
