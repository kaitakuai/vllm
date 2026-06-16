"""Unit tests for routes.check_params_match.

Validates that artifact-defining params (model, seq_len, k_dim, max_tokens) are
matched against the deployed config and rejected with 409 on mismatch. No server
needed — check_params_match is a plain function over request.app.state.
"""
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from vllm.poc.routes import check_params_match, PoCParamsModel


def _request(deployed=None, serving_models=None):
    state = SimpleNamespace(
        openai_serving_models=serving_models,
        poc_deployed=deployed,
    )
    return SimpleNamespace(app=SimpleNamespace(state=state))


def _params(model="m", seq_len=256, k_dim=12, max_tokens=8):
    return PoCParamsModel(model=model, seq_len=seq_len, k_dim=k_dim, max_tokens=max_tokens)


DEPLOYED = {"model": "m", "seq_len": 256, "k_dim": 12, "max_tokens": 8}


class TestCheckParamsMatch:
    def test_no_deployed_config_is_noop(self):
        # No poc_deployed and no serving_models -> nothing to match, no raise.
        check_params_match(_request(), _params())

    def test_matching_params_ok(self):
        check_params_match(_request(deployed=DEPLOYED), _params())

    def test_max_tokens_mismatch_raises_409(self):
        with pytest.raises(HTTPException) as ei:
            check_params_match(_request(deployed=DEPLOYED), _params(max_tokens=16))
        assert ei.value.status_code == 409
        assert "max_tokens" in ei.value.detail["fields"]

    def test_seq_len_mismatch_raises_409(self):
        with pytest.raises(HTTPException) as ei:
            check_params_match(_request(deployed=DEPLOYED), _params(seq_len=128))
        assert "seq_len" in ei.value.detail["fields"]

    def test_k_dim_mismatch_raises_409(self):
        with pytest.raises(HTTPException) as ei:
            check_params_match(_request(deployed=DEPLOYED), _params(k_dim=8))
        assert "k_dim" in ei.value.detail["fields"]

    def test_max_tokens_zero_is_a_real_configured_value(self):
        # max_tokens=0 (prefill-only) must be treated as a value to match, not
        # as "unset" — so a request asking for decode against a prefill-only
        # deployment is rejected.
        deployed = {**DEPLOYED, "max_tokens": 0}
        check_params_match(_request(deployed=deployed), _params(max_tokens=0))
        with pytest.raises(HTTPException) as ei:
            check_params_match(_request(deployed=deployed), _params(max_tokens=4))
        assert "max_tokens" in ei.value.detail["fields"]

    def test_requested_detail_includes_max_tokens(self):
        with pytest.raises(HTTPException) as ei:
            check_params_match(_request(deployed=DEPLOYED), _params(max_tokens=99))
        assert ei.value.detail["requested"]["max_tokens"] == 99
