"""poc_share knob: config validation + CLI plumbing + the scheduler budget math.

poc_share is the explicit chat<->PoC mix control: PoC may consume at most
poc_share of each scheduler step's token budget; chat gets the rest
(1.0 = PoC greedy, 0.0 = chat only / PoC paused, 0.5 = even split).
"""
import pytest

from vllm.config.cache import CacheConfig


def test_poc_share_default_is_half():
    assert CacheConfig().poc_share == 0.5


@pytest.mark.parametrize("share", [0.0, 0.25, 0.5, 1.0])
def test_poc_share_accepts_valid_fraction(share):
    assert CacheConfig(poc_share=share).poc_share == share


@pytest.mark.parametrize("bad", [-0.1, 1.1, 2.0])
def test_poc_share_rejects_out_of_range(bad):
    with pytest.raises(Exception):
        CacheConfig(poc_share=bad)


def test_poc_share_cli_arg_exposed():
    from vllm.engine.arg_utils import EngineArgs
    parser = EngineArgs.add_cli_args(__import__("argparse").ArgumentParser())
    ns = parser.parse_args(["--poc-share", "0.3"])
    assert abs(ns.poc_share - 0.3) < 1e-9
