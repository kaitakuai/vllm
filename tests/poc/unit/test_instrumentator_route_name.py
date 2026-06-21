"""The prometheus instrumentator's per-request route-name lookup does
`route.path` over app.routes. FastAPI >=0.117 `include_router` inserts
`_IncludedRouter` objects that lack `.path`, so every non-PoC request 500s
(seen only on the docker image: fastapi 0.138 vs 0.116 locally). The fix hardens
the resolver to tolerate missing `.path` and descend into nested routers.
"""
from starlette.routing import Match

from vllm.entrypoints.serve.instrumentator.metrics import (
    _install_path_safe_route_name,
)


class _FullRoute:
    """Minimal stand-in for a matched route."""

    def __init__(self, path=None, routes=None):
        self._path = path
        self.routes = routes
        if path is not None:
            self.path = path  # _IncludedRouter omits this attribute

    def matches(self, scope):
        return Match.FULL, {}


def _resolver():
    import prometheus_fastapi_instrumentator.routing as r

    _install_path_safe_route_name()
    return r._get_route_name


def test_plain_route_resolves_to_path():
    get = _resolver()
    assert get({}, [_FullRoute(path="/v1/chat/completions")]) == "/v1/chat/completions"


def test_included_router_without_path_does_not_crash():
    # _IncludedRouter: matches FULL, no `.path`, wraps the real route in `.routes`.
    inner = _FullRoute(path="/v1/models")
    wrapper = _FullRoute(path=None, routes=[inner])
    get = _resolver()
    # Pre-fix this raised AttributeError: '_IncludedRouter' has no attribute 'path'
    assert get({}, [wrapper]) == "/v1/models"


def test_install_is_idempotent():
    _install_path_safe_route_name()
    get1 = _resolver()
    get2 = _resolver()
    assert get1 is get2
