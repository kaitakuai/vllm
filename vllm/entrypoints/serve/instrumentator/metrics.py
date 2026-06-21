# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import prometheus_client
import regex as re
from fastapi import FastAPI, Response
from prometheus_client import make_asgi_app
from prometheus_fastapi_instrumentator import Instrumentator
from starlette.routing import Mount

from vllm.v1.metrics.prometheus import get_prometheus_registry


class PrometheusResponse(Response):
    media_type = prometheus_client.CONTENT_TYPE_LATEST


def _install_path_safe_route_name():
    """FastAPI >=0.117 `include_router` inserts `_IncludedRouter` objects (no
    `.path`) into app.routes. prometheus_fastapi_instrumentator's per-request
    route-name lookup does `route.path` unconditionally and 500s every request.
    Harden it: tolerate missing `.path` and descend into nested routers."""
    from prometheus_fastapi_instrumentator import routing as _r
    from starlette.routing import Match

    if getattr(_r, "_vllm_path_safe", False):
        return

    def _safe(scope, routes, route_name=None):
        for route in routes:
            match, child_scope = route.matches(scope)
            if match == Match.FULL:
                path = getattr(route, "path", "") or ""
                sub = getattr(route, "routes", None)
                if sub:
                    child = _safe({**scope, **child_scope}, sub, path)
                    return None if child is None else path + child
                return path or None
            if match == Match.PARTIAL and route_name is None:
                route_name = getattr(route, "path", None)
        return route_name

    _r._get_route_name = _safe
    _r._vllm_path_safe = True


def attach_router(app: FastAPI):
    """Mount prometheus metrics to a FastAPI app."""
    _install_path_safe_route_name()

    registry = get_prometheus_registry()

    # `response_class=PrometheusResponse` is needed to return an HTTP response
    # with header "Content-Type: text/plain; version=0.0.4; charset=utf-8"
    # instead of the default "application/json" which is incorrect.
    # See https://github.com/trallnag/prometheus-fastapi-instrumentator/issues/163#issue-1296092364
    Instrumentator(
        excluded_handlers=[
            "/metrics",
            "/health",
            "/load",
            "/ping",
            "/version",
            "/server_info",
        ],
        registry=registry,
    ).add().instrument(app).expose(app, response_class=PrometheusResponse)

    # Add prometheus asgi middleware to route /metrics requests
    metrics_route = Mount("/metrics", make_asgi_app(registry=registry))

    # Workaround for 307 Redirect for /metrics
    metrics_route.path_regex = re.compile("^/metrics(?P<path>.*)$")
    app.routes.append(metrics_route)
