"""Expose both llama-server replicas at one OpenAI-compatible base URL."""
from contextlib import asynccontextmanager
from itertools import cycle

import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route
from starlette.types import Receive, Scope, Send

from infer import SERVERS

HOP_BY_HOP_HEADERS = {
    b"connection", b"keep-alive", b"proxy-authenticate", b"proxy-authorization",
    b"proxy-connection", b"te", b"trailer", b"transfer-encoding", b"upgrade",
}


def forwarded_headers(
    headers: list[tuple[bytes, bytes]], exclude: tuple[bytes, ...] = ()
) -> list[tuple[bytes, bytes]]:
    excluded = HOP_BY_HOP_HEADERS.union(exclude)
    for name, value in headers:
        if name.lower() == b"connection":
            excluded.update(token.strip().lower() for token in value.split(b","))
    return [(name.lower(), value) for name, value in headers if name.lower() not in excluded]


class ReplicaResponse(StreamingResponse):
    def __init__(self, response: httpx.Response) -> None:
        super().__init__(response.aiter_raw(), status_code=response.status_code)
        self.raw_headers = forwarded_headers(response.headers.raw)
        self.replica_response = response

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            await self.replica_response.aclose()


@asynccontextmanager
async def lifespan(app: Starlette):
    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0), trust_env=False) as client:
        app.state.client = client
        app.state.replicas = cycle([httpx.URL(server) for server in SERVERS])
        yield


async def proxy(request: Request) -> StreamingResponse | JSONResponse:
    replica = next(request.app.state.replicas)
    path = request.scope.get("raw_path") or httpx.URL(path=request.url.path).raw_path
    url = replica.copy_with(raw_path=replica.raw_path.rstrip(b"/") + path)
    url = url.copy_with(query=request.scope["query_string"] or None)
    upstream_request = httpx.Request(
        method=request.method,
        url=url,
        headers=forwarded_headers(request.headers.raw, (b"host", b"content-length")),
        content=await request.body(),
    )
    try:
        response = await request.app.state.client.send(upstream_request, stream=True)
    except httpx.TimeoutException:
        return JSONResponse(
            {"error": {"message": "llama-server timed out", "type": "server_error"}},
            status_code=504,
        )
    except httpx.RequestError:
        return JSONResponse(
            {"error": {"message": "Could not reach llama-server", "type": "server_error"}},
            status_code=502,
        )
    return ReplicaResponse(response)


app = Starlette(
    routes=[Route("/v1/{path:path}", proxy, methods=["GET", "POST", "PUT", "PATCH", "DELETE"])],
    lifespan=lifespan,
)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)
