"""Bound request bytes before JSON parsing, including streamed/chunked bodies."""

from __future__ import annotations

import asyncio

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send


class RequestBodyLimitMiddleware:
    """Buffer at most max_bytes, then replay a bounded body to the application."""

    def __init__(self, app: ASGIApp, max_bytes: int, timeout_seconds: float = 15):
        self.app = app
        self.max_bytes = max_bytes
        self.timeout_seconds = timeout_seconds

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers", []))
        try:
            declared = int(headers.get(b"content-length", b"0"))
            if declared < 0:
                raise ValueError
        except ValueError:
            await JSONResponse(status_code=400, content={"detail": "Invalid Content-Length."})(
                scope, receive, send
            )
            return
        if declared > self.max_bytes:
            await self._too_large(scope, receive, send)
            return
        chunks = []
        total = 0
        try:
            async with asyncio.timeout(self.timeout_seconds):
                while True:
                    message = await receive()
                    if message["type"] == "http.disconnect":
                        return
                    chunk = message.get("body", b"")
                    total += len(chunk)
                    if total > self.max_bytes:
                        await self._too_large(scope, receive, send)
                        return
                    chunks.append(chunk)
                    if not message.get("more_body", False):
                        break
        except TimeoutError:
            await JSONResponse(status_code=408, content={"detail": "Request body read timed out."})(
                scope, receive, send
            )
            return
        body = b"".join(chunks)
        replayed = False

        async def replay():
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, replay, send)

    async def _too_large(self, scope: Scope, receive: Receive, send: Send) -> None:
        await JSONResponse(
            status_code=413,
            content={"detail": f"Request body exceeds {self.max_bytes} bytes."},
        )(scope, receive, send)
