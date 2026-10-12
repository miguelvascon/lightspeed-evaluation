"""Loopback HTTP server shared by the golden judge and lightspeed-stack fakes.

Public API
----------
``serve(handler)``
    Context manager. Yields ``http://127.0.0.1:<port>``.
``handler(method, path, headers, body) -> (status, headers, body)``
    ``method`` and ``path`` are ``str``, ``headers`` is a ``str`` mapping,
    ``body`` is ``bytes``. The return is ``(int, dict[str, str], bytes)``.
``Router(*handlers)``
    Calls the first handler whose ``handles(path) -> bool`` is true.
``prefix(mount, handler)``
    Route that owns ``mount`` and everything under it. Use ``/v1`` for the
    judge fake and ``/lsc`` for the lightspeed-stack fake, then pass both to
    one ``Router`` and ``serve`` that router.
"""

from __future__ import annotations

import socket
import threading
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Protocol, cast

HeaderMap = Mapping[str, str]
Response = tuple[int, dict[str, str], bytes]
Handler = Callable[[str, str, HeaderMap, bytes], Response]


class Routed(Protocol):
    """Callable route that claims paths via ``handles``."""

    def handles(self, path: str) -> bool:
        """Return True when this route owns ``path``."""
        raise NotImplementedError

    def __call__(
        self, method: str, path: str, headers: HeaderMap, body: bytes
    ) -> Response:
        """Answer one request."""
        raise NotImplementedError


class Router:
    """Dispatch to the first handler whose ``handles`` matches."""

    def __init__(self, *handlers: Routed) -> None:
        """Store routes. Earlier routes win."""
        self._handlers = handlers

    def handles(self, path: str) -> bool:
        """Return True when any route claims ``path``."""
        return any(handler.handles(path) for handler in self._handlers)

    def __call__(
        self, method: str, path: str, headers: HeaderMap, body: bytes
    ) -> Response:
        """Dispatch one request, or 404 when nothing claims the path."""
        for handler in self._handlers:
            if handler.handles(path):
                return handler(method, path, headers, body)
        return 404, {"Content-Type": "text/plain"}, b"no golden route"


class _Prefix:
    """Mount a callable under a path prefix such as ``/v1`` or ``/lsc``."""

    def __init__(self, mount: str, handler: Handler) -> None:
        """Remember the mount point and the wrapped callable."""
        text = mount if mount.startswith("/") else f"/{mount}"
        self._mount = text.rstrip("/") or "/"
        self._handler = handler

    def handles(self, path: str) -> bool:
        """True for the mount itself and for paths nested under it."""
        root = path.split("?", 1)[0].rstrip("/") or "/"
        return root == self._mount or root.startswith(f"{self._mount}/")

    def __call__(
        self, method: str, path: str, headers: HeaderMap, body: bytes
    ) -> Response:
        """Forward the original path to the wrapped handler."""
        return self._handler(method, path, headers, body)


def prefix(mount: str, handler: Handler) -> _Prefix:
    """Return a route for ``mount``. Shared mounts are ``/v1`` and ``/lsc``."""
    return _Prefix(mount, handler)


class _Server(ThreadingHTTPServer):
    """Threading server that keeps the golden handler on the instance."""

    golden_handler: Handler

    def __init__(self, handler: Handler) -> None:
        """Bind 127.0.0.1 on an ephemeral port."""
        self.golden_handler = handler
        super().__init__(("127.0.0.1", 0), _RequestHandler)


class _RequestHandler(BaseHTTPRequestHandler):
    """Read one HTTP request and write the handler response."""

    def handle_one(self) -> None:
        """Dispatch the current request to ``golden_handler``."""
        length = int(self.headers.get("Content-Length", "0") or 0)
        payload = self.rfile.read(length) if length else b""
        incoming = dict(self.headers.items())
        server = cast(_Server, self.server)
        status, out_headers, body = server.golden_handler(
            self.command, self.path, incoming, payload
        )
        self.send_response(status)
        for key, value in out_headers.items():
            self.send_header(key, value)
        lowered = {key.lower() for key in out_headers}
        if "content-length" not in lowered:
            self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_request(self, code: int | str = "-", size: int | str = "-") -> None:
        """Drop the default stderr access log."""
        del code, size


def _bind_verbs() -> None:
    """Map every verb the fakes use onto ``handle_one``."""
    for verb in ("GET", "POST", "PUT", "DELETE"):
        setattr(_RequestHandler, f"do_{verb}", _RequestHandler.handle_one)


_bind_verbs()


def _wait_until_listening(port: int) -> None:
    """Return once the ephemeral port accepts a connection."""
    last_error = "not started"
    for _ in range(50):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.05):
                return
        except OSError as exc:
            last_error = str(exc)
            threading.Event().wait(0.02)
    raise RuntimeError(f"golden server did not listen: {last_error}")


@contextmanager
def serve(handler: Handler) -> Iterator[str]:
    """Serve ``handler`` on 127.0.0.1 and yield ``http://127.0.0.1:<port>``."""
    server = _Server(handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = int(server.server_address[1])
    try:
        _wait_until_listening(port)
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
