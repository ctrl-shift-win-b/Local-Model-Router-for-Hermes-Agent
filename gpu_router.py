#!/usr/bin/env python3
"""OpenAI-compatible router for the local Qwen servers Hermes is allowed to use.

Text preference is the order of the backends in the config: the 5090 router,
then the 3090, then the V100. A remote backend that failed its last probe is
skipped, so a dead machine does not add a connect timeout. Managed backends
stay eligible when they are down, because a request is allowed to start them.

A managed server is started only for a request that selects it, and it is
stopped after it has had no requests for ``idle_seconds`` (default 10 minutes).
The router only kills processes it started. A manual hold under the state
directory keeps that server up.

Requests that carry an image or a video use a vision backend instead of the
text model. The weakest free GPU is unloaded first (the V100, then the 3090,
then a remote machine). If every candidate is busy, held, or missing its
weights, the router returns a 503 Hermes can show, and it says to try again
later. Image generation is a separate capability and gets its own error when
no image engine is configured.

Cross-machine calls send ``X-Router-Scope: local``. The far router then uses
only its own GPUs, so the two routers cannot bounce one request forever.

A client may name a conversation with ``X-Router-Session``. That key stays on
the backend that first accepted it until the key has been quiet for
``idle_seconds``. A new key takes the first eligible backend that has a free
slot (``slots`` in the config; 0 means unlimited). A request with no session
header keeps the plain preference order and does not take a slot. If the
pinned backend is down, the key moves to whichever backend actually answers.
Retiring a pin does not unload a process; the process idle timer still does.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from http.client import HTTPConnection, HTTPException, HTTPSConnection, HTTPResponse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterable, Mapping
from urllib.parse import urlsplit


PUBLIC_MODEL = "qwen3.8-27b"
VISION_MODEL = "qwen3.5-2b"
VISION_CONTEXT = 8192
SCOPE_HEADER = "X-Router-Scope"
SESSION_HEADER = "X-Router-Session"
_FALLBACK_SESSION_HEADER = "X-Hermes-Session-Id"
_SESSION_LIMIT = 200
_HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-connection",
    "transfer-encoding",
    "upgrade",
    "te",
    "trailer",
}
# The V100 server rejects minimal, high, and max. The 3090 folds those onto
# the same three template rungs itself, so only the V100 needs this rewrite.
_V100_EFFORT = {"minimal": "low", "high": "xhigh", "max": "xhigh"}
_VISUAL_INPUTS = {
    "image_url",
    "image",
    "input_image",
    "video_url",
    "video",
    "input_video",
}
_WEIGHTS_MISSING = "weights are not installed"


class RouterError(Exception):
    def __init__(self, message: str, *, retry: bool = True, permanent: bool = False) -> None:
        super().__init__(message)
        self.message = message
        self.retry = retry
        self.permanent = permanent


class GpuBusy(RouterError):
    """The GPU that would serve this request cannot be freed right now."""


class StartFailed(RouterError):
    """The selected server did not come up."""


@dataclass(frozen=True)
class Backend:
    name: str
    origin: str
    remote: bool = False
    command: tuple[str, ...] = ()
    preflight: tuple[str, ...] = ()
    idle_seconds: float | None = None
    cwd: str | None = None
    env: tuple[tuple[str, str], ...] = ()
    capabilities: tuple[str, ...] = ("text",)
    gpu: str | None = None
    weakness: int = 0
    slots: int = 0

    @property
    def managed(self) -> bool:
        return bool(self.command) and not self.remote

    def split(self) -> tuple[str, str, int]:
        parts = urlsplit(self.origin)
        if parts.scheme not in {"http", "https"} or not parts.hostname:
            raise ValueError(f"backend {self.name} origin is not an http(s) URL: {self.origin}")
        port = parts.port or (443 if parts.scheme == "https" else 80)
        return parts.scheme, parts.hostname, port


@dataclass(frozen=True)
class Health:
    up: bool
    models: tuple[str, ...]
    error: str
    checked_at: float
    sessions: int = 0


class Pool:
    """Health of each backend. Probe and request threads share this."""

    def __init__(self, backends: Iterable[Backend]) -> None:
        self.backends = tuple(backends)
        if not self.backends:
            raise ValueError("at least one backend is required")
        names = [backend.name for backend in self.backends]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate backend name in {names}")
        now = time.monotonic()
        self._health = {
            backend.name: Health(False, (), "unchecked", now) for backend in self.backends
        }
        self._lock = threading.Lock()

    def mark(self, name: str, health: Health) -> None:
        with self._lock:
            self._health[name] = health

    def snapshot(self) -> list[tuple[Backend, Health]]:
        with self._lock:
            return [(backend, self._health[backend.name]) for backend in self.backends]

    def health_of(self, name: str) -> Health:
        with self._lock:
            return self._health[name]

    def candidates(self, kind: str = "text", local_only: bool = False) -> list[Backend]:
        """Backends that may serve ``kind``, in configured preference order.

        A remote whose last probe failed is left out. A managed local server
        stays in the list while it is stopped, so the request can start it.
        An unchecked remote is tried once; that is how a not-yet-probed
        machine is still the first choice.
        """
        with self._lock:
            chosen = []
            for backend in self.backends:
                if kind not in backend.capabilities:
                    continue
                if local_only and backend.remote:
                    continue
                health = self._health[backend.name]
                if health.up or health.error == "unchecked" or backend.managed:
                    chosen.append(backend)
            return chosen

    def models_of(self, name: str) -> tuple[str, ...]:
        with self._lock:
            return self._health[name].models


def probe(backend: Backend, timeout: float) -> Health:
    """GET /v1/models. A non-200 or a transport error marks the backend down.

    A remote router that answered the model list also reports how many text
    sessions it has pinned to its own GPUs. A failed ``/health`` read leaves
    that count at 0 and does not mark the backend down.
    """
    try:
        status, _headers, body = request_bytes(
            backend, "GET", "/v1/models", {}, None, timeout
        )
    except (OSError, HTTPException) as exc:
        return Health(False, (), str(exc), time.monotonic())
    if status != 200:
        return Health(False, (), f"HTTP {status}", time.monotonic())
    try:
        payload = json.loads(body.decode("utf-8"))
        rows = payload.get("data", [])
        models = tuple(
            str(row["id"]) for row in rows if isinstance(row, dict) and row.get("id")
        )
    except (UnicodeError, json.JSONDecodeError, TypeError, AttributeError):
        return Health(False, (), "model list is not JSON", time.monotonic())
    if not models:
        return Health(False, (), "model list is empty", time.monotonic())
    sessions = _remote_sessions(backend, timeout) if backend.remote else 0
    return Health(True, models, "", time.monotonic(), sessions)


def _remote_sessions(backend: Backend, timeout: float) -> int:
    try:
        status, _headers, body = request_bytes(backend, "GET", "/health", {}, None, timeout)
    except (OSError, HTTPException):
        return 0
    if status != 200:
        return 0
    try:
        payload = json.loads(body.decode("utf-8"))
        sessions = int(payload.get("local_sessions", 0))
    except (UnicodeError, json.JSONDecodeError, TypeError, ValueError, AttributeError):
        return 0
    return sessions if sessions > 0 else 0


def prepare_body(
    backend_name: str,
    body: bytes,
    content_type: str,
    advertised_models: tuple[str, ...],
) -> bytes:
    """Return the request body to send to this backend.

    JSON object bodies may have the V100 effort names folded and ``model``
    pointed at an id that backend actually serves. Anything else is unchanged.
    """
    if not body:
        return body
    media = content_type.split(";", 1)[0].strip().lower()
    stripped = body.lstrip()
    if media not in {"application/json", ""} and not stripped.startswith((b"{", b"[")):
        return body
    if media and "json" not in media and not stripped.startswith((b"{", b"[")):
        return body
    try:
        payload = json.loads(body)
    except (UnicodeError, json.JSONDecodeError):
        return body
    if not isinstance(payload, dict):
        return body

    changed = False
    if backend_name == "v100":
        effort = payload.get("reasoning_effort")
        folded = _V100_EFFORT.get(effort) if isinstance(effort, str) else None
        if folded is not None:
            payload["reasoning_effort"] = folded
            changed = True
    model = payload.get("model")
    if advertised_models and isinstance(model, str) and model not in advertised_models:
        payload["model"] = advertised_models[0]
        changed = True
    if not changed:
        return body
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def request_kind(method: str, path: str, body: bytes, content_type: str) -> str:
    """Classify a proxied call as text, vision, or image generation.

    Vision is an image or video inside the request. Image generation is a
    request that asks for a picture back, including ``/v1/images``.
    """
    del method, content_type
    route = urlsplit(path).path
    if route.startswith("/v1/images"):
        return "image"
    if not body:
        return "text"
    try:
        payload = json.loads(body)
    except (UnicodeError, json.JSONDecodeError):
        return "text"
    if isinstance(payload, dict) and _wants_image_output(payload):
        return "image"
    if _has_visual_input(payload):
        return "vision"
    return "text"


def _wants_image_output(payload: dict) -> bool:
    for key in ("modalities", "output_modalities"):
        value = payload.get(key)
        if isinstance(value, list) and any(str(item).lower() == "image" for item in value):
            return True
    return False


def _has_visual_input(value: object) -> bool:
    if isinstance(value, dict):
        kind = value.get("type")
        if isinstance(kind, str) and kind.lower() in _VISUAL_INPUTS:
            return True
        if any(key in value for key in ("image_url", "input_image", "video_url")):
            return True
        return any(_has_visual_input(item) for item in value.values())
    if isinstance(value, list):
        return any(_has_visual_input(item) for item in value)
    return False


def default_state_dir() -> Path:
    root = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
    return Path(root) / "hermes-gpus"


class Supervisor:
    """Start managed servers for a request and stop them once they are idle.

    The lock is held across a cold start. A second request waits, then shares
    the server. One GPU is never given two models: an idle server this
    supervisor started is stopped before the new one is spawned.
    """

    def __init__(
        self,
        backends: Iterable[Backend],
        idle_seconds: float,
        start_timeout: float,
        state_dir: Path | None,
        fail_cooldown: float = 30.0,
    ) -> None:
        self.backends = tuple(backends)
        self.idle_seconds = idle_seconds
        self.start_timeout = start_timeout
        self.state_dir = state_dir
        self.fail_cooldown = fail_cooldown
        self._lock = threading.RLock()
        self._procs: dict[str, subprocess.Popen[bytes]] = {}
        self._inflight: dict[str, int] = {}
        self._idle_since: dict[str, float] = {}
        self._fail_until: dict[str, float] = {}
        self._fail_reason: dict[str, str] = {}
        self._starting: set[str] = set()
        self._logs: dict[str, object] = {}
        self._pin_check = lambda _name: False
        self.reclaim_orphans()

    def set_pin_check(self, check) -> None:
        """``check(name)`` is true while a conversation pin still holds that backend."""
        self._pin_check = check

    def running(self, name: str) -> bool:
        with self._lock:
            return self._owned_alive(name)

    def acquire(self, backend: Backend, probe_fn) -> None:
        with self._lock:
            if (
                time.monotonic() < self._fail_until.get(backend.name, 0)
                and not self._owned_alive(backend.name)
            ):
                raise StartFailed(self._fail_reason.get(backend.name, "the last start failed"))
            if self._owned_alive(backend.name):
                self._take(backend)
                return
            self._preflight(backend)
            victim = self._victim(backend, probe_fn)
            if victim:
                self._kill(victim, f"unload for {backend.name}")
                # The process is gone. Give the driver a moment to release the device.
                time.sleep(0.3)
            health = probe_fn(backend)
            if health.up:
                self._take(backend)
                return
            self._starting.add(backend.name)
            try:
                process = self._spawn(backend)
                self._wait_ready(backend, process, probe_fn)
                self._take(backend)
            except Exception as exc:
                self._kill(backend.name, "start failed")
                message = exc.message if isinstance(exc, RouterError) else str(exc)
                permanent = isinstance(exc, StartFailed) and exc.permanent
                self._fail_until[backend.name] = time.monotonic() + self.fail_cooldown
                self._fail_reason[backend.name] = message
                if isinstance(exc, StartFailed):
                    raise
                raise StartFailed(message, permanent=permanent) from exc
            finally:
                self._starting.discard(backend.name)

    def release(self, name: str, stop: bool = False) -> None:
        with self._lock:
            current = self._inflight.get(name, 0)
            if current > 0:
                self._inflight[name] = current - 1
            if self._inflight.get(name, 0) == 0:
                self._idle_since[name] = time.monotonic()
            should_stop = stop and self._inflight.get(name, 0) == 0
        if should_stop:
            self._kill(name, "request failed")

    def reap_idle(self) -> None:
        now = time.monotonic()
        with self._lock:
            victims = []
            for name in list(self._procs):
                if name in self._starting or self._inflight.get(name, 0) > 0:
                    continue
                if self._held(name):
                    continue
                limit = self._idle_limit(name)
                if limit <= 0:
                    continue
                since = self._idle_since.get(name)
                if since is None:
                    self._idle_since[name] = now
                    continue
                if now - since >= limit:
                    victims.append(name)
        for name in victims:
            self._kill(name, "idle")

    def shutdown(self) -> None:
        with self._lock:
            names = list(self._procs)
        for name in names:
            if self._held(name):
                sys.stderr.write(f"leave {name} running; a manual hold is set\n")
                continue
            self._kill(name, "router exit")

    def reclaim_orphans(self) -> None:
        """Stop servers a previous router left behind, unless a manual hold is set."""
        if self.state_dir is None:
            return
        directory = self.state_dir / "supervisors"
        if not directory.is_dir():
            return
        for path in directory.glob("*.pid"):
            name = path.stem
            pid = _read_pid(path)
            if pid is None or not _pid_alive(pid):
                path.unlink(missing_ok=True)
                continue
            if self._held(name):
                sys.stderr.write(f"leave held server {name} (pid {pid})\n")
                continue
            sys.stderr.write(f"stop orphaned {name} (pid {pid})\n")
            _signal_group(pid, signal.SIGTERM)
            deadline = time.monotonic() + 10
            while _pid_alive(pid) and time.monotonic() < deadline:
                time.sleep(0.2)
            if _pid_alive(pid):
                _signal_group(pid, signal.SIGKILL)
            path.unlink(missing_ok=True)

    def _take(self, backend: Backend) -> None:
        self._inflight[backend.name] = self._inflight.get(backend.name, 0) + 1
        self._idle_since.pop(backend.name, None)
        self._fail_until.pop(backend.name, None)

    def _owned_alive(self, name: str) -> bool:
        process = self._procs.get(name)
        return process is not None and process.poll() is None

    def _idle_limit(self, name: str) -> float:
        for backend in self.backends:
            if backend.name == name and backend.idle_seconds is not None:
                return backend.idle_seconds
        return self.idle_seconds

    def _held(self, name: str) -> bool:
        if self.state_dir is None:
            return False
        return (self.state_dir / "holds" / name / "manual").is_file()

    def _siblings(self, backend: Backend) -> list[Backend]:
        if not backend.gpu:
            return []
        return [
            other
            for other in self.backends
            if other.gpu == backend.gpu and other.name != backend.name
        ]

    def _victim(self, backend: Backend, probe_fn) -> str | None:
        for other in self._siblings(backend):
            if self._inflight.get(other.name, 0) > 0:
                raise GpuBusy(
                    f"The {backend.gpu} is busy with {other.name}. Try again later."
                )
            if self._held(other.name):
                raise GpuBusy(
                    f"The {backend.gpu} is held by a manual start of {other.name}. "
                    "Stop that server, or try again later."
                )
            if self._pin_check(other.name):
                raise GpuBusy(
                    f"The {backend.gpu} is assigned to a running agent on {other.name}. "
                    "Try again later."
                )
            if self._owned_alive(other.name):
                return other.name
            try:
                occupied = probe_fn(other).up
            except Exception:
                occupied = False
            if occupied:
                raise GpuBusy(
                    f"The {backend.gpu} is serving {other.name}, which this router did not start. "
                    "Stop that server, or try again later."
                )
        return None

    def _preflight(self, backend: Backend) -> None:
        if not backend.preflight:
            return
        try:
            done = subprocess.run(
                list(backend.preflight),
                check=False,
                capture_output=True,
                text=True,
                timeout=15,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise StartFailed(f"{backend.name} preflight failed: {exc}", permanent=True) from exc
        if done.returncode != 0:
            detail = (done.stderr or done.stdout or "preflight failed").strip()
            raise StartFailed(detail, permanent=_WEIGHTS_MISSING in detail)

    def _spawn(self, backend: Backend) -> subprocess.Popen[bytes]:
        if not backend.command:
            raise StartFailed(f"{backend.name} has no start command", permanent=True)
        sys.stderr.write(f"start {backend.name}\n")
        log_handle = self._open_log(backend.name)
        env = os.environ.copy()
        env.update(dict(backend.env))
        try:
            process = subprocess.Popen(
                list(backend.command),
                cwd=backend.cwd,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except OSError as exc:
            raise StartFailed(f"could not start {backend.name}: {exc}") from exc
        self._procs[backend.name] = process
        self._write_pid(backend.name, process.pid)
        return process

    def _wait_ready(self, backend: Backend, process: subprocess.Popen[bytes], probe_fn) -> None:
        limit = self.start_timeout
        deadline = time.monotonic() + limit
        while time.monotonic() < deadline:
            code = process.poll()
            if code is not None:
                detail = self._tail(backend.name)
                raise StartFailed(
                    f"{backend.name} exited with status {code}. {detail}".strip(),
                    permanent=_WEIGHTS_MISSING in detail,
                )
            health = probe_fn(backend)
            if health.up:
                sys.stderr.write(f"{backend.name} is ready\n")
                return
            time.sleep(0.2)
        detail = self._tail(backend.name)
        raise StartFailed(
            f"{backend.name} was not ready within {limit:.0f}s. {detail}".strip(),
            permanent=_WEIGHTS_MISSING in detail,
        )

    def _kill(self, name: str, reason: str) -> None:
        with self._lock:
            if reason == "idle" and (
                self._inflight.get(name, 0) > 0 or name in self._starting or self._held(name)
            ):
                return
            process = self._procs.get(name)
            if process is None or process.poll() is not None:
                self._procs.pop(name, None)
                self._remove_pid(name)
                self._close_log(name)
                return
            sys.stderr.write(f"stop {name} ({reason})\n")
            _signal_group(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                _signal_group(process.pid, signal.SIGKILL)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    sys.stderr.write(f"{name} did not exit after SIGKILL\n")
            self._procs.pop(name, None)
            self._idle_since.pop(name, None)
            self._remove_pid(name)
            self._close_log(name)

    def _open_log(self, name: str):
        if self.state_dir is None:
            return subprocess.DEVNULL
        directory = self.state_dir / "logs"
        directory.mkdir(parents=True, exist_ok=True)
        handle = open(directory / f"{name}.log", "ab")  # noqa: SIM115
        self._logs[name] = handle
        return handle

    def _close_log(self, name: str) -> None:
        handle = self._logs.pop(name, None)
        if handle not in (None, subprocess.DEVNULL):
            try:
                handle.close()
            except Exception:
                pass

    def _tail(self, name: str) -> str:
        if self.state_dir is None:
            return ""
        path = self.state_dir / "logs" / f"{name}.log"
        try:
            data = path.read_bytes()
        except OSError:
            return ""
        text = data[-2000:].decode("utf-8", "replace").strip()
        return text

    def _write_pid(self, name: str, pid: int) -> None:
        if self.state_dir is None:
            return
        directory = self.state_dir / "supervisors"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{name}.pid").write_text(str(pid))

    def _remove_pid(self, name: str) -> None:
        if self.state_dir is None:
            return
        path = self.state_dir / "supervisors" / f"{name}.pid"
        path.unlink(missing_ok=True)


def _read_pid(path: Path) -> int | None:
    try:
        pid = int(path.read_text().strip())
    except (OSError, ValueError):
        return None
    return pid if pid > 0 else None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _signal_group(pid: int, sig: int) -> None:
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        return
    except PermissionError:
        try:
            os.kill(pid, sig)
        except OSError:
            return


def request_bytes(
    backend: Backend,
    method: str,
    path: str,
    headers: Mapping[str, str],
    body: bytes | None,
    timeout: float,
) -> tuple[int, list[tuple[str, str]], bytes]:
    status, response_headers, response = _open(backend, method, path, headers, body, timeout)
    try:
        payload = response.read()
    finally:
        _close_response(response)
    return status, response_headers, payload


def _connection(backend: Backend, timeout: float) -> HTTPConnection:
    scheme, host, port = backend.split()
    if scheme == "https":
        return HTTPSConnection(host, port, timeout=timeout)
    return HTTPConnection(host, port, timeout=timeout)


def _open(
    backend: Backend,
    method: str,
    path: str,
    headers: Mapping[str, str],
    body: bytes | None,
    timeout: float,
) -> tuple[int, list[tuple[str, str]], HTTPResponse]:
    connection = _connection(backend, timeout)
    try:
        connection.request(method, path, body=body, headers=dict(headers))
        response = connection.getresponse()
    except Exception:
        connection.close()
        raise
    # The caller reads ``response`` and must close the connection.
    response._router_connection = connection  # type: ignore[attr-defined]
    header_rows = [(key, value) for key, value in response.getheaders()]
    return response.status, header_rows, response


class Affinity:
    """Conversation key to backend. The key is opaque; the router does not read prompts.

    A pin retires after ``idle_seconds`` with no successful request for that
    key and kind. Zero disables retirement. Text and vision pins are separate,
    so a vision call does not consume a text slot.
    """

    def __init__(self, idle_seconds: float) -> None:
        self.idle_seconds = idle_seconds
        self._lock = threading.Lock()
        self._pins: dict[tuple[str, str], tuple[str, float]] = {}

    def reap(self) -> None:
        if self.idle_seconds <= 0:
            return
        now = time.monotonic()
        with self._lock:
            stale = [
                key
                for key, (_name, used) in self._pins.items()
                if now - used >= self.idle_seconds
            ]
            for key in stale:
                del self._pins[key]

    def get(self, session: str, kind: str) -> str | None:
        with self._lock:
            row = self._pins.get((session, kind))
        return None if row is None else row[0]

    def note(self, session: str | None, kind: str, backend: str) -> None:
        if not session:
            return
        with self._lock:
            self._pins[(session, kind)] = (backend, time.monotonic())

    def count(self, backend: str, kind: str) -> int:
        with self._lock:
            return sum(
                1
                for (_session, pin_kind), (name, _used) in self._pins.items()
                if pin_kind == kind and name == backend
            )

    def pinned(self, backend: str) -> bool:
        with self._lock:
            return any(name == backend for name, _used in self._pins.values())

    def local_text_sessions(self, names: set[str]) -> int:
        with self._lock:
            return sum(
                1
                for (_session, kind), (name, _used) in self._pins.items()
                if kind == "text" and name in names
            )


class Router:
    def __init__(
        self,
        pool: Pool,
        public_model: str,
        context_length: int,
        probe_timeout: float,
        request_timeout: float,
        supervisor: Supervisor | None = None,
        *,
        idle_seconds: float = 600.0,
        affinity: Affinity | None = None,
    ) -> None:
        self.pool = pool
        self.public_model = public_model
        self.context_length = context_length
        self.probe_timeout = probe_timeout
        self.request_timeout = request_timeout
        self.supervisor = supervisor
        self.affinity = affinity if affinity is not None else Affinity(idle_seconds)
        if supervisor is not None:
            supervisor.set_pin_check(self.affinity.pinned)

    def probe_all(self) -> None:
        for backend, _health in self.pool.snapshot():
            # Probing a stopped server only records that its port is closed.
            # It does not start the process.
            self.pool.mark(backend.name, probe(backend, self.probe_timeout))

    def ordered(self, kind: str, local_only: bool) -> list[Backend]:
        rows = self.pool.candidates(kind, local_only)
        if kind == "text":
            return rows
        health = {backend.name: item for backend, item in self.pool.snapshot()}

        def key(backend: Backend) -> tuple[int, int]:
            # Reuse a vision server that is already up. Otherwise start the
            # weakest GPU. A remote router is the last resort.
            if backend.remote:
                return (2, 0)
            if health[backend.name].up:
                return (0, -backend.weakness)
            return (1, -backend.weakness)

        return sorted(rows, key=key)

    def _occupied(self, backend: Backend, kind: str) -> int:
        local = self.affinity.count(backend.name, kind)
        if not backend.remote:
            return local
        reported = self.pool.health_of(backend.name).sessions
        return local if local > reported else reported

    def place(self, kind: str, local_only: bool, session: str | None) -> list[Backend]:
        """Order backends for one request.

        An existing pin stays first while that backend is still a candidate.
        A new key takes the first candidate with a free slot. No session key
        keeps the plain preference order; the caller must not record a pin.
        """
        self.affinity.reap()
        rows = self.ordered(kind, local_only)
        if not session or not rows:
            return rows
        pinned = self.affinity.get(session, kind)
        if pinned:
            for backend in rows:
                if backend.name == pinned:
                    return [backend, *[item for item in rows if item.name != pinned]]
        chosen = rows[0]
        for backend in rows:
            if backend.slots <= 0 or self._occupied(backend, kind) < backend.slots:
                chosen = backend
                break
        return [chosen, *[item for item in rows if item.name != chosen.name]]

    def note(self, session: str | None, kind: str, backend: str) -> None:
        self.affinity.note(session, kind, backend)

    def activate(self, backend: Backend) -> bool:
        """Start a managed backend if this request needs it.

        The returned flag is true when the caller holds a lease it must release.
        """
        if self.supervisor is None or not backend.managed:
            return False
        self.supervisor.acquire(backend, lambda item: probe(item, self.probe_timeout))
        health = probe(backend, self.probe_timeout)
        self.pool.mark(backend.name, health)
        if not health.up:
            self.supervisor.release(backend.name, stop=True)
            raise StartFailed(health.error or f"{backend.name} is not ready")
        return True

    def release(self, backend: Backend, failed: bool) -> None:
        if self.supervisor is not None:
            self.supervisor.release(backend.name, stop=failed)

    def health_document(self) -> dict:
        local_names = {backend.name for backend in self.pool.backends if not backend.remote}
        rows = []
        active = None
        for backend, health in self.pool.snapshot():
            if active is None and health.up and "text" in backend.capabilities:
                active = backend.name
            rows.append(
                {
                    "name": backend.name,
                    "origin": backend.origin,
                    "up": health.up,
                    "managed": backend.managed,
                    "capabilities": list(backend.capabilities),
                    "gpu": backend.gpu,
                    "models": list(health.models),
                    "error": health.error,
                    "sessions": self.affinity.count(backend.name, "text"),
                    "slots": backend.slots,
                }
            )
        return {
            "active": active,
            "public_model": self.public_model,
            "local_sessions": self.affinity.local_text_sessions(local_names),
            "backends": rows,
        }

    def models_document(self) -> dict:
        data = [
            {
                "id": self.public_model,
                "object": "model",
                "owned_by": "local",
                "context_length": self.context_length,
                "max_model_len": self.context_length,
            }
        ]
        if any("vision" in backend.capabilities for backend in self.pool.backends):
            data.append(
                {
                    "id": VISION_MODEL,
                    "object": "model",
                    "owned_by": "local",
                    "context_length": VISION_CONTEXT,
                    "max_model_len": VISION_CONTEXT,
                }
            )
        return {"object": "list", "data": data}


def session_of(headers: Mapping[str, str]) -> str | None:
    """Conversation key from the request, or None when the client did not name one.

    ``X-Router-Session`` wins. ``X-Hermes-Session-Id`` is only a fallback, so a
    physical session id does not override the conversation-root key.
    """
    raw = ""
    for name in (SESSION_HEADER, _FALLBACK_SESSION_HEADER):
        value = headers.get(name)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            raw = text
            break
    if not raw:
        return None
    cleaned = "".join(ch for ch in raw if ch.isprintable() and ch not in "\r\n")
    cleaned = cleaned.strip()
    if not cleaned:
        return None
    return cleaned[:_SESSION_LIMIT]


def _forward_headers(source: Mapping[str, str], body: bytes | None, remote: bool) -> dict[str, str]:
    headers = {
        key: value
        for key, value in source.items()
        if key.lower() not in _HOP_BY_HOP
        and key.lower() != "host"
        and key.lower() != "content-length"
        and key.lower() != SCOPE_HEADER.lower()
    }
    if body is not None:
        headers["Content-Length"] = str(len(body))
    if remote:
        headers[SCOPE_HEADER] = "local"
    return headers


def _send_upstream(
    handler: BaseHTTPRequestHandler,
    status: int,
    headers: list[tuple[str, str]],
    response: HTTPResponse,
) -> None:
    handler.send_response(status)
    length = None
    for key, value in headers:
        if key.lower() in _HOP_BY_HOP:
            continue
        if key.lower() == "content-length":
            length = int(value)
        handler.send_header(key, value)
    if length is None:
        handler.send_header("Transfer-Encoding", "chunked")
    handler.end_headers()
    try:
        if length is not None:
            remaining = length
            while remaining:
                chunk = response.read(min(65536, remaining))
                if not chunk:
                    break
                handler.wfile.write(chunk)
                remaining -= len(chunk)
        else:
            while True:
                chunk = response.read(65536)
                if not chunk:
                    break
                handler.wfile.write(f"{len(chunk):X}\r\n".encode("ascii"))
                handler.wfile.write(chunk)
                handler.wfile.write(b"\r\n")
            handler.wfile.write(b"0\r\n\r\n")
        handler.wfile.flush()
    finally:
        connection = getattr(response, "_router_connection", None)
        if connection is not None:
            connection.close()


def _close_response(response: HTTPResponse) -> None:
    connection = getattr(response, "_router_connection", None)
    try:
        response.read()
    except Exception:
        pass
    if connection is not None:
        connection.close()


def _unavailable(
    kind: str,
    busy: list[str],
    failed: list[str],
    last_error: str,
) -> tuple[int, dict]:
    if kind == "vision":
        reason = " ".join(failed + busy).strip()
        if not reason:
            reason = (
                "No vision engine is configured. Qwen3.5-2B has to be loaded on a "
                "free GPU, weakest first, before an image can be read."
            )
        message = (
            "Vision is not available right now. Reading an image unloads the text "
            "model on the weakest free GPU and loads Qwen3.5-2B there "
            "(V100, then the RTX 3090, then the 5090). "
            f"{reason} Try again later."
        )
        error_type = "vision_unavailable"
    elif kind == "image":
        reason = " ".join(failed + busy).strip()
        if not reason:
            reason = "No image engine is configured, so a picture cannot be generated."
        message = f"Image generation is not available. {reason} Try again later."
        error_type = "image_unavailable"
    else:
        detail = " ".join(failed + busy).strip() or last_error
        message = f"no inference backend is available ({detail})"
        if busy:
            message += " Try again later."
        error_type = "backend_unavailable"
    return 503, {"error": {"message": message, "type": error_type}}


class RouterHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    router: Router

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def do_GET(self) -> None:  # noqa: N802
        self._dispatch()

    def do_POST(self) -> None:  # noqa: N802
        self._dispatch()

    def do_DELETE(self) -> None:  # noqa: N802
        self._dispatch()

    def _dispatch(self) -> None:
        path = urlsplit(self.path).path
        if self.command == "GET" and path == "/health":
            self._write_json(200, self.router.health_document())
            return
        if self.command == "GET" and path == "/v1/models":
            self._write_json(200, self.router.models_document())
            return
        if not path.startswith("/v1/"):
            self._write_json(404, {"error": {"message": "not found", "type": "not_found"}})
            return
        # Hermes probes /v1/props and similar discovery URLs. Those must not
        # load a model. Only a POST carries a completion, a vision request,
        # or an image generation.
        if self.command != "POST":
            self._write_json(404, {"error": {"message": "not found", "type": "not_found"}})
            return
        length = int(self.headers.get("Content-Length", "0") or "0")
        body = self.rfile.read(length) if length else b""
        self._proxy(body)

    def _proxy(self, body: bytes) -> None:
        content_type = self.headers.get("Content-Type", "")
        kind = request_kind(self.command, self.path, body, content_type)
        local_only = (self.headers.get(SCOPE_HEADER) or "").strip().lower() == "local"
        session = session_of(self.headers)
        candidates = self.router.place(kind, local_only, session)
        busy: list[str] = []
        failed: list[str] = []
        last_error = "no backend is reachable"
        if not candidates:
            self._write_json(*_unavailable(kind, busy, failed, last_error))
            return
        for backend in candidates:
            leased = False
            try:
                leased = self.router.activate(backend)
            except GpuBusy as exc:
                busy.append(exc.message)
                continue
            except StartFailed as exc:
                failed.append(exc.message)
                if exc.permanent:
                    break
                continue
            outgoing = prepare_body(
                backend.name, body, content_type, self.router.pool.models_of(backend.name)
            )
            headers = _forward_headers(self.headers, outgoing if outgoing else None, backend.remote)
            sent = False
            try:
                try:
                    status, response_headers, response = _open(
                        backend,
                        self.command,
                        self.path,
                        headers,
                        outgoing if outgoing else None,
                        self.router.request_timeout,
                    )
                except (OSError, HTTPException) as exc:
                    self.router.pool.mark(
                        backend.name, Health(False, (), str(exc), time.monotonic())
                    )
                    last_error = f"{backend.name}: {exc}"
                    continue
                if status in {502, 503, 504}:
                    _close_response(response)
                    self.router.pool.mark(
                        backend.name, Health(False, (), f"HTTP {status}", time.monotonic())
                    )
                    last_error = f"{backend.name}: HTTP {status}"
                    continue
                sys.stderr.write(
                    f"route {self.command} {urlsplit(self.path).path} -> {backend.name} {status}\n"
                )
                self.router.note(session, kind, backend.name)
                _send_upstream(self, status, response_headers, response)
                sent = True
                return
            finally:
                if leased:
                    self.router.release(backend, failed=not sent)
        self._write_json(*_unavailable(kind, busy, failed, last_error))

    def _write_json(self, status: int, payload: dict) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def _parse_backend(text: str) -> Backend:
    name, separator, origin = text.partition("=")
    if not separator or not name or not origin:
        raise argparse.ArgumentTypeError("backend must be name=http://host:port")
    return Backend(name.strip(), origin.strip().rstrip("/"))


def _backend_from_json(row: dict) -> Backend:
    if not isinstance(row, dict):
        raise ValueError("each backend must be an object")
    name = str(row["name"])
    origin = str(row["origin"]).rstrip("/")
    command = tuple(str(part) for part in row.get("command", []))
    preflight = tuple(str(part) for part in row.get("preflight", []))
    env_row = row.get("env") or {}
    if not isinstance(env_row, dict):
        raise ValueError(f"backend {name} env must be an object")
    capabilities = tuple(str(item) for item in row.get("capabilities", ["text"]))
    idle = row.get("idle_seconds")
    return Backend(
        name=name,
        origin=origin,
        remote=bool(row.get("remote", False)),
        command=command,
        preflight=preflight,
        idle_seconds=None if idle is None else float(idle),
        cwd=row.get("cwd"),
        env=tuple((str(key), str(value)) for key, value in env_row.items()),
        capabilities=capabilities or ("text",),
        gpu=row.get("gpu"),
        weakness=int(row.get("weakness", 0)),
        slots=int(row.get("slots", 0)),
    )


def load_router_config(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read router config {path}: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("backends"), list):
        raise SystemExit(f"{path} must be an object with a backends list")
    return data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Route Hermes to the 5090, 3090, or V100")
    parser.add_argument("--config", help="JSON file describing backends, idle time, and the listen address")
    parser.add_argument("--listen", default="127.0.0.1:8092")
    parser.add_argument("--public-model", default=PUBLIC_MODEL)
    parser.add_argument("--context-length", type=int, default=240000)
    parser.add_argument("--probe-timeout", type=float, default=1.0)
    parser.add_argument("--probe-interval", type=float, default=2.0)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--idle-seconds", type=float, default=600.0)
    parser.add_argument("--start-timeout", type=float, default=180.0)
    parser.add_argument("--fail-cooldown", type=float, default=30.0)
    parser.add_argument("--state-dir", default="")
    parser.add_argument("--backend", action="append", type=_parse_backend)
    return parser


def router_from_args(args: argparse.Namespace) -> tuple[Router, str, int, float]:
    state_text = args.state_dir
    idle_seconds = args.idle_seconds
    start_timeout = args.start_timeout
    fail_cooldown = args.fail_cooldown
    listen = args.listen
    public_model = args.public_model
    context_length = args.context_length
    probe_timeout = args.probe_timeout
    probe_interval = args.probe_interval
    request_timeout = args.request_timeout
    if args.config:
        data = load_router_config(Path(args.config))
        backends = tuple(_backend_from_json(row) for row in data["backends"])
        listen = str(data.get("listen", listen))
        public_model = str(data.get("public_model", public_model))
        context_length = int(data.get("context_length", context_length))
        probe_timeout = float(data.get("probe_timeout", probe_timeout))
        probe_interval = float(data.get("probe_interval", probe_interval))
        request_timeout = float(data.get("request_timeout", request_timeout))
        idle_seconds = float(data.get("idle_seconds", idle_seconds))
        start_timeout = float(data.get("start_timeout", start_timeout))
        fail_cooldown = float(data.get("fail_cooldown", fail_cooldown))
        if not state_text:
            state_text = str(data.get("state_dir", ""))
    else:
        if not args.backend:
            raise SystemExit("pass --config or at least one --backend name=url")
        backends = tuple(args.backend)
    state_dir = Path(state_text).expanduser() if state_text else default_state_dir()
    pool = Pool(backends)
    supervisor = Supervisor(
        backends,
        idle_seconds=idle_seconds,
        start_timeout=start_timeout,
        state_dir=state_dir,
        fail_cooldown=fail_cooldown,
    )
    router = Router(
        pool,
        public_model,
        context_length,
        probe_timeout,
        request_timeout,
        supervisor,
        idle_seconds=idle_seconds,
    )
    host, separator, port_text = listen.rpartition(":")
    if not separator:
        raise SystemExit("--listen must be host:port")
    return router, host or "127.0.0.1", int(port_text), probe_interval


def serve(router: Router, host: str, port: int, probe_interval: float) -> None:
    stop = threading.Event()

    def probe_loop() -> None:
        while not stop.is_set():
            try:
                router.probe_all()
            except Exception as exc:
                sys.stderr.write(f"probe: {exc}\n")
            stop.wait(probe_interval)

    def idle_loop() -> None:
        while not stop.is_set():
            try:
                router.affinity.reap()
            except Exception as exc:
                sys.stderr.write(f"session reap: {exc}\n")
            if router.supervisor is not None:
                try:
                    router.supervisor.reap_idle()
                except Exception as exc:
                    sys.stderr.write(f"idle reap: {exc}\n")
            stop.wait(1.0)

    threading.Thread(target=probe_loop, name="gpu-router-probe", daemon=True).start()
    threading.Thread(target=idle_loop, name="gpu-router-idle", daemon=True).start()
    router.probe_all()

    class BoundHandler(RouterHandler):
        pass

    BoundHandler.router = router
    server = ThreadingHTTPServer((host, port), BoundHandler)

    def _shutdown(signum: int, _frame) -> None:
        # shutdown() waits for serve_forever, which is this thread. A helper
        # thread has to call it or the signal handler deadlocks.
        sys.stderr.write(f"gpu router caught signal {signum}\n")
        stop.set()
        threading.Thread(target=server.shutdown, name="gpu-router-shutdown", daemon=True).start()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)
    sys.stderr.write(f"gpu router listening on http://{host}:{port}\n")
    try:
        server.serve_forever()
    finally:
        stop.set()
        if router.supervisor is not None:
            router.supervisor.shutdown()
        server.server_close()


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    router, host, port, probe_interval = router_from_args(args)
    serve(router, host, port, probe_interval)


if __name__ == "__main__":
    main()
