#!/usr/bin/env python3
"""Directory of router addresses. Standard library only.

A router POSTs /registry/register when it comes up. The same machine id
replaces the stored address. A new machine id is added. The address is
the one this connection came from when that host answers /health.
When the path rewrites the source, the connection comes from a gateway
that is not the router, so the directory checks the listen address the
router sends and keeps that one if it answers. Every interval this
process asks each stored address for /health. A machine with no
successful register and no successful health check for offline_seconds
is removed. The next POST adds it again.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import threading
import time
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


_MAX_BODY = 4096
_ID_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")


class Directory:
    """machine id -> last address that registered, and when it was last alive."""

    def __init__(self, path: Path, offline_seconds: float) -> None:
        self.path = path
        self.offline_seconds = offline_seconds
        self._lock = threading.Lock()
        self._members: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeError):
            return
        rows = raw.get("members") if isinstance(raw, dict) else None
        if not isinstance(rows, list):
            return
        for row in rows:
            if not isinstance(row, dict):
                continue
            machine_id = str(row.get("machine_id") or "")
            host = str(row.get("host") or "")
            try:
                port = int(row.get("port") or 0)
                last_seen = float(row.get("last_seen") or 0)
            except (TypeError, ValueError):
                continue
            if not _valid_id(machine_id) or not host or not _valid_port(port):
                continue
            self._members[machine_id] = {
                "machine_id": machine_id,
                "name": _clean_name(str(row.get("name") or "")),
                "host": host,
                "port": port,
                "last_seen": last_seen,
            }

    def register(self, machine_id: str, name: str, host: str, port: int, now: float) -> str:
        with self._lock:
            status = "updated" if machine_id in self._members else "added"
            self._members[machine_id] = {
                "machine_id": machine_id,
                "name": name,
                "host": host,
                "port": port,
                "last_seen": now,
            }
            self._save_unlocked()
            return status

    def sweep(self, now: float, reachable) -> list[str]:
        """Refresh anyone ``reachable(host, port)`` can see, then drop the quiet."""
        with self._lock:
            rows = list(self._members.values())
        refreshed = False
        for row in rows:
            try:
                ok = bool(reachable(row["host"], int(row["port"])))
            except Exception:
                ok = False
            if not ok:
                continue
            with self._lock:
                current = self._members.get(row["machine_id"])
                if (
                    current
                    and current["host"] == row["host"]
                    and int(current["port"]) == int(row["port"])
                ):
                    current["last_seen"] = now
                    refreshed = True
        removed: list[str] = []
        with self._lock:
            for machine_id, row in list(self._members.items()):
                if now - float(row["last_seen"]) >= self.offline_seconds:
                    self._members.pop(machine_id, None)
                    removed.append(machine_id)
            if removed or refreshed:
                self._save_unlocked()
        return removed

    def members(self) -> list[dict]:
        with self._lock:
            rows = list(self._members.values())
        rows.sort(key=lambda row: (row["name"], row["machine_id"]))
        return rows

    def _save_unlocked(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"members": list(self._members.values())}, indent=2) + "\n"
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(payload, encoding="utf-8")
        tmp.replace(self.path)


def _valid_id(value: str) -> bool:
    return 1 <= len(value) <= 80 and set(value) <= _ID_CHARS


def _clean_name(value: str) -> str:
    cleaned = "".join(ch for ch in value if ch in _ID_CHARS)
    return cleaned[:64] or "router"


def _valid_port(value: int) -> bool:
    return 1 <= value <= 65535


def _peer_host(value: str) -> str:
    host = value.strip()
    if host.startswith("::ffff:"):
        host = host[7:]
    return host


def _routable_host(value: str) -> str:
    """An address other routers can dial. Loopback and wildcards are not that."""
    host = _peer_host(value)
    try:
        parsed = ipaddress.ip_address(host)
    except ValueError:
        return ""
    if parsed.is_unspecified or parsed.is_multicast or parsed.is_loopback:
        return ""
    return host


def choose_host(
    peer: str,
    announced: str,
    port: int,
    reachable,
    current: str = "",
) -> str:
    """Pick the address other routers should dial.

    The peer wins when it answers. A rewritten source is the gateway, and
    that gateway does not answer the router port, so the listen address
    the router announced is used once it answers. A dead new address does
    not replace one that still answers.
    """
    for host in (peer, announced, current):
        if not host or host in {"0.0.0.0"}:
            continue
        try:
            ok = bool(reachable(host, port))
        except Exception:
            ok = False
        if ok:
            return host
    return peer


def parse_register(body: bytes, peer_host: str) -> tuple[str, str, str, int, str]:
    """Return machine_id, name, peer host, port, and the announced listen address."""
    try:
        data = json.loads(body.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("register body is not JSON") from exc
    if not isinstance(data, dict):
        raise ValueError("register body must be an object")
    machine_id = str(data.get("machine_id") or "")
    if not _valid_id(machine_id):
        raise ValueError("machine_id must be 1-80 letters, digits, dot, underscore, or dash")
    try:
        port = int(data.get("port"))
    except (TypeError, ValueError) as exc:
        raise ValueError("port is required") from exc
    if not _valid_port(port):
        raise ValueError("port is out of range")
    host = _peer_host(peer_host)
    if not host or host == "0.0.0.0":
        raise ValueError("missing peer address")
    return machine_id, _clean_name(str(data.get("name") or "")), host, port, _routable_host(str(data.get("listen_host") or ""))


def ping(host: str, port: int, timeout: float = 2.0) -> bool:
    """True when that router answers GET /health. A refused connection is offline."""
    connection = HTTPConnection(host, port, timeout=timeout)
    try:
        connection.request("GET", "/health")
        response = connection.getresponse()
        response.read()
        return response.status == 200
    except OSError:
        return False
    finally:
        connection.close()


def make_handler(directory: Directory, reachable=ping):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args) -> None:
            return

        def do_GET(self) -> None:  # noqa: N802
            path = urlsplit(self.path).path
            if path == "/health":
                self._write(200, {"ok": True})
                return
            if path == "/registry":
                self._write(200, {"members": directory.members()})
                return
            self._write(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            path = urlsplit(self.path).path
            if path != "/registry/register":
                self._write(404, {"error": "not found"})
                return
            length = int(self.headers.get("Content-Length") or "0")
            if length < 0 or length > _MAX_BODY:
                self._write(400, {"error": "body is too large"})
                return
            body = self.rfile.read(length) if length else b""
            try:
                machine_id, name, peer, port, announced = parse_register(body, self.client_address[0])
            except ValueError as exc:
                self._write(400, {"error": str(exc)})
                return
            current = ""
            for row in directory.members():
                if row.get("machine_id") == machine_id:
                    current = str(row.get("host") or "")
                    break
            host = choose_host(peer, announced, port, reachable, current)
            status = directory.register(machine_id, name, host, port, time.time())
            self._write(200, {"status": status, "members": directory.members()})

        def _write(self, status: int, payload: dict) -> None:
            encoded = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    return Handler


def serve(
    host: str,
    port: int,
    directory: Directory,
    interval: float,
    ping_timeout: float,
) -> None:
    stop = threading.Event()

    def loop() -> None:
        while not stop.is_set():
            directory.sweep(time.time(), lambda item_host, item_port: ping(item_host, item_port, ping_timeout))
            stop.wait(interval)

    threading.Thread(target=loop, name="registry-sweep", daemon=True).start()
    server = ThreadingHTTPServer((host, port), make_handler(directory))

    def _shutdown(signum: int, _frame) -> None:
        stop.set()
        threading.Thread(target=server.shutdown, name="registry-shutdown", daemon=True).start()

    import signal

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)
    try:
        server.serve_forever()
    finally:
        stop.set()
        server.server_close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Keep the current address of each router")
    parser.add_argument("--listen", default="0.0.0.0:8096")
    parser.add_argument("--state", default="members.json")
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--offline", type=float, default=300.0)
    parser.add_argument("--ping-timeout", type=float, default=2.0)
    args = parser.parse_args(argv)
    listen_host, separator, listen_port = args.listen.rpartition(":")
    if not separator:
        raise SystemExit("--listen must be host:port")
    directory = Directory(Path(args.state), args.offline)
    serve(listen_host or "0.0.0.0", int(listen_port), directory, args.interval, args.ping_timeout)


if __name__ == "__main__":
    main()
