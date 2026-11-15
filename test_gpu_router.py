#!/usr/bin/env python3
"""Router tests. They use in-process HTTP servers and do not touch the GPUs."""

from __future__ import annotations

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import socket
import sys
import tempfile
import textwrap
import time
from pathlib import Path

from gpu_router import (
    Backend,
    GpuBusy,
    Health,
    Pool,
    Router,
    Supervisor,
    _forward_headers,
    apply_member_origins,
    load_machine_id,
    prepare_body,
    probe,
    request_kind,
    role_of,
    session_of,
)
from hermes_needs_engines import needs_engines


class _Catalog(BaseHTTPRequestHandler):
    model_id = "qwen3.8-27b"
    fail = False
    seen: list[dict] = []

    def log_message(self, fmt: str, *args) -> None:
        return

    def do_GET(self) -> None:  # noqa: N802
        if self.fail:
            self.send_error(503)
            return
        body = json.dumps({"data": [{"id": self.model_id}]}).encode()
        self._send(200, body)

    before_post = None

    def do_POST(self) -> None:  # noqa: N802
        hook = type(self).__dict__.get("before_post")
        if hook is not None:
            hook()
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        self.seen.append(json.loads(raw.decode()))
        if self.fail:
            self.send_error(503)
            return
        if self.path.endswith("/stream"):
            payload = b"data: hi\n\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        self._send(200, json.dumps({"ok": self.model_id, "echo": json.loads(raw)}).encode())

    def _send(self, status: int, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _serve(model_id: str, fail: bool = False, before_post=None) -> tuple[ThreadingHTTPServer, str]:
    class Handler(_Catalog):
        pass

    Handler.model_id = model_id
    Handler.fail = fail
    Handler.seen = []
    Handler.before_post = before_post
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    return server, f"http://127.0.0.1:{port}"


class PrepareBodyTests(unittest.TestCase):
    def test_v100_folds_efforts_the_template_rejects(self) -> None:
        body = json.dumps({"model": "qwen3.8-27b", "reasoning_effort": "high"}).encode()
        folded = json.loads(prepare_body("v100", body, "application/json", ("qwen3.8-27b",)))
        self.assertEqual(folded["reasoning_effort"], "xhigh")

    def test_other_backends_keep_the_client_effort(self) -> None:
        body = json.dumps({"reasoning_effort": "high"}).encode()
        self.assertEqual(prepare_body("3090", body, "application/json", ()), body)
        self.assertEqual(prepare_body("5090", body, "application/json", ()), body)

    def test_xhigh_is_already_valid_on_the_v100(self) -> None:
        body = json.dumps({"reasoning_effort": "xhigh"}).encode()
        self.assertEqual(prepare_body("v100", body, "application/json", ()), body)

    def test_model_id_is_rewritten_to_the_backend_catalog(self) -> None:
        body = json.dumps({"model": "qwen3.8-27b"}).encode()
        rewritten = json.loads(prepare_body("5090", body, "application/json", ("other-id",)))
        self.assertEqual(rewritten["model"], "other-id")


class PoolTests(unittest.TestCase):
    def test_preference_skips_a_down_primary(self) -> None:
        first, origin_a = _serve("remote-model")
        second, origin_b = _serve("qwen3.8-27b")
        self.addCleanup(first.shutdown)
        self.addCleanup(first.server_close)
        self.addCleanup(second.shutdown)
        self.addCleanup(second.server_close)
        pool = Pool([Backend("5090", origin_a), Backend("3090", origin_b)])
        pool.mark("5090", Health(False, (), "down", 0))
        pool.mark("3090", Health(True, ("qwen3.8-27b",), "", 0))
        self.assertEqual([item.name for item in pool.candidates()], ["3090"])

    def test_probe_marks_a_closed_port_down_and_an_open_one_up(self) -> None:
        server, origin = _serve("qwen3.8-27b")
        self.addCleanup(server.shutdown)
        self.addCleanup(server.server_close)
        self.assertTrue(probe(Backend("3090", origin), 1.0).up)
        down = probe(Backend("5090", "http://127.0.0.1:1"), 0.4)
        self.assertFalse(down.up)

    def test_router_forwards_to_the_first_healthy_backend(self) -> None:
        remote, remote_origin = _serve("remote-model")
        local, local_origin = _serve("qwen3.8-27b")
        self.addCleanup(remote.shutdown)
        self.addCleanup(remote.server_close)
        self.addCleanup(local.shutdown)
        self.addCleanup(local.server_close)
        pool = Pool([Backend("5090", remote_origin), Backend("3090", local_origin)])
        router = Router(pool, "qwen3.8-27b", 240000, 1.0, 5.0)
        router.probe_all()
        self.assertEqual(router.health_document()["active"], "5090")
        from gpu_router import RouterHandler, _open

        class Handler(RouterHandler):
            pass

        Handler.router = router
        front = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=front.serve_forever, daemon=True).start()
        self.addCleanup(front.shutdown)
        port = front.server_address[1]
        status, _headers, response = _open(
            Backend("front", f"http://127.0.0.1:{port}"),
            "POST",
            "/v1/chat/completions",
            {"Content-Type": "application/json"},
            json.dumps({"model": "qwen3.8-27b", "reasoning_effort": "xhigh"}).encode(),
            5.0,
        )
        try:
            body = json.loads(response.read().decode())
        finally:
            response._router_connection.close()
        self.assertEqual(status, 200)
        self.assertEqual(body["ok"], "remote-model")
        self.assertEqual(body["echo"]["model"], "remote-model")
        self.addCleanup(front.server_close)

    def test_a_dead_primary_is_skipped_for_the_request(self) -> None:
        local, local_origin = _serve("qwen3.8-27b")
        self.addCleanup(local.shutdown)
        self.addCleanup(local.server_close)
        pool = Pool(
            [Backend("5090", "http://127.0.0.1:1"), Backend("3090", local_origin)]
        )
        # Nothing has been probed, so the request itself has to walk past the
        # closed port instead of trusting a health cache.
        router = Router(pool, "qwen3.8-27b", 240000, 0.4, 5.0)
        from gpu_router import RouterHandler, _open

        class Handler(RouterHandler):
            pass

        Handler.router = router
        front = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=front.serve_forever, daemon=True).start()
        self.addCleanup(front.shutdown)
        self.addCleanup(front.server_close)
        port = front.server_address[1]
        status, _headers, response = _open(
            Backend("front", f"http://127.0.0.1:{port}"),
            "POST",
            "/v1/chat/completions",
            {"Content-Type": "application/json"},
            json.dumps({"model": "qwen3.8-27b"}).encode(),
            5.0,
        )
        try:
            body = json.loads(response.read().decode())
        finally:
            response._router_connection.close()
        self.assertEqual(status, 200)
        self.assertEqual(body["ok"], "qwen3.8-27b")


class NeedsEnginesTests(unittest.TestCase):
    def test_chat_invocations_start_engines(self) -> None:
        self.assertTrue(needs_engines([]))
        self.assertTrue(needs_engines(["chat"]))
        self.assertTrue(needs_engines(["-m", "qwen3.8-27b"]))
        self.assertTrue(needs_engines(["--reasoning", "xhigh"]))
        self.assertTrue(needs_engines(["-z", "ping"]))
        self.assertTrue(needs_engines(["gateway", "run"]))
        self.assertTrue(needs_engines(["serve"]))
        self.assertTrue(needs_engines(["--tui"]))

    def test_management_invocations_do_not(self) -> None:
        self.assertFalse(needs_engines(["--help"]))
        self.assertFalse(needs_engines(["--version"]))
        self.assertFalse(needs_engines(["chat", "--help"]))
        self.assertFalse(needs_engines(["status"]))
        self.assertFalse(needs_engines(["config", "set", "model.default", "x"]))
        self.assertFalse(needs_engines(["doctor"]))
        self.assertFalse(needs_engines(["gateway", "status"]))
        self.assertFalse(needs_engines(["gateway", "stop"]))
        self.assertFalse(needs_engines(["model"]))


_FAKE_SERVER = textwrap.dedent(
    """\
    import json
    import sys
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            return

        def _send(self, payload):
            raw = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            self._send({"data": [{"id": sys.argv[2]}]})

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0") or "0")
            raw = self.rfile.read(length) if length else b"{}"
            self._send({"ok": sys.argv[2], "echo": json.loads(raw.decode() or "{}")})

    ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), Handler).serve_forever()
    """
)


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


class RequestKindTests(unittest.TestCase):
    def test_an_image_part_is_vision_and_image_output_is_not(self) -> None:
        image = json.dumps(
            {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,aa"}}]}]}
        ).encode()
        self.assertEqual(request_kind("POST", "/v1/chat/completions", image, "application/json"), "vision")
        picture = json.dumps({"modalities": ["image"]}).encode()
        self.assertEqual(request_kind("POST", "/v1/chat/completions", picture, "application/json"), "image")
        self.assertEqual(request_kind("POST", "/v1/images/generations", b"{}", "application/json"), "image")
        self.assertEqual(
            request_kind("POST", "/v1/chat/completions", b'{"messages":[{"content":"hi"}]}', "application/json"),
            "text",
        )

    def test_a_tool_schema_that_names_image_url_stays_text(self) -> None:
        body = json.dumps({
            "model": "qwen3.8-27b",
            "messages": [{"role": "user", "content": "good morning"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "vision_analyze",
                    "parameters": {
                        "type": "object",
                        "properties": {"image_url": {"type": "string"}, "question": {"type": "string"}},
                        "required": ["image_url", "question"],
                    },
                },
            }],
        }).encode()
        self.assertEqual(request_kind("POST", "/v1/chat/completions", body, "application/json"), "text")
        picture = json.dumps({
            "messages": [{"role": "tool", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,aa"}},
            ]}],
            "tools": [{"function": {"parameters": {"properties": {"image_url": {"type": "string"}}}}}],
        }).encode()
        self.assertEqual(request_kind("POST", "/v1/chat/completions", picture, "application/json"), "vision")


class ScopeTests(unittest.TestCase):
    def test_local_scope_does_not_call_the_other_router(self) -> None:
        remote, remote_origin = _serve("remote-model")
        local, local_origin = _serve("qwen3.8-27b")
        self.addCleanup(remote.shutdown)
        self.addCleanup(remote.server_close)
        self.addCleanup(local.shutdown)
        self.addCleanup(local.server_close)
        pool = Pool(
            [
                Backend("5090", remote_origin, remote=True, capabilities=("text", "vision")),
                Backend("3090", local_origin),
            ]
        )
        router = Router(pool, "qwen3.8-27b", 240000, 1.0, 5.0)
        router.probe_all()
        from gpu_router import RouterHandler, _open

        class Handler(RouterHandler):
            pass

        Handler.router = router
        front = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=front.serve_forever, daemon=True).start()
        self.addCleanup(front.shutdown)
        self.addCleanup(front.server_close)
        port = front.server_address[1]
        status, _headers, response = _open(
            Backend("front", f"http://127.0.0.1:{port}"),
            "POST",
            "/v1/chat/completions",
            {"Content-Type": "application/json", "X-Router-Scope": "local"},
            json.dumps({"model": "qwen3.8-27b"}).encode(),
            5.0,
        )
        try:
            body = json.loads(response.read().decode())
        finally:
            response._router_connection.close()
        self.assertEqual(status, 200)
        self.assertEqual(body["ok"], "qwen3.8-27b")


class IdleAndVisionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.state = Path(self._tmp.name)
        self.script = self.state / "fake_server.py"
        self.script.write_text(_FAKE_SERVER)
        self.supervisors: list[Supervisor] = []

    def tearDown(self) -> None:
        for supervisor in self.supervisors:
            supervisor.shutdown()

    def _backend(self, name: str, port: int, gpu: str, capabilities: tuple[str, ...], weakness: int) -> Backend:
        return Backend(
            name=name,
            origin=f"http://127.0.0.1:{port}",
            command=(sys.executable, str(self.script), str(port), name),
            capabilities=capabilities,
            gpu=gpu,
            weakness=weakness,
        )

    def _supervisor(self, backends: list[Backend], idle: float = 600.0) -> Supervisor:
        supervisor = Supervisor(backends, idle_seconds=idle, start_timeout=5.0, state_dir=self.state)
        self.supervisors.append(supervisor)
        return supervisor

    def test_an_idle_server_the_router_started_is_stopped(self) -> None:
        port = _free_port()
        backend = self._backend("v100", port, "v100", ("text",), 30)
        supervisor = self._supervisor([backend], idle=0.3)
        supervisor.acquire(backend, lambda item: probe(item, 0.5))
        self.assertTrue(supervisor.running("v100"))
        supervisor.release("v100")
        time.sleep(0.5)
        supervisor.reap_idle()
        self.assertFalse(supervisor.running("v100"))

    def test_a_manual_hold_keeps_the_idle_server(self) -> None:
        port = _free_port()
        backend = self._backend("v100", port, "v100", ("text",), 30)
        hold = self.state / "holds" / "v100" / "manual"
        hold.parent.mkdir(parents=True)
        hold.write_text("")
        supervisor = self._supervisor([backend], idle=0.2)
        supervisor.acquire(backend, lambda item: probe(item, 0.5))
        supervisor.release("v100")
        time.sleep(0.4)
        supervisor.reap_idle()
        self.assertTrue(supervisor.running("v100"))
        supervisor._kill("v100", "test done")

    def test_vision_unloads_the_idle_text_model_on_that_gpu(self) -> None:
        text_port = _free_port()
        vision_port = _free_port()
        text = self._backend("v100", text_port, "v100", ("text",), 30)
        vision = self._backend("vision-v100", vision_port, "v100", ("vision",), 30)
        supervisor = self._supervisor([text, vision])
        probe_fn = lambda item: probe(item, 0.5)
        supervisor.acquire(text, probe_fn)
        supervisor.release("v100")
        supervisor.acquire(vision, probe_fn)
        self.assertFalse(supervisor.running("v100"))
        self.assertTrue(supervisor.running("vision-v100"))
        supervisor.release("vision-v100")

    def test_a_busy_gpu_is_not_unloaded_and_the_error_says_to_retry(self) -> None:
        text_port = _free_port()
        vision_port = _free_port()
        text = self._backend("v100", text_port, "v100", ("text",), 30)
        vision = self._backend("vision-v100", vision_port, "v100", ("vision",), 30)
        supervisor = self._supervisor([text, vision])
        probe_fn = lambda item: probe(item, 0.5)
        supervisor.acquire(text, probe_fn)
        with self.assertRaises(GpuBusy) as caught:
            supervisor.acquire(vision, probe_fn)
        self.assertIn("Try again later", caught.exception.message)
        self.assertTrue(supervisor.running("v100"))
        self.assertFalse(supervisor.running("vision-v100"))
        supervisor.release("v100")

    def test_missing_weights_do_not_unload_a_gpu(self) -> None:
        text_port = _free_port()
        text = self._backend("v100", text_port, "v100", ("text",), 30)
        vision = Backend(
            name="vision-v100",
            origin="http://127.0.0.1:9",
            command=(sys.executable, "-c", "import sys; sys.exit(1)"),
            preflight=(sys.executable, "-c", "import sys; sys.stderr.write('weights are not installed\\n'); sys.exit(1)"),
            capabilities=("vision",),
            gpu="v100",
            weakness=30,
        )
        supervisor = self._supervisor([text, vision])
        probe_fn = lambda item: probe(item, 0.5)
        supervisor.acquire(text, probe_fn)
        supervisor.release("v100")
        with self.assertRaises(Exception) as caught:
            supervisor.acquire(vision, probe_fn)
        self.assertIn("weights are not installed", str(caught.exception))
        self.assertTrue(getattr(caught.exception, "permanent", False))
        self.assertTrue(supervisor.running("v100"))
        supervisor.release("v100")

    def test_http_image_request_is_rejected_without_starting_a_gpu(self) -> None:
        port = _free_port()
        text = self._backend("v100", port, "v100", ("text",), 30)
        supervisor = self._supervisor([text])
        pool = Pool([text])
        router = Router(pool, "qwen3.8-27b", 240000, 0.5, 5.0, supervisor)
        from gpu_router import RouterHandler, _open

        class Handler(RouterHandler):
            pass

        Handler.router = router
        front = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=front.serve_forever, daemon=True).start()
        self.addCleanup(front.shutdown)
        self.addCleanup(front.server_close)
        status, _headers, response = _open(
            Backend("front", f"http://127.0.0.1:{front.server_address[1]}"),
            "POST",
            "/v1/images/generations",
            {"Content-Type": "application/json"},
            b"{}",
            5.0,
        )
        try:
            body = json.loads(response.read().decode())
        finally:
            response._router_connection.close()
        self.assertEqual(status, 503)
        self.assertIn("Try again later", body["error"]["message"])
        self.assertEqual(body["error"]["type"], "image_unavailable")
        self.assertFalse(supervisor.running("v100"))

    def test_a_discovery_get_does_not_start_a_gpu(self) -> None:
        port = _free_port()
        text = self._backend("v100", port, "v100", ("text",), 30)
        supervisor = self._supervisor([text])
        router = Router(Pool([text]), "qwen3.8-27b", 240000, 0.5, 5.0, supervisor)
        from gpu_router import RouterHandler, _open

        class Handler(RouterHandler):
            pass

        Handler.router = router
        front = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=front.serve_forever, daemon=True).start()
        self.addCleanup(front.shutdown)
        self.addCleanup(front.server_close)
        status, _headers, response = _open(
            Backend("front", f"http://127.0.0.1:{front.server_address[1]}"),
            "GET",
            "/v1/props",
            {},
            None,
            5.0,
        )
        try:
            response.read()
        finally:
            response._router_connection.close()
        self.assertEqual(status, 404)
        self.assertFalse(supervisor.running("v100"))

    def test_a_pinned_idle_text_model_is_not_unloaded_for_vision(self) -> None:
        from gpu_router import Affinity

        text_port = _free_port()
        vision_port = _free_port()
        text = self._backend("v100", text_port, "v100", ("text",), 30)
        vision = self._backend("vision-v100", vision_port, "v100", ("vision",), 30)
        supervisor = self._supervisor([text, vision])
        affinity = Affinity(600)
        affinity.note("alpha", "text", "v100")
        router = Router(
            Pool([text, vision]),
            "qwen3.8-27b",
            240000,
            0.5,
            5.0,
            supervisor,
            affinity=affinity,
        )
        probe_fn = lambda item: probe(item, 0.5)
        supervisor.acquire(text, probe_fn)
        supervisor.release("v100")
        status, body = _post(
            router,
            "/v1/chat/completions",
            {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]},
            {"X-Router-Session": "alpha"},
        )
        self.assertEqual(status, 503)
        self.assertIn("running agent", body["error"]["message"])
        self.assertIn("Try again later", body["error"]["message"])
        self.assertTrue(supervisor.running("v100"))
        self.assertFalse(supervisor.running("vision-v100"))

    def test_a_retired_pin_lets_vision_unload_the_text_model(self) -> None:
        from gpu_router import Affinity

        text_port = _free_port()
        vision_port = _free_port()
        text = self._backend("v100", text_port, "v100", ("text",), 30)
        vision = self._backend("vision-v100", vision_port, "v100", ("vision",), 30)
        supervisor = self._supervisor([text, vision])
        affinity = Affinity(0.05)
        affinity.note("alpha", "text", "v100")
        router = Router(
            Pool([text, vision]),
            "qwen3.8-27b",
            240000,
            0.5,
            5.0,
            supervisor,
            affinity=affinity,
        )
        probe_fn = lambda item: probe(item, 0.5)
        supervisor.acquire(text, probe_fn)
        supervisor.release("v100")
        time.sleep(0.08)
        status, body = _post(
            router,
            "/v1/chat/completions",
            {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]},
            {"X-Router-Session": "alpha"},
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["ok"], "vision-v100")
        self.assertFalse(supervisor.running("v100"))
        self.assertTrue(supervisor.running("vision-v100"))


def _post(router: Router, path: str, payload: dict, headers: dict | None = None) -> tuple[int, dict]:
    from gpu_router import RouterHandler, _open

    class Handler(RouterHandler):
        pass

    Handler.router = router
    front = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=front.serve_forever, daemon=True).start()
    try:
        request_headers = {"Content-Type": "application/json"}
        if headers:
            request_headers.update(headers)
        status, _headers, response = _open(
            Backend("front", f"http://127.0.0.1:{front.server_address[1]}"),
            "POST",
            path,
            request_headers,
            json.dumps(payload).encode(),
            5.0,
        )
        try:
            body = json.loads(response.read().decode())
        finally:
            response._router_connection.close()
        return status, body
    finally:
        front.shutdown()
        front.server_close()


class SessionPinTests(unittest.TestCase):
    def test_the_session_header_is_capped_and_the_router_name_wins(self) -> None:
        self.assertIsNone(session_of({}))
        self.assertIsNone(session_of({"X-Router-Session": "  "}))
        self.assertEqual(session_of({"X-Hermes-Session-Id": "phys"}), "phys")
        self.assertEqual(
            session_of({"X-Router-Session": "root", "X-Hermes-Session-Id": "phys"}),
            "root",
        )
        self.assertEqual(len(session_of({"X-Router-Session": "x" * 500}) or ""), 200)
        self.assertEqual(role_of({}), "")
        self.assertEqual(role_of({"X-Router-Role": "main"}), "main")
        self.assertEqual(role_of({"X-Router-Role": "agent"}), "")

    def test_the_session_header_is_forwarded_with_local_scope(self) -> None:
        headers = _forward_headers(
            {
                "X-Router-Session": "abc",
                "Host": "h",
                "X-Router-Scope": "nope",
                "Content-Length": "1",
            },
            b"{}",
            True,
        )
        self.assertEqual(headers["X-Router-Session"], "abc")
        self.assertEqual(headers["X-Router-Scope"], "local")
        self.assertNotIn("Host", headers)

    def test_idle_agents_share_the_fast_card_and_a_busy_card_spreads_them(self) -> None:
        started = threading.Event()
        release = threading.Event()

        def hold() -> None:
            started.set()
            release.wait(5)

        first, origin_a = _serve("fast", before_post=hold)
        second, origin_b = _serve("next")
        self.addCleanup(release.set)
        self.addCleanup(first.shutdown)
        self.addCleanup(first.server_close)
        self.addCleanup(second.shutdown)
        self.addCleanup(second.server_close)
        router = Router(
            Pool(
                [
                    Backend("5090", origin_a, slots=1),
                    Backend("3090", origin_b, slots=1),
                ]
            ),
            "qwen3.8-27b",
            240000,
            1.0,
            5.0,
        )
        router.probe_all()
        body = {"model": "qwen3.8-27b"}
        landed: dict[str, dict] = {}

        def run_alpha() -> None:
            _status, payload = _post(router, "/v1/chat/completions", body, {"X-Router-Session": "alpha"})
            landed["alpha"] = payload

        worker = threading.Thread(target=run_alpha)
        worker.start()
        self.assertTrue(started.wait(5))
        _status, plain = _post(router, "/v1/chat/completions", body, {})
        self.assertEqual(plain["ok"], "next")
        _status, beta = _post(
            router,
            "/v1/chat/completions",
            body,
            {"X-Router-Session": "beta", "X-Router-Role": "agent"},
        )
        self.assertEqual(beta["ok"], "next")
        release.set()
        worker.join(5)
        self.assertFalse(worker.is_alive())
        self.assertEqual(landed["alpha"]["ok"], "fast")
        # Alpha still holds the only fast slot, so the agent that spilled stays.
        _status, stayed_beta = _post(router, "/v1/chat/completions", body, {"X-Router-Session": "beta"})
        self.assertEqual(stayed_beta["ok"], "next")
        _status, stayed = _post(
            router,
            "/v1/chat/completions",
            body,
            {"X-Router-Session": "alpha", "X-Router-Role": "main"},
        )
        self.assertEqual(stayed["ok"], "fast")
        pins = {row["name"]: row["pins"] for row in router.health_document()["backends"]}
        self.assertEqual(pins, {"5090": 1, "3090": 1})
        self.assertEqual(router.health_document()["local_sessions"], 0)

    def test_a_dead_pin_moves_and_returns_when_the_faster_card_is_back(self) -> None:
        first, origin_a = _serve("fast")
        second, origin_b = _serve("next")
        self.addCleanup(first.shutdown)
        self.addCleanup(first.server_close)
        self.addCleanup(second.shutdown)
        self.addCleanup(second.server_close)
        router = Router(
            Pool([Backend("5090", origin_a, slots=1), Backend("3090", origin_b, slots=1)]),
            "qwen3.8-27b",
            240000,
            0.4,
            5.0,
        )
        router.probe_all()
        body = {"model": "qwen3.8-27b"}
        _status, landed = _post(router, "/v1/chat/completions", body, {"X-Router-Session": "alpha"})
        self.assertEqual(landed["ok"], "fast")
        first.RequestHandlerClass.fail = True
        _status, moved = _post(router, "/v1/chat/completions", body, {"X-Router-Session": "alpha"})
        self.assertEqual(moved["ok"], "next")
        first.RequestHandlerClass.fail = False
        router.probe_all()
        _status, returned = _post(router, "/v1/chat/completions", body, {"X-Router-Session": "alpha"})
        self.assertEqual(returned["ok"], "fast")

    def test_the_hermes_header_is_only_a_fallback(self) -> None:
        first, origin_a = _serve("fast")
        second, origin_b = _serve("next")
        self.addCleanup(first.shutdown)
        self.addCleanup(first.server_close)
        self.addCleanup(second.shutdown)
        self.addCleanup(second.server_close)
        router = Router(
            Pool([Backend("5090", origin_a, slots=1), Backend("3090", origin_b, slots=1)]),
            "qwen3.8-27b",
            240000,
            1.0,
            5.0,
        )
        router.probe_all()
        body = {"model": "qwen3.8-27b"}
        _post(router, "/v1/chat/completions", body, {"X-Router-Session": "alpha"})
        # Alpha holds the only fast slot, so the other session takes the next card.
        _status, beta = _post(router, "/v1/chat/completions", body, {"X-Hermes-Session-Id": "beta"})
        self.assertEqual(beta["ok"], "next")
        _status, mixed = _post(
            router,
            "/v1/chat/completions",
            body,
            {"X-Router-Session": "alpha", "X-Hermes-Session-Id": "beta"},
        )
        self.assertEqual(mixed["ok"], "fast")

    def test_a_quiet_session_is_retired_and_zero_idle_keeps_it(self) -> None:
        server, origin = _serve("fast")
        self.addCleanup(server.shutdown)
        self.addCleanup(server.server_close)
        retiring = Router(
            Pool([Backend("5090", origin, slots=1)]),
            "qwen3.8-27b",
            240000,
            1.0,
            5.0,
            idle_seconds=0.05,
        )
        _post(retiring, "/v1/chat/completions", {"model": "qwen3.8-27b"}, {"X-Router-Session": "alpha"})
        self.assertEqual(retiring.affinity.count("5090", "text"), 1)
        self.assertEqual(retiring.health_document()["local_sessions"], 0)
        time.sleep(0.08)
        retiring.affinity.reap()
        self.assertEqual(retiring.affinity.count("5090", "text"), 0)
        kept = Router(
            Pool([Backend("5090", origin, slots=1)]),
            "qwen3.8-27b",
            240000,
            1.0,
            5.0,
            idle_seconds=0,
        )
        _post(kept, "/v1/chat/completions", {"model": "qwen3.8-27b"}, {"X-Router-Session": "alpha"})
        time.sleep(0.05)
        kept.affinity.reap()
        self.assertEqual(kept.affinity.count("5090", "text"), 1)

    def test_discovery_does_not_create_a_pin(self) -> None:
        server, origin = _serve("fast")
        self.addCleanup(server.shutdown)
        self.addCleanup(server.server_close)
        router = Router(Pool([Backend("5090", origin, slots=1)]), "qwen3.8-27b", 240000, 1.0, 5.0)
        from gpu_router import RouterHandler, _open

        class Handler(RouterHandler):
            pass

        Handler.router = router
        front = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=front.serve_forever, daemon=True).start()
        self.addCleanup(front.shutdown)
        self.addCleanup(front.server_close)
        status, _headers, response = _open(
            Backend("front", f"http://127.0.0.1:{front.server_address[1]}"),
            "GET",
            "/health",
            {"X-Router-Session": "alpha"},
            None,
            5.0,
        )
        try:
            body = json.loads(response.read().decode())
        finally:
            response._router_connection.close()
        self.assertEqual(status, 200)
        self.assertEqual(body["local_sessions"], 0)

    def test_remote_session_count_fills_a_slot_and_a_stale_zero_does_not(self) -> None:
        remote, remote_origin = _serve_health(local_sessions=2, health_status=200)
        local, local_origin = _serve("next")
        self.addCleanup(remote.shutdown)
        self.addCleanup(remote.server_close)
        self.addCleanup(local.shutdown)
        self.addCleanup(local.server_close)
        full = Router(
            Pool(
                [
                    Backend("5090", remote_origin, remote=True, slots=2),
                    Backend("3090", local_origin, slots=1),
                ]
            ),
            "qwen3.8-27b",
            240000,
            1.0,
            5.0,
        )
        full.probe_all()
        self.assertTrue(full.pool.health_of("5090").up)
        self.assertEqual(full.pool.health_of("5090").sessions, 2)
        self.assertEqual(full.place("text", False, "gamma")[0].name, "3090")

        quiet, quiet_origin = _serve_health(local_sessions=0, health_status=500)
        other, other_origin = _serve("other")
        self.addCleanup(quiet.shutdown)
        self.addCleanup(quiet.server_close)
        self.addCleanup(other.shutdown)
        self.addCleanup(other.server_close)
        held = Router(
            Pool(
                [
                    Backend("5090", quiet_origin, remote=True, slots=1),
                    Backend("3090", other_origin, slots=1),
                ]
            ),
            "qwen3.8-27b",
            240000,
            1.0,
            5.0,
        )
        held.probe_all()
        self.assertTrue(held.pool.health_of("5090").up)
        self.assertEqual(held.pool.health_of("5090").sessions, 0)
        held.note("alpha", "text", "5090")
        # Alpha's quiet pin holds the only fast slot. Alpha itself still fits.
        self.assertEqual(held.place("text", False, "beta")[0].name, "3090")
        claimed = held.claim(held.place("text", False, "alpha"), "text", "alpha")
        self.assertIsNotNone(claimed)
        assert claimed is not None
        self.assertEqual(claimed.name, "5090")
        self.assertEqual(held.place("text", False, "beta")[0].name, "3090")
        held.release_slot(claimed, "text")
        self.assertEqual(held.place("text", False, "beta")[0].name, "3090")

    def test_a_slower_pin_promotes_only_while_the_faster_card_has_room(self) -> None:
        router = Router(
            Pool(
                [
                    Backend("5090", "http://127.0.0.1:9", slots=1),
                    Backend("3090", "http://127.0.0.1:9", slots=1),
                ]
            ),
            "qwen3.8-27b",
            240000,
            0.2,
            1.0,
        )
        router.note("alpha", "text", "3090")
        self.assertEqual(router.place("text", False, "alpha")[0].name, "5090")
        claimed = router.claim(router.place("text", False, "other"), "text")
        self.assertIsNotNone(claimed)
        assert claimed is not None
        self.assertEqual(claimed.name, "5090")
        self.assertEqual(router.place("text", False, "alpha")[0].name, "3090")
        router.release_slot(claimed, "text")
        self.assertEqual(router.place("text", False, "alpha")[0].name, "5090")

    def test_a_vision_pin_does_not_promote(self) -> None:
        router = Router(
            Pool(
                [
                    Backend("vision-v100", "http://127.0.0.1:9", slots=1, capabilities=("vision",), weakness=30),
                    Backend("vision-3090", "http://127.0.0.1:9", slots=1, capabilities=("vision",), weakness=20),
                ]
            ),
            "qwen3.8-27b",
            240000,
            0.2,
            1.0,
        )
        router.note("alpha", "vision", "vision-3090")
        self.assertEqual(router.place("vision", False, "alpha")[0].name, "vision-3090")

    def test_vision_pins_do_not_consume_text_slots(self) -> None:
        router = Router(
            Pool([Backend("5090", "http://127.0.0.1:9", slots=1), Backend("3090", "http://127.0.0.1:9", slots=1)]),
            "qwen3.8-27b",
            240000,
            0.2,
            1.0,
        )
        router.note("alpha", "vision", "5090")
        self.assertEqual(router.place("text", False, "beta")[0].name, "5090")
        self.assertEqual(router.health_document()["local_sessions"], 0)

    def test_a_main_and_four_agents_spread_across_the_three_cards(self) -> None:
        router = Router(
            Pool(
                [
                    Backend("5090", "http://127.0.0.1:9", slots=2),
                    Backend("3090", "http://127.0.0.1:9", slots=2),
                    Backend("v100", "http://127.0.0.1:9", slots=1),
                ]
            ),
            "qwen3.8-27b",
            240000,
            0.2,
            1.0,
        )
        names = ("main", "a0", "a1", "a2", "a3")
        landed = []
        for name in names:
            chosen = router.place("text", False, name)[0].name
            landed.append(chosen)
            router.note(name, "text", chosen)
        self.assertEqual(landed, ["5090", "5090", "3090", "3090", "v100"])
        again = [router.place("text", False, name)[0].name for name in names]
        self.assertEqual(again, landed)

    def test_a_directory_address_replaces_only_the_matching_backend(self) -> None:
        pool = Pool(
            [
                Backend("5090", "https://10.0.0.1:8092", remote=True, machine_id="fast"),
                Backend("3090", "http://127.0.0.1:8090"),
            ]
        )
        held = pool.backends[0]
        changed = apply_member_origins(
            pool,
            [
                {"machine_id": "fast", "host": "10.0.0.2", "port": 8092},
                {"machine_id": "other", "host": "10.0.0.3", "port": 1},
            ],
        )
        self.assertEqual(changed, ["5090"])
        self.assertEqual(pool.backends[0].origin, "https://10.0.0.2:8092")
        self.assertEqual(pool.backends[1].origin, "http://127.0.0.1:8090")
        self.assertEqual(held.origin, "https://10.0.0.1:8092")
        self.assertEqual(
            apply_member_origins(pool, [{"machine_id": "fast", "host": "10.0.0.2", "port": 8092}]),
            [],
        )
        self.assertEqual(
            apply_member_origins(
                pool,
                [{"machine_id": "fast", "host": "10.0.0.6", "port": 8092}],
                lambda host, _port: host == "10.0.0.2",
            ),
            [],
        )
        self.assertEqual(pool.backends[0].origin, "https://10.0.0.2:8092")

    def test_machine_id_is_created_once_and_a_bad_file_is_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp)
            first = load_machine_id(state)
            self.assertEqual(load_machine_id(state), first)
            (state / "machine-id").write_text("has space\n", encoding="utf-8")
            second = load_machine_id(state)
            self.assertNotEqual(second, "has space")
            self.assertEqual(load_machine_id(state), second)

    def test_publish_registers_this_process_and_adopts_a_peer(self) -> None:
        from http.server import BaseHTTPRequestHandler

        from registry_server import Directory, make_handler

        class Health(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                body = b'{"ok":true}'
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, fmt: str, *args) -> None:
                return

        with tempfile.TemporaryDirectory() as tmp:
            answering = ThreadingHTTPServer(("127.0.0.1", 0), Health)
            threading.Thread(target=answering.serve_forever, daemon=True).start()
            peer_port = answering.server_address[1]
            directory = Directory(Path(tmp) / "members.json", 300)
            directory.register("peer", "fast", "127.0.0.1", peer_port, time.time())
            server = ThreadingHTTPServer(
                ("127.0.0.1", 0),
                make_handler(directory, lambda _host, _port: False),
            )
            threading.Thread(target=server.serve_forever, daemon=True).start()
            port = server.server_address[1]
            pool = Pool(
                [
                    Backend("5090", "http://10.0.0.1:8092", remote=True, machine_id="peer"),
                    Backend("3090", "http://127.0.0.1:8090"),
                ]
            )
            router = Router(
                pool,
                "qwen3.8-27b",
                240000,
                0.2,
                1.0,
                registry_server=f"http://127.0.0.1:{port}",
                machine_id="self-id",
                machine_name="workstation",
            )
            try:
                router.publish_registry(8092)
            finally:
                server.shutdown()
                server.server_close()
                answering.shutdown()
                answering.server_close()
            members = {row["machine_id"]: row for row in directory.members()}
            self.assertEqual(members["self-id"]["host"], "127.0.0.1")
            self.assertEqual(members["self-id"]["port"], 8092)
            self.assertEqual(members["self-id"]["name"], "workstation")
            self.assertEqual(pool.backends[0].origin, f"http://127.0.0.1:{peer_port}")
            self.assertEqual(pool.backends[1].origin, "http://127.0.0.1:8090")

    def test_publish_reports_a_closed_directory(self) -> None:
        router = Router(
            Pool([Backend("3090", "http://127.0.0.1:8090")]),
            "qwen3.8-27b",
            240000,
            0.2,
            1.0,
            registry_server="http://127.0.0.1:1",
            machine_id="self-id",
        )
        with self.assertRaises(OSError):
            router.publish_registry(8092)


def _serve_health(local_sessions: int, health_status: int) -> tuple[ThreadingHTTPServer, str]:
    class Handler(_Catalog):
        pass

    Handler.model_id = "remote-model"
    Handler.fail = False
    Handler.seen = []
    Handler.local_sessions = local_sessions
    Handler.health_status = health_status

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == "/health":
            if self.health_status != 200:
                self.send_error(self.health_status)
                return
            body = json.dumps({"local_sessions": self.local_sessions}).encode()
            self._send(200, body)
            return
        _Catalog.do_GET(self)

    Handler.do_GET = do_GET  # type: ignore[method-assign]
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


if __name__ == "__main__":
    unittest.main()
