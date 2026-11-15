#!/usr/bin/env python3
"""Directory tests. They do not open a GPU or a real router."""

from __future__ import annotations

import json
import threading
import time
import unittest
from http.client import HTTPConnection
from pathlib import Path
from tempfile import TemporaryDirectory

from registry_server import Directory, choose_host, make_handler, parse_register, ping
from http.server import ThreadingHTTPServer


class DirectoryTest(unittest.TestCase):
    def test_the_same_machine_updates_and_a_new_one_is_added(self) -> None:
        with TemporaryDirectory() as tmp:
            directory = Directory(Path(tmp) / "members.json", 300)
            self.assertEqual(directory.register("alpha", "workstation", "10.0.0.8", 8092, 1000), "added")
            self.assertEqual(directory.register("alpha", "workstation", "10.0.0.9", 8092, 1010), "updated")
            self.assertEqual(directory.register("beta", "frame", "10.0.0.4", 8092, 1010), "added")
            rows = {row["machine_id"]: row for row in directory.members()}
            self.assertEqual(set(rows), {"alpha", "beta"})
            self.assertEqual(rows["alpha"]["host"], "10.0.0.9")

    def test_a_quiet_machine_stays_until_five_minutes_then_can_return(self) -> None:
        with TemporaryDirectory() as tmp:
            directory = Directory(Path(tmp) / "members.json", 300)
            directory.register("alpha", "workstation", "10.0.0.8", 8092, 1000)
            self.assertEqual(directory.sweep(1200, lambda _host, _port: False), [])
            self.assertEqual([row["machine_id"] for row in directory.members()], ["alpha"])
            self.assertEqual(directory.sweep(1300, lambda _host, _port: False), ["alpha"])
            self.assertEqual(directory.members(), [])
            self.assertEqual(directory.register("alpha", "workstation", "10.0.0.10", 8092, 1400), "added")
            self.assertEqual(directory.members()[0]["host"], "10.0.0.10")

    def test_a_successful_ping_refreshes_the_clock(self) -> None:
        with TemporaryDirectory() as tmp:
            directory = Directory(Path(tmp) / "members.json", 300)
            directory.register("alpha", "workstation", "10.0.0.8", 8092, 1000)
            self.assertEqual(directory.sweep(1400, lambda host, port: host == "10.0.0.8"), [])
            self.assertEqual(directory.members()[0]["last_seen"], 1400)

    def test_the_peer_address_wins_over_a_claimed_host(self) -> None:
        machine_id, name, host, port, announced = parse_register(
            b'{"machine_id":"alpha","name":"workstation","host":"1.2.3.4","listen_host":"10.0.0.9","port":8092}',
            "::ffff:10.0.0.8",
        )
        self.assertEqual((machine_id, name, host, port, announced), ("alpha", "workstation", "10.0.0.8", 8092, "10.0.0.9"))

    def test_a_rewritten_source_keeps_the_listen_address_that_answers(self) -> None:
        def gateway_only(host: str, _port: int) -> bool:
            return host == "10.0.0.70"

        self.assertEqual(choose_host("10.0.0.6", "10.0.0.70", 8092, gateway_only), "10.0.0.70")
        self.assertEqual(
            choose_host("10.0.0.6", "", 8092, gateway_only, current="10.0.0.70"),
            "10.0.0.70",
        )
        self.assertEqual(
            choose_host("10.0.0.8", "10.0.0.70", 8092, lambda host, _port: True),
            "10.0.0.8",
        )

        def health(host: str, _port: int, timeout: float = 2.0) -> bool:
            return host == "10.0.0.70" and timeout > 0

        self.assertEqual(choose_host("10.0.0.6", "10.0.0.70", 8092, health), "10.0.0.70")

    def test_http_register_stores_the_connection_address(self) -> None:
        with TemporaryDirectory() as tmp:
            directory = Directory(Path(tmp) / "members.json", 300)
            server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(directory, lambda _host, _port: False))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            host, port = server.server_address[:2]
            try:
                connection = HTTPConnection(host, port, timeout=2)
                body = json.dumps({"machine_id": "alpha", "name": "workstation", "port": 8092}).encode()
                connection.request("POST", "/registry/register", body=body, headers={"Content-Type": "application/json"})
                response = connection.getresponse()
                payload = json.loads(response.read().decode())
                connection.close()
                self.assertEqual(response.status, 200)
                self.assertEqual(payload["status"], "added")
                self.assertEqual(payload["members"][0]["host"], "127.0.0.1")
                self.assertEqual(payload["members"][0]["port"], 8092)
            finally:
                server.shutdown()
                server.server_close()

    def test_ping_is_false_when_nothing_listens(self) -> None:
        self.assertFalse(ping("127.0.0.1", 1, 0.2))

    def test_a_ping_of_the_old_address_does_not_refresh_the_replacement(self) -> None:
        with TemporaryDirectory() as tmp:
            directory = Directory(Path(tmp) / "members.json", 300)
            directory.register("alpha", "workstation", "10.0.0.8", 8092, 1000)

            def reachable(host: str, _port: int) -> bool:
                directory.register("alpha", "workstation", "10.0.0.9", 8092, 1100)
                return host == "10.0.0.8"

            directory.sweep(1200, reachable)
            row = directory.members()[0]
            self.assertEqual(row["host"], "10.0.0.9")
            self.assertEqual(row["last_seen"], 1100)

    def test_a_bad_machine_id_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            parse_register(b'{"machine_id":"has space","port":1}', "10.0.0.1")


if __name__ == "__main__":
    unittest.main()
