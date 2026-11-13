# Local Model Router for Hermes Agent

OpenAI-compatible router that sits in front of local inference servers.
Hermes Agent talks to one URL. The router picks a backend, starts a local
engine for the request that needs it, and stops that engine after it has
been idle.

This repository is the router only. It does not include model weights or
an inference engine. A backend is any server that speaks
`/v1/chat/completions` and `/v1/models`.

## Requirements

- Python 3.10 or newer
- `curl`, for the shell helpers
- On Linux, a user systemd instance if you want `ninfer-router start`

The Mac system Python is too old. Point `NINFER_ROUTER_PYTHON` at a
Python 3.10+ binary.

## Configure

Copy one example and edit the hosts, commands, and slot counts:

```bash
mkdir -p "${XDG_CONFIG_HOME:-$HOME/.config}/ninfer-router"
cp examples/router.gpu-host.json \
  "${XDG_CONFIG_HOME:-$HOME/.config}/ninfer-router/router.json"
```

`examples/router.gpu-host.json` is a machine with local engines and one
remote router in front of a faster GPU. `examples/router.fast-gpu.json` is
that faster machine. `examples/router.client.json` is a computer with no
GPU: it listens on `127.0.0.1` and forwards to the other routers.

`ninfer-router` reads, in order:

1. `NINFER_ROUTER_CONFIG`
2. `${XDG_CONFIG_HOME:-$HOME/.config}/ninfer-router/router.json`
3. `./router.local.json`
4. `./router.json`

Those last two names are gitignored. Keep hostnames, addresses, and start
commands there, not in the repository.

Backends are tried in the order they appear. A remote backend that failed
its last probe is skipped. A managed backend (one with a `command`) can be
down; the request is allowed to start it. `slots` is how many conversations
may sit on that backend. `0` means unlimited. `weakness` is used when a
vision request has to choose which local GPU to unload. Higher means weaker.

Set `idle_seconds` to `600` for ten minutes with no request, then unload.
`0` leaves a started engine running. Discovery calls such as `GET /v1/props`
do not start an engine and do not reset the idle timer.

A forwarded call sends `X-Router-Scope: local`. The far router then uses
only its own backends, so two routers cannot bounce one request forever.

## Conversation pin

Hermes can name a conversation. On the custom provider that points at this
router:

```bash
hermes config set providers.<your-provider>.session_affinity_header X-Router-Session
```

Hermes sends the conversation-root key on each model call. Sub-agents and
auxiliary calls for that conversation share the key. The router keeps the
key on the backend that accepted it until the key has been quiet for
`idle_seconds`. A new key takes the first eligible backend that still has
a free slot and stays there. The pin moves only when that backend fails
the request.

A request with no session header follows the plain preference order and
does not take a slot. `X-Hermes-Session-Id` is accepted only when
`X-Router-Session` is absent.

The process idle timer is separate. Retiring a pin does not unload an
engine. The engine stops when it has had no requests for `idle_seconds`.

## Run

```bash
./ninfer-router run
```

The process listens on the `listen` address from the config, by default
port `8092`. Hermes should use `http://127.0.0.1:8092/v1`.

`hermes-launch auto /path/to/hermes ...` starts the router for a chat or
a gateway run, then execs Hermes. Management commands such as `config` and
`status` do not start it.

Install `systemd/ninfer-router.service` as a user unit if the checkout is
at `~/Local-Model-Router-for-Hermes-Agent`. Edit `ExecStart` otherwise.

```bash
systemctl --user daemon-reload
systemctl --user enable --now ninfer-router.service
```

## Tests

```bash
python3 -m unittest test_gpu_router
```

The tests use in-process HTTP servers. They do not start a GPU.

## License

MIT. Copyright (c) 2026 ctrl-shift-win-b.
