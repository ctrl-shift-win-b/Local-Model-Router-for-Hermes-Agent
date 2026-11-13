# Local Model Router for Hermes Agent

OpenAI-compatible router that sits in front of local inference servers.
Hermes Agent talks to one URL. The router picks a backend, starts a local
engine for the request that needs it, and stops that engine after it has
been idle.

This repository is the router only. It does not include model weights or
an inference engine. A backend is any server that speaks
`/v1/chat/completions` and `/v1/models`.

A new Hermes client is [docs/new-hermes-client.md](docs/new-hermes-client.md).
Measured engine choices and how they connect to Hermes are
[docs/engines-and-results.md](docs/engines-and-results.md). Hostnames and
addresses stay in the untracked router file, not in this repository.

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

Backends are tried in the order they appear. For text, that order is the
speed order: fastest first. A remote backend that failed its last probe is
skipped. A managed backend (one with a `command`) can be down; the request
is allowed to start it. `slots` is how many text requests may run on that
backend at once. `0` means unlimited. An idle agent does not occupy a slot.
`weakness` is used when a vision request has to choose which local GPU to
unload. Higher means weaker.

Set `idle_seconds` to `600` for ten minutes with no request, then unload.
`0` leaves a started engine running. Discovery calls such as `GET /v1/props`
do not start an engine and do not reset the idle timer.

A forwarded call sends `X-Router-Scope: local`. The far router then uses
only its own backends, so two routers cannot bounce one request forever.

## Placement

Hermes names an agent on the custom provider that points at this router:

```bash
hermes config set providers.<your-provider>.session_affinity_header X-Router-Session
```

Stock Hermes puts the conversation root in that header, so the main agent
and its delegates share one backend. A turn on this cluster publishes a
different key for each delegate (its subagent id) and keeps the main agent
on the stock key. Auxiliary calls during the turn, such as titles and
compression, inherit the caller's key. The same turn sends `X-Router-Role:
main` or `X-Router-Role: agent`. The role is recorded. It does not cancel
a generation that is already running.

Text placement is sticky until a faster card is free:

- A new agent takes the fastest backend with a free in-flight slot.
- It stays there while that backend is up and every faster backend is full.
- When a faster backend has a free slot, the next request moves. One prefix
  rebuild, then it sticks to the new card.
- A request already streaming on a card runs to completion. The next request
  from that agent is the one that can move.
- If the pinned card is down or returns 502, 503, or 504, the agent moves
  to whichever backend answers. It comes back when the faster card is
  healthy and has room.

`slots` is concurrency, not a reservation. Two sequential agents both use
the fastest card. Two agents at the same time use two cards when the first
card's slots are full.

Vision does not promote. A vision pin stays on the GPU that loaded it.
Only an image or a video inside a message is a vision request. A tool
schema that names `image_url` stays on the text model.
A request with no session header is not pinned; text still prefers a card
that has room. `X-Hermes-Session-Id` is accepted only when
`X-Router-Session` is absent.

`/health` field `sessions` is in-flight text requests. `pins` is the idle
preference count. `local_sessions` is the in-flight count on this machine's
own backends, which is what a peer router reads.

The process idle timer is separate. Retiring a pin does not unload an
engine. The engine stops when it has had no requests for `idle_seconds`.
A live pin still blocks unloading that GPU to make room for vision.

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
