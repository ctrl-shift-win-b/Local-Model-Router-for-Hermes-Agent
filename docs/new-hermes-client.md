# New Hermes client and router

This is the setup for a new PC or Mac that should talk to the local Qwen
cluster the way the existing Hermes installs do. Hermes uses one local
provider. The router on that computer listens on `127.0.0.1:8092` and
forwards to the GPU hosts. A machine that owns GPUs uses the same Hermes
profile and a different router file. Serve flags, weights, and the bake-off
are in [engines-and-results.md](engines-and-results.md).

Hostnames, addresses, and start commands stay in the untracked router file.
The examples in this repo use placeholder hosts.

## Code to install

Clone this repository and use that checkout. `gpu_router.py` from this
commit hashes to:

```text
5da2ed8544f1033188489f3028d0a6616e212112d805a80df3abfadc5498dcb0
```

Python 3.10 or newer is required. macOS system Python is too old. Point
`NINFER_ROUTER_PYTHON` at the Hermes interpreter under
`~/.hermes/tools/` (the `python-3.14*` binary for that OS). Linux
`/usr/bin/python3` is fine when it is 3.10 or newer.

From the repo directory:

```bash
python3 -m unittest test_gpu_router
```

The tests use in-process HTTP servers. They do not start a GPU.

## Router file

```bash
mkdir -p "${XDG_CONFIG_HOME:-$HOME/.config}/ninfer-router"
cp examples/router.client.json \
  "${XDG_CONFIG_HOME:-$HOME/.config}/ninfer-router/router.json"
chmod 600 "${XDG_CONFIG_HOME:-$HOME/.config}/ninfer-router/router.json"
```

Edit that copy. `examples/router.client.json` is the no-GPU shape:

- `listen` is `127.0.0.1:8092`. A client does not accept LAN connections.
- `public_model` is `qwen3.8-27b`, `context_length` is `240000`.
- `idle_seconds` is `600`. Leave it there. `0` would keep a started engine
  loaded.
- Backends are remote, fastest first. On this cluster that is the fast GPU
  host (`slots` 2, `capabilities` `["text"]`), then the workstation
  (`slots` 3, `capabilities` `["text", "vision", "image"]`). Slot 3 on the
  workstation is the sum of its text slots (2 + 1).
- Replace `fast-gpu.example` and `workstation.example` with the real router
  URLs. Each of those machines listens on port `8092`.

`ninfer-router` reads, in order: `NINFER_ROUTER_CONFIG`, then
`${XDG_CONFIG_HOME:-$HOME/.config}/ninfer-router/router.json`, then
`./router.local.json`, then `./router.json`. The last two names are
gitignored.

Give `vision` and `image` only to a router that actually has those engines.
On this cluster the fast GPU host is text. Vision is the workstation.
Advertising vision on the fast GPU host sends pictures to a machine that
does not serve them.

A GPU host uses `examples/router.gpu-host.json` or
`examples/router.fast-gpu.json` and listens on `0.0.0.0:8092` so peers can
probe it. The V100 text backend must be named exactly `v100`. The router
rewrites `reasoning_effort` only for that name (`minimal` to `low`, `high`
and `max` to `xhigh`). Any other name forwards the effort unchanged, and
the V100 server rejects `minimal`, `high`, and `max`.

## Keep the router running

Hermes talks to `http://127.0.0.1:8092/v1`. A `GET` (`/health`,
`/v1/models`, `/v1/props`) does not start an engine and does not reset the
idle timer. The first `POST /v1/*` that selects a backend is what starts it.
The engine stops after 600 seconds with no requests. The router kills only
processes it spawned.

Restart a router only when no chat is in flight. A restart stops the
engines that router started. The first request after a fast restart can
race the bind. Retry that request.

### Linux

The unit assumes the checkout is `~/Local-Model-Router-for-Hermes-Agent`.
Edit `ExecStart` and `WorkingDirectory` when it lives somewhere else.

```bash
mkdir -p "${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
cp systemd/ninfer-router.service \
  "${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user/"
systemctl --user daemon-reload
systemctl --user enable --now ninfer-router.service
```

`KillMode=process`, so a server you started by hand is outside this unit.
A host that other machines probe should also run `loginctl enable-linger`
for the account that owns the unit, so the router comes back at boot.
A client that only listens on loopback can stay a user unit without linger.

`hermes-launch auto /path/to/hermes ...` starts that unit for a chat or a
gateway run, then execs Hermes. `config`, `status`, and other management
commands do not start it. On this cluster the Linux Hermes commands are
wrapped that way. Wrapping is optional when the unit is already active:
Hermes only needs the base URL.

### macOS

There is no systemd. `ninfer-router start` calls `systemctl` and will not
launch the process. Run `ninfer-router run` from a LaunchAgent. Copy
`examples/com.hermes.ninfer-router.plist` to `~/Library/LaunchAgents/`,
replace `REPLACE_HOME` and the Python path, then:

```bash
launchctl bootstrap gui/$(id -u) \
  ~/Library/LaunchAgents/com.hermes.ninfer-router.plist
```

Restart it with:

```bash
launchctl kickstart -k gui/$(id -u)/com.hermes.ninfer-router
```

`hermes-launch` on macOS still tries `systemctl`. That call fails, and the
wrapper then waits for `http://127.0.0.1:8092/health`. The LaunchAgent has
to be up first. Leave the agent loaded. A Hermes exit does not stop it.

A non-interactive shell often misses `~/.local/bin`. The Hermes command on
the existing Mac is `~/.local/bin/hermes`.

## Hermes profile

Install Hermes the usual way, then set this profile. It is the same on
every machine in the cluster. The selected model is local. Auxiliary calls
stay on that same provider. Cloud keys may remain in the environment. They
are not the route.

Leave `fallback_providers` unset. Hermes fallback switches after one failed
request and stays there for the session. The router already fails over
among local backends.

Leave `delegation.provider`, `delegation.model`, and `delegation.base_url`
unset, and leave any advanced-delegation model map unset. Children then
inherit `qwen3.8-27b` at `http://127.0.0.1:8092/v1` with reasoning `xhigh`.
Setting those keys sends children somewhere else.

`hermes config set` on some builds warns that `agent.reasoning_effort` is
unrecognized and suggests `agent.reasoning_echo`. The key the agent reads
is `agent.reasoning_effort`. If the CLI will not write a key, put it in
`~/.hermes/config.yaml`. Use Hermes' own writer when you can
(`utils.atomic_roundtrip_yaml_update` after `hermes_bootstrap`, or
`hermes_cli.config_backend.write_config_keys` on a newer tree). That keeps
comments and the file mode. Copy the live file aside before the first edit.

| Key | Value |
| --- | --- |
| `model.default` | `qwen3.8-27b` |
| `model.provider` | `custom:qwen-local` |
| `model.base_url` | `http://127.0.0.1:8092/v1` |
| `model.context_length` | `240000` |
| `providers.qwen-local.api` | `http://127.0.0.1:8092/v1` |
| `providers.qwen-local.transport` | `chat_completions` |
| `providers.qwen-local.default_model` | `qwen3.8-27b` |
| `providers.qwen-local.context_length` | `240000` |
| `providers.qwen-local.session_affinity_header` | `X-Router-Session` |
| `agent.reasoning_effort` | `xhigh` |
| `agent.max_turns` | `90` |
| `compression.threshold` | `0.757` |
| `compression.target_ratio` | `0.20` |
| `compression.protect_last_n` | `20` |
| `compression.protect_first_n` | `3` |
| `compression.idle_compact_after_seconds` | `0` |
| `compression.proactive_prune_tokens` | `0` |
| `web.backend` | `ddgs` |

`0.757` is about 181,700 tokens of the 240,000 window. Hermes raises any
window under 512,000 to at least `0.75`. `0.757` is already above that
floor. The summary call is capped by
`auxiliary.compression.extra_body.max_tokens: 32768`. That number is only
the summary request. The compressor does not reserve `max_tokens` on this
custom provider. The vision model is 8,192 tokens and is not a compaction
target. The shared 240,000 ceiling is what matters: a session can move to
another card, and the 3090's extra tokens above 240,000 are unused.

These seven auxiliary tasks are `provider: main`, so they stay on the
router. Compression also uses a lower effort:

```yaml
auxiliary:
  vision:
    provider: main
  web_extract:
    provider: main
  tts_audio_tags:
    provider: main
  title_generation:
    provider: main
  session_search:
    provider: main
  background_review:
    provider: main
  compression:
    provider: main
    reasoning_effort: low
    extra_body:
      max_tokens: 32768
```

`auxiliary.vision.provider: main` forces the text path. Hermes still sends
the picture to Qwen3.5-2B as a second, short request (the question plus the
image, no agent system prompt and no tool list). The 27B never sees pixels.
Leave `verify_on_stop` unset. Leave `display.personality` unset, so
`agent.system_prompt` is the prompt that is always injected.
`agent.coding_instructions` is added only in the coding-posture block.

`agent.disabled_toolsets`:

```yaml
disabled_toolsets:
  - bot_room
  - browser
  - catalog
  - computer_use
  - connections
  - desktop_ui
  - discord
  - discord_admin
  - feishu_doc
  - feishu_drive
  - kanban
  - onboarding
  - project
  - stt
  - video
  - video_gen
```

Web, terminal, file, skills, and vision stay enabled. The CLI toolset
`hermes-cli` includes `web_search` and `web_extract`. Browser stays
disabled.

`agent.system_prompt` (always on):

```text
Use a skill only for a real coding job. If the user is not asking you to write, change, debug, review, or test code, do not call skill_view. Questions, research, and ordinary chat stay skill-free.

When the request is a coding job, call skill_view on exactly one matching skill before the first edit, then follow it. New behavior or a behavior change uses test-driven-development. A bug, a failure, or something that does not work uses systematic-debugging. A request to review, commit, or ship uses requesting-code-review. A request to simplify, clean up, or refactor code that already works uses simplify-code. A question about whether an approach will work before building it uses spike.

A question about existing code, or an edit the user spelled out line by line, matches none of these. Do not load a skill. Read the file, make the change, and run the project's test command if behavior can change.

Do not load a second skill in the same turn unless the task changes kind. Ignore the general instruction to load any partially relevant skill. Only the match above counts. Judge the request, not the working directory.
```

`agent.coding_instructions` (coding posture only):

```text
Use a skill only for a real coding job. If the user is not asking you to write, change, debug, review, or test code, no skill is relevant: do not call skill_view.

When the request is a coding job, call skill_view on exactly one matching skill before the first edit, then follow it:
- new behavior or a behavior change: test-driven-development
- a bug, a failure, or something that does not work: systematic-debugging
- the user asked to review, commit, or ship: requesting-code-review
- the user asked to simplify, clean up, or refactor code that already works: simplify-code
- the user is asking whether an approach will work before building it: spike

A question about existing code, or an edit the user spelled out line by line, matches none of these. Do not load a skill. Read the file, make the change, and run the project's test command if behavior can change.

Do not load a second skill in the same turn unless the task changes kind. Ignore the general instruction to load any partially relevant skill; only the match above counts.
```

The five skills named there ship with Hermes under
`~/.hermes/skills/software-development/`. The prompts name the skill. They
do not paste the skill text. Host-only skills on one machine stay on that
machine.

`~/.hermes/SOUL.md` on the existing installs is 667 bytes, SHA-256
`36c1f5a2e92cd1d018311eaf4c8f1e8886672eae78212e033c681d0e3d5d506f`:

```text
You are Hermes Agent, built by Nous Research. Be direct: match the length of your reply to the weight of the ask — a one-line question gets a one-line answer, and finished work gets a short report of what changed, what's verified, and what's left, never a replay of the process. No filler ("Great question," "I'd be happy to"), no restating the request back, no re-summarizing what you already said, no narrating tool calls the user can see. Plain claims over adjectives; when unsure, say so plainly. Agree because it's right, not because the user said it. Depth is earned — give it when the user asks for detail, teaches, or the stakes demand it, not by default.
```

Prompt and config changes apply on the next session, or on the next
auxiliary call when Hermes re-reads the file. A process that is already
running keeps the system prompt it started with.

## One key per agent

Stock Hermes puts the conversation root in `X-Router-Session`. The main
agent and every delegate then share one backend. This cluster patches the
installed Hermes tree so each agent sends its own key. Auxiliary calls
during the turn inherit the caller's key. The same request sends
`X-Router-Role: main` or `X-Router-Role: agent`. The router records the
role. The role does not cancel a generation that is already running.

The patch is in the Hermes install, not in this repo. A Hermes upgrade
overwrites it. Reapply these three edits after an upgrade.

`agent/portal_tags.py`, next to the other context vars:

```python
_router_agent: ContextVar[Optional[tuple[str, str]]] = ContextVar(
    "hermes_router_agent", default=None
)


def set_router_agent(key: Optional[str], role: Optional[str]):
    """Publish ``(agent key, role)`` for X-Router-Session. Empty key clears it."""
    cleaned = str(key or "").strip()
    if not cleaned:
        return _router_agent.set(None)
    rank = "main" if str(role or "").strip().lower() == "main" else "agent"
    return _router_agent.set((cleaned, rank))


def reset_router_agent(token) -> None:
    """Restore the previous router-agent identity (pair with ``set_router_agent``)."""
    _reset_var(_router_agent, token)


def get_router_agent() -> Optional[tuple[str, str]]:
    return _router_agent.get()
```

`agent/opencode_affinity.py`. Add `router_agent_identity`, and at the start
of the body of `custom_provider_session_affinity_headers`, after the header
name has been resolved and before the stock affinity key is used:

```python
def router_agent_identity(agent: Any) -> tuple[str, str]:
    """``(key, role)`` for the local GPU router. ``("", "main")`` when unknown.

    A delegate is its subagent id, else its own session id. The main agent keeps
    the stock affinity key, so compaction and a declared gateway scope do not
    move it. Callers publish this for the turn; the header builder reads it back.
    """
    try:
        sub = str(getattr(agent, "_subagent_id", "") or "").strip()
        platform = str(getattr(agent, "platform", "") or "").strip().lower()
        if sub or platform == "subagent":
            key = sub or str(getattr(agent, "session_id", "") or "").strip()
            return key, "agent"
        return resolve_affinity_key(getattr(agent, "session_id", None)), "main"
    except Exception:
        return "", "main"
```

```python
if header.lower() == "x-router-session":
    try:
        from agent.portal_tags import get_router_agent

        bound = get_router_agent()
    except Exception:
        bound = None
    if bound and bound[0]:
        role = bound[1] if bound[1] == "main" else "agent"
        return {header: bound[0], "X-Router-Role": role}
```

Headers the caller already set still win. `merge_session_affinity_headers`
uses `setdefault`.

`agent/turn_facade.py`, in the turn that already sets the conversation and
affinity context vars:

- Import `router_agent_identity` from `agent.opencode_affinity`.
- Import `set_router_agent` and `reset_router_agent` from `agent.portal_tags`.
- Initialize `router_token = None` with the other scope tokens, before the
  `try`, so an early return still has a token to reset.
- After `set_affinity_scope(...)`, publish the identity. The main agent's
  key then matches stock affinity. A delegate publishes its own id.

```python
agent_key, agent_role = router_agent_identity(self)
if agent_key:
    router_token = set_router_agent(agent_key, agent_role)
```

- In the outer `finally`, after the other context resets:

```python
if router_token is not None:
    reset_router_agent(router_token)
```

Without this patch, a new client still routes, and every agent in one
conversation shares one GPU.

What the router does with the key is in the README. Short version: a new
agent takes the fastest backend with a free in-flight slot. It stays while
that card is up and every faster card is full. The next request moves when
a faster card has a free slot. A request that is already streaming runs to
completion. Slots count in-flight requests, not idle pins. Vision does not
promote. A live pin still blocks unloading that GPU to make room for
vision. Routers do not share pin tables. Each one probes peers about every
2 seconds (`GET /v1/models` and `GET /health`) and treats a remote backend's
occupancy as the max of its own forwards and the peer's `local_sessions`.
A forwarded call sends `X-Router-Scope: local`, so the far router uses only
its own backends.

## Web search

`web.backend: ddgs` is DuckDuckGo through the `ddgs` extra Hermes already
pins. Writing the key does not install the package. With Hermes on `PATH`
(or `~/.local/bin/hermes`):

```bash
hermes tools post-setup ddgs
```

That runs `pm.sync_venv(["ddgs"], explicit=True)` inside the selected
environment. The aligned installs resolved `ddgs` 9.16.0. The process that
ran the install stays on the old environment and cannot `import ddgs`. The
hook says to restart Hermes. Start a new process. On a machine with a
gateway:

```bash
hermes gateway restart --detach -y
```

A chat that is already open keeps the interpreter it started with. Start a
new chat to search.

`ddgs` is search only. `web_extract` on that backend returns
`DuckDuckGo (ddgs) is a search-only backend and cannot extract URL content.`
That error does not fall through to another extractor. Leave it that way.
The tool is still in the `web` toolset. There is no separate extract
backend in this profile.

## Check

```bash
curl -fsS http://127.0.0.1:8092/health
curl -fsS http://127.0.0.1:8092/v1/models
```

`/health` answers while every GPU is unloaded. `sessions` and
`local_sessions` are in-flight text requests. `pins` is the idle preference
count. A model id in `/v1/models` is the router's public id. It does not
mean a GPU is loaded.

Then open a new Hermes session and send a short text message. That `POST`
is what loads a card. Vision is a message that contains an image or a
video. A tool schema that names `image_url` stays on the text model. Image
generation has no engine. The router returns an unavailable error and asks
the caller to try later.

After a Hermes upgrade, confirm three things before using the machine:
the router file hash above, the three affinity edits, and
`hermes tools post-setup ddgs` if the new environment cannot import `ddgs`.
