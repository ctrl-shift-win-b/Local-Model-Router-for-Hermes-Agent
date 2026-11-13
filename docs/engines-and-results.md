# Engines, measurements, and Hermes

What was measured, which engine each GPU runs, and how that connects to
the router and to Hermes. A new client follows
[new-hermes-client.md](new-hermes-client.md). Addresses and account names
stay in each machine's untracked router file.

Roles:

| Role | What it runs |
| --- | --- |
| Fast GPU host | One fast card. Its router prefers that card, then the workstation. Text only. |
| Workstation | Two local cards, called A and B below. Text on both. Vision on both, weakest first. |
| Client | Hermes plus a loopback router. No local engine. Forwards to the two hosts. |

Card A is the faster of the two local cards (24 GB, sm_86). Card B is the
larger, slower card (32 GB, sm_70). One process cannot span both. Hermes
on every role uses `http://127.0.0.1:8092/v1`, model `qwen3.8-27b`,
provider `custom:qwen-local`, context 240000, reasoning `xhigh`.

## Bake-off

One prompt, 57,410 tokens. Thinking off, greedy decode, prefix cache off,
so client time-to-first-token matches the server. MTP checked every
accepted token against the target model. The answer did not separate
engines: every winner named the same layer, mix, and page rule.

Pick order: answer quality, then decode, then context at or above 240k,
then concurrency above 1, then prefill. The winners pin no model and no KV
in host RAM.

| Card | Winner | Context | KV | Concurrency | Long prefill | Long decode |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| A, before a consumer-card unlock | NInfer groupwise | 262144 | rk4v4 | 2 | 2163 tok/s | 104 tok/s |
| A, after that unlock | same config | 262144 | rk4v4 | 2 | 2243 tok/s | 109 tok/s |
| B | NInfer NVFP4 v2 | 240000 | int8 | 1 | 688 tok/s | 47 tok/s |
| Fast GPU host | NInfer NVFP4 | 240000 | FP8 | 2 | 8152 tok/s | 153 tok/s |

The fast GPU host was already measured on this same prompt (TTFT 7.0 s)
and was not re-benched. Against unlocked card A that is about 3.6× prefill
and about 1.4× decode. Against card B it is about 3.3× decode. An agent
may rebuild its prefix once to move onto that faster card.

Card A at concurrency 2 matches concurrency 1 on the long prompt. Two
short requests fit. The KV pool is one 262,144-token allocation, so two
full 240k contexts do not. Card B at concurrency 2 also boots, and the
long prompt slows to 504 / 41, so concurrency 1 is the serve config.
llama.cpp Q4_K_M is slower on both (card A about 1062 / 30 with q4_0 KV;
card B about 416 / 17 with q8_0 KV). q8_0 KV does not fit beside Q4 weights
on the 24 GB card.

vLLM, SGLang, and TensorRT-LLM do not show a 262k context with host cache
off on 24 GB, and those kernels are not built for sm_70.

The unlock raised an FP16 GEMM at n=8192 from 68.8 TFLOPS (FP32 accumulate)
to 123.7. This groupwise path dequantizes into an FP16 prefill GEMM and
does not pick up that jump. Decode stays bandwidth-bound. The long prompt
moved from 2163 / 104 to 2243 / 109. Keep the unlock on card A only. It is
not part of bringing up a client.

The three text servers are the same Qwen3.8-27B family. On the published
cards, the NVFP4 file is a few points ahead on GPQA and about a point on
RealWorldQA. The groupwise file is a hair ahead on IFBench. AIME and ERQA
tie. Card B's file is NVFP4 version 2. Its binary rejects the version-3
file, and there is no separate version-2 exam. The unmeasured long-context
gap is rk4v4 versus FP8 versus int8. Hermes compacts near 181k tokens, so
card A's extra room above 240000 does not change short answers.

## Bench flags are not the serve flags

The bake-off passed `--no-thinking`, `--greedy`, and turned prefix reuse
off, so client time-to-first-token matched the server.

Live Hermes servers leave prefix reuse on and do not pass `--greedy` or
`--no-thinking`. Hermes sends `reasoning_effort: xhigh`.

On card A, prefix reuse is the binary default (8192 MiB plus 8
StateImages). Leave that default. Do not pass `--host-context-mib` or
`--no-prefix-reuse`. Card B does not pass `--no-prefix-reuse` or
`--host-kv-mib`. Both keep `--media-cache-mib 0`. Card B has no
`--prefill-cublas` flag. The fast GPU host's script already omits
`--no-prefix-reuse`: NVFP4, FP8 KV, context 240000, MTP 3, concurrency 2.

## Live engines

`CUDA_DEVICE_ORDER=PCI_BUS_ID`. Index 0 is card A. Index 1 is card B.
Bind every engine to `127.0.0.1`. The public model id on every text server
is `qwen3.8-27b`.

### Card A, port 8090

NInfer built for sm_86. Weights are the groupwise artifact
`qwen3_8_27b.ninfer` (18982458624 bytes, SHA-256
`7f2a0086a071ee932c66106e1552d118a85cfec8ddef13c67a54cd64abb39706`).

- `--max-concurrency 2`
- `--max-context 262144 --kv-capacity 262144 --kv-dtype rk4v4`
- `--spec mtp --draft-tokens 3 --lm-head-draft --gdn-state-fp16`
- `--prefill-cublas --prefill-chunk 2048`
- `--media-cache-mib 0`
- `--reasoning-effort xhigh`

The NVFP4 file plus a long cache does not fit this 24 GB card.

### Card B, port 8091

NInfer built for sm_70. Weights are
`qwen3_8_27b_nvfp4_v2.ninfer` (23719496192 bytes, SHA-256
`552c374c685dce302603b95fbe940fb04243c0cd44c083efc644ad3d980d462c`,
header `4e494e4645520002`). The version-3 NVFP4 file is a different
header (`4e494e4645520003`) and this binary rejects it.

- `--max-concurrency 1`
- `--max-context 240000 --kv-capacity 240000 --kv-dtype int8`
- `--prefill-chunk 2048`
- `--spec mtp --draft-tokens 3 --lm-head-draft`
- `--media-cache-mib 0`
- no reasoning-effort flag on this binary

This server accepts `none`, `low`, `medium`, and `xhigh`. It rejects
`minimal`, `high`, and `max`. The router rewrites those three only when
the backend name is exactly `v100`.

### Vision

llama.cpp `llama-server`, built for CUDA architectures 70 and 86, with
`qwen35` and a `qwen3vl` projector. Weights are a Qwen3.5-2B GGUF plus an
mmproj GGUF. Alias `qwen3.5-2b`, context 8192, all layers on the GPU,
jinja on, parallel 1.

Card B's vision server is port 8093. Card A's is port 8094. A preflight
checks that both files exist and does not start a GPU. A single-stream
download of the projector succeeded where a multi-connection download
returned HTTP 416.

Vision unloads the weakest free text GPU first (card B, then card A). A
live pin blocks that unload. If every candidate is busy, held, or missing
weights, the router returns 503. Image generation has no engine and
returns its own unavailable error.

A vision request is an image or a video inside a message. A tool schema
that names `image_url` stays on the text model.

### Fast GPU host

NInfer there binds to localhost. The example origin is `127.0.0.1:18090`
with 2 slots. The router on that machine starts it. Other machines only
probe that router.

## Router wiring

Public model `qwen3.8-27b`, context 240000. Vision model `qwen3.5-2b`,
context 8192. `idle_seconds` 600. Probe interval about 2 seconds.
`gpu_router.py` in this commit is the router to run.

Workstation backends, in speed order:

| Name | Origin | Slots | Capability |
| --- | --- | ---: | --- |
| fast GPU router | that host's port 8092 | 2 | text |
| card A | `127.0.0.1:8090` | 2 | text, weakness 20 |
| `v100` | `127.0.0.1:8091` | 1 | text, weakness 30. The name must be `v100`. |
| vision on B | `127.0.0.1:8093` | 1 | vision, weakness 30 |
| vision on A | `127.0.0.1:8094` | 1 | vision, weakness 20 |

The workstation and the fast GPU host listen on all interfaces, port 8092.
The fast GPU host prefers its own card, then the workstation with 3 text
slots (2 + 1) and with vision and image advertised, because those engines
are on the workstation. The client listens on `127.0.0.1:8092` only.
`examples/router.gpu-host.json` lists vision on the fast GPU remote as
well. On this cluster that remote is text only.

Placement:

- Each Hermes agent has its own `X-Router-Session` key. Auxiliary calls
  use the caller's key. `X-Router-Role` is logged and does not move work.
- A new agent takes the fastest backend with a free in-flight slot and
  stays while that card is up and every faster card is full. The next
  request moves when a faster card has room.
- A generation that is already streaming is never cancelled.
- A dead backend fails over. The agent returns when the faster card is
  healthy and has room.
- Slots count in-flight requests. An idle pin does not occupy a slot.
  The text cap is 5.
- Headerless posts are not pinned. Text still prefers a card with room.
- Vision does not promote.
- A live pin blocks vision from unloading that GPU.
- Process unload is a separate 600 second clock.
- A forwarded call sends `X-Router-Scope: local`.
- Routers do not share pins. Occupancy of a remote backend is the max of
  this router's forwards and the peer's `local_sessions`.
- `GET` discovery does not start a GPU and does not reset the idle timer.

`/health` answers while every GPU is unloaded. With the model resident,
card A and the fast GPU host sit near their empty-desktop power. Card B
sits about 13 W higher (about 40 W resident, about 27 W empty). The idle
timer is what drops that, and what keeps a short pause from throwing away
the prefix cache.

Card B's power limit stays 250 W. Stop it if the GPU reaches 88°C or
memory reaches 85°C. Stop card A at 95°C. Any unlock targets card A only.

## Bring up a replacement

1. Build the sm_86 engine, the sm_70 engine, and llama.cpp for
   architectures 70 and 86. CUDA toolkit 12.9 is what these builds used.
2. Check the two text weight hashes above. Put the vision GGUF and its
   projector beside them. Run the vision preflight.
3. Match the serve flags in this file. Leave prefix reuse on.
4. Install this repo's router and confirm the `gpu_router.py` hash in the
   client guide. Write the untracked router file from the table above.
5. Install the Hermes profile and the per-agent header patch from the
   client guide. Run `hermes tools post-setup ddgs`, then start a new
   Hermes process.
6. Start the router while no chat is in flight. `GET /health` and
   `GET /v1/models` must succeed without loading a GPU.
7. A short Hermes text chat is the load test. It should land on the fast
   GPU host while that host has a free slot.

A new client skips steps 1 through 3 and uses
`examples/router.client.json`.
