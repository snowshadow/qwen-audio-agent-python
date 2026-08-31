# qwen-audio-agent — LangGraph Gateway (Python)

[中文](README_ZH.md) | [English](README.md)

[![CI](https://github.com/snowshadow/qwen-audio-agent-python/actions/workflows/ci.yml/badge.svg)](https://github.com/snowshadow/qwen-audio-agent-python/actions/workflows/ci.yml)
[![npm](https://img.shields.io/npm/v/qwen-audio-agent)](https://www.npmjs.com/package/qwen-audio-agent)
[![node](https://img.shields.io/badge/node-%E2%89%A522.22.2-brightgreen)](https://nodejs.org/)
[![python](https://img.shields.io/badge/python-3.12–3.14-blue)](https://www.python.org/)
[![license](https://img.shields.io/github/license/snowshadow/qwen-audio-agent-python)](LICENSE)

Independent **LangGraph / Python** fork of
[QwenAudio/qwen-audio-agent](https://github.com/QwenAudio/qwen-audio-agent).
Apache-2.0, not an official QwenAudio release. See [`NOTICE`](NOTICE).

This repository exists to run the **v2 Gateway** with **two LangGraph state
machines** in Python. Hosts (CLI, WebUI, TUI, Desktop) and **ACP backend
adapters stay on Node**. It is a voice Gateway product — not a LangGraph
Platform app and not a reusable graph library.

For the upstream product overview, desktop releases, and community channels,
see [QwenAudio/qwen-audio-agent](https://github.com/QwenAudio/qwen-audio-agent).
`npm install -g qwen-audio-agent` still installs the **official** npm package.

## What is different here

| Area | This repo | Upstream |
| --- | --- | --- |
| Gateway control flow | Python + **LangGraph** (`FrontendTurnGraph`, `WorkGraph`) | JavaScript Gateway (v1) |
| Long I/O | Realtime, ACP, models **outside** the graphs | Same product idea, different runtime |
| Backend agents | Existing **Node ACP adapters** via `BackendPort` | Same adapters |
| Status | v2 alpha (`2.0.0a0`), active development | Stable v1 line |

The old JavaScript Gateway runtime is **removed** from this tree. Do not expect
v1-only extension paths (frontend MCP, OpenAPI, knowledge retrieval, web search,
session-end memory extraction, local wake word) on the Python Gateway until they
are ported — configuration keys for them currently have **no effect**.

## Architecture

![LangGraph Gateway overview](docs/architecture-overview-en.png)

```text
Desktop / WebUI / TUI
          │ HTTP, WebSocket, playback receipts
          ▼
Python transport + realtime app ◄──► Realtime provider (DashScope, etc.)
          │ normalized tool call
          ▼
FrontendTurnGraph          one realtime turn: validate, policy, inline/control/background
          │
          └─ spawn_thinking ──► WorkGraph     one Task: checkpointed lifecycle
                                      │ BackendPort (Node bridge)
                                      ▼
                              ACP backend adapter ──► coding agent
                                      │ result
                                      ▼
                          ResultDeliveryCoordinator → voice / text client
```

**LangGraph owns bounded control flow only.** Audio streaming, model sessions,
scheduler lanes, and ACP I/O stay in ordinary Python / Node code around the
graphs.

| Graph | Scope | Responsibility |
| --- | --- | --- |
| `FrontendTurnGraph` | One voice turn | Tool admission, limits, inline vs background routing |
| `WorkGraph` | One background Task | Event reduction, checkpoints, public task projection |

**Process boundary** (unchanged host contract):

```text
Node/Electron host → python-gateway-entry.mjs → python -m qwen_audio_agent.cli
WorkGraph backend  → NodeBackendBridge → python-backend-bridge.mjs → JS ACP adapters
```

Source layout:

- `src/qwen_audio_agent/graphs/` — LangGraph definitions and facades
- `src/qwen_audio_agent/transport/` — FastAPI / WebSocket Gateway
- `server/src/agent/` — ACP adapters (still JavaScript)

Full invariants, state ownership, and the not-yet-wired extension list:
[docs/architecture.md](docs/architecture.md).

Gateway contract version: **4.0.0** ([docs/contract.md](docs/contract.md)).

## Requirements

- Node.js **22.22.2+** or **24.15.0+**, npm **10+**
- Python **3.12–3.14** and [uv](https://docs.astral.sh/uv/)
- DashScope API key when using the default cloud Realtime frontend

## Install

From this repository (LangGraph Gateway):

```bash
npm install -g git+https://github.com/snowshadow/qwen-audio-agent-python.git
```

From source:

```bash
git clone https://github.com/snowshadow/qwen-audio-agent-python.git
cd qwen-audio-agent-python
uv sync --frozen
npm install
npm run install:global
```

Details: [docs/getting-started/install.md](docs/getting-started/install.md).

## Quick start

1. Create config and set your Realtime API key:

```bash
qwenaudio config
```

```dotenv
DASHSCOPE_API_KEY=your-key
QWEN_AUDIO_REALTIME_MODEL=qwen-audio-3.0-realtime-plus
# Optional backend; omit for frontend-only mode
AGENT_PROTOCOL=openclaw
```

2. Start Gateway and a client:

```bash
qwenaudio          # terminal 1
qwenaudio tui      # terminal 2 — or: qwenaudio webui
```

After changing Gateway settings from the CLI, restart the Gateway process.

Configuration reference: [docs/configuration.md](docs/configuration.md).

## Development

Gateway changes belong in Python; backend adapter changes in
`server/src/agent/`.

```bash
uv sync --frozen
npm ci
npm run test:python    # pytest + ruff
npm test               # Node workspaces + ACP adapters
npm run dev            # Gateway + WebUI hot reload
```

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Links

- Issues: [github.com/snowshadow/qwen-audio-agent-python/issues](https://github.com/snowshadow/qwen-audio-agent-python/issues)
- Upstream: [github.com/QwenAudio/qwen-audio-agent](https://github.com/QwenAudio/qwen-audio-agent)
- License: [Apache-2.0](LICENSE) (upstream attribution in [`NOTICE`](NOTICE))
- Security: [SECURITY.md](SECURITY.md) · Privacy: [PRIVACY.md](PRIVACY.md)
