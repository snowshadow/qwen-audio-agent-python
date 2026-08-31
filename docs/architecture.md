# qwen-audio-agent architecture

This document defines the product boundary. Changes that contradict these
invariants are architecture changes, not local feature work.

See the
[Realtime Voice Chatbot Runtime Roadmap](https://github.com/QwenAudio/qwen-audio-agent/blob/main/docs/roadmap/frontend-chatbot-runtime.md)
for the target boundary and migration history of the Realtime Voice Chatbot,
asynchronous Task bridge, and single user-configured backend agent. This
document, together with the tests, describes the current runtime behavior.

## Current v2 implementation

The Gateway runtime is implemented in Python under `src/qwen_audio_agent/`.
LangGraph owns two bounded state machines, not the whole product. The public
v2 baseline is this hybrid: Python Gateway graphs and runtime, Node ACP
adapters, existing Desktop/Web/TUI hosts. It is not a LangGraph Platform
application and does not publish the graphs as a standalone library. Do not
rewrite ACP adapters in Python to “finish” the port.

```text
Desktop / WebUI / TUI
          │ HTTP, WebSocket, playback receipts
          ▼
Python transport + realtime application ◄──► Realtime Provider
          │ normalized tool call
          ▼
FrontendTurnGraph
├── inline / control result returns to the current turn
└── spawn_thinking ──► intake receipt ──► WorkGraph
                                              │ BackendPort
                                              ▼
                                      configured backend
                                              │ final Work result
                                              ▼
                                  ResultDeliveryCoordinator
                                  safe window ──► voice: playback.started
                                              text: response completed
                                              │
                                              ▼
                                      durable Task acknowledgement
```

`FrontendTurnGraph` is scoped to one realtime turn. It validates and
deduplicates calls, enforces capability, count, time, and result-size limits,
and routes inline, control, and background tools. Its extension contract can
checkpoint an approval interrupt before a side effect, although no tool in the
current public catalog enables that path. It does not own audio, model
responses, backend selection, or Work execution.

`WorkGraph` is scoped to one Task. The owner scheduler, long-running backend
operation, and adapter connection stay outside the graph. Normalized backend
events resume the graph, which reduces them into bounded state and checkpoints
before waiting again. The Task repository is updated from that state and
publishes the public projection.

State has one owner for each purpose:

| State | Durable owner | Meaning |
| --- | --- | --- |
| Frontend and Work execution | LangGraph SQLite checkpointer, one thread per turn or Task | Bounded control-flow snapshot; safe graph resume point |
| Small cross-thread indexes | LangGraph SQLite Store | Runtime lookup data such as a pending frontend approval |
| Task lifecycle, result, and notification lease | JSON Task repository | Product ledger and delivery source of truth |
| Conversation transcript | Conversation store | Recent user/assistant timeline; not graph state |
| Native backend Session | Backend adapter | Backend-private continuity and recovery evidence |

A checkpoint does not prove that an external side effect happened, that a
backend Session still exists, or that audio reached the client. Those facts
remain with their owning repository, adapter, or playback receipt.

The Gateway launch entry used by the existing Node and Electron hosts is a
compatibility shell. The process path is:

```text
Node / Electron host → python-gateway-entry.mjs → Python Gateway
Python Work Runtime → NodeBackendBridge → python-backend-bridge.mjs
                    → existing JavaScript backend adapters
```

The entry starts `python -m qwen_audio_agent.cli` and forwards structured
readiness, signals, and exit status. CLI, TUI, Desktop UI/process code, and the
retained backend adapters still have substantive JavaScript responsibilities;
the active runtime path does not maintain a second Gateway or Task state
machine there.

The public `qwenaudio` and Desktop configuration flow is unchanged. The shell
projects the existing environment contract, including `HOST`, `PORT`, Realtime
settings, and `AGENT_PROTOCOL`, into the Python CLI. `QWAUDIO_DATA_DIR` owns the
shared identity, profile, memory, configuration, and notes;
`QWAUDIO_CONFIG_DIR` owns Gateway runtime state such as the lease, Task ledger,
conversations, checkpoints, and skins. If the data directory is not set, it
falls back to the configuration directory. The launcher uses
`QWEN_AUDIO_AGENT_PYTHON` when set, the project virtual environment when
present, and otherwise the locked `uv` project. Direct Python development may
use `uv run python -m qwen_audio_agent.cli`. Packaged v2 Desktop builds include
the locked project but not a Python interpreter or `uv`; the host must provide
`uv` or point `QWEN_AUDIO_AGENT_PYTHON` at an environment with the runtime
dependencies installed.

The live catalog is the built-in realtime tools in section 3 plus the ACP
backend bridge. These documented extension points are **not connected** on the
Python Gateway path. The previous JavaScript Gateway runtime has been removed;
public contract modules remain for a later port, and the related configuration
keys currently have no effect:

- Frontend MCP client (`QWEN_AUDIO_FRONTEND_MCP_CONFIG`)
- Frontend OpenAPI adapter (`QWEN_AUDIO_FRONTEND_OPENAPI_CONFIG`)
- Frontend Profile tool sources (`QWEN_AUDIO_FRONTEND_PROFILE`)
- Knowledge retrieval provider
- Frontend web search (`QWEN_AUDIO_WEB_SEARCH_PROVIDER`)
- Session-end automatic memory reconciliation
- Local wake-word detection (`QWEN_AUDIO_WAKE_WORD_ENABLED`)

## 1. User-visible model

The user talks to one qwen-audio assistant. Internally there are two qwen-audio-agent
layers:

1. **Realtime frontend** — full-duplex speech, simple direct answers, and basic
   local time/memory tools.
2. **Backend Agent** — one configured action Agent that handles requests
   requiring tools, files, applications, code, device control, or multi-step
   execution.

The public Python CLI currently registers the retained ACP drivers, including
OpenCode, OpenClaw, Qoder, Qwen Code, Kimi Code, and Pi. The BackendPort SDK
also exports an A2A adapter for custom composition roots, but the public CLI
does not select it. A backend may internally use tools, skills, agents, or
other Sessions. Those are backend-private implementation details and do not
create additional qwen-audio-agent layers. Protocol details remain inside
their adapters; backend-specific launch and capability behavior lives in
registered drivers.

## 2. Nonblocking request flow

```text
final ASR → FrontendTurnGraph
   │
   ├─ immediately answerable ───────────────► Realtime speech
   │
   └─ requires work
          │ spawn_thinking(objective)
          ▼
      Task accepted
          │ response returns to Realtime immediately
          ▼
      WorkGraph owner FIFO queue
          │
          ▼
      configured BackendPort
          │ the backend decides how to work
          ▼
      final presentation
          │ waits for a safe duplex insertion window
          ▼
      Realtime naturally speaks the result
```

`spawn_thinking` never waits for the requested work. The user can continue
speaking while multiple Task items are queued. Coordinator submissions use a
FIFO lane per owner. A Task releases that lane when verified delegation begins,
so delegated target work does not block the next coordinator submission.

## 3. Realtime boundary

Realtime keeps a deliberately small tool set — few tools, low latency, no
multi-step orchestration. The maximum built-in catalog is:

```text
spawn_thinking
schedule_reminder
cancel_agent_task
get_agent_task_status
get_current_time
memory
notes
respond_agent_permission
enter_sleep
```

`spawn_thinking` is exposed only when a backend is configured. `enter_sleep`
is exposed only when the connected client advertises the `sleeping` state.

`memory` maintains two ordinary Markdown documents through one flat interface. Each call is one
atomic `read`, `append`, or `replace` operation. `replace` locates a unique source fragment, and
fails safely if that fragment is missing or ambiguous. Each call returns its own result to the
Realtime model. The documents have different authority:

- `user` is the current user's long-term personalization overlay: forms of
  address, relationship, the assistant's name for that user, language, expression
  style, and default behavior. It is injected as user-authorized directive material,
  overrides the instance-wide defaults from `ASSISTANT.md`, and yields to the user's
  current utterance.
- `memory` contains durable facts and decisions used for understanding and answers,
  never as instructions. The scopes are separated by behavioral authority, not topic.

Neither scope can authorize leaking internal structure, skipping permission
checks, or changing task and safety protocols. The packaged `PROMPT.md` remains
the core policy and cannot be overridden by personalization. Local `ASSISTANT.md`
contains only the assistant instance's default identity, personality, relationship
stance, and expression style. The assistant never edits it through memory. It is
created from the packaged template once, preserved across upgrades, and reloaded
for the next voice session after an edit.

`notes` manages user-named lists (shopping lists, todos, reading lists) as
frontend-owned, locally persisted collections: single-call add, show,
match-remove, clear, and drop with no backend involvement. Lists are item data,
not memory; stable facts remain in `memory`, and list items are never written
into the user preferences or factual memory. Item and list resolution matches
exact text first, then a unique case-insensitive substring, and otherwise
reports ambiguity with the candidate names back to the model for clarification.
The tool contract tells the model to call `clear` and `drop` only after an
explicit destructive request in the current turn; the data store does not add
a second confirmation gate.

`get_agent_task_status` is the single Realtime entry point for lifecycle,
progress, and interim-result questions. The Gateway reads its own Task record
directly, including the latest adapter-normalized message, activity, and
artifact descriptors. A status check does not create another Task, invoke the
coordinator, or enter the asynchronous announcement queue.

It does not have tools for:

- selecting, creating, continuing, or cancelling backend Sessions;
- choosing synchronous, asynchronous, foreground, or background execution;
- selecting backend execution strategy;
- selecting tools, Agents, or subagents.

`respond_agent_permission` is the only exception to the rule that Realtime does
not control backend execution. It may relay only an explicit current-turn user
decision for a pending, owner-scoped permission request supplied by the
Gateway. It may understand natural affirmative or negative wording such as
“可以” or “不允许”, but it cannot invent consent without a current-turn user
utterance, create a request, choose a tool, or modify a backend permission
policy. Replies are limited to `always` and `reject`, and resolve only the
pending request. `always` is retained as a wire-compatible approval token; it
does not create a session grant. Approval maps only to a backend `allow_once`
option. If the backend does not offer one, the request is cancelled rather than
silently widened to `allow_always`. Rejection prefers `reject_once` and may use
`reject_always` only to deny more broadly. The frontend never auto-approves a
later request.

The `objective` passed to `spawn_thinking` is a conservative interpretation of
the user's request, not an execution plan. It must resolve references such as
“continue that page” into one self-contained instruction before submission.
That instruction is the only model-visible task text sent to the backend; the
verbatim ASR is not appended as a second task description. The backend Agent
does not receive the frontend persona, durable memory, or recent chat history.
Execution-relevant facts must be resolved into the instruction instead of
forwarding those documents.

The bridge gives the backend Agent a Task's `inputParts` as native protocol
parts rather than a JSON manifest. `inputRefs` remains an internal compatibility
field; no current public tool populates it or resolves earlier conversation
attachments. Task IDs, owner identity, lifecycle, timestamps, and routing
remain structured Gateway/BackendPort data and are not placed in the backend
Agent's instruction. The working directory and user time zone are not repeated
in every prompt; protocol or backend runtime context owns them.

## 4. Fixed Backend Agent Session

The ACP adapter owns one persistent coordinator Session identity per protocol
and owner. The exact storage key is adapter-private; it is derived from:

```text
<protocol> × <owner>
```

The Gateway stores the native ACP Session ID behind that stable key and calls
`session/resume` on later turns. Project delegation likewise resumes the
selected native Session in its recorded working directory, so voice-originated
work remains in the backend's own Session history rather than a Gateway copy.

Voice browser session IDs and Task IDs never change that identity. A new voice
conversation therefore continues using the same backend Agent context.

Both the Gateway queue and the ACP adapter serialize writes. This double guard
prevents concurrent messages from racing inside one backend Session.

The backend Agent owns its execution strategy. qwen-audio-agent supplies one
self-contained natural task instruction and current-turn native attachments;
it does not forward frontend history or preferences, prescribe task-state
JSON, or instruct the backend Agent how to use backend-specific capabilities.

## 5. Task state

A qwen-audio-agent Task record is a delivery receipt, not a mirror of the backend's
internal task graph.

```text
scheduled ──► queued
queued → running ─────────────────────────→ completed
   │        └→ delegated → finalizing ────────┘
   └────────────→ cancelling → cancelled
                            ↘ failed
```

Public fields are limited to the user request, timestamps, final result/error,
generic activity, a bounded pending permission summary with optional safe
operation details, and notification state. There is no execution mode,
delivery mode, subagent state, backend permission identifier, backend topology,
or backend cancellation internals.

The UI presents both `queued` and `running` as the same “processing” state.
Queue position is an internal scheduling detail and does not change the user's
duplex conversation.

Queued and directly running Tasks cannot be safely resumed after a Gateway
restart, so they become failed with an explicit restart reason. A delegated
Task may be reattached only when its adapter can verify the persisted native
Session and the matching WorkGraph checkpoint exists. Private recovery
correlation is persisted but omitted from the public Task projection.
Completed results and notification delivery state are persisted. Scheduled
Work is rearmed from the Task ledger rather than inferred from a graph
checkpoint.

## 6. Progress animation

Progress is observability, not control. The ACP adapter projects standard
`session/update` notifications into generic activity:

- tool name, bounded user-safe detail, and running/completed state;
- plan progress;
- a generic thinking signal without thought content;
- bounded Session title and current mode metadata.

The UI maps this to the stable task objective or phrases such as “searching”,
“reading”, “generating an image”, or the current mode. A generic thinking signal
keeps showing the task objective, while the desktop pet maps it to its existing
processing animation. Session IDs,
subagent IDs, raw permission payloads, and raw thought content are not shown. A
pending permission may show the exact bounded title, description, command, or
path needed for informed consent after secret-like values are redacted.

Activity never produces spoken status updates and never affects the queue.

## 7. Final result delivery

The backend Agent returns a standard ACP turn. Text arrives in
`agent_message_chunk` updates, while images, audio, and resources remain native
`ContentBlock` values; the turn ends with the `session/prompt`
`PromptResponse`. The ACP adapter no longer flattens these values and then asks
the model to reconstruct `state` or `presentation` JSON. It projects text and
non-text content into the BackendPort `content` and `artifacts` fields, and the
Gateway accepts only `stopReason=end_turn` as a successfully completed turn.
Cancellation, refusal, and token or agent-request limits enter the Gateway's
cancelled or failed paths. The Gateway then chooses the appropriate
conversation, resource, and voice presentation for each client.

Completed results prefer the originating conversation. On a fresh connection,
unfinished results from older conversations may be recovered for the same
owner. A renewable claim prevents two live frontends from presenting the same
result. Voice results are injected into Realtime context and marked delivered
only after the active client reports `playback.started`. For an audio response,
model response completion, audio generation, `playback.ended`, and cancellation
before playback starts do not acknowledge delivery. If the user interrupts, is
speaking, or another response is pending, delivery waits. A failed attempt
releases or retries its claim; an expired lease can be claimed by another live
frontend. Retries are bounded so one malformed result cannot block later
completions.

Text-only clients have no playback receipt. For them, a successful response
completion is the delivery acknowledgement; a failed response remains pending.

When the backend Agent calls `session_start` or `session_send`, delegation is
established only after the Session tool creates or continues the target work
and returns run and Session identifiers validated by the adapter. It is not
established by a model-authored field. ACP faithfully carries the tool call,
tool result, and turn termination; the adapter publishes the validated
correlation to the Work runtime, which moves the original Task to `delegated`
and releases both the backend Agent serialization lock and the Task scheduler
lane. Other voice requests can therefore use the coordinator while the target
Session runs.

The coordinator may naturally end its ACP turn after the tool succeeds, but
that text neither controls task state nor acts as a completion signal.
Correlation IDs, the target Session, and lifecycle stay in the Gateway Task
registry and adapter runtime; the model never has to echo them.

The adapter independently keeps the Task lifecycle and event subscription
alive. Only the matching ACP target prompt completion correlated to the
delegation ID can advance the Task. The adapter then briefly reacquires the
backend Agent lock and sends the verified result, including its native
ContentBlocks, back for a final natural answer. A busy target, an empty result,
an unrelated Session update, or an older result cannot complete the Task.

ACP Agent turns have no artificial wall-clock deadline. An initial coordinator
turn, a delegated target turn, and the final synthesis turn end only when ACP
reports completion, the user explicitly cancels the Task, or the backend exits
or shuts down. Connection initialization and bounded control RPCs retain
timeouts so an unavailable backend cannot block Gateway startup indefinitely.

Cancellation is confirmed rather than optimistic. `queued` Task is cancelled
locally. `running` or `finalizing` Task aborts its active backend request. For
`delegated` Task, an idle coordinator is first asked to call
`session_cancel`; if the coordinator Session is occupied, the ACP adapter
directly sends `session/cancel` to the exact correlated target Session. The Task remains
`cancelling` until one of those paths confirms the stop, then becomes
`cancelled`. A failed stop becomes `failed` with the cancellation error.
After a direct adapter abort, the Gateway records a cancellation fact and
injects it once into the next safe coordinator turn. This reconciles the
coordinator's history without delaying cancellation or repeating the stop.

The frontend acknowledgement comes from the Task that the Gateway actually
accepted, not from a coordinator-authored delegation state. The coordinator
turn still ends according to ACP lifecycle signals; the Gateway does not infer
its completion from acknowledgement text or from a successful Session tool
call.

## 8. Backend-internal capabilities

For ACP backends that accept client-supplied MCP servers, including OpenCode,
Qoder, Qwen Code, and Kimi Code, the Gateway injects the same five tools into the
coordinator: Session list, start, send, status, and cancel. OpenClaw ACP does
not accept client-supplied MCP servers, so the same coordination contract maps
to OpenClaw's native Session tools. `session_start` and `session_send` return an
opaque delegation ID. After either succeeds, the backend Agent must not poll,
repeat the work, or answer from its own context; the adapter owns waiting,
cancellation, permission routing, and result correlation.

The coordinator MCP server also publishes the stable coordination contract
through the MCP initialization `instructions` field. Backends whose drivers
declare `coordinatorMcpInstructions` receive only the dynamic natural task
instruction on each turn; this avoids appending the same routing and response
rules to the persistent Session history. The capability is enabled only after the Agent host
has been verified to project MCP server instructions into model context; it
currently applies to OpenCode, Qoder, Qwen Code, and Claude Code. The shared
payload stays within a 2 KiB portability budget. Unverified Agents—or a future
payload that exceeds that budget—keep the complete per-turn prompt as a safe
fallback. This flag does not describe general MCP support. Project Sessions
never receive the coordinator MCP server.

`session_status` is observational only. If the query fails, the backend Agent
must report the failure; it must not inspect the target directory with native
tools or duplicate the delegated work.

Frontend code must not depend on which internal capability was chosen.
Frontend task snapshots may expose only a bounded title and generic delegated
state, never delegation IDs, target Session IDs, directories, or raw events.

## 9. Dependency direction

```text
WebUI / TUI / Desktop
   ↓ WebSocket and HTTP
Python transport and Realtime application
   ↓ normalized tool call
FrontendTurnGraph
   ↓ WorkSubmissionPort
WorkGraph + Task repository + scheduler
   ↓ protocol-neutral BackendPort
Node backend bridge
   ↓ Adapter projection: natural instruction + native attachment parts
OpenCode ACP, OpenClaw ACP bridge, Qoder ACP, Qwen Code ACP,
Kimi Code ACP, or another supported Agent
```

Backend-specific JavaScript adapter details remain under `server/src/agent`;
the Python Work Runtime sees only `BackendPort`. Realtime tools must not import
backend adapters. The UI consumes only public Task events and final timeline
content. Transport depends on application and domain contracts, while provider
and backend adapters point inward through ports.

Gateway may serve the immutable `web/dist` artifact as a deployment
convenience, but this is static hosting only. Gateway source must not import UI
components, presentation text, styling, terminal behavior, or desktop behavior.
All three UIs own their rendering and map structured protocol fields to their
own labels and interaction patterns.

## 10. Process ownership

The Gateway is the only core product service. Backend lifecycles use one shared
`owned/external` ownership model:

- `owned`: Gateway starts the required local backend processes and stops them
  on exit. The native backend process loads its own user configuration, models,
  tools, and MCP servers; the adapter supplies only protocol parameters and
  required shared capabilities.
- `external`: available only to backends declaring external-service support.
  Gateway does not start, move, or stop that backend. It connects through the
  backend's published protocol address and leaves configuration and state under
  the external service's control.

Backend service ownership and the ACP connection are independent axes. Each
backend profile declares an `acpConnection`; the connection factory currently
implements `process`, which launches one local ACP stdio child. A future remote
ACP bridge can add another connection kind without changing coordinator,
permission, Task, or Session lifecycle code. Declaring an external backend
service does not by itself make the ACP connection remote.

Each backend is registered through one validated plugin contract. Its catalog
entry owns identity, installation, native onboarding, process environment and
ownership metadata; its Agent and Runtime drivers declare explicit boolean
capabilities and are rejected at startup when incomplete or inconsistent.
Backend child processes receive only portable operating-system variables and
the selected plugin's declared credential namespace. Gateway identity,
Realtime, memory, and other backend secrets never cross that boundary. A
generic ACP command may opt in additional names explicitly through
`QWEN_AUDIO_AGENT_ACP_FORWARD_ENV`.

The Python HTTP/WebSocket application is constructed by an injectable
composition root. Importing the application factory does not bind a port. The
CLI performs the concrete wiring, while tests and future hosts may supply
isolated input arbitration, Task and Conversation sources, identity, health,
and Realtime event handlers. The Node/Electron bootstrap remains only for
launch compatibility and process ownership; it does not implement application
behavior.

The shared adapter usually owns one ACP stdio child and stops it with Gateway.
OpenCode, Qoder, Qwen Code, and Kimi Code run directly as ACP agents; OpenCode may also
start its native local Session UI service. `OPENCODE_BASE_URL` currently names
that UI service, not a remote ACP execution endpoint, so OpenCode remains
`owned`.

OpenClaw uses a small ACP bridge. Without an explicit address, Gateway starts
an OpenClaw Gateway with isolated runtime and Session state. When
`OPENCLAW_BASE_URL` is explicit, it connects to the user's existing OpenClaw
Gateway without reading, copying, or modifying its authentication or Agent
state. Service ownership is then `external`, while the ACP connection remains
a local `process`: the official local bridge connects to the remote OpenClaw
Gateway over WebSocket/WSS. External connections bypass the short local-startup
port probe so the bridge can report the real network, TLS, and authentication
result. A local bridge exit interrupts ACP only and never changes the remote
Gateway lifecycle.

Codex follows the same boundary: qwen-audio-agent starts `codex-acp` over ACP
stdio, and that adapter starts Codex App Server over its own local stdio
protocol. Codex App Server may expose other transports, but they are not a
remote ACP endpoint and must not leak into the shared ACP adapter.

Desktop, TUI and WebUI are replaceable Gateway clients. The Python CLI selects
the Realtime model from the existing environment contract; clients do not
override that model per connection. Health publishes the active provider,
exact model profile, model catalog, input sample rate, configuration signature,
selectable providers, and backend summary. Desktop may configure and restart
only its locally owned Gateway. Closing a UI cannot
affect queued work or the fixed backend Agent Session. Configuration that
changes Realtime or backend behavior takes effect on the next Gateway start;
changing a UI's Gateway URL only reconnects that UI.

The macOS desktop renderer is packaged inside the application. Electron serves
those immutable assets from a private, random loopback path and proxies only
Gateway HTTP API and Realtime WebSocket traffic. Desktop UI assets must not be
loaded from the Gateway: rebuilding the desktop application must be sufficient
to update its appearance without upgrading the running Gateway frontend.

## 11. Review checklist

Before merging a change, verify:

1. Can Realtime still converse while backend work is queued or running?
2. Does every executable request enter the same persistent backend Agent
   Session?
3. Did any frontend API gain knowledge of Session, subagent, permission, or
   execution mode?
4. Are tool events used only for generic UI progress?
5. Is completion spoken only from a final backend Agent result?
6. Did any UI begin managing a Gateway or backend process?
7. Can interruption postpone speech without cancelling submitted Task?
8. Do tests cover FIFO serialization, fixed Session reuse, checkpoint and
   ledger boundaries, tool animation, and playback-start delivery retry?
