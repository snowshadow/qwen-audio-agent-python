# qwen-audio-agent 架构

本文档定义产品边界。违反这些不变性的变更属于架构变更，而非局部功能开发。

前台 Realtime Voice Chatbot、异步 Task Bridge 与单一用户后台 Agent 的目标边界及
迁移记录见
[Realtime Voice Chatbot Runtime Roadmap](https://github.com/QwenAudio/qwen-audio-agent/blob/main/docs/roadmap/frontend-chatbot-runtime.zh.md)。
本文与测试共同描述当前运行时行为。

## 当前 v2 实现

Gateway 运行时位于 `src/qwen_audio_agent/`，使用 Python 实现。
LangGraph 只负责两个有界状态机，不是整个产品的状态容器。v2 公开基线是这套混合架构：
Python Gateway 的图与运行时、Node ACP 适配器、现有 Desktop/Web/TUI 宿主。
它不是 LangGraph Platform 应用，也不把这两张图作为独立库发布。
不要把 ACP 适配器改写成 Python 来「补完」迁移。

```text
Desktop / WebUI / TUI
          │ HTTP、WebSocket、播放回执
          ▼
Python Transport + Realtime Application ◄──► Realtime Provider
          │ 归一化工具调用
          ▼
FrontendTurnGraph
├── inline / control 结果返回当前轮次
└── spawn_thinking ──► 受理回执 ──► WorkGraph
                                         │ BackendPort
                                         ▼
                                     已配置后台
                                         │ 最终 Work 结果
                                         ▼
                             ResultDeliveryCoordinator
                             安全窗口 ──► 语音：playback.started
                                         纯文本：响应成功完成
                                         │
                                         ▼
                                  Task 交付状态持久化
```

`FrontendTurnGraph` 的作用域是一个实时对话轮次。它校验并去重工具调用，
执行能力、次数、时间和结果大小限制，路由 inline、control 与 background 工具。
它的扩展契约支持在副作用前 checkpoint 并等待授权，但当前公开工具目录没有启用该路径。
它不管理音频、模型响应、后台选择或 Work 执行。

`WorkGraph` 的作用域是一个 Task。owner 调度器、长耗时后台操作和 Adapter
连接留在图外。归一化后台事件恢复图执行；图将事件归并为有界状态，并在
下一次等待前 checkpoint。Task Repository 再由该状态更新并发布公开投影。

各类状态只有一个负责方：

| 状态 | 持久化负责方 | 含义 |
| --- | --- | --- |
| Frontend 与 Work 执行 | LangGraph SQLite Checkpointer，每个轮次或 Task 一个 thread | 有界控制流快照，用于安全恢复图执行 |
| 少量跨 thread 索引 | LangGraph SQLite Store | 前台待授权索引等运行时查找数据 |
| Task 生命周期、结果与通知租约 | JSON Task Repository | 产品账本与交付事实源 |
| 对话记录 | Conversation Store | 近期用户/助手时间线，不是图状态 |
| 后台原生 Session | Backend Adapter | 后台私有连续性与恢复依据 |

checkpoint 不能证明外部副作用已发生、后台 Session 仍存在，或音频已到达客户端。
这些事实仍由各自的 Repository、Adapter 或播放回执负责。

现有 Node 与 Electron Host 使用的 Gateway 启动入口是兼容薄壳，进程链为：

```text
Node / Electron Host → python-gateway-entry.mjs → Python Gateway
Python Work Runtime → NodeBackendBridge → python-backend-bridge.mjs
                    → 现有 JavaScript 后台 Adapter
```

该入口启动 `python -m qwen_audio_agent.cli`，并转发结构化就绪消息、信号与退出状态。
CLI、TUI、Desktop UI/进程代码与保留的后台 Adapter 仍承担实际 JavaScript 职责；
当前运行链路不在 JavaScript 内维护第二套 Gateway 或 Task 状态机。

公开 `qwenaudio` 与桌面版配置流程保持不变。薄壳将已有环境变量契约传给
Python CLI，包括 `HOST`、`PORT`、Realtime 设置与 `AGENT_PROTOCOL`。
`QWAUDIO_DATA_DIR` 保存共享身份、画像、记忆、配置和清单；
`QWAUDIO_CONFIG_DIR` 保存 Gateway 租约、Task 账本、对话、checkpoint 和皮肤等
运行时状态。未设置数据目录时，它回退到配置目录。启动器优先使用
`QWEN_AUDIO_AGENT_PYTHON`，其次使用工程虚拟环境，否则使用锁定的 `uv` 工程。
Python 源码开发可直接运行 `uv run python -m qwen_audio_agent.cli`。v2 Desktop 包携带
锁定工程，但不内置 Python 解释器或 `uv`；主机必须提供 `uv`，或让
`QWEN_AUDIO_AGENT_PYTHON` 指向已安装运行依赖的环境。

当前生效的能力是第 3 节的内置实时工具目录，以及 ACP 后台桥。下列文档中的扩展点
**尚未接入** Python Gateway。旧版 JavaScript Gateway 运行时已移除；公开合同模块
仍保留，供后续移植。相关配置项在完成移植前不会生效：

- 前台 MCP Client（`QWEN_AUDIO_FRONTEND_MCP_CONFIG`）
- 前台 OpenAPI Adapter（`QWEN_AUDIO_FRONTEND_OPENAPI_CONFIG`）
- Frontend Profile 工具源（`QWEN_AUDIO_FRONTEND_PROFILE`）
- 知识检索 Provider
- 前台联网搜索（`QWEN_AUDIO_WEB_SEARCH_PROVIDER`）
- 会话结束自动记忆整理
- 本地唤醒词（`QWEN_AUDIO_WAKE_WORD_ENABLED`）

## 1. 用户可见模型

用户与一个 qwen-audio 助手对话。内部存在两个 qwen-audio-agent 层：

1. **实时前端** — 全双工语音、简单直接回答，以及基本的本地时间/记忆工具。
2. **后端 Agent** — 一个用户配置的办事 Agent，负责处理需要工具、文件、应用程序、代码、设备控制或多步执行的请求。

公开 Python CLI 当前注册了保留的 ACP Driver，包括 OpenCode、OpenClaw、
Qoder、Qwen Code、Kimi Code 和 Pi。BackendPort SDK 也导出 A2A Adapter，
可用于自定义组合根，但公开 CLI 不选择它。后端内部可以使用工具、技能、
Agent 或其他 Session。这些都是后端私有实现细节，不会创建额外的
qwen-audio-agent 层。协议细节只存在于各自 Adapter 内；后端特定的启动和能力行为
位于已注册的 Driver 中。

## 2. 非阻塞请求流

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

`spawn_thinking` 永不等待所请求的工作完成。用户可以在多个 Task 项排队期间继续
说话。Coordinator Submission 按 owner 使用 FIFO Lane。Task 进入已验证的委派后即释放
该 Lane，因此委派的目标工作不会阻塞下一个 Coordinator Submission。

## 3. 实时边界

实时前端有意保持极小的工具集——工具少、延迟低、无多步编排。内置工具上限为：

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

`spawn_thinking` 仅在已配置后台时暴露。`enter_sleep` 仅在连接客户端声明
`sleeping` 状态时暴露。

`memory` 通过一个扁平接口维护两份普通 Markdown 文档。每次调用只执行一个
原子操作：`read`、`append` 或 `replace`；`replace` 使用唯一匹配的原文定位，
找不到或匹配多处时安全失败。每次调用都将自己的结果回传给 Realtime 模型。
两个文档的权限边界不同：

- `user` 是当前用户的长期个性化覆盖：称呼、关系、助手在其面前的名称、语言、表达
  风格和默认做法。它作为用户授权的指令材料注入，覆盖 `ASSISTANT.md` 中的实例级
  默认人设，并服从用户当前的话语。
- `memory` 是只用于理解和回答的持久事实与决定，绝不作为行为指令。两者按是否具有
  行为权威区分，而不是按主题机械分类。

两者都不能授权泄露内部结构、跳过权限检查，或改变任务和安全协议。随包的
`PROMPT.md` 是不可被个性化覆盖的核心规则；本地 `ASSISTANT.md` 只保存助手实例的默认
身份、人格、关系定位和表达风格。首次启动从随包模板创建，升级不覆盖，下一次语音会话
读取修改。助手不会通过记忆工具修改它。

`notes` 将用户命名的列表（购物清单、待办事项、阅读列表）作为前端自有的本地持久化集合
进行管理：单次调用即可完成添加、展示、匹配移除、清空和删除，无需后端参与。列表是
条目数据，而非记忆；稳定事实保留在 `memory` 中，列表条目不会被写入用户偏好或
事实记忆。条目和列表解析首先匹配精确文本，然后匹配唯一的不区分大小写子串，
否则报告歧义并将候选名称返回给模型以澄清。工具契约要求模型只在当前轮次
用户明确提出破坏性请求后调用 `clear` 和 `drop`；数据 Store 不再加第二道确认。

`get_agent_task_status` 是生命周期、进度和中间结果问题的唯一实时入口。
Gateway 直接读取自身持有的 Task 记录，包括 Adapter 归一化后的最新消息、活动和
产物摘要。状态查询不会创建另一个 Task，不调用协调 Agent，也不进入异步播报队列。

实时前端没有以下工具：

- 选择、创建、继续或取消后端 Session；
- 选择同步、异步、前台或后台执行；
- 选择后端执行策略；
- 选择工具、Agent 或子 Agent。

`respond_agent_permission` 是实时前端不控制后端执行这一规则的唯一例外。
它只能转发由 Gateway 提供的、针对待处理的、owner 作用域权限请求的明确当前轮次
用户决策。它可以理解自然的肯定或否定措辞，如"可以"或"不允许"，但不能在没有
当前轮次用户话语的情况下虚构同意、创建请求、选择工具或修改后端权限策略。
回复仅限于 `always` 和 `reject`，并只处理当前待决请求。`always` 只是为兼容现有
接口保留的批准字段值，不会形成会话授权。批准只映射到后端 `allow_once`；如果后端
没有提供单次允许选项，请求会被取消，不会静默扩大为 `allow_always`。拒绝优先选择
`reject_once`，必要时可用 `reject_always` 扩大拒绝范围。后续权限请求始终重新询问。

传递给 `spawn_thinking` 的 `objective` 是对用户请求的保守解释，而非执行计划。
提交前必须结合当前对话把"继续那个页面"之类的指代解析成一条自包含指令。该指令就是
后台 Agent 唯一收到的模型可见文本；ASR 原文不再作为第二份任务描述附加。后台 Agent
不接收前台人格、长期记忆或最近聊天历史；与执行有关的事实必须先解析进指令，而不是
转发这些文档。

Bridge 将 Task 的 `inputParts` 作为协议原生 Part 传给后台 Agent，而不是
JSON 清单。`inputRefs` 仅保留为内部兼容字段；当前公开工具不会填写它，也不会用它
解析更早对话轮次的附件。Task ID、owner、生命周期、时间戳和路由继续作为
Gateway/BackendPort 的结构化数据，不进入后台 Agent 的任务指令。工作目录和用户时区也不
重复拼入每轮文本；协议或后台自身的运行上下文负责这些信息。

## 4. 固定后端 Agent Session

ACP Adapter 为每组 protocol 与 owner 拥有一个持久协调器 Session 身份。
确切存储键是 Adapter 私有细节，由下列组合派生：

```text
<protocol> × <owner>
```

Gateway 在该稳定键之后存储原生 ACP Session ID，并在后续轮次调用
`session/resume`。项目委派同样在其记录的工作目录中恢复选定的原生 Session，
因此语音发起的工作保留在后端自己的 Session 历史中，而非 Gateway 副本中。

语音浏览器会话 ID 和 Task ID 不会更改该身份。因此，新的语音对话会继续使用
相同的后端 Agent 上下文。

Gateway 队列和 ACP 适配器都对写入进行串行化。这种双重保护防止并发消息在一个
后端 Session 内部发生竞争。

后端 Agent 拥有自己的执行策略。qwen-audio-agent 只提供一条自包含自然任务指令和
当前轮次的协议原生附件；它不转发前台历史或偏好，不规定状态 JSON，也不指导后端
Agent 如何使用后端特定能力。

## 5. Task 状态

qwen-audio-agent Task 记录是交付回执，而非后端内部任务图的镜像。

```text
scheduled ──► queued
queued → running ─────────────────────────→ completed
   │        └→ delegated → finalizing ────────┘
   └────────────→ cancelling → cancelled
                            ↘ failed
```

公共字段仅限于用户请求、时间戳、最终结果/错误、通用活动、带可选安全操作详情的
有界待处理权限摘要和通知状态。不存在执行模式、交付模式、子 Agent 状态、后端
权限标识符、后端拓扑或后端取消内部信息。

UI 将 `queued` 和 `running` 呈现为相同的"处理中"状态。队列位置是内部调度细节，
不会改变用户的双工对话。

排队中和直接执行中的 Task 在 Gateway 重启后无法安全恢复，因此会变为 failed 并附带
明确的重启原因。委派 Task 只有在 Adapter 能确认持久化原生 Session，且存在对应
WorkGraph checkpoint 时才会重新挂接。私有恢复关联会被持久化，但不进入公开 Task 投影。
已完成的结果和通知交付状态会被持久化。定时 Work 从 Task 账本重新设置，不从
Graph checkpoint 推断。

## 6. 进度动画

进度是可观测性，而非控制。ACP 适配器将标准 `session/update` 通知投射为通用活动：

- 工具名称、有界的用户安全详情和运行中/已完成状态；
- 计划进度；
- 不含思考内容的通用思考信号；
- 有界的 Session 标题和当前模式元数据。

UI 将此映射为稳定的任务说明或短语，如"搜索中"、"读取中"、"生成图像"或当前模式；
通用思考信号继续显示任务目标，桌面宠物则映射到已有的处理动画。Session ID、
子 Agent ID、原始权限载荷和原始思考内容不显示。待处理权限可以在类密钥值被脱敏后，
显示精确且有界的标题、说明、命令或路径，以便知情同意。

活动绝不会产生语音状态更新，也绝不影响队列。

## 7. 最终结果交付

后端 Agent 通过标准 ACP 回合返回结果：正文来自 `agent_message_chunk`，图片、音频和
资源保留为原生 `ContentBlock`，回合以 `session/prompt` 的 `PromptResponse` 结束。
ACP Adapter 不再把这些内容压成一段文本，也不要求模型补写 `state` 或
`presentation` JSON；它将文本和非文本内容分别投射为 BackendPort 的 `content` 与
`artifacts`。只有 `stopReason=end_turn` 代表回合成功完成；取消、拒绝、Token 或
Agent 请求次数耗尽分别进入 Gateway 的取消或失败路径。Gateway 再根据客户端能力
决定对话展示、资源卡片和语音表达。

已完成的结果优先返回到发起对话。在全新连接时，可以恢复同一 owner 的旧对话中
未完成的结果。可续期声明防止两个实时前端呈现相同结果。语音结果被注入实时上下文，
仅在当前客户端上报 `playback.started` 后标记为已交付。对有音频的响应，模型响应完成、
音频生成、`playback.ended` 和播放开始前的取消都不会确认交付。如果用户打断、正在说话
或有其他响应待处理，交付会等待。失败的尝试会释放声明或重试；过期租约可由其他在线前端
重新领取。重试有次数上限，因此一个格式异常的结果不会阻塞后续完成。

纯文本客户端没有播放回执。对它们而言，响应成功完成即表示交付确认；
响应失败时仍保持待交付。

当后端 Agent 调用 `session_start` 或 `session_send` 时，委派成立的权威事实是
Session 工具已经成功创建或续接目标任务，并返回 Adapter 验证过的运行与 Session
标识，而不是模型输出的某个字段。ACP 只负责如实传递工具调用、工具结果和当前回合的
终止；Adapter 将验证后的关联发布给 Work 运行时，后者据此把原始 Task 移至
`delegated`，并释放后端 Agent 串行化锁和 Task 调度通道。因此，其他语音请求可以在
目标 Session 运行期间使用协调器。

协调 Agent 可以在工具成功后自然结束当前 ACP 回合，但该文本不控制任务状态，也不会
被解释为完成信号。关联 ID、目标 Session 和生命周期完全保留在 Gateway 的 Task
Registry 与 Adapter 运行时中，不要求模型回显。

适配器独立地保持 Task 生命周期和事件订阅存活。只有与委派 ID 关联的匹配 ACP
目标提示完成才能完成 Task。然后适配器短暂重新获取后端 Agent 锁，并将经验证的
结果及其原生 ContentBlock 发送给它整理最终答复。繁忙的目标、空结果、无关的
Session 更新或旧结果
都无法完成 Task。

ACP Agent 轮次不设人为墙钟超时。初始协调轮次、委派目标轮次和最终整理轮次，
只会在 ACP 报告完成、用户显式取消 Task，或后端进程退出/关闭时结束。连接初始化
和有界控制 RPC 仍保留超时，避免不可用的后端无限阻塞 Gateway 启动。

取消是确认式的，而非乐观式的。`queued` Task 在本地取消。`running` 或
`finalizing` Task 中止其活跃后端请求。对于 `delegated` Task，首先请求空闲的
协调器调用 `session_cancel`；如果协调器 Session 被占用，ACP 适配器直接向精确
关联的目标 Session 发送 `session/cancel`。Task 保持 `cancelling` 状态，
直到其中一条路径确认停止，然后变为 `cancelled`。停止失败则变为 `failed` 并
附带取消错误。在适配器直接中止后，Gateway 会记录一个取消事实，并在下一个安全的
协调器轮次中注入一次。这样可以在不延迟取消或重复停止的情况下协调器的历史。

前台的受理确认来自 Gateway 已经创建的 Task，而不是协调 Agent 自报的委派状态。
协调轮次仍依 ACP 生命周期信号自然结束；Gateway 不会根据受理文本或 Session
工具调用成功来推断该轮次已完成。

## 8. 后端内部能力

对于接受客户端提供的 MCP 服务器的 ACP 后端（包括 OpenCode、Qoder、Qwen Code 和
Kimi Code），Gateway 向协调器注入相同的五个工具：Session list、start、
send、status 和 cancel。OpenClaw ACP 不接受客户端提供的 MCP 服务器，
因此相同的协调契约映射到 OpenClaw 的原生 Session 工具。`session_start`
和 `session_send` 返回不透明的委派 ID。在任一成功后，后端 Agent 不得轮询、
重复工作或从自己的上下文中回答；适配器负责等待、取消、权限路由和结果关联。

协调 MCP Server 还会通过 MCP 初始化响应的 `instructions` 字段发布稳定协调契约。
后台 Driver 只有在确认 Agent Host 会把 MCP Server instructions 投射进模型上下文后，
才声明 `coordinatorMcpInstructions`；这些后台每轮只接收动态自然任务指令，避免把
相同的路由与返回规则反复追加到持久 Session 历史。目前已确认 OpenCode、Qoder、Qwen Code
和 Claude Code，并将共享内容控制在 2 KiB 的可移植预算内。尚未验证的后台，或未来
超过预算的内容，继续使用完整的逐轮 Prompt 安全回退。该标志不表示后台是否普遍支持
MCP。项目 Session 不会连接协调 MCP Server。

`session_status` 仅用于观察。如果查询失败，后端 Agent 必须报告失败；
不得使用原生工具检查目标目录或复制委派的工作。

前端代码不得依赖于选择了哪个内部能力。前端任务快照只能暴露有界的标题和
通用委派状态，绝不暴露委派 ID、目标 Session ID、目录或原始事件。

## 9. 依赖方向

```text
WebUI / TUI / Desktop
   ↓ WebSocket and HTTP
Python Transport + Realtime Application
   ↓ 归一化工具调用
FrontendTurnGraph
   ↓ WorkSubmissionPort
WorkGraph + Task Repository + Scheduler
   ↓ 与协议无关的 BackendPort
Node Backend Bridge
   ↓ Adapter 投影：自然任务指令 + 原生附件 Part
OpenCode ACP, OpenClaw ACP bridge, Qoder ACP, Qwen Code ACP,
Kimi Code ACP, or another supported Agent
```

后台特定的 JavaScript Adapter 细节保留在 `server/src/agent`；Python Work Runtime
只能看到 `BackendPort`。Realtime 工具不得导入后台 Adapter。UI 仅消费公开 Task
事件和最终时间线内容。Transport 依赖 Application 与 Domain 契约，Provider 与后台
Adapter 则通过 Port 指向内部。

Gateway 可以将不可变的 `web/dist` 产物作为部署便利来提供，但这仅是静态托管。
Gateway 源码不得导入 UI 组件、呈现文本、样式、终端行为或桌面行为。
所有三个 UI 拥有自己的渲染，并将结构化协议字段映射到各自的标签和交互模式。

## 10. 进程所有权

Gateway 是唯一的核心产品服务。后台生命周期由共享的 `owned/external` 归属模型管理：

- `owned`：Gateway 启动后端所需的本地进程，并在退出时停止它们。后台原生进程负责
  加载自己的用户配置、模型、工具和 MCP；适配器只提供协议参数和必要的公共能力。
- `external`：仅适用于声明了外部服务能力的后端。Gateway 不启动、改端口或停止该
  后台，只通过后端公开的协议地址连接，把配置与状态完全留给外部服务管理。

后台服务归属与 ACP 连接方式是两个相互独立的维度。每个后台 profile 声明一个
`acpConnection`；连接工厂当前实现 `process`，即启动一个本地 ACP stdio 子进程。
未来的远程 ACP bridge 可以新增另一种连接类型，而无需修改协调、权限、Task 或
Session 生命周期代码。声明外部后台服务，并不意味着 ACP 连接也自动变成远程连接。

每个后台通过一份经过校验的 Plugin 契约注册。目录项统一拥有身份、安装、原生配置
入口、进程环境和归属元数据；Agent Driver 与 Runtime Driver 必须显式声明完整的布尔
能力，缺失或互相矛盾时在启动阶段直接拒绝。后台子进程只接收跨平台运行所需的系统
变量和当前 Plugin 声明的凭证命名空间，Gateway 身份、Realtime、Memory 以及其他
后台的密钥不会跨过该边界。通用 ACP 命令如确有需要，可通过
`QWEN_AUDIO_AGENT_ACP_FORWARD_ENV` 显式列出额外变量名。

Python HTTP/WebSocket 应用由可注入的组合根构造。导入应用工厂不会监听端口；
CLI 负责实际组装，测试及未来 Host 可以注入彼此隔离的 Input Arbitration、
Task 与 Conversation Source、Identity、Health 和 Realtime Event Handler。
Node/Electron bootstrap 只保留启动兼容与进程所有权，不实现应用行为。

共享适配器通常拥有一个 ACP stdio 子进程，并随 Gateway 一起停止。OpenCode、Qoder、
Qwen Code 和 Kimi Code 直接作为 ACP Agent 运行；OpenCode 还可以额外启动其原生本地 Session
UI 服务。当前 `OPENCODE_BASE_URL` 表示这个 UI 服务地址，不是远程 ACP 执行端点，
因此 OpenCode 仍属于 `owned`。

OpenClaw 使用一个小型 ACP bridge。未显式配置地址时，Gateway 启动一个具有隔离运行时
和 Session 状态的 OpenClaw Gateway；显式配置 `OPENCLAW_BASE_URL` 时，则直接连接用户
已有的 OpenClaw Gateway，并且不读取、复制或修改其认证和 Agent 状态。此时服务归属为
`external`，ACP 连接仍是本地 `process`：本地官方 bridge 通过 WebSocket/WSS 连接远程
OpenClaw Gateway。外部连接不使用面向本地启动的短时端口探测，而由 bridge 报告实际的
网络、TLS 和认证结果。本地 bridge 退出只会中断 ACP 连接，不会触碰远程 Gateway。

Codex 也遵循同一边界：qwen-audio-agent 通过 ACP stdio 启动 `codex-acp`，该适配器再
通过自己的本地 stdio 协议启动 Codex App Server。Codex App Server 可以提供其他传输，
但它们不是远程 ACP 端点，不应泄漏进共享 ACP 适配层。

Desktop、TUI 和 WebUI 是可替换的 Gateway 客户端。Python CLI 从现有环境变量
契约选择 Realtime 模型；客户端不按连接覆盖该模型。health 发布当前 Provider、
精确模型档案、模型目录、输入采样率、配置签名、可选 Provider 与后台摘要。
Desktop 只能配置和重启其本地自有 Gateway。
关闭 UI 不会影响排队中的工作或固定的后台 Agent Session。更改实时或后端行为的配置
在下次 Gateway 启动时生效；更改 UI 的 Gateway URL 仅重新连接该 UI。

macOS 桌面渲染器打包在应用程序内部。Electron 从私有的随机回环路径提供这些
不可变资源，并仅代理 Gateway HTTP API 和 Realtime WebSocket 流量。桌面 UI
资源不得从 Gateway 加载：重新构建桌面应用程序必须足以更新其外观，而无需升级
正在运行的 Gateway 前端。

## 11. 审查清单

合并变更前，请验证：

1. 后端工作排队或运行时，实时前端是否仍能对话？
2. 每个可执行请求是否进入同一个持久后端 Agent Session？
3. 任何前端 API 是否获得了 Session、子 Agent、权限或执行模式的知识？
4. 工具事件是否仅用于通用 UI 进度？
5. 完成播报是否仅来自最终后端 Agent 结果？
6. 任何 UI 是否开始管理 Gateway 或后端进程？
7. 打断是否能在不取消已提交 Task 的情况下推迟语音？
8. 测试是否覆盖 FIFO 串行化、固定 Session 复用、checkpoint 与账本边界、
   工具动画和播放开始确认的交付重试？
