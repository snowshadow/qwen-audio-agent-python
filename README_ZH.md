# qwen-audio-agent — LangGraph Gateway（Python）

[中文](README_ZH.md) | [English](README.md)

[![CI](https://github.com/snowshadow/qwen-audio-agent-python/actions/workflows/ci.yml/badge.svg)](https://github.com/snowshadow/qwen-audio-agent-python/actions/workflows/ci.yml)
[![npm](https://img.shields.io/npm/v/qwen-audio-agent)](https://www.npmjs.com/package/qwen-audio-agent)
[![node](https://img.shields.io/badge/node-%E2%89%A522.22.2-brightgreen)](https://nodejs.org/)
[![python](https://img.shields.io/badge/python-3.12–3.14-blue)](https://www.python.org/)
[![license](https://img.shields.io/github/license/snowshadow/qwen-audio-agent-python)](LICENSE)

本仓库是 [QwenAudio/qwen-audio-agent](https://github.com/QwenAudio/qwen-audio-agent)
的**独立 LangGraph / Python 分支**，使用 Apache-2.0，**不是** QwenAudio 官方发行。
上游署名见 [`NOTICE`](NOTICE)。

这里的重点是 **v2 Gateway**：用 **两张 LangGraph 状态机**（Python）负责有界控制流；
CLI、WebUI、TUI、Desktop 宿主与 **ACP 后台适配器**仍留在 Node。这是语音 Gateway
产品，不是 LangGraph Platform 应用，也不把图当作独立库发布。

产品宣传、桌面版发布与社区渠道请看上游
[QwenAudio/qwen-audio-agent](https://github.com/QwenAudio/qwen-audio-agent)。
`npm install -g qwen-audio-agent` 安装的仍是**官方** npm 包。

## 与上游的差异

| 方面 | 本仓库 | 上游 |
| --- | --- | --- |
| Gateway 控制流 | Python + **LangGraph**（`FrontendTurnGraph`、`WorkGraph`） | JavaScript Gateway（v1） |
| 长耗时 I/O | Realtime、ACP、模型在**图外** | 同一产品方向，不同运行时 |
| 后台 Agent | 经 `BackendPort` 走现有 **Node ACP 适配器** | 同一套适配器 |
| 状态 | v2 alpha（`2.0.0a0`），开发中 | 稳定 v1 线 |

本树已**删除**旧版 JavaScript Gateway。下列 v1 扩展尚未接到 Python Gateway，
相关配置项目前**无效**：前台 MCP、OpenAPI、知识检索、联网搜索、会话结束记忆抽取、
本地唤醒词。

## 架构

![LangGraph Gateway 总览](docs/architecture-overview.png)

```text
Desktop / WebUI / TUI
          │ HTTP、WebSocket、播放回执
          ▼
Python Transport + Realtime Application ◄──► Realtime Provider（DashScope 等）
          │ 归一化工具调用
          ▼
FrontendTurnGraph          一轮实时对话：校验、策略、inline/control/background
          │
          └─ spawn_thinking ──► WorkGraph     一个 Task：checkpoint 生命周期
                                      │ BackendPort（Node 桥）
                                      ▼
                              ACP 后台适配器 ──► 办事 Agent
                                      │ 结果
                                      ▼
                          ResultDeliveryCoordinator → 语音 / 文本客户端
```

**LangGraph 只管有界控制流。** 音频流、模型 Session、调度 lane、ACP I/O 都在图外的
普通 Python / Node 代码里完成。

| 图 | 作用域 | 职责 |
| --- | --- | --- |
| `FrontendTurnGraph` | 一轮语音对话 | 工具准入、限额、inline 与后台路由 |
| `WorkGraph` | 一个后台 Task | 事件归并、checkpoint、公开任务投影 |

**进程边界**（宿主契约不变）：

```text
Node/Electron 宿主 → python-gateway-entry.mjs → python -m qwen_audio_agent.cli
WorkGraph 后台     → NodeBackendBridge → python-backend-bridge.mjs → JS ACP 适配器
```

源码位置：

- `src/qwen_audio_agent/graphs/` — LangGraph 定义与门面
- `src/qwen_audio_agent/transport/` — FastAPI / WebSocket Gateway
- `server/src/agent/` — ACP 适配器（仍为 JavaScript）

完整不变量、状态归属与未接入扩展清单：
[docs/architecture.zh.md](docs/architecture.zh.md)。

Gateway 合同版本：**4.0.0**（[docs/contract.zh.md](docs/contract.zh.md)）。

## 环境要求

- Node.js **22.22.2+** 或 **24.15.0+**，npm **10+**
- Python **3.12–3.14** 与 [uv](https://docs.astral.sh/uv/)
- 使用默认云端 Realtime 前台时需要 DashScope API Key

## 安装

安装**本仓库**（LangGraph Gateway）：

```bash
npm install -g git+https://github.com/snowshadow/qwen-audio-agent-python.git
```

从源码安装：

```bash
git clone https://github.com/snowshadow/qwen-audio-agent-python.git
cd qwen-audio-agent-python
uv sync --frozen
npm install
npm run install:global
```

详见 [docs/getting-started/install.zh.md](docs/getting-started/install.zh.md)。

## 快速开始

1. 创建配置并填入 Realtime API Key：

```bash
qwenaudio config
```

```dotenv
DASHSCOPE_API_KEY=your-key
QWEN_AUDIO_REALTIME_MODEL=qwen-audio-3.0-realtime-plus
# 可选后台；留空则仅前台模式
AGENT_PROTOCOL=openclaw
```

2. 启动 Gateway 与客户端：

```bash
qwenaudio          # 终端 1
qwenaudio tui      # 终端 2 — 或：qwenaudio webui
```

通过 CLI 修改 Gateway 配置后，需要重启 Gateway 进程。

配置说明：[docs/configuration.zh.md](docs/configuration.zh.md)。

## 开发

改 Gateway 走 Python；改后台适配器走 `server/src/agent/`。

```bash
uv sync --frozen
npm ci
npm run test:python    # pytest + ruff
npm test               # Node 工作区 + ACP 适配器
npm run dev            # Gateway 与 WebUI 热更新
```

见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 链接

- Issue：[github.com/snowshadow/qwen-audio-agent-python/issues](https://github.com/snowshadow/qwen-audio-agent-python/issues)
- 上游：[github.com/QwenAudio/qwen-audio-agent](https://github.com/QwenAudio/qwen-audio-agent)
- 许可证：[Apache-2.0](LICENSE)（上游署名见 [`NOTICE`](NOTICE)）
- 安全：[SECURITY.md](SECURITY.md) · 隐私：[PRIVACY.md](PRIVACY.md)
