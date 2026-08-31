# 助手画像、用户偏好与记忆

前台上下文分成四层，职责互不重叠：

| 层级 | 载体 | 职责 |
| --- | --- | --- |
| 核心规则 | `config/frontend-agent/PROMPT.md` | 工具协议、权限、安全和任务边界，用户记忆不能覆盖 |
| 助手画像 | `ASSISTANT.md` | 助手实例的默认身份、人格、关系定位和表达风格，由用户或二次开发者配置 |
| 用户偏好 | `USER.md` / `user` | 当前用户明确设定的长期个性化覆盖，覆盖助手默认人设 |
| 长期记忆 | `MEMORY.md` / `memory` | 用于理解用户和回答问题的长期事实与决定，不具有行为权威 |

指令冲突按“核心规则 → 用户当前明确要求 → 用户偏好 → 助手画像”处理。长期记忆
不在指令优先级中，它只是回答依据；与用户当前陈述冲突时，以当前陈述为准。
因此，对话中说“以后回答短一点”或“以后你叫小舟”会更新当前用户的 `USER.md`，
不会修改实例级 `ASSISTANT.md`；本轮临时要求只在本轮生效。

用户数据保存在配置目录下（CLI 为 `~/.config/qwaudio/`）：

| 文件 | 说明 |
| --- | --- |
| `ASSISTANT.md` | 实例级默认人设；名称、人格、关系定位和表达风格 |
| `USER.md` | 当前用户的长期个性化覆盖 |
| `MEMORY.md` | 关于用户的长期事实与决定 |
| `memory-audit.jsonl` | 自动记忆的诊断日志（补丁、跳过、失败逐条追加，仅供事后查阅） |
| `tasks.json` | 后台任务结果和待通知状态 |
| `state.env` | 本地身份密钥（首次启动自动生成，仅当前用户可读写） |
| `logs/` | 经过凭据脱敏并自动轮转的本地运行日志 |

这些文件只保存在本机，不会写入源码仓库，文件权限为仅当前用户可读写。

## 助手画像

首次启动会从随包模板 `config/frontend-agent/ASSISTANT.md` 创建本地
`ASSISTANT.md`，之后升级不会覆盖。直接编辑本地文件即可更改整个助手实例的默认名称、
人格、关系定位和表达风格，下一次建立语音会话时生效；也可用
`QWEN_AUDIO_AGENT_ASSISTANT_PROFILE_PATH` 指向其他文件。

`ASSISTANT.md` 不是对话记忆，也不是运行规则。助手不会通过 `memory` 工具修改它；
写在其中的工具、权限、安全、记忆、任务路由或能力声明不会覆盖 `PROMPT.md`。

## 用户偏好

`USER.md` 是当前用户对默认人设的长期个性化覆盖，不是第二份助手人设，也不是通用事实
仓库。它可以保存助手如何称呼用户、用户如何称呼助手，以及用户明确要求长期采用的语言、
回复风格和默认做法。只有用户明确设定或纠正时才会修改；会话后自动整理尚未接入当前 Python Gateway，
在完成移植前不能补记这类指令。

判断标准不是描述对象，而是作用域：“助手默认叫千问 Audio”属于 `ASSISTANT.md`；当前
用户在对话中说“以后你叫小舟”，则“小舟”是当前用户的覆盖，属于 `USER.md`。同理，
“默认继续 A 项目”属于 `USER.md`，而“A 项目使用 React”只是事实，属于 `MEMORY.md`。
文件是普通 Markdown，工具写入立即生效，直接编辑则在下一次语音会话生效。如需放在
其他位置，可设置
`QWEN_AUDIO_AGENT_USER_MODEL_PATH`（旧名称
`QWEN_AUDIO_AGENT_USER_PROFILE_PATH` 仍可读取）。

请勿在其中保存密码、API Key、验证码或令牌。

旧版 `frontend-memory.json` 中的 `profile`、`rules` 和 `user` 内容会在首次启动时
迁移到 `USER.md`。

## 长期记忆

`MEMORY.md` 使用普通 Markdown 保存关于用户的长期事实与决定，例如所在地、习惯、兴趣、
关系、项目、目标和计划。它只帮助理解和回答，不直接支配行为。内容来源有两种：

- **明确要求**：对话中说“记住、改成、不再”等，助手会生成精确 Markdown 修改；
  一句话中的多项信息会在同一轮逐项处理，并只生成一次最终回应。
- **自动整理（尚未接入 v2 Python Gateway）**：旧版 JavaScript Gateway 可在会话结束后
  用轻量文本模型查漏补缺。当前 Python Gateway 未接入该路径。在完成移植前，请通过
  `memory` 工具写入长期内容。
  `QWEN_AUDIO_MEMORY_AUTO=off` 仍记录后续移植后的关闭开关。
  见[架构文档](../architecture.zh.md#当前-v2-实现)。

Realtime 通过记忆服务提交受限 Markdown 变更，不直接写文件。密码、密钥等敏感内容
会被双重过滤拦截。`memory-audit.jsonl` 仍是后续自动整理移植后的诊断日志格式。
觉得内容不对，直接在对话中说“那条记错了”或“忘掉它”即可；助手会修改或删除
对应 Markdown 原文。

前台只暴露一个 `memory` 工具，每次调用执行一个原子操作：`read` 读取文档，
`append` 追加 Markdown，`replace` 用文档中唯一匹配的 `old_text` 替换或删除内容。
一句话包含多项持久修改时，Realtime 可在同一轮逐项调用，Gateway 只生成一次
后续回应。写入前会重新读取最新文档，精确替换找不到或匹配多处时安全失败。

## 替换记忆 Provider

> **v2 Python Gateway。** 当前 Gateway 使用内置 Markdown 文档。
> 下面的 `MemoryProvider` 合同尚未接到 Python 组合根。v2 不再导出嵌入式
> JavaScript Gateway。

内置的 `USER.md` 和 `MEMORY.md` 是默认实现，不是 Gateway 的固定存储依赖。宿主应用
可以从公开入口实现版本化的 `MemoryProvider`，并在 Composition Root 注入：

```js
import { MEMORY_PROVIDER_PROTOCOL_VERSION } from 'qwen-audio-agent/memory-provider'

const memoryProvider = {
  describe: () => ({
    protocolVersion: MEMORY_PROVIDER_PROTOCOL_VERSION,
    key: 'company-memory',
    label: 'Company Memory',
  }),
  list(ownerId, options) {
    return []
  },
  async apply(ownerId, changes, context) {
    return { changed: 0, documents: [] }
  },
  health: () => ({ ok: true }),
  async close() {},
}
```

`list()` 必须返回同步、有界的 Realtime 上下文快照；远程 Provider 应在 Adapter 内维护
本地缓存。`apply()` 可以异步，`context` 中的来源、Session、Turn 和 Trace 由 Gateway
提供，不属于模型可控的修改内容。Provider 返回的文档会统一限制长度、规范 scope，并
丢弃重复或无效文档。

Realtime、自动整理器和工具处理器只依赖 `FrontendMemoryRuntime`，不会访问供应商 SDK、
数据库或 Markdown 文件。未注入 Provider 时继续使用现有 Markdown 实现，现有配置和数据
无需迁移。第三方 Adapter 自行负责远程认证、租户映射、缓存刷新和底层记录到 `user`、
`memory` 两种公开文档语义的转换。

## 日志

日志采用 JSON Lines 格式，API Key、Token、Authorization、Cookie、密码和
Secret 字段会在写入前脱敏，默认不记录麦克风音频、用户转写正文、模型回复正文
或任务结果。桌面版可在“设置 → 应用 → 日志”中打开日志目录。详见
[配置说明](../configuration.zh.md#本地日志)。
