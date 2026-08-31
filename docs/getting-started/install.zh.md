# 安装

需要 Node.js 22.22.2+ 或 24.15.0+、npm 10+、Python 3.12–3.14 和
[uv](https://docs.astral.sh/uv/)。首次启动 Gateway 时，程序会通过
`uv run --frozen --no-dev` 准备锁定的 Python 运行环境。使用默认的 DashScope
实时语音前台时，还需要 DashScope API Key。
仓库提供 `.nvmrc` 和 `.node-version`；使用 nvm 时可直接运行 `nvm use`。

本仓库是 [QwenAudio/qwen-audio-agent](https://github.com/QwenAudio/qwen-audio-agent)
的独立 Apache-2.0 衍生作品，不是 QwenAudio 组织的官方发行。下面的命令安装的是
**本仓库**。`npm install -g qwen-audio-agent` 安装的仍是上游项目。

## 从 GitHub 安装

```bash
npm install -g git+https://github.com/snowshadow/qwen-audio-agent-python.git
```

## 从源码安装

```bash
git clone https://github.com/snowshadow/qwen-audio-agent-python.git
cd qwen-audio-agent-python
uv sync --frozen
npm install
npm run install:global
```

## 升级

升级到本仓库的最新代码：

```bash
npm install -g git+https://github.com/snowshadow/qwen-audio-agent-python.git
```

升级后如果以后台服务方式运行 Gateway，需执行 `qwenaudio gateway restart` 让新版本生效。

## 验证安装

查看配置文件的准确位置并确认安装就绪：

```bash
qwenaudio config
```

配置后台 Agent 后，可运行只读检查确认后台可执行文件、ACP 接入和适配器是否就绪：

```bash
qwenaudio setup
```

## 配置文件位置

CLI 使用 `~/.config/qwaudio/config.env`；桌面版使用系统标准应用数据目录
（macOS 为 `~/Library/Application Support/Qwen Audio Agent`），两者的 Gateway、
锁、日志与设置互不干扰，可以同时运行。设置 `QWAUDIO_CONFIG_DIR` 或
`XDG_CONFIG_HOME` 可以更改配置目录。详见[配置说明](../configuration.zh.md)。

## 获取 DashScope API Key

阿里云百炼为 Qwen Audio 3.0 Realtime 提供
[新人免费额度](https://help.aliyun.com/zh/model-studio/new-free-quota)，创建 API Key 后
即可免费开始使用 qwen-audio-agent。

1. 打开百炼控制台的 [API Key 页面](https://bailian.console.aliyun.com/?tab=model#/api-key)，
   登录账号，单击**创建 API Key**。
2. 复制生成的 Key，稍后填入 `config.env`。请勿公开或提交 API Key。

详细说明见[百炼官方文档](https://help.aliyun.com/zh/model-studio/get-api-key)。
